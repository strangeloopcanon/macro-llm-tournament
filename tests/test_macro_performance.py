import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.macro_performance_gate import (
    DEFAULT_TARGET_CATALOG,
    _bottom_line,
    _vintage_oos_empirical_eligible,
    build_oos_pairwise_comparison,
    build_performance_attribution,
    build_performance_vintage_readiness,
    catalog_sha256,
    load_performance_target_catalog,
    macro_performance_verdict,
)
from macro_llm_tournament.demand_vintage_oos import (
    _balanced_origin_sample,
    _normalize_context,
    _normalize_origins,
    build_vintage_cards_and_targets,
    fixture_vintage_panel,
    vintage_forecast_cache_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, env=_env(), text=True, capture_output=True, check=False)


class MacroPerformanceTests(unittest.TestCase):
    def test_target_catalog_is_executable_and_hashable(self):
        catalog = load_performance_target_catalog(DEFAULT_TARGET_CATALOG)

        self.assertFalse(catalog.empty)
        self.assertEqual(catalog["target_id"].nunique(), catalog.shape[0])
        self.assertEqual(len(catalog_sha256(catalog)), 64)
        self.assertTrue({"lab", "oos"}.issubset(set(catalog["split"])))
        self.assertTrue(catalog["blocking"].map(type).eq(bool).all())

        with TemporaryDirectory() as temp_dir:
            bad_catalog = Path(temp_dir) / "bad_catalog.csv"
            catalog.drop(columns=["target_id"]).to_csv(bad_catalog, index=False)
            with self.assertRaises(ValueError):
                load_performance_target_catalog(bad_catalog)

    def test_demand_vintage_oos_fixture_hides_dates_and_targets(self):
        with TemporaryDirectory() as temp_dir:
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--max-origins",
                    "8",
                    "--output-dir",
                    temp_dir,
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(temp_dir)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            cards = pd.read_csv(root / "demand_vintage_oos_cards.csv")
            targets = pd.read_csv(root / "demand_vintage_oos_targets.csv")
            scores = pd.read_csv(root / "demand_vintage_oos_scores.csv")
            leakage = pd.read_csv(root / "demand_vintage_oos_leakage_audit.csv")

            self.assertEqual(manifest["verdict"], "demand_vintage_oos_fixture_ready")
            self.assertTrue(manifest["passed"])
            self.assertFalse(cards.empty)
            self.assertFalse(targets.empty)
            self.assertFalse(scores.empty)
            self.assertTrue(leakage.empty)

            payload_text = "\n".join(cards["prompt_payload_json"].astype(str).head(25).tolist())
            self.assertIsNone(re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", payload_text))
            self.assertNotIn("target_value", payload_text)
            self.assertNotIn("target_raw_value", payload_text)
            self.assertNotIn("target_observation_date", payload_text)
            self.assertNotIn("as_of_date", payload_text)
            self.assertNotIn("realized", payload_text.lower())

    def test_demand_vintage_oos_replay_reads_cached_model_forecasts(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            provider_dir = cache_dir / "codex_cli"
            provider_dir.mkdir(parents=True)

            origins, context = fixture_vintage_panel()
            sampled = _balanced_origin_sample(_normalize_origins(origins), 8)
            cards, _targets = build_vintage_cards_and_targets(sampled, _normalize_context(context), history_periods=8)
            self.assertFalse(cards.empty)
            for _, card in cards.iterrows():
                prompt_payload = json.loads(card["prompt_payload_json"])
                cache_name = vintage_forecast_cache_name("codex_cli", "gpt-5.5", prompt_payload)
                cache_payload = {
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "payload": {
                        "forecast_value": 0.0,
                        "confidence": 0.61,
                        "reason": "cached replay forecast from relative history",
                    },
                    "cache_hit": False,
                }
                (provider_dir / f"{cache_name}.json").write_text(json.dumps(cache_payload), encoding="utf-8")

            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--forecast-mode",
                    "replay",
                    "--provider",
                    "codex_cli",
                    "--models",
                    "gpt-5.5",
                    "--cache-dir",
                    str(cache_dir),
                    "--max-origins",
                    "8",
                    "--output-dir",
                    str(root / "replay_out"),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            out = root / "replay_out"
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            forecasts = pd.read_csv(out / "demand_vintage_oos_forecasts.csv")
            raw_records = json.loads((out / "demand_vintage_oos_raw_records.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["forecast_mode"], "replay")
            self.assertEqual(manifest["live_call_count"], 0)
            self.assertEqual(manifest["cache_hit_count"], len(cards))
            self.assertEqual(len(raw_records), len(cards))
            self.assertIn("llm_codex_cli_gpt-5.5", set(forecasts["source"]))
            self.assertIn("llm_belief", set(forecasts["variant"]))

    def test_demand_vintage_oos_live_preflights_call_budget_before_provider(self):
        with TemporaryDirectory() as temp_dir:
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--forecast-mode",
                    "live",
                    "--provider",
                    "codex_cli",
                    "--models",
                    "gpt-5.5",
                    "--fresh-cache",
                    "--max-live-calls",
                    "1",
                    "--max-origins",
                    "8",
                    "--output-dir",
                    temp_dir,
                ]
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("--max-live-calls must be at least", result.stderr)

    def test_demand_vintage_oos_live_fresh_cache_rejects_existing_cache_json(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            (cache_dir / "codex_cli").mkdir(parents=True)
            (cache_dir / "codex_cli" / "stale.json").write_text("{}", encoding="utf-8")

            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--forecast-mode",
                    "live",
                    "--provider",
                    "codex_cli",
                    "--models",
                    "gpt-5.5",
                    "--fresh-cache",
                    "--cache-dir",
                    str(cache_dir),
                    "--max-live-calls",
                    "100",
                    "--max-origins",
                    "8",
                    "--output-dir",
                    str(root / "out"),
                ]
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("require an empty cache directory", result.stderr)

    def test_macro_performance_fixture_scores_lab_without_empirical_overclaim(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            demand_dir = root / "demand"
            vintage_dir = root / "vintage_oos"
            gate_dir = root / "performance"

            demand = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_economy",
                    "--belief-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--models",
                    "gpt-5.5",
                    "--household-source",
                    "fixture",
                    "--household-count",
                    "12",
                    "--period-count",
                    "20",
                    "--variants",
                    "representative,adaptive,llm_belief",
                    "--feedback-mode",
                    "closed_loop",
                    "--scenarios",
                    "baseline,transfer_shock,rate_hike,job_risk_shock,belief_feedback",
                    "--output-dir",
                    str(demand_dir),
                ]
            )
            self.assertEqual(demand.returncode, 0, demand.stderr)

            vintage = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--max-origins",
                    "8",
                    "--output-dir",
                    str(vintage_dir),
                ]
            )
            self.assertEqual(vintage.returncode, 0, vintage.stderr)

            gate = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_performance_gate",
                    "--mode",
                    "fixture",
                    "--demand-run-dir",
                    str(demand_dir),
                    "--vintage-oos-dir",
                    str(vintage_dir),
                    "--output-dir",
                    str(gate_dir),
                ]
            )
            self.assertEqual(gate.returncode, 0, gate.stderr)

            manifest = json.loads((gate_dir / "manifest.json").read_text(encoding="utf-8"))
            summary = pd.read_csv(gate_dir / "macro_performance_variant_summary.csv")
            attribution = pd.read_csv(gate_dir / "macro_performance_attribution.csv")
            report = (gate_dir / "macro_performance_report.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["verdict"], "macro_lab_performance_ready")
            self.assertTrue(manifest["passed"])
            self.assertFalse(manifest["empirical_ready"])
            self.assertIn("macro_lab_performance_ready", report)

            llm_lab = summary[(summary["split"] == "lab") & (summary["variant"] == "llm_belief")]
            self.assertFalse(llm_lab.empty)
            self.assertEqual(int(llm_lab["blocking_fail_count"].max()), 0)
            self.assertEqual(int(llm_lab["blocking_gap_count"].max()), 0)
            self.assertFalse(attribution.empty)
            self.assertIn("lab", set(attribution["split"]))
            self.assertNotIn("no_change", set(summary.loc[summary["split"] == "lab", "variant"]))

            replay_with_fixture_oos = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_performance_gate",
                    "--mode",
                    "replay",
                    "--demand-run-dir",
                    str(demand_dir),
                    "--vintage-oos-dir",
                    str(vintage_dir),
                    "--output-dir",
                    str(root / "performance_replay_fixture_oos"),
                ]
            )
            self.assertEqual(replay_with_fixture_oos.returncode, 0, replay_with_fixture_oos.stderr)
            replay_manifest = json.loads((root / "performance_replay_fixture_oos" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(replay_manifest["verdict"], "macro_lab_performance_ready")
            self.assertFalse(replay_manifest["empirical_ready"])
            self.assertFalse(replay_manifest["vintage_oos_empirical_eligible"])

    def test_macro_performance_empirical_verdict_requires_oos_baseline_beat(self):
        scores = pd.DataFrame([{"placeholder": 1}])
        summary = pd.DataFrame(
            [
                {
                    "source": "llm_lab",
                    "variant": "llm_belief",
                    "split": "lab",
                    "scored_count": 4,
                    "blocking_fail_count": 0,
                    "blocking_gap_count": 0,
                    "critical_fail_count": 0,
                    "weighted_normalized_loss": 0.10,
                },
                {
                    "source": "llm_oos",
                    "variant": "llm_belief",
                    "split": "oos",
                    "scored_count": 1,
                    "blocking_fail_count": 0,
                    "blocking_gap_count": 0,
                    "critical_fail_count": 0,
                    "weighted_normalized_loss": 0.25,
                },
            ]
        )
        worse_attribution = pd.DataFrame(
            [
                {
                    "split": "oos",
                    "llm_weighted_loss": 0.25,
                    "best_baseline_weighted_loss": 0.20,
                    "loss_improvement_pct": -25.0,
                }
            ]
        )
        better_attribution = pd.DataFrame(
            [
                {
                    "split": "oos",
                    "llm_weighted_loss": 0.15,
                    "best_baseline_weighted_loss": 0.20,
                    "loss_improvement_pct": 25.0,
                }
            ]
        )

        self.assertEqual(
            macro_performance_verdict(scores, summary, worse_attribution, mode="replay", oos_empirical_eligible=True),
            "macro_lab_performance_ready",
        )
        self.assertEqual(
            macro_performance_verdict(scores, summary, better_attribution, mode="replay", oos_empirical_eligible=True),
            "macro_empirical_oos_ready",
        )

    def test_oos_attribution_uses_raw_error_when_target_loss_is_clipped(self):
        summary = pd.DataFrame(
            [
                {
                    "source": "llm_live",
                    "variant": "llm_belief",
                    "split": "oos",
                    "weighted_normalized_loss": 1.0,
                    "macro_performance_score": 0.0,
                },
                {
                    "source": "rolling_trend",
                    "variant": "rolling_trend",
                    "split": "oos",
                    "weighted_normalized_loss": 1.0,
                    "macro_performance_score": 0.0,
                },
            ]
        )
        scores = pd.DataFrame(
            [
                {
                    "source": "llm_live",
                    "variant": "llm_belief",
                    "split": "oos",
                    "metric": "weighted_normalized_abs_error",
                    "status": "fail",
                    "value": 2.0,
                    "normalized_loss": 1.0,
                },
                {
                    "source": "rolling_trend",
                    "variant": "rolling_trend",
                    "split": "oos",
                    "metric": "weighted_normalized_abs_error",
                    "status": "fail",
                    "value": 3.0,
                    "normalized_loss": 1.0,
                },
            ]
        )

        attribution = build_performance_attribution(summary, scores=scores)

        self.assertEqual(attribution.loc[0, "llm_weighted_loss"], 2.0)
        self.assertEqual(attribution.loc[0, "best_baseline_weighted_loss"], 3.0)
        self.assertAlmostEqual(float(attribution.loc[0, "loss_improvement_pct"]), 33.3333333333)

    def test_oos_pairwise_comparison_clusters_by_origin_against_best_baseline(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for origin_id in ["origin_a", "origin_b"]:
                for card_idx in range(2):
                    card_id = f"{origin_id}_card_{card_idx}"
                    for source, variant, loss in [
                        ("llm_live", "llm_belief", 1.0),
                        ("rolling_trend", "rolling_trend", 3.0),
                        ("rolling_mean", "rolling_mean", 4.0),
                        ("no_change", "no_change", 5.0),
                    ]:
                        rows.append(
                            {
                                "source": source,
                                "variant": variant,
                                "origin_id": origin_id,
                                "card_id": card_id,
                                "normalized_abs_error": loss,
                            }
                        )
            pd.DataFrame(rows).to_csv(root / "demand_vintage_oos_joined_errors.csv", index=False)

            pairwise = build_oos_pairwise_comparison(root, bootstrap_samples=200, seed=7)

            self.assertEqual(pairwise.loc[0, "llm_source"], "llm_live")
            self.assertEqual(pairwise.loc[0, "best_baseline_source"], "rolling_trend")
            self.assertEqual(int(pairwise.loc[0, "n_clusters"]), 2)
            self.assertEqual(pairwise.loc[0, "mean_loss_reduction"], 2.0)
            self.assertAlmostEqual(float(pairwise.loc[0, "improvement_pct"]), 66.6666666667)
            self.assertEqual(pairwise.loc[0, "bootstrap_share_positive"], 1.0)

    def test_vintage_readiness_uses_configured_oos_artifact_dir(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            panel_dir = root / "panel"
            oos_dir = root / "oos"
            panel_dir.mkdir()
            oos_dir.mkdir()
            (panel_dir / "fred_vintage_status.json").write_text(
                json.dumps({"origin_count": 24, "series_ids": ["PCECC96", "PSAVERT", "CPIAUCSL", "UNRATE", "FEDFUNDS", "UMCSENT", "GDPC1"]}),
                encoding="utf-8",
            )
            pd.DataFrame({"origin": [f"origin_{idx}" for idx in range(8)], "split": ["test"] * 8}).to_csv(
                panel_dir / "forecast_origins_for_vintage_context.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "as_of_date": ["2026-01-01"],
                    "realtime_start": ["2026-01-01"],
                    "realtime_end": ["2026-01-01"],
                    "observation_date": ["2025-10-01"],
                    "series_id": ["PCECC96"],
                }
            ).to_csv(panel_dir / "fred_vintage_context.csv", index=False)
            for filename in [
                "demand_vintage_oos_cards.csv",
                "demand_vintage_oos_targets.csv",
                "demand_vintage_oos_scores.csv",
            ]:
                (oos_dir / filename).write_text("ok\n", encoding="utf-8")

            readiness = build_performance_vintage_readiness(panel_dir, oos_dir)
            artifact_rows = readiness[readiness["metric"].astype(str).str.startswith("demand_vintage_oos_")]

            self.assertEqual(set(artifact_rows["status"]), {"pass"})
            self.assertEqual(set(artifact_rows["source"]), {str(oos_dir)})

    def test_live_oos_empirical_eligibility_rejects_cache_hits(self):
        base = {
            "passed": True,
            "mode": "panel",
            "forecast_mode": "live",
            "verdict": "demand_vintage_oos_scored",
            "live_call_count": 10,
            "cache_hit_count": 0,
        }

        self.assertTrue(_vintage_oos_empirical_eligible(base))
        self.assertFalse(_vintage_oos_empirical_eligible({**base, "cache_hit_count": 1}))
        self.assertFalse(_vintage_oos_empirical_eligible({**base, "live_call_count": 0}))
        self.assertTrue(
            _vintage_oos_empirical_eligible(
                {
                    "passed": True,
                    "mode": "panel",
                    "forecast_mode": "replay",
                    "verdict": "demand_vintage_oos_scored",
                    "live_call_count": 0,
                    "cache_hit_count": 10,
                }
            )
        )

    def test_performance_report_distinguishes_oos_baseline_win_from_empirical_pass(self):
        manifest = {
            "verdict": "macro_lab_performance_ready",
            "vintage_oos_scores_available": True,
            "vintage_oos_empirical_eligible": True,
            "vintage_oos_llm_baseline_improvement_pct": 9.8,
        }

        text = _bottom_line(manifest, pd.DataFrame())

        self.assertIn("beats the strongest deterministic baseline", text)
        self.assertIn("absolute OOS target is still missed", text)
        self.assertNotIn("has not beaten", text)

    def test_macro_performance_live_mode_blocks_without_spending_calls(self):
        with TemporaryDirectory() as temp_dir:
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_performance_gate",
                    "--mode",
                    "live",
                    "--output-dir",
                    temp_dir,
                ]
            )

            self.assertEqual(result.returncode, 1)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["verdict"], "macro_performance_live_blocked")
            self.assertFalse(manifest["passed"])


if __name__ == "__main__":
    unittest.main()
