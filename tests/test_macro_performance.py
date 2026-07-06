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
    _vintage_oos_provenance,
    _vintage_oos_empirical_eligible,
    build_oos_family_pairwise_comparison,
    build_oos_pairwise_comparison,
    build_performance_attribution,
    build_performance_vintage_readiness,
    catalog_sha256,
    load_performance_target_catalog,
    macro_performance_verdict,
)
from macro_llm_tournament.demand_vintage_oos import (
    _balanced_origin_sample,
    _filter_origins_by_splits,
    _legacy_prompt_payloads_by_card,
    _normalize_context,
    _normalize_origins,
    build_vintage_cards_and_targets,
    fixture_vintage_panel,
    vintage_forecast_cache_name,
)
from macro_llm_tournament.belief_calibration import BELIEF_CALIBRATION_VERSION
from macro_llm_tournament.demand_vintage_audit import run_qualitative_recall_adaptive
from macro_llm_tournament.forecast_audit import (
    DirectRecallClient,
    fixture_qualitative_recall_payload,
    qualitative_recall_cache_name,
    qualitative_recall_prompt,
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
            forecasts = pd.read_csv(root / "demand_vintage_oos_forecasts.csv")
            scores = pd.read_csv(root / "demand_vintage_oos_scores.csv")
            leakage = pd.read_csv(root / "demand_vintage_oos_leakage_audit.csv")

            self.assertEqual(manifest["verdict"], "demand_vintage_oos_fixture_ready")
            self.assertTrue(manifest["passed"])
            self.assertFalse(cards.empty)
            self.assertFalse(targets.empty)
            self.assertTrue(
                {"no_change", "rolling_mean", "rolling_trend", "ar2", "recursive_least_squares"}.issubset(
                    set(forecasts["source"])
                )
            )
            self.assertFalse(scores.empty)
            self.assertTrue(leakage.empty)

            payload_text = "\n".join(cards["prompt_payload_json"].astype(str).head(25).tolist())
            self.assertIsNone(re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", payload_text))
            self.assertNotIn("target_value", payload_text)
            self.assertNotIn("target_raw_value", payload_text)
            self.assertNotIn("target_observation_date", payload_text)
            self.assertNotIn("as_of_date", payload_text)
            self.assertNotIn("realized", payload_text.lower())

    def test_demand_vintage_oos_split_filter_writes_only_requested_split(self):
        with TemporaryDirectory() as temp_dir:
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--splits",
                    "val",
                    "--output-dir",
                    temp_dir,
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(temp_dir)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            cards = pd.read_csv(root / "demand_vintage_oos_cards.csv")
            targets = pd.read_csv(root / "demand_vintage_oos_targets.csv")

            self.assertEqual(manifest["splits"], ["val"])
            self.assertEqual(set(cards["split"]), {"val"})
            self.assertEqual(set(targets["split"]), {"val"})
            self.assertFalse(cards.empty)

            all_result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--output-dir",
                    str(root / "all"),
                ]
            )
            self.assertEqual(all_result.returncode, 0, all_result.stderr)
            all_cards = pd.read_csv(root / "all" / "demand_vintage_oos_cards.csv")
            split_keys = set(cards[["origin_id", "target_name"]].itertuples(index=False, name=None))
            all_val_keys = set(
                all_cards[all_cards["split"].astype(str) == "val"][["origin_id", "target_name"]].itertuples(
                    index=False,
                    name=None,
                )
            )
            self.assertEqual(split_keys, all_val_keys)

    def test_vintage_pct_change_targets_use_final_current_denominator(self):
        origins = pd.DataFrame(
            [
                {"origin": "origin_test", "as_of_date": "2025-01-15", "split": "test"},
                {"origin": "origin_later", "as_of_date": "2025-04-15", "split": "train"},
            ]
        )
        context = pd.DataFrame(
            [
                {
                    "origin": "origin_test",
                    "as_of_date": "2025-01-15",
                    "series_id": "PCECC96",
                    "label": "real consumption",
                    "observation_date": "2024-07-01",
                    "value": 95.0,
                    "realtime_start": "2025-01-15",
                    "realtime_end": "2025-01-15",
                },
                {
                    "origin": "origin_test",
                    "as_of_date": "2025-01-15",
                    "series_id": "PCECC96",
                    "label": "real consumption",
                    "observation_date": "2024-10-01",
                    "value": 100.0,
                    "realtime_start": "2025-01-15",
                    "realtime_end": "2025-01-15",
                },
                {
                    "origin": "origin_later",
                    "as_of_date": "2025-04-15",
                    "series_id": "PCECC96",
                    "label": "real consumption",
                    "observation_date": "2024-10-01",
                    "value": 200.0,
                    "realtime_start": "2025-04-15",
                    "realtime_end": "2025-04-15",
                },
                {
                    "origin": "origin_later",
                    "as_of_date": "2025-04-15",
                    "series_id": "PCECC96",
                    "label": "real consumption",
                    "observation_date": "2025-01-01",
                    "value": 220.0,
                    "realtime_start": "2025-04-15",
                    "realtime_end": "2025-04-15",
                },
            ]
        )

        cards, targets = build_vintage_cards_and_targets(_normalize_origins(origins), _normalize_context(context), history_periods=2)
        target = targets[targets["target_name"] == "real_consumption_growth_pct"].iloc[0]

        self.assertEqual(float(target["current_value"]), 100.0)
        self.assertEqual(float(target["target_current_value"]), 200.0)
        self.assertAlmostEqual(float(target["target_value"]), 10.0)
        self.assertIn('"value":100.0', str(cards.iloc[0]["history_json"]).replace(" ", ""))

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

    def test_demand_vintage_oos_replay_can_read_legacy_split_local_cache(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            provider_dir = cache_dir / "codex_cli"
            provider_dir.mkdir(parents=True)

            origins, context = fixture_vintage_panel()
            origins = _filter_origins_by_splits(_normalize_origins(origins), ["val"])
            cards, _targets = build_vintage_cards_and_targets(origins, _normalize_context(context), history_periods=8)
            legacy_payloads = _legacy_prompt_payloads_by_card(cards)
            self.assertFalse(cards.empty)
            self.assertFalse(cards["origin_id"].astype(str).str.startswith("vintage_origin_0000").all())
            self.assertEqual(set(legacy_payloads), set(cards["card_id"].astype(str)))
            for _, card in cards.iterrows():
                legacy_prompt_payload = legacy_payloads[str(card["card_id"])]
                cache_name = vintage_forecast_cache_name("codex_cli", "gpt-5.5", legacy_prompt_payload)
                cache_payload = {
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "payload": {
                        "forecast_value": 0.0,
                        "confidence": 0.61,
                        "reason": "legacy split-local cached replay forecast",
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
                    "--splits",
                    "val",
                    "--output-dir",
                    str(root / "replay_out"),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            out = root / "replay_out"
            raw_records = json.loads((out / "demand_vintage_oos_raw_records.json").read_text(encoding="utf-8"))
            forecasts = pd.read_csv(out / "demand_vintage_oos_forecasts.csv")

            self.assertTrue(all(record.get("legacy_cache_hit") for record in raw_records))
            self.assertTrue(all(record.get("legacy_cache_name") for record in raw_records))
            self.assertEqual(set(forecasts["origin_id"].astype(str)), set(cards["origin_id"].astype(str)))

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
                        ("ar2", "ar2", 2.5),
                        ("recursive_least_squares", "recursive_least_squares", 2.0),
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
            self.assertEqual(pairwise.loc[0, "best_baseline_source"], "recursive_least_squares")
            self.assertEqual(int(pairwise.loc[0, "n_clusters"]), 2)
            self.assertEqual(pairwise.loc[0, "mean_loss_reduction"], 1.0)
            self.assertAlmostEqual(float(pairwise.loc[0, "improvement_pct"]), 50.0)
            self.assertEqual(pairwise.loc[0, "bootstrap_share_positive"], 1.0)

    def test_oos_family_pairwise_comparison_reports_family_bootstrap_and_dm(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for origin_id in ["origin_a", "origin_b", "origin_c", "origin_d"]:
                for target_name, llm_loss, baseline_loss in [
                    ("inflation_growth_pct", 1.0, 2.0),
                    ("output_growth_pct", 3.0, 1.0),
                ]:
                    card_id = f"{origin_id}_{target_name}"
                    rows.extend(
                        [
                            {
                                "source": "llm_live",
                                "variant": "llm_belief",
                                "origin_id": origin_id,
                                "card_id": card_id,
                                "target_name": target_name,
                                "normalized_abs_error": llm_loss,
                            },
                            {
                                "source": "recursive_least_squares",
                                "variant": "recursive_least_squares",
                                "origin_id": origin_id,
                                "card_id": card_id,
                                "target_name": target_name,
                                "normalized_abs_error": baseline_loss,
                            },
                            {
                                "source": "no_change",
                                "variant": "no_change",
                                "origin_id": origin_id,
                                "card_id": card_id,
                                "target_name": target_name,
                                "normalized_abs_error": baseline_loss + 1.0,
                            },
                        ]
                    )
            pd.DataFrame(rows).to_csv(root / "demand_vintage_oos_joined_errors.csv", index=False)

            family = build_oos_family_pairwise_comparison(root, bootstrap_samples=200, seed=7)

            self.assertEqual(set(family["target_family"]), {"inflation_growth_pct", "output_growth_pct"})
            inflation = family[family["target_family"] == "inflation_growth_pct"].iloc[0]
            output = family[family["target_family"] == "output_growth_pct"].iloc[0]
            self.assertEqual(inflation["best_baseline_source"], "recursive_least_squares")
            self.assertGreater(float(inflation["mean_loss_reduction"]), 0.0)
            self.assertLess(float(output["mean_loss_reduction"]), 0.0)
            self.assertIn("dm_z_stat", family.columns)
            self.assertIn("dm_two_sided_p", family.columns)

    def test_demand_vintage_audit_fixture_writes_recall_and_belief_outputs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vintage_dir = root / "vintage"
            audit_dir = root / "audit"
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

            audit = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_audit",
                    "--run-dir",
                    str(vintage_dir),
                    "--recall-mode",
                    "fixture",
                    "--qualitative-recall-mode",
                    "fixture",
                    "--models",
                    "gpt-5.5,gpt-5.4",
                    "--output-dir",
                    str(audit_dir),
                ]
            )

            self.assertEqual(audit.returncode, 0, audit.stderr)
            manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
            recall = pd.read_csv(audit_dir / "direct_recall_scores.csv")
            qualitative = pd.read_csv(audit_dir / "qualitative_recall_scores.csv")
            cutoff = pd.read_csv(audit_dir / "cutoff_status.csv")
            belief = pd.read_csv(audit_dir / "audit_belief_structure_summary.csv")

            self.assertEqual(manifest["status"], "ok")
            self.assertGreater(manifest["card_count"], 0)
            self.assertEqual(manifest["live_call_count"], 0)
            self.assertIn("adaptive_live_call_ceiling", manifest)
            self.assertEqual(manifest["qualitative_recall_batch_size"], 49)
            self.assertEqual(manifest["qualitative_recall_min_batch_size"], 12)
            self.assertTrue(manifest["cache_root"])
            self.assertEqual(set(recall["model"]), {"gpt-5.5", "gpt-5.4"})
            self.assertIn("ALL", set(recall["variable"]))
            self.assertIn("ALL", set(qualitative["group"]))
            self.assertIn("gpt-5.4-user-supplied", set(cutoff["cutoff_label"]))
            self.assertIn("ALL", set(belief["variable"]))

    def test_demand_vintage_audit_replay_splits_qualitative_child_caches(self):
        with TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            provider = "codex_cli"
            model = "gpt-5.4"
            targets = pd.DataFrame(
                [
                    {
                        "card_id": f"card_{idx}",
                        "variable": "real_consumption_growth_pct",
                        "variable_name": "Real consumption growth",
                        "origin": f"period_{idx}",
                        "horizon": 1,
                        "target_quarter_label": f"period_{idx + 1}",
                    }
                    for idx in range(4)
                ]
            )
            write_client = DirectRecallClient(provider, model, cache_root, mode="replay", max_live_calls=0)
            provider_dir = cache_root / provider
            provider_dir.mkdir(parents=True, exist_ok=True)
            for start in (0, 2):
                batch = targets.iloc[start : start + 2].copy()
                prompt = qualitative_recall_prompt(batch)
                cache_name = qualitative_recall_cache_name(write_client, prompt)
                path = provider_dir / f"{cache_name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "provider": provider,
                            "model": model,
                            "payload": fixture_qualitative_recall_payload(batch),
                            "cache_hit": False,
                            "cache_path": str(path),
                            "response_created_utc": "2026-01-01T00:00:00+00:00",
                        }
                    ),
                    encoding="utf-8",
                )

            predictions, raw_records, replay_client = run_qualitative_recall_adaptive(
                targets,
                write_client,
                batch_size=4,
                min_batch_size=2,
            )

            self.assertEqual(predictions.shape[0], 4)
            self.assertEqual(replay_client.cache_hit_count, 2)
            self.assertTrue(any(record["split_batch"] for record in raw_records))

    def test_demand_vintage_audit_failed_replay_marks_manifest_failed(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vintage_dir = root / "vintage"
            audit_dir = root / "audit"
            vintage = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--max-origins",
                    "2",
                    "--output-dir",
                    str(vintage_dir),
                ]
            )
            self.assertEqual(vintage.returncode, 0, vintage.stderr)

            audit = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_audit",
                    "--run-dir",
                    str(vintage_dir),
                    "--recall-mode",
                    "off",
                    "--qualitative-recall-mode",
                    "replay",
                    "--qualitative-recall-batch-size",
                    "4",
                    "--qualitative-recall-min-batch-size",
                    "2",
                    "--models",
                    "gpt-5.4",
                    "--output-dir",
                    str(audit_dir),
                ]
            )

            self.assertNotEqual(audit.returncode, 0)
            manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("Replay mode cache miss", manifest["error"])

    def test_belief_calibration_fits_validation_and_scores_evaluation(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_dir = root / "val"
            evaluation_dir = root / "test"
            output_dir = root / "calibration"

            calibration = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--splits",
                    "val",
                    "--output-dir",
                    str(calibration_dir),
                ]
            )
            self.assertEqual(calibration.returncode, 0, calibration.stderr)
            evaluation = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--splits",
                    "test",
                    "--output-dir",
                    str(evaluation_dir),
                ]
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)

            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.belief_calibration",
                    "--calibration-run-dir",
                    str(calibration_dir),
                    "--evaluation-run-dir",
                    str(evaluation_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            profile = json.loads((output_dir / "belief_calibration_profile.json").read_text(encoding="utf-8"))
            forecasts = pd.read_csv(output_dir / "demand_vintage_oos_forecasts.csv")
            model = pd.read_csv(output_dir / "belief_calibration_model.csv")
            summary = pd.read_csv(output_dir / "demand_vintage_oos_summary.csv")

            self.assertEqual(profile["schema_version"], BELIEF_CALIBRATION_VERSION)
            self.assertEqual(manifest["calibration_splits"], ["val"])
            self.assertEqual(manifest["evaluation_splits"], ["test"])
            self.assertTrue(manifest["passed"])
            self.assertFalse(model.empty)
            self.assertTrue(any(str(source).endswith("_calibrated") for source in forecasts["source"]))
            self.assertTrue(any(str(source).endswith("_calibrated") for source in summary["source"]))
            provenance = _vintage_oos_provenance(output_dir)
            self.assertEqual(provenance["verdict"], "demand_vintage_oos_scored")
            self.assertEqual(provenance["artifact_verdict"], "belief_calibration_empirical_ready")
            self.assertEqual(provenance["forecast_mode"], "fixture")

    def test_belief_calibration_fails_when_calibration_and_evaluation_splits_overlap(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluation_dir = root / "test"
            output_dir = root / "calibration"

            evaluation = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_vintage_oos",
                    "--mode",
                    "fixture",
                    "--splits",
                    "test",
                    "--output-dir",
                    str(evaluation_dir),
                ]
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)

            result = _run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.belief_calibration",
                    "--calibration-run-dir",
                    str(evaluation_dir),
                    "--evaluation-run-dir",
                    str(evaluation_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertNotEqual(result.returncode, 0)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["verdict"], "belief_calibration_split_leakage_failed")
            self.assertFalse(manifest["passed"])
            self.assertEqual(manifest["calibration_split_overlap"], ["test"])
            provenance = _vintage_oos_provenance(output_dir)
            self.assertEqual(provenance["verdict"], "belief_calibration_split_leakage_failed")

    def test_demand_economy_calibration_profile_changes_source_and_preserves_accounting(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": BELIEF_CALIBRATION_VERSION,
                        "profile_id": "test_profile",
                        "demand_adjustments": {
                            "calibration_strength": 0.1,
                            "income_rebound_gain": 0.05,
                            "inflation_attention_gain": 0.03,
                            "job_risk_regime_gain": 0.06,
                            "confidence_rebound_gain": 0.05,
                            "precaution_uncertainty_gain": 0.05,
                            "active_targets": ["output_growth_pct"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "demand"
            result = _run(
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
                    "llm_belief",
                    "--feedback-mode",
                    "closed_loop",
                    "--scenarios",
                    "baseline,transfer_shock,rate_hike,job_risk_shock,belief_feedback",
                    "--belief-calibration-profile",
                    str(profile_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            beliefs = pd.read_csv(output_dir / "demand_beliefs.csv")
            accounting = pd.read_csv(output_dir / "demand_accounting.csv")

            self.assertEqual(manifest["belief_calibration_profile_id"], "test_profile")
            self.assertTrue(set(beliefs["source"].astype(str)).pop().endswith("_calibrated"))
            self.assertLess(float(accounting["abs_residual"].max()), 1e-6)
            self.assertIn("calibrated_dynamics", " ".join(beliefs["reason_codes_json"].astype(str).tolist()))

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
