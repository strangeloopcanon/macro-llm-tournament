import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.phase4_matched_twins import (
    default_output_mapping,
    mapping_sha256,
    normalized_mapping_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase4MatchedTwinsTests(unittest.TestCase):
    def test_phase4_mapping_hash_is_canonical(self):
        payload = normalized_mapping_payload(default_output_mapping())
        round_tripped = json.loads(json.dumps(payload, sort_keys=True))

        self.assertEqual(mapping_sha256(payload), mapping_sha256(round_tripped))
        self.assertEqual(len(payload["rows"]), 5)
        self.assertIn("leakage_rule", payload)

    def test_phase4_fixture_cli_writes_locked_mapping_and_scores(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.phase4_matched_twins",
                    "--mode",
                    "fixture",
                    "--data-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--household-source",
                    "fixture",
                    "--household-count",
                    "6",
                    "--period-count",
                    "8",
                    "--asof-start",
                    "2025-12-15",
                    "--asof-end",
                    "2026-02-15",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(temp_dir)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            scores = pd.read_csv(root / "phase4_proxy_scores.csv")
            comparison = pd.read_csv(root / "phase4_twin_path_comparison.csv")
            accounting = pd.read_csv(root / "phase4_twin_accounting.csv")
            mapping = json.loads((root / "phase4_output_mapping.json").read_text(encoding="utf-8"))
            report = (root / "phase4_matched_twins_report.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["verdict"], "phase4_matched_twin_fixture_ready")
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["live_call_count"], 0)
            self.assertEqual(manifest["mapping_sha256"], mapping_sha256(mapping))
            self.assertIn("phase4_matched_twin_fixture_ready", report)
            self.assertIn("ALL", set(scores["target_name"].astype(str)))
            self.assertEqual({"adaptive", "llm_belief_fixture_gpt-5.5"}, set(scores["source"].astype(str)))
            self.assertFalse(comparison.empty)
            self.assertLessEqual(float(accounting["abs_residual"].max()), 1e-6)

    def test_phase4_live_mode_is_blocked_before_spending(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.phase4_matched_twins",
                    "--mode",
                    "live",
                    "--data-mode",
                    "fixture",
                    "--max-live-calls",
                    "1",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            root = Path(temp_dir)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("blocked", manifest["error"])
            self.assertFalse((root / "phase4_proxy_scores.csv").exists())

    def test_phase4_persona_ecology_replay_cli_scores_matched_twins(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ecology_dir = root / "ecology"
            output_dir = root / "phase4"
            _write_minimal_ecology_dir(ecology_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.phase4_matched_twins",
                    "--mode",
                    "replay",
                    "--belief-source",
                    "persona_ecology_replay",
                    "--persona-ecology-dir",
                    str(ecology_dir),
                    "--data-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--asof-start",
                    "2025-12-15",
                    "--asof-end",
                    "2025-12-15",
                    "--period-count",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            scores = pd.read_csv(output_dir / "phase4_proxy_scores.csv")
            beliefs = pd.read_csv(output_dir / "phase4_beliefs.csv")

            self.assertEqual(manifest["verdict"], "phase4_matched_twin_replay_scored")
            self.assertEqual(manifest["belief_source"], "persona_ecology_replay")
            self.assertEqual(manifest["persona_ecology_input"]["source"], "llm_codex_cli_gpt-5.5")
            self.assertIn("llm_codex_cli_gpt-5.5__phase4_prior_replay", set(scores["source"].astype(str)))
            self.assertIn("adaptive", set(scores["source"].astype(str)))
            self.assertIn("sce_prior_update_replay", "\n".join(beliefs["reason_codes_json"].astype(str)))


def _write_minimal_ecology_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "ok",
        "provider": "codex_cli",
        "models": ["gpt-5.5"],
        "primary_update_source": "llm_codex_cli_gpt-5.5",
        "prior_mode": "empirical",
        "feedback_mode": "none",
        "date_mode": "relative",
        "prior_update_evidence": {"evidence_verdict": "clears_prior_update_gate"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    panel_rows = []
    for period_index, period_id in enumerate(["sce_2024_10", "sce_2024_11"]):
        for respondent_id, income, liquidity, weight in [
            ("respondent_a", "low", "low", 0.55),
            ("respondent_b", "high", "high", 0.45),
        ]:
            panel_rows.append(
                {
                    "panel_row_id": f"{respondent_id}__{period_id}",
                    "period_id": period_id,
                    "period_index": period_index,
                    "respondent_id": respondent_id,
                    "survey_source": "ny_fed_sce_microdata",
                    "survey_date": "2024-10-01" if period_index == 0 else "2024-11-01",
                    "weight": weight,
                    "age_group": "35_54",
                    "income_group": income,
                    "education_group": "college_plus",
                    "gender": "female",
                    "region": "south",
                    "employment_status": "employed",
                    "homeownership": "owner",
                    "liquid_wealth_group": liquidity,
                    "prior_expected_inflation_1y": 3.0 + period_index,
                    "prior_expected_unemployment_higher_prob": 35.0 + 2.0 * period_index,
                    "prior_expected_real_income_growth": 1.0 - 0.2 * period_index,
                    "actual_expected_inflation_1y": 3.1 + period_index,
                    "actual_expected_unemployment_higher_prob": 36.0 + 2.0 * period_index,
                    "actual_expected_real_income_growth": 1.1 - 0.2 * period_index,
                }
            )
    pd.DataFrame(panel_rows).to_csv(root / "persona_ecology_panel.csv", index=False)
    prediction_rows = []
    for panel in panel_rows:
        for target, prior, prediction in [
            ("expected_inflation_1y", panel["prior_expected_inflation_1y"], panel["prior_expected_inflation_1y"] + 0.2),
            (
                "expected_unemployment_higher_prob",
                panel["prior_expected_unemployment_higher_prob"],
                panel["prior_expected_unemployment_higher_prob"] + 1.0,
            ),
            ("expected_real_income_growth", panel["prior_expected_real_income_growth"], panel["prior_expected_real_income_growth"] - 0.1),
        ]:
            prediction_rows.append(
                {
                    "schema_version": "persona_belief_ecology_v1",
                    "panel_row_id": panel["panel_row_id"],
                    "respondent_id": panel["respondent_id"],
                    "period_id": panel["period_id"],
                    "period_index": panel["period_index"],
                    "survey_source": "ny_fed_sce_microdata",
                    "survey_date": panel["survey_date"],
                    "source": "llm_codex_cli_gpt-5.5",
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "target_name": target,
                    "prior_prediction": prior,
                    "prediction": prediction,
                    "p10": prediction - 0.5,
                    "p50": prediction,
                    "p90": prediction + 0.5,
                    "confidence": 0.7,
                    "uncertainty": 0.5,
                    "profile_weight": 0.1,
                    "prior_weight": 0.5,
                    "environment_weight": 0.3,
                    "aggregate_feedback_weight": 0.1,
                    "cache_hit": True,
                    "call_source": "cache",
                }
            )
    pd.DataFrame(prediction_rows).to_csv(root / "persona_ecology_predictions.csv", index=False)


if __name__ == "__main__":
    unittest.main()
