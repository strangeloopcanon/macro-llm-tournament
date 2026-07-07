import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.agent_common import WORK_ROOT
from macro_llm_tournament.agent_types import build_household_type_cells
from macro_llm_tournament.behavior_ecology import fixture_policy_payload
from macro_llm_tournament.phase4_matched_twins import (
    OutputMappingSpec,
    default_output_mapping,
    mapping_sha256,
    mapped_period_value,
    normalized_mapping_payload,
    phase4_scoring_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase4MatchedTwinsTests(unittest.TestCase):
    def test_phase4_mapping_hash_is_canonical(self):
        payload = normalized_mapping_payload(default_output_mapping())
        round_tripped = json.loads(json.dumps(payload, sort_keys=True))

        self.assertEqual(mapping_sha256(payload), mapping_sha256(round_tripped))
        self.assertEqual(len(payload["rows"]), 5)
        self.assertIn("leakage_rule", payload)
        self.assertEqual(payload["schema_version"], "phase4_prior_update_matched_twins_v2")
        saving_row = next(row for row in payload["rows"] if row["target_name"] == "personal_saving_rate_pct")
        self.assertEqual(saving_row["economy_variable"], "saving_rate")
        self.assertEqual(saving_row["economy_transform"], "diff")
        self.assertEqual(saving_row["target_transform"], "diff")
        self.assertLess(saving_row["lower"], 0.0)
        self.assertGreater(saving_row["upper"], 0.0)

    def test_phase4_saving_rate_diff_transform_applies_to_economy_and_target(self):
        spec = OutputMappingSpec(
            target_name="personal_saving_rate_pct",
            series_id="PSAVERT",
            target_label="Personal saving rate",
            target_units="percent",
            target_transform="diff",
            economy_variable="saving_rate",
            economy_transform="diff",
            period_alignment="card_i_scores_economy_period_i_plus_1_against_next_month_target",
            lower=-25.0,
            upper=25.0,
            note="test mapping",
        )
        path = pd.DataFrame(
            [
                {"aggregate_saving": 10.0, "aggregate_income": 100.0},
                {"aggregate_saving": 15.0, "aggregate_income": 100.0},
            ]
        )
        self.assertAlmostEqual(mapped_period_value(path, 1, spec), 5.0)

        targets = pd.DataFrame(
            [
                {
                    "card_id": "card_1",
                    "period_id": "period_1",
                    "target_month": "2026-01-01",
                    "target_name": "personal_saving_rate_pct",
                    "target_label": "Personal saving rate",
                    "target_value": 4.4,
                    "target_available": True,
                    "last_signal": 3.6,
                    "history_scale": 0.05,
                    "series_id": "PSAVERT",
                    "units": "percent",
                    "contamination_label": "post_model_cutoff_clean",
                    "source_url": "https://fred.stlouisfed.org/series/PSAVERT",
                }
            ]
        )
        adjusted = phase4_scoring_targets(targets, [spec])
        row = adjusted.iloc[0]

        self.assertAlmostEqual(row["target_value"], 0.8)
        self.assertAlmostEqual(row["last_signal"], 0.0)
        self.assertAlmostEqual(row["raw_target_value"], 4.4)
        self.assertAlmostEqual(row["raw_last_signal"], 3.6)
        self.assertEqual(row["phase4_target_transform"], "diff")
        self.assertGreaterEqual(row["history_scale"], 0.25)

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

    def test_phase4_policy_schedule_mode_records_shared_behavior_executor(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ecology_dir = root / "ecology"
            policy_records = root / "behavior_policy_records.json"
            output_dir = root / "phase4"
            _write_minimal_ecology_dir(ecology_dir)
            _write_fixture_policy_records(policy_records)

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
                    "--behavior-policy-mode",
                    "schedule",
                    "--behavior-policy-raw-records-json",
                    str(policy_records),
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
            decisions = pd.read_csv(output_dir / "phase4_household_decisions.csv")
            report = (output_dir / "phase4_matched_twins_report.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["behavior_policy"]["mode"], "schedule")
            self.assertEqual(manifest["behavior_policy"]["schema_version"], "demand_behavior_policy_schedule_v1")
            self.assertIn("schedule", set(decisions["behavior_policy_mode"].astype(str)))
            self.assertTrue(decisions["behavior_policy_type_id"].astype(str).str.len().gt(0).any())
            self.assertIn("LLM-authored behavior-policy schedule", report)

    def test_phase4_state_schedule_mode_uses_state_conditioned_policy_profile(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ecology_dir = root / "ecology"
            state_policy_dir = root / "state_policy"
            output_dir = root / "phase4"
            _write_minimal_ecology_dir(ecology_dir)

            state_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.state_policy_schedules",
                    "--mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--household-source",
                    "persona_ecology_replay",
                    "--persona-ecology-dir",
                    str(ecology_dir),
                    "--output-dir",
                    str(state_policy_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)

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
                    "--behavior-policy-mode",
                    "state_schedule",
                    "--behavior-policy-state-profile-json",
                    str(state_policy_dir / "state_behavior_policy_profile.json"),
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
            decisions = pd.read_csv(output_dir / "phase4_household_decisions.csv")
            report = (output_dir / "phase4_matched_twins_report.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["behavior_policy"]["mode"], "state_schedule")
            self.assertEqual(manifest["behavior_policy"]["schema_version"], "demand_behavior_state_policy_schedule_v1")
            self.assertIn("state_schedule", set(decisions["behavior_policy_mode"].astype(str)))
            self.assertTrue(decisions["behavior_policy_type_id"].astype(str).str.len().gt(0).any())
            self.assertIn("state-conditioned behavior-policy schedule", report)


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


def _write_fixture_policy_records(path: Path) -> None:
    type_cells, _ = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=2022)
    records = [
        {
            "record_type": "policy_transfer",
            "cache_hit": True,
            "payload": fixture_policy_payload(type_cells, family="transfer"),
        },
        {
            "record_type": "policy_income_loss",
            "cache_hit": True,
            "payload": fixture_policy_payload(type_cells, family="income_loss"),
        },
    ]
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
