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
from macro_llm_tournament.behavior_gate import (
    BEHAVIOR_CTC_HOLDOUT_SPLIT,
    BEHAVIOR_SCENARIOS,
    aggregate_behavior_actions,
    behavior_targets_frame,
    run_behavior_controls,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class BehaviorEcologyTests(unittest.TestCase):
    def test_ctc_holdout_targets_are_split_and_income_columns_exist(self):
        targets = behavior_targets_frame(target_scope="aggregate", evaluation_split=BEHAVIOR_CTC_HOLDOUT_SPLIT)
        self.assertEqual(targets.shape[0], 5)
        self.assertEqual({"ctc_2021_monthly_child_credit_style"}, set(targets["scenario_id"].astype(str)))
        self.assertIn("low_minus_high_income_debt_repayment_share", set(targets["prediction_column"].astype(str)))
        self.assertIn("high_minus_low_income_liquid_saving_share", set(targets["prediction_column"].astype(str)))

        type_cells, _ = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=2022)
        actions = run_behavior_controls(
            [scenario for scenario in BEHAVIOR_SCENARIOS if scenario.scenario_id == "ctc_2021_monthly_child_credit_style"],
            type_cells,
        )
        aggregates = aggregate_behavior_actions(actions)
        self.assertIn("low_minus_high_income_debt_repayment_share", aggregates.columns)
        self.assertIn("high_minus_low_income_liquid_saving_share", aggregates.columns)

    def test_behavior_ecology_replays_existing_policy_schedules_on_ctc_holdout(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_records = root / "policy_records.json"
            output_dir = root / "ctc"
            _write_fixture_policy_records(policy_records)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.behavior_ecology",
                    "--mode",
                    "replay",
                    "--arms",
                    "policy",
                    "--scenario-ids",
                    "ctc_2021_monthly_child_credit_style",
                    "--policy-raw-records-json",
                    str(policy_records),
                    "--max-live-calls",
                    "0",
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
            scores = pd.read_csv(output_dir / "ecology_target_scores.csv")
            actions = pd.read_csv(output_dir / "ecology_actions.csv")

            self.assertEqual(manifest["live_call_count"], 0)
            self.assertEqual(manifest["policy_raw_records_json"], str(policy_records))
            self.assertIn("Confirmatory behavior-holdout replay", manifest["claim_scope"])
            self.assertIn("behavior_holdout_ctc_v1", manifest["claim_scope"])
            self.assertEqual({BEHAVIOR_CTC_HOLDOUT_SPLIT}, set(scores["evaluation_split"].astype(str)))
            self.assertIn("policy_codex_cli_gpt-5.5", set(actions["source"].astype(str)))


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
