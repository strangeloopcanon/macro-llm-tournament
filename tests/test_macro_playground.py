import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.demand_economy import DemandScenario, build_fixture_demand_households, fixture_belief_payload
from macro_llm_tournament.llm_common import LLMUnavailable
from macro_llm_tournament.macro_playground import (
    FIRM_REACTION_PROMPT_VERSION,
    POLICY_NARRATIVE_PROMPT_VERSION,
    SCENARIO_SPEC_VERSION,
    canonical_sha256,
    load_macro_playground_spec,
    normalize_firm_reaction_payload,
    normalize_household_belief_payload,
    normalize_macro_playground_spec,
    normalize_policy_narrative_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SPEC = REPO_ROOT / "configs" / "macro_playground_fixture_spec.json"


class MacroPlaygroundTests(unittest.TestCase):
    def test_macro_playground_spec_normalizes_and_hashes_canonically(self):
        spec = load_macro_playground_spec(FIXTURE_SPEC)
        first_hash = canonical_sha256(spec)
        second_hash = canonical_sha256(json.loads(json.dumps(spec, sort_keys=True)))

        self.assertEqual(spec["scenario_spec_version"], SCENARIO_SPEC_VERSION)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(spec["branches"][0]["branch_id"], "baseline")

        duplicate = dict(spec)
        duplicate["branches"] = [spec["branches"][0], spec["branches"][0]]
        with self.assertRaises(ValueError):
            normalize_macro_playground_spec(duplicate)

        dated = dict(spec)
        dated["branches"] = [dict(spec["branches"][0], label="simulate 2008")]
        with self.assertRaises(ValueError):
            normalize_macro_playground_spec(dated)

    def test_macro_playground_actor_payloads_fail_closed(self):
        firm_payload = {
            "prompt_version": FIRM_REACTION_PROMPT_VERSION,
            "branch_id": "baseline",
            "period_id": "period_0",
            "planned_output_gap_pct": 0.0,
            "hiring_gap_pp": 0.0,
            "price_pressure_pp": 0.0,
            "credit_tightening_index": 0.0,
            "reason": "ok",
        }
        self.assertEqual(normalize_firm_reaction_payload(firm_payload, branch_id="baseline", period_id="period_0")["hiring_gap_pp"], 0.0)
        bad_firm = dict(firm_payload, hiring_gap_pp=9.0)
        with self.assertRaises(LLMUnavailable):
            normalize_firm_reaction_payload(bad_firm, branch_id="baseline", period_id="period_0")

        policy_payload = {
            "prompt_version": POLICY_NARRATIVE_PROMPT_VERSION,
            "branch_id": "baseline",
            "period_id": "period_0",
            "rate_rule_shift_pp": 0.0,
            "transfer_per_household": 0.0,
            "communication_confidence_shift": 0.0,
            "job_risk_attention_shift_pp": 0.0,
            "dispersion_multiplier": 1.0,
            "reason": "ok",
        }
        bad_policy = dict(policy_payload, desired_consumption=100.0)
        with self.assertRaises(LLMUnavailable):
            normalize_policy_narrative_payload(bad_policy, branch_id="baseline", period_id="period_0")

        households = build_fixture_demand_households(3)
        states = households.assign(
            source="macro_playground_fixture_gpt-5.5",
            variant="llm_belief",
            labor_income=households["annual_income"] / 4.0,
            baseline_consumption=households["baseline_consumption_annual"] / 4.0,
            liquid_buffer_months=1.0,
            job_loss_probability=households["baseline_job_loss_probability"],
            transfer_buffer_relief=0.0,
        ).to_dict(orient="records")
        payload = fixture_belief_payload(
            DemandScenario("baseline", "baseline"),
            {
                "period_id": "period_0",
                "period_index": 0,
                "scenario_id": "baseline",
                "output_gap_pct": 0.0,
                "employment_rate": 0.955,
                "inflation_rate": 2.0,
                "policy_rate": 3.0,
                "policy_rate_shock_pp": 0.0,
                "job_risk_shock_pp": 0.0,
                "transfer_per_household": 0.0,
                "aggregate_job_loss_belief": 5.0,
                "aggregate_confidence_index": 50.0,
                "aggregate_liquid_buffer_months": 3.0,
            },
            states,
        )
        payload["beliefs"][0]["desired_consumption"] = 100.0
        with self.assertRaises(LLMUnavailable):
            normalize_household_belief_payload(states, {"payload": payload})

    def test_macro_playground_fixture_cli_writes_ready_report(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
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
            report = (root / "macro_playground_report.md").read_text(encoding="utf-8")
            qa = pd.read_csv(root / "macro_playground_qa_scorecard.csv")
            periods = pd.read_csv(root / "macro_playground_periods.csv")
            accounting = pd.read_csv(root / "macro_playground_accounting.csv")
            actor_payloads = (root / "macro_playground_actor_payloads.jsonl").read_text(encoding="utf-8")

            self.assertEqual(manifest["verdict"], "macro_playground_fixture_ready")
            self.assertTrue(manifest["passed"])
            self.assertTrue(qa["passed"].astype(bool).all())
            self.assertIn("internal engine/playability result", report)
            self.assertIn("macro_playground_fixture_ready", report)
            self.assertEqual(set(periods["branch_id"]), {"baseline", "transfer_shock", "rate_hike", "job_risk_shock", "belief_feedback"})
            self.assertLessEqual(float(accounting["abs_residual"].max()), 1e-6)
            self.assertIn("household_beliefs", actor_payloads)
            self.assertIn("firm", actor_payloads)
            self.assertIn("policy_narrative", actor_payloads)
            self.assertIn("critic", actor_payloads)

    def test_macro_playground_replay_reproduces_fixture_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            replay_dir = root / "replay"
            fixture = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--output-dir",
                    str(fixture_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(fixture.returncode, 0, fixture.stderr)

            replay = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "replay",
                    "--max-live-calls",
                    "0",
                    "--replay-records-json",
                    str(fixture_dir / "macro_playground_actor_payloads.jsonl"),
                    "--output-dir",
                    str(replay_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)

            fixture_periods = pd.read_csv(fixture_dir / "macro_playground_periods.csv").sort_values(["branch_id", "period_index"]).reset_index(drop=True)
            replay_periods = pd.read_csv(replay_dir / "macro_playground_periods.csv").sort_values(["branch_id", "period_index"]).reset_index(drop=True)
            fixture_periods = fixture_periods.drop(columns=["source"])
            replay_periods = replay_periods.drop(columns=["source"])
            pd.testing.assert_frame_equal(fixture_periods, replay_periods)

    def test_macro_playground_replay_rejects_tampered_actor_payload(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            tampered_path = root / "tampered_actor_payloads.jsonl"
            fixture = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--output-dir",
                    str(fixture_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(fixture.returncode, 0, fixture.stderr)
            rows = [
                json.loads(line)
                for line in (fixture_dir / "macro_playground_actor_payloads.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            first_firm = next(row for row in rows if row["actor_role"] == "firm")
            first_firm["payload"]["hiring_gap_pp"] = 9.0
            first_firm["payload_sha256"] = canonical_sha256(first_firm["payload"])
            tampered_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            replay = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "replay",
                    "--max-live-calls",
                    "0",
                    "--replay-records-json",
                    str(tampered_path),
                    "--output-dir",
                    str(root / "tampered_replay"),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(replay.returncode, 0)
            self.assertIn("hiring_gap_pp", replay.stderr)

    def test_macro_playground_rejects_reusing_output_dir_for_different_spec(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
            second_spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
            second_spec["run_id"] = "fixture_playable_macro_v0_2_changed"
            first_spec_path = root / "first_spec.json"
            second_spec_path = root / "second_spec.json"
            output_dir = root / "shared_output"
            first_spec_path.write_text(json.dumps(first_spec), encoding="utf-8")
            second_spec_path.write_text(json.dumps(second_spec), encoding="utf-8")

            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(first_spec_path),
                    "--mode",
                    "fixture",
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
            self.assertEqual(first.returncode, 0, first.stderr)

            second = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(second_spec_path),
                    "--mode",
                    "fixture",
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
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("different scenario spec", second.stderr)

    def test_macro_playground_live_mode_writes_blocked_manifest(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_playground",
                    "--spec",
                    str(FIXTURE_SPEC),
                    "--mode",
                    "live",
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

            self.assertEqual(result.returncode, 2)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["verdict"], "macro_playground_live_blocked")
            self.assertFalse(manifest["passed"])


if __name__ == "__main__":
    unittest.main()
