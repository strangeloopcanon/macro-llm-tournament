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
    catalog_sha256,
    load_performance_target_catalog,
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
