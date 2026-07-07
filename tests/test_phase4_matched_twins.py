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


if __name__ == "__main__":
    unittest.main()
