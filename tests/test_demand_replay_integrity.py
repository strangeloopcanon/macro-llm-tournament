import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from macro_llm_tournament.demand_economy import DemandEconomyClient, _raw_replay_record_map


class DemandReplayIntegrityTests(unittest.TestCase):
    def test_raw_replay_records_keep_model_identity_in_key(self):
        records = [
            {
                "provider": "codex_cli",
                "model": "model-a",
                "variant": "llm_belief",
                "scenario_id": "baseline",
                "period_index": 1,
                "payload": {"household_beliefs": []},
            },
            {
                "provider": "codex_cli",
                "model": "model-b",
                "variant": "llm_belief",
                "scenario_id": "baseline",
                "period_index": 1,
                "payload": {"household_beliefs": []},
            },
        ]

        mapped = _raw_replay_record_map(records)

        self.assertEqual(len(mapped), 2)
        self.assertIn(("codex_cli", "model-a", "llm_belief", "baseline", 1), mapped)
        self.assertIn(("codex_cli", "model-b", "llm_belief", "baseline", 1), mapped)

    def test_raw_replay_rejects_duplicate_identity(self):
        record = {
            "provider": "codex_cli",
            "model": "model-a",
            "variant": "llm_belief",
            "scenario_id": "baseline",
            "period_index": 1,
            "payload": {"household_beliefs": []},
        }

        with self.assertRaisesRegex(ValueError, "Duplicate raw replay record"):
            _raw_replay_record_map([record, dict(record)])

    def test_replay_and_live_sources_are_distinct(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            replay = DemandEconomyClient(
                "codex_cli",
                "gpt-5.5",
                root / "replay",
                mode="replay",
                variant="llm_belief",
            )
            live = DemandEconomyClient(
                "codex_cli",
                "gpt-5.5",
                root / "live",
                mode="live",
                variant="llm_belief",
                max_live_calls=1,
            )

        self.assertNotEqual(replay.source, live.source)
        self.assertIn("replay", replay.source)
        self.assertIn("live", live.source)


if __name__ == "__main__":
    unittest.main()
