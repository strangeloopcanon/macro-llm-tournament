from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from macro_llm_tournament.ecology_households import (
    LiveCallBudget,
    household_request_identity,
)
from macro_llm_tournament.ecology_inputs import (
    ORIGIN_SNAPSHOT_SCHEMA_VERSION,
    _canonical_sha256,
    load_origin_information,
)
from macro_llm_tournament.ecology_provider import CodexJSONClient, ProviderUnavailable


class EcologyIOTests(unittest.TestCase):
    def test_origin_snapshot_is_hash_bound_and_month_bound(self) -> None:
        information = {
            "origin_month": "2026-07-01",
            "as_of_date": "2026-07-10",
            "origin_visible_macro_context": {"UNRATE": {"observation_date": "2026-06-01", "value": 4.2}},
            "origin_visible_macro_history": {"UNRATE": [{"observation_date": "2026-06-01", "value": 4.2}]},
            "public_events": [],
        }
        payload = {
            "schema_version": ORIGIN_SNAPSHOT_SCHEMA_VERSION,
            "origin_information": information,
            "source": "test",
            "source_requests": [],
        }
        payload["snapshot_sha256"] = _canonical_sha256(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "origin.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, digest = load_origin_information(path, "2026-07-01")
            self.assertEqual(loaded, information)
            self.assertEqual(digest, payload["snapshot_sha256"])
            with self.assertRaisesRegex(ValueError, "month mismatch"):
                load_origin_information(path, "2026-08-01")
            payload["origin_information"]["as_of_date"] = "2026-07-11"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_origin_information(path, "2026-07-01")

    def test_request_identity_binds_model_prompt_and_card(self) -> None:
        card = {
            "prompt_version": "household_ecology_monthly_v1",
            "household": {"household_id": "h1"},
            "public_information": {"origin_month": "2026-07-01"},
        }
        first = household_request_identity("codex_cli", "gpt-5.5", card)[2]
        other_model = household_request_identity("codex_cli", "gpt-5.4", card)[2]
        other_card = household_request_identity(
            "codex_cli", "gpt-5.5", card | {"public_information": {"origin_month": "2026-08-01"}}
        )[2]
        self.assertNotEqual(first, other_model)
        self.assertNotEqual(first, other_card)

    def test_replay_rejects_cached_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = CodexJSONClient(
                model="gpt-5.5",
                cache_dir=Path(directory),
                mode="replay",
                max_live_calls=0,
                execution_cwd=Path(directory),
            )
            path = client.cache_path("response")
            path.write_text(
                json.dumps({"provider": "codex_cli", "model": "gpt-5.4", "payload": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProviderUnavailable, "identity mismatch"):
                client.json_call("prompt", "response", instructions="instructions")

    def test_live_call_journal_records_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            budget = LiveCallBudget(1, Path(directory))
            budget.reserve("request")
            budget.complete("request", response={"ok": True})
            record = json.loads((Path(directory) / "attempt_0001.json").read_text())
            self.assertEqual(record["status"], "accepted")
            self.assertIsNotNone(record["response_sha256"])
            with self.assertRaisesRegex(ValueError, "cap reached"):
                budget.reserve("second")


if __name__ == "__main__":
    unittest.main()
