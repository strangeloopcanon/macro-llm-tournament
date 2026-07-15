from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from macro_llm_tournament.ecology_households import (
    LiveCallBudget,
    household_request_identity,
)
from macro_llm_tournament.ecology_inputs import (
    ORIGIN_SNAPSHOT_SCHEMA_VERSION,
    _canonical_sha256,
    load_origin_information,
)
from macro_llm_tournament.frozen_vintage_bundle import (
    BUNDLE_SCHEMA_VERSION,
    HISTORY_COLUMNS,
    ORIGIN_COLUMNS,
    _bundle_sha256,
    _file_sha256,
    _write_canonical_csv,
)
from macro_llm_tournament.ecology_provider import CodexJSONClient, ProviderUnavailable
from macro_llm_tournament.ecology_provider import CODEX_TOOL_ISOLATION_VERSION


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

    def test_directory_origin_loader_does_not_require_or_open_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origins = [{"origin_month": "2026-01-01", "as_of_date": "2026-01-15"}]
            history = [
                {
                    "origin_month": "2026-01-01",
                    "as_of_date": "2026-01-15",
                    "series_id": "PCE",
                    "series_role": "target_origin_history",
                    "observation_date": "2025-12-01",
                    "value": 100.0,
                    "realtime_start": "2026-01-15",
                    "realtime_end": "2026-01-15",
                }
            ]
            _write_canonical_csv(root / "origins.csv", ORIGIN_COLUMNS, origins)
            _write_canonical_csv(root / "history.csv", HISTORY_COLUMNS, history)
            manifest = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "origins": origins,
                "payload_sha256": {
                    "origins.csv": _file_sha256(root / "origins.csv"),
                    "history.csv": _file_sha256(root / "history.csv"),
                    "targets.csv": "target-file-deliberately-absent",
                },
            }
            manifest["bundle_sha256"] = _bundle_sha256(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            information, digest = load_origin_information(root, "2026-01-01")
            self.assertEqual(information["as_of_date"], "2026-01-15")
            self.assertEqual(information["origin_visible_macro_context"]["PCE"]["value"], 100.0)
            self.assertEqual(digest, manifest["bundle_sha256"])

    def test_request_identity_binds_model_prompt_and_card(self) -> None:
        card = {
            "prompt_version": "household_ecology_monthly_v7",
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

    def test_replay_rejects_cached_request_mismatch_without_deleting_evidence(self) -> None:
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
                json.dumps({
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "request_sha256": "different-request",
                    "payload": {},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProviderUnavailable, "request identity or execution context mismatch"):
                client.json_call("prompt", "response", instructions="instructions")
            self.assertTrue(path.exists())

    def test_live_call_journal_records_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            budget = LiveCallBudget(1, Path(directory))
            budget.reserve("request")
            budget.complete("request", response={"ok": True})
            journals = list(Path(directory).glob("attempt_0001_*.json"))
            self.assertEqual(len(journals), 1)
            record = json.loads(journals[0].read_text())
            self.assertEqual(record["status"], "accepted")
            self.assertIsNotNone(record["response_sha256"])
            with self.assertRaisesRegex(ValueError, "cap reached"):
                budget.reserve("second")

    def test_live_codex_call_disables_local_tools_in_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / "source-codex-home"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text("{}", encoding="utf-8")
            client = CodexJSONClient(
                model="gpt-5.5",
                cache_dir=root / "cache",
                mode="live",
                max_live_calls=1,
                execution_cwd=root,
            )
            seen: dict[str, object] = {}

            def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
                seen["command"] = command
                seen["cwd"] = kwargs["cwd"]
                seen["env"] = kwargs["env"]
                isolated_home = Path(kwargs["env"]["CODEX_HOME"])
                seen["isolated_home_parent"] = isolated_home.parent
                seen["isolated_agents_exists"] = (isolated_home / "AGENTS.md").exists()
                seen["isolated_auth_is_symlink"] = (isolated_home / "auth.json").is_symlink()
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text('{"ok": true}', encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with mock.patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}), mock.patch(
                "macro_llm_tournament.ecology_provider.shutil.which",
                return_value="/usr/bin/codex",
            ), mock.patch(
                "macro_llm_tournament.ecology_provider.subprocess.run", side_effect=fake_run
            ):
                result = client.json_call("prompt", "request", instructions="instructions")

            command = seen["command"]
            assert isinstance(command, list)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("shell_tool", command)
            self.assertIn("plugins", command)
            self.assertIn("skill_mcp_dependency_install", command)
            self.assertIn('web_search="disabled"', command)
            self.assertNotEqual(Path(seen["cwd"]), root.resolve())
            self.assertEqual(result["tool_isolation_version"], CODEX_TOOL_ISOLATION_VERSION)
            self.assertEqual(seen["isolated_home_parent"], Path(seen["cwd"]))
            self.assertFalse(seen["isolated_agents_exists"])
            self.assertTrue(seen["isolated_auth_is_symlink"])


if __name__ == "__main__":
    unittest.main()
