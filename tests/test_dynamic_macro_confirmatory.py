from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from macro_llm_tournament import dynamic_macro_confirmatory as confirmation
from macro_llm_tournament.dynamic_macro_economy import OUTPUT_FILES, score_macro
from macro_llm_tournament.frozen_vintage_bundle import FrozenVintageBundleError
from macro_llm_tournament.source_provenance import build_source_contract


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def locked(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def build_lock(root: Path, *, bundle: bool) -> Path:
    source_root = root / "src" / "macro_llm_tournament"
    source_root.mkdir(parents=True)
    for name, contents in (("dynamic_macro_economy.py", "economy\n"), ("dynamic_macro_confirmatory.py", "confirmatory\n")):
        (source_root / name).write_text(contents, encoding="utf-8")
    source_contract_path = root / "configs" / "source_contract.json"
    write_json(source_contract_path, build_source_contract(root))
    config = root / "configs" / "development.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}", encoding="utf-8")
    household = root / "work" / "households.csv"
    household.parent.mkdir(parents=True)
    household.write_text("type_id\nh\n", encoding="utf-8")
    profile = root / "configs" / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    replay = root / "work" / "replay.json"
    write_json(replay, [{"cache_identity": {"period_index": index}} for index in range(5)])
    bundle_manifest = root / "work" / "development_bundle.json"
    write_json(bundle_manifest, {"bundle_sha256": "a" * 64})
    candidate = {
        "candidate_id": "winner", "behavior_policy_mode": "empirical_bridge", "empirical_bridge_json": None,
        "replay_prefix_raw_records_json": None, "replay_prefix_period_count": 5,
        "hybrid_state_weight": 1.0, "feedback_mode": "closed_loop", "feedback_gain": 1.0,
        "policy_rate_smoothing": 0.85, "policy_state_mode": "origin_visible", "policy_state_weight": 1.0,
        "belief_gain_global": 3.0, "belief_gain_inflation": 1.5, "belief_gain_income": 0.5,
        "belief_gain_unemployment": 1.0, "household_flow_anchor": "origin_saving_rate",
    }
    candidate["empirical_bridge_json"] = locked(root, profile)
    candidate["replay_prefix_raw_records_json"] = locked(root, replay)
    child_spec = {"household_provenance": {"raw_input_file_sha256": sha(household)}, "behavior_policy_content_sha256": "profile-content"}
    child_manifest = {"bundle_sha256": "a" * 64, "normalized_spec_sha256": canonical_sha(child_spec), "behavior_policy_content_sha256": "profile-content"}
    candidate_spec = root / "outputs" / "candidate_spec.json"
    candidate_manifest_path = root / "outputs" / "candidate_manifest.json"
    write_json(candidate_spec, child_spec)
    write_json(candidate_manifest_path, child_manifest)
    winner = {"candidate_id": "winner", "candidate": candidate, "child_manifest_sha256": sha(candidate_manifest_path), "child_normalized_spec_sha256": canonical_sha(child_spec), "development_result": {"llm_macro_score": 1.0, "adaptive_macro_score": 2.0}}
    winner_path = root / "outputs" / "winner.json"
    write_json(winner_path, winner)
    tournament = root / "outputs" / "tournament.json"
    write_json(tournament, {"status": "complete", "winner_candidate_id": "winner"})
    if bundle:
        future = root / "work" / "dynamic_macro" / "frozen_2026_01_2026_06_common_month_v1"
        future.mkdir(parents=True)
        origins = [(f"2026-{month:02d}-01", f"2026-{month:02d}-15") for month in range(1, 7)]
        for name, fields, rows in (
            ("origins.csv", ["origin_month", "as_of_date"], [dict(zip(("origin_month", "as_of_date"), row)) for row in origins]),
            ("targets.csv", ["origin_month", "target_observation_date", "first_release_as_of_date", "release_detection_method", "target_name"], [{"origin_month": month, "target_observation_date": month, "first_release_as_of_date": "2026-07-01", "release_detection_method": "vintage_dates", "target_name": f"target_{index}"} for month, _ in origins for index in range(10)]),
        ):
            with (future / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        write_json(future / "manifest.json", {"schema_version": "frozen_rolling_origin_vintage_bundle_v4", "mode": "fred", "bundle_sha256": "b" * 64, "origins": [{"origin_month": month} for month, _ in origins], "target_observation_semantics": {"rule": "common_origin_observation_month"}, "payload_sha256": {"origins.csv": sha(future / "origins.csv"), "targets.csv": sha(future / "targets.csv")}})
    lock = {
        "schema_version": confirmation.SCHEMA_VERSION, "claim_scope": "future_confirmatory_only",
        "development": {"winner_candidate_id": "winner", "bundle_sha256": "a" * 64, "candidate_normalized_spec_sha256": canonical_sha(child_spec), "winner_scores": {"llm_macro_score": 1.0, "adaptive_macro_score": 2.0}, "spec": locked(root, config), "manifest": locked(root, tournament), "winner_manifest": locked(root, winner_path), "candidate_manifest": locked(root, candidate_manifest_path), "candidate_spec": locked(root, candidate_spec), "candidate": candidate},
        "inputs": {"bundle_manifest": locked(root, bundle_manifest), "households": locked(root, household), "profile": locked(root, profile), "replay_records": locked(root, replay)},
        "execution_source": locked(root, source_contract_path),
        "confirmation": {"bundle_dir": "work/dynamic_macro/frozen_2026_01_2026_06_common_month_v1", "origins": list(confirmation.ORIGINS), "target_rows_per_origin": 10, "score_origin_month": "2026-06-01", "replay_prefix_period_count": 5, "provider": "codex_cli", "model": "gpt-5.5", "semantic_retry_limit": 2, "max_live_calls": 3, "receipt_path": "work/receipt.json"},
    }
    path = root / "configs" / "lock.json"
    write_json(path, lock)
    return path


def write_child_output(root: Path, output: Path) -> None:
    output.mkdir(parents=True)
    spec = {"mode": "replay_live", "provider": "codex_cli", "model": "gpt-5.5", "behavior_policy_content_sha256": "profile-content", "household_provenance": {"raw_input_file_sha256": sha(root / "work" / "households.csv")}, "replay_provenance": {"raw_records_sha256": sha(root / "work" / "replay.json"), "replay_prefix_period_count": 5}}
    write_json(output / "normalized_spec.json", spec)
    pd.DataFrame([{"candidate": "llm", "residual": 0.0, "abs_residual": 0.0, "passed": True}]).to_csv(output / "accounting.csv", index=False)
    write_json(
        output / "raw_records.json",
        [
            {
                "candidate": "llm_belief",
                "cache_identity": {"period_index": index},
                "cache_hit": index != 5,
            }
            for index in range(6)
        ]
        + [
            {"candidate": "adaptive", "cache_hit": True}
            for _ in range(6)
        ],
    )
    write_json(
        output / "live_attempts.json",
        [
            {
                "schema_version": "dynamic_macro_live_attempt_v1",
                "attempt_number": 1,
                "attempt_id": "live_attempt_0001",
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "period_index": 5,
                "status": "accepted",
                "started_at_utc": "2026-07-09T12:00:00+00:00",
                "finished_at_utc": "2026-07-09T12:00:01+00:00",
                "cache_file": "demand_belief_example.json",
                "cache_file_sha256": "c" * 64,
                "error_sha256": None,
                "journal_sha256": "d" * 64,
            }
        ],
    )
    joined = pd.DataFrame([
        {"candidate": candidate, "origin_month": "2026-06-01", "family": f"family_{index // 2}", "target_name": f"target_{index}", "scaled_squared_error": float(index + (0 if candidate == "llm" else 1)), "absolute_scaled_error": float(index + (0 if candidate == "llm" else 1)) ** 0.5, "direction_correct": index % 2 == 0}
        for candidate in ("llm", "adaptive")
        for index in range(10)
    ])
    joined.to_csv(output / "joined_errors.csv", index=False)
    target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    target_scores.to_csv(output / "target_scores.csv", index=False)
    family_scores.to_csv(output / "family_scores.csv", index=False)
    origin_scores.to_csv(output / "origin_scores.csv", index=False)
    for name in OUTPUT_FILES:
        path = output / name
        if not path.exists():
            if name.endswith(".csv"):
                path.write_text("placeholder\n", encoding="utf-8")
            elif name == "report.md":
                path.write_text("confirmatory fixture\n", encoding="utf-8")
            elif name not in {"manifest.json"}:
                write_json(path, [])
    source_contract = json.loads((root / "configs" / "source_contract.json").read_text(encoding="utf-8"))
    manifest = {"status": "complete", "bundle_sha256": "b" * 64, "normalized_spec_sha256": canonical_sha(spec), "execution_source": source_contract, "output_contract": list(OUTPUT_FILES), "score_origin_contract": {"scored_origins": ["2026-06-01"], "scored_origin_count": 1}, "scored_target_row_count": 10, "replayed_record_count": 5, "live_call_count": 1, "max_accounting_abs_residual": 0.0, "macro_scores": macro_scores, "llm_minus_adaptive": macro_scores["llm"] - macro_scores["adaptive"]}
    manifest["outputs"] = {name: sha(output / name) for name in OUTPUT_FILES if name != "manifest.json"}
    write_json(output / "manifest.json", manifest)


class DynamicMacroConfirmatoryTests(unittest.TestCase):
    def test_missing_bundle_fails_before_receipt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=False)
            with patch.object(confirmation, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "June bundle is missing"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", "outputs/run"]))
            self.assertFalse((root / "work" / "receipt.json").exists())

    def test_complete_run_receipts_once_and_scores_only_june(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            output = root / "outputs" / "run"
            manifest = json.loads(
                (root / "work" / "dynamic_macro" / "frozen_2026_01_2026_06_common_month_v1" / "manifest.json").read_text()
            )
            with patch.object(confirmation, "PROJECT_ROOT", root), patch.object(
                confirmation, "validate_frozen_vintage_bundle", return_value=manifest
            ), patch.object(
                confirmation,
                "_run_child",
                side_effect=lambda _lock, _candidate, path: write_child_output(root, path),
            ):
                receipt = confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", str(output)]))
                self.assertEqual(receipt["status"], "complete")
                self.assertEqual(receipt["replayed_record_count"], 5)
                self.assertEqual(receipt["scored_origin_month"], "2026-06-01")
                self.assertEqual(receipt["live_attempt_count"], 1)
                self.assertEqual(receipt["live_attempts_sha256"], sha(output / "live_attempts.json"))
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "surface is spent"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", str(root / "outputs" / "again")]))

    def test_source_drift_fails_before_receipt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            (root / "src" / "macro_llm_tournament" / "dynamic_macro_economy.py").write_text("drifted\n", encoding="utf-8")
            with patch.object(confirmation, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "source tree does not match"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", "outputs/run"]))
            self.assertFalse((root / "work" / "receipt.json").exists())

    def test_tampered_score_table_fails_receipt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            manifest = json.loads((root / "work" / "dynamic_macro" / "frozen_2026_01_2026_06_common_month_v1" / "manifest.json").read_text())
            def write_and_tamper(_lock, _candidate, path):
                write_child_output(root, path)
                score_path = path / "target_scores.csv"
                score_path.write_text(score_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with patch.object(confirmation, "PROJECT_ROOT", root), patch.object(confirmation, "validate_frozen_vintage_bundle", return_value=manifest), patch.object(confirmation, "_run_child", side_effect=write_and_tamper):
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "output hash mismatch: target_scores.csv"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", "outputs/run"]))
            self.assertEqual(json.loads((root / "work" / "receipt.json").read_text())["status"], "failed")

    def test_failed_receipt_blocks_rerun(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            manifest = json.loads(
                (root / "work" / "dynamic_macro" / "frozen_2026_01_2026_06_common_month_v1" / "manifest.json").read_text()
            )
            with patch.object(confirmation, "PROJECT_ROOT", root), patch.object(
                confirmation, "validate_frozen_vintage_bundle", return_value=manifest
            ), patch.object(
                confirmation,
                "_run_child",
                side_effect=confirmation.DynamicMacroConfirmatoryError("provider failed"),
            ):
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "provider failed"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", "outputs/run"]))
                self.assertEqual(json.loads((root / "work" / "receipt.json").read_text())["status"], "failed")
                with self.assertRaisesRegex(confirmation.DynamicMacroConfirmatoryError, "surface is spent"):
                    confirmation.run_confirmation(confirmation.parse_args(["--lock", str(lock), "--output-dir", "outputs/again"]))

    def test_invalid_complete_bundle_fails_before_receipt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            with patch.object(confirmation, "PROJECT_ROOT", root), patch.object(
                confirmation,
                "validate_frozen_vintage_bundle",
                side_effect=FrozenVintageBundleError("history.csv hash mismatch"),
            ):
                with self.assertRaisesRegex(
                    confirmation.DynamicMacroConfirmatoryError,
                    "June bundle is invalid",
                ):
                    confirmation.run_confirmation(
                        confirmation.parse_args(
                            ["--lock", str(lock), "--output-dir", "outputs/run"]
                        )
                    )
            self.assertFalse((root / "work" / "receipt.json").exists())

    def test_rejects_tampered_or_unreconciled_live_attempt_ledger(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = build_lock(root, bundle=True)
            output = root / "outputs" / "run"
            write_child_output(root, output)
            attempts_path = output / "live_attempts.json"
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            attempts.append(
                {
                    **attempts[0],
                    "attempt_number": 2,
                    "attempt_id": "live_attempt_0002",
                    "status": "failed",
                    "cache_file_sha256": None,
                    "error_sha256": "e" * 64,
                    "journal_sha256": "f" * 64,
                }
            )
            write_json(attempts_path, attempts)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["live_attempts.json"] = sha(attempts_path)
            write_json(manifest_path, manifest)
            bundle = {"bundle_sha256": "b" * 64}
            with patch.object(confirmation, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(
                    confirmation.DynamicMacroConfirmatoryError,
                    "does not reconcile",
                ):
                    confirmation._validate_output(
                        json.loads(lock.read_text(encoding="utf-8")), bundle, output
                    )


if __name__ == "__main__":
    unittest.main()
