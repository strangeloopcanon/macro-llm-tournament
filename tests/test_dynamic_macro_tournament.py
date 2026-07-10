from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from macro_llm_tournament.dynamic_macro_tournament import (
    SCHEMA_VERSION,
    DynamicMacroTournamentError,
    _build_failed_candidate_cache_seed,
    _summarize_failed_candidate_attempt,
    normalize_spec,
    parse_args,
    run_tournament,
    select_winner,
)
from macro_llm_tournament.dynamic_macro_economy import score_macro


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(root: Path, *, include_reserved: bool = False) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    origins = [
        {"origin_month": "2026-01-01", "as_of_date": "2026-01-15"},
        {"origin_month": "2026-02-01", "as_of_date": "2026-02-15"},
    ]
    if include_reserved:
        origins.append({"origin_month": "2026-03-01", "as_of_date": "2026-03-15"})
    bundle_manifest = {
        "bundle_sha256": "fixture-bundle",
        "origins": origins,
        "target_observation_semantics": {"rule": "common_origin_observation_month"},
    }
    (bundle / "manifest.json").write_text(json.dumps(bundle_manifest), encoding="utf-8")
    households = root / "households.csv"
    households.write_text("type_id,value\nh,1\n", encoding="utf-8")
    spec = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "fixture_tournament",
        "shared": {
            "bundle_dir": str(bundle),
            "bundle_sha256": "fixture-bundle",
            "households_csv": str(households),
            "households_sha256": sha(households),
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "contamination_policy": "unavailable_at_cutoff",
            "score_origin_start": "2026-02-01",
            "score_origin_end": "2026-02-01",
            "bootstrap_replicates": 10,
        },
        "candidates": [
            {"candidate_id": "candidate_a", "behavior_policy_mode": "fixed_kernel", "max_live_calls": 6},
            {"candidate_id": "candidate_b", "behavior_policy_mode": "fixed_kernel", "max_live_calls": 6},
        ],
        "reserved_confirmatory": {"origin_month": "2026-03-01", "bundle_dir": "reserved"},
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def fake_child(spec, candidate, *, mode, output_dir):
    output_dir.mkdir(parents=True)
    score = 0.4 if candidate["candidate_id"] == "candidate_a" else 0.3
    adaptive = 0.1 if candidate["candidate_id"] == "candidate_a" else 0.5
    household_hash = spec["shared"]["households_sha256"]
    child_spec = {
        "feedback_mode": candidate["feedback_mode"],
        "feedback_gain": candidate["feedback_gain"],
        "policy_rate_smoothing": candidate["policy_rate_smoothing"],
        "policy_state_mode": candidate["policy_state_mode"],
        "policy_state_weight": candidate["policy_state_weight"],
        "behavior_policy_mode": candidate["behavior_policy_mode"],
        "belief_gains": {
            "global": candidate["belief_gain_global"],
            "inflation": candidate["belief_gain_inflation"],
            "income": candidate["belief_gain_income"],
            "unemployment": candidate["belief_gain_unemployment"],
            "confidence": 1.0,
        },
        "score_origin_contract": {
            "score_origin_start": spec["shared"]["score_origin_start"],
            "score_origin_end": spec["shared"]["score_origin_end"],
        },
        "replay_provenance": {
            "raw_records_sha256": None,
            "replay_prefix_period_count": 0,
        },
    }
    child_spec_sha = hashlib.sha256(
        json.dumps(
            child_spec,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (output_dir / "normalized_spec.json").write_text(
        json.dumps(child_spec), encoding="utf-8"
    )
    joined = pd.DataFrame(
        [
            {
                "candidate": "llm",
                "origin_month": "2026-02-01",
                "family": "demand",
                "target_name": "fixture_target",
                "scaled_squared_error": score**2,
                "absolute_scaled_error": score,
                "direction_correct": True,
            },
            {
                "candidate": "adaptive",
                "origin_month": "2026-02-01",
                "family": "demand",
                "target_name": "fixture_target",
                "scaled_squared_error": adaptive**2,
                "absolute_scaled_error": adaptive,
                "direction_correct": True,
            },
        ]
    )
    target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    joined.to_csv(output_dir / "joined_errors.csv", index=False)
    target_scores.to_csv(output_dir / "target_scores.csv", index=False)
    family_scores.to_csv(output_dir / "family_scores.csv", index=False)
    origin_scores.to_csv(output_dir / "origin_scores.csv", index=False)
    for name in (
        "households.csv",
        "prompt_cards.csv",
        "beliefs.csv",
        "decisions.csv",
        "periods.csv",
        "accounting.csv",
        "forecasts.csv",
    ):
        (output_dir / name).write_text("fixture\n", encoding="utf-8")
    (output_dir / "raw_records.json").write_text("[]\n", encoding="utf-8")
    (output_dir / "report.md").write_text("# Fixture\n", encoding="utf-8")
    output_contract = [
        "normalized_spec.json",
        "manifest.json",
        "households.csv",
        "prompt_cards.csv",
        "beliefs.csv",
        "decisions.csv",
        "periods.csv",
        "accounting.csv",
        "forecasts.csv",
        "joined_errors.csv",
        "target_scores.csv",
        "family_scores.csv",
        "origin_scores.csv",
        "raw_records.json",
        "report.md",
    ]
    manifest = {
        "status": "complete",
        "bundle_sha256": spec["shared"]["bundle_sha256"],
        "normalized_spec_sha256": child_spec_sha,
        "household_provenance": {"raw_input_file_sha256": household_hash},
        "max_accounting_abs_residual": 0.0,
        "macro_scores": macro_scores,
        "llm_minus_adaptive": macro_scores["llm"] - macro_scores["adaptive"],
        "live_call_count": 2 if mode == "live" else 0,
        "output_contract": output_contract,
        "outputs": {
            name: sha(output_dir / name)
            for name in output_contract
            if name != "manifest.json"
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class DynamicMacroTournamentTests(unittest.TestCase):
    def test_attempt_journal_is_authoritative_and_valid_cache_is_seeded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attempt = (
                root
                / "failed_attempts"
                / "candidate_a_attempt_1"
            )
            cache = attempt / ".cache" / "codex_cli"
            cache.mkdir(parents=True)
            valid = {
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "cache_hit": False,
                "response_created_utc": "2026-07-09T00:00:00+00:00",
                "payload": {"beliefs": [{"type_id": "h1"}, {"type_id": "h2"}]},
            }
            (cache / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
            invalid = {**valid, "payload": {"beliefs": [{"type_id": "h1"}, {"type_id": "h1"}]}}
            (cache / "duplicate.json").write_text(json.dumps(invalid), encoding="utf-8")
            journals = attempt / ".cache" / "live_attempts"
            journals.mkdir()
            for index in range(2):
                (journals / f"attempt_{index + 1:04d}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "dynamic_macro_live_attempt_v1",
                            "status": "failed" if index else "complete",
                            "provider": "codex_cli",
                            "model": "gpt-5.5",
                            "period_index": index + 1,
                            "started_at_utc": "2026-07-09T00:00:00+00:00",
                            "finished_at_utc": "2026-07-09T00:00:01+00:00",
                        }
                    ),
                    encoding="utf-8",
                )
            summary = _summarize_failed_candidate_attempt(
                root,
                candidate_id="candidate_a",
                attempt_number=1,
                archived=attempt,
            )
            self.assertEqual(summary["live_call_count"], 2)
            self.assertEqual(summary["live_call_count_basis"], "attempt_journal")
            seed = _build_failed_candidate_cache_seed(
                root, candidate_id="candidate_a"
            )
            self.assertIsNotNone(seed)
            self.assertEqual(
                [path.name for path in (seed / "codex_cli").glob("*.json")],
                ["valid.json"],
            )

    def test_normalized_spec_is_stable_and_reserves_future_origin(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = write_inputs(Path(temp_dir))
            first = normalize_spec(json.loads(path.read_text()), spec_path=path)
            second = normalize_spec(json.loads(path.read_text()), spec_path=path)
            self.assertEqual(first, second)
            self.assertEqual(first["reserved_confirmatory"]["origin_month"], "2026-03-01")
            self.assertEqual(first["maximum_authorized_live_calls"], 12)
            self.assertTrue(
                all(
                    candidate["policy_rate_smoothing"] == 0.0
                    and candidate["policy_state_mode"] == "recursive"
                    and candidate["policy_state_weight"] == 1.0
                    for candidate in first["candidates"]
                )
            )

    def test_reserved_origin_cannot_enter_development_bundle(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = write_inputs(Path(temp_dir), include_reserved=True)
            with self.assertRaisesRegex(DynamicMacroTournamentError, "Reserved confirmatory origin"):
                normalize_spec(json.loads(path.read_text()), spec_path=path)

    def test_winner_uses_llm_score_and_ignores_adaptive_result(self) -> None:
        scores = pd.DataFrame(
            [
                {"candidate_id": "a", "status": "complete", "llm_macro_score": 0.4, "adaptive_macro_score": 0.1, "llm_direction_accuracy": 0.9, "mechanism_complexity": 0},
                {"candidate_id": "b", "status": "complete", "llm_macro_score": 0.3, "adaptive_macro_score": 0.8, "llm_direction_accuracy": 0.5, "mechanism_complexity": 1},
            ]
        )
        self.assertEqual(select_winner(scores)["candidate_id"], "b")

    def test_fixture_tournament_writes_contract_and_selects_best_llm(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec = write_inputs(root)
            args = parse_args(["--spec", str(spec), "--mode", "fixture", "--max-live-calls", "0", "--output-dir", str(root / "out")])
            with patch("macro_llm_tournament.dynamic_macro_tournament._run_candidate", side_effect=fake_child):
                manifest = run_tournament(args)
            self.assertEqual(manifest["winner_candidate_id"], "candidate_b")
            self.assertEqual(json.loads((root / "out" / "winner_manifest.json").read_text())["candidate_id"], "candidate_b")
            self.assertTrue((root / "out" / "report.md").is_file())

    def test_accounting_failure_disqualifies_candidate(self) -> None:
        def bad_first(spec, candidate, *, mode, output_dir):
            fake_child(spec, candidate, mode=mode, output_dir=output_dir)
            if candidate["candidate_id"] == "candidate_b":
                path = output_dir / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["max_accounting_abs_residual"] = 1.0
                path.write_text(json.dumps(manifest), encoding="utf-8")

        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec = write_inputs(root)
            args = parse_args(["--spec", str(spec), "--mode", "fixture", "--max-live-calls", "0", "--output-dir", str(root / "out")])
            with patch("macro_llm_tournament.dynamic_macro_tournament._run_candidate", side_effect=bad_first):
                manifest = run_tournament(args)
            self.assertEqual(manifest["winner_candidate_id"], "candidate_a")
            scores = pd.read_csv(root / "out" / "candidate_scores.csv").set_index("candidate_id")
            self.assertEqual(scores.loc["candidate_b", "status"], "disqualified")

    def test_live_call_cap_is_checked_before_candidates_run(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec = write_inputs(root)
            args = parse_args(["--spec", str(spec), "--mode", "live", "--max-live-calls", "11", "--output-dir", str(root / "out")])
            with self.assertRaisesRegex(DynamicMacroTournamentError, "must be at least 12"):
                run_tournament(args)

    def test_live_candidate_failure_aborts_without_selecting_from_partial_field(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec = write_inputs(root)
            args = parse_args(
                [
                    "--spec",
                    str(spec),
                    "--mode",
                    "live",
                    "--max-live-calls",
                    "12",
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_tournament._run_candidate",
                side_effect=DynamicMacroTournamentError("provider failed after calls"),
            ):
                with self.assertRaisesRegex(
                    DynamicMacroTournamentError, "Live tournament aborted"
                ):
                    run_tournament(args)
            self.assertFalse((root / "out" / "winner_manifest.json").exists())

    def test_live_resume_reuses_complete_children_and_counts_failed_attempts(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec_path = write_inputs(root)
            raw_spec = json.loads(spec_path.read_text())
            normalized = normalize_spec(raw_spec, spec_path=spec_path)
            output = root / "out"
            output.mkdir()
            (output / "normalized_spec.json").write_text(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_a = next(
                row for row in normalized["candidates"] if row["candidate_id"] == "candidate_a"
            )
            fake_child(
                normalized,
                candidate_a,
                mode="live",
                output_dir=output / "candidates" / "candidate_a",
            )
            partial = output / "candidates" / "candidate_b" / ".cache" / "codex_cli"
            partial.mkdir(parents=True)
            (partial / "bad.json").write_text(
                json.dumps(
                    {
                        "provider": "codex_cli",
                        "model": "gpt-5.5",
                        "cache_hit": False,
                        "response_created_utc": "2026-07-09T00:00:00+00:00",
                        "payload": {"beliefs": []},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "live",
                    "--max-live-calls",
                    "12",
                    "--output-dir",
                    str(output),
                    "--resume",
                ]
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_tournament._run_candidate",
                side_effect=fake_child,
            ) as run_child:
                manifest = run_tournament(args)
            self.assertEqual(run_child.call_count, 1)
            self.assertEqual(manifest["failed_live_call_count"], 1)
            self.assertEqual(manifest["successful_live_call_count"], 4)
            self.assertEqual(manifest["live_call_count"], 5)
            scores = pd.read_csv(output / "candidate_scores.csv").set_index("candidate_id")
            self.assertTrue(bool(scores.loc["candidate_a", "resumed_existing_result"]))
            self.assertEqual(scores.loc["candidate_b", "prior_failed_live_call_count"], 1)
            self.assertTrue((output / "failed_attempts" / "candidate_b_attempt_1").is_dir())

    def test_live_resume_rejects_tampered_complete_child(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec_path = write_inputs(root)
            normalized = normalize_spec(
                json.loads(spec_path.read_text()), spec_path=spec_path
            )
            output = root / "out"
            output.mkdir()
            (output / "normalized_spec.json").write_text(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_a = next(
                row
                for row in normalized["candidates"]
                if row["candidate_id"] == "candidate_a"
            )
            child = output / "candidates" / "candidate_a"
            fake_child(normalized, candidate_a, mode="live", output_dir=child)
            family = pd.read_csv(child / "family_scores.csv")
            family.loc[family["candidate"].eq("llm"), "macro_score"] = 0.0
            family.to_csv(child / "family_scores.csv", index=False)
            args = parse_args(
                [
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "live",
                    "--max-live-calls",
                    "12",
                    "--output-dir",
                    str(output),
                    "--resume",
                ]
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_tournament._run_candidate"
            ) as run_child:
                with self.assertRaisesRegex(
                    DynamicMacroTournamentError,
                    "Child output hash mismatch",
                ):
                    run_tournament(args)
            run_child.assert_not_called()
            self.assertFalse((output / "winner_manifest.json").exists())

    def test_live_multi_resume_fails_before_exceeding_candidate_cap(self) -> None:
        def write_live_calls(directory: Path, count: int) -> None:
            cache = directory / ".cache" / "codex_cli"
            cache.mkdir(parents=True)
            for index in range(count):
                (cache / f"call_{index}.json").write_text(
                    json.dumps(
                        {
                            "provider": "codex_cli",
                            "model": "gpt-5.5",
                            "cache_hit": False,
                            "response_created_utc": f"2026-07-09T00:00:0{index}+00:00",
                        }
                    ),
                    encoding="utf-8",
                )

        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            spec_path = write_inputs(root)
            normalized = normalize_spec(
                json.loads(spec_path.read_text()), spec_path=spec_path
            )
            output = root / "out"
            output.mkdir()
            (output / "normalized_spec.json").write_text(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_live_calls(
                output / "failed_attempts" / "candidate_a_attempt_1", 2
            )
            write_live_calls(
                output / "failed_attempts" / "candidate_a_attempt_2", 2
            )
            write_live_calls(output / "candidates" / "candidate_a", 2)
            args = parse_args(
                [
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "live",
                    "--max-live-calls",
                    "12",
                    "--output-dir",
                    str(output),
                    "--resume",
                ]
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_tournament._run_candidate"
            ) as run_child:
                with self.assertRaisesRegex(
                    DynamicMacroTournamentError, "exhausted its cumulative live-call cap"
                ):
                    run_tournament(args)
            run_child.assert_not_called()
            self.assertTrue(
                (output / "failed_attempts" / "candidate_a_attempt_3").is_dir()
            )


if __name__ == "__main__":
    unittest.main()
