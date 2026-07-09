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
    normalize_spec,
    parse_args,
    run_tournament,
    select_winner,
)


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
    manifest = {
        "status": "complete",
        "bundle_sha256": spec["shared"]["bundle_sha256"],
        "normalized_spec_sha256": child_spec_sha,
        "household_provenance": {"raw_input_file_sha256": household_hash},
        "max_accounting_abs_residual": 0.0,
        "macro_scores": {"llm": score, "adaptive": adaptive},
        "llm_minus_adaptive": score - adaptive,
        "live_call_count": 2 if mode == "live" else 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "candidate": "llm",
                "family": "demand",
                "direction_accuracy": 0.6,
                "family_mean_squared_scaled_error": score**2,
            },
            {
                "candidate": "adaptive",
                "family": "demand",
                "direction_accuracy": 0.9,
                "family_mean_squared_scaled_error": adaptive**2,
            },
        ]
    ).to_csv(output_dir / "family_scores.csv", index=False)
    pd.DataFrame([{"candidate": "llm", "origin_month": "2026-01-01", "macro_score": score}]).to_csv(
        output_dir / "origin_scores.csv", index=False
    )


class DynamicMacroTournamentTests(unittest.TestCase):
    def test_normalized_spec_is_stable_and_reserves_future_origin(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = write_inputs(Path(temp_dir))
            first = normalize_spec(json.loads(path.read_text()), spec_path=path)
            second = normalize_spec(json.loads(path.read_text()), spec_path=path)
            self.assertEqual(first, second)
            self.assertEqual(first["reserved_confirmatory"]["origin_month"], "2026-03-01")
            self.assertEqual(first["maximum_authorized_live_calls"], 12)

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


if __name__ == "__main__":
    unittest.main()
