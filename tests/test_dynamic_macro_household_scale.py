from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.dynamic_macro_household_scale import (
    CLAIM_SCOPE,
    ECONOMY_OUTPUT_FILES,
    DynamicMacroHouseholdScaleError,
    compare_runs,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(root: Path, households: int, llm_error: float, *, mechanism: str = "shared") -> None:
    root.mkdir()
    normalized_spec = {
        "provider": "codex_cli",
        "model": "gpt-5.5",
        "provider_reasoning_effort": "high",
        "execution_source_tree_sha256": "source-lock",
        "prompt_version": "prompt-lock",
        "behavior_policy_mode": mechanism,
        "feedback_mode": "closed_loop",
        "feedback_gain": 1.0,
        "policy_state_mode": "recursive",
        "household_count": households,
        "household_provenance": {"raw_input_file_sha256": f"households-{households}"},
    }
    write_json(root / "normalized_spec.json", normalized_spec)
    household_frame = pd.DataFrame(
        {
            "type_id": [f"h{i:03d}" for i in range(households)],
            "population_weight": [1.0 / households] * households,
        }
    )
    household_frame.to_csv(root / "households.csv", index=False)

    rows: list[dict[str, object]] = []
    for origin_index, origin in enumerate(("2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01")):
        for candidate, error in (("llm", llm_error), ("adaptive", 0.4)):
            rows.append(
                {
                    "candidate": candidate,
                    "origin_month": origin,
                    "as_of_date": origin.replace("-01", "-15"),
                    "target_observation_date": origin,
                    "target_name": "ALL",
                    "series_id": "ALL",
                    "family": "demand",
                    "economy_measure": "fixture",
                    "economy_transform": "level",
                    "transform": "level",
                    "default_scale": 1.0,
                    "target_value": 0.0,
                    "origin_visible_denominator_value": 0.0,
                    "prediction": error,
                    "scaled_squared_error": error**2,
                    "absolute_scaled_error": abs(error),
                    "direction_correct": True,
                }
            )
    pd.DataFrame(rows).to_csv(root / "joined_errors.csv", index=False)

    accounting = pd.DataFrame(
        [{"residual": 0.0, "abs_residual": 0.0, "passed": True}]
    )
    accounting.to_csv(root / "accounting.csv", index=False)
    household_ids = [f"h{i:03d}" for i in range(households)]
    llm_batches = [household_ids[index : index + 100] for index in range(0, households, 100)]
    records = []
    live_attempts = []
    attempt_number = 0
    for period in range(5):
        for batch_index, batch_ids in enumerate(llm_batches):
            attempt_number += 1
            payload = {"beliefs": [{"type_id": type_id} for type_id in batch_ids]}
            records.append(
                {
                    "candidate": "llm_belief",
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "period_id": f"period_{period}",
                    "period_index": period,
                    "batch_index": batch_index,
                    "batch_count": len(llm_batches),
                    "household_type_ids": batch_ids,
                    "household_type_ids_sha256": canonical_sha(batch_ids),
                    "prompt_payload_sha256": "a" * 64,
                    "response_sha256": canonical_sha(payload),
                    "cache_identity": {"period_index": period, "state_identity_sha256": "b" * 64},
                    "payload": payload,
                }
            )
            live_attempts.append(
                {
                    "schema_version": "dynamic_macro_live_attempt_v2",
                    "attempt_number": attempt_number,
                    "attempt_id": f"live_attempt_{attempt_number:04d}",
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "period_index": period,
                    "status": "accepted",
                    "started_at_utc": "2026-07-09T00:00:00+00:00",
                    "finished_at_utc": "2026-07-09T00:01:00+00:00",
                    "cache_file": f"cache_{period}_{batch_index}.json",
                    "cache_file_sha256": "c" * 64,
                    "error_sha256": None,
                    "batch_index": batch_index,
                    "batch_count": len(llm_batches),
                    "household_type_ids": batch_ids,
                    "household_type_ids_sha256": canonical_sha(batch_ids),
                    "prompt_payload_sha256": "a" * 64,
                    "provider_called": True,
                    "response_sha256": canonical_sha(payload),
                    "journal_sha256": "d" * 64,
                }
            )
        records.append(
            {
                "candidate": "adaptive",
                "provider": "deterministic",
                "model": "adaptive",
                "period_id": f"period_{period}",
                "period_index": period,
                "batch_index": 0,
                "batch_count": 1,
                "payload": {"beliefs": [{"type_id": type_id} for type_id in household_ids]},
            }
        )
    write_json(root / "raw_records.json", records)
    write_json(root / "live_attempts.json", live_attempts)

    for name in ECONOMY_OUTPUT_FILES:
        path = root / name
        if path.exists():
            continue
        if name.endswith(".csv"):
            path.write_text("fixture\n", encoding="utf-8")
        elif name.endswith(".json"):
            write_json(path, [])
        else:
            path.write_text("# fixture\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "provider": "codex_cli",
        "model": "gpt-5.5",
        "provider_reasoning_effort": "high",
        "execution_source_tree_sha256": "source-lock",
        "prompt_version": "prompt-lock",
        "household_count": households,
        "live_call_count": len(live_attempts),
        "cache_hit_count": 0,
        "replayed_record_count": 0,
        "semantic_retry_count": 0,
        "normalized_spec_sha256": canonical_sha(normalized_spec),
        "max_accounting_abs_residual": 0.0,
        "output_contract": list(ECONOMY_OUTPUT_FILES),
        "outputs": {name: sha(root / name) for name in ECONOMY_OUTPUT_FILES if name != "manifest.json"},
        "macro_scores": {"llm": 999.0, "adaptive": 999.0},
    }
    write_json(root / "manifest.json", manifest)


def write_inputs(root: Path, *, llm81: float = 0.500, llm200: float = 0.504) -> tuple[Path, Path, Path, Path, Path]:
    run81 = root / "corrected81"
    run200 = root / "corrected200"
    write_run(run81, 81, llm81)
    write_run(run200, 200, llm200)
    evidence = root / "historical.json"
    write_json(
        evidence,
        {
            "label": "uncorrected81_context_only",
            "role": "historical context",
            "household_count": 81,
            "eligible_for_promotion": False,
            "historical_uncorrected81_context": {
                "household_count": 81,
                "llm_macro_score": 0.5497889640842802,
                "adaptive_macro_score": 0.5486674359188668,
                "llm_direction_accuracy": 0.4916666666666667,
                "adaptive_direction_accuracy": 0.4916666666666667,
                "effective_weighted_sample_size": 55.34476682942994,
                "max_normalized_household_weight": 0.0472132319658739,
                "max_accounting_abs_residual": 7.275957614183426e-12,
                "live_call_count": 0,
                "semantic_retry_count": 0,
                "replayed_record_count": 5,
                "cache_hit_count": 0,
                "eligible_for_promotion": False,
            },
        },
    )
    spec = root / "comparison.json"
    write_json(
        spec,
        {
            "schema_version": "dynamic_macro_household_scale_comparison_v1",
            "claim_scope": CLAIM_SCOPE,
            "origins": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"],
            "score_origins": ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"],
            "historical_evidence": {"path": str(evidence), "sha256": sha(evidence)},
            "locks": {
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "provider_reasoning_effort": "high",
                "execution_source_tree_sha256": "source-lock",
                "prompt_version": "prompt-lock",
                "mechanism_fields": ["behavior_policy_mode", "feedback_mode", "feedback_gain", "policy_state_mode"],
                "mechanism_values": {
                    "behavior_policy_mode": "shared",
                    "feedback_mode": "closed_loop",
                    "feedback_gain": 1.0,
                    "policy_state_mode": "recursive",
                },
            },
            "runs": {
                "corrected81": {"household_count": 81, "households_sha256": "households-81", "accepted_live_calls": 5, "raw_record_batch_counts": {"llm_belief": 5, "adaptive": 5}},
                "corrected200": {"household_count": 200, "households_sha256": "households-200", "accepted_live_calls": 10, "raw_record_batch_counts": {"llm_belief": 10, "adaptive": 5}},
            },
        },
    )
    return spec, evidence, run81, run200, root / "comparison-output"


class DynamicMacroHouseholdScaleTests(unittest.TestCase):
    def test_recomputes_scores_and_prefers_200_within_one_percent(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            manifest = compare_runs(spec, evidence, run81, run200, output)
            self.assertEqual(manifest["winner"], "corrected200")
            comparison = pd.read_csv(output / "comparison.csv")
            scores = comparison.set_index("cohort")["llm_ALL_rmse_scaled"]
            self.assertAlmostEqual(float(scores["corrected81"]), 0.5)
            self.assertAlmostEqual(float(scores["corrected200"]), 0.504)
            self.assertEqual(len(comparison), 3)
            self.assertEqual(set(manifest["outputs"]), set(manifest["output_contract"]) - {"manifest.json"})
            self.assertEqual(json.loads((output / "promotion_receipt.json").read_text())["status"], "complete")
            self.assertIn("recomputed", (output / "report.md").read_text())

    def test_hash_tampering_fails_closed(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            (run200 / "report.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "output hash mismatch"):
                compare_runs(spec, evidence, run81, run200, output)
            self.assertFalse(output.exists())

    def test_june_rows_fail_closed_even_when_hashes_are_updated(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            path = run200 / "joined_errors.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "origin_month"] = "2026-06-01"
            frame.to_csv(path, index=False)
            manifest = json.loads((run200 / "manifest.json").read_text())
            manifest["outputs"]["joined_errors.csv"] = sha(path)
            write_json(run200 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "June"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_mechanism_drift_and_accounting_violation_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            normalized = json.loads((run200 / "normalized_spec.json").read_text())
            normalized["behavior_policy_mode"] = "changed"
            write_json(run200 / "normalized_spec.json", normalized)
            manifest = json.loads((run200 / "manifest.json").read_text())
            manifest["normalized_spec_sha256"] = canonical_sha(normalized)
            manifest["outputs"]["normalized_spec.json"] = sha(run200 / "normalized_spec.json")
            write_json(run200 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "mechanism"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_accounting_and_batch_contracts_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            accounting_path = run200 / "accounting.csv"
            accounting_path.write_text("residual,abs_residual,passed\n0.1,0.1,False\n", encoding="utf-8")
            manifest = json.loads((run200 / "manifest.json").read_text())
            manifest["outputs"]["accounting.csv"] = sha(accounting_path)
            write_json(run200 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "accounting"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            records_path = run200 / "raw_records.json"
            records = json.loads(records_path.read_text())
            records.pop()
            write_json(records_path, records)
            manifest = json.loads((run200 / "manifest.json").read_text())
            manifest["outputs"]["raw_records.json"] = sha(records_path)
            write_json(run200 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "batch"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_rejects_confirmatory_historical_evidence_and_stale_output(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            spec, evidence, run81, run200, output = write_inputs(root)
            bad_evidence = root / "bad-evidence.json"
            write_json(bad_evidence, {"label": "uncorrected81_context_only_confirmatory"})
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "confirmatory"):
                compare_runs(spec, bad_evidence, run81, run200, output)
            output.mkdir()
            (output / "old.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "stale"):
                compare_runs(spec, evidence, run81, run200, output)


if __name__ == "__main__":
    unittest.main()
