from __future__ import annotations

import hashlib
import json
import math
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


def refresh_output_hash(run: Path, name: str) -> None:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][name] = sha(run / name)
    write_json(manifest_path, manifest)


FIXTURE_BUNDLE_SHA256 = "f" * 64
FIXTURE_FAMILIES = (
    "demand", "demand", "balance_sheet", "balance_sheet", "labor",
    "labor", "prices", "prices", "income_policy", "income_policy",
)


def fixture_targets() -> list[dict[str, object]]:
    targets = [
        {
            "target_name": f"target_{index}",
            "series_id": f"SERIES_{index}",
            "family": family,
            "transform": "level",
            "economy_measure": "next_policy_rate",
            "economy_transform": "level",
            "default_scale": 1.0,
        }
        for index, family in enumerate(FIXTURE_FAMILIES)
    ]
    targets[2].update(
        series_id="PSAVERT",
        transform="diff",
        economy_measure="saving_rate_pct",
        economy_transform="diff",
    )
    targets[3].update(
        series_id="REVOLSL",
        transform="pct_change",
        economy_measure="aggregate_debt",
        economy_transform="pct_change",
    )
    return targets


def fixture_bundle_targets() -> list[dict[str, object]]:
    return [
        {
            "origin_month": origin,
            "as_of_date": origin.replace("-01", "-15"),
            "target_observation_date": origin,
            "target_name": target["target_name"],
            "series_id": target["series_id"],
            "family": target["family"],
            "transform": target["transform"],
            "default_scale": target["default_scale"],
            "target_value": 0.0,
            "origin_visible_denominator_value": 0.0,
        }
        for origin in ("2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01")
        for target in fixture_targets()
    ]


def fixture_source_contract() -> dict[str, object]:
    files = {"src/macro_llm_tournament/fixture.py": "b" * 64}
    return {
        "schema_version": "macro_llm_source_contract_v1",
        "source_root": "src/macro_llm_tournament",
        "file_count": len(files),
        "files": files,
        "tree_sha256": canonical_sha(files),
    }


def write_run(root: Path, households: int, llm_error: float, *, mechanism: str = "shared") -> None:
    root.mkdir()
    targets = fixture_targets()
    source_contract = fixture_source_contract()
    source_tree_sha256 = str(source_contract["tree_sha256"])
    source_contract_sha256 = canonical_sha(source_contract)
    normalized_spec = {
        "provider": "codex_cli",
        "model": "gpt-5.5",
        "provider_reasoning_effort": "high",
        "execution_source_tree_sha256": source_tree_sha256,
        "source_contract_sha256": source_contract_sha256,
        "bundle_sha256": FIXTURE_BUNDLE_SHA256,
        "prompt_version": "prompt-lock",
        "behavior_policy_mode": mechanism,
        "feedback_mode": "closed_loop",
        "feedback_gain": 1.0,
        "policy_state_mode": "recursive",
        "household_count": households,
        "household_provenance": {"raw_input_file_sha256": f"households-{households}"},
        "target_mappings": targets,
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
    for origin in ("2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"):
        for candidate, error in (("llm", llm_error), ("adaptive", 0.4)):
            for target in targets:
                prediction = (
                    0.0
                    if target["series_id"] in {"PSAVERT", "REVOLSL"}
                    else error
                )
                rows.append({
                    "candidate": candidate,
                    "origin_month": origin,
                    "as_of_date": origin.replace("-01", "-15"),
                    "target_observation_date": origin,
                    **target,
                    "prediction": prediction,
                    "target_value": 0.0,
                    "origin_visible_denominator_value": 0.0,
                    "error": 999.0,
                    "scaled_error": 999.0,
                    "scaled_squared_error": 999.0,
                    "absolute_scaled_error": 999.0,
                    "direction_correct": False,
                })
    pd.DataFrame(rows).to_csv(root / "joined_errors.csv", index=False)
    pd.DataFrame(rows)[
        [
            "candidate", "origin_month", "as_of_date", "target_observation_date",
            "target_name", "series_id", "family", "prediction", "economy_measure",
            "economy_transform",
        ]
    ].to_csv(root / "forecasts.csv", index=False)

    household_ids = [f"h{i:03d}" for i in range(households)]
    weighted_consumption = float((pd.Series([1.0 / households] * households) * 4.0).sum())
    weighted_income = float((pd.Series([1.0 / households] * households) * 5.0).sum())
    llm_batches = [household_ids[index : index + 100] for index in range(0, households, 100)]
    records = []
    live_attempts = []
    prompt_cards = []
    decision_rows = []
    period_rows = []
    accounting_rows = []
    attempt_number = 0
    for period in range(5):
        for batch_index, batch_ids in enumerate(llm_batches):
            attempt_number += 1
            payload = {"beliefs": [{"type_id": type_id} for type_id in batch_ids]}
            prompt_payload = {"household_type_ids": batch_ids}
            cache_file = f"cache_{period}_{batch_index}.json"
            write_json(
                root / cache_file,
                {"provider": "codex_cli", "model": "gpt-5.5", "payload": payload},
            )
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
                    "prompt_payload_sha256": canonical_sha(prompt_payload),
                    "response_sha256": canonical_sha(payload),
                    "cache_hit": False,
                    "provider_called": True,
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
                    "cache_file": cache_file,
                    "cache_file_sha256": sha(root / cache_file),
                    "error_sha256": None,
                    "batch_index": batch_index,
                    "batch_count": len(llm_batches),
                    "household_type_ids": batch_ids,
                    "household_type_ids_sha256": canonical_sha(batch_ids),
                    "prompt_payload_sha256": canonical_sha(prompt_payload),
                    "provider_called": True,
                    "response_sha256": canonical_sha(payload),
                    "journal_sha256": "d" * 64,
                }
            )
            prompt_cards.append({
                "candidate": "llm",
                "period_index": period,
                "batch_index": batch_index,
                "batch_count": len(llm_batches),
                "household_type_ids": json.dumps(batch_ids),
                "household_type_ids_sha256": canonical_sha(batch_ids),
                "prompt_payload_sha256": canonical_sha(prompt_payload),
                "prompt_payload": json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")),
            })
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
        for candidate, policy_rate in (("llm", llm_error), ("adaptive", 0.4)):
            for type_id in household_ids:
                decision_rows.append({
                    "candidate": candidate,
                    "period_index": period,
                    "type_id": type_id,
                    "population_weight": 1.0 / households,
                    "liquid_assets_before": 10.0,
                    "labor_income": 5.0,
                    "transfer": 0.0,
                    "consumption": 4.0,
                    "debt_repayment": 1.0,
                    "liquid_assets_after": 10.0,
                    "debt_before": 2.0,
                    "debt_after": 1.0,
                })
                accounting_rows.extend([
                    {"candidate": candidate, "period_index": period, "unit": type_id, "identity": "household_cash_budget", "residual": 0.0, "abs_residual": 0.0, "passed": True},
                    {"candidate": candidate, "period_index": period, "unit": type_id, "identity": "household_debt_stock", "residual": 0.0, "abs_residual": 0.0, "passed": True},
                ])
            period_rows.append({
                "candidate": candidate,
                "period_index": period,
                "origin_month": f"2026-{period + 1:02d}-01",
                "aggregate_consumption": weighted_consumption,
                "aggregate_income": weighted_income,
                "aggregate_transfer": 0.0,
                "aggregate_saving": 0.0,
                "aggregate_debt_repayment": 1.0,
                "safe_asset_absorption": 0.0,
                "aggregate_liquid_assets": 10.0,
                "aggregate_debt": 1.0,
                "output": weighted_consumption,
                "goods_market_residual": 0.0,
                "next_employment_rate": 0.95,
                "next_inflation_rate": 2.0,
                "next_policy_rate": policy_rate,
                "next_aggregate_income": weighted_income,
            })
            accounting_rows.append({"candidate": candidate, "period_index": period, "unit": "aggregate", "identity": "one_good_output_equals_consumption", "residual": 0.0, "abs_residual": 0.0, "passed": True})
    write_json(root / "raw_records.json", records)
    write_json(root / "live_attempts.json", live_attempts)
    pd.DataFrame(prompt_cards).to_csv(root / "prompt_cards.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(root / "decisions.csv", index=False)
    pd.DataFrame(period_rows).to_csv(root / "periods.csv", index=False)
    pd.DataFrame(accounting_rows).to_csv(root / "accounting.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": candidate,
                "type_id": type_id,
                "population_weight": 1.0 / households,
                "labor_income": 5.0,
            }
            for candidate in ("llm", "adaptive")
            for type_id in household_ids
        ]
    ).to_csv(root / "final_household_states.csv", index=False)

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
        "execution_source": source_contract,
        "execution_source_tree_sha256": source_tree_sha256,
        "source_contract_sha256": source_contract_sha256,
        "bundle_sha256": FIXTURE_BUNDLE_SHA256,
        "prompt_version": "prompt-lock",
        "household_count": households,
        "live_call_count": len(live_attempts),
        "cache_hit_count": 0,
        "replayed_record_count": 0,
        "semantic_retry_count": 0,
        "rejected_semantic_payloads": [],
        "normalized_spec_sha256": canonical_sha(normalized_spec),
        "max_accounting_abs_residual": 0.0,
        "scored_target_row_count": 10,
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
    bundle_targets = fixture_bundle_targets()
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
                "execution_source_tree_sha256": fixture_source_contract()["tree_sha256"],
                "prompt_version": "prompt-lock",
                "bundle_sha256": FIXTURE_BUNDLE_SHA256,
                "bundle_targets": bundle_targets,
                "bundle_targets_sha256": canonical_sha(bundle_targets),
                "mechanism_fields": ["behavior_policy_mode", "feedback_mode", "feedback_gain", "policy_state_mode"],
                "mechanism_values": {
                    "behavior_policy_mode": "shared",
                    "feedback_mode": "closed_loop",
                    "feedback_gain": 1.0,
                    "policy_state_mode": "recursive",
                },
            },
            "runs": {
                "corrected81": {"household_count": 81, "households_sha256": "households-81", "accepted_live_calls": 5, "maximum_live_calls": 5, "raw_record_batch_counts": {"llm_belief": 5, "adaptive": 5}},
                "corrected200": {"household_count": 200, "households_sha256": "households-200", "accepted_live_calls": 10, "maximum_live_calls": 10, "raw_record_batch_counts": {"llm_belief": 10, "adaptive": 5}},
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
            self.assertAlmostEqual(float(scores["corrected81"]), 0.5 * math.sqrt(0.8))
            self.assertAlmostEqual(float(scores["corrected200"]), 0.504 * math.sqrt(0.8))
            self.assertEqual(len(comparison), 3)
            self.assertEqual(set(manifest["outputs"]), set(manifest["output_contract"]) - {"manifest.json"})
            self.assertEqual(json.loads((output / "promotion_receipt.json").read_text())["status"], "complete")
            self.assertIn("recomputed", (output / "report.md").read_text())

    def test_200_cell_aggregate_roundoff_uses_accounting_tolerance(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            periods = pd.read_csv(run200 / "periods.csv")
            periods["aggregate_liquid_assets"] += 2e-10
            periods.to_csv(run200 / "periods.csv", index=False)
            refresh_output_hash(run200, "periods.csv")
            manifest = compare_runs(spec, evidence, run81, run200, output)
            self.assertEqual(manifest["runs"]["corrected200"]["household_count"], 200)

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

    def test_source_tree_identity_cannot_be_replaced_by_source_contract(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            normalized = json.loads((run200 / "normalized_spec.json").read_text())
            normalized["execution_source_tree_sha256"] = "c" * 64
            normalized["source_contract_sha256"] = "a" * 64
            write_json(run200 / "normalized_spec.json", normalized)
            manifest = json.loads((run200 / "manifest.json").read_text())
            manifest["normalized_spec_sha256"] = canonical_sha(normalized)
            manifest["outputs"]["normalized_spec.json"] = sha(run200 / "normalized_spec.json")
            write_json(run200 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "source tree"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_accepts_both_live_attempt_cap_spellings(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            spec_data = json.loads(spec.read_text())
            for cohort in ("corrected81", "corrected200"):
                lock = spec_data["runs"][cohort]
                lock["max_live_calls"] = lock.pop("maximum_live_calls")
            write_json(spec, spec_data)
            manifest = compare_runs(spec, evidence, run81, run200, output)
            self.assertEqual(manifest["winner"], "corrected200")

    def test_derived_score_columns_are_recomputed_from_primitives(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            joined = pd.read_csv(run200 / "joined_errors.csv")
            joined["error"] = 123.0
            joined["scaled_error"] = 456.0
            joined["scaled_squared_error"] = 789.0
            joined["absolute_scaled_error"] = 987.0
            joined["direction_correct"] = True
            joined.to_csv(run200 / "joined_errors.csv", index=False)
            refresh_output_hash(run200, "joined_errors.csv")
            manifest = compare_runs(spec, evidence, run81, run200, output)
            self.assertAlmostEqual(
                manifest["recomputed_scores"]["corrected200"]["llm"],
                0.504 * math.sqrt(0.8),
            )

    def test_target_inventory_and_prompt_cards_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            joined = pd.read_csv(run200 / "joined_errors.csv")
            joined.loc[0, "target_name"] = joined.loc[1, "target_name"]
            joined.to_csv(run200 / "joined_errors.csv", index=False)
            refresh_output_hash(run200, "joined_errors.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "duplicate|ten locked targets"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            cards = pd.read_csv(run200 / "prompt_cards.csv")
            payload = {"population_weight": 1.0}
            cards.loc[0, "prompt_payload"] = json.dumps(payload)
            cards.loc[0, "prompt_payload_sha256"] = canonical_sha(payload)
            cards.to_csv(run200 / "prompt_cards.csv", index=False)
            records = json.loads((run200 / "raw_records.json").read_text())
            records[0]["prompt_payload_sha256"] = canonical_sha(payload)
            write_json(run200 / "raw_records.json", records)
            attempts = json.loads((run200 / "live_attempts.json").read_text())
            attempts[0]["prompt_payload_sha256"] = canonical_sha(payload)
            write_json(run200 / "live_attempts.json", attempts)
            for name in ("prompt_cards.csv", "raw_records.json", "live_attempts.json"):
                refresh_output_hash(run200, name)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "forbidden target or weight"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_prompt_coverage_attempt_cap_cache_binding_and_accounting_fail_closed(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            cards = pd.read_csv(run200 / "prompt_cards.csv").iloc[1:]
            cards.to_csv(run200 / "prompt_cards.csv", index=False)
            refresh_output_hash(run200, "prompt_cards.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "prompt cards"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            spec_data = json.loads(spec.read_text())
            spec_data["runs"]["corrected81"]["maximum_live_calls"] = 4
            write_json(spec, spec_data)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "cap"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            (run200 / "cache_0_0.json").unlink()
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "cache file"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            records = json.loads((run200 / "raw_records.json").read_text())
            records[0]["response_sha256"] = "e" * 64
            write_json(run200 / "raw_records.json", records)
            refresh_output_hash(run200, "raw_records.json")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "raw record batch provenance|does not bind response_sha256"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            decisions = pd.read_csv(run200 / "decisions.csv")
            decisions.loc[0, "consumption"] = 9.0
            decisions.to_csv(run200 / "decisions.csv", index=False)
            refresh_output_hash(run200, "decisions.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "aggregate_consumption|accounting evidence"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_bundle_joined_forecast_and_period_lineage_tampering_fails(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            normalized = json.loads((run81 / "normalized_spec.json").read_text())
            normalized["bundle_sha256"] = "e" * 64
            write_json(run81 / "normalized_spec.json", normalized)
            manifest = json.loads((run81 / "manifest.json").read_text())
            manifest["normalized_spec_sha256"] = canonical_sha(normalized)
            manifest["outputs"]["normalized_spec.json"] = sha(run81 / "normalized_spec.json")
            write_json(run81 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "frozen-bundle identity"):
                compare_runs(spec, evidence, run81, run200, output)

        for column, message in (
            ("prediction", "joined predictions differ"),
            ("target_value", "joined targets differ"),
        ):
            with TemporaryDirectory() as temp:
                spec, evidence, run81, run200, output = write_inputs(Path(temp))
                joined = pd.read_csv(run81 / "joined_errors.csv")
                joined.loc[0, column] = float(joined.loc[0, column]) + 1.0
                joined.to_csv(run81 / "joined_errors.csv", index=False)
                refresh_output_hash(run81, "joined_errors.csv")
                with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, message):
                    compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            forecasts = pd.read_csv(run81 / "forecasts.csv")
            forecasts.loc[0, "prediction"] = float(forecasts.loc[0, "prediction"]) + 1.0
            forecasts.to_csv(run81 / "forecasts.csv", index=False)
            refresh_output_hash(run81, "forecasts.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "forecasts do not reproduce"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            periods = pd.read_csv(run81 / "periods.csv")
            periods.loc[periods["period_index"].eq(1), "next_policy_rate"] += 1.0
            periods.to_csv(run81 / "periods.csv", index=False)
            refresh_output_hash(run81, "periods.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "forecasts do not reproduce"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            periods = pd.read_csv(run81 / "periods.csv")
            periods.loc[
                periods["period_index"].eq(4), "next_aggregate_income"
            ] += 1.0
            periods.to_csv(run81 / "periods.csv", index=False)
            refresh_output_hash(run81, "periods.csv")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "terminal aggregate income"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_raw_response_and_cache_payload_tampering_fails(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            records = json.loads((run81 / "raw_records.json").read_text())
            records[0]["payload"]["beliefs"][0]["extra"] = "tampered"
            write_json(run81 / "raw_records.json", records)
            refresh_output_hash(run81, "raw_records.json")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "raw record batch provenance"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            cache_path = run81 / "cache_0_0.json"
            cache = json.loads(cache_path.read_text())
            cache["payload"]["beliefs"][0]["extra"] = "tampered"
            write_json(cache_path, cache)
            attempts = json.loads((run81 / "live_attempts.json").read_text())
            attempts[0]["cache_file_sha256"] = sha(cache_path)
            write_json(run81 / "live_attempts.json", attempts)
            refresh_output_hash(run81, "live_attempts.json")
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "cache payload does not bind"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_saving_ratio_debt_and_retry_accounting_tampering_fails(self) -> None:
        for column in ("aggregate_saving", "aggregate_debt"):
            with TemporaryDirectory() as temp:
                spec, evidence, run81, run200, output = write_inputs(Path(temp))
                periods = pd.read_csv(run81 / "periods.csv")
                periods.loc[0, column] = float(periods.loc[0, column]) + 1.0
                periods.to_csv(run81 / "periods.csv", index=False)
                refresh_output_hash(run81, "periods.csv")
                with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, column):
                    compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            attempts = json.loads((run81 / "live_attempts.json").read_text())
            failed = []
            for _ in range(3):
                row = dict(attempts[0])
                row.update(status="failed", response_sha256=None, error_sha256="e" * 64)
                failed.append(row)
            attempts = failed + attempts
            for number, row in enumerate(attempts, start=1):
                row["attempt_number"] = number
                row["attempt_id"] = f"live_attempt_{number:04d}"
            write_json(run81 / "live_attempts.json", attempts)
            manifest = json.loads((run81 / "manifest.json").read_text())
            manifest["live_call_count"] = len(attempts)
            manifest["semantic_retry_count"] = 3
            manifest["rejected_semantic_payloads"] = [{"attempt": index} for index in range(3)]
            manifest["outputs"]["live_attempts.json"] = sha(run81 / "live_attempts.json")
            write_json(run81 / "manifest.json", manifest)
            spec_data = json.loads(spec.read_text())
            spec_data["runs"]["corrected81"]["maximum_live_calls"] = len(attempts)
            write_json(spec, spec_data)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "semantic-retry contract"):
                compare_runs(spec, evidence, run81, run200, output)

        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            manifest = json.loads((run81 / "manifest.json").read_text())
            manifest["cache_hit_count"] = 1
            write_json(run81 / "manifest.json", manifest)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "replay/cache accounting"):
                compare_runs(spec, evidence, run81, run200, output)

    def test_prompt_value_alias_and_source_contract_tampering_fails(self) -> None:
        with TemporaryDirectory() as temp:
            spec, evidence, run81, run200, output = write_inputs(Path(temp))
            cards = pd.read_csv(run81 / "prompt_cards.csv")
            payload = {"note": "hidden survey wgt field"}
            cards.loc[0, "prompt_payload"] = json.dumps(payload)
            cards.loc[0, "prompt_payload_sha256"] = canonical_sha(payload)
            cards.to_csv(run81 / "prompt_cards.csv", index=False)
            records = json.loads((run81 / "raw_records.json").read_text())
            records[0]["prompt_payload_sha256"] = canonical_sha(payload)
            write_json(run81 / "raw_records.json", records)
            attempts = json.loads((run81 / "live_attempts.json").read_text())
            attempts[0]["prompt_payload_sha256"] = canonical_sha(payload)
            write_json(run81 / "live_attempts.json", attempts)
            for name in ("prompt_cards.csv", "raw_records.json", "live_attempts.json"):
                refresh_output_hash(run81, name)
            with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, "forbidden target or weight"):
                compare_runs(spec, evidence, run81, run200, output)

        for mutation, message in (
            ("file_map", "tree hash does not reproduce"),
            ("top_tree", "top-level execution-source tree hash"),
            ("contract_sha", "source-contract hash"),
        ):
            with TemporaryDirectory() as temp:
                spec, evidence, run81, run200, output = write_inputs(Path(temp))
                manifest = json.loads((run81 / "manifest.json").read_text())
                if mutation == "file_map":
                    manifest["execution_source"]["files"]["src/macro_llm_tournament/fixture.py"] = "c" * 64
                elif mutation == "top_tree":
                    manifest["execution_source_tree_sha256"] = "c" * 64
                    normalized = json.loads((run81 / "normalized_spec.json").read_text())
                    normalized["execution_source_tree_sha256"] = "c" * 64
                    write_json(run81 / "normalized_spec.json", normalized)
                    manifest["normalized_spec_sha256"] = canonical_sha(normalized)
                    manifest["outputs"]["normalized_spec.json"] = sha(run81 / "normalized_spec.json")
                    spec_data = json.loads(spec.read_text())
                    spec_data["locks"]["execution_source_tree_sha256"] = "c" * 64
                    write_json(spec, spec_data)
                else:
                    manifest["source_contract_sha256"] = "c" * 64
                write_json(run81 / "manifest.json", manifest)
                with self.assertRaisesRegex(DynamicMacroHouseholdScaleError, message):
                    compare_runs(spec, evidence, run81, run200, output)


if __name__ == "__main__":
    unittest.main()
