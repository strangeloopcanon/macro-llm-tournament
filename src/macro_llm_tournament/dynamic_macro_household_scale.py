"""Fail-closed retrospective comparison of corrected 81- and 200-household runs.

This module only consumes completed dynamic-macro run directories.  It never
reruns a model or modifies a run directory.  The comparison lock owns the
claim scope, cohort sizes, batch contract, provenance locks, and score
surface; the run artifacts own the observations and predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE
from .dynamic_macro_economy import OUTPUT_FILES as ECONOMY_OUTPUT_FILES
from .dynamic_macro_economy import score_macro
from .dynamic_macro_clients import validate_live_attempt_ledger


SCHEMA_VERSION = "dynamic_macro_household_scale_comparison_v1"
CLAIM_SCOPE = "developmental_retrospective_only"
COHORTS = ("corrected81", "corrected200")
EXPECTED_ORIGINS = tuple(f"2026-{month:02d}-01" for month in range(1, 6))
EXPECTED_SCORE_ORIGINS = EXPECTED_ORIGINS[1:]
OUTPUT_FILES = (
    "manifest.json",
    "comparison.csv",
    "family_scores.csv",
    "origin_scores.csv",
    "promotion_receipt.json",
    "report.md",
)
RUN_OUTPUT_FILES = tuple(ECONOMY_OUTPUT_FILES)
REQUIRED_JOINED_COLUMNS = {
    "candidate",
    "origin_month",
    "target_name",
    "family",
    "scaled_squared_error",
    "absolute_scaled_error",
    "direction_correct",
}
SURFACE_COLUMNS = (
    "origin_month",
    "as_of_date",
    "target_observation_date",
    "target_name",
    "series_id",
    "family",
    "economy_measure",
    "economy_transform",
    "transform",
    "default_scale",
    "target_value",
    "origin_visible_denominator_value",
)
COHORT_SPEC_ALIASES = {
    "corrected81": ("corrected81", "81", "run81", "run_81"),
    "corrected200": ("corrected200", "200", "run200", "run_200"),
}
PROMPT_LOCK_KEYS = (
    "prompt_sha256",
    "prompt_hash",
    "prompt_contract_sha256",
    "prompt_lock_sha256",
    "prompt_version",
)
SOURCE_LOCK_KEYS = (
    "source_sha256",
    "source_hash",
    "source_contract_sha256",
    "source_lock_sha256",
    "execution_source_sha256",
    "execution_source_tree_sha256",
)
MECHANISM_EXCLUDED_KEYS = {
    "household_count",
    "household_provenance",
    "households_sha256",
    "normalized_household_state_sha256",
    "raw_input_file_sha256",
    "raw_records_sha256",
    "raw_record_count",
    "raw_record_batch_count",
    "raw_record_batch_counts",
    "batch_count",
    "batch_counts",
    "cohort",
    "cohort_id",
    "run_id",
    "run_name",
}


class DynamicMacroHouseholdScaleError(ValueError):
    """Raised for any invalid lock, run, score surface, or output state."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare corrected 81- and 200-household retrospective macro runs."
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--historical-evidence", required=True)
    parser.add_argument("--run-81-dir", "--run81-dir", dest="run81_dir", required=True)
    parser.add_argument("--run-200-dir", "--run200-dir", dest="run200_dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run_comparison(parse_args(argv))
    except (DynamicMacroHouseholdScaleError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(_canonical_json({"output_dir": manifest["output_dir"], "winner": manifest["winner"]}))
    return 0


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Validate two finished runs, score them, and write a promotion receipt."""

    spec_path = _resolve(args.spec)
    evidence_path = _resolve(args.historical_evidence)
    output_dir = _resolve(args.output_dir)
    spec = _read_object(spec_path, "comparison spec")
    evidence = _read_object(evidence_path, "historical evidence")
    _validate_spec(spec, spec_path, evidence_path)
    _validate_historical_evidence(evidence)
    _validate_locked_path(spec.get("historical_evidence"), evidence_path, evidence_path, "historical evidence")
    _reject_stale_output(output_dir)

    run_dirs = {
        "corrected81": _resolve(args.run81_dir),
        "corrected200": _resolve(args.run200_dir),
    }
    runs = {
        cohort: validate_run(run_dir, spec=spec, cohort=cohort)
        for cohort, run_dir in run_dirs.items()
    }
    _validate_run_pair(runs, spec)
    result = _build_result(spec, evidence, evidence_path, spec_path, runs)
    return _write_result(output_dir, result)


def compare_runs(
    spec_path: str | Path,
    historical_evidence_path: str | Path,
    run81_dir: str | Path,
    run200_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convenience API for callers that do not need to construct argparse args."""

    args = argparse.Namespace(
        spec=str(spec_path),
        historical_evidence=str(historical_evidence_path),
        run81_dir=str(run81_dir),
        run200_dir=str(run200_dir),
        output_dir=str(output_dir),
    )
    return run_comparison(args)


def validate_run(run_dir: Path | str, *, spec: Mapping[str, Any], cohort: str) -> dict[str, Any]:
    """Return a validated, recomputed run record; never trust manifest scores."""

    if cohort not in COHORTS:
        raise DynamicMacroHouseholdScaleError(f"Unknown cohort: {cohort}")
    root = _resolve(run_dir)
    if not root.is_dir():
        raise DynamicMacroHouseholdScaleError(f"Missing {cohort} run directory: {root}")
    manifest_path = root / "manifest.json"
    spec_path = root / "normalized_spec.json"
    manifest = _read_object(manifest_path, f"{cohort} manifest")
    normalized_spec = _read_object(spec_path, f"{cohort} normalized spec")
    _validate_run_contract(root, manifest, cohort)
    _validate_normalized_spec_hash(manifest, normalized_spec, spec_path, cohort)
    cohort_lock = _cohort_lock(spec, cohort) or {}
    _validate_locked_path(cohort_lock.get("path"), root, root, f"{cohort} run")
    if cohort_lock.get("manifest_sha256") and _sha256_file(manifest_path) != cohort_lock["manifest_sha256"]:
        raise DynamicMacroHouseholdScaleError(f"Locked {cohort} manifest hash changed")
    if cohort_lock.get("normalized_spec_sha256") and _sha256_file(spec_path) != cohort_lock["normalized_spec_sha256"]:
        raise DynamicMacroHouseholdScaleError(f"Locked {cohort} normalized spec hash changed")
    _validate_run_locks(manifest, normalized_spec, spec, cohort)
    _validate_run_calendar(manifest, normalized_spec, cohort)

    expected_households = _expected_household_count(spec, cohort)
    households = _read_csv(root / "households.csv", f"{cohort} households")
    if len(households) != expected_households:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} household count is {len(households)}, expected {expected_households}"
        )
    if "type_id" in households and households["type_id"].duplicated().any():
        raise DynamicMacroHouseholdScaleError(f"{cohort} households contain duplicate type_id values")
    manifest_count = _first_int(manifest, "household_count", "normalized_household_count")
    if manifest_count != expected_households:
        raise DynamicMacroHouseholdScaleError(f"{cohort} manifest household count is invalid")

    if "type_id" not in households or "population_weight" not in households:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} households lack identity or population weights"
        )
    weights = pd.to_numeric(households["population_weight"], errors="coerce")
    if (
        weights.isna().any()
        or (weights <= 0).any()
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-10)
    ):
        raise DynamicMacroHouseholdScaleError(f"{cohort} population weights are invalid")
    household_ids = tuple(sorted(households["type_id"].astype(str)))
    cohort_households_sha = str(
        normalized_spec.get("household_provenance", {}).get("raw_input_file_sha256", "")
    )
    expected_households_sha = str(cohort_lock.get("households_sha256", ""))
    if expected_households_sha and cohort_households_sha != expected_households_sha:
        raise DynamicMacroHouseholdScaleError(f"{cohort} household input hash changed")

    raw_records = _read_list(root / "raw_records.json", f"{cohort} raw records")
    batch_counts = _validate_raw_records(
        raw_records, household_ids, spec, cohort
    )
    live_attempts = _read_list(root / "live_attempts.json", f"{cohort} live attempts")
    call_counts = _validate_live_attempts(
        live_attempts, manifest, spec, cohort, household_ids
    )
    accounting = _read_csv(root / "accounting.csv", f"{cohort} accounting")
    max_residual = _validate_accounting(accounting, manifest, cohort)

    joined = _read_csv(root / "joined_errors.csv", f"{cohort} joined errors")
    _validate_joined_surface(joined, cohort)
    try:
        target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface cannot be recomputed: {exc}"
        ) from exc
    all_scores = _all_scores(target_scores, macro_scores, cohort)

    return {
        "cohort": cohort,
        "run_dir": str(root),
        "manifest": manifest,
        "normalized_spec": normalized_spec,
        "manifest_sha256": _sha256_file(manifest_path),
        "normalized_spec_sha256": _sha256_file(spec_path),
        "output_hashes": {
            name: _sha256_file(root / name)
            for name in RUN_OUTPUT_FILES
            if name != "manifest.json"
        },
        "joined": joined,
        "target_scores": target_scores,
        "family_scores": family_scores,
        "origin_scores": origin_scores,
        "macro_scores": {str(key): float(value) for key, value in macro_scores.items()},
        "all_scores": all_scores,
        "household_count": expected_households,
        "raw_record_batch_counts": batch_counts,
        "call_counts": call_counts,
        "effective_weighted_sample_size": float(1.0 / float((weights**2).sum())),
        "max_normalized_household_weight": float(weights.max()),
        "direction_accuracy": {
            str(candidate): float(group["direction_accuracy"].mean())
            for candidate, group in target_scores.groupby("candidate")
        },
        "max_accounting_abs_residual": max_residual,
        "mechanism_fingerprint": _mechanism_fingerprint(normalized_spec),
        "lock_values": _actual_lock_values(manifest, normalized_spec),
    }


def _validate_spec(spec: Mapping[str, Any], spec_path: Path, evidence_path: Path) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise DynamicMacroHouseholdScaleError("Unsupported household-scale comparison spec")
    if spec.get("claim_scope") != CLAIM_SCOPE:
        raise DynamicMacroHouseholdScaleError("Household-scale comparison must be developmental and retrospective only")
    _reject_confirmatory_labels(spec, "comparison spec")
    locks = spec.get("locks")
    if not isinstance(locks, Mapping):
        raise DynamicMacroHouseholdScaleError("Comparison spec must contain locked provenance values")
    if not _expected_value(locks, "provider") or not _expected_value(locks, "model"):
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock provider and model")
    if _expected_value(locks, "provider_reasoning_effort") != "high":
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock Codex CLI reasoning effort to high"
        )
    if _first_locked_value(locks, SOURCE_LOCK_KEYS) is None:
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock source provenance")
    if _first_locked_value(locks, PROMPT_LOCK_KEYS) is None:
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock the prompt")
    if not _has_any(locks, "mechanism", "mechanism_sha256", "mechanism_fields"):
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock the mechanism")
    for cohort in COHORTS:
        if _cohort_lock(spec, cohort) is None:
            raise DynamicMacroHouseholdScaleError(f"Comparison spec is missing {cohort} lock")
    if spec_path == evidence_path:
        raise DynamicMacroHouseholdScaleError("Comparison spec and historical evidence must be separate inputs")


def _validate_historical_evidence(evidence: Mapping[str, Any]) -> None:
    _reject_confirmatory_labels(evidence, "historical evidence")
    if not _contains_string(evidence, "uncorrected81"):
        raise DynamicMacroHouseholdScaleError("Historical evidence must identify uncorrected81 context")
    if not _contains_string(evidence, "context"):
        raise DynamicMacroHouseholdScaleError("Historical uncorrected81 evidence must be context only")
    counts = _named_ints(evidence, {"household_count", "respondent_count", "normalized_household_count"})
    if counts and any(value != 81 for value in counts):
        raise DynamicMacroHouseholdScaleError("Historical evidence household count must be 81")
    eligibility = _named_bools(evidence, {"eligible_for_promotion", "promotion_eligible"})
    if not eligibility or any(eligibility):
        raise DynamicMacroHouseholdScaleError("Historical uncorrected81 evidence is ineligible for promotion")
    context = evidence.get("historical_uncorrected81_context")
    required = {
        "household_count",
        "llm_macro_score",
        "adaptive_macro_score",
        "llm_direction_accuracy",
        "adaptive_direction_accuracy",
        "effective_weighted_sample_size",
        "max_normalized_household_weight",
        "max_accounting_abs_residual",
        "live_call_count",
        "semantic_retry_count",
        "replayed_record_count",
        "cache_hit_count",
        "eligible_for_promotion",
    }
    if not isinstance(context, Mapping) or not required.issubset(context):
        raise DynamicMacroHouseholdScaleError(
            "Historical evidence lacks the uncorrected81 comparison context"
        )


def _validate_locked_path(lock: Any, actual: Path, base: Path, label: str) -> None:
    if lock is None:
        return
    if isinstance(lock, str):
        if _resolve(lock) != actual:
            raise DynamicMacroHouseholdScaleError(f"Locked {label} path changed")
        return
    if not isinstance(lock, Mapping):
        raise DynamicMacroHouseholdScaleError(f"Locked {label} is invalid")
    path = lock.get("path")
    if isinstance(path, str) and _resolve(path) != actual:
            raise DynamicMacroHouseholdScaleError(f"Locked {label} path changed")
    expected = lock.get("sha256")
    if expected and _sha256_file(actual) != expected:
        raise DynamicMacroHouseholdScaleError(f"Locked {label} hash changed")


def _validate_run_contract(root: Path, manifest: Mapping[str, Any], cohort: str) -> None:
    if manifest.get("status") != "complete":
        raise DynamicMacroHouseholdScaleError(f"{cohort} run is not complete")
    contract = manifest.get("output_contract")
    expected = set(RUN_OUTPUT_FILES)
    if not isinstance(contract, list) or set(contract) != expected or len(contract) != len(expected):
        raise DynamicMacroHouseholdScaleError(f"{cohort} output contract is incomplete or contains unknown files")
    outputs = manifest.get("outputs")
    expected_hashes = expected - {"manifest.json"}
    if not isinstance(outputs, Mapping) or set(outputs) != expected_hashes:
        raise DynamicMacroHouseholdScaleError(f"{cohort} output hashes do not cover the contract")
    for name, expected_hash in outputs.items():
        path = root / str(name)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise DynamicMacroHouseholdScaleError(f"{cohort} output hash mismatch: {name}")
    if any(not (root / name).is_file() for name in RUN_OUTPUT_FILES):
        raise DynamicMacroHouseholdScaleError(f"{cohort} output contract is incomplete")


def _validate_normalized_spec_hash(manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any], path: Path, cohort: str) -> None:
    expected = manifest.get("normalized_spec_sha256")
    if not isinstance(expected, str) or _canonical_sha(normalized_spec) != expected:
        raise DynamicMacroHouseholdScaleError(f"{cohort} normalized spec hash does not reproduce")
    if _sha256_file(path) == "":
        raise DynamicMacroHouseholdScaleError(f"{cohort} normalized spec is unreadable")


def _validate_run_locks(manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any], spec: Mapping[str, Any], cohort: str) -> None:
    locks = spec["locks"]
    expected_provider = _expected_value(locks, "provider")
    expected_model = _expected_value(locks, "model")
    actual_provider = _first_value(manifest, normalized_spec, "provider")
    actual_model = _first_value(manifest, normalized_spec, "model")
    if manifest.get("provider") is not None and normalized_spec.get("provider") is not None and manifest["provider"] != normalized_spec["provider"]:
        raise DynamicMacroHouseholdScaleError(f"{cohort} manifest/spec provider identity differs")
    if manifest.get("model") is not None and normalized_spec.get("model") is not None and manifest["model"] != normalized_spec["model"]:
        raise DynamicMacroHouseholdScaleError(f"{cohort} manifest/spec model identity differs")
    if actual_provider != expected_provider or actual_model != expected_model:
        raise DynamicMacroHouseholdScaleError(f"{cohort} provider/model lock changed")
    if (
        _first_value(manifest, normalized_spec, "provider_reasoning_effort")
        != locks.get("provider_reasoning_effort")
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} provider reasoning-effort lock changed"
        )
    for label, keys in (("source", SOURCE_LOCK_KEYS), ("prompt", PROMPT_LOCK_KEYS)):
        expected = _first_locked_value(locks, keys)
        actual = _first_locked_value({**normalized_spec, **manifest}, keys)
        if expected is not None and actual != expected:
            raise DynamicMacroHouseholdScaleError(f"{cohort} {label} lock changed")
    mechanism_lock = locks.get("mechanism")
    if isinstance(mechanism_lock, Mapping) and _mechanism_projection(normalized_spec) != _canonical_obj(mechanism_lock):
        raise DynamicMacroHouseholdScaleError(f"{cohort} mechanism does not match the lock")
    if isinstance(mechanism_lock, str) and normalized_spec.get("mechanism", normalized_spec.get("behavior_policy_mode")) != mechanism_lock:
        raise DynamicMacroHouseholdScaleError(f"{cohort} mechanism does not match the lock")
    mechanism_fields = locks.get("mechanism_fields")
    if isinstance(mechanism_fields, list):
        observed = {str(field): _lookup_path(normalized_spec, str(field)) for field in mechanism_fields}
        expected = locks.get("mechanism_values", {})
        if not isinstance(expected, Mapping) or _canonical_obj(observed) != _canonical_obj(expected):
            raise DynamicMacroHouseholdScaleError(f"{cohort} mechanism fields do not match the lock")
    expected_mechanism_sha = locks.get("mechanism_sha256")
    if expected_mechanism_sha and _canonical_sha(_mechanism_fingerprint(normalized_spec)) != expected_mechanism_sha:
        raise DynamicMacroHouseholdScaleError(f"{cohort} mechanism hash changed")


def _validate_raw_records(
    records: list[dict[str, Any]],
    household_ids: tuple[str, ...],
    spec: Mapping[str, Any],
    cohort: str,
) -> dict[str, int]:
    expected = _expected_batch_counts(spec, cohort)
    expected_ids = set(household_ids)
    actual: dict[str, int] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen: set[tuple[str, int, int]] = set()
    for row in records:
        candidate = str(row.get("candidate", ""))
        if candidate not in {"llm_belief", "adaptive"}:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw record has an unknown candidate"
            )
        identity = row.get("cache_identity")
        period = (
            identity.get("period_index")
            if isinstance(identity, Mapping)
            else row.get("period_index")
        )
        if period is None:
            raw_period_id = str(row.get("period_id", ""))
            if raw_period_id.startswith("period_"):
                period = raw_period_id.removeprefix("period_")
        try:
            period_value = int(period)
            batch_index = int(row.get("batch_index", 0))
            batch_count = int(row.get("batch_count", 1))
        except (TypeError, ValueError) as exc:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw record has malformed period or batch identity"
            ) from exc
        if period_value not in range(5) or batch_index < 0 or batch_count <= batch_index:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw record period or batch identity is out of range"
            )
        key = (candidate, period_value, batch_index)
        if key in seen:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw records contain duplicate batches"
            )
        seen.add(key)
        payload = row.get("payload")
        beliefs = payload.get("beliefs") if isinstance(payload, Mapping) else None
        if not isinstance(beliefs, list) or not beliefs:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw record batch lacks beliefs"
            )
        ids = [str(belief.get("type_id", "")) for belief in beliefs if isinstance(belief, Mapping)]
        if len(ids) != len(beliefs) or len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw record batch has invalid household coverage"
            )
        if candidate == "llm_belief":
            recorded_ids = row.get("household_type_ids")
            if (
                not isinstance(recorded_ids, list)
                or recorded_ids != sorted(recorded_ids)
                or set(recorded_ids) != set(ids)
                or row.get("household_type_ids_sha256") != _canonical_sha(recorded_ids)
                or not _is_sha256(row.get("prompt_payload_sha256"))
                or not _is_sha256(row.get("response_sha256"))
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} LLM raw record batch provenance is invalid"
                )
        grouped.setdefault((candidate, period_value), []).append(row)
        actual[candidate] = actual.get(candidate, 0) + 1

    if actual != expected:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record batch counts {actual} do not match {expected}"
        )
    for candidate in ("llm_belief", "adaptive"):
        for period in range(5):
            rows = grouped.get((candidate, period), [])
            if not rows:
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} {candidate} raw-record periods are incomplete"
                )
            counts = {int(row.get("batch_count", 1)) for row in rows}
            indexes = sorted(int(row.get("batch_index", 0)) for row in rows)
            if len(counts) != 1 or indexes != list(range(next(iter(counts)))):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} {candidate} raw-record batch layout is incomplete"
                )
            union: set[str] = set()
            for row in rows:
                beliefs = row["payload"]["beliefs"]
                ids = {str(item["type_id"]) for item in beliefs}
                if union.intersection(ids):
                    raise DynamicMacroHouseholdScaleError(
                        f"{cohort} {candidate} batches overlap"
                    )
                union.update(ids)
            if union != expected_ids:
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} {candidate} batches do not cover the cohort exactly once"
                )
    return actual


def _validate_live_attempts(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    cohort: str,
    household_ids: tuple[str, ...],
) -> dict[str, int]:
    try:
        validate_live_attempt_ledger(rows)
    except Exception as exc:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} live-attempt ledger is invalid: {exc}"
        ) from exc
    expected_accepted = int(
        (_cohort_lock(spec, cohort) or {}).get("accepted_live_calls", -1)
    )
    accepted = [row for row in rows if row.get("status") == "accepted"]
    failed = [row for row in rows if row.get("status") == "failed"]
    if expected_accepted < 0 or len(accepted) != expected_accepted:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accepted live-call count does not match the lock"
        )
    if int(manifest.get("live_call_count", -1)) != len(rows):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} live-call manifest does not reconcile with its ledger"
        )
    expected_ids = set(household_ids)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in accepted:
        if (
            row.get("provider") != "codex_cli"
            or row.get("model") != "gpt-5.5"
            or row.get("provider_called") is not True
            or not _is_sha256(row.get("response_sha256"))
        ):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} live attempt provider or response provenance is invalid"
            )
        grouped.setdefault(int(row["period_index"]), []).append(row)
    if set(grouped) != set(range(5)):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accepted live attempts do not cover all five periods"
        )
    for period, period_rows in grouped.items():
        counts = {int(row["batch_count"]) for row in period_rows}
        indexes = sorted(int(row["batch_index"]) for row in period_rows)
        if len(counts) != 1 or indexes != list(range(next(iter(counts)))):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} live-attempt batches are incomplete in period {period}"
            )
        union: set[str] = set()
        for row in period_rows:
            ids = set(row["household_type_ids"])
            if union.intersection(ids):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} live-attempt batches overlap in period {period}"
                )
            union.update(ids)
        if union != expected_ids:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} live-attempt batches do not cover the cohort"
            )
    return {
        "live": len(rows),
        "accepted": len(accepted),
        "failed": len(failed),
        "semantic_retries": int(manifest.get("semantic_retry_count", 0)),
        "replayed": int(manifest.get("replayed_record_count", 0)),
        "cache_hits": int(manifest.get("cache_hit_count", 0)),
    }


def _validate_accounting(accounting: pd.DataFrame, manifest: Mapping[str, Any], cohort: str) -> float:
    required = {"residual", "abs_residual", "passed"}
    if accounting.empty or not required.issubset(accounting.columns):
        raise DynamicMacroHouseholdScaleError(f"{cohort} accounting table is incomplete")
    residual = pd.to_numeric(accounting["residual"], errors="coerce")
    absolute = pd.to_numeric(accounting["abs_residual"], errors="coerce")
    if residual.isna().any() or absolute.isna().any() or not (absolute - residual.abs()).abs().le(1e-15).all():
        raise DynamicMacroHouseholdScaleError(f"{cohort} accounting residuals are invalid")
    if not accounting["passed"].map(_as_bool).all():
        raise DynamicMacroHouseholdScaleError(f"{cohort} has accounting violations")
    maximum = float(absolute.max())
    if not math.isfinite(maximum) or maximum > ACCOUNTING_TOLERANCE:
        raise DynamicMacroHouseholdScaleError(f"{cohort} accounting residual exceeds tolerance")
    recorded = float(manifest.get("max_accounting_abs_residual", math.nan))
    if not math.isclose(maximum, recorded, rel_tol=0.0, abs_tol=1e-15):
        raise DynamicMacroHouseholdScaleError(f"{cohort} accounting maximum does not reproduce")
    return maximum


def _validate_joined_surface(joined: pd.DataFrame, cohort: str) -> None:
    if not REQUIRED_JOINED_COLUMNS.issubset(joined.columns):
        raise DynamicMacroHouseholdScaleError(f"{cohort} joined score surface is incomplete")
    if set(joined["candidate"].astype(str)) != {"llm", "adaptive"}:
        raise DynamicMacroHouseholdScaleError(f"{cohort} joined score surface is not a matched pair")
    origins = set(joined["origin_month"].astype(str))
    if origins != set(EXPECTED_SCORE_ORIGINS):
        if any("2026-06" in origin for origin in origins):
            raise DynamicMacroHouseholdScaleError(f"{cohort} contains June rows")
        raise DynamicMacroHouseholdScaleError(f"{cohort} joined score surface must be Feb-May only")
    if joined.astype(str).apply(lambda column: column.str.contains(r"2026-06", regex=True).any()).any():
        raise DynamicMacroHouseholdScaleError(f"{cohort} contains June rows")
    identity = joined[["candidate", "origin_month", "target_name"]]
    if identity.duplicated().any():
        raise DynamicMacroHouseholdScaleError(f"{cohort} joined score surface contains duplicate target rows")
    for column in ("scaled_squared_error", "absolute_scaled_error"):
        values = pd.to_numeric(joined[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise DynamicMacroHouseholdScaleError(f"{cohort} score column {column} is invalid")
    squared = pd.to_numeric(joined["scaled_squared_error"], errors="coerce")
    absolute = pd.to_numeric(joined["absolute_scaled_error"], errors="coerce")
    if not (squared - absolute.pow(2)).abs().le(1e-12).all():
        raise DynamicMacroHouseholdScaleError(f"{cohort} derived score columns are inconsistent")


def _validate_run_calendar(manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any], cohort: str) -> None:
    for payload in (manifest, normalized_spec):
        contract = payload.get("score_origin_contract")
        if not isinstance(contract, Mapping):
            continue
        scored = tuple(str(value) for value in contract.get("scored_origins", []))
        if scored and scored != EXPECTED_SCORE_ORIGINS:
            if any("2026-06" in value for value in scored):
                raise DynamicMacroHouseholdScaleError(f"{cohort} contains June score rows")
            raise DynamicMacroHouseholdScaleError(f"{cohort} score-origin contract is not Feb-May")
        warmup = tuple(str(value) for value in contract.get("warmup_origins", []))
        if warmup and warmup != (EXPECTED_ORIGINS[0],):
            raise DynamicMacroHouseholdScaleError(f"{cohort} warm-up origin contract changed")


def _validate_run_pair(runs: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any]) -> None:
    first = runs[COHORTS[0]]["joined"]
    second = runs[COHORTS[1]]["joined"]
    for column in SURFACE_COLUMNS:
        if column not in first.columns or column not in second.columns:
            raise DynamicMacroHouseholdScaleError(f"Score surface lacks locked column {column}")
    left = _surface_frame(first)
    right = _surface_frame(second)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=0.0, atol=1e-12)
    except AssertionError as exc:
        raise DynamicMacroHouseholdScaleError("Corrected 81 and 200 Jan-May score surfaces differ") from exc
    if runs[COHORTS[0]]["mechanism_fingerprint"] != runs[COHORTS[1]]["mechanism_fingerprint"]:
        raise DynamicMacroHouseholdScaleError("Corrected 81 and 200 mechanisms differ")
    if runs[COHORTS[0]]["lock_values"] != runs[COHORTS[1]]["lock_values"]:
        raise DynamicMacroHouseholdScaleError("Corrected 81 and 200 source/prompt/model locks differ")
    expected_origins = tuple(spec.get("origins", EXPECTED_ORIGINS))
    if expected_origins != EXPECTED_ORIGINS or tuple(spec.get("score_origins", EXPECTED_SCORE_ORIGINS)) != EXPECTED_SCORE_ORIGINS:
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock January-May with February-May scored")


def _build_result(
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_path: Path,
    spec_path: Path,
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    score81 = runs["corrected81"]["all_scores"]["llm"]
    score200 = runs["corrected200"]["all_scores"]["llm"]
    relative_gap = math.inf if score81 == 0 and score200 != 0 else (0.0 if score81 == score200 == 0 else abs(score200 - score81) / abs(score81))
    tolerance = float(spec.get("relative_gap_tolerance", 0.01))
    winner = "corrected200" if score200 <= score81 or relative_gap <= tolerance else "corrected81"
    comparison_rows = [_historical_comparison_row(evidence)]
    comparison_rows.extend(
        _comparison_row(runs[cohort], cohort, score81, score200, relative_gap, winner)
        for cohort in COHORTS
    )
    comparison = pd.DataFrame(comparison_rows)
    family = pd.concat([_tag_scores(runs[c]["family_scores"], c) for c in COHORTS], ignore_index=True)
    origin = pd.concat([_tag_scores(runs[c]["origin_scores"], c) for c in COHORTS], ignore_index=True)
    family = family.sort_values(["cohort", "candidate", "family"], kind="mergesort").reset_index(drop=True)
    origin = origin.sort_values(["cohort", "candidate", "origin_month"], kind="mergesort").reset_index(drop=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_scope": CLAIM_SCOPE,
        "output_dir": "",
        "winner": winner,
        "decision_rule": {
            "metric": "absolute_llm_ALL_rmse_scaled",
            "lower_is_better": True,
            "relative_gap_tolerance": tolerance,
            "near_tie_preference": "corrected200",
            "adaptive_role": "diagnostic_only_not_a_selection_veto",
        },
        "spec_sha256": _sha256_file(spec_path),
        "historical_evidence_sha256": _sha256_file(evidence_path),
        "origins": list(EXPECTED_ORIGINS),
        "score_origins": list(EXPECTED_SCORE_ORIGINS),
        "runs": {
            cohort: {
                key: value
                for key, value in runs[cohort].items()
                if key in {"run_dir", "manifest_sha256", "normalized_spec_sha256", "output_hashes", "household_count", "raw_record_batch_counts", "call_counts", "effective_weighted_sample_size", "max_normalized_household_weight", "direction_accuracy", "max_accounting_abs_residual", "macro_scores", "all_scores", "mechanism_fingerprint", "lock_values"}
            }
            for cohort in COHORTS
        },
        "recomputed_scores": {
            cohort: runs[cohort]["all_scores"] for cohort in COHORTS
        },
        "relative_llm_gap_corrected200_vs_corrected81": relative_gap,
        "output_contract": list(OUTPUT_FILES),
    }
    return {"manifest": manifest, "comparison": comparison, "family": family, "origin": origin, "receipt": {"winner": winner, "llm_all_rmse_scaled": {c: runs[c]["all_scores"]["llm"] for c in COHORTS}, "relative_gap": relative_gap}}


def _write_result(output_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison = result["comparison"]
    family = result["family"]
    origin = result["origin"]
    manifest = dict(result["manifest"])
    manifest["output_dir"] = str(output_dir)
    _write_csv(output_dir / "comparison.csv", comparison)
    _write_csv(output_dir / "family_scores.csv", family)
    _write_csv(output_dir / "origin_scores.csv", origin)
    report = _build_report(manifest, comparison, family, origin)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest["outputs"] = {
        name: _sha256_file(output_dir / name)
        for name in ("comparison.csv", "family_scores.csv", "origin_scores.csv", "report.md")
    }
    receipt = dict(result["receipt"])
    receipt.update({"schema_version": "dynamic_macro_household_scale_promotion_receipt_v1", "status": "complete", "claim_scope": CLAIM_SCOPE, "artifact_hashes": dict(manifest["outputs"])})
    _write_json(output_dir / "promotion_receipt.json", receipt)
    manifest["outputs"]["promotion_receipt.json"] = _sha256_file(output_dir / "promotion_receipt.json")
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _comparison_row(run: Mapping[str, Any], cohort: str, score81: float, score200: float, relative_gap: float, winner: str) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "result_role": "current_corrected_development_comparison",
        "eligible_for_promotion": True,
        "household_count": run["household_count"],
        "effective_weighted_sample_size": run["effective_weighted_sample_size"],
        "max_normalized_household_weight": run["max_normalized_household_weight"],
        "raw_record_batch_count": sum(run["raw_record_batch_counts"].values()),
        "llm_ALL_rmse_scaled": run["all_scores"]["llm"],
        "adaptive_ALL_rmse_scaled": run["all_scores"]["adaptive"],
        "llm_macro_score_recomputed": run["macro_scores"]["llm"],
        "adaptive_macro_score_recomputed": run["macro_scores"]["adaptive"],
        "llm_minus_adaptive_recomputed": run["macro_scores"]["llm"] - run["macro_scores"]["adaptive"],
        "llm_direction_accuracy": run["direction_accuracy"]["llm"],
        "adaptive_direction_accuracy": run["direction_accuracy"]["adaptive"],
        "max_accounting_abs_residual": run["max_accounting_abs_residual"],
        "live_call_count": run["call_counts"]["live"],
        "accepted_live_call_count": run["call_counts"]["accepted"],
        "failed_live_call_count": run["call_counts"]["failed"],
        "semantic_retry_count": run["call_counts"]["semantic_retries"],
        "replayed_record_count": run["call_counts"]["replayed"],
        "cache_hit_count": run["call_counts"]["cache_hits"],
        "relative_gap_corrected200_vs_corrected81": relative_gap,
        "selected": cohort == winner,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
    }


def _historical_comparison_row(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = evidence["historical_uncorrected81_context"]
    return {
        "cohort": "historical_uncorrected81",
        "result_role": "historical_context_only",
        "eligible_for_promotion": False,
        "household_count": int(context["household_count"]),
        "effective_weighted_sample_size": float(context["effective_weighted_sample_size"]),
        "max_normalized_household_weight": float(context["max_normalized_household_weight"]),
        "raw_record_batch_count": int(context["replayed_record_count"]),
        "llm_ALL_rmse_scaled": float(context["llm_macro_score"]),
        "adaptive_ALL_rmse_scaled": float(context["adaptive_macro_score"]),
        "llm_macro_score_recomputed": float(context["llm_macro_score"]),
        "adaptive_macro_score_recomputed": float(context["adaptive_macro_score"]),
        "llm_minus_adaptive_recomputed": float(context["llm_macro_score"]) - float(context["adaptive_macro_score"]),
        "llm_direction_accuracy": float(context["llm_direction_accuracy"]),
        "adaptive_direction_accuracy": float(context["adaptive_direction_accuracy"]),
        "max_accounting_abs_residual": float(context["max_accounting_abs_residual"]),
        "live_call_count": int(context["live_call_count"]),
        "accepted_live_call_count": 0,
        "failed_live_call_count": 0,
        "semantic_retry_count": int(context["semantic_retry_count"]),
        "replayed_record_count": int(context["replayed_record_count"]),
        "cache_hit_count": int(context["cache_hit_count"]),
        "relative_gap_corrected200_vs_corrected81": math.nan,
        "selected": False,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
    }


def _all_scores(target_scores: pd.DataFrame, macro_scores: Mapping[str, float], cohort: str) -> dict[str, float]:
    names = target_scores["target_name"].astype(str).str.upper()
    if names.eq("ALL").any():
        all_frame = target_scores[names.eq("ALL")]
        if set(all_frame["candidate"].astype(str)) != {"llm", "adaptive"}:
            raise DynamicMacroHouseholdScaleError(f"{cohort} ALL score is not a matched pair")
        return {str(row["candidate"]): float(row["target_score"]) for _, row in all_frame.iterrows()}
    return {str(candidate): float(value) for candidate, value in macro_scores.items()}


def _tag_scores(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "cohort", cohort)
    return result


def _surface_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[list(SURFACE_COLUMNS)].copy()
    return result.sort_values(list(SURFACE_COLUMNS), kind="mergesort").reset_index(drop=True)


def _mechanism_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return _mechanism_projection(value)


def _mechanism_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        lowered = key.lower()
        if key in MECHANISM_EXCLUDED_KEYS or "batch" in lowered or "household" in lowered or "raw_record" in lowered or lowered.endswith("_path"):
            continue
        if isinstance(child, Mapping):
            result[key] = _mechanism_projection(child)
        elif isinstance(child, list):
            result[key] = [_mechanism_projection(item) if isinstance(item, Mapping) else item for item in child]
        else:
            result[key] = child
    return result


def _actual_lock_values(manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any]) -> dict[str, Any]:
    combined = {**normalized_spec, **manifest}
    return {"provider": _first_value(manifest, normalized_spec, "provider"), "model": _first_value(manifest, normalized_spec, "model"), "source": _first_locked_value(combined, SOURCE_LOCK_KEYS), "prompt": _first_locked_value(combined, PROMPT_LOCK_KEYS)}


def _expected_household_count(spec: Mapping[str, Any], cohort: str) -> int:
    lock = _cohort_lock(spec, cohort) or {}
    value = lock.get("household_count", _mapping_value(spec.get("household_counts"), cohort))
    if isinstance(value, bool) or not isinstance(value, int) or value != (81 if cohort == "corrected81" else 200):
        raise DynamicMacroHouseholdScaleError(f"{cohort} household count lock is invalid")
    return value


def _expected_batch_counts(spec: Mapping[str, Any], cohort: str) -> dict[str, int]:
    lock = _cohort_lock(spec, cohort) or {}
    value = lock.get("raw_record_batch_counts", lock.get("batch_counts"))
    if value is None:
        value = spec.get("raw_record_batch_counts")
    if not isinstance(value, Mapping):
        raise DynamicMacroHouseholdScaleError(f"{cohort} raw-record batch count lock is missing")
    if not value or any(isinstance(val, bool) or not isinstance(val, int) for val in value.values()):
        raise DynamicMacroHouseholdScaleError(f"{cohort} raw-record batch count lock is invalid")
    result = {str(key): int(val) for key, val in value.items()}
    if any(val < 1 for val in result.values()):
        raise DynamicMacroHouseholdScaleError(f"{cohort} raw-record batch count lock is invalid")
    return result


def _cohort_lock(spec: Mapping[str, Any], cohort: str) -> Mapping[str, Any] | None:
    runs = spec.get("runs")
    if isinstance(runs, Mapping):
        for alias in COHORT_SPEC_ALIASES[cohort]:
            value = runs.get(alias)
            if isinstance(value, Mapping):
                return value
    for alias in COHORT_SPEC_ALIASES[cohort]:
        value = spec.get(alias)
        if isinstance(value, Mapping):
            return value
    return None


def _reject_stale_output(path: Path) -> None:
    if path.exists():
        raise DynamicMacroHouseholdScaleError(f"Output directory is stale or non-empty: {path}")


def _reject_confirmatory_labels(value: Any, label: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        negated = any(
            marker in lowered
            for marker in (
                "not_confirmatory",
                "non_confirmatory",
                "not confirmatory",
                "not-confirmatory",
            )
        )
        if "confirmatory" in lowered and not negated:
            raise DynamicMacroHouseholdScaleError(f"{label} contains a confirmatory label")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_confirmatory_labels(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_confirmatory_labels(child, label)


def _contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle.lower() in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_string(child, needle) for child in value.values())
    if isinstance(value, list):
        return any(_contains_string(child, needle) for child in value)
    return False


def _contains_false(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return any((name == key and child is False) or _contains_false(child, key) for name, child in value.items())
    if isinstance(value, list):
        return any(_contains_false(child, key) for child in value)
    return False


def _named_bools(value: Any, names: set[str]) -> list[bool]:
    if isinstance(value, Mapping):
        found = [child for name, child in value.items() if name in names and isinstance(child, bool)]
        return found + [item for child in value.values() for item in _named_bools(child, names)]
    if isinstance(value, list):
        return [item for child in value for item in _named_bools(child, names)]
    return []


def _named_ints(value: Any, names: set[str]) -> list[int]:
    if isinstance(value, Mapping):
        found = [int(child) for name, child in value.items() if name in names and isinstance(child, int) and not isinstance(child, bool)]
        return found + [item for child in value.values() for item in _named_ints(child, names)]
    if isinstance(value, list):
        return [item for child in value for item in _named_ints(child, names)]
    return []


def _first_int(value: Mapping[str, Any], *names: str) -> int | None:
    found = _named_ints(value, set(names))
    return found[0] if found else None


def _expected_value(value: Mapping[str, Any], key: str) -> Any:
    return value.get(key)


def _has_any(value: Mapping[str, Any], *keys: str) -> bool:
    return any(value.get(key) is not None for key in keys)


def _first_locked_value(value: Mapping[str, Any], keys: Iterable[str]) -> Any:
    keys = tuple(keys)
    for key in keys:
        if key in value:
            return value[key]
    for group in ("source", "source_lock", "execution_source", "prompt", "prompt_lock", "prompt_contract"):
        child = value.get(group)
        if isinstance(child, Mapping):
            for key in keys:
                if key in child:
                    return child[key]
            if "sha256" in child and any("sha" in key for key in keys):
                return child["sha256"]
            if "version" in child and "prompt_version" in keys:
                return child["version"]
    return None


def _first_value(first: Mapping[str, Any], second: Mapping[str, Any], key: str) -> Any:
    return first.get(key, second.get(key))


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _lookup_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DynamicMacroHouseholdScaleError(f"{label} must be a JSON object")
    return value


def _read_list(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DynamicMacroHouseholdScaleError(f"{label} must be a JSON list of objects")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def _build_report(manifest: Mapping[str, Any], comparison: pd.DataFrame, family: pd.DataFrame, origin: pd.DataFrame) -> str:
    rows = [
        "# Corrected Household-Scale Comparison",
        "",
        f"Claim scope: `{manifest['claim_scope']}`.",
        "",
        f"Winner: `{manifest['winner']}`. The rule minimizes absolute LLM `ALL.rmse_scaled`; a relative gap at or below 1% prefers corrected200.",
        "",
        "Adaptive scores are diagnostic only and do not veto promotion.",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Integrity",
        "",
        "- Both runs reproduce MacroScore from `joined_errors.csv`.",
        "- The Jan-May surface and mechanisms match; only cohort-specific household and batch fields differ.",
        "- June rows and accounting violations were rejected by the validator.",
        "- Historical uncorrected81 evidence is context only and is not eligible for promotion.",
        "",
        f"Family rows: `{len(family)}`; origin rows: `{len(origin)}`.",
        "",
    ]
    return "\n".join(rows)


def _canonical_obj(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
