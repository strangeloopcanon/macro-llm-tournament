"""Fail-closed retrospective comparison of corrected 81- and 200-household runs.

This module only consumes completed dynamic-macro run directories.  It never
reruns a model or modifies a run directory.  The comparison lock owns the
claim scope, cohort sizes, batch contract, provenance locks, and score
surface; the run artifacts own the observations and predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .dynamic_macro_clients import validate_live_attempt_ledger
from .dynamic_macro_economy import OUTPUT_FILES as ECONOMY_OUTPUT_FILES
from .dynamic_macro_economy import score_macro
from .dynamic_macro_household_scale_artifacts import (
    CLAIM_SCOPE,
    OUTPUT_FILES,
    SCHEMA_VERSION,
    _build_result,
    _validate_run_pair,
    _write_result,
)
from .dynamic_macro_household_scale_validation import (
    COHORTS,
    COHORT_SPEC_ALIASES,
    EXPECTED_ORIGINS,
    EXPECTED_SCORE_ORIGINS,
    MECHANISM_EXCLUDED_KEYS,
    PROJECT_ROOT,
    PROMPT_LOCK_KEYS,
    REQUIRED_JOINED_COLUMNS,
    RUN_OUTPUT_FILES,
    SOURCE_LOCK_KEYS,
    SURFACE_COLUMNS,
    DynamicMacroHouseholdScaleError,
    _actual_lock_values,
    _all_scores,
    _attempt_cache_path,
    _canonical_obj,
    _canonical_sha,
    _canonical_json,
    _cohort_lock,
    _contains_string,
    _execution_source_tree_sha256,
    _expected_batch_counts,
    _expected_household_count,
    _expected_value,
    _first_int,
    _first_locked_value,
    _first_value,
    _has_any,
    _is_sha256,
    _lookup_path,
    _mechanism_fingerprint,
    _mechanism_projection,
    _named_bools,
    _named_ints,
    _numeric_frame,
    _read_csv,
    _read_list,
    _read_object,
    _reject_confirmatory_labels,
    _resolve,
    _sha256_file,
    _validate_accounting,
    _validate_joined_surface,
    _validate_prompt_cards,
    _validate_score_lineage,
)
from .frozen_vintage_bundle import load_frozen_vintage_bundle


__all__ = [
    "CLAIM_SCOPE",
    "COHORTS",
    "COHORT_SPEC_ALIASES",
    "ECONOMY_OUTPUT_FILES",
    "EXPECTED_ORIGINS",
    "EXPECTED_SCORE_ORIGINS",
    "MECHANISM_EXCLUDED_KEYS",
    "OUTPUT_FILES",
    "PROJECT_ROOT",
    "PROMPT_LOCK_KEYS",
    "REQUIRED_JOINED_COLUMNS",
    "RUN_OUTPUT_FILES",
    "SCHEMA_VERSION",
    "SOURCE_LOCK_KEYS",
    "SURFACE_COLUMNS",
    "DynamicMacroHouseholdScaleError",
    "compare_runs",
    "main",
    "parse_args",
    "run_comparison",
    "validate_run",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare corrected 81- and 200-household retrospective macro runs."
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--historical-evidence", required=True)
    parser.add_argument("--run-81-dir", "--run81-dir", dest="run81_dir", required=True)
    parser.add_argument(
        "--run-200-dir", "--run200-dir", dest="run200_dir", required=True
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run_comparison(parse_args(argv))
    except (DynamicMacroHouseholdScaleError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        _canonical_json(
            {"output_dir": manifest["output_dir"], "winner": manifest["winner"]}
        )
    )
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
    _validate_locked_path(
        spec.get("historical_evidence"),
        evidence_path,
        evidence_path,
        "historical evidence",
    )
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


def validate_run(
    run_dir: Path | str, *, spec: Mapping[str, Any], cohort: str
) -> dict[str, Any]:
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
    if (
        cohort_lock.get("manifest_sha256")
        and _sha256_file(manifest_path) != cohort_lock["manifest_sha256"]
    ):
        raise DynamicMacroHouseholdScaleError(f"Locked {cohort} manifest hash changed")
    if (
        cohort_lock.get("normalized_spec_sha256")
        and _sha256_file(spec_path) != cohort_lock["normalized_spec_sha256"]
    ):
        raise DynamicMacroHouseholdScaleError(
            f"Locked {cohort} normalized spec hash changed"
        )
    _validate_run_locks(manifest, normalized_spec, spec, cohort)
    _validate_execution_source_contract(manifest, normalized_spec, cohort)
    bundle_targets = _validate_bundle_lock(manifest, normalized_spec, spec, cohort)
    _validate_run_calendar(manifest, normalized_spec, cohort)

    expected_households = _expected_household_count(spec, cohort)
    households = _read_csv(root / "households.csv", f"{cohort} households")
    if len(households) != expected_households:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} household count is {len(households)}, expected {expected_households}"
        )
    if "type_id" in households and households["type_id"].duplicated().any():
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} households contain duplicate type_id values"
        )
    manifest_count = _first_int(
        manifest, "household_count", "normalized_household_count"
    )
    if manifest_count != expected_households:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest household count is invalid"
        )

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
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} population weights are invalid"
        )
    household_ids = tuple(sorted(households["type_id"].astype(str)))
    cohort_households_sha = str(
        normalized_spec.get("household_provenance", {}).get("raw_input_file_sha256", "")
    )
    expected_households_sha = str(cohort_lock.get("households_sha256", ""))
    if expected_households_sha and cohort_households_sha != expected_households_sha:
        raise DynamicMacroHouseholdScaleError(f"{cohort} household input hash changed")

    raw_records = _read_list(root / "raw_records.json", f"{cohort} raw records")
    batch_counts = _validate_raw_records(raw_records, household_ids, spec, cohort)
    live_attempts = _read_list(root / "live_attempts.json", f"{cohort} live attempts")
    call_counts = _validate_live_attempts(
        live_attempts, raw_records, root, manifest, spec, cohort, household_ids
    )
    prompt_cards = _read_csv(root / "prompt_cards.csv", f"{cohort} prompt cards")
    _validate_prompt_cards(prompt_cards, raw_records, household_ids, cohort)
    decisions = _read_csv(root / "decisions.csv", f"{cohort} decisions")
    periods = _read_csv(root / "periods.csv", f"{cohort} periods")
    final_states = _read_csv(
        root / "final_household_states.csv", f"{cohort} final household states"
    )
    accounting = _read_csv(root / "accounting.csv", f"{cohort} accounting")
    max_residual = _validate_accounting(
        accounting, decisions, periods, final_states, households, manifest, cohort
    )

    joined = _read_csv(root / "joined_errors.csv", f"{cohort} joined errors")
    forecasts = _read_csv(root / "forecasts.csv", f"{cohort} forecasts")
    _validate_joined_surface(joined, normalized_spec, manifest, cohort)
    _validate_score_lineage(
        forecasts,
        joined,
        periods,
        normalized_spec,
        bundle_targets,
        cohort,
    )
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


def _validate_locked_path(lock: Any, actual: Path, base: Path, label: str) -> None:
    if lock is None:
        return
    if isinstance(lock, str):
        if not _locked_path_matches(lock, actual):
            raise DynamicMacroHouseholdScaleError(f"Locked {label} path changed")
        return
    if not isinstance(lock, Mapping):
        raise DynamicMacroHouseholdScaleError(f"Locked {label} is invalid")
    path = lock.get("path")
    if isinstance(path, str) and not _locked_path_matches(path, actual):
        raise DynamicMacroHouseholdScaleError(f"Locked {label} path changed")
    expected = lock.get("sha256")
    if expected and _sha256_file(actual) != expected:
        raise DynamicMacroHouseholdScaleError(f"Locked {label} hash changed")


def _locked_path_matches(value: str, actual: Path) -> bool:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve() == actual
    return actual in {(Path.cwd() / path).resolve(), (PROJECT_ROOT / path).resolve()}


def _validate_run_contract(
    root: Path, manifest: Mapping[str, Any], cohort: str
) -> None:
    if manifest.get("status") != "complete":
        raise DynamicMacroHouseholdScaleError(f"{cohort} run is not complete")
    contract = manifest.get("output_contract")
    expected = set(RUN_OUTPUT_FILES)
    if (
        not isinstance(contract, list)
        or set(contract) != expected
        or len(contract) != len(expected)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} output contract is incomplete or contains unknown files"
        )
    outputs = manifest.get("outputs")
    expected_hashes = expected - {"manifest.json"}
    if not isinstance(outputs, Mapping) or set(outputs) != expected_hashes:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} output hashes do not cover the contract"
        )
    for name, expected_hash in outputs.items():
        path = root / str(name)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} output hash mismatch: {name}"
            )
    if any(not (root / name).is_file() for name in RUN_OUTPUT_FILES):
        raise DynamicMacroHouseholdScaleError(f"{cohort} output contract is incomplete")


def _validate_normalized_spec_hash(
    manifest: Mapping[str, Any],
    normalized_spec: Mapping[str, Any],
    path: Path,
    cohort: str,
) -> None:
    expected = manifest.get("normalized_spec_sha256")
    if not isinstance(expected, str) or _canonical_sha(normalized_spec) != expected:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} normalized spec hash does not reproduce"
        )
    if _sha256_file(path) == "":
        raise DynamicMacroHouseholdScaleError(f"{cohort} normalized spec is unreadable")


def _validate_run_calendar(
    manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any], cohort: str
) -> None:
    for payload in (manifest, normalized_spec):
        contract = payload.get("score_origin_contract")
        if not isinstance(contract, Mapping):
            continue
        scored = tuple(str(value) for value in contract.get("scored_origins", []))
        if scored and scored != EXPECTED_SCORE_ORIGINS:
            if any("2026-06" in value for value in scored):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} contains June score rows"
                )
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} score-origin contract is not Feb-May"
            )
        warmup = tuple(str(value) for value in contract.get("warmup_origins", []))
        if warmup and warmup != (EXPECTED_ORIGINS[0],):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} warm-up origin contract changed"
            )


def _validate_run_locks(
    manifest: Mapping[str, Any],
    normalized_spec: Mapping[str, Any],
    spec: Mapping[str, Any],
    cohort: str,
) -> None:
    locks = spec["locks"]
    expected_provider = _expected_value(locks, "provider")
    expected_model = _expected_value(locks, "model")
    actual_provider = _first_value(manifest, normalized_spec, "provider")
    actual_model = _first_value(manifest, normalized_spec, "model")
    if (
        manifest.get("provider") is not None
        and normalized_spec.get("provider") is not None
        and manifest["provider"] != normalized_spec["provider"]
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest/spec provider identity differs"
        )
    if (
        manifest.get("model") is not None
        and normalized_spec.get("model") is not None
        and manifest["model"] != normalized_spec["model"]
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest/spec model identity differs"
        )
    if actual_provider != expected_provider or actual_model != expected_model:
        raise DynamicMacroHouseholdScaleError(f"{cohort} provider/model lock changed")
    if _first_value(
        manifest, normalized_spec, "provider_reasoning_effort"
    ) != locks.get("provider_reasoning_effort"):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} provider reasoning-effort lock changed"
        )
    expected_source = _execution_source_tree_sha256(locks)
    manifest_source = _execution_source_tree_sha256(manifest)
    spec_source = _execution_source_tree_sha256(normalized_spec)
    if not expected_source or not manifest_source or not spec_source:
        raise DynamicMacroHouseholdScaleError(f"{cohort} source tree lock is missing")
    if manifest_source != spec_source:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest/spec source tree identity differs"
        )
    if manifest_source != expected_source:
        raise DynamicMacroHouseholdScaleError(f"{cohort} source tree lock changed")
    expected_prompt = _first_locked_value(locks, PROMPT_LOCK_KEYS)
    manifest_prompt = _first_locked_value(manifest, PROMPT_LOCK_KEYS)
    spec_prompt = _first_locked_value(normalized_spec, PROMPT_LOCK_KEYS)
    if expected_prompt is None or manifest_prompt is None or spec_prompt is None:
        raise DynamicMacroHouseholdScaleError(f"{cohort} prompt lock is missing")
    if manifest_prompt != spec_prompt:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest/spec prompt identity differs"
        )
    if manifest_prompt != expected_prompt:
        raise DynamicMacroHouseholdScaleError(f"{cohort} prompt lock changed")
    mechanism_lock = locks.get("mechanism")
    if isinstance(mechanism_lock, Mapping) and _mechanism_projection(
        normalized_spec
    ) != _canonical_obj(mechanism_lock):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} mechanism does not match the lock"
        )
    if (
        isinstance(mechanism_lock, str)
        and normalized_spec.get(
            "mechanism", normalized_spec.get("behavior_policy_mode")
        )
        != mechanism_lock
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} mechanism does not match the lock"
        )
    mechanism_fields = locks.get("mechanism_fields")
    if isinstance(mechanism_fields, list):
        observed = {
            str(field): _lookup_path(normalized_spec, str(field))
            for field in mechanism_fields
        }
        expected = locks.get("mechanism_values", {})
        if not isinstance(expected, Mapping) or _canonical_obj(
            observed
        ) != _canonical_obj(expected):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} mechanism fields do not match the lock"
            )
    expected_mechanism_sha = locks.get("mechanism_sha256")
    if (
        expected_mechanism_sha
        and _canonical_sha(_mechanism_fingerprint(normalized_spec))
        != expected_mechanism_sha
    ):
        raise DynamicMacroHouseholdScaleError(f"{cohort} mechanism hash changed")


def _validate_execution_source_contract(
    manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any], cohort: str
) -> None:
    contract = manifest.get("execution_source")
    if not isinstance(contract, Mapping):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} embedded execution-source contract is missing"
        )
    files = contract.get("files")
    if (
        contract.get("schema_version") != "macro_llm_source_contract_v1"
        or contract.get("source_root") != "src/macro_llm_tournament"
        or not isinstance(files, Mapping)
        or not files
        or contract.get("file_count") != len(files)
        or any(
            not isinstance(path, str)
            or not path.startswith("src/macro_llm_tournament/")
            for path in files
        )
        or any(not _is_sha256(digest) for digest in files.values())
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} embedded execution-source file map is invalid"
        )
    tree_sha = _canonical_sha(dict(files))
    contract_sha = _canonical_sha(contract)
    if contract.get("tree_sha256") != tree_sha:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} embedded execution-source tree hash does not reproduce"
        )
    if manifest.get("execution_source_tree_sha256") != tree_sha:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} top-level execution-source tree hash does not reproduce"
        )
    if manifest.get("source_contract_sha256") != contract_sha:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} source-contract hash does not reproduce"
        )
    if (
        normalized_spec.get("execution_source_tree_sha256") != tree_sha
        or normalized_spec.get("source_contract_sha256") != contract_sha
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} normalized spec source contract differs"
        )


def _validate_bundle_lock(
    manifest: Mapping[str, Any],
    normalized_spec: Mapping[str, Any],
    spec: Mapping[str, Any],
    cohort: str,
) -> pd.DataFrame:
    locks = spec.get("locks")
    expected = locks.get("bundle_sha256") if isinstance(locks, Mapping) else None
    if not _is_sha256(expected):
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock the frozen bundle"
        )
    if (
        manifest.get("bundle_sha256") != expected
        or normalized_spec.get("bundle_sha256") != expected
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} frozen-bundle identity changed"
        )
    embedded = locks.get("bundle_targets") if isinstance(locks, Mapping) else None
    if embedded is not None:
        if not isinstance(embedded, list) or locks.get(
            "bundle_targets_sha256"
        ) != _canonical_sha(embedded):
            raise DynamicMacroHouseholdScaleError(
                "Comparison spec embedded bundle targets are invalid"
            )
        frame = pd.DataFrame(embedded)
    else:
        bundle_root = _find_locked_bundle(expected, spec)
        try:
            bundle = load_frozen_vintage_bundle(bundle_root)
        except Exception as exc:
            raise DynamicMacroHouseholdScaleError(
                f"Cannot validate locked frozen bundle: {bundle_root}"
            ) from exc
        if bundle.manifest.get("bundle_sha256") != expected:
            raise DynamicMacroHouseholdScaleError(
                "Resolved frozen bundle does not match its lock"
            )
        frame = pd.DataFrame(bundle.targets)
    return _normalize_bundle_targets(frame, cohort)


def _find_locked_bundle(expected_sha: str, spec: Mapping[str, Any]) -> Path:
    locks = spec.get("locks")
    configured = spec.get("bundle_dir")
    if configured is None and isinstance(locks, Mapping):
        configured = locks.get("bundle_dir")
    if isinstance(configured, str):
        path = _resolve(configured)
        if path.is_dir():
            return path
        raise DynamicMacroHouseholdScaleError(
            f"Locked frozen-bundle directory is missing: {path}"
        )
    search_root = PROJECT_ROOT / "work" / "dynamic_macro"
    matches: list[Path] = []
    if search_root.is_dir():
        for manifest_path in sorted(search_root.glob("frozen_*/manifest.json")):
            lowered = str(manifest_path).lower()
            if "june" in lowered or "2026_06" in lowered:
                continue
            try:
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if candidate.get("bundle_sha256") == expected_sha:
                matches.append(manifest_path.parent.resolve())
    if len(matches) != 1:
        raise DynamicMacroHouseholdScaleError(
            "Locked frozen bundle cannot be resolved uniquely; add bundle_dir to the comparison spec"
        )
    return matches[0]


def _normalize_bundle_targets(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    required = {
        "origin_month",
        "as_of_date",
        "target_observation_date",
        "target_name",
        "series_id",
        "family",
        "transform",
        "default_scale",
        "target_value",
        "origin_visible_denominator_value",
    }
    if not required.issubset(frame.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} locked frozen-bundle targets are incomplete"
        )
    result = frame[list(required)].copy()
    for column in required - {
        "default_scale",
        "target_value",
        "origin_visible_denominator_value",
    }:
        result[column] = result[column].astype(str)
    result = _numeric_frame(
        result,
        {"default_scale", "target_value", "origin_visible_denominator_value"},
        cohort,
        "locked bundle targets",
    )
    result = result[result["origin_month"].isin(EXPECTED_SCORE_ORIGINS)].copy()
    keys = ["origin_month", "target_name"]
    if len(result) != 40 or result.duplicated(keys).any():
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} locked frozen bundle must contain ten unique targets per scored origin"
        )
    return result.sort_values(keys, kind="mergesort").reset_index(drop=True)


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
        if (
            period_value not in range(5)
            or batch_index < 0
            or batch_count <= batch_index
        ):
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
        ids = [
            str(belief.get("type_id", ""))
            for belief in beliefs
            if isinstance(belief, Mapping)
        ]
        if (
            len(ids) != len(beliefs)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(expected_ids)
        ):
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
                or row.get("response_sha256") != _canonical_sha(payload)
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
    raw_records: list[dict[str, Any]],
    root: Path,
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
    cohort_lock = _cohort_lock(spec, cohort) or {}
    cap = cohort_lock.get(
        "maximum_live_calls",
        cohort_lock.get(
            "max_live_calls",
            spec.get("maximum_live_calls", spec.get("max_live_calls")),
        ),
    )
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < expected_accepted:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} live-attempt cap is missing or invalid"
        )
    if len(rows) > cap:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} live-attempt ledger exceeds its cap"
        )
    attempt_batches: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["period_index"]), int(row.get("batch_index", 0)))
        attempt_batches.setdefault(key, []).append(row)
    for key, attempts in attempt_batches.items():
        failures = [row for row in attempts if row.get("status") == "failed"]
        successes = [row for row in attempts if row.get("status") == "accepted"]
        if len(failures) > 2 or len(successes) != 1:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} live-attempt batch {key} exceeds the semantic-retry contract"
            )
        if attempts[-1].get("status") != "accepted":
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} live-attempt batch {key} has attempts after acceptance"
            )
    rejected = manifest.get("rejected_semantic_payloads")
    if (
        int(manifest.get("semantic_retry_count", -1)) != len(failed)
        or not isinstance(rejected, list)
        or len(rejected) != len(failed)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} semantic-retry manifest does not reconcile with its ledger"
        )
    raw_by_batch: dict[tuple[int, int], dict[str, Any]] = {}
    for record in raw_records:
        if record.get("candidate") != "llm_belief":
            continue
        try:
            key = (int(record.get("period_index")), int(record.get("batch_index")))
        except (TypeError, ValueError) as exc:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM raw record batch identity is invalid"
            ) from exc
        if key in raw_by_batch:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM raw records contain duplicate batches"
            )
        raw_by_batch[key] = record
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
        for row in period_rows:
            key = (period, int(row["batch_index"]))
            record = raw_by_batch.get(key)
            if record is None:
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} accepted live attempt has no LLM raw record"
                )
            for field in (
                "response_sha256",
                "prompt_payload_sha256",
                "household_type_ids_sha256",
            ):
                if row.get(field) != record.get(field):
                    raise DynamicMacroHouseholdScaleError(
                        f"{cohort} accepted live attempt does not bind {field} to its raw record"
                    )
            if row.get("household_type_ids") != record.get("household_type_ids"):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} accepted live attempt does not bind household IDs to its raw record"
                )
            cache_path = _attempt_cache_path(
                root, row.get("cache_file"), record.get("cache_path"), cohort
            )
            if not cache_path.is_file() or _sha256_file(cache_path) != row.get(
                "cache_file_sha256"
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} accepted live-attempt cache file is missing or changed"
                )
            cache = _read_object(cache_path, f"{cohort} accepted live-attempt cache")
            cache_payload = cache.get("payload")
            if (
                not isinstance(cache_payload, Mapping)
                or cache_payload != record.get("payload")
                or _canonical_sha(cache_payload) != row.get("response_sha256")
                or _canonical_sha(cache_payload) != record.get("response_sha256")
                or cache.get("provider") != row.get("provider")
                or cache.get("model") != row.get("model")
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} accepted cache payload does not bind its raw record and ledger"
                )
    accepted_keys = {
        (int(row["period_index"]), int(row["batch_index"])) for row in accepted
    }
    called_keys = {
        key
        for key, record in raw_by_batch.items()
        if record.get("provider_called") is True
    }
    if called_keys != accepted_keys:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} provider-called LLM raw records do not exactly match accepted live attempts"
        )
    llm_records = list(raw_by_batch.values())
    called_records = sum(
        record.get("provider_called") is True for record in llm_records
    )
    cached_records = sum(
        record.get("cache_hit") is True and record.get("provider_called") is not True
        for record in llm_records
    )
    replayed = int(manifest.get("replayed_record_count", -1))
    cache_hits = int(manifest.get("cache_hit_count", -1))
    if (
        called_records != len(accepted)
        or replayed < 0
        or cache_hits < 0
        or replayed + cache_hits != cached_records
        or called_records + cached_records != len(llm_records)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record replay/cache accounting does not reconcile with the manifest"
        )
    return {
        "live": len(rows),
        "accepted": len(accepted),
        "failed": len(failed),
        "semantic_retries": int(manifest.get("semantic_retry_count", 0)),
        "replayed": int(manifest.get("replayed_record_count", 0)),
        "cache_hits": int(manifest.get("cache_hit_count", 0)),
    }


def _validate_spec(
    spec: Mapping[str, Any], spec_path: Path, evidence_path: Path
) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise DynamicMacroHouseholdScaleError(
            "Unsupported household-scale comparison spec"
        )
    if spec.get("claim_scope") != CLAIM_SCOPE:
        raise DynamicMacroHouseholdScaleError(
            "Household-scale comparison must be developmental and retrospective only"
        )
    _reject_confirmatory_labels(spec, "comparison spec")
    locks = spec.get("locks")
    if not isinstance(locks, Mapping):
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must contain locked provenance values"
        )
    if not _expected_value(locks, "provider") or not _expected_value(locks, "model"):
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock provider and model"
        )
    if _expected_value(locks, "provider_reasoning_effort") != "high":
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock Codex CLI reasoning effort to high"
        )
    if _first_locked_value(locks, SOURCE_LOCK_KEYS) is None:
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock source provenance"
        )
    if _first_locked_value(locks, PROMPT_LOCK_KEYS) is None:
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock the prompt")
    if not _has_any(locks, "mechanism", "mechanism_sha256", "mechanism_fields"):
        raise DynamicMacroHouseholdScaleError("Comparison spec must lock the mechanism")
    for cohort in COHORTS:
        if _cohort_lock(spec, cohort) is None:
            raise DynamicMacroHouseholdScaleError(
                f"Comparison spec is missing {cohort} lock"
            )
    if spec_path == evidence_path:
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec and historical evidence must be separate inputs"
        )


def _validate_historical_evidence(evidence: Mapping[str, Any]) -> None:
    _reject_confirmatory_labels(evidence, "historical evidence")
    if not _contains_string(evidence, "uncorrected81"):
        raise DynamicMacroHouseholdScaleError(
            "Historical evidence must identify uncorrected81 context"
        )
    if not _contains_string(evidence, "context"):
        raise DynamicMacroHouseholdScaleError(
            "Historical uncorrected81 evidence must be context only"
        )
    counts = _named_ints(
        evidence, {"household_count", "respondent_count", "normalized_household_count"}
    )
    if counts and any(value != 81 for value in counts):
        raise DynamicMacroHouseholdScaleError(
            "Historical evidence household count must be 81"
        )
    eligibility = _named_bools(
        evidence, {"eligible_for_promotion", "promotion_eligible"}
    )
    if not eligibility or any(eligibility):
        raise DynamicMacroHouseholdScaleError(
            "Historical uncorrected81 evidence is ineligible for promotion"
        )
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


def _reject_stale_output(path: Path) -> None:
    if path.exists():
        raise DynamicMacroHouseholdScaleError(
            f"Output directory is stale or non-empty: {path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
