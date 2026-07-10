"""Fail-closed one-shot runner for the locked June 2026 confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE
from .dynamic_macro_clients import validate_live_attempt_ledger
from .dynamic_macro_common import DynamicMacroError
from .dynamic_macro_economy import OUTPUT_FILES as ECONOMY_OUTPUT_FILES
from .dynamic_macro_economy import score_macro
from .frozen_vintage_bundle import (
    FrozenVintageBundleError,
    validate_frozen_vintage_bundle,
)
from .source_provenance import SourceProvenanceError, validate_source_contract


SCHEMA_VERSION = "dynamic_macro_confirmatory_lock_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINS = tuple(f"2026-{month:02d}-01" for month in range(1, 7))
ROWS_PER_ORIGIN = 10


class DynamicMacroConfirmatoryError(ValueError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked June 2026 confirmation once.")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_confirmation(parse_args(argv))
    except (DynamicMacroConfirmatoryError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = _resolve(args.lock)
    lock = _read_json(lock_path)
    source = _validate_lock(lock)
    candidate = _validate_development(lock)
    bundle = _validate_bundle(lock)
    receipt_path = _resolve(lock["confirmation"]["receipt_path"])
    output_dir = _resolve(args.output_dir)
    if receipt_path.exists():
        raise DynamicMacroConfirmatoryError("Confirmatory receipt already exists; the June surface is spent")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DynamicMacroConfirmatoryError("Confirmatory output directory must be absent or empty")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": _sha(lock_path),
        "bundle": bundle,
        "execution_source": source,
        "output_dir": _relative(output_dir),
    }
    _create_receipt(receipt_path, receipt)
    try:
        _run_child(lock, candidate, output_dir)
        result = _validate_output(lock, bundle, output_dir)
    except Exception as exc:
        receipt.update(status="failed", finished_at_utc=datetime.now(timezone.utc).isoformat(), error=str(exc)[:1000])
        _replace_receipt(receipt_path, receipt)
        if isinstance(exc, DynamicMacroConfirmatoryError):
            raise
        raise DynamicMacroConfirmatoryError(str(exc)) from exc
    receipt.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(), **result)
    _replace_receipt(receipt_path, receipt)
    return receipt


def _validate_lock(lock: dict[str, Any]) -> dict[str, Any]:
    confirmation = lock.get("confirmation")
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise DynamicMacroConfirmatoryError("Unsupported confirmatory lock schema")
    if lock.get("claim_scope") != "future_confirmatory_only":
        raise DynamicMacroConfirmatoryError("Retrospective work cannot be called confirmatory")
    if not isinstance(confirmation, dict):
        raise DynamicMacroConfirmatoryError("Confirmatory lock is incomplete")
    if confirmation.get("bundle_dir") != "work/dynamic_macro/frozen_2026_01_2026_06_common_month_v1":
        raise DynamicMacroConfirmatoryError("Confirmatory bundle path is not locked")
    if tuple(confirmation.get("origins", [])) != ORIGINS or confirmation.get("score_origin_month") != "2026-06-01":
        raise DynamicMacroConfirmatoryError("Confirmatory origins must be January-June with June scored")
    if int(confirmation.get("replay_prefix_period_count", -1)) != 5 or int(confirmation.get("target_rows_per_origin", -1)) != ROWS_PER_ORIGIN:
        raise DynamicMacroConfirmatoryError("Confirmatory replay or target contract changed")
    if (
        confirmation.get("provider") != "codex_cli"
        or confirmation.get("model") != "gpt-5.5"
        or int(confirmation.get("semantic_retry_limit", -1)) != 2
        or int(confirmation.get("max_live_calls", 0)) != 3
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory provider/model/live-call contract changed")
    source_lock = lock.get("execution_source")
    _locked_file(source_lock, "execution source contract")
    try:
        return validate_source_contract(_read_json(_path(source_lock)), PROJECT_ROOT)
    except SourceProvenanceError as exc:
        raise DynamicMacroConfirmatoryError(str(exc)) from exc


def _validate_development(lock: dict[str, Any]) -> dict[str, Any]:
    development = lock.get("development")
    inputs = lock.get("inputs")
    if not isinstance(development, dict) or not isinstance(inputs, dict):
        raise DynamicMacroConfirmatoryError("Confirmatory development lock is incomplete")
    for name in ("spec", "manifest", "winner_manifest", "candidate_manifest", "candidate_spec"):
        _locked_file(development.get(name), name)
    for name in ("bundle_manifest", "households", "profile", "replay_records"):
        _locked_file(inputs.get(name), name)
    manifest = _read_json(_path(development["manifest"]))
    winner = _read_json(_path(development["winner_manifest"]))
    candidate_manifest = _read_json(_path(development["candidate_manifest"]))
    candidate_spec = _read_json(_path(development["candidate_spec"]))
    candidate_id = development.get("winner_candidate_id")
    if manifest.get("status") != "complete" or manifest.get("winner_candidate_id") != candidate_id:
        raise DynamicMacroConfirmatoryError("Development tournament does not record the locked winner")
    if winner.get("candidate_id") != candidate_id or winner.get("candidate") != development.get("candidate"):
        raise DynamicMacroConfirmatoryError("Winner manifest does not match the locked candidate")
    if winner.get("child_manifest_sha256") != _sha(_path(development["candidate_manifest"])):
        raise DynamicMacroConfirmatoryError("Winner child manifest hash mismatch")
    if winner.get("child_normalized_spec_sha256") != candidate_manifest.get("normalized_spec_sha256"):
        raise DynamicMacroConfirmatoryError("Winner child spec identity mismatch")
    if winner.get("child_normalized_spec_sha256") != development.get("candidate_normalized_spec_sha256"):
        raise DynamicMacroConfirmatoryError("Confirmatory child spec identity is not explicitly locked")
    if _canonical_sha(candidate_spec) != candidate_manifest.get("normalized_spec_sha256"):
        raise DynamicMacroConfirmatoryError("Development child normalized spec is invalid")
    candidate = winner["candidate"]
    if candidate.get("replay_prefix_period_count") != 5:
        raise DynamicMacroConfirmatoryError("Winner must have five replay prefix periods")
    if candidate.get("empirical_bridge_json") != inputs["profile"] or candidate.get("replay_prefix_raw_records_json") != inputs["replay_records"]:
        raise DynamicMacroConfirmatoryError("Winner inputs do not match the confirmatory lock")
    if candidate_manifest.get("bundle_sha256") != lock["development"]["bundle_sha256"]:
        raise DynamicMacroConfirmatoryError("Development bundle identity changed")
    scores = development.get("winner_scores")
    result = winner.get("development_result", {})
    if not isinstance(scores, dict) or any(
        not math.isclose(float(result.get(key, math.nan)), float(expected), rel_tol=0.0, abs_tol=1e-15)
        for key, expected in scores.items()
    ):
        raise DynamicMacroConfirmatoryError("Development winner scores do not match the lock")
    if candidate_spec.get("household_provenance", {}).get("raw_input_file_sha256") != inputs["households"]["sha256"]:
        raise DynamicMacroConfirmatoryError("Development household provenance changed")
    return candidate


def _validate_bundle(lock: dict[str, Any]) -> dict[str, Any]:
    root = _resolve(lock["confirmation"]["bundle_dir"])
    manifest_path, origins_path, targets_path = root / "manifest.json", root / "origins.csv", root / "targets.csv"
    if not root.is_dir() or not all(path.is_file() for path in (manifest_path, origins_path, targets_path)):
        raise DynamicMacroConfirmatoryError("Locked June bundle is missing")
    try:
        manifest = validate_frozen_vintage_bundle(root)
    except FrozenVintageBundleError as exc:
        raise DynamicMacroConfirmatoryError(f"Locked June bundle is invalid: {exc}") from exc
    if manifest.get("schema_version") != "frozen_rolling_origin_vintage_bundle_v4" or manifest.get("mode") != "fred":
        raise DynamicMacroConfirmatoryError("June bundle must be a frozen FRED bundle")
    if manifest.get("target_observation_semantics", {}).get("rule") != "common_origin_observation_month":
        raise DynamicMacroConfirmatoryError("June bundle must use common origin observation months")
    expected_origins = [(month, month[:-2] + "15") for month in ORIGINS]
    origins = _csv(origins_path)
    if [(row.get("origin_month"), row.get("as_of_date")) for row in origins] != expected_origins:
        raise DynamicMacroConfirmatoryError("June bundle origins must be exactly January-June 2026")
    if [row.get("origin_month") for row in manifest.get("origins", [])] != list(ORIGINS):
        raise DynamicMacroConfirmatoryError("June bundle manifest origins are invalid")
    targets = _csv(targets_path)
    counts = {month: 0 for month in ORIGINS}
    for row in targets:
        month = row.get("origin_month")
        if month not in counts or row.get("target_observation_date") != month:
            raise DynamicMacroConfirmatoryError("June bundle target observation month is invalid")
        if not row.get("first_release_as_of_date") or row.get("release_detection_method") != "vintage_dates":
            raise DynamicMacroConfirmatoryError("June bundle lacks complete vintage-date first releases")
        counts[month] += 1
    if len(targets) != 60 or any(count != ROWS_PER_ORIGIN for count in counts.values()):
        raise DynamicMacroConfirmatoryError("June bundle must have exactly 60 rows and ten per origin")
    score_target_names = sorted(
        row.get("target_name", "")
        for row in targets
        if row.get("origin_month") == "2026-06-01"
    )
    if len(score_target_names) != ROWS_PER_ORIGIN or any(
        not name for name in score_target_names
    ):
        raise DynamicMacroConfirmatoryError("June bundle target identities are incomplete")
    payloads = manifest.get("payload_sha256", {})
    if payloads.get("origins.csv") != _sha(origins_path) or payloads.get("targets.csv") != _sha(targets_path):
        raise DynamicMacroConfirmatoryError("June bundle payload hash mismatch")
    bundle_sha = str(manifest.get("bundle_sha256", ""))
    if len(bundle_sha) != 64:
        raise DynamicMacroConfirmatoryError("June bundle hash is absent")
    return {
        "bundle_dir": _relative(root),
        "bundle_sha256": bundle_sha,
        "manifest_sha256": _sha(manifest_path),
        "targets_sha256": _sha(targets_path),
        "score_target_names": score_target_names,
    }


def _run_child(lock: dict[str, Any], candidate: dict[str, Any], output_dir: Path) -> None:
    c, inputs = lock["confirmation"], lock["inputs"]
    command = [
        sys.executable,
        "-m",
        "macro_llm_tournament.dynamic_macro_economy",
        "--bundle-dir",
        c["bundle_dir"],
        "--households-csv",
        inputs["households"]["path"],
        "--mode",
        "replay_live",
        "--provider",
        "codex_cli",
        "--model",
        "gpt-5.5",
        "--contamination-policy",
        "unavailable_at_cutoff",
        "--score-origin-start",
        "2026-06-01",
        "--score-origin-end",
        "2026-06-01",
        "--behavior-policy-mode",
        candidate["behavior_policy_mode"],
        "--empirical-bridge-json",
        inputs["profile"]["path"],
        "--hybrid-state-weight",
        str(candidate["hybrid_state_weight"]),
        "--feedback-mode",
        candidate["feedback_mode"],
        "--feedback-gain",
        str(candidate["feedback_gain"]),
        "--policy-rate-smoothing",
        str(candidate["policy_rate_smoothing"]),
        "--policy-state-mode",
        candidate["policy_state_mode"],
        "--policy-state-weight",
        str(candidate["policy_state_weight"]),
        "--belief-gain-global",
        str(candidate["belief_gain_global"]),
        "--belief-gain-inflation",
        str(candidate["belief_gain_inflation"]),
        "--belief-gain-income",
        str(candidate["belief_gain_income"]),
        "--belief-gain-unemployment",
        str(candidate["belief_gain_unemployment"]),
        "--household-flow-anchor",
        candidate["household_flow_anchor"],
        "--raw-records-json",
        inputs["replay_records"]["path"],
        "--replay-prefix-period-count",
        "5",
        "--semantic-retry-limit",
        str(c["semantic_retry_limit"]),
        "--max-live-calls",
        str(c["max_live_calls"]),
        "--fresh-cache",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        raise DynamicMacroConfirmatoryError(f"Confirmatory child failed: {(result.stderr or result.stdout).strip()[:1000]}")


def _validate_output(lock: dict[str, Any], bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    spec_path = root / "normalized_spec.json"
    records_path = root / "raw_records.json"
    attempts_path = root / "live_attempts.json"
    if not all((root / name).is_file() for name in ECONOMY_OUTPUT_FILES):
        raise DynamicMacroConfirmatoryError("Confirmatory child output is incomplete")
    manifest, spec = _read_json(manifest_path), _read_json(spec_path)
    if manifest.get("status") != "complete" or manifest.get("bundle_sha256") != bundle["bundle_sha256"] or _canonical_sha(spec) != manifest.get("normalized_spec_sha256"):
        raise DynamicMacroConfirmatoryError("Confirmatory child manifest identity is invalid")
    if spec.get("mode") != "replay_live" or spec.get("provider") != "codex_cli" or spec.get("model") != "gpt-5.5":
        raise DynamicMacroConfirmatoryError("Confirmatory child provider mode changed")
    expected_source = _read_json(_path(lock["execution_source"]))
    if manifest.get("execution_source") != expected_source:
        raise DynamicMacroConfirmatoryError("Confirmatory executable source provenance changed")
    if spec.get("household_provenance", {}).get("raw_input_file_sha256") != lock["inputs"]["households"]["sha256"]:
        raise DynamicMacroConfirmatoryError("Confirmatory household provenance changed")
    replay = spec.get("replay_provenance", {})
    if replay.get("raw_records_sha256") != lock["inputs"]["replay_records"]["sha256"] or replay.get("replay_prefix_period_count") != 5:
        raise DynamicMacroConfirmatoryError("Confirmatory replay provenance changed")
    development_profile_sha = _read_json(_path(lock["development"]["candidate_manifest"])).get("behavior_policy_content_sha256")
    if not development_profile_sha or spec.get("behavior_policy_content_sha256") != development_profile_sha:
        raise DynamicMacroConfirmatoryError("Confirmatory behavior profile provenance changed")
    contract = manifest.get("score_origin_contract", {})
    if contract.get("scored_origins") != ["2026-06-01"] or contract.get("scored_origin_count") != 1 or manifest.get("scored_target_row_count") != 10:
        raise DynamicMacroConfirmatoryError("Confirmatory child must score only June and ten rows")
    live_call_count = manifest.get("live_call_count")
    if (
        manifest.get("replayed_record_count") != 5
        or not isinstance(live_call_count, int)
        or isinstance(live_call_count, bool)
        or not 1 <= live_call_count <= int(lock["confirmation"]["max_live_calls"])
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory replay/live accounting is invalid")
    accounting = pd.read_csv(root / "accounting.csv")
    required_accounting = {"residual", "abs_residual", "passed"}
    if accounting.empty or not required_accounting.issubset(accounting.columns):
        raise DynamicMacroConfirmatoryError("Confirmatory accounting table is incomplete")
    residual_values = pd.to_numeric(accounting["residual"], errors="coerce")
    absolute_values = pd.to_numeric(accounting["abs_residual"], errors="coerce")
    if (
        residual_values.isna().any()
        or absolute_values.isna().any()
        or not (absolute_values - residual_values.abs()).abs().le(1e-15).all()
        or not accounting["passed"].map(_as_bool).all()
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory accounting table is invalid")
    residual = float(absolute_values.max())
    if (
        not math.isfinite(residual)
        or residual > ACCOUNTING_TOLERANCE
        or not math.isclose(
            residual,
            float(manifest.get("max_accounting_abs_residual", math.inf)),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory accounting residual is invalid")
    records = _read_list(records_path)
    attempts = _read_list(attempts_path)
    try:
        validate_live_attempt_ledger(attempts)
    except DynamicMacroError as exc:
        raise DynamicMacroConfirmatoryError(
            f"Confirmatory live-attempt ledger is invalid: {exc}"
        ) from exc
    if len(attempts) != live_call_count:
        raise DynamicMacroConfirmatoryError(
            "Confirmatory live-attempt ledger does not reconcile with live_call_count"
        )
    accepted_attempts = [row for row in attempts if row["status"] == "accepted"]
    if (
        len(accepted_attempts) != 1
        or accepted_attempts[0]["period_index"] != 5
        or accepted_attempts[0]["provider"] != "codex_cli"
        or accepted_attempts[0]["model"] != "gpt-5.5"
    ):
        raise DynamicMacroConfirmatoryError(
            "Confirmatory live-attempt ledger must contain one accepted June payload"
        )
    llm_records = [row for row in records if row.get("candidate") == "llm_belief"]
    adaptive_records = [row for row in records if row.get("candidate") == "adaptive"]
    llm_periods = [
        row.get("cache_identity", {}).get("period_index") for row in llm_records
    ]
    if (
        len(records) != 12
        or len(llm_records) != 6
        or len(adaptive_records) != 6
        or llm_periods != list(range(6))
        or any(row.get("cache_hit") is not True for row in llm_records[:5])
        or llm_records[5].get("cache_hit") is not False
        or any(row.get("cache_hit") is not True for row in adaptive_records)
    ):
        raise DynamicMacroConfirmatoryError(
            "Confirmatory run must replay five LLM periods, accept one June LLM payload, and retain six adaptive audit records"
        )
    output_contract = manifest.get("output_contract")
    expected_contract = set(ECONOMY_OUTPUT_FILES)
    if (
        not isinstance(output_contract, list)
        or len(output_contract) != len(set(output_contract))
        or set(output_contract) != expected_contract
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory output contract is incomplete or unknown")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != expected_contract - {"manifest.json"}:
        raise DynamicMacroConfirmatoryError("Confirmatory output hashes do not cover the contract")
    for name, expected in outputs.items():
        if not (root / name).is_file() or _sha(root / name) != expected:
            raise DynamicMacroConfirmatoryError(f"Confirmatory output hash mismatch: {name}")

    joined = pd.read_csv(root / "joined_errors.csv")
    required_score_columns = {
        "candidate",
        "origin_month",
        "family",
        "target_name",
        "scaled_squared_error",
        "absolute_scaled_error",
        "direction_correct",
    }
    if (
        not required_score_columns.issubset(joined.columns)
        or len(joined) != 20
        or set(joined["candidate"].astype(str)) != {"llm", "adaptive"}
        or set(joined["origin_month"].astype(str)) != {"2026-06-01"}
        or joined.groupby("candidate").size().to_dict() != {"adaptive": 10, "llm": 10}
        or joined.groupby("candidate")["target_name"].nunique().to_dict() != {"adaptive": 10, "llm": 10}
        or any(
            sorted(group["target_name"].astype(str)) != bundle["score_target_names"]
            for _, group in joined.groupby("candidate")
        )
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory joined score surface is not the locked June matched pair")
    try:
        target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    except (DynamicMacroError, KeyError, ValueError) as exc:
        raise DynamicMacroConfirmatoryError(f"Confirmatory score surface is invalid: {exc}") from exc
    for name, expected_frame, sort_by in (
        ("target_scores.csv", target_scores, ["candidate", "family", "target_name"]),
        ("family_scores.csv", family_scores, ["candidate", "family"]),
        ("origin_scores.csv", origin_scores, ["candidate", "origin_month"]),
    ):
        _assert_score_frame_matches(
            pd.read_csv(root / name), expected_frame, sort_by=sort_by, label=name
        )
    recorded_scores = manifest.get("macro_scores")
    if not isinstance(recorded_scores, dict) or any(
        not math.isclose(
            float(recorded_scores.get(name, math.nan)),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, value in macro_scores.items()
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory macro scores do not reproduce")
    score_delta = macro_scores["llm"] - macro_scores["adaptive"]
    if not math.isclose(
        float(manifest.get("llm_minus_adaptive", math.nan)),
        score_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DynamicMacroConfirmatoryError("Confirmatory score difference does not reproduce")
    return {
        "child_manifest_sha256": _sha(manifest_path),
        "child_normalized_spec_sha256": _sha(spec_path),
        "accounting_sha256": outputs["accounting.csv"],
        "joined_errors_sha256": outputs["joined_errors.csv"],
        "target_scores_sha256": outputs["target_scores.csv"],
        "family_scores_sha256": outputs["family_scores.csv"],
        "origin_scores_sha256": outputs["origin_scores.csv"],
        "report_sha256": outputs["report.md"],
        "raw_records_sha256": outputs["raw_records.json"],
        "live_attempts_sha256": outputs["live_attempts.json"],
        "live_attempt_count": len(attempts),
        "output_hashes": outputs,
        "macro_scores": macro_scores,
        "llm_minus_adaptive": score_delta,
        "max_accounting_abs_residual": residual,
        "replayed_record_count": 5,
        "scored_origin_month": "2026-06-01",
        "scored_target_row_count": 10,
    }


def _assert_score_frame_matches(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: list[str],
    label: str,
) -> None:
    if set(observed.columns) != set(expected.columns):
        raise DynamicMacroConfirmatoryError(f"{label} columns do not reproduce")
    columns = list(expected.columns)
    left = observed[columns].sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    right = expected[columns].sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise DynamicMacroConfirmatoryError(
            f"{label} does not reproduce from joined_errors.csv"
        ) from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _locked_file(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str) or not _path(value).is_file() or _sha(_path(value)) != value["sha256"]:
        raise DynamicMacroConfirmatoryError(f"Locked {label} hash changed or is missing")


def _create_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise DynamicMacroConfirmatoryError("Confirmatory receipt already exists; the June surface is spent") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_receipt(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError as exc:
        raise DynamicMacroConfirmatoryError(f"Path must be inside project root: {path}") from exc


def _path(value: dict[str, str]) -> Path:
    return _resolve(value["path"])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicMacroConfirmatoryError(f"Expected JSON object: {path}")
    return value


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DynamicMacroConfirmatoryError(f"Expected JSON list: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
