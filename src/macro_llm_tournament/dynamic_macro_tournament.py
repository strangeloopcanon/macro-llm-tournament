from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, markdown_table


SCHEMA_VERSION = "dynamic_macro_tournament_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
OUTPUT_FILES = (
    "normalized_spec.json",
    "candidate_scores.csv",
    "family_scores.csv",
    "origin_scores.csv",
    "winner_manifest.json",
    "manifest.json",
    "report.md",
)
BEHAVIOR_MODES = {
    "fixed_kernel",
    "schedule",
    "state_schedule",
    "empirical_bridge",
    "empirical_bridge_state_schedule",
}


class DynamicMacroTournamentError(ValueError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a locked developmental recursive-macro tournament.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--mode", choices=("fixture", "live"), required=True)
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_tournament(args)
    except (DynamicMacroTournamentError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(_canonical_json({"output_dir": args.output_dir, "winner": manifest["winner_candidate_id"]}))
    return 0


def run_tournament(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = _resolve_path(args.spec)
    raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = normalize_spec(raw_spec, spec_path=spec_path)
    output_dir = Path(args.output_dir).resolve()
    existing_output = output_dir.exists() and any(output_dir.iterdir())
    if existing_output and not args.resume:
        raise DynamicMacroTournamentError("Tournament output directory must be absent or empty")
    if args.resume and not existing_output:
        raise DynamicMacroTournamentError("--resume requires an existing incomplete tournament output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "live" and int(args.max_live_calls) < int(spec["maximum_authorized_live_calls"]):
        raise DynamicMacroTournamentError(
            f"--max-live-calls must be at least {spec['maximum_authorized_live_calls']} for the locked campaign"
        )
    if args.mode == "fixture" and int(args.max_live_calls) != 0:
        raise DynamicMacroTournamentError("Fixture tournament requires --max-live-calls 0")

    normalized_spec_path = output_dir / "normalized_spec.json"
    if args.resume:
        if (output_dir / "manifest.json").is_file():
            raise DynamicMacroTournamentError("Completed tournament output cannot be resumed")
        if not normalized_spec_path.is_file() or _read_json(normalized_spec_path) != spec:
            raise DynamicMacroTournamentError("Resume spec does not match the existing locked tournament")
    else:
        normalized_spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    candidate_rows: list[dict[str, Any]] = []
    family_frames: list[pd.DataFrame] = []
    origin_frames: list[pd.DataFrame] = []
    used_live_calls = 0
    failed_live_calls = 0
    failed_attempts: list[dict[str, Any]] = []
    for index, candidate in enumerate(spec["candidates"], start=1):
        candidate_id = candidate["candidate_id"]
        candidate_dir = output_dir / "candidates" / candidate_id
        prior_failed_for_candidate = 0
        try:
            if args.resume and _candidate_output_is_complete(candidate_dir):
                resumed_existing = True
            else:
                resumed_existing = False
                if args.resume and candidate_dir.exists():
                    archived = _archive_failed_candidate_attempt(
                        output_dir, candidate_id, candidate_dir
                    )
                    prior_failed_for_candidate = int(archived["live_call_count"])
                    failed_live_calls += prior_failed_for_candidate
                    failed_attempts.append(archived)
                _run_candidate(
                    spec,
                    candidate,
                    mode=args.mode,
                    output_dir=candidate_dir,
                )
            row, family, origin = collect_candidate_result(spec, candidate, candidate_dir)
            row["resumed_existing_result"] = resumed_existing
            row["prior_failed_live_call_count"] = prior_failed_for_candidate
            used_live_calls += int(row["live_call_count"])
            if prior_failed_for_candidate + int(row["live_call_count"]) > int(
                candidate["max_live_calls"]
            ):
                raise DynamicMacroTournamentError(
                    f"Candidate {candidate_id} exceeded its cumulative live-call cap"
                )
            candidate_rows.append(row)
            family.insert(0, "tournament_candidate_id", candidate_id)
            origin.insert(0, "tournament_candidate_id", candidate_id)
            family_frames.append(family)
            origin_frames.append(origin)
        except Exception as exc:
            if args.mode == "live":
                raise DynamicMacroTournamentError(
                    f"Live tournament aborted after candidate {candidate_id} failed: {exc}"
                ) from exc
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "disqualified",
                    "disqualification_reason": str(exc)[:500],
                    "llm_macro_score": math.nan,
                    "adaptive_macro_score": math.nan,
                    "llm_minus_adaptive": math.nan,
                    "llm_direction_accuracy": math.nan,
                    "max_accounting_abs_residual": math.nan,
                    "live_call_count": 0,
                    "replayed_record_count": 0,
                    "resumed_existing_result": False,
                    "prior_failed_live_call_count": prior_failed_for_candidate,
                    "mechanism_complexity": int(candidate["mechanism_complexity"]),
                }
            )
        if used_live_calls + failed_live_calls > int(args.max_live_calls):
            raise DynamicMacroTournamentError(
                "Observed live calls exceeded the tournament authorization"
            )
        print(f"dynamic_macro_tournament progress: {index}/{len(spec['candidates'])} {candidate_id}", flush=True)

    scores = pd.DataFrame(candidate_rows)
    winner = select_winner(scores)
    winner_id = str(winner["candidate_id"])
    winner_child_manifest = json.loads(
        (output_dir / "candidates" / winner_id / "manifest.json").read_text(encoding="utf-8")
    )
    winner_manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": winner_id,
        "selection_rule": spec["selection_rule"],
        "candidate": next(row for row in spec["candidates"] if row["candidate_id"] == winner_id),
        "development_result": winner.to_dict(),
        "child_manifest_sha256": _file_sha256(output_dir / "candidates" / winner_id / "manifest.json"),
        "child_normalized_spec_sha256": winner_child_manifest["normalized_spec_sha256"],
        "reserved_confirmatory": spec["reserved_confirmatory"],
    }
    scores.to_csv(output_dir / "candidate_scores.csv", index=False)
    _concat_or_empty(family_frames).to_csv(output_dir / "family_scores.csv", index=False)
    _concat_or_empty(origin_frames).to_csv(output_dir / "origin_scores.csv", index=False)
    (output_dir / "winner_manifest.json").write_text(
        json.dumps(_jsonable(winner_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "mode": args.mode,
        "normalized_spec_sha256": _sha256_json(spec),
        "candidate_count": len(spec["candidates"]),
        "eligible_candidate_count": int(scores["status"].eq("complete").sum()),
        "winner_candidate_id": winner_id,
        "live_call_count": int(used_live_calls + failed_live_calls),
        "successful_live_call_count": int(used_live_calls),
        "failed_live_call_count": int(failed_live_calls),
        "failed_attempts": failed_attempts,
        "maximum_authorized_live_calls": int(args.max_live_calls),
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
        "reserved_confirmatory_origin": spec["reserved_confirmatory"]["origin_month"],
        "outputs": {},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(manifest, spec, scores), encoding="utf-8")
    manifest["outputs"] = {
        name: _file_sha256(output_dir / name)
        for name in OUTPUT_FILES
        if name != "manifest.json" and (output_dir / name).is_file()
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def normalize_spec(raw: dict[str, Any], *, spec_path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise DynamicMacroTournamentError(f"Spec schema_version must be {SCHEMA_VERSION}")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DynamicMacroTournamentError("Tournament spec requires a non-empty candidates list")
    shared = raw.get("shared")
    reserved = raw.get("reserved_confirmatory")
    if not isinstance(shared, dict) or not isinstance(reserved, dict):
        raise DynamicMacroTournamentError("Spec requires shared and reserved_confirmatory objects")
    required_shared = ("bundle_dir", "households_csv", "provider", "model", "contamination_policy")
    missing = [key for key in required_shared if not shared.get(key)]
    if missing:
        raise DynamicMacroTournamentError("Shared spec missing: " + ", ".join(missing))
    bundle_dir = _resolve_path(shared["bundle_dir"])
    households_csv = _resolve_path(shared["households_csv"])
    bundle_manifest = _read_json(bundle_dir / "manifest.json")
    if str(bundle_manifest.get("bundle_sha256")) != str(shared.get("bundle_sha256")):
        raise DynamicMacroTournamentError("Development bundle hash does not match the locked spec")
    if _file_sha256(households_csv) != str(shared.get("households_sha256")):
        raise DynamicMacroTournamentError("Household input hash does not match the locked spec")
    reserved_origin = str(reserved.get("origin_month", ""))
    dev_origins = [str(row["origin_month"]) for row in bundle_manifest.get("origins", [])]
    if not reserved_origin or reserved_origin in dev_origins:
        raise DynamicMacroTournamentError("Reserved confirmatory origin must not be present in the development bundle")
    if dev_origins and reserved_origin <= max(dev_origins):
        raise DynamicMacroTournamentError("Reserved confirmatory origin must follow all development origins")
    semantics = bundle_manifest.get("target_observation_semantics")
    if not isinstance(semantics, dict) or semantics.get("rule") != "common_origin_observation_month":
        raise DynamicMacroTournamentError(
            "Development bundle must use common-observation-month targets"
        )
    score_start = str(shared.get("score_origin_start", ""))
    score_end = str(shared.get("score_origin_end", ""))
    if score_start not in dev_origins or score_end not in dev_origins:
        raise DynamicMacroTournamentError("Shared score-origin range must use development origins")
    if dev_origins.index(score_start) > dev_origins.index(score_end):
        raise DynamicMacroTournamentError("Shared score-origin range is reversed")
    if dev_origins.index(score_start) == 0:
        raise DynamicMacroTournamentError("Development tournament requires at least one warm-up origin")

    normalized_candidates = []
    seen: set[str] = set()
    total_cap = 0
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            raise DynamicMacroTournamentError("Each candidate must be an object")
        candidate_id = str(raw_candidate.get("candidate_id", ""))
        if not CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in seen:
            raise DynamicMacroTournamentError(f"Invalid or duplicate candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        behavior_mode = str(raw_candidate.get("behavior_policy_mode", "fixed_kernel"))
        if behavior_mode not in BEHAVIOR_MODES:
            raise DynamicMacroTournamentError(f"Unsupported behavior mode for {candidate_id}: {behavior_mode}")
        candidate = {
            "candidate_id": candidate_id,
            "behavior_policy_mode": behavior_mode,
            "behavior_policy_profile": _locked_optional_path(raw_candidate.get("behavior_policy_profile")),
            "empirical_bridge_json": _locked_optional_path(raw_candidate.get("empirical_bridge_json")),
            "behavior_policy_state_profile_json": _locked_optional_path(
                raw_candidate.get("behavior_policy_state_profile_json")
            ),
            "replay_prefix_raw_records_json": _locked_optional_path(
                raw_candidate.get("replay_prefix_raw_records_json")
            ),
            "replay_prefix_period_count": int(
                raw_candidate.get("replay_prefix_period_count", 0)
            ),
            "hybrid_state_weight": float(raw_candidate.get("hybrid_state_weight", 1.0)),
            "feedback_mode": str(raw_candidate.get("feedback_mode", "closed_loop")),
            "feedback_gain": float(raw_candidate.get("feedback_gain", 1.0)),
            "policy_rate_smoothing": float(
                raw_candidate.get("policy_rate_smoothing", 0.0)
            ),
            "policy_state_mode": str(
                raw_candidate.get("policy_state_mode", "recursive")
            ),
            "belief_gain_global": float(raw_candidate.get("belief_gain_global", 1.0)),
            "belief_gain_inflation": float(raw_candidate.get("belief_gain_inflation", 1.0)),
            "belief_gain_income": float(raw_candidate.get("belief_gain_income", 1.0)),
            "belief_gain_unemployment": float(raw_candidate.get("belief_gain_unemployment", 1.0)),
            "household_flow_anchor": str(raw_candidate.get("household_flow_anchor", "origin_saving_rate")),
            "mechanism_complexity": int(raw_candidate.get("mechanism_complexity", 0)),
            "max_live_calls": int(raw_candidate.get("max_live_calls", len(dev_origins) * 3)),
        }
        numeric = [
            candidate["hybrid_state_weight"],
            candidate["feedback_gain"],
            candidate["policy_rate_smoothing"],
            candidate["belief_gain_global"],
            candidate["belief_gain_inflation"],
            candidate["belief_gain_income"],
            candidate["belief_gain_unemployment"],
        ]
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
            raise DynamicMacroTournamentError(f"Candidate {candidate_id} has invalid numeric controls")
        if candidate["policy_rate_smoothing"] > 1.0:
            raise DynamicMacroTournamentError(
                f"Candidate {candidate_id} policy_rate_smoothing must not exceed one"
            )
        if candidate["policy_state_mode"] not in {"recursive", "origin_visible"}:
            raise DynamicMacroTournamentError(
                f"Candidate {candidate_id} has invalid policy_state_mode"
            )
        if bool(candidate["replay_prefix_raw_records_json"]) != bool(
            candidate["replay_prefix_period_count"]
        ):
            raise DynamicMacroTournamentError(
                f"Candidate {candidate_id} must declare both replay prefix path and period count"
            )
        total_cap += candidate["max_live_calls"]
        normalized_candidates.append(candidate)
    normalized_candidates.sort(key=lambda row: row["candidate_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(raw.get("run_id", "dynamic_macro_tournament")),
        "claim_scope": "developmental_model_selection_not_confirmatory",
        "source_spec_sha256": _file_sha256(spec_path),
        "shared": {
            **shared,
            "bundle_dir": _repo_relative(bundle_dir),
            "households_csv": _repo_relative(households_csv),
            "bundle_manifest_file_sha256": _file_sha256(bundle_dir / "manifest.json"),
            "households_sha256": _file_sha256(households_csv),
        },
        "candidates": normalized_candidates,
        "selection_rule": {
            "primary": "minimum_llm_family_equal_macro_score",
            "tie_breakers": ["higher_llm_direction_accuracy", "lower_mechanism_complexity", "candidate_id"],
            "hard_gates": ["complete_child_output", "accounting_tolerance", "input_provenance"],
            "adaptive_role": "diagnostic_only_not_a_selection_veto",
        },
        "maximum_authorized_live_calls": total_cap,
        "reserved_confirmatory": reserved,
    }


def _run_candidate(spec: dict[str, Any], candidate: dict[str, Any], *, mode: str, output_dir: Path) -> None:
    shared = spec["shared"]
    _assert_locked_paths_unchanged(candidate)
    child_mode = (
        "replay_live"
        if mode == "live" and candidate.get("replay_prefix_raw_records_json")
        else mode
    )
    command = [
        sys.executable,
        "-m",
        "macro_llm_tournament.dynamic_macro_economy",
        "--bundle-dir",
        shared["bundle_dir"],
        "--households-csv",
        shared["households_csv"],
        "--mode",
        child_mode,
        "--provider",
        str(shared["provider"]),
        "--model",
        str(shared["model"]),
        "--contamination-policy",
        str(shared["contamination_policy"]),
        "--score-origin-start",
        str(shared["score_origin_start"]),
        "--score-origin-end",
        str(shared["score_origin_end"]),
        "--behavior-policy-mode",
        candidate["behavior_policy_mode"],
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
        "--bootstrap-replicates",
        str(shared.get("bootstrap_replicates", 1000)),
        "--bootstrap-seed",
        str(shared.get("bootstrap_seed", 20260709)),
        "--max-live-calls",
        str(candidate["max_live_calls"] if mode == "live" else 0),
        "--output-dir",
        str(output_dir),
    ]
    if mode == "live":
        command.append("--fresh-cache")
    replay_prefix = candidate.get("replay_prefix_raw_records_json")
    if replay_prefix and mode == "live":
        command.extend(
            [
                "--raw-records-json",
                replay_prefix["path"],
                "--replay-prefix-period-count",
                str(candidate["replay_prefix_period_count"]),
            ]
        )
    for option, key in (
        ("--behavior-policy-profile", "behavior_policy_profile"),
        ("--empirical-bridge-json", "empirical_bridge_json"),
        ("--behavior-policy-state-profile-json", "behavior_policy_state_profile_json"),
    ):
        if candidate.get(key):
            command.extend([option, candidate[key]["path"]])
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise DynamicMacroTournamentError(
            f"Candidate {candidate['candidate_id']} failed: {(result.stderr or result.stdout).strip()[:1000]}"
        )
    _assert_locked_paths_unchanged(candidate)


def collect_candidate_result(
    spec: dict[str, Any], candidate: dict[str, Any], candidate_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = candidate_dir / "manifest.json"
    required = (manifest_path, candidate_dir / "family_scores.csv", candidate_dir / "origin_scores.csv")
    if any(not path.is_file() for path in required):
        raise DynamicMacroTournamentError("Child output contract is incomplete")
    manifest = _read_json(manifest_path)
    child_spec_path = candidate_dir / "normalized_spec.json"
    if not child_spec_path.is_file():
        raise DynamicMacroTournamentError("Child normalized spec is missing")
    child_spec = _read_json(child_spec_path)
    if _sha256_json(child_spec) != str(manifest.get("normalized_spec_sha256")):
        raise DynamicMacroTournamentError("Child normalized spec hash mismatch")
    shared = spec["shared"]
    if manifest.get("status") != "complete" or manifest.get("bundle_sha256") != shared["bundle_sha256"]:
        raise DynamicMacroTournamentError("Child manifest status or bundle identity mismatch")
    household = manifest.get("household_provenance", {})
    if household.get("raw_input_file_sha256") != shared["households_sha256"]:
        raise DynamicMacroTournamentError("Child household provenance mismatch")
    _assert_locked_paths_unchanged(candidate)
    _assert_child_spec_matches_candidate(shared, candidate, child_spec)
    residual = float(manifest.get("max_accounting_abs_residual", math.inf))
    if not math.isfinite(residual) or residual > ACCOUNTING_TOLERANCE:
        raise DynamicMacroTournamentError(f"Accounting residual {residual} exceeds tolerance")
    family = pd.read_csv(candidate_dir / "family_scores.csv")
    origin = pd.read_csv(candidate_dir / "origin_scores.csv")
    llm_family = family[family["candidate"].eq("llm")]
    if llm_family.empty:
        raise DynamicMacroTournamentError("Child family scores lack the LLM candidate")
    direction = float(llm_family["direction_accuracy"].mean())
    recomputed_scores = {
        str(name): float(math.sqrt(group["family_mean_squared_scaled_error"].mean()))
        for name, group in family.groupby("candidate")
    }
    if set(recomputed_scores) != {"llm", "adaptive"}:
        raise DynamicMacroTournamentError("Child family table cannot reproduce matched macro scores")
    scores = manifest.get("macro_scores", {})
    if any(
        not math.isclose(
            float(scores.get(name, math.nan)),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, value in recomputed_scores.items()
    ):
        raise DynamicMacroTournamentError("Child manifest macro scores do not reproduce from family scores")
    expected_delta = recomputed_scores["llm"] - recomputed_scores["adaptive"]
    if not math.isclose(
        float(manifest.get("llm_minus_adaptive", math.nan)),
        expected_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DynamicMacroTournamentError("Child LLM-minus-adaptive score is inconsistent")
    return (
        {
            "candidate_id": candidate["candidate_id"],
            "status": "complete",
            "disqualification_reason": "",
            "llm_macro_score": float(scores["llm"]),
            "adaptive_macro_score": float(scores["adaptive"]),
            "llm_minus_adaptive": float(manifest["llm_minus_adaptive"]),
            "llm_direction_accuracy": direction,
            "max_accounting_abs_residual": residual,
            "live_call_count": int(manifest.get("live_call_count", 0)),
            "replayed_record_count": int(manifest.get("replayed_record_count", 0)),
            "mechanism_complexity": int(candidate["mechanism_complexity"]),
        },
        family,
        origin,
    )


def select_winner(scores: pd.DataFrame) -> pd.Series:
    eligible = scores[scores["status"].eq("complete")].copy()
    if eligible.empty:
        reasons = "; ".join(scores["disqualification_reason"].astype(str))
        raise DynamicMacroTournamentError(f"All candidates were disqualified: {reasons}")
    eligible = eligible.sort_values(
        ["llm_macro_score", "llm_direction_accuracy", "mechanism_complexity", "candidate_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return eligible.iloc[0]


def build_report(manifest: dict[str, Any], spec: dict[str, Any], scores: pd.DataFrame) -> str:
    display = scores[
        [
            "candidate_id",
            "status",
            "llm_macro_score",
            "adaptive_macro_score",
            "llm_minus_adaptive",
            "llm_direction_accuracy",
            "mechanism_complexity",
        ]
    ]
    return "\n".join(
        [
            "# Dynamic Macro Tournament",
            "",
            "## Result",
            f"- Winner: `{manifest['winner_candidate_id']}`.",
            f"- Development candidates: `{manifest['candidate_count']}`; eligible: `{manifest['eligible_candidate_count']}`.",
            f"- Live calls: `{manifest['live_call_count']}`.",
            "- Adaptive expectations is a strong diagnostic benchmark, not a development veto.",
            f"- `{spec['reserved_confirmatory']['origin_month']}` remained outside this tournament.",
            "",
            "## Scores",
            markdown_table(display),
            "",
            "## Scope",
            "This is developmental model selection on frozen January-May 2026 first-release targets. It is not confirmatory evidence.",
            "",
        ]
    )


def _locked_optional_path(value: Any) -> dict[str, str] | None:
    if value in (None, ""):
        return None
    path = _resolve_path(value)
    if not path.is_file():
        raise DynamicMacroTournamentError(f"Locked profile does not exist: {path}")
    return {"path": _repo_relative(path), "sha256": _file_sha256(path)}


def _assert_locked_paths_unchanged(candidate: dict[str, Any]) -> None:
    for key in (
        "behavior_policy_profile",
        "empirical_bridge_json",
        "behavior_policy_state_profile_json",
        "replay_prefix_raw_records_json",
    ):
        locked = candidate.get(key)
        if locked and _file_sha256(_resolve_path(locked["path"])) != locked["sha256"]:
            raise DynamicMacroTournamentError(
                f"Locked candidate input changed after normalization: {key}"
            )


def _candidate_output_is_complete(candidate_dir: Path) -> bool:
    required = (
        candidate_dir / "manifest.json",
        candidate_dir / "normalized_spec.json",
        candidate_dir / "family_scores.csv",
        candidate_dir / "origin_scores.csv",
    )
    if any(not path.is_file() for path in required):
        return False
    try:
        return _read_json(candidate_dir / "manifest.json").get("status") == "complete"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _archive_failed_candidate_attempt(
    output_dir: Path, candidate_id: str, candidate_dir: Path
) -> dict[str, Any]:
    failed_root = output_dir / "failed_attempts"
    failed_root.mkdir(parents=True, exist_ok=True)
    attempt_number = 1 + len(list(failed_root.glob(f"{candidate_id}_attempt_*")))
    archived = failed_root / f"{candidate_id}_attempt_{attempt_number}"
    candidate_dir.replace(archived)
    live_cache_records: list[dict[str, Any]] = []
    for path in sorted(archived.glob(".cache/**/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("cache_hit") is False
            and payload.get("response_created_utc")
        ):
            live_cache_records.append(
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "sha256": _file_sha256(path),
                    "provider": payload.get("provider"),
                    "model": payload.get("model"),
                    "response_created_utc": payload.get("response_created_utc"),
                }
            )
    return {
        "candidate_id": candidate_id,
        "attempt_number": attempt_number,
        "archive_path": str(archived.relative_to(output_dir)),
        "live_call_count": len(live_cache_records),
        "live_cache_records": live_cache_records,
    }


def _assert_child_spec_matches_candidate(
    shared: dict[str, Any], candidate: dict[str, Any], child: dict[str, Any]
) -> None:
    expected_controls = {
        "feedback_mode": candidate["feedback_mode"],
        "feedback_gain": candidate["feedback_gain"],
        "policy_rate_smoothing": candidate["policy_rate_smoothing"],
        "policy_state_mode": candidate["policy_state_mode"],
        "behavior_policy_mode": candidate["behavior_policy_mode"],
    }
    for key, expected in expected_controls.items():
        if child.get(key) != expected:
            raise DynamicMacroTournamentError(f"Child normalized spec changed {key}")
    expected_gains = {
        "global": candidate["belief_gain_global"],
        "inflation": candidate["belief_gain_inflation"],
        "income": candidate["belief_gain_income"],
        "unemployment": candidate["belief_gain_unemployment"],
        "confidence": 1.0,
    }
    if child.get("belief_gains") != expected_gains:
        raise DynamicMacroTournamentError("Child normalized spec changed belief gains")
    score_contract = child.get("score_origin_contract", {})
    if (
        score_contract.get("score_origin_start") != shared["score_origin_start"]
        or score_contract.get("score_origin_end") != shared["score_origin_end"]
    ):
        raise DynamicMacroTournamentError("Child normalized spec changed the score-origin range")
    replay = child.get("replay_provenance", {})
    locked_replay = candidate.get("replay_prefix_raw_records_json")
    if locked_replay and child.get("mode") == "replay_live":
        if (
            replay.get("raw_records_sha256") != locked_replay["sha256"]
            or int(replay.get("replay_prefix_period_count", -1))
            != int(candidate["replay_prefix_period_count"])
        ):
            raise DynamicMacroTournamentError("Child replay-prefix provenance mismatch")


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise DynamicMacroTournamentError(f"Tournament input must live under the project root: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicMacroTournamentError(f"Expected a JSON object: {path}")
    return value


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


if __name__ == "__main__":
    raise SystemExit(main())
