from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, PROJECT_ROOT, WORK_ROOT, markdown_table
from .demand_economy import DemandEconomyClient, behavior_policy_manifest, run_demand_economy
from .phase4_matched_twins import (
    PHASE4_VERSION,
    PersonaEcologyReplayDemandClient,
    build_path_comparison,
    build_phase4_clients,
    cards_to_frame,
    default_output_mapping,
    economy_proxy_forecasts,
    load_persona_ecology_bundle,
    load_phase4_behavior_policy_profile,
    load_phase4_households,
    mapping_sha256,
    max_accounting_abs_residual,
    normalized_mapping_payload,
    phase4_scenario_with_feedback_multiplier,
    phase4_scoring_targets,
)
from .postcutoff_behavior_gate import build_postcutoff_behavior_cards, load_proxy_data, score_proxy_forecasts


TOURNAMENT_VERSION = "macro_economy_tournament_v1"
RETROSPECTIVE_ONLY_ERROR = "macro_tournament development search may not use confirmatory scoring surfaces"
CONFIRMATORY_LOCK_ERROR = "macro_tournament confirmatory scoring requires a locked two-candidate spec and one unused score surface"
CONFIRMATORY_REGISTRY_VERSION = "macro_tournament_confirmatory_registry_v1"
DEFAULT_CONFIRMATORY_REGISTRY = PROJECT_ROOT / "reports" / "macro_tournament_confirmatory_registry.json"

CANDIDATE_PAYLOAD_KEYS = [
    "belief_gain_global",
    "belief_gain_inflation",
    "belief_gain_income",
    "belief_gain_unemployment",
    "behavior_mechanism",
    "hybrid_state_weight",
    "feedback_mode",
    "feedback_gain_multiplier",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exploratory macro economy tournament over Phase 4 candidates.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"macro_tournament_{timestamp_slug()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(Path(args.spec))
    result = run_tournament(spec, args=args, output_dir=output_dir)
    write_tournament_outputs(output_dir, result)
    print(output_dir)
    return 0


def load_spec(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--spec must contain a JSON object")
    data["spec_path"] = str(path)
    data["spec_sha256"] = file_sha256(path)
    return data


def run_tournament(spec: dict[str, Any], *, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    validate_spec(spec, args=args)
    normalized_spec = normalize_spec(spec)
    candidates = expand_candidates(spec)
    surfaces = [prepare_surface(surface, spec=spec) for surface in spec["surfaces"]]
    behavior_profile_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    manifest = base_manifest(spec, args=args, normalized_spec=normalized_spec, candidates=candidates, surfaces=surfaces)
    if (
        spec_scoring_label(spec) == "confirmatory"
        and not bool(getattr(args, "allow_test_unfrozen_confirmatory", False))
        and manifest["git"].get("dirty") is not False
    ):
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: clean git state is required")
    manifest["behavior_profiles_by_candidate"] = preload_behavior_profiles(
        candidates,
        surfaces,
        spec=spec,
        cache=behavior_profile_cache,
    )
    reservation: dict[str, Any] | None = None
    if spec_scoring_label(spec) == "confirmatory":
        reservation = reserve_confirmatory_surfaces(spec, output_dir=output_dir, manifest=manifest)
        manifest["confirmatory_registry_record"] = reservation
    candidate_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []

    try:
        for index, candidate in enumerate(candidates, start=1):
            candidate_result = score_candidate(candidate, surfaces, spec=spec, behavior_profile_cache=behavior_profile_cache)
            candidate_rows.append(candidate_result["summary"])
            score_rows.extend(candidate_result["scores"])
            accounting_rows.extend(candidate_result["accounting"])
            if index == 1 or index % 100 == 0 or index == len(candidates):
                print(f"macro_tournament progress: {index}/{len(candidates)} candidates", file=sys.stderr, flush=True)

        candidate_table = pd.DataFrame(candidate_rows)
        score_table = pd.DataFrame(score_rows)
        accounting_table = pd.DataFrame(accounting_rows)
        if spec_scoring_label(spec) == "confirmatory":
            validate_confirmatory_candidate_results(candidate_table, spec)
            validate_confirmatory_score_results(score_table, spec)
        winner = select_winner(candidate_table)
        verdict = "macro_tournament_confirmatory_scored" if spec_scoring_label(spec) == "confirmatory" else "macro_tournament_development_scored"
        manifest.update(
            {
                "status": "ok",
                "verdict": verdict,
                "winner_candidate_id": winner.get("candidate_id"),
                "winner_mean_llm_rmse_scaled": winner.get("mean_llm_rmse_scaled"),
                "winner_behavior_mechanism": winner.get("behavior_mechanism"),
                "winner_is_incumbent": winner.get("is_incumbent"),
                "winner_is_promoted_incumbent": winner.get("is_promoted_incumbent"),
                "disqualified_candidates": int(candidate_table["disqualified"].sum()) if not candidate_table.empty else 0,
                "outputs": output_filenames(),
            }
        )
        report = build_report(manifest, candidate_table, score_table)
    except Exception as exc:
        if reservation is not None:
            fail_confirmatory_reservation(reservation, status="failed_after_reservation", error=str(exc))
        raise
    return {
        "normalized_spec": normalized_spec,
        "manifest": manifest,
        "candidate_table": candidate_table,
        "score_table": score_table,
        "accounting_table": accounting_table,
        "winner": winner,
        "report": report,
    }


def validate_spec(spec: dict[str, Any], *, args: argparse.Namespace) -> None:
    if str(spec.get("schema_version")) != TOURNAMENT_VERSION:
        raise ValueError(f"Unsupported macro tournament schema_version: {spec.get('schema_version')!r}")
    scoring_label = spec_scoring_label(spec)
    if args.max_live_calls != 0:
        raise ValueError("macro_tournament currently consumes replayed Phase 4 inputs only; use --max-live-calls 0")
    if not isinstance(spec.get("surfaces"), list) or not spec["surfaces"]:
        raise ValueError("Tournament spec must include at least one surface")
    if scoring_label == "confirmatory":
        validate_confirmatory_spec(spec, args=args)
        return
    if scoring_label != "retrospective":
        raise ValueError(RETROSPECTIVE_ONLY_ERROR)
    if not isinstance(spec.get("candidate_grid"), dict):
        raise ValueError("Tournament spec must include candidate_grid")
    for surface in spec["surfaces"]:
        if str(surface.get("scoring_label", "retrospective")) != "retrospective":
            raise ValueError(RETROSPECTIVE_ONLY_ERROR)
        if str(surface.get("data_mode", "fred")) == "fred" and str(surface.get("scoring_label")) == "confirmatory":
            raise ValueError(RETROSPECTIVE_ONLY_ERROR)


def validate_confirmatory_spec(spec: dict[str, Any], *, args: argparse.Namespace) -> None:
    if args.mode != "replay":
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    if spec.get("confirmatory_lock") is not True:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    if "candidate_grid" in spec:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    candidates = spec.get("candidate_list")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    # Validate candidate IDs before any scoring or data access.
    expand_candidates(spec)
    surfaces = spec.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != 1:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    surface = surfaces[0]
    if str(surface.get("scoring_label", spec.get("scoring_label"))) != "confirmatory":
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    if str(surface.get("data_mode", "fred")) != "fred":
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    if bool(surface.get("scoreable_only")) is not True:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    score_dates = surface_score_asof_dates(surface)
    if len(score_dates) != 1:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    start = pd.Timestamp(surface["asof_start"])
    end = pd.Timestamp(surface["asof_end"])
    score_date = pd.Timestamp(score_dates[0])
    if score_date < start or score_date > end:
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    registry_path = confirmatory_registry_path(spec)
    if registry_path.resolve() != DEFAULT_CONFIRMATORY_REGISTRY.resolve():
        raise ValueError(CONFIRMATORY_LOCK_ERROR)
    registry = load_confirmatory_registry(registry_path)
    spent = confirmatory_spent_keys(registry)
    keys = confirmatory_surface_keys(spec)
    if spent.intersection(keys):
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: surface already spent")
    if not bool(getattr(args, "allow_test_unfrozen_confirmatory", False)):
        raise ValueError(
            f"{CONFIRMATORY_LOCK_ERROR}: frozen vintage inputs are required and the production loader is not implemented"
        )
    required_targets = spec.get("required_target_names")
    if (
        not isinstance(required_targets, list)
        or not required_targets
        or any(not isinstance(target, str) or not target for target in required_targets)
        or len(set(required_targets)) != len(required_targets)
        or "ALL" in required_targets
    ):
        raise ValueError(CONFIRMATORY_LOCK_ERROR)


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(spec, sort_keys=True))


def expand_candidates(spec: dict[str, Any]) -> list[Candidate]:
    if isinstance(spec.get("candidate_list"), list):
        return expand_explicit_candidates(spec)
    grid = spec["candidate_grid"]
    global_gains = list(grid.get("belief_gain_global", [1.0]))
    inflation_gains = list(grid.get("belief_gain_inflation", [1.0]))
    income_gains = list(grid.get("belief_gain_income", [1.0]))
    unemployment_gains = list(grid.get("belief_gain_unemployment", [1.0]))
    behavior_mechanisms = list(grid.get("behavior_mechanisms", ["empirical_bridge_v5_stabilized"]))
    hybrid_weights = list(grid.get("hybrid_state_weights", [1.0]))
    feedback_modes = list(grid.get("feedback_modes", ["closed_loop"]))
    feedback_multipliers = list(grid.get("feedback_gain_multipliers", [1.0]))
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for global_gain, pi_gain, inc_gain, unemp_gain, behavior, feedback_mode, feedback_multiplier in itertools.product(
        global_gains,
        inflation_gains,
        income_gains,
        unemployment_gains,
        behavior_mechanisms,
        feedback_modes,
        feedback_multipliers,
    ):
        weights = hybrid_weights if behavior == "empirical_bridge_state_schedule" else [None]
        for hybrid_weight in weights:
            payload = {
                "belief_gain_global": float(global_gain),
                "belief_gain_inflation": float(pi_gain),
                "belief_gain_income": float(inc_gain),
                "belief_gain_unemployment": float(unemp_gain),
                "behavior_mechanism": str(behavior),
                "hybrid_state_weight": None if hybrid_weight is None else float(hybrid_weight),
                "feedback_mode": str(feedback_mode),
                "feedback_gain_multiplier": float(feedback_multiplier),
            }
            candidate_id = candidate_id_for(payload)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(Candidate(candidate_id=candidate_id, payload={**payload, "candidate_id": candidate_id}))
    return sorted(candidates, key=lambda candidate: candidate.candidate_id)


def expand_explicit_candidates(spec: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for raw in spec["candidate_list"]:
        if not isinstance(raw, dict):
            raise ValueError("candidate_list entries must be JSON objects")
        missing = [key for key in CANDIDATE_PAYLOAD_KEYS if key not in raw]
        if missing:
            raise ValueError(f"candidate_list entry missing keys: {', '.join(missing)}")
        payload = {
            "belief_gain_global": float(raw["belief_gain_global"]),
            "belief_gain_inflation": float(raw["belief_gain_inflation"]),
            "belief_gain_income": float(raw["belief_gain_income"]),
            "belief_gain_unemployment": float(raw["belief_gain_unemployment"]),
            "behavior_mechanism": str(raw["behavior_mechanism"]),
            "hybrid_state_weight": None if raw.get("hybrid_state_weight") is None else float(raw["hybrid_state_weight"]),
            "feedback_mode": str(raw["feedback_mode"]),
            "feedback_gain_multiplier": float(raw["feedback_gain_multiplier"]),
        }
        candidate_id = candidate_id_for(payload)
        supplied_id = raw.get("candidate_id")
        if supplied_id is not None and str(supplied_id) != candidate_id:
            raise ValueError(f"candidate_list supplied candidate_id {supplied_id!r} but payload hashes to {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_list candidate_id: {candidate_id}")
        seen.add(candidate_id)
        candidates.append(Candidate(candidate_id=candidate_id, payload={**payload, "candidate_id": candidate_id}))
    return candidates


def candidate_id_for(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "cand_" + hashlib.sha256(encoded).hexdigest()[:12]


def prepare_surface(surface: dict[str, Any], *, spec: dict[str, Any]) -> dict[str, Any]:
    surface_args = namespace_for_surface(surface, spec=spec, candidate=None)
    mapping = default_output_mapping()
    mapping_payload = normalized_mapping_payload(mapping)
    frames, data_status = load_proxy_data(
        data_mode=surface_args.data_mode,
        refresh=False,
        cutoff_date=surface_args.cutoff_date,
        asof_end=surface_args.asof_end,
    )
    cards, targets, context = build_postcutoff_behavior_cards(
        frames,
        cutoff_date=surface_args.cutoff_date,
        asof_start=surface_args.asof_start,
        asof_end=surface_args.asof_end,
        history_months=int(surface_args.history_months),
        scoreable_only=bool(surface_args.scoreable_only),
    )
    if not cards:
        raise ValueError(f"Surface {surface.get('surface_id')} built no Phase 4 cards")
    targets = filter_surface_targets_for_score_dates(targets, surface)
    ecology_bundle = load_persona_ecology_bundle(surface_args) if surface_args.belief_source == "persona_ecology_replay" else None
    households = load_phase4_households(surface_args, ecology_bundle=ecology_bundle)
    period_count = max(int(surface_args.period_count), len(cards) + 1, 2)
    scoring_targets = phase4_scoring_targets(targets, mapping)
    if spec_scoring_label(spec) == "confirmatory":
        validate_confirmatory_target_set(scoring_targets, spec)
    return {
        "surface_id": str(surface["surface_id"]),
        "surface_spec": surface,
        "args": surface_args,
        "mapping": mapping,
        "mapping_payload": mapping_payload,
        "mapping_sha256": mapping_sha256(mapping_payload),
        "cards": cards,
        "cards_frame": cards_to_frame(cards),
        "targets": scoring_targets,
        "context": context,
        "data_status": data_status,
        "input_hashes": {
            "proxy_frames_sha256": {
                str(series_id): dataframe_sha256(frame)
                for series_id, frame in sorted(frames.items())
            },
            "persona_ecology_manifest_sha256": (
                ecology_bundle.get("manifest_sha256") if ecology_bundle else None
            ),
            "persona_ecology_panel_sha256": (
                ecology_bundle.get("panel_sha256") if ecology_bundle else None
            ),
            "persona_ecology_predictions_sha256": (
                ecology_bundle.get("predictions_sha256") if ecology_bundle else None
            ),
        },
        "ecology_bundle": ecology_bundle,
        "households": households,
        "period_count": period_count,
        "scored_asof_dates": surface_score_asof_dates(surface),
    }


def filter_surface_targets_for_score_dates(targets: pd.DataFrame, surface: dict[str, Any]) -> pd.DataFrame:
    score_dates = surface_score_asof_dates(surface)
    if not score_dates:
        return targets
    allowed = {pd.Timestamp(value).date().isoformat() for value in score_dates}
    filtered = targets[targets["as_of_date"].astype(str).isin(allowed)].copy()
    if filtered.empty:
        raise ValueError(f"Surface {surface.get('surface_id')} has no targets for score_asof_dates={sorted(allowed)}")
    return filtered.reset_index(drop=True)


def surface_score_asof_dates(surface: dict[str, Any]) -> list[str]:
    raw = surface.get("score_asof_dates", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("score_asof_dates must be a list")
    return [pd.Timestamp(value).date().isoformat() for value in raw]


def validate_confirmatory_target_set(targets: pd.DataFrame, spec: dict[str, Any]) -> None:
    required = {str(value) for value in spec.get("required_target_names", [])}
    required_columns = {"target_name", "target_available", "target_value", "as_of_date"}
    if not required_columns.issubset(targets.columns):
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: missing required targets: malformed target table")
    rows = targets.copy()
    actual = set(rows["target_name"].astype(str))
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"undeclared={','.join(extra)}")
        raise ValueError(
            f"{CONFIRMATORY_LOCK_ERROR}: target contract mismatch: {'; '.join(details)}"
        )
    rows["target_value"] = pd.to_numeric(rows["target_value"], errors="coerce")
    valid = rows[
        rows["target_available"].astype(bool)
        & rows["target_value"].map(np.isfinite)
    ]
    row_counts = rows.groupby(rows["target_name"].astype(str)).size()
    valid_counts = valid.groupby(valid["target_name"].astype(str)).size()
    missing = sorted(
        target
        for target in required
        if int(row_counts.get(target, 0)) != 1 or int(valid_counts.get(target, 0)) != 1
    )
    if rows["as_of_date"].astype(str).nunique() != 1:
        missing.append("single_score_date_contract")
    if missing:
        raise ValueError(
            f"{CONFIRMATORY_LOCK_ERROR}: missing required targets: {', '.join(missing)}"
        )


def validate_confirmatory_candidate_results(candidate_table: pd.DataFrame, spec: dict[str, Any]) -> None:
    expected_candidates = len(spec.get("candidate_list", []))
    expected_surfaces = len(spec.get("surfaces", []))
    complete = (
        candidate_table.shape[0] == expected_candidates
        and not candidate_table.empty
        and not candidate_table["disqualified"].astype(bool).any()
        and candidate_table["surface_count"].astype(int).eq(expected_surfaces).all()
    )
    if not complete:
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: incomplete candidate comparison")


def validate_confirmatory_score_results(score_table: pd.DataFrame, spec: dict[str, Any]) -> None:
    required_columns = {"candidate_id", "source", "target_name", "rmse_scaled", "n"}
    if not required_columns.issubset(score_table.columns):
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: incomplete score rows")
    required_targets = {str(value) for value in spec.get("required_target_names", [])}
    expected_candidates = {
        str(candidate.get("candidate_id"))
        for candidate in spec.get("candidate_list", [])
    }
    target_names = set(score_table["target_name"].astype(str))
    scored_targets = target_names - {"ALL"}
    actual_candidates = set(score_table["candidate_id"].astype(str))
    if scored_targets != required_targets or target_names != required_targets | {"ALL"}:
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: score target contract mismatch")
    if actual_candidates != expected_candidates:
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: incomplete score rows")
    rows = score_table.copy()
    rows["rmse_scaled"] = pd.to_numeric(rows["rmse_scaled"], errors="coerce")
    rows["n"] = pd.to_numeric(rows["n"], errors="coerce")
    complete = True
    for candidate_id in expected_candidates:
        candidate_rows = rows[
            rows["candidate_id"].astype(str).eq(candidate_id)
            & rows["rmse_scaled"].map(np.isfinite)
        ]
        for target_name in required_targets:
            target_rows = candidate_rows[
                candidate_rows["target_name"].astype(str).eq(target_name)
                & candidate_rows["n"].eq(1)
            ]
            adaptive_count = int(target_rows["source"].astype(str).eq("adaptive").sum())
            llm_count = int((~target_rows["source"].astype(str).eq("adaptive")).sum())
            if adaptive_count != 1 or llm_count != 1:
                complete = False
        overall_rows = candidate_rows[candidate_rows["target_name"].astype(str).eq("ALL")]
        valid_overall = overall_rows["n"].eq(len(required_targets))
        adaptive_count = int(
            (valid_overall & overall_rows["source"].astype(str).eq("adaptive")).sum()
        )
        llm_count = int(
            (valid_overall & ~overall_rows["source"].astype(str).eq("adaptive")).sum()
        )
        if adaptive_count != 1 or llm_count != 1:
            complete = False
    if not complete:
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: incomplete score rows")


def locked_confirmatory_scoring_inputs(
    forecasts: pd.DataFrame,
    targets: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_confirmatory_target_set(targets, spec)
    required_targets = {str(value) for value in spec["required_target_names"]}
    locked_forecasts = forecasts[
        forecasts["target_name"].astype(str).isin(required_targets)
    ].copy()
    locked_targets = targets[
        targets["target_name"].astype(str).isin(required_targets)
    ].copy()
    return locked_forecasts, locked_targets


def dataframe_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    if not canonical.empty:
        canonical = canonical.sort_values(list(canonical.columns), kind="mergesort").reset_index(drop=True)
    encoded = canonical.to_csv(
        index=False,
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def namespace_for_surface(surface: dict[str, Any], *, spec: dict[str, Any], candidate: Candidate | None) -> SimpleNamespace:
    payload = candidate.payload if candidate is not None else {}
    profiles = spec.get("profiles", {})
    behavior_mode, bridge_json, state_json, hybrid_weight = behavior_args(payload, profiles)
    return SimpleNamespace(
        provider=str(surface.get("provider", spec.get("provider", "codex_cli"))),
        model=str(surface.get("model", spec.get("model", "gpt-5.5"))),
        mode="replay" if str(surface.get("belief_source", "persona_ecology_replay")) == "persona_ecology_replay" else "fixture",
        data_mode=str(surface.get("data_mode", "fred")),
        refresh_fred=False,
        cutoff_date=str(surface.get("cutoff_date", "2025-12-01")),
        asof_start=str(surface["asof_start"]),
        asof_end=str(surface["asof_end"]),
        history_months=int(surface.get("history_months", 18)),
        scoreable_only=bool(surface.get("scoreable_only", False)),
        scoring_label=str(surface.get("scoring_label", spec.get("scoring_label", "retrospective"))),
        belief_source=str(surface.get("belief_source", "persona_ecology_replay")),
        persona_ecology_dir=str(surface.get("persona_ecology_dir", "")),
        primary_ecology_source=str(surface.get("primary_ecology_source", "")),
        ecology_period_policy=str(surface.get("ecology_period_policy", "strict")),
        household_source=str(surface.get("household_source", "fixture")),
        household_csv=surface.get("household_csv"),
        household_count=int(surface.get("household_count", 24)),
        period_count=int(surface.get("period_count", 2)),
        feedback_mode=str(payload.get("feedback_mode", surface.get("feedback_mode", "closed_loop"))),
        behavior_policy_mode=behavior_mode,
        behavior_policy_raw_records_json=surface.get("behavior_policy_raw_records_json"),
        behavior_policy_state_profile_json=state_json,
        empirical_bridge_json=bridge_json,
        hybrid_state_weight=hybrid_weight,
        belief_gain_global=float(payload.get("belief_gain_global", 1.0)),
        belief_gain_inflation=float(payload.get("belief_gain_inflation", 1.0)),
        belief_gain_income=float(payload.get("belief_gain_income", 1.0)),
        belief_gain_unemployment=float(payload.get("belief_gain_unemployment", 1.0)),
        feedback_gain_multiplier=float(payload.get("feedback_gain_multiplier", 1.0)),
        max_live_calls=0,
        fresh_cache=False,
        output_dir=None,
    )


def behavior_args(payload: dict[str, Any], profiles: dict[str, Any]) -> tuple[str, str | None, str | None, float]:
    behavior = str(payload.get("behavior_mechanism", "empirical_bridge_v5_stabilized"))
    state_json = profiles.get("state_schedule")
    if behavior == "empirical_bridge_v4":
        return "empirical_bridge", str(profiles["empirical_bridge_v4"]), None, 1.0
    if behavior == "empirical_bridge_v5_stabilized":
        return "empirical_bridge", str(profiles["empirical_bridge_v5_stabilized"]), None, 1.0
    if behavior == "state_schedule":
        return "state_schedule", None, str(state_json), 1.0
    if behavior == "empirical_bridge_state_schedule":
        return (
            "empirical_bridge_state_schedule",
            str(profiles.get("empirical_bridge_v5_stabilized") or profiles.get("empirical_bridge_v4")),
            str(state_json),
            float(payload.get("hybrid_state_weight", 1.0)),
        )
    raise ValueError(f"Unsupported behavior_mechanism: {behavior}")


def score_candidate(
    candidate: Candidate,
    surfaces: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    behavior_profile_cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> dict[str, Any]:
    score_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for surface in surfaces:
        try:
            result = run_candidate_surface(candidate, surface, spec=spec, behavior_profile_cache=behavior_profile_cache)
            score_rows.extend(result["scores"])
            accounting_rows.extend(result["accounting"])
        except Exception as exc:
            errors.append(f"{surface['surface_id']}: {exc}")
    summary = summarize_candidate(candidate, score_rows, accounting_rows, errors, spec=spec)
    return {"summary": summary, "scores": score_rows, "accounting": accounting_rows}


def run_candidate_surface(
    candidate: Candidate,
    surface: dict[str, Any],
    *,
    spec: dict[str, Any],
    behavior_profile_cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> dict[str, Any]:
    args = namespace_for_surface(surface["surface_spec"], spec=spec, candidate=candidate)
    behavior_profile = cached_behavior_policy_profile(args, behavior_profile_cache)
    scenario = phase4_scenario_with_feedback_multiplier(float(args.feedback_gain_multiplier))
    cache_dir = WORK_ROOT / "macro_tournament_phase4_cache" / candidate.candidate_id / surface["surface_id"]
    clients = build_phase4_clients(args, cache_dir=cache_dir, ecology_bundle=surface["ecology_bundle"], period_count=surface["period_count"])
    all_beliefs: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_periods: list[pd.DataFrame] = []
    all_accounting: list[pd.DataFrame] = []
    for client in clients:
        _initial, beliefs, decisions, periods, accounting, _prompt_rows = run_demand_economy(
            surface["households"],
            [scenario],
            client,
            period_count=surface["period_count"],
            feedback_mode=args.feedback_mode,
            behavior_policy_profile=behavior_profile,
        )
        all_beliefs.append(beliefs)
        all_decisions.append(decisions)
        all_periods.append(periods)
        all_accounting.append(accounting)
    beliefs_frame = pd.concat(all_beliefs, ignore_index=True)
    decisions_frame = pd.concat(all_decisions, ignore_index=True)
    periods_frame = pd.concat(all_periods, ignore_index=True)
    accounting_frame = pd.concat(all_accounting, ignore_index=True)
    forecasts = economy_proxy_forecasts(periods_frame, surface["cards"], surface["mapping"])
    scoring_targets = surface["targets"]
    if spec_scoring_label(spec) == "confirmatory":
        forecasts, scoring_targets = locked_confirmatory_scoring_inputs(
            forecasts,
            scoring_targets,
            spec,
        )
    scores, _joined = score_proxy_forecasts(forecasts, scoring_targets)
    score_rows = decorate_score_rows(scores, candidate, surface, behavior_profile)
    accounting_rows = decorate_accounting_rows(accounting_frame, decisions_frame, candidate, surface)
    return {"scores": score_rows, "accounting": accounting_rows, "beliefs": beliefs_frame}


def cached_behavior_policy_profile(
    args: SimpleNamespace,
    cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> dict[str, Any] | None:
    key = (
        args.behavior_policy_mode,
        args.behavior_policy_raw_records_json,
        args.behavior_policy_state_profile_json,
        args.empirical_bridge_json,
        float(args.hybrid_state_weight),
    )
    if key not in cache:
        cache[key] = load_phase4_behavior_policy_profile(args)
    return cache[key]


def preload_behavior_profiles(
    candidates: list[Candidate],
    surfaces: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> dict[str, Any]:
    if not surfaces:
        return {}
    manifests: dict[str, Any] = {}
    for candidate in candidates:
        args = namespace_for_surface(surfaces[0]["surface_spec"], spec=spec, candidate=candidate)
        profile = cached_behavior_policy_profile(args, cache)
        manifests[candidate.candidate_id] = behavior_policy_manifest(
            profile,
            mode=behavior_mode_for_candidate(candidate.payload),
        )
    return manifests


def decorate_score_rows(
    scores: pd.DataFrame,
    candidate: Candidate,
    surface: dict[str, Any],
    behavior_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    behavior_manifest = behavior_policy_manifest(behavior_profile, mode=behavior_mode_for_candidate(candidate.payload))
    for _, row in scores.iterrows():
        out = row.to_dict()
        out.update(
            {
                "candidate_id": candidate.candidate_id,
                "surface_id": surface["surface_id"],
                "behavior_mechanism": candidate.payload["behavior_mechanism"],
                "feedback_mode": candidate.payload["feedback_mode"],
                "feedback_gain_multiplier": candidate.payload["feedback_gain_multiplier"],
                "belief_gain_global": candidate.payload["belief_gain_global"],
                "belief_gain_inflation": candidate.payload["belief_gain_inflation"],
                "belief_gain_income": candidate.payload["belief_gain_income"],
                "belief_gain_unemployment": candidate.payload["belief_gain_unemployment"],
                "behavior_policy_manifest_json": json.dumps(behavior_manifest, sort_keys=True),
            }
        )
        rows.append(out)
    return rows


def decorate_accounting_rows(accounting: pd.DataFrame, decisions: pd.DataFrame, candidate: Candidate, surface: dict[str, Any]) -> list[dict[str, Any]]:
    max_residual = max_accounting_abs_residual(accounting)
    clipped_inputs = count_clipped_bridge_inputs(decisions)
    return [
        {
            "candidate_id": candidate.candidate_id,
            "surface_id": surface["surface_id"],
            "max_accounting_abs_residual": max_residual,
            "empirical_bridge_clipped_inputs": clipped_inputs,
            "accounting_passed": bool(max_residual <= ACCOUNTING_TOLERANCE),
        }
    ]


def summarize_candidate(
    candidate: Candidate,
    score_rows: list[dict[str, Any]],
    accounting_rows: list[dict[str, Any]],
    errors: list[str],
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    scores = pd.DataFrame(score_rows)
    accounting = pd.DataFrame(accounting_rows)
    llm_overall = scores[(scores["target_name"].astype(str) == "ALL") & (scores["source"].astype(str) != "adaptive")] if not scores.empty else pd.DataFrame()
    adaptive_overall = scores[(scores["target_name"].astype(str) == "ALL") & (scores["source"].astype(str) == "adaptive")] if not scores.empty else pd.DataFrame()
    accounting_passed = bool(not accounting.empty and accounting["accounting_passed"].astype(bool).all())
    disqualified = bool(errors or not accounting_passed or llm_overall.empty)
    return {
        **candidate.payload,
        "is_incumbent": is_incumbent(candidate.payload),
        "is_promoted_incumbent": is_promoted_incumbent(candidate.candidate_id, spec),
        "surface_count": int(llm_overall["surface_id"].nunique()) if not llm_overall.empty else 0,
        "mean_llm_rmse_scaled": float(llm_overall["rmse_scaled"].mean()) if not llm_overall.empty else np.nan,
        "mean_llm_direction_accuracy": float(llm_overall["direction_accuracy"].mean()) if "direction_accuracy" in llm_overall else np.nan,
        "mean_adaptive_rmse_scaled": float(adaptive_overall["rmse_scaled"].mean()) if not adaptive_overall.empty else np.nan,
        "llm_minus_adaptive_rmse_scaled": (
            float(llm_overall["rmse_scaled"].mean() - adaptive_overall["rmse_scaled"].mean())
            if not llm_overall.empty and not adaptive_overall.empty
            else np.nan
        ),
        "empirical_bridge_clipped_inputs": int(accounting["empirical_bridge_clipped_inputs"].sum()) if not accounting.empty else 0,
        "max_accounting_abs_residual": float(accounting["max_accounting_abs_residual"].max()) if not accounting.empty else np.nan,
        "simplicity_score": simplicity_score(candidate.payload),
        "disqualified": disqualified,
        "disqualification_reason": "; ".join(errors) if errors else ("" if accounting_passed else "accounting failed"),
    }


def count_clipped_bridge_inputs(decisions: pd.DataFrame) -> int:
    if "empirical_bridge_clipped_inputs_json" not in decisions:
        return 0
    total = 0
    for value in decisions["empirical_bridge_clipped_inputs_json"].dropna():
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            total += sum(1 for clipped in payload.values() if bool(clipped))
    return total


def behavior_mode_for_candidate(payload: dict[str, Any]) -> str:
    behavior = str(payload.get("behavior_mechanism"))
    if behavior.startswith("empirical_bridge") and behavior != "empirical_bridge_state_schedule":
        return "empirical_bridge"
    return behavior


def is_incumbent(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("behavior_mechanism")) == "empirical_bridge_v5_stabilized"
        and float(payload.get("belief_gain_global")) == 1.0
        and float(payload.get("belief_gain_inflation")) == 1.0
        and float(payload.get("belief_gain_income")) == 1.0
        and float(payload.get("belief_gain_unemployment")) == 1.0
        and str(payload.get("feedback_mode")) == "closed_loop"
        and float(payload.get("feedback_gain_multiplier")) == 1.0
    )


def is_promoted_incumbent(candidate_id: str, spec: dict[str, Any]) -> bool:
    promoted = spec.get("promoted_candidate_id")
    if isinstance(promoted, str):
        return candidate_id == promoted
    promoted_many = spec.get("promoted_candidate_ids")
    if isinstance(promoted_many, list):
        return candidate_id in {str(value) for value in promoted_many}
    return False


def simplicity_score(payload: dict[str, Any]) -> float:
    behavior_penalty = {
        "empirical_bridge_v4": 1.0,
        "empirical_bridge_v5_stabilized": 1.1,
        "state_schedule": 1.5,
        "empirical_bridge_state_schedule": 2.0,
    }.get(str(payload.get("behavior_mechanism")), 3.0)
    gain_penalty = sum(
        abs(float(payload[key]) - 1.0)
        for key in ["belief_gain_global", "belief_gain_inflation", "belief_gain_income", "belief_gain_unemployment", "feedback_gain_multiplier"]
    )
    feedback_penalty = 0.25 if str(payload.get("feedback_mode")) != "closed_loop" else 0.0
    return float(behavior_penalty + gain_penalty + feedback_penalty)


def select_winner(candidate_table: pd.DataFrame) -> dict[str, Any]:
    if candidate_table.empty:
        return {}
    eligible = candidate_table[~candidate_table["disqualified"].astype(bool)].copy()
    if eligible.empty:
        return {}
    ordered = eligible.sort_values(
        [
            "mean_llm_rmse_scaled",
            "mean_llm_direction_accuracy",
            "empirical_bridge_clipped_inputs",
            "max_accounting_abs_residual",
            "simplicity_score",
            "candidate_id",
        ],
        ascending=[True, False, True, True, True, True],
    )
    return ordered.iloc[0].to_dict()


def base_manifest(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    normalized_spec: dict[str, Any],
    candidates: list[Candidate],
    surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    scoring_label = spec_scoring_label(spec)
    claim_scope = (
        "locked_confirmatory_score_once"
        if scoring_label == "confirmatory"
        else "exploratory_development_search_only"
    )
    return {
        "schema_version": TOURNAMENT_VERSION,
        "phase4_schema_version": PHASE4_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec.get("run_id"),
        "mode": args.mode,
        "max_live_calls": int(args.max_live_calls),
        "claim_scope": claim_scope,
        "scoring_label": scoring_label,
        "confirmatory_surface_policy": spec.get("confirmatory_surface_policy"),
        "confirmatory_lock": bool(spec.get("confirmatory_lock", False)),
        "confirmatory_spent_registry_path": repo_relative_path(confirmatory_registry_path(spec)),
        "confirmatory_surface_keys": confirmatory_surface_keys(spec) if scoring_label == "confirmatory" else [],
        "spec_path": spec.get("spec_path"),
        "spec_sha256": spec.get("spec_sha256"),
        "normalized_spec_sha256": hashlib.sha256(json.dumps(normalized_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "candidate_count": len(candidates),
        "surface_count": len(surfaces),
        "surface_ids": [surface["surface_id"] for surface in surfaces],
        "scored_asof_dates_by_surface": {surface["surface_id"]: surface["scored_asof_dates"] for surface in surfaces},
        "mapping_sha256_by_surface": {surface["surface_id"]: surface["mapping_sha256"] for surface in surfaces},
        "input_hashes_by_surface": {
            surface["surface_id"]: surface["input_hashes"]
            for surface in surfaces
        },
        "git": git_metadata(),
    }


def build_report(manifest: dict[str, Any], candidate_table: pd.DataFrame, score_table: pd.DataFrame) -> str:
    winner_id = manifest.get("winner_candidate_id")
    winner_rows = candidate_table[candidate_table["candidate_id"].astype(str).eq(str(winner_id))] if not candidate_table.empty else pd.DataFrame()
    top = candidate_table[~candidate_table["disqualified"].astype(bool)].sort_values("mean_llm_rmse_scaled").head(12) if not candidate_table.empty else pd.DataFrame()
    incumbent = candidate_table[candidate_table["is_incumbent"].astype(bool)] if not candidate_table.empty else pd.DataFrame()
    confirmatory = manifest.get("scoring_label") == "confirmatory"
    guardrails = [
        "- Adaptive and persistence-style baselines are diagnostics, not vetoes.",
        "- Accounting, provenance, score labels, and locked mappings are hard constraints.",
        f"- Confirmatory surface policy: `{manifest.get('confirmatory_surface_policy')}`.",
    ]
    if confirmatory:
        guardrails.insert(0, "- This is locked confirmatory scoring, not candidate search.")
        guardrails.append(f"- Spent-surface keys: `{json.dumps(manifest.get('confirmatory_surface_keys', []), sort_keys=True)}`.")
    else:
        guardrails.insert(0, "- This is exploratory model-building, not confirmatory scoring.")
    return "\n".join(
        [
            "# Macro Economy Tournament",
            "",
            "## Bottom Line",
            tournament_bottom_line(winner_rows, incumbent),
            "",
            "## Guardrails",
            *guardrails,
            "",
            "## Winner",
            markdown_table(winner_rows),
            "",
            "## Top Candidates",
            markdown_table(top),
            "",
            "## Score Rows",
            markdown_table(score_table.head(80)),
            "",
        ]
    )


def tournament_bottom_line(winner_rows: pd.DataFrame, incumbent: pd.DataFrame) -> str:
    if winner_rows.empty:
        return "No eligible LLM economy candidate cleared the hard constraints."
    winner = winner_rows.iloc[0]
    if incumbent.empty:
        return (
            f"Winner `{winner['candidate_id']}` scores mean LLM scaled RMSE "
            f"`{winner['mean_llm_rmse_scaled']:.4f}` across the retrospective surfaces."
        )
    inc = incumbent.sort_values("mean_llm_rmse_scaled").iloc[0]
    delta = float(inc["mean_llm_rmse_scaled"]) - float(winner["mean_llm_rmse_scaled"])
    return (
        f"Winner `{winner['candidate_id']}` scores `{winner['mean_llm_rmse_scaled']:.4f}` mean LLM scaled RMSE, "
        f"versus incumbent `{inc['candidate_id']}` at `{inc['mean_llm_rmse_scaled']:.4f}`. "
        f"Improvement over incumbent: `{delta:.4f}`."
    )


def write_tournament_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    reservation = result["manifest"].get("confirmatory_registry_record")
    try:
        (output_dir / "macro_tournament_spec.normalized.json").write_text(
            json.dumps(result["normalized_spec"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2, sort_keys=True), encoding="utf-8")
        result["candidate_table"].to_csv(output_dir / "macro_tournament_candidates.csv", index=False)
        result["score_table"].to_csv(output_dir / "macro_tournament_scores.csv", index=False)
        result["accounting_table"].to_csv(output_dir / "macro_tournament_accounting.csv", index=False)
        (output_dir / "winner_manifest.json").write_text(json.dumps(result["winner"], indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "macro_tournament_report.md").write_text(result["report"], encoding="utf-8")
        if result["manifest"].get("scoring_label") == "confirmatory":
            completed = complete_confirmatory_reservation(reservation, result)
            result["manifest"]["confirmatory_registry_record"] = completed
            atomic_write_json(output_dir / "manifest.json", result["manifest"])
    except Exception as exc:
        if reservation is not None:
            fail_confirmatory_reservation(
                reservation,
                status="output_incomplete",
                error=str(exc),
            )
        raise


def spec_scoring_label(spec: dict[str, Any]) -> str:
    return str(spec.get("scoring_label", "retrospective"))


def confirmatory_registry_path(spec: dict[str, Any]) -> Path:
    raw = spec.get("confirmatory_spent_registry_path") or spec.get("spent_registry_path")
    if raw is None:
        return DEFAULT_CONFIRMATORY_REGISTRY
    path = Path(str(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def confirmatory_surface_keys(spec: dict[str, Any]) -> list[str]:
    if spec_scoring_label(spec) != "confirmatory":
        return []
    keys: list[str] = []
    for surface in spec.get("surfaces", []):
        for score_date in surface_score_asof_dates(surface):
            payload = {
                "schema_version": TOURNAMENT_VERSION,
                "data_mode": str(surface.get("data_mode", "fred")),
                "score_asof_date": score_date,
            }
            keys.append(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
    return keys


def confirmatory_spent_keys(registry: dict[str, Any]) -> set[str]:
    spent = {str(value) for value in registry.get("spent_surface_keys", [])}
    for record in registry.get("records", []):
        if not isinstance(record, dict):
            continue
        for score_dates in record.get("scored_asof_dates_by_surface", {}).values():
            for score_date in score_dates:
                payload = {
                    "schema_version": TOURNAMENT_VERSION,
                    "data_mode": "fred",
                    "score_asof_date": pd.Timestamp(score_date).date().isoformat(),
                }
                spent.add(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
    return spent


def load_confirmatory_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": CONFIRMATORY_REGISTRY_VERSION,
            "spent_surface_keys": [],
            "records": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed confirmatory registry: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Malformed confirmatory registry: {path}")
    if data.get("schema_version") != CONFIRMATORY_REGISTRY_VERSION:
        raise ValueError(f"Unsupported confirmatory registry schema_version: {data.get('schema_version')!r}")
    spent_surface_keys = data.setdefault("spent_surface_keys", [])
    records = data.setdefault("records", [])
    if not isinstance(spent_surface_keys, list) or any(
        not isinstance(value, str) or not value for value in spent_surface_keys
    ):
        raise ValueError(f"Malformed confirmatory registry: {path}")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Malformed confirmatory registry: {path}")
    reservation_ids = []
    for record in records:
        if "reservation_id" not in record:
            continue
        reservation_id = record["reservation_id"]
        if not isinstance(reservation_id, str) or not reservation_id:
            raise ValueError(f"Malformed confirmatory registry: {path}")
        reservation_ids.append(reservation_id)
    if len(reservation_ids) != len(set(reservation_ids)):
        raise ValueError(f"Malformed confirmatory registry: duplicate reservation_id in {path}")
    return data


def build_confirmatory_registry_record(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id"),
        "output_dir": str(output_dir),
        "spec_path": manifest.get("spec_path"),
        "spec_sha256": manifest.get("spec_sha256"),
        "normalized_spec_sha256": manifest.get("normalized_spec_sha256"),
        "registry_path": manifest.get("confirmatory_spent_registry_path"),
        "surface_ids": manifest.get("surface_ids", []),
        "scored_asof_dates_by_surface": manifest.get("scored_asof_dates_by_surface", {}),
        "surface_keys": manifest.get("confirmatory_surface_keys", []),
        "winner_candidate_id": manifest.get("winner_candidate_id"),
        "winner_mean_llm_rmse_scaled": manifest.get("winner_mean_llm_rmse_scaled"),
    }


def reserve_confirmatory_surfaces(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    registry_path = confirmatory_registry_path(spec)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    keys = confirmatory_surface_keys(spec)
    reservation_id = hashlib.sha256(
        json.dumps(
            {
                "run_id": spec.get("run_id"),
                "score_keys": keys,
                "spec_sha256": spec.get("spec_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with registry_lock(registry_path):
        registry = load_confirmatory_registry(registry_path)
        if confirmatory_spent_keys(registry).intersection(keys):
            raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: surface already spent")
        record = {
            "reservation_id": reservation_id,
            "status": "reserved_before_scoring",
            "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": spec.get("run_id"),
            "output_dir": repo_relative_path(output_dir),
            "spec_path": spec.get("spec_path"),
            "spec_sha256": spec.get("spec_sha256"),
            "normalized_spec_sha256": manifest.get("normalized_spec_sha256"),
            "registry_path": repo_relative_path(registry_path),
            "git": manifest.get("git"),
            "mapping_sha256_by_surface": manifest.get("mapping_sha256_by_surface"),
            "input_hashes_by_surface": manifest.get("input_hashes_by_surface"),
            "behavior_profiles_by_candidate": manifest.get("behavior_profiles_by_candidate"),
            "scored_asof_dates_by_surface": {
                str(surface.get("surface_id")): surface_score_asof_dates(surface)
                for surface in spec.get("surfaces", [])
            },
            "surface_keys": keys,
        }
        registry["spent_surface_keys"] = sorted(
            {str(value) for value in registry.get("spent_surface_keys", [])} | set(keys)
        )
        registry.setdefault("records", []).append(record)
        atomic_write_json(registry_path, registry)
    return record


def complete_confirmatory_reservation(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    registry_path = path_from_registry_record(record)
    with registry_lock(registry_path):
        registry = load_confirmatory_registry(registry_path)
        reservation_id = str(record.get("reservation_id"))
        matches = [
            item
            for item in registry.get("records", [])
            if isinstance(item, dict) and str(item.get("reservation_id")) == reservation_id
        ]
        if len(matches) != 1:
            raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: reservation record missing or duplicated")
        completed = build_confirmatory_registry_record(Path(record["output_dir"]), result)
        matches[0].update(completed)
        matches[0]["reservation_id"] = reservation_id
        matches[0]["status"] = "completed"
        matches[0]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        matches[0]["registry_path"] = repo_relative_path(registry_path)
        atomic_write_json(registry_path, registry)
        return dict(matches[0])


def fail_confirmatory_reservation(
    record: dict[str, Any],
    *,
    status: str,
    error: str,
) -> None:
    registry_path = path_from_registry_record(record)
    with registry_lock(registry_path):
        registry = load_confirmatory_registry(registry_path)
        reservation_id = str(record.get("reservation_id"))
        matches = [
            item
            for item in registry.get("records", [])
            if isinstance(item, dict) and str(item.get("reservation_id")) == reservation_id
        ]
        if len(matches) != 1:
            raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: reservation record missing or duplicated")
        matches[0]["status"] = status
        matches[0]["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        matches[0]["error"] = error
        atomic_write_json(registry_path, registry)


def path_from_registry_record(record: dict[str, Any]) -> Path:
    raw = record.get("registry_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{CONFIRMATORY_LOCK_ERROR}: reservation record has no registry path")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


class registry_lock:
    def __init__(self, registry_path: Path):
        self.path = registry_path.with_suffix(registry_path.suffix + ".lock")
        self.handle: Any = None

    def __enter__(self) -> None:
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def output_filenames() -> list[str]:
    return [
        "macro_tournament_spec.normalized.json",
        "manifest.json",
        "macro_tournament_candidates.csv",
        "macro_tournament_scores.csv",
        "macro_tournament_accounting.csv",
        "winner_manifest.json",
        "macro_tournament_report.md",
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
