"""Canonical recursive monthly macro runner over a frozen vintage bundle.

The runner keeps the information boundary deliberately narrow: only
origin-visible history enters period overrides and prompts. Frozen outcomes are
joined after both matched economies have completed their continuous paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .agent_common import WORK_ROOT, markdown_table
from .demand_economy import (
    DEMAND_ECONOMY_PROMPT_VERSION,
    DemandEconomyClient,
    DemandScenario,
    run_demand_economy,
)
from .dynamic_macro_clients import (
    GainAdjustedDemandClient,
    JournaledLiveDemandClient,
    LiveAttemptJournal,
    ObservedSignalAdaptiveClient,
    ReplayThenLiveDemandClient,
    apply_belief_gains,
    belief_gains_from_args,
    canonical_live_attempts,
    cache_identity,
    observed_signal_adaptive_payload,
    seed_live_cache,
    validate_replay_prefix_records,
)
from .dynamic_macro_common import (
    BundleView,
    DynamicMacroError,
    FORBIDDEN_PROMPT_KEYS,
    _canonical_json,
    _sha256_json,
)
from .dynamic_macro_inputs import (
    CONTAMINATION_LABELS,
    CONTAMINATION_POLICIES,
    MAPPING_BY_SERIES,
    REQUIRED_FAMILIES,
    TargetMapping,
    anchor_household_flows,
    assert_no_prompt_target_leakage,
    build_initial_environment_anchor,
    build_period_overrides,
    canonical_behavior_profile_sha256,
    filter_bundle_targets,
    load_behavior_profile,
    load_bundle_view,
    load_households,
    select_score_origins,
    validate_household_temporal_availability,
    household_input_provenance,
)
from .llm_common import LLMUnavailable
from .source_provenance import build_source_contract


SCHEMA_VERSION = "recursive_dynamic_macro_economy_v3"
PERIODS_PER_YEAR = 12.0
ACCOUNTING_TOLERANCE = 1e-8
DEFAULT_BOOTSTRAP_SEED = 20260709
DEFAULT_BOOTSTRAP_REPLICATES = 1000
OUTPUT_FILES = (
    "normalized_spec.json",
    "manifest.json",
    "households.csv",
    "final_household_states.csv",
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
    "live_attempts.json",
    "report.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched recursive monthly macro economies over a frozen bundle."
    )
    parser.add_argument("--bundle-dir", required=True)
    household_group = parser.add_mutually_exclusive_group(required=True)
    household_group.add_argument(
        "--household-panel", help="Prepared real-SCE respondent panel CSV."
    )
    household_group.add_argument(
        "--households-csv", help="Demand-economy-ready household CSV."
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "replay", "replay_live", "live"),
        default="fixture",
    )
    parser.add_argument("--provider", choices=("codex_cli",), default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--contamination-policy",
        choices=CONTAMINATION_POLICIES,
        default="unavailable_at_cutoff",
        help="Select by first-release availability, strict post-cutoff event date, or explicit all-target scoring.",
    )
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--max-households-per-call", type=int, default=100)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument(
        "--seed-cache-dir",
        help="Validated prior live-cache records to seed into an otherwise fresh output cache.",
    )
    parser.add_argument(
        "--raw-records-json",
        help="Identity-bearing raw records to use instead of replay cache files.",
    )
    parser.add_argument(
        "--replay-prefix-period-count",
        type=int,
        default=0,
        help="In replay_live mode, require raw records for exactly periods 0..N-1 and call Codex thereafter.",
    )
    parser.add_argument(
        "--semantic-retry-limit",
        type=int,
        default=2,
        help="Maximum fresh retries after a structurally invalid live belief payload.",
    )
    parser.add_argument(
        "--score-origin-start",
        help="First origin month included in scoring; earlier origins remain recursive warm-up periods.",
    )
    parser.add_argument(
        "--score-origin-end",
        help="Last origin month included in scoring; later origins remain in the path but are not scored.",
    )
    parser.add_argument(
        "--behavior-policy-mode",
        choices=(
            "fixed_kernel",
            "schedule",
            "state_schedule",
            "empirical_bridge",
            "empirical_bridge_state_schedule",
        ),
        default="fixed_kernel",
    )
    parser.add_argument(
        "--behavior-policy-profile",
        help="Generic profile path for the selected non-hybrid policy mode.",
    )
    parser.add_argument(
        "--behavior-policy-raw-records-json",
        help="Behavior-ecology raw records for schedule mode.",
    )
    parser.add_argument(
        "--behavior-policy-state-profile-json",
        help="State-conditioned policy profile JSON.",
    )
    parser.add_argument(
        "--empirical-bridge-json", help="Accepted empirical bridge profile JSON."
    )
    parser.add_argument("--hybrid-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--feedback-mode", choices=("closed_loop", "none"), default="closed_loop"
    )
    parser.add_argument("--feedback-gain", type=float, default=1.0)
    parser.add_argument(
        "--policy-rate-smoothing",
        type=float,
        default=0.0,
        help="Monthly inertia applied to the Taylor-rule policy target; zero preserves the original rule.",
    )
    parser.add_argument(
        "--policy-state-mode",
        choices=("recursive", "origin_visible"),
        default="recursive",
        help="Use the recursive policy state or assimilate the frozen origin-visible FEDFUNDS state each month.",
    )
    parser.add_argument(
        "--policy-state-weight",
        type=float,
        default=1.0,
        help="Weight on the origin-visible policy observation when policy-state assimilation is enabled.",
    )
    parser.add_argument(
        "--household-flow-anchor",
        choices=("origin_saving_rate", "none"),
        default="origin_saving_rate",
        help="Calibrate the initial aggregate consumption/saving level from origin-visible PSAVERT.",
    )
    parser.add_argument("--belief-gain-global", type=float, default=1.0)
    parser.add_argument("--belief-gain-inflation", type=float, default=1.0)
    parser.add_argument("--belief-gain-income", type=float, default=1.0)
    parser.add_argument("--belief-gain-unemployment", type=float, default=1.0)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-block-length", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_dynamic_macro(args)
    except (DynamicMacroError, LLMUnavailable, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        _canonical_json(
            {
                "output_dir": str(Path(args.output_dir)),
                "macro_scores": manifest["macro_scores"],
            }
        )
    )
    return 0


def run_dynamic_macro(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir)
    if args.mode in {"live", "replay_live"} and output_dir.exists():
        raise DynamicMacroError(
            "Live-style mode refuses to reuse an existing output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_bundle = load_bundle_view(Path(args.bundle_dir))
    bundle, contamination_coverage = filter_bundle_targets(
        loaded_bundle,
        model=args.model,
        policy=args.contamination_policy,
    )
    _validate_replay_live_horizon(args, origin_count=len(bundle.origins))
    score_bundle, score_origin_contract = select_score_origins(
        bundle,
        start=args.score_origin_start,
        end=args.score_origin_end,
    )
    households = load_households(args)
    household_temporal_coverage = validate_household_temporal_availability(
        args,
        households,
        first_origin=bundle.origins[0],
    )
    period_overrides = build_period_overrides(
        bundle,
        policy_state_mode=str(args.policy_state_mode),
    )
    households, household_flow_anchor = anchor_household_flows(
        households,
        period_overrides,
        mode=args.household_flow_anchor,
    )
    household_provenance = household_input_provenance(
        args,
        households,
        temporal_coverage=household_temporal_coverage,
    )
    initial_environment_anchor = build_initial_environment_anchor(period_overrides)
    behavior_profile = load_behavior_profile(args)
    behavior_profile_content_sha256 = canonical_behavior_profile_sha256(
        behavior_profile
    )
    gains = belief_gains_from_args(args)
    execution_source = build_source_contract(Path(__file__).resolve().parents[2])
    raw_replay_records = (
        load_raw_records(Path(args.raw_records_json)) if args.raw_records_json else None
    )

    cache_dir = (
        output_dir / ".cache"
        if args.fresh_cache
        else WORK_ROOT / "dynamic_macro_economy_cache"
    )
    journal_dir = output_dir / ".cache" / "live_attempts"
    seed_cache_provenance = (
        seed_live_cache(
            Path(args.seed_cache_dir),
            cache_dir,
            provider=args.provider,
            model=args.model,
        )
        if args.seed_cache_dir
        else None
    )
    provider_cwd = (
        output_dir / "provider_cwd"
        if args.mode in {"live", "replay", "replay_live"}
        else None
    )
    if provider_cwd is not None:
        provider_cwd.mkdir(parents=True, exist_ok=True)
        if any(provider_cwd.iterdir()):
            raise DynamicMacroError("Isolated provider_cwd must be empty at run start")
    if args.mode == "replay_live":
        llm_base: Any = ReplayThenLiveDemandClient(
            args.provider,
            args.model,
            cache_dir,
            replay_records=raw_replay_records or [],
            replay_prefix_period_count=int(args.replay_prefix_period_count),
            max_live_calls=int(args.max_live_calls),
            semantic_retry_limit=int(args.semantic_retry_limit),
            max_households_per_call=int(args.max_households_per_call),
            execution_cwd=provider_cwd,
            journal_dir=journal_dir,
        )
    else:
        llm_mode = (
            "raw_replay"
            if args.mode == "replay" and raw_replay_records is not None
            else args.mode
        )
        llm_base = DemandEconomyClient(
            args.provider,
            args.model,
            cache_dir,
            mode=llm_mode,
            variant="llm_belief",
            max_live_calls=args.max_live_calls,
            raw_replay_records=raw_replay_records,
            execution_cwd=provider_cwd,
            max_households_per_call=int(args.max_households_per_call),
            semantic_retry_limit=int(args.semantic_retry_limit),
        )
        if args.mode == "live":
            llm_base = JournaledLiveDemandClient(
                llm_base,
                LiveAttemptJournal(
                    journal_dir, provider=args.provider, model=args.model
                ),
            )
    llm_client = GainAdjustedDemandClient(
        llm_base,
        gains=gains,
        requested_mode=args.mode,
        replay_records=raw_replay_records,
        replay_prefix_period_count=int(args.replay_prefix_period_count),
    )
    adaptive_client = ObservedSignalAdaptiveClient()
    scenario = DemandScenario(
        scenario_id="recursive_monthly_path",
        label="Continuous frozen-origin monthly path",
        feedback_gain=float(args.feedback_gain),
        policy_rate_smoothing=float(args.policy_rate_smoothing),
        policy_state_mode=str(args.policy_state_mode),
        policy_state_weight=float(args.policy_state_weight),
        notes="One continuous trajectory; targets are withheld until scoring.",
    )

    result_frames: dict[str, list[pd.DataFrame]] = {
        "initial": [],
        "beliefs": [],
        "decisions": [],
        "periods": [],
        "accounting": [],
    }
    prompt_rows: list[dict[str, Any]] = []
    final_state_frames: list[pd.DataFrame] = []
    for candidate, client in (("llm", llm_client), ("adaptive", adaptive_client)):
        initial, beliefs, decisions, periods, accounting, prompts = run_demand_economy(
            households,
            [scenario],
            client,
            period_count=len(bundle.origins),
            feedback_mode=args.feedback_mode,
            behavior_policy_profile=behavior_profile,
            periods_per_year=PERIODS_PER_YEAR,
            period_overrides=period_overrides,
            initial_environment_override=initial_environment_anchor,
            max_households_per_call=int(args.max_households_per_call),
        )
        for frame_name, frame in (
            ("initial", initial),
            ("beliefs", beliefs),
            ("decisions", decisions),
            ("periods", periods),
            ("accounting", accounting),
        ):
            tagged = frame.copy()
            tagged.insert(0, "candidate", candidate)
            result_frames[frame_name].append(tagged)
        for row in prompts:
            prompt_rows.append({"candidate": candidate, **row})
        final_states = pd.DataFrame(
            getattr(client, "final_household_states", {}).get(scenario.scenario_id, [])
        )
        if final_states.empty:
            raise DynamicMacroError("Demand economy did not expose final household states")
        final_states.insert(0, "candidate", candidate)
        final_state_frames.append(final_states)

    initial_frame = pd.concat(result_frames["initial"], ignore_index=True)
    beliefs_frame = pd.concat(result_frames["beliefs"], ignore_index=True)
    decisions_frame = pd.concat(result_frames["decisions"], ignore_index=True)
    periods_frame = pd.concat(result_frames["periods"], ignore_index=True)
    accounting_frame = pd.concat(result_frames["accounting"], ignore_index=True)
    final_household_states_frame = pd.concat(final_state_frames, ignore_index=True)
    attach_origin_identity(beliefs_frame, bundle.origins)
    attach_origin_identity(decisions_frame, bundle.origins)
    attach_origin_identity(periods_frame, bundle.origins)
    attach_origin_identity(accounting_frame, bundle.origins)

    assert_matched_initial_state(initial_frame)
    assert_complete_periods(periods_frame, bundle.origins)
    assert_accounting(accounting_frame)
    assert_no_prompt_target_leakage(prompt_rows)

    forecasts = build_forecasts(periods_frame, households, score_bundle)
    joined = join_and_score_forecasts(forecasts, score_bundle.targets)
    target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    bootstrap = origin_block_bootstrap(
        joined,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        block_length=args.bootstrap_block_length,
    )
    raw_records = [*llm_client.raw_records, *adaptive_client.raw_records]
    live_attempts = (
        canonical_live_attempts(journal_dir)
        if args.mode in {"live", "replay_live"}
        else []
    )
    if args.mode in {"live", "replay_live"} and int(llm_client.live_call_count) != len(
        live_attempts
    ):
        raise DynamicMacroError(
            "Live-attempt ledger does not reconcile with live_call_count"
        )

    spec = normalized_spec(
        args,
        bundle=bundle,
        gains=gains,
        period_overrides=period_overrides,
        initial_environment_anchor=initial_environment_anchor,
        behavior_profile_content_sha256=behavior_profile_content_sha256,
        household_provenance=household_provenance,
        household_flow_anchor=household_flow_anchor,
        contamination_coverage=contamination_coverage,
        score_origin_contract=score_origin_contract,
        seed_cache_provenance=seed_cache_provenance,
        execution_source=execution_source,
    )
    spec_sha256 = _sha256_json(spec)
    prompt_frame = prompt_rows_to_frame(prompt_rows)
    write_outputs(
        output_dir,
        spec=spec,
        households=households,
        prompts=prompt_frame,
        final_household_states=final_household_states_frame,
        beliefs=beliefs_frame,
        decisions=decisions_frame,
        periods=periods_frame,
        accounting=accounting_frame,
        forecasts=forecasts,
        joined=joined,
        target_scores=target_scores,
        family_scores=family_scores,
        origin_scores=origin_scores,
        raw_records=raw_records,
        live_attempts=live_attempts,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "bundle_sha256": bundle.bundle_sha256,
        "normalized_spec_sha256": spec_sha256,
        "periods_per_year": PERIODS_PER_YEAR,
        "origin_count": len(bundle.origins),
        "target_count": len(bundle.target_specs),
        "scored_origin_count": score_origin_contract["scored_origin_count"],
        "scored_target_row_count": len(score_bundle.targets),
        "score_origin_contract": score_origin_contract,
        "contamination_coverage": contamination_coverage,
        "provider_execution_isolation": provider_execution_isolation(args.mode),
        "initial_environment_anchor": initial_environment_anchor,
        "initial_environment_anchor_origin": bundle.origins[0],
        "household_provenance": household_provenance,
        "household_flow_anchor": household_flow_anchor,
        "behavior_policy_content_sha256": behavior_profile_content_sha256,
        "household_count": int(len(households)),
        "max_households_per_call": int(args.max_households_per_call),
        "live_call_count": int(llm_client.live_call_count),
        "cache_hit_count": llm_client.cache_hit_count,
        "replayed_record_count": llm_client.replayed_record_count,
        "semantic_retry_count": llm_client.semantic_retry_count,
        "rejected_semantic_payloads": llm_client.rejected_semantic_payloads,
        "seed_cache_provenance": seed_cache_provenance,
        "max_accounting_abs_residual": _max_accounting_residual(accounting_frame),
        "macro_scores": macro_scores,
        "llm_minus_adaptive": macro_scores["llm"] - macro_scores["adaptive"],
        "execution_source": execution_source,
        "execution_source_tree_sha256": execution_source["tree_sha256"],
        "source_contract_sha256": _sha256_json(execution_source),
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "provider_reasoning_effort": os.environ.get("CODEX_CLI_REASONING_EFFORT"),
        "origin_block_bootstrap": bootstrap,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
        "output_contract": list(OUTPUT_FILES),
        "outputs": output_hashes(output_dir, exclude={"manifest.json", "report.md"}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(manifest, target_scores, family_scores)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest["outputs"] = output_hashes(output_dir, exclude={"manifest.json"})
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest



def assert_matched_initial_state(initial: pd.DataFrame) -> None:
    ignored = {"candidate", "source", "variant"}
    columns = [column for column in initial.columns if column not in ignored]
    llm = (
        initial[initial["candidate"].eq("llm")][columns]
        .sort_values("type_id")
        .reset_index(drop=True)
    )
    adaptive = (
        initial[initial["candidate"].eq("adaptive")][columns]
        .sort_values("type_id")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            llm, adaptive, check_dtype=False, check_exact=True
        )
    except AssertionError as exc:
        raise DynamicMacroError(
            "Matched twins do not share identical initial state"
        ) from exc


def assert_complete_periods(
    periods: pd.DataFrame, origins: tuple[dict[str, str], ...]
) -> None:
    expected = set(range(len(origins)))
    for candidate, frame in periods.groupby("candidate"):
        actual = set(pd.to_numeric(frame["period_index"], errors="raise").astype(int))
        if actual != expected or len(frame) != len(expected):
            raise DynamicMacroError(
                f"Missing periods for {candidate}: expected 0..{len(origins) - 1}"
            )


def assert_accounting(accounting: pd.DataFrame) -> None:
    maximum = _max_accounting_residual(accounting)
    if not math.isfinite(maximum) or maximum > ACCOUNTING_TOLERANCE:
        raise DynamicMacroError(
            f"Accounting residual {maximum:.12g} exceeds {ACCOUNTING_TOLERANCE:.1e}"
        )


def _max_accounting_residual(accounting: pd.DataFrame) -> float:
    if accounting.empty or "abs_residual" not in accounting:
        return float("inf")
    return float(pd.to_numeric(accounting["abs_residual"], errors="coerce").max())



def attach_origin_identity(
    frame: pd.DataFrame, origins: tuple[dict[str, str], ...]
) -> None:
    if frame.empty:
        return
    by_index = {index: origin for index, origin in enumerate(origins)}
    frame["origin_month"] = (
        frame["period_index"]
        .astype(int)
        .map(lambda index: by_index[index]["origin_month"])
    )
    frame["as_of_date"] = (
        frame["period_index"]
        .astype(int)
        .map(lambda index: by_index[index]["as_of_date"])
    )


def build_forecasts(
    periods: pd.DataFrame, households: pd.DataFrame, bundle: BundleView
) -> pd.DataFrame:
    initial = _initial_macro_row(households)
    rows: list[dict[str, Any]] = []
    targets_by_origin: dict[str, list[dict[str, Any]]] = {}
    for target in bundle.targets:
        targets_by_origin.setdefault(target["origin_month"], []).append(target)
    for candidate, path in periods.groupby("candidate"):
        path = path.sort_values("period_index").reset_index(drop=True)
        for index, origin in enumerate(bundle.origins):
            current = path.iloc[index]
            previous: Mapping[str, Any] = (
                initial if index == 0 else path.iloc[index - 1]
            )
            for target in targets_by_origin.get(origin["origin_month"], []):
                mapping = MAPPING_BY_SERIES[target["series_id"]]
                prediction = mapped_value(current, previous, mapping)
                rows.append(
                    {
                        "candidate": candidate,
                        "origin_month": origin["origin_month"],
                        "as_of_date": origin["as_of_date"],
                        "target_observation_date": target["target_observation_date"],
                        "target_name": target["target_name"],
                        "series_id": target["series_id"],
                        "family": target["family"],
                        "prediction": prediction,
                        "economy_measure": mapping.economy_measure,
                        "economy_transform": mapping.economy_transform,
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(bundle.targets) * 2
    if (
        len(frame) != expected
        or frame[["candidate", "origin_month", "target_name"]].duplicated().any()
    ):
        raise DynamicMacroError("Incomplete target coverage in economy forecasts")
    return frame.sort_values(
        ["candidate", "origin_month", "family", "target_name"]
    ).reset_index(drop=True)


def mapped_value(
    current: Mapping[str, Any], previous: Mapping[str, Any], mapping: TargetMapping
) -> float:
    current_measure = economy_measure(current, mapping.economy_measure)
    previous_measure = economy_measure(previous, mapping.economy_measure)
    if mapping.economy_transform == "pct_change":
        return _pct_change(current_measure, previous_measure)
    if mapping.economy_transform == "nominal_pct_change":
        real_growth = _pct_change(current_measure, previous_measure)
        monthly_price = _annual_rate_to_monthly_pct(
            float(current["next_inflation_rate"])
        )
        return 100.0 * (
            (1.0 + real_growth / 100.0) * (1.0 + monthly_price / 100.0) - 1.0
        )
    if mapping.economy_transform == "diff":
        return current_measure - previous_measure
    if mapping.economy_transform == "level":
        return current_measure
    if mapping.economy_transform == "annual_rate_to_monthly_pct":
        return _annual_rate_to_monthly_pct(current_measure)
    raise DynamicMacroError(f"Unknown economy transform: {mapping.economy_transform}")


def economy_measure(row: Mapping[str, Any], measure: str) -> float:
    if measure == "saving_rate_pct":
        income = float(row["aggregate_income"])
        return (
            0.0
            if abs(income) < 1e-12
            else 100.0 * float(row["aggregate_saving"]) / income
        )
    if measure == "unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["employment_rate"]))
    if measure == "next_unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["next_employment_rate"]))
    return float(row[measure])


def _initial_macro_row(households: pd.DataFrame) -> dict[str, float]:
    weights = households["population_weight"].astype(float)
    aggregate_income = float(
        (weights * households["annual_income"].astype(float) / PERIODS_PER_YEAR).sum()
    )
    aggregate_saving = float(
        (
            weights
            * households["annual_income"].astype(float)
            / PERIODS_PER_YEAR
            * households["base_saving_rate"].astype(float)
        ).sum()
    )
    aggregate_consumption = float(
        (
            weights
            * households["baseline_consumption_annual"].astype(float)
            / PERIODS_PER_YEAR
        ).sum()
    )
    aggregate_debt = float((weights * households["debt"].astype(float)).sum())
    return {
        "aggregate_consumption": aggregate_consumption,
        "aggregate_income": aggregate_income,
        "aggregate_saving": aggregate_saving,
        "aggregate_debt": aggregate_debt,
        "employment_rate": 0.955,
        "inflation_rate": 2.0,
        "policy_rate": 2.5,
        "next_employment_rate": 0.955,
        "next_inflation_rate": 2.0,
        "next_policy_rate": 2.5,
        "next_aggregate_income": aggregate_income,
    }


def join_and_score_forecasts(
    forecasts: pd.DataFrame, targets: tuple[dict[str, Any], ...]
) -> pd.DataFrame:
    target_frame = pd.DataFrame(targets)
    joined = forecasts.merge(
        target_frame,
        on=[
            "origin_month",
            "as_of_date",
            "target_observation_date",
            "target_name",
            "series_id",
            "family",
        ],
        how="left",
        validate="many_to_one",
    )
    if joined["target_value"].isna().any() or len(joined) != len(forecasts):
        raise DynamicMacroError("Forecast-target join is incomplete")
    joined["error"] = joined["prediction"] - joined["target_value"]
    joined["scaled_error"] = joined["error"] / joined["default_scale"]
    joined["scaled_squared_error"] = joined["scaled_error"] ** 2
    joined["absolute_scaled_error"] = joined["scaled_error"].abs()
    transformed = ~joined["transform"].eq("level")
    joined["target_direction"] = np.sign(
        np.where(
            transformed,
            joined["target_value"],
            joined["target_value"] - joined["origin_visible_denominator_value"],
        )
    ).astype(int)
    joined["forecast_direction"] = np.sign(
        np.where(
            transformed,
            joined["prediction"],
            joined["prediction"] - joined["origin_visible_denominator_value"],
        )
    ).astype(int)
    joined["direction_correct"] = joined["target_direction"].eq(
        joined["forecast_direction"]
    )
    pivot = joined.pivot(
        index=["origin_month", "target_name"],
        columns="candidate",
        values="scaled_squared_error",
    )
    if set(pivot.columns) != {"llm", "adaptive"}:
        raise DynamicMacroError("Matched score rows are incomplete")
    delta = (pivot["llm"] - pivot["adaptive"]).rename(
        "llm_minus_adaptive_scaled_squared_error"
    )
    joined = joined.merge(
        delta, on=["origin_month", "target_name"], how="left", validate="many_to_one"
    )
    return joined.sort_values(
        ["candidate", "origin_month", "family", "target_name"]
    ).reset_index(drop=True)


def score_macro(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target_scores = joined.groupby(
        ["candidate", "family", "target_name"], as_index=False
    ).agg(
        observations=("scaled_squared_error", "size"),
        mean_squared_scaled_error=("scaled_squared_error", "mean"),
        mean_absolute_scaled_error=("absolute_scaled_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
    )
    target_scores["target_score"] = np.sqrt(target_scores["mean_squared_scaled_error"])
    target_pivot = target_scores.pivot(
        index=["family", "target_name"], columns="candidate", values="target_score"
    )
    target_delta = (target_pivot["llm"] - target_pivot["adaptive"]).rename(
        "llm_minus_adaptive_target_score"
    )
    target_scores = target_scores.merge(
        target_delta,
        on=["family", "target_name"],
        how="left",
        validate="many_to_one",
    )
    family_scores = target_scores.groupby(["candidate", "family"], as_index=False).agg(
        target_count=("target_name", "size"),
        family_mean_squared_scaled_error=("mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    family_scores["family_score"] = np.sqrt(
        family_scores["family_mean_squared_scaled_error"]
    )
    family_pivot = family_scores.pivot(
        index="family", columns="candidate", values="family_score"
    )
    family_delta = (family_pivot["llm"] - family_pivot["adaptive"]).rename(
        "llm_minus_adaptive_family_score"
    )
    family_scores = family_scores.merge(
        family_delta, on="family", how="left", validate="many_to_one"
    )
    macro_scores = {
        str(candidate): float(
            math.sqrt(group["family_mean_squared_scaled_error"].mean())
        )
        for candidate, group in family_scores.groupby("candidate")
    }
    if set(macro_scores) != {"llm", "adaptive"}:
        raise DynamicMacroError("MacroScore requires both matched candidates")
    family_scores["family_equal_weight"] = family_scores.groupby("candidate")[
        "family"
    ].transform(lambda values: 1.0 / len(values))
    family_scores["macro_score"] = family_scores["candidate"].map(macro_scores)
    family_scores["llm_minus_adaptive"] = macro_scores["llm"] - macro_scores["adaptive"]

    origin_target = joined.groupby(
        ["candidate", "origin_month", "family", "target_name"], as_index=False
    ).agg(
        mean_squared_scaled_error=("scaled_squared_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
    )
    origin_family = origin_target.groupby(
        ["candidate", "origin_month", "family"], as_index=False
    ).agg(
        family_mean_squared_scaled_error=("mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    origin_scores = origin_family.groupby(
        ["candidate", "origin_month"], as_index=False
    ).agg(
        family_count=("family", "size"),
        macro_mean_squared_scaled_error=("family_mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    origin_scores["macro_score"] = np.sqrt(
        origin_scores["macro_mean_squared_scaled_error"]
    )
    origin_pivot = origin_scores.pivot(
        index="origin_month", columns="candidate", values="macro_score"
    )
    origin_delta = (origin_pivot["llm"] - origin_pivot["adaptive"]).rename(
        "llm_minus_adaptive"
    )
    origin_scores = origin_scores.merge(
        origin_delta, on="origin_month", how="left", validate="many_to_one"
    )
    return target_scores, family_scores, origin_scores, macro_scores


def origin_block_bootstrap(
    joined: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_length: int | None,
) -> dict[str, Any]:
    origins = sorted(joined["origin_month"].astype(str).unique())
    n_origins = len(origins)
    if replicates <= 0:
        raise DynamicMacroError("bootstrap replicates must be positive")
    length = int(block_length or max(1, round(math.sqrt(n_origins))))
    if length < 1 or length > n_origins:
        raise DynamicMacroError(
            "bootstrap block length must be between 1 and the origin count"
        )
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    starts = np.arange(n_origins)
    for replicate in range(replicates):
        selected: list[str] = []
        while len(selected) < n_origins:
            start = int(rng.choice(starts))
            selected.extend(
                origins[(start + offset) % n_origins] for offset in range(length)
            )
        pieces: list[pd.DataFrame] = []
        for draw_index, origin in enumerate(selected[:n_origins]):
            piece = joined[joined["origin_month"].astype(str).eq(origin)].copy()
            piece["origin_month"] = f"bootstrap_{replicate}_{draw_index}"
            pieces.append(piece)
        _, _, _, scores = score_macro(pd.concat(pieces, ignore_index=True))
        deltas.append(scores["llm"] - scores["adaptive"])
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975])
    return {
        "method": "circular_moving_origin_block_bootstrap",
        "estimand": "llm_minus_adaptive_family_equal_macro_score",
        "seed": int(seed),
        "replicates": int(replicates),
        "block_length": length,
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def normalized_spec(
    args: argparse.Namespace,
    *,
    bundle: BundleView,
    gains: dict[str, float],
    period_overrides: dict[int, dict[str, Any]],
    initial_environment_anchor: dict[str, float],
    behavior_profile_content_sha256: str | None,
    household_provenance: dict[str, Any],
    household_flow_anchor: dict[str, Any],
    contamination_coverage: dict[str, Any],
    score_origin_contract: dict[str, Any],
    seed_cache_provenance: dict[str, Any] | None,
    execution_source: dict[str, Any],
) -> dict[str, Any]:
    mappings = []
    for spec in bundle.target_specs:
        mapping = MAPPING_BY_SERIES[spec["series_id"]]
        mappings.append({**spec, **asdict(mapping)})
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_sha256": bundle.bundle_sha256,
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "provider_reasoning_effort": os.environ.get("CODEX_CLI_REASONING_EFFORT"),
        "execution_source_tree_sha256": execution_source["tree_sha256"],
        "source_contract_sha256": _sha256_json(execution_source),
        "provider_execution_isolation": provider_execution_isolation(args.mode),
        "periods_per_year": PERIODS_PER_YEAR,
        "feedback_mode": args.feedback_mode,
        "feedback_gain": float(args.feedback_gain),
        "policy_rate_smoothing": float(args.policy_rate_smoothing),
        "policy_state_mode": str(args.policy_state_mode),
        "policy_state_weight": float(args.policy_state_weight),
        "max_households_per_call": int(args.max_households_per_call),
        "belief_gains": gains,
        "initial_environment_anchor": initial_environment_anchor,
        "initial_environment_anchor_origin": bundle.origins[0],
        "household_provenance": household_provenance,
        "household_flow_anchor": household_flow_anchor,
        "behavior_policy_mode": args.behavior_policy_mode,
        "behavior_policy_content_sha256": behavior_profile_content_sha256,
        "replay_provenance": {
            "raw_records_json": str(args.raw_records_json) if args.raw_records_json else None,
            "raw_records_sha256": (
                hashlib.sha256(Path(args.raw_records_json).read_bytes()).hexdigest()
                if args.raw_records_json
                else None
            ),
            "replay_prefix_period_count": int(args.replay_prefix_period_count),
            "semantic_retry_limit": int(args.semantic_retry_limit),
        },
        "seed_cache_provenance": seed_cache_provenance,
        "period_overrides": {
            str(key): value for key, value in period_overrides.items()
        },
        "target_mappings": mappings,
        "contamination_coverage": contamination_coverage,
        "score_origin_contract": score_origin_contract,
        "macro_score": {
            "target_score": "sqrt(mean squared scaled error across scored origins)",
            "family_score": "sqrt(mean target mean-squared-scaled-error within family)",
            "macro_score": "sqrt(mean family mean-squared-scaled-error with equal family weight)",
            "adaptive_role": "diagnostic_only_not_a_selection_veto",
            "adaptive_information": "same raw origin-visible history and derived observed signals as the LLM, plus endogenous recursive state",
        },
    }


def provider_execution_isolation(mode: str) -> dict[str, Any]:
    enabled = mode in {"live", "replay", "replay_live"}
    return {
        "enabled": enabled,
        "relative_path": "provider_cwd" if enabled else None,
        "workspace_policy": "empty_isolated_no_repo_access" if enabled else "not_used",
        "included_in_output_contract": False,
    }


def prompt_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "prompt_payload"},
                "prompt_payload": _canonical_json(row["prompt_payload"]),
            }
            for row in rows
        ]
    )


def write_outputs(
    output_dir: Path,
    *,
    spec: dict[str, Any],
    households: pd.DataFrame,
    prompts: pd.DataFrame,
    final_household_states: pd.DataFrame,
    beliefs: pd.DataFrame,
    decisions: pd.DataFrame,
    periods: pd.DataFrame,
    accounting: pd.DataFrame,
    forecasts: pd.DataFrame,
    joined: pd.DataFrame,
    target_scores: pd.DataFrame,
    family_scores: pd.DataFrame,
    origin_scores: pd.DataFrame,
    raw_records: list[dict[str, Any]],
    live_attempts: list[dict[str, Any]],
) -> None:
    (output_dir / "normalized_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, frame in (
        ("households.csv", households),
        ("final_household_states.csv", final_household_states),
        ("prompt_cards.csv", prompts),
        ("beliefs.csv", beliefs),
        ("decisions.csv", decisions),
        ("periods.csv", periods),
        ("accounting.csv", accounting),
        ("forecasts.csv", forecasts),
        ("joined_errors.csv", joined),
        ("target_scores.csv", target_scores),
        ("family_scores.csv", family_scores),
        ("origin_scores.csv", origin_scores),
    ):
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "raw_records.json").write_text(
        json.dumps(raw_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "live_attempts.json").write_text(
        json.dumps(live_attempts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_report(
    manifest: dict[str, Any], target_scores: pd.DataFrame, family_scores: pd.DataFrame
) -> str:
    bootstrap = manifest["origin_block_bootstrap"]
    contamination = manifest["contamination_coverage"]
    isolation = manifest["provider_execution_isolation"]
    household_provenance = manifest["household_provenance"]
    household_temporal = household_provenance["temporal_coverage"]
    summary = family_scores[
        ["candidate", "family", "family_score", "direction_accuracy", "macro_score"]
    ].sort_values(["candidate", "family"])
    return "\n".join(
        [
            "# Recursive Dynamic Macro Economy",
            "",
            "## Result",
            f"- LLM family-equal MacroScore: `{manifest['macro_scores']['llm']:.6f}`.",
            f"- Adaptive diagnostic MacroScore: `{manifest['macro_scores']['adaptive']:.6f}`.",
            f"- LLM minus adaptive: `{manifest['llm_minus_adaptive']:.6f}`.",
            f"- 95% origin-block bootstrap interval: `[{bootstrap['lower']:.6f}, {bootstrap['upper']:.6f}]`.",
            "- The adaptive twin is diagnostic only and does not veto or select the LLM candidate.",
            f"- Scored origins: `{', '.join(manifest['score_origin_contract']['scored_origins'])}`; "
            f"warm-up origins: `{', '.join(manifest['score_origin_contract']['warmup_origins']) or 'none'}`.",
            "",
            "## Family Scores",
            markdown_table(summary),
            "",
            "## Target Scores",
            markdown_table(
                target_scores[
                    [
                        "candidate",
                        "family",
                        "target_name",
                        "target_score",
                        "direction_accuracy",
                        "observations",
                    ]
                ].sort_values(["candidate", "family", "target_name"])
            ),
            "",
            "## Integrity",
            f"- Bundle SHA-256: `{manifest['bundle_sha256']}`.",
            f"- Normalized spec SHA-256: `{manifest['normalized_spec_sha256']}`.",
            f"- Normalized household-state SHA-256: `{household_provenance['normalized_household_state_sha256']}`; "
            f"raw input SHA-256: `{household_provenance['raw_input_file_sha256']}`.",
            f"- Behavior-policy canonical content SHA-256: `{manifest['behavior_policy_content_sha256']}`.",
            f"- Household public availability: `{household_temporal['status']}`; latest source availability "
            f"`{household_temporal['source_availability_date_max']}` versus first macro as-of "
            f"`{household_temporal['first_macro_as_of_date']}`.",
            f"- Shared first-origin environment anchor: `{_canonical_json(manifest['initial_environment_anchor'])}`.",
            f"- Household flow anchor: `{_canonical_json(manifest['household_flow_anchor'])}`.",
            f"- Maximum accounting residual: `{manifest['max_accounting_abs_residual']:.12g}`.",
            f"- Live calls: `{manifest['live_call_count']}`; replayed identity-bearing records: "
            f"`{manifest['replayed_record_count']}`; ordinary cache hits: `{manifest['cache_hit_count']}`.",
            f"- Monthly origins: `{manifest['origin_count']}`; frozen targets: `{manifest['target_count']}`.",
            f"- Contamination policy `{contamination['policy']}` for `{contamination['model']}` selected "
            f"`{contamination['selected_rows']}` of `{contamination['catalogue_rows']}` origin-target rows.",
            f"- Original label counts available: `{_canonical_json(contamination['available_label_counts'])}`; "
            f"selected: `{_canonical_json(contamination['selected_label_counts'])}`; "
            f"excluded: `{_canonical_json(contamination['excluded_label_counts'])}`.",
            f"- Contamination-filtered origin coverage: `{contamination['complete_origin_count']}` complete, "
            f"`{contamination['partial_origin_count']}` partial, `{contamination['empty_origin_count']}` empty.",
            f"- Provider execution isolation: `{isolation['workspace_policy']}` at relative path `{isolation['relative_path']}`.",
            "- Both twins receive the same raw origin-visible history and observed-signal summary; each retains its own endogenous recursive feedback state.",
            "- Frozen outcomes, target-contamination labels, and revision audit never enter prompts.",
            "",
        ]
    )


def output_hashes(output_dir: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILES
        if name not in exclude and (output_dir / name).is_file()
    }


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("raw_records") or payload.get("records")
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise DynamicMacroError(
            "--raw-records-json must contain a JSON list of records"
        )
    return list(payload)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_households_per_call) <= 0:
        raise DynamicMacroError("--max-households-per-call must be positive")
    if args.mode == "live" and int(args.max_live_calls) <= 0:
        raise DynamicMacroError("--max-live-calls must be positive in live mode")
    if args.mode == "replay_live" and int(args.max_live_calls) < 0:
        raise DynamicMacroError("--max-live-calls must be non-negative in replay_live mode")
    if args.mode not in {"replay", "replay_live"} and args.raw_records_json:
        raise DynamicMacroError(
            "--raw-records-json is only valid in replay or replay_live mode"
        )
    if args.mode == "replay_live":
        if not args.raw_records_json:
            raise DynamicMacroError("replay_live requires --raw-records-json")
        if int(args.replay_prefix_period_count) <= 0:
            raise DynamicMacroError(
                "replay_live requires --replay-prefix-period-count greater than zero"
            )
    elif int(args.replay_prefix_period_count) != 0:
        raise DynamicMacroError(
            "--replay-prefix-period-count is only valid in replay_live mode"
        )
    if int(args.semantic_retry_limit) < 0:
        raise DynamicMacroError("--semantic-retry-limit must be non-negative")
    if args.seed_cache_dir and (
        args.mode not in {"live", "replay_live"} or not args.fresh_cache
    ):
        raise DynamicMacroError(
            "--seed-cache-dir requires live/replay_live mode with --fresh-cache"
        )
    if not math.isfinite(float(args.feedback_gain)) or float(args.feedback_gain) < 0.0:
        raise DynamicMacroError("--feedback-gain must be finite and non-negative")
    if not math.isfinite(float(args.policy_rate_smoothing)) or not 0.0 <= float(
        args.policy_rate_smoothing
    ) <= 1.0:
        raise DynamicMacroError("--policy-rate-smoothing must be between zero and one")
    if not math.isfinite(float(args.policy_state_weight)) or not 0.0 <= float(
        args.policy_state_weight
    ) <= 1.0:
        raise DynamicMacroError("--policy-state-weight must be between zero and one")
    if (
        not math.isfinite(float(args.hybrid_state_weight))
        or not 0.0 <= float(args.hybrid_state_weight) <= 1.0
    ):
        raise DynamicMacroError("--hybrid-state-weight must be between zero and one")


def _validate_replay_live_horizon(
    args: argparse.Namespace, *, origin_count: int
) -> None:
    if args.mode != "replay_live":
        return
    prefix = int(args.replay_prefix_period_count)
    if prefix > origin_count:
        raise DynamicMacroError(
            "Replay prefix cannot exceed the recursive economy horizon"
        )
    if prefix < origin_count and int(args.max_live_calls) <= 0:
        raise DynamicMacroError(
            "replay_live requires positive --max-live-calls when the replay prefix does not cover the horizon"
        )


def _pct_change(current: float, previous: float) -> float:
    if abs(previous) < 1e-12:
        raise DynamicMacroError(
            "Cannot map a percentage-change target from a zero economy baseline"
        )
    return 100.0 * (current / previous - 1.0)


def _annual_rate_to_monthly_pct(rate: float) -> float:
    if rate <= -100.0:
        raise DynamicMacroError(
            "Annual inflation rate cannot be at or below -100 percent"
        )
    return 100.0 * ((1.0 + rate / 100.0) ** (1.0 / PERIODS_PER_YEAR) - 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
