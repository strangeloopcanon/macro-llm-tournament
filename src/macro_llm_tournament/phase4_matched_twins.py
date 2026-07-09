from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, WORK_ROOT, markdown_table
from .demand_economy import (
    BEHAVIOR_POLICY_MODES,
    DemandEconomyClient,
    DemandScenario,
    behavior_policy_manifest,
    build_hybrid_behavior_policy_profile,
    build_fixture_demand_households,
    load_empirical_bridge_profile,
    load_behavior_policy_profile,
    load_state_behavior_policy_profile,
    normalize_demand_households,
    run_demand_economy,
)
from .empirical_bridge import BRIDGE_SPEC_VERSION, DEFAULT_BRIDGE
from .postcutoff_behavior_gate import (
    TARGET_SPECS,
    build_postcutoff_behavior_cards,
    load_proxy_data,
    score_proxy_forecasts,
)


PHASE4_VERSION = "phase4_prior_update_matched_twins_v2"
DEFAULT_SCENARIO = DemandScenario(
    "baseline",
    "No exogenous shock; households react only to endogenous feedback.",
)
COMPARISON_VARIABLES = (
    "aggregate_consumption",
    "output_gap_pct",
    "employment_rate",
    "inflation_rate",
    "policy_rate",
    "aggregate_job_loss_belief",
    "aggregate_confidence_index",
    "aggregate_liquid_buffer_months",
)


@dataclass(frozen=True)
class OutputMappingSpec:
    target_name: str
    series_id: str
    target_label: str
    target_units: str
    target_transform: str
    economy_variable: str
    economy_transform: str
    period_alignment: str
    lower: float
    upper: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 prior-update matched-twin economy gate.")
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=("fixture", "replay", "live"), default="fixture")
    parser.add_argument("--data-mode", choices=("fixture", "fred"), default="fixture")
    parser.add_argument("--refresh-fred", action="store_true")
    parser.add_argument("--cutoff-date", default="2025-12-01")
    parser.add_argument("--asof-start", default="2025-12-15")
    parser.add_argument("--asof-end", default="2026-04-15")
    parser.add_argument("--history-months", type=int, default=18)
    parser.add_argument("--scoreable-only", action="store_true")
    parser.add_argument("--scoring-label", choices=("retrospective", "confirmatory"), default="retrospective")
    parser.add_argument("--belief-source", choices=("fixture", "persona_ecology_replay"), default="fixture")
    parser.add_argument("--persona-ecology-dir", default=None)
    parser.add_argument("--primary-ecology-source", default="")
    parser.add_argument("--ecology-period-policy", choices=("strict", "hold_last"), default="strict")
    parser.add_argument("--household-source", choices=("fixture", "csv"), default="fixture")
    parser.add_argument("--household-csv", default=None)
    parser.add_argument("--household-count", type=int, default=24)
    parser.add_argument("--period-count", type=int, default=12)
    parser.add_argument("--feedback-mode", choices=("closed_loop", "none"), default="closed_loop")
    parser.add_argument("--behavior-policy-mode", choices=BEHAVIOR_POLICY_MODES, default="fixed_kernel")
    parser.add_argument("--behavior-policy-raw-records-json", default=None)
    parser.add_argument("--behavior-policy-state-profile-json", default=None)
    parser.add_argument("--empirical-bridge-json", default=None)
    parser.add_argument("--hybrid-state-weight", type=float, default=1.0)
    parser.add_argument("--belief-gain-global", type=float, default=1.0)
    parser.add_argument("--belief-gain-inflation", type=float, default=1.0)
    parser.add_argument("--belief-gain-income", type=float, default=1.0)
    parser.add_argument("--belief-gain-unemployment", type=float, default=1.0)
    parser.add_argument("--feedback-gain-multiplier", type=float, default=1.0)
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"phase4_matched_twins_{timestamp_slug()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(args)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        validate_args(args)
        mapping = default_output_mapping()
        mapping_payload = normalized_mapping_payload(mapping)
        mapping_sha = mapping_sha256(mapping_payload)

        behavior_policy_profile = load_phase4_behavior_policy_profile(args)
        frames, data_status = load_proxy_data(
            data_mode=args.data_mode,
            refresh=args.refresh_fred,
            cutoff_date=args.cutoff_date,
            asof_end=args.asof_end,
        )
        cards, targets, context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date=args.cutoff_date,
            asof_start=args.asof_start,
            asof_end=args.asof_end,
            history_months=args.history_months,
            scoreable_only=args.scoreable_only,
        )
        if not cards:
            raise ValueError("No post-cutoff proxy cards were built.")

        ecology_bundle = load_persona_ecology_bundle(args) if args.belief_source == "persona_ecology_replay" else None
        households = load_phase4_households(args, ecology_bundle=ecology_bundle)
        period_count = max(int(args.period_count), len(cards) + 1, 2)
        cache_dir = output_dir / "fresh_phase4_matched_twins_cache" if args.fresh_cache else WORK_ROOT / "phase4_matched_twins_cache"
        scenarios = [phase4_scenario_with_feedback_multiplier(float(args.feedback_gain_multiplier))]
        twins = build_phase4_clients(args, cache_dir=cache_dir, ecology_bundle=ecology_bundle, period_count=period_count)

        all_initial: list[pd.DataFrame] = []
        all_beliefs: list[pd.DataFrame] = []
        all_decisions: list[pd.DataFrame] = []
        all_periods: list[pd.DataFrame] = []
        all_accounting: list[pd.DataFrame] = []
        raw_records: list[dict[str, Any]] = []
        live_used = 0
        cache_hits = 0
        for client in twins:
            initial, beliefs, decisions, periods, accounting, _prompt_rows = run_demand_economy(
                households,
                scenarios,
                client,
                period_count=period_count,
                feedback_mode=args.feedback_mode,
                behavior_policy_profile=behavior_policy_profile,
            )
            all_initial.append(initial)
            all_beliefs.append(beliefs)
            all_decisions.append(decisions)
            all_periods.append(periods)
            all_accounting.append(accounting)
            raw_records.extend(client.raw_records)
            live_used += client.live_call_count
            cache_hits += client.cache_hit_count

        initial_frame = pd.concat(all_initial, ignore_index=True)
        beliefs_frame = pd.concat(all_beliefs, ignore_index=True)
        decisions_frame = pd.concat(all_decisions, ignore_index=True)
        periods_frame = pd.concat(all_periods, ignore_index=True)
        accounting_frame = pd.concat(all_accounting, ignore_index=True)
        path_comparison = build_path_comparison(periods_frame)
        scoring_targets = phase4_scoring_targets(targets, mapping)
        proxy_forecasts = economy_proxy_forecasts(periods_frame, cards, mapping)
        proxy_scores, joined = score_proxy_forecasts(proxy_forecasts, scoring_targets)
        verdict = classify_phase4_run(
            periods_frame,
            accounting_frame,
            proxy_scores,
            belief_source=args.belief_source,
            data_mode=args.data_mode,
        )

        write_outputs(
            output_dir,
            mapping_payload=mapping_payload,
            cards=cards_to_frame(cards),
            context=context,
            targets=scoring_targets,
            households=households,
            initial=initial_frame,
            beliefs=beliefs_frame,
            decisions=decisions_frame,
            periods=periods_frame,
            accounting=accounting_frame,
            comparison=path_comparison,
            forecasts=proxy_forecasts,
            joined=joined,
            scores=proxy_scores,
            raw_records=raw_records,
        )
        manifest.update(
            {
                "status": "ok",
                "verdict": verdict["verdict"],
                "claim_scope": verdict["claim_scope"],
                "passed": bool(verdict["passed"]),
                "belief_source": args.belief_source,
                "ecology_period_policy": args.ecology_period_policy,
                "scoring_label": args.scoring_label,
                "v2_confirmatory_status": "first_mapping_v2_confirmatory_surface_spent_by_macro_confirmatory_fred_2026_02_v1",
                "behavior_policy": behavior_policy_manifest(behavior_policy_profile, mode=args.behavior_policy_mode),
                "mapping_sha256": mapping_sha,
                "mapping_spec_version": PHASE4_VERSION,
                "data_status": data_status,
                "persona_ecology_input": ecology_input_manifest(
                    ecology_bundle,
                    period_count=period_count,
                    belief_transform=belief_transform_from_args(args),
                ),
                "card_count": int(len(cards)),
                "target_rows": int(scoring_targets.shape[0]),
                "scoreable_target_rows": int(scoring_targets["target_available"].sum()) if not scoring_targets.empty else 0,
                "period_count_effective": int(period_count),
                "household_count": int(households.shape[0]),
                "sources": sorted(periods_frame["source"].astype(str).unique()),
                "live_call_count": int(live_used),
                "cache_hit_count": int(cache_hits),
                "max_accounting_abs_residual": max_accounting_abs_residual(accounting_frame),
                "score_rows": int(proxy_scores.shape[0]),
                "winner_source": phase4_winner(proxy_scores),
                "outputs": output_filenames(),
            }
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "phase4_matched_twins_report.md").write_text(build_report(manifest, proxy_scores, path_comparison), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "live_call_count": 0, "cache_hit_count": 0})
        if args.behavior_policy_mode == "empirical_bridge":
            manifest.update(empirical_bridge_failure_fields(args))
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


def validate_args(args: argparse.Namespace) -> None:
    if args.scoring_label == "confirmatory":
        raise ValueError(
            "Standalone Phase 4 confirmatory scoring is disabled; use the macro tournament only after "
            "a frozen-vintage loader and pre-score reservation are available."
        )
    if args.belief_source == "fixture" and args.mode != "fixture":
        raise ValueError("Phase 4 live/replay is blocked for fixture belief source; use --mode fixture.")
    if args.belief_source == "persona_ecology_replay" and args.mode != "replay":
        raise ValueError("Phase 4 persona ecology replay must use --mode replay.")
    if args.max_live_calls != 0:
        raise ValueError("Phase 4 matched-twin runs must use --max-live-calls 0; run persona ecology live upstream first.")
    if args.behavior_policy_mode == "schedule":
        if not args.behavior_policy_raw_records_json:
            raise ValueError("--behavior-policy-raw-records-json is required when --behavior-policy-mode schedule")
        if not Path(args.behavior_policy_raw_records_json).exists():
            raise ValueError(f"--behavior-policy-raw-records-json does not exist: {args.behavior_policy_raw_records_json}")
    if args.behavior_policy_mode == "state_schedule":
        if not args.behavior_policy_state_profile_json:
            raise ValueError("--behavior-policy-state-profile-json is required when --behavior-policy-mode state_schedule")
        if not Path(args.behavior_policy_state_profile_json).exists():
            raise ValueError(f"--behavior-policy-state-profile-json does not exist: {args.behavior_policy_state_profile_json}")
    if args.behavior_policy_mode == "empirical_bridge_state_schedule":
        if not args.behavior_policy_state_profile_json:
            raise ValueError("--behavior-policy-state-profile-json is required when --behavior-policy-mode empirical_bridge_state_schedule")
        if not Path(args.behavior_policy_state_profile_json).exists():
            raise ValueError(f"--behavior-policy-state-profile-json does not exist: {args.behavior_policy_state_profile_json}")
        if args.empirical_bridge_json and not Path(args.empirical_bridge_json).exists():
            raise ValueError(f"--empirical-bridge-json does not exist: {args.empirical_bridge_json}")
        if not 0.0 <= float(args.hybrid_state_weight) <= 1.0:
            raise ValueError("--hybrid-state-weight must be between 0 and 1")
    for name in ["belief_gain_global", "belief_gain_inflation", "belief_gain_income", "belief_gain_unemployment", "feedback_gain_multiplier"]:
        value = float(getattr(args, name))
        if value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.belief_source == "persona_ecology_replay":
        if not args.persona_ecology_dir:
            raise ValueError("--persona-ecology-dir is required with --belief-source persona_ecology_replay")
        if not Path(args.persona_ecology_dir).exists():
            raise ValueError(f"--persona-ecology-dir does not exist: {args.persona_ecology_dir}")
    if args.household_source == "csv":
        if not args.household_csv:
            raise ValueError("--household-csv is required when --household-source csv")
        if not Path(args.household_csv).exists():
            raise ValueError(f"--household-csv does not exist: {args.household_csv}")


def load_phase4_behavior_policy_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.behavior_policy_mode == "schedule":
        return load_behavior_policy_profile(Path(args.behavior_policy_raw_records_json))
    if args.behavior_policy_mode == "state_schedule":
        return load_state_behavior_policy_profile(Path(args.behavior_policy_state_profile_json))
    if args.behavior_policy_mode == "empirical_bridge":
        return load_empirical_bridge_profile(Path(args.empirical_bridge_json) if args.empirical_bridge_json else DEFAULT_BRIDGE)
    if args.behavior_policy_mode == "empirical_bridge_state_schedule":
        return build_hybrid_behavior_policy_profile(
            load_empirical_bridge_profile(Path(args.empirical_bridge_json) if args.empirical_bridge_json else DEFAULT_BRIDGE),
            load_state_behavior_policy_profile(Path(args.behavior_policy_state_profile_json)),
            state_weight=float(args.hybrid_state_weight),
        )
    return None


def phase4_scenario_with_feedback_multiplier(multiplier: float) -> DemandScenario:
    return replace(DEFAULT_SCENARIO, feedback_gain=float(DEFAULT_SCENARIO.feedback_gain) * float(multiplier))


def empirical_bridge_failure_fields(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.empirical_bridge_json) if args.empirical_bridge_json else DEFAULT_BRIDGE
    if not path.exists():
        return {"bridge_spec_version": BRIDGE_SPEC_VERSION, "empirical_bridge_path": str(path), "empirical_bridge_status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"bridge_spec_version": BRIDGE_SPEC_VERSION, "empirical_bridge_path": str(path), "empirical_bridge_status": "unreadable"}
    return {
        "bridge_spec_version": payload.get("bridge_spec_version") or payload.get("schema_version"),
        "empirical_bridge_path": str(path),
        "empirical_bridge_sha256": payload.get("canonical_payload_sha256"),
        "empirical_bridge_status": payload.get("status"),
        "empirical_bridge_constraints": payload.get("constraints"),
        "fit_waves": payload.get("fit_waves"),
        "validation_waves": payload.get("validation_waves"),
    }


def load_phase4_households(args: argparse.Namespace, *, ecology_bundle: dict[str, Any] | None) -> pd.DataFrame:
    if ecology_bundle is not None:
        return ecology_bundle["households"]
    return load_households(args)


def load_households(args: argparse.Namespace) -> pd.DataFrame:
    if args.household_source == "csv":
        return normalize_demand_households(pd.read_csv(Path(args.household_csv)))
    return build_fixture_demand_households(args.household_count)


def build_phase4_clients(
    args: argparse.Namespace,
    *,
    cache_dir: Path,
    ecology_bundle: dict[str, Any] | None,
    period_count: int,
) -> list[Any]:
    if args.belief_source == "persona_ecology_replay":
        if ecology_bundle is None:
            raise ValueError("persona ecology replay source was requested but no ecology bundle was loaded")
        return [
            PersonaEcologyReplayDemandClient(
                ecology_bundle,
                period_count=period_count,
                period_policy=args.ecology_period_policy,
                belief_transform=belief_transform_from_args(args),
            ),
            DemandEconomyClient(args.provider, "adaptive", cache_dir, mode="fixture", variant="adaptive", max_live_calls=0),
        ]
    return [
        DemandEconomyClient(args.provider, args.model, cache_dir, mode=args.mode, variant="llm_belief", max_live_calls=args.max_live_calls),
        DemandEconomyClient(args.provider, "adaptive", cache_dir, mode="fixture", variant="adaptive", max_live_calls=0),
    ]


def belief_transform_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "global": float(args.belief_gain_global),
        "expected_inflation_1y": float(args.belief_gain_inflation),
        "expected_real_income_growth": float(args.belief_gain_income),
        "expected_unemployment_higher_prob": float(args.belief_gain_unemployment),
    }


def load_persona_ecology_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.persona_ecology_dir)
    manifest_path = root / "manifest.json"
    panel_path = root / "persona_ecology_panel.csv"
    predictions_path = root / "persona_ecology_predictions.csv"
    if not manifest_path.exists():
        raise ValueError(f"Persona ecology manifest missing: {manifest_path}")
    if not panel_path.exists():
        raise ValueError(f"Persona ecology panel missing: {panel_path}")
    if not predictions_path.exists():
        raise ValueError(f"Persona ecology predictions missing: {predictions_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = args.primary_ecology_source or str(manifest.get("primary_update_source") or "")
    if not source:
        raise ValueError("--primary-ecology-source is required because the ecology manifest has no primary_update_source")

    panel = pd.read_csv(panel_path)
    predictions = pd.read_csv(predictions_path)
    required_targets = {
        "expected_inflation_1y",
        "expected_unemployment_higher_prob",
        "expected_real_income_growth",
    }
    predictions = predictions[predictions["source"].astype(str).eq(source)].copy()
    if predictions.empty:
        raise ValueError(f"Persona ecology predictions contain no rows for source `{source}`")
    missing_targets = required_targets - set(predictions["target_name"].astype(str))
    if missing_targets:
        raise ValueError(f"Persona ecology predictions missing required targets: {', '.join(sorted(missing_targets))}")

    wide = build_ecology_prediction_wide(predictions)
    period_indices = sorted(int(value) for value in wide["period_index"].dropna().unique())
    if not period_indices:
        raise ValueError("Persona ecology predictions contain no period_index values")
    households = build_households_from_ecology_panel(panel, wide)
    coverage = ecology_period_coverage(wide, households)
    return {
        "root": root,
        "manifest": manifest,
        "panel": panel,
        "predictions": predictions,
        "wide": wide,
        "households": households,
        "coverage": coverage,
        "source": source,
        "manifest_sha256": file_sha256(manifest_path),
        "panel_sha256": file_sha256(panel_path),
        "predictions_sha256": file_sha256(predictions_path),
    }


def build_ecology_prediction_wide(predictions: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["respondent_id", "period_index", "period_id", "source", "provider", "model"]
    for column in index_cols:
        if column not in predictions:
            raise ValueError(f"Persona ecology predictions missing `{column}`")
    value_pieces = []
    for value_column in ["prior_prediction", "prediction", "p10", "p50", "p90"]:
        pivot = predictions.pivot_table(
            index=index_cols,
            columns="target_name",
            values=value_column,
            aggfunc="first",
        )
        pivot.columns = [f"{value_column}_{column}" for column in pivot.columns]
        value_pieces.append(pivot)
    metadata = (
        predictions.groupby(index_cols, dropna=False)[
            [
                "confidence",
                "uncertainty",
                "profile_weight",
                "prior_weight",
                "environment_weight",
                "aggregate_feedback_weight",
                "cache_hit",
                "call_source",
            ]
        ]
        .first()
        .reset_index()
        .set_index(index_cols)
    )
    wide = pd.concat([metadata, *value_pieces], axis=1).reset_index()
    return wide.sort_values(["period_index", "respondent_id"]).reset_index(drop=True)


def ecology_period_coverage(wide: pd.DataFrame, households: pd.DataFrame) -> dict[str, Any]:
    respondent_ids = set(households["type_id"].astype(str))
    period_counts = wide[wide["respondent_id"].astype(str).isin(respondent_ids)].groupby("period_index")["respondent_id"].nunique()
    return {
        "period_indices": [int(value) for value in sorted(period_counts.index.tolist())],
        "respondents_by_period": {str(int(index)): int(value) for index, value in period_counts.items()},
        "household_count": int(len(respondent_ids)),
    }


def build_households_from_ecology_panel(panel: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    required = {"respondent_id", "period_index", "weight", "income_group", "liquid_wealth_group", "age_group", "employment_status"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Persona ecology panel missing columns required for households: {', '.join(sorted(missing))}")
    first_period = int(pd.to_numeric(panel["period_index"], errors="coerce").min())
    base = panel[pd.to_numeric(panel["period_index"], errors="coerce").eq(first_period)].copy()
    available_ids = set(wide[wide["period_index"].astype(int).eq(first_period)]["respondent_id"].astype(str))
    base = base[base["respondent_id"].astype(str).isin(available_ids)].copy()
    if base.empty:
        raise ValueError("No first-period persona ecology respondents have replay predictions")

    rows: list[dict[str, Any]] = []
    for _, row in base.drop_duplicates("respondent_id").iterrows():
        respondent_id = str(row["respondent_id"])
        prior_inflation = float(row.get("prior_expected_inflation_1y", row.get("actual_expected_inflation_1y", 3.0)))
        prior_unemployment_higher = float(
            row.get("prior_expected_unemployment_higher_prob", row.get("actual_expected_unemployment_higher_prob", 35.0))
        )
        prior_income = float(row.get("prior_expected_real_income_growth", row.get("actual_expected_real_income_growth", 1.0)))
        income_group = normalize_group(row.get("income_group"), default="middle", allowed=("low", "middle", "high"))
        liquidity_group = normalize_group(row.get("liquid_wealth_group"), default="middle", allowed=("low", "middle", "high"))
        annual_income = income_to_annual_income(income_group)
        baseline_consumption_annual = annual_income * income_consumption_ratio(income_group)
        liquid_assets = liquidity_months(liquidity_group, income_group) * (baseline_consumption_annual / 12.0)
        job_loss = unemployment_higher_to_job_loss(prior_unemployment_higher)
        job_loss_risk_type = "high" if job_loss >= 9.0 or str(row.get("employment_status", "")).lower() in {"unemployed", "not_employed"} else "low"
        rows.append(
            {
                "type_id": respondent_id,
                "label": f"SCE prior-update respondent {respondent_id}",
                "population_weight": float(row.get("weight", 1.0)),
                "age_bucket": age_to_bucket(row.get("age_group")),
                "income_group": income_group,
                "liquidity_group": "low" if liquidity_group == "low" else "high",
                "job_loss_risk_type": job_loss_risk_type,
                "employment_status": str(row.get("employment_status", "unknown")),
                "annual_income": annual_income,
                "baseline_consumption_annual": baseline_consumption_annual,
                "liquid_assets": liquid_assets,
                "debt": annual_income * debt_service_burden(income_group) * (1.3 if liquidity_group == "low" else 0.8),
                "debt_service_burden": debt_service_burden(income_group),
                "base_mpc": base_mpc(liquidity_group, income_group, job_loss_risk_type),
                "base_saving_rate": base_saving_rate(liquidity_group, income_group),
                "rate_sensitivity": 0.35 + (0.20 if liquidity_group != "low" else 0.04) + (0.08 if income_group == "high" else 0.0),
                "income_sensitivity": {"low": 0.82, "middle": 0.58, "high": 0.36}[income_group],
                "precautionary_sensitivity": 0.34 + (0.28 if liquidity_group == "low" else 0.08) + (0.16 if job_loss_risk_type == "high" else 0.0),
                "baseline_job_loss_probability": job_loss,
                "unemployment_higher_probability_1y": prior_unemployment_higher,
                "target_buffer_months": 1.6 if liquidity_group == "low" else 5.2,
                "inflation_expectation_1y": prior_inflation,
                "income_growth_expectation_1y": prior_income,
                "confidence_index": confidence_from_priors(prior_unemployment_higher, prior_income, prior_inflation, income_group),
                "attention_weight_prices": 0.68 if income_group == "low" else 0.56 if income_group == "middle" else 0.46,
                "attention_weight_jobs": 0.75 if job_loss_risk_type == "high" else 0.48,
                "attention_weight_rates": 0.66 if liquidity_group != "low" or income_group == "high" else 0.40,
                "income_volatility": 0.10 + (0.08 if job_loss_risk_type == "high" else 0.02) + (0.04 if income_group == "low" else 0.0),
                "subsistence_floor_share": 0.56 if income_group == "low" else 0.48 if income_group == "middle" else 0.38,
            }
        )
    return normalize_demand_households(pd.DataFrame(rows))


class PersonaEcologyReplayDemandClient:
    variant = "llm_belief"

    def __init__(
        self,
        ecology_bundle: dict[str, Any],
        *,
        period_count: int,
        period_policy: str,
        belief_transform: dict[str, float] | None = None,
    ):
        self.ecology_bundle = ecology_bundle
        self.period_count = int(period_count)
        self.period_policy = period_policy
        self.belief_transform = normalize_belief_transform(belief_transform)
        self.raw_records: list[dict[str, Any]] = []
        self._wide_by_key = {
            (str(row["respondent_id"]), int(row["period_index"])): row
            for _, row in ecology_bundle["wide"].iterrows()
        }
        self._available_periods = sorted({period for _, period in self._wide_by_key})
        if not self._available_periods:
            raise ValueError("Persona ecology replay has no available periods")
        if self.period_policy == "strict" and max(self._available_periods) < self.period_count - 1:
            raise ValueError(
                "Persona ecology replay does not cover the requested Phase 4 periods: "
                f"needs 0..{self.period_count - 1}, has 0..{max(self._available_periods)}"
            )

    @property
    def source(self) -> str:
        return f"{self.ecology_bundle['source']}__phase4_prior_replay"

    @property
    def live_call_count(self) -> int:
        return 0

    @property
    def cache_hit_count(self) -> int:
        predictions = self.ecology_bundle["predictions"]
        if "cache_hit" not in predictions:
            return 0
        calls = predictions.drop_duplicates(["respondent_id", "period_index"])
        return int(calls["cache_hit"].astype(bool).sum())

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        requested_period = int(period_state["period_index"])
        ecology_period = self._ecology_period_for(requested_period)
        beliefs: dict[str, dict[str, Any]] = {}
        for state in household_states:
            type_id = str(state["type_id"])
            row = self._wide_by_key.get((type_id, ecology_period))
            if row is None:
                raise ValueError(f"Persona ecology replay missing respondent={type_id}, period_index={ecology_period}")
            belief = ecology_row_to_demand_belief(row, state=state, belief_transform=self.belief_transform)
            beliefs[type_id] = belief
        payload = {
            "prompt_version": PHASE4_VERSION,
            "beliefs_by_type": beliefs,
            "direct_actions_by_type": {},
        }
        self.raw_records.append(
            {
                "source": self.source,
                "variant": self.variant,
                "scenario_id": scenario.scenario_id,
                "period_id": period_state["period_id"],
                "period_index": requested_period,
                "provider": "persona_ecology_replay",
                "model": self.ecology_bundle["source"],
                "cache_hit": True,
                "cache_path": str(self.ecology_bundle["root"] / "persona_ecology_predictions.csv"),
                "ecology_period_index": ecology_period,
                "belief_transform": self.belief_transform,
                "payload": payload,
            }
        )
        return payload

    def _ecology_period_for(self, requested_period: int) -> int:
        if requested_period in self._available_periods:
            return requested_period
        if self.period_policy == "hold_last":
            return max(period for period in self._available_periods if period <= requested_period)
        raise ValueError(f"Persona ecology replay has no period_index={requested_period}")


def normalize_belief_transform(transform: dict[str, float] | None) -> dict[str, float]:
    values = {
        "global": 1.0,
        "expected_inflation_1y": 1.0,
        "expected_real_income_growth": 1.0,
        "expected_unemployment_higher_prob": 1.0,
    }
    if transform:
        for key in values:
            values[key] = float(transform.get(key, values[key]))
    return values


def transformed_ecology_prediction(row: pd.Series, target_name: str, transform: dict[str, float]) -> float:
    prediction = float(row[f"prediction_{target_name}"])
    prior_key = f"prior_prediction_{target_name}"
    prior = float(row[prior_key]) if prior_key in row and pd.notna(row[prior_key]) else prediction
    gain = float(transform.get("global", 1.0)) * float(transform.get(target_name, 1.0))
    return float(prior + gain * (prediction - prior))


def ecology_row_to_demand_belief(
    row: pd.Series,
    *,
    state: dict[str, Any],
    belief_transform: dict[str, float] | None = None,
) -> dict[str, Any]:
    transform = normalize_belief_transform(belief_transform)
    inflation = transformed_ecology_prediction(row, "expected_inflation_1y", transform)
    unemployment_higher = transformed_ecology_prediction(row, "expected_unemployment_higher_prob", transform)
    income_growth = transformed_ecology_prediction(row, "expected_real_income_growth", transform)
    uncertainty = float(row.get("uncertainty", 0.6))
    confidence = confidence_from_priors(unemployment_higher, income_growth, inflation, str(state.get("income_group", "middle")))
    confidence = float(np.clip(confidence + 8.0 * (float(row.get("confidence", 0.6)) - 0.6), 0.0, 100.0))
    low_liquid = str(state.get("liquidity_group", "")).lower() == "low"
    precaution = 2.4 + 0.055 * unemployment_higher + 0.45 * uncertainty + (1.0 if low_liquid else 0.1) + max(0.0, -income_growth) * 0.12
    return {
        "type_id": str(state["type_id"]),
        "expected_inflation_next_period": float(np.clip(inflation, -5.0, 15.0)),
        "expected_income_growth_next_period": float(np.clip(income_growth, -12.0, 12.0)),
        "perceived_job_loss_probability": unemployment_higher_to_job_loss(unemployment_higher),
        "expected_unemployment_higher_probability_next_period": float(np.clip(unemployment_higher, 0.0, 100.0)),
        "confidence_index": confidence,
        "precautionary_saving_score": float(np.clip(precaution, 0.0, 10.0)),
        "attention_weight_prices": float(np.clip(0.45 + 0.45 * float(row.get("environment_weight", 0.3)), 0.0, 1.0)),
        "attention_weight_jobs": float(np.clip(0.35 + 0.35 * float(row.get("prior_weight", 0.4)) + 0.20 * float(row.get("aggregate_feedback_weight", 0.1)), 0.0, 1.0)),
        "attention_weight_rates": float(np.clip(0.35 + 0.40 * float(row.get("environment_weight", 0.3)), 0.0, 1.0)),
        "reason_codes": [
            "sce_prior_update_replay",
            f"ecology_period_{int(row['period_index'])}",
            (
                f"belief_gain_g{transform['global']:.2f}"
                f"_pi{transform['expected_inflation_1y']:.2f}"
                f"_inc{transform['expected_real_income_growth']:.2f}"
                f"_unemp{transform['expected_unemployment_higher_prob']:.2f}"
            ),
        ],
        "causal_path": ["real respondent prior", "codex_cli belief update", "deterministic demand policy"],
    }


def normalize_group(value: Any, *, default: str, allowed: tuple[str, ...]) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default


def age_to_bucket(value: Any) -> str:
    text = str(value).strip().lower()
    return "older" if text in {"55_plus", "55+", "older", "retired"} else "prime"


def income_to_annual_income(group: str) -> float:
    return {"low": 38000.0, "middle": 76000.0, "high": 135000.0}.get(group, 76000.0)


def income_consumption_ratio(group: str) -> float:
    return {"low": 0.94, "middle": 0.82, "high": 0.68}.get(group, 0.82)


def liquidity_months(liquidity_group: str, income_group: str) -> float:
    if liquidity_group == "low":
        return {"low": 0.6, "middle": 0.9, "high": 1.4}.get(income_group, 0.9)
    if liquidity_group == "middle":
        return {"low": 1.8, "middle": 2.8, "high": 4.2}.get(income_group, 2.8)
    return {"low": 3.8, "middle": 5.2, "high": 8.5}.get(income_group, 5.2)


def debt_service_burden(income_group: str) -> float:
    return {"low": 0.16, "middle": 0.13, "high": 0.09}.get(income_group, 0.13)


def base_mpc(liquidity_group: str, income_group: str, job_loss_risk_type: str) -> float:
    out = 0.72 if liquidity_group == "low" else 0.24
    out += {"low": 0.09, "middle": 0.0, "high": -0.08}.get(income_group, 0.0)
    out += 0.04 if job_loss_risk_type == "high" else -0.02
    return float(np.clip(out, 0.08, 0.92))


def base_saving_rate(liquidity_group: str, income_group: str) -> float:
    return float(np.clip(0.12 + (0.08 if liquidity_group != "low" else -0.03) + (0.05 if income_group == "high" else 0.0), 0.02, 0.35))


def unemployment_higher_to_job_loss(value: float) -> float:
    return float(np.clip(0.24 * float(value), 1.0, 24.0))


def confidence_from_priors(unemployment_higher: float, income_growth: float, inflation: float, income_group: str) -> float:
    income_adjustment = {"low": -4.0, "middle": 0.0, "high": 5.0}.get(income_group, 0.0)
    return float(np.clip(63.0 + income_adjustment - 0.36 * float(unemployment_higher) + 1.2 * float(income_growth) - 0.45 * max(0.0, float(inflation) - 2.0), 0.0, 100.0))


def ecology_input_manifest(
    ecology_bundle: dict[str, Any] | None,
    *,
    period_count: int,
    belief_transform: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    if ecology_bundle is None:
        return None
    manifest = ecology_bundle["manifest"]
    return {
        "root": str(ecology_bundle["root"]),
        "source": ecology_bundle["source"],
        "upstream_status": manifest.get("status"),
        "upstream_prior_update_evidence": manifest.get("prior_update_evidence"),
        "upstream_provider": manifest.get("provider"),
        "upstream_models": manifest.get("models"),
        "upstream_prior_mode": manifest.get("prior_mode"),
        "upstream_feedback_mode": manifest.get("feedback_mode"),
        "upstream_date_mode": manifest.get("date_mode"),
        "requested_phase4_period_count": int(period_count),
        "belief_transform": normalize_belief_transform(belief_transform),
        "coverage": ecology_bundle["coverage"],
        "manifest_sha256": ecology_bundle["manifest_sha256"],
        "panel_sha256": ecology_bundle["panel_sha256"],
        "predictions_sha256": ecology_bundle["predictions_sha256"],
        "adapter_note": (
            "SCE unemployment-higher probabilities are mapped to personal job-loss-risk inputs by a fixed "
            "0.24 multiplier; this is an explicit bridge assumption, not a fitted parameter."
        ),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_output_mapping() -> list[OutputMappingSpec]:
    by_name = {spec.target_name: spec for spec in TARGET_SPECS}
    rows = [
        (
            "pce_mom_pct",
            "aggregate_consumption",
            "pct_change",
            None,
            None,
            None,
            "Nominal PCE proxy maps to aggregate consumption growth.",
        ),
        (
            "real_pce_mom_pct",
            "output",
            "pct_change",
            None,
            None,
            None,
            "Real PCE proxy maps to real output growth in the one-good economy.",
        ),
        (
            "retail_sales_mom_pct",
            "aggregate_consumption",
            "pct_change",
            None,
            None,
            None,
            "Retail proxy maps to aggregate consumption growth; this is deliberately locked before scoring.",
        ),
        (
            "personal_saving_rate_pct",
            "saving_rate",
            "diff",
            "diff",
            -25.0,
            25.0,
            "Saving-rate proxy maps to month-over-month change in aggregate saving divided by aggregate income.",
        ),
        (
            "revolving_credit_mom_pct",
            "aggregate_debt",
            "pct_change",
            None,
            None,
            None,
            "Revolving-credit proxy maps to aggregate household debt growth.",
        ),
    ]
    mapping: list[OutputMappingSpec] = []
    for target_name, variable, economy_transform, target_transform, lower, upper, note in rows:
        target = by_name[target_name]
        mapping.append(
            OutputMappingSpec(
                target_name=target.target_name,
                series_id=target.series_id,
                target_label=target.label,
                target_units=target.units,
                target_transform=target_transform or target.transform,
                economy_variable=variable,
                economy_transform=economy_transform,
                period_alignment="card_i_scores_economy_period_i_plus_1_against_next_month_target",
                lower=target.lower if lower is None else lower,
                upper=target.upper if upper is None else upper,
                note=note,
            )
        )
    return mapping


def normalized_mapping_payload(mapping: Iterable[OutputMappingSpec]) -> dict[str, Any]:
    rows = [spec.__dict__ for spec in mapping]
    return {
        "schema_version": PHASE4_VERSION,
        "leakage_rule": "This mapping is written before forecasts are scored and keyed only by card_id, period_id, and target_name.",
        "rows": sorted(rows, key=lambda row: row["target_name"]),
    }


def mapping_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def economy_proxy_forecasts(
    periods: pd.DataFrame,
    cards: Iterable[Any],
    mapping: Iterable[OutputMappingSpec],
) -> pd.DataFrame:
    period_by_source = {source: group.sort_values("period_index").reset_index(drop=True) for source, group in periods.groupby("source", dropna=False)}
    rows: list[dict[str, Any]] = []
    cards_list = list(cards)
    for source, path in period_by_source.items():
        for card_index, card in enumerate(cards_list):
            target_period = card_index + 1
            if target_period >= path.shape[0]:
                continue
            for spec in mapping:
                prediction = mapped_period_value(path, target_period, spec)
                rows.append(
                    {
                        "card_id": card.card_id,
                        "period_id": card.period_id,
                        "target_name": spec.target_name,
                        "source": str(source),
                        "prediction": float(np.clip(prediction, spec.lower, spec.upper)),
                        "method": f"phase4_locked_{spec.economy_transform}_{spec.economy_variable}",
                    }
                )
    return pd.DataFrame(rows)


def phase4_scoring_targets(targets: pd.DataFrame, mapping: Iterable[OutputMappingSpec]) -> pd.DataFrame:
    if targets.empty:
        return targets.copy()
    by_target = {spec.target_name: spec for spec in mapping}
    out = targets.copy()
    out["raw_target_value"] = out["target_value"]
    out["raw_last_signal"] = out["last_signal"]
    out["phase4_target_transform"] = out["target_name"].astype(str).map(
        {target_name: spec.target_transform for target_name, spec in by_target.items()}
    )
    missing = sorted(set(out["target_name"].astype(str)) - set(by_target))
    if missing:
        raise ValueError(f"Targets have no Phase 4 mapping rows: {', '.join(missing)}")
    for spec in by_target.values():
        mask = out["target_name"].astype(str).eq(spec.target_name)
        if spec.target_transform in {"level", "pct_change"}:
            continue
        if spec.target_transform == "diff":
            available = mask & out["target_value"].map(np.isfinite) & out["last_signal"].map(np.isfinite)
            out.loc[mask, "target_available"] = out.loc[mask, "target_available"].astype(bool) & available
            out.loc[available, "target_value"] = (
                out.loc[available, "raw_target_value"].astype(float) - out.loc[available, "raw_last_signal"].astype(float)
            )
            out.loc[mask, "last_signal"] = 0.0
            out.loc[mask, "history_scale"] = out.loc[mask, "history_scale"].astype(float).clip(lower=0.25)
            continue
        raise ValueError(f"Unsupported Phase 4 target transform: {spec.target_transform}")
    return out


def period_measure(row: pd.Series, spec: OutputMappingSpec) -> float:
    if spec.economy_variable == "saving_rate":
        income = float(row["aggregate_income"])
        if abs(income) <= 1e-9:
            return 0.0
        return 100.0 * float(row["aggregate_saving"]) / income
    return float(row[spec.economy_variable])


def mapped_period_value(path: pd.DataFrame, target_period: int, spec: OutputMappingSpec) -> float:
    current = path.iloc[target_period]
    previous = path.iloc[target_period - 1]
    if spec.economy_transform == "pct_change":
        base = period_measure(previous, spec)
        if abs(base) <= 1e-9:
            return 0.0
        return 100.0 * (period_measure(current, spec) / base - 1.0)
    if spec.economy_transform == "diff":
        return period_measure(current, spec) - period_measure(previous, spec)
    if spec.economy_transform == "level":
        return period_measure(current, spec)
    raise ValueError(f"Unsupported economy transform: {spec.economy_transform}")


def build_path_comparison(periods: pd.DataFrame) -> pd.DataFrame:
    adaptive = periods[periods["variant"].astype(str).eq("adaptive")].copy()
    llm = periods[periods["variant"].astype(str).eq("llm_belief")].copy()
    rows: list[dict[str, Any]] = []
    for _, llm_row in llm.iterrows():
        match = adaptive[
            (adaptive["scenario_id"].astype(str) == str(llm_row["scenario_id"]))
            & (adaptive["period_index"].astype(int) == int(llm_row["period_index"]))
        ]
        if match.empty:
            continue
        adaptive_row = match.iloc[0]
        for variable in COMPARISON_VARIABLES:
            llm_value = float(llm_row[variable])
            adaptive_value = float(adaptive_row[variable])
            rows.append(
                {
                    "scenario_id": str(llm_row["scenario_id"]),
                    "period_id": str(llm_row["period_id"]),
                    "period_index": int(llm_row["period_index"]),
                    "llm_source": str(llm_row["source"]),
                    "adaptive_source": str(adaptive_row["source"]),
                    "variable": variable,
                    "llm_value": llm_value,
                    "adaptive_value": adaptive_value,
                    "delta": llm_value - adaptive_value,
                    "abs_delta": abs(llm_value - adaptive_value),
                }
            )
    return pd.DataFrame(rows)


def classify_phase4_run(
    periods: pd.DataFrame,
    accounting: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    belief_source: str,
    data_mode: str,
) -> dict[str, Any]:
    variants = set(periods["variant"].astype(str)) if not periods.empty else set()
    has_twins = {"llm_belief", "adaptive"}.issubset(variants)
    accounting_ok = max_accounting_abs_residual(accounting) <= ACCOUNTING_TOLERANCE
    has_scores = not scores.empty and "ALL" in set(scores["target_name"].astype(str))
    passed = bool(has_twins and accounting_ok and has_scores)
    if belief_source == "persona_ecology_replay":
        verdict = "phase4_matched_twin_replay_scored" if passed else "phase4_matched_twin_replay_needs_work"
        claim_scope = (
            "Exploratory replay: the runner feeds banked real-SCE prior-update predictions into the deterministic demand economy "
            "and compares that path to an adaptive-expectations twin. This isolates the belief-updater channel, but it is not a "
            "final empirical macro-validity claim until the household panel, replay horizon, and output mapping are all pre-registered "
            "for the same confirmatory run."
        )
        if data_mode == "fixture":
            claim_scope += " Proxy targets are deterministic fixtures in this run."
        return {"passed": passed, "verdict": verdict, "claim_scope": claim_scope}
    return {
        "passed": passed,
        "verdict": "phase4_matched_twin_fixture_ready" if passed else "phase4_matched_twin_fixture_needs_work",
        "claim_scope": (
            "Fixture readiness only: the runner locks the proxy mapping and compares matched LLM-updater and adaptive twins, "
            "but does not establish empirical macro validity without real household state and live/replay belief updates."
        ),
    }


def phase4_winner(scores: pd.DataFrame) -> str | None:
    if scores.empty:
        return None
    overall = scores[scores["target_name"].astype(str).eq("ALL")].sort_values(["rmse_scaled", "source"])
    if overall.empty:
        return None
    return str(overall.iloc[0]["source"])


def max_accounting_abs_residual(accounting: pd.DataFrame) -> float:
    if accounting.empty or "abs_residual" not in accounting:
        return float("inf")
    return float(pd.to_numeric(accounting["abs_residual"], errors="coerce").fillna(float("inf")).max())


def cards_to_frame(cards: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "card_id": card.card_id,
                "period_id": card.period_id,
                "as_of_date": card.as_of_date,
                "target_month": card.target_month,
                "contamination_label": card.contamination_label,
                "prompt_payload": json.dumps(card.prompt_payload, sort_keys=True),
            }
            for card in cards
        ]
    )


def write_outputs(
    output_dir: Path,
    *,
    mapping_payload: dict[str, Any],
    cards: pd.DataFrame,
    context: pd.DataFrame,
    targets: pd.DataFrame,
    households: pd.DataFrame,
    initial: pd.DataFrame,
    beliefs: pd.DataFrame,
    decisions: pd.DataFrame,
    periods: pd.DataFrame,
    accounting: pd.DataFrame,
    comparison: pd.DataFrame,
    forecasts: pd.DataFrame,
    joined: pd.DataFrame,
    scores: pd.DataFrame,
    raw_records: list[dict[str, Any]],
) -> None:
    (output_dir / "phase4_output_mapping.json").write_text(json.dumps(mapping_payload, indent=2, sort_keys=True), encoding="utf-8")
    cards.to_csv(output_dir / "phase4_proxy_cards.csv", index=False)
    context.to_csv(output_dir / "phase4_proxy_context.csv", index=False)
    targets.to_csv(output_dir / "phase4_proxy_targets.csv", index=False)
    households.to_csv(output_dir / "phase4_households.csv", index=False)
    initial.to_csv(output_dir / "phase4_initial_state.csv", index=False)
    beliefs.to_csv(output_dir / "phase4_beliefs.csv", index=False)
    decisions.to_csv(output_dir / "phase4_household_decisions.csv", index=False)
    periods.to_csv(output_dir / "phase4_twin_periods.csv", index=False)
    accounting.to_csv(output_dir / "phase4_twin_accounting.csv", index=False)
    comparison.to_csv(output_dir / "phase4_twin_path_comparison.csv", index=False)
    forecasts.to_csv(output_dir / "phase4_proxy_forecasts.csv", index=False)
    joined.to_csv(output_dir / "phase4_proxy_joined_errors.csv", index=False)
    scores.to_csv(output_dir / "phase4_proxy_scores.csv", index=False)
    (output_dir / "phase4_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True), encoding="utf-8")


def build_report(manifest: dict[str, Any], scores: pd.DataFrame, comparison: pd.DataFrame) -> str:
    overall = scores[scores["target_name"].astype(str).eq("ALL")].sort_values(["rmse_scaled", "source"]) if not scores.empty else scores
    lines = [
        "# Phase 4 Prior-Update Matched Twins",
        "",
        "## Bottom Line",
        phase4_bottom_line(manifest),
        "",
        "## What This Tests",
        phase4_test_description(manifest),
        "",
        "## Behavior Execution",
        phase4_behavior_policy_description(manifest),
        "",
        "## Locked Output Mapping",
        f"- Mapping SHA-256: `{manifest.get('mapping_sha256')}`",
        f"- Mapping schema: `{manifest.get('mapping_spec_version')}`.",
        f"- Scoring label: `{manifest.get('scoring_label')}`.",
        "- Mapping is written to `phase4_output_mapping.json` before proxy forecasts are scored.",
        "- Forecasts join to targets only on `card_id`, `period_id`, and `target_name`.",
        "- In v2, `personal_saving_rate_pct` is scored as month-over-month change in the saving-rate proxy, not the saving-rate level. The transform is applied identically to both twins and to the target series before scoring.",
        "- The first mapping-v2 confirmatory surface has been spent by `macro_confirmatory_fred_2026_02_v1`; existing v2 rescoring remains labeled retrospective.",
        "",
        "## Proxy Scoreboard",
        markdown_table(overall),
        "",
        "## Path Differences",
        markdown_table(comparison.groupby("variable", dropna=False)["abs_delta"].mean().reset_index(name="mean_abs_delta") if not comparison.empty else comparison),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def phase4_test_description(manifest: dict[str, Any]) -> str:
    if manifest.get("belief_source") == "persona_ecology_replay":
        return (
            "This runner compares the same accounting-constrained demand economy under two belief-updating rules: "
            "a banked prior-conditioned LLM belief-updater replay and an adaptive-expectations twin. Real SCE respondent "
            "heterogeneity seeds the household table, while deterministic code owns behavior, aggregation, feedback, and accounting. "
            "The result is exploratory unless the household panel horizon, belief replay horizon, and proxy scoring horizon are "
            "pre-registered for a confirmatory run."
        )
    return (
        "This runner compares the same accounting-constrained demand economy under two belief-updating rules: an LLM belief-updater "
        "fixture and an adaptive-expectations twin. Real household heterogeneity and live prior-update caches are not used in fixture "
        "mode, so this is a readiness gate, not an empirical macro-validity result."
    )


def phase4_behavior_policy_description(manifest: dict[str, Any]) -> str:
    policy = manifest.get("behavior_policy") or {}
    if policy.get("mode") == "schedule":
        return (
            "Both twins use the same LLM-authored behavior-policy schedule. The model supplied transfer and "
            "income-risk response functions upstream; this runner only interpolates those functions against each "
            "household state, enforces budgets, and aggregates. The matched-twin comparison therefore isolates the "
            "belief-updater channel while holding the behavior executor fixed."
        )
    if policy.get("mode") == "state_schedule":
        return (
            "Both twins use the same LLM-authored state-conditioned behavior-policy schedule. The model supplied "
            "bounded policy functions over household balance sheets and belief gaps upstream; this runner matches "
            "each respondent-derived household to the nearest policy profile, interpolates those functions, enforces "
            "budgets, and aggregates. The matched-twin comparison therefore tests whether prior-conditioned LLM belief "
            "updates add value once behavior is executed through the most natural policy-function bridge currently in the repo."
        )
    if policy.get("mode") == "empirical_bridge":
        return (
            "Both twins use the same empirically estimated SCE spending-belief bridge. The bridge maps household belief "
            "changes into a real consumption-growth margin using the accepted v4 Mundlak coefficients; the executor divides "
            "the annual deviation by four for the quarterly demand period, then deterministic code enforces budgets, "
            "saving/debt closure, accounting, and aggregation. The matched-twin comparison therefore isolates the "
            "belief-updater channel while using the measured belief-to-spending bridge."
        )
    if policy.get("mode") == "empirical_bridge_state_schedule":
        return (
            "Both twins use the same hybrid behavior executor. The empirical bridge supplies baseline consumption-growth "
            "movement from belief changes, while the state-conditioned schedule supplies transfer allocation and shock "
            f"response with state weight `{policy.get('state_weight')}`. The executor records both profile hashes, enforces "
            "budgets, and keeps the belief-updater channel isolated."
        )
    return (
        "Both twins use the fixed deterministic demand kernel. This is the older behavior layer and remains available "
        "as the baseline executor for direct comparison with schedule-mode runs."
    )


def phase4_bottom_line(manifest: dict[str, Any]) -> str:
    verdict = manifest.get("verdict", "unknown")
    winner = manifest.get("winner_source")
    if manifest.get("passed"):
        if manifest.get("belief_source") == "persona_ecology_replay":
            return (
                f"`{verdict}`. The replay runner fed banked real-SCE prior-update beliefs into the demand economy, "
                f"ran the adaptive-expectations twin from the same respondent-derived household state, preserved accounting, "
                f"and emitted proxy scores. Best source by scaled RMSE: `{winner}`. This is an exploratory end-to-end result, "
                "not the final macro-validity claim."
            )
        return (
            f"`{verdict}`. The fixture runner locks the post-cutoff proxy mapping, runs matched LLM-updater and "
            f"adaptive twins from the same initial state, preserves accounting, and emits comparable proxy scores. "
            f"Best fixture source by scaled RMSE: `{winner}`. This is readiness, not empirical validity."
        )
    return f"`{verdict}`. The matched-twin fixture did not clear readiness checks."


def output_filenames() -> list[str]:
    return [
        "phase4_output_mapping.json",
        "phase4_proxy_cards.csv",
        "phase4_proxy_context.csv",
        "phase4_proxy_targets.csv",
        "phase4_households.csv",
        "phase4_initial_state.csv",
        "phase4_beliefs.csv",
        "phase4_household_decisions.csv",
        "phase4_twin_periods.csv",
        "phase4_twin_accounting.csv",
        "phase4_twin_path_comparison.csv",
        "phase4_proxy_forecasts.csv",
        "phase4_proxy_joined_errors.csv",
        "phase4_proxy_scores.csv",
        "phase4_raw_records.json",
        "phase4_matched_twins_report.md",
        "manifest.json",
    ]


def base_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": PHASE4_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "data_mode": args.data_mode,
        "cutoff_date": args.cutoff_date,
        "asof_start": args.asof_start,
        "asof_end": args.asof_end,
        "history_months": int(args.history_months),
        "scoreable_only": bool(args.scoreable_only),
        "scoring_label": args.scoring_label,
        "v2_confirmatory_status": "first_mapping_v2_confirmatory_surface_spent_by_macro_confirmatory_fred_2026_02_v1",
        "belief_source": args.belief_source,
        "persona_ecology_dir": args.persona_ecology_dir,
        "primary_ecology_source": args.primary_ecology_source,
        "ecology_period_policy": args.ecology_period_policy,
        "household_source": args.household_source,
        "household_count_requested": int(args.household_count),
        "period_count_requested": int(args.period_count),
        "feedback_mode": args.feedback_mode,
        "behavior_policy_mode": args.behavior_policy_mode,
        "behavior_policy_raw_records_json": args.behavior_policy_raw_records_json,
        "behavior_policy_state_profile_json": args.behavior_policy_state_profile_json,
        "empirical_bridge_json": args.empirical_bridge_json,
        "hybrid_state_weight": float(args.hybrid_state_weight),
        "belief_transform": belief_transform_from_args(args),
        "feedback_gain_multiplier": float(args.feedback_gain_multiplier),
        "fresh_cache": bool(args.fresh_cache),
        "max_live_calls": int(args.max_live_calls),
        "status": "running",
        "live_mode_blocked_until": "real household prior-state panel, replay cache contract, and empirical mapping approval are supplied",
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
