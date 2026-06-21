from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .agent_common import (
    AGENT_ECONOMY_VERSION,
    AGENT_LLM_CACHE_ROOT,
    LLM_CACHE_ROOT,
    OUTPUT_ROOT,
    WORK_ROOT,
    max_abs,
)
from .agent_llm import AgentLLMClient, agent_prompt, fixture_agent_payload, normalize_agent_payload
from .agent_report import build_agent_economy_report
from .agent_runtime import FEEDBACK_MODE_CHOICES, HOUSEHOLD_POLICY_CHOICES, run_agent_economy
from .agent_targets import build_agent_belief_target_rows, score_agent_belief_targets
from .agent_types import AgentTypeDefinition, build_household_type_cells
from .forecast_cards import (
    FORECAST_CARD_PROMPT_VERSION,
    ForecastCard,
    assert_no_prompt_target_leakage,
    build_forecast_cards,
    cards_to_frame,
    enrich_forecast_cards,
)
from .forecast_controls import build_control_forecasts
from .forecast_data import WORK_ROOT as SPF_WORK_ROOT
from .forecast_data import load_spf_error_data, parse_variable_list
from .forecast_llm import ForecastLLMClient, run_llm_forecasts
from .forecast_scoring import score_forecasts, verdict_from_scores
from .fred_vintage import WORK_ROOT as FRED_VINTAGE_WORK_ROOT
from .fred_vintage import build_vintage_context_for_cards
from .llm_common import LLMUnavailable
from .survey_beliefs import WORK_ROOT as SURVEY_BELIEF_WORK_ROOT
from .survey_beliefs import load_survey_belief_targets, survey_context_by_card


__all__ = [
    "AgentLLMClient",
    "AgentTypeDefinition",
    "add_counterfactual_forecasts",
    "agent_prompt",
    "build_agent_belief_target_rows",
    "build_agent_economy_report",
    "build_agent_forecast_inputs",
    "build_household_type_cells",
    "fixture_agent_payload",
    "normalize_agent_payload",
    "run_agent_economy",
    "score_agent_belief_targets",
]


COUNTERFACTUAL_SHOCKS: dict[str, dict[str, float]] = {
    "rate_hike": {"TBILL": 1.00, "TBOND": 0.70, "RGDP": -0.40, "UNEMP": 0.25, "CPI": -0.15},
    "inflation_spike": {"CPI": 1.50, "TBILL": 0.80, "TBOND": 0.55, "RGDP": -0.30, "UNEMP": 0.20},
    "growth_slump": {"RGDP": -1.50, "UNEMP": 1.00, "CPI": -0.35, "TBILL": -0.40, "TBOND": -0.30},
    "credit_crunch": {"RGDP": -0.90, "UNEMP": 0.65, "CPI": -0.20, "TBILL": 0.20, "TBOND": 0.35},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forecast-first typed agent economy with accounting discipline.")
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--llm-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-forecast-cache", action="store_true")
    parser.add_argument("--agent-mode", choices=["rules", "fixture", "replay", "live"], default="rules")
    parser.add_argument("--max-agent-live-calls", type=int, default=0)
    parser.add_argument("--fresh-agent-cache", action="store_true")
    parser.add_argument("--variables", default="CPI,RGDP,UNEMP,TBILL")
    parser.add_argument("--horizons", default="1")
    parser.add_argument("--holdout-start-year", type=int, default=2019)
    parser.add_argument("--holdout-end-year", type=int, default=2024)
    parser.add_argument("--history-quarters", type=int, default=24)
    parser.add_argument("--card-count", type=int, default=8)
    parser.add_argument("--refresh-spf", action="store_true")
    parser.add_argument("--vintage-context", choices=["off", "best_effort", "require"], default="best_effort")
    parser.add_argument("--refresh-fred-vintage", action="store_true")
    parser.add_argument("--belief-targets", choices=["off", "best_effort", "require"], default="best_effort")
    parser.add_argument("--refresh-belief-targets", action="store_true")
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--belief-sources", default="llm,spf_consensus,constant_gain,recursive_least_squares")
    parser.add_argument("--household-policy", choices=HOUSEHOLD_POLICY_CHOICES, default="direct")
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODE_CHOICES, default="closed_loop")
    parser.add_argument("--counterfactual-shocks", default="")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.llm_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --llm-mode live is used")
    if args.agent_mode == "live" and args.max_agent_live_calls <= 0:
        raise SystemExit("--max-agent-live-calls must be positive when --agent-mode live is used")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"agent_economy_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = parse_variable_list(args.variables)
    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]
    source_filters = [part.strip() for part in args.belief_sources.split(",") if part.strip()]
    counterfactual_shocks = [part.strip() for part in args.counterfactual_shocks.split(",") if part.strip()]
    manifest: dict[str, Any] = {
        "schema_version": AGENT_ECONOMY_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "llm_mode": args.llm_mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_forecast_cache": bool(args.fresh_forecast_cache),
        "agent_mode": args.agent_mode,
        "max_agent_live_calls": int(args.max_agent_live_calls),
        "fresh_agent_cache": bool(args.fresh_agent_cache),
        "variables": variables,
        "horizons": horizons,
        "holdout_start_year": int(args.holdout_start_year),
        "holdout_end_year": int(args.holdout_end_year),
        "history_quarters": int(args.history_quarters),
        "requested_card_count": int(args.card_count),
        "card_prompt_version": FORECAST_CARD_PROMPT_VERSION,
        "vintage_context_mode": args.vintage_context,
        "belief_targets_mode": args.belief_targets,
        "scf_wave": int(args.scf_wave),
        "belief_source_filters": source_filters,
        "household_policy": args.household_policy,
        "feedback_mode": args.feedback_mode,
        "counterfactual_shocks": counterfactual_shocks,
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    try:
        cards, cards_frame, data_frames, context_status = build_agent_forecast_inputs(
            variables=variables,
            horizons=horizons,
            holdout_start_year=args.holdout_start_year,
            holdout_end_year=args.holdout_end_year,
            history_quarters=args.history_quarters,
            card_count=args.card_count,
            refresh_spf=args.refresh_spf,
            vintage_context=args.vintage_context,
            refresh_fred_vintage=args.refresh_fred_vintage,
            belief_targets=args.belief_targets,
            refresh_belief_targets=args.refresh_belief_targets,
        )
        _write_input_frames(output_dir, data_frames, context_status)

        control_forecasts = build_control_forecasts(data_frames["spf_error_observations"], cards, tune_end_year=args.holdout_start_year - 1)
        control_forecasts.to_csv(output_dir / "control_forecasts.csv", index=False)
        cache_dir = output_dir / "fresh_llm_cache" if args.fresh_forecast_cache else LLM_CACHE_ROOT
        client = ForecastLLMClient(args.provider, args.model, cache_dir, mode=args.llm_mode, max_live_calls=args.max_live_calls)
        llm_forecasts, raw_records = run_llm_forecasts(client, cards)
        recall_probe = client.recall_probe()
        llm_forecasts.to_json(output_dir / "llm_forecasts.jsonl", orient="records", lines=True)
        (output_dir / "llm_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "recall_probe.json").write_text(json.dumps(recall_probe, indent=2, sort_keys=True), encoding="utf-8")

        all_forecasts = pd.concat([control_forecasts, llm_forecasts], ignore_index=True)
        all_forecasts.to_csv(output_dir / "all_forecasts.csv", index=False)
        scores, _behavior, _joined = score_forecasts(cards_frame, all_forecasts)
        verdict = verdict_from_scores(scores)
        scores.to_csv(output_dir / "forecast_scores.csv", index=False)
        agent_forecasts, counterfactual_rows = add_counterfactual_forecasts(all_forecasts, counterfactual_shocks)
        agent_forecasts.to_csv(output_dir / "agent_forecasts.csv", index=False)
        counterfactual_rows.to_csv(output_dir / "counterfactual_scenarios.csv", index=False)

        type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        type_cells.to_csv(output_dir / "household_type_cells.csv", index=False)
        agent_client = _build_agent_client(args, output_dir)
        state_initial, desired_actions, feasible_actions, aggregates, diagnostics, agent_scores = run_agent_economy(
            cards,
            agent_forecasts,
            type_cells,
            source_filters=source_filters,
            agent_client=agent_client,
            household_policy=args.household_policy,
            feedback_mode=args.feedback_mode,
        )
        _write_agent_outputs(output_dir, state_initial, desired_actions, feasible_actions, aggregates, diagnostics, agent_scores)
        agent_belief_targets = build_agent_belief_target_rows(cards, data_frames["survey_belief_targets"])
        agent_belief_target_scores = score_agent_belief_targets(aggregates, agent_belief_targets)
        agent_belief_targets.to_csv(output_dir / "agent_belief_targets.csv", index=False)
        agent_belief_target_scores.to_csv(output_dir / "agent_belief_target_scores.csv", index=False)
        agent_raw_records = agent_client.raw_records if agent_client is not None else []
        (output_dir / "agent_llm_raw_records.json").write_text(json.dumps(agent_raw_records, indent=2, sort_keys=True), encoding="utf-8")

        manifest.update(
            {
                "status": "ok",
                "card_count": int(len(cards)),
                "card_regime_counts": cards_frame["regime_label"].value_counts().sort_index().to_dict(),
                "card_contamination_counts": cards_frame["contamination_label"].value_counts().sort_index().to_dict(),
                "vintage_context_status": context_status["vintage_context_status"],
                "survey_belief_status": context_status["survey_belief_status"],
                "survey_belief_target_rows": int(data_frames["survey_belief_targets"].shape[0]),
                "survey_belief_card_rows": int(data_frames["survey_belief_targets_by_card"].shape[0]),
                "forecast_rows": int(all_forecasts.shape[0]),
                "agent_forecast_rows": int(agent_forecasts.shape[0]),
                "counterfactual_added_forecast_rows": int(agent_forecasts.shape[0] - all_forecasts.shape[0]),
                "counterfactual_scenario_rows": int(counterfactual_rows.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "llm_cache_dir": _safe_relative(cache_dir),
                "agent_live_call_count": int(agent_client.live_call_count if agent_client is not None else 0),
                "agent_cache_hit_count": int(agent_client.cache_hit_count if agent_client is not None else 0),
                "agent_cache_dir": _safe_relative(_agent_cache_dir(args, output_dir)),
                "forecast_verdict": verdict,
                "household_type_status": type_status,
                "household_type_count": int(type_cells.shape[0]),
                "agent_llm_raw_record_count": int(len(agent_raw_records)),
                "agent_desired_action_rows": int(desired_actions.shape[0]),
                "agent_feasible_action_rows": int(feasible_actions.shape[0]),
                "agent_aggregate_rows": int(aggregates.shape[0]),
                "accounting_max_abs_cash_residual": max_abs(diagnostics, "max_abs_cash_residual"),
                "accounting_max_abs_networth_residual": max_abs(diagnostics, "max_abs_networth_residual"),
                "agent_score_rows": int(agent_scores.shape[0]),
                "agent_belief_target_rows": int(agent_belief_targets.shape[0]),
                "agent_belief_target_score_rows": int(agent_belief_target_scores.shape[0]),
                "agent_state_update_granularity": "origin",
                "outputs": _output_names(),
            }
        )
        report = build_agent_economy_report(
            manifest,
            scores,
            type_cells,
            aggregates,
            diagnostics,
            agent_scores,
            agent_belief_target_scores=agent_belief_target_scores,
        )
        (output_dir / "agent_economy_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except (Exception, LLMUnavailable) as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


def add_counterfactual_forecasts(
    forecasts: pd.DataFrame,
    shocks: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shock_names = [str(shock).strip() for shock in shocks if str(shock).strip()]
    unknown = sorted(set(shock_names) - set(COUNTERFACTUAL_SHOCKS))
    if unknown:
        raise ValueError(f"Unknown counterfactual shocks: {', '.join(unknown)}")
    if forecasts.empty or not shock_names:
        return forecasts.copy(), pd.DataFrame()

    rows: list[pd.DataFrame] = []
    scenario_rows: list[dict[str, Any]] = []
    adjustable_columns = ["point_forecast", "p10", "p50", "p90", "panel_mean"]
    for shock in shock_names:
        adjustments = COUNTERFACTUAL_SHOCKS[shock]
        frame = forecasts.copy()
        frame["base_source"] = frame["source"].astype(str)
        frame["source"] = frame["base_source"].map(lambda source: f"{source}__cf_{shock}")
        frame["counterfactual_shock"] = shock
        for variable, delta in adjustments.items():
            mask = frame["variable"].astype(str).eq(variable)
            for column in adjustable_columns:
                if column in frame:
                    frame.loc[mask, column] = pd.to_numeric(frame.loc[mask, column], errors="coerce") + float(delta)
            scenario_rows.append(
                {
                    "counterfactual_shock": shock,
                    "variable": variable,
                    "forecast_delta": float(delta),
                    "source_count": int(forecasts["source"].nunique()),
                    "row_count": int(mask.sum()),
                }
            )
        rows.append(frame)
    return pd.concat([forecasts, *rows], ignore_index=True), pd.DataFrame(scenario_rows)


def build_agent_forecast_inputs(
    *,
    variables: Iterable[str],
    horizons: Iterable[int],
    holdout_start_year: int,
    holdout_end_year: int,
    history_quarters: int,
    card_count: int,
    refresh_spf: bool,
    vintage_context: str,
    refresh_fred_vintage: bool,
    belief_targets: str,
    refresh_belief_targets: bool,
) -> tuple[list[ForecastCard], pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    spf_data = load_spf_error_data(variables, work_dir=SPF_WORK_ROOT, refresh=refresh_spf)
    cards = build_forecast_cards(
        spf_data,
        variables=variables,
        horizons=horizons,
        holdout_start_year=holdout_start_year,
        holdout_end_year=holdout_end_year,
        history_quarters=history_quarters,
        card_count=card_count,
    )
    if not cards:
        raise ValueError("No forecast cards built for requested agent-economy selection.")
    base_card_ids = [card.card_id for card in cards]
    vintage_contexts, vintage_rows, vintage_status = build_vintage_context_for_cards(
        cards,
        work_dir=FRED_VINTAGE_WORK_ROOT,
        refresh=refresh_fred_vintage,
        mode=vintage_context,
    )
    survey_targets, survey_status = load_survey_belief_targets(
        work_dir=SURVEY_BELIEF_WORK_ROOT,
        refresh=refresh_belief_targets,
        mode=belief_targets,
    )
    survey_contexts, survey_by_card = survey_context_by_card(cards, survey_targets)
    cards = enrich_forecast_cards(
        cards,
        vintage_context_by_card=vintage_contexts if vintage_context != "off" else None,
        survey_context_by_card=survey_contexts if belief_targets != "off" else None,
    )
    id_map = {old: new.card_id for old, new in zip(base_card_ids, cards)}
    if not vintage_rows.empty and "card_id" in vintage_rows:
        vintage_rows["card_id"] = vintage_rows["card_id"].map(id_map).fillna(vintage_rows["card_id"])
    if not survey_by_card.empty and "card_id" in survey_by_card:
        survey_by_card["card_id"] = survey_by_card["card_id"].map(id_map).fillna(survey_by_card["card_id"])
    assert_no_prompt_target_leakage(cards)
    cards_frame = cards_to_frame(cards)
    return (
        cards,
        cards_frame,
        {
            "forecast_cards": cards_frame,
            "spf_error_observations": spf_data,
            "fred_vintage_context": vintage_rows,
            "survey_belief_targets": survey_targets,
            "survey_belief_targets_by_card": survey_by_card,
        },
        {"vintage_context_status": vintage_status, "survey_belief_status": survey_status},
    )


def _build_agent_client(args: argparse.Namespace, output_dir: Path) -> AgentLLMClient | None:
    if args.agent_mode == "rules":
        return None
    return AgentLLMClient(
        args.provider,
        args.model,
        _agent_cache_dir(args, output_dir),
        mode=args.agent_mode,
        max_live_calls=args.max_agent_live_calls,
    )


def _agent_cache_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    return output_dir / "fresh_agent_cache" if args.fresh_agent_cache else AGENT_LLM_CACHE_ROOT


def _safe_relative(path: Path) -> str:
    return str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)


def _write_input_frames(output_dir: Path, data_frames: dict[str, pd.DataFrame], context_status: dict[str, Any]) -> None:
    for name, frame in data_frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "fred_vintage_status.json").write_text(json.dumps(context_status["vintage_context_status"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "survey_belief_status.json").write_text(json.dumps(context_status["survey_belief_status"], indent=2, sort_keys=True), encoding="utf-8")


def _write_agent_outputs(
    output_dir: Path,
    state_initial: pd.DataFrame,
    desired_actions: pd.DataFrame,
    feasible_actions: pd.DataFrame,
    aggregates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    agent_scores: pd.DataFrame,
) -> None:
    state_initial.to_json(output_dir / "agent_state_initial.jsonl", orient="records", lines=True)
    desired_actions.to_json(output_dir / "agent_desired_actions.jsonl", orient="records", lines=True)
    feasible_actions.to_csv(output_dir / "agent_feasible_actions.csv", index=False)
    aggregates.to_csv(output_dir / "agent_aggregate_outcomes.csv", index=False)
    diagnostics.to_csv(output_dir / "agent_accounting_diagnostics.csv", index=False)
    agent_scores.to_csv(output_dir / "agent_scores.csv", index=False)


def _output_names() -> list[str]:
    return [
        "forecast_cards.csv",
        "llm_forecasts.jsonl",
        "control_forecasts.csv",
        "agent_forecasts.csv",
        "counterfactual_scenarios.csv",
        "forecast_scores.csv",
        "household_type_cells.csv",
        "agent_state_initial.jsonl",
        "agent_desired_actions.jsonl",
        "agent_feasible_actions.csv",
        "agent_aggregate_outcomes.csv",
        "agent_accounting_diagnostics.csv",
        "agent_scores.csv",
        "agent_belief_targets.csv",
        "agent_belief_target_scores.csv",
        "agent_llm_raw_records.json",
        "agent_economy_report.md",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
