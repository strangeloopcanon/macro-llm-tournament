from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .forecast_agent_panel import build_forecast_agent_panel
from .forecast_cards import (
    FORECAST_CARD_PROMPT_VERSION,
    assert_no_prompt_target_leakage,
    build_forecast_cards,
    cards_to_frame,
    enrich_forecast_cards,
)
from .forecast_controls import build_control_forecasts
from .forecast_data import WORK_ROOT as SPF_WORK_ROOT
from .forecast_data import load_spf_error_data, parse_variable_list
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS, run_llm_forecasts
from .forecast_report import build_forecast_report
from .forecast_scoring import score_forecast_slices, score_forecasts, verdict_from_scores
from .fred_vintage import WORK_ROOT as FRED_VINTAGE_WORK_ROOT
from .fred_vintage import build_vintage_context_for_cards
from .llm_common import LLMUnavailable
from .survey_beliefs import WORK_ROOT as SURVEY_BELIEF_WORK_ROOT
from .survey_beliefs import load_survey_belief_targets, survey_context_by_card


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LLM_CACHE_ROOT = PROJECT_ROOT / "work" / "llm_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct SPF forecast tournament for LLM beliefs.")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--llm-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--variables", default="CPI,RGDP,UNEMP,TBILL")
    parser.add_argument("--horizons", default="1")
    parser.add_argument("--holdout-start-year", type=int, default=2015)
    parser.add_argument("--holdout-end-year", type=int, default=2024)
    parser.add_argument("--history-quarters", type=int, default=24)
    parser.add_argument("--card-count", type=int, default=12)
    parser.add_argument("--refresh-spf", action="store_true")
    parser.add_argument("--vintage-context", choices=["off", "best_effort", "require"], default="best_effort")
    parser.add_argument("--refresh-fred-vintage", action="store_true")
    parser.add_argument("--belief-targets", choices=["off", "best_effort", "require"], default="best_effort")
    parser.add_argument("--refresh-belief-targets", action="store_true")
    parser.add_argument("--typed-agent-panel", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.llm_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --llm-mode live is used")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"spf_forecast_tournament_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = parse_variable_list(args.variables)
    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]
    manifest: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "llm_mode": args.llm_mode,
        "max_live_calls": int(args.max_live_calls),
        "variables": variables,
        "horizons": horizons,
        "holdout_start_year": int(args.holdout_start_year),
        "holdout_end_year": int(args.holdout_end_year),
        "history_quarters": int(args.history_quarters),
        "requested_card_count": int(args.card_count),
        "card_prompt_version": FORECAST_CARD_PROMPT_VERSION,
        "vintage_context_mode": args.vintage_context,
        "belief_targets_mode": args.belief_targets,
        "typed_agent_panel": bool(args.typed_agent_panel),
        "status": "running",
        "caveats": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    try:
        spf_data = load_spf_error_data(variables, work_dir=SPF_WORK_ROOT, refresh=args.refresh_spf)
        cards = build_forecast_cards(
            spf_data,
            variables=variables,
            horizons=horizons,
            holdout_start_year=args.holdout_start_year,
            holdout_end_year=args.holdout_end_year,
            history_quarters=args.history_quarters,
            card_count=args.card_count,
        )
        if not cards:
            raise ValueError("No forecast cards built for requested SPF selection.")
        base_card_ids = [card.card_id for card in cards]
        vintage_contexts, vintage_rows, vintage_status = build_vintage_context_for_cards(
            cards,
            work_dir=FRED_VINTAGE_WORK_ROOT,
            refresh=args.refresh_fred_vintage,
            mode=args.vintage_context,
        )
        survey_targets, survey_status = load_survey_belief_targets(
            work_dir=SURVEY_BELIEF_WORK_ROOT,
            refresh=args.refresh_belief_targets,
            mode=args.belief_targets,
        )
        survey_contexts, survey_by_card = survey_context_by_card(cards, survey_targets)
        cards = enrich_forecast_cards(
            cards,
            vintage_context_by_card=vintage_contexts if args.vintage_context != "off" else None,
            survey_context_by_card=survey_contexts if args.belief_targets != "off" else None,
        )
        id_map = {old: new.card_id for old, new in zip(base_card_ids, cards)}
        if not vintage_rows.empty and "card_id" in vintage_rows:
            vintage_rows["card_id"] = vintage_rows["card_id"].map(id_map).fillna(vintage_rows["card_id"])
        if not survey_by_card.empty and "card_id" in survey_by_card:
            survey_by_card["card_id"] = survey_by_card["card_id"].map(id_map).fillna(survey_by_card["card_id"])
        assert_no_prompt_target_leakage(cards)
        cards_frame = cards_to_frame(cards)
        cards_frame.to_csv(output_dir / "forecast_cards.csv", index=False)
        spf_data.to_csv(output_dir / "spf_error_observations.csv", index=False)
        vintage_rows.to_csv(output_dir / "fred_vintage_context.csv", index=False)
        (output_dir / "fred_vintage_status.json").write_text(
            json.dumps(vintage_status, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        survey_targets.to_csv(output_dir / "survey_belief_targets.csv", index=False)
        survey_by_card.to_csv(output_dir / "survey_belief_targets_by_card.csv", index=False)
        (output_dir / "survey_belief_status.json").write_text(
            json.dumps(survey_status, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        control_forecasts = build_control_forecasts(spf_data, cards, tune_end_year=args.holdout_start_year - 1)
        control_forecasts.to_csv(output_dir / "control_forecasts.csv", index=False)

        client = ForecastLLMClient(
            args.provider,
            args.model,
            LLM_CACHE_ROOT,
            mode=args.llm_mode,
            max_live_calls=args.max_live_calls,
        )
        llm_forecasts, raw_records = run_llm_forecasts(client, cards)
        recall_probe = client.recall_probe()
        llm_forecasts.to_json(output_dir / "llm_forecasts.jsonl", orient="records", lines=True)
        (output_dir / "llm_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "recall_probe.json").write_text(json.dumps(recall_probe, indent=2, sort_keys=True), encoding="utf-8")

        all_forecasts = pd.concat([control_forecasts, llm_forecasts], ignore_index=True)
        all_forecasts.to_csv(output_dir / "all_forecasts.csv", index=False)
        agent_panel_rows = pd.DataFrame()
        agent_aggregate_rows = pd.DataFrame()
        if args.typed_agent_panel:
            agent_panel_rows, agent_aggregate_rows = build_forecast_agent_panel(cards, all_forecasts)
            agent_panel_rows.to_csv(output_dir / "forecast_agent_panel.csv", index=False)
            agent_aggregate_rows.to_csv(output_dir / "forecast_agent_aggregates.csv", index=False)
        scores, behavior, joined = score_forecasts(cards_frame, all_forecasts)
        slice_scores = score_forecast_slices(cards_frame, joined)
        verdict = verdict_from_scores(scores)
        scores.to_csv(output_dir / "forecast_scores.csv", index=False)
        slice_scores.to_csv(output_dir / "forecast_slice_scores.csv", index=False)
        behavior.to_csv(output_dir / "behavioral_coefficients.csv", index=False)
        joined.to_csv(output_dir / "forecast_joined_errors.csv", index=False)

        manifest.update(
            {
                "status": "ok",
                "card_count": int(len(cards)),
                "card_regime_counts": cards_frame["regime_label"].value_counts().sort_index().to_dict(),
                "card_evaluation_split_counts": cards_frame["evaluation_split"].value_counts().sort_index().to_dict(),
                "card_contamination_counts": cards_frame["contamination_label"].value_counts().sort_index().to_dict(),
                "vintage_context_status": vintage_status,
                "survey_belief_status": survey_status,
                "survey_belief_target_rows": int(survey_targets.shape[0]),
                "survey_belief_card_rows": int(survey_by_card.shape[0]),
                "typed_agent_panel_rows": int(agent_panel_rows.shape[0]),
                "typed_agent_aggregate_rows": int(agent_aggregate_rows.shape[0]),
                "forecast_rows": int(all_forecasts.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "verdict": verdict,
                "outputs": [
                    "forecast_cards.csv",
                    "fred_vintage_context.csv",
                    "fred_vintage_status.json",
                    "survey_belief_targets.csv",
                    "survey_belief_targets_by_card.csv",
                    "survey_belief_status.json",
                    "llm_forecasts.jsonl",
                    "control_forecasts.csv",
                    "forecast_scores.csv",
                    "forecast_slice_scores.csv",
                    "behavioral_coefficients.csv",
                    "forecast_tournament_report.md",
                ],
            }
        )
        if args.typed_agent_panel:
            manifest["outputs"].extend(["forecast_agent_panel.csv", "forecast_agent_aggregates.csv"])
        report = build_forecast_report(
            manifest=manifest,
            scores=scores,
            behavior=behavior,
            verdict=verdict,
            slice_scores=slice_scores,
        )
        (output_dir / "forecast_tournament_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except (Exception, LLMUnavailable) as exc:
        manifest.update({"status": "failed", "error": str(exc), "trace": traceback.format_exc(limit=4)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
