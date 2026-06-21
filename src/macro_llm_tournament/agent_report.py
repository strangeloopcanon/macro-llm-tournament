from __future__ import annotations

import json

import pandas as pd

from .agent_common import markdown_table


def build_agent_economy_report(
    manifest: dict,
    forecast_scores: pd.DataFrame,
    type_cells: pd.DataFrame,
    aggregates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    agent_scores: pd.DataFrame,
    *,
    agent_belief_target_scores: pd.DataFrame | None = None,
) -> str:
    llm_rows = forecast_scores[forecast_scores["source"].astype(str).str.startswith("llm_")] if not forecast_scores.empty else pd.DataFrame()
    overall_forecasts = forecast_scores[forecast_scores["variable"] == "ALL"].copy() if not forecast_scores.empty else pd.DataFrame()
    max_coverage = int(overall_forecasts["n"].max()) if not overall_forecasts.empty else 0
    comparable_forecasts = overall_forecasts[overall_forecasts["n"] == max_coverage].copy() if max_coverage else pd.DataFrame()
    partial_forecasts = overall_forecasts[overall_forecasts["n"] < max_coverage].copy() if max_coverage else pd.DataFrame()
    best_forecast = comparable_forecasts.sort_values("rmse").head(1) if not comparable_forecasts.empty else pd.DataFrame()
    lines = [
        "# Forecast-First Typed Agent Economy",
        "",
        "## Bottom Line",
        _agent_bottom_line(manifest, best_forecast, llm_rows, diagnostics),
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- LLM mode: `{manifest.get('llm_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Fresh forecast cache: `{manifest.get('fresh_forecast_cache')}`",
        f"- Agent mode: `{manifest.get('agent_mode')}`",
        f"- Agent live calls used: `{manifest.get('agent_live_call_count', 0)}` of cap `{manifest.get('max_agent_live_calls', 0)}`",
        f"- Agent cache hits: `{manifest.get('agent_cache_hit_count', 0)}`",
        f"- Fresh agent cache: `{manifest.get('fresh_agent_cache')}`",
        f"- Household policy: `{manifest.get('household_policy', 'direct')}`",
        f"- Feedback mode: `{manifest.get('feedback_mode', 'closed_loop')}`",
        f"- Counterfactual shocks: `{', '.join(manifest.get('counterfactual_shocks', [])) or 'none'}`",
        f"- Forecast cards: `{manifest.get('card_count')}`",
        f"- Holdout: `{manifest.get('holdout_start_year')}`-`{manifest.get('holdout_end_year')}`",
        f"- Variables: `{', '.join(manifest.get('variables', []))}`",
        f"- Belief sources: `{', '.join(manifest.get('belief_source_filters', []))}`",
        f"- SCF type source: `{manifest.get('household_type_status', {}).get('status', 'unknown')}`",
        f"- Vintage context: `{manifest.get('vintage_context_status', {}).get('status', 'unknown')}`",
        f"- Survey belief context: `{manifest.get('survey_belief_status', {}).get('status', 'unknown')}`",
        "",
        "## Forecast Leaderboard",
        markdown_table(
            comparable_forecasts.sort_values("rmse")
            [["source", "n", "rmse", "mae", "bias", "direction_accuracy"]]
            .head(12)
            if not comparable_forecasts.empty
            else pd.DataFrame()
        ),
        "",
        *_partial_forecast_section(partial_forecasts),
        "## Household Type Cells",
        markdown_table(
            type_cells[
                [
                    "type_id",
                    "population_weight",
                    "annual_income",
                    "liquid_assets",
                    "illiquid_assets",
                    "debt",
                    "liquid_buffer_months",
                    "credit_limit_proxy",
                    "source",
                ]
            ]
        ),
        "",
        "## Agent Economy Aggregates",
        markdown_table(
            aggregates[
                [
                    "source",
                    "origin",
                    "variable",
                    "aggregate_consumption_change_pct",
                    "aggregate_liquid_buffer_change",
                    "aggregate_borrowing",
                    "aggregate_debt_repayment",
                    "credit_rationing_ratio",
                    "firm_hiring_index",
                    "firm_price_pressure_index",
                ]
            ].head(16)
            if not aggregates.empty
            else pd.DataFrame()
        ),
        "",
        "## Accounting Diagnostics",
        markdown_table(
            diagnostics[
                [
                    "source",
                    "origin",
                    "variable",
                    "household_rows",
                    "max_abs_cash_residual",
                    "max_abs_networth_residual",
                    "credit_market_gap",
                    "passes_accounting",
                ]
            ].head(16)
            if not diagnostics.empty
            else pd.DataFrame()
        ),
        "",
        "## Agent Scores",
        markdown_table(agent_scores.sort_values(["score", "source"]).head(16) if not agent_scores.empty else pd.DataFrame()),
        "",
        "## Household Belief Target Scores",
        markdown_table(
            agent_belief_target_scores.sort_values(["rmse", "source"]).head(16)
            if agent_belief_target_scores is not None and not agent_belief_target_scores.empty
            else pd.DataFrame()
        ),
        "",
        "## What This Run Means",
        (
            "This run uses LLM or control macro beliefs as inputs to typed household, firm, and bank agents, "
            "then lets deterministic accounting decide feasible spending, borrowing, portfolio movement, and "
            "aggregation. Agent state advances once per SPF origin, so multiple variable cards from the same "
            "survey date do not become artificial time steps. In closed-loop feedback mode, firm hiring, price "
            "pressure, and bank credit supply from each event update the next-origin household income and credit "
            "state. Household belief targets are scored once per origin rather than once per variable card."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _agent_bottom_line(
    manifest: dict,
    best_forecast: pd.DataFrame,
    llm_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    accounting_ok = bool(not diagnostics.empty and diagnostics["passes_accounting"].all())
    forecast_live_calls = int(manifest.get("live_call_count") or 0)
    agent_live_calls = int(manifest.get("agent_live_call_count") or 0)
    if best_forecast.empty:
        forecast_text = "Forecast scoring did not produce a leaderboard."
    else:
        row = best_forecast.iloc[0]
        forecast_text = f"Best forecast source is `{row['source']}` with RMSE `{float(row['rmse']):.4f}`."
    llm_text = "An LLM forecast row is present." if not llm_rows.empty else "No LLM forecast row is present."
    accounting_text = "Accounting diagnostics pass." if accounting_ok else "Accounting diagnostics need inspection."
    return (
        f"{forecast_text} {llm_text} The run used `{forecast_live_calls}` forecast live calls and "
        f"`{agent_live_calls}` agent live calls, then produced persistent typed-agent state, desired actions, "
        f"feasible actions, aggregate outcomes, and diagnostics. {accounting_text}"
    )


def _partial_forecast_section(partial_forecasts: pd.DataFrame) -> list[str]:
    if partial_forecasts.empty:
        return []
    return [
        "## Partial-Coverage Forecast Diagnostics",
        (
            "These forecast rows do not cover the same card set as the main leaderboard and should not be "
            "read as the headline rank."
        ),
        markdown_table(
            partial_forecasts.sort_values("rmse")[
                ["source", "n", "rmse", "mae", "bias", "direction_accuracy"]
            ]
        ),
        "",
    ]
