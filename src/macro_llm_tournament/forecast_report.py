from __future__ import annotations

import json
from typing import Any

import pandas as pd


def build_forecast_report(
    *,
    manifest: dict[str, Any],
    scores: pd.DataFrame,
    behavior: pd.DataFrame,
    verdict: dict[str, Any],
    slice_scores: pd.DataFrame | None = None,
) -> str:
    overall = scores[scores["variable"] == "ALL"].sort_values("rmse").copy()
    llm_source = str(verdict.get("llm_source", ""))
    llm_n = int(verdict.get("llm_n") or 0)
    comparable_rows = overall[overall["n"] >= llm_n].head(8) if llm_n else overall.head(8)
    partial_rows = overall[overall["n"] < llm_n].head(8) if llm_n else overall.iloc[0:0]
    lines = [
        "# SPF Direct Forecast Tournament",
        "",
        "## Bottom Line",
        _verdict_sentence(verdict),
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- LLM mode: `{manifest.get('llm_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Forecast cards: `{manifest.get('card_count')}`",
        f"- Holdout: `{manifest.get('holdout_start_year')}`-`{manifest.get('holdout_end_year')}`",
        f"- Variables: `{', '.join(manifest.get('variables', []))}`",
        f"- Horizons: `{', '.join(str(value) for value in manifest.get('horizons', []))}`",
        f"- Card prompt version: `{manifest.get('card_prompt_version')}`",
        f"- Vintage macro context: `{_status_label(manifest.get('vintage_context_status'))}`",
        f"- Survey belief targets: `{_status_label(manifest.get('survey_belief_status'))}`"
        f" ({manifest.get('survey_belief_target_rows', 0)} rows)",
        f"- Forecast-first typed panel rows: `{manifest.get('typed_agent_panel_rows', 0)}`",
        f"- Regime counts: `{json.dumps(manifest.get('card_regime_counts', {}), sort_keys=True)}`",
        f"- Contamination labels: `{json.dumps(manifest.get('card_contamination_counts', {}), sort_keys=True)}`",
        "",
        "## Full-Coverage Leaderboard",
        _markdown_table(
            comparable_rows[
                [
                    "source",
                    "n",
                    "rmse",
                    "mae",
                    "bias",
                    "direction_accuracy",
                    "cache_hit_rate",
                ]
            ]
        ),
        "",
        *_partial_coverage_section(partial_rows),
        "",
        "## Variable Breakdown",
        _variable_breakdown(scores, llm_source=llm_source),
        "",
        *_slice_section(slice_scores),
        "",
        "## Behavioral Coefficients",
        _markdown_table(
            behavior[
                [
                    "source",
                    "n",
                    "underreaction_slope_error_on_revision",
                    "extrapolation_slope_forecast_change_on_recent_signal",
                    "mean_panel_std",
                    "mean_interval_width_p90_p10",
                ]
            ].head(12)
        ),
        "",
        "## Interpretation",
        (
            "This generated report summarizes one forecast-tournament run. Compare full-coverage rows "
            "for the primary leaderboard, then inspect variable and regime slices before interpreting "
            "any model-level result."
        ),
        "",
        "## Caveats",
        "- This first run is a historical/confounded screen; the selected holdout is pre-model-cutoff.",
        "- Forecast cards use official SPF forecast-error files for aligned forecasts and realizations.",
        "- The prompt excludes each card's realized target and same-card SPF consensus forecast.",
        *_caveat_lines(manifest),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _slice_section(slice_scores: pd.DataFrame | None) -> list[str]:
    if slice_scores is None or slice_scores.empty:
        return []
    llm = slice_scores[
        (slice_scores["slice"] == "regime") & (slice_scores["source"].astype(str).str.startswith("llm_"))
    ].sort_values("regime_label")
    if llm.empty:
        return []
    return [
        "## LLM Regime Slices",
        _markdown_table(llm[["regime_label", "n", "rmse", "mae", "bias", "direction_accuracy"]]),
        "",
    ]


def _partial_coverage_section(partial_rows: pd.DataFrame) -> list[str]:
    if partial_rows.empty:
        return []
    return [
        "## Partial-Coverage Benchmarks",
        (
            "These rows do not cover the same card set as the LLM and are shown for diagnostics, "
            "not for the main rank."
        ),
        _markdown_table(partial_rows[["source", "n", "rmse", "mae", "bias", "direction_accuracy"]]),
        "",
    ]


def _variable_breakdown(scores: pd.DataFrame, *, llm_source: str) -> str:
    if not llm_source:
        return "_No LLM row available._"
    rows: list[dict[str, Any]] = []
    variables = sorted(value for value in scores["variable"].dropna().unique() if value != "ALL")
    for variable in variables:
        var_rows = scores[scores["variable"] == variable]
        llm = var_rows[var_rows["source"] == llm_source]
        spf = var_rows[var_rows["source"] == "spf_consensus"]
        if llm.empty:
            continue
        llm_row = llm.iloc[0]
        spf_row = spf.iloc[0] if not spf.empty else None
        rows.append(
            {
                "variable": variable,
                "llm_rmse": float(llm_row["rmse"]),
                "spf_consensus_rmse": float(spf_row["rmse"]) if spf_row is not None else None,
                "llm_n": int(llm_row["n"]),
                "read": _variable_read(float(llm_row["rmse"]), float(spf_row["rmse"])) if spf_row is not None else "",
            }
        )
    return _markdown_table(pd.DataFrame(rows))


def _variable_read(llm_rmse: float, spf_rmse: float) -> str:
    if llm_rmse < spf_rmse * 0.98:
        return "LLM better"
    if spf_rmse < llm_rmse * 0.98:
        return "SPF better"
    return "rough tie"


def _verdict_sentence(verdict: dict[str, Any]) -> str:
    if verdict.get("status") != "ok":
        return f"Verdict unavailable: `{verdict.get('status')}`."
    llm = verdict.get("llm_source")
    if verdict.get("beats_all_primary_behavioral_controls") and verdict.get("beats_spf_consensus"):
        return (
            f"`{llm}` beats both primary behavioral controls and SPF consensus on overall RMSE "
            f"(`{verdict.get('llm_rmse'):.4f}`), ranking `{verdict.get('llm_rank_by_rmse')}`."
        )
    if verdict.get("beats_all_primary_behavioral_controls"):
        better = ", ".join(verdict.get("sources_with_lower_rmse_than_llm", [])[:5])
        return (
            f"`{llm}` passes the narrow primary behavioral gate, beating constant-gain and recursive "
            f"least squares on overall RMSE (`{verdict.get('llm_rmse'):.4f}`), but it does not win "
            f"the tournament. Lower-RMSE sources include: `{better}`."
        )
    controls = verdict.get("primary_control_results", [])
    failed = [row["control"] for row in controls if not row.get("llm_beats")]
    return (
        f"`{llm}` does not clear the primary behavioral-control gate. "
        f"It fails to beat: `{', '.join(failed)}`. "
        f"Best overall source is `{verdict.get('best_source')}` with RMSE `{verdict.get('best_rmse'):.4f}`."
    )


def _status_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "unknown"))
    return str(value or "unknown")


def _caveat_lines(manifest: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    vintage = manifest.get("vintage_context_status") or {}
    if isinstance(vintage, dict) and vintage.get("status") == "ok":
        lines.append("- Vintage macro context is real-time FRED/ALFRED data as of the card date.")
    elif manifest.get("vintage_context_mode") == "off":
        lines.append("- Vintage macro context was disabled for this run.")
    else:
        lines.append("- Real-time FRED/ALFRED vintage context was unavailable; rerun with `FRED_API_KEY` for the hard gate.")
    survey = manifest.get("survey_belief_status") or {}
    if isinstance(survey, dict) and survey.get("status") in {"ok", "partial"}:
        lines.append("- Survey beliefs are included as as-of household-belief context and diagnostics, not as SPF outcome targets.")
    elif manifest.get("belief_targets_mode") == "off":
        lines.append("- Survey-belief targets were disabled for this run.")
    else:
        lines.append("- Survey-belief targets were unavailable for this run.")
    return lines


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    clean = clean.fillna("").astype(str)
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)
