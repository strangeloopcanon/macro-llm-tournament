from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRIMARY_BEHAVIORAL_CONTROLS = ["constant_gain", "recursive_least_squares"]


def score_forecasts(cards: pd.DataFrame, forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = forecasts.merge(
        cards[
            [
                "card_id",
                "variable",
                "origin",
                "origin_index",
                "horizon",
                "target_realized",
                "asof_reference_value",
                "prior_spf_forecast",
                "rolling_signal_mean_4",
                "recent_signal_change_4",
                "contamination_label",
            ]
        ],
        on=["card_id", "variable", "origin", "origin_index", "horizon"],
        how="inner",
    )
    joined["error"] = joined["point_forecast"] - joined["target_realized"]
    joined["abs_error"] = joined["error"].abs()
    joined["squared_error"] = joined["error"] ** 2
    joined["target_change"] = joined["target_realized"] - joined["asof_reference_value"]
    joined["forecast_change"] = joined["point_forecast"] - joined["asof_reference_value"]
    joined["direction_correct"] = np.sign(joined["target_change"]) == np.sign(joined["forecast_change"])

    scores = []
    for keys, group in joined.groupby(["source", "provider", "model"], dropna=False):
        source, provider, model = keys
        scores.append(_score_row(group, source=source, provider=provider, model=model, variable="ALL"))
        for variable, var_group in group.groupby("variable"):
            scores.append(_score_row(var_group, source=source, provider=provider, model=model, variable=variable))
    score_df = pd.DataFrame(scores).sort_values(["variable", "rmse", "mae", "source"]).reset_index(drop=True)
    behavior = behavioral_coefficients(joined)
    return score_df, behavior, joined


def score_forecast_slices(cards: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        col
        for col in ["regime_label", "evaluation_split", "contamination_label"]
        if col in cards.columns and col not in joined.columns
    ]
    enriched = joined.merge(cards[["card_id", *label_cols]].copy(), on="card_id", how="left") if label_cols else joined.copy()
    rows: list[dict[str, Any]] = []
    for slice_name, group_cols in [
        ("regime", ["regime_label"]),
        ("evaluation_split", ["evaluation_split"]),
        ("contamination", ["contamination_label"]),
        ("variable_regime", ["variable", "regime_label"]),
    ]:
        for keys, group in enriched.groupby(["source", "provider", "model", *group_cols], dropna=False):
            source, provider, model, *values = keys
            row_variable = str(values[group_cols.index("variable")]) if "variable" in group_cols else "ALL"
            row = _score_row(group, source=source, provider=provider, model=model, variable=row_variable)
            row["slice"] = slice_name
            for col, value in zip(group_cols, values):
                row[col] = value
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["slice", "regime_label", "variable", "rmse", "source"], na_position="last")


def _score_row(group: pd.DataFrame, *, source: str, provider: str, model: str, variable: str) -> dict[str, Any]:
    errors = group["error"].astype(float)
    return {
        "source": source,
        "provider": provider,
        "model": model,
        "variable": variable,
        "n": int(group.shape[0]),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "bias": float(np.mean(errors)) if len(errors) else np.nan,
        "direction_accuracy": float(group["direction_correct"].mean()) if len(group) else np.nan,
        "mean_abs_target_change": float(np.mean(np.abs(group["target_change"]))) if len(group) else np.nan,
        "mean_panel_std": float(group["panel_std"].dropna().mean()) if group["panel_std"].notna().any() else np.nan,
        "cache_hit_rate": float(group["cache_hit"].mean()) if "cache_hit" in group else np.nan,
    }


def behavioral_coefficients(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, group in joined.groupby("source"):
        revision = group["point_forecast"].astype(float) - group["prior_spf_forecast"].astype(float)
        error = group["target_realized"].astype(float) - group["point_forecast"].astype(float)
        recent_signal = group["asof_reference_value"].astype(float) - group["rolling_signal_mean_4"].astype(float)
        forecast_change = group["point_forecast"].astype(float) - group["asof_reference_value"].astype(float)
        rows.append(
            {
                "source": source,
                "n": int(group.shape[0]),
                "underreaction_slope_error_on_revision": _slope(revision, error),
                "extrapolation_slope_forecast_change_on_recent_signal": _slope(recent_signal, forecast_change),
                "mean_forecast_revision": float(np.mean(revision)) if len(revision) else np.nan,
                "mean_forecast_error_realized_minus_forecast": float(np.mean(error)) if len(error) else np.nan,
                "mean_panel_std": float(group["panel_std"].dropna().mean()) if group["panel_std"].notna().any() else np.nan,
                "mean_interval_width_p90_p10": float((group["p90"] - group["p10"]).dropna().mean())
                if group["p90"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def _slope(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 3 or np.nanstd(x_values[mask]) < 1e-12:
        return float("nan")
    x_centered = x_values[mask] - np.mean(x_values[mask])
    y_centered = y_values[mask] - np.mean(y_values[mask])
    return float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))


def verdict_from_scores(scores: pd.DataFrame, *, llm_source_prefix: str = "llm_") -> dict[str, Any]:
    overall = scores[scores["variable"] == "ALL"].copy()
    if overall.empty:
        return {"status": "no_scores"}
    llm_rows = overall[overall["source"].str.startswith(llm_source_prefix)].sort_values("rmse")
    if llm_rows.empty:
        return {"status": "no_llm_rows"}
    llm = llm_rows.iloc[0]
    llm_n = int(llm["n"])
    control_rows = overall[overall["source"].isin(PRIMARY_BEHAVIORAL_CONTROLS)].copy()
    beat_controls = []
    for _, row in control_rows.iterrows():
        beat_controls.append(
            {
                "control": row["source"],
                "control_rmse": float(row["rmse"]),
                "llm_rmse": float(llm["rmse"]),
                "llm_beats": bool(float(llm["rmse"]) < float(row["rmse"])),
            }
        )
    ranked = overall[overall["n"] >= llm_n].sort_values("rmse").reset_index(drop=True)
    llm_rank = int(ranked.index[ranked["source"] == llm["source"]][0] + 1)
    spf = overall[overall["source"] == "spf_consensus"]
    beats_spf = bool(not spf.empty and float(llm["rmse"]) < float(spf.iloc[0]["rmse"]))
    better_sources = ranked[ranked["rmse"] < float(llm["rmse"])]["source"].astype(str).tolist()
    return {
        "status": "ok",
        "llm_source": llm["source"],
        "llm_rmse": float(llm["rmse"]),
        "llm_mae": float(llm["mae"]),
        "llm_n": llm_n,
        "llm_rank_by_rmse": llm_rank,
        "primary_control_results": beat_controls,
        "beats_all_primary_behavioral_controls": bool(beat_controls and all(row["llm_beats"] for row in beat_controls)),
        "beats_spf_consensus": beats_spf,
        "sources_with_lower_rmse_than_llm": better_sources,
        "full_coverage_ranked": True,
        "best_source": str(ranked.iloc[0]["source"]),
        "best_rmse": float(ranked.iloc[0]["rmse"]),
    }
