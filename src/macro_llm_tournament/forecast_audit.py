from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import PROJECT_ROOT, WORK_ROOT, cache_key, extract_json, markdown_table
from .llm_common import LLMUnavailable


AUDIT_VERSION = "forecast_tournament_audit_v1"
DIRECT_RECALL_VERSION = "direct_realization_recall_probe_v1"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "current" / "historical_48card_gpt55" / "live"
DEFAULT_CACHE_DIR = WORK_ROOT / "direct_recall_cache"


@dataclass(frozen=True)
class DirectRecallClient:
    provider: str
    model: str
    cache_dir: Path
    mode: str
    max_live_calls: int
    live_call_count: int = 0
    cache_hit_count: int = 0

    def with_counts(
        self,
        *,
        live_call_count: int | None = None,
        cache_hit_count: int | None = None,
    ) -> "DirectRecallClient":
        return DirectRecallClient(
            self.provider,
            self.model,
            self.cache_dir,
            self.mode,
            self.max_live_calls,
            self.live_call_count if live_call_count is None else live_call_count,
            self.cache_hit_count if cache_hit_count is None else cache_hit_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an existing SPF forecast tournament run.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--recall-mode", choices=["off", "fixture", "replay", "live"], default="off")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--recall-batch-size", type=int, default=48)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recall_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --recall-mode live is used")
    run_dir = Path(args.run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / f"audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fresh_direct_recall_cache" if args.fresh_cache else DEFAULT_CACHE_DIR
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_VERSION,
        "timestamp_utc": timestamp,
        "run_dir": str(run_dir),
        "provider": args.provider,
        "model": args.model,
        "recall_mode": args.recall_mode,
        "max_live_calls": int(args.max_live_calls),
        "recall_batch_size": int(args.recall_batch_size),
        "fresh_cache": bool(args.fresh_cache),
        "status": "running",
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        cards = load_cards(run_dir)
        forecasts = load_forecasts(run_dir)
        joined = load_joined(run_dir, cards, forecasts)
        surprise = build_surprise_audit(joined, cards)
        theil = build_theil_u(joined)
        source_scores = build_source_scores(joined)
        paired = build_paired_loss_tests(joined)
        belief_detail, belief_summary = build_belief_structure_audit(joined)
        recall_predictions = pd.DataFrame()
        recall_scores = pd.DataFrame()
        recall_raw: list[dict[str, Any]] = []
        recall_client = DirectRecallClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.recall_mode,
            max_live_calls=args.max_live_calls,
        )
        if args.recall_mode != "off":
            recall_predictions, recall_raw, recall_client = run_direct_recall(
                cards,
                recall_client,
                batch_size=args.recall_batch_size,
            )
            recall_scores = score_direct_recall(cards, recall_predictions)
        source_scores.to_csv(output_dir / "audit_source_scores.csv", index=False)
        surprise.to_csv(output_dir / "audit_surprise_split.csv", index=False)
        theil.to_csv(output_dir / "audit_theils_u.csv", index=False)
        paired.to_csv(output_dir / "audit_paired_loss_tests.csv", index=False)
        belief_detail.to_csv(output_dir / "audit_belief_structure.csv", index=False)
        belief_summary.to_csv(output_dir / "audit_belief_structure_summary.csv", index=False)
        recall_predictions.to_csv(output_dir / "direct_recall_predictions.csv", index=False)
        recall_scores.to_csv(output_dir / "direct_recall_scores.csv", index=False)
        (output_dir / "direct_recall_raw_records.json").write_text(json.dumps(recall_raw, indent=2, sort_keys=True), encoding="utf-8")
        manifest.update(
            {
                "status": "ok",
                "card_count": int(cards.shape[0]),
                "forecast_rows": int(forecasts.shape[0]),
                "joined_rows": int(joined.shape[0]),
                "source_count": int(joined["source"].nunique()),
                "surprise_rows": int(surprise.shape[0]),
                "theil_rows": int(theil.shape[0]),
                "paired_test_rows": int(paired.shape[0]),
                "belief_structure_rows": int(belief_detail.shape[0]),
                "belief_structure_summary_rows": int(belief_summary.shape[0]),
                "recall_prediction_rows": int(recall_predictions.shape[0]),
                "recall_score_rows": int(recall_scores.shape[0]),
                "recall_live_call_count": int(recall_client.live_call_count),
                "recall_cache_hit_count": int(recall_client.cache_hit_count),
                "outputs": [
                    "audit_source_scores.csv",
                    "audit_surprise_split.csv",
                    "audit_theils_u.csv",
                    "audit_paired_loss_tests.csv",
                    "audit_belief_structure.csv",
                    "audit_belief_structure_summary.csv",
                    "direct_recall_predictions.csv",
                    "direct_recall_scores.csv",
                    "direct_recall_raw_records.json",
                    "forecast_audit_report.md",
                ],
            }
        )
        report = build_audit_report(manifest, source_scores, surprise, theil, paired, recall_scores, belief_summary=belief_summary)
        (output_dir / "forecast_audit_report.md").write_text(report, encoding="utf-8")
        (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


def load_cards(run_dir: Path) -> pd.DataFrame:
    candidates = [run_dir / "forecast_cards.csv", run_dir / "postcutoff_forecast_cards.csv"]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if "card_id" not in frame:
                raise ValueError(f"{path} is missing card_id")
            return frame
    raise FileNotFoundError(f"No forecast card file found in {run_dir}")


def load_forecasts(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "all_forecasts.csv"
    if path.exists():
        return pd.read_csv(path)
    control = pd.read_csv(run_dir / "control_forecasts.csv")
    llm = pd.read_json(run_dir / "llm_forecasts.jsonl", orient="records", lines=True)
    return pd.concat([control, llm], ignore_index=True)


def load_joined(run_dir: Path, cards: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    path = run_dir / "forecast_joined_errors.csv"
    if path.exists():
        joined = pd.read_csv(path)
    else:
        card_columns = [
            "card_id",
            "variable",
            "origin",
            "origin_index",
            "horizon",
            "target_realized",
            "asof_reference_value",
        ]
        for optional in ["prior_spf_forecast", "rolling_signal_mean_4", "recent_signal_change_4"]:
            if optional in cards:
                card_columns.append(optional)
        joined = forecasts.merge(
            cards[card_columns],
            on=["card_id", "variable", "origin", "origin_index", "horizon"],
            how="inner",
        )
        joined["error"] = joined["point_forecast"].astype(float) - joined["target_realized"].astype(float)
        joined["target_change"] = joined["target_realized"].astype(float) - joined["asof_reference_value"].astype(float)
        joined["forecast_change"] = joined["point_forecast"].astype(float) - joined["asof_reference_value"].astype(float)
        joined["direction_correct"] = np.sign(joined["target_change"]) == np.sign(joined["forecast_change"])
    for column in ["regime_label", "evaluation_split", "contamination_label"]:
        if column in cards and column not in joined:
            joined = joined.merge(cards[["card_id", column]], on="card_id", how="left")
    joined["squared_error"] = joined["error"].astype(float) ** 2
    joined["abs_error"] = joined["error"].abs()
    return joined


def build_source_scores(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "variable"], dropna=False):
        source, variable = keys
        rows.append(score_group(group, source=str(source), variable=str(variable)))
    for source, group in joined.groupby("source", dropna=False):
        rows.append(score_group(group, source=str(source), variable="ALL"))
    return pd.DataFrame(rows).sort_values(["variable", "rmse", "source"]).reset_index(drop=True)


def build_surprise_audit(joined: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    spf = joined[joined["source"] == "spf_consensus"][["card_id", "error"]].copy()
    if spf.empty:
        return pd.DataFrame()
    spf["spf_abs_error"] = spf["error"].abs()
    median_error = float(spf["spf_abs_error"].median())
    card_columns = ["card_id", "variable", "origin"]
    if "regime_label" in cards.columns:
        card_columns.append("regime_label")
    bucketed_cards = cards[card_columns].merge(spf[["card_id", "spf_abs_error"]], on="card_id", how="inner")
    bucketed_cards["surprise_bucket"] = np.where(bucketed_cards["spf_abs_error"] > median_error, "surprising_spf_error_gt_median", "calm_spf_error_le_median")
    if "regime_label" not in bucketed_cards.columns:
        bucketed_cards["regime_label"] = "unknown"
    bucketed_cards["event_bucket"] = np.where(
        bucketed_cards["regime_label"].isin(["covid_shock", "inflation_surge"]),
        "covid_or_inflation_surge",
        "other_regimes",
    )
    enriched = joined.merge(bucketed_cards[["card_id", "surprise_bucket", "event_bucket", "spf_abs_error"]], on="card_id", how="inner")
    rows: list[dict[str, Any]] = []
    for bucket_col in ["surprise_bucket", "event_bucket"]:
        for bucket_value, group in enriched.groupby(bucket_col, dropna=False):
            rows.extend(score_bucket(group, bucket_col=bucket_col, bucket_value=str(bucket_value), variable="ALL"))
            for variable, variable_group in group.groupby("variable", dropna=False):
                rows.extend(score_bucket(variable_group, bucket_col=bucket_col, bucket_value=str(bucket_value), variable=str(variable)))
    return pd.DataFrame(rows).sort_values(["bucket_type", "bucket", "variable", "source"]).reset_index(drop=True)


def score_bucket(group: pd.DataFrame, *, bucket_col: str, bucket_value: str, variable: str) -> list[dict[str, Any]]:
    rows = []
    base = {
        "bucket_type": bucket_col,
        "bucket": bucket_value,
        "variable": variable,
    }
    for source, source_group in group.groupby("source", dropna=False):
        row = score_group(source_group, source=str(source), variable=variable)
        row.update(base)
        rows.append(row)
    llm = next((row for row in rows if row["source"].startswith("llm_")), None)
    spf = next((row for row in rows if row["source"] == "spf_consensus"), None)
    if llm and spf:
        rows.append(
            {
                **base,
                "source": "llm_minus_spf_gap",
                "n": min(llm["n"], spf["n"]),
                "rmse": llm["rmse"] - spf["rmse"],
                "mae": llm["mae"] - spf["mae"],
                "bias": np.nan,
                "direction_accuracy": np.nan,
                "rmse_ci_low": np.nan,
                "rmse_ci_high": np.nan,
            }
        )
    return rows


def build_theil_u(joined: pd.DataFrame) -> pd.DataFrame:
    score = build_source_scores(joined)
    rows: list[dict[str, Any]] = []
    for variable, group in score.groupby("variable", dropna=False):
        baseline = group[group["source"] == "no_change"]
        baseline_rmse = float(baseline.iloc[0]["rmse"]) if not baseline.empty else np.nan
        for _, row in group.iterrows():
            theil = float(row["rmse"]) / baseline_rmse if np.isfinite(baseline_rmse) and baseline_rmse > 0 else np.nan
            rows.append(
                {
                    "source": row["source"],
                    "variable": variable,
                    "n": int(row["n"]),
                    "rmse": float(row["rmse"]),
                    "no_change_rmse": baseline_rmse,
                    "theils_u_vs_no_change": theil,
                    "sane_vs_no_change": bool(np.isfinite(theil) and theil <= 1.5),
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "theils_u_vs_no_change", "source"], na_position="last").reset_index(drop=True)


def build_paired_loss_tests(joined: pd.DataFrame, *, baseline: str = "spf_consensus") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = joined[joined["source"] == baseline][["card_id", "variable", "squared_error", "abs_error"]].rename(
        columns={"squared_error": "baseline_squared_error", "abs_error": "baseline_abs_error"}
    )
    if base.empty:
        return pd.DataFrame()
    for keys, group in joined.groupby(["source", "variable"], dropna=False):
        source, variable = keys
        rows.append(paired_loss_row(group, base, source=str(source), variable=str(variable), baseline=baseline))
    for source, group in joined.groupby("source", dropna=False):
        rows.append(paired_loss_row(group, base, source=str(source), variable="ALL", baseline=baseline))
    return pd.DataFrame(rows).sort_values(["variable", "mean_squared_loss_diff_vs_baseline", "source"]).reset_index(drop=True)


def paired_loss_row(group: pd.DataFrame, base: pd.DataFrame, *, source: str, variable: str, baseline: str) -> dict[str, Any]:
    if variable != "ALL":
        base = base[base["variable"] == variable]
    merged = group.merge(base[["card_id", "baseline_squared_error", "baseline_abs_error"]], on="card_id", how="inner")
    diff = merged["squared_error"].astype(float) - merged["baseline_squared_error"].astype(float)
    abs_diff = merged["abs_error"].astype(float) - merged["baseline_abs_error"].astype(float)
    return {
        "source": source,
        "baseline": baseline,
        "variable": variable,
        "n": int(merged.shape[0]),
        "mean_squared_loss_diff_vs_baseline": float(diff.mean()) if len(diff) else np.nan,
        "mean_abs_loss_diff_vs_baseline": float(abs_diff.mean()) if len(abs_diff) else np.nan,
        "dm_t_approx_squared_loss": t_stat(diff),
        "normal_approx_p_two_sided": normal_p_value(t_stat(diff)),
    }


def build_belief_structure_audit(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    enriched = joined.copy()
    spf_abs = (
        enriched[enriched["source"] == "spf_consensus"][["card_id", "error"]]
        .rename(columns={"error": "spf_error"})
        .copy()
    )
    if not spf_abs.empty:
        spf_abs["spf_abs_error"] = spf_abs["spf_error"].abs()
        enriched = enriched.merge(spf_abs[["card_id", "spf_abs_error"]], on="card_id", how="left")
    elif "spf_abs_error" not in enriched:
        enriched["spf_abs_error"] = np.nan

    summary_rows: list[dict[str, Any]] = []
    for keys, group in enriched.groupby(["source", "variable"], dropna=False):
        source, variable = keys
        summary_rows.append(belief_structure_summary_row(group, source=str(source), variable=str(variable)))
    for source, group in enriched.groupby("source", dropna=False):
        summary_rows.append(belief_structure_summary_row(group, source=str(source), variable="ALL"))
    summary = pd.DataFrame(summary_rows).sort_values(["variable", "source"]).reset_index(drop=True)
    detail_rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        base = {"source": row["source"], "variable": row["variable"], "n": int(row["n"])}
        for metric_name in [column for column in summary.columns if column not in {"source", "variable", "n"}]:
            detail_rows.append(
                {
                    **base,
                    "metric_family": belief_metric_family(metric_name),
                    "metric_name": metric_name,
                    "value": row[metric_name],
                }
            )
    detail = pd.DataFrame(detail_rows).sort_values(["variable", "source", "metric_family", "metric_name"]).reset_index(drop=True)
    return detail, summary


def belief_structure_summary_row(group: pd.DataFrame, *, source: str, variable: str) -> dict[str, Any]:
    point = numeric_column(group, "point_forecast")
    target = numeric_column(group, "target_realized")
    asof = numeric_column(group, "asof_reference_value")
    prior_spf = numeric_column(group, "prior_spf_forecast")
    rolling_mean = numeric_column(group, "rolling_signal_mean_4")
    recent_change = numeric_column(group, "recent_signal_change_4")
    panel_std = numeric_column(group, "panel_std")
    p10 = numeric_column(group, "p10")
    p90 = numeric_column(group, "p90")
    confidence = numeric_column(group, "confidence")
    spf_abs_error = numeric_column(group, "spf_abs_error")
    error = point - target
    abs_error = error.abs()
    revision = point - prior_spf
    realized_minus_forecast = target - point
    forecast_change = point - asof
    recent_signal = asof - rolling_mean
    interval_width = p90 - p10
    interval_mask = finite_mask(target, p10, p90) & (p10 <= p90)
    interval_coverage = (
        ((target[interval_mask] >= p10[interval_mask]) & (target[interval_mask] <= p90[interval_mask])).mean()
        if interval_mask.any()
        else np.nan
    )
    confidence_mask = finite_mask(confidence, abs_error)
    confidence_median = float(confidence[confidence_mask].median()) if confidence_mask.any() else np.nan
    high_conf_mae = float(abs_error[confidence_mask & (confidence >= confidence_median)].mean()) if np.isfinite(confidence_median) else np.nan
    low_conf_mae = float(abs_error[confidence_mask & (confidence < confidence_median)].mean()) if np.isfinite(confidence_median) else np.nan
    surprise_mask = finite_mask(spf_abs_error, error)
    surprise_median = float(spf_abs_error[surprise_mask].median()) if surprise_mask.any() else np.nan
    high_surprise = error[surprise_mask & (spf_abs_error > surprise_median)]
    low_surprise = error[surprise_mask & (spf_abs_error <= surprise_median)]
    high_surprise_rmse = rmse(high_surprise)
    low_surprise_rmse = rmse(low_surprise)
    return {
        "source": source,
        "variable": variable,
        "n": int(group.shape[0]),
        "underreaction_slope_error_on_revision": slope(revision, realized_minus_forecast),
        "mean_forecast_revision": finite_mean(revision),
        "mean_realized_minus_forecast": finite_mean(realized_minus_forecast),
        "extrapolation_slope_forecast_change_on_recent_signal": slope(recent_signal, forecast_change),
        "extrapolation_slope_forecast_change_on_recent_spf_change": slope(recent_change, forecast_change),
        "mean_panel_std": finite_mean(panel_std),
        "dispersion_slope_abs_error_on_panel_std": slope(panel_std, abs_error),
        "mean_interval_width_p90_p10": finite_mean(interval_width[interval_width >= 0]),
        "interval_coverage_p10_p90": float(interval_coverage) if np.isfinite(interval_coverage) else np.nan,
        "interval_n": int(interval_mask.sum()),
        "median_abs_error": finite_median(abs_error),
        "mean_confidence": finite_mean(confidence),
        "confidence_abs_error_corr": correlation(confidence, abs_error),
        "confidence_slope_abs_error_on_confidence": slope(confidence, abs_error),
        "high_confidence_mae": high_conf_mae,
        "low_confidence_mae": low_conf_mae,
        "high_minus_low_confidence_mae": high_conf_mae - low_conf_mae if np.isfinite(high_conf_mae) and np.isfinite(low_conf_mae) else np.nan,
        "surprise_slope_abs_error_on_spf_abs_error": slope(spf_abs_error, abs_error),
        "surprise_rmse_gap_high_minus_low_spf_error": high_surprise_rmse - low_surprise_rmse
        if np.isfinite(high_surprise_rmse) and np.isfinite(low_surprise_rmse)
        else np.nan,
    }


def belief_metric_family(metric_name: str) -> str:
    if metric_name.startswith("underreaction") or metric_name.startswith("mean_forecast_revision") or metric_name.startswith("mean_realized"):
        return "underreaction"
    if metric_name.startswith("extrapolation"):
        return "extrapolation"
    if "panel_std" in metric_name or metric_name.startswith("dispersion"):
        return "disagreement_dispersion"
    if "interval" in metric_name:
        return "interval_calibration"
    if "confidence" in metric_name or metric_name == "median_abs_error":
        return "confidence_calibration"
    if metric_name.startswith("surprise"):
        return "surprise_response"
    return "other"


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def finite_mask(*values: pd.Series) -> pd.Series:
    if not values:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=values[0].index)
    for value in values:
        mask &= np.isfinite(pd.to_numeric(value, errors="coerce"))
    return mask


def finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if len(finite) else np.nan


def finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.median()) if len(finite) else np.nan


def rmse(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else np.nan


def slope(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 3 or np.nanstd(x_values[mask]) < 1e-12:
        return np.nan
    centered_x = x_values[mask] - np.mean(x_values[mask])
    centered_y = y_values[mask] - np.mean(y_values[mask])
    return float(np.sum(centered_x * centered_y) / np.sum(centered_x**2))


def correlation(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 3 or np.nanstd(x_values[mask]) < 1e-12 or np.nanstd(y_values[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x_values[mask], y_values[mask])[0, 1])


def score_group(group: pd.DataFrame, *, source: str, variable: str) -> dict[str, Any]:
    errors = group["error"].astype(float)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan
    low, high = rmse_ci(errors)
    return {
        "source": source,
        "variable": variable,
        "n": int(group.shape[0]),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "bias": float(np.mean(errors)) if len(errors) else np.nan,
        "direction_accuracy": float(group["direction_correct"].mean()) if "direction_correct" in group else np.nan,
        "rmse_ci_low": low,
        "rmse_ci_high": high,
    }


def rmse_ci(errors: pd.Series, *, draws: int = 800) -> tuple[float, float]:
    values = pd.to_numeric(errors, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        rmse = float(np.sqrt(np.mean(np.square(values)))) if values.size else np.nan
        return rmse, rmse
    rng = np.random.default_rng(17)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    rmses = np.sqrt(np.mean(np.square(values[idx]), axis=1))
    return float(np.quantile(rmses, 0.025)), float(np.quantile(rmses, 0.975))


def t_stat(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size < 3:
        return float("nan")
    se = float(np.std(finite, ddof=1) / math.sqrt(finite.size))
    if se <= 1e-12:
        return float("nan")
    return float(np.mean(finite) / se)


def normal_p_value(t_value: float) -> float:
    if not np.isfinite(t_value):
        return float("nan")
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def run_direct_recall(
    cards: pd.DataFrame,
    client: DirectRecallClient,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], DirectRecallClient]:
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for start in range(0, cards.shape[0], batch_size):
        batch = cards.iloc[start : start + batch_size].copy()
        if client.mode == "fixture":
            payload = fixture_recall_payload(batch)
            data = {
                "provider": client.provider,
                "model": client.model,
                "payload": payload,
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = direct_recall_prompt(batch)
            data, client = direct_recall_call(client, prompt, direct_recall_cache_name(client, prompt))
        normalized = normalize_direct_recall_payload(batch, data)
        rows.extend(normalized)
        raw_records.append(
            {
                "batch_start": start,
                "batch_size": int(batch.shape[0]),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
    return pd.DataFrame(rows), raw_records, client


def direct_recall_prompt(cards: pd.DataFrame) -> str:
    items = [
        {
            "card_id": row["card_id"],
            "variable": row["variable"],
            "variable_name": row.get("variable_name", row["variable"]),
            "origin": row["origin"],
            "horizon": int(row["horizon"]),
            "units": row.get("units", ""),
        }
        for _, row in cards.iterrows()
    ]
    payload = {
        "prompt_version": DIRECT_RECALL_VERSION,
        "task": (
            "Without using tools, browsing, files, or calculations from hidden data, state whether you recall "
            "the realized value for each listed SPF-style macro target. If you do not know the value from memory, "
            "return null."
        ),
        "required_response": {
            "items": [
                {
                    "card_id": "matching id",
                    "recalled_realized": "number or null",
                    "confidence": "0 to 1",
                    "reason": "short memory-status reason",
                }
            ]
        },
        "items": items,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def direct_recall_cache_name(client: DirectRecallClient, prompt: str) -> str:
    return f"direct_recall_{cache_key({'provider': client.provider, 'model': client.model, 'prompt': prompt})}"


def direct_recall_call(client: DirectRecallClient, prompt: str, cache_name: str) -> tuple[dict[str, Any], DirectRecallClient]:
    cache_dir = client.cache_dir / client.provider
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_name}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cache_hit"] = True
        data["cache_path"] = str(path)
        return data, client.with_counts(cache_hit_count=client.cache_hit_count + 1)
    if client.mode == "replay":
        raise LLMUnavailable(f"Direct recall replay cache miss for {cache_name}")
    if client.mode != "live":
        raise LLMUnavailable(f"Direct recall mode {client.mode} cannot make live calls")
    if client.live_call_count >= client.max_live_calls:
        raise LLMUnavailable(f"Direct recall live-call cap reached ({client.max_live_calls}); cache miss for {cache_name}")
    binary = os.getenv("CODEX_CLI_BIN") or shutil.which("codex")
    if not binary:
        raise LLMUnavailable("codex CLI binary not found; set CODEX_CLI_BIN or install codex")
    last_message_path = path.with_name(f"{path.stem}.{os.getpid()}.last_message.txt")
    command = [
        binary,
        "exec",
        "--model",
        client.model,
        "--cd",
        str(PROJECT_ROOT),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    system_prompt = f"""
Return only valid JSON. This is a direct recall contamination probe.
Do not browse, inspect files, run commands, or infer using hidden outcome data. If you do not
know a realized value from memory, return null for that item.

{prompt.strip()}
""".strip()
    try:
        result = subprocess.run(
            command,
            input=system_prompt,
            text=True,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            timeout=float(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "240")),
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise LLMUnavailable(f"Codex CLI failed with exit code {result.returncode}: {stderr[:700]}")
        text = last_message_path.read_text(encoding="utf-8") if last_message_path.exists() else result.stdout
        payload = extract_json(text or result.stdout or "")
        data = {
            "provider": client.provider,
            "model": client.model,
            "payload": payload,
            "cache_hit": False,
            "cache_path": str(path),
            "response_created_utc": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data, client.with_counts(live_call_count=client.live_call_count + 1)
    except subprocess.TimeoutExpired as exc:
        raise LLMUnavailable(f"Codex CLI timed out after {exc.timeout} seconds") from exc
    finally:
        if last_message_path.exists():
            last_message_path.unlink()


def normalize_direct_recall_payload(cards: pd.DataFrame, data: dict[str, Any]) -> list[dict[str, Any]]:
    payload = data.get("payload", data)
    items = payload.get("items")
    if not isinstance(items, list):
        raise LLMUnavailable("Direct recall payload is missing items list")
    expected = set(cards["card_id"].astype(str))
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id", ""))
        if card_id not in expected:
            continue
        value = item.get("recalled_realized")
        try:
            recalled = float(value) if value is not None else np.nan
        except (TypeError, ValueError):
            recalled = np.nan
        confidence = item.get("confidence", np.nan)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = np.nan
        by_id[card_id] = {
            "card_id": card_id,
            "recalled_realized": recalled,
            "confidence": confidence_value,
            "reason": str(item.get("reason", ""))[:300],
            "cache_hit": bool(data.get("cache_hit", False)),
            "cache_path": data.get("cache_path"),
        }
    missing = sorted(expected - set(by_id))
    if missing:
        raise LLMUnavailable(f"Direct recall payload missing card ids: {', '.join(missing[:8])}")
    return [by_id[str(card_id)] for card_id in cards["card_id"].astype(str)]


def fixture_recall_payload(cards: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": DIRECT_RECALL_VERSION,
        "items": [
            {
                "card_id": row["card_id"],
                "recalled_realized": None,
                "confidence": 0.0,
                "reason": "fixture mode returns no recall",
            }
            for _, row in cards.iterrows()
        ],
    }


def score_direct_recall(cards: pd.DataFrame, recall_predictions: pd.DataFrame) -> pd.DataFrame:
    if cards.empty or recall_predictions.empty:
        return pd.DataFrame()
    merge_columns = ["card_id", "variable", "origin", "horizon", "target_realized"]
    for optional in ["regime_label", "contamination_label"]:
        if optional in cards.columns:
            merge_columns.append(optional)
    joined = recall_predictions.merge(
        cards[merge_columns],
        on="card_id",
        how="inner",
    )
    joined = joined[joined["recalled_realized"].map(np.isfinite)].copy()
    if joined.empty:
        return pd.DataFrame(
            [
                {
                    "source": "direct_recall",
                    "variable": "ALL",
                    "n": 0,
                    "coverage": 0.0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                }
            ]
        )
    joined["error"] = joined["recalled_realized"].astype(float) - joined["target_realized"].astype(float)
    rows = []
    total_cards = cards.shape[0]
    for variable, group in joined.groupby("variable", dropna=False):
        rows.append(recall_score_row(group, variable=str(variable), total_cards=int((cards["variable"] == variable).sum())))
    rows.append(recall_score_row(joined, variable="ALL", total_cards=total_cards))
    return pd.DataFrame(rows).sort_values(["variable"]).reset_index(drop=True)


def recall_score_row(group: pd.DataFrame, *, variable: str, total_cards: int) -> dict[str, Any]:
    errors = group["error"].astype(float)
    return {
        "source": "direct_recall",
        "variable": variable,
        "n": int(group.shape[0]),
        "coverage": float(group.shape[0] / total_cards) if total_cards else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "bias": float(np.mean(errors)) if len(errors) else np.nan,
    }


def build_audit_report(
    manifest: dict[str, Any],
    source_scores: pd.DataFrame,
    surprise: pd.DataFrame,
    theil: pd.DataFrame,
    paired: pd.DataFrame,
    recall_scores: pd.DataFrame,
    *,
    belief_summary: pd.DataFrame | None = None,
) -> str:
    overall = filter_variable(source_scores, "ALL").sort_values("rmse") if "rmse" in source_scores else pd.DataFrame()
    belief_all = filter_variable(belief_summary if belief_summary is not None else pd.DataFrame(), "ALL")
    surprise_gap = filter_variable(surprise, "ALL")
    if not surprise_gap.empty and "source" in surprise_gap:
        surprise_gap = surprise_gap[surprise_gap["source"] == "llm_minus_spf_gap"].copy()
    theil_all = filter_variable(theil, "ALL")
    if not theil_all.empty and "theils_u_vs_no_change" in theil_all:
        theil_all = theil_all.sort_values("theils_u_vs_no_change")
    paired_all = filter_variable(paired, "ALL")
    if not paired_all.empty and "mean_squared_loss_diff_vs_baseline" in paired_all:
        paired_all = paired_all.sort_values("mean_squared_loss_diff_vs_baseline")
    lines = [
        "# Forecast Tournament Audit",
        "",
        "## Bottom Line",
        audit_bottom_line(overall, surprise_gap, theil_all, recall_scores),
        "",
        "## Overall Leaderboard With RMSE CI",
        markdown_table(select_columns(overall, ["source", "n", "rmse", "rmse_ci_low", "rmse_ci_high", "mae", "direction_accuracy"])),
        "",
        "## Surprise Split: GPT Minus SPF",
        markdown_table(select_columns(surprise_gap, ["bucket_type", "bucket", "n", "rmse", "mae"])),
        "",
        "## Theil's U Vs No-Change",
        markdown_table(select_columns(theil_all, ["source", "n", "rmse", "no_change_rmse", "theils_u_vs_no_change", "sane_vs_no_change"])),
        "",
        "## Paired Loss Difference Vs SPF",
        markdown_table(
            select_columns(
                paired_all,
                [
                    "source",
                    "n",
                    "mean_squared_loss_diff_vs_baseline",
                    "mean_abs_loss_diff_vs_baseline",
                    "dm_t_approx_squared_loss",
                    "normal_approx_p_two_sided",
                ],
            )
        ),
        "",
        "## Belief Structure",
        markdown_table(
            select_columns(
                belief_all,
                [
                    "source",
                    "n",
                    "underreaction_slope_error_on_revision",
                    "extrapolation_slope_forecast_change_on_recent_signal",
                    "mean_panel_std",
                    "interval_coverage_p10_p90",
                    "confidence_abs_error_corr",
                    "surprise_rmse_gap_high_minus_low_spf_error",
                ],
            )
        ),
        "",
        "## Direct Recall Scores",
        markdown_table(recall_scores if not recall_scores.empty else pd.DataFrame()),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def filter_variable(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    if frame.empty or "variable" not in frame:
        return pd.DataFrame()
    return frame[frame["variable"] == variable].copy()


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    selected = frame.copy()
    for column in columns:
        if column not in selected:
            selected[column] = np.nan
    return selected[columns]


def audit_bottom_line(
    overall: pd.DataFrame,
    surprise_gap: pd.DataFrame,
    theil_all: pd.DataFrame,
    recall_scores: pd.DataFrame,
) -> str:
    parts: list[str] = []
    if not overall.empty:
        best = overall.iloc[0]
        parts.append(f"Best overall source is `{best['source']}` with RMSE `{float(best['rmse']):.4f}`.")
    if not surprise_gap.empty:
        gap_text = ", ".join(f"{row['bucket']}={float(row['rmse']):.4f}" for _, row in surprise_gap.iterrows())
        parts.append(f"LLM-minus-SPF RMSE gaps by bucket: {gap_text}. Negative favors LLM.")
    if "theils_u_vs_no_change" in theil_all and "source" in theil_all:
        bad_controls = theil_all[(theil_all["theils_u_vs_no_change"] > 1.5) & (~theil_all["source"].isin(["no_change"]))]
    else:
        bad_controls = pd.DataFrame()
    if not bad_controls.empty:
        parts.append(f"Controls with Theil's U above 1.5: `{', '.join(bad_controls['source'].astype(str).head(6))}`.")
    else:
        parts.append("No overall control has Theil's U above 1.5 versus no-change in this run.")
    if recall_scores.empty:
        parts.append("Direct realized-value recall was not run.")
    else:
        overall_recall = recall_scores[recall_scores["variable"] == "ALL"]
        if not overall_recall.empty:
            row = overall_recall.iloc[0]
            parts.append(f"Direct recall coverage is `{float(row['coverage']):.2%}` with MAE `{row['mae']}`.")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
