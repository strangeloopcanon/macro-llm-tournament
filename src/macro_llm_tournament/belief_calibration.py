from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, markdown_table
from .demand_vintage_oos import (
    DEMAND_VINTAGE_OOS_VERSION,
    TARGET_SPECS,
    audit_card_leakage,
    frame_sha256,
    score_vintage_forecasts,
    summarize_vintage_scores,
)
from .macro_performance_gate import build_oos_pairwise_comparison


BELIEF_CALIBRATION_VERSION = "belief_dynamics_calibration_v1"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "belief_calibration"
LLM_VARIANT = "llm_belief"
BASELINE_SOURCES = {"no_change", "rolling_mean", "rolling_trend", "ar2", "recursive_least_squares"}
FEATURE_COLUMNS = [
    "forecast_value",
    "confidence",
    "no_change",
    "rolling_mean",
    "rolling_trend",
    "ar2",
    "recursive_least_squares",
    "recent_change",
    "mean_change_4",
    "trend_gap",
    "volatility_4",
    "acceleration",
    "drawdown_signal",
    "rebound_signal",
    "regime_shift_signal",
]


@dataclass(frozen=True)
class VintageRun:
    root: Path
    manifest: dict[str, Any]
    cards: pd.DataFrame
    targets: pd.DataFrame
    forecasts: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit validation-only belief dynamics calibration and score it on held-out OOS cards.")
    parser.add_argument("--calibration-run-dir", required=True, help="Validation-split demand_vintage_oos run used to fit calibration.")
    parser.add_argument("--evaluation-run-dir", required=True, help="Held-out demand_vintage_oos run to score after applying calibration.")
    parser.add_argument("--behavior-baseline-dir", default=None, help="Optional uncalibrated demand_economy run for behavior comparison.")
    parser.add_argument("--behavior-calibrated-dir", default=None, help="Optional calibrated demand_economy run for behavior comparison.")
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_belief_calibration(
        calibration_run_dir=Path(args.calibration_run_dir),
        evaluation_run_dir=Path(args.evaluation_run_dir),
        behavior_baseline_dir=Path(args.behavior_baseline_dir) if args.behavior_baseline_dir else None,
        behavior_calibrated_dir=Path(args.behavior_calibrated_dir) if args.behavior_calibrated_dir else None,
        ridge_alpha=float(args.ridge_alpha),
    )
    write_belief_calibration_outputs(result, Path(args.output_dir))
    print(f"Wrote belief calibration run to {args.output_dir}")
    print(json.dumps({"verdict": result["manifest"]["verdict"], "passed": result["manifest"]["passed"]}, indent=2, sort_keys=True))
    return 0 if result["manifest"]["passed"] else 1


def build_belief_calibration(
    *,
    calibration_run_dir: Path,
    evaluation_run_dir: Path,
    behavior_baseline_dir: Path | None = None,
    behavior_calibrated_dir: Path | None = None,
    ridge_alpha: float = 2.0,
) -> dict[str, Any]:
    calibration = load_vintage_run(calibration_run_dir)
    evaluation = load_vintage_run(evaluation_run_dir)
    calibration_features = build_calibration_feature_frame(calibration.cards, calibration.forecasts, calibration.targets)
    evaluation_features = build_calibration_feature_frame(evaluation.cards, evaluation.forecasts, evaluation.targets)
    model, validation_scores = fit_calibration_models(calibration_features, ridge_alpha=ridge_alpha)
    calibrated_forecasts, application = apply_calibration_models(evaluation_features, model)
    forecasts = pd.concat([evaluation.forecasts, calibrated_forecasts], ignore_index=True) if not calibrated_forecasts.empty else evaluation.forecasts.copy()
    scores, joined = score_vintage_forecasts(forecasts, evaluation.targets)
    summary = summarize_vintage_scores(scores)
    leakage = audit_card_leakage(evaluation.cards)
    pairwise = _pairwise_from_joined(joined)
    behavior = build_behavior_comparison(behavior_baseline_dir, behavior_calibrated_dir)
    profile = build_calibration_profile(model, validation_scores, application)
    manifest = build_calibration_manifest(
        calibration,
        evaluation,
        profile,
        model,
        validation_scores,
        application,
        summary,
        pairwise,
        behavior,
        leakage,
        ridge_alpha=ridge_alpha,
    )
    report = build_belief_calibration_report(manifest, summary, validation_scores, model, pairwise, behavior, leakage)
    return {
        "manifest": manifest,
        "profile": profile,
        "calibration_features": calibration_features,
        "evaluation_features": evaluation_features,
        "model": model,
        "validation_scores": validation_scores,
        "application": application,
        "cards": evaluation.cards,
        "targets": evaluation.targets,
        "forecasts": forecasts,
        "scores": scores,
        "joined": joined,
        "summary": summary,
        "pairwise": pairwise,
        "behavior": behavior,
        "leakage": leakage,
        "report": report,
    }


def load_vintage_run(root: Path) -> VintageRun:
    required = {
        "manifest.json": "manifest",
        "demand_vintage_oos_cards.csv": "cards",
        "demand_vintage_oos_targets.csv": "targets",
        "demand_vintage_oos_forecasts.csv": "forecasts",
    }
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Vintage OOS run missing required file(s) under {root}: {', '.join(missing)}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return VintageRun(
        root=root,
        manifest=manifest,
        cards=pd.read_csv(root / "demand_vintage_oos_cards.csv"),
        targets=pd.read_csv(root / "demand_vintage_oos_targets.csv"),
        forecasts=pd.read_csv(root / "demand_vintage_oos_forecasts.csv"),
    )


def build_calibration_feature_frame(cards: pd.DataFrame, forecasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if cards.empty or forecasts.empty or targets.empty:
        return pd.DataFrame()
    card_features = [_card_feature_row(row) for _, row in cards.iterrows()]
    features = pd.DataFrame(card_features)
    baseline = (
        forecasts[forecasts["source"].astype(str).isin(BASELINE_SOURCES)]
        .pivot_table(index="card_id", columns="source", values="forecast_value", aggfunc="first")
        .reset_index()
    )
    for source in BASELINE_SOURCES:
        if source not in baseline:
            baseline[source] = np.nan
    features = features.merge(
        baseline[["card_id", "no_change", "rolling_mean", "rolling_trend", "ar2", "recursive_least_squares"]],
        on="card_id",
        how="left",
    )
    features["trend_gap"] = pd.to_numeric(features["rolling_trend"], errors="coerce") - pd.to_numeric(
        features["rolling_mean"], errors="coerce"
    )
    target_cols = ["card_id", "target_value", "default_scale", "target_available"]
    features = features.merge(targets[target_cols], on="card_id", how="left")
    llm = forecasts[forecasts["variant"].astype(str) == LLM_VARIANT].copy()
    llm = llm[~llm["source"].astype(str).str.endswith("_calibrated")]
    merged = llm.merge(features, on=["card_id", "origin_id", "split", "target_name"], how="inner", suffixes=("", "_feature"))
    for column in FEATURE_COLUMNS + ["target_value", "default_scale"]:
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["forecast_value", "target_value", "default_scale"])
    return merged.reset_index(drop=True)


def fit_calibration_models(features: pd.DataFrame, *, ridge_alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (source, target_name), group in features.groupby(["source", "target_name"], sort=True):
        train = group.copy()
        X = train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = pd.to_numeric(train["target_value"], errors="coerce").to_numpy(dtype=float)
        scales = pd.to_numeric(train["default_scale"], errors="coerce").replace(0.0, np.nan).fillna(1.0).to_numpy(dtype=float)
        if len(y) < 3 or not np.isfinite(y).all():
            continue
        means = X.mean(axis=0).to_numpy(dtype=float)
        stds = X.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
        Z = (X.to_numpy(dtype=float) - means) / stds
        design = np.column_stack([np.ones(Z.shape[0]), Z])
        penalty = np.eye(design.shape[1]) * max(0.0, float(ridge_alpha))
        penalty[0, 0] = 0.0
        try:
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        fitted = design @ coef
        original = pd.to_numeric(train["forecast_value"], errors="coerce").to_numpy(dtype=float)
        blends = np.linspace(0.0, 1.0, 5)
        losses = [float(np.nanmean(np.abs(((1.0 - blend) * original + blend * fitted) - y) / scales)) for blend in blends]
        best_idx = int(np.nanargmin(losses))
        selected_blend = float(blends[best_idx])
        uncalibrated_loss = float(losses[0])
        calibrated_loss = float(losses[best_idx])
        active = bool(selected_blend > 0.0 and calibrated_loss <= uncalibrated_loss)
        residual = y - ((1.0 - selected_blend) * original + selected_blend * fitted)
        model_id = _hash_json({"source": source, "target_name": target_name, "coef": coef.tolist(), "blend": selected_blend})
        row = {
            "model_id": model_id,
            "source": str(source),
            "target_name": str(target_name),
            "n_validation": int(len(train)),
            "ridge_alpha": float(ridge_alpha),
            "selected_blend": selected_blend if active else 0.0,
            "active": active,
            "validation_uncalibrated_wnae": uncalibrated_loss,
            "validation_calibrated_wnae": calibrated_loss if active else uncalibrated_loss,
            "validation_improvement_pct": 100.0 * (uncalibrated_loss - calibrated_loss) / uncalibrated_loss if active and uncalibrated_loss > 0 else 0.0,
            "residual_abs_p50": float(np.nanquantile(np.abs(residual), 0.50)),
            "residual_abs_p90": float(np.nanquantile(np.abs(residual), 0.90)),
            "intercept": float(coef[0]),
            "feature_means_json": json.dumps(dict(zip(FEATURE_COLUMNS, means)), sort_keys=True),
            "feature_stds_json": json.dumps(dict(zip(FEATURE_COLUMNS, stds)), sort_keys=True),
            "feature_coefs_json": json.dumps(dict(zip(FEATURE_COLUMNS, coef[1:])), sort_keys=True),
        }
        model_rows.append(row)
        score_rows.append(
            {
                "source": str(source),
                "target_name": str(target_name),
                "n_validation": int(len(train)),
                "uncalibrated_wnae": uncalibrated_loss,
                "calibrated_wnae": row["validation_calibrated_wnae"],
                "improvement_pct": row["validation_improvement_pct"],
                "selected_blend": row["selected_blend"],
                "active": active,
            }
        )
    model = pd.DataFrame(model_rows)
    scores = pd.DataFrame(score_rows)
    if not model.empty:
        model = model.sort_values(["source", "target_name"]).reset_index(drop=True)
    if not scores.empty:
        scores = scores.sort_values(["source", "target_name"]).reset_index(drop=True)
    return model, scores


def apply_calibration_models(features: pd.DataFrame, model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty or model.empty:
        return pd.DataFrame(), pd.DataFrame()
    model_by_key = {(str(row["source"]), str(row["target_name"])): row for _, row in model.iterrows()}
    forecast_rows: list[dict[str, Any]] = []
    app_rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        key = (str(row["source"]), str(row["target_name"]))
        spec = model_by_key.get(key)
        if spec is None:
            continue
        prediction = _apply_model_row(row, spec)
        original = float(row["forecast_value"])
        blend = float(spec.get("selected_blend", 0.0) or 0.0)
        scale = float(row.get("default_scale", 1.0) or 1.0)
        baseline_bounds = [
            original,
            float(row.get("no_change", original)),
            float(row.get("rolling_mean", original)),
            float(row.get("rolling_trend", original)),
            float(row.get("ar2", original)),
            float(row.get("recursive_least_squares", original)),
        ]
        lower = min(baseline_bounds) - 8.0 * scale
        upper = max(baseline_bounds) + 8.0 * scale
        calibrated = float(np.clip((1.0 - blend) * original + blend * prediction, lower, upper))
        calibrated_source = f"{row['source']}_calibrated"
        confidence = _calibrated_confidence(float(row.get("confidence", 0.5)), spec, row)
        forecast_rows.append(
            {
                "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                "card_id": row["card_id"],
                "origin_id": row["origin_id"],
                "split": row["split"],
                "target_name": row["target_name"],
                "source": calibrated_source,
                "variant": LLM_VARIANT,
                "forecast_value": calibrated,
                "confidence": confidence,
                "reason": f"validation-only calibrated belief dynamics model {spec['model_id']}",
            }
        )
        app_rows.append(
            {
                "source": row["source"],
                "calibrated_source": calibrated_source,
                "target_name": row["target_name"],
                "card_id": row["card_id"],
                "model_id": spec["model_id"],
                "selected_blend": blend,
                "original_forecast": original,
                "model_forecast": float(prediction),
                "calibrated_forecast": calibrated,
                "forecast_delta": calibrated - original,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(forecast_rows), pd.DataFrame(app_rows)


def build_calibration_profile(model: pd.DataFrame, validation_scores: pd.DataFrame, application: pd.DataFrame) -> dict[str, Any]:
    active = model[model["active"].astype(bool)].copy() if not model.empty and "active" in model else pd.DataFrame()
    profile_id = _hash_json(
        {
            "schema_version": BELIEF_CALIBRATION_VERSION,
            "models": model.where(pd.notna(model), None).to_dict(orient="records") if not model.empty else [],
        }
    )
    demand_adjustments = _demand_adjustments_from_models(active, validation_scores)
    return {
        "schema_version": BELIEF_CALIBRATION_VERSION,
        "profile_id": profile_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "source_count": int(model["source"].nunique()) if not model.empty else 0,
        "model_rows": model.where(pd.notna(model), None).to_dict(orient="records") if not model.empty else [],
        "validation_scores": validation_scores.where(pd.notna(validation_scores), None).to_dict(orient="records")
        if not validation_scores.empty
        else [],
        "application_rows": int(application.shape[0]),
        "demand_adjustments": demand_adjustments,
    }


def build_behavior_comparison(baseline_dir: Path | None, calibrated_dir: Path | None) -> pd.DataFrame:
    if baseline_dir is None or calibrated_dir is None:
        return pd.DataFrame()
    baseline_val = _read_optional_csv(baseline_dir / "demand_validation_scores.csv")
    calibrated_val = _read_optional_csv(calibrated_dir / "demand_validation_scores.csv")
    baseline_beliefs = _read_optional_csv(baseline_dir / "demand_belief_target_scores.csv")
    calibrated_beliefs = _read_optional_csv(calibrated_dir / "demand_belief_target_scores.csv")
    rows: list[dict[str, Any]] = []
    if not baseline_val.empty and not calibrated_val.empty:
        base = _best_llm_source(baseline_val)
        cal = _best_llm_source(calibrated_val, calibrated=True)
        rows.extend(_compare_validation_frames(baseline_val, calibrated_val, base, cal))
    if not baseline_beliefs.empty and not calibrated_beliefs.empty:
        base = _best_llm_source(baseline_beliefs)
        cal = _best_llm_source(calibrated_beliefs, calibrated=True)
        rows.extend(
            _compare_metric_frames(
                baseline_beliefs,
                calibrated_beliefs,
                base,
                cal,
                key_column="belief_variable",
                value_column="normalized_mae",
                surface="belief_target",
                lower_is_better=True,
            )
        )
    return pd.DataFrame(rows)


def _compare_validation_frames(
    baseline: pd.DataFrame,
    calibrated: pd.DataFrame,
    baseline_source: str | None,
    calibrated_source: str | None,
) -> list[dict[str, Any]]:
    if not baseline_source or not calibrated_source:
        return []
    left_columns = ["metric", "value", "target_low", "target_high", "passed"]
    right_columns = ["metric", "value", "target_low", "target_high", "passed"]
    left = baseline[baseline["source"].astype(str) == baseline_source][left_columns].copy()
    right = calibrated[calibrated["source"].astype(str) == calibrated_source][right_columns].copy()
    merged = left.merge(right, on="metric", suffixes=("_baseline", "_calibrated"))
    rows = []
    for _, row in merged.iterrows():
        base = float(row["value_baseline"])
        cal = float(row["value_calibrated"])
        base_distance = _target_distance(base, row["target_low_baseline"], row["target_high_baseline"])
        cal_distance = _target_distance(cal, row["target_low_calibrated"], row["target_high_calibrated"])
        base_passed = bool(row["passed_baseline"])
        cal_passed = bool(row["passed_calibrated"])
        improved = bool((cal_passed and not base_passed) or (cal_distance < base_distance))
        regressed = bool(base_passed and not cal_passed)
        rows.append(
            {
                "surface": "validation",
                "metric": row["metric"],
                "baseline_source": baseline_source,
                "calibrated_source": calibrated_source,
                "baseline_value": base,
                "calibrated_value": cal,
                "delta": cal - base,
                "baseline_passed": base_passed,
                "calibrated_passed": cal_passed,
                "baseline_distance": base_distance,
                "calibrated_distance": cal_distance,
                "improved": improved,
                "regressed": regressed,
            }
        )
    return rows


def build_calibration_manifest(
    calibration: VintageRun,
    evaluation: VintageRun,
    profile: dict[str, Any],
    model: pd.DataFrame,
    validation_scores: pd.DataFrame,
    application: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    behavior: pd.DataFrame,
    leakage: pd.DataFrame,
    *,
    ridge_alpha: float,
) -> dict[str, Any]:
    calibration_splits = _manifest_splits(calibration)
    evaluation_splits = _manifest_splits(evaluation)
    split_overlap = sorted(set(calibration_splits) & set(evaluation_splits))
    split_disjoint = bool(calibration_splits) and bool(evaluation_splits) and not split_overlap
    best_calibrated = _best_summary_loss(summary, calibrated=True)
    best_uncalibrated = _best_summary_loss(summary, calibrated=False)
    best_baseline = _best_baseline_loss(summary)
    empirical_ready = bool(split_disjoint and np.isfinite(best_calibrated) and best_calibrated <= 1.0)
    relative_improved = bool(split_disjoint and np.isfinite(best_calibrated) and np.isfinite(best_uncalibrated) and best_calibrated < best_uncalibrated)
    beats_baseline = bool(split_disjoint and np.isfinite(best_calibrated) and np.isfinite(best_baseline) and best_calibrated < best_baseline)
    if not split_disjoint:
        verdict = "belief_calibration_split_leakage_failed"
    else:
        verdict = "belief_calibration_empirical_ready" if empirical_ready else "belief_calibration_evaluated"
    passed = bool(split_disjoint and leakage.empty and not model.empty and not application.empty)
    return {
        "schema_version": BELIEF_CALIBRATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "verdict": verdict,
        "passed": passed,
        "mode": evaluation.manifest.get("mode"),
        "forecast_mode": evaluation.manifest.get("forecast_mode"),
        "provider": evaluation.manifest.get("provider"),
        "models": evaluation.manifest.get("models", []),
        "live_call_count": int(evaluation.manifest.get("live_call_count", 0) or 0),
        "cache_hit_count": int(evaluation.manifest.get("cache_hit_count", 0) or 0),
        "raw_record_count": int(evaluation.manifest.get("raw_record_count", 0) or 0),
        "empirical_ready": empirical_ready,
        "relative_improved_vs_uncalibrated": relative_improved,
        "beats_best_baseline": beats_baseline,
        "ridge_alpha": float(ridge_alpha),
        "profile_id": profile.get("profile_id"),
        "calibration_run_dir": str(calibration.root),
        "evaluation_run_dir": str(evaluation.root),
        "calibration_manifest_sha256": _hash_json(calibration.manifest),
        "evaluation_manifest_sha256": _hash_json(evaluation.manifest),
        "calibration_splits": calibration_splits,
        "evaluation_splits": evaluation_splits,
        "calibration_split_disjoint": split_disjoint,
        "calibration_split_overlap": split_overlap,
        "model_rows": int(model.shape[0]),
        "active_model_rows": int(model["active"].astype(bool).sum()) if not model.empty and "active" in model else 0,
        "validation_score_rows": int(validation_scores.shape[0]),
        "application_rows": int(application.shape[0]),
        "best_calibrated_wnae": best_calibrated,
        "best_uncalibrated_wnae": best_uncalibrated,
        "best_baseline_wnae": best_baseline,
        "best_calibrated_improvement_vs_uncalibrated_pct": _improvement_pct(best_uncalibrated, best_calibrated),
        "best_calibrated_improvement_vs_baseline_pct": _improvement_pct(best_baseline, best_calibrated),
        "pairwise_best_bootstrap_share_positive": _pairwise_best_share(pairwise),
        "behavior_rows": int(behavior.shape[0]),
        "leakage_issue_count": int(leakage.shape[0]),
        "cards_sha256": frame_sha256(evaluation.cards),
        "targets_sha256": frame_sha256(evaluation.targets),
        "outputs": [
            "belief_calibration_profile.json",
            "belief_calibration_model.csv",
            "belief_calibration_validation_scores.csv",
            "belief_calibration_application.csv",
            "belief_calibration_features_validation.csv",
            "belief_calibration_features_evaluation.csv",
            "demand_vintage_oos_cards.csv",
            "demand_vintage_oos_targets.csv",
            "demand_vintage_oos_forecasts.csv",
            "demand_vintage_oos_scores.csv",
            "demand_vintage_oos_joined_errors.csv",
            "demand_vintage_oos_summary.csv",
            "demand_vintage_oos_leakage_audit.csv",
            "belief_calibration_pairwise.csv",
            "belief_calibration_behavior_comparison.csv",
            "belief_calibration_report.md",
            "manifest.json",
        ],
    }


def write_belief_calibration_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "belief_calibration_profile.json").write_text(
        json.dumps(_jsonable(result["profile"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["model"].to_csv(output_dir / "belief_calibration_model.csv", index=False)
    result["validation_scores"].to_csv(output_dir / "belief_calibration_validation_scores.csv", index=False)
    result["application"].to_csv(output_dir / "belief_calibration_application.csv", index=False)
    result["calibration_features"].to_csv(output_dir / "belief_calibration_features_validation.csv", index=False)
    result["evaluation_features"].to_csv(output_dir / "belief_calibration_features_evaluation.csv", index=False)
    result["cards"].to_csv(output_dir / "demand_vintage_oos_cards.csv", index=False)
    result["targets"].to_csv(output_dir / "demand_vintage_oos_targets.csv", index=False)
    result["forecasts"].to_csv(output_dir / "demand_vintage_oos_forecasts.csv", index=False)
    result["scores"].to_csv(output_dir / "demand_vintage_oos_scores.csv", index=False)
    result["joined"].to_csv(output_dir / "demand_vintage_oos_joined_errors.csv", index=False)
    result["summary"].to_csv(output_dir / "demand_vintage_oos_summary.csv", index=False)
    result["leakage"].to_csv(output_dir / "demand_vintage_oos_leakage_audit.csv", index=False)
    result["pairwise"].to_csv(output_dir / "belief_calibration_pairwise.csv", index=False)
    result["behavior"].to_csv(output_dir / "belief_calibration_behavior_comparison.csv", index=False)
    (output_dir / "belief_calibration_report.md").write_text(result["report"], encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(result["manifest"]), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_belief_calibration_report(
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    validation_scores: pd.DataFrame,
    model: pd.DataFrame,
    pairwise: pd.DataFrame,
    behavior: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    lines = [
        "# Belief Dynamics Calibration",
        "",
        "## Bottom Line",
        _calibration_bottom_line(manifest),
        "",
        "## OOS Score Summary",
        markdown_table(summary),
        "",
        "## Validation Fit",
        markdown_table(validation_scores),
        "",
        "## Calibration Models",
        markdown_table(model.drop(columns=["feature_means_json", "feature_stds_json", "feature_coefs_json"], errors="ignore") if not model.empty else model),
        "",
        "## OOS Paired Comparison",
        markdown_table(pairwise),
        "",
        "## Behavior Comparison",
        markdown_table(behavior),
        "",
        "## Leakage Audit",
        markdown_table(leakage),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _card_feature_row(card: pd.Series) -> dict[str, Any]:
    history = json.loads(str(card["history_json"]))
    values = [float(item["value"]) for item in history if np.isfinite(float(item["value"]))]
    transform = str(card["transform"])
    if len(values) < 2:
        changes = [0.0]
    elif transform == "pct_change":
        changes = [100.0 * (cur / prev - 1.0) for prev, cur in zip(values[:-1], values[1:]) if abs(prev) > 1e-12]
    else:
        changes = [cur - prev for prev, cur in zip(values[:-1], values[1:])]
    recent = float(changes[-1]) if changes else 0.0
    trailing = np.asarray(changes[-4:] if changes else [0.0], dtype=float)
    prior = np.asarray(changes[-5:-1] if len(changes) >= 5 else changes[:-1] or [0.0], dtype=float)
    current = float(values[-1]) if values else 0.0
    high = max(values) if values else current
    low = min(values) if values else current
    if transform == "pct_change" and abs(high) > 1e-12:
        drawdown = min(0.0, 100.0 * (current / high - 1.0))
        runup = max(0.0, 100.0 * (current / max(abs(low), 1e-12) - 1.0))
    else:
        drawdown = min(0.0, current - high)
        runup = max(0.0, current - low)
    mean_change = float(np.mean(trailing))
    acceleration = float(recent - np.mean(prior)) if prior.size else 0.0
    rebound = max(0.0, recent) * max(0.0, -float(np.mean(prior))) + max(0.0, runup) * max(0.0, -drawdown)
    volatility = float(np.std(trailing))
    return {
        "card_id": card["card_id"],
        "origin_id": card["origin_id"],
        "split": card["split"],
        "target_name": card["target_name"],
        "recent_change": recent,
        "mean_change_4": mean_change,
        "volatility_4": volatility,
        "acceleration": acceleration,
        "drawdown_signal": abs(float(drawdown)),
        "rebound_signal": float(rebound),
        "regime_shift_signal": float(abs(acceleration) + volatility),
    }


def _apply_model_row(row: pd.Series, spec: pd.Series) -> float:
    means = json.loads(str(spec["feature_means_json"]))
    stds = json.loads(str(spec["feature_stds_json"]))
    coefs = json.loads(str(spec["feature_coefs_json"]))
    value = float(spec["intercept"])
    for feature in FEATURE_COLUMNS:
        raw = float(row.get(feature, 0.0) or 0.0)
        mean = float(means.get(feature, 0.0) or 0.0)
        std = float(stds.get(feature, 1.0) or 1.0) or 1.0
        value += ((raw - mean) / std) * float(coefs.get(feature, 0.0) or 0.0)
    return float(value)


def _calibrated_confidence(confidence: float, spec: pd.Series, row: pd.Series) -> float:
    residual_p90 = float(spec.get("residual_abs_p90", 1.0) or 1.0)
    scale = max(1e-6, float(row.get("default_scale", 1.0) or 1.0))
    uncertainty = min(1.0, residual_p90 / (4.0 * scale))
    return float(np.clip(0.15 + 0.70 * confidence * (1.0 - 0.45 * uncertainty), 0.05, 0.95))


def _demand_adjustments_from_models(model: pd.DataFrame, validation_scores: pd.DataFrame) -> dict[str, Any]:
    if model.empty:
        strength = 0.0
    else:
        improvements = pd.to_numeric(model.get("validation_improvement_pct", pd.Series(dtype=float)), errors="coerce").clip(lower=0.0)
        strength = float(np.nanmean(improvements) / 100.0) if not improvements.empty else 0.0
    strength = float(np.clip(strength, 0.0, 0.12))
    active_targets = set(model["target_name"].astype(str)) if not model.empty else set()
    return {
        "calibration_strength": strength,
        "income_rebound_gain": 0.025 + 0.10 * strength if {"real_consumption_growth_pct", "output_growth_pct"} & active_targets else 0.015,
        "inflation_attention_gain": 0.015 + 0.07 * strength if "inflation_growth_pct" in active_targets else 0.010,
        "job_risk_regime_gain": 0.030 + 0.12 * strength if "unemployment_rate_level" in active_targets else 0.020,
        "confidence_rebound_gain": 0.030 + 0.12 * strength if "sentiment_growth_pct" in active_targets else 0.015,
        "precaution_uncertainty_gain": 0.025 + 0.10 * strength,
        "active_targets": sorted(active_targets),
        "source": "validation_only_vintage_oos_profile",
    }


def _pairwise_from_joined(joined: pd.DataFrame) -> pd.DataFrame:
    with pd.option_context("mode.copy_on_write", True):
        temp_root = OUTPUT_ROOT / ".tmp_belief_calibration_pairwise"
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            joined.to_csv(temp_root / "demand_vintage_oos_joined_errors.csv", index=False)
            return build_oos_pairwise_comparison(temp_root)
        finally:
            try:
                (temp_root / "demand_vintage_oos_joined_errors.csv").unlink()
                temp_root.rmdir()
            except OSError:
                pass


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _best_llm_source(frame: pd.DataFrame, *, calibrated: bool = False) -> str | None:
    if frame.empty or "source" not in frame:
        return None
    sources = sorted(frame["source"].dropna().astype(str).unique().tolist())
    candidates = [source for source in sources if "llm_belief" in source and ("calibrated" in source) == calibrated]
    return candidates[0] if candidates else None


def _compare_metric_frames(
    baseline: pd.DataFrame,
    calibrated: pd.DataFrame,
    baseline_source: str | None,
    calibrated_source: str | None,
    *,
    key_column: str,
    value_column: str,
    surface: str,
    lower_is_better: bool = False,
) -> list[dict[str, Any]]:
    if not baseline_source or not calibrated_source:
        return []
    left = baseline[baseline["source"].astype(str) == baseline_source][[key_column, value_column]].copy()
    right = calibrated[calibrated["source"].astype(str) == calibrated_source][[key_column, value_column]].copy()
    merged = left.merge(right, on=key_column, suffixes=("_baseline", "_calibrated"))
    rows = []
    for _, row in merged.iterrows():
        base = float(row[f"{value_column}_baseline"])
        cal = float(row[f"{value_column}_calibrated"])
        delta = cal - base
        improved = cal < base if lower_is_better else abs(cal) >= abs(base)
        rows.append(
            {
                "surface": surface,
                "metric": row[key_column],
                "baseline_source": baseline_source,
                "calibrated_source": calibrated_source,
                "baseline_value": base,
                "calibrated_value": cal,
                "delta": delta,
                "improved": bool(improved),
            }
        )
    return rows


def _target_distance(value: float, target_low: Any, target_high: Any) -> float:
    low = float(target_low) if target_low is not None and not pd.isna(target_low) else -np.inf
    high = float(target_high) if target_high is not None and not pd.isna(target_high) else np.inf
    if low <= value <= high:
        return 0.0
    if value < low:
        return float(low - value)
    return float(value - high)


def _best_summary_loss(summary: pd.DataFrame, *, calibrated: bool) -> float:
    if summary.empty:
        return np.nan
    rows = summary[summary["variant"].astype(str) == LLM_VARIANT].copy()
    rows = rows[rows["source"].astype(str).str.endswith("_calibrated") == calibrated]
    if rows.empty:
        return np.nan
    return float(pd.to_numeric(rows["weighted_normalized_abs_error"], errors="coerce").min())


def _best_baseline_loss(summary: pd.DataFrame) -> float:
    if summary.empty:
        return np.nan
    rows = summary[summary["source"].astype(str).isin(BASELINE_SOURCES)].copy()
    if rows.empty:
        return np.nan
    return float(pd.to_numeric(rows["weighted_normalized_abs_error"], errors="coerce").min())


def _improvement_pct(baseline: float, candidate: float) -> float | None:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline <= 0:
        return None
    return float(100.0 * (baseline - candidate) / baseline)


def _pairwise_best_share(pairwise: pd.DataFrame) -> float | None:
    if pairwise.empty or "bootstrap_share_positive" not in pairwise:
        return None
    rows = pairwise[pairwise["llm_source"].astype(str).str.endswith("_calibrated")]
    values = pd.to_numeric(rows["bootstrap_share_positive"], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _manifest_splits(run: VintageRun) -> list[str]:
    splits = run.manifest.get("splits")
    if isinstance(splits, list) and splits:
        return [str(item) for item in splits]
    return sorted(run.cards["split"].dropna().astype(str).unique().tolist()) if "split" in run.cards else []


def _calibration_bottom_line(manifest: dict[str, Any]) -> str:
    if manifest.get("empirical_ready"):
        return "Verdict: `belief_calibration_empirical_ready`. Calibrated beliefs clear the absolute OOS threshold."
    improvement = manifest.get("best_calibrated_improvement_vs_uncalibrated_pct")
    if improvement is not None and float(improvement) > 0:
        return (
            "Verdict: `belief_calibration_evaluated`. Validation-locked calibration improves held-out OOS error "
            f"by {float(improvement):.2f}% versus the uncalibrated LLM, but the absolute empirical threshold is still not cleared."
        )
    return "Verdict: `belief_calibration_evaluated`. Calibration ran and scored, but it did not improve the held-out LLM OOS score."


def _hash_json(data: Any) -> str:
    encoded = json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if np.isfinite(item) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
