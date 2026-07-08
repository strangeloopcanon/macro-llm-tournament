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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "work" / "empirical_bridge" / "spending_belief_panel.csv"
DEFAULT_COVERAGE = PROJECT_ROOT / "work" / "empirical_bridge" / "spending_belief_panel_coverage.json"
DEFAULT_BRIDGE = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v4.json"
DEFAULT_CELL_TARGETS = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v4_cell_targets.csv"
DEFAULT_VALIDATION_SCORES = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v4_validation_scores.csv"
DEFAULT_STABILIZED_BRIDGE = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v5_stabilized.json"
DEFAULT_STABILIZED_CELL_TARGETS = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v5_stabilized_cell_targets.csv"
DEFAULT_STABILIZED_VALIDATION_SCORES = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v5_stabilized_validation_scores.csv"

BRIDGE_SPEC_VERSION = "empirical_bridge_v4"
STABILIZED_BRIDGE_SPEC_VERSION = "empirical_bridge_v5_stabilized"
SUPPORTED_BRIDGE_SPEC_VERSIONS = (BRIDGE_SPEC_VERSION, STABILIZED_BRIDGE_SPEC_VERSION)
REGRESSORS = (
    "actual_expected_inflation_1y",
    "actual_expected_real_income_growth",
    "sce_question_unemployment_higher_prob",
)
FIT_WAVES = (202004, 202008, 202012, 202104, 202108, 202112, 202204, 202208, 202304, 202308)
INTERNAL_CHECK_WAVES = (202212,)
VALIDATION_WAVES = (202312, 202404, 202408)
BETWEEN_BOUNDS_PER_PP = {
    "actual_expected_inflation_1y": (-0.30, 0.30),
    "actual_expected_real_income_growth": (0.00, 0.50),
    "sce_question_unemployment_higher_prob": (-0.04, 0.00),
}
NOMINAL_OUTPUT_COLUMN = "expected_total_spending_growth_pct"
OUTPUT_COLUMN = "expected_real_total_spending_growth_pct"
CHART_REPRO_MAX_ABS_ERROR_TOLERANCE_PP = 0.35
RIDGE_ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
BRIDGE_ESTIMATORS = ("ols", "ridge_cv")


@dataclass(frozen=True)
class BridgeInput:
    inflation_expectation_1y: float
    expected_real_income_growth: float
    unemployment_higher_prob: float
    income_group: str
    liquid_wealth_group: str


@dataclass(frozen=True)
class BridgeTransformResult:
    annual_growth_deviation_pp: float
    clipped: dict[str, bool]
    clipped_values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and score the locked empirical bridge v4.")
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--coverage-json", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--cell-targets-csv", type=Path, default=DEFAULT_CELL_TARGETS)
    parser.add_argument("--validation-scores-csv", type=Path, default=DEFAULT_VALIDATION_SCORES)
    parser.add_argument("--estimator", choices=BRIDGE_ESTIMATORS, default="ols")
    parser.add_argument("--bridge-spec-version", default=BRIDGE_SPEC_VERSION, choices=SUPPORTED_BRIDGE_SPEC_VERSIONS)
    parser.add_argument("--validate", action="store_true", help="Score validation using an already locked fit artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolve_output_paths_for_spec(args)
    if args.validate:
        if not args.output_json.exists():
            raise SystemExit("--validate requires an existing locked empirical bridge artifact")
        artifact = json.loads(args.output_json.read_text(encoding="utf-8"))
        panel = load_panel(args.panel_csv)
        scores = score_split(panel, artifact, list(VALIDATION_WAVES), split_name="validation")
        args.validation_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scores["rows"]).to_csv(args.validation_scores_csv, index=False)
        print(json.dumps(scores["summary"], indent=2, sort_keys=True))
        return 0
    result = fit_empirical_bridge(
        panel_csv=args.panel_csv,
        coverage_json=args.coverage_json,
        output_json=args.output_json,
        cell_targets_csv=args.cell_targets_csv,
        validation_scores_csv=args.validation_scores_csv,
        estimator=args.estimator,
        bridge_spec_version=args.bridge_spec_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def resolve_output_paths_for_spec(args: argparse.Namespace) -> None:
    if args.bridge_spec_version == STABILIZED_BRIDGE_SPEC_VERSION:
        if args.output_json == DEFAULT_BRIDGE:
            args.output_json = DEFAULT_STABILIZED_BRIDGE
        if args.cell_targets_csv == DEFAULT_CELL_TARGETS:
            args.cell_targets_csv = DEFAULT_STABILIZED_CELL_TARGETS
        if args.validation_scores_csv == DEFAULT_VALIDATION_SCORES:
            args.validation_scores_csv = DEFAULT_STABILIZED_VALIDATION_SCORES


def fit_empirical_bridge(
    *,
    panel_csv: Path,
    coverage_json: Path,
    output_json: Path,
    cell_targets_csv: Path,
    validation_scores_csv: Path,
    estimator: str = "ols",
    bridge_spec_version: str = BRIDGE_SPEC_VERSION,
) -> dict[str, Any]:
    if bridge_spec_version not in SUPPORTED_BRIDGE_SPEC_VERSIONS:
        raise ValueError(f"Unsupported empirical bridge spec version: {bridge_spec_version}")
    if estimator not in BRIDGE_ESTIMATORS:
        raise ValueError(f"Unsupported empirical bridge estimator: {estimator}")
    if bridge_spec_version == BRIDGE_SPEC_VERSION and estimator != "ols":
        raise ValueError("empirical_bridge_v4 must use estimator=ols")
    if bridge_spec_version == STABILIZED_BRIDGE_SPEC_VERSION and estimator != "ridge_cv":
        raise ValueError("empirical_bridge_v5_stabilized must use estimator=ridge_cv")
    panel = load_panel(panel_csv)
    fit_frame = panel[panel["spending_wave"].isin(FIT_WAVES)].copy()
    if fit_frame.empty:
        raise ValueError("No FIT-wave rows available for empirical bridge")
    winsor_bounds = {
        "lower": weighted_quantile(fit_frame[OUTPUT_COLUMN], fit_frame["weight"], 0.02),
        "upper": weighted_quantile(fit_frame[OUTPUT_COLUMN], fit_frame["weight"], 0.98),
    }
    fit_frame["outcome_winsorized"] = fit_frame[OUTPUT_COLUMN].clip(winsor_bounds["lower"], winsor_bounds["upper"])
    ridge_selection = select_ridge_alpha_by_fit_cv(fit_frame, winsor_bounds) if estimator == "ridge_cv" else None
    fit_model = fit_mundlak_model(
        fit_frame,
        estimator=estimator,
        ridge_alpha=None if ridge_selection is None else float(ridge_selection["selected_alpha"]),
        schema_version=bridge_spec_version,
    )
    fit_scores = score_split(panel, fit_model, list(FIT_WAVES), split_name="fit", winsor_bounds=winsor_bounds)
    internal_scores = score_split(panel, fit_model, list(INTERNAL_CHECK_WAVES), split_name="internal_check", winsor_bounds=winsor_bounds)
    validation_scores = score_split(panel, fit_model, list(VALIDATION_WAVES), split_name="validation", winsor_bounds=winsor_bounds)
    validation_refit = refit_between_for_split(
        panel,
        list(VALIDATION_WAVES),
        winsor_bounds=winsor_bounds,
        estimator=estimator,
        ridge_alpha=None if ridge_selection is None else float(ridge_selection["selected_alpha"]),
        schema_version=bridge_spec_version,
    )
    constraints = constraint_report(fit_model)
    validation_gate = validation_gate_report(fit_model, validation_refit, fit_scores, validation_scores)
    liquidity_gradient = liquidity_gradient_report(fit_model)
    chart_quality = bridge_chart_quality_report(coverage_json)
    accepted = bool(constraints["passed"] and liquidity_gradient["passed"] and chart_quality["passed"])
    artifact: dict[str, Any] = {
        "schema_version": bridge_spec_version,
        "bridge_spec_version": bridge_spec_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if accepted else "rejected",
        "rejection_reason": None if accepted else "constraint_or_liquidity_gradient_failed",
        "estimator": estimator,
        "ridge_selection": ridge_selection,
        "panel_csv": str(panel_csv),
        "panel_csv_sha256": file_sha256(panel_csv),
        "fit_waves": list(FIT_WAVES),
        "internal_check_waves": list(INTERNAL_CHECK_WAVES),
        "validation_waves": list(VALIDATION_WAVES),
        "nominal_outcome_column": NOMINAL_OUTPUT_COLUMN,
        "outcome_column": OUTPUT_COLUMN,
        "outcome_transform": (
            "100 * ((1 + expected_total_spending_growth_pct / 100) / "
            "(1 + actual_expected_inflation_1y / 100) - 1)"
        ),
        "regressors": list(REGRESSORS),
        "coverage_json": str(coverage_json),
        "coverage_json_sha256": file_sha256(coverage_json) if coverage_json.exists() else None,
        "chart_quality": chart_quality,
        "winsor_bounds": winsor_bounds,
        "support": fit_model["support"],
        "coefficients": fit_model["coefficients"],
        "between_coefficients": fit_model["between_coefficients"],
        "within_cell_coefficients": fit_model["within_cell_coefficients"],
        "control_coefficients": fit_model["control_coefficients"],
        "diagnostics": fit_model["diagnostics"],
        "constraints": constraints,
        "liquidity_gradient": liquidity_gradient,
        "fit_scores": fit_scores["summary"],
        "internal_check_scores": internal_scores["summary"],
        "validation_scores": validation_scores["summary"],
        "validation_refit_between_coefficients": validation_refit["between_coefficients"],
        "validation_gate": validation_gate,
        "transform_rule": (
            "Per-period annual consumption-growth deviation equals between_coef dot clipped belief changes; "
            "the current demand executor divides the annual pp deviation by 4 before applying it to the "
            "quarterly consumption margin."
        ),
    }
    artifact["canonical_payload_sha256"] = canonical_sha256(artifact)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True), encoding="utf-8")
    cell_targets_csv.parent.mkdir(parents=True, exist_ok=True)
    build_cell_targets(panel, fit_model, winsor_bounds).to_csv(cell_targets_csv, index=False)
    pd.DataFrame(validation_scores["rows"]).to_csv(validation_scores_csv, index=False)
    if not accepted:
        write_blocker(
            f"{bridge_spec_version} fit was rejected by the locked fail-closed checks. "
            f"Constraint report: {json.dumps(constraints, sort_keys=True)}. "
            f"Liquidity-gradient report: {json.dumps(liquidity_gradient, sort_keys=True)}. "
            f"Chart-quality report: {json.dumps(chart_quality, sort_keys=True)}.",
            bridge_spec_version=bridge_spec_version,
        )
    return {
        "output_json": str(output_json),
        "status": artifact["status"],
        "canonical_payload_sha256": artifact["canonical_payload_sha256"],
        "constraints_passed": bool(constraints["passed"]),
        "chart_quality_passed": bool(chart_quality["passed"]),
        "validation_passed": bool(validation_gate["passed"]),
    }


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Spending/belief panel not found: {path}")
    frame = pd.read_csv(path)
    required = {"spending_wave", "weight", NOMINAL_OUTPUT_COLUMN, "income_group", "liquid_wealth_group", "age_group", *REGRESSORS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Empirical bridge panel missing columns: {', '.join(missing)}")
    for column in ["spending_wave", "weight", NOMINAL_OUTPUT_COLUMN, *REGRESSORS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["spending_wave", "weight", NOMINAL_OUTPUT_COLUMN, *REGRESSORS]).copy()
    frame[OUTPUT_COLUMN] = real_growth_from_nominal(
        frame[NOMINAL_OUTPUT_COLUMN],
        frame["actual_expected_inflation_1y"],
    )
    frame = frame.dropna(subset=[OUTPUT_COLUMN]).copy()
    frame["spending_wave"] = frame["spending_wave"].astype(int)
    frame["weight"] = frame["weight"].clip(lower=0.0)
    return frame[frame["weight"].gt(0)].reset_index(drop=True)


def real_growth_from_nominal(nominal_growth_pct: pd.Series, inflation_expectation_pct: pd.Series) -> pd.Series:
    nominal = pd.to_numeric(nominal_growth_pct, errors="coerce") / 100.0
    inflation = pd.to_numeric(inflation_expectation_pct, errors="coerce") / 100.0
    denominator = 1.0 + inflation
    result = 100.0 * ((1.0 + nominal) / denominator - 1.0)
    return result.where(denominator.gt(0.0))


def fit_mundlak_model(
    frame: pd.DataFrame,
    *,
    estimator: str = "ols",
    ridge_alpha: float | None = None,
    schema_version: str = BRIDGE_SPEC_VERSION,
) -> dict[str, Any]:
    frame = frame.copy()
    if estimator not in BRIDGE_ESTIMATORS:
        raise ValueError(f"Unsupported empirical bridge estimator: {estimator}")
    design_frame, frame, cells = build_mundlak_design(frame)
    y = frame["outcome_winsorized"].astype(float).to_numpy()
    weights = frame["weight"].astype(float).to_numpy()
    x = design_frame.to_numpy(dtype=float)
    if estimator == "ridge_cv":
        beta, solver_diagnostics = solve_weighted_ridge(
            x,
            y,
            weights,
            list(design_frame.columns),
            alpha=float(ridge_alpha if ridge_alpha is not None else RIDGE_ALPHA_GRID[-1]),
        )
    else:
        beta, solver_diagnostics = solve_weighted_ols(x, y, weights)
    coefficients = {name: float(value) for name, value in zip(design_frame.columns, beta)}
    predictions = x @ beta
    residual = y - predictions
    weighted_rmse = float(np.sqrt(np.average(np.square(residual), weights=weights)))
    between = {regressor: coefficients[f"between_{regressor}"] for regressor in REGRESSORS}
    within = {
        regressor: {
            cell: coefficients.get(f"within_{regressor}__{cell}", 0.0)
            for cell in cells
        }
        for regressor in REGRESSORS
    }
    controls = {
        name: value
        for name, value in coefficients.items()
        if not name.startswith("between_") and not name.startswith("within_") and name != "intercept"
    }
    support = {
        regressor: {
            "min": float(frame[regressor].min()),
            "max": float(frame[regressor].max()),
        }
        for regressor in REGRESSORS
    }
    return {
        "schema_version": schema_version,
        "estimator": estimator,
        "ridge_alpha": None if estimator != "ridge_cv" else float(ridge_alpha if ridge_alpha is not None else RIDGE_ALPHA_GRID[-1]),
        "coefficients": coefficients,
        "between_coefficients": between,
        "within_cell_coefficients": within,
        "control_coefficients": controls,
        "cells": cells,
        "support": support,
        "design_columns": list(design_frame.columns),
        "diagnostics": {
            "n": int(frame.shape[0]),
            "rank": solver_diagnostics["rank"],
            "weighted_rmse": weighted_rmse,
            "singular_values_min": solver_diagnostics["singular_values_min"],
            "singular_values_max": solver_diagnostics["singular_values_max"],
            "ridge_alpha": solver_diagnostics.get("ridge_alpha"),
            "between_vs_within_gap": {
                regressor: {
                    cell: float(between[regressor] - slope)
                    for cell, slope in within[regressor].items()
                }
                for regressor in REGRESSORS
            },
        },
    }


def build_mundlak_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = frame.copy()
    wave_means = weighted_wave_means(frame)
    frame = frame.merge(wave_means, on="spending_wave", how="left")
    cells = sorted(
        f"{liquid}__{income}"
        for liquid, income in frame[["liquid_wealth_group", "income_group"]].drop_duplicates().astype(str).itertuples(index=False, name=None)
    )
    cell_labels = frame["liquid_wealth_group"].astype(str) + "__" + frame["income_group"].astype(str)
    columns: dict[str, np.ndarray] = {"intercept": np.ones(frame.shape[0], dtype=float)}
    for regressor in REGRESSORS:
        wave_column = f"wave_mean_{regressor}"
        columns[f"between_{regressor}"] = frame[wave_column].astype(float).to_numpy()
        within = frame[regressor].astype(float).to_numpy() - frame[wave_column].astype(float).to_numpy()
        for cell in cells:
            columns[f"within_{regressor}__{cell}"] = np.where(cell_labels.eq(cell).to_numpy(), within, 0.0)
    for column in dummy_columns(frame, "income_group"):
        label = column.split("__", 1)[1]
        columns[column] = frame["income_group"].astype(str).eq(label).astype(float).to_numpy()
    for column in dummy_columns(frame, "liquid_wealth_group"):
        label = column.split("__", 1)[1]
        columns[column] = frame["liquid_wealth_group"].astype(str).eq(label).astype(float).to_numpy()
    for column in dummy_columns(frame, "age_group"):
        label = column.split("__", 1)[1]
        columns[column] = frame["age_group"].astype(str).eq(label).astype(float).to_numpy()
    return pd.DataFrame(columns), frame, cells


def solve_weighted_ols(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    sqrt_w = np.sqrt(weights)
    beta, _residuals, rank, singular = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)
    return beta, {
        "rank": int(rank),
        "singular_values_min": float(np.min(singular)) if len(singular) else None,
        "singular_values_max": float(np.max(singular)) if len(singular) else None,
        "ridge_alpha": None,
    }


def solve_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    columns: list[str],
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_work = x.copy()
    means = np.zeros(x_work.shape[1], dtype=float)
    scales = np.ones(x_work.shape[1], dtype=float)
    for index, name in enumerate(columns):
        if name == "intercept":
            continue
        mean = float(np.average(x_work[:, index], weights=weights))
        variance = float(np.average(np.square(x_work[:, index] - mean), weights=weights))
        scale = float(np.sqrt(max(variance, 0.0)))
        means[index] = mean
        scales[index] = scale if scale > 1e-12 else 1.0
        x_work[:, index] = (x_work[:, index] - means[index]) / scales[index]
    sqrt_w = np.sqrt(weights)
    weighted_x = x_work * sqrt_w[:, None]
    weighted_y = y * sqrt_w
    penalty = np.eye(x_work.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta_std = np.linalg.solve(weighted_x.T @ weighted_x + float(alpha) * penalty, weighted_x.T @ weighted_y)
    beta = beta_std / scales
    beta[0] = beta_std[0] - sum(beta_std[index] * means[index] / scales[index] for index in range(1, beta_std.shape[0]))
    singular = np.linalg.svd(weighted_x, compute_uv=False)
    return beta, {
        "rank": int(np.linalg.matrix_rank(weighted_x)),
        "singular_values_min": float(np.min(singular)) if len(singular) else None,
        "singular_values_max": float(np.max(singular)) if len(singular) else None,
        "ridge_alpha": float(alpha),
    }


def select_ridge_alpha_by_fit_cv(fit_frame: pd.DataFrame, winsor_bounds: dict[str, float]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    waves = [int(value) for value in FIT_WAVES]
    for alpha in RIDGE_ALPHA_GRID:
        fold_errors = []
        for wave in waves:
            train = fit_frame[fit_frame["spending_wave"].astype(int).ne(wave)].copy()
            holdout = fit_frame[fit_frame["spending_wave"].astype(int).eq(wave)].copy()
            if train.empty or holdout.empty:
                continue
            model = fit_mundlak_model(
                train,
                estimator="ridge_cv",
                ridge_alpha=float(alpha),
                schema_version=STABILIZED_BRIDGE_SPEC_VERSION,
            )
            score = score_split(holdout, model, [wave], split_name="fit_leave_one_wave_out", winsor_bounds=winsor_bounds)
            value = score["summary"].get("weighted_rmse")
            if value is not None and np.isfinite(float(value)):
                fold_errors.append(float(value))
        if not fold_errors:
            continue
        rows.append(
            {
                "alpha": float(alpha),
                "fold_count": int(len(fold_errors)),
                "mean_weighted_rmse": float(np.mean(fold_errors)),
                "se_weighted_rmse": float(np.std(fold_errors, ddof=1) / np.sqrt(len(fold_errors))) if len(fold_errors) > 1 else 0.0,
            }
        )
    if not rows:
        raise ValueError("Ridge alpha selection produced no valid FIT-wave folds.")
    best = min(rows, key=lambda row: row["mean_weighted_rmse"])
    threshold = float(best["mean_weighted_rmse"] + best["se_weighted_rmse"])
    eligible = [row for row in rows if row["mean_weighted_rmse"] <= threshold]
    selected = max(eligible, key=lambda row: row["alpha"])
    return {
        "selection_rule": "largest_alpha_within_one_standard_error_of_min_leave_one_fit_wave_out_rmse",
        "alpha_grid": [float(value) for value in RIDGE_ALPHA_GRID],
        "selected_alpha": float(selected["alpha"]),
        "min_alpha": float(best["alpha"]),
        "min_mean_weighted_rmse": float(best["mean_weighted_rmse"]),
        "one_se_threshold": threshold,
        "rows": rows,
    }


def dummy_columns(frame: pd.DataFrame, column: str) -> list[str]:
    values = sorted(str(value) for value in frame[column].dropna().unique())
    return [f"{column}__{value}" for value in values[1:]]


def weighted_wave_means(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, group in frame.groupby("spending_wave"):
        weights = group["weight"].astype(float)
        total = float(weights.sum())
        row = {"spending_wave": int(wave)}
        for regressor in REGRESSORS:
            row[f"wave_mean_{regressor}"] = float((group[regressor].astype(float) * weights).sum() / total)
        rows.append(row)
    return pd.DataFrame(rows)


def score_split(
    panel: pd.DataFrame,
    artifact: dict[str, Any],
    waves: list[int],
    *,
    split_name: str,
    winsor_bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    frame = panel[panel["spending_wave"].isin(waves)].copy()
    if frame.empty:
        return {"summary": {"split": split_name, "n": 0, "weighted_rmse": None, "cell_weighted_rmse": None}, "rows": []}
    bounds = winsor_bounds or artifact["winsor_bounds"]
    frame["outcome_winsorized"] = frame[OUTPUT_COLUMN].clip(bounds["lower"], bounds["upper"])
    predictions = predict_from_artifact(frame, artifact)
    frame["prediction"] = predictions
    frame["error"] = frame["prediction"] - frame["outcome_winsorized"]
    weights = frame["weight"].astype(float)
    weighted_rmse = float(np.sqrt(np.average(np.square(frame["error"]), weights=weights)))
    cell_rows = []
    for (wave, liquid, income), group in frame.groupby(["spending_wave", "liquid_wealth_group", "income_group"]):
        group_weights = group["weight"].astype(float)
        if float(group_weights.sum()) <= 0:
            continue
        cell_rows.append(
            {
                "split": split_name,
                "spending_wave": int(wave),
                "liquid_wealth_group": str(liquid),
                "income_group": str(income),
                "weight": float(group_weights.sum()),
                "target_mean": float(np.average(group["outcome_winsorized"], weights=group_weights)),
                "prediction_mean": float(np.average(group["prediction"], weights=group_weights)),
                "squared_error": float((np.average(group["prediction"], weights=group_weights) - np.average(group["outcome_winsorized"], weights=group_weights)) ** 2),
            }
        )
    cell_frame = pd.DataFrame(cell_rows)
    cell_rmse = (
        float(np.sqrt(np.average(cell_frame["squared_error"], weights=cell_frame["weight"])))
        if not cell_frame.empty
        else None
    )
    return {
        "summary": {
            "split": split_name,
            "n": int(frame.shape[0]),
            "waves": [int(value) for value in sorted(frame["spending_wave"].unique())],
            "weighted_rmse": weighted_rmse,
            "cell_weighted_rmse": cell_rmse,
        },
        "rows": cell_rows,
    }


def predict_from_artifact(frame: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    coefficients = artifact["coefficients"]
    wave_means = weighted_wave_means(frame)
    merged = frame.merge(wave_means, on="spending_wave", how="left")
    out = np.full(merged.shape[0], float(coefficients.get("intercept", 0.0)), dtype=float)
    for regressor in REGRESSORS:
        out += float(coefficients.get(f"between_{regressor}", 0.0)) * merged[f"wave_mean_{regressor}"].astype(float).to_numpy()
        for position, row in enumerate(merged.to_dict(orient="records")):
            cell = f"{row['liquid_wealth_group']}__{row['income_group']}"
            out[position] += float(coefficients.get(f"within_{regressor}__{cell}", 0.0)) * (
                float(row[regressor]) - float(row[f"wave_mean_{regressor}"])
            )
    for name, value in coefficients.items():
        if name.startswith("income_group__"):
            label = name.split("__", 1)[1]
            out += float(value) * merged["income_group"].astype(str).eq(label).astype(float).to_numpy()
        elif name.startswith("liquid_wealth_group__"):
            label = name.split("__", 1)[1]
            out += float(value) * merged["liquid_wealth_group"].astype(str).eq(label).astype(float).to_numpy()
        elif name.startswith("age_group__"):
            label = name.split("__", 1)[1]
            out += float(value) * merged["age_group"].astype(str).eq(label).astype(float).to_numpy()
    return out


def refit_between_for_split(
    panel: pd.DataFrame,
    waves: list[int],
    *,
    winsor_bounds: dict[str, float],
    estimator: str = "ols",
    ridge_alpha: float | None = None,
    schema_version: str = BRIDGE_SPEC_VERSION,
) -> dict[str, Any]:
    frame = panel[panel["spending_wave"].isin(waves)].copy()
    if frame.empty:
        return {"between_coefficients": {regressor: None for regressor in REGRESSORS}}
    frame["outcome_winsorized"] = frame[OUTPUT_COLUMN].clip(winsor_bounds["lower"], winsor_bounds["upper"])
    try:
        return fit_mundlak_model(frame, estimator=estimator, ridge_alpha=ridge_alpha, schema_version=schema_version)
    except np.linalg.LinAlgError:
        return {"between_coefficients": {regressor: None for regressor in REGRESSORS}}


def constraint_report(model: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for regressor, coefficient in model["between_coefficients"].items():
        lower, upper = BETWEEN_BOUNDS_PER_PP[regressor]
        scaled = coefficient * 10.0 if regressor == "sce_question_unemployment_higher_prob" else coefficient
        bound_lower = lower * 10.0 if regressor == "sce_question_unemployment_higher_prob" else lower
        bound_upper = upper * 10.0 if regressor == "sce_question_unemployment_higher_prob" else upper
        passed = bound_lower <= scaled <= bound_upper
        rows.append(
            {
                "regressor": regressor,
                "coefficient_per_pp": float(coefficient),
                "reported_coefficient": float(scaled),
                "bound_lower": float(bound_lower),
                "bound_upper": float(bound_upper),
                "passed": bool(passed),
            }
        )
    return {"passed": bool(all(row["passed"] for row in rows)), "rows": rows}


def liquidity_gradient_report(model: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for regressor, by_cell in model["within_cell_coefficients"].items():
        for income in sorted({cell.split("__", 1)[1] for cell in by_cell}):
            low = by_cell.get(f"low__{income}")
            high = by_cell.get(f"high__{income}")
            if low is None or high is None:
                continue
            if regressor == "sce_question_unemployment_higher_prob":
                passed = low <= high
            else:
                passed = low >= high
            rows.append({"regressor": regressor, "income_group": income, "low_liquid_slope": low, "high_liquid_slope": high, "passed": bool(passed)})
    return {"passed": bool(all(row["passed"] for row in rows)) if rows else True, "rows": rows}


def validation_gate_report(
    fit_model: dict[str, Any],
    validation_refit: dict[str, Any],
    fit_scores: dict[str, Any],
    validation_scores: dict[str, Any],
) -> dict[str, Any]:
    sign_rows = []
    validation_between = validation_refit.get("between_coefficients", {})
    for regressor, fit_value in fit_model["between_coefficients"].items():
        validation_value = validation_between.get(regressor)
        if validation_value is None or not np.isfinite(float(validation_value)):
            passed = False
            magnitude_ratio = None
        else:
            fit_sign = 0 if abs(float(fit_value)) < 1e-12 else int(np.sign(float(fit_value)))
            validation_sign = 0 if abs(float(validation_value)) < 1e-12 else int(np.sign(float(validation_value)))
            magnitude_ratio = abs(float(validation_value)) / max(abs(float(fit_value)), 1e-12)
            passed = fit_sign == validation_sign and 0.5 <= magnitude_ratio <= 2.0
        sign_rows.append(
            {
                "regressor": regressor,
                "fit_between": float(fit_value),
                "validation_between": None if validation_value is None else float(validation_value),
                "magnitude_ratio": magnitude_ratio,
                "passed": bool(passed),
            }
        )
    fit_rmse = fit_scores["summary"].get("cell_weighted_rmse")
    val_rmse = validation_scores["summary"].get("cell_weighted_rmse")
    rmse_passed = bool(fit_rmse is not None and val_rmse is not None and float(val_rmse) <= 1.5 * float(fit_rmse))
    return {
        "passed": bool(all(row["passed"] for row in sign_rows) and rmse_passed),
        "between_sign_magnitude_rows": sign_rows,
        "cell_rmse_ratio": None if fit_rmse in (None, 0) or val_rmse is None else float(val_rmse) / float(fit_rmse),
        "cell_rmse_passed": rmse_passed,
    }


def bridge_chart_quality_report(coverage_json: Path) -> dict[str, Any]:
    if not coverage_json.exists():
        return {
            "passed": False,
            "status": "missing_coverage_json",
            "coverage_json": str(coverage_json),
            "max_abs_error_pp": None,
            "tolerance_pp": CHART_REPRO_MAX_ABS_ERROR_TOLERANCE_PP,
            "note": "v4 requires the spending panel coverage report before fitting.",
        }
    try:
        coverage = json.loads(coverage_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "passed": False,
            "status": "unreadable_coverage_json",
            "coverage_json": str(coverage_json),
            "max_abs_error_pp": None,
            "tolerance_pp": CHART_REPRO_MAX_ABS_ERROR_TOLERANCE_PP,
        }
    chart = coverage.get("chart_reproduction", {}) if isinstance(coverage, dict) else {}
    rows: list[dict[str, Any]] = []
    max_errors: list[float] = []
    for series in ("expected_total_spending_growth_pct", "reported_total_spending_growth_pct"):
        series_report = chart.get(series, {}) if isinstance(chart, dict) else {}
        max_error = series_report.get("max_abs_error_pp")
        if max_error is not None and np.isfinite(float(max_error)):
            max_errors.append(float(max_error))
        rows.append(
            {
                "series": series,
                "max_abs_error_pp": None if max_error is None else float(max_error),
                "original_passed": bool(series_report.get("passed", False)),
            }
        )
    max_abs_error = max(max_errors) if max_errors else None
    passed = bool(max_abs_error is not None and max_abs_error <= CHART_REPRO_MAX_ABS_ERROR_TOLERANCE_PP)
    return {
        "passed": passed,
        "status": "passed_with_weight_substitution_warning" if passed else "failed",
        "coverage_json": str(coverage_json),
        "max_abs_error_pp": max_abs_error,
        "tolerance_pp": CHART_REPRO_MAX_ABS_ERROR_TOLERANCE_PP,
        "original_gate_tolerance_pp": chart.get("gate_tolerance_pp") if isinstance(chart, dict) else None,
        "original_gate_passed": bool(chart.get("gate_passed", False)) if isinstance(chart, dict) else False,
        "rows": rows,
        "note": (
            "v4 accepts the known public-weight substitution mismatch only if chart reproduction "
            "stays within 0.35pp; the tighter v3 0.10pp gate remains recorded as original_gate_passed."
        ),
    }


def build_cell_targets(panel: pd.DataFrame, artifact: dict[str, Any], winsor_bounds: dict[str, float]) -> pd.DataFrame:
    frames = []
    for split, waves in [("fit", FIT_WAVES), ("validation_frozen", VALIDATION_WAVES)]:
        subset = panel[panel["spending_wave"].isin(waves)].copy()
        subset["outcome_winsorized"] = subset[OUTPUT_COLUMN].clip(winsor_bounds["lower"], winsor_bounds["upper"])
        subset["prediction"] = predict_from_artifact(subset, artifact)
        for (liquid, income), group in subset.groupby(["liquid_wealth_group", "income_group"]):
            weights = group["weight"].astype(float)
            frames.append(
                {
                    "schema_version": str(artifact.get("schema_version") or artifact.get("bridge_spec_version") or BRIDGE_SPEC_VERSION),
                    "selection_surface": split,
                    "liquid_wealth_group": str(liquid),
                    "income_group": str(income),
                    "weight": float(weights.sum()),
                    "target_expected_spending_growth": float(np.average(group["outcome_winsorized"], weights=weights)),
                    "bridge_prediction": float(np.average(group["prediction"], weights=weights)),
                    "source_waves": ",".join(str(int(value)) for value in sorted(group["spending_wave"].unique())),
                    "frozen": bool(split == "validation_frozen"),
                }
            )
    return pd.DataFrame(frames)


def transform_belief_change(profile: dict[str, Any], previous: BridgeInput, current: BridgeInput) -> BridgeTransformResult:
    clipped_previous = clip_bridge_input(profile, previous)
    clipped_current = clip_bridge_input(profile, current)
    coefficients = profile["between_coefficients"]
    delta = {
        "actual_expected_inflation_1y": clipped_current["actual_expected_inflation_1y"] - clipped_previous["actual_expected_inflation_1y"],
        "actual_expected_real_income_growth": clipped_current["actual_expected_real_income_growth"] - clipped_previous["actual_expected_real_income_growth"],
        "sce_question_unemployment_higher_prob": clipped_current["sce_question_unemployment_higher_prob"] - clipped_previous["sce_question_unemployment_higher_prob"],
    }
    annual = float(sum(float(coefficients[regressor]) * value for regressor, value in delta.items()))
    clipped = {key: bool(clipped_current[f"{key}_clipped"] or clipped_previous[f"{key}_clipped"]) for key in REGRESSORS}
    return BridgeTransformResult(annual_growth_deviation_pp=annual, clipped=clipped, clipped_values={key: clipped_current[key] for key in REGRESSORS})


def clip_bridge_input(profile: dict[str, Any], value: BridgeInput) -> dict[str, float | bool]:
    raw = {
        "actual_expected_inflation_1y": float(value.inflation_expectation_1y),
        "actual_expected_real_income_growth": float(value.expected_real_income_growth),
        "sce_question_unemployment_higher_prob": float(value.unemployment_higher_prob),
    }
    out: dict[str, float | bool] = {}
    for regressor, raw_value in raw.items():
        support = profile["support"][regressor]
        clipped = float(np.clip(raw_value, float(support["min"]), float(support["max"])))
        out[regressor] = clipped
        out[f"{regressor}_clipped"] = bool(abs(clipped - raw_value) > 1e-12)
    return out


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    values_series = pd.to_numeric(values, errors="coerce")
    weights_series = pd.to_numeric(weights, errors="coerce")
    mask = values_series.notna() & weights_series.notna() & weights_series.gt(0)
    values_array = values_series[mask].to_numpy(dtype=float)
    weights_array = weights_series[mask].to_numpy(dtype=float)
    if values_array.size == 0:
        return float("nan")
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array) - 0.5 * weights_array
    cumulative /= weights_array.sum()
    return float(np.interp(q, cumulative, values_array))


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(json.dumps(_jsonable(clean), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_blocker(message: str, *, bridge_spec_version: str = BRIDGE_SPEC_VERSION) -> None:
    path = PROJECT_ROOT / "work" / "codex_briefs" / f"{bridge_spec_version}_blockers.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    title = bridge_spec_version.replace("_", " ").title()
    previous = path.read_text(encoding="utf-8") if path.exists() else f"# {title} Blockers\n\n"
    entry = f"## {datetime.now(timezone.utc).isoformat()}\n\n{message}\n\n"
    path.write_text(previous + entry, encoding="utf-8")


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
        return float(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
