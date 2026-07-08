from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import PROJECT_ROOT, markdown_table
from .empirical_bridge import (
    FIT_WAVES,
    INTERNAL_CHECK_WAVES,
    OUTPUT_COLUMN,
    REGRESSORS,
    VALIDATION_WAVES,
    file_sha256,
    fit_mundlak_model,
    load_panel,
    weighted_wave_means,
)


DIAGNOSTIC_SCHEMA_VERSION = "phase4_v4_diagnostics_v1"
DEFAULT_STRICT_DIR = PROJECT_ROOT / "outputs" / "phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_onecard"
DEFAULT_HOLDLAST_DIR = PROJECT_ROOT / "outputs" / "phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_holdlast_5cards"
DEFAULT_BRIDGE_JSON = PROJECT_ROOT / "work" / "empirical_bridge" / "empirical_bridge_v4.json"
DEFAULT_PANEL_CSV = PROJECT_ROOT / "work" / "empirical_bridge" / "spending_belief_panel.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase4_v4_diagnostics"
OUTPUT_FILES = [
    "phase4_v4_scaled_error_contributions.csv",
    "empirical_bridge_v4_wave_diagnostics.csv",
    "empirical_bridge_v4_coefficient_summary.csv",
    "phase4_v4_diagnostics_report.md",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose empirical bridge v4 Phase 4 scoring and validation instability.")
    parser.add_argument("--strict-dir", type=Path, default=DEFAULT_STRICT_DIR)
    parser.add_argument("--holdlast-dir", type=Path, default=DEFAULT_HOLDLAST_DIR)
    parser.add_argument("--bridge-json", type=Path, default=DEFAULT_BRIDGE_JSON)
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [RunSpec("strict", args.strict_dir), RunSpec("holdlast", args.holdlast_dir)]
    contribution = pd.concat([scaled_error_contribution(run) for run in runs], ignore_index=True)
    if contribution.empty:
        raise ValueError("No Phase 4 error-contribution rows were produced.")
    bridge = json.loads(args.bridge_json.read_text(encoding="utf-8"))
    panel = load_panel(args.panel_csv)
    wave_diagnostics = bridge_wave_diagnostics(panel, bridge)
    if wave_diagnostics.empty:
        raise ValueError("No empirical bridge wave diagnostics were produced.")
    coefficient_summary = bridge_coefficient_summary(wave_diagnostics, bridge)
    if coefficient_summary.empty:
        raise ValueError("No empirical bridge coefficient diagnostics were produced.")

    contribution.to_csv(output_dir / OUTPUT_FILES[0], index=False)
    wave_diagnostics.to_csv(output_dir / OUTPUT_FILES[1], index=False)
    coefficient_summary.to_csv(output_dir / OUTPUT_FILES[2], index=False)
    report = build_report(contribution, wave_diagnostics, coefficient_summary, bridge)
    (output_dir / OUTPUT_FILES[3]).write_text(report, encoding="utf-8")
    manifest = build_manifest(args, output_dir, bridge)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "status": "ok", "verdict": manifest["verdict"]}, indent=2, sort_keys=True))
    return 0


def scaled_error_contribution(run: RunSpec) -> pd.DataFrame:
    path = run.path / "phase4_proxy_joined_errors.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing joined error file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Joined error file is empty: {path}")
    sources = sorted(frame["source"].astype(str).unique())
    llm_sources = [source for source in sources if source != "adaptive"]
    if "adaptive" not in sources or len(llm_sources) != 1:
        raise ValueError(f"{path} must contain exactly one adaptive and one non-adaptive source")
    llm_source = llm_sources[0]
    rows: list[dict[str, Any]] = []
    for target_name, group in frame.groupby("target_name", dropna=False):
        adaptive = group[group["source"].astype(str).eq("adaptive")]
        llm = group[group["source"].astype(str).eq(llm_source)]
        if adaptive.empty or llm.empty:
            continue
        adaptive_scaled_sse = float(np.square(adaptive["scaled_error"].astype(float)).sum())
        llm_scaled_sse = float(np.square(llm["scaled_error"].astype(float)).sum())
        adaptive_sse = float(np.square(adaptive["error"].astype(float)).sum())
        llm_sse = float(np.square(llm["error"].astype(float)).sum())
        rows.append(
            {
                "run": run.label,
                "target_name": str(target_name),
                "n": int(adaptive.shape[0]),
                "history_scale_mean": float(adaptive["history_scale"].astype(float).mean()),
                "adaptive_scaled_sse": adaptive_scaled_sse,
                "llm_scaled_sse": llm_scaled_sse,
                "delta_scaled_sse_llm_minus_adaptive": llm_scaled_sse - adaptive_scaled_sse,
                "adaptive_unscaled_sse": adaptive_sse,
                "llm_unscaled_sse": llm_sse,
                "delta_unscaled_sse_llm_minus_adaptive": llm_sse - adaptive_sse,
                "adaptive_rmse_scaled": float(np.sqrt(np.mean(np.square(adaptive["scaled_error"].astype(float))))),
                "llm_rmse_scaled": float(np.sqrt(np.mean(np.square(llm["scaled_error"].astype(float))))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No target-level adaptive/LLM matches found in {path}")
    total_scaled = out.groupby("run")["delta_scaled_sse_llm_minus_adaptive"].transform(lambda value: float(value.abs().sum()))
    out["abs_delta_scaled_sse_share"] = out["delta_scaled_sse_llm_minus_adaptive"].abs() / total_scaled.replace(0.0, np.nan)
    return out.sort_values(["run", "delta_scaled_sse_llm_minus_adaptive"], ascending=[True, False]).reset_index(drop=True)


def bridge_wave_diagnostics(panel: pd.DataFrame, bridge: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_specs = [
        ("fit", list(FIT_WAVES)),
        ("internal_check", list(INTERNAL_CHECK_WAVES)),
        ("validation", list(VALIDATION_WAVES)),
    ]
    for split, waves in split_specs:
        subset = panel[panel["spending_wave"].isin(waves)].copy()
        if subset.empty:
            continue
        wave_means = weighted_wave_means(subset)
        outcome_rows = []
        for wave, group in subset.groupby("spending_wave"):
            weights = group["weight"].astype(float)
            outcome_rows.append(
                {
                    "spending_wave": int(wave),
                    f"wave_mean_{OUTPUT_COLUMN}": float(np.average(group[OUTPUT_COLUMN].astype(float), weights=weights)),
                    "n": int(group.shape[0]),
                    "weight_sum": float(weights.sum()),
                }
            )
        merged = wave_means.merge(pd.DataFrame(outcome_rows), on="spending_wave", how="left")
        condition = wave_design_condition_number(merged)
        subset["outcome_winsorized"] = subset[OUTPUT_COLUMN].clip(
            float(bridge["winsor_bounds"]["lower"]),
            float(bridge["winsor_bounds"]["upper"]),
        )
        between = fit_mundlak_model(subset)["between_coefficients"]
        for _, row in merged.sort_values("spending_wave").iterrows():
            out = {
                "split": split,
                "spending_wave": int(row["spending_wave"]),
                "n": int(row["n"]),
                "weight_sum": float(row["weight_sum"]),
                "design_condition_number": condition,
                f"wave_mean_{OUTPUT_COLUMN}": float(row[f"wave_mean_{OUTPUT_COLUMN}"]),
            }
            for regressor in REGRESSORS:
                out[f"wave_mean_{regressor}"] = float(row[f"wave_mean_{regressor}"])
                out[f"split_between_{regressor}"] = None if pd.isna(between[regressor]) else float(between[regressor])
            rows.append(out)
    return pd.DataFrame(rows)


def wave_design_condition_number(wave_means: pd.DataFrame) -> float | None:
    if wave_means.shape[0] < 2:
        return None
    matrix = wave_means[[f"wave_mean_{regressor}" for regressor in REGRESSORS]].astype(float).to_numpy()
    matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
    return float(np.linalg.cond(matrix))


def bridge_coefficient_summary(wave_diagnostics: pd.DataFrame, bridge: dict[str, Any]) -> pd.DataFrame:
    if wave_diagnostics.empty or "split" not in wave_diagnostics:
        raise ValueError("Wave diagnostics must contain split rows before coefficient summary.")
    rows: list[dict[str, Any]] = []
    for split, group in wave_diagnostics.groupby("split", dropna=False):
        for regressor in REGRESSORS:
            values = group[f"wave_mean_{regressor}"].astype(float)
            outcome = group[f"wave_mean_{OUTPUT_COLUMN}"].astype(float)
            if group.shape[0] >= 2:
                corr = float(values.corr(outcome))
                wave_range = float(values.max() - values.min())
                outcome_range = float(outcome.max() - outcome.min())
            else:
                corr = None
                wave_range = None
                outcome_range = None
            coefficient = group[f"split_between_{regressor}"].dropna()
            fit_value = float(bridge["between_coefficients"][regressor])
            split_value = None if coefficient.empty else float(coefficient.iloc[0])
            rows.append(
                {
                    "split": str(split),
                    "regressor": regressor,
                    "bridge_fit_between_coefficient": fit_value,
                    "split_refit_between_coefficient": split_value,
                    "magnitude_ratio_to_fit": None if split_value is None or abs(fit_value) < 1e-12 else abs(split_value) / abs(fit_value),
                    "wave_mean_range": wave_range,
                    "outcome_mean_range": outcome_range,
                    "wave_mean_correlation_with_outcome": corr,
                    "design_condition_number": group["design_condition_number"].dropna().iloc[0]
                    if group["design_condition_number"].notna().any()
                    else None,
                }
            )
    return pd.DataFrame(rows)


def build_manifest(args: argparse.Namespace, output_dir: Path, bridge: dict[str, Any]) -> dict[str, Any]:
    strict_joined = args.strict_dir / "phase4_proxy_joined_errors.csv"
    holdlast_joined = args.holdlast_dir / "phase4_proxy_joined_errors.csv"
    strict_manifest = args.strict_dir / "manifest.json"
    holdlast_manifest = args.holdlast_dir / "manifest.json"
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "ok",
        "passed": True,
        "verdict": "phase4_v4_diagnostics_complete",
        "claim_scope": (
            "Retrospective diagnostic decomposition only; does not consume a confirmatory score surface "
            "and does not change the locked Phase 4 v2 mapping."
        ),
        "bridge": {
            "path": _project_path(args.bridge_json),
            "sha256": file_sha256(args.bridge_json),
            "schema_version": bridge.get("schema_version"),
            "bridge_spec_version": bridge.get("bridge_spec_version"),
            "status": bridge.get("status"),
            "canonical_payload_sha256": bridge.get("canonical_payload_sha256"),
        },
        "inputs": {
            "strict_manifest": _file_manifest(strict_manifest),
            "strict_joined_errors": _file_manifest(strict_joined),
            "holdlast_manifest": _file_manifest(holdlast_manifest),
            "holdlast_joined_errors": _file_manifest(holdlast_joined),
            "panel_csv": _file_manifest(args.panel_csv),
        },
        "outputs": {
            name: {
                "path": _project_path(output_dir / name),
                "sha256": file_sha256(output_dir / name),
            }
            for name in OUTPUT_FILES
        },
    }


def _file_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnostic input: {path}")
    return {
        "path": _project_path(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_report(
    contribution: pd.DataFrame,
    wave_diagnostics: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    bridge: dict[str, Any],
) -> str:
    strict = contribution[contribution["run"].eq("strict")].copy()
    holdlast = contribution[contribution["run"].eq("holdlast")].copy()
    validation_income = coefficient_summary[
        coefficient_summary["split"].eq("validation")
        & coefficient_summary["regressor"].eq("actual_expected_real_income_growth")
    ]
    validation_income_row = validation_income.iloc[0].to_dict() if not validation_income.empty else {}
    lines = [
        "# Phase 4 v4 Diagnostics",
        "",
        "## Bottom Line",
        "",
        (
            "The v4 empirical bridge fixes most of the prior over-contraction, but the remaining scaled-score "
            "gap is mechanical and target-specific. In the strict run, the LLM path wins on unscaled RMSE, "
            "but scaled RMSE gives more weight to PCE and real-PCE misses than to the saving-rate improvement."
        ),
        "",
        (
            "The validation real-income coefficient instability is not a parser bug. It comes from a three-wave "
            "validation refit with tiny between-wave movement in expected real income growth and near-perfect "
            "collinearity with validation-wave real spending means."
        ),
        "",
        "## Strict Scaled-Error Contributions",
        "",
        markdown_table(
            strict[
                [
                    "target_name",
                    "n",
                    "history_scale_mean",
                    "delta_scaled_sse_llm_minus_adaptive",
                    "delta_unscaled_sse_llm_minus_adaptive",
                    "abs_delta_scaled_sse_share",
                ]
            ]
            if not strict.empty
            else strict
        ),
        "",
        "## Hold-Last Scaled-Error Contributions",
        "",
        markdown_table(
            holdlast[
                [
                    "target_name",
                    "n",
                    "history_scale_mean",
                    "delta_scaled_sse_llm_minus_adaptive",
                    "delta_unscaled_sse_llm_minus_adaptive",
                    "abs_delta_scaled_sse_share",
                ]
            ]
            if not holdlast.empty
            else holdlast
        ),
        "",
        "## Bridge Coefficient Stability",
        "",
        markdown_table(
            coefficient_summary[
                [
                    "split",
                    "regressor",
                    "bridge_fit_between_coefficient",
                    "split_refit_between_coefficient",
                    "magnitude_ratio_to_fit",
                    "wave_mean_range",
                    "outcome_mean_range",
                    "wave_mean_correlation_with_outcome",
                    "design_condition_number",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        (
            f"v4 bridge status: `{bridge.get('status')}`. The fit between coefficient for expected real income "
            f"growth is `{bridge['between_coefficients']['actual_expected_real_income_growth']:.4f}`. In validation, "
            f"the refit coefficient is `{validation_income_row.get('split_refit_between_coefficient', float('nan')):.4f}` "
            f"with magnitude ratio `{validation_income_row.get('magnitude_ratio_to_fit', float('nan')):.2f}`."
        ),
        "",
        (
            "The practical next step is not another blind refit. It is to either pre-register a more stable bridge "
            "estimator for weak between-wave variation, or collect/use a longer same-horizon panel before treating "
            "the bridge as confirmatory."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
