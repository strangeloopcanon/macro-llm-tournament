from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, markdown_table
from .behavior_gate import behavior_target_catalog
from .macro_validity import (
    build_demand_irf_paths,
    load_demand_run,
    score_demand_irf_shape,
    score_micro_behavior_gate,
    score_vintage_oos_readiness,
)


MACRO_PERFORMANCE_VERSION = "macro_simulation_performance_gate_v1"
DEFAULT_DEMAND_RUN_DIR = OUTPUT_ROOT / "demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5"
DEFAULT_VINTAGE_PANEL_DIR = WORK_ROOT / "fred_vintage_panel"
DEFAULT_VINTAGE_OOS_DIR = OUTPUT_ROOT / "demand_vintage_oos_fixture"
DEFAULT_TARGET_CATALOG = Path(__file__).resolve().parent / "data" / "macro_performance_targets.csv"
ALLOWED_SOURCE_STATUSES = {"verified_public", "internal_mechanism", "empirical_shape", "vintage_oos"}
DETERMINISTIC_BASELINE_VARIANTS = {"representative", "adaptive"}
VINTAGE_OOS_BASELINE_VARIANTS = {"no_change", "rolling_mean", "rolling_trend"}
LLM_VARIANT = "llm_belief"
PERFORMANCE_MODES = ("fixture", "replay", "live")

REQUIRED_TARGET_COLUMNS = {
    "target_id",
    "split",
    "family",
    "scenario_id",
    "variant_scope",
    "metric",
    "extractor",
    "prediction_column",
    "target_low",
    "target_high",
    "target_value",
    "score_direction",
    "weight",
    "blocking",
    "critical",
    "source_label",
    "source_url",
    "source_status",
    "scored",
    "target_scope",
    "type_id",
    "baseline_scenario_id",
    "window_start",
    "window_end",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score lab and OOS performance for the macro simulation.")
    parser.add_argument("--demand-run-dir", default=str(DEFAULT_DEMAND_RUN_DIR))
    parser.add_argument("--vintage-panel-dir", default=str(DEFAULT_VINTAGE_PANEL_DIR))
    parser.add_argument("--vintage-oos-dir", default=str(DEFAULT_VINTAGE_OOS_DIR))
    parser.add_argument("--target-catalog", default=str(DEFAULT_TARGET_CATALOG))
    parser.add_argument("--mode", choices=PERFORMANCE_MODES, default="fixture")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "macro_performance_gate"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "live":
        result = _live_blocked_manifest(args)
        write_macro_performance_outputs(result, Path(args.output_dir))
        print(f"Wrote macro performance gate to {args.output_dir}")
        print(json.dumps({"verdict": result["manifest"]["verdict"], "passed": False}, indent=2, sort_keys=True))
        return 1
    result = build_macro_performance_gate(
        demand_run_dir=Path(args.demand_run_dir),
        vintage_panel_dir=Path(args.vintage_panel_dir),
        vintage_oos_dir=Path(args.vintage_oos_dir),
        target_catalog_path=Path(args.target_catalog),
        mode=args.mode,
    )
    write_macro_performance_outputs(result, Path(args.output_dir))
    print(f"Wrote macro performance gate to {args.output_dir}")
    print(json.dumps({"verdict": result["manifest"]["verdict"], "passed": result["manifest"]["passed"]}, indent=2, sort_keys=True))
    return 0 if result["manifest"]["passed"] else 1


def load_performance_target_catalog(path: Path = DEFAULT_TARGET_CATALOG) -> pd.DataFrame:
    catalog = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_TARGET_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError(f"Macro performance target catalog missing required columns: {', '.join(sorted(missing))}")
    out = catalog.copy()
    for column in ["target_low", "target_high", "target_value", "weight", "window_start", "window_end"]:
        out[column] = out[column].map(_coerce_catalog_number)
    for column in ["blocking", "critical", "scored"]:
        out[column] = out[column].map(_coerce_bool)
    if out["target_id"].duplicated().any():
        duplicated = sorted(out.loc[out["target_id"].duplicated(), "target_id"].astype(str).unique().tolist())
        raise ValueError(f"Macro performance target_id values must be unique: {', '.join(duplicated)}")
    return out


def catalog_sha256(catalog: pd.DataFrame) -> str:
    records = _jsonable(catalog.where(pd.notna(catalog), None).to_dict(orient="records"))
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_macro_performance_gate(
    *,
    demand_run_dir: Path,
    vintage_panel_dir: Path,
    vintage_oos_dir: Path,
    target_catalog_path: Path = DEFAULT_TARGET_CATALOG,
    mode: str = "fixture",
) -> dict[str, Any]:
    if mode not in PERFORMANCE_MODES:
        raise ValueError(f"Unsupported macro performance mode: {mode}")
    if mode == "live":
        raise ValueError("Macro performance live mode is blocked; use fixture or replay.")
    artifacts = load_demand_run(demand_run_dir)
    catalog = load_performance_target_catalog(target_catalog_path)
    catalog_hash = catalog_sha256(catalog)
    score_inputs = build_score_input_frames(artifacts, vintage_oos_dir=vintage_oos_dir)
    scores = score_performance_targets(catalog, score_inputs)
    variant_summary = summarize_variant_performance(scores)
    attribution = build_performance_attribution(variant_summary, scores=scores)
    oos_pairwise = build_oos_pairwise_comparison(vintage_oos_dir)
    vintage_readiness = build_performance_vintage_readiness(vintage_panel_dir, vintage_oos_dir)
    manifest = build_macro_performance_manifest(
        artifacts_manifest=artifacts.manifest,
        demand_run_dir=demand_run_dir,
        vintage_panel_dir=vintage_panel_dir,
        vintage_oos_dir=vintage_oos_dir,
        target_catalog_path=target_catalog_path,
        target_catalog_hash=catalog_hash,
        mode=mode,
        scores=scores,
        variant_summary=variant_summary,
        attribution=attribution,
        oos_pairwise=oos_pairwise,
        vintage_readiness=vintage_readiness,
    )
    report = build_macro_performance_report(manifest, variant_summary, attribution, oos_pairwise, scores, vintage_readiness)
    return {
        "manifest": manifest,
        "target_catalog": catalog,
        "scores": scores,
        "variant_summary": variant_summary,
        "attribution": attribution,
        "oos_pairwise": oos_pairwise,
        "vintage_readiness": vintage_readiness,
        "report": report,
    }


def build_score_input_frames(artifacts: Any, *, vintage_oos_dir: Path) -> dict[str, pd.DataFrame]:
    validation = artifacts.validation.copy()
    if not validation.empty and "status" not in validation:
        validation["status"] = validation["passed"].map(lambda value: "pass" if bool(value) else "fail")
    irfs = build_demand_irf_paths(artifacts.periods)
    expected_sources = sorted(validation["source"].dropna().astype(str).unique().tolist()) if "source" in validation else []
    irf_scores = score_demand_irf_shape(irfs, validation, expected_sources=expected_sources)
    micro_scores = _build_all_micro_scores(validation, artifacts.decisions)
    vintage_scores = _load_vintage_oos_summary(vintage_oos_dir)
    return {
        "validation_metric": validation,
        "irf_score_metric": irf_scores,
        "micro_metric": micro_scores,
        "vintage_oos_summary_metric": vintage_scores,
    }


def build_performance_vintage_readiness(vintage_panel_dir: Path, vintage_oos_dir: Path) -> pd.DataFrame:
    readiness = score_vintage_oos_readiness(vintage_panel_dir).copy()
    if readiness.empty:
        return readiness
    runner_files = [
        "demand_vintage_oos_cards.csv",
        "demand_vintage_oos_targets.csv",
        "demand_vintage_oos_scores.csv",
    ]
    for filename in runner_files:
        metric = filename.replace(".csv", "_available")
        exists = (vintage_oos_dir / filename).exists()
        mask = readiness["metric"].astype(str) == metric
        if not mask.any():
            continue
        readiness.loc[mask, "source"] = str(vintage_oos_dir)
        readiness.loc[mask, "value"] = 1.0 if exists else np.nan
        readiness.loc[mask, "target_low"] = 1.0
        readiness.loc[mask, "target_high"] = 1.0
        readiness.loc[mask, "status"] = "pass" if exists else "gap"
        readiness.loc[mask, "passed"] = bool(exists)
        readiness.loc[mask, "target_kind"] = "scored_oos_artifact"
        readiness.loc[mask, "interpretation"] = (
            f"Scored date-free demand-vintage OOS artifact is present: {filename}."
            if exists
            else f"Scored date-free demand-vintage OOS artifact is not present yet: {filename}."
        )
    return readiness


def score_performance_targets(catalog: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_catalog = _source_variant_catalog(inputs)
    for _, target in catalog.iterrows():
        if not bool(target["scored"]) or str(target["source_status"]) not in ALLOWED_SOURCE_STATUSES:
            rows.append(_target_gap_row(target, source="", variant="", status="excluded", interpretation="Target row excluded by scored/source_status policy."))
            continue
        extractor = str(target["extractor"])
        frame = inputs.get(extractor, pd.DataFrame())
        if extractor == "vintage_oos_summary_metric":
            target_sources = _vintage_sources(frame)
        else:
            target_sources = _sources_for_scope(source_catalog, str(target["variant_scope"]))
        if not target_sources:
            rows.append(_target_gap_row(target, source="", variant=str(target["variant_scope"]), status="gap", interpretation="No source matched target variant_scope."))
            continue
        for source, variant in target_sources:
            value = _extract_metric_value(frame, source=source, variant=variant, metric=str(target["metric"]))
            rows.append(_score_target_value(target, source=source, variant=variant, value=value))
    return pd.DataFrame(rows).sort_values(["split", "variant", "source", "family", "target_id"]).reset_index(drop=True)


def summarize_variant_performance(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = scores[scores["status"].isin(["pass", "fail", "gap"])].groupby(["source", "variant", "split"], dropna=False, sort=True)
    for (source, variant, split), group in grouped:
        scored = group[group["status"].isin(["pass", "fail"])].copy()
        weights = pd.to_numeric(scored["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        losses = pd.to_numeric(scored["normalized_loss"], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
        denominator = float(weights.sum())
        weighted_loss = float((weights * losses).sum() / denominator) if denominator > 0 else np.nan
        blocking = group["blocking"].astype(bool)
        critical = group["critical"].astype(bool)
        rows.append(
            {
                "source": source,
                "variant": variant,
                "split": split,
                "target_count": int(group.shape[0]),
                "scored_count": int(scored.shape[0]),
                "passed_count": int((scored["status"] == "pass").sum()),
                "failed_count": int((scored["status"] == "fail").sum()),
                "gap_count": int((group["status"] == "gap").sum()),
                "blocking_fail_count": int(((group["status"] == "fail") & blocking).sum()),
                "blocking_gap_count": int(((group["status"] == "gap") & blocking).sum()),
                "critical_fail_count": int(((group["status"] == "fail") & critical).sum()),
                "weighted_normalized_loss": weighted_loss,
                "macro_performance_score": 1.0 - weighted_loss if np.isfinite(weighted_loss) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "variant", "source"]).reset_index(drop=True)


def build_performance_attribution(summary: pd.DataFrame, *, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for split, split_rows in summary.groupby("split", sort=True):
        baseline_variants = VINTAGE_OOS_BASELINE_VARIANTS if str(split) == "oos" else DETERMINISTIC_BASELINE_VARIANTS
        attribution_rows = _oos_raw_attribution_rows(scores, split=str(split), baseline_variants=baseline_variants)
        if attribution_rows is not None:
            rows.append(attribution_rows)
            continue
        llm = split_rows[split_rows["variant"].astype(str) == LLM_VARIANT]
        baselines = split_rows[split_rows["variant"].astype(str).isin(baseline_variants)]
        if llm.empty or baselines.empty:
            continue
        llm_row = llm.sort_values(["weighted_normalized_loss", "source"]).iloc[0]
        baseline_row = baselines.sort_values(["weighted_normalized_loss", "source"]).iloc[0]
        llm_loss = float(llm_row["weighted_normalized_loss"])
        baseline_loss = float(baseline_row["weighted_normalized_loss"])
        improvement = (baseline_loss - llm_loss) / baseline_loss if np.isfinite(baseline_loss) and baseline_loss > 0 else np.nan
        rows.append(
            {
                "split": split,
                "llm_source": llm_row["source"],
                "best_baseline_source": baseline_row["source"],
                "best_baseline_variant": baseline_row["variant"],
                "llm_score": llm_row["macro_performance_score"],
                "best_baseline_score": baseline_row["macro_performance_score"],
                "llm_weighted_loss": llm_loss,
                "best_baseline_weighted_loss": baseline_loss,
                "loss_improvement_pct": 100.0 * improvement if np.isfinite(improvement) else np.nan,
                "interpretation": "Positive means the LLM belief variant lowers weighted normalized loss versus the strongest deterministic baseline.",
            }
        )
    return pd.DataFrame(rows)


def _oos_raw_attribution_rows(
    scores: pd.DataFrame | None,
    *,
    split: str,
    baseline_variants: set[str],
) -> dict[str, Any] | None:
    if split != "oos" or scores is None or scores.empty:
        return None
    rows = scores[
        (scores["split"].astype(str) == "oos")
        & (scores["metric"].astype(str) == "weighted_normalized_abs_error")
        & (scores["status"].astype(str).isin({"pass", "fail"}))
    ].copy()
    if rows.empty:
        return None
    llm = rows[rows["variant"].astype(str) == LLM_VARIANT].copy()
    baselines = rows[rows["variant"].astype(str).isin(baseline_variants)].copy()
    if llm.empty or baselines.empty:
        return None
    llm["raw_loss"] = pd.to_numeric(llm["value"], errors="coerce")
    baselines["raw_loss"] = pd.to_numeric(baselines["value"], errors="coerce")
    llm = llm[np.isfinite(llm["raw_loss"])]
    baselines = baselines[np.isfinite(baselines["raw_loss"])]
    if llm.empty or baselines.empty:
        return None
    llm_row = llm.sort_values(["raw_loss", "source"]).iloc[0]
    baseline_row = baselines.sort_values(["raw_loss", "source"]).iloc[0]
    llm_loss = float(llm_row["raw_loss"])
    baseline_loss = float(baseline_row["raw_loss"])
    improvement = (baseline_loss - llm_loss) / baseline_loss if baseline_loss > 0 else np.nan
    return {
        "split": split,
        "llm_source": llm_row["source"],
        "best_baseline_source": baseline_row["source"],
        "best_baseline_variant": baseline_row["variant"],
        "llm_score": float(1.0 - float(llm_row["normalized_loss"])),
        "best_baseline_score": float(1.0 - float(baseline_row["normalized_loss"])),
        "llm_weighted_loss": llm_loss,
        "best_baseline_weighted_loss": baseline_loss,
        "loss_improvement_pct": 100.0 * improvement if np.isfinite(improvement) else np.nan,
        "interpretation": "Positive means the LLM belief variant lowers raw vintage OOS weighted normalized absolute error versus the strongest deterministic baseline.",
    }


def build_oos_pairwise_comparison(vintage_oos_dir: Path, *, bootstrap_samples: int = 10000, seed: int = 20260630) -> pd.DataFrame:
    joined_path = vintage_oos_dir / "demand_vintage_oos_joined_errors.csv"
    if not joined_path.exists():
        return pd.DataFrame()
    joined = pd.read_csv(joined_path)
    required = {"source", "variant", "origin_id", "card_id", "normalized_abs_error"}
    if joined.empty or not required.issubset(set(joined.columns)):
        return pd.DataFrame()
    joined = joined.copy()
    joined["normalized_abs_error"] = pd.to_numeric(joined["normalized_abs_error"], errors="coerce")
    joined = joined.dropna(subset=["normalized_abs_error", "origin_id", "card_id"])
    llm_sources = sorted(joined.loc[joined["variant"].astype(str) == LLM_VARIANT, "source"].astype(str).unique().tolist())
    baselines = joined[joined["variant"].astype(str).isin(VINTAGE_OOS_BASELINE_VARIANTS)].copy()
    if not llm_sources or baselines.empty:
        return pd.DataFrame()
    baseline_losses = (
        baselines.groupby(["source", "variant"], as_index=False)["normalized_abs_error"]
        .mean()
        .sort_values(["normalized_abs_error", "source"])
    )
    if baseline_losses.empty:
        return pd.DataFrame()
    best_baseline_source = str(baseline_losses.iloc[0]["source"])
    best_baseline_variant = str(baseline_losses.iloc[0]["variant"])
    best_baseline = baselines[baselines["source"].astype(str) == best_baseline_source][
        ["card_id", "origin_id", "normalized_abs_error"]
    ].rename(columns={"normalized_abs_error": "baseline_loss"})
    rows: list[dict[str, Any]] = []
    for llm_source in llm_sources:
        llm = joined[joined["source"].astype(str) == llm_source][["card_id", "origin_id", "normalized_abs_error"]].rename(
            columns={"normalized_abs_error": "llm_loss"}
        )
        paired = llm.merge(best_baseline, on=["card_id", "origin_id"], how="inner")
        if paired.empty:
            continue
        origin_diffs = (
            paired.assign(loss_reduction=paired["baseline_loss"] - paired["llm_loss"])
            .groupby("origin_id", as_index=False)
            .agg(
                llm_loss=("llm_loss", "mean"),
                baseline_loss=("baseline_loss", "mean"),
                loss_reduction=("loss_reduction", "mean"),
            )
        )
        diffs = pd.to_numeric(origin_diffs["loss_reduction"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(diffs) == 0:
            continue
        boot = _bootstrap_mean(diffs, samples=bootstrap_samples, seed=seed)
        mean_llm_loss = float(origin_diffs["llm_loss"].mean())
        mean_baseline_loss = float(origin_diffs["baseline_loss"].mean())
        mean_reduction = float(np.mean(diffs))
        rows.append(
            {
                "llm_source": llm_source,
                "best_baseline_source": best_baseline_source,
                "best_baseline_variant": best_baseline_variant,
                "n_clusters": int(len(diffs)),
                "mean_llm_loss": mean_llm_loss,
                "mean_baseline_loss": mean_baseline_loss,
                "mean_loss_reduction": mean_reduction,
                "improvement_pct": 100.0 * mean_reduction / mean_baseline_loss if mean_baseline_loss > 0 else np.nan,
                "bootstrap_mean_ci_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
                "bootstrap_mean_ci_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
                "bootstrap_share_positive": float(np.mean(boot > 0.0)) if len(boot) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_loss_reduction", "llm_source"], ascending=[False, True]).reset_index(drop=True)


def _bootstrap_mean(values: np.ndarray, *, samples: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(max(1, int(samples)), values.size), replace=True)
    return draws.mean(axis=1)


def build_macro_performance_manifest(
    *,
    artifacts_manifest: dict[str, Any],
    demand_run_dir: Path,
    vintage_panel_dir: Path,
    vintage_oos_dir: Path,
    target_catalog_path: Path,
    target_catalog_hash: str,
    mode: str,
    scores: pd.DataFrame,
    variant_summary: pd.DataFrame,
    attribution: pd.DataFrame,
    oos_pairwise: pd.DataFrame,
    vintage_readiness: pd.DataFrame,
) -> dict[str, Any]:
    oos_available = _oos_scores_available(vintage_oos_dir)
    vintage_oos_provenance = _vintage_oos_provenance(vintage_oos_dir)
    oos_empirical_eligible = _vintage_oos_empirical_eligible(vintage_oos_provenance)
    oos_improvement_pct = _oos_loss_improvement_pct(attribution)
    verdict = macro_performance_verdict(
        scores,
        variant_summary,
        attribution,
        mode=mode,
        oos_empirical_eligible=oos_empirical_eligible,
    )
    return {
        "schema_version": MACRO_PERFORMANCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "ok",
        "verdict": verdict,
        "passed": verdict in {"macro_lab_performance_ready", "macro_empirical_oos_ready"},
        "empirical_ready": verdict == "macro_empirical_oos_ready",
        "demand_run_dir": str(demand_run_dir),
        "demand_run_verdict": artifacts_manifest.get("verdict"),
        "vintage_panel_dir": str(vintage_panel_dir),
        "vintage_oos_dir": str(vintage_oos_dir),
        "vintage_oos_scores_available": bool(oos_available),
        "vintage_oos_mode": vintage_oos_provenance.get("mode"),
        "vintage_oos_forecast_mode": vintage_oos_provenance.get("forecast_mode"),
        "vintage_oos_verdict": vintage_oos_provenance.get("verdict"),
        "vintage_oos_artifact_verdict": vintage_oos_provenance.get("artifact_verdict"),
        "vintage_oos_empirical_eligible": bool(oos_empirical_eligible),
        "vintage_oos_llm_baseline_improvement_pct": oos_improvement_pct,
        "target_catalog_path": str(target_catalog_path),
        "target_catalog_sha256": target_catalog_hash,
        "target_rows": int(scores.shape[0]),
        "variant_summary_rows": int(variant_summary.shape[0]),
        "oos_pairwise_rows": int(oos_pairwise.shape[0]),
        "oos_pairwise_best_bootstrap_share_positive": _best_oos_pairwise_share_positive(oos_pairwise),
        "vintage_readiness_status": _scorecard_status(vintage_readiness),
        "outputs": [
            "macro_performance_target_catalog.csv",
            "macro_performance_scores.csv",
            "macro_performance_variant_summary.csv",
            "macro_performance_attribution.csv",
            "macro_performance_oos_pairwise.csv",
            "macro_performance_vintage_readiness.csv",
            "macro_performance_report.md",
            "manifest.json",
        ],
    }


def macro_performance_verdict(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    attribution: pd.DataFrame | None = None,
    *,
    mode: str,
    oos_empirical_eligible: bool = False,
) -> str:
    if scores.empty or summary.empty:
        return "macro_lab_performance_needs_work"
    lab_llm = summary[(summary["split"].astype(str) == "lab") & (summary["variant"].astype(str) == LLM_VARIANT)]
    if lab_llm.empty:
        return "macro_lab_performance_needs_work"
    lab_ready = bool(
        (lab_llm["blocking_fail_count"].astype(int) == 0).all()
        and (lab_llm["blocking_gap_count"].astype(int) == 0).all()
        and (lab_llm["critical_fail_count"].astype(int) == 0).all()
        and (lab_llm["scored_count"].astype(int) > 0).all()
    )
    if not lab_ready:
        return "macro_lab_performance_needs_work"
    oos_llm = summary[(summary["split"].astype(str) == "oos") & (summary["variant"].astype(str) == LLM_VARIANT)]
    oos_best_ready = False
    if not oos_llm.empty:
        best_oos_llm = oos_llm.sort_values(["weighted_normalized_loss", "source"], na_position="last").iloc[0]
        oos_best_ready = bool(
            int(best_oos_llm["blocking_fail_count"]) == 0
            and int(best_oos_llm["blocking_gap_count"]) == 0
            and int(best_oos_llm["scored_count"]) > 0
        )
    if (
        mode != "fixture"
        and oos_empirical_eligible
        and oos_best_ready
        and _oos_llm_beats_baseline(attribution)
    ):
        return "macro_empirical_oos_ready"
    return "macro_lab_performance_ready"


def build_macro_performance_report(
    manifest: dict[str, Any],
    variant_summary: pd.DataFrame,
    attribution: pd.DataFrame,
    oos_pairwise: pd.DataFrame,
    scores: pd.DataFrame,
    vintage_readiness: pd.DataFrame,
) -> str:
    failed = scores[scores["status"].isin(["fail", "gap"]) & scores["blocking"].astype(bool)] if not scores.empty else pd.DataFrame()
    lines = [
        "# Macro Simulation Performance Gate",
        "",
        "## Bottom Line",
        _bottom_line(manifest, failed),
        "",
        "## Variant Summary",
        markdown_table(variant_summary),
        "",
        "## Baseline Comparison",
        markdown_table(attribution),
        "",
        "## OOS Paired Comparison",
        markdown_table(oos_pairwise),
        "",
        "## Blocking Target Misses",
        markdown_table(failed.head(40) if not failed.empty else failed),
        "",
        "## Vintage OOS Readiness",
        markdown_table(vintage_readiness),
        "",
        "## Target Scores",
        markdown_table(scores.head(80)),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_macro_performance_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.get("target_catalog", pd.DataFrame()).to_csv(output_dir / "macro_performance_target_catalog.csv", index=False)
    result.get("scores", pd.DataFrame()).to_csv(output_dir / "macro_performance_scores.csv", index=False)
    result.get("variant_summary", pd.DataFrame()).to_csv(output_dir / "macro_performance_variant_summary.csv", index=False)
    result.get("attribution", pd.DataFrame()).to_csv(output_dir / "macro_performance_attribution.csv", index=False)
    result.get("oos_pairwise", pd.DataFrame()).to_csv(output_dir / "macro_performance_oos_pairwise.csv", index=False)
    result.get("vintage_readiness", pd.DataFrame()).to_csv(output_dir / "macro_performance_vintage_readiness.csv", index=False)
    (output_dir / "macro_performance_report.md").write_text(result.get("report", ""), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(result["manifest"]), indent=2, sort_keys=True), encoding="utf-8")


def _build_all_micro_scores(validation: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows = []
    targets = behavior_target_catalog(include_unscored=True)
    for source, group in validation.groupby("source", sort=True):
        variant = str(group["variant"].iloc[0]) if "variant" in group and not group.empty else _variant_from_source(source)
        rows.append(score_micro_behavior_gate(validation, decisions, targets, source=str(source), variant=variant))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _source_variant_catalog(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in inputs.items():
        if name == "vintage_oos_summary_metric":
            continue
        if frame.empty or "source" not in frame:
            continue
        for source, group in frame.groupby("source", sort=True):
            if str(source) == "":
                continue
            variant = str(group["variant"].iloc[0]) if "variant" in group and not group.empty else _variant_from_source(source)
            rows.append({"source": str(source), "variant": variant})
    if not rows:
        return pd.DataFrame(columns=["source", "variant"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["variant", "source"]).reset_index(drop=True)


def _sources_for_scope(source_catalog: pd.DataFrame, scope: str) -> list[tuple[str, str]]:
    if source_catalog.empty:
        return []
    if scope == "all":
        frame = source_catalog
    else:
        frame = source_catalog[source_catalog["variant"].astype(str) == scope]
    return [(str(row["source"]), str(row["variant"])) for _, row in frame.iterrows()]


def _vintage_sources(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if frame.empty:
        return []
    return [(str(row["source"]), str(row.get("variant", "vintage_oos"))) for _, row in frame[["source", "variant"]].drop_duplicates().iterrows()]


def _extract_metric_value(frame: pd.DataFrame, *, source: str, variant: str, metric: str) -> float:
    if frame.empty:
        return np.nan
    rows = frame[(frame["source"].astype(str) == source) & (frame["metric"].astype(str) == metric)]
    if "variant" in frame and rows.empty:
        rows = frame[(frame["variant"].astype(str) == variant) & (frame["metric"].astype(str) == metric)]
    if rows.empty:
        return np.nan
    return _finite_or_nan(rows["value"].iloc[0])


def _score_target_value(target: pd.Series, *, source: str, variant: str, value: float) -> dict[str, Any]:
    low = _finite_or_inf(target["target_low"])
    high = _finite_or_inf(target["target_high"])
    direction = str(target["score_direction"])
    if not np.isfinite(value):
        passed = False
        status = "gap"
        loss = 1.0
    elif direction == "lower_is_better":
        passed = bool(value <= high)
        loss = _bounded_loss(value, _target_scale(target))
        status = "pass" if passed else "fail"
    elif direction == "higher_is_better":
        passed = bool(value >= low)
        loss = 0.0 if passed else _bounded_loss(low - value, _target_scale(target))
        status = "pass" if passed else "fail"
    else:
        passed = bool(value >= low and value <= high)
        if passed:
            loss = 0.0
        elif value < low:
            loss = _bounded_loss(low - value, _target_scale(target))
        else:
            loss = _bounded_loss(value - high, _target_scale(target))
        status = "pass" if passed else "fail"
    return {
        **_target_metadata(target),
        "source": source,
        "variant": variant,
        "value": value,
        "status": status,
        "passed": bool(passed),
        "normalized_loss": float(loss),
        "score": float(1.0 - loss),
        "interpretation": str(target["notes"]),
    }


def _target_gap_row(target: pd.Series, *, source: str, variant: str, status: str, interpretation: str) -> dict[str, Any]:
    return {
        **_target_metadata(target),
        "source": source,
        "variant": variant,
        "value": np.nan,
        "status": status,
        "passed": False,
        "normalized_loss": 1.0,
        "score": 0.0,
        "interpretation": interpretation,
    }


def _target_metadata(target: pd.Series) -> dict[str, Any]:
    fields = [
        "target_id",
        "split",
        "family",
        "scenario_id",
        "variant_scope",
        "metric",
        "extractor",
        "target_low",
        "target_high",
        "target_value",
        "score_direction",
        "weight",
        "blocking",
        "critical",
        "source_status",
        "source_label",
        "target_scope",
        "baseline_scenario_id",
        "window_start",
        "window_end",
    ]
    return {field: target[field] for field in fields}


def _load_vintage_oos_summary(vintage_oos_dir: Path) -> pd.DataFrame:
    summary_path = vintage_oos_dir / "demand_vintage_oos_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=["source", "variant", "metric", "value", "status", "passed"])
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return pd.DataFrame(columns=["source", "variant", "metric", "value", "status", "passed"])
    rows = []
    test_rows = summary[summary["split"].astype(str).isin(["test", "holdout", "oos"])]
    for _, row in test_rows.iterrows():
        rows.append(
            {
                "source": str(row["source"]),
                "variant": str(row.get("variant", row["source"])),
                "metric": "weighted_normalized_abs_error",
                "value": float(row["weighted_normalized_abs_error"]),
                "status": "pass",
                "passed": True,
            }
        )
    return pd.DataFrame(rows)


def _oos_scores_available(vintage_oos_dir: Path) -> bool:
    return (vintage_oos_dir / "demand_vintage_oos_scores.csv").exists() and (vintage_oos_dir / "demand_vintage_oos_targets.csv").exists()


def _vintage_oos_provenance(vintage_oos_dir: Path) -> dict[str, Any]:
    manifest_path = vintage_oos_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "missing"}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    schema_version = data.get("schema_version")
    verdict = data.get("verdict")
    if schema_version == "belief_dynamics_calibration_v1" and str(verdict).startswith("belief_calibration_"):
        split_disjoint = bool(data.get("calibration_split_disjoint", False))
        verdict = "demand_vintage_oos_scored" if bool(data.get("passed", False)) and split_disjoint else verdict
    return {
        "status": str(data.get("status", "unknown")),
        "mode": data.get("mode"),
        "forecast_mode": data.get("forecast_mode"),
        "verdict": verdict,
        "artifact_verdict": data.get("verdict"),
        "passed": bool(data.get("passed", False)),
        "schema_version": schema_version,
        "live_call_count": int(data.get("live_call_count", 0) or 0),
        "cache_hit_count": int(data.get("cache_hit_count", 0) or 0),
    }


def _vintage_oos_empirical_eligible(provenance: dict[str, Any]) -> bool:
    forecast_mode = str(provenance.get("forecast_mode") or "")
    base_eligible = bool(
        provenance.get("passed")
        and provenance.get("mode") not in {None, "", "fixture"}
        and forecast_mode not in {"", "fixture"}
        and provenance.get("verdict") == "demand_vintage_oos_scored"
    )
    if not base_eligible:
        return False
    if forecast_mode == "live":
        return int(provenance.get("live_call_count", 0) or 0) > 0 and int(provenance.get("cache_hit_count", 0) or 0) == 0
    if forecast_mode == "replay":
        return int(provenance.get("cache_hit_count", 0) or 0) > 0
    return False


def _oos_llm_beats_baseline(attribution: pd.DataFrame | None) -> bool:
    if attribution is None or attribution.empty:
        return False
    rows = attribution[attribution["split"].astype(str) == "oos"].copy()
    if rows.empty:
        return False
    llm_loss = pd.to_numeric(rows["llm_weighted_loss"], errors="coerce")
    baseline_loss = pd.to_numeric(rows["best_baseline_weighted_loss"], errors="coerce")
    improvement = pd.to_numeric(rows["loss_improvement_pct"], errors="coerce")
    return bool(((llm_loss < baseline_loss) & (improvement > 0.0)).any())


def _oos_loss_improvement_pct(attribution: pd.DataFrame | None) -> float | None:
    if attribution is None or attribution.empty:
        return None
    rows = attribution[attribution["split"].astype(str) == "oos"].copy()
    if rows.empty:
        return None
    values = pd.to_numeric(rows["loss_improvement_pct"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _best_oos_pairwise_share_positive(oos_pairwise: pd.DataFrame | None) -> float | None:
    if oos_pairwise is None or oos_pairwise.empty:
        return None
    values = pd.to_numeric(oos_pairwise.get("bootstrap_share_positive"), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _scorecard_status(scores: pd.DataFrame) -> str:
    if scores.empty:
        return "missing"
    if "status" in scores and (scores["status"].astype(str) == "gap").any():
        return "partial"
    if "status" in scores and (scores["status"].astype(str) == "fail").any():
        return "fail"
    return "pass"


def _bottom_line(manifest: dict[str, Any], failed: pd.DataFrame) -> str:
    verdict = manifest.get("verdict")
    if verdict == "macro_empirical_oos_ready":
        return (
            f"Verdict: `{verdict}`. The lab gate passes and the LLM vintage OOS forecast beats the strongest deterministic baseline. "
            "This is the first empirical performance-ready state."
        )
    if verdict == "macro_lab_performance_ready":
        if manifest.get("vintage_oos_scores_available") and not manifest.get("vintage_oos_empirical_eligible"):
            cap = " Vintage OOS scores are present as diagnostics, but their provenance is fixture/non-empirical."
        elif manifest.get("vintage_oos_scores_available") and manifest.get("vintage_oos_empirical_eligible"):
            improvement = _finite_or_nan(manifest.get("vintage_oos_llm_baseline_improvement_pct"))
            if np.isfinite(improvement) and improvement > 0.0:
                cap = (
                    " Vintage OOS scores are present and the LLM forecast beats the strongest deterministic baseline, "
                    "but the absolute OOS target is still missed."
                )
            else:
                cap = " Vintage OOS scores are present, but the LLM forecast has not beaten the strongest deterministic OOS baseline yet."
        elif manifest.get("vintage_oos_scores_available"):
            cap = " Vintage OOS scores are present as diagnostics."
        else:
            cap = " Vintage OOS is still missing, so empirical readiness is capped."
        return f"Verdict: `{verdict}`. The LLM belief economy clears the lab performance gate.{cap}"
    misses = ", ".join(failed["target_id"].astype(str).head(8).tolist()) if not failed.empty else "missing score surface"
    return f"Verdict: `{verdict}`. The performance gate needs work before further live spend. Blocking misses: {misses}."


def _live_blocked_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "schema_version": MACRO_PERFORMANCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "status": "blocked",
        "verdict": "macro_performance_live_blocked",
        "passed": False,
        "reason": "Live performance mode needs actor-aware call budgeting and scored OOS artifacts first.",
        "demand_run_dir": args.demand_run_dir,
        "vintage_panel_dir": args.vintage_panel_dir,
        "vintage_oos_dir": args.vintage_oos_dir,
        "target_catalog_path": args.target_catalog,
        "outputs": ["manifest.json", "macro_performance_report.md"],
    }
    return {
        "manifest": manifest,
        "target_catalog": pd.DataFrame(),
        "scores": pd.DataFrame(),
        "variant_summary": pd.DataFrame(),
        "attribution": pd.DataFrame(),
        "vintage_readiness": pd.DataFrame(),
        "oos_pairwise": pd.DataFrame(),
        "report": build_macro_performance_report(manifest, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    }


def _target_scale(target: pd.Series) -> float:
    low = _finite_or_inf(target["target_low"])
    high = _finite_or_inf(target["target_high"])
    value = _finite_or_nan(target["target_value"])
    if np.isfinite(low) and np.isfinite(high) and abs(high - low) > 1e-12:
        return float(abs(high - low))
    if np.isfinite(value) and abs(value) > 1e-12:
        return float(abs(value))
    finite_bound = high if np.isfinite(high) else low
    if np.isfinite(finite_bound) and abs(finite_bound) > 1e-12:
        return float(abs(finite_bound))
    return 1.0


def _bounded_loss(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 1.0
    return float(np.clip(max(0.0, distance) / max(float(scale), 1e-12), 0.0, 1.0))


def _coerce_catalog_number(value: Any) -> float:
    text = str(value).strip()
    if text == "":
        return np.nan
    if text.lower() == "inf":
        return np.inf
    if text.lower() == "-inf":
        return -np.inf
    return float(text)


def _coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return float(numeric) if np.isfinite(numeric) else np.nan


def _finite_or_inf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _variant_from_source(source: str) -> str:
    text = str(source)
    if text.startswith("llm_belief"):
        return LLM_VARIANT
    for variant in ["representative", "adaptive", "naive_persona", "no_change", "rolling_mean", "rolling_trend", "llm_belief_fixture"]:
        if text.startswith(variant):
            return variant
    return text


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
        if np.isfinite(item):
            return item
        return "Infinity" if item > 0 else "-Infinity" if item < 0 else None
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
