from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, markdown_table
from .behavior_gate import behavior_target_catalog


MACRO_VALIDITY_VERSION = "macro_validity_scorecard_v1"
DEFAULT_DEMAND_RUN_DIR = OUTPUT_ROOT / "demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "macro_validity_scorecard"
DEFAULT_VINTAGE_PANEL_DIR = WORK_ROOT / "fred_vintage_panel"
DEFAULT_REPORT_PATH = Path("reports") / "macro_validity_scorecard_report.md"
LLM_BELIEF_VARIANT = "llm_belief"
IRF_VARIABLES = (
    "aggregate_consumption",
    "aggregate_income",
    "output_gap_pct",
    "employment_rate",
    "inflation_rate",
    "policy_rate",
)
VINTAGE_REQUIREMENT_GROUPS = {
    "real_consumption": {"PCECC96", "PCE"},
    "saving_rate": {"PSAVERT"},
    "retail_spending": {"RSAFS", "RSXFS"},
    "inflation": {"CPIAUCSL", "PCEPI", "CPILFESL", "PCEPILFE"},
    "labor_market": {"UNRATE", "PAYEMS"},
    "policy_rate": {"FEDFUNDS", "TB3MS"},
    "sentiment": {"UMCSENT", "MICH"},
    "income": {"DSPIC96"},
    "output": {"GDPC1", "INDPRO"},
}


@dataclass(frozen=True)
class DemandRunArtifacts:
    run_dir: Path
    manifest: dict[str, Any]
    periods: pd.DataFrame
    validation: pd.DataFrame
    decisions: pd.DataFrame
    accounting: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score macro-validity gates for a demand-economy run.")
    parser.add_argument("--demand-run-dir", default=str(DEFAULT_DEMAND_RUN_DIR))
    parser.add_argument("--vintage-panel-dir", default=str(DEFAULT_VINTAGE_PANEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--variant", default=LLM_BELIEF_VARIANT)
    parser.add_argument("--source", default=None, help="Exact demand-run source to score. Defaults to the selected variant's first source.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.demand_run_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path) if args.report_path else None
    artifacts = load_demand_run(run_dir)
    result = build_macro_validity_scorecard(
        artifacts,
        vintage_panel_dir=Path(args.vintage_panel_dir),
        variant=args.variant,
        source=args.source,
    )
    write_macro_validity_outputs(result, output_dir=output_dir, report_path=report_path)
    print(f"Wrote macro validity scorecard to {output_dir}")
    if report_path is not None:
        print(f"Wrote macro validity report to {report_path}")
    return 0


def load_demand_run(run_dir: Path) -> DemandRunArtifacts:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Demand run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DemandRunArtifacts(
        run_dir=run_dir,
        manifest=manifest,
        periods=_read_csv(run_dir / "demand_periods.csv"),
        validation=_read_csv(run_dir / "demand_validation_scores.csv"),
        decisions=_read_csv(run_dir / "demand_household_decisions.csv"),
        accounting=_read_csv(run_dir / "demand_accounting.csv"),
    )


def build_macro_validity_scorecard(
    artifacts: DemandRunArtifacts,
    *,
    vintage_panel_dir: Path,
    variant: str = LLM_BELIEF_VARIANT,
    source: str | None = None,
) -> dict[str, Any]:
    scored_source = select_source(artifacts.validation, variant=variant, source=source)
    target_catalog = behavior_target_catalog(include_unscored=True)
    micro_scores = score_micro_behavior_gate(
        artifacts.validation,
        artifacts.decisions,
        target_catalog,
        source=scored_source,
        variant=variant,
    )
    irf_paths = build_demand_irf_paths(artifacts.periods)
    irf_scores = score_demand_irf_shape(
        irf_paths,
        artifacts.validation,
        expected_sources=_variant_sources(artifacts.periods, variant=variant, source=scored_source),
    )
    vintage_scores = score_vintage_oos_readiness(vintage_panel_dir)
    scorecard = build_scorecard(micro_scores, irf_scores, vintage_scores)
    manifest = {
        "schema_version": MACRO_VALIDITY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "demand_run_dir": str(artifacts.run_dir),
        "demand_run_verdict": artifacts.manifest.get("verdict"),
        "demand_run_evidence_verdict": (artifacts.manifest.get("evidence") or {}).get("evidence_verdict"),
        "scored_source": scored_source,
        "scored_variant": variant,
        "vintage_panel_dir": str(vintage_panel_dir),
        "overall_verdict": macro_validity_verdict(scorecard),
        "outputs": [
            "macro_validity_scorecard.csv",
            "macro_validity_micro_behavior_scores.csv",
            "macro_validity_irf_paths.csv",
            "macro_validity_irf_scores.csv",
            "macro_validity_vintage_readiness.csv",
            "macro_validity_report.md",
            "manifest.json",
        ],
    }
    return {
        "manifest": manifest,
        "scorecard": scorecard,
        "micro_scores": micro_scores,
        "irf_paths": irf_paths,
        "irf_scores": irf_scores,
        "vintage_scores": vintage_scores,
    }


def write_macro_validity_outputs(result: dict[str, Any], *, output_dir: Path, report_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["scorecard"].to_csv(output_dir / "macro_validity_scorecard.csv", index=False)
    result["micro_scores"].to_csv(output_dir / "macro_validity_micro_behavior_scores.csv", index=False)
    result["irf_paths"].to_csv(output_dir / "macro_validity_irf_paths.csv", index=False)
    result["irf_scores"].to_csv(output_dir / "macro_validity_irf_scores.csv", index=False)
    result["vintage_scores"].to_csv(output_dir / "macro_validity_vintage_readiness.csv", index=False)
    report = build_macro_validity_report(
        result["manifest"],
        result["scorecard"],
        result["micro_scores"],
        result["irf_scores"],
        result["vintage_scores"],
    )
    (output_dir / "macro_validity_report.md").write_text(report, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(result["manifest"]), indent=2, sort_keys=True), encoding="utf-8")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_dir / "macro_validity_report.md", report_path)


def score_micro_behavior_gate(
    validation: pd.DataFrame,
    decisions: pd.DataFrame,
    target_catalog: pd.DataFrame,
    *,
    source: str,
    variant: str = LLM_BELIEF_VARIANT,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aggregate_total_range = _catalog_range(
        target_catalog,
        target_family="mpc",
        response_variable="total_spending_share",
        target_scope="aggregate",
    )
    low_liquidity_range = _catalog_target_range(target_catalog, "eip_2020_low_liquidity_spend")
    high_liquidity_range = _catalog_target_range(target_catalog, "eip_2020_high_liquidity_spend")
    liquidity_ratio_range = _catalog_target_range(target_catalog, "eip_2020_liquidity_ratio")
    rows.append(
        _metric_score(
            "micro_behavior",
            source,
            variant,
            "aggregate_transfer_mpc_public_bridge",
            _validation_value(validation, source, "transfer_impact_mpc"),
            aggregate_total_range[0],
            aggregate_total_range[1],
            "public_target_bridge",
            "Existing transfer-shock impact MPC against public total-spending transfer-response bands.",
        )
    )
    low_mpc = _validation_value(validation, source, "low_liquidity_impact_mpc")
    high_mpc = _validation_value(validation, source, "high_liquidity_impact_mpc")
    rows.append(
        _metric_score(
            "micro_behavior",
            source,
            variant,
            "low_liquidity_mpc_public_bridge",
            low_mpc,
            low_liquidity_range[0],
            low_liquidity_range[1],
            "public_target_bridge",
            "Low-liquidity impact MPC against the public low-balance bridge band.",
        )
    )
    rows.append(
        _metric_score(
            "micro_behavior",
            source,
            variant,
            "high_liquidity_mpc_public_bridge",
            high_mpc,
            high_liquidity_range[0],
            high_liquidity_range[1],
            "public_target_bridge",
            "High-liquidity impact MPC against the public high-balance bridge band.",
        )
    )
    rows.append(
        _metric_score(
            "micro_behavior",
            source,
            variant,
            "low_high_liquidity_mpc_ratio_public_bridge",
            low_mpc / high_mpc if np.isfinite(low_mpc) and np.isfinite(high_mpc) and abs(high_mpc) > 1e-12 else np.nan,
            liquidity_ratio_range[0],
            liquidity_ratio_range[1],
            "public_target_bridge",
            "Low-liquidity to high-liquidity MPC ratio against the public liquidity-gradient bridge band.",
        )
    )
    for metric, target_kind, interpretation in [
        ("liquidity_mpc_gradient", "hank_shape_constraint", "Low-liquidity impact MPC minus high-liquidity impact MPC."),
        ("income_mpc_gradient", "hank_shape_constraint", "Low-income impact MPC minus high-income impact MPC."),
        ("job_risk_impact_consumption_delta", "behavior_mechanism_constraint", "Consumption falls when perceived job risk rises before income changes."),
        ("job_risk_impact_income_delta_abs", "behavior_mechanism_constraint", "Income is unchanged on impact in a pure perceived-risk shock."),
        ("max_accounting_abs_residual", "accounting_constraint", "Household and goods-market identities hold to numerical tolerance."),
    ]:
        row = validation[(validation["source"] == source) & (validation["metric"] == metric)]
        if row.empty:
            rows.append(_missing_score("micro_behavior", source, variant, metric, target_kind, interpretation))
            continue
        rows.append(
            _metric_score(
                "micro_behavior",
                source,
                variant,
                metric,
                float(row["value"].iloc[0]),
                float(row["target_low"].iloc[0]),
                float(row["target_high"].iloc[0]),
                target_kind,
                interpretation,
            )
        )
    for metric, interpretation in [
        ("transfer_allocation_share_sum_residual", "Transfer allocation shares should exhaust each transfer dollar across consumption, debt repayment, and liquid saving."),
    ]:
        rows.append(_transfer_allocation_residual_score(decisions, source, variant, metric, interpretation))
    for target_id, metric, column, interpretation in [
        (
            "eip_2020_debt_share",
            "debt_repayment_action_target",
            "transfer_debt_repayment_share",
            "Demand-economy transfer mechanism allocates a public-target-consistent share to debt repayment.",
        ),
        (
            "eip_2020_liquid_saving_share",
            "liquid_saving_action_target",
            "transfer_liquid_saving_share",
            "Demand-economy transfer mechanism allocates a public-target-consistent share to liquid saving.",
        ),
    ]:
        low, high = _catalog_target_range(target_catalog, target_id)
        value = _weighted_transfer_share(decisions, source, column)
        if np.isfinite(value):
            rows.append(_metric_score("micro_behavior", source, variant, metric, value, low, high, "public_direct_behavior_target", interpretation))
        else:
            rows.append(_gap_score("micro_behavior", source, variant, metric, "direct_micro_target_gap", interpretation, blocking=True))
    for metric, interpretation in [
        ("labor_response_action_target", "Demand-economy output does not yet score direct household labor-response targets."),
        ("portfolio_liquidity_shift_action_target", "Demand-economy output does not yet score direct portfolio or liquidity-shift targets."),
    ]:
        rows.append(_gap_score("micro_behavior", source, variant, metric, "future_scope_gap", interpretation, blocking=False))
    return pd.DataFrame(rows)


def build_demand_irf_paths(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_periods in periods.groupby("source", sort=True):
        if "baseline" not in set(source_periods["scenario_id"].astype(str)):
            continue
        baseline = source_periods[source_periods["scenario_id"] == "baseline"].copy()
        baseline_columns = ["period_index", *[column for column in IRF_VARIABLES if column in baseline]]
        for scenario_id, scenario_periods in source_periods[source_periods["scenario_id"] != "baseline"].groupby("scenario_id", sort=True):
            joined = scenario_periods.merge(baseline[baseline_columns], on="period_index", suffixes=("", "_baseline"))
            for _, row in joined.iterrows():
                for variable in IRF_VARIABLES:
                    baseline_column = f"{variable}_baseline"
                    if variable not in row or baseline_column not in row:
                        continue
                    scenario_value = float(row[variable])
                    baseline_value = float(row[baseline_column])
                    response = scenario_value - baseline_value
                    rows.append(
                        {
                            "source": source,
                            "variant": str(row.get("variant", _variant_from_source(source))),
                            "scenario_id": str(scenario_id),
                            "period_id": str(row.get("period_id", f"period_{int(row['period_index'])}")),
                            "period_index": int(row["period_index"]),
                            "variable": variable,
                            "baseline_value": baseline_value,
                            "scenario_value": scenario_value,
                            "response": response,
                            "shock_scale": _shock_scale(row, scenario_id),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["source", "scenario_id", "variable", "period_index"]).reset_index(drop=True)


def score_demand_irf_shape(
    irfs: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    expected_sources: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sorted(set(expected_sources)):
        variant = _variant_from_source(source)
        rows.extend(_score_rate_hike_irfs(irfs, source, variant))
        rows.extend(_score_transfer_irfs(irfs, source, variant))
        rows.extend(_score_job_risk_irfs(irfs, source, variant))
        amplification = _validation_value(validation, source, "belief_feedback_amplification_ratio")
        rows.append(
            _metric_score(
                "impulse_response",
                source,
                variant,
                "belief_feedback_output_rms_ratio",
                amplification,
                1.05,
                np.inf,
                "model_sanity",
                "Belief-feedback output movement divided by baseline movement.",
            )
        )
    return pd.DataFrame(rows)


def score_vintage_oos_readiness(vintage_panel_dir: Path) -> pd.DataFrame:
    status_path = vintage_panel_dir / "fred_vintage_status.json"
    origins_path = vintage_panel_dir / "forecast_origins_for_vintage_context.csv"
    context_path = vintage_panel_dir / "fred_vintage_context.csv"
    status = _read_json(status_path) if status_path.exists() else {}
    origins = _read_csv(origins_path) if origins_path.exists() else pd.DataFrame()
    context = _read_csv(context_path) if context_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    source = str(vintage_panel_dir)
    variant = "vintage_oos"
    rows.append(
        _metric_score(
            "vintage_oos",
            source,
            variant,
            "fred_vintage_panel_present",
            1.0 if status_path.exists() and context_path.exists() else 0.0,
            1.0,
            1.0,
            "readiness",
            "As-of FRED/ALFRED vintage panel files are present.",
        )
    )
    rows.append(
        _metric_score(
            "vintage_oos",
            source,
            variant,
            "vintage_origin_count",
            float(status.get("origin_count", origins["origin"].nunique() if "origin" in origins else 0)),
            24.0,
            np.inf,
            "readiness",
            "Vintage context has enough origins for a first OOS scoring split.",
        )
    )
    test_count = int(origins[origins["split"].astype(str).str.lower().isin({"test", "holdout", "oos"})].shape[0]) if "split" in origins else 0
    rows.append(
        _metric_score(
            "vintage_oos",
            source,
            variant,
            "vintage_test_origin_count",
            float(test_count),
            8.0,
            np.inf,
            "readiness",
            "Vintage context has a held-out split large enough for a first scored run.",
        )
    )
    rows.append(
        _metric_score(
            "vintage_oos",
            source,
            variant,
            "vintage_realtime_fields_present",
            1.0 if {"as_of_date", "realtime_start", "realtime_end", "observation_date"}.issubset(set(context.columns)) else 0.0,
            1.0,
            1.0,
            "leakage_control",
            "Vintage rows retain as-of and real-time fields needed for prompt leakage audits.",
        )
    )
    series_ids = set(status.get("series_ids") or (context["series_id"].dropna().astype(str).unique().tolist() if "series_id" in context else []))
    for group_name, allowed in VINTAGE_REQUIREMENT_GROUPS.items():
        rows.append(
            _metric_score(
                "vintage_oos",
                source,
                variant,
                f"vintage_series_group_{group_name}",
                1.0 if series_ids & allowed else 0.0,
                1.0,
                1.0,
                "coverage",
                f"Vintage panel covers {group_name.replace('_', ' ')} via one of {', '.join(sorted(allowed))}.",
            )
        )
    runner_files = [
        "demand_vintage_oos_cards.csv",
        "demand_vintage_oos_targets.csv",
        "demand_vintage_oos_scores.csv",
    ]
    for filename in runner_files:
        rows.append(
            _gap_score(
                "vintage_oos",
                source,
                variant,
                filename.replace(".csv", "_available"),
                "scored_oos_gap",
                f"Scored date-free demand-vintage OOS artifact is not present yet: {filename}.",
            )
        )
    return pd.DataFrame(rows)


def build_scorecard(micro_scores: pd.DataFrame, irf_scores: pd.DataFrame, vintage_scores: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _scorecard_row(
            "micro_behavior",
            micro_scores,
            "Micro behavior is scored against transfer-MPC bridges, HANK-style heterogeneity, precaution, and direct-action gaps.",
            "Use this as the core consumption-behavior gate; labor and portfolio targets remain future-scope extensions.",
        ),
        _scorecard_row(
            "impulse_response",
            irf_scores,
            "Impulse responses are scored as scenario-minus-baseline shape constraints.",
            "Keep this as the qualitative IRF gate; add sourced magnitude bands before calling it empirical IRF validation.",
        ),
        _scorecard_row(
            "vintage_oos",
            vintage_scores,
            "Vintage context coverage is scored separately from hidden-outcome OOS performance.",
            "Build the date-free demand-vintage card/target/scoring runner and replay this scorecard on those artifacts.",
        ),
    ]
    return pd.DataFrame(rows)


def macro_validity_verdict(scorecard: pd.DataFrame) -> str:
    if scorecard.empty:
        return "macro_validity_not_scored"
    statuses = dict(zip(scorecard["gate"], scorecard["status"]))
    if statuses.get("micro_behavior") == "pass" and statuses.get("impulse_response") == "pass" and statuses.get("vintage_oos") == "pass":
        return "macro_validity_ready"
    if statuses.get("micro_behavior") == "pass" and statuses.get("impulse_response") == "pass" and statuses.get("vintage_oos") == "partial":
        return "macro_behavior_dynamics_ready_but_vintage_oos_unscored"
    if statuses.get("impulse_response") == "pass" and statuses.get("vintage_oos") in {"partial", "pass"}:
        return "macro_validity_bridge_ready_but_behavior_and_oos_need_work"
    return "macro_validity_needs_work"


def build_macro_validity_report(
    manifest: dict[str, Any],
    scorecard: pd.DataFrame,
    micro_scores: pd.DataFrame,
    irf_scores: pd.DataFrame,
    vintage_scores: pd.DataFrame,
) -> str:
    micro_public_misses = micro_scores[
        (micro_scores["target_kind"] == "public_target_bridge") & (micro_scores["status"] == "fail")
    ]
    lines = [
        "# Macro Validity Scorecard",
        "",
        "## Bottom Line",
        _macro_validity_bottom_line(manifest, scorecard, micro_public_misses),
        "",
        "## Scorecard",
        markdown_table(scorecard),
        "",
        "## Micro Behavior Gate",
        markdown_table(micro_scores),
        "",
        "## Impulse-Response Shape Gate",
        markdown_table(irf_scores),
        "",
        "## Vintage OOS Readiness Gate",
        markdown_table(vintage_scores),
        "",
        "## What This Means",
        (
            "The current HANK-lite demand run has moved past pure mechanics. It now has a repeatable macro-validity "
            "surface: direct micro-behavior constraints, scenario-minus-baseline impulse responses, and vintage OOS "
            "readiness are checked in one place. The result should be read as a bridge scorecard. Passing the IRF "
            "shape gate says the abstract economy reacts in the right qualitative directions. It does not yet prove "
            "real-world vintage OOS macro accuracy, because the date-free vintage demand cards and hidden outcome "
            "scores are still missing."
        ),
        "",
        "## Next Gate",
        (
            "The next implementation target is the scored demand-vintage OOS runner: build hidden date-free cards from "
            "the vintage panel, map as-of macro states into the demand economy, withhold future demand outcomes, run "
            "the belief module, and compare against no-change, rolling/trend, and SPF-style controls."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def select_source(validation: pd.DataFrame, *, variant: str, source: str | None = None) -> str:
    if source:
        if validation.empty or source in set(validation.get("source", pd.Series(dtype=str)).astype(str)):
            return source
        raise ValueError(f"Source not found in validation scores: {source}")
    if validation.empty:
        raise ValueError("Cannot infer source from empty validation scores.")
    candidates = validation[validation["variant"].astype(str) == variant]["source"].dropna().astype(str).unique().tolist()
    if not candidates:
        raise ValueError(f"No source found for variant {variant!r}")
    return sorted(candidates)[0]


def _score_rate_hike_irfs(irfs: pd.DataFrame, source: str, variant: str) -> list[dict[str, Any]]:
    return [
        _share_score(
            irfs,
            source,
            variant,
            "rate_hike",
            "output_gap_pct",
            1,
            8,
            lambda series: series < 0.0,
            "rate_hike_output_negative_share_p1_p8",
            0.75,
            np.inf,
            "empirical_shape",
            "Rate-hike output-gap response is negative in most early periods.",
        ),
        _share_score(
            irfs,
            source,
            variant,
            "rate_hike",
            "aggregate_consumption",
            1,
            8,
            lambda series: series < 0.0,
            "rate_hike_consumption_negative_share_p1_p8",
            0.75,
            np.inf,
            "empirical_shape",
            "Rate-hike consumption response is negative in most early periods.",
        ),
        _share_score(
            irfs,
            source,
            variant,
            "rate_hike",
            "inflation_rate",
            4,
            12,
            lambda series: series <= 0.0,
            "rate_hike_inflation_lagged_negative_share_p4_p12",
            0.60,
            np.inf,
            "empirical_shape",
            "Rate-hike inflation response turns negative over the lagged window.",
        ),
        _trough_period_score(
            irfs,
            source,
            variant,
            "rate_hike",
            "output_gap_pct",
            "rate_hike_output_trough_period",
            2.0,
            12.0,
            "empirical_shape",
            "Rate-hike output trough occurs after impact and inside the medium-run window.",
        ),
        _early_positive_count_score(
            irfs,
            source,
            variant,
            "rate_hike",
            "output_gap_pct",
            "rate_hike_output_sign_reversal_pre_p8",
            "Rate-hike output response does not flip positive before period 8.",
        ),
    ]


def _score_transfer_irfs(irfs: pd.DataFrame, source: str, variant: str) -> list[dict[str, Any]]:
    transfer = _irf_slice(irfs, source, "transfer_shock", "aggregate_consumption", 1, 8)
    if transfer.empty:
        peak_period = np.nan
        decay_ratio = np.nan
    else:
        peak_period = float(transfer.loc[transfer["response"].idxmax(), "period_index"])
        response_p1 = _response_at(transfer, 1)
        response_p4 = _response_at(transfer, 4)
        decay_ratio = abs(response_p4) / abs(response_p1) if np.isfinite(response_p1) and abs(response_p1) > 1e-12 else np.nan
    return [
        _metric_score(
            "impulse_response",
            source,
            variant,
            "transfer_consumption_peak_period",
            peak_period,
            1.0,
            2.0,
            "empirical_shape",
            "Transfer consumption response peaks on impact or soon after.",
        ),
        _metric_score(
            "impulse_response",
            source,
            variant,
            "transfer_consumption_decay_ratio_p4_vs_p1",
            decay_ratio,
            0.0,
            1.0,
            "empirical_shape",
            "Transfer consumption response fades by period 4 relative to impact.",
        ),
    ]


def _score_job_risk_irfs(irfs: pd.DataFrame, source: str, variant: str) -> list[dict[str, Any]]:
    income = _irf_slice(irfs, source, "job_risk_shock", "aggregate_income", 1, 1)
    income_impact_abs = abs(_response_at(income, 1)) if not income.empty else np.nan
    return [
        _share_score(
            irfs,
            source,
            variant,
            "job_risk_shock",
            "aggregate_consumption",
            1,
            4,
            lambda series: series < 0.0,
            "job_risk_consumption_negative_share_p1_p4",
            0.75,
            np.inf,
            "behavior_mechanism_constraint",
            "Job-risk news lowers consumption before income mechanically changes.",
        ),
        _metric_score(
            "impulse_response",
            source,
            variant,
            "job_risk_income_impact_abs",
            income_impact_abs,
            0.0,
            1e-6,
            "behavior_mechanism_constraint",
            "Job-risk shock does not move income mechanically on impact.",
        ),
    ]


def _scorecard_row(gate: str, scores: pd.DataFrame, interpretation: str, next_action: str) -> dict[str, Any]:
    if scores.empty:
        return {
            "gate": gate,
            "status": "missing",
            "passed_count": 0,
            "scored_count": 0,
            "gap_count": 1,
            "failed_count": 0,
            "blocking_issue_count": 1,
            "interpretation": interpretation,
            "next_action": next_action,
        }
    scored = scores[scores["status"].isin(["pass", "fail"])]
    failed = scores[scores["status"] == "fail"]
    gaps = scores[scores["status"] == "gap"]
    blocking = scores["blocking"].astype(bool) if "blocking" in scores else pd.Series(True, index=scores.index)
    blocking_failed = scores[(scores["status"] == "fail") & blocking]
    blocking_gaps = scores[(scores["status"] == "gap") & blocking]
    if not blocking_failed.empty:
        status = "fail"
    elif not blocking_gaps.empty:
        status = "partial"
    else:
        status = "pass"
    return {
        "gate": gate,
        "status": status,
        "passed_count": int((scored["status"] == "pass").sum()),
        "scored_count": int(scored.shape[0]),
        "gap_count": int(gaps.shape[0]),
        "failed_count": int(failed.shape[0]),
        "blocking_issue_count": int(blocking_failed.shape[0] + blocking_gaps.shape[0]),
        "interpretation": interpretation,
        "next_action": next_action,
    }


def _macro_validity_bottom_line(manifest: dict[str, Any], scorecard: pd.DataFrame, micro_public_misses: pd.DataFrame) -> str:
    statuses = dict(zip(scorecard["gate"], scorecard["status"])) if not scorecard.empty else {}
    miss_text = ""
    if not micro_public_misses.empty:
        missed = ", ".join(micro_public_misses["metric"].astype(str).tolist())
        miss_text = f" The public micro-behavior bridge misses are: {missed}."
    return (
        f"Verdict: `{manifest.get('overall_verdict')}`. "
        f"Micro behavior is `{statuses.get('micro_behavior', 'missing')}`, impulse-response shape is "
        f"`{statuses.get('impulse_response', 'missing')}`, and vintage OOS readiness is "
        f"`{statuses.get('vintage_oos', 'missing')}`.{miss_text} "
        "So the current demand economy has a real qualitative macro-dynamics score. Broader macro validity still "
        "depends on a scored date-free vintage OOS run."
    )


def _catalog_target_range(target_catalog: pd.DataFrame, target_id: str) -> tuple[float, float]:
    row = target_catalog[target_catalog["target_id"] == target_id]
    if row.empty:
        return np.nan, np.nan
    return float(row["target_low"].iloc[0]), float(row["target_high"].iloc[0])


def _catalog_range(target_catalog: pd.DataFrame, *, target_family: str, response_variable: str, target_scope: str) -> tuple[float, float]:
    rows = target_catalog[
        (target_catalog["scored"].astype(bool))
        & (target_catalog["target_family"].astype(str) == target_family)
        & (target_catalog["response_variable"].astype(str) == response_variable)
        & (target_catalog["target_scope"].astype(str) == target_scope)
    ]
    if rows.empty:
        return np.nan, np.nan
    return float(rows["target_low"].min()), float(rows["target_high"].max())


def _variant_sources(periods: pd.DataFrame, *, variant: str, source: str | None = None) -> list[str]:
    if source is not None:
        return [source]
    if periods.empty or "source" not in periods:
        return []
    if "variant" in periods:
        candidates = periods[periods["variant"].astype(str) == variant]["source"].dropna().astype(str).unique().tolist()
    else:
        candidates = [src for src in periods["source"].dropna().astype(str).unique().tolist() if _variant_from_source(src) == variant]
    return sorted(candidates)


def _metric_score(
    gate: str,
    source: str,
    variant: str,
    metric: str,
    value: float,
    target_low: float,
    target_high: float,
    target_kind: str,
    interpretation: str,
) -> dict[str, Any]:
    value = float(value) if np.isfinite(value) else np.nan
    passed = bool(np.isfinite(value) and value >= float(target_low) and value <= float(target_high))
    return {
        "gate": gate,
        "source": source,
        "variant": variant,
        "metric": metric,
        "value": value,
        "target_low": float(target_low) if np.isfinite(target_low) else target_low,
        "target_high": float(target_high) if np.isfinite(target_high) else target_high,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "blocking": True,
        "target_kind": target_kind,
        "interpretation": interpretation,
    }


def _missing_score(gate: str, source: str, variant: str, metric: str, target_kind: str, interpretation: str) -> dict[str, Any]:
    return _metric_score(gate, source, variant, metric, np.nan, 0.0, 0.0, target_kind, interpretation)


def _gap_score(
    gate: str,
    source: str,
    variant: str,
    metric: str,
    target_kind: str,
    interpretation: str,
    *,
    blocking: bool = True,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "source": source,
        "variant": variant,
        "metric": metric,
        "value": np.nan,
        "target_low": np.nan,
        "target_high": np.nan,
        "status": "gap",
        "passed": False,
        "blocking": bool(blocking),
        "target_kind": target_kind,
        "interpretation": interpretation,
    }


def _weighted_transfer_share(decisions: pd.DataFrame, source: str, column: str) -> float:
    required = {"source", "scenario_id", "period_index", "population_weight", "transfer", column}
    if decisions.empty or not required.issubset(set(decisions.columns)):
        return np.nan
    rows = decisions[
        (decisions["source"].astype(str) == source)
        & (decisions["scenario_id"].astype(str) == "transfer_shock")
        & (decisions["period_index"].astype(int) == 1)
        & (pd.to_numeric(decisions["transfer"], errors="coerce") > 0)
    ].copy()
    if rows.empty:
        return np.nan
    weights = pd.to_numeric(rows["population_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    transfer = pd.to_numeric(rows["transfer"], errors="coerce").fillna(0.0).clip(lower=0.0)
    values = pd.to_numeric(rows[column], errors="coerce")
    denominator = float((weights * transfer).sum())
    if denominator <= 0:
        return np.nan
    return float((weights * transfer * values).sum() / denominator)


def _transfer_allocation_residual_score(
    decisions: pd.DataFrame,
    source: str,
    variant: str,
    metric: str,
    interpretation: str,
) -> dict[str, Any]:
    columns = {"transfer_consumption_share", "transfer_debt_repayment_share", "transfer_liquid_saving_share"}
    required = {"source", "scenario_id", "period_index", "transfer", *columns}
    if decisions.empty or not required.issubset(set(decisions.columns)):
        return _gap_score("micro_behavior", source, variant, metric, "mechanism_accounting", interpretation, blocking=True)
    rows = decisions[
        (decisions["source"].astype(str) == source)
        & (decisions["scenario_id"].astype(str) == "transfer_shock")
        & (decisions["period_index"].astype(int) == 1)
        & (pd.to_numeric(decisions["transfer"], errors="coerce") > 0)
    ].copy()
    if rows.empty:
        return _gap_score("micro_behavior", source, variant, metric, "mechanism_accounting", interpretation, blocking=True)
    residual = (
        pd.to_numeric(rows["transfer_consumption_share"], errors="coerce")
        + pd.to_numeric(rows["transfer_debt_repayment_share"], errors="coerce")
        + pd.to_numeric(rows["transfer_liquid_saving_share"], errors="coerce")
        - 1.0
    ).abs()
    value = float(residual.max())
    return _metric_score("micro_behavior", source, variant, metric, value, 0.0, 1e-9, "mechanism_accounting", interpretation)


def _share_score(
    irfs: pd.DataFrame,
    source: str,
    variant: str,
    scenario_id: str,
    variable: str,
    start: int,
    end: int,
    predicate: Any,
    metric: str,
    target_low: float,
    target_high: float,
    target_kind: str,
    interpretation: str,
) -> dict[str, Any]:
    frame = _irf_slice(irfs, source, scenario_id, variable, start, end)
    if frame.empty:
        value = np.nan
    else:
        value = float(predicate(frame["response"].astype(float)).mean())
    return _metric_score("impulse_response", source, variant, metric, value, target_low, target_high, target_kind, interpretation)


def _trough_period_score(
    irfs: pd.DataFrame,
    source: str,
    variant: str,
    scenario_id: str,
    variable: str,
    metric: str,
    target_low: float,
    target_high: float,
    target_kind: str,
    interpretation: str,
) -> dict[str, Any]:
    frame = _irf_slice(irfs, source, scenario_id, variable, 1, 20)
    if frame.empty:
        value = np.nan
    else:
        value = float(frame.loc[frame["response"].idxmin(), "period_index"])
    return _metric_score("impulse_response", source, variant, metric, value, target_low, target_high, target_kind, interpretation)


def _early_positive_count_score(
    irfs: pd.DataFrame,
    source: str,
    variant: str,
    scenario_id: str,
    variable: str,
    metric: str,
    interpretation: str,
) -> dict[str, Any]:
    frame = _irf_slice(irfs, source, scenario_id, variable, 1, 8)
    value = float((frame["response"].astype(float) > 0.0).sum()) if not frame.empty else np.nan
    return _metric_score("impulse_response", source, variant, metric, value, 0.0, 0.0, "empirical_shape", interpretation)


def _irf_slice(irfs: pd.DataFrame, source: str, scenario_id: str, variable: str, start: int, end: int) -> pd.DataFrame:
    if irfs.empty:
        return pd.DataFrame()
    return irfs[
        (irfs["source"].astype(str) == source)
        & (irfs["scenario_id"].astype(str) == scenario_id)
        & (irfs["variable"].astype(str) == variable)
        & (irfs["period_index"].astype(int) >= start)
        & (irfs["period_index"].astype(int) <= end)
    ].copy()


def _response_at(frame: pd.DataFrame, period_index: int) -> float:
    row = frame[frame["period_index"].astype(int) == int(period_index)]
    if row.empty:
        return np.nan
    return float(row["response"].iloc[0])


def _validation_value(validation: pd.DataFrame, source: str, metric: str) -> float:
    if validation.empty:
        return np.nan
    row = validation[(validation["source"].astype(str) == source) & (validation["metric"].astype(str) == metric)]
    if row.empty:
        return np.nan
    value = pd.to_numeric(row["value"], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def _shock_scale(row: pd.Series, scenario_id: str) -> float:
    if scenario_id == "transfer_shock":
        return float(row.get("transfer_per_household", 0.0))
    if scenario_id == "rate_hike":
        return float(row.get("policy_rate_shock_pp", 0.0))
    if scenario_id == "job_risk_shock":
        return float(row.get("job_risk_shock_pp", 0.0))
    return 1.0


def _variant_from_source(source: str) -> str:
    if str(source).startswith("llm_belief"):
        return LLM_BELIEF_VARIANT
    for variant in ["representative", "adaptive", "naive_persona"]:
        if str(source).startswith(variant):
            return variant
    return str(source)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
