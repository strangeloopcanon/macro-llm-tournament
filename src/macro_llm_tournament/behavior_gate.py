from __future__ import annotations

import argparse
import json
from importlib import resources
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import AGENT_ECONOMY_VERSION, OUTPUT_ROOT, WORK_ROOT, cache_key, markdown_table, round_or_none
from .agent_llm import AgentLLMClient
from .agent_types import build_household_type_cells
from .llm_common import LLMUnavailable


BEHAVIOR_GATE_VERSION = "household_behavior_target_gate_v4"
BEHAVIOR_PROMPT_VERSION = "household_behavior_target_gate_v1"
BEHAVIOR_PROMPT_DESCRIPTIVE_VERSION = "household_behavior_target_gate_descriptive_v1"
BEHAVIOR_PROMPT_VARIANTS = ("baseline", "descriptive")
BEHAVIOR_DESCRIPTIVE_FRAMING = (
    "Predict the average behavior that real US households of each type were actually measured to do in this "
    "situation, as found in spending diaries, bank transaction data, and household panel studies. Do not answer "
    "with what a financially prudent household should do; measured behavior often departs from prudent advice. "
    "Before answering, silently reason about each household type's monthly cash flow, liquid buffer in months, "
    "and how binding its liquidity constraint is, then commit to shares consistent with that reasoning."
)
BEHAVIOR_PRIMITIVE_PROMPT_VERSION = "household_behavior_primitives_v2"
BEHAVIOR_PRIMITIVE_POLICY_VERSION = "primitive_to_action_policy_log_shock_v1"
BEHAVIOR_PRIMITIVE_CALIBRATED_POLICY_VERSION = "primitive_to_action_policy_selection_calibrated_v1"
TARGET_CATALOG_PACKAGE = "macro_llm_tournament"
TARGET_CATALOG_RESOURCE = "data/public_behavior_targets.csv"
BEHAVIOR_SHARE_COLUMNS = [
    "total_spending_share",
    "nondurable_spending_share",
    "durable_spending_share",
    "debt_repayment_share",
    "liquid_saving_share",
]
BEHAVIOR_BASELINE_SOURCES = ("liquidity_rule", "flat_30pct_rule", "permanent_income_rule")
BEHAVIOR_BASELINE_TIE_TOLERANCE = 1e-9
BEHAVIOR_SELECTION_SPLIT = "behavior_selection_v1"
BEHAVIOR_HOLDOUT_SPLIT = "behavior_holdout_v1"
BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT = "behavior_holdout_ui_v1"
BEHAVIOR_PRESPECIFIED_SUFFIX = "__liquidity_prior_50"
BEHAVIOR_PRIMITIVE_FIELDS = [
    "perceived_job_loss_risk_pp",
    "expected_income_growth_pct",
    "precautionary_saving_motive",
    "liquidity_stress",
    "debt_repayment_urgency",
    "durable_purchase_pull_forward",
    "shock_size_normalized",
    "shock_size_log_income_ratio",
    "confidence",
]
BEHAVIOR_BASELINE_COMPARISON_COLUMNS = [
    "source",
    "source_kind",
    "target_scope",
    "evaluation_split",
    "target_family",
    "n",
    "effective_weight",
    "rmse_range",
    "mae_range",
    "rmse_point",
    "mae_point",
    "best_baseline_source",
    "best_baseline_rmse_range",
    "best_baseline_mae_range",
    "rmse_range_delta_vs_baseline",
    "mae_range_delta_vs_baseline",
    "rmse_range_pct_improvement_vs_baseline",
    "mae_range_pct_improvement_vs_baseline",
    "beats_best_baseline",
    "ties_best_baseline",
    "baseline_verdict",
]


@dataclass(frozen=True)
class BehaviorScenario:
    scenario_id: str
    label: str
    as_of_date: str
    transfer_amount: float
    horizon_months: int
    prompt_context: str
    contamination_label: str
    scenario_type: str = "transfer"
    income_loss_pct: float = 0.0
    ui_month: int = 0


BEHAVIOR_SCENARIOS: tuple[BehaviorScenario, ...] = (
    BehaviorScenario(
        "rebate_2001_style",
        "One-time tax rebate in a weak labor market",
        "2001-07-01",
        500.0,
        6,
        (
            "A household receives a one-time federal tax rebate during a weak labor market. "
            "The payment is mailed or deposited over the summer. Durable-goods markets are normal."
        ),
        "historical_pre_model_cutoff_confounded",
    ),
    BehaviorScenario(
        "stimulus_2008_style",
        "One-time stimulus payment during a recession",
        "2008-05-01",
        950.0,
        6,
        (
            "A household receives a one-time fiscal stimulus payment during a recession and credit stress. "
            "Vehicle and durable-goods purchases are possible but credit conditions are tighter than normal."
        ),
        "historical_pre_model_cutoff_confounded",
    ),
    BehaviorScenario(
        "eip_2020_style",
        "Emergency impact payment during a shutdown shock",
        "2020-04-15",
        1200.0,
        1,
        (
            "A household receives an emergency impact payment during an abrupt shutdown shock. "
            "Service consumption opportunities are constrained, unemployment risk is elevated, and liquid buffers matter strongly."
        ),
        "historical_pre_model_cutoff_confounded",
    ),
    BehaviorScenario(
        "lottery_windfall_style",
        "Unexpected medium-sized lottery windfall",
        "date_free_holdout",
        5000.0,
        12,
        (
            "A household unexpectedly receives a medium-sized lottery windfall. "
            "The payment is a transitory gain, not a recurring income change. Durable goods are available."
        ),
        "external_lottery_holdout",
    ),
    BehaviorScenario(
        "small_lottery_windfall_style",
        "Unexpected small lottery windfall",
        "date_free_holdout",
        1500.0,
        12,
        (
            "A household unexpectedly receives a small lottery windfall. "
            "The payment is salient but modest relative to annual income."
        ),
        "external_lottery_holdout",
    ),
    BehaviorScenario(
        "large_lottery_windfall_style",
        "Unexpected large lottery windfall",
        "date_free_holdout",
        50000.0,
        12,
        (
            "A household unexpectedly receives a large lottery windfall. "
            "The payment is much larger than a routine rebate, so gradual saving and durable adjustment are plausible."
        ),
        "external_lottery_holdout",
    ),
    BehaviorScenario(
        "ui_onset_income_loss_style",
        "Job loss with unemployment insurance beginning",
        "date_free_holdout",
        0.0,
        1,
        (
            "A worker loses their job and begins receiving unemployment insurance benefits. "
            "Monthly household income falls, but UI partially replaces lost wages. "
            "The response is the spending drop relative to the household's pre-unemployment monthly spending."
        ),
        "external_ui_exhaustion_holdout",
        "income_loss",
        0.22,
        0,
    ),
    BehaviorScenario(
        "ui_receipt_monthly_path_style",
        "Ongoing unemployment insurance receipt before exhaustion",
        "date_free_holdout",
        0.0,
        1,
        (
            "A worker remains unemployed while receiving regular unemployment insurance benefits. "
            "The job loss is already known, benefits are still arriving, and the task is to estimate the additional "
            "month-to-month spending drift relative to the prior month."
        ),
        "external_ui_exhaustion_holdout",
        "income_loss",
        0.00,
        3,
    ),
    BehaviorScenario(
        "ui_exhaustion_income_loss_style",
        "Predictable unemployment insurance benefit exhaustion",
        "date_free_holdout",
        0.0,
        1,
        (
            "A worker has been unemployed long enough that unemployment insurance benefits now expire. "
            "The benefit exhaustion is predictable and creates a large monthly income drop. "
            "The response is the spending drop relative to spending while benefits were still arriving."
        ),
        "external_ui_exhaustion_holdout",
        "income_loss",
        0.33,
        6,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct household behavior target gate.")
    parser.add_argument("--provider", choices=["codex_cli", "cursor_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--prompt-variant", choices=list(BEHAVIOR_PROMPT_VARIANTS), default="baseline")
    parser.add_argument("--behavior-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--primitive-mode", choices=["auto", "fixture", "replay", "live", "off"], default="auto")
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primitive_mode = args.behavior_mode if args.primitive_mode == "auto" else args.primitive_mode
    if args.behavior_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --behavior-mode live is used")
    if primitive_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when primitive behavior is live")
    if primitive_mode in {"live", "replay"} and primitive_mode != args.behavior_mode:
        raise SystemExit("--primitive-mode live/replay must match --behavior-mode so call accounting and cache semantics stay shared")
    if args.fresh_cache and args.cache_dir:
        raise SystemExit("--fresh-cache and --cache-dir cannot be combined; use one fresh run or one explicit resume cache")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"behavior_gate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "fresh_behavior_cache"
        if args.fresh_cache
        else WORK_ROOT / "behavior_llm_cache"
    )

    manifest: dict[str, Any] = {
        "schema_version": BEHAVIOR_GATE_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "prompt_variant": args.prompt_variant,
        "behavior_prompt_version": (
            BEHAVIOR_PROMPT_DESCRIPTIVE_VERSION if args.prompt_variant == "descriptive" else BEHAVIOR_PROMPT_VERSION
        ),
        "behavior_mode": args.behavior_mode,
        "primitive_mode": primitive_mode,
        "primitive_fixed_policy_version": BEHAVIOR_PRIMITIVE_POLICY_VERSION if primitive_mode != "off" else None,
        "primitive_fixed_policy_parameter_source": "fixed_theory_coefficients_declared_before_scoring" if primitive_mode != "off" else None,
        "primitive_fixed_policy_parameters": primitive_policy_parameters() if primitive_mode != "off" else {},
        "primitive_calibrated_policy_version": BEHAVIOR_PRIMITIVE_CALIBRATED_POLICY_VERSION if primitive_mode != "off" else None,
        "primitive_calibration_split": BEHAVIOR_SELECTION_SPLIT if primitive_mode != "off" else None,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "explicit_cache_dir": bool(args.cache_dir),
        "scf_wave": int(args.scf_wave),
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        scenarios = scenarios_to_frame(BEHAVIOR_SCENARIOS)
        target_catalog = behavior_target_catalog(include_unscored=True)
        targets = behavior_targets_frame(target_scope="aggregate")
        cell_targets = behavior_targets_frame(target_scope="cell")
        llm_client = BehaviorLLMClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.behavior_mode,
            max_live_calls=args.max_live_calls,
            prompt_variant=args.prompt_variant,
        )
        actions = run_behavior_gate(BEHAVIOR_SCENARIOS, type_cells, llm_client=llm_client)
        primitive_actions = pd.DataFrame(columns=actions.columns)
        primitive_fixed_actions = pd.DataFrame(columns=actions.columns)
        primitive_calibrated_actions = pd.DataFrame(columns=actions.columns)
        primitive_payloads = pd.DataFrame()
        primitive_sign_audit = pd.DataFrame()
        primitive_calibrated_policy_parameters: dict[str, Any] = {}
        primitive_calibration_report: dict[str, Any] = {}
        if primitive_mode != "off":
            primitive_fixed_actions, primitive_payloads = run_primitive_behavior_gate(
                BEHAVIOR_SCENARIOS,
                type_cells,
                llm_client=llm_client,
                primitive_mode=primitive_mode,
            )
            selection_targets = targets[targets["evaluation_split"] == BEHAVIOR_SELECTION_SPLIT].copy()
            selection_cell_targets = cell_targets[cell_targets["evaluation_split"] == BEHAVIOR_SELECTION_SPLIT].copy()
            primitive_calibrated_policy_parameters, primitive_calibration_report = calibrate_primitive_policy_parameters(
                BEHAVIOR_SCENARIOS,
                type_cells,
                primitive_payloads,
                selection_targets,
                selection_cell_targets,
            )
            primitive_calibrated_actions = build_primitive_actions_from_payloads(
                BEHAVIOR_SCENARIOS,
                type_cells,
                primitive_payloads,
                source=f"primitive_{args.provider}_{args.model}",
                policy_params=primitive_calibrated_policy_parameters,
            )
            primitive_actions = pd.concat([primitive_fixed_actions, primitive_calibrated_actions], ignore_index=True)
            primitive_sign_audit = primitive_behavior_sign_audit(
                primitive_calibrated_actions,
                primitive_payloads,
                policy_params=primitive_calibrated_policy_parameters,
            )
        controls = run_behavior_controls(BEHAVIOR_SCENARIOS, type_cells)
        ablations = run_behavior_ablations(pd.concat([actions, controls], ignore_index=True))
        all_actions = pd.concat([actions, primitive_actions, controls, ablations], ignore_index=True)
        aggregates = aggregate_behavior_actions(all_actions)
        scores = score_behavior_targets(aggregates, targets)
        cell_joined_errors = join_cell_behavior_target_errors(all_actions, cell_targets)
        cell_scores = score_cell_behavior_targets(all_actions, cell_targets)
        baseline_comparison = build_behavior_baseline_comparison(scores, cell_scores)
        holdout_targets = behavior_targets_frame(target_scope="aggregate", evaluation_split=BEHAVIOR_HOLDOUT_SPLIT)
        holdout_cell_targets = behavior_targets_frame(target_scope="cell", evaluation_split=BEHAVIOR_HOLDOUT_SPLIT)
        holdout_scores = score_behavior_targets(aggregates, holdout_targets)
        holdout_cell_scores = score_cell_behavior_targets(all_actions, holdout_cell_targets)
        holdout_baseline_comparison = build_behavior_baseline_comparison(holdout_scores, holdout_cell_scores)
        ui_holdout_targets = behavior_targets_frame(target_scope="aggregate", evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
        ui_holdout_cell_targets = behavior_targets_frame(target_scope="cell", evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
        ui_holdout_scores = score_behavior_targets(aggregates, ui_holdout_targets)
        ui_holdout_cell_scores = score_cell_behavior_targets(all_actions, ui_holdout_cell_targets)
        ui_holdout_baseline_comparison = build_behavior_baseline_comparison(ui_holdout_scores, ui_holdout_cell_scores)
        prespecified_source = f"llm_{args.provider}_{args.model}{BEHAVIOR_PRESPECIFIED_SUFFIX}"
        raw_llm_source = f"llm_{args.provider}_{args.model}"
        primitive_source = f"primitive_{args.provider}_{args.model}"
        holdout_verdict = prespecified_behavior_holdout_verdict(
            holdout_baseline_comparison,
            candidate_source=prespecified_source,
            target_scope="aggregate",
        )
        primitive_holdout_verdict = prespecified_behavior_holdout_verdict(
            holdout_baseline_comparison,
            candidate_source=primitive_source,
            target_scope="aggregate",
        )
        ui_raw_holdout_verdict = prespecified_behavior_holdout_verdict(
            ui_holdout_baseline_comparison,
            candidate_source=raw_llm_source,
            target_scope="aggregate",
            evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
        )
        ui_primitive_holdout_verdict = prespecified_behavior_holdout_verdict(
            ui_holdout_baseline_comparison,
            candidate_source=primitive_source,
            target_scope="aggregate",
            evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
        )
        scenarios.to_csv(output_dir / "behavior_scenarios.csv", index=False)
        targets.to_csv(output_dir / "behavior_targets.csv", index=False)
        cell_targets.to_csv(output_dir / "behavior_cell_targets.csv", index=False)
        holdout_targets.to_csv(output_dir / "behavior_holdout_targets.csv", index=False)
        holdout_cell_targets.to_csv(output_dir / "behavior_holdout_cell_targets.csv", index=False)
        ui_holdout_targets.to_csv(output_dir / "behavior_ui_exhaustion_holdout_targets.csv", index=False)
        ui_holdout_cell_targets.to_csv(output_dir / "behavior_ui_exhaustion_holdout_cell_targets.csv", index=False)
        target_catalog.to_csv(output_dir / "behavior_target_catalog.csv", index=False)
        type_cells.to_csv(output_dir / "household_type_cells.csv", index=False)
        primitive_actions.to_csv(output_dir / "household_behavior_primitive_actions.csv", index=False)
        primitive_payloads.to_csv(output_dir / "household_behavior_primitives.csv", index=False)
        primitive_sign_audit.to_csv(output_dir / "behavior_primitive_sign_audit.csv", index=False)
        (output_dir / "primitive_policy_calibration.json").write_text(
            json.dumps(
                {
                    "policy_version": BEHAVIOR_PRIMITIVE_CALIBRATED_POLICY_VERSION if primitive_mode != "off" else None,
                    "parameters": primitive_calibrated_policy_parameters,
                    "calibration_report": primitive_calibration_report,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        ablations.to_csv(output_dir / "household_behavior_ablations.csv", index=False)
        all_actions.to_csv(output_dir / "household_behavior_actions.csv", index=False)
        aggregates.to_csv(output_dir / "behavior_aggregates.csv", index=False)
        scores.to_csv(output_dir / "behavior_target_scores.csv", index=False)
        cell_joined_errors.to_csv(output_dir / "behavior_cell_target_joined_errors.csv", index=False)
        cell_scores.to_csv(output_dir / "behavior_cell_target_scores.csv", index=False)
        baseline_comparison.to_csv(output_dir / "behavior_baseline_comparison.csv", index=False)
        holdout_scores.to_csv(output_dir / "behavior_holdout_target_scores.csv", index=False)
        holdout_cell_scores.to_csv(output_dir / "behavior_holdout_cell_target_scores.csv", index=False)
        holdout_baseline_comparison.to_csv(output_dir / "behavior_holdout_baseline_comparison.csv", index=False)
        ui_holdout_scores.to_csv(output_dir / "behavior_ui_exhaustion_holdout_target_scores.csv", index=False)
        ui_holdout_cell_scores.to_csv(output_dir / "behavior_ui_exhaustion_holdout_cell_target_scores.csv", index=False)
        ui_holdout_baseline_comparison.to_csv(output_dir / "behavior_ui_exhaustion_holdout_baseline_comparison.csv", index=False)
        (output_dir / "behavior_llm_raw_records.json").write_text(json.dumps(llm_client.raw_records, indent=2, sort_keys=True), encoding="utf-8")
        manifest.update(
            {
                "status": "ok",
                "scenario_count": len(BEHAVIOR_SCENARIOS),
                "target_count": int(targets.shape[0]),
                "cell_target_count": int(cell_targets.shape[0]),
                "target_catalog_rows": int(target_catalog.shape[0]),
                "unscored_target_gap_count": int((~target_catalog["scored"]).sum()) if "scored" in target_catalog else 0,
                "household_type_status": type_status,
                "household_type_count": int(type_cells.shape[0]),
                "action_rows": int(all_actions.shape[0]),
                "primitive_action_rows": int(primitive_actions.shape[0]),
                "primitive_fixed_action_rows": int(primitive_fixed_actions.shape[0]),
                "primitive_calibrated_action_rows": int(primitive_calibrated_actions.shape[0]),
                "primitive_payload_rows": int(primitive_payloads.shape[0]),
                "primitive_calibrated_policy_parameters": primitive_calibrated_policy_parameters,
                "primitive_calibration_report": primitive_calibration_report,
                "primitive_sign_audit_rows": int(primitive_sign_audit.shape[0]),
                "primitive_sign_audit_passed": bool(primitive_sign_audit["passed"].all()) if not primitive_sign_audit.empty else False,
                "ablation_action_rows": int(ablations.shape[0]),
                "ablation_sources": sorted(ablations["source"].unique().tolist()) if not ablations.empty else [],
                "aggregate_rows": int(aggregates.shape[0]),
                "score_rows": int(scores.shape[0]),
                "cell_score_rows": int(cell_scores.shape[0]),
                "cell_joined_error_rows": int(cell_joined_errors.shape[0]),
                "baseline_comparison_rows": int(baseline_comparison.shape[0]),
                "holdout_split": BEHAVIOR_HOLDOUT_SPLIT,
                "holdout_target_rows": int(holdout_targets.shape[0] + holdout_cell_targets.shape[0]),
                "holdout_score_rows": int(holdout_scores.shape[0] + holdout_cell_scores.shape[0]),
                "holdout_baseline_comparison_rows": int(holdout_baseline_comparison.shape[0]),
                "ui_exhaustion_holdout_split": BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
                "ui_exhaustion_holdout_target_rows": int(ui_holdout_targets.shape[0] + ui_holdout_cell_targets.shape[0]),
                "ui_exhaustion_holdout_score_rows": int(ui_holdout_scores.shape[0] + ui_holdout_cell_scores.shape[0]),
                "ui_exhaustion_holdout_baseline_comparison_rows": int(ui_holdout_baseline_comparison.shape[0]),
                "prespecified_behavior_source": prespecified_source,
                "prespecified_holdout_verdict": holdout_verdict,
                "primitive_behavior_source": primitive_source,
                "primitive_holdout_verdict": primitive_holdout_verdict,
                "ui_exhaustion_raw_llm_holdout_verdict": ui_raw_holdout_verdict,
                "ui_exhaustion_primitive_holdout_verdict": ui_primitive_holdout_verdict,
                "aggregate_baseline_verdict": behavior_baseline_verdict(baseline_comparison, target_scope="aggregate"),
                "cell_baseline_verdict": behavior_baseline_verdict(baseline_comparison, target_scope="cell"),
                "raw_llm_holdout_baseline_verdict": behavior_baseline_verdict(
                    holdout_baseline_comparison,
                    target_scope="aggregate",
                    evaluation_split=BEHAVIOR_HOLDOUT_SPLIT,
                    source_kinds=("llm",),
                ),
                "primitive_holdout_baseline_verdict": behavior_baseline_verdict(
                    holdout_baseline_comparison,
                    target_scope="aggregate",
                    evaluation_split=BEHAVIOR_HOLDOUT_SPLIT,
                    source_kinds=("primitive",),
                ),
                "ui_exhaustion_raw_llm_baseline_verdict": behavior_baseline_verdict(
                    ui_holdout_baseline_comparison,
                    target_scope="aggregate",
                    evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
                    source_kinds=("llm",),
                ),
                "ui_exhaustion_primitive_baseline_verdict": behavior_baseline_verdict(
                    ui_holdout_baseline_comparison,
                    target_scope="aggregate",
                    evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
                    source_kinds=("primitive",),
                ),
                "primitive_aggregate_baseline_verdict": behavior_baseline_verdict(
                    baseline_comparison,
                    target_scope="aggregate",
                    source_kinds=("primitive",),
                ),
                "primitive_cell_baseline_verdict": behavior_baseline_verdict(
                    baseline_comparison,
                    target_scope="cell",
                    source_kinds=("primitive",),
                ),
                "raw_llm_aggregate_baseline_verdict": behavior_baseline_verdict(
                    baseline_comparison,
                    target_scope="aggregate",
                    source_kinds=("llm",),
                ),
                "raw_llm_cell_baseline_verdict": behavior_baseline_verdict(
                    baseline_comparison,
                    target_scope="cell",
                    source_kinds=("llm",),
                ),
                "live_call_count": int(llm_client.live_call_count),
                "cache_hit_count": int(llm_client.cache_hit_count),
                "cache_dir": str(cache_dir.relative_to(Path.cwd()) if cache_dir.is_relative_to(Path.cwd()) else cache_dir),
                "outputs": [
                    "behavior_scenarios.csv",
                    "behavior_targets.csv",
                    "behavior_cell_targets.csv",
                    "behavior_holdout_targets.csv",
                    "behavior_holdout_cell_targets.csv",
                    "behavior_ui_exhaustion_holdout_targets.csv",
                    "behavior_ui_exhaustion_holdout_cell_targets.csv",
                    "behavior_target_catalog.csv",
                    "household_type_cells.csv",
                    "household_behavior_primitive_actions.csv",
                    "household_behavior_primitives.csv",
                    "behavior_primitive_sign_audit.csv",
                    "primitive_policy_calibration.json",
                    "household_behavior_ablations.csv",
                    "household_behavior_actions.csv",
                    "behavior_aggregates.csv",
                    "behavior_target_scores.csv",
                    "behavior_cell_target_joined_errors.csv",
                    "behavior_cell_target_scores.csv",
                    "behavior_baseline_comparison.csv",
                    "behavior_holdout_target_scores.csv",
                    "behavior_holdout_cell_target_scores.csv",
                    "behavior_holdout_baseline_comparison.csv",
                    "behavior_ui_exhaustion_holdout_target_scores.csv",
                    "behavior_ui_exhaustion_holdout_cell_target_scores.csv",
                    "behavior_ui_exhaustion_holdout_baseline_comparison.csv",
                    "behavior_llm_raw_records.json",
                    "behavior_gate_report.md",
                ],
            }
        )
        report = build_behavior_gate_report(
            manifest,
            scenarios,
            targets,
            aggregates,
            scores,
            target_catalog=target_catalog,
            cell_targets=cell_targets,
            cell_scores=cell_scores,
            cell_joined_errors=cell_joined_errors,
            baseline_comparison=baseline_comparison,
            holdout_baseline_comparison=holdout_baseline_comparison,
            holdout_targets=holdout_targets,
            ui_holdout_baseline_comparison=ui_holdout_baseline_comparison,
            ui_holdout_targets=ui_holdout_targets,
            primitive_sign_audit=primitive_sign_audit,
        )
        (output_dir / "behavior_gate_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


class BehaviorLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str,
        max_live_calls: int,
        prompt_variant: str = "baseline",
    ):
        if prompt_variant not in BEHAVIOR_PROMPT_VARIANTS:
            raise ValueError(f"Unsupported behavior prompt variant: {prompt_variant}")
        self.provider = provider
        self.model = model
        self.mode = mode
        self.prompt_variant = prompt_variant
        self._client = AgentLLMClient(provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls)
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._client.cache_hit_count

    def behavior_panel(self, scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_behavior_payload(scenario, type_cells),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = behavior_prompt(scenario, type_cells, variant=self.prompt_variant)
            data = self._client._codex_call(prompt, f"behavior_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}")
        normalized = normalize_behavior_payload(scenario, type_cells, data)
        self.raw_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "record_type": "behavior_allocations",
                "prompt_variant": self.prompt_variant,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized

    def primitive_panel(self, scenario: BehaviorScenario, type_cells: pd.DataFrame, *, primitive_mode: str) -> dict[str, Any]:
        if primitive_mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_behavior_primitive_payload(scenario, type_cells),
                "cache_hit": True,
                "cache_path": None,
            }
        elif primitive_mode in {"live", "replay"}:
            prompt = behavior_primitive_prompt(scenario, type_cells)
            data = self._client._codex_call(
                prompt,
                f"behavior_primitives_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}",
            )
        else:
            raise LLMUnavailable(f"Unsupported primitive mode: {primitive_mode}")
        normalized = normalize_behavior_primitive_payload(scenario, type_cells, data)
        self.raw_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "record_type": "behavior_primitives",
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized


def run_behavior_gate(scenarios: Iterable[BehaviorScenario], type_cells: pd.DataFrame, *, llm_client: BehaviorLLMClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        payload = llm_client.behavior_panel(scenario, type_cells)
        for _, type_cell in type_cells.iterrows():
            response = payload["household_by_type"][str(type_cell["type_id"])]
            rows.append(_behavior_action_row(scenario, type_cell, response, source=f"llm_{llm_client.provider}_{llm_client.model}"))
    return pd.DataFrame(rows)


def run_primitive_behavior_gate(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    *,
    llm_client: BehaviorLLMClient,
    primitive_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_rows: list[dict[str, Any]] = []
    primitive_rows: list[dict[str, Any]] = []
    action_source = f"primitive_fixed_{llm_client.provider}_{llm_client.model}"
    payload_source = f"primitive_payload_{llm_client.provider}_{llm_client.model}"
    fixed_params = primitive_policy_parameters()
    for scenario in scenarios:
        payload = llm_client.primitive_panel(scenario, type_cells, primitive_mode=primitive_mode)
        for _, type_cell in type_cells.iterrows():
            primitive = payload["primitives_by_type"][str(type_cell["type_id"])]
            action = primitive_policy_action(scenario, type_cell, primitive, policy_params=fixed_params)
            action_rows.append(_behavior_action_row(scenario, type_cell, action, source=action_source))
            primitive_rows.append(
                {
                    "schema_version": BEHAVIOR_PRIMITIVE_PROMPT_VERSION,
                    "scenario_id": scenario.scenario_id,
                    "source": payload_source,
                    "type_id": str(type_cell["type_id"]),
                    "population_weight": float(type_cell["population_weight"]),
                    "liquidity_group": _liquidity_group(type_cell),
                    **{field: float(primitive[field]) for field in BEHAVIOR_PRIMITIVE_FIELDS},
                    "reason": str(primitive.get("reason", ""))[:300],
                }
            )
    return pd.DataFrame(action_rows), pd.DataFrame(primitive_rows)


def build_primitive_actions_from_payloads(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    primitive_payloads: pd.DataFrame,
    *,
    source: str,
    policy_params: dict[str, Any],
) -> pd.DataFrame:
    if primitive_payloads.empty:
        return pd.DataFrame()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    type_by_id = {str(row["type_id"]): row for _, row in type_cells.iterrows()}
    rows: list[dict[str, Any]] = []
    for _, primitive_row in primitive_payloads.iterrows():
        scenario_id = str(primitive_row["scenario_id"])
        type_id = str(primitive_row["type_id"])
        scenario = scenario_by_id.get(scenario_id)
        type_cell = type_by_id.get(type_id)
        if scenario is None or type_cell is None:
            continue
        primitive = {field: float(primitive_row[field]) for field in BEHAVIOR_PRIMITIVE_FIELDS}
        primitive["confidence"] = float(primitive_row.get("confidence", 0.5))
        action = primitive_policy_action(scenario, type_cell, primitive, policy_params=policy_params)
        rows.append(_behavior_action_row(scenario, type_cell, action, source=source))
    return pd.DataFrame(rows)


def run_behavior_controls(scenarios: Iterable[BehaviorScenario], type_cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controls = {
        "liquidity_rule": _liquidity_rule_response,
        "flat_30pct_rule": _flat_rule_response,
        "permanent_income_rule": _permanent_income_response,
    }
    for scenario in scenarios:
        for source, fn in controls.items():
            for _, type_cell in type_cells.iterrows():
                rows.append(_behavior_action_row(scenario, type_cell, fn(scenario, type_cell), source=source))
    return pd.DataFrame(rows)


def run_behavior_ablations(actions: pd.DataFrame, *, baseline_source: str = "liquidity_rule") -> pd.DataFrame:
    if actions.empty or baseline_source not in set(actions["source"]):
        return pd.DataFrame(columns=actions.columns)
    llm_sources = sorted(
        source
        for source in actions["source"].dropna().unique()
        if str(source).startswith("llm_") and "__" not in str(source)
    )
    if not llm_sources:
        return pd.DataFrame(columns=actions.columns)

    baseline = actions[actions["source"] == baseline_source].copy()
    rows: list[pd.DataFrame] = []
    for llm_source in llm_sources:
        llm = actions[actions["source"] == llm_source].copy()
        merged = llm.merge(
            baseline,
            on=["scenario_id", "type_id"],
            suffixes=("_llm", "_base"),
            validate="one_to_one",
        )
        if merged.empty:
            continue
        rows.append(
            _blend_behavior_sources(
                merged,
                source=f"{llm_source}__liquidity_prior_75",
                llm_weight=0.25,
                reason="75 percent liquidity rule, 25 percent LLM action",
            )
        )
        rows.append(
            _blend_behavior_sources(
                merged,
                source=f"{llm_source}__liquidity_prior_50",
                llm_weight=0.50,
                reason="50 percent liquidity rule, 50 percent LLM action",
            )
        )
        rows.append(
            _residual_behavior_sources(
                merged,
                source=f"{llm_source}__residual_over_liquidity",
                reason="liquidity-rule group means plus LLM within-group residuals",
            )
        )
    if not rows:
        return pd.DataFrame(columns=actions.columns)
    return pd.concat(rows, ignore_index=True)[list(actions.columns)]


def _blend_behavior_sources(merged: pd.DataFrame, *, source: str, llm_weight: float, reason: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_weight = 1.0 - llm_weight
    for _, row in merged.iterrows():
        output = _base_action_from_merged(row, source=source, reason=reason)
        for column in BEHAVIOR_SHARE_COLUMNS:
            output[column] = base_weight * float(row[f"{column}_base"]) + llm_weight * float(
                row[f"{column}_llm"]
            )
        output["confidence"] = base_weight * float(row.get("confidence_base", 0.5)) + llm_weight * float(
            row.get("confidence_llm", 0.5)
        )
        rows.append(_normalize_action_row(output))
    return pd.DataFrame(rows)


def _residual_behavior_sources(merged: pd.DataFrame, *, source: str, reason: str) -> pd.DataFrame:
    frame = merged.copy()
    rows: list[dict[str, Any]] = []
    for column in BEHAVIOR_SHARE_COLUMNS:
        frame[f"{column}_residual"] = frame[f"{column}_llm"].astype(float) - frame[f"{column}_base"].astype(
            float
        )
        frame[f"{column}_centered_residual"] = frame[f"{column}_residual"] - frame.groupby(
            ["scenario_id", "liquidity_group_llm"]
        )[f"{column}_residual"].transform(
            lambda values: _weighted_mean(values, frame.loc[values.index, "population_weight_llm"])
        )
    for _, row in frame.iterrows():
        output = _base_action_from_merged(row, source=source, reason=reason)
        for column in BEHAVIOR_SHARE_COLUMNS:
            output[column] = float(row[f"{column}_base"]) + float(row[f"{column}_centered_residual"])
        output["confidence"] = min(float(row.get("confidence_base", 0.5)), float(row.get("confidence_llm", 0.5)))
        rows.append(_normalize_action_row(output))
    return pd.DataFrame(rows)


def _base_action_from_merged(row: pd.Series, *, source: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": row["schema_version_llm"],
        "scenario_id": row["scenario_id"],
        "scenario_type": row.get("scenario_type_llm", row.get("scenario_type", "transfer")),
        "source": source,
        "type_id": row["type_id"],
        "population_weight": float(row["population_weight_llm"]),
        "liquidity_group": row["liquidity_group_llm"],
        "transfer_amount": float(row["transfer_amount_llm"]),
        "income_loss_pct": float(row.get("income_loss_pct_llm", row.get("income_loss_pct", 0.0))),
        "confidence": 0.5,
        "reason": reason,
    }


def _normalize_action_row(row: dict[str, Any]) -> dict[str, Any]:
    total = float(np.clip(row["total_spending_share"], 0.0, 1.0))
    debt = float(np.clip(row["debt_repayment_share"], 0.0, 1.0))
    liquid = float(np.clip(row["liquid_saving_share"], 0.0, 1.0))
    if total + debt + liquid > 1.0:
        remainder = max(0.0, 1.0 - total)
        non_spending = debt + liquid
        if non_spending > 0:
            debt = remainder * debt / non_spending
            liquid = remainder * liquid / non_spending
        else:
            debt = 0.0
            liquid = remainder

    nondurable = float(np.clip(row["nondurable_spending_share"], 0.0, total))
    durable = float(np.clip(row["durable_spending_share"], 0.0, total))
    if nondurable + durable > total and nondurable + durable > 0:
        scale = total / (nondurable + durable)
        nondurable *= scale
        durable *= scale
    if total > 0 and nondurable + durable == 0:
        nondurable = 0.8 * total
        durable = 0.2 * total

    row.update(
        {
            "total_spending_share": total,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": debt,
            "liquid_saving_share": liquid,
            "confidence": float(np.clip(row.get("confidence", 0.5), 0.0, 1.0)),
        }
    )
    return row


def behavior_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame, *, variant: str = "baseline") -> str:
    if variant not in BEHAVIOR_PROMPT_VARIANTS:
        raise ValueError(f"Unsupported behavior prompt variant: {variant}")
    if scenario.scenario_type == "income_loss":
        return behavior_income_loss_prompt(scenario, type_cells, variant=variant)
    payload = {
        "prompt_version": BEHAVIOR_PROMPT_DESCRIPTIVE_VERSION if variant == "descriptive" else BEHAVIOR_PROMPT_VERSION,
        "task": "Allocate a one-time household transfer into spending, debt repayment, and liquid saving.",
        "as_of_rule": "Use only the scenario and type-cell information below. Do not cite realized study estimates.",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "as_of_date": scenario.as_of_date,
            "transfer_amount": scenario.transfer_amount,
            "horizon_months": scenario.horizon_months,
            "context": scenario.prompt_context,
        },
        "household_type_cells": [
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": round_or_none(row["population_weight"]),
                "annual_income": round_or_none(row["annual_income"]),
                "liquid_assets": round_or_none(row["liquid_assets"]),
                "illiquid_assets": round_or_none(row["illiquid_assets"]),
                "debt": round_or_none(row["debt"]),
                "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
            }
            for _, row in type_cells.iterrows()
        ],
        "required_response": {
            "household_actions": [
                {
                    "type_id": "one supplied type_id",
                    "total_spending_share": "0 to 1 share of transfer spent over the scenario horizon",
                    "nondurable_spending_share": "0 to 1 share of transfer spent on nondurables/services",
                    "durable_spending_share": "0 to 1 share of transfer spent on durables",
                    "debt_repayment_share": "0 to 1 share of transfer used to repay debt",
                    "liquid_saving_share": "0 to 1 share of transfer retained as liquid buffer",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    if variant == "descriptive":
        payload["behavior_framing"] = BEHAVIOR_DESCRIPTIVE_FRAMING
    return json.dumps(payload, indent=2, sort_keys=True)


def behavior_income_loss_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame, *, variant: str = "baseline") -> str:
    payload = {
        "prompt_version": BEHAVIOR_PROMPT_DESCRIPTIVE_VERSION if variant == "descriptive" else BEHAVIOR_PROMPT_VERSION,
        "task": "Estimate bounded household spending and balance-sheet responses to an income-loss scenario.",
        "as_of_rule": "Use only the scenario and type-cell information below. Do not cite realized study estimates.",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "as_of_date": scenario.as_of_date,
            "scenario_type": scenario.scenario_type,
            "income_loss_pct": scenario.income_loss_pct,
            "ui_month": scenario.ui_month,
            "horizon_months": scenario.horizon_months,
            "context": scenario.prompt_context,
        },
        "household_type_cells": [
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": round_or_none(row["population_weight"]),
                "annual_income": round_or_none(row["annual_income"]),
                "liquid_assets": round_or_none(row["liquid_assets"]),
                "illiquid_assets": round_or_none(row["illiquid_assets"]),
                "debt": round_or_none(row["debt"]),
                "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
            }
            for _, row in type_cells.iterrows()
        ],
        "required_response": {
            "household_actions": [
                {
                    "type_id": "one supplied type_id",
                    "total_spending_share": "0 to 1 spending-drop share relative to the relevant pre-shock monthly spending baseline",
                    "nondurable_spending_share": "0 to 1 nondurable-spending-drop share relative to the relevant pre-shock monthly nondurable baseline",
                    "durable_spending_share": "0 to 1 durable-spending-drop share relative to the relevant pre-shock monthly durable baseline",
                    "debt_repayment_share": "0 to 1 debt-repayment reduction or missed-payment adjustment share",
                    "liquid_saving_share": "0 to 1 liquid-saving reduction or buffer drawdown share",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    if variant == "descriptive":
        payload["behavior_framing"] = BEHAVIOR_DESCRIPTIVE_FRAMING
    return json.dumps(payload, indent=2, sort_keys=True)


def behavior_primitive_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> str:
    if scenario.scenario_type == "income_loss":
        return behavior_income_loss_primitive_prompt(scenario, type_cells)
    payload = {
        "prompt_version": BEHAVIOR_PRIMITIVE_PROMPT_VERSION,
        "task": "Infer household behavioral primitives for a one-time transfer. Do not output spending, saving, or debt allocation shares.",
        "as_of_rule": "Use only the scenario and type-cell information below. Do not cite realized study estimates.",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "as_of_date": scenario.as_of_date,
            "transfer_amount": scenario.transfer_amount,
            "horizon_months": scenario.horizon_months,
            "context": scenario.prompt_context,
        },
        "household_type_cells": [
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": round_or_none(row["population_weight"]),
                "annual_income": round_or_none(row["annual_income"]),
                "liquid_assets": round_or_none(row["liquid_assets"]),
                "illiquid_assets": round_or_none(row["illiquid_assets"]),
                "debt": round_or_none(row["debt"]),
                "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
            }
            for _, row in type_cells.iterrows()
        ],
        "required_response": {
            "household_primitives": [
                {
                    "type_id": "one supplied type_id",
                    "perceived_job_loss_risk_pp": "0 to 100 subjective job-loss risk in percentage points",
                    "expected_income_growth_pct": "-20 to 20 expected income growth over the scenario horizon",
                    "precautionary_saving_motive": "0 to 1",
                    "liquidity_stress": "0 to 1",
                    "debt_repayment_urgency": "0 to 1",
                    "durable_purchase_pull_forward": "0 to 1",
                    "shock_size_normalized": "0 to 1, where 0 is a small routine payment and 1 is life-changing relative to income",
                    "shock_size_log_income_ratio": "-4 to 2, log10(transfer_amount / annual_income); keep prize-size gradation instead of clipping large windfalls",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def behavior_income_loss_primitive_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> str:
    payload = {
        "prompt_version": BEHAVIOR_PRIMITIVE_PROMPT_VERSION,
        "task": "Infer household behavioral primitives for an income-loss scenario. Do not output spending, saving, or debt allocation shares.",
        "as_of_rule": "Use only the scenario and type-cell information below. Do not cite realized study estimates.",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "as_of_date": scenario.as_of_date,
            "scenario_type": scenario.scenario_type,
            "income_loss_pct": scenario.income_loss_pct,
            "ui_month": scenario.ui_month,
            "horizon_months": scenario.horizon_months,
            "context": scenario.prompt_context,
        },
        "household_type_cells": [
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": round_or_none(row["population_weight"]),
                "annual_income": round_or_none(row["annual_income"]),
                "liquid_assets": round_or_none(row["liquid_assets"]),
                "illiquid_assets": round_or_none(row["illiquid_assets"]),
                "debt": round_or_none(row["debt"]),
                "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
            }
            for _, row in type_cells.iterrows()
        ],
        "required_response": {
            "household_primitives": [
                {
                    "type_id": "one supplied type_id",
                    "perceived_job_loss_risk_pp": "0 to 100 subjective job-loss or continued-unemployment risk in percentage points",
                    "expected_income_growth_pct": "-20 to 20 expected income growth over the scenario horizon",
                    "precautionary_saving_motive": "0 to 1",
                    "liquidity_stress": "0 to 1",
                    "debt_repayment_urgency": "0 to 1",
                    "durable_purchase_pull_forward": "0 to 1, usually low for income-loss scenarios",
                    "shock_size_normalized": "0 to 1 severity of the income-loss shock",
                    "shock_size_log_income_ratio": "-4 to 2, log10(abs(monthly_income_loss) / annual_income)",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_behavior_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": BEHAVIOR_PROMPT_VERSION,
        "household_actions": [
            {
                "type_id": str(row["type_id"]),
                **_liquidity_rule_response(scenario, row),
                "reason": "deterministic liquidity fixture",
            }
            for _, row in type_cells.iterrows()
        ],
    }


def fixture_behavior_primitive_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": BEHAVIOR_PRIMITIVE_PROMPT_VERSION,
        "household_primitives": [
            {
                "type_id": str(row["type_id"]),
                **fixture_behavior_primitives(scenario, row),
                "reason": "deterministic primitive fixture",
            }
            for _, row in type_cells.iterrows()
        ],
    }


def normalize_behavior_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    actions = payload.get("household_actions")
    if not isinstance(actions, list):
        raise LLMUnavailable(f"Behavior payload for {scenario.scenario_id} is missing household_actions list")
    expected_ids = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        type_id = str(action.get("type_id", ""))
        if type_id not in expected_ids:
            continue
        total = _bounded_share(action, "total_spending_share")
        nondurable = _bounded_share(action, "nondurable_spending_share")
        durable = _bounded_share(action, "durable_spending_share")
        if nondurable + durable > total and nondurable + durable > 0:
            scale = total / (nondurable + durable)
            nondurable *= scale
            durable *= scale
        debt = _bounded_share(action, "debt_repayment_share")
        liquid = _bounded_share(action, "liquid_saving_share")
        total_use = total + debt + liquid
        if total_use > 1.0 and total_use > 0:
            debt *= (1.0 - total) / max(debt + liquid, 1e-12)
            liquid = max(0.0, 1.0 - total - debt)
        by_type[type_id] = {
            "total_spending_share": total,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": debt,
            "liquid_saving_share": liquid,
            "confidence": _bounded_share(action, "confidence"),
            "reason": str(action.get("reason", ""))[:300],
        }
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Behavior payload for {scenario.scenario_id} is missing type ids: {', '.join(missing)}")
    return {"household_by_type": by_type}


def normalize_behavior_primitive_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    if "household_actions" in payload:
        raise LLMUnavailable(
            f"Primitive behavior payload for {scenario.scenario_id} must not include household_actions"
        )
    primitives = payload.get("household_primitives")
    if not isinstance(primitives, list):
        raise LLMUnavailable(f"Primitive behavior payload for {scenario.scenario_id} is missing household_primitives list")
    expected_ids = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, Any]] = {}
    for primitive in primitives:
        if not isinstance(primitive, dict):
            continue
        type_id = str(primitive.get("type_id", ""))
        if type_id not in expected_ids:
            continue
        mutation_fields = sorted(set(BEHAVIOR_SHARE_COLUMNS) & set(primitive))
        if mutation_fields:
            raise LLMUnavailable(
                f"Primitive behavior payload for {scenario.scenario_id} includes final allocation fields: "
                f"{', '.join(mutation_fields)}"
            )
        by_type[type_id] = {
            "perceived_job_loss_risk_pp": bounded_number(primitive, "perceived_job_loss_risk_pp", 0.0, 100.0),
            "expected_income_growth_pct": bounded_number(primitive, "expected_income_growth_pct", -20.0, 20.0),
            "precautionary_saving_motive": bounded_number(primitive, "precautionary_saving_motive", 0.0, 1.0),
            "liquidity_stress": bounded_number(primitive, "liquidity_stress", 0.0, 1.0),
            "debt_repayment_urgency": bounded_number(primitive, "debt_repayment_urgency", 0.0, 1.0),
            "durable_purchase_pull_forward": bounded_number(primitive, "durable_purchase_pull_forward", 0.0, 1.0),
            "shock_size_normalized": bounded_number(primitive, "shock_size_normalized", 0.0, 1.0),
            "shock_size_log_income_ratio": bounded_number(primitive, "shock_size_log_income_ratio", -4.0, 2.0),
            "confidence": bounded_number(primitive, "confidence", 0.0, 1.0),
            "reason": str(primitive.get("reason", ""))[:300],
        }
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Primitive behavior payload for {scenario.scenario_id} is missing type ids: {', '.join(missing)}")
    return {"primitives_by_type": by_type}


def fixture_behavior_primitives(scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, float]:
    buffer_months = float(type_cell.get("liquid_buffer_months", 2.0))
    debt_ratio = float(type_cell.get("debt_to_asset", 0.0))
    shock_income_ratio = _scenario_shock_income_ratio(scenario, type_cell)
    if scenario.scenario_type == "income_loss":
        severity = float(np.clip(float(scenario.income_loss_pct) / 0.35, 0.0, 1.0))
        exhaustion = 1.0 if scenario.scenario_id == "ui_exhaustion_income_loss_style" else 0.0
        ongoing = 1.0 if scenario.scenario_id == "ui_receipt_monthly_path_style" else 0.0
        return {
            "perceived_job_loss_risk_pp": float(np.clip(45.0 + 18.0 * exhaustion - 12.0 * ongoing - 0.45 * buffer_months, 0.0, 100.0)),
            "expected_income_growth_pct": float(np.clip(-3.0 - 8.0 * severity - 3.0 * exhaustion + 2.0 * ongoing, -20.0, 20.0)),
            "precautionary_saving_motive": float(np.clip(0.32 + 0.34 * severity + 0.16 * exhaustion + 0.03 * buffer_months, 0.0, 1.0)),
            "liquidity_stress": float(np.clip(0.62 + 0.22 * severity + 0.12 * exhaustion - 0.10 * buffer_months + 0.08 * debt_ratio, 0.0, 1.0)),
            "debt_repayment_urgency": float(np.clip(0.22 + 0.38 * debt_ratio + 0.10 * severity, 0.0, 1.0)),
            "durable_purchase_pull_forward": float(np.clip(0.04 - 0.03 * severity, 0.0, 1.0)),
            "shock_size_normalized": float(np.clip(severity, 0.0, 1.0)),
            "shock_size_log_income_ratio": float(np.clip(np.log10(max(shock_income_ratio, 1e-4)), -4.0, 2.0)),
            "confidence": 0.60,
        }
    shutdown = 1.0 if scenario.scenario_id == "eip_2020_style" else 0.0
    recession = 1.0 if scenario.scenario_id in {"stimulus_2008_style", "eip_2020_style"} else 0.0
    return {
        "perceived_job_loss_risk_pp": float(np.clip(4.0 + 10.0 * recession + 18.0 * shutdown - 0.35 * buffer_months, 0.0, 100.0)),
        "expected_income_growth_pct": float(np.clip(1.5 - 2.5 * recession - 3.0 * shutdown, -20.0, 20.0)),
        "precautionary_saving_motive": float(np.clip(0.20 + 0.12 * recession + 0.16 * shutdown + 0.04 * buffer_months, 0.0, 1.0)),
        "liquidity_stress": float(np.clip(0.70 - 0.10 * buffer_months + 0.08 * debt_ratio, 0.0, 1.0)),
        "debt_repayment_urgency": float(np.clip(0.12 + 0.45 * debt_ratio, 0.0, 1.0)),
        "durable_purchase_pull_forward": float(np.clip(0.16 + 0.10 * (scenario.horizon_months >= 6) - 0.15 * shutdown, 0.0, 1.0)),
        "shock_size_normalized": float(np.clip(shock_income_ratio / 0.50, 0.0, 1.0)),
        "shock_size_log_income_ratio": float(np.clip(np.log10(max(shock_income_ratio, 1e-4)), -4.0, 2.0)),
        "confidence": 0.60,
    }


def primitive_policy_parameters() -> dict[str, float | str]:
    return {
        "source": "fixed_theory_coefficients_no_target_fit",
        "base_spending_share": 0.24,
        "liquidity_stress_to_spending": 0.34,
        "durable_pull_to_spending": 0.10,
        "positive_income_growth_to_spending": 0.05,
        "precaution_to_spending": -0.16,
        "job_risk_to_spending": -0.10,
        "shock_log_to_spending": -0.12,
        "base_debt_repayment_share": 0.06,
        "debt_urgency_to_debt_repayment": 0.34,
        "job_risk_to_debt_repayment": 0.04,
        "base_durable_fraction": 0.18,
        "durable_pull_to_durable_fraction": 0.28,
        "shutdown_durable_fraction_penalty": -0.12,
    }


def primitive_policy_action(
    scenario: BehaviorScenario,
    type_cell: pd.Series,
    primitive: dict[str, Any],
    *,
    policy_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        return primitive_income_loss_policy_action(scenario, type_cell, primitive, policy_params=policy_params)
    params = policy_params if policy_params is not None else primitive_policy_parameters()
    job_risk = float(primitive["perceived_job_loss_risk_pp"]) / 100.0
    income_growth = float(primitive["expected_income_growth_pct"]) / 10.0
    liquidity_stress = float(primitive["liquidity_stress"])
    precaution = float(primitive["precautionary_saving_motive"])
    debt_urgency = float(primitive["debt_repayment_urgency"])
    durable_pull = float(primitive["durable_purchase_pull_forward"])
    shock_size_log = float(primitive["shock_size_log_income_ratio"])
    shutdown = 1.0 if scenario.scenario_id == "eip_2020_style" else 0.0

    total = (
        float(params["base_spending_share"])
        + float(params["liquidity_stress_to_spending"]) * liquidity_stress
        + float(params["durable_pull_to_spending"]) * durable_pull
        + float(params["positive_income_growth_to_spending"]) * max(income_growth, 0.0)
        + float(params["precaution_to_spending"]) * precaution
        + float(params["job_risk_to_spending"]) * job_risk
        + float(params["shock_log_to_spending"]) * shock_size_log
    )
    debt = (
        float(params["base_debt_repayment_share"])
        + float(params["debt_urgency_to_debt_repayment"]) * debt_urgency
        + float(params["job_risk_to_debt_repayment"]) * job_risk
    )
    durable_fraction = (
        float(params["base_durable_fraction"])
        + float(params["durable_pull_to_durable_fraction"]) * durable_pull
        + float(params["shutdown_durable_fraction_penalty"]) * shutdown
    )
    total = float(np.clip(total, 0.02, 0.85))
    debt = float(np.clip(debt, 0.0, 0.50))
    durable_fraction = float(np.clip(durable_fraction, 0.02, 0.65))
    durable = total * durable_fraction
    nondurable = max(0.0, total - durable)
    liquid = max(0.0, 1.0 - total - debt)
    return _normalize_action_row(
        {
            "total_spending_share": total,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": debt,
            "liquid_saving_share": liquid,
            "confidence": float(primitive["confidence"]),
            "reason": (
                f"{params.get('source', 'primitive_policy')}: liquidity stress raises spending; "
                "precaution and job risk raise saving; debt urgency raises repayment; "
                "larger windfalls lower immediate MPC through log shock size"
            ),
        }
    )


def primitive_income_loss_policy_action(
    scenario: BehaviorScenario,
    type_cell: pd.Series,
    primitive: dict[str, Any],
    *,
    policy_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = policy_params if policy_params is not None else primitive_policy_parameters()
    job_risk = float(primitive["perceived_job_loss_risk_pp"]) / 100.0
    income_loss = max(0.0, -float(primitive["expected_income_growth_pct"]) / 20.0)
    liquidity_stress = float(primitive["liquidity_stress"])
    precaution = float(primitive["precautionary_saving_motive"])
    debt_urgency = float(primitive["debt_repayment_urgency"])
    shock_size = float(primitive["shock_size_normalized"])
    exhaustion = 1.0 if scenario.scenario_id == "ui_exhaustion_income_loss_style" else 0.0
    ongoing = 1.0 if scenario.scenario_id == "ui_receipt_monthly_path_style" else 0.0

    total_drop = (
        0.010
        + 0.045 * liquidity_stress
        + 0.035 * precaution
        + 0.030 * job_risk
        + 0.080 * income_loss
        + 0.035 * shock_size
        + 0.040 * exhaustion
        - 0.060 * ongoing
    )
    debt_adjustment = 0.020 + 0.090 * debt_urgency + 0.030 * shock_size
    buffer_adjustment = 0.040 + 0.100 * liquidity_stress + 0.050 * precaution
    total_drop = float(np.clip(total_drop, 0.0, 0.30))
    nondurable = float(np.clip(0.72 * total_drop, 0.0, total_drop))
    durable = max(0.0, total_drop - nondurable)
    return _normalize_action_row(
        {
            "total_spending_share": total_drop,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": float(np.clip(debt_adjustment, 0.0, 0.50)),
            "liquid_saving_share": float(np.clip(buffer_adjustment, 0.0, 0.50)),
            "confidence": float(primitive["confidence"]),
            "reason": (
                f"{params.get('source', 'primitive_income_loss_policy')}: income loss, liquidity stress, "
                "precaution, and exhaustion raise spending-drop responses; ongoing UI receipt dampens drift"
            ),
        }
    )


PRIMITIVE_CALIBRATION_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "base_spending_share": (0.02, 0.60),
    "liquidity_stress_to_spending": (0.00, 0.90),
    "durable_pull_to_spending": (0.00, 0.30),
    "positive_income_growth_to_spending": (0.00, 0.20),
    "precaution_to_spending": (-0.60, 0.00),
    "job_risk_to_spending": (-0.60, -0.05),
    "shock_log_to_spending": (-0.60, 0.00),
    "base_debt_repayment_share": (0.00, 0.35),
    "debt_urgency_to_debt_repayment": (0.02, 0.90),
    "job_risk_to_debt_repayment": (0.00, 0.20),
}


def calibrate_primitive_policy_parameters(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    primitive_payloads: pd.DataFrame,
    aggregate_targets: pd.DataFrame,
    cell_targets: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = primitive_policy_parameters()
    params = dict(fixed)
    params["source"] = "selection_calibrated_from_behavior_selection_v1"
    if primitive_payloads.empty or aggregate_targets.empty:
        report = {
            "status": "skipped",
            "reason": "missing primitive payloads or selection targets",
            "calibration_split": BEHAVIOR_SELECTION_SPLIT,
            "objective_surface": "aggregate_selection_targets_only",
            "objective_before": float("nan"),
            "objective_after": float("nan"),
            "parameter_bounds": PRIMITIVE_CALIBRATION_PARAMETER_BOUNDS,
        }
        return params, report

    before = _primitive_policy_objective(
        scenarios,
        type_cells,
        primitive_payloads,
        fixed,
        aggregate_targets,
        cell_targets,
    )
    best = _primitive_policy_objective(
        scenarios,
        type_cells,
        primitive_payloads,
        params,
        aggregate_targets,
        cell_targets,
    )
    trace: list[dict[str, Any]] = [{"iteration": 0, "parameter": "initial", "objective": best["objective"]}]
    for iteration in range(1, 3):
        improved = False
        for name, bounds in PRIMITIVE_CALIBRATION_PARAMETER_BOUNDS.items():
            low, high = bounds
            center = float(params[name])
            width = (high - low) / float(2 ** (iteration + 1))
            candidates = np.linspace(max(low, center - width), min(high, center + width), 5)
            candidates = np.unique(np.append(candidates, [center, low, high]))
            local_best_value = center
            local_best = best
            for candidate in candidates:
                candidate_params = dict(params)
                candidate_params[name] = float(candidate)
                result = _primitive_policy_objective(
                    scenarios,
                    type_cells,
                    primitive_payloads,
                    candidate_params,
                    aggregate_targets,
                    cell_targets,
                )
                if result["objective"] < local_best["objective"] - 1e-10:
                    local_best = result
                    local_best_value = float(candidate)
            if local_best_value != center:
                params[name] = local_best_value
                best = local_best
                improved = True
                trace.append(
                    {
                        "iteration": iteration,
                        "parameter": name,
                        "value": local_best_value,
                        "objective": best["objective"],
                    }
                )
        if not improved:
            break

    report = {
        "status": "ok",
        "calibration_split": BEHAVIOR_SELECTION_SPLIT,
        "objective_surface": "aggregate_selection_targets_only",
        "objective_before": before["objective"],
        "objective_after": best["objective"],
        "aggregate_rmse_before": before["aggregate_rmse"],
        "aggregate_rmse_after": best["aggregate_rmse"],
        "cell_rmse_before": before["cell_rmse"],
        "cell_rmse_after": best["cell_rmse"],
        "selection_aggregate_target_rows": int(aggregate_targets.shape[0]),
        "selection_cell_target_rows": int(cell_targets.shape[0]),
        "parameter_bounds": PRIMITIVE_CALIBRATION_PARAMETER_BOUNDS,
        "trace": trace[-20:],
    }
    return params, report


def _primitive_policy_objective(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    primitive_payloads: pd.DataFrame,
    policy_params: dict[str, Any],
    aggregate_targets: pd.DataFrame,
    cell_targets: pd.DataFrame,
) -> dict[str, float]:
    source = "primitive_calibration_candidate"
    actions = build_primitive_actions_from_payloads(
        scenarios,
        type_cells,
        primitive_payloads,
        source=source,
        policy_params=policy_params,
    )
    aggregates = aggregate_behavior_actions(actions)
    scores = score_behavior_targets(aggregates, aggregate_targets)
    cell_scores = score_cell_behavior_targets(actions, cell_targets)
    aggregate_rmse = _score_lookup_rmse(scores, source=source, target_scope="aggregate", evaluation_split=BEHAVIOR_SELECTION_SPLIT)
    cell_rmse = _score_lookup_rmse(cell_scores, source=source, target_scope="cell", evaluation_split=BEHAVIOR_SELECTION_SPLIT)
    objective = aggregate_rmse if np.isfinite(aggregate_rmse) else float("inf")
    return {
        "objective": objective,
        "aggregate_rmse": aggregate_rmse,
        "cell_rmse": cell_rmse,
    }


def _score_lookup_rmse(scores: pd.DataFrame, *, source: str, target_scope: str, evaluation_split: str) -> float:
    if scores.empty:
        return float("nan")
    rows = scores[
        (scores["source"].astype(str) == source)
        & (scores["target_scope"].astype(str) == target_scope)
        & (scores["evaluation_split"].astype(str) == evaluation_split)
        & (scores["target_family"].astype(str) == "ALL")
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["rmse_range"])


def primitive_behavior_sign_audit(
    actions: pd.DataFrame,
    primitives: pd.DataFrame,
    *,
    policy_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = ["audit_id", "expected_sign", "slope_or_delta", "n", "passed", "notes"]
    if actions.empty or primitives.empty:
        return pd.DataFrame(columns=columns)
    params = policy_params if policy_params is not None else primitive_policy_parameters()
    job_risk_liquid_saving_derivative = (
        -float(params["job_risk_to_spending"]) - float(params["job_risk_to_debt_repayment"])
    )
    joined = actions.merge(
        primitives.drop(columns=["source"], errors="ignore"),
        on=["scenario_id", "type_id"],
        suffixes=("", "_primitive"),
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            _coefficient_audit(
                "liquidity_stress_raises_mpc",
                float(params["liquidity_stress_to_spending"]),
                expected_positive=True,
                n=joined.shape[0],
                notes="Fixed policy derivative: liquidity_stress -> total_spending_share",
            ),
            _coefficient_audit(
                "precaution_raises_liquid_saving",
                -float(params["precaution_to_spending"]),
                expected_positive=True,
                n=joined.shape[0],
                notes="Fixed policy derivative: precaution lowers spending and therefore raises residual liquid saving.",
            ),
            _coefficient_audit(
                "job_risk_raises_liquid_saving",
                job_risk_liquid_saving_derivative,
                expected_positive=True,
                n=joined.shape[0],
                notes="Fixed policy derivative: job risk lowers spending more than it raises debt repayment.",
            ),
            _coefficient_audit(
                "debt_urgency_raises_repayment",
                float(params["debt_urgency_to_debt_repayment"]),
                expected_positive=True,
                n=joined.shape[0],
                notes="Fixed policy derivative: debt urgency -> debt_repayment_share",
            ),
            _coefficient_audit(
                "larger_windfall_lowers_mpc",
                float(params["shock_log_to_spending"]),
                expected_positive=False,
                n=joined.shape[0],
                notes="Policy derivative: shock_size_log_income_ratio -> total_spending_share",
            ),
        ]
    )
    eip = joined[joined["scenario_id"].astype(str) == "eip_2020_style"].copy()
    low = eip[eip["liquidity_group"] == "low"]
    high = eip[eip["liquidity_group"] == "high"]
    delta = _weighted_average(low, "total_spending_share") - _weighted_average(high, "total_spending_share") if not low.empty and not high.empty else float("nan")
    rows.append(
        {
            "audit_id": "low_liquidity_cells_higher_mpc",
            "expected_sign": "positive",
            "slope_or_delta": delta,
            "n": int(eip.shape[0]),
            "passed": bool(np.isfinite(delta) and delta > 0.0),
            "notes": "Low-liquidity EIP cells should spend more than high-liquidity cells through primitive stress.",
        }
    )
    small = joined[joined["scenario_id"].astype(str) == "small_lottery_windfall_style"]
    large = joined[joined["scenario_id"].astype(str) == "large_lottery_windfall_style"]
    shock_delta = _weighted_average(small, "total_spending_share") - _weighted_average(large, "total_spending_share") if not small.empty and not large.empty else float("nan")
    rows.append(
        {
            "audit_id": "small_lottery_higher_mpc_than_large",
            "expected_sign": "positive",
            "slope_or_delta": shock_delta,
            "n": int(small.shape[0] + large.shape[0]),
            "passed": bool(np.isfinite(shock_delta) and shock_delta > 0.0),
            "notes": "Small lottery windfalls should have higher MPC than large lottery windfalls.",
        }
    )
    return pd.DataFrame(rows, columns=columns)


def _coefficient_audit(audit_id: str, value: float, *, expected_positive: bool, n: int, notes: str) -> dict[str, Any]:
    passed = bool(np.isfinite(value) and (value > 0.0 if expected_positive else value < 0.0))
    return {
        "audit_id": audit_id,
        "expected_sign": "positive" if expected_positive else "negative",
        "slope_or_delta": float(value),
        "n": int(n),
        "passed": passed,
        "notes": notes,
    }


def _slope_audit(frame: pd.DataFrame, x_column: str, y_column: str, *, expected_positive: bool, audit_id: str) -> dict[str, Any]:
    finite = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.shape[0] < 3 or float(finite[x_column].std(ddof=0)) <= 1e-12:
        slope = float("nan")
    else:
        slope = float(np.polyfit(finite[x_column].astype(float), finite[y_column].astype(float), 1)[0])
    passed = bool(np.isfinite(slope) and (slope > 0.0 if expected_positive else slope < 0.0))
    return {
        "audit_id": audit_id,
        "expected_sign": "positive" if expected_positive else "negative",
        "slope_or_delta": slope,
        "n": int(finite.shape[0]),
        "passed": passed,
        "notes": f"{x_column} -> {y_column}",
    }


def aggregate_behavior_actions(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in actions.groupby(["scenario_id", "source"], dropna=False):
        scenario_id, source = keys
        low = group[group["liquidity_group"] == "low"]
        high = group[group["liquidity_group"] == "high"]
        aggregate_total = _weighted_average(group, "total_spending_share")
        aggregate_nondurable = _weighted_average(group, "nondurable_spending_share")
        aggregate_durable = _weighted_average(group, "durable_spending_share")
        low_spend = _weighted_average(low, "total_spending_share")
        high_spend = _weighted_average(high, "total_spending_share")
        low_debt = _weighted_average(low, "debt_repayment_share")
        high_debt = _weighted_average(high, "debt_repayment_share")
        low_liquid_saving = _weighted_average(low, "liquid_saving_share")
        high_liquid_saving = _weighted_average(high, "liquid_saving_share")
        rows.append(
            {
                "scenario_id": scenario_id,
                "source": source,
                "n_types": int(group.shape[0]),
                "aggregate_total_spending_share": aggregate_total,
                "aggregate_nondurable_spending_share": aggregate_nondurable,
                "aggregate_durable_spending_share": aggregate_durable,
                "aggregate_total_spending_drop_share": aggregate_total,
                "aggregate_nondurable_spending_drop_share": aggregate_nondurable,
                "aggregate_durable_spending_drop_share": aggregate_durable,
                "aggregate_debt_repayment_share": _weighted_average(group, "debt_repayment_share"),
                "aggregate_liquid_saving_share": _weighted_average(group, "liquid_saving_share"),
                "low_liquidity_total_spending_share": low_spend,
                "high_liquidity_total_spending_share": high_spend,
                "low_high_liquidity_spending_ratio": low_spend / max(high_spend, 1e-9),
                "low_liquidity_total_spending_drop_share": low_spend,
                "high_liquidity_total_spending_drop_share": high_spend,
                "low_high_liquidity_spending_drop_ratio": low_spend / max(high_spend, 1e-9),
                "low_liquidity_debt_repayment_share": low_debt,
                "high_liquidity_debt_repayment_share": high_debt,
                "low_minus_high_debt_repayment_share": low_debt - high_debt,
                "low_liquidity_liquid_saving_share": low_liquid_saving,
                "high_liquidity_liquid_saving_share": high_liquid_saving,
                "high_minus_low_liquid_saving_share": high_liquid_saving - low_liquid_saving,
            }
        )
    return pd.DataFrame(rows)


def score_behavior_targets(aggregates: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if aggregates.empty or targets.empty:
        return pd.DataFrame()
    joined = aggregates.merge(targets, on="scenario_id", how="inner")
    joined["prediction"] = [float(row[row["prediction_column"]]) for _, row in joined.iterrows()]
    joined["range_error"] = joined.apply(_range_error, axis=1)
    joined["point_error"] = joined["prediction"] - joined["target_value"].astype(float)
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "evaluation_split", "target_family"], dropna=False):
        source, evaluation_split, target_family = keys
        rows.append(
            _score_group(
                group,
                source=source,
                evaluation_split=evaluation_split,
                target_family=target_family,
                target_scope="aggregate",
            )
        )
    for keys, group in joined.groupby(["source", "evaluation_split"], dropna=False):
        source, evaluation_split = keys
        rows.append(
            _score_group(
                group,
                source=source,
                evaluation_split=evaluation_split,
                target_family="ALL",
                target_scope="aggregate",
            )
        )
    return pd.DataFrame(rows).sort_values(["evaluation_split", "target_family", "rmse_range", "source"]).reset_index(drop=True)


def join_cell_behavior_target_errors(actions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if actions.empty or targets.empty:
        return pd.DataFrame()
    required = {"scenario_id", "type_id", "prediction_column", "target_low", "target_high", "target_value"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Cell behavior targets missing columns: {', '.join(sorted(missing))}")
    joined = actions.merge(targets, on=["scenario_id", "type_id"], how="inner", suffixes=("", "_target"))
    if joined.empty:
        return joined
    joined["prediction"] = [float(row[row["prediction_column"]]) for _, row in joined.iterrows()]
    joined["range_error"] = joined.apply(_range_error, axis=1)
    joined["point_error"] = joined["prediction"] - joined["target_value"].astype(float)
    joined["abs_range_error"] = joined["range_error"].abs()
    joined["score_weight"] = joined["population_weight"].astype(float).clip(lower=0.0)
    return joined.sort_values(["scenario_id", "source", "type_id", "target_id"]).reset_index(drop=True)


def score_cell_behavior_targets(actions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    joined = join_cell_behavior_target_errors(actions, targets)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "evaluation_split", "target_family"], dropna=False):
        source, evaluation_split, target_family = keys
        rows.append(
            _score_group(
                group,
                source=source,
                evaluation_split=evaluation_split,
                target_family=target_family,
                target_scope="cell",
                weight_column="score_weight",
            )
        )
    for keys, group in joined.groupby(["source", "evaluation_split"], dropna=False):
        source, evaluation_split = keys
        rows.append(
            _score_group(
                group,
                source=source,
                evaluation_split=evaluation_split,
                target_family="ALL",
                target_scope="cell",
                weight_column="score_weight",
            )
        )
    return pd.DataFrame(rows).sort_values(["evaluation_split", "target_family", "rmse_range", "source"]).reset_index(drop=True)


def build_behavior_baseline_comparison(
    scores: pd.DataFrame,
    cell_scores: pd.DataFrame | None = None,
    *,
    baseline_sources: Iterable[str] = BEHAVIOR_BASELINE_SOURCES,
) -> pd.DataFrame:
    frames = [scores]
    if cell_scores is not None and not cell_scores.empty:
        frames.append(cell_scores)
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=BEHAVIOR_BASELINE_COMPARISON_COLUMNS)
    combined = pd.concat(valid_frames, ignore_index=True)
    baselines = set(str(source) for source in baseline_sources)
    rows: list[dict[str, Any]] = []
    for keys, group in combined.groupby(["target_scope", "evaluation_split", "target_family"], dropna=False):
        target_scope, evaluation_split, target_family = keys
        baseline_group = group[group["source"].astype(str).isin(baselines)].sort_values(["rmse_range", "mae_range", "source"])
        if baseline_group.empty:
            continue
        baseline = baseline_group.iloc[0]
        baseline_rmse = float(baseline["rmse_range"])
        baseline_mae = float(baseline["mae_range"])
        for _, row in group.sort_values(["rmse_range", "source"]).iterrows():
            source = str(row["source"])
            if source in baselines:
                continue
            rmse = float(row["rmse_range"])
            mae = float(row["mae_range"])
            rows.append(
                {
                    "source": source,
                    "source_kind": _behavior_source_kind(source),
                    "target_scope": str(target_scope),
                    "evaluation_split": str(evaluation_split),
                    "target_family": str(target_family),
                    "n": int(row["n"]),
                    "effective_weight": float(row.get("effective_weight", np.nan)),
                    "rmse_range": rmse,
                    "mae_range": mae,
                    "rmse_point": float(row.get("rmse_point", np.nan)),
                    "mae_point": float(row.get("mae_point", np.nan)),
                    "best_baseline_source": str(baseline["source"]),
                    "best_baseline_rmse_range": baseline_rmse,
                    "best_baseline_mae_range": baseline_mae,
                    "rmse_range_delta_vs_baseline": rmse - baseline_rmse,
                    "mae_range_delta_vs_baseline": mae - baseline_mae,
                    "rmse_range_pct_improvement_vs_baseline": _pct_improvement(baseline_rmse, rmse),
                    "mae_range_pct_improvement_vs_baseline": _pct_improvement(baseline_mae, mae),
                    "beats_best_baseline": bool(rmse < baseline_rmse - BEHAVIOR_BASELINE_TIE_TOLERANCE),
                    "ties_best_baseline": bool(abs(rmse - baseline_rmse) <= BEHAVIOR_BASELINE_TIE_TOLERANCE),
                    "baseline_verdict": _baseline_row_verdict(rmse, baseline_rmse),
                }
            )
    if not rows:
        return pd.DataFrame(columns=BEHAVIOR_BASELINE_COMPARISON_COLUMNS)
    return pd.DataFrame(rows, columns=BEHAVIOR_BASELINE_COMPARISON_COLUMNS).sort_values(["target_scope", "evaluation_split", "target_family", "rmse_range", "source"]).reset_index(drop=True)


def behavior_baseline_verdict(
    comparison: pd.DataFrame,
    *,
    target_scope: str,
    evaluation_split: str = BEHAVIOR_SELECTION_SPLIT,
    source_kinds: Iterable[str] = ("llm", "llm_ablation"),
) -> dict[str, Any]:
    source_kind_list = tuple(str(kind) for kind in source_kinds)
    if comparison.empty:
        return {
            "verdict": "behavior_baseline_unmeasured",
            "reason": "No behavior baseline comparison rows were produced.",
            "target_scope": target_scope,
            "evaluation_split": evaluation_split,
            "source_kinds": list(source_kind_list),
        }
    overall = comparison[
        (comparison["target_scope"].astype(str) == target_scope)
        & (comparison["evaluation_split"].astype(str) == evaluation_split)
        & (comparison["target_family"].astype(str) == "ALL")
        & (comparison["source_kind"].isin(source_kind_list))
    ].copy()
    if overall.empty:
        return {
            "verdict": "behavior_baseline_unmeasured",
            "reason": f"No {target_scope} ALL rows were available for source kinds {', '.join(source_kind_list)}.",
            "target_scope": target_scope,
            "evaluation_split": evaluation_split,
            "source_kinds": list(source_kind_list),
        }
    best = overall.sort_values(["rmse_range", "source"]).iloc[0]
    if bool(best["beats_best_baseline"]):
        verdict = "behavior_beats_best_baseline"
    elif bool(best["ties_best_baseline"]):
        verdict = "behavior_ties_best_baseline"
    else:
        verdict = "behavior_loses_to_best_baseline"
    return {
        "verdict": verdict,
        "target_scope": target_scope,
        "evaluation_split": evaluation_split,
        "source_kinds": list(source_kind_list),
        "best_source": str(best["source"]),
        "best_source_kind": str(best["source_kind"]),
        "best_baseline_source": str(best["best_baseline_source"]),
        "rmse_range": float(best["rmse_range"]),
        "best_baseline_rmse_range": float(best["best_baseline_rmse_range"]),
        "rmse_range_delta_vs_baseline": float(best["rmse_range_delta_vs_baseline"]),
        "rmse_range_pct_improvement_vs_baseline": float(best["rmse_range_pct_improvement_vs_baseline"]),
        "n": int(best["n"]),
    }


def prespecified_behavior_holdout_verdict(
    comparison: pd.DataFrame,
    *,
    candidate_source: str,
    target_scope: str,
    evaluation_split: str = BEHAVIOR_HOLDOUT_SPLIT,
) -> dict[str, Any]:
    if comparison.empty:
        return {
            "verdict": "holdout_unmeasured",
            "reason": "No holdout baseline comparison rows were produced.",
            "candidate_source": candidate_source,
            "target_scope": target_scope,
            "evaluation_split": evaluation_split,
        }
    overall = comparison[
        (comparison["target_scope"].astype(str) == target_scope)
        & (comparison["evaluation_split"].astype(str) == evaluation_split)
        & (comparison["target_family"].astype(str) == "ALL")
        & (comparison["source"].astype(str) == candidate_source)
    ].copy()
    if overall.empty:
        return {
            "verdict": "holdout_unmeasured",
            "reason": f"Candidate source {candidate_source} has no {target_scope} ALL holdout row.",
            "candidate_source": candidate_source,
            "target_scope": target_scope,
            "evaluation_split": evaluation_split,
        }
    row = overall.iloc[0]
    if bool(row["beats_best_baseline"]):
        verdict = "holdout_beats_best_baseline"
    elif bool(row["ties_best_baseline"]):
        verdict = "holdout_ties_best_baseline"
    else:
        verdict = "holdout_loses_to_best_baseline"
    return {
        "verdict": verdict,
        "candidate_source": candidate_source,
        "target_scope": target_scope,
        "evaluation_split": evaluation_split,
        "best_baseline_source": str(row["best_baseline_source"]),
        "rmse_range": float(row["rmse_range"]),
        "best_baseline_rmse_range": float(row["best_baseline_rmse_range"]),
        "rmse_range_delta_vs_baseline": float(row["rmse_range_delta_vs_baseline"]),
        "rmse_range_pct_improvement_vs_baseline": float(row["rmse_range_pct_improvement_vs_baseline"]),
        "n": int(row["n"]),
    }


def behavior_target_catalog(*, include_unscored: bool = True) -> pd.DataFrame:
    with resources.files(TARGET_CATALOG_PACKAGE).joinpath(TARGET_CATALOG_RESOURCE).open("r", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    numeric_columns = ["target_low", "target_high", "target_value"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["scored"] = frame["scored"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "target_scope" not in frame:
        frame["target_scope"] = np.where(frame["scored"], "aggregate", "unscored_gap")
    frame["target_scope"] = frame["target_scope"].fillna("").replace("", "aggregate")
    if "type_id" not in frame:
        frame["type_id"] = ""
    frame["type_id"] = frame["type_id"].fillna("").astype(str)
    if "evaluation_split" not in frame:
        frame["evaluation_split"] = ""
    frame["evaluation_split"] = frame["evaluation_split"].fillna("").astype(str)
    inferred_split = _infer_behavior_evaluation_split(frame)
    frame["evaluation_split"] = np.where(frame["evaluation_split"].str.strip().eq(""), inferred_split, frame["evaluation_split"])
    frame["evaluation_split"] = np.where(~frame["scored"], "unscored_gap", frame["evaluation_split"])
    if not include_unscored:
        frame = frame[frame["scored"] & (frame["source_status"] == "verified_public")].copy()
    return frame.reset_index(drop=True)


def _infer_behavior_evaluation_split(frame: pd.DataFrame) -> np.ndarray:
    family = frame["target_family"].fillna("").astype(str)
    inferred = np.full(frame.shape[0], BEHAVIOR_SELECTION_SPLIT, dtype=object)
    inferred[family.str.startswith("holdout_ui_").to_numpy()] = BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT
    inferred[family.str.startswith("holdout_lottery").to_numpy()] = BEHAVIOR_HOLDOUT_SPLIT
    other_holdout = family.str.startswith("holdout_").to_numpy() & (inferred == BEHAVIOR_SELECTION_SPLIT)
    inferred[other_holdout] = BEHAVIOR_HOLDOUT_SPLIT
    scored = frame["scored"].astype(bool).to_numpy()
    inferred[~scored] = "unscored_gap"
    return inferred


def behavior_targets_frame(
    *,
    include_unscored: bool = False,
    target_scope: str | None = "aggregate",
    evaluation_split: str | None = None,
) -> pd.DataFrame:
    frame = behavior_target_catalog(include_unscored=include_unscored)
    if not include_unscored:
        frame = frame[frame["scored"] & (frame["source_status"] == "verified_public")].copy()
    if target_scope is not None:
        frame = frame[frame["target_scope"] == target_scope].copy()
    if evaluation_split is not None:
        frame = frame[frame["evaluation_split"] == evaluation_split].copy()
    return frame.reset_index(drop=True)


def scenarios_to_frame(scenarios: Iterable[BehaviorScenario]) -> pd.DataFrame:
    return pd.DataFrame([scenario.__dict__ for scenario in scenarios])


def build_behavior_gate_report(
    manifest: dict[str, Any],
    scenarios: pd.DataFrame,
    targets: pd.DataFrame,
    aggregates: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    target_catalog: pd.DataFrame | None = None,
    cell_targets: pd.DataFrame | None = None,
    cell_scores: pd.DataFrame | None = None,
    cell_joined_errors: pd.DataFrame | None = None,
    baseline_comparison: pd.DataFrame | None = None,
    holdout_baseline_comparison: pd.DataFrame | None = None,
    holdout_targets: pd.DataFrame | None = None,
    ui_holdout_baseline_comparison: pd.DataFrame | None = None,
    ui_holdout_targets: pd.DataFrame | None = None,
    primitive_sign_audit: pd.DataFrame | None = None,
) -> str:
    target_catalog = target_catalog if target_catalog is not None else behavior_target_catalog(include_unscored=True)
    cell_targets = cell_targets if cell_targets is not None else behavior_targets_frame(target_scope="cell")
    cell_scores = cell_scores if cell_scores is not None else pd.DataFrame()
    cell_joined_errors = cell_joined_errors if cell_joined_errors is not None else pd.DataFrame()
    baseline_comparison = baseline_comparison if baseline_comparison is not None else build_behavior_baseline_comparison(scores, cell_scores)
    holdout_baseline_comparison = holdout_baseline_comparison if holdout_baseline_comparison is not None else pd.DataFrame()
    holdout_targets = holdout_targets if holdout_targets is not None else pd.DataFrame()
    ui_holdout_baseline_comparison = ui_holdout_baseline_comparison if ui_holdout_baseline_comparison is not None else pd.DataFrame()
    ui_holdout_targets = ui_holdout_targets if ui_holdout_targets is not None else pd.DataFrame()
    primitive_sign_audit = primitive_sign_audit if primitive_sign_audit is not None else pd.DataFrame()
    gaps = target_catalog[~target_catalog["scored"]].copy() if "scored" in target_catalog else pd.DataFrame()
    lines = [
        "# Household Behavior Target Gate",
        "",
        "## Bottom Line",
        _behavior_bottom_line(scores, cell_scores, baseline_comparison=baseline_comparison),
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- Prompt variant: `{manifest.get('prompt_variant', 'baseline')}`",
        f"- Behavior mode: `{manifest.get('behavior_mode')}`",
        f"- Primitive mode: `{manifest.get('primitive_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Scenario count: `{manifest.get('scenario_count')}`",
        f"- Aggregate target count: `{manifest.get('target_count')}`",
        f"- Cell-level target count: `{manifest.get('cell_target_count')}`",
        f"- Unscored target gaps: `{manifest.get('unscored_target_gap_count', 0)}`",
        f"- Ablation sources: `{len(manifest.get('ablation_sources', []))}`",
        f"- SCF type source: `{manifest.get('household_type_status', {}).get('status', 'unknown')}`",
        "",
        "## Behavior Ablation Tournament",
        markdown_table(_ablation_summary_table(scores, cell_scores)),
        "",
        "## Baseline Comparison",
        markdown_table(_baseline_comparison_table(baseline_comparison)),
        "",
        "## Prespecified Holdout Gate",
        _prespecified_holdout_sentence(manifest.get("prespecified_holdout_verdict", {})),
        _prespecified_holdout_sentence(manifest.get("primitive_holdout_verdict", {})),
        "",
        markdown_table(_baseline_comparison_table(holdout_baseline_comparison)),
        "",
        "## UI-Exhaustion Holdout Gate",
        _prespecified_holdout_sentence(manifest.get("ui_exhaustion_raw_llm_holdout_verdict", {})),
        _prespecified_holdout_sentence(manifest.get("ui_exhaustion_primitive_holdout_verdict", {})),
        "",
        markdown_table(_baseline_comparison_table(ui_holdout_baseline_comparison)),
        "",
        "## Primitive Mechanism Audit",
        f"- Fixed policy version: `{manifest.get('primitive_fixed_policy_version')}`",
        f"- Calibrated policy version: `{manifest.get('primitive_calibrated_policy_version')}`",
        f"- Calibration split: `{manifest.get('primitive_calibration_split')}`",
        f"- Calibration objective: `{manifest.get('primitive_calibration_report', {}).get('objective_before')}` -> `{manifest.get('primitive_calibration_report', {}).get('objective_after')}`",
        f"- All sign audits passed: `{manifest.get('primitive_sign_audit_passed')}`",
        "",
        markdown_table(primitive_sign_audit if not primitive_sign_audit.empty else primitive_sign_audit),
        "",
        "## Holdout Targets",
        markdown_table(
            holdout_targets[
                [
                    "scenario_id",
                    "target_name",
                    "target_family",
                    "response_variable",
                    "target_low",
                    "target_high",
                    "target_value",
                    "source_label",
                    "evaluation_split",
                    "notes",
                ]
            ]
            if not holdout_targets.empty
            else holdout_targets
        ),
        "",
        "## UI-Exhaustion Holdout Targets",
        markdown_table(
            ui_holdout_targets[
                [
                    "scenario_id",
                    "target_name",
                    "target_family",
                    "response_variable",
                    "target_low",
                    "target_high",
                    "target_value",
                    "source_label",
                    "evaluation_split",
                    "notes",
                ]
            ]
            if not ui_holdout_targets.empty
            else ui_holdout_targets
        ),
        "",
        "## Aggregate Scoreboard By Split",
        markdown_table(scores.sort_values(["evaluation_split", "target_family", "rmse_range", "source"])),
        "",
        "## Cell-Level Scoreboard By Split",
        markdown_table(cell_scores.sort_values(["evaluation_split", "target_family", "rmse_range", "source"]) if not cell_scores.empty else cell_scores),
        "",
        "## Cell-Level Joined Errors",
        markdown_table(
            cell_joined_errors[
                [
                    "scenario_id",
                    "source",
                    "type_id",
                    "target_name",
                    "target_family",
                    "evaluation_split",
                    "prediction",
                    "target_low",
                    "target_high",
                    "range_error",
                    "score_weight",
                ]
            ].sort_values(["scenario_id", "source", "type_id"]).head(48)
            if not cell_joined_errors.empty
            else cell_joined_errors
        ),
        "",
        "## Aggregates",
        markdown_table(aggregates.sort_values(["scenario_id", "source"]).head(32)),
        "",
        "## Aggregate Targets",
        markdown_table(
            targets[
                [
                    "scenario_id",
                    "target_name",
                    "target_family",
                    "evaluation_split",
                    "household_bucket",
                    "window",
                    "target_low",
                    "target_high",
                    "target_value",
                    "prediction_column",
                    "source_label",
                ]
            ]
        ),
        "",
        "## Cell-Level Targets",
        markdown_table(
            cell_targets[
                [
                    "scenario_id",
                    "type_id",
                    "target_name",
                    "target_family",
                    "evaluation_split",
                    "household_bucket",
                    "window",
                    "target_low",
                    "target_high",
                    "target_value",
                    "prediction_column",
                    "source_label",
                ]
            ]
            if not cell_targets.empty
            else cell_targets
        ),
        "",
        "## Unscored Direct-Target Gaps",
        markdown_table(gaps[["scenario_id", "target_name", "target_family", "response_variable", "household_bucket", "notes"]] if not gaps.empty else gaps),
        "",
        "## What This Gate Means",
        (
            "This gate scores household behavior directly against public stimulus-response moments. "
            "The aggregate surface covers spending, debt repayment, saving, and directional debt/saving gradients "
            "by liquidity. The cell-level surface applies public low- and high-liquidity response ranges to matching "
            "SCF household types, weighted by population share, so the bridge asks whether typed agents beat a "
            "liquidity baseline at the household-cell grain. "
            "Rows in the unscored gap table are intentionally held out until a public direct target is verified."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _behavior_action_row(scenario: BehaviorScenario, type_cell: pd.Series, response: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "schema_version": AGENT_ECONOMY_VERSION,
        "scenario_id": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "source": source,
        "type_id": str(type_cell["type_id"]),
        "population_weight": float(type_cell["population_weight"]),
        "liquidity_group": _liquidity_group(type_cell),
        "transfer_amount": float(scenario.transfer_amount),
        "income_loss_pct": float(scenario.income_loss_pct),
        "total_spending_share": float(response["total_spending_share"]),
        "nondurable_spending_share": float(response["nondurable_spending_share"]),
        "durable_spending_share": float(response["durable_spending_share"]),
        "debt_repayment_share": float(response["debt_repayment_share"]),
        "liquid_saving_share": float(response["liquid_saving_share"]),
        "confidence": float(response.get("confidence", 0.5)),
        "reason": str(response.get("reason", ""))[:300],
    }


def _liquidity_rule_response(scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        return _liquidity_rule_income_loss_response(scenario, type_cell)
    buffer_months = float(type_cell.get("liquid_buffer_months", 2.0))
    debt_ratio = float(type_cell.get("debt_to_asset", 0.0))
    spend = float(np.clip(0.52 - 0.045 * buffer_months + 0.05 * debt_ratio, 0.08, 0.62))
    durable = float(np.clip(0.18 * spend, 0.0, 0.16))
    nondurable = max(0.0, spend - durable)
    debt = float(np.clip(0.10 + 0.12 * debt_ratio, 0.0, 0.35))
    liquid = max(0.0, 1.0 - spend - debt)
    return {
        "total_spending_share": spend,
        "nondurable_spending_share": nondurable,
        "durable_spending_share": durable,
        "debt_repayment_share": debt,
        "liquid_saving_share": liquid,
        "confidence": 0.65,
        "reason": "liquidity rule based on cash buffer and debt intensity",
    }


def _liquidity_rule_income_loss_response(scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, Any]:
    buffer_months = float(type_cell.get("liquid_buffer_months", 2.0))
    debt_ratio = float(type_cell.get("debt_to_asset", 0.0))
    severity = float(np.clip(float(scenario.income_loss_pct) / 0.35, 0.0, 1.0))
    exhaustion = 1.0 if scenario.scenario_id == "ui_exhaustion_income_loss_style" else 0.0
    ongoing = 1.0 if scenario.scenario_id == "ui_receipt_monthly_path_style" else 0.0
    total_drop = float(
        np.clip(
            0.015 + 0.060 * severity + 0.025 * exhaustion - 0.055 * ongoing - 0.010 * buffer_months + 0.015 * debt_ratio,
            0.0,
            0.22,
        )
    )
    nondurable = float(np.clip(0.78 * total_drop, 0.0, total_drop))
    durable = max(0.0, total_drop - nondurable)
    debt = float(np.clip(0.035 + 0.10 * debt_ratio + 0.04 * severity, 0.0, 0.35))
    liquid = float(np.clip(0.07 + 0.08 * severity + 0.06 * max(0.0, 2.0 - buffer_months), 0.0, 0.50))
    return {
        "total_spending_share": total_drop,
        "nondurable_spending_share": nondurable,
        "durable_spending_share": durable,
        "debt_repayment_share": debt,
        "liquid_saving_share": liquid,
        "confidence": 0.65,
        "reason": "liquidity rule for income-loss spending drops based on cash buffer and debt intensity",
    }


def _flat_rule_response(scenario: BehaviorScenario, _type_cell: pd.Series) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        total_drop = 0.08 if scenario.scenario_id != "ui_receipt_monthly_path_style" else 0.005
        nondurable = 0.75 * total_drop
        return {
            "total_spending_share": total_drop,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": total_drop - nondurable,
            "debt_repayment_share": 0.08,
            "liquid_saving_share": 0.10,
            "confidence": 0.50,
            "reason": "flat spending-drop rule for UI income-loss scenarios",
        }
    return {
        "total_spending_share": 0.30,
        "nondurable_spending_share": 0.24,
        "durable_spending_share": 0.06,
        "debt_repayment_share": 0.20,
        "liquid_saving_share": 0.50,
        "confidence": 0.50,
        "reason": "flat 30 percent spending rule",
    }


def _permanent_income_response(scenario: BehaviorScenario, _type_cell: pd.Series) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        exhaustion = scenario.scenario_id == "ui_exhaustion_income_loss_style"
        total_drop = 0.025 if exhaustion else (0.040 if scenario.scenario_id == "ui_onset_income_loss_style" else 0.002)
        nondurable = 0.80 * total_drop
        return {
            "total_spending_share": total_drop,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": total_drop - nondurable,
            "debt_repayment_share": 0.04,
            "liquid_saving_share": 0.12,
            "confidence": 0.50,
            "reason": "permanent-income benchmark smooths predictable benefit exhaustion",
        }
    return {
        "total_spending_share": 0.10,
        "nondurable_spending_share": 0.08,
        "durable_spending_share": 0.02,
        "debt_repayment_share": 0.15,
        "liquid_saving_share": 0.75,
        "confidence": 0.50,
        "reason": "low-MPC permanent-income benchmark",
    }


def _bounded_share(mapping: dict[str, Any], key: str) -> float:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        raise LLMUnavailable(f"Behavior payload field {key} must be numeric") from None
    if not np.isfinite(value):
        raise LLMUnavailable(f"Behavior payload field {key} must be finite")
    return float(np.clip(value, 0.0, 1.0))


def bounded_number(mapping: dict[str, Any], key: str, low: float, high: float) -> float:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        raise LLMUnavailable(f"Primitive behavior payload field {key} must be numeric") from None
    if not np.isfinite(value):
        raise LLMUnavailable(f"Primitive behavior payload field {key} must be finite")
    return float(np.clip(value, low, high))


def _liquidity_group(type_cell: pd.Series) -> str:
    type_id = str(type_cell["type_id"])
    if type_id in {"liquid_poor_renter", "wealthy_htm_homeowner", "unemployed_low_liquid"}:
        return "low"
    if type_id in {"retiree_liquid_assets", "high_income_illiquid_rich", "business_owner_top_wealth"}:
        return "high"
    return "middle"


def _scenario_shock_income_ratio(scenario: BehaviorScenario, type_cell: pd.Series) -> float:
    annual_income = max(float(type_cell.get("annual_income", 50000.0)), 1.0)
    if scenario.scenario_type == "income_loss":
        monthly_income = annual_income / 12.0
        return abs(float(scenario.income_loss_pct)) * monthly_income / annual_income
    return float(scenario.transfer_amount) / annual_income


def _weighted_average(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["population_weight"].astype(float).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        weights = pd.Series(np.ones(len(group)) / max(1, len(group)), index=group.index)
        total = 1.0
    return float((group[column].astype(float) * weights).sum() / total)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_weights = weights.astype(float).clip(lower=0.0)
    total = float(clean_weights.sum())
    if total <= 0:
        return float(values.astype(float).mean())
    return float((values.astype(float) * clean_weights).sum() / total)


def _range_error(row: pd.Series) -> float:
    prediction = float(row["prediction"])
    low = float(row["target_low"])
    high = float(row["target_high"])
    if low <= prediction <= high:
        return 0.0
    return low - prediction if prediction < low else prediction - high


def _score_group(
    group: pd.DataFrame,
    *,
    source: str,
    evaluation_split: str,
    target_family: str,
    target_scope: str,
    weight_column: str | None = None,
) -> dict[str, Any]:
    range_error = group["range_error"].astype(float)
    point_error = group["point_error"].astype(float)
    if weight_column and weight_column in group:
        weights = group[weight_column].astype(float).clip(lower=0.0)
        if float(weights.sum()) <= 0.0:
            weights = pd.Series(np.ones(len(group)), index=group.index, dtype=float)
    else:
        weights = pd.Series(np.ones(len(group)), index=group.index, dtype=float)
    weight_sum = float(weights.sum())
    return {
        "source": source,
        "target_family": target_family,
        "target_scope": target_scope,
        "evaluation_split": str(evaluation_split),
        "n": int(group.shape[0]),
        "effective_weight": weight_sum,
        "rmse_range": float(np.sqrt((weights * np.square(range_error)).sum() / weight_sum)),
        "mae_range": float((weights * np.abs(range_error)).sum() / weight_sum),
        "rmse_point": float(np.sqrt((weights * np.square(point_error)).sum() / weight_sum)),
        "mae_point": float((weights * np.abs(point_error)).sum() / weight_sum),
        "mean_prediction": float((weights * group["prediction"].astype(float)).sum() / weight_sum),
        "mean_target": float((weights * group["target_value"].astype(float)).sum() / weight_sum),
    }


def _behavior_source_kind(source: str) -> str:
    if source in BEHAVIOR_BASELINE_SOURCES:
        return "baseline"
    if source.startswith("primitive_fixed_"):
        return "primitive_fixed"
    if source.startswith("primitive_"):
        return "primitive"
    if source.startswith("choice_"):
        return "choice"
    if source.startswith("llm_") and "__" in source:
        return "llm_ablation"
    if source.startswith("llm_"):
        return "llm"
    return "other"


def _pct_improvement(baseline_value: float, candidate_value: float) -> float:
    if not np.isfinite(baseline_value) or abs(baseline_value) <= BEHAVIOR_BASELINE_TIE_TOLERANCE:
        return float("nan")
    return float((baseline_value - candidate_value) / abs(baseline_value))


def _baseline_row_verdict(candidate_rmse: float, baseline_rmse: float) -> str:
    if candidate_rmse < baseline_rmse - BEHAVIOR_BASELINE_TIE_TOLERANCE:
        return "beats_best_baseline"
    if abs(candidate_rmse - baseline_rmse) <= BEHAVIOR_BASELINE_TIE_TOLERANCE:
        return "ties_best_baseline"
    return "loses_to_best_baseline"


def _behavior_bottom_line(
    scores: pd.DataFrame,
    cell_scores: pd.DataFrame | None = None,
    *,
    baseline_comparison: pd.DataFrame | None = None,
) -> str:
    if scores.empty:
        return "Behavior target scoring produced no rows."
    selection_overall = scores[
        (scores["evaluation_split"] == BEHAVIOR_SELECTION_SPLIT) & (scores["target_family"] == "ALL")
    ].sort_values("rmse_range")
    holdout_overall = scores[
        (scores["evaluation_split"] == BEHAVIOR_HOLDOUT_SPLIT) & (scores["target_family"] == "ALL")
    ].sort_values("rmse_range")
    ui_holdout_overall = scores[
        (scores["evaluation_split"] == BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT) & (scores["target_family"] == "ALL")
    ].sort_values("rmse_range")
    if selection_overall.empty and holdout_overall.empty and ui_holdout_overall.empty:
        return "Behavior target scoring produced no split-level overall row."
    selection_line = "Selection-split aggregate scoring is unmeasured."
    if not selection_overall.empty:
        best = selection_overall.iloc[0]
        selection_line = (
            f"Selection-split aggregate best source is `{best['source']}` with range RMSE "
            f"`{float(best['rmse_range']):.4f}` across `{int(best['n'])}` targets."
        )
    holdout_line = _split_best_line(holdout_overall, label="Lottery holdout aggregate")
    ui_holdout_line = _split_best_line(ui_holdout_overall, label="UI-exhaustion holdout aggregate")
    cell_line = ""
    if cell_scores is not None and not cell_scores.empty:
        cell_overall = cell_scores[
            (cell_scores["evaluation_split"] == BEHAVIOR_SELECTION_SPLIT)
            & (cell_scores["target_family"] == "ALL")
        ].sort_values("rmse_range")
        if not cell_overall.empty:
            cell_best = cell_overall.iloc[0]
            cell_line = (
                f" Selection-split cell-level best source is `{cell_best['source']}` with population-weighted range RMSE "
                f"`{float(cell_best['rmse_range']):.4f}` across `{int(cell_best['n'])}` cell targets."
            )
    return (
        f"{selection_line}"
        f"{holdout_line}"
        f"{ui_holdout_line}"
        f"{cell_line}"
        f" {_baseline_verdict_sentence(baseline_comparison if baseline_comparison is not None else build_behavior_baseline_comparison(scores, cell_scores))}"
    )


def _split_best_line(overall: pd.DataFrame, *, label: str) -> str:
    if overall.empty:
        return f" {label} scoring is unmeasured."
    best = overall.iloc[0]
    return (
        f" {label} best source is `{best['source']}` with range RMSE "
        f"`{float(best['rmse_range']):.4f}` across `{int(best['n'])}` targets."
    )


def _baseline_verdict_sentence(comparison: pd.DataFrame) -> str:
    aggregate = behavior_baseline_verdict(comparison, target_scope="aggregate", evaluation_split=BEHAVIOR_SELECTION_SPLIT)
    cell = behavior_baseline_verdict(comparison, target_scope="cell", evaluation_split=BEHAVIOR_SELECTION_SPLIT)
    raw_aggregate = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_SELECTION_SPLIT,
        source_kinds=("llm",),
    )
    raw_cell = behavior_baseline_verdict(
        comparison,
        target_scope="cell",
        evaluation_split=BEHAVIOR_SELECTION_SPLIT,
        source_kinds=("llm",),
    )
    primitive_aggregate = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_SELECTION_SPLIT,
        source_kinds=("primitive",),
    )
    primitive_cell = behavior_baseline_verdict(
        comparison,
        target_scope="cell",
        evaluation_split=BEHAVIOR_SELECTION_SPLIT,
        source_kinds=("primitive",),
    )
    raw_holdout = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_HOLDOUT_SPLIT,
        source_kinds=("llm",),
    )
    primitive_holdout = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_HOLDOUT_SPLIT,
        source_kinds=("primitive",),
    )
    raw_ui_holdout = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
        source_kinds=("llm",),
    )
    primitive_ui_holdout = behavior_baseline_verdict(
        comparison,
        target_scope="aggregate",
        evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
        source_kinds=("primitive",),
    )

    def clause(label: str, verdict: dict[str, Any], *, subject: str) -> str:
        verdict_name = str(verdict.get("verdict", "behavior_baseline_unmeasured"))
        if verdict_name == "behavior_baseline_unmeasured":
            return f"{label} baseline comparison is unmeasured."
        verb = {
            "behavior_beats_best_baseline": "beats",
            "behavior_ties_best_baseline": "ties",
            "behavior_loses_to_best_baseline": "loses to",
        }.get(verdict_name, "compares with")
        return (
            f"{label} {subject} source `{verdict.get('best_source')}` {verb} "
            f"best rule baseline `{verdict.get('best_baseline_source')}` "
            f"on `{verdict.get('evaluation_split')}` "
            f"(delta `{float(verdict.get('rmse_range_delta_vs_baseline', np.nan)):.4f}`)."
        )

    return " ".join(
        [
            clause("Aggregate", raw_aggregate, subject="raw LLM"),
            clause("Aggregate", primitive_aggregate, subject="primitive-driven"),
            clause("Aggregate", aggregate, subject="best LLM/ablation"),
            clause("Holdout aggregate", raw_holdout, subject="raw LLM"),
            clause("Holdout aggregate", primitive_holdout, subject="primitive-driven"),
            clause("UI holdout aggregate", raw_ui_holdout, subject="raw LLM"),
            clause("UI holdout aggregate", primitive_ui_holdout, subject="primitive-driven"),
            clause("Cell-level", raw_cell, subject="raw LLM"),
            clause("Cell-level", primitive_cell, subject="primitive-driven"),
            clause("Cell-level", cell, subject="best LLM/ablation"),
        ]
    )


def _prespecified_holdout_sentence(verdict: dict[str, Any]) -> str:
    verdict_name = str(verdict.get("verdict", "holdout_unmeasured"))
    if verdict_name == "holdout_unmeasured":
        return f"Holdout verdict is unmeasured: {verdict.get('reason', 'no reason provided')}."
    verb = {
        "holdout_beats_best_baseline": "beats",
        "holdout_ties_best_baseline": "ties",
        "holdout_loses_to_best_baseline": "loses to",
    }.get(verdict_name, "compares with")
    return (
        f"Pre-specified source `{verdict.get('candidate_source')}` {verb} "
        f"best rule baseline `{verdict.get('best_baseline_source')}` on `{verdict.get('target_scope')}` "
        f"`{verdict.get('evaluation_split')}` targets (delta `{float(verdict.get('rmse_range_delta_vs_baseline', np.nan)):.4f}`, "
        f"n `{int(verdict.get('n', 0))}`)."
    )


def _ablation_summary_table(scores: pd.DataFrame, cell_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    aggregate = scores[scores["target_family"] == "ALL"][
        ["evaluation_split", "source", "n", "rmse_range", "mae_range", "rmse_point", "mae_point"]
    ].copy()
    aggregate = aggregate.rename(
        columns={
            "n": "aggregate_n",
            "rmse_range": "aggregate_rmse_range",
            "mae_range": "aggregate_mae_range",
            "rmse_point": "aggregate_rmse_point",
            "mae_point": "aggregate_mae_point",
        }
    )
    if cell_scores is not None and not cell_scores.empty:
        cell = cell_scores[cell_scores["target_family"] == "ALL"][
            ["evaluation_split", "source", "n", "rmse_range", "mae_range"]
        ].copy()
        cell = cell.rename(
            columns={
                "n": "cell_n",
                "rmse_range": "cell_rmse_range",
                "mae_range": "cell_mae_range",
            }
        )
        aggregate = aggregate.merge(cell, on=["evaluation_split", "source"], how="left")
    interesting = aggregate[
        aggregate["source"].eq("liquidity_rule")
        | aggregate["source"].str.startswith("llm_")
        | aggregate["source"].str.startswith("primitive_")
        | aggregate["source"].str.contains("__liquidity_prior|__residual_over_liquidity", regex=True)
    ].copy()
    if interesting.empty:
        interesting = aggregate
    return interesting.sort_values(["evaluation_split", "aggregate_rmse_range", "source"]).reset_index(drop=True)


def _baseline_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    columns = [
        "target_scope",
        "evaluation_split",
        "target_family",
        "source",
        "source_kind",
        "n",
        "rmse_range",
        "best_baseline_source",
        "best_baseline_rmse_range",
        "rmse_range_delta_vs_baseline",
        "rmse_range_pct_improvement_vs_baseline",
        "baseline_verdict",
    ]
    selected = comparison[columns].copy()
    return selected.sort_values(["target_scope", "evaluation_split", "target_family", "rmse_range", "source"]).reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
