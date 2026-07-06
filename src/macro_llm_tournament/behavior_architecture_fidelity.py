from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, cache_key, markdown_table, round_or_none
from .agent_llm import AgentLLMClient
from .agent_types import build_household_type_cells
from .behavior_gate import (
    BEHAVIOR_SCENARIOS,
    BEHAVIOR_SELECTION_SPLIT,
    BEHAVIOR_SHARE_COLUMNS,
    BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
    BehaviorLLMClient,
    BehaviorScenario,
    aggregate_behavior_actions,
    behavior_targets_frame,
    bounded_number,
    build_behavior_baseline_comparison,
    run_behavior_controls,
    run_behavior_gate,
    score_behavior_targets,
)
from .llm_common import LLMUnavailable


ARCHITECTURE_FIDELITY_VERSION = "behavior_architecture_fidelity_v1"
CHOICE_PROMPT_VERSION = "behavior_constrained_choice_v1"
PRIMITIVE_V3_PROMPT_VERSION = "household_behavior_primitives_v3"
PRIMITIVE_V3_POLICY_VERSION = "primitive_v3_policy_fixed_selection_only_v1"
CHOICE_SOURCE_PREFIX = "choice"
PRIMITIVE_V3_SOURCE_PREFIX = "primitive_v3"
INTERPRETABILITY_RANK = {"constrained_raw": 0, "constrained_choice": 1, "primitive_v3": 2}


CHOICE_FIELDS = [
    "base_response_share",
    "liquidity_adjustment",
    "shock_size_damping",
    "debt_repayment_share",
    "durable_fraction",
    "income_loss_sensitivity",
    "exhaustion_boost",
    "confidence",
]
PRIMITIVE_V3_FIELDS = [
    "perceived_job_loss_risk_pp",
    "expected_income_growth_pct",
    "precautionary_saving_motive",
    "liquidity_stress",
    "debt_repayment_urgency",
    "durable_purchase_pull_forward",
    "shock_size_normalized",
    "shock_size_log_income_ratio",
    "income_change_attention",
    "predictable_drop_attention",
    "windfall_permanence_belief",
    "spending_commitment_share",
    "confidence",
]


@dataclass(frozen=True)
class ArchitectureResult:
    source: str
    architecture: str
    interpretability_tier: str
    selection_rmse: float
    selection_cell_rmse: float
    ui_holdout_rmse: float
    best_ui_baseline_rmse: float
    ui_delta_vs_best_baseline: float
    within_25pct_of_raw_ui: bool
    recommended: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavior architecture fidelity experiment.")
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fidelity-tolerance", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --mode live is used")
    if args.mode == "live" and not args.fresh_cache and not args.cache_dir:
        raise SystemExit("--fresh-cache or --cache-dir is required when --mode live is used")
    if args.fresh_cache and args.cache_dir:
        raise SystemExit("--fresh-cache and --cache-dir cannot be combined; use one fresh run or one explicit resume cache")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"behavior_architecture_fidelity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "fresh_behavior_architecture_cache"
        if args.fresh_cache
        else WORK_ROOT / "behavior_llm_cache"
    )
    manifest: dict[str, Any] = {
        "schema_version": ARCHITECTURE_FIDELITY_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "explicit_cache_dir": bool(args.cache_dir),
        "cache_dir": str(cache_dir),
        "selection_split": BEHAVIOR_SELECTION_SPLIT,
        "holdout_split": BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT,
        "fidelity_tolerance": float(args.fidelity_tolerance),
        "lottery_holdout_status": "spent_and_not_scored_for_new_architectures",
        "ui_holdout_leakage_note": (
            "Phase 1 raw/rule UI results were known before this run; constrained-choice and primitive-v3 "
            "configurations are locked in this manifest before their UI score is interpreted."
        ),
        "locked_architectures": locked_architecture_manifest(),
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        selection_targets = behavior_targets_frame(target_scope="aggregate", evaluation_split=BEHAVIOR_SELECTION_SPLIT)
        ui_targets = behavior_targets_frame(target_scope="aggregate", evaluation_split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
        targets = pd.concat([selection_targets, ui_targets], ignore_index=True)

        raw_client = BehaviorLLMClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.mode,
            max_live_calls=args.max_live_calls,
        )
        raw_actions = run_behavior_gate(BEHAVIOR_SCENARIOS, type_cells, llm_client=raw_client)
        remaining_cap = max(0, int(args.max_live_calls) - raw_client.live_call_count)
        choice_client = ArchitectureLLMClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.mode,
            max_live_calls=remaining_cap,
        )
        choice_actions = run_choice_architecture(BEHAVIOR_SCENARIOS, type_cells, client=choice_client)
        remaining_cap = max(0, remaining_cap - choice_client.live_call_count)
        primitive_client = ArchitectureLLMClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.mode,
            max_live_calls=remaining_cap,
        )
        primitive_actions = run_primitive_v3_architecture(BEHAVIOR_SCENARIOS, type_cells, client=primitive_client)
        controls = run_behavior_controls(BEHAVIOR_SCENARIOS, type_cells)

        all_actions = pd.concat([raw_actions, choice_actions, primitive_actions, controls], ignore_index=True)
        aggregates = aggregate_behavior_actions(all_actions)
        scores = score_behavior_targets(aggregates, targets)
        selection_scores = score_behavior_targets(aggregates, selection_targets)
        ui_scores = score_behavior_targets(aggregates, ui_targets)
        comparison = build_behavior_baseline_comparison(scores)
        ui_comparison = build_behavior_baseline_comparison(ui_scores)
        fidelity = build_fidelity_table(
            scores,
            comparison,
            provider=args.provider,
            model=args.model,
            tolerance=float(args.fidelity_tolerance),
        )
        report = build_report(manifest, fidelity, ui_comparison, selection_scores, ui_scores)

        all_actions.to_csv(output_dir / "behavior_architecture_actions.csv", index=False)
        aggregates.to_csv(output_dir / "behavior_architecture_aggregates.csv", index=False)
        scores.to_csv(output_dir / "behavior_architecture_scores.csv", index=False)
        comparison.to_csv(output_dir / "behavior_architecture_baseline_comparison.csv", index=False)
        ui_comparison.to_csv(output_dir / "behavior_architecture_ui_baseline_comparison.csv", index=False)
        fidelity.to_csv(output_dir / "behavior_architecture_fidelity.csv", index=False)
        (output_dir / "behavior_architecture_raw_records.json").write_text(
            json.dumps(raw_client.raw_records + choice_client.raw_records + primitive_client.raw_records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "behavior_architecture_report.md").write_text(report, encoding="utf-8")

        manifest.update(
            {
                "status": "ok",
                "household_type_status": type_status,
                "household_type_count": int(type_cells.shape[0]),
                "action_rows": int(all_actions.shape[0]),
                "score_rows": int(scores.shape[0]),
                "selection_score_rows": int(selection_scores.shape[0]),
                "ui_score_rows": int(ui_scores.shape[0]),
                "live_call_count": int(raw_client.live_call_count + choice_client.live_call_count + primitive_client.live_call_count),
                "cache_hit_count": int(raw_client.cache_hit_count + choice_client.cache_hit_count + primitive_client.cache_hit_count),
                "raw_live_call_count": int(raw_client.live_call_count),
                "choice_live_call_count": int(choice_client.live_call_count),
                "primitive_v3_live_call_count": int(primitive_client.live_call_count),
                "raw_cache_hit_count": int(raw_client.cache_hit_count),
                "choice_cache_hit_count": int(choice_client.cache_hit_count),
                "primitive_v3_cache_hit_count": int(primitive_client.cache_hit_count),
                "recommended_architecture": _recommended_architecture(fidelity),
                "outputs": [
                    "behavior_architecture_actions.csv",
                    "behavior_architecture_aggregates.csv",
                    "behavior_architecture_scores.csv",
                    "behavior_architecture_baseline_comparison.csv",
                    "behavior_architecture_ui_baseline_comparison.csv",
                    "behavior_architecture_fidelity.csv",
                    "behavior_architecture_raw_records.json",
                    "behavior_architecture_report.md",
                ],
            }
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


class ArchitectureLLMClient:
    def __init__(self, provider: str, model: str, cache_dir: Path, *, mode: str, max_live_calls: int):
        self.provider = provider
        self.model = model
        self.mode = mode
        self._client = AgentLLMClient(provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls)
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._client.cache_hit_count

    def choice_panel(self, scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_choice_payload(scenario, type_cells),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = choice_prompt(scenario, type_cells)
            data = self._client._codex_call(
                prompt,
                f"behavior_choice_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}",
            )
        normalized = normalize_choice_payload(scenario, type_cells, data)
        self.raw_records.append(_raw_record("behavior_choice", scenario, data))
        return normalized

    def primitive_v3_panel(self, scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_primitive_v3_payload(scenario, type_cells),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = primitive_v3_prompt(scenario, type_cells)
            data = self._client._codex_call(
                prompt,
                f"behavior_primitives_v3_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}",
            )
        normalized = normalize_primitive_v3_payload(scenario, type_cells, data)
        self.raw_records.append(_raw_record("behavior_primitives_v3", scenario, data))
        return normalized


def _raw_record(record_type: str, scenario: BehaviorScenario, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "record_type": record_type,
        "provider": data.get("provider"),
        "model": data.get("model"),
        "cache_hit": bool(data.get("cache_hit", False)),
        "cache_path": data.get("cache_path"),
        "payload": data.get("payload", data),
    }


def run_choice_architecture(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    *,
    client: ArchitectureLLMClient,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = f"{CHOICE_SOURCE_PREFIX}_{client.provider}_{client.model}"
    for scenario in scenarios:
        payload = client.choice_panel(scenario, type_cells)
        for _, type_cell in type_cells.iterrows():
            choice = payload["choices_by_type"][str(type_cell["type_id"])]
            rows.append(action_row(scenario, type_cell, choice_policy_action(scenario, type_cell, choice), source=source))
    return pd.DataFrame(rows)


def run_primitive_v3_architecture(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    *,
    client: ArchitectureLLMClient,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = f"{PRIMITIVE_V3_SOURCE_PREFIX}_{client.provider}_{client.model}"
    for scenario in scenarios:
        payload = client.primitive_v3_panel(scenario, type_cells)
        for _, type_cell in type_cells.iterrows():
            primitive = payload["primitives_by_type"][str(type_cell["type_id"])]
            rows.append(action_row(scenario, type_cell, primitive_v3_policy_action(scenario, type_cell, primitive), source=source))
    return pd.DataFrame(rows)


def choice_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> str:
    payload = {
        "prompt_version": CHOICE_PROMPT_VERSION,
        "task": (
            "Choose one bounded, named household policy for each type cell. Do not output final spending shares; "
            "deterministic code executes the selected policy and clips it to feasibility."
        ),
        "as_of_rule": "Use only the supplied scenario and type cells. Do not cite realized study estimates or target values.",
        "scenario": scenario_payload(scenario),
        "allowed_policy_families": [
            "liquidity_buffer_rule",
            "windfall_damped_rule",
            "income_loss_smoothing_rule",
        ],
        "household_type_cells": type_rows(type_cells),
        "required_response": {
            "household_policy_choices": [
                {
                    "type_id": "one supplied type_id",
                    "policy_family": "one allowed policy family",
                    "base_response_share": "0 to 0.85 baseline spending response or spending-drop response",
                    "liquidity_adjustment": "0 to 0.35 added for low-liquidity cells and subtracted for high-liquidity cells",
                    "shock_size_damping": "0 to 0.60 damping for large transitory windfalls",
                    "debt_repayment_share": "0 to 0.50",
                    "durable_fraction": "0 to 0.70 fraction of spending response in durables",
                    "income_loss_sensitivity": "0 to 0.60 spending-drop sensitivity to income loss",
                    "exhaustion_boost": "0 to 0.30 extra response for predictable UI exhaustion",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def primitive_v3_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> str:
    payload = {
        "prompt_version": PRIMITIVE_V3_PROMPT_VERSION,
        "task": (
            "Infer behavior primitives only. Do not output final spending, saving, debt, or allocation shares. "
            "The deterministic primitive-v3 policy will map primitives into household actions."
        ),
        "as_of_rule": "Use only the supplied scenario and type cells. Do not cite realized study estimates or target values.",
        "scenario": scenario_payload(scenario),
        "household_type_cells": type_rows(type_cells),
        "required_response": {
            "household_primitives": [
                {
                    "type_id": "one supplied type_id",
                    "perceived_job_loss_risk_pp": "0 to 100",
                    "expected_income_growth_pct": "-20 to 20",
                    "precautionary_saving_motive": "0 to 1",
                    "liquidity_stress": "0 to 1",
                    "debt_repayment_urgency": "0 to 1",
                    "durable_purchase_pull_forward": "0 to 1",
                    "shock_size_normalized": "0 to 1",
                    "shock_size_log_income_ratio": "-4 to 2",
                    "income_change_attention": "0 to 1 salience of the income change itself",
                    "predictable_drop_attention": "0 to 1 attention to predictable future income drops",
                    "windfall_permanence_belief": "0 to 1 belief that the shock is persistent rather than transitory",
                    "spending_commitment_share": "0 to 1 share of spending that is hard to adjust quickly",
                    "confidence": "0 to 1",
                    "reason": "short reason",
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def scenario_payload(scenario: BehaviorScenario) -> dict[str, Any]:
    payload = {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "as_of_date": scenario.as_of_date,
        "scenario_type": scenario.scenario_type,
        "transfer_amount": scenario.transfer_amount,
        "horizon_months": scenario.horizon_months,
        "income_loss_pct": scenario.income_loss_pct,
        "ui_month": scenario.ui_month,
        "context": scenario.prompt_context,
    }
    return payload


def type_rows(type_cells: pd.DataFrame) -> list[dict[str, Any]]:
    return [
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
    ]


def normalize_choice_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    choices = payload.get("household_policy_choices")
    if not isinstance(choices, list):
        raise LLMUnavailable(f"Choice payload for {scenario.scenario_id} is missing household_policy_choices list")
    expected_ids = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, Any]] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        type_id = str(choice.get("type_id", ""))
        if type_id not in expected_ids:
            continue
        family = str(choice.get("policy_family", "liquidity_buffer_rule"))
        if family not in {"liquidity_buffer_rule", "windfall_damped_rule", "income_loss_smoothing_rule"}:
            family = "liquidity_buffer_rule"
        by_type[type_id] = {
            "policy_family": family,
            "base_response_share": bounded_number(choice, "base_response_share", 0.0, 0.85),
            "liquidity_adjustment": bounded_number(choice, "liquidity_adjustment", 0.0, 0.35),
            "shock_size_damping": bounded_number(choice, "shock_size_damping", 0.0, 0.60),
            "debt_repayment_share": bounded_number(choice, "debt_repayment_share", 0.0, 0.50),
            "durable_fraction": bounded_number(choice, "durable_fraction", 0.0, 0.70),
            "income_loss_sensitivity": bounded_number(choice, "income_loss_sensitivity", 0.0, 0.60),
            "exhaustion_boost": bounded_number(choice, "exhaustion_boost", 0.0, 0.30),
            "confidence": bounded_number(choice, "confidence", 0.0, 1.0),
            "reason": str(choice.get("reason", ""))[:300],
        }
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Choice payload for {scenario.scenario_id} is missing type ids: {', '.join(missing)}")
    return {"choices_by_type": by_type}


def normalize_primitive_v3_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    if "household_actions" in payload:
        raise LLMUnavailable(f"Primitive-v3 payload for {scenario.scenario_id} must not include household_actions")
    primitives = payload.get("household_primitives")
    if not isinstance(primitives, list):
        raise LLMUnavailable(f"Primitive-v3 payload for {scenario.scenario_id} is missing household_primitives list")
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
                f"Primitive-v3 payload for {scenario.scenario_id} includes final allocation fields: {', '.join(mutation_fields)}"
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
            "income_change_attention": bounded_number(primitive, "income_change_attention", 0.0, 1.0),
            "predictable_drop_attention": bounded_number(primitive, "predictable_drop_attention", 0.0, 1.0),
            "windfall_permanence_belief": bounded_number(primitive, "windfall_permanence_belief", 0.0, 1.0),
            "spending_commitment_share": bounded_number(primitive, "spending_commitment_share", 0.0, 1.0),
            "confidence": bounded_number(primitive, "confidence", 0.0, 1.0),
            "reason": str(primitive.get("reason", ""))[:300],
        }
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Primitive-v3 payload for {scenario.scenario_id} is missing type ids: {', '.join(missing)}")
    return {"primitives_by_type": by_type}


def fixture_choice_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": CHOICE_PROMPT_VERSION,
        "household_policy_choices": [
            {
                "type_id": str(row["type_id"]),
                **fixture_choice(scenario, row),
                "reason": "deterministic constrained-choice fixture",
            }
            for _, row in type_cells.iterrows()
        ],
    }


def fixture_choice(scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, Any]:
    buffer_months = float(type_cell.get("liquid_buffer_months", 2.0))
    low_buffer = float(np.clip((2.0 - buffer_months) / 2.0, 0.0, 1.0))
    if scenario.scenario_type == "income_loss":
        return {
            "policy_family": "income_loss_smoothing_rule",
            "base_response_share": 0.015,
            "liquidity_adjustment": 0.025 * low_buffer,
            "shock_size_damping": 0.0,
            "debt_repayment_share": 0.06,
            "durable_fraction": 0.22,
            "income_loss_sensitivity": 0.18,
            "exhaustion_boost": 0.05,
            "confidence": 0.55,
        }
    return {
        "policy_family": "windfall_damped_rule" if "lottery" in scenario.scenario_id else "liquidity_buffer_rule",
        "base_response_share": 0.30,
        "liquidity_adjustment": 0.16,
        "shock_size_damping": 0.25 if "lottery" in scenario.scenario_id else 0.05,
        "debt_repayment_share": 0.18,
        "durable_fraction": 0.20,
        "income_loss_sensitivity": 0.0,
        "exhaustion_boost": 0.0,
        "confidence": 0.55,
    }


def fixture_primitive_v3_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": PRIMITIVE_V3_PROMPT_VERSION,
        "household_primitives": [
            {
                "type_id": str(row["type_id"]),
                **fixture_primitive_v3(scenario, row),
                "reason": "deterministic primitive-v3 fixture",
            }
            for _, row in type_cells.iterrows()
        ],
    }


def fixture_primitive_v3(scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, Any]:
    buffer_months = float(type_cell.get("liquid_buffer_months", 2.0))
    transfer_ratio = scenario.transfer_amount / max(float(type_cell.get("annual_income", 50000.0)), 1.0)
    if scenario.scenario_type == "income_loss":
        severity = float(np.clip(scenario.income_loss_pct / 0.35, 0.0, 1.0))
        exhaustion = 1.0 if scenario.scenario_id == "ui_exhaustion_income_loss_style" else 0.0
        ongoing = 1.0 if scenario.scenario_id == "ui_receipt_monthly_path_style" else 0.0
        return {
            "perceived_job_loss_risk_pp": float(np.clip(40.0 + 25.0 * exhaustion - 12.0 * ongoing, 0.0, 100.0)),
            "expected_income_growth_pct": float(np.clip(-3.0 - 9.0 * severity - 3.0 * exhaustion + 2.0 * ongoing, -20.0, 20.0)),
            "precautionary_saving_motive": float(np.clip(0.35 + 0.25 * severity + 0.12 * exhaustion, 0.0, 1.0)),
            "liquidity_stress": float(np.clip(0.55 + 0.25 * severity - 0.08 * buffer_months, 0.0, 1.0)),
            "debt_repayment_urgency": 0.35,
            "durable_purchase_pull_forward": 0.03,
            "shock_size_normalized": severity,
            "shock_size_log_income_ratio": float(np.clip(np.log10(max(abs(scenario.income_loss_pct) / 12.0, 1e-4)), -4.0, 2.0)),
            "income_change_attention": float(np.clip(0.55 + 0.25 * severity, 0.0, 1.0)),
            "predictable_drop_attention": float(np.clip(0.25 + 0.50 * exhaustion, 0.0, 1.0)),
            "windfall_permanence_belief": 0.05,
            "spending_commitment_share": 0.55,
            "confidence": 0.60,
        }
    return {
        "perceived_job_loss_risk_pp": 8.0,
        "expected_income_growth_pct": 0.8,
        "precautionary_saving_motive": 0.30,
        "liquidity_stress": float(np.clip(0.65 - 0.10 * buffer_months, 0.0, 1.0)),
        "debt_repayment_urgency": 0.25,
        "durable_purchase_pull_forward": 0.22,
        "shock_size_normalized": float(np.clip(transfer_ratio / 0.50, 0.0, 1.0)),
        "shock_size_log_income_ratio": float(np.clip(np.log10(max(transfer_ratio, 1e-4)), -4.0, 2.0)),
        "income_change_attention": 0.55,
        "predictable_drop_attention": 0.05,
        "windfall_permanence_belief": 0.10 if "lottery" in scenario.scenario_id else 0.25,
        "spending_commitment_share": 0.45,
        "confidence": 0.60,
    }


def choice_policy_action(scenario: BehaviorScenario, type_cell: pd.Series, choice: dict[str, Any]) -> dict[str, Any]:
    liquidity_group = liquidity_group_for_type(type_cell)
    liquidity_sign = 1.0 if liquidity_group == "low" else (-1.0 if liquidity_group == "high" else 0.0)
    base = float(choice["base_response_share"]) + liquidity_sign * float(choice["liquidity_adjustment"])
    if scenario.scenario_type == "income_loss":
        response = (
            base
            + float(choice["income_loss_sensitivity"]) * float(scenario.income_loss_pct)
            + (float(choice["exhaustion_boost"]) if scenario.scenario_id == "ui_exhaustion_income_loss_style" else 0.0)
        )
        if scenario.scenario_id == "ui_receipt_monthly_path_style":
            response *= 0.20
    else:
        shock_log = np.log10(max(scenario.transfer_amount / max(float(type_cell.get("annual_income", 50000.0)), 1.0), 1e-4))
        response = base - float(choice["shock_size_damping"]) * max(0.0, shock_log + 1.0)
    total = float(np.clip(response, 0.0, 0.90))
    durable = total * float(np.clip(choice["durable_fraction"], 0.0, 0.70))
    nondurable = max(0.0, total - durable)
    debt = float(np.clip(choice["debt_repayment_share"], 0.0, 0.50))
    liquid = max(0.0, 1.0 - total - debt)
    return normalize_action(
        {
            "total_spending_share": total,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": debt,
            "liquid_saving_share": liquid,
            "confidence": float(choice["confidence"]),
            "reason": f"{choice['policy_family']}: {choice.get('reason', '')}"[:300],
        }
    )


def primitive_v3_policy_action(scenario: BehaviorScenario, type_cell: pd.Series, primitive: dict[str, Any]) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        income_loss = max(0.0, -float(primitive["expected_income_growth_pct"]) / 20.0)
        total = (
            0.006
            + 0.025 * float(primitive["liquidity_stress"])
            + 0.030 * float(primitive["income_change_attention"])
            + 0.020 * float(primitive["precautionary_saving_motive"])
            + 0.040 * income_loss
            + 0.035 * float(primitive["predictable_drop_attention"])
            - 0.020 * float(primitive["spending_commitment_share"])
        )
        if scenario.scenario_id == "ui_receipt_monthly_path_style":
            total *= 0.20
        total = float(np.clip(total, 0.0, 0.30))
        durable_fraction = 0.18
        debt = 0.04 + 0.08 * float(primitive["debt_repayment_urgency"])
    else:
        shock_log = float(primitive["shock_size_log_income_ratio"])
        transitory = 1.0 - float(primitive["windfall_permanence_belief"])
        total = (
            0.20
            + 0.26 * float(primitive["liquidity_stress"])
            + 0.08 * float(primitive["income_change_attention"])
            + 0.08 * float(primitive["durable_purchase_pull_forward"])
            - 0.14 * float(primitive["precautionary_saving_motive"])
            - 0.18 * transitory * max(0.0, shock_log + 0.4)
        )
        total = float(np.clip(total, 0.02, 0.85))
        durable_fraction = float(np.clip(0.12 + 0.35 * float(primitive["durable_purchase_pull_forward"]), 0.02, 0.65))
        debt = 0.04 + 0.30 * float(primitive["debt_repayment_urgency"])
    durable = total * durable_fraction
    nondurable = max(0.0, total - durable)
    return normalize_action(
        {
            "total_spending_share": total,
            "nondurable_spending_share": nondurable,
            "durable_spending_share": durable,
            "debt_repayment_share": float(np.clip(debt, 0.0, 0.50)),
            "liquid_saving_share": max(0.0, 1.0 - total - float(np.clip(debt, 0.0, 0.50))),
            "confidence": float(primitive["confidence"]),
            "reason": f"{PRIMITIVE_V3_POLICY_VERSION}: richer primitive interface",
        }
    )


def liquidity_group_for_type(type_cell: pd.Series) -> str:
    type_id = str(type_cell["type_id"])
    if type_id in {"liquid_poor_renter", "wealthy_htm_homeowner", "unemployed_low_liquid"}:
        return "low"
    if type_id in {"retiree_liquid_assets", "high_income_illiquid_rich", "business_owner_top_wealth"}:
        return "high"
    return "middle"


def action_row(scenario: BehaviorScenario, type_cell: pd.Series, response: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "schema_version": ARCHITECTURE_FIDELITY_VERSION,
        "scenario_id": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "source": source,
        "type_id": str(type_cell["type_id"]),
        "population_weight": float(type_cell["population_weight"]),
        "liquidity_group": liquidity_group_for_type(type_cell),
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


def normalize_action(row: dict[str, Any]) -> dict[str, Any]:
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


def build_fidelity_table(
    scores: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    provider: str,
    model: str,
    tolerance: float,
) -> pd.DataFrame:
    raw_source = f"llm_{provider}_{model}"
    sources = [
        (raw_source, "constrained_raw", "low"),
        (f"{CHOICE_SOURCE_PREFIX}_{provider}_{model}", "constrained_choice", "medium"),
        (f"{PRIMITIVE_V3_SOURCE_PREFIX}_{provider}_{model}", "primitive_v3", "high"),
    ]
    raw_ui_rmse = lookup_rmse(scores, source=raw_source, split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
    rows: list[dict[str, Any]] = []
    for source, architecture, tier in sources:
        ui_rmse = lookup_rmse(scores, source=source, split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
        baseline = lookup_comparison(comparison, source=source, split=BEHAVIOR_UI_EXHAUSTION_HOLDOUT_SPLIT)
        rows.append(
            {
                "source": source,
                "architecture": architecture,
                "interpretability_tier": tier,
                "interpretability_rank": INTERPRETABILITY_RANK[architecture],
                "selection_rmse": lookup_rmse(scores, source=source, split=BEHAVIOR_SELECTION_SPLIT),
                "ui_holdout_rmse": ui_rmse,
                "raw_ui_holdout_rmse": raw_ui_rmse,
                "within_25pct_of_raw_ui": bool(np.isfinite(ui_rmse) and np.isfinite(raw_ui_rmse) and ui_rmse <= raw_ui_rmse * (1.0 + tolerance)),
                "best_ui_baseline_source": baseline.get("best_baseline_source"),
                "best_ui_baseline_rmse": baseline.get("best_baseline_rmse_range", np.nan),
                "ui_delta_vs_best_baseline": baseline.get("rmse_range_delta_vs_baseline", np.nan),
                "ui_baseline_verdict": baseline.get("baseline_verdict", "unmeasured"),
            }
        )
    out = pd.DataFrame(rows)
    eligible = out[out["within_25pct_of_raw_ui"]].sort_values(["interpretability_rank", "ui_holdout_rmse"], ascending=[False, True])
    out["recommended"] = False
    if not eligible.empty:
        out.loc[out["source"].eq(str(eligible.iloc[0]["source"])), "recommended"] = True
    return out.sort_values(["interpretability_rank", "ui_holdout_rmse"], ascending=[False, True]).reset_index(drop=True)


def lookup_rmse(scores: pd.DataFrame, *, source: str, split: str) -> float:
    row = scores[
        (scores["source"].astype(str) == source)
        & (scores["evaluation_split"].astype(str) == split)
        & (scores["target_scope"].astype(str) == "aggregate")
        & (scores["target_family"].astype(str) == "ALL")
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["rmse_range"])


def lookup_comparison(comparison: pd.DataFrame, *, source: str, split: str) -> dict[str, Any]:
    row = comparison[
        (comparison["source"].astype(str) == source)
        & (comparison["evaluation_split"].astype(str) == split)
        & (comparison["target_scope"].astype(str) == "aggregate")
        & (comparison["target_family"].astype(str) == "ALL")
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def locked_architecture_manifest() -> dict[str, Any]:
    return {
        "A_constrained_raw": {
            "source_prefix": "llm",
            "description": "LLM emits final bounded allocation/drop shares; deterministic code clips budgets.",
        },
        "B_constrained_choice": {
            "source_prefix": CHOICE_SOURCE_PREFIX,
            "prompt_version": CHOICE_PROMPT_VERSION,
            "policy_families": ["liquidity_buffer_rule", "windfall_damped_rule", "income_loss_smoothing_rule"],
            "fields": CHOICE_FIELDS,
            "development_split": BEHAVIOR_SELECTION_SPLIT,
        },
        "C_primitive_v3": {
            "source_prefix": PRIMITIVE_V3_SOURCE_PREFIX,
            "prompt_version": PRIMITIVE_V3_PROMPT_VERSION,
            "policy_version": PRIMITIVE_V3_POLICY_VERSION,
            "fields": PRIMITIVE_V3_FIELDS,
            "added_fields_rationale": {
                "income_change_attention": "separates salience of income changes from generic job risk",
                "predictable_drop_attention": "represents whether a predictable income drop is actually attended to",
                "windfall_permanence_belief": "separates transitory windfalls from persistent income changes",
                "spending_commitment_share": "captures slow-to-adjust baseline spending commitments in income-loss scenarios",
            },
            "development_split": BEHAVIOR_SELECTION_SPLIT,
        },
    }


def _recommended_architecture(fidelity: pd.DataFrame) -> dict[str, Any]:
    if fidelity.empty or not bool(fidelity["recommended"].any()):
        return {"status": "none", "reason": "No architecture was within tolerance of raw UI holdout fidelity."}
    row = fidelity[fidelity["recommended"]].iloc[0]
    return {
        "status": "selected",
        "source": str(row["source"]),
        "architecture": str(row["architecture"]),
        "interpretability_tier": str(row["interpretability_tier"]),
        "ui_holdout_rmse": float(row["ui_holdout_rmse"]),
        "raw_ui_holdout_rmse": float(row["raw_ui_holdout_rmse"]),
    }


def build_report(
    manifest: dict[str, Any],
    fidelity: pd.DataFrame,
    ui_comparison: pd.DataFrame,
    selection_scores: pd.DataFrame,
    ui_scores: pd.DataFrame,
) -> str:
    recommendation = _recommended_architecture(fidelity)
    verdict = (
        f"Recommended architecture: `{recommendation['architecture']}` (`{recommendation['source']}`), "
        f"UI RMSE `{recommendation['ui_holdout_rmse']:.4f}` versus raw `{recommendation['raw_ui_holdout_rmse']:.4f}`."
        if recommendation.get("status") == "selected"
        else "No interpretable architecture stayed within the locked fidelity tolerance."
    )
    lines = [
        "# Behavior Architecture Fidelity",
        "",
        "## Bottom Line",
        verdict,
        "",
        "This is an architecture-fidelity experiment, not a new broad macro-validity claim. "
        "The UI holdout is the confirmatory surface for B/C in this run; lottery remains frozen.",
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- Mode: `{manifest.get('mode')}`",
        f"- Selection split: `{manifest.get('selection_split')}`",
        f"- Holdout split: `{manifest.get('holdout_split')}`",
        f"- Fidelity tolerance: `{manifest.get('fidelity_tolerance')}`",
        f"- UI leakage note: {manifest.get('ui_holdout_leakage_note')}",
        "",
        "## Fidelity Table",
        markdown_table(fidelity),
        "",
        "## UI Baseline Comparison",
        markdown_table(ui_comparison),
        "",
        "## Selection Scores",
        markdown_table(selection_scores.sort_values(["target_family", "rmse_range", "source"])),
        "",
        "## UI Scores",
        markdown_table(ui_scores.sort_values(["target_family", "rmse_range", "source"])),
        "",
        "## Locked Architectures",
        "```json",
        json.dumps(manifest.get("locked_architectures", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
