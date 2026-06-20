from __future__ import annotations

import argparse
import json
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


BEHAVIOR_GATE_VERSION = "household_behavior_target_gate_v1"


@dataclass(frozen=True)
class BehaviorScenario:
    scenario_id: str
    label: str
    as_of_date: str
    transfer_amount: float
    horizon_months: int
    prompt_context: str
    contamination_label: str


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
)


BEHAVIOR_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "rebate_2001_style",
        "target_name": "aggregate_nondurable_spending_share_6mo",
        "target_family": "mpc",
        "target_low": 0.20,
        "target_high": 0.40,
        "target_value": 0.30,
        "prediction_column": "aggregate_nondurable_spending_share",
        "source_label": "Johnson Parker Souleles 2006 / CFPB summary",
        "source_url": "https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf",
        "notes": "Prior tax-rebate literature estimates 20-40 percent of checks spent on nondurables and services over six months.",
    },
    {
        "scenario_id": "stimulus_2008_style",
        "target_name": "aggregate_nondurable_spending_share_3mo",
        "target_family": "mpc",
        "target_low": 0.12,
        "target_high": 0.30,
        "target_value": 0.21,
        "prediction_column": "aggregate_nondurable_spending_share",
        "source_label": "Parker Souleles Johnson McClelland 2013",
        "source_url": "https://www.openicpsr.org/openicpsr/project/116117/version/V1/view",
        "notes": "2008 ESP estimates show 12-30 percent of payments spent on nondurables in the receipt quarter.",
    },
    {
        "scenario_id": "stimulus_2008_style",
        "target_name": "aggregate_total_spending_share_6mo",
        "target_family": "mpc",
        "target_low": 0.50,
        "target_high": 0.90,
        "target_value": 0.70,
        "prediction_column": "aggregate_total_spending_share",
        "source_label": "Parker Souleles Johnson McClelland 2013",
        "source_url": "https://www.openicpsr.org/openicpsr/project/116117/version/V1/view",
        "notes": "Including durables, the average total spending response is reported as 50-90 percent of payments.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "aggregate_total_spending_share_10d",
        "target_family": "mpc",
        "target_low": 0.266,
        "target_high": 0.369,
        "target_value": 0.33,
        "prediction_column": "aggregate_total_spending_share",
        "source_label": "CFPB/Pagel et al. 2020",
        "source_url": "https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf",
        "notes": "Estimated weighted MPC is about 0.266 and immediate spending increases by roughly one third of the payment.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "aggregate_debt_repayment_share",
        "target_family": "debt_saving",
        "target_low": 0.30,
        "target_high": 0.40,
        "target_value": 0.35,
        "prediction_column": "aggregate_debt_repayment_share",
        "source_label": "NY Fed SCE / Liberty Street Economics 2020",
        "source_url": "https://libertystreeteconomics.newyorkfed.org/2020/10/how-have-households-used-their-stimulus-payments-and-how-would-they-spend-the-next/",
        "notes": "The June 2020 SCE module reports 35 percent of first-round EIP funds used to pay down debt.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "aggregate_liquid_saving_share",
        "target_family": "debt_saving",
        "target_low": 0.31,
        "target_high": 0.41,
        "target_value": 0.36,
        "prediction_column": "aggregate_liquid_saving_share",
        "source_label": "NY Fed SCE / Liberty Street Economics 2020",
        "source_url": "https://libertystreeteconomics.newyorkfed.org/2020/10/how-have-households-used-their-stimulus-payments-and-how-would-they-spend-the-next/",
        "notes": "The June 2020 SCE module reports 36 percent of first-round EIP funds saved.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "low_liquidity_total_spending_share",
        "target_family": "liquidity_gradient",
        "target_low": 0.40,
        "target_high": 0.70,
        "target_value": 0.45,
        "prediction_column": "low_liquidity_total_spending_share",
        "source_label": "CFPB/Pagel et al. 2020",
        "source_url": "https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf",
        "notes": "Users with very low balances have MPCs of 0.4 or above.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "high_liquidity_total_spending_share",
        "target_family": "liquidity_gradient",
        "target_low": 0.05,
        "target_high": 0.15,
        "target_value": 0.10,
        "prediction_column": "high_liquidity_total_spending_share",
        "source_label": "CFPB/Pagel et al. 2020",
        "source_url": "https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf",
        "notes": "Users with high balances have MPCs on the order of 0.1.",
    },
    {
        "scenario_id": "eip_2020_style",
        "target_name": "low_high_liquidity_spending_ratio",
        "target_family": "liquidity_gradient",
        "target_low": 3.0,
        "target_high": 6.0,
        "target_value": 4.0,
        "prediction_column": "low_high_liquidity_spending_ratio",
        "source_label": "CFPB/Pagel et al. 2020",
        "source_url": "https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf",
        "notes": "Low-balance MPCs are roughly four times high-balance MPCs using 0.4+ versus about 0.1.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct household behavior target gate.")
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--behavior-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.behavior_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --behavior-mode live is used")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"behavior_gate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fresh_behavior_cache" if args.fresh_cache else WORK_ROOT / "behavior_llm_cache"

    manifest: dict[str, Any] = {
        "schema_version": BEHAVIOR_GATE_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "behavior_mode": args.behavior_mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "scf_wave": int(args.scf_wave),
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        scenarios = scenarios_to_frame(BEHAVIOR_SCENARIOS)
        targets = behavior_targets_frame()
        llm_client = BehaviorLLMClient(args.provider, args.model, cache_dir, mode=args.behavior_mode, max_live_calls=args.max_live_calls)
        actions = run_behavior_gate(BEHAVIOR_SCENARIOS, type_cells, llm_client=llm_client)
        controls = run_behavior_controls(BEHAVIOR_SCENARIOS, type_cells)
        all_actions = pd.concat([actions, controls], ignore_index=True)
        aggregates = aggregate_behavior_actions(all_actions)
        scores = score_behavior_targets(aggregates, targets)
        scenarios.to_csv(output_dir / "behavior_scenarios.csv", index=False)
        targets.to_csv(output_dir / "behavior_targets.csv", index=False)
        type_cells.to_csv(output_dir / "household_type_cells.csv", index=False)
        all_actions.to_csv(output_dir / "household_behavior_actions.csv", index=False)
        aggregates.to_csv(output_dir / "behavior_aggregates.csv", index=False)
        scores.to_csv(output_dir / "behavior_target_scores.csv", index=False)
        (output_dir / "behavior_llm_raw_records.json").write_text(json.dumps(llm_client.raw_records, indent=2, sort_keys=True), encoding="utf-8")
        manifest.update(
            {
                "status": "ok",
                "scenario_count": len(BEHAVIOR_SCENARIOS),
                "target_count": int(targets.shape[0]),
                "household_type_status": type_status,
                "household_type_count": int(type_cells.shape[0]),
                "action_rows": int(all_actions.shape[0]),
                "aggregate_rows": int(aggregates.shape[0]),
                "score_rows": int(scores.shape[0]),
                "live_call_count": int(llm_client.live_call_count),
                "cache_hit_count": int(llm_client.cache_hit_count),
                "cache_dir": str(cache_dir.relative_to(Path.cwd()) if cache_dir.is_relative_to(Path.cwd()) else cache_dir),
                "outputs": [
                    "behavior_scenarios.csv",
                    "behavior_targets.csv",
                    "household_type_cells.csv",
                    "household_behavior_actions.csv",
                    "behavior_aggregates.csv",
                    "behavior_target_scores.csv",
                    "behavior_llm_raw_records.json",
                    "behavior_gate_report.md",
                ],
            }
        )
        report = build_behavior_gate_report(manifest, scenarios, targets, aggregates, scores)
        (output_dir / "behavior_gate_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


class BehaviorLLMClient:
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
            prompt = behavior_prompt(scenario, type_cells)
            data = self._client._codex_call(prompt, f"behavior_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}")
        normalized = normalize_behavior_payload(scenario, type_cells, data)
        self.raw_records.append(
            {
                "scenario_id": scenario.scenario_id,
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


def behavior_prompt(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> str:
    payload = {
        "prompt_version": BEHAVIOR_GATE_VERSION,
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
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_behavior_payload(scenario: BehaviorScenario, type_cells: pd.DataFrame) -> dict[str, Any]:
    return {
        "prompt_version": BEHAVIOR_GATE_VERSION,
        "household_actions": [
            {
                "type_id": str(row["type_id"]),
                **_liquidity_rule_response(scenario, row),
                "reason": "deterministic liquidity fixture",
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


def aggregate_behavior_actions(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in actions.groupby(["scenario_id", "source"], dropna=False):
        scenario_id, source = keys
        low = group[group["liquidity_group"] == "low"]
        high = group[group["liquidity_group"] == "high"]
        low_spend = _weighted_average(low, "total_spending_share")
        high_spend = _weighted_average(high, "total_spending_share")
        rows.append(
            {
                "scenario_id": scenario_id,
                "source": source,
                "n_types": int(group.shape[0]),
                "aggregate_total_spending_share": _weighted_average(group, "total_spending_share"),
                "aggregate_nondurable_spending_share": _weighted_average(group, "nondurable_spending_share"),
                "aggregate_durable_spending_share": _weighted_average(group, "durable_spending_share"),
                "aggregate_debt_repayment_share": _weighted_average(group, "debt_repayment_share"),
                "aggregate_liquid_saving_share": _weighted_average(group, "liquid_saving_share"),
                "low_liquidity_total_spending_share": low_spend,
                "high_liquidity_total_spending_share": high_spend,
                "low_high_liquidity_spending_ratio": low_spend / max(high_spend, 1e-9),
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
    for keys, group in joined.groupby(["source", "target_family"], dropna=False):
        source, target_family = keys
        rows.append(_score_group(group, source=source, target_family=target_family))
    for source, group in joined.groupby("source", dropna=False):
        rows.append(_score_group(group, source=source, target_family="ALL"))
    return pd.DataFrame(rows).sort_values(["target_family", "rmse_range", "source"]).reset_index(drop=True)


def behavior_targets_frame() -> pd.DataFrame:
    return pd.DataFrame(BEHAVIOR_TARGETS)


def scenarios_to_frame(scenarios: Iterable[BehaviorScenario]) -> pd.DataFrame:
    return pd.DataFrame([scenario.__dict__ for scenario in scenarios])


def build_behavior_gate_report(manifest: dict[str, Any], scenarios: pd.DataFrame, targets: pd.DataFrame, aggregates: pd.DataFrame, scores: pd.DataFrame) -> str:
    lines = [
        "# Household Behavior Target Gate",
        "",
        "## Bottom Line",
        _behavior_bottom_line(scores),
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- Behavior mode: `{manifest.get('behavior_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Scenario count: `{manifest.get('scenario_count')}`",
        f"- Target count: `{manifest.get('target_count')}`",
        f"- SCF type source: `{manifest.get('household_type_status', {}).get('status', 'unknown')}`",
        "",
        "## Scoreboard",
        markdown_table(scores.sort_values(["target_family", "rmse_range", "source"])),
        "",
        "## Aggregates",
        markdown_table(aggregates.sort_values(["scenario_id", "source"]).head(32)),
        "",
        "## Targets",
        markdown_table(targets[["scenario_id", "target_name", "target_low", "target_high", "target_value", "prediction_column", "source_label"]]),
        "",
        "## What This Gate Means",
        (
            "This gate scores household behavior directly against public stimulus-response moments. "
            "It is still historical and contamination-confounded, but it tests the agent layer on spending, saving, "
            "debt repayment, and liquidity gradients rather than only on macro forecasts or survey beliefs."
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
        "source": source,
        "type_id": str(type_cell["type_id"]),
        "population_weight": float(type_cell["population_weight"]),
        "liquidity_group": _liquidity_group(type_cell),
        "transfer_amount": float(scenario.transfer_amount),
        "total_spending_share": float(response["total_spending_share"]),
        "nondurable_spending_share": float(response["nondurable_spending_share"]),
        "durable_spending_share": float(response["durable_spending_share"]),
        "debt_repayment_share": float(response["debt_repayment_share"]),
        "liquid_saving_share": float(response["liquid_saving_share"]),
        "confidence": float(response.get("confidence", 0.5)),
        "reason": str(response.get("reason", ""))[:300],
    }


def _liquidity_rule_response(_scenario: BehaviorScenario, type_cell: pd.Series) -> dict[str, Any]:
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


def _flat_rule_response(_scenario: BehaviorScenario, _type_cell: pd.Series) -> dict[str, Any]:
    return {
        "total_spending_share": 0.30,
        "nondurable_spending_share": 0.24,
        "durable_spending_share": 0.06,
        "debt_repayment_share": 0.20,
        "liquid_saving_share": 0.50,
        "confidence": 0.50,
        "reason": "flat 30 percent spending rule",
    }


def _permanent_income_response(_scenario: BehaviorScenario, _type_cell: pd.Series) -> dict[str, Any]:
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


def _liquidity_group(type_cell: pd.Series) -> str:
    type_id = str(type_cell["type_id"])
    if type_id in {"liquid_poor_renter", "wealthy_htm_homeowner", "unemployed_low_liquid"}:
        return "low"
    if type_id in {"retiree_liquid_assets", "high_income_illiquid_rich", "business_owner_top_wealth"}:
        return "high"
    return "middle"


def _weighted_average(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return float("nan")
    weights = group["population_weight"].astype(float).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        weights = pd.Series(np.ones(len(group)) / max(1, len(group)), index=group.index)
        total = 1.0
    return float((group[column].astype(float) * weights).sum() / total)


def _range_error(row: pd.Series) -> float:
    prediction = float(row["prediction"])
    low = float(row["target_low"])
    high = float(row["target_high"])
    if low <= prediction <= high:
        return 0.0
    return low - prediction if prediction < low else prediction - high


def _score_group(group: pd.DataFrame, *, source: str, target_family: str) -> dict[str, Any]:
    range_error = group["range_error"].astype(float)
    point_error = group["point_error"].astype(float)
    return {
        "source": source,
        "target_family": target_family,
        "n": int(group.shape[0]),
        "rmse_range": float(np.sqrt(np.mean(np.square(range_error)))),
        "mae_range": float(np.mean(np.abs(range_error))),
        "rmse_point": float(np.sqrt(np.mean(np.square(point_error)))),
        "mae_point": float(np.mean(np.abs(point_error))),
        "mean_prediction": float(group["prediction"].mean()),
        "mean_target": float(group["target_value"].mean()),
    }


def _behavior_bottom_line(scores: pd.DataFrame) -> str:
    if scores.empty:
        return "Behavior target scoring produced no rows."
    overall = scores[scores["target_family"] == "ALL"].sort_values("rmse_range")
    if overall.empty:
        return "Behavior target scoring produced no overall row."
    best = overall.iloc[0]
    return f"Best overall source is `{best['source']}` with range RMSE `{float(best['rmse_range']):.4f}` across `{int(best['n'])}` behavior targets."


if __name__ == "__main__":
    raise SystemExit(main())
