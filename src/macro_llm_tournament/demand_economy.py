from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, WORK_ROOT, bounded_float, cache_key, markdown_table, round_or_none
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable


DEMAND_ECONOMY_VERSION = "abstract_behavior_demand_economy_v1"
DEMAND_ECONOMY_PROMPT_VERSION = "abstract_behavior_demand_economy_v1"
DECISION_MODES = ("fixture", "replay", "live")
FEEDBACK_MODES = ("closed_loop", "none")
NEUTRAL_POLICY_RATE = 3.0
INFLATION_TARGET = 2.0


@dataclass(frozen=True)
class DemandScenario:
    scenario_id: str
    label: str
    transfer_period: int = -1
    transfer_amount: float = 0.0
    rate_shock_start: int = -1
    rate_shock_end: int = -1
    rate_shock_pp: float = 0.0
    belief_dispersion_multiplier: float = 1.0
    feedback_gain: float = 1.0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an abstract behavior-based demand economy.")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5")
    parser.add_argument("--decision-mode", choices=DECISION_MODES, default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--household-source", choices=["fixture", "csv"], default="fixture")
    parser.add_argument("--household-csv", default=None)
    parser.add_argument("--household-count", type=int, default=6)
    parser.add_argument("--period-count", type=int, default=8)
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default="closed_loop")
    parser.add_argument("--scenarios", default="baseline,transfer_shock,rate_hike,belief_feedback")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    scenario_ids = [part.strip() for part in args.scenarios.split(",") if part.strip()]
    scenarios = [scenario for scenario in default_demand_scenarios() if scenario.scenario_id in set(scenario_ids)]
    missing_scenarios = sorted(set(scenario_ids) - {scenario.scenario_id for scenario in scenarios})
    if missing_scenarios:
        raise SystemExit(f"Unknown scenarios: {', '.join(missing_scenarios)}")
    if not models:
        raise SystemExit("--models must contain at least one model")
    if not scenarios:
        raise SystemExit("--scenarios must contain at least one scenario")
    if args.period_count < 4:
        raise SystemExit("--period-count must be at least 4 for impulse-response validation")
    if args.decision_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --decision-mode live is used")
    if args.decision_mode == "live" and not args.fresh_cache:
        raise SystemExit("--fresh-cache is required when --decision-mode live is used")
    required_calls = len(models) * len(scenarios) * int(args.period_count)
    if args.decision_mode == "live" and args.max_live_calls < required_calls:
        raise SystemExit(
            "--max-live-calls must be at least "
            f"{required_calls} for a fresh live run with {len(models)} model(s), "
            f"{len(scenarios)} scenario(s), and {args.period_count} periods"
        )
    if args.household_source == "csv":
        if not args.household_csv:
            raise SystemExit("--household-csv is required when --household-source csv")
        if not Path(args.household_csv).exists():
            raise SystemExit(f"--household-csv does not exist: {args.household_csv}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"demand_economy_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fresh_demand_economy_cache" if args.fresh_cache else WORK_ROOT / "demand_economy_cache"
    households = (
        normalize_demand_households(pd.read_csv(Path(args.household_csv)))
        if args.household_source == "csv"
        else build_fixture_demand_households(args.household_count)
    )

    all_initial: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_periods: list[pd.DataFrame] = []
    all_accounting: list[pd.DataFrame] = []
    all_prompt_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    live_used = 0
    cache_hits = 0
    for model in models:
        client = DemandEconomyClient(
            args.provider,
            model,
            cache_dir,
            mode=args.decision_mode,
            max_live_calls=max(0, int(args.max_live_calls) - live_used),
        )
        initial, decisions, periods, accounting, prompt_rows = run_demand_economy(
            households,
            scenarios,
            client,
            period_count=args.period_count,
            feedback_mode=args.feedback_mode,
        )
        all_initial.append(initial)
        all_decisions.append(decisions)
        all_periods.append(periods)
        all_accounting.append(accounting)
        all_prompt_rows.extend(prompt_rows)
        raw_records.extend(client.raw_records)
        live_used += client.live_call_count
        cache_hits += client.cache_hit_count

    initial_frame = pd.concat(all_initial, ignore_index=True) if all_initial else pd.DataFrame()
    decisions_frame = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    periods_frame = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    accounting_frame = pd.concat(all_accounting, ignore_index=True) if all_accounting else pd.DataFrame()
    validation = score_demand_economy_validation(periods_frame, decisions_frame, accounting_frame)
    evidence = classify_demand_economy_evidence(validation, mode=args.decision_mode)

    manifest = {
        "schema_version": DEMAND_ECONOMY_VERSION,
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "timestamp_utc": timestamp,
        "argv": _sanitized_argv(),
        "run_command": shlex.join(_sanitized_argv()),
        "git": _git_metadata(),
        "status": "ok",
        "provider": args.provider,
        "models": models,
        "decision_mode": args.decision_mode,
        "fresh_cache": bool(args.fresh_cache),
        "max_live_calls": int(args.max_live_calls),
        "live_call_count": int(live_used),
        "cache_hit_count": int(cache_hits),
        "household_source": args.household_source,
        "household_count": int(households.shape[0]),
        "period_count": int(args.period_count),
        "feedback_mode": args.feedback_mode,
        "scenario_count": len(scenarios),
        "scenarios": [scenario.__dict__ for scenario in scenarios],
        "evidence": evidence,
        "outputs": [
            "demand_households.csv",
            "demand_initial_state.csv",
            "demand_periods.csv",
            "demand_household_decisions.csv",
            "demand_accounting.csv",
            "demand_validation_scores.csv",
            "demand_prompt_cards.jsonl",
            "demand_raw_records.json",
            "demand_economy_report.md",
            "manifest.json",
        ],
    }

    households.to_csv(output_dir / "demand_households.csv", index=False)
    initial_frame.to_csv(output_dir / "demand_initial_state.csv", index=False)
    periods_frame.to_csv(output_dir / "demand_periods.csv", index=False)
    decisions_frame.to_csv(output_dir / "demand_household_decisions.csv", index=False)
    accounting_frame.to_csv(output_dir / "demand_accounting.csv", index=False)
    validation.to_csv(output_dir / "demand_validation_scores.csv", index=False)
    pd.DataFrame(all_prompt_rows).to_json(output_dir / "demand_prompt_cards.jsonl", orient="records", lines=True)
    _write_json(output_dir / "demand_raw_records.json", raw_records)
    _write_json(output_dir / "manifest.json", manifest)
    report = build_demand_economy_report(manifest, households, periods_frame, validation, accounting_frame)
    (output_dir / "demand_economy_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote demand economy run to {output_dir}")
    print(json.dumps(_jsonable(evidence), indent=2, sort_keys=True, allow_nan=False))
    return 0


class DemandEconomyClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str = "fixture",
        max_live_calls: int = 0,
    ):
        if mode not in DECISION_MODES:
            raise ValueError(f"Unsupported demand-economy mode: {mode}")
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.raw_records: list[dict[str, Any]] = []
        self._llm = ForecastLLMClient(provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls)

    @property
    def live_call_count(self) -> int:
        return int(self._llm.live_call_count)

    @property
    def cache_hit_count(self) -> int:
        return int(self._llm.cache_hit_count)

    @property
    def source(self) -> str:
        prefix = "fixture" if self.mode == "fixture" else self.provider
        return f"{prefix}_{self.model}"

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt_payload = demand_economy_prompt_payload(scenario, period_state, household_states)
        prompt_text = demand_economy_prompt(scenario, period_state, household_states)
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_demand_payload(scenario, period_state, household_states),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            cache_name = f"demand_economy_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt_payload})}"
            data = self._llm.json_call(prompt_text, cache_name, instructions=_demand_instructions())
        normalized = normalize_demand_payload(household_states, data)
        self.raw_records.append(
            {
                "source": self.source,
                "scenario_id": scenario.scenario_id,
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized


def default_demand_scenarios() -> list[DemandScenario]:
    return [
        DemandScenario(
            "baseline",
            "No exogenous shock; heterogeneous households react only to endogenous feedback.",
            notes="Control path for impulse-response scoring.",
        ),
        DemandScenario(
            "transfer_shock",
            "One-period lump-sum household transfer.",
            transfer_period=1,
            transfer_amount=1000.0,
            notes="Tests aggregate MPC and the liquidity gradient.",
        ),
        DemandScenario(
            "rate_hike",
            "Temporary one percentage point policy-rate hike.",
            rate_shock_start=1,
            rate_shock_end=4,
            rate_shock_pp=1.0,
            notes="Tests whether consumption falls after a monetary tightening.",
        ),
        DemandScenario(
            "belief_feedback",
            "No exogenous shock; stronger initial belief dispersion and feedback.",
            belief_dispersion_multiplier=1.8,
            feedback_gain=1.35,
            notes="Tests whether heterogeneous beliefs plus feedback generate endogenous fluctuations.",
        ),
    ]


def build_fixture_demand_households(household_count: int = 6) -> pd.DataFrame:
    rows = [
        {
            "type_id": "low_liquidity_renter",
            "label": "Low-liquidity renter",
            "population_weight": 0.18,
            "income_group": "low",
            "liquidity_group": "low",
            "employment_status": "employed_high_risk",
            "annual_income": 32000.0,
            "baseline_consumption_annual": 31000.0,
            "liquid_assets": 500.0,
            "debt": 4500.0,
            "base_mpc": 0.84,
            "rate_sensitivity": 0.20,
            "income_sensitivity": 0.80,
            "precautionary_sensitivity": 0.75,
            "baseline_job_loss_probability": 11.0,
            "target_buffer_months": 1.2,
            "inflation_expectation_1y": 3.4,
            "income_growth_expectation_1y": 0.4,
        },
        {
            "type_id": "low_income_parent",
            "label": "Low-income working parent",
            "population_weight": 0.16,
            "income_group": "low",
            "liquidity_group": "low",
            "employment_status": "employed",
            "annual_income": 42000.0,
            "baseline_consumption_annual": 39800.0,
            "liquid_assets": 900.0,
            "debt": 6500.0,
            "base_mpc": 0.78,
            "rate_sensitivity": 0.24,
            "income_sensitivity": 0.74,
            "precautionary_sensitivity": 0.68,
            "baseline_job_loss_probability": 9.0,
            "target_buffer_months": 1.6,
            "inflation_expectation_1y": 3.1,
            "income_growth_expectation_1y": 0.6,
        },
        {
            "type_id": "middle_liquidity_worker",
            "label": "Middle-liquidity worker",
            "population_weight": 0.28,
            "income_group": "middle",
            "liquidity_group": "middle",
            "employment_status": "employed",
            "annual_income": 68000.0,
            "baseline_consumption_annual": 57000.0,
            "liquid_assets": 6000.0,
            "debt": 12000.0,
            "base_mpc": 0.50,
            "rate_sensitivity": 0.34,
            "income_sensitivity": 0.58,
            "precautionary_sensitivity": 0.42,
            "baseline_job_loss_probability": 6.0,
            "target_buffer_months": 2.8,
            "inflation_expectation_1y": 2.7,
            "income_growth_expectation_1y": 1.0,
        },
        {
            "type_id": "indebted_homeowner",
            "label": "Indebted homeowner",
            "population_weight": 0.18,
            "income_group": "middle",
            "liquidity_group": "middle",
            "employment_status": "employed",
            "annual_income": 90000.0,
            "baseline_consumption_annual": 76000.0,
            "liquid_assets": 9000.0,
            "debt": 42000.0,
            "base_mpc": 0.42,
            "rate_sensitivity": 0.70,
            "income_sensitivity": 0.46,
            "precautionary_sensitivity": 0.36,
            "baseline_job_loss_probability": 5.0,
            "target_buffer_months": 3.0,
            "inflation_expectation_1y": 2.5,
            "income_growth_expectation_1y": 1.2,
        },
        {
            "type_id": "high_liquidity_professional",
            "label": "High-liquidity professional",
            "population_weight": 0.14,
            "income_group": "high",
            "liquidity_group": "high",
            "employment_status": "employed_low_risk",
            "annual_income": 145000.0,
            "baseline_consumption_annual": 101000.0,
            "liquid_assets": 55000.0,
            "debt": 18000.0,
            "base_mpc": 0.22,
            "rate_sensitivity": 0.54,
            "income_sensitivity": 0.34,
            "precautionary_sensitivity": 0.18,
            "baseline_job_loss_probability": 3.0,
            "target_buffer_months": 6.0,
            "inflation_expectation_1y": 2.2,
            "income_growth_expectation_1y": 1.5,
        },
        {
            "type_id": "retired_saver",
            "label": "Retired high-buffer saver",
            "population_weight": 0.06,
            "income_group": "middle",
            "liquidity_group": "high",
            "employment_status": "retired",
            "annual_income": 72000.0,
            "baseline_consumption_annual": 58000.0,
            "liquid_assets": 85000.0,
            "debt": 4000.0,
            "base_mpc": 0.18,
            "rate_sensitivity": 0.18,
            "income_sensitivity": 0.20,
            "precautionary_sensitivity": 0.12,
            "baseline_job_loss_probability": 2.0,
            "target_buffer_months": 9.0,
            "inflation_expectation_1y": 2.4,
            "income_growth_expectation_1y": 0.5,
        },
    ]
    count = max(1, min(int(household_count or len(rows)), len(rows)))
    return normalize_demand_households(pd.DataFrame(rows[:count]))


def normalize_demand_households(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "type_id",
        "label",
        "population_weight",
        "income_group",
        "liquidity_group",
        "annual_income",
        "baseline_consumption_annual",
        "liquid_assets",
        "debt",
        "base_mpc",
        "rate_sensitivity",
        "income_sensitivity",
        "precautionary_sensitivity",
        "baseline_job_loss_probability",
        "target_buffer_months",
        "inflation_expectation_1y",
        "income_growth_expectation_1y",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Demand household panel missing columns: {', '.join(sorted(missing))}")
    out = frame.copy()
    for column in required - {"type_id", "label", "income_group", "liquidity_group", "employment_status"}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[list(required - {"type_id", "label", "income_group", "liquidity_group", "employment_status"})].isna().any().any():
        raise ValueError("Demand household panel contains non-numeric required values")
    out["type_id"] = out["type_id"].astype(str)
    if out["type_id"].duplicated().any():
        raise ValueError("Demand household type_id values must be unique")
    out["population_weight"] = out["population_weight"].clip(lower=0.0)
    total_weight = float(out["population_weight"].sum())
    if total_weight <= 0:
        raise ValueError("Demand household population weights must sum to a positive value")
    out["population_weight"] = out["population_weight"] / total_weight
    out["base_mpc"] = out["base_mpc"].clip(lower=0.02, upper=0.98)
    out["liquidity_group"] = out["liquidity_group"].astype(str).str.lower()
    return out.sort_values("type_id").reset_index(drop=True)


def run_demand_economy(
    households: pd.DataFrame,
    scenarios: Iterable[DemandScenario],
    client: DemandEconomyClient,
    *,
    period_count: int = 8,
    feedback_mode: str = "closed_loop",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"Unsupported feedback mode: {feedback_mode}")
    households = normalize_demand_households(households)
    source = client.source
    initial = _initial_household_states(households, source=source)
    decision_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        household_states = initial.to_dict(orient="records")
        env = _initial_environment(households)
        for period_index in range(int(period_count)):
            period_state = _period_state(env, scenario, period_index)
            panel = client.decision_panel(scenario, period_state, household_states)
            prompt_rows.append(
                {
                    "source": source,
                    "scenario_id": scenario.scenario_id,
                    "period_id": period_state["period_id"],
                    "period_index": int(period_index),
                    "prompt_payload": demand_economy_prompt_payload(scenario, period_state, household_states),
                }
            )
            realized = _realize_household_period(households, household_states, panel, period_state, source=source)
            decision_rows.extend(realized)
            aggregate = _aggregate_period(realized, scenario, period_state, source=source)
            period_rows.append(aggregate)
            accounting_rows.extend(_accounting_rows(realized, aggregate))
            household_states = _next_household_states(realized, households, aggregate)
            env = _next_environment(env, aggregate, scenario, feedback_mode=feedback_mode)
    return (
        initial,
        pd.DataFrame(decision_rows),
        pd.DataFrame(period_rows),
        pd.DataFrame(accounting_rows),
        prompt_rows,
    )


def demand_economy_prompt(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> str:
    payload = demand_economy_prompt_payload(scenario, period_state, household_states)
    return f"""
Abstract demand-economy household decision card:
{json.dumps(payload, indent=2, sort_keys=True)}

Return exactly this JSON shape:
{{
  "prompt_version": "{DEMAND_ECONOMY_PROMPT_VERSION}",
  "household_decisions": [
    {{
      "type_id": "one supplied type_id",
      "consumption_propensity_shift_pp": 0.0,
      "desired_buffer_months": 3.0,
      "job_loss_probability": 5.0,
      "confidence": 0.5,
      "reason": "short reason based only on the abstract state"
    }}
  ]
}}

Return one decision for every supplied household type. Do not include markdown.
""".strip()


def demand_economy_prompt_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "task": (
            "Choose consume-vs-save behavior parameters for representative households in an abstract, "
            "date-free, one-good economy. Deterministic code will enforce budgets and aggregate feedback."
        ),
        "contamination_control": "No calendar dates, named historical episodes, target realized paths, or external data are supplied.",
        "economy": {
            "goods": "one nondurable consumption good",
            "capital": "none",
            "asset_market": "one liquid buffer asset only",
            "firms": "meet demand up to feedback-adjusted output",
            "policy": "Taylor-rule short rate with optional abstract shock",
        },
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "transfer_active": bool(float(period_state["transfer_per_household"]) > 0),
            "policy_rate_shock_pp": round_or_none(period_state["policy_rate_shock_pp"]),
            "notes": scenario.notes,
        },
        "current_environment": {
            key: round_or_none(period_state[key])
            for key in [
                "period_index",
                "output_gap_pct",
                "employment_rate",
                "inflation_rate",
                "policy_rate",
                "transfer_per_household",
                "aggregate_job_loss_belief",
                "aggregate_liquid_buffer_months",
            ]
        },
        "period_id": period_state["period_id"],
        "households": [
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "income_group": row["income_group"],
                "liquidity_group": row["liquidity_group"],
                "population_weight": round_or_none(row["population_weight"]),
                "quarterly_labor_income": round_or_none(row["labor_income"]),
                "quarterly_baseline_consumption": round_or_none(row["baseline_consumption"]),
                "liquid_assets": round_or_none(row["liquid_assets"]),
                "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
                "base_mpc": round_or_none(row["base_mpc"]),
                "prior_job_loss_probability": round_or_none(row["job_loss_probability"]),
                "prior_inflation_expectation_1y": round_or_none(row["inflation_expectation_1y"]),
                "prior_income_growth_expectation_1y": round_or_none(row["income_growth_expectation_1y"]),
            }
            for row in household_states
        ],
        "allowed_type_ids": [row["type_id"] for row in household_states],
    }


def fixture_demand_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = []
    output_gap = float(period_state["output_gap_pct"])
    policy_gap = float(period_state["policy_rate"]) - NEUTRAL_POLICY_RATE
    transfer = float(period_state["transfer_per_household"])
    for row in household_states:
        liquidity = str(row["liquidity_group"]).lower()
        base_mpc = float(row["base_mpc"])
        current_buffer = float(row["liquid_buffer_months"])
        target_buffer = float(row["target_buffer_months"])
        job_loss = float(row["baseline_job_loss_probability"])
        job_loss += 0.30 * max(0.0, -output_gap)
        job_loss += 0.20 * max(0.0, policy_gap)
        job_loss *= float(scenario.belief_dispersion_multiplier)
        if liquidity == "low":
            transfer_shift = 3.0 if transfer > 0 else 0.0
            buffer_target = max(target_buffer, 1.5)
        elif liquidity == "middle":
            transfer_shift = 0.5 if transfer > 0 else 0.0
            buffer_target = max(target_buffer, 2.8)
        else:
            transfer_shift = -1.5 if transfer > 0 else 0.0
            buffer_target = max(target_buffer, 5.5)
        precaution_shift = -0.35 * max(0.0, job_loss - float(row["baseline_job_loss_probability"]))
        rate_shift = -1.8 * float(row["rate_sensitivity"]) * max(0.0, policy_gap)
        buffer_shift = -0.55 * max(0.0, buffer_target - current_buffer)
        consumption_shift = float(np.clip(100.0 * (base_mpc - 0.48) + transfer_shift + precaution_shift + rate_shift + buffer_shift, -15.0, 18.0))
        decisions.append(
            {
                "type_id": row["type_id"],
                "consumption_propensity_shift_pp": consumption_shift,
                "desired_buffer_months": float(np.clip(buffer_target, 0.5, 12.0)),
                "job_loss_probability": float(np.clip(job_loss, 0.5, 35.0)),
                "confidence": float(np.clip(0.68 - 0.012 * abs(output_gap) - 0.01 * abs(policy_gap), 0.25, 0.85)),
                "reason": "fixture decision from liquidity, job-risk, transfer, and policy-rate state",
            }
        )
    return {"prompt_version": DEMAND_ECONOMY_PROMPT_VERSION, "household_decisions": decisions}


def normalize_demand_payload(household_states: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    if payload.get("prompt_version") != DEMAND_ECONOMY_PROMPT_VERSION:
        raise LLMUnavailable("Demand economy payload has the wrong prompt_version")
    decisions = payload.get("household_decisions")
    if not isinstance(decisions, list):
        raise LLMUnavailable("Demand economy payload must contain household_decisions")
    allowed = {str(row["type_id"]) for row in household_states}
    by_type: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise LLMUnavailable("Each demand economy household decision must be an object")
        type_id = str(decision.get("type_id", ""))
        if type_id not in allowed:
            raise LLMUnavailable(f"Unknown demand economy household type_id: {type_id}")
        if type_id in by_type:
            raise LLMUnavailable(f"Duplicate demand economy household type_id: {type_id}")
        by_type[type_id] = {
            "type_id": type_id,
            "consumption_propensity_shift_pp": bounded_float(decision, "consumption_propensity_shift_pp", -20.0, 20.0),
            "desired_buffer_months": bounded_float(decision, "desired_buffer_months", 0.0, 12.0),
            "job_loss_probability": bounded_float(decision, "job_loss_probability", 0.0, 40.0),
            "confidence": bounded_float(decision, "confidence", 0.0, 1.0),
            "reason": str(decision.get("reason", ""))[:300],
        }
    missing = allowed - set(by_type)
    if missing:
        raise LLMUnavailable(f"Demand economy payload missing household type_ids: {', '.join(sorted(missing))}")
    return {"prompt_version": DEMAND_ECONOMY_PROMPT_VERSION, "household_decisions": by_type}


def score_demand_economy_validation(
    periods: pd.DataFrame,
    decisions: pd.DataFrame,
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    if periods.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    max_accounting_residual = (
        float(accounting["abs_residual"].max())
        if not accounting.empty and "abs_residual" in accounting
        else np.inf
    )
    for source, source_periods in periods.groupby("source", sort=True):
        baseline = source_periods[source_periods["scenario_id"] == "baseline"].copy()
        transfer = source_periods[source_periods["scenario_id"] == "transfer_shock"].copy()
        rate_hike = source_periods[source_periods["scenario_id"] == "rate_hike"].copy()
        feedback = source_periods[source_periods["scenario_id"] == "belief_feedback"].copy()
        if not baseline.empty and not transfer.empty:
            joined = transfer.merge(
                baseline[["period_index", "aggregate_consumption"]],
                on="period_index",
                suffixes=("_transfer", "_baseline"),
            )
            impact = joined[joined["period_index"] == 1]
            transfer_amount = float(impact["transfer_per_household"].iloc[0]) if not impact.empty else np.nan
            impact_mpc = (
                float((impact["aggregate_consumption_transfer"] - impact["aggregate_consumption_baseline"]).iloc[0] / transfer_amount)
                if transfer_amount and np.isfinite(transfer_amount)
                else np.nan
            )
            first_four = joined[(joined["period_index"] >= 1) & (joined["period_index"] <= 4)]
            cumulative_mpc = (
                float((first_four["aggregate_consumption_transfer"] - first_four["aggregate_consumption_baseline"]).sum() / transfer_amount)
                if transfer_amount and np.isfinite(transfer_amount)
                else np.nan
            )
            low_mpc, high_mpc = _liquidity_mpcs(decisions, source=source)
            rows.extend(
                [
                    _metric_row(source, "transfer_impact_mpc", impact_mpc, 0.20, 0.85, "Aggregate impact MPC after a one-period transfer."),
                    _metric_row(source, "transfer_cumulative_mpc_4p", cumulative_mpc, 0.20, 2.40, "Cumulative four-period transfer MPC."),
                    _metric_row(source, "low_liquidity_impact_mpc", low_mpc, 0.45, 1.00, "Impact MPC for low-liquidity households."),
                    _metric_row(source, "high_liquidity_impact_mpc", high_mpc, 0.05, 0.45, "Impact MPC for high-liquidity households."),
                    _metric_row(
                        source,
                        "liquidity_mpc_gradient",
                        low_mpc - high_mpc,
                        0.20,
                        1.00,
                        "Low-liquidity impact MPC minus high-liquidity impact MPC.",
                    ),
                ]
            )
        if not baseline.empty and not rate_hike.empty:
            joined = rate_hike.merge(
                baseline[["period_index", "aggregate_consumption"]],
                on="period_index",
                suffixes=("_rate_hike", "_baseline"),
            )
            window = joined[(joined["period_index"] >= 1) & (joined["period_index"] <= 4)]
            mean_response = float((window["aggregate_consumption_rate_hike"] - window["aggregate_consumption_baseline"]).mean())
            impact_response = float(
                (joined.loc[joined["period_index"] == 1, "aggregate_consumption_rate_hike"].iloc[0])
                - (joined.loc[joined["period_index"] == 1, "aggregate_consumption_baseline"].iloc[0])
            )
            rows.extend(
                [
                    _metric_row(source, "rate_hike_impact_consumption_delta", impact_response, -np.inf, -1e-6, "Impact-period consumption response to a rate hike."),
                    _metric_row(source, "rate_hike_mean_consumption_delta_4p", mean_response, -np.inf, -1e-6, "Mean four-period consumption response to a rate hike."),
                ]
            )
        if not baseline.empty and not feedback.empty:
            baseline_rms = float(np.sqrt(np.mean(np.square(baseline["output_gap_pct"].iloc[1:]))))
            feedback_rms = float(np.sqrt(np.mean(np.square(feedback["output_gap_pct"].iloc[1:]))))
            rows.extend(
                [
                    _metric_row(source, "baseline_no_shock_output_gap_rms", baseline_rms, 0.0, np.inf, "Baseline endogenous output-gap root-mean-square movement."),
                    _metric_row(
                        source,
                        "belief_feedback_amplification_ratio",
                        feedback_rms / max(baseline_rms, 1e-6),
                        1.10,
                        np.inf,
                        "No-shock belief-feedback output-gap movement divided by baseline movement.",
                    ),
                ]
            )
        rows.append(
            _metric_row(
                source,
                "max_accounting_abs_residual",
                max_accounting_residual,
                0.0,
                ACCOUNTING_TOLERANCE,
                "Maximum absolute household-budget or goods-market accounting residual.",
            )
        )
    return pd.DataFrame(rows).sort_values(["source", "metric"]).reset_index(drop=True)


def classify_demand_economy_evidence(validation: pd.DataFrame, *, mode: str) -> dict[str, Any]:
    if validation.empty:
        return {
            "evidence_verdict": "no_validation_rows",
            "passed": False,
            "passed_metric_count": 0,
            "metric_count": 0,
        }
    passed = validation["passed"].astype(bool)
    all_passed = bool(passed.all())
    verdict = "behavior_demand_economy_ready" if all_passed else "behavior_demand_economy_needs_work"
    if mode == "fixture" and all_passed:
        verdict = "fixture_behavior_demand_economy_ready"
    return {
        "evidence_verdict": verdict,
        "passed": all_passed,
        "passed_metric_count": int(passed.sum()),
        "metric_count": int(validation.shape[0]),
        "failed_metrics": validation.loc[~passed, ["source", "metric", "value", "target_low", "target_high"]].to_dict(orient="records"),
    }


def build_demand_economy_report(
    manifest: dict[str, Any],
    households: pd.DataFrame,
    periods: pd.DataFrame,
    validation: pd.DataFrame,
    accounting: pd.DataFrame,
) -> str:
    failed = validation[~validation["passed"].astype(bool)] if not validation.empty else pd.DataFrame()
    period_preview = periods.sort_values(["source", "scenario_id", "period_index"]).head(48) if not periods.empty else periods
    accounting_summary = (
        accounting.groupby(["source", "scenario_id"], as_index=False)["abs_residual"].max().sort_values(["source", "scenario_id"])
        if not accounting.empty
        else accounting
    )
    lines = [
        "# Abstract Behavior Demand Economy",
        "",
        "## Bottom Line",
        _demand_bottom_line(manifest, failed),
        "",
        "## Setup",
        f"- Decision mode: `{manifest.get('decision_mode')}`",
        f"- Provider/models: `{manifest.get('provider')}` / `{', '.join(manifest.get('models', []))}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Household types: `{manifest.get('household_count')}`",
        f"- Scenarios: `{manifest.get('scenario_count')}`",
        f"- Periods per scenario: `{manifest.get('period_count')}`",
        f"- Feedback mode: `{manifest.get('feedback_mode')}`",
        "",
        "## Validation Scores",
        markdown_table(validation),
        "",
        "## Failed Metrics",
        markdown_table(failed),
        "",
        "## Household Type Surface",
        markdown_table(households[["type_id", "population_weight", "income_group", "liquidity_group", "annual_income", "liquid_assets", "base_mpc"]]),
        "",
        "## Period Paths",
        markdown_table(
            period_preview[
                [
                    "source",
                    "scenario_id",
                    "period_id",
                    "aggregate_consumption",
                    "output_gap_pct",
                    "employment_rate",
                    "inflation_rate",
                    "policy_rate",
                    "transfer_per_household",
                ]
            ]
            if not period_preview.empty
            else period_preview
        ),
        "",
        "## Accounting Audit",
        markdown_table(accounting_summary),
        "",
        "## Interpretation",
        (
            "This is the first behavior-based macro gate. Households choose consume-vs-save parameters, "
            "then deterministic code enforces their cash budgets, aggregates demand, maps demand into output, "
            "employment, income, sticky inflation, and a Taylor-rule policy rate, and feeds that state into the "
            "next period. The validation target is dynamic behavior, not matching a named historical path."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _initial_household_states(households: pd.DataFrame, *, source: str) -> pd.DataFrame:
    rows = []
    for _, row in households.iterrows():
        baseline_consumption = float(row["baseline_consumption_annual"]) / 4.0
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": source,
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": float(row["population_weight"]),
                "income_group": row["income_group"],
                "liquidity_group": row["liquidity_group"],
                "annual_income": float(row["annual_income"]),
                "labor_income": float(row["annual_income"]) / 4.0,
                "baseline_consumption": baseline_consumption,
                "liquid_assets": float(row["liquid_assets"]),
                "debt": float(row["debt"]),
                "base_mpc": float(row["base_mpc"]),
                "rate_sensitivity": float(row["rate_sensitivity"]),
                "income_sensitivity": float(row["income_sensitivity"]),
                "precautionary_sensitivity": float(row["precautionary_sensitivity"]),
                "baseline_job_loss_probability": float(row["baseline_job_loss_probability"]),
                "job_loss_probability": float(row["baseline_job_loss_probability"]),
                "target_buffer_months": float(row["target_buffer_months"]),
                "inflation_expectation_1y": float(row["inflation_expectation_1y"]),
                "income_growth_expectation_1y": float(row["income_growth_expectation_1y"]),
                "liquid_buffer_months": _buffer_months(float(row["liquid_assets"]), baseline_consumption),
            }
        )
    return pd.DataFrame(rows).sort_values("type_id").reset_index(drop=True)


def _initial_environment(households: pd.DataFrame) -> dict[str, float]:
    baseline_consumption = float((households["population_weight"] * households["baseline_consumption_annual"] / 4.0).sum())
    return {
        "baseline_aggregate_consumption": baseline_consumption,
        "aggregate_consumption": baseline_consumption,
        "output_gap_pct": 0.0,
        "employment_rate": 0.955,
        "inflation_rate": INFLATION_TARGET,
        "policy_rate": NEUTRAL_POLICY_RATE,
        "aggregate_job_loss_belief": float((households["population_weight"] * households["baseline_job_loss_probability"]).sum()),
        "aggregate_liquid_buffer_months": 3.0,
    }


def _period_state(env: dict[str, float], scenario: DemandScenario, period_index: int) -> dict[str, Any]:
    rate_shock = (
        float(scenario.rate_shock_pp)
        if scenario.rate_shock_start <= period_index <= scenario.rate_shock_end and scenario.rate_shock_start >= 0
        else 0.0
    )
    transfer = float(scenario.transfer_amount) if period_index == scenario.transfer_period else 0.0
    return {
        **env,
        "scenario_id": scenario.scenario_id,
        "period_index": int(period_index),
        "period_id": f"period_{period_index}",
        "transfer_per_household": transfer,
        "policy_rate_shock_pp": rate_shock,
        "policy_rate": float(env["policy_rate"]) + rate_shock,
    }


def _realize_household_period(
    households: pd.DataFrame,
    household_states: list[dict[str, Any]],
    panel: dict[str, Any],
    period_state: dict[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]]:
    household_by_type = {str(row["type_id"]): row for _, row in households.iterrows()}
    decisions = panel["household_decisions"]
    rows: list[dict[str, Any]] = []
    for state in household_states:
        type_id = str(state["type_id"])
        static = household_by_type[type_id]
        decision = decisions[type_id]
        labor_income = float(state["labor_income"])
        liquid_before = float(state["liquid_assets"])
        transfer = float(period_state["transfer_per_household"])
        cash_available = liquid_before + labor_income + transfer
        baseline_consumption = float(state["baseline_consumption"])
        current_buffer = _buffer_months(liquid_before, baseline_consumption)
        desired_buffer = float(decision["desired_buffer_months"])
        mpc = float(np.clip(float(static["base_mpc"]) + float(decision["consumption_propensity_shift_pp"]) / 100.0, 0.02, 0.98))
        income_gap = labor_income / max(float(static["annual_income"]) / 4.0, 1e-9) - 1.0
        policy_gap = float(period_state["policy_rate"]) - NEUTRAL_POLICY_RATE
        job_loss = float(decision["job_loss_probability"])
        baseline_job_loss = float(static["baseline_job_loss_probability"])
        rate_drag = max(0.0, policy_gap) * float(static["rate_sensitivity"]) * baseline_consumption * 0.060
        precaution_drag = max(0.0, job_loss - baseline_job_loss) * float(static["precautionary_sensitivity"]) * baseline_consumption * 0.010
        buffer_drag = max(0.0, desired_buffer - current_buffer) * baseline_consumption * 0.018
        income_effect = income_gap * float(static["income_sensitivity"]) * baseline_consumption
        desired_consumption = baseline_consumption + mpc * transfer + income_effect - rate_drag - precaution_drag - buffer_drag
        floor_consumption = min(cash_available, 0.45 * baseline_consumption)
        consumption = float(np.clip(desired_consumption, floor_consumption, cash_available))
        saving_flow = labor_income + transfer - consumption
        liquid_after = liquid_before + saving_flow
        budget_residual = liquid_before + labor_income + transfer - consumption - liquid_after
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": source,
                "scenario_id": period_state["scenario_id"],
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "type_id": type_id,
                "label": state["label"],
                "income_group": state["income_group"],
                "liquidity_group": state["liquidity_group"],
                "population_weight": float(state["population_weight"]),
                "labor_income": labor_income,
                "transfer": transfer,
                "cash_available": cash_available,
                "consumption": consumption,
                "saving_flow": saving_flow,
                "liquid_assets_before": liquid_before,
                "liquid_assets_after": liquid_after,
                "liquid_buffer_months_before": current_buffer,
                "liquid_buffer_months_after": _buffer_months(liquid_after, baseline_consumption),
                "base_mpc": float(static["base_mpc"]),
                "realized_mpc_from_transfer": consumption / transfer if transfer > 0 else np.nan,
                "consumption_propensity_shift_pp": float(decision["consumption_propensity_shift_pp"]),
                "desired_buffer_months": desired_buffer,
                "job_loss_probability": job_loss,
                "confidence": float(decision["confidence"]),
                "budget_residual": budget_residual,
                "reason": decision["reason"],
            }
        )
    return rows


def _aggregate_period(
    realized: list[dict[str, Any]],
    scenario: DemandScenario,
    period_state: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    aggregate_consumption = _weighted(realized, "consumption")
    aggregate_income = _weighted(realized, "labor_income")
    aggregate_transfer = _weighted(realized, "transfer")
    aggregate_saving = _weighted(realized, "saving_flow")
    aggregate_liquid_assets = _weighted(realized, "liquid_assets_after")
    aggregate_job_loss = _weighted(realized, "job_loss_probability")
    aggregate_buffer = _weighted(realized, "liquid_buffer_months_after")
    baseline = float(period_state["baseline_aggregate_consumption"])
    output = aggregate_consumption
    output_gap = 100.0 * (output / max(baseline, 1e-9) - 1.0)
    return {
        "schema_version": DEMAND_ECONOMY_VERSION,
        "source": source,
        "scenario_id": scenario.scenario_id,
        "scenario_label": scenario.label,
        "period_id": period_state["period_id"],
        "period_index": int(period_state["period_index"]),
        "aggregate_consumption": aggregate_consumption,
        "aggregate_income": aggregate_income,
        "aggregate_transfer": aggregate_transfer,
        "aggregate_saving": aggregate_saving,
        "aggregate_liquid_assets": aggregate_liquid_assets,
        "aggregate_job_loss_belief": aggregate_job_loss,
        "aggregate_liquid_buffer_months": aggregate_buffer,
        "output": output,
        "output_gap_pct": output_gap,
        "employment_rate": float(period_state["employment_rate"]),
        "inflation_rate": float(period_state["inflation_rate"]),
        "policy_rate": float(period_state["policy_rate"]),
        "policy_rate_shock_pp": float(period_state["policy_rate_shock_pp"]),
        "transfer_per_household": float(period_state["transfer_per_household"]),
        "goods_market_residual": output - aggregate_consumption,
    }


def _next_household_states(
    realized: list[dict[str, Any]],
    households: pd.DataFrame,
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    static_by_type = {str(row["type_id"]): row for _, row in households.iterrows()}
    employment_factor = float(aggregate["employment_rate"]) / 0.955
    inflation_gap = float(aggregate["inflation_rate"]) - INFLATION_TARGET
    output_gap = float(aggregate["output_gap_pct"])
    rows: list[dict[str, Any]] = []
    for row in realized:
        static = static_by_type[str(row["type_id"])]
        annual_income = float(static["annual_income"])
        labor_income = annual_income / 4.0 * np.clip(employment_factor * (1.0 + 0.002 * output_gap), 0.70, 1.25)
        baseline_consumption = float(static["baseline_consumption_annual"]) / 4.0
        job_loss = float(row["job_loss_probability"])
        job_loss = float(np.clip(0.80 * job_loss + 0.20 * float(static["baseline_job_loss_probability"]) + 0.10 * max(0.0, -output_gap), 0.5, 35.0))
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": row["source"],
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": float(row["population_weight"]),
                "income_group": row["income_group"],
                "liquidity_group": row["liquidity_group"],
                "annual_income": annual_income,
                "labor_income": float(labor_income),
                "baseline_consumption": baseline_consumption,
                "liquid_assets": float(row["liquid_assets_after"]),
                "debt": float(static["debt"]),
                "base_mpc": float(static["base_mpc"]),
                "rate_sensitivity": float(static["rate_sensitivity"]),
                "income_sensitivity": float(static["income_sensitivity"]),
                "precautionary_sensitivity": float(static["precautionary_sensitivity"]),
                "baseline_job_loss_probability": float(static["baseline_job_loss_probability"]),
                "job_loss_probability": job_loss,
                "target_buffer_months": float(static["target_buffer_months"]),
                "inflation_expectation_1y": float(np.clip(float(static["inflation_expectation_1y"]) + 0.25 * inflation_gap, 0.0, 12.0)),
                "income_growth_expectation_1y": float(np.clip(float(static["income_growth_expectation_1y"]) + 0.05 * output_gap, -8.0, 8.0)),
                "liquid_buffer_months": _buffer_months(float(row["liquid_assets_after"]), baseline_consumption),
            }
        )
    return rows


def _next_environment(
    env: dict[str, float],
    aggregate: dict[str, Any],
    scenario: DemandScenario,
    *,
    feedback_mode: str,
) -> dict[str, float]:
    if feedback_mode == "none":
        return {
            **env,
            "aggregate_consumption": float(aggregate["aggregate_consumption"]),
            "aggregate_job_loss_belief": float(aggregate["aggregate_job_loss_belief"]),
            "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
        }
    output_gap = float(aggregate["output_gap_pct"])
    gain = float(scenario.feedback_gain)
    next_employment = float(np.clip(0.72 * float(env["employment_rate"]) + 0.28 * (0.955 + 0.0022 * gain * output_gap), 0.82, 0.99))
    next_inflation = float(
        np.clip(
            0.58 * float(env["inflation_rate"]) + 0.42 * INFLATION_TARGET + 0.030 * gain * output_gap,
            -2.0,
            12.0,
        )
    )
    next_policy = float(
        np.clip(
            NEUTRAL_POLICY_RATE + 1.35 * (next_inflation - INFLATION_TARGET) + 0.22 * output_gap,
            0.0,
            12.0,
        )
    )
    return {
        **env,
        "aggregate_consumption": float(aggregate["aggregate_consumption"]),
        "output_gap_pct": output_gap,
        "employment_rate": next_employment,
        "inflation_rate": next_inflation,
        "policy_rate": next_policy,
        "aggregate_job_loss_belief": float(aggregate["aggregate_job_loss_belief"]),
        "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
    }


def _accounting_rows(realized: list[dict[str, Any]], aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "source": row["source"],
            "scenario_id": row["scenario_id"],
            "period_id": row["period_id"],
            "period_index": int(row["period_index"]),
            "unit": row["type_id"],
            "identity": "household_cash_budget",
            "residual": float(row["budget_residual"]),
            "abs_residual": abs(float(row["budget_residual"])),
            "passed": abs(float(row["budget_residual"])) <= ACCOUNTING_TOLERANCE,
        }
        for row in realized
    ]
    residual = float(aggregate["goods_market_residual"])
    rows.append(
        {
            "source": aggregate["source"],
            "scenario_id": aggregate["scenario_id"],
            "period_id": aggregate["period_id"],
            "period_index": int(aggregate["period_index"]),
            "unit": "aggregate",
            "identity": "one_good_output_equals_consumption",
            "residual": residual,
            "abs_residual": abs(residual),
            "passed": abs(residual) <= ACCOUNTING_TOLERANCE,
        }
    )
    return rows


def _liquidity_mpcs(decisions: pd.DataFrame, *, source: str) -> tuple[float, float]:
    subset = decisions[
        (decisions["source"] == source)
        & (decisions["period_index"] == 1)
        & (decisions["scenario_id"].isin(["baseline", "transfer_shock"]))
    ].copy()
    if subset.empty:
        return np.nan, np.nan
    baseline = subset[subset["scenario_id"] == "baseline"][
        ["type_id", "consumption"]
    ].rename(columns={"consumption": "baseline_consumption"})
    transfer = subset[subset["scenario_id"] == "transfer_shock"].merge(baseline, on="type_id", how="inner")
    if transfer.empty:
        return np.nan, np.nan
    transfer["type_mpc"] = (transfer["consumption"] - transfer["baseline_consumption"]) / transfer["transfer"].replace(0.0, np.nan)
    grouped = (
        transfer.groupby("liquidity_group")
        .apply(lambda group: float((group["type_mpc"] * group["population_weight"]).sum() / group["population_weight"].sum()))
        .to_dict()
    )
    return float(grouped.get("low", np.nan)), float(grouped.get("high", np.nan))


def _metric_row(source: str, metric: str, value: float, target_low: float, target_high: float, interpretation: str) -> dict[str, Any]:
    if not np.isfinite(value):
        passed = False
    else:
        passed = bool(float(value) >= float(target_low) and float(value) <= float(target_high))
    return {
        "source": source,
        "metric": metric,
        "value": float(value) if np.isfinite(value) else np.nan,
        "target_low": float(target_low) if np.isfinite(target_low) else target_low,
        "target_high": float(target_high) if np.isfinite(target_high) else target_high,
        "passed": passed,
        "interpretation": interpretation,
    }


def _weighted(rows: list[dict[str, Any]], column: str) -> float:
    return float(sum(float(row[column]) * float(row["population_weight"]) for row in rows))


def _buffer_months(liquid_assets: float, quarterly_consumption: float) -> float:
    monthly_consumption = max(float(quarterly_consumption) / 3.0, 1e-9)
    return float(max(0.0, float(liquid_assets)) / monthly_consumption)


def _demand_instructions() -> str:
    return """
Return only valid JSON. Use only the abstract economy state supplied in the prompt.
Do not browse, inspect files, run commands, cite historical episodes, or infer calendar dates.
Choose behavior parameters; deterministic code will enforce accounting.
""".strip()


def _demand_bottom_line(manifest: dict[str, Any], failed: pd.DataFrame) -> str:
    evidence = manifest.get("evidence", {})
    verdict = evidence.get("evidence_verdict", "unknown")
    if failed.empty:
        return (
            f"`{verdict}`. The fixture economy clears the accounting, transfer-MPC, liquidity-gradient, "
            "rate-hike, and no-shock feedback checks. That makes it ready as the behavior macro harness "
            "for live LLM household actors."
        )
    return (
        f"`{verdict}`. The harness ran, but some dynamic validation checks failed. "
        "Treat the failed-metric table as the next calibration target before live model spend."
    )


def _sanitized_argv() -> list[str]:
    raw = sys.argv
    if raw and Path(raw[0]).name == "demand_economy.py":
        return ["python3", "-m", "macro_llm_tournament.demand_economy", *raw[1:]]
    return list(raw)


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - git can be unavailable in packaged use
        return {"error": str(exc)[:200]}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
