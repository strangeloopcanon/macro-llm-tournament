"""Household-first rolling microeconomy runner."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .ecology_engine import run_monthly_ecology
from .ecology_inputs import load_origin_information
from .ecology_households import (
    HOUSEHOLD_PROMPT_VERSION,
    HouseholdElicitor,
    LiveCallBudget,
    canonical_sha256,
    household_card,
    normalize_household_payload,
)
from .ecology_models import CreditIntermediaryState, EmployerState, HouseholdState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOUSEHOLDS = PROJECT_ROOT / "work/persona_beliefs/persistent_household_scale_v1/initial_households_200.csv"
DEFAULT_HISTORY = PROJECT_ROOT / "work/persona_beliefs/persistent_household_scale_v1/selected_observed_history.csv"
DEFAULT_BUNDLE = PROJECT_ROOT / "work/dynamic_macro/frozen_2026_01_2026_05_common_month_v1"
DEFAULT_CACHE = PROJECT_ROOT / "work/ecology_cache"
SCHEMA_VERSION = "household_first_rolling_microeconomy_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="Monthly origin, YYYY-MM-DD")
    parser.add_argument("--mode", choices=("fixture", "replay", "live"), required=True)
    parser.add_argument("--provider", default="codex_cli", choices=("codex_cli",))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--household-count", type=int, default=200)
    parser.add_argument("--households", type=Path, default=DEFAULT_HOUSEHOLDS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--expected-replay-sha256")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src/macro_llm_tournament").glob("ecology*.py"))
    paths.extend(
        PROJECT_ROOT / "src/macro_llm_tournament" / name
        for name in ("frozen_vintage_bundle.py", "persistent_households.py")
    )
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    if args.household_count <= 0:
        raise ValueError("--household-count must be positive")
    households = pd.read_csv(args.households)
    if args.household_count > len(households):
        raise ValueError("--household-count exceeds available cohort")
    households = _coverage_sample(households, args.household_count)
    history = pd.read_csv(args.history)
    origin, bundle_sha = load_origin_information(args.bundle, args.origin)
    return households, history, origin, bundle_sha


def _coverage_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Choose a stable canary that covers observable household-state categories."""

    ordered = frame.copy()
    ordered["_sample_hash"] = ordered["type_id"].map(
        lambda value: hashlib.sha256(f"ecology-canary-v1|{value}".encode()).hexdigest()
    )
    ordered = ordered.sort_values("_sample_hash").reset_index(drop=True)
    if count >= len(ordered):
        return ordered.drop(columns="_sample_hash").sort_values("type_id").reset_index(drop=True)
    columns = [
        column
        for column in ("income_group", "liquidity_group", "age_bucket", "employment_status")
        if column in ordered
    ]
    tokens = {
        index: {(column, str(row[column])) for column in columns}
        for index, row in ordered.iterrows()
    }
    uncovered = set().union(*tokens.values()) if tokens else set()
    chosen: list[int] = []
    remaining = set(ordered.index)
    while remaining and len(chosen) < count:
        index = min(
            remaining,
            key=lambda item: (-len(tokens[item] & uncovered), ordered.loc[item, "_sample_hash"]),
        )
        chosen.append(index)
        uncovered -= tokens[index]
        remaining.remove(index)
    return (
        ordered.loc[chosen]
        .drop(columns="_sample_hash")
        .sort_values("type_id")
        .reset_index(drop=True)
    )


def _state_from_row(row: pd.Series) -> HouseholdState:
    monthly_consumption = float(row["baseline_consumption_annual"]) / 12.0
    income = float(row["annual_income"])
    status = str(row.get("employment_status", "unknown")).lower()
    employed = status not in {"unemployed", "not_employed"}
    hours = 160.0 if employed else 0.0
    hourly_wage = income / 2080.0 if employed else max(7.25, income / 2080.0)
    debt = max(0.0, float(row.get("debt", 0.0)))
    return HouseholdState(
        household_id=str(row["type_id"]),
        employer_id="aggregate_employer",
        deposit_balance_usd=max(0.0, float(row.get("liquid_assets", 0.0))),
        revolving_debt_usd=debt,
        revolving_credit_limit_usd=max(debt, debt * 1.5, monthly_consumption * 2.0),
        hourly_wage_usd=hourly_wage,
        baseline_monthly_hours=hours,
        baseline_monthly_consumption_usd=monthly_consumption,
        layoff_threshold_pct=50.0,
        liquid_buffer_floor_months=float(row.get("subsistence_floor_share", 0.5)),
        subsistence_consumption_share=float(row.get("subsistence_floor_share", 0.45)),
        population_weight=float(row.get("population_weight", 1.0 / 200.0)),
    )


def _initial_employer(states: list[HouseholdState]) -> EmployerState:
    employed_hours = sum(row.baseline_monthly_hours for row in states)
    baseline_demand = sum(row.baseline_monthly_consumption_usd for row in states)
    productivity = baseline_demand / max(employed_hours, 1.0)
    employed = sum(row.baseline_monthly_hours > 0 for row in states)
    wages = [row.hourly_wage_usd for row in states if row.baseline_monthly_hours > 0]
    return EmployerState(
        employer_id="aggregate_employer",
        productivity_per_hour=productivity,
        monthly_capacity_units=baseline_demand * 1.15,
        inventory_units=baseline_demand * 0.08,
        price_per_unit_usd=1.0,
        target_headcount=max(1, employed),
        wage_offer_usd=sum(wages) / max(len(wages), 1),
        target_inventory_units=baseline_demand * 0.08,
        variable_nonlabor_cost_per_unit_usd=0.15,
        inventory_carry_cost_per_unit_usd=0.005,
    )


def _initial_credit(states: list[HouseholdState], origin: dict[str, Any]) -> CreditIntermediaryState:
    context = origin["origin_visible_macro_context"]
    policy_rate = float(context.get("FEDFUNDS", {}).get("value", 4.0))
    return CreditIntermediaryState(
        intermediary_id="aggregate_credit_intermediary",
        annual_interest_rate_pct=max(0.0, policy_rate + 14.0),
        minimum_payment_rate_pct=3.0,
        minimum_payment_floor_usd=25.0,
        loss_given_default_pct=55.0,
        new_lending_budget_usd=sum(
            max(0.0, row.revolving_credit_limit_usd - row.revolving_debt_usd) for row in states
        ) * 0.2,
    )


def _load_recursive_state(
    path: Path | None,
    states: list[HouseholdState],
    employer: EmployerState,
    credit: CreditIntermediaryState,
    origin_month: str,
) -> tuple[list[HouseholdState], EmployerState, CreditIntermediaryState, dict[str, Any], dict[str, Any], str | None]:
    if path is None:
        return states, employer, credit, {}, {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "household_ecology_recursive_state_v2":
        raise ValueError("recursive state schema mismatch")
    if payload.get("next_origin_month") != origin_month:
        raise ValueError("recursive state month continuity mismatch")
    parent_scenario = payload.get("scenario")
    if parent_scenario not in {"downside", "median", "upside"}:
        raise ValueError("recursive state scenario mismatch")
    rows = payload.get("households")
    if not isinstance(rows, list):
        raise ValueError("recursive state households must be a list")
    by_id = {row["household_id"]: row for row in rows}
    if set(by_id) != {row.household_id for row in states}:
        raise ValueError("recursive state household membership mismatch")
    restored = [
        HouseholdState(**{field.name: by_id[state.household_id][field.name] for field in dataclasses.fields(HouseholdState)})
        for state in states
    ]
    restored_employer = EmployerState(
        **{field.name: payload["employer"][field.name] for field in dataclasses.fields(EmployerState)}
    )
    restored_credit = CreditIntermediaryState(
        **{field.name: payload["credit"][field.name] for field in dataclasses.fields(CreditIntermediaryState)}
    )
    macro_state = payload.get("macro_state", {})
    if not isinstance(macro_state, dict):
        raise ValueError("recursive macro state must be an object")
    return restored, restored_employer, restored_credit, by_id, macro_state, str(parent_scenario)


def _next_recursive_state(
    *,
    scenario: str,
    result: Any,
    states: list[HouseholdState],
    responses: dict[str, Any],
    raw_payloads: dict[str, dict[str, Any]],
    employer: EmployerState,
    credit: CreditIntermediaryState,
    origin: dict[str, Any],
) -> dict[str, Any]:
    prior_by_id = {row.household_id: row for row in states}
    household_rows: list[dict[str, Any]] = []
    for outcome in result.households:
        prior = prior_by_id[outcome.household_id]
        response = responses[outcome.household_id]
        wage_ratio = result.employer.next_wage_offer_usd / max(result.employer.current_wage_offer_usd, 1.0)
        next_household = dataclasses.asdict(
            HouseholdState(
                household_id=prior.household_id,
                employer_id=prior.employer_id,
                deposit_balance_usd=outcome.deposit_balance_end_usd,
                revolving_debt_usd=outcome.revolving_debt_end_usd,
                revolving_credit_limit_usd=prior.revolving_credit_limit_usd,
                hourly_wage_usd=max(0.0, prior.hourly_wage_usd * wage_ratio),
                baseline_monthly_hours=0.7 * prior.baseline_monthly_hours + 0.3 * outcome.actual_hours_worked,
                baseline_monthly_consumption_usd=max(
                    1.0,
                    0.7 * prior.baseline_monthly_consumption_usd + 0.3 * outcome.consumption_usd,
                ),
                layoff_threshold_pct=prior.layoff_threshold_pct,
                liquid_buffer_floor_months=prior.liquid_buffer_floor_months,
                subsistence_consumption_share=prior.subsistence_consumption_share,
                population_weight=prior.population_weight,
            )
        )
        next_household.update(
            {
                "inflation_expectation_1y": response.expected_inflation_pct.p50,
                "income_growth_expectation_1y": response.expected_income_growth_pct.p50,
                "job_loss_probability": response.job_loss_probability_pct.p50,
                "previous_intentions": raw_payloads[outcome.household_id],
                "previous_outcomes": {
                    "consumption_usd": outcome.consumption_usd,
                    "hours_worked": outcome.actual_hours_worked,
                    "job_search_hours": outcome.actual_job_search_hours,
                    "wage_income_usd": outcome.wage_income_usd,
                    "debt_payment_usd": outcome.debt_payment_usd,
                    "borrowing_usd": outcome.borrowing_usd,
                    "defaulted": outcome.defaulted,
                },
            }
        )
        household_rows.append(next_household)
    pressure = max(0.75, min(1.25, result.employer.demand_pressure))
    next_employer = dataclasses.asdict(
        EmployerState(
            employer_id=employer.employer_id,
            productivity_per_hour=employer.productivity_per_hour,
            monthly_capacity_units=max(
                result.employer.output_units,
                employer.monthly_capacity_units * (1.0 + 0.15 * (pressure - 1.0)),
            ),
            inventory_units=max(0.0, result.employer.inventory_end_units),
            price_per_unit_usd=result.employer.next_price_per_unit_usd,
            target_headcount=max(1, round(result.employer.employment_count * pressure)),
            wage_offer_usd=result.employer.next_wage_offer_usd,
            target_inventory_units=employer.target_inventory_units,
            fixed_cost_usd=employer.fixed_cost_usd,
            variable_nonlabor_cost_per_unit_usd=employer.variable_nonlabor_cost_per_unit_usd,
            vacancy_cost_per_opening_usd=employer.vacancy_cost_per_opening_usd,
            inventory_carry_cost_per_unit_usd=employer.inventory_carry_cost_per_unit_usd,
        )
    )
    headroom = sum(
        max(0.0, row["revolving_credit_limit_usd"] - row["revolving_debt_usd"])
        for row in household_rows
    )
    next_credit = dataclasses.asdict(
        CreditIntermediaryState(
            intermediary_id=credit.intermediary_id,
            annual_interest_rate_pct=credit.annual_interest_rate_pct,
            minimum_payment_rate_pct=credit.minimum_payment_rate_pct,
            minimum_payment_floor_usd=credit.minimum_payment_floor_usd,
            loss_given_default_pct=credit.loss_given_default_pct,
            new_lending_budget_usd=headroom * 0.2,
        )
    )
    return {
        "schema_version": "household_ecology_recursive_state_v2",
        "source_origin_month": origin["origin_month"],
        "next_origin_month": (
            pd.Timestamp(origin["origin_month"]) + pd.offsets.MonthBegin(1)
        ).date().isoformat(),
        "scenario": scenario,
        "households": household_rows,
        "employer": next_employer,
        "credit": next_credit,
        "macro_state": _weighted_macro(result, states, origin),
    }


def _weighted_macro(result: Any, states: list[HouseholdState], origin: dict[str, Any]) -> dict[str, float]:
    weights = {row.household_id: row.population_weight for row in states}
    total_weight = sum(weights.values()) or 1.0
    def weighted(field: str) -> float:
        return sum(getattr(row, field) * weights[row.household_id] for row in result.households) / total_weight
    baseline_consumption = sum(
        row.baseline_consumption_usd * weights[row.household_id] for row in result.households
    ) / total_weight
    debt_start = sum(row.revolving_debt_start_usd * weights[row.household_id] for row in result.households) / total_weight
    debt_end = sum(row.revolving_debt_end_usd * weights[row.household_id] for row in result.households) / total_weight
    employed_weight = sum(
        weights[row.household_id] for row in result.households if row.actual_hours_worked > 0
    ) / total_weight
    saving = weighted("wage_income_usd") + weighted("borrowing_usd") - weighted("consumption_usd") - weighted("debt_payment_usd")
    saving_rate = 100.0 * saving / max(weighted("wage_income_usd"), 1.0)
    price_growth = 100.0 * (result.employer.next_price_per_unit_usd / result.employer.current_price_per_unit_usd - 1.0)
    return {
        "consumption_growth_pct": 100.0 * (weighted("consumption_usd") / max(baseline_consumption, 1.0) - 1.0),
        "saving_rate_pct": saving_rate,
        "revolving_credit_growth_pct": 100.0 * (debt_end / max(debt_start, 1.0) - 1.0),
        "employment_rate_pct": 100.0 * employed_weight,
        "unemployment_rate_pct": 100.0 * (1.0 - employed_weight),
        "price_growth_pct": price_growth,
        "output_units": result.employer.output_units,
        "units_sold": result.employer.units_sold,
        "inventory_end_units": result.employer.inventory_end_units,
        "policy_rate_pct": float(origin["origin_visible_macro_context"].get("FEDFUNDS", {}).get("value", 0.0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    households, history, origin, bundle_sha = _load_inputs(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    states = [_state_from_row(row) for _, row in households.iterrows()]
    histories = {
        household_id: group.to_dict("records")
        for household_id, group in history.groupby("household_id", sort=True)
    }
    worker_count = int(getattr(args, "workers", 1))
    if worker_count <= 0:
        raise ValueError("--workers must be positive")
    call_budget = LiveCallBudget(args.max_live_calls, output / "live_attempts")
    employer = _initial_employer(states)
    credit = _initial_credit(states, origin)
    states, employer, credit, recursive_rows, prior_macro_state, parent_scenario = _load_recursive_state(
        getattr(args, "state_json", None), states, employer, credit, args.origin
    )
    if prior_macro_state:
        origin = dict(origin)
        origin["prior_simulated_state"] = prior_macro_state
    cards: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    responses: dict[str, Any] = {}
    raw_payloads: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for state in states:
        own_history = [
            row for row in histories.get(state.household_id, [])
            if str(row.get("public_availability_date", "")) <= origin["as_of_date"]
        ]
        card_state = households.loc[households["type_id"].eq(state.household_id)].iloc[0].to_dict()
        card_state.update(recursive_rows.get(state.household_id, {}))
        card = household_card(
            card_state,
            origin=origin,
            own_history=own_history,
        )
        cards.append(card)

    def elicit_one(card: dict[str, Any]) -> dict[str, Any]:
        elicitor = HouseholdElicitor(
            args.provider,
            args.model,
            args.cache_dir,
            args.mode,
            args.max_live_calls,
            PROJECT_ROOT,
            call_budget,
        )
        return elicitor.elicit(card)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        records = list(executor.map(elicit_one, cards))

    for state, card, record in zip(states, cards, records, strict=True):
        payload = record["payload"]
        responses[state.household_id] = normalize_household_payload(payload, state.household_id)
        raw_payloads[state.household_id] = payload
        events.append({
            "household_id": state.household_id,
            "origin_month": args.origin,
            "prompt_version": HOUSEHOLD_PROMPT_VERSION,
            "card_sha256": canonical_sha256(card),
            "payload_sha256": canonical_sha256(payload),
            "cache_hit": bool(record.get("cache_hit", False)),
        })

    scenarios = {
        scenario: run_monthly_ecology(states, responses, employer, credit, scenario=scenario)
        for scenario in ("downside", "median", "upside")
    }
    replay_equivalence_sha256 = canonical_sha256(
        {
            "cards": cards,
            "payloads": raw_payloads,
            "scenarios": {name: _jsonable(result) for name, result in scenarios.items()},
        }
    )
    expected_replay_sha256 = getattr(args, "expected_replay_sha256", None)
    if expected_replay_sha256 is not None and expected_replay_sha256 != replay_equivalence_sha256:
        raise ValueError("replay equivalence hash mismatch")
    for scenario, result in scenarios.items():
        _write_json(output / f"{scenario}_economy.json", result)
        _write_json(
            output / f"{scenario}_next_state.json",
            _next_recursive_state(
                scenario=scenario,
                result=result,
                states=states,
                responses=responses,
                raw_payloads=raw_payloads,
                employer=employer,
                credit=credit,
                origin=origin,
            ),
        )
    (output / "next_state.json").write_bytes((output / "median_next_state.json").read_bytes())
    _write_json(output / "normalized_origin_information.json", origin)
    _write_json(output / "household_cards.json", cards)
    _write_json(output / "household_responses.json", records)
    _write_json(output / "event_log.json", events)
    macro_rows = [
        _weighted_macro(result, states, origin) | {"scenario": name}
        for name, result in scenarios.items()
    ]
    pd.DataFrame(macro_rows).to_csv(output / "macro_forecast_paths.csv", index=False)
    pd.DataFrame([
        _jsonable(row) | {"scenario": scenario}
        for scenario, result in scenarios.items()
        for row in result.households
    ]).to_csv(output / "household_decisions.csv", index=False)
    pd.DataFrame([
        _jsonable(row) | {"scenario": scenario}
        for scenario, result in scenarios.items()
        for row in result.accounting_residuals
    ]).to_csv(output / "accounting_audit.csv", index=False)
    target_month = (pd.Timestamp(args.origin) + pd.offsets.MonthBegin(1)).date().isoformat()
    forecast_frozen = pd.Timestamp(origin["as_of_date"]) < pd.Timestamp(target_month)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "origin_month": args.origin,
        "target_month": target_month,
        "as_of_date": origin["as_of_date"],
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "household_count": len(states),
        "accepted_household_response_count": len(records),
        "one_call_per_household": True,
        "live_call_count": call_budget.used,
        "cache_hit_count": sum(bool(record.get("cache_hit", False)) for record in records),
        "provider_response_count": sum("response_created_utc" in record for record in records),
        "worker_count": worker_count,
        "household_prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "bundle_sha256": bundle_sha,
        "household_input_sha256": _file_sha256(args.households),
        "history_input_sha256": _file_sha256(args.history),
        "recursive_state_input_sha256": (
            _file_sha256(args.state_json) if getattr(args, "state_json", None) else None
        ),
        "parent_scenario": parent_scenario,
        "accounting_passed": all(
            residual.passed for result in scenarios.values() for residual in result.accounting_residuals
        ),
        "max_abs_accounting_residual": max(result.max_abs_residual() for result in scenarios.values()),
        "target_values_loaded_into_prompts": False,
        "forecast_frozen_before_realization": forecast_frozen,
        "evaluation_status": "prospective_frozen" if forecast_frozen else "retrospective",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_sha256": _source_sha256(),
        "replay_equivalence_sha256": replay_equivalence_sha256,
        "expected_replay_sha256": expected_replay_sha256,
        "replay_verified": expected_replay_sha256 == replay_equivalence_sha256 if expected_replay_sha256 else None,
        "artifacts": {},
    }
    weights = {state.household_id: state.population_weight for state in states}
    weight_sum = sum(weights.values()) or 1.0
    def weighted_response(field: str) -> float:
        return sum(
            getattr(responses[household_id], field).p50 * weight
            for household_id, weight in weights.items()
        ) / weight_sum
    median_result = scenarios["median"]
    median_baseline = sum(row.baseline_consumption_usd for row in median_result.households)
    median_desired = sum(row.desired_consumption_usd for row in median_result.households)
    target_label = pd.Timestamp(target_month).strftime("%B %Y")
    report = [
        "# Household-First Rolling Microeconomy",
        "",
        "## Bottom Line",
        "",
        f"This is the first complete **{len(states)}-household** forecast from the new ecology. "
        "Each household was elicited separately; deterministic institutions then reconciled "
        "their choices into production, employment, inventories, credit, and settlement.",
        "",
        f"The median path predicts consumption growth of **{next(row['consumption_growth_pct'] for row in macro_rows if row['scenario'] == 'median'):.2f}%**. "
        f"Households themselves intended **{100.0 * (median_desired / max(median_baseline, 1.0) - 1.0):.2f}%**; "
        "the additional contraction comes from mandatory debt service and binding household resources, not an arbitrary aggregate gain.",
        "",
        (
            f"This is a frozen forecast, not an accuracy claim. {target_label} outcomes were not used and are not yet scored."
            if forecast_frozen
            else f"This is a retrospective simulation for the {target_label} target month, not a prospective forecast."
        ),
        "",
        "## Architecture",
        "",
        "```text",
        "survey-seeded household state + own history + as-of public information",
        "                              |",
        "                    one isolated LLM call",
        "                              |",
        "                 beliefs and intended choices",
        "                              |",
        "          deterministic budgets, credit, and production",
        "                              |",
        "       household actions -> employer -> macro state -> next origin",
        "```",
        "",
        "## Run Facts",
        "",
        f"- Forecast origin: `{args.origin}`; information cutoff: `{origin['as_of_date']}`.",
        f"- One-month-ahead target: `{target_month}`; status: `{manifest['evaluation_status']}`.",
        f"- Provider/model: `{args.provider}` / `{args.model}`.",
        f"- Accepted household responses: `{manifest['accepted_household_response_count']}`; "
        f"provider-created records represented: `{manifest['provider_response_count']}`.",
        f"- Replay cache hits in this execution: `{manifest['cache_hit_count']}`; fresh calls: `{manifest['live_call_count']}`.",
        f"- Accounting: **{'PASS' if manifest['accounting_passed'] else 'FAIL'}**; maximum residual `{manifest['max_abs_accounting_residual']:.3g}`.",
        f"- Origin snapshot hash: `{bundle_sha}`.",
        f"- Executable source hash: `{manifest['source_sha256']}`.",
        f"- Replay equivalence: `{manifest['replay_equivalence_sha256']}`; verified against an expected hash: `{manifest['replay_verified']}`.",
        "",
        "## Forecast Paths",
        "",
        "| Scenario | Consumption growth | Saving rate | Revolving-credit growth | Employment rate | Price growth |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in macro_rows:
        report.append(
            f"| {row['scenario']} | {row['consumption_growth_pct']:.2f}% | {row['saving_rate_pct']:.2f}% | "
            f"{row['revolving_credit_growth_pct']:.2f}% | {row['employment_rate_pct']:.2f}% | {row['price_growth_pct']:.2f}% |"
        )
    report.extend(
        [
            "",
            "## Household Signal",
            "",
            f"Population-weighted median beliefs imply inflation of **{weighted_response('expected_inflation_pct'):.2f}%**, "
            f"income growth of **{weighted_response('expected_income_growth_pct'):.2f}%**, and a "
            f"**{weighted_response('job_loss_probability_pct'):.2f}%** one-year job-loss probability.",
            "",
            f"The median economy executes `${sum(row.consumption_usd for row in median_result.households):,.0f}` of consumption, "
            f"`${sum(row.debt_payment_usd for row in median_result.households):,.0f}` of debt payments, and "
            f"`${sum(row.borrowing_usd for row in median_result.households):,.0f}` of new borrowing across the {len(states)} simulated household units.",
            "",
            "The cross-section is not a repeated representative household: low-liquidity agents plan larger consumption cuts, beliefs vary materially, and balance-sheet constraints alter feasible actions household by household.",
            "",
            "## What This Establishes",
            "",
            "The system is now a branchable, recursive microeconomy rather than an aggregate demand identity. Production comes from labor and capacity; sales clear against production plus inventories; employment and vacancies are explicit; loans, payments, deposits, and defaults have counterparties; and every scenario emits the exact state used by the next origin.",
            "",
            "The cohort is initialized from March-April 2025 SCE observations, the latest common two-wave panel used here, while the public macro card is current to the forecast cutoff. Household balance sheets are coarse survey mappings rather than contemporaneous measured accounts.",
            "",
            f"It does not yet establish predictive accuracy. Employment is still governed by one aggregate employer, and one untouched forecast origin cannot validate dynamics. The next evidence comes from appending realized {target_label} outcomes and repeating the same frozen procedure over several new months.",
        ]
    )
    (output / "ecology_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for path in sorted(output.iterdir()):
        if path.name != "manifest.json":
            manifest["artifacts"][path.name] = _artifact_sha256(path)
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
