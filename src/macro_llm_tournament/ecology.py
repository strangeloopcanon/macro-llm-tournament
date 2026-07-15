"""Household-first rolling microeconomy runner."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .ecology_engine import run_monthly_ecology
from .ecology_inputs import load_origin_information
from .ecology_information import build_macro_information_card
from .ecology_households import (
    HOUSEHOLD_PROMPT_VERSION,
    HouseholdElicitor,
    LiveCallBudget,
    canonical_sha256,
    household_card,
    normalize_household_payload,
)
from .ecology_models import CreditIntermediaryState, EmployerState, HouseholdState
from .ecology_provider import (
    CODEX_INSTRUCTION_CONTEXT_VERSION,
    CODEX_TOOL_ISOLATION_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "work/ecology_cache"
SCHEMA_VERSION = "household_first_rolling_microeconomy_v1"
LIVE_REFERENCE_SCHEMA_VERSION = "household_ecology_live_reference_v2"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="Monthly origin, YYYY-MM-DD")
    parser.add_argument("--mode", choices=("fixture", "replay", "live"), required=True)
    parser.add_argument("--provider", default="codex_cli", choices=("codex_cli",))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--household-count", type=int, default=200)
    parser.add_argument("--households", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--state-json", type=Path)
    parser.add_argument(
        "--state-policy",
        choices=("rolling_reanchored", "recursive"),
        default="rolling_reanchored",
    )
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


def _live_reference_path(
    cache_dir: Path,
    *,
    origin: str,
    provider: str,
    model: str,
    card_sha256s: Sequence[str],
) -> tuple[Path, str]:
    request_set_sha256 = canonical_sha256(
        {
            "origin": origin,
            "provider": provider,
            "model": model,
            "prompt_version": HOUSEHOLD_PROMPT_VERSION,
            "tool_isolation_version": CODEX_TOOL_ISOLATION_VERSION,
            "instruction_context_version": CODEX_INSTRUCTION_CONTEXT_VERSION,
            "engine_source_sha256": _source_sha256(),
            "card_sha256s": list(card_sha256s),
        }
    )
    return (
        cache_dir / "live_references" / f"{request_set_sha256}.json",
        request_set_sha256,
    )


def _read_live_reference(path: Path, request_set_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = payload.get("reference_sha256")
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "reference_sha256"}
    )
    if (
        payload.get("schema_version") != LIVE_REFERENCE_SCHEMA_VERSION
        or payload.get("request_set_sha256") != request_set_sha256
        or supplied_hash != expected_hash
    ):
        raise ValueError("live replay reference is malformed or mismatched")
    return payload


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
    committed = row.get("baseline_committed_consumption_monthly_usd")
    discretionary = row.get("baseline_discretionary_consumption_monthly_usd")
    has_components = pd.notna(committed) and pd.notna(discretionary)
    monthly_consumption = (
        float(committed) + float(discretionary)
        if has_components
        else float(row["baseline_consumption_annual"]) / 12.0
    )
    income = float(row.get("annual_income_usd", row.get("annual_income", 0.0)))
    reported_monthly_earned = row.get(
        "monthly_earned_income_usd", row.get("monthly_wage_income_usd")
    )
    has_reported_earned = pd.notna(reported_monthly_earned)
    monthly_earned = (
        float(reported_monthly_earned) if has_reported_earned else income / 12.0
    )
    status = (
        str(row.get("employment_status", "unknown"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    explicitly_employed = {
        "employed",
        "self_employed",
        "full_time",
        "part_time",
        "employed_full_time",
        "employed_part_time",
        "working",
    }
    explicitly_not_employed = {
        "unemployed",
        "not_employed",
        "retired",
        "out_of_labor_force",
        "not_in_labor_force",
        "disabled",
        "student",
        "homemaker",
    }
    if status in explicitly_employed:
        employed = True
    elif status in explicitly_not_employed:
        employed = False
    else:
        donor_status = str(row.get("scf_donor_employment_group", "")).strip().lower()
        # Unknown SCE employment inherits the matched SCF donor's labor-force
        # state. Legacy rows without donor provenance fall back to earned income.
        employed = (
            donor_status == "employed"
            if donor_status in {"employed", "not_employed"}
            else monthly_earned > 0.0
        )
    hours = 160.0 if employed else 0.0
    # SCF earned income is measured for the family, not this SCE respondent.
    # Real matched states therefore keep it fixed at the household boundary.
    # Synthetic legacy fixtures without an explicit earned-income field retain
    # their respondent-wage interpretation for dynamic-engine tests.
    household_earned = monthly_earned if has_reported_earned else 0.0
    respondent_earned = monthly_earned if employed and not has_reported_earned else 0.0
    hourly_wage = respondent_earned / 160.0 if respondent_earned > 0.0 else 0.0
    debt = max(0.0, float(row.get("revolving_debt_usd", row.get("debt", 0.0))))
    credit_limit = float(
        row.get(
            "revolving_credit_limit_usd",
            max(debt, debt * 1.5, monthly_consumption * 2.0),
        )
    )
    return HouseholdState(
        household_id=str(row["type_id"]),
        employer_id="aggregate_employer",
        deposit_balance_usd=max(
            0.0,
            float(row.get("liquid_deposits_usd", row.get("liquid_assets", 0.0))),
        ),
        revolving_debt_usd=debt,
        revolving_credit_limit_usd=max(debt, credit_limit),
        hourly_wage_usd=hourly_wage,
        baseline_monthly_hours=hours,
        baseline_monthly_consumption_usd=monthly_consumption,
        employment_share=1.0 if employed else 0.0,
        layoff_threshold_pct=50.0,
        liquid_buffer_floor_months=float(row.get("subsistence_floor_share", 0.5)),
        subsistence_consumption_share=float(row.get("subsistence_floor_share", 0.45)),
        population_weight=float(row.get("population_weight", 1.0 / 200.0)),
        monthly_household_earned_income_usd=household_earned,
        monthly_nonwage_income_usd=max(
            0.0, float(row.get("monthly_nonwage_income_usd", 0.0))
        ),
        monthly_transfer_income_usd=max(
            0.0, float(row.get("monthly_transfers_benefits_usd", 0.0))
        ),
        baseline_committed_consumption_usd=(float(committed) if has_components else None),
        baseline_discretionary_consumption_usd=(
            float(discretionary) if has_components else None
        ),
        minimum_debt_payment_usd=max(
            0.0, float(row.get("recurring_minimum_debt_payment_usd", 0.0))
        ),
    )


def _initial_employer(states: list[HouseholdState]) -> EmployerState:
    weight_total = sum(row.population_weight for row in states) or 1.0
    masses = {
        row.household_id: len(states) * row.population_weight / weight_total for row in states
    }
    employed_hours = sum(
        masses[row.household_id] * row.baseline_monthly_hours for row in states
    )
    baseline_demand = sum(
        masses[row.household_id] * row.baseline_monthly_consumption_usd for row in states
    )
    productivity = baseline_demand / max(employed_hours, 1.0)
    employed = sum(
        masses[row.household_id]
        * (row.employment_share if row.employment_share is not None else float(row.baseline_monthly_hours > 0))
        for row in states
    )
    employed_mass = max(employed, 1.0)
    weighted_wage = sum(
        masses[row.household_id]
        * (row.employment_share if row.employment_share is not None else float(row.baseline_monthly_hours > 0))
        * row.hourly_wage_usd
        for row in states
    ) / employed_mass
    return EmployerState(
        employer_id="aggregate_employer",
        productivity_per_hour=productivity,
        monthly_capacity_units=baseline_demand * 1.15,
        inventory_units=baseline_demand * 0.08,
        price_per_unit_usd=1.0,
        target_headcount=max(1.0, employed),
        wage_offer_usd=weighted_wage,
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
        new_lending_budget_usd=(
            len(states)
            * sum(
                row.population_weight
                * max(0.0, row.revolving_credit_limit_usd - row.revolving_debt_usd)
                for row in states
            )
            / max(sum(row.population_weight for row in states), 1e-12)
            * 0.2
        ),
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
    if payload.get("schema_version") not in {
        "household_ecology_recursive_state_v2",
        "household_ecology_recursive_state_v3",
    }:
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
        HouseholdState(
            **{
                field.name: by_id[state.household_id].get(
                    field.name, getattr(state, field.name)
                )
                for field in dataclasses.fields(HouseholdState)
            }
        )
        for state in states
    ]
    restored_employer = employer
    if isinstance(payload.get("employer"), dict):
        restored_employer = EmployerState(
            **{
                field.name: payload["employer"][field.name]
                for field in dataclasses.fields(EmployerState)
            }
        )
    restored_credit = credit
    if isinstance(payload.get("credit"), dict):
        restored_credit = CreditIntermediaryState(
            **{
                field.name: payload["credit"][field.name]
                for field in dataclasses.fields(CreditIntermediaryState)
            }
        )
    macro_state = payload.get("macro_state", {})
    if not isinstance(macro_state, dict):
        raise ValueError("recursive macro state must be an object")
    return restored, restored_employer, restored_credit, by_id, macro_state, str(parent_scenario)


def _latest_visible_value(origin: dict[str, Any], series_id: str) -> float | None:
    row = origin.get("origin_visible_macro_context", {}).get(series_id)
    if not isinstance(row, dict):
        return None
    try:
        value = float(row["value"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rolling_reanchor(
    *,
    restored: list[HouseholdState],
    anchors: list[HouseholdState],
    prior_macro_state: dict[str, Any],
    origin: dict[str, Any],
) -> tuple[list[HouseholdState], dict[str, Any]]:
    """Anchor each one-step forecast to the latest visible consumption level.

    Individual deposits, revolving debt, and employment state continue from the
    preceding simulation. The spending baseline is rebuilt from each household's
    fixed SCF-conditioned anchor and the aggregate PCE movement visible at the
    current origin, so forecast errors do not become next month's observed fact.
    """

    anchor_by_id = {row.household_id: row for row in anchors}
    current_pce = _latest_visible_value(origin, "PCE")
    reference_pce = prior_macro_state.get("anchor_reference_pce_value")
    if reference_pce is None:
        reference_pce = prior_macro_state.get("latest_visible_pce_value")
    if current_pce is None or reference_pce is None or float(reference_pce) <= 0.0:
        ratio = 1.0
        status = "pce_anchor_unavailable"
    else:
        ratio = max(0.75, min(1.25, current_pce / float(reference_pce)))
        status = "pce_level_reanchored"
    rows: list[HouseholdState] = []
    for state in restored:
        anchor = anchor_by_id[state.household_id]
        committed = (
            anchor.baseline_committed_consumption_usd * ratio
            if anchor.baseline_committed_consumption_usd is not None
            else None
        )
        discretionary = (
            anchor.baseline_discretionary_consumption_usd * ratio
            if anchor.baseline_discretionary_consumption_usd is not None
            else None
        )
        rows.append(
            dataclasses.replace(
                state,
                baseline_monthly_consumption_usd=(
                    anchor.baseline_monthly_consumption_usd * ratio
                ),
                baseline_committed_consumption_usd=committed,
                baseline_discretionary_consumption_usd=discretionary,
                hourly_wage_usd=anchor.hourly_wage_usd,
            )
        )
    return rows, {
        "status": status,
        "series_id": "PCE",
        "reference_visible_value": reference_pce,
        "current_visible_value": current_pce,
        "applied_level_ratio": ratio,
    }


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
    institution_mode: str = "dynamic",
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
                hourly_wage_usd=(
                    max(0.0, outcome.wage_income_usd / outcome.actual_hours_worked * wage_ratio)
                    if outcome.actual_hours_worked > 1e-12
                    else prior.hourly_wage_usd
                ),
                baseline_monthly_hours=outcome.actual_hours_worked,
                baseline_monthly_consumption_usd=max(1.0, outcome.consumption_usd),
                employment_share=outcome.employment_share_end,
                layoff_threshold_pct=prior.layoff_threshold_pct,
                liquid_buffer_floor_months=prior.liquid_buffer_floor_months,
                subsistence_consumption_share=prior.subsistence_consumption_share,
                population_weight=prior.population_weight,
                monthly_household_earned_income_usd=prior.monthly_household_earned_income_usd,
                monthly_nonwage_income_usd=prior.monthly_nonwage_income_usd,
                monthly_transfer_income_usd=prior.monthly_transfer_income_usd,
                baseline_committed_consumption_usd=prior.baseline_committed_consumption_usd,
                baseline_discretionary_consumption_usd=prior.baseline_discretionary_consumption_usd,
                minimum_debt_payment_usd=prior.minimum_debt_payment_usd,
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
            target_headcount=(
                employer.target_headcount
                if institution_mode == "household_demand"
                else max(1.0, result.employer.employment_count * pressure)
            ),
            wage_offer_usd=result.employer.next_wage_offer_usd,
            target_inventory_units=employer.target_inventory_units,
            fixed_cost_usd=employer.fixed_cost_usd,
            variable_nonlabor_cost_per_unit_usd=employer.variable_nonlabor_cost_per_unit_usd,
            vacancy_cost_per_opening_usd=employer.vacancy_cost_per_opening_usd,
            inventory_carry_cost_per_unit_usd=employer.inventory_carry_cost_per_unit_usd,
        )
    )
    weight_total = sum(row.population_weight for row in states) or 1.0
    headroom = len(states) * sum(
        prior_by_id[row["household_id"]].population_weight
        * max(0.0, row["revolving_credit_limit_usd"] - row["revolving_debt_usd"])
        for row in household_rows
    ) / weight_total
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
    payload = {
        "schema_version": "household_ecology_recursive_state_v3",
        "source_origin_month": origin["origin_month"],
        "next_origin_month": (
            pd.Timestamp(origin["origin_month"]) + pd.offsets.MonthBegin(1)
        ).date().isoformat(),
        "scenario": scenario,
        "households": household_rows,
        "macro_state": _weighted_macro(result, states, origin),
    }
    if institution_mode == "household_demand":
        macro = payload["macro_state"]
        payload["macro_state"] = {
            key: macro[key]
            for key in (
                "latest_visible_pce_value",
                "anchor_reference_pce_value",
                "state_policy",
            )
        }
    else:
        payload["employer"] = next_employer
        payload["credit"] = next_credit
    return payload


def _weighted_macro(result: Any, states: list[HouseholdState], origin: dict[str, Any]) -> dict[str, Any]:
    weights = {row.household_id: row.population_weight for row in states}
    total_weight = sum(weights.values()) or 1.0
    def weighted(field: str) -> float:
        return sum(getattr(row, field) * weights[row.household_id] for row in result.households) / total_weight
    baseline_consumption = sum(
        row.baseline_consumption_usd * weights[row.household_id] for row in result.households
    ) / total_weight
    debt_start = sum(row.revolving_debt_start_usd * weights[row.household_id] for row in result.households) / total_weight
    debt_end = sum(row.revolving_debt_end_usd * weights[row.household_id] for row in result.households) / total_weight
    prior_employed_weight = sum(
        weights[row.household_id]
        * (row.employment_share if row.employment_share is not None else float(row.baseline_monthly_hours > 0))
        for row in states
    ) / total_weight
    employed_weight = sum(
        weights[row.household_id] * row.employment_share_end for row in result.households
    ) / total_weight
    wage_income = weighted("wage_income_usd")
    household_earned_income = sum(
        row.monthly_household_earned_income_usd * weights[row.household_id]
        for row in states
    ) / total_weight
    nonwage_income = sum(
        row.monthly_nonwage_income_usd * weights[row.household_id]
        for row in states
    ) / total_weight
    transfer_income = sum(
        row.monthly_transfer_income_usd * weights[row.household_id]
        for row in states
    ) / total_weight
    disposable_income = (
        wage_income + household_earned_income + nonwage_income + transfer_income
    )
    consumption = weighted("consumption_usd")
    # Borrowing and debt repayment change financing composition; they are not
    # personal saving or dissaving. Household earned, nonwage, and transfer
    # income are resources in the engine, so reported saving includes them.
    saving = disposable_income - consumption
    saving_rate = 100.0 * saving / max(disposable_income, 1.0)
    baseline_wage_income = sum(
        row.hourly_wage_usd
        * row.baseline_monthly_hours
        * weights[row.household_id]
        for row in states
    ) / total_weight
    baseline_disposable_income = (
        baseline_wage_income
        + household_earned_income
        + nonwage_income
        + transfer_income
    )
    baseline_saving_rate = 100.0 * (
        baseline_disposable_income - baseline_consumption
    ) / max(baseline_disposable_income, 1.0)
    price_growth = 100.0 * (result.employer.next_price_per_unit_usd / result.employer.current_price_per_unit_usd - 1.0)
    latest_visible_pce = _latest_visible_value(origin, "PCE")
    consumption_growth = 100.0 * (
        weighted("consumption_usd") / max(baseline_consumption, 1.0) - 1.0
    )
    routine_drift = float(origin.get("routine_nominal_spending_drift_pct", 0.0))
    return {
        "consumption_growth_pct": consumption_growth,
        "routine_nominal_spending_drift_pct": routine_drift,
        "household_residual_contribution_pct": consumption_growth - routine_drift,
        "saving_rate_pct": saving_rate,
        "baseline_saving_rate_pct": baseline_saving_rate,
        "saving_rate_change_pp": saving_rate - baseline_saving_rate,
        "wage_income_usd": wage_income,
        "household_earned_income_usd": household_earned_income,
        "nonwage_income_usd": nonwage_income,
        "transfer_income_usd": transfer_income,
        "disposable_income_usd": disposable_income,
        "personal_saving_usd": saving,
        "revolving_credit_growth_pct": 100.0 * (debt_end / max(debt_start, 1.0) - 1.0),
        "employment_rate_pct": 100.0 * employed_weight,
        "employment_rate_change_pp": 100.0 * (employed_weight - prior_employed_weight),
        "unemployment_rate_pct": 100.0 * (1.0 - employed_weight),
        "price_growth_pct": price_growth,
        "output_units": result.employer.output_units,
        "units_sold": result.employer.units_sold,
        "inventory_end_units": result.employer.inventory_end_units,
        "policy_rate_pct": float(origin["origin_visible_macro_context"].get("FEDFUNDS", {}).get("value", 0.0)),
        "latest_visible_pce_value": latest_visible_pce,
        "anchor_reference_pce_value": origin.get(
            "anchor_reference_pce_value", latest_visible_pce
        ),
        "state_policy": origin.get("state_policy", "recursive"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    households, history, origin, bundle_sha = _load_inputs(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    states = [_state_from_row(row) for _, row in households.iterrows()]
    anchor_states = list(states)
    histories = {
        household_id: group.to_dict("records")
        for household_id, group in history.groupby("household_id", sort=True)
    }
    worker_count = int(getattr(args, "workers", 1))
    if worker_count <= 0:
        raise ValueError("--workers must be positive")
    attempt_journal_dir = (
        args.cache_dir
        / "live_attempts"
        / HOUSEHOLD_PROMPT_VERSION
        / str(args.origin)
    )
    call_budget = LiveCallBudget(args.max_live_calls, attempt_journal_dir)
    employer = _initial_employer(states)
    credit = _initial_credit(states, origin)
    states, employer, credit, recursive_rows, prior_macro_state, parent_scenario = _load_recursive_state(
        getattr(args, "state_json", None), states, employer, credit, args.origin
    )
    state_policy = getattr(args, "state_policy", "rolling_reanchored")
    reanchor = {
        "status": "survey_seeded_initial_state",
        "applied_level_ratio": 1.0,
    }
    if prior_macro_state and state_policy == "rolling_reanchored":
        states, reanchor = _rolling_reanchor(
            restored=states,
            anchors=anchor_states,
            prior_macro_state=prior_macro_state,
            origin=origin,
        )
        employer = _initial_employer(states)
        credit = _initial_credit(states, origin)
    elif prior_macro_state:
        reanchor = {
            "status": "recursive_unanchored",
            "applied_level_ratio": None,
        }
    origin = dict(origin)
    origin["state_policy"] = state_policy
    origin["anchor_reference_pce_value"] = prior_macro_state.get(
        "anchor_reference_pce_value",
        _latest_visible_value(origin, "PCE"),
    )
    origin["compact_macro_information"] = build_macro_information_card(
        origin,
        policy_declarations={
            "declared_spread_bps": 1400.0,
            "declared_pass_through_fraction": 1.0,
        },
    )
    pce_change = (
        origin["compact_macro_information"]
        .get("series", {})
        .get("PCE", {})
        .get("changes", {})
        .get("1m")
    )
    routine_nominal_drift_pct = (
        float(pce_change["value"])
        if isinstance(pce_change, dict) and pce_change.get("value") is not None
        else 0.0
    )
    origin["routine_nominal_spending_drift_pct"] = routine_nominal_drift_pct
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
        # Recursive rows contribute belief and outcome memory only. The exact
        # re-anchored engine state must win every overlapping field.
        card_state.update(recursive_rows.get(state.household_id, {}))
        card_state.update(dataclasses.asdict(state))
        card_state["employed"] = state.employment_share is not None and state.employment_share > 0.0
        card_state["annual_income"] = (
            state.hourly_wage_usd * state.baseline_monthly_hours * 12.0
        )
        card_state["state_provenance"] = (
            "rolling_observed_reanchor"
            if state.household_id in recursive_rows
            and state_policy == "rolling_reanchored"
            else (
                "prior_simulated_month"
                if state.household_id in recursive_rows
                else "survey_seeded_initial_state"
            )
        )
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

    institution_mode = "household_demand" if state_policy == "rolling_reanchored" else "dynamic"
    scenarios = {
        scenario: run_monthly_ecology(
            states,
            responses,
            employer,
            credit,
            scenario=scenario,
            institution_mode=institution_mode,
        )
        for scenario in ("downside", "median", "upside")
    }
    replay_equivalence_sha256 = canonical_sha256(
        {
            "cards": cards,
            "payloads": raw_payloads,
            "scenarios": {name: _jsonable(result) for name, result in scenarios.items()},
        }
    )
    reference_path, request_set_sha256 = _live_reference_path(
        args.cache_dir,
        origin=args.origin,
        provider=args.provider,
        model=args.model,
        card_sha256s=[event["card_sha256"] for event in events],
    )
    explicit_expected = getattr(args, "expected_replay_sha256", None)
    reference_payload: dict[str, Any] | None = None
    if args.mode == "live":
        reference_payload = {
            "schema_version": LIVE_REFERENCE_SCHEMA_VERSION,
            "request_set_sha256": request_set_sha256,
            "replay_equivalence_sha256": replay_equivalence_sha256,
            "provider": args.provider,
            "model": args.model,
            "origin_month": args.origin,
            "prompt_version": HOUSEHOLD_PROMPT_VERSION,
            "tool_isolation_version": CODEX_TOOL_ISOLATION_VERSION,
            "instruction_context_version": CODEX_INSTRUCTION_CONTEXT_VERSION,
            "engine_source_sha256": _source_sha256(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        reference_payload["reference_sha256"] = canonical_sha256(reference_payload)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        if reference_path.exists():
            existing = _read_live_reference(reference_path, request_set_sha256)
            if existing["replay_equivalence_sha256"] != replay_equivalence_sha256:
                raise ValueError("live replay reference already exists with a different result")
            reference_payload = existing
        else:
            _write_json(reference_path, reference_payload)
        expected_replay_sha256 = explicit_expected
    elif args.mode == "replay":
        if not reference_path.exists():
            raise ValueError("replay requires an immutable live reference")
        reference_payload = _read_live_reference(reference_path, request_set_sha256)
        reference_expected = str(reference_payload["replay_equivalence_sha256"])
        if explicit_expected is not None and explicit_expected != reference_expected:
            raise ValueError("explicit replay hash disagrees with immutable live reference")
        expected_replay_sha256 = reference_expected
    else:
        expected_replay_sha256 = explicit_expected
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
                institution_mode=institution_mode,
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
        "one_accepted_response_per_household": True,
        "population_mass_formula": "household_count * population_weight / sum(population_weight)",
        "institutional_aggregation": "population_mass_weighted",
        "state_policy": state_policy,
        "reanchor": reanchor,
        "institution_mode": institution_mode,
        "macro_information_card_sha256": origin["compact_macro_information"][
            "card_sha256"
        ],
        "routine_nominal_spending_drift_pct": routine_nominal_drift_pct,
        "routine_nominal_spending_drift_source": (
            "latest origin-visible one-month PCE change; zero only when unavailable"
        ),
        "employment_transition": (
            "origin_employment_state_frozen_for_household_demand_diagnostic"
            if institution_mode == "household_demand"
            else "expected_mass_from_monthly_job_loss_probability_with_fractional_hiring"
        ),
        "one_call_per_household": (
            args.mode == "live" and call_budget.used == len(records) and call_budget.failed == 0
        ),
        "live_call_count": call_budget.used,
        "fresh_accepted_response_count": call_budget.accepted,
        "failed_provider_attempt_count": call_budget.failed,
        "live_attempt_journal_sha256": (
            _artifact_sha256(attempt_journal_dir)
            if attempt_journal_dir.exists() and any(attempt_journal_dir.iterdir())
            else None
        ),
        "cache_hit_count": sum(bool(record.get("cache_hit", False)) for record in records),
        "provider_response_count": sum("response_created_utc" in record for record in records),
        "worker_count": worker_count,
        "household_prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "codex_tool_isolation_version": (
            CODEX_TOOL_ISOLATION_VERSION
            if args.mode in {"live", "replay"}
            else None
        ),
        "codex_instruction_context_version": (
            CODEX_INSTRUCTION_CONTEXT_VERSION
            if args.mode in {"live", "replay"}
            else None
        ),
        "local_text_file_tools_available_to_model": (
            False if args.mode in {"live", "replay"} else None
        ),
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
        "live_reference_sha256": (
            reference_payload.get("reference_sha256") if reference_payload else None
        ),
        "live_reference_request_set_sha256": request_set_sha256,
        "replay_reference_kind": (
            "immutable_live_run" if reference_payload is not None else None
        ),
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
    median_by_id = {row.household_id: row for row in median_result.households}
    population_scale = float(len(states))
    median_baseline = population_scale * sum(
        weights[row.household_id] * median_by_id[row.household_id].baseline_consumption_usd
        for row in states
    ) / weight_sum
    median_desired = population_scale * sum(
        weights[row.household_id] * median_by_id[row.household_id].desired_consumption_usd
        for row in states
    ) / weight_sum
    intended_growth = 100.0 * (median_desired / max(median_baseline, 1.0) - 1.0)
    executed_growth = next(
        row["consumption_growth_pct"]
        for row in macro_rows
        if row["scenario"] == "median"
    )
    feasibility_wedge = executed_growth - intended_growth
    execution_sentence = (
        "Deterministic execution leaves that aggregate intention unchanged."
        if abs(feasibility_wedge) < 0.005
        else (
            f"Deterministic feasibility changes it by **{feasibility_wedge:+.2f} percentage points** "
            "through debt service and binding household resources."
        )
    )
    target_label = pd.Timestamp(target_month).strftime("%B %Y")
    report = [
        "# Household-First Rolling Microeconomy",
        "",
        "## Bottom Line",
        "",
        f"This is a complete **{len(states)}-household** forecast from the household-first ecology. "
        "Each household was elicited separately; deterministic code then enforced budgets, "
        "credit limits, production feasibility, and settlement.",
        "",
        f"The median path predicts consumption growth of **{executed_growth:.2f}%**. "
        f"Households themselves intended **{intended_growth:.2f}%**. {execution_sentence}",
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
        "              one tool-isolated LLM elicitation",
        "                              |",
        "                 beliefs and intended choices",
        "                              |",
        "          deterministic budgets, credit, and production",
        "                              |",
        "          household actions -> aggregate demand and balances",
        "```",
        "",
        "## Run Facts",
        "",
        f"- Forecast origin: `{args.origin}`; information cutoff: `{origin['as_of_date']}`.",
        f"- One-month-ahead target: `{target_month}`; status: `{manifest['evaluation_status']}`.",
        f"- Provider/model: `{args.provider}` / `{args.model}`.",
        f"- Accepted household responses: `{manifest['accepted_household_response_count']}`; "
        f"provider-created records represented: `{manifest['provider_response_count']}`.",
        f"- Provider attempts in this execution: `{manifest['live_call_count']}`; "
        f"accepted fresh responses: `{manifest['fresh_accepted_response_count']}`; "
        f"failed attempts: `{manifest['failed_provider_attempt_count']}`.",
        f"- Model tool isolation: `{manifest['codex_tool_isolation_version']}`; "
        "each live call ran in a fresh empty directory with local-file, shell, web, browser, app, memory, and multi-agent access disabled.",
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
            f"The population-weighted mean of household p50 beliefs implies inflation of **{weighted_response('expected_inflation_pct'):.2f}%**, "
            f"income growth of **{weighted_response('expected_income_growth_pct'):.2f}%**, and a "
            f"**{weighted_response('job_loss_probability_pct'):.2f}%** next-month job-loss probability.",
            "",
            f"The population-mass-weighted median economy executes `${median_result.employer.revenue_usd:,.0f}` of consumption, "
            f"`${median_result.credit.debt_payments_received_usd:,.0f}` of debt payments, and "
            f"`${median_result.credit.borrowing_total_usd:,.0f}` of new borrowing across {len(states)} household types representing {len(states)} population-equivalent units.",
            "",
            "The cross-section is not a repeated representative household: beliefs, intended spending, debt actions, and binding balance-sheet constraints differ household by household.",
            "",
            "## What This Establishes",
            "",
            "This run is a household-demand microeconomy rather than an aggregate demand identity. Household policies generate demand from the bottom up; deterministic production follows expected sales with gradual inventory adjustment; loans, payments, deposits, and sales have explicit counterparties. Wages and respondent employment are held fixed in this diagnostic so an uncalibrated labor market cannot manufacture the consumption result.",
            "",
            "The cohort is initialized from March-April 2025 SCE observations, the latest common two-wave panel used here, while the public macro card is current to the forecast cutoff. Financial states are deterministic SCE-conditioned matches to public 2022 SCF households, not contemporaneous linked household accounts.",
            "",
            f"It does not yet establish predictive accuracy. The current run deliberately omits firm and bank decision agents, and one untouched forecast origin cannot validate dynamics. The next evidence comes from appending realized {target_label} outcomes and repeating the same frozen procedure over several new months.",
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
