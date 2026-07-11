from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .ecology_models import (
    ACCOUNTING_TOLERANCE,
    ECOLOGY_SCHEMA_VERSION,
    AccountingResidual,
    CounterpartyFlow,
    CreditIntermediaryState,
    CreditMonthResult,
    EmployerMonthResult,
    EmployerState,
    HouseholdMonthResult,
    HouseholdResponse,
    HouseholdState,
    HouseholdTrajectory,
    HouseholdTrajectoryPoint,
    MonthlyEcologyResult,
)


@dataclass(frozen=True)
class _Plan:
    household: HouseholdState
    response: HouseholdResponse
    trajectory: HouseholdTrajectory
    realized_job_loss: bool
    actual_hours_worked: float
    actual_job_search_hours: float
    wage_income_usd: float
    interest_accrued_usd: float
    minimum_payment_due_usd: float
    borrowing_requested_usd: float
    desired_consumption_usd: float
    desired_buffer_end_usd: float
    debt_payment_usd: float
    consumption_before_goods_usd: float
    default_chargeoff_usd: float
    defaulted: bool
    revolving_debt_end_usd: float
    remaining_cash_after_payment_usd: float


def build_household_trajectory(response: HouseholdResponse) -> HouseholdTrajectory:
    response.validate()
    return HouseholdTrajectory(
        downside=HouseholdTrajectoryPoint(
            inflation_pct=float(response.expected_inflation_pct.p90),
            income_growth_pct=float(response.expected_income_growth_pct.p10),
            job_loss_probability_pct=float(response.job_loss_probability_pct.p90),
            consumption_change_pct=float(response.planned_consumption_change_pct.p10),
            planned_work_hours=float(response.planned_work_hours.p10),
            planned_job_search_hours=float(response.planned_job_search_hours.p90),
        ),
        median=HouseholdTrajectoryPoint(
            inflation_pct=float(response.expected_inflation_pct.p50),
            income_growth_pct=float(response.expected_income_growth_pct.p50),
            job_loss_probability_pct=float(response.job_loss_probability_pct.p50),
            consumption_change_pct=float(response.planned_consumption_change_pct.p50),
            planned_work_hours=float(response.planned_work_hours.p50),
            planned_job_search_hours=float(response.planned_job_search_hours.p50),
        ),
        upside=HouseholdTrajectoryPoint(
            inflation_pct=float(response.expected_inflation_pct.p10),
            income_growth_pct=float(response.expected_income_growth_pct.p90),
            job_loss_probability_pct=float(response.job_loss_probability_pct.p10),
            consumption_change_pct=float(response.planned_consumption_change_pct.p90),
            planned_work_hours=float(response.planned_work_hours.p90),
            planned_job_search_hours=float(response.planned_job_search_hours.p10),
        ),
    )


def run_monthly_ecology(
    households: list[HouseholdState],
    responses: Mapping[str, HouseholdResponse],
    employer: EmployerState,
    credit: CreditIntermediaryState,
    *,
    scenario: str = "median",
) -> MonthlyEcologyResult:
    if scenario not in {"downside", "median", "upside"}:
        raise ValueError("scenario must be downside, median, or upside")
    if not households:
        raise ValueError("households must be non-empty")
    employer.validate()
    credit.validate()
    ordered_households = sorted(households, key=lambda row: row.household_id)
    for household in ordered_households:
        household.validate()
        if household.employer_id != employer.employer_id:
            raise ValueError(
                f"household {household.household_id} employer_id does not match employer"
            )
        if household.household_id not in responses:
            raise ValueError(f"missing response for household {household.household_id}")

    headroom_start = sum(
        max(0.0, row.revolving_credit_limit_usd - row.revolving_debt_usd)
        for row in ordered_households
    )
    trajectories = {
        household.household_id: build_household_trajectory(responses[household.household_id])
        for household in ordered_households
    }
    points = {household_id: getattr(trajectory, scenario) for household_id, trajectory in trajectories.items()}
    retained = sum(
        household.baseline_monthly_hours > 0
        and points[household.household_id].job_loss_probability_pct < household.layoff_threshold_pct
        for household in ordered_households
    )
    openings = max(0, int(employer.target_headcount) - retained)
    candidates = sorted(
        (
            household
            for household in ordered_households
            if household.baseline_monthly_hours <= ACCOUNTING_TOLERANCE
            and points[household.household_id].planned_job_search_hours > 0
        ),
        key=lambda household: (
            -points[household.household_id].planned_job_search_hours,
            household.household_id,
        ),
    )
    hired_ids = {household.household_id for household in candidates[:openings]}
    plans = [
        _build_plan(
            household,
            responses[household.household_id],
            credit,
            scenario=scenario,
            hired=household.household_id in hired_ids,
        )
        for household in ordered_households
    ]

    total_requested = sum(row.borrowing_requested_usd for row in plans)
    if total_requested <= 0.0:
        rationing_ratio = 1.0
    elif math.isfinite(credit.new_lending_budget_usd):
        rationing_ratio = min(1.0, credit.new_lending_budget_usd / total_requested)
    else:
        rationing_ratio = 1.0

    goods_ready = []
    for plan in plans:
        borrowing_usd = plan.borrowing_requested_usd * rationing_ratio
        resources_after_rationing = (
            plan.household.deposit_balance_usd + plan.wage_income_usd + borrowing_usd
        )
        debt_available = (
            plan.household.revolving_debt_usd
            + plan.interest_accrued_usd
            + borrowing_usd
        )
        mandatory_payment_usd = min(
            debt_available,
            plan.minimum_payment_due_usd,
            resources_after_rationing,
        )
        remaining_after_mandatory = max(0.0, resources_after_rationing - mandatory_payment_usd)
        consumption_before_goods_usd = min(
            plan.desired_consumption_usd,
            remaining_after_mandatory,
        )
        remaining_after_consumption = remaining_after_mandatory - consumption_before_goods_usd
        desired_total_payment_usd = max(
            plan.minimum_payment_due_usd,
            plan.response.debt_payment_intent_usd,
        )
        extra_payment_target_usd = max(0.0, desired_total_payment_usd - mandatory_payment_usd)
        extra_payment_capacity_usd = max(
            0.0,
            remaining_after_consumption - plan.desired_buffer_end_usd,
        )
        debt_payment_usd = min(
            debt_available,
            mandatory_payment_usd + min(extra_payment_target_usd, extra_payment_capacity_usd),
        )
        debt_stock_before_chargeoff = (
            plan.household.revolving_debt_usd
            + plan.interest_accrued_usd
            + borrowing_usd
            - debt_payment_usd
        )
        defaulted = (
            plan.minimum_payment_due_usd - debt_payment_usd > ACCOUNTING_TOLERANCE
            and plan.realized_job_loss
        )
        default_chargeoff_usd = (
            debt_stock_before_chargeoff * credit.loss_given_default_pct / 100.0
            if defaulted
            else 0.0
        )
        debt_after = max(0.0, debt_stock_before_chargeoff - default_chargeoff_usd)
        remaining_cash_after_payment = (
            plan.household.deposit_balance_usd
            + plan.wage_income_usd
            + borrowing_usd
            - debt_payment_usd
        )
        goods_ready.append(
            {
                "plan": plan,
                "borrowing_usd": borrowing_usd,
                "debt_payment_usd": debt_payment_usd,
                "default_chargeoff_usd": default_chargeoff_usd,
                "defaulted": defaulted,
                "revolving_debt_end_usd": debt_after,
                "remaining_cash_after_payment_usd": remaining_cash_after_payment,
                "consumption_before_goods_usd": consumption_before_goods_usd,
            }
        )

    total_hours = sum(float(row["plan"].actual_hours_worked) for row in goods_ready)
    output_units = min(
        employer.monthly_capacity_units,
        employer.productivity_per_hour * total_hours,
    )
    available_units = employer.inventory_units + output_units
    desired_consumption_total = sum(
        float(row["consumption_before_goods_usd"]) for row in goods_ready
    )
    nominal_sales_capacity = available_units * employer.price_per_unit_usd
    goods_rationing_ratio = (
        min(1.0, nominal_sales_capacity / desired_consumption_total)
        if desired_consumption_total > 0.0
        else 1.0
    )

    household_results: list[HouseholdMonthResult] = []
    all_flows: list[CounterpartyFlow] = []
    for row in goods_ready:
        plan = row["plan"]
        borrowing_usd = float(row["borrowing_usd"])
        debt_payment_usd = float(row["debt_payment_usd"])
        consumption_usd = float(row["consumption_before_goods_usd"]) * goods_rationing_ratio
        deposit_end_usd = row["remaining_cash_after_payment_usd"] - consumption_usd
        cash_residual = (
            plan.household.deposit_balance_usd
            + plan.wage_income_usd
            + borrowing_usd
            - debt_payment_usd
            - consumption_usd
            - deposit_end_usd
        )
        debt_residual = (
            plan.household.revolving_debt_usd
            + plan.interest_accrued_usd
            + borrowing_usd
            - debt_payment_usd
            - row["default_chargeoff_usd"]
            - row["revolving_debt_end_usd"]
        )
        flows = _counterparty_flows(
            household=plan.household,
            employer=employer,
            credit=credit,
            wage_income_usd=plan.wage_income_usd,
            borrowing_usd=borrowing_usd,
            debt_payment_usd=debt_payment_usd,
            consumption_usd=consumption_usd,
            default_chargeoff_usd=float(row["default_chargeoff_usd"]),
        )
        all_flows.extend(flows)
        household_results.append(
            HouseholdMonthResult(
                household_id=plan.household.household_id,
                employer_id=plan.household.employer_id,
                trajectory=plan.trajectory,
                realized_job_loss=plan.realized_job_loss,
                actual_hours_worked=plan.actual_hours_worked,
                actual_job_search_hours=plan.actual_job_search_hours,
                wage_income_usd=plan.wage_income_usd,
                baseline_consumption_usd=plan.household.baseline_monthly_consumption_usd,
                desired_consumption_usd=plan.desired_consumption_usd,
                consumption_usd=consumption_usd,
                goods_rationing_ratio=goods_rationing_ratio,
                desired_buffer_end_usd=plan.desired_buffer_end_usd,
                deposit_balance_start_usd=plan.household.deposit_balance_usd,
                deposit_balance_end_usd=deposit_end_usd,
                revolving_debt_start_usd=plan.household.revolving_debt_usd,
                interest_accrued_usd=plan.interest_accrued_usd,
                minimum_payment_due_usd=plan.minimum_payment_due_usd,
                debt_payment_usd=debt_payment_usd,
                borrowing_requested_usd=plan.borrowing_requested_usd,
                borrowing_usd=borrowing_usd,
                default_chargeoff_usd=float(row["default_chargeoff_usd"]),
                defaulted=bool(row["defaulted"]),
                revolving_debt_end_usd=float(row["revolving_debt_end_usd"]),
                cash_residual_usd=cash_residual,
                debt_residual_usd=debt_residual,
                counterparties=tuple(flows),
            )
        )

    aggregate_consumption_usd = sum(row.consumption_usd for row in household_results)
    units_sold = (
        aggregate_consumption_usd / employer.price_per_unit_usd
        if employer.price_per_unit_usd > 0.0
        else 0.0
    )
    inventory_end_units = employer.inventory_units + output_units - units_sold
    employment_count = sum(
        1 for row in household_results if row.actual_hours_worked > ACCOUNTING_TOLERANCE
    )
    vacancies = max(0, int(employer.target_headcount) - employment_count)
    wage_bill_usd = sum(row.wage_income_usd for row in household_results)
    variable_cost_usd = output_units * employer.variable_nonlabor_cost_per_unit_usd
    vacancy_cost_usd = vacancies * employer.vacancy_cost_per_opening_usd
    inventory_carry_cost_usd = (
        max(inventory_end_units, 0.0) * employer.inventory_carry_cost_per_unit_usd
    )
    revenue_usd = aggregate_consumption_usd
    profit_usd = (
        revenue_usd
        - wage_bill_usd
        - variable_cost_usd
        - employer.fixed_cost_usd
        - vacancy_cost_usd
        - inventory_carry_cost_usd
    )
    goods_residual_units = (
        employer.inventory_units + output_units - units_sold - inventory_end_units
    )
    target_inventory = max(employer.target_inventory_units, 1.0)
    demand_pressure = (
        desired_consumption_total / nominal_sales_capacity
        if nominal_sales_capacity > 0.0
        else 0.0
    )
    inventory_gap = (target_inventory - inventory_end_units) / target_inventory
    next_price = employer.price_per_unit_usd * (
        1.0 + max(-0.10, min(0.10, 0.03 * (demand_pressure - 1.0) + 0.02 * inventory_gap))
    )
    vacancy_rate = vacancies / max(int(employer.target_headcount), 1)
    next_wage = employer.wage_offer_usd * (
        1.0 + max(-0.05, min(0.08, 0.02 * vacancy_rate + 0.01 * inventory_gap))
    )
    employer_result = EmployerMonthResult(
        employer_id=employer.employer_id,
        output_units=output_units,
        capacity_units=employer.monthly_capacity_units,
        inventory_start_units=employer.inventory_units,
        inventory_end_units=inventory_end_units,
        units_sold=units_sold,
        employment_count=employment_count,
        vacancies=vacancies,
        average_hourly_wage_usd=(
            wage_bill_usd / total_hours if total_hours > 0.0 else employer.wage_offer_usd
        ),
        current_price_per_unit_usd=employer.price_per_unit_usd,
        next_price_per_unit_usd=next_price,
        current_wage_offer_usd=employer.wage_offer_usd,
        next_wage_offer_usd=next_wage,
        wage_bill_usd=wage_bill_usd,
        revenue_usd=revenue_usd,
        variable_cost_usd=variable_cost_usd,
        fixed_cost_usd=employer.fixed_cost_usd,
        vacancy_cost_usd=vacancy_cost_usd,
        inventory_carry_cost_usd=inventory_carry_cost_usd,
        profit_usd=profit_usd,
        goods_residual_units=goods_residual_units,
        demand_pressure=demand_pressure,
    )

    deposits_start = sum(row.deposit_balance_usd for row in ordered_households)
    deposits_end = sum(row.deposit_balance_end_usd for row in household_results)
    debt_start = sum(row.revolving_debt_usd for row in ordered_households)
    debt_end = sum(row.revolving_debt_end_usd for row in household_results)
    credit_limits_total = sum(
        row.revolving_credit_limit_usd for row in ordered_households
    )
    borrowing_total = sum(row.borrowing_usd for row in household_results)
    interest_income = sum(row.interest_accrued_usd for row in household_results)
    minimum_payments_due = sum(
        row.minimum_payment_due_usd for row in household_results
    )
    debt_payments_received = sum(row.debt_payment_usd for row in household_results)
    chargeoffs = sum(row.default_chargeoff_usd for row in household_results)
    deposit_stock_residual = deposits_end - (
        deposits_start
        + wage_bill_usd
        + borrowing_total
        - debt_payments_received
        - aggregate_consumption_usd
    )
    debt_stock_residual = debt_end - (
        debt_start
        + interest_income
        + borrowing_total
        - debt_payments_received
        - chargeoffs
    )
    credit_result = CreditMonthResult(
        intermediary_id=credit.intermediary_id,
        deposits_start_usd=deposits_start,
        deposits_end_usd=deposits_end,
        revolving_debt_start_usd=debt_start,
        revolving_debt_end_usd=debt_end,
        credit_limits_total_usd=credit_limits_total,
        available_headroom_start_usd=headroom_start,
        new_lending_budget_usd=credit.new_lending_budget_usd,
        rationing_ratio=rationing_ratio,
        borrowing_total_usd=borrowing_total,
        interest_income_usd=interest_income,
        minimum_payments_due_usd=minimum_payments_due,
        debt_payments_received_usd=debt_payments_received,
        chargeoffs_usd=chargeoffs,
        default_count=sum(1 for row in household_results if row.defaulted),
        profit_usd=interest_income - chargeoffs,
        deposit_stock_residual_usd=deposit_stock_residual,
        debt_stock_residual_usd=debt_stock_residual,
    )

    accounting_residuals = _accounting_residuals(
        households=household_results,
        employer=employer_result,
        credit=credit_result,
    )
    return MonthlyEcologyResult(
        schema_version=ECOLOGY_SCHEMA_VERSION,
        households=tuple(household_results),
        employer=employer_result,
        credit=credit_result,
        counterparties=tuple(all_flows),
        accounting_residuals=tuple(accounting_residuals),
    )


def _build_plan(
    household: HouseholdState,
    response: HouseholdResponse,
    credit: CreditIntermediaryState,
    *,
    scenario: str,
    hired: bool = False,
) -> _Plan:
    trajectory = build_household_trajectory(response)
    point = getattr(trajectory, scenario)
    was_employed = household.baseline_monthly_hours > ACCOUNTING_TOLERANCE
    realized_job_loss = (
        was_employed
        and point.job_loss_probability_pct >= household.layoff_threshold_pct
    )
    employed_now = (was_employed and not realized_job_loss) or hired
    actual_hours_worked = (
        0.0
        if not employed_now
        else min(
            point.planned_work_hours if point.planned_work_hours > 0 else 160.0,
            max(160.0, household.baseline_monthly_hours * 1.25),
        )
    )
    actual_job_search_hours = max(
        point.planned_job_search_hours,
        12.0 if realized_job_loss else 0.0,
    )
    wage_income_usd = actual_hours_worked * household.hourly_wage_usd
    interest_accrued_usd = (
        household.revolving_debt_usd * credit.annual_interest_rate_pct / 1200.0
    )
    minimum_payment_due_usd = min(
        household.revolving_debt_usd + interest_accrued_usd,
        max(
            credit.minimum_payment_floor_usd,
            household.revolving_debt_usd * credit.minimum_payment_rate_pct / 100.0,
        ),
    )
    borrowing_headroom = max(
        0.0,
        household.revolving_credit_limit_usd - household.revolving_debt_usd,
    )
    desired_consumption_usd = household.baseline_monthly_consumption_usd * (
        1.0 + point.consumption_change_pct / 100.0
    )
    desired_consumption_usd = max(0.0, desired_consumption_usd)
    target_buffer_usd = max(
        household.liquid_buffer_floor_months * household.baseline_monthly_consumption_usd,
        response.target_buffer_months * household.baseline_monthly_consumption_usd,
    )
    # A buffer target is a stock objective, not an instruction to close the
    # entire gap this month. The stated contribution controls the monthly flow.
    desired_buffer_end_usd = max(
        household.deposit_balance_usd,
        min(
            target_buffer_usd,
            household.deposit_balance_usd + response.buffer_contribution_intent_usd,
        ),
    )
    resources_before_borrow = household.deposit_balance_usd + wage_income_usd
    subsistence_usd = (
        household.subsistence_consumption_share
        * household.baseline_monthly_consumption_usd
    )
    emergency_need = max(
        0.0,
        minimum_payment_due_usd + subsistence_usd - resources_before_borrow,
    )
    borrowing_requested_usd = min(
        borrowing_headroom,
        response.borrowing_intent_usd + emergency_need,
    )
    resources_after_borrow = resources_before_borrow + borrowing_requested_usd
    desired_total_payment = min(
        household.revolving_debt_usd + interest_accrued_usd + borrowing_requested_usd,
        max(minimum_payment_due_usd, response.debt_payment_intent_usd),
    )
    debt_payment_cap = max(0.0, resources_after_borrow - subsistence_usd)
    debt_payment_usd = min(desired_total_payment, debt_payment_cap)
    remaining_cash_after_payment_usd = resources_after_borrow - debt_payment_usd
    preferred_consumption_ceiling = max(
        0.0,
        remaining_cash_after_payment_usd - desired_buffer_end_usd,
    )
    consumption_before_goods_usd = min(
        desired_consumption_usd,
        remaining_cash_after_payment_usd,
    )
    if consumption_before_goods_usd > preferred_consumption_ceiling:
        consumption_before_goods_usd = max(
            min(subsistence_usd, remaining_cash_after_payment_usd),
            preferred_consumption_ceiling,
        )
    consumption_before_goods_usd = min(
        consumption_before_goods_usd,
        remaining_cash_after_payment_usd,
    )
    debt_after_before_chargeoff = (
        household.revolving_debt_usd
        + interest_accrued_usd
        + borrowing_requested_usd
        - debt_payment_usd
    )
    minimum_shortfall = minimum_payment_due_usd - debt_payment_usd
    defaulted = minimum_shortfall > ACCOUNTING_TOLERANCE and realized_job_loss
    default_chargeoff_usd = (
        debt_after_before_chargeoff * credit.loss_given_default_pct / 100.0
        if defaulted
        else 0.0
    )
    revolving_debt_end_usd = max(
        0.0,
        debt_after_before_chargeoff - default_chargeoff_usd,
    )
    return _Plan(
        household=household,
        response=response,
        trajectory=trajectory,
        realized_job_loss=realized_job_loss,
        actual_hours_worked=actual_hours_worked,
        actual_job_search_hours=actual_job_search_hours,
        wage_income_usd=wage_income_usd,
        interest_accrued_usd=interest_accrued_usd,
        minimum_payment_due_usd=minimum_payment_due_usd,
        borrowing_requested_usd=borrowing_requested_usd,
        desired_consumption_usd=desired_consumption_usd,
        desired_buffer_end_usd=desired_buffer_end_usd,
        debt_payment_usd=debt_payment_usd,
        consumption_before_goods_usd=consumption_before_goods_usd,
        default_chargeoff_usd=default_chargeoff_usd,
        defaulted=defaulted,
        revolving_debt_end_usd=revolving_debt_end_usd,
        remaining_cash_after_payment_usd=remaining_cash_after_payment_usd,
    )


def _counterparty_flows(
    *,
    household: HouseholdState,
    employer: EmployerState,
    credit: CreditIntermediaryState,
    wage_income_usd: float,
    borrowing_usd: float,
    debt_payment_usd: float,
    consumption_usd: float,
    default_chargeoff_usd: float,
) -> list[CounterpartyFlow]:
    flows: list[CounterpartyFlow] = []
    if wage_income_usd > 0.0:
        flows.append(
            CounterpartyFlow(
                from_party_id=employer.employer_id,
                from_party_type="employer",
                to_party_id=household.household_id,
                to_party_type="household",
                category="wages",
                amount_usd=wage_income_usd,
            )
        )
    if borrowing_usd > 0.0:
        flows.append(
            CounterpartyFlow(
                from_party_id=credit.intermediary_id,
                from_party_type="credit_intermediary",
                to_party_id=household.household_id,
                to_party_type="household",
                category="revolving_borrowing",
                amount_usd=borrowing_usd,
            )
        )
    if debt_payment_usd > 0.0:
        flows.append(
            CounterpartyFlow(
                from_party_id=household.household_id,
                from_party_type="household",
                to_party_id=credit.intermediary_id,
                to_party_type="credit_intermediary",
                category="debt_payment",
                amount_usd=debt_payment_usd,
            )
        )
    if consumption_usd > 0.0:
        flows.append(
            CounterpartyFlow(
                from_party_id=household.household_id,
                from_party_type="household",
                to_party_id=employer.employer_id,
                to_party_type="employer",
                category="consumption_spending",
                amount_usd=consumption_usd,
            )
        )
    if default_chargeoff_usd > 0.0:
        flows.append(
            CounterpartyFlow(
                from_party_id=credit.intermediary_id,
                from_party_type="credit_intermediary",
                to_party_id=household.household_id,
                to_party_type="household",
                category="default_chargeoff",
                amount_usd=default_chargeoff_usd,
                cash_flow=False,
            )
        )
    return flows


def _accounting_residuals(
    *,
    households: list[HouseholdMonthResult],
    employer: EmployerMonthResult,
    credit: CreditMonthResult,
) -> list[AccountingResidual]:
    residuals: list[AccountingResidual] = []
    for row in households:
        residuals.append(
            AccountingResidual(
                name=f"{row.household_id}:cash_budget",
                residual=row.cash_residual_usd,
            )
        )
        residuals.append(
            AccountingResidual(
                name=f"{row.household_id}:revolving_debt_stock",
                residual=row.debt_residual_usd,
            )
        )
    residuals.extend(
        [
            AccountingResidual(
                name="employer:goods_inventory",
                residual=employer.goods_residual_units,
            ),
            AccountingResidual(
                name="credit:deposit_stock_match",
                residual=credit.deposit_stock_residual_usd,
            ),
            AccountingResidual(
                name="credit:debt_stock_match",
                residual=credit.debt_stock_residual_usd,
            ),
            AccountingResidual(
                name="counterparty:wages",
                residual=employer.wage_bill_usd
                - sum(
                    flow.amount_usd
                    for row in households
                    for flow in row.counterparties
                    if flow.category == "wages"
                ),
            ),
            AccountingResidual(
                name="counterparty:consumption",
                residual=employer.revenue_usd
                - sum(
                    flow.amount_usd
                    for row in households
                    for flow in row.counterparties
                    if flow.category == "consumption_spending"
                ),
            ),
            AccountingResidual(
                name="counterparty:borrowing",
                residual=credit.borrowing_total_usd
                - sum(
                    flow.amount_usd
                    for row in households
                    for flow in row.counterparties
                    if flow.category == "revolving_borrowing"
                ),
            ),
            AccountingResidual(
                name="counterparty:debt_payment",
                residual=credit.debt_payments_received_usd
                - sum(
                    flow.amount_usd
                    for row in households
                    for flow in row.counterparties
                    if flow.category == "debt_payment"
                ),
            ),
        ]
    )
    return residuals
