from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


# Sub-micro-dollar tolerance absorbs floating-point summation at 200-household
# scale while remaining far below any economically meaningful transaction.
ACCOUNTING_TOLERANCE = 1e-6
ECOLOGY_SCHEMA_VERSION = "household_first_monthly_ecology_v4"
HOUSEHOLD_RESPONSE_SCHEMA_VERSION = "household_conditional_nominal_policy_v5"


def _require_finite(value: float, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def _require_nonnegative(value: float, field_name: str) -> float:
    out = _require_finite(value, field_name)
    if out < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")
    return out


def _require_probability(value: float, field_name: str) -> float:
    out = _require_finite(value, field_name)
    if out < 0.0 or out > 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return out


@dataclass(frozen=True)
class QuantileTriplet:
    p10: float
    p50: float
    p90: float

    def validate(
        self,
        field_name: str,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> None:
        values = {
            "p10": _require_finite(self.p10, f"{field_name}.p10"),
            "p50": _require_finite(self.p50, f"{field_name}.p50"),
            "p90": _require_finite(self.p90, f"{field_name}.p90"),
        }
        if not values["p10"] <= values["p50"] <= values["p90"]:
            raise ValueError(f"{field_name} must satisfy p10 <= p50 <= p90")
        if lower_bound is not None:
            for key, value in values.items():
                if value < lower_bound:
                    raise ValueError(f"{field_name}.{key} must be at least {lower_bound}")
        if upper_bound is not None:
            for key, value in values.items():
                if value > upper_bound:
                    raise ValueError(f"{field_name}.{key} must be at most {upper_bound}")


@dataclass(frozen=True)
class HouseholdTrajectoryPoint:
    inflation_pct: float
    income_growth_pct: float
    job_loss_probability_pct: float
    consumption_change_pct: float
    planned_work_hours: float
    planned_job_search_hours: float


@dataclass(frozen=True)
class HouseholdTrajectory:
    downside: HouseholdTrajectoryPoint
    median: HouseholdTrajectoryPoint
    upside: HouseholdTrajectoryPoint


@dataclass(frozen=True)
class HouseholdPolicyBranch:
    """A one-month household plan conditional on employment state."""

    next_month_committed_consumption_nominal_usd: float
    next_month_discretionary_consumption_nominal_usd: float
    deposit_change_intent_usd: float
    extra_debt_payment_usd: float
    borrowing_intent_usd: float

    def validate(self, field_name: str) -> None:
        _require_nonnegative(
            self.next_month_committed_consumption_nominal_usd,
            f"{field_name}.next_month_committed_consumption_nominal_usd",
        )
        _require_nonnegative(
            self.next_month_discretionary_consumption_nominal_usd,
            f"{field_name}.next_month_discretionary_consumption_nominal_usd",
        )
        _require_finite(
            self.deposit_change_intent_usd,
            f"{field_name}.deposit_change_intent_usd",
        )
        _require_nonnegative(
            self.extra_debt_payment_usd,
            f"{field_name}.extra_debt_payment_usd",
        )
        _require_nonnegative(
            self.borrowing_intent_usd,
            f"{field_name}.borrowing_intent_usd",
        )


@dataclass(frozen=True)
class HouseholdResponse:
    expected_inflation_pct: QuantileTriplet
    expected_income_growth_pct: QuantileTriplet
    job_loss_probability_pct: QuantileTriplet
    planned_consumption_change_pct: QuantileTriplet | None
    planned_work_hours: QuantileTriplet
    planned_job_search_hours: QuantileTriplet
    target_buffer_months: float
    buffer_contribution_intent_usd: float
    debt_payment_intent_usd: float
    borrowing_intent_usd: float
    employed_policy: HouseholdPolicyBranch | None = None
    not_employed_policy: HouseholdPolicyBranch | None = None

    def validate(self) -> None:
        self.expected_inflation_pct.validate(
            "expected_inflation_pct",
            lower_bound=-25.0,
            upper_bound=40.0,
        )
        self.expected_income_growth_pct.validate(
            "expected_income_growth_pct",
            lower_bound=-50.0,
            upper_bound=50.0,
        )
        self.job_loss_probability_pct.validate(
            "job_loss_probability_pct",
            lower_bound=0.0,
            upper_bound=100.0,
        )
        has_legacy_consumption = self.planned_consumption_change_pct is not None
        has_conditional_policy = (
            self.employed_policy is not None and self.not_employed_policy is not None
        )
        if has_legacy_consumption == has_conditional_policy:
            raise ValueError(
                "household response must provide exactly one of legacy consumption "
                "change or employed/not-employed policy branches"
            )
        if self.planned_consumption_change_pct is not None:
            self.planned_consumption_change_pct.validate(
                "planned_consumption_change_pct",
                lower_bound=-100.0,
                upper_bound=200.0,
            )
        else:
            assert self.employed_policy is not None
            assert self.not_employed_policy is not None
            self.employed_policy.validate("employed_policy")
            self.not_employed_policy.validate("not_employed_policy")
        self.planned_work_hours.validate(
            "planned_work_hours",
            lower_bound=0.0,
            upper_bound=320.0,
        )
        self.planned_job_search_hours.validate(
            "planned_job_search_hours",
            lower_bound=0.0,
            upper_bound=200.0,
        )
        _require_nonnegative(self.target_buffer_months, "target_buffer_months")
        _require_nonnegative(
            self.buffer_contribution_intent_usd,
            "buffer_contribution_intent_usd",
        )
        _require_nonnegative(self.debt_payment_intent_usd, "debt_payment_intent_usd")
        _require_nonnegative(self.borrowing_intent_usd, "borrowing_intent_usd")


@dataclass(frozen=True)
class HouseholdState:
    household_id: str
    employer_id: str
    deposit_balance_usd: float
    revolving_debt_usd: float
    revolving_credit_limit_usd: float
    hourly_wage_usd: float
    baseline_monthly_hours: float
    baseline_monthly_consumption_usd: float
    employment_share: float | None = None
    layoff_threshold_pct: float = 50.0
    liquid_buffer_floor_months: float = 0.5
    subsistence_consumption_share: float = 0.45
    population_weight: float = 1.0
    monthly_household_earned_income_usd: float = 0.0
    monthly_nonwage_income_usd: float = 0.0
    monthly_transfer_income_usd: float = 0.0
    baseline_committed_consumption_usd: float | None = None
    baseline_discretionary_consumption_usd: float | None = None
    minimum_debt_payment_usd: float = 0.0

    def validate(self) -> None:
        if not self.household_id:
            raise ValueError("household_id must be non-empty")
        if not self.employer_id:
            raise ValueError("employer_id must be non-empty")
        _require_nonnegative(self.deposit_balance_usd, "deposit_balance_usd")
        _require_nonnegative(self.revolving_debt_usd, "revolving_debt_usd")
        _require_nonnegative(
            self.revolving_credit_limit_usd,
            "revolving_credit_limit_usd",
        )
        if self.revolving_debt_usd - self.revolving_credit_limit_usd > ACCOUNTING_TOLERANCE:
            raise ValueError("revolving_debt_usd cannot exceed revolving_credit_limit_usd")
        _require_nonnegative(self.hourly_wage_usd, "hourly_wage_usd")
        _require_nonnegative(self.baseline_monthly_hours, "baseline_monthly_hours")
        _require_nonnegative(
            self.baseline_monthly_consumption_usd,
            "baseline_monthly_consumption_usd",
        )
        if self.employment_share is not None:
            share = _require_finite(self.employment_share, "employment_share")
            if share < 0.0 or share > 1.0:
                raise ValueError("employment_share must be between 0 and 1")
        _require_probability(self.layoff_threshold_pct, "layoff_threshold_pct")
        _require_nonnegative(
            self.liquid_buffer_floor_months,
            "liquid_buffer_floor_months",
        )
        share = _require_finite(
            self.subsistence_consumption_share,
            "subsistence_consumption_share",
        )
        if share < 0.0 or share > 1.0:
            raise ValueError("subsistence_consumption_share must be between 0 and 1")
        _require_nonnegative(self.population_weight, "population_weight")
        _require_nonnegative(
            self.monthly_household_earned_income_usd,
            "monthly_household_earned_income_usd",
        )
        _require_nonnegative(
            self.monthly_nonwage_income_usd,
            "monthly_nonwage_income_usd",
        )
        _require_nonnegative(
            self.monthly_transfer_income_usd,
            "monthly_transfer_income_usd",
        )
        for field_name, value in (
            ("baseline_committed_consumption_usd", self.baseline_committed_consumption_usd),
            ("baseline_discretionary_consumption_usd", self.baseline_discretionary_consumption_usd),
        ):
            if value is not None:
                _require_nonnegative(value, field_name)
        _require_nonnegative(
            self.minimum_debt_payment_usd,
            "minimum_debt_payment_usd",
        )


@dataclass(frozen=True)
class EmployerState:
    employer_id: str
    productivity_per_hour: float
    monthly_capacity_units: float
    inventory_units: float
    price_per_unit_usd: float
    target_headcount: float
    wage_offer_usd: float
    target_inventory_units: float = 0.0
    fixed_cost_usd: float = 0.0
    variable_nonlabor_cost_per_unit_usd: float = 0.0
    vacancy_cost_per_opening_usd: float = 0.0
    inventory_carry_cost_per_unit_usd: float = 0.0

    def validate(self) -> None:
        if not self.employer_id:
            raise ValueError("employer_id must be non-empty")
        _require_nonnegative(self.productivity_per_hour, "productivity_per_hour")
        _require_nonnegative(self.monthly_capacity_units, "monthly_capacity_units")
        _require_nonnegative(self.inventory_units, "inventory_units")
        _require_nonnegative(self.price_per_unit_usd, "price_per_unit_usd")
        _require_nonnegative(self.target_headcount, "target_headcount")
        _require_nonnegative(self.wage_offer_usd, "wage_offer_usd")
        _require_nonnegative(self.target_inventory_units, "target_inventory_units")
        _require_nonnegative(self.fixed_cost_usd, "fixed_cost_usd")
        _require_nonnegative(
            self.variable_nonlabor_cost_per_unit_usd,
            "variable_nonlabor_cost_per_unit_usd",
        )
        _require_nonnegative(
            self.vacancy_cost_per_opening_usd,
            "vacancy_cost_per_opening_usd",
        )
        _require_nonnegative(
            self.inventory_carry_cost_per_unit_usd,
            "inventory_carry_cost_per_unit_usd",
        )


@dataclass(frozen=True)
class CreditIntermediaryState:
    intermediary_id: str
    annual_interest_rate_pct: float
    minimum_payment_rate_pct: float
    minimum_payment_floor_usd: float = 25.0
    loss_given_default_pct: float = 60.0
    new_lending_budget_usd: float = math.inf

    def validate(self) -> None:
        if not self.intermediary_id:
            raise ValueError("intermediary_id must be non-empty")
        _require_nonnegative(
            self.annual_interest_rate_pct,
            "annual_interest_rate_pct",
        )
        _require_nonnegative(
            self.minimum_payment_rate_pct,
            "minimum_payment_rate_pct",
        )
        _require_nonnegative(
            self.minimum_payment_floor_usd,
            "minimum_payment_floor_usd",
        )
        _require_probability(
            self.loss_given_default_pct,
            "loss_given_default_pct",
        )
        budget = _require_finite(
            self.new_lending_budget_usd,
            "new_lending_budget_usd",
        ) if math.isfinite(self.new_lending_budget_usd) else self.new_lending_budget_usd
        if budget != math.inf and budget < 0.0:
            raise ValueError("new_lending_budget_usd must be nonnegative")


@dataclass(frozen=True)
class CounterpartyFlow:
    from_party_id: str
    from_party_type: str
    to_party_id: str
    to_party_type: str
    category: str
    amount_usd: float
    cash_flow: bool = True


@dataclass(frozen=True)
class AccountingResidual:
    name: str
    residual: float
    tolerance: float = ACCOUNTING_TOLERANCE

    @property
    def passed(self) -> bool:
        return abs(self.residual) <= self.tolerance


@dataclass(frozen=True)
class HouseholdMonthResult:
    household_id: str
    employer_id: str
    trajectory: HouseholdTrajectory
    realized_job_loss: bool
    employment_share_start: float
    job_loss_share: float
    hired_share: float
    employment_share_end: float
    actual_hours_worked: float
    actual_job_search_hours: float
    wage_income_usd: float
    baseline_consumption_usd: float
    desired_consumption_usd: float
    consumption_usd: float
    goods_rationing_ratio: float
    desired_buffer_end_usd: float
    deposit_balance_start_usd: float
    deposit_balance_end_usd: float
    revolving_debt_start_usd: float
    interest_accrued_usd: float
    minimum_payment_due_usd: float
    debt_payment_usd: float
    borrowing_requested_usd: float
    borrowing_usd: float
    default_chargeoff_usd: float
    defaulted: bool
    revolving_debt_end_usd: float
    cash_residual_usd: float
    debt_residual_usd: float
    counterparties: tuple[CounterpartyFlow, ...]


@dataclass(frozen=True)
class EmployerMonthResult:
    employer_id: str
    output_units: float
    capacity_units: float
    inventory_start_units: float
    inventory_end_units: float
    units_sold: float
    employment_count: float
    vacancies: float
    average_hourly_wage_usd: float
    current_price_per_unit_usd: float
    next_price_per_unit_usd: float
    current_wage_offer_usd: float
    next_wage_offer_usd: float
    wage_bill_usd: float
    revenue_usd: float
    variable_cost_usd: float
    fixed_cost_usd: float
    vacancy_cost_usd: float
    inventory_carry_cost_usd: float
    profit_usd: float
    goods_residual_units: float
    demand_pressure: float


@dataclass(frozen=True)
class CreditMonthResult:
    intermediary_id: str
    deposits_start_usd: float
    deposits_end_usd: float
    revolving_debt_start_usd: float
    revolving_debt_end_usd: float
    credit_limits_total_usd: float
    available_headroom_start_usd: float
    new_lending_budget_usd: float
    rationing_ratio: float
    borrowing_total_usd: float
    interest_income_usd: float
    minimum_payments_due_usd: float
    debt_payments_received_usd: float
    chargeoffs_usd: float
    default_count: int
    profit_usd: float
    deposit_stock_residual_usd: float
    debt_stock_residual_usd: float


@dataclass(frozen=True)
class MonthlyEcologyResult:
    schema_version: str
    households: tuple[HouseholdMonthResult, ...]
    employer: EmployerMonthResult
    credit: CreditMonthResult
    counterparties: tuple[CounterpartyFlow, ...]
    accounting_residuals: tuple[AccountingResidual, ...]

    @property
    def aggregate_consumption_usd(self) -> float:
        return float(self.employer.revenue_usd)

    @property
    def aggregate_borrowing_usd(self) -> float:
        return float(self.credit.borrowing_total_usd)

    def max_abs_residual(self) -> float:
        return max((abs(row.residual) for row in self.accounting_residuals), default=0.0)


def household_response_schema() -> dict[str, Any]:
    quantile_block = {
        "type": "object",
        "required": ["p10", "p50", "p90"],
        "properties": {
            "p10": {"type": "number"},
            "p50": {"type": "number"},
            "p90": {"type": "number"},
        },
        "rule": "p10 <= p50 <= p90",
    }
    policy_block = {
        "type": "object",
        "required": [
            "next_month_committed_consumption_nominal_usd",
            "next_month_discretionary_consumption_nominal_usd",
            "deposit_change_intent_usd",
            "extra_debt_payment_usd",
            "borrowing_intent_usd",
        ],
        "properties": {
            "next_month_committed_consumption_nominal_usd": {"type": "number", "minimum": 0.0},
            "next_month_discretionary_consumption_nominal_usd": {"type": "number", "minimum": 0.0},
            "deposit_change_intent_usd": {"type": "number"},
            "extra_debt_payment_usd": {"type": "number", "minimum": 0.0},
            "borrowing_intent_usd": {"type": "number", "minimum": 0.0},
        },
    }
    return {
        "schema_version": HOUSEHOLD_RESPONSE_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "prompt_version",
            "household_id",
            "expected_inflation_pct",
            "expected_income_growth_pct",
            "job_loss_probability_pct",
            "planned_work_hours",
            "planned_job_search_hours",
            "employed_policy",
            "not_employed_policy",
            "reason_codes",
        ],
        "properties": {
            "prompt_version": {"type": "string", "const": "household_ecology_monthly_v14"},
            "household_id": {"type": "string", "minLength": 1},
            "expected_inflation_pct": quantile_block,
            "expected_income_growth_pct": quantile_block,
            "job_loss_probability_pct": quantile_block,
            "planned_work_hours": quantile_block,
            "planned_job_search_hours": quantile_block,
            "employed_policy": policy_block,
            "not_employed_policy": policy_block,
            "reason_codes": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "maxLength": 240},
            },
        },
        "trajectory_mapping": {
            "downside": {
                "inflation_pct": "expected_inflation_pct.p90",
                "income_growth_pct": "expected_income_growth_pct.p10",
                "job_loss_probability_pct": "job_loss_probability_pct.p90",
                "consumption_policy": "employed/not_employed_policy mixed by employment share",
                "planned_work_hours": "planned_work_hours.p10",
                "planned_job_search_hours": "planned_job_search_hours.p90",
            },
            "median": {
                "inflation_pct": "expected_inflation_pct.p50",
                "income_growth_pct": "expected_income_growth_pct.p50",
                "job_loss_probability_pct": "job_loss_probability_pct.p50",
                "consumption_policy": "employed/not_employed_policy mixed by employment share",
                "planned_work_hours": "planned_work_hours.p50",
                "planned_job_search_hours": "planned_job_search_hours.p50",
            },
            "upside": {
                "inflation_pct": "expected_inflation_pct.p10",
                "income_growth_pct": "expected_income_growth_pct.p90",
                "job_loss_probability_pct": "job_loss_probability_pct.p10",
                "consumption_policy": "employed/not_employed_policy mixed by employment share",
                "planned_work_hours": "planned_work_hours.p90",
                "planned_job_search_hours": "planned_job_search_hours.p10",
            },
        },
    }
