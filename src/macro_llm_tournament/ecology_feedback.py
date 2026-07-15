"""Pure two-period aggregate producer feedback for the household ecology.

This module deliberately does not mutate household or engine state.  It makes
the small aggregate firm-to-household feedback leg available for isolated
experiments before it is wired into a recursive simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


INVENTORY_TARGET_SHARE = 0.08
INVENTORY_GAP_CLOSURE = 0.35
EMPLOYMENT_GAP_CLOSURE = 0.25
WAGE_EMPLOYMENT_ELASTICITY = 0.10
MAX_MONTHLY_WAGE_CHANGE = 0.02
_STOCK_TOLERANCE = 1e-9


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = _nonnegative_finite(value, name)
    if value == 0.0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class AggregateFirmPeriodOne:
    """Observed period-one firm aggregate, expressed in physical units."""

    demand_units: float
    sales_units: float
    opening_inventory_units: float
    closing_inventory_units: float
    base_output_units: float
    productivity_index: float
    producer_employment_index: float
    producer_wage_index: float

    def validate(self) -> None:
        demand = _nonnegative_finite(self.demand_units, "demand_units")
        sales = _nonnegative_finite(self.sales_units, "sales_units")
        opening_inventory = _nonnegative_finite(
            self.opening_inventory_units, "opening_inventory_units"
        )
        closing_inventory = _nonnegative_finite(
            self.closing_inventory_units, "closing_inventory_units"
        )
        output = _positive_finite(self.base_output_units, "base_output_units")
        _positive_finite(self.productivity_index, "productivity_index")
        _positive_finite(
            self.producer_employment_index, "producer_employment_index"
        )
        _positive_finite(self.producer_wage_index, "producer_wage_index")
        if sales - demand > _STOCK_TOLERANCE:
            raise ValueError("sales_units cannot exceed demand_units")
        if abs(opening_inventory + output - sales - closing_inventory) > _STOCK_TOLERANCE:
            raise ValueError(
                "period-one stock continuity requires opening inventory + output "
                "to equal sales + closing inventory"
            )


@dataclass(frozen=True)
class AggregateFirmFeedback:
    """Period-two aggregate producer plan implied by period-one observations."""

    expected_sales_units: float
    target_inventory_units: float
    inventory_gap_units: float
    planned_output_units: float
    required_labor_index: float
    producer_employment_index: float
    producer_wage_index: float
    employment_index_change: float
    wage_index_change: float
    wage_change_pct: float


def compute_aggregate_firm_feedback(
    period_one: AggregateFirmPeriodOne,
) -> AggregateFirmFeedback:
    """Compute the deterministic period-two producer response.

    Period-one demand is the expected-sales signal.  The observed sales value
    remains in the input so the physical stock identity is checked rather than
    silently assumed.
    """

    period_one.validate()
    expected_sales = float(period_one.demand_units)
    target_inventory = INVENTORY_TARGET_SHARE * expected_sales
    inventory_gap = target_inventory - float(period_one.closing_inventory_units)
    planned_output = max(0.0, expected_sales + INVENTORY_GAP_CLOSURE * inventory_gap)
    required_labor = planned_output / (
        float(period_one.base_output_units) * float(period_one.productivity_index)
    )
    employment = float(period_one.producer_employment_index) + EMPLOYMENT_GAP_CLOSURE * (
        required_labor - float(period_one.producer_employment_index)
    )
    employment_change = employment / float(period_one.producer_employment_index)
    wage_change_pct = max(
        -MAX_MONTHLY_WAGE_CHANGE,
        min(
            MAX_MONTHLY_WAGE_CHANGE,
            WAGE_EMPLOYMENT_ELASTICITY * (employment_change - 1.0),
        ),
    )
    wage_change = 1.0 + wage_change_pct
    return AggregateFirmFeedback(
        expected_sales_units=expected_sales,
        target_inventory_units=target_inventory,
        inventory_gap_units=inventory_gap,
        planned_output_units=planned_output,
        required_labor_index=required_labor,
        producer_employment_index=employment,
        producer_wage_index=float(period_one.producer_wage_index) * wage_change,
        employment_index_change=employment_change,
        wage_index_change=wage_change,
        wage_change_pct=wage_change_pct,
    )


@dataclass(frozen=True)
class HouseholdFamilyIncome:
    """Household income components kept separate for aggregate wage feedback."""

    respondent_employment_share: float
    family_wage_income_usd: float
    business_income_usd: float
    nonwage_income_usd: float
    transfer_income_usd: float

    @property
    def gross_income_usd(self) -> float:
        return (
            self.family_wage_income_usd
            + self.business_income_usd
            + self.nonwage_income_usd
            + self.transfer_income_usd
        )

    def validate(self) -> None:
        employment_share = _nonnegative_finite(
            self.respondent_employment_share, "respondent_employment_share"
        )
        if employment_share > 1.0:
            raise ValueError("respondent_employment_share must be between 0 and 1")
        for name, value in (
            ("family_wage_income_usd", self.family_wage_income_usd),
            ("business_income_usd", self.business_income_usd),
            ("nonwage_income_usd", self.nonwage_income_usd),
            ("transfer_income_usd", self.transfer_income_usd),
        ):
            _nonnegative_finite(value, name)


def apply_producer_income_feedback(
    income: HouseholdFamilyIncome,
    feedback: AggregateFirmFeedback,
) -> HouseholdFamilyIncome:
    """Apply aggregate employment and wage changes without changing respondent status."""

    income.validate()
    wage_income_multiplier = (
        float(feedback.employment_index_change) * float(feedback.wage_index_change)
    )
    if not math.isfinite(wage_income_multiplier) or wage_income_multiplier < 0.0:
        raise ValueError("feedback wage-income multiplier must be finite and nonnegative")
    return HouseholdFamilyIncome(
        respondent_employment_share=income.respondent_employment_share,
        family_wage_income_usd=income.family_wage_income_usd * wage_income_multiplier,
        business_income_usd=income.business_income_usd,
        nonwage_income_usd=income.nonwage_income_usd,
        transfer_income_usd=income.transfer_income_usd,
    )


def build_simulated_environment_payload(
    feedback: AggregateFirmFeedback,
) -> dict[str, Any]:
    """Return an explicitly labelled simulated input block for household cards."""

    return {
        "simulated_environment": {
            "source": "aggregate_firm_feedback",
            "period": 2,
            "producer": {
                "expected_sales_units": feedback.expected_sales_units,
                "target_inventory_units": feedback.target_inventory_units,
                "inventory_gap_units": feedback.inventory_gap_units,
                "planned_output_units": feedback.planned_output_units,
                "required_labor_index": feedback.required_labor_index,
                "producer_employment_index": feedback.producer_employment_index,
                "producer_wage_index": feedback.producer_wage_index,
                "employment_index_change": feedback.employment_index_change,
                "wage_index_change": feedback.wage_index_change,
                "wage_change_pct": feedback.wage_change_pct,
            },
        }
    }
