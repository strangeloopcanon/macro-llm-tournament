"""Canonical household-state transition between monthly ecology periods."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import pandas as pd

from .ecology_feedback import (
    AggregateFirmFeedback,
    HouseholdFamilyIncome,
    apply_producer_income_feedback,
)
from .ecology_models import ACCOUNTING_TOLERANCE, HouseholdState


def transition_household_states(
    states: Sequence[HouseholdState],
    decisions: pd.DataFrame,
    feedback: AggregateFirmFeedback,
) -> tuple[list[HouseholdState], list[dict[str, Any]]]:
    """Carry settled stocks and producer income feedback into the next month."""

    if decisions["household_id"].astype(str).duplicated().any():
        raise ValueError("monthly decisions must contain unique household IDs")
    decision_by_id = decisions.set_index(
        decisions["household_id"].astype(str)
    ).to_dict("index")
    expected_ids = {state.household_id for state in states}
    if set(decision_by_id) != expected_ids:
        raise ValueError("monthly decisions must cover every household state exactly")

    next_states: list[HouseholdState] = []
    transitions: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda item: item.household_id):
        decision = decision_by_id[state.household_id]
        updated_income = apply_producer_income_feedback(
            HouseholdFamilyIncome(
                respondent_employment_share=float(state.employment_share or 0.0),
                family_wage_income_usd=state.monthly_family_wage_income_usd,
                business_income_usd=state.monthly_family_business_income_usd,
                nonwage_income_usd=state.monthly_nonwage_income_usd,
                transfer_income_usd=state.monthly_transfer_income_usd,
            ),
            feedback,
        )
        executed_consumption = float(decision["consumption_usd"])
        settled_committed = float(decision["committed_consumption_usd"])
        settled_discretionary = float(decision["discretionary_consumption_usd"])
        settled_one_off = float(decision["one_off_purchase_usd"])
        category_total = settled_committed + settled_discretionary + settled_one_off
        if not abs(executed_consumption - category_total) <= max(
            ACCOUNTING_TOLERANCE,
            1e-9 * max(1.0, abs(executed_consumption)),
        ):
            raise ValueError(
                f"household {state.household_id} consumption categories do not "
                "reconcile to settled consumption"
            )
        next_recurring_consumption = settled_committed + settled_discretionary
        next_state = dataclasses.replace(
            state,
            deposit_balance_usd=float(decision["deposit_balance_end_usd"]),
            revolving_debt_usd=float(decision["revolving_debt_end_usd"]),
            baseline_monthly_consumption_usd=next_recurring_consumption,
            baseline_committed_consumption_usd=settled_committed,
            baseline_discretionary_consumption_usd=settled_discretionary,
            monthly_family_wage_income_usd=updated_income.family_wage_income_usd,
            monthly_family_business_income_usd=updated_income.business_income_usd,
            monthly_household_earned_income_usd=(
                updated_income.family_wage_income_usd
                + updated_income.business_income_usd
            ),
        )
        next_state.validate()
        next_states.append(next_state)
        transitions.append(
            {
                "household_id": state.household_id,
                "respondent_employment_share_period_1": state.employment_share,
                "respondent_employment_share_period_2": next_state.employment_share,
                "deposit_balance_period_1_open_usd": state.deposit_balance_usd,
                "deposit_balance_period_1_close_period_2_open_usd": (
                    next_state.deposit_balance_usd
                ),
                "revolving_debt_period_1_open_usd": state.revolving_debt_usd,
                "revolving_debt_period_1_close_period_2_open_usd": (
                    next_state.revolving_debt_usd
                ),
                "family_wage_income_period_1_usd": state.monthly_family_wage_income_usd,
                "family_wage_income_period_2_usd": (
                    next_state.monthly_family_wage_income_usd
                ),
                "family_business_income_period_1_usd": (
                    state.monthly_family_business_income_usd
                ),
                "family_business_income_period_2_usd": (
                    next_state.monthly_family_business_income_usd
                ),
                "family_earned_income_period_2_usd": (
                    next_state.monthly_household_earned_income_usd
                ),
                "period_1_executed_consumption_usd": executed_consumption,
                "period_2_recurring_consumption_baseline_usd": next_recurring_consumption,
                "period_2_committed_consumption_baseline_usd": settled_committed,
                "period_2_discretionary_consumption_baseline_usd": settled_discretionary,
            }
        )
    return next_states, transitions
