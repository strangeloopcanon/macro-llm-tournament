from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_behavior import (
    bank_credit_multiplier,
    first_finite,
    firm_hiring_index,
    firm_price_pressure_index,
    forecast_uncertainty,
    response_by_variable,
    sector_value,
    standardized_signal,
    updated_belief,
)
from .agent_common import (
    ACCOUNTING_TOLERANCE,
    ADJUSTMENT_COST_RATE,
    AGENT_ECONOMY_VERSION,
    pct_change,
    weighted_sum,
)
from .agent_llm import AgentLLMClient
from .forecast_cards import ForecastCard

AGENT_BEHAVIOR_COLUMNS = (
    "consumption_change_pct",
    "desired_liquid_buffer_change_pct",
    "borrowing_desire_index",
    "portfolio_rebalance_to_liquid_pct",
    "job_search_intensity_index",
)
HOUSEHOLD_POLICY_CHOICES = ("direct", "rule", "residual_over_liquidity")
FEEDBACK_MODE_CHOICES = ("none", "closed_loop")


def run_agent_economy(
    cards: Iterable[ForecastCard],
    forecasts: pd.DataFrame,
    type_cells: pd.DataFrame,
    *,
    source_filters: Iterable[str] | None = None,
    agent_client: AgentLLMClient | None = None,
    household_policy: str = "direct",
    feedback_mode: str = "closed_loop",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if household_policy not in HOUSEHOLD_POLICY_CHOICES:
        raise ValueError(f"Unsupported household policy: {household_policy}")
    if feedback_mode not in FEEDBACK_MODE_CHOICES:
        raise ValueError(f"Unsupported feedback mode: {feedback_mode}")
    ordered_cards = sorted(list(cards), key=lambda card: (card.origin_index, card.variable, card.horizon))
    forecasts = forecasts.copy()
    forecasts["source"] = forecasts["source"].astype(str)
    selected_forecasts = _select_belief_forecasts(forecasts, source_filters=source_filters)
    state_by_key = _initial_state_by_source_type(selected_forecasts, type_cells)
    state_initial = pd.DataFrame(state_by_key.values()).sort_values(["belief_source", "type_id"]).reset_index(drop=True)
    desired_rows: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    state_history_rows: list[dict[str, Any]] = []

    origins = sorted({(card.origin_index, card.origin) for card in ordered_cards})
    for origin_index, origin in origins:
        origin_cards = [card for card in ordered_cards if card.origin_index == origin_index]
        origin_state_updates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for card in origin_cards:
            card_forecasts = selected_forecasts[selected_forecasts["card_id"].astype(str) == card.card_id]
            for source, source_forecasts in card_forecasts.groupby("source", sort=True):
                forecast = source_forecasts.iloc[0]
                event_desired = []
                policy_inputs = []
                prior_states = [dict(state_by_key[(source, str(type_cell["type_id"]))]) for _, type_cell in type_cells.iterrows()]
                agent_payload = (
                    agent_client.agent_panel(card, forecast, type_cells, prior_states)
                    if agent_client is not None
                    else None
                )
                for _, type_cell in type_cells.iterrows():
                    key = (source, str(type_cell["type_id"]))
                    prior_state = dict(state_by_key[key])
                    agent_response = (
                        agent_payload["household_by_type"].get(str(type_cell["type_id"]))
                        if agent_payload is not None
                        else None
                    )
                    policy_inputs.append(
                        {
                            "type_cell": type_cell,
                            "prior_state": prior_state,
                            "agent_response": agent_response,
                            "direct_response": _direct_behavior_response(card, forecast, type_cell, agent_response),
                            "rule_response": _rule_behavior_response(card, forecast, type_cell),
                            "liquidity_group": _liquidity_group(type_cell),
                            "population_weight": float(type_cell["population_weight"]),
                        }
                    )
                policy_inputs = _apply_household_policy(policy_inputs, household_policy)
                for item in policy_inputs:
                    desired = _desired_action(
                        card,
                        forecast,
                        item["type_cell"],
                        item["prior_state"],
                        agent_response=item["agent_response"],
                        behavior_response=item["policy_response"],
                        household_policy=household_policy,
                        liquidity_group=item["liquidity_group"],
                        sector_response=agent_payload if agent_payload is not None else None,
                    )
                    event_desired.append(desired)
                    desired_rows.append(desired)
                feasible = _reconcile_event(card, str(source), event_desired)
                aggregate = _aggregate_event(card, str(source), feasible)
                feasible = _attach_event_feedback(feasible, aggregate, feedback_mode=feedback_mode)
                feasible_rows.extend(feasible)
                aggregate_rows.append(aggregate)
                diagnostic_rows.append(_diagnose_event(card, str(source), feasible, aggregate))
                for row in feasible:
                    origin_state_updates[(str(source), str(row["type_id"]))].append(row)
        for key, rows in origin_state_updates.items():
            updated_state = _next_state_from_origin_rows(rows)
            state_by_key[key] = updated_state
            state_history_rows.append(
                {
                    **updated_state,
                    "origin_index": int(origin_index),
                    "origin": str(origin),
                    "state_stage": "post_origin",
                }
            )

    desired_frame = pd.DataFrame(desired_rows)
    feasible_frame = pd.DataFrame(feasible_rows)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    agent_scores = _score_agent_aggregates(aggregate_frame)
    state_final_rows = (
        pd.DataFrame(state_by_key.values())
        .sort_values(["belief_source", "type_id"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    desired_frame.attrs["state_history_records"] = state_history_rows
    desired_frame.attrs["state_final_records"] = state_final_rows
    return state_initial, desired_frame, feasible_frame, aggregate_frame, diagnostics, agent_scores


def _select_belief_forecasts(forecasts: pd.DataFrame, *, source_filters: Iterable[str] | None) -> pd.DataFrame:
    if source_filters is None:
        return forecasts.copy()
    filters = list(source_filters)
    if not filters:
        return forecasts.copy()
    mask = pd.Series(False, index=forecasts.index)
    for item in filters:
        if item == "llm":
            mask |= forecasts["source"].astype(str).str.startswith("llm_")
        else:
            source_text = forecasts["source"].astype(str)
            mask |= source_text.eq(item) | source_text.str.startswith(f"{item}__cf_")
    selected = forecasts[mask].copy()
    if selected.empty:
        raise ValueError(f"No forecasts match belief source filters: {filters}")
    return selected


def _initial_state_by_source_type(forecasts: pd.DataFrame, type_cells: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    states: dict[tuple[str, str], dict[str, Any]] = {}
    sources = sorted(forecasts["source"].astype(str).unique())
    for source in sources:
        for _, type_cell in type_cells.iterrows():
            type_id = str(type_cell["type_id"])
            states[(source, type_id)] = {
                "schema_version": AGENT_ECONOMY_VERSION,
                "belief_source": source,
                "type_id": type_id,
                "population_weight": float(type_cell["population_weight"]),
                "annual_income": float(type_cell["annual_income"]),
                "liquid_assets": float(type_cell["liquid_assets"]),
                "illiquid_assets": float(type_cell["illiquid_assets"]),
                "debt": float(type_cell["debt"]),
                "consumption_proxy_annual": float(type_cell["consumption_proxy_annual"]),
                "credit_limit_proxy": float(type_cell["credit_limit_proxy"]),
                "expected_inflation_1y": 2.5,
                "expected_real_income_growth": 1.5,
                "expected_unemployment_rate": 4.5,
                "expected_short_rate": 3.0,
                "confidence": 0.50,
                "uncertainty": 0.50,
                "desired_liquid_buffer_months": float(type_cell.get("liquid_buffer_months", 2.0)),
                "credit_access": "normal",
                "recent_forecast_error": 0.0,
                "last_origin": "initial",
            }
    return states


def _direct_behavior_response(
    card: ForecastCard,
    forecast: pd.Series,
    type_cell: pd.Series,
    agent_response: dict[str, Any] | None,
) -> dict[str, float]:
    if agent_response is not None:
        return _bounded_behavior_response(
            {
                "consumption_change_pct": float(agent_response["consumption_change_pct"]),
                "desired_liquid_buffer_change_pct": float(agent_response["liquid_buffer_change_pct"]),
                "borrowing_desire_index": float(agent_response["borrowing_desire_index"]),
                "portfolio_rebalance_to_liquid_pct": float(agent_response["portfolio_rebalance_to_liquid_pct"]),
                "job_search_intensity_index": float(agent_response["job_search_intensity_index"]),
            }
        )
    return _rule_behavior_response(card, forecast, type_cell)


def _rule_behavior_response(card: ForecastCard, forecast: pd.Series, type_cell: pd.Series) -> dict[str, float]:
    signal = standardized_signal(card, float(forecast["point_forecast"]))
    uncertainty = forecast_uncertainty(forecast, signal)
    return _bounded_behavior_response(
        response_by_variable(
            card.variable,
            signal,
            liquidity=float(type_cell["liquidity_sensitivity"]),
            rate=float(type_cell["rate_sensitivity"]),
            unemployment=float(type_cell["unemployment_sensitivity"]),
            portfolio=float(type_cell["portfolio_sensitivity"]),
            uncertainty=uncertainty,
        )
    )


def _apply_household_policy(policy_inputs: list[dict[str, Any]], household_policy: str) -> list[dict[str, Any]]:
    if household_policy == "direct":
        for item in policy_inputs:
            item["policy_response"] = dict(item["direct_response"])
        return policy_inputs
    if household_policy == "rule":
        for item in policy_inputs:
            item["policy_response"] = dict(item["rule_response"])
        return policy_inputs
    if household_policy != "residual_over_liquidity":
        raise ValueError(f"Unsupported household policy: {household_policy}")

    frame = pd.DataFrame(
        [
            {
                "idx": idx,
                "liquidity_group": item["liquidity_group"],
                "population_weight": item["population_weight"],
                **{f"{column}_direct": item["direct_response"][column] for column in AGENT_BEHAVIOR_COLUMNS},
                **{f"{column}_rule": item["rule_response"][column] for column in AGENT_BEHAVIOR_COLUMNS},
            }
            for idx, item in enumerate(policy_inputs)
        ]
    )
    if frame.empty:
        return policy_inputs
    for column in AGENT_BEHAVIOR_COLUMNS:
        residual = frame[f"{column}_direct"].astype(float) - frame[f"{column}_rule"].astype(float)
        centered = residual - residual.groupby(frame["liquidity_group"]).transform(
            lambda values: _weighted_mean(values, frame.loc[values.index, "population_weight"])
        )
        frame[f"{column}_policy"] = frame[f"{column}_rule"].astype(float) + centered

    for idx, item in enumerate(policy_inputs):
        item["policy_response"] = _bounded_behavior_response(
            {column: float(frame.loc[idx, f"{column}_policy"]) for column in AGENT_BEHAVIOR_COLUMNS}
        )
    return policy_inputs


def _bounded_behavior_response(response: dict[str, float]) -> dict[str, float]:
    bounds = {
        "consumption_change_pct": (-20.0, 20.0),
        "desired_liquid_buffer_change_pct": (-25.0, 25.0),
        "borrowing_desire_index": (-5.0, 5.0),
        "portfolio_rebalance_to_liquid_pct": (-15.0, 15.0),
        "job_search_intensity_index": (-3.0, 6.0),
    }
    return {column: float(np.clip(float(response[column]), low, high)) for column, (low, high) in bounds.items()}


def _liquidity_group(type_cell: pd.Series) -> str:
    type_id = str(type_cell["type_id"])
    if type_id in {"liquid_poor_renter", "wealthy_htm_homeowner", "unemployed_low_liquid"}:
        return "low"
    if type_id in {"retiree_liquid_assets", "high_income_illiquid_rich", "business_owner_top_wealth"}:
        return "high"
    return "middle"


def _agent_behavior_mode(agent_response: dict[str, Any] | None, household_policy: str) -> str:
    if household_policy == "residual_over_liquidity":
        return "llm_residual_over_liquidity" if agent_response is not None else "rule_residual_over_liquidity"
    if household_policy == "rule":
        return "policy_rule"
    return "llm_agent" if agent_response is not None else "rules"


def _desired_action(
    card: ForecastCard,
    forecast: pd.Series,
    type_cell: pd.Series,
    prior_state: dict[str, Any],
    *,
    agent_response: dict[str, Any] | None = None,
    behavior_response: dict[str, float] | None = None,
    household_policy: str = "direct",
    liquidity_group: str | None = None,
    sector_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal = standardized_signal(card, float(forecast["point_forecast"]))
    uncertainty = forecast_uncertainty(forecast, signal)
    if behavior_response is not None:
        response = behavior_response
    elif agent_response is not None:
        response = {
            "consumption_change_pct": float(agent_response["consumption_change_pct"]),
            "desired_liquid_buffer_change_pct": float(agent_response["liquid_buffer_change_pct"]),
            "borrowing_desire_index": float(agent_response["borrowing_desire_index"]),
            "portfolio_rebalance_to_liquid_pct": float(agent_response["portfolio_rebalance_to_liquid_pct"]),
            "job_search_intensity_index": float(agent_response["job_search_intensity_index"]),
        }
    else:
        response = response_by_variable(
            card.variable,
            signal,
            liquidity=float(type_cell["liquidity_sensitivity"]),
            rate=float(type_cell["rate_sensitivity"]),
            unemployment=float(type_cell["unemployment_sensitivity"]),
            portfolio=float(type_cell["portfolio_sensitivity"]),
            uncertainty=uncertainty,
        )
    baseline_consumption = max(0.0, float(prior_state["consumption_proxy_annual"]) / 4.0)
    liquid_assets = max(0.0, float(prior_state["liquid_assets"]))
    illiquid_assets = max(0.0, float(prior_state["illiquid_assets"]))
    annual_income = max(0.0, float(prior_state["annual_income"]))
    return {
        "schema_version": AGENT_ECONOMY_VERSION,
        "card_id": card.card_id,
        "source": str(forecast["source"]),
        "origin": card.origin,
        "origin_index": card.origin_index,
        "variable": card.variable,
        "horizon": card.horizon,
        "type_id": str(type_cell["type_id"]),
        "population_weight": float(type_cell["population_weight"]),
        "liquidity_group": liquidity_group or _liquidity_group(type_cell),
        "household_policy": household_policy,
        "belief_point_forecast": float(forecast["point_forecast"]),
        "belief_signal_vs_history": signal,
        "prior_liquid_assets": liquid_assets,
        "prior_illiquid_assets": illiquid_assets,
        "prior_debt": max(0.0, float(prior_state["debt"])),
        "annual_income": annual_income,
        "period_income": annual_income / 4.0,
        "baseline_consumption": baseline_consumption,
        "credit_limit_proxy": max(0.0, float(prior_state["credit_limit_proxy"])),
        "desired_consumption": max(0.0, baseline_consumption * (1.0 + response["consumption_change_pct"] / 100.0)),
        "desired_consumption_change_pct": response["consumption_change_pct"],
        "desired_liquid_buffer_change": liquid_assets * response["desired_liquid_buffer_change_pct"] / 100.0,
        "desired_borrowing": max(0.0, response["borrowing_desire_index"] * max(annual_income, 1.0) * 0.025),
        "desired_debt_repayment": max(0.0, -response["borrowing_desire_index"] * max(annual_income, 1.0) * 0.025),
        "desired_portfolio_to_liquid": illiquid_assets * response["portfolio_rebalance_to_liquid_pct"] / 100.0,
        "desired_job_search_intensity_index": response["job_search_intensity_index"],
        "updated_expected_inflation_1y": _agent_or_rule_belief(agent_response, "expected_inflation_1y", prior_state, card, "CPI", forecast),
        "updated_expected_real_income_growth": _agent_or_rule_belief(agent_response, "expected_real_income_growth", prior_state, card, "RGDP", forecast),
        "updated_expected_unemployment_rate": _agent_or_rule_belief(agent_response, "expected_unemployment_rate", prior_state, card, "UNEMP", forecast),
        "updated_expected_short_rate": _agent_or_rule_belief(agent_response, "expected_short_rate", prior_state, card, "TBILL", forecast),
        "updated_confidence": float(agent_response["confidence"]) if agent_response is not None else float(np.clip(1.0 - uncertainty / 8.0, 0.05, 0.95)),
        "updated_uncertainty": float(agent_response["uncertainty"]) if agent_response is not None else float(np.clip(uncertainty / 4.0, 0.05, 1.50)),
        "agent_behavior_mode": _agent_behavior_mode(agent_response, household_policy),
        "agent_firm_hiring_index": sector_value(sector_response, "firm", "hiring_index"),
        "agent_firm_price_pressure_index": sector_value(sector_response, "firm", "price_pressure_index"),
        "agent_bank_credit_multiplier": sector_value(sector_response, "bank", "credit_supply_multiplier"),
    }


def _reconcile_event(card: ForecastCard, source: str, desired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not desired_rows:
        return []
    desired_borrowing_total = weighted_sum(desired_rows, "desired_borrowing")
    credit_capacity_total = weighted_sum(desired_rows, "credit_limit_proxy")
    agent_bank_multiplier = first_finite(desired_rows, "agent_bank_credit_multiplier")
    bank_multiplier = (
        float(np.clip(agent_bank_multiplier, 0.35, 1.20))
        if np.isfinite(agent_bank_multiplier)
        else bank_credit_multiplier(card, desired_rows)
    )
    credit_supply = max(0.0, credit_capacity_total * bank_multiplier)
    credit_rationing_ratio = 1.0 if desired_borrowing_total <= credit_supply or desired_borrowing_total <= 0.0 else credit_supply / desired_borrowing_total
    feasible: list[dict[str, Any]] = []
    for row in desired_rows:
        out = dict(row)
        liquid_before = float(row["prior_liquid_assets"])
        illiquid_before = float(row["prior_illiquid_assets"])
        debt_before = float(row["prior_debt"])
        income = float(row["period_income"])
        borrowing = min(float(row["desired_borrowing"]) * credit_rationing_ratio, float(row["credit_limit_proxy"]))
        desired_portfolio = float(row["desired_portfolio_to_liquid"])
        portfolio_to_liquid = min(max(desired_portfolio, 0.0), illiquid_before)
        cash_sources = liquid_before + income + borrowing + portfolio_to_liquid
        repayment_capacity = max(0.0, cash_sources - 0.35 * float(row["baseline_consumption"]))
        debt_repayment = min(float(row["desired_debt_repayment"]), debt_before, repayment_capacity)
        cash_after_repayment = liquid_before + income + borrowing + portfolio_to_liquid - debt_repayment
        sale_adjustment_cost = ADJUSTMENT_COST_RATE * portfolio_to_liquid
        max_portfolio_to_illiquid = max(0.0, (cash_after_repayment - sale_adjustment_cost) / (1.0 + ADJUSTMENT_COST_RATE))
        portfolio_to_illiquid = min(max(-desired_portfolio, 0.0), max_portfolio_to_illiquid)
        adjustment_cost = ADJUSTMENT_COST_RATE * (portfolio_to_liquid + portfolio_to_illiquid)
        available_cash = liquid_before + income + borrowing + portfolio_to_liquid - debt_repayment - portfolio_to_illiquid - adjustment_cost
        if -ACCOUNTING_TOLERANCE <= available_cash < 0.0:
            available_cash = 0.0
        reserve_target = max(0.0, float(row["desired_liquid_buffer_change"]))
        consumption = min(float(row["desired_consumption"]), max(0.0, available_cash - reserve_target))
        liquid_after = available_cash - consumption
        illiquid_after = illiquid_before - portfolio_to_liquid + portfolio_to_illiquid
        debt_after = debt_before + borrowing - debt_repayment
        cash_residual = (
            liquid_before
            + income
            + borrowing
            + portfolio_to_liquid
            - debt_repayment
            - portfolio_to_illiquid
            - adjustment_cost
            - consumption
            - liquid_after
        )
        networth_before = liquid_before + illiquid_before - debt_before
        networth_after = liquid_after + illiquid_after - debt_after
        networth_residual = networth_after - (networth_before + income - consumption - adjustment_cost)
        balance_sheet_floor_ok = (
            liquid_after >= -ACCOUNTING_TOLERANCE
            and illiquid_after >= -ACCOUNTING_TOLERANCE
            and debt_after >= -ACCOUNTING_TOLERANCE
        )
        identity_ok = abs(cash_residual) <= ACCOUNTING_TOLERANCE and abs(networth_residual) <= ACCOUNTING_TOLERANCE
        out.update(
            {
                "bank_credit_supply_aggregate": credit_supply,
                "bank_credit_multiplier": bank_multiplier,
                "credit_rationing_ratio": credit_rationing_ratio,
                "feasible_borrowing": borrowing,
                "feasible_debt_repayment": debt_repayment,
                "feasible_portfolio_to_liquid": portfolio_to_liquid,
                "feasible_portfolio_to_illiquid": portfolio_to_illiquid,
                "portfolio_adjustment_cost": adjustment_cost,
                "feasible_consumption": consumption,
                "feasible_consumption_change_pct": pct_change(consumption, float(row["baseline_consumption"])),
                "liquid_assets_after": liquid_after,
                "illiquid_assets_after": illiquid_after,
                "debt_after": max(0.0, debt_after),
                "cash_accounting_residual": cash_residual,
                "networth_accounting_residual": networth_residual,
                "passes_balance_sheet_floors": balance_sheet_floor_ok,
                "passes_accounting": identity_ok and balance_sheet_floor_ok,
            }
        )
        feasible.append(out)
    return feasible


def _aggregate_event(card: ForecastCard, source: str, feasible_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_consumption = weighted_sum(feasible_rows, "baseline_consumption")
    consumption = weighted_sum(feasible_rows, "feasible_consumption")
    desired_consumption = weighted_sum(feasible_rows, "desired_consumption")
    aggregate_consumption_change = pct_change(consumption, baseline_consumption)
    demand_gap = pct_change(consumption, desired_consumption) if desired_consumption > 0 else 0.0
    belief_signal = weighted_sum(feasible_rows, "belief_signal_vs_history")
    agent_firm_hiring = first_finite(feasible_rows, "agent_firm_hiring_index")
    agent_price_pressure = first_finite(feasible_rows, "agent_firm_price_pressure_index")
    firm_hiring = float(np.clip(agent_firm_hiring, -5.0, 5.0)) if np.isfinite(agent_firm_hiring) else firm_hiring_index(card, belief_signal, aggregate_consumption_change)
    price_pressure = float(np.clip(agent_price_pressure, -5.0, 5.0)) if np.isfinite(agent_price_pressure) else firm_price_pressure_index(card, belief_signal, aggregate_consumption_change)
    return {
        "schema_version": AGENT_ECONOMY_VERSION,
        "card_id": card.card_id,
        "source": source,
        "origin": card.origin,
        "origin_index": card.origin_index,
        "variable": card.variable,
        "horizon": card.horizon,
        "aggregate_baseline_consumption": baseline_consumption,
        "aggregate_desired_consumption": desired_consumption,
        "aggregate_feasible_consumption": consumption,
        "aggregate_consumption_change_pct": aggregate_consumption_change,
        "aggregate_demand_gap_pct": demand_gap,
        "aggregate_liquid_buffer_change": weighted_sum(feasible_rows, "desired_liquid_buffer_change"),
        "aggregate_borrowing": weighted_sum(feasible_rows, "feasible_borrowing"),
        "aggregate_debt_repayment": weighted_sum(feasible_rows, "feasible_debt_repayment"),
        "aggregate_portfolio_to_liquid": weighted_sum(feasible_rows, "feasible_portfolio_to_liquid"),
        "aggregate_portfolio_to_illiquid": weighted_sum(feasible_rows, "feasible_portfolio_to_illiquid"),
        "aggregate_liquid_assets_after": weighted_sum(feasible_rows, "liquid_assets_after"),
        "aggregate_debt_after": weighted_sum(feasible_rows, "debt_after"),
        "aggregate_expected_inflation_1y": weighted_sum(feasible_rows, "updated_expected_inflation_1y"),
        "aggregate_expected_real_income_growth": weighted_sum(feasible_rows, "updated_expected_real_income_growth"),
        "aggregate_expected_unemployment_rate": weighted_sum(feasible_rows, "updated_expected_unemployment_rate"),
        "aggregate_expected_short_rate": weighted_sum(feasible_rows, "updated_expected_short_rate"),
        "aggregate_confidence": weighted_sum(feasible_rows, "updated_confidence"),
        "aggregate_uncertainty": weighted_sum(feasible_rows, "updated_uncertainty"),
        "credit_rationing_ratio": float(feasible_rows[0]["credit_rationing_ratio"]) if feasible_rows else np.nan,
        "firm_hiring_index": firm_hiring,
        "firm_price_pressure_index": price_pressure,
        "weighted_belief_signal": belief_signal,
        "target_realized": card.target_realized,
    }


def _attach_event_feedback(
    feasible_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    feedback_mode: str,
) -> list[dict[str, Any]]:
    if feedback_mode == "none":
        income_feedback_pct = 0.0
        credit_feedback_multiplier = 1.0
    else:
        firm_hiring = float(aggregate.get("firm_hiring_index", 0.0))
        price_pressure = float(aggregate.get("firm_price_pressure_index", 0.0))
        credit_shortfall = 1.0 - float(np.clip(aggregate.get("credit_rationing_ratio", 1.0), 0.0, 1.0))
        bank_multiplier = first_finite(feasible_rows, "bank_credit_multiplier")
        if not np.isfinite(bank_multiplier):
            bank_multiplier = 1.0
        income_feedback_pct = float(
            np.clip(0.20 * firm_hiring - 0.08 * max(price_pressure, 0.0) - 0.60 * credit_shortfall, -2.0, 2.0)
        )
        credit_feedback_multiplier = float(
            np.clip(1.0 + 0.08 * (bank_multiplier - 1.0) + 0.01 * firm_hiring - 0.05 * credit_shortfall, 0.90, 1.08)
        )
    out: list[dict[str, Any]] = []
    for row in feasible_rows:
        enriched = dict(row)
        enriched.update(
            {
                "feedback_mode": feedback_mode,
                "event_firm_hiring_index": float(aggregate.get("firm_hiring_index", np.nan)),
                "event_firm_price_pressure_index": float(aggregate.get("firm_price_pressure_index", np.nan)),
                "event_income_feedback_pct": income_feedback_pct,
                "event_credit_limit_feedback_multiplier": credit_feedback_multiplier,
            }
        )
        out.append(enriched)
    return out


def _diagnose_event(card: ForecastCard, source: str, feasible_rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    desired_borrowing = weighted_sum(feasible_rows, "desired_borrowing")
    feasible_borrowing = weighted_sum(feasible_rows, "feasible_borrowing")
    return {
        "schema_version": AGENT_ECONOMY_VERSION,
        "card_id": card.card_id,
        "source": source,
        "origin": card.origin,
        "variable": card.variable,
        "household_rows": len(feasible_rows),
        "max_abs_cash_residual": max((abs(float(row["cash_accounting_residual"])) for row in feasible_rows), default=0.0),
        "max_abs_networth_residual": max((abs(float(row["networth_accounting_residual"])) for row in feasible_rows), default=0.0),
        "min_liquid_assets_after": min((float(row["liquid_assets_after"]) for row in feasible_rows), default=0.0),
        "min_illiquid_assets_after": min((float(row["illiquid_assets_after"]) for row in feasible_rows), default=0.0),
        "min_debt_after": min((float(row["debt_after"]) for row in feasible_rows), default=0.0),
        "credit_market_gap": desired_borrowing - feasible_borrowing,
        "credit_rationing_ratio": aggregate.get("credit_rationing_ratio", np.nan),
        "passes_accounting": all(bool(row["passes_accounting"]) for row in feasible_rows),
    }


def _next_state_from_origin_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    annual_income = float(first["annual_income"]) * (1.0 + _mean_if_present(rows, "event_income_feedback_pct", default=0.0) / 100.0)
    credit_limit_proxy = float(first["credit_limit_proxy"]) * _mean_if_present(
        rows, "event_credit_limit_feedback_multiplier", default=1.0
    )
    return {
        "schema_version": AGENT_ECONOMY_VERSION,
        "belief_source": str(first["source"]),
        "type_id": str(first["type_id"]),
        "population_weight": float(first["population_weight"]),
        "annual_income": max(0.0, annual_income),
        "liquid_assets": _mean(rows, "liquid_assets_after"),
        "illiquid_assets": _mean(rows, "illiquid_assets_after"),
        "debt": _mean(rows, "debt_after"),
        "consumption_proxy_annual": max(0.0, _mean(rows, "feasible_consumption") * 4.0),
        "credit_limit_proxy": max(0.0, credit_limit_proxy),
        "expected_inflation_1y": _field_from_variable(rows, {"CPI"}, "updated_expected_inflation_1y"),
        "expected_real_income_growth": _field_from_variable(rows, {"RGDP"}, "updated_expected_real_income_growth"),
        "expected_unemployment_rate": _field_from_variable(rows, {"UNEMP"}, "updated_expected_unemployment_rate"),
        "expected_short_rate": _field_from_variable(rows, {"TBILL", "TBOND"}, "updated_expected_short_rate"),
        "confidence": _mean(rows, "updated_confidence"),
        "uncertainty": _mean(rows, "updated_uncertainty"),
        "desired_liquid_buffer_months": 12.0 * _mean(rows, "liquid_assets_after") / max(annual_income, 1.0),
        "credit_access": "tight" if _mean(rows, "credit_rationing_ratio") < 0.85 else "normal",
        "recent_forecast_error": 0.0,
        "last_origin": str(first["origin"]),
        "state_update_card_count": len(rows),
        "mean_income_feedback_pct": _mean_if_present(rows, "event_income_feedback_pct", default=0.0),
        "mean_credit_limit_feedback_multiplier": _mean_if_present(
            rows, "event_credit_limit_feedback_multiplier", default=1.0
        ),
    }


def _field_from_variable(rows: list[dict[str, Any]], variables: set[str], column: str) -> float:
    matching = [float(row[column]) for row in rows if str(row["variable"]) in variables]
    if matching:
        return float(np.mean(matching))
    return _mean(rows, column)


def _mean(rows: list[dict[str, Any]], column: str) -> float:
    values = [float(row[column]) for row in rows if np.isfinite(float(row[column]))]
    return float(np.mean(values)) if values else 0.0


def _mean_if_present(rows: list[dict[str, Any]], column: str, *, default: float) -> float:
    values = [float(row[column]) for row in rows if column in row and np.isfinite(float(row[column]))]
    return float(np.mean(values)) if values else float(default)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_weights = weights.astype(float).clip(lower=0.0)
    total = float(clean_weights.sum())
    if total <= 0:
        return float(values.astype(float).mean())
    return float((values.astype(float) * clean_weights).sum() / total)


def _score_agent_aggregates(aggregates: pd.DataFrame) -> pd.DataFrame:
    if aggregates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, group in aggregates.groupby("source", sort=True):
        score = float(
            group["aggregate_demand_gap_pct"].abs().mean()
            + 10.0 * (1.0 - group["credit_rationing_ratio"].clip(0.0, 1.0)).mean()
            + 0.1 * group["aggregate_consumption_change_pct"].abs().mean()
        )
        rows.append(
            {
                "source": source,
                "n": int(group.shape[0]),
                "score": score,
                "mean_abs_demand_gap_pct": float(group["aggregate_demand_gap_pct"].abs().mean()),
                "mean_credit_rationing": float((1.0 - group["credit_rationing_ratio"].clip(0.0, 1.0)).mean()),
                "mean_abs_consumption_change_pct": float(group["aggregate_consumption_change_pct"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def _agent_or_rule_belief(
    agent_response: dict[str, Any] | None,
    agent_field: str,
    prior_state: dict[str, Any],
    card: ForecastCard,
    target_variable: str,
    forecast: pd.Series,
) -> float:
    if agent_response is not None:
        return float(agent_response[agent_field])
    return updated_belief(prior_state, card.variable, target_variable, agent_field, forecast)
