from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .forecast_cards import ForecastCard


FORECAST_AGENT_PANEL_VERSION = "forecast_first_typed_agent_panel_v1"


FORECAST_AGENT_TYPES: tuple[dict[str, Any], ...] = (
    {"type_id": "liquid_poor_renter", "population_weight": 0.22, "liquidity_sensitivity": 1.45, "rate_sensitivity": 0.65},
    {"type_id": "wealthy_htm_homeowner", "population_weight": 0.10, "liquidity_sensitivity": 1.20, "rate_sensitivity": 1.05},
    {"type_id": "leveraged_homeowner", "population_weight": 0.13, "liquidity_sensitivity": 1.10, "rate_sensitivity": 1.35},
    {"type_id": "middle_income_buffer", "population_weight": 0.24, "liquidity_sensitivity": 0.80, "rate_sensitivity": 0.75},
    {"type_id": "retiree_liquid_assets", "population_weight": 0.12, "liquidity_sensitivity": 0.55, "rate_sensitivity": 0.90},
    {"type_id": "high_income_illiquid_rich", "population_weight": 0.09, "liquidity_sensitivity": 0.40, "rate_sensitivity": 0.80},
    {"type_id": "unemployed_low_liquid", "population_weight": 0.06, "liquidity_sensitivity": 1.70, "rate_sensitivity": 0.50},
    {"type_id": "business_owner_top_wealth", "population_weight": 0.04, "liquidity_sensitivity": 0.70, "rate_sensitivity": 1.15},
)


def build_forecast_agent_panel(cards: Iterable[ForecastCard], forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cards_by_id = {card.card_id: card for card in cards}
    llm_rows = forecasts[forecasts["source"].astype(str).str.startswith("llm_")].copy()
    rows: list[dict[str, Any]] = []
    for _, forecast in llm_rows.iterrows():
        card = cards_by_id.get(str(forecast["card_id"]))
        if card is None:
            continue
        point = float(forecast["point_forecast"])
        signal = _standardized_signal(card, point)
        for agent_type in FORECAST_AGENT_TYPES:
            row = _agent_response_row(card, forecast, agent_type, signal)
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel, pd.DataFrame()
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in panel.groupby(["card_id", "variable", "origin", "source"], dropna=False):
        card_id, variable, origin, source = keys
        row = _weighted_panel_average(group).to_dict()
        row.update({"card_id": card_id, "variable": variable, "origin": origin, "source": source})
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    return panel, aggregate


def _agent_response_row(
    card: ForecastCard,
    forecast: pd.Series,
    agent_type: dict[str, Any],
    signal: float,
) -> dict[str, Any]:
    liquidity = float(agent_type["liquidity_sensitivity"])
    rate = float(agent_type["rate_sensitivity"])
    uncertainty = max(0.0, float(forecast.get("p90", np.nan)) - float(forecast.get("p10", np.nan)))
    if not np.isfinite(uncertainty):
        uncertainty = max(0.1, abs(signal))
    response = _response_by_variable(card.variable, signal, liquidity=liquidity, rate=rate)
    return {
        "schema_version": FORECAST_AGENT_PANEL_VERSION,
        "card_id": card.card_id,
        "source": forecast["source"],
        "variable": card.variable,
        "origin": card.origin,
        "type_id": agent_type["type_id"],
        "population_weight": float(agent_type["population_weight"]),
        "belief_point_forecast": float(forecast["point_forecast"]),
        "belief_signal_vs_history": float(signal),
        "consumption_change_pct": response["consumption_change_pct"],
        "desired_liquid_buffer_change_pct": response["desired_liquid_buffer_change_pct"],
        "borrowing_desire_index": response["borrowing_desire_index"],
        "job_search_intensity_index": response["job_search_intensity_index"],
        "portfolio_rebalance_to_liquid_pct": response["portfolio_rebalance_to_liquid_pct"],
        "uncertainty_index": float(np.clip(0.25 + 0.15 * uncertainty + 0.2 * abs(signal), 0.0, 1.5)),
    }


def _response_by_variable(variable: str, signal: float, *, liquidity: float, rate: float) -> dict[str, float]:
    if variable == "CPI":
        consumption = -0.35 * liquidity * max(signal, -1.5)
        buffer = 0.60 * liquidity * signal
        borrowing = -0.25 * liquidity * max(signal, 0.0)
        job = 0.05 * max(signal, 0.0)
        portfolio = 0.10 * liquidity * signal
    elif variable == "RGDP":
        consumption = 0.45 * signal / max(liquidity, 0.4)
        buffer = -0.20 * signal
        borrowing = 0.20 * signal
        job = -0.25 * signal
        portfolio = -0.05 * signal
    elif variable == "UNEMP":
        consumption = -0.55 * liquidity * signal
        buffer = 0.70 * liquidity * signal
        borrowing = -0.40 * liquidity * signal
        job = 0.75 * max(signal, 0.0)
        portfolio = 0.20 * liquidity * signal
    elif variable in {"TBILL", "TBOND"}:
        consumption = -0.20 * rate * max(signal, 0.0)
        buffer = 0.35 * rate * signal
        borrowing = -0.55 * rate * signal
        job = 0.0
        portfolio = 0.45 * rate * signal
    else:
        consumption = -0.10 * signal
        buffer = 0.10 * signal
        borrowing = -0.10 * signal
        job = 0.0
        portfolio = 0.0
    return {
        "consumption_change_pct": float(np.clip(consumption, -8.0, 8.0)),
        "desired_liquid_buffer_change_pct": float(np.clip(buffer, -10.0, 10.0)),
        "borrowing_desire_index": float(np.clip(borrowing, -3.0, 3.0)),
        "job_search_intensity_index": float(np.clip(job, -2.0, 4.0)),
        "portfolio_rebalance_to_liquid_pct": float(np.clip(portfolio, -8.0, 8.0)),
    }


def _standardized_signal(card: ForecastCard, point_forecast: float) -> float:
    scale = max(float(card.recent_signal_volatility_8 or 0.0), 0.25)
    return float(np.clip((point_forecast - float(card.rolling_signal_mean_4)) / scale, -4.0, 4.0))


def _weighted_panel_average(group: pd.DataFrame) -> pd.Series:
    weights = group["population_weight"].astype(float)
    total = float(weights.sum())
    if total <= 0:
        weights = pd.Series(np.ones(len(group)) / max(1, len(group)), index=group.index)
        total = 1.0
    out = {"schema_version": FORECAST_AGENT_PANEL_VERSION, "household_type_count": int(group.shape[0])}
    for column in [
        "belief_point_forecast",
        "belief_signal_vs_history",
        "consumption_change_pct",
        "desired_liquid_buffer_change_pct",
        "borrowing_desire_index",
        "job_search_intensity_index",
        "portfolio_rebalance_to_liquid_pct",
        "uncertainty_index",
    ]:
        out[column] = float((group[column].astype(float) * weights).sum() / total)
    return pd.Series(out)
