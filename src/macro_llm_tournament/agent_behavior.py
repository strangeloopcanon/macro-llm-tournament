from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .agent_common import finite_float, round_or_none, weighted_sum
from .forecast_cards import ForecastCard


def response_by_variable(
    variable: str,
    signal: float,
    *,
    liquidity: float,
    rate: float,
    unemployment: float,
    portfolio: float,
    uncertainty: float,
) -> dict[str, float]:
    caution = 1.0 + 0.12 * max(uncertainty - 1.0, 0.0)
    if variable == "CPI":
        consumption = -0.45 * liquidity * max(signal, -1.5) * caution
        buffer = 0.70 * liquidity * signal * caution
        borrowing = -0.30 * liquidity * max(signal, 0.0)
        job = 0.06 * unemployment * max(signal, 0.0)
        portfolio_move = 0.14 * portfolio * signal
    elif variable == "RGDP":
        consumption = 0.50 * signal / max(liquidity, 0.4)
        buffer = -0.25 * signal
        borrowing = 0.24 * signal
        job = -0.30 * unemployment * signal
        portfolio_move = -0.08 * portfolio * signal
    elif variable == "UNEMP":
        consumption = -0.65 * liquidity * unemployment * signal * caution
        buffer = 0.85 * liquidity * signal * caution
        borrowing = -0.48 * liquidity * signal
        job = 0.95 * unemployment * max(signal, 0.0)
        portfolio_move = 0.24 * portfolio * signal
    elif variable in {"TBILL", "TBOND"}:
        consumption = -0.26 * rate * max(signal, 0.0)
        buffer = 0.40 * rate * signal
        borrowing = -0.70 * rate * signal
        job = 0.02 * max(signal, 0.0)
        portfolio_move = 0.58 * portfolio * rate * signal
    else:
        consumption = -0.10 * signal
        buffer = 0.10 * signal
        borrowing = -0.10 * signal
        job = 0.0
        portfolio_move = 0.0
    return {
        "consumption_change_pct": float(np.clip(consumption, -15.0, 15.0)),
        "desired_liquid_buffer_change_pct": float(np.clip(buffer, -20.0, 20.0)),
        "borrowing_desire_index": float(np.clip(borrowing, -5.0, 5.0)),
        "job_search_intensity_index": float(np.clip(job, -3.0, 6.0)),
        "portfolio_rebalance_to_liquid_pct": float(np.clip(portfolio_move, -12.0, 12.0)),
    }


def firm_hiring_index(card: ForecastCard, belief_signal: float, aggregate_consumption_change: float) -> float:
    unemployment_pressure = max(0.0, belief_signal if card.variable == "UNEMP" else 0.0)
    return float(np.clip(0.35 * aggregate_consumption_change - 0.50 * unemployment_pressure, -5.0, 5.0))


def firm_price_pressure_index(card: ForecastCard, belief_signal: float, aggregate_consumption_change: float) -> float:
    inflation_pressure = max(0.0, belief_signal if card.variable == "CPI" else 0.0)
    rate_pressure = max(0.0, belief_signal if card.variable in {"TBILL", "TBOND"} else 0.0)
    return float(np.clip(0.45 * inflation_pressure + 0.08 * aggregate_consumption_change + 0.10 * rate_pressure, -5.0, 5.0))


def updated_belief(prior_state: dict[str, Any], variable: str, target_variable: str, field: str, forecast: pd.Series) -> float:
    prior = float(prior_state[field])
    if variable != target_variable:
        return prior
    return float(0.35 * prior + 0.65 * float(forecast["point_forecast"]))


def bank_credit_multiplier(card: ForecastCard, desired_rows: list[dict[str, Any]]) -> float:
    signal = weighted_sum(desired_rows, "belief_signal_vs_history")
    if card.variable in {"TBILL", "TBOND"}:
        multiplier = 1.0 - 0.18 * max(signal, 0.0)
    elif card.variable == "UNEMP":
        multiplier = 1.0 - 0.15 * max(signal, 0.0)
    elif card.variable == "RGDP":
        multiplier = 1.0 + 0.08 * max(signal, 0.0)
    else:
        multiplier = 1.0 - 0.04 * max(signal, 0.0)
    return float(np.clip(multiplier, 0.35, 1.20))


def forecast_uncertainty(forecast: pd.Series, signal: float) -> float:
    width = float(forecast.get("p90", np.nan)) - float(forecast.get("p10", np.nan))
    if not np.isfinite(width):
        width = abs(signal)
    return float(np.clip(width + 0.25 * abs(signal), 0.05, 12.0))


def standardized_signal(card: ForecastCard, point_forecast: float) -> float:
    scale = max(float(card.recent_signal_volatility_8 or 0.0), 0.25)
    return float(np.clip((point_forecast - float(card.rolling_signal_mean_4)) / scale, -4.0, 4.0))


def forecast_for_prompt(forecast: pd.Series) -> dict[str, Any]:
    return {
        "source": str(forecast["source"]),
        "variable": str(forecast["variable"]),
        "origin": str(forecast["origin"]),
        "horizon": int(forecast["horizon"]),
        "point_forecast": round_or_none(forecast["point_forecast"]),
        "p10": round_or_none(forecast.get("p10")),
        "p50": round_or_none(forecast.get("p50")),
        "p90": round_or_none(forecast.get("p90")),
        "confidence": round_or_none(forecast.get("confidence")),
        "panel_mean": round_or_none(forecast.get("panel_mean")),
        "panel_std": round_or_none(forecast.get("panel_std")),
    }


def sector_value(sector_response: dict[str, Any] | None, sector: str, field: str) -> float:
    if sector_response is None:
        return float("nan")
    section = sector_response.get(sector)
    if not isinstance(section, dict):
        return float("nan")
    return finite_float(section.get(field), default=np.nan)


def first_finite(rows: list[dict[str, Any]], column: str) -> float:
    for row in rows:
        value = finite_float(row.get(column), default=np.nan)
        if np.isfinite(value):
            return float(value)
    return float("nan")
