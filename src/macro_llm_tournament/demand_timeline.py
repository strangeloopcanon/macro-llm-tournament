"""Frequency-aware recursive timeline helpers for the demand economy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


NEUTRAL_POLICY_RATE = 3.0
INFLATION_TARGET = 2.0
STEADY_EMPLOYMENT_RATE = 0.955
DEFAULT_PERIODS_PER_YEAR = 4.0
PROTECTED_PERIOD_OVERRIDE_FIELDS = frozenset({"scenario_id", "period_id", "period_index"})


def validate_timeline_scenario(scenario: Any) -> None:
    if not 0.0 <= float(scenario.policy_rate_smoothing) <= 1.0:
        raise ValueError("policy_rate_smoothing must be between 0 and 1")
    if scenario.policy_state_mode not in {"recursive", "origin_visible"}:
        raise ValueError(f"Unsupported policy_state_mode: {scenario.policy_state_mode}")
    if not 0.0 <= float(scenario.policy_state_weight) <= 1.0:
        raise ValueError("policy_state_weight must be between 0 and 1")


def _validated_periods_per_year(periods_per_year: float) -> float:
    value = float(periods_per_year)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("periods_per_year must be a positive finite float")
    return value


def _months_per_period(periods_per_year: float) -> float:
    return 12.0 / _validated_periods_per_year(periods_per_year)


def _per_period_amount(annual_amount: float, periods_per_year: float) -> float:
    return float(annual_amount) / _validated_periods_per_year(periods_per_year)


def _quarterly_persistence_for_frequency(quarterly_persistence: float, periods_per_year: float) -> float:
    persistence = float(quarterly_persistence)
    if not 0.0 <= persistence < 1.0:
        raise ValueError("quarterly_persistence must be in [0, 1)")
    frequency = _validated_periods_per_year(periods_per_year)
    if np.isclose(frequency, DEFAULT_PERIODS_PER_YEAR):
        return persistence
    return float(persistence ** (DEFAULT_PERIODS_PER_YEAR / frequency))


def _quarterly_flow_for_frequency(
    quarterly_flow: float,
    quarterly_persistence: float,
    periods_per_year: float,
) -> float:
    persistence = float(quarterly_persistence)
    if not 0.0 <= persistence < 1.0:
        raise ValueError("quarterly_persistence must be in [0, 1)")
    frequency_persistence = _quarterly_persistence_for_frequency(persistence, periods_per_year)
    return float(quarterly_flow) * (1.0 - frequency_persistence) / (1.0 - persistence)


def _normalize_period_overrides(period_overrides: dict[int, dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if period_overrides is None:
        return {}
    if not isinstance(period_overrides, dict):
        raise ValueError("period_overrides must be a dict keyed by period index")
    out: dict[int, dict[str, Any]] = {}
    for raw_index, raw_override in period_overrides.items():
        try:
            period_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"period_overrides key must be an integer period index, got {raw_index!r}") from exc
        if period_index < 0:
            raise ValueError(f"period_overrides period index must be non-negative, got {period_index}")
        if not isinstance(raw_override, dict):
            raise ValueError(f"period_overrides[{period_index}] must be a dict of exogenous fields")
        forbidden = sorted(PROTECTED_PERIOD_OVERRIDE_FIELDS.intersection(raw_override))
        if forbidden:
            raise ValueError(
                "period_overrides cannot override protected period identity fields: " + ", ".join(forbidden)
            )
        override = dict(raw_override)
        _assert_finite_override_values(override, path=f"period_overrides[{period_index}]")
        out[period_index] = override
    return out


def _normalize_initial_environment_override(value: dict[str, Any] | None) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("initial_environment_override must be a dict")
    allowed = {
        "output_gap_pct",
        "employment_rate",
        "inflation_rate",
        "policy_rate",
        "aggregate_job_loss_belief",
        "aggregate_confidence_index",
        "aggregate_liquid_buffer_months",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("initial_environment_override contains unsupported fields: " + ", ".join(unknown))
    _assert_finite_override_values(value, path="initial_environment_override")
    out = {key: float(raw) for key, raw in value.items()}
    if "employment_rate" in out and not 0.0 <= out["employment_rate"] <= 1.0:
        raise ValueError("initial_environment_override employment_rate must be between zero and one")
    return out


def _assert_finite_override_values(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_finite_override_values(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_finite_override_values(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise ValueError(f"{path} contains a nonfinite numeric value")


def _buffer_months(liquid_assets: float, period_consumption: float, *, periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
    monthly_consumption = max(float(period_consumption) / _months_per_period(periods_per_year), 1e-9)
    return float(max(0.0, float(liquid_assets)) / monthly_consumption)


def _initial_environment(
    households: pd.DataFrame,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, float]:
    baseline_consumption = float((households["population_weight"] * households["baseline_consumption_annual"] / periods_per_year).sum())
    return {
        "periods_per_year": periods_per_year,
        "baseline_aggregate_consumption": baseline_consumption,
        "aggregate_consumption": baseline_consumption,
        "output_gap_pct": 0.0,
        "employment_rate": STEADY_EMPLOYMENT_RATE,
        "inflation_rate": INFLATION_TARGET,
        "policy_rate": NEUTRAL_POLICY_RATE,
        "aggregate_job_loss_belief": float((households["population_weight"] * households["baseline_job_loss_probability"]).sum()),
        "aggregate_confidence_index": float((households["population_weight"] * households["confidence_index"]).sum()),
        "aggregate_liquid_buffer_months": 3.0,
    }


def _period_state(
    env: dict[str, float],
    scenario: Any,
    period_index: int,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    period_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rate_shock = (
        float(scenario.rate_shock_pp)
        if scenario.rate_shock_start <= period_index <= scenario.rate_shock_end and scenario.rate_shock_start >= 0
        else 0.0
    )
    job_risk_shock = (
        float(scenario.job_risk_shock_pp)
        if scenario.job_risk_shock_start <= period_index <= scenario.job_risk_shock_end and scenario.job_risk_shock_start >= 0
        else 0.0
    )
    transfer = float(scenario.transfer_amount) if period_index == scenario.transfer_period else 0.0
    state: dict[str, Any] = {
        **env,
        "scenario_id": scenario.scenario_id,
        "period_index": int(period_index),
        "period_id": f"period_{period_index}",
        "periods_per_year": periods_per_year,
        "months_per_period": _months_per_period(periods_per_year),
        "transfer_per_household": transfer,
        "policy_rate_shock_pp": rate_shock,
        "job_risk_shock_pp": job_risk_shock,
        "policy_rate": float(env["policy_rate"]) + rate_shock,
    }
    if period_override:
        public_override = {key: value for key, value in period_override.items() if key != "origin_visible_state_assimilation"}
        state.update(public_override)
        state["supplied_exogenous_conditions"] = public_override
    else:
        state["supplied_exogenous_conditions"] = {}
    return state


def _environment_for_period(
    env: dict[str, float],
    scenario: Any,
    *,
    period_override: dict[str, Any] | None,
) -> dict[str, float]:
    if scenario.policy_state_mode == "recursive":
        return env
    assimilation = period_override.get("origin_visible_state_assimilation") if period_override else None
    policy = assimilation.get("policy_rate") if isinstance(assimilation, dict) else None
    value = policy.get("value") if isinstance(policy, dict) else None
    try:
        policy_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("origin_visible policy state requires a finite policy_rate value") from exc
    if not math.isfinite(policy_rate) or not 0.0 <= policy_rate <= 30.0:
        raise ValueError("origin_visible policy state requires a finite policy_rate value")
    weight = float(scenario.policy_state_weight)
    return {
        **env,
        "policy_rate": weight * policy_rate + (1.0 - weight) * float(env["policy_rate"]),
    }


def _next_environment(
    env: dict[str, float],
    aggregate: dict[str, Any],
    scenario: Any,
    *,
    feedback_mode: str,
) -> dict[str, float]:
    if feedback_mode == "none":
        return {
            **env,
            "aggregate_consumption": float(aggregate["aggregate_consumption"]),
            "aggregate_job_loss_belief": float(aggregate["aggregate_job_loss_belief"]),
            "aggregate_confidence_index": float(aggregate["aggregate_confidence_index"]),
            "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
        }
    periods_per_year = _validated_periods_per_year(env.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR))
    output_gap = float(aggregate["output_gap_pct"])
    gain = float(scenario.feedback_gain)
    employment_persistence = _quarterly_persistence_for_frequency(0.82, periods_per_year)
    inflation_persistence = _quarterly_persistence_for_frequency(0.64, periods_per_year)
    inflation_output_feedback = _quarterly_flow_for_frequency(0.024, 0.64, periods_per_year)
    next_employment = float(
        np.clip(
            employment_persistence * float(env["employment_rate"])
            + (1.0 - employment_persistence) * (STEADY_EMPLOYMENT_RATE + 0.0010 * gain * output_gap),
            0.82,
            0.99,
        )
    )
    next_inflation = float(
        np.clip(
            inflation_persistence * float(env["inflation_rate"])
            + (1.0 - inflation_persistence) * INFLATION_TARGET
            + inflation_output_feedback * gain * output_gap,
            -2.0,
            12.0,
        )
    )
    policy_target = NEUTRAL_POLICY_RATE + 1.35 * (next_inflation - INFLATION_TARGET) + 0.18 * output_gap
    smoothing = float(scenario.policy_rate_smoothing)
    next_policy = float(
        np.clip(
            smoothing * float(env["policy_rate"]) + (1.0 - smoothing) * policy_target,
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
        "aggregate_confidence_index": float(aggregate["aggregate_confidence_index"]),
        "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
    }
