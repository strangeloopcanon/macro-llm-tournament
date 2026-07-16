"""Standardized monthly macro predictions for the household ecology.

This module is deliberately pure: it turns ecology-level household and firm
signals plus an origin-safe macro information card into one scoreable monthly
prediction record.  It does not read realized macro data or fill missing
visible baselines with defaults.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


MACRO_PREDICTION_SCHEMA_VERSION = "household_ecology_standardized_macro_predictions_v2"
FIRM_PLAN_VERSION = "contemporaneous_demand_inventory_firm_plan_v2"

# Kept exactly aligned with the transparent observability shadow calculation.
TARGET_INVENTORY_SHARE = 0.08
INVENTORY_ADJUSTMENT_SPEED = 0.35
EMPLOYMENT_ADJUSTMENT_SPEED = 0.25
PRICE_PRESSURE_DEMAND_WEIGHT = 0.15
PRICE_PRESSURE_INVENTORY_WEIGHT = 0.10


# ``visible_baseline_field`` is the flat prediction-record field emitted for a
# target. ``visible_baseline_source`` documents its origin-safe card location.
MACRO_TARGET_CONTRACT: dict[str, dict[str, str]] = {
    "consumption_growth_pct": {
        "target_name": "pce_growth_pct",
        "series_id": "PCE",
        "mapping_quality": "closest_aggregate_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "Nominal consumption (PCE)",
        "note": "Executed household spending growth is the closest aggregate model output to nominal PCE growth.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_consumption_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.PCE.changes.1m.value",
    },
    "real_consumption_growth_pct": {
        "target_name": "real_pce_growth_pct",
        "series_id": "PCEC96",
        "mapping_quality": "mechanical_deflation_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "Real consumption (PCEC96)",
        "note": "Nominal household spending growth is deflated by the household-belief inflation prediction.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_real_consumption_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.PCEC96.changes.1m.value",
    },
    "price_growth_pct": {
        "target_name": "pce_price_growth_pct",
        "series_id": "PCEPI",
        "mapping_quality": "mechanical_household_expectation_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "PCE price index (PCEPI)",
        "note": "The population-weighted 12-month household inflation belief is compounded down to one month.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_price_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.PCEPI.changes.1m.value",
    },
    "real_disposable_income_growth_pct": {
        "target_name": "real_disposable_income_growth_pct",
        "series_id": "DSPIC96",
        "mapping_quality": "mechanical_deflation_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "Real disposable income (DSPIC96)",
        "note": "The population-weighted 12-month household nominal-income belief is compounded to one month and deflated by predicted inflation.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_real_disposable_income_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.DSPIC96.changes.1m.value",
    },
    "payroll_growth_pct": {
        "target_name": "payroll_growth_pct",
        "series_id": "PAYEMS",
        "mapping_quality": "mechanical_firm_feedback_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "Payroll employment (PAYEMS)",
        "note": "The target-month firm plan maps predicted household demand and origin-visible inventories into a bounded planned-employment change.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_payroll_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.PAYEMS.changes.1m.value",
    },
    "unemployment_rate_level": {
        "target_name": "unemployment_rate_level",
        "series_id": "UNRATE",
        "mapping_quality": "mechanical_firm_feedback_proxy",
        "score_mode": "full",
        "direction_mode": "change_from_visible_baseline",
        "unit": "percent",
        "unit_label": "Unemployment rate (%)",
        "chart_title": "Unemployment rate (UNRATE)",
        "note": "The origin-visible unemployment rate is advanced using the target-month planned payroll change with a fixed labor force.",
        "visible_baseline_kind": "latest_level",
        "visible_baseline_field": "visible_baseline_unemployment_rate_level",
        "visible_baseline_source": "compact_macro_information.series.UNRATE.latest_value",
    },
    "personal_saving_rate_change_pp": {
        "target_name": "personal_saving_rate_change",
        "series_id": "PSAVERT",
        "mapping_quality": "household_budget_residual_proxy",
        "score_mode": "full",
        "unit": "percentage_points",
        "unit_label": "Monthly change (pp)",
        "chart_title": "Personal saving rate (PSAVERT)",
        "note": "The model value is the change in a gross household-income budget residual rate, not a national-accounts saving identity.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_personal_saving_rate_change_pp",
        "visible_baseline_source": "compact_macro_information.series.PSAVERT.changes.1m.value",
    },
    "revolving_credit_growth_pct": {
        "target_name": "revolving_credit_growth_pct",
        "series_id": "REVOLSL",
        "mapping_quality": "directional_proxy",
        "score_mode": "direction_only",
        "unit": "percent",
        "unit_label": "Direction of monthly growth",
        "chart_title": "Revolving credit (REVOLSL)",
        "note": "The simulated revolving-debt stock is compared with aggregate revolving credit on direction only.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_revolving_credit_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.REVOLSL.changes.1m.value",
    },
    "retail_sales_growth_pct": {
        "target_name": "retail_sales_growth_pct",
        "series_id": "RSAFS",
        "mapping_quality": "declared_demand_proxy",
        "score_mode": "full",
        "unit": "percent",
        "unit_label": "Monthly growth (%)",
        "chart_title": "Retail sales (RSAFS)",
        "note": "Executed household spending growth is reused as a declared retail-demand proxy.",
        "visible_baseline_kind": "1m_change",
        "visible_baseline_field": "visible_baseline_retail_sales_growth_pct",
        "visible_baseline_source": "compact_macro_information.series.RSAFS.changes.1m.value",
    },
}

# Backward-compatible read-only alias for callers that imported the worker's
# provisional name while the full contract was being integrated.
STANDARDIZED_MACRO_TARGETS = MACRO_TARGET_CONTRACT


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def annual_to_monthly_growth(annual_growth_pct: float) -> float:
    """Convert a compound annual percentage growth expectation to one month."""

    annual_growth = _require_finite(annual_growth_pct, "annual_growth_pct")
    if annual_growth < -100.0:
        raise ValueError("annual_growth_pct must be at least -100 percent")
    return 100.0 * ((1.0 + annual_growth / 100.0) ** (1.0 / 12.0) - 1.0)


def fisher_real_growth(nominal_growth_pct: float, price_growth_pct: float) -> float:
    """Return monthly real growth using the exact Fisher-style price deflator."""

    nominal_growth = _require_finite(nominal_growth_pct, "nominal_growth_pct")
    price_growth = _require_finite(price_growth_pct, "price_growth_pct")
    if nominal_growth < -100.0:
        raise ValueError("nominal_growth_pct must be at least -100 percent")
    if price_growth <= -100.0:
        raise ValueError("price_growth_pct must be greater than -100 percent")
    return 100.0 * ((1.0 + nominal_growth / 100.0) / (1.0 + price_growth / 100.0) - 1.0)


def unemployment_level_from_payroll_growth(
    unemployment_rate_level: float, payroll_growth_pct: float
) -> float:
    """Apply payroll growth to employment while holding the labor force fixed.

    The labor force is normalized to 100 people.  The visible unemployment rate
    determines the starting employment count, so no hidden labor-force series
    is introduced at this boundary.
    """

    unemployment_rate = _require_finite(unemployment_rate_level, "unemployment_rate_level")
    payroll_growth = _require_finite(payroll_growth_pct, "payroll_growth_pct")
    if not 0.0 <= unemployment_rate <= 100.0:
        raise ValueError("unemployment_rate_level must be between 0 and 100 percent")
    if payroll_growth < -100.0:
        raise ValueError("payroll_growth_pct must be at least -100 percent")
    labor_force = 100.0
    employed = labor_force - unemployment_rate
    next_employed = float(np.clip(employed * (1.0 + payroll_growth / 100.0), 0.0, 100.0))
    return 100.0 * (labor_force - next_employed) / labor_force


def visible_baseline_from_card(
    card: Mapping[str, Any], series_id: str, kind: str
) -> float:
    """Read a required origin-visible baseline from a compact macro card."""

    if not isinstance(card, Mapping):
        raise ValueError("compact_macro_information card must be a mapping")
    if kind not in {"1m_change", "latest_level"}:
        raise ValueError("baseline kind must be 1m_change or latest_level")
    series = card.get("series")
    if not isinstance(series, Mapping):
        raise ValueError("compact_macro_information card lacks series mapping")
    entry = series.get(series_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"compact_macro_information card lacks {series_id}")
    if entry.get("available") is not True:
        raise ValueError(f"compact_macro_information {series_id} is unavailable")
    if kind == "latest_level":
        return _require_finite(entry.get("latest_value"), f"{series_id}.latest_value")
    changes = entry.get("changes")
    if not isinstance(changes, Mapping):
        raise ValueError(f"compact_macro_information {series_id} lacks changes")
    one_month = changes.get("1m")
    if not isinstance(one_month, Mapping):
        raise ValueError(f"compact_macro_information {series_id} lacks 1m change")
    return _require_finite(one_month.get("value"), f"{series_id}.changes.1m.value")


def contemporaneous_firm_plan(
    *,
    consumption_growth_pct: float,
    inventory_start_units: float,
    baseline_sales_units: float,
) -> dict[str, float]:
    """Plan target-month production and employment from origin-safe state.

    The predicted target-month household demand change is combined with the
    inventory stock visible at the forecast origin. No target-month realized or
    settled inventory enters this calculation.
    """

    demand_growth = _require_finite(consumption_growth_pct, "consumption_growth_pct")
    inventory = _require_finite(inventory_start_units, "inventory_start_units")
    sales = _require_finite(baseline_sales_units, "baseline_sales_units")
    if inventory < 0.0:
        raise ValueError("inventory_start_units must be nonnegative")
    if sales <= 0.0:
        raise ValueError("baseline_sales_units must be positive")
    inventory_share = inventory / sales
    inventory_gap_pp = 100.0 * (TARGET_INVENTORY_SHARE - inventory_share)
    output_growth = float(np.clip(
        demand_growth + INVENTORY_ADJUSTMENT_SPEED * inventory_gap_pp,
        -10.0,
        10.0,
    ))
    required_labor_growth = output_growth
    employment_growth = EMPLOYMENT_ADJUSTMENT_SPEED * required_labor_growth
    price_pressure = (
        PRICE_PRESSURE_DEMAND_WEIGHT * demand_growth
        + PRICE_PRESSURE_INVENTORY_WEIGHT * inventory_gap_pp
    )
    return {
        "firm_expected_sales_index": 100.0 + demand_growth,
        "firm_target_output_index": 100.0 + output_growth,
        "firm_required_labor_index": 100.0 + required_labor_growth,
        "firm_planned_employment_index": 100.0 + employment_growth,
        "firm_price_pressure_pp": price_pressure,
        "firm_inventory_share_pct": 100.0 * inventory_share,
        "firm_inventory_gap_pp": inventory_gap_pp,
    }


def build_standardized_macro_predictions(
    *,
    nominal_consumption_growth_pct: float,
    annual_nominal_household_income_expectation_pct: float,
    price_growth_pct: float,
    personal_saving_rate_change_pp: float,
    revolving_credit_growth_pct: float,
    inventory_start_units: float,
    baseline_sales_units: float,
    compact_macro_information: Mapping[str, Any],
) -> dict[str, float]:
    """Build all standardized household-ecology monthly macro predictions.

    Nominal consumption is the declared household-demand signal.  Retail sales
    deliberately reuses it as a demand proxy.  Annual household income is
    compounded down to one month before Fisher deflation. Payroll comes from a
    target-month firm plan formed from predicted demand and origin inventory;
    unemployment applies that payroll change to the origin-visible rate under
    a fixed labor-force identity.
    """

    consumption_growth = _require_finite(
        nominal_consumption_growth_pct, "nominal_consumption_growth_pct"
    )
    price_growth = _require_finite(price_growth_pct, "price_growth_pct")
    saving_rate_change = _require_finite(
        personal_saving_rate_change_pp, "personal_saving_rate_change_pp"
    )
    revolving_growth = _require_finite(
        revolving_credit_growth_pct, "revolving_credit_growth_pct"
    )
    annual_income_growth = _require_finite(
        annual_nominal_household_income_expectation_pct,
        "annual_nominal_household_income_expectation_pct",
    )

    baselines = {
        metadata["visible_baseline_field"]: visible_baseline_from_card(
            compact_macro_information,
            metadata["series_id"],
            metadata["visible_baseline_kind"],
        )
        for metadata in MACRO_TARGET_CONTRACT.values()
    }
    firm_plan = contemporaneous_firm_plan(
        consumption_growth_pct=consumption_growth,
        inventory_start_units=inventory_start_units,
        baseline_sales_units=baseline_sales_units,
    )
    payroll_growth = firm_plan["firm_planned_employment_index"] - 100.0
    monthly_income_growth = annual_to_monthly_growth(annual_income_growth)
    predictions = {
        "consumption_growth_pct": consumption_growth,
        "real_consumption_growth_pct": fisher_real_growth(consumption_growth, price_growth),
        "price_growth_pct": price_growth,
        "real_disposable_income_growth_pct": fisher_real_growth(
            monthly_income_growth, price_growth
        ),
        "payroll_growth_pct": payroll_growth,
        "unemployment_rate_level": unemployment_level_from_payroll_growth(
            baselines["visible_baseline_unemployment_rate_level"], payroll_growth
        ),
        "personal_saving_rate_change_pp": saving_rate_change,
        "revolving_credit_growth_pct": revolving_growth,
        "retail_sales_growth_pct": consumption_growth,
    }
    result = predictions | baselines | firm_plan
    for field, value in result.items():
        _require_finite(value, field)
    return result
