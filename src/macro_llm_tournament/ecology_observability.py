"""Build a full diagnostic surface from banked household-ecology runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ecology import _artifact_sha256, _file_sha256, _write_json


SCHEMA_VERSION = "household_ecology_observability_v2"
FIRM_SHADOW_VERSION = "demand_inventory_firm_shadow_v1"
FEEDBACK_SCHEMA_VERSION = "household_ecology_two_period_feedback_v3"
TARGET_INVENTORY_SHARE = 0.08
INVENTORY_ADJUSTMENT_SPEED = 0.35
EMPLOYMENT_ADJUSTMENT_SPEED = 0.25
PRICE_PRESSURE_DEMAND_WEIGHT = 0.15
PRICE_PRESSURE_INVENTORY_WEIGHT = 0.10
LLM_HOUSEHOLD_ECONOMY_SETTLEMENT_LABEL = (
    "LLM household economy - code-enforced budgets and settlement"
)
LLM_HOUSEHOLD_ECONOMY_CHART_LABEL = "LLM household economy"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrospective-run", type=Path, required=True)
    parser.add_argument("--prospective-run", type=Path)
    parser.add_argument("--feedback-run", type=Path)
    parser.add_argument("--households", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_artifact(run_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    path = run_dir / name
    expected = manifest.get("artifacts", {}).get(name)
    if not isinstance(expected, str):
        raise ValueError(f"manifest does not bind required artifact: {name}")
    actual = _artifact_sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash mismatch for {path}: {actual} != {expected}")
    return path


def _load_weights(path: Path, expected_count: int) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"type_id", "population_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"household file missing fields: {', '.join(sorted(missing))}")
    if frame["type_id"].astype(str).duplicated().any():
        raise ValueError("household IDs must be unique")
    weights = {
        str(row["type_id"]): _require_finite(row["population_weight"], "population_weight")
        for _, row in frame.iterrows()
    }
    if len(weights) != expected_count or any(value <= 0.0 for value in weights.values()):
        raise ValueError("household weights must be positive and cover the run exactly")
    return weights


def _weighted_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    if set(values) != set(weights):
        missing = sorted(set(weights).difference(values))
        extra = sorted(set(values).difference(weights))
        raise ValueError(f"weighted values do not match households; missing={missing[:3]} extra={extra[:3]}")
    total = sum(weights.values())
    return sum(values[key] * weights[key] for key in weights) / total


def firm_response_shadow(
    *, consumption_growth_pct: float, inventory_end_units: float, units_sold: float
) -> dict[str, float]:
    """Project a transparent next-period firm response without changing the economy.

    Sales inherit household-demand growth. Output also closes 35 percent of the
    inventory gap. Fixed productivity maps output one-for-one into required labor;
    employment closes one quarter of that requirement. Price pressure is an index,
    not a price forecast.
    """

    demand_growth = _require_finite(consumption_growth_pct, "consumption_growth_pct")
    inventory = _require_finite(inventory_end_units, "inventory_end_units")
    sales = _require_finite(units_sold, "units_sold")
    if inventory < 0.0:
        raise ValueError("inventory_end_units must be nonnegative")
    if sales <= 0.0:
        raise ValueError("units_sold must be positive")
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


def _evaluation_status(manifest: dict[str, Any]) -> str:
    status = str(manifest.get("evaluation_status", ""))
    if status not in {"retrospective", "prospective_frozen"}:
        raise ValueError(f"unsupported child evaluation status: {status}")
    return status


def _employment_policy_name(card: dict[str, Any]) -> str:
    employed = card["household"]["current_state"].get("employed")
    if type(employed) is not bool:
        raise ValueError("household card employed field must be a JSON boolean")
    return "employed_policy" if employed else "not_employed_policy"


def _run_payload(run_dir: Path, weights: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(run_dir / "manifest.json")
    if not manifest.get("accounting_passed"):
        raise ValueError(f"source run did not pass accounting: {run_dir}")
    if int(manifest.get("household_count", -1)) != len(weights):
        raise ValueError(f"source run household count does not match weights: {run_dir}")
    for name in (
        "household_cards.json",
        "household_responses.json",
        "household_decisions.csv",
        "macro_forecast_paths.csv",
        "median_economy.json",
    ):
        _validate_artifact(run_dir, manifest, name)

    cards = _read_json(run_dir / "household_cards.json")
    records = _read_json(run_dir / "household_responses.json")
    decisions = pd.read_csv(run_dir / "household_decisions.csv")
    macro = pd.read_csv(run_dir / "macro_forecast_paths.csv")
    economy = _read_json(run_dir / "median_economy.json")
    if len(cards) != len(weights) or len(records) != len(weights) or len(decisions) != len(weights):
        raise ValueError(f"source run does not contain one row per household: {run_dir}")
    if len(macro) != 1 or str(macro.iloc[0].get("scenario")) != "median":
        raise ValueError(f"source run must contain one median macro path: {run_dir}")

    payload_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload", {})
        household_id = str(payload.get("household_id", ""))
        if not household_id or household_id in payload_by_id:
            raise ValueError(f"invalid household response identities: {run_dir}")
        payload_by_id[household_id] = payload
    card_by_id = {str(card["household"]["household_id"]): card for card in cards}
    if set(payload_by_id) != set(weights) or set(card_by_id) != set(weights):
        raise ValueError(f"source run household identities do not match weights: {run_dir}")

    target_month = str(manifest["target_month"])
    origin_month = str(manifest["origin_month"])
    status = _evaluation_status(manifest)
    common = {
        "origin_month": origin_month,
        "target_month": target_month,
        "evaluation_status": status,
    }
    rows: list[dict[str, Any]] = []

    def add(layer: str, metric: str, value: float, unit: str, source_class: str, note: str) -> None:
        rows.append({
            **common,
            "layer": layer,
            "metric": metric,
            "value": _require_finite(value, metric),
            "unit": unit,
            "source_class": source_class,
            "interpretation": note,
        })

    for field, label in (
        ("expected_inflation_pct", "Expected inflation"),
        ("expected_income_growth_pct", "Expected income growth"),
        ("job_loss_probability_pct", "Personal job-loss probability"),
    ):
        values = {
            household_id: _require_finite(payload[field]["p50"], field)
            for household_id, payload in payload_by_id.items()
        }
        add("beliefs", field, _weighted_mean(values, weights), "percent", "llm_household_intention", label)

    policy_fields = (
        "committed_consumption_change_usd",
        "discretionary_consumption_change_usd",
        "one_off_purchase_usd",
        "extra_debt_payment_usd",
        "borrowing_intent_usd",
    )
    policy_means: dict[str, float] = {}
    for field in policy_fields:
        values: dict[str, float] = {}
        for household_id, payload in payload_by_id.items():
            branch = payload[_employment_policy_name(card_by_id[household_id])]
            values[household_id] = _require_finite(branch[field], field)
        policy_means[field] = _weighted_mean(values, weights)
        add(
            "intended_policy",
            field,
            policy_means[field],
            "usd_per_represented_household",
            "llm_household_intention",
            "Population-weighted household intention on the origin employment branch.",
        )
    baseline_consumption: dict[str, float] = {}
    intended_consumption: dict[str, float] = {}
    intended_committed: dict[str, float] = {}
    intended_discretionary: dict[str, float] = {}
    for household_id in weights:
        state = card_by_id[household_id]["household"]["current_state"]
        baseline_total = _require_finite(state["monthly_consumption"], "monthly_consumption")
        baseline_committed = _require_finite(
            state.get("current_month_committed_consumption", baseline_total * 0.65),
            "current_month_committed_consumption",
        )
        baseline_discretionary = _require_finite(
            state.get(
                "current_month_discretionary_consumption",
                max(0.0, baseline_total - baseline_committed),
            ),
            "current_month_discretionary_consumption",
        )
        branch = payload_by_id[household_id][
            _employment_policy_name(card_by_id[household_id])
        ]
        committed = max(
            0.0,
            baseline_committed
            + _require_finite(
                branch["committed_consumption_change_usd"],
                "committed_consumption_change_usd",
            ),
        )
        discretionary = max(
            0.0,
            baseline_discretionary
            + _require_finite(
                branch["discretionary_consumption_change_usd"],
                "discretionary_consumption_change_usd",
            ),
        )
        one_off = _require_finite(branch["one_off_purchase_usd"], "one_off_purchase_usd")
        baseline_consumption[household_id] = baseline_total
        intended_committed[household_id] = committed
        intended_discretionary[household_id] = discretionary
        intended_consumption[household_id] = committed + discretionary + one_off

    baseline_consumption_mean = _weighted_mean(baseline_consumption, weights)
    intended_consumption_mean = _weighted_mean(intended_consumption, weights)
    add(
        "intended_policy",
        "intended_committed_consumption_usd",
        _weighted_mean(intended_committed, weights),
        "usd_per_represented_household",
        "llm_household_intention",
        "Population-weighted intended committed spending after applying the signed household change.",
    )
    add(
        "intended_policy",
        "intended_discretionary_consumption_usd",
        _weighted_mean(intended_discretionary, weights),
        "usd_per_represented_household",
        "llm_household_intention",
        "Population-weighted intended discretionary spending after applying the signed household change.",
    )
    add(
        "intended_policy",
        "intended_total_consumption_usd",
        intended_consumption_mean,
        "usd_per_represented_household",
        "llm_household_intention",
        "Population-weighted total intended spending, including one-off purchases.",
    )
    normalized_policy = {
        "intended_consumption_growth_pct": 100.0
        * (intended_consumption_mean / max(baseline_consumption_mean, 1.0) - 1.0),
        "extra_debt_payment_pct_of_baseline_consumption": 100.0
        * policy_means["extra_debt_payment_usd"]
        / max(baseline_consumption_mean, 1.0),
        "borrowing_intent_pct_of_baseline_consumption": 100.0
        * policy_means["borrowing_intent_usd"]
        / max(baseline_consumption_mean, 1.0),
    }
    for field, value in normalized_policy.items():
        add(
            "intended_policy_normalized",
            field,
            value,
            "percent",
            "llm_household_intention",
            "Population-weighted intention normalized by population-weighted baseline consumption.",
        )

    decision_by_id = decisions.set_index(decisions["household_id"].astype(str), drop=False)
    if set(decision_by_id.index) != set(weights):
        raise ValueError(f"decision identities do not match weights: {run_dir}")
    for field in (
        "baseline_consumption_usd",
        "desired_consumption_usd",
        "consumption_usd",
        "debt_payment_usd",
        "borrowing_usd",
    ):
        values = {household_id: _require_finite(decision_by_id.loc[household_id, field], field) for household_id in weights}
        add(
            "household_execution",
            field,
            _weighted_mean(values, weights),
            "usd_per_represented_household",
            "code_enforced_budgets_and_settlement",
            "Population-weighted feasible outcome after code-enforced budgets, credit limits, and settlement.",
        )
    deposit_change = {
        household_id: _require_finite(decision_by_id.loc[household_id, "deposit_balance_end_usd"], "deposit_end")
        - _require_finite(decision_by_id.loc[household_id, "deposit_balance_start_usd"], "deposit_start")
        for household_id in weights
    }
    debt_change = {
        household_id: _require_finite(decision_by_id.loc[household_id, "revolving_debt_end_usd"], "debt_end")
        - _require_finite(decision_by_id.loc[household_id, "revolving_debt_start_usd"], "debt_start")
        for household_id in weights
    }
    add(
        "household_execution",
        "executed_liquid_deposit_residual_usd",
        _weighted_mean(deposit_change, weights),
        "usd_per_represented_household",
        "code_enforced_budgets_and_settlement",
        "Population-weighted change in the liquid cash residual after code-enforced budgets and settlement.",
    )
    add("household_execution", "revolving_debt_change_usd", _weighted_mean(debt_change, weights), "usd_per_represented_household", "code_enforced_budgets_and_settlement", "Closing minus opening revolving debt after code-enforced settlement.")

    macro_row = macro.iloc[0]
    for field in (
        "consumption_growth_pct",
        "routine_nominal_spending_drift_pct",
        "gross_income_residual_rate_pct",
        "gross_income_residual_rate_change_pp",
        "revolving_credit_growth_pct",
        "employment_rate_pct",
        "price_growth_pct",
        "output_units",
        "units_sold",
        "inventory_end_units",
    ):
        if field.endswith("_pp"):
            unit = "percentage_points"
        elif field.endswith(("_pct", "rate_pct")):
            unit = "percent"
        else:
            unit = "population_weighted_units"
        add("macro_execution", field, macro_row[field], unit, "code_enforced_budgets_and_settlement", "Settled aggregate from the LLM household economy after code-enforced budgets and settlement.")

    employer = economy["employer"]
    credit = economy["credit"]
    add("institution_execution", "employer_demand_pressure_index", 100.0 * employer["demand_pressure"], "index", "code_enforced_budgets_and_settlement", "Current demand relative to the employer's baseline demand after settlement.")
    add("institution_execution", "employer_profit_usd", employer["profit_usd"], "population_weighted_usd", "code_enforced_budgets_and_settlement", "No-capital firm accounting residual after explicit costs.")
    add("institution_execution", "credit_rationing_ratio", credit["rationing_ratio"], "ratio", "code_enforced_budgets_and_settlement", "Share of requested new borrowing funded by code-enforced settlement.")
    add("institution_execution", "credit_profit_usd", credit["profit_usd"], "population_weighted_usd", "code_enforced_budgets_and_settlement", "Interest income less chargeoffs after settlement.")

    shadow = firm_response_shadow(
        consumption_growth_pct=macro_row["consumption_growth_pct"],
        inventory_end_units=macro_row["inventory_end_units"],
        units_sold=macro_row["units_sold"],
    )
    for field, value in shadow.items():
        unit = "index" if field.endswith("_index") else "percentage_points" if field.endswith("_pp") else "percent"
        add(
            "firm_response_shadow",
            field,
            value,
            unit,
            "mechanical_firm_feedback",
            "One-step mechanical firm feedback from household demand; not fed back into households and not an empirical forecast.",
        )
    return manifest, rows


def _observed_panel(retrospective_dir: Path, prospective_dir: Path | None) -> pd.DataFrame:
    parent = _read_json(retrospective_dir / "manifest.json")
    joined_path = _validate_artifact(retrospective_dir, parent, "predicted_vs_actual.csv")
    joined = pd.read_csv(joined_path)
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        common = {
            "origin_month": str(row["origin_month"]),
            "target_month": str(row["target_month"]),
            "metric": str(row["metric"]),
            "evaluation_status": "retrospective_diagnostic_not_confirmatory",
            "mapping_quality": str(row["mapping_quality"]),
            "mapping_note": str(row["mapping_note"]),
        }
        rows.append({**common, "series_role": "llm_household_economy", "value": float(row["prediction"]), "source_class": "llm_household_economy_code_enforced_budgets_and_settlement"})
        rows.append({**common, "series_role": "first_release_actual", "value": float(row["actual"]), "source_class": "observed_first_release"})
        if row["metric"] == "consumption_growth_pct":
            rows.append({**common, "series_role": "routine_visible_drift", "value": float(row["routine_nominal_spending_drift_pct"]), "source_class": "origin_visible_statistical_anchor"})
    if prospective_dir is not None:
        manifest = _read_json(prospective_dir / "manifest.json")
        path = _validate_artifact(prospective_dir, manifest, "macro_forecast_paths.csv")
        frame = pd.read_csv(path)
        if len(frame) != 1 or manifest.get("evaluation_status") != "prospective_frozen":
            raise ValueError("prospective run must contain one frozen median path")
        row = frame.iloc[0]
        for metric, quality, note in (
            ("consumption_growth_pct", "closest_aggregate_proxy", "Frozen ecology spending-growth proxy; realization not yet available."),
            ("revolving_credit_growth_pct", "directional_proxy", "Frozen ecology revolving-credit direction proxy; realization not yet available."),
        ):
            common = {
                "origin_month": str(manifest["origin_month"]),
                "target_month": str(manifest["target_month"]),
                "metric": metric,
                "evaluation_status": "prospective_frozen_unscored",
                "mapping_quality": quality,
                "mapping_note": note,
            }
            rows.append({**common, "series_role": "llm_household_economy", "value": float(row[metric]), "source_class": "llm_household_economy_code_enforced_budgets_and_settlement"})
            if metric == "consumption_growth_pct":
                rows.append({**common, "series_role": "routine_visible_drift", "value": float(row["routine_nominal_spending_drift_pct"]), "source_class": "origin_visible_statistical_anchor"})
    result = pd.DataFrame(rows)
    return result.sort_values(["target_month", "metric", "series_role"]).reset_index(drop=True)


def _feedback_period_two_rows(
    feedback_dir: Path | None,
    *,
    expected_period_one_replay_equivalence_sha256: str | None = None,
    expected_household_count: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return the bound period-two feedback marker, when the optional artifact exists."""

    if feedback_dir is None:
        return [], None
    if not feedback_dir.is_dir():
        raise ValueError("explicit feedback run directory does not exist")
    path = feedback_dir / "dynamic_macro_paths.csv"
    if not path.is_file():
        raise ValueError("explicit feedback run is missing dynamic_macro_paths.csv")

    manifest = _read_json(feedback_dir / "manifest.json")
    if manifest.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
        raise ValueError("feedback run schema version mismatch")
    if manifest.get("accounting_passed") is not True:
        raise ValueError("feedback run must pass accounting")
    if manifest.get("replay_verified") is not True:
        raise ValueError("feedback run must be a verified replay")
    if (
        expected_household_count is not None
        and int(manifest.get("household_count", -1)) != expected_household_count
    ):
        raise ValueError("feedback household count does not match observability panel")
    if (
        expected_period_one_replay_equivalence_sha256 is not None
        and manifest.get("period_1_replay_equivalence_sha256")
        != expected_period_one_replay_equivalence_sha256
    ):
        raise ValueError("feedback run is not bound to the prospective period-1 run")
    _validate_artifact(feedback_dir, manifest, "dynamic_macro_paths.csv")
    frame = pd.read_csv(path)
    required = {
        "period",
        "consumption_usd",
        "output_units",
        "producer_employment_index",
        "producer_wage_index",
        "consumption_growth_from_period_1_pct",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "feedback dynamic macro artifact missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    numeric_period = pd.to_numeric(frame["period"], errors="coerce")
    period_one = frame.loc[
        numeric_period.eq(1) | frame["period"].astype(str).eq("period_1")
    ]
    period_two = frame.loc[
        numeric_period.eq(2) | frame["period"].astype(str).eq("period_2")
    ]
    if len(period_one) != 1 or len(period_two) != 1:
        raise ValueError("feedback dynamic macro artifact must contain exactly one period_2 row")
    period_one_row = period_one.iloc[0]
    row = period_two.iloc[0]
    for period_name, period_row in (("period_1", period_one_row), ("period_2", row)):
        for field in required.difference({"period"}):
            _require_finite(period_row[field], f"feedback {period_name} {field}")
    target_month = row.get("target_month", manifest.get("period_2_target_month"))
    if not isinstance(target_month, str) or not target_month:
        raise ValueError("feedback run requires a period_2_target_month")
    try:
        pd.Timestamp(target_month)
    except (TypeError, ValueError) as exc:
        raise ValueError("feedback period_2 target_month must be parseable") from exc

    common = {
        "origin_month": str(manifest.get("origin_month", "")),
        "target_month": target_month,
        "evaluation_status": "prospective_feedback_period_2_unscored",
    }
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            **common,
            "layer": "macro_execution",
            "metric": "consumption_growth_pct",
            "value": _require_finite(
                row["consumption_growth_from_period_1_pct"],
                "consumption_growth_from_period_1_pct",
            ),
            "unit": "percent",
            "source_class": "llm_household_economy",
            "interpretation": "Unscored period_2 LLM household economy outcome after mechanical firm feedback.",
        }
    )
    for metric, period_two_field, period_one_field in (
        ("recursive_producer_output_index", "output_units", "output_units"),
        (
            "recursive_producer_employment_index",
            "producer_employment_index",
            "producer_employment_index",
        ),
        (
            "recursive_producer_wage_index",
            "producer_wage_index",
            "producer_wage_index",
        ),
    ):
        baseline = _require_finite(period_one_row[period_one_field], period_one_field)
        if baseline <= 0.0:
            raise ValueError(f"feedback period_1 {period_one_field} must be positive")
        rows.append(
            {
                **common,
                "layer": "recursive_firm_feedback",
                "metric": metric,
                "value": 100.0
                * _require_finite(row[period_two_field], period_two_field)
                / baseline,
                "unit": "index",
                "source_class": "mechanical_firm_feedback",
                "interpretation": "Unscored period_2 realized aggregate firm feedback; it is mechanical, not an LLM output.",
            }
        )
    return rows, _artifact_sha256(feedback_dir / "manifest.json")


def _write_chart(observed: pd.DataFrame, simulation: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 14.0), sharex=True)
    colors = {"actual": "#2463A6", "llm": "#B23A2B", "routine": "#777777", "mechanical": "#4E6E58", "shadow": "#865D36"}

    def observed_axis(axis: Any, metric: str, title: str, direction_only: bool = False) -> None:
        subset = observed.loc[observed["metric"].eq(metric)]
        styles = {
            "first_release_actual": ("actual", "o", "-", "First-release actual"),
            "llm_household_economy": ("llm", "s", "-", LLM_HOUSEHOLD_ECONOMY_CHART_LABEL),
            "routine_visible_drift": ("routine", "^", "--", "Origin-visible drift"),
        }
        for role, (color, marker, line, label) in styles.items():
            rows = subset.loc[subset["series_role"].eq(role)].sort_values("target_month")
            if rows.empty:
                continue
            dates = pd.to_datetime(rows["target_month"])
            historical = ~rows["evaluation_status"].eq("prospective_frozen_unscored")
            values = np.sign(rows["value"]) if direction_only else rows["value"]
            axis.plot(dates[historical], values.loc[historical], color=colors[color], marker=marker, linestyle=line, linewidth=2, label=label)
            prospective = ~historical
            if prospective.any():
                axis.scatter(dates[prospective], values.loc[prospective], facecolors="none", edgecolors=colors[color], marker="D", s=70, linewidths=1.8, label=f"{label} (frozen)")
        axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
        axis.set_title(title)
        if direction_only:
            axis.set_ylabel("Direction only")
            axis.set_yticks([-1.0, 0.0, 1.0], ["contraction", "zero", "expansion"])
            axis.set_ylim(-1.25, 1.25)
        else:
            axis.set_ylabel("Monthly growth (%)")
        axis.legend(fontsize=8, frameon=False, ncol=1, loc="best")

    observed_axis(axes[0, 0], "consumption_growth_pct", "Observed comparison: nominal consumption")
    observed_axis(axes[0, 1], "revolving_credit_growth_pct", "Observed comparison: revolving credit (sign-only proxy)", True)

    def plot_simulation(
        axis: Any,
        rows: pd.DataFrame,
        *,
        label: str,
        color: str | None = None,
        linestyle: str = "-",
    ) -> None:
        rows = rows.sort_values("target_month")
        historical = rows["evaluation_status"].eq(
            "retrospective_diagnostic_not_confirmatory"
        )
        line, = axis.plot(
            pd.to_datetime(rows.loc[historical, "target_month"]),
            rows.loc[historical, "value"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
            linestyle=linestyle,
        )
        future = rows["evaluation_status"].isin(
            {"prospective_frozen", "prospective_feedback_period_2_unscored"}
        )
        if future.any():
            feedback = rows["evaluation_status"].eq(
                "prospective_feedback_period_2_unscored"
            )
            axis.scatter(
                pd.to_datetime(rows.loc[future, "target_month"]),
                rows.loc[future, "value"],
                facecolors="none",
                edgecolors=color or line.get_color(),
                marker="D",
                s=62,
                linewidths=1.6,
                label=(
                    f"{label} (period_2, unscored)"
                    if feedback.any()
                    else None
                ),
            )

    belief_labels = {
        "expected_inflation_pct": "Inflation",
        "expected_income_growth_pct": "Income growth",
        "job_loss_probability_pct": "Personal job loss",
    }
    for metric, label in belief_labels.items():
        rows = simulation.loc[(simulation["layer"] == "beliefs") & (simulation["metric"] == metric)].sort_values("target_month")
        plot_simulation(axes[1, 0], rows, label=label)
    axes[1, 0].set_title("Population-weighted LLM beliefs")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].legend(fontsize=8, frameon=False)

    policy_labels = {
        "intended_consumption_growth_pct": "Consumption growth",
        "extra_debt_payment_pct_of_baseline_consumption": "Extra debt payment",
        "borrowing_intent_pct_of_baseline_consumption": "Borrowing",
    }
    policy_colors = {
        "intended_consumption_growth_pct": "#2463A6",
        "extra_debt_payment_pct_of_baseline_consumption": "#4E8B57",
        "borrowing_intent_pct_of_baseline_consumption": "#7A5AA6",
    }
    for metric, label in policy_labels.items():
        rows = simulation.loc[(simulation["layer"] == "intended_policy_normalized") & (simulation["metric"] == metric)].sort_values("target_month")
        plot_simulation(
            axes[1, 1],
            rows,
            label=label,
            color=policy_colors[metric],
        )
    axes[1, 1].axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axes[1, 1].set_title("LLM intended household policy")
    axes[1, 1].set_ylabel("Consumption / debt actions (%)")
    axes[1, 1].legend(fontsize=8, frameon=False, ncol=2)

    execution_labels = {
        "consumption_growth_pct": "Consumption growth",
        "revolving_credit_growth_pct": "Revolving debt growth",
        "gross_income_residual_rate_change_pp": "Gross-income residual change",
        "price_growth_pct": "Current code-enforced price growth",
    }
    execution_colors = {
        "consumption_growth_pct": "#2463A6",
        "revolving_credit_growth_pct": "#E07A1F",
        "gross_income_residual_rate_change_pp": "#4E8B57",
        "price_growth_pct": "#7A5AA6",
    }
    credit_axis = axes[2, 0].twinx()
    for metric, label in execution_labels.items():
        rows = simulation.loc[(simulation["layer"] == "macro_execution") & (simulation["metric"] == metric)].sort_values("target_month")
        target_axis = credit_axis if metric == "revolving_credit_growth_pct" else axes[2, 0]
        plot_simulation(
            target_axis,
            rows,
            label=label,
            color=execution_colors[metric],
            linestyle="--" if target_axis is credit_axis else "-",
        )
    axes[2, 0].axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axes[2, 0].set_title("LLM household economy: settled outcomes")
    axes[2, 0].set_ylabel("Growth (%) / residual change (pp)")
    credit_axis.set_ylabel("Revolving debt growth (%)", color="#E07A1F")
    handles, labels = axes[2, 0].get_legend_handles_labels()
    handles2, labels2 = credit_axis.get_legend_handles_labels()
    axes[2, 0].legend(handles + handles2, labels + labels2, fontsize=8, frameon=False, ncol=2)

    shadow_labels = {
        "firm_expected_sales_index": "Expected sales",
        "firm_target_output_index": "Target output",
        "firm_required_labor_index": "Required labor",
        "firm_planned_employment_index": "Planned employment",
    }
    for metric, label in shadow_labels.items():
        rows = simulation.loc[(simulation["layer"] == "firm_response_shadow") & (simulation["metric"] == metric)].sort_values("target_month")
        plot_simulation(axes[2, 1], rows, label=label)
    recursive_labels = {
        "recursive_producer_output_index": "Recursive output (period 2)",
        "recursive_producer_employment_index": "Recursive employment (period 2)",
        "recursive_producer_wage_index": "Recursive wage (period 2)",
    }
    for metric, label in recursive_labels.items():
        rows = simulation.loc[
            (simulation["layer"] == "recursive_firm_feedback")
            & (simulation["metric"] == metric)
        ].sort_values("target_month")
        plot_simulation(axes[2, 1], rows, label=label)
    axes[2, 1].axhline(100.0, color="#B8B8B8", linewidth=0.8)
    axes[2, 1].set_title("Producer response and two-period feedback")
    axes[2, 1].set_ylabel("Index (origin baseline = 100)")
    axes[2, 1].legend(fontsize=8, frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle("Household Ecology: Diagnostic Economic Observability Surface", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.945, "Four-point retrospective proxy diagnostic (Jan-Apr origins); observed outcomes, LLM household intentions, settlement, and firm feedback are separate; hollow diamonds are frozen and unscored.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    retrospective = args.retrospective_run.resolve()
    prospective = args.prospective_run.resolve() if args.prospective_run else None
    feedback = args.feedback_run.resolve() if args.feedback_run else None
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    parent = _read_json(retrospective / "manifest.json")
    origins = [str(value) for value in parent.get("origin_months", [])]
    if not origins:
        raise ValueError("retrospective manifest has no origin months")
    expected_count = int(parent.get("household_count", -1))
    weights = _load_weights(args.households, expected_count)

    simulation_rows: list[dict[str, Any]] = []
    source_runs: dict[str, dict[str, Any]] = {}
    for origin in origins:
        child = retrospective / "runs" / origin
        expected_hash = parent.get("child_run_manifest_sha256", {}).get(origin)
        actual_hash = _artifact_sha256(child / "manifest.json")
        if actual_hash != expected_hash:
            raise ValueError(f"child manifest hash mismatch for {origin}")
        manifest, rows = _run_payload(child, weights)
        # Child runs were frozen before the parent loaded realizations, so their
        # own manifests correctly say prospective_frozen. Once assembled by the
        # retrospective parent, every diagnostic row must inherit the parent's
        # retrospective status instead of masquerading as still prospective.
        for row in rows:
            row["evaluation_status"] = "retrospective_diagnostic_not_confirmatory"
        simulation_rows.extend(rows)
        source_runs[origin] = {
            "path": str(child),
            "manifest_sha256": actual_hash,
            "source_child_status": manifest["evaluation_status"],
            "panel_evaluation_status": "retrospective_diagnostic_not_confirmatory",
        }
    prospective_manifest_sha256: str | None = None
    prospective_replay_equivalence_sha256: str | None = None
    if prospective is not None:
        manifest, rows = _run_payload(prospective, weights)
        simulation_rows.extend(rows)
        prospective_manifest_sha256 = _artifact_sha256(prospective / "manifest.json")
        prospective_replay_equivalence_sha256 = manifest.get(
            "replay_equivalence_sha256"
        )
        if not isinstance(prospective_replay_equivalence_sha256, str):
            raise ValueError("prospective run lacks replay equivalence provenance")
        source_runs[str(manifest["origin_month"])] = {
            "path": str(prospective),
            "manifest_sha256": prospective_manifest_sha256,
            "source_child_status": manifest["evaluation_status"],
            "panel_evaluation_status": "prospective_frozen",
        }
    feedback_rows, feedback_manifest_sha256 = _feedback_period_two_rows(
        feedback,
        expected_period_one_replay_equivalence_sha256=(
            prospective_replay_equivalence_sha256
        ),
        expected_household_count=expected_count,
    )
    simulation_rows.extend(feedback_rows)

    observed = _observed_panel(retrospective, prospective)
    simulation = pd.DataFrame(simulation_rows).sort_values(["target_month", "layer", "metric"]).reset_index(drop=True)
    observed.to_csv(output / "observed_comparison_panel.csv", index=False)
    simulation.to_csv(output / "simulation_observability_panel.csv", index=False)
    _write_chart(observed, simulation, output / "economic_observability_surface.png")

    status_counts = simulation.groupby(["evaluation_status", "source_class"]).size().to_dict()
    recursive_description = (
        "The bound period-two run then carries household deposits and debt forward, "
        "realizes aggregate producer output, employment, wages, and family wage income, "
        "and elicits a fresh decision from every LLM household. It uses no future "
        "observed data and remains an unscored mechanism trace."
        if feedback is not None
        else "No recursive period-two feedback run was supplied."
    )
    report = [
        "# Full Economic Observability Surface",
        "",
        "This artifact exposes the whole active household economy without upgrading diagnostics into evidence. It separates observed first-release outcomes, LLM household beliefs and intentions, code-enforced budget and institutional settlement, and unscored mechanical firm feedback. When a bound feedback run supplies period_2, it is shown only as a hollow, unscored future marker.",
        "",
        "## Structural Addition",
        "",
        "The mechanical firm-feedback shadow inherits household-demand growth as expected sales, closes 35% of the inventory gap in target output, maps output one-for-one to required labor at fixed productivity, and closes 25% of that labor requirement through planned employment. The price-pressure index combines 15% of demand growth with 10% of the inventory gap. These declared coefficients are diagnostic mechanics, not fitted estimates.",
        "",
        "That historical shadow does not feed wages, employment, prices, or income back into household calls. It is distinct from the recursive mechanism and uses separate metric names.",
        "",
        recursive_description,
        "",
        "## Evidence Boundary",
        "",
        "- Historical observed comparisons remain retrospective diagnostics.",
        "- The prospective point remains frozen and unscored.",
        "- The gross-income residual is internal household accounting, not national saving.",
        "- Revolving credit remains a direction-only proxy.",
        "- Both firm mechanisms are unscored; neither can improve the historical forecast score and neither is an LLM output.",
        "- Household budgets, goods inventory, bank stocks, and named counterparty flows are audited; a firm balance sheet and full external-sector stocks are not modeled.",
        "",
        "## Output Contract",
        "",
        "- `observed_comparison_panel.csv`: tidy observed, LLM-household-economy, and routine-anchor rows.",
        "- `simulation_observability_panel.csv`: LLM intentions, code-enforced settlement, and mechanical firm-feedback rows.",
        "- `economic_observability_surface.png`: the six-panel diagnostic figure.",
        "- `manifest.json`: source bindings, declared mechanics, and artifact hashes.",
        "",
        f"Rows by status and source class: `{status_counts}`.",
    ]
    (output / "observability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "firm_shadow_version": FIRM_SHADOW_VERSION,
        "firm_shadow_role": "unscored_one_step_mechanical_diagnostic",
        "firm_shadow_parameters": {
            "target_inventory_share": TARGET_INVENTORY_SHARE,
            "inventory_adjustment_speed": INVENTORY_ADJUSTMENT_SPEED,
            "employment_adjustment_speed": EMPLOYMENT_ADJUSTMENT_SPEED,
            "price_pressure_demand_weight": PRICE_PRESSURE_DEMAND_WEIGHT,
            "price_pressure_inventory_weight": PRICE_PRESSURE_INVENTORY_WEIGHT,
        },
        "retrospective_manifest_sha256": _artifact_sha256(retrospective / "manifest.json"),
        "prospective_manifest_sha256": prospective_manifest_sha256,
        "feedback_manifest_sha256": feedback_manifest_sha256,
        "household_input_sha256": _file_sha256(args.households),
        "source_runs": source_runs,
        "retrospective_is_confirmatory": False,
        "prospective_is_scored": False,
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.name != "manifest.json":
            manifest["artifacts"][path.name] = _artifact_sha256(path)
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
