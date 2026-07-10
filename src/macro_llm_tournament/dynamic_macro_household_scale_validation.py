"""Run-artifact validation and integrity helpers for household-scale comparison."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE
from .dynamic_macro_common import FORBIDDEN_PROMPT_KEYS, FORBIDDEN_PROMPT_PREFIXES
from .dynamic_macro_economy import OUTPUT_FILES as ECONOMY_OUTPUT_FILES
from .dynamic_macro_inputs import REQUIRED_FAMILIES


COHORTS = ("corrected81", "corrected200")
EXPECTED_ORIGINS = tuple(f"2026-{month:02d}-01" for month in range(1, 6))
EXPECTED_SCORE_ORIGINS = EXPECTED_ORIGINS[1:]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_OUTPUT_FILES = tuple(ECONOMY_OUTPUT_FILES)
REQUIRED_JOINED_COLUMNS = {
    "candidate",
    "origin_month",
    "target_name",
    "family",
    "prediction",
    "target_value",
    "default_scale",
    "transform",
    "origin_visible_denominator_value",
}
SURFACE_COLUMNS = (
    "origin_month",
    "as_of_date",
    "target_observation_date",
    "target_name",
    "series_id",
    "family",
    "economy_measure",
    "economy_transform",
    "transform",
    "default_scale",
    "target_value",
    "origin_visible_denominator_value",
)
COHORT_SPEC_ALIASES = {
    "corrected81": ("corrected81", "81", "run81", "run_81"),
    "corrected200": ("corrected200", "200", "run200", "run_200"),
}
PROMPT_LOCK_KEYS = (
    "prompt_sha256",
    "prompt_hash",
    "prompt_contract_sha256",
    "prompt_lock_sha256",
    "prompt_version",
)
SOURCE_LOCK_KEYS = ("execution_source_tree_sha256",)
MECHANISM_EXCLUDED_KEYS = {
    "household_count",
    "household_provenance",
    "households_sha256",
    "normalized_household_state_sha256",
    "raw_input_file_sha256",
    "raw_records_sha256",
    "raw_record_count",
    "raw_record_batch_count",
    "raw_record_batch_counts",
    "batch_count",
    "batch_counts",
    "cohort",
    "cohort_id",
    "run_id",
    "run_name",
}


class DynamicMacroHouseholdScaleError(ValueError):
    """Raised for any invalid lock, run, score surface, or output state."""


def _validate_accounting(
    accounting: pd.DataFrame,
    decisions: pd.DataFrame,
    periods: pd.DataFrame,
    final_states: pd.DataFrame,
    households: pd.DataFrame,
    manifest: Mapping[str, Any],
    cohort: str,
) -> float:
    required = {
        "candidate",
        "period_index",
        "unit",
        "identity",
        "residual",
        "abs_residual",
        "passed",
    }
    decision_columns = {
        "candidate",
        "period_index",
        "type_id",
        "population_weight",
        "liquid_assets_before",
        "labor_income",
        "transfer",
        "consumption",
        "debt_repayment",
        "liquid_assets_after",
        "debt_before",
        "debt_after",
    }
    period_columns = {
        "candidate",
        "period_index",
        "aggregate_consumption",
        "aggregate_income",
        "aggregate_transfer",
        "aggregate_saving",
        "aggregate_debt_repayment",
        "safe_asset_absorption",
        "aggregate_liquid_assets",
        "aggregate_debt",
        "output",
        "goods_market_residual",
        "next_aggregate_income",
    }
    if accounting.empty or not required.issubset(accounting.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting table is incomplete"
        )
    if not decision_columns.issubset(decisions.columns) or not period_columns.issubset(
        periods.columns
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} primitive accounting ledgers are incomplete"
        )
    final_columns = {"candidate", "type_id", "population_weight", "labor_income"}
    if not final_columns.issubset(final_states.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} terminal household states are incomplete"
        )
    numeric_decisions = _numeric_frame(
        decisions, decision_columns - {"candidate", "type_id"}, cohort, "decisions"
    )
    numeric_periods = _numeric_frame(
        periods, period_columns - {"candidate"}, cohort, "periods"
    )
    expected_weights = households.set_index(households["type_id"].astype(str))[
        "population_weight"
    ].astype(float)
    decisions = decisions.copy()
    decisions["type_id"] = decisions["type_id"].astype(str)
    final_states = final_states.copy()
    final_states["type_id"] = final_states["type_id"].astype(str)
    _numeric_frame(
        final_states,
        {"population_weight", "labor_income"},
        cohort,
        "terminal household states",
    )
    expected_keys = {
        (candidate, period, type_id)
        for candidate in ("llm", "adaptive")
        for period in range(5)
        for type_id in expected_weights.index
    }
    actual_keys = {
        (str(row.candidate), int(row.period_index), str(row.type_id))
        for row in decisions.itertuples(index=False)
    }
    if len(actual_keys) != len(decisions) or actual_keys != expected_keys:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} decisions do not cover every household exactly once"
        )
    if any(
        not math.isclose(
            float(row.population_weight),
            float(expected_weights[str(row.type_id)]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in decisions.itertuples(index=False)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} decision population weights do not match households"
        )
    final_keys = {
        (str(row.candidate), str(row.type_id))
        for row in final_states.itertuples(index=False)
    }
    expected_final_keys = {
        (candidate, type_id)
        for candidate in ("llm", "adaptive")
        for type_id in expected_weights.index
    }
    if (
        len(final_states) != len(expected_final_keys)
        or final_keys != expected_final_keys
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} terminal household states do not cover both candidates"
        )
    if any(
        not math.isclose(
            float(row.population_weight),
            float(expected_weights[str(row.type_id)]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in final_states.itertuples(index=False)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} terminal household weights do not match households"
        )
    budget = (
        numeric_decisions["liquid_assets_before"]
        + numeric_decisions["labor_income"]
        + numeric_decisions["transfer"]
        - numeric_decisions["consumption"]
        - numeric_decisions["debt_repayment"]
        - numeric_decisions["liquid_assets_after"]
    )
    debt = (
        numeric_decisions["debt_before"]
        - numeric_decisions["debt_repayment"]
        - numeric_decisions["debt_after"]
    )
    expected_rows: list[dict[str, Any]] = []
    for index, row in decisions.reset_index(drop=True).iterrows():
        for identity, residual in (
            ("household_cash_budget", float(budget.iloc[index])),
            ("household_debt_stock", float(debt.iloc[index])),
        ):
            expected_rows.append(
                {
                    "candidate": str(row["candidate"]),
                    "period_index": int(row["period_index"]),
                    "unit": str(row["type_id"]),
                    "identity": identity,
                    "residual": residual,
                }
            )
    for candidate in ("llm", "adaptive"):
        for period in range(5):
            mask = decisions["candidate"].astype(str).eq(candidate) & numeric_decisions[
                "period_index"
            ].eq(period)
            subset = decisions.loc[mask]
            weights = pd.to_numeric(subset["population_weight"], errors="coerce")
            consumption = float(
                (weights * pd.to_numeric(subset["consumption"], errors="coerce")).sum()
            )
            income = float(
                (weights * pd.to_numeric(subset["labor_income"], errors="coerce")).sum()
            )
            transfer = float(
                (weights * pd.to_numeric(subset["transfer"], errors="coerce")).sum()
            )
            debt_repayment = float(
                (
                    weights * pd.to_numeric(subset["debt_repayment"], errors="coerce")
                ).sum()
            )
            saving = float(
                (
                    weights
                    * (
                        pd.to_numeric(subset["labor_income"], errors="coerce")
                        + pd.to_numeric(subset["transfer"], errors="coerce")
                        - pd.to_numeric(subset["consumption"], errors="coerce")
                        - pd.to_numeric(subset["debt_repayment"], errors="coerce")
                    )
                ).sum()
            )
            liquid_assets = float(
                (
                    weights
                    * pd.to_numeric(subset["liquid_assets_after"], errors="coerce")
                ).sum()
            )
            debt_stock = float(
                (weights * pd.to_numeric(subset["debt_after"], errors="coerce")).sum()
            )
            period_rows = periods[
                periods["candidate"].astype(str).eq(candidate)
                & numeric_periods["period_index"].eq(period)
            ]
            if len(period_rows) != 1:
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} periods are incomplete or duplicated"
                )
            period_row = period_rows.iloc[0]
            for column, expected in (
                ("aggregate_consumption", consumption),
                ("aggregate_income", income),
                ("aggregate_transfer", transfer),
                ("aggregate_saving", saving),
                ("aggregate_debt_repayment", debt_repayment),
                ("safe_asset_absorption", saving),
                ("aggregate_liquid_assets", liquid_assets),
                ("aggregate_debt", debt_stock),
            ):
                actual = float(period_row[column])
                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=ACCOUNTING_TOLERANCE,
                ):
                    raise DynamicMacroHouseholdScaleError(
                        f"{cohort} {column} does not reproduce from decisions"
                    )
            goods = float(period_row["output"]) - consumption
            if not math.isclose(
                float(period_row["goods_market_residual"]),
                goods,
                rel_tol=0.0,
                abs_tol=ACCOUNTING_TOLERANCE,
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} goods-market residual does not reproduce"
                )
            expected_rows.append(
                {
                    "candidate": candidate,
                    "period_index": period,
                    "unit": "aggregate",
                    "identity": "one_good_output_equals_consumption",
                    "residual": goods,
                }
            )
            if period < 4:
                next_decisions = decisions[
                    decisions["candidate"].astype(str).eq(candidate)
                    & numeric_decisions["period_index"].eq(period + 1)
                ]
                next_income = float(
                    (
                        pd.to_numeric(
                            next_decisions["population_weight"], errors="coerce"
                        )
                        * pd.to_numeric(next_decisions["labor_income"], errors="coerce")
                    ).sum()
                )
                if not math.isclose(
                    float(period_row["next_aggregate_income"]),
                    next_income,
                    rel_tol=0.0,
                    abs_tol=ACCOUNTING_TOLERANCE,
                ):
                    raise DynamicMacroHouseholdScaleError(
                        f"{cohort} next aggregate income does not reconcile with decisions"
                    )
            else:
                terminal = final_states[
                    final_states["candidate"].astype(str).eq(candidate)
                ]
                terminal_income = float(
                    (
                        pd.to_numeric(terminal["population_weight"], errors="coerce")
                        * pd.to_numeric(terminal["labor_income"], errors="coerce")
                    ).sum()
                )
                if not math.isclose(
                    float(period_row["next_aggregate_income"]),
                    terminal_income,
                    rel_tol=0.0,
                    abs_tol=ACCOUNTING_TOLERANCE,
                ):
                    raise DynamicMacroHouseholdScaleError(
                        f"{cohort} terminal aggregate income does not reconcile with household states"
                    )
    expected_accounting = pd.DataFrame(expected_rows)
    observed = accounting.copy()
    observed["candidate"] = observed["candidate"].astype(str)
    observed["unit"] = observed["unit"].astype(str)
    observed["identity"] = observed["identity"].astype(str)
    observed["period_index"] = pd.to_numeric(observed["period_index"], errors="coerce")
    observed["residual"] = pd.to_numeric(observed["residual"], errors="coerce")
    observed["abs_residual"] = pd.to_numeric(observed["abs_residual"], errors="coerce")
    if (
        observed[["period_index", "residual", "abs_residual"]].isna().any().any()
        or observed.duplicated(["candidate", "period_index", "unit", "identity"]).any()
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting identities are invalid"
        )
    keys = ["candidate", "period_index", "unit", "identity"]
    merged = expected_accounting.merge(
        observed,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_expected", "_observed"),
    )
    if len(merged) != len(expected_accounting) or not merged["_merge"].eq("both").all():
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting evidence does not cover recomputed identities"
        )
    if (
        not (merged["residual_expected"] - merged["residual_observed"])
        .abs()
        .le(1e-10)
        .all()
        or not (merged["abs_residual"] - merged["residual_expected"].abs())
        .abs()
        .le(1e-10)
        .all()
        or not observed["passed"].map(_as_bool).all()
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting evidence does not reproduce"
        )
    maximum = float(merged["residual_expected"].abs().max())
    if not math.isfinite(maximum) or maximum > ACCOUNTING_TOLERANCE:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting residual exceeds tolerance"
        )
    recorded = float(manifest.get("max_accounting_abs_residual", math.nan))
    if not math.isclose(maximum, recorded, rel_tol=0.0, abs_tol=1e-10):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accounting maximum does not reproduce"
        )
    return maximum


def _validate_score_lineage(
    forecasts: pd.DataFrame,
    joined: pd.DataFrame,
    periods: pd.DataFrame,
    normalized_spec: Mapping[str, Any],
    bundle_targets: pd.DataFrame,
    cohort: str,
) -> None:
    contract = _target_contract(normalized_spec, {}, cohort)
    period_required = {
        "candidate",
        "period_index",
        "origin_month",
        "aggregate_consumption",
        "aggregate_income",
        "aggregate_saving",
        "aggregate_debt",
        "next_employment_rate",
        "next_inflation_rate",
        "next_policy_rate",
        "next_aggregate_income",
    }
    if not period_required.issubset(periods.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} periods lack score-bearing state"
        )
    _numeric_frame(
        periods,
        period_required - {"candidate", "origin_month"},
        cohort,
        "score-bearing periods",
    )
    expected_period_keys = {
        (candidate, index, origin)
        for candidate in ("llm", "adaptive")
        for index, origin in enumerate(EXPECTED_ORIGINS)
    }
    actual_period_keys = {
        (str(row.candidate), int(row.period_index), str(row.origin_month))
        for row in periods.itertuples(index=False)
    }
    if (
        len(periods) != len(expected_period_keys)
        or actual_period_keys != expected_period_keys
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} score-bearing period path is incomplete"
        )
    bundle_by_origin = {
        origin: frame
        for origin, frame in bundle_targets.groupby("origin_month", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for candidate in ("llm", "adaptive"):
        path = periods[periods["candidate"].astype(str).eq(candidate)].copy()
        path = path.sort_values("period_index", kind="mergesort").reset_index(drop=True)
        for index, origin in enumerate(EXPECTED_ORIGINS):
            if origin not in EXPECTED_SCORE_ORIGINS:
                continue
            current = path.iloc[index]
            previous = path.iloc[index - 1]
            for target in bundle_by_origin[origin].to_dict(orient="records"):
                mapping = contract[str(target["target_name"])]
                rows.append(
                    {
                        "candidate": candidate,
                        "origin_month": origin,
                        "as_of_date": str(target["as_of_date"]),
                        "target_observation_date": str(
                            target["target_observation_date"]
                        ),
                        "target_name": str(target["target_name"]),
                        "series_id": str(target["series_id"]),
                        "family": str(target["family"]),
                        "prediction": _mapped_period_value(
                            current, previous, mapping, cohort
                        ),
                        "economy_measure": mapping["economy_measure"],
                        "economy_transform": mapping["economy_transform"],
                    }
                )
    expected_forecasts = pd.DataFrame(rows)
    forecast_columns = list(expected_forecasts.columns)
    if not set(forecast_columns).issubset(forecasts.columns):
        raise DynamicMacroHouseholdScaleError(f"{cohort} forecasts are incomplete")
    _assert_keyed_frame_matches(
        forecasts[forecast_columns],
        expected_forecasts,
        keys=["candidate", "origin_month", "target_name"],
        numeric={"prediction"},
        label=f"{cohort} forecasts do not reproduce from periods",
    )
    expected_targets = (
        bundle_targets.assign(_join_key=1)
        .merge(
            pd.DataFrame({"candidate": ["llm", "adaptive"], "_join_key": [1, 1]}),
            on="_join_key",
        )
        .drop(columns="_join_key")
    )
    target_columns = [
        "candidate",
        "origin_month",
        "as_of_date",
        "target_observation_date",
        "target_name",
        "series_id",
        "family",
        "transform",
        "default_scale",
        "target_value",
        "origin_visible_denominator_value",
    ]
    if not set(target_columns + ["prediction"]).issubset(joined.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score rows lack bundle lineage"
        )
    _assert_keyed_frame_matches(
        joined[target_columns],
        expected_targets[target_columns],
        keys=["candidate", "origin_month", "target_name"],
        numeric={"default_scale", "target_value", "origin_visible_denominator_value"},
        label=f"{cohort} joined targets differ from the locked frozen bundle",
    )
    _assert_keyed_frame_matches(
        joined[["candidate", "origin_month", "target_name", "prediction"]],
        forecasts[["candidate", "origin_month", "target_name", "prediction"]],
        keys=["candidate", "origin_month", "target_name"],
        numeric={"prediction"},
        label=f"{cohort} joined predictions differ from forecasts.csv",
    )


def _mapped_period_value(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    mapping: Mapping[str, Any],
    cohort: str,
) -> float:
    current_measure = _period_measure(current, str(mapping["economy_measure"]), cohort)
    previous_measure = _period_measure(
        previous, str(mapping["economy_measure"]), cohort
    )
    transform = str(mapping["economy_transform"])
    if transform == "pct_change":
        return _period_pct_change(current_measure, previous_measure, cohort)
    if transform == "nominal_pct_change":
        real_growth = _period_pct_change(current_measure, previous_measure, cohort)
        monthly_price = _annual_rate_to_monthly_pct(
            float(current["next_inflation_rate"]), cohort
        )
        return 100.0 * (
            (1.0 + real_growth / 100.0) * (1.0 + monthly_price / 100.0) - 1.0
        )
    if transform == "diff":
        return current_measure - previous_measure
    if transform == "level":
        return current_measure
    if transform == "annual_rate_to_monthly_pct":
        return _annual_rate_to_monthly_pct(current_measure, cohort)
    raise DynamicMacroHouseholdScaleError(
        f"{cohort} target mapping has unknown economy transform"
    )


def _period_measure(row: Mapping[str, Any], measure: str, cohort: str) -> float:
    if measure == "saving_rate_pct":
        income = float(row["aggregate_income"])
        return (
            0.0
            if abs(income) < 1e-12
            else 100.0 * float(row["aggregate_saving"]) / income
        )
    if measure == "unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["employment_rate"]))
    if measure == "next_unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["next_employment_rate"]))
    try:
        value = float(row[measure])
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} periods lack economy measure {measure}"
        ) from exc
    if not math.isfinite(value):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} economy measure {measure} is invalid"
        )
    return value


def _period_pct_change(current: float, previous: float, cohort: str) -> float:
    if abs(previous) < 1e-12:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} percentage-change forecast has a zero baseline"
        )
    return 100.0 * (current / previous - 1.0)


def _annual_rate_to_monthly_pct(rate: float, cohort: str) -> float:
    if rate <= -100.0:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} annual rate is at or below -100 percent"
        )
    return 100.0 * ((1.0 + rate / 100.0) ** (1.0 / 12.0) - 1.0)


def _assert_keyed_frame_matches(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
    numeric: set[str],
    label: str,
) -> None:
    if set(observed.columns) != set(expected.columns):
        raise DynamicMacroHouseholdScaleError(label)
    left = observed.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = (
        expected[list(left.columns)]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(left) != len(right)
        or left.duplicated(keys).any()
        or right.duplicated(keys).any()
    ):
        raise DynamicMacroHouseholdScaleError(label)
    for column in left.columns:
        if column in numeric:
            left_values = pd.to_numeric(left[column], errors="coerce")
            right_values = pd.to_numeric(right[column], errors="coerce")
            if (
                left_values.isna().any()
                or right_values.isna().any()
                or not (left_values - right_values).abs().le(1e-10).all()
            ):
                raise DynamicMacroHouseholdScaleError(label)
        elif not left[column].astype(str).equals(right[column].astype(str)):
            raise DynamicMacroHouseholdScaleError(label)


def _validate_joined_surface(
    joined: pd.DataFrame,
    normalized_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cohort: str,
) -> None:
    if not REQUIRED_JOINED_COLUMNS.issubset(joined.columns):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface is incomplete"
        )
    if set(joined["candidate"].astype(str)) != {"llm", "adaptive"}:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface is not a matched pair"
        )
    origins = set(joined["origin_month"].astype(str))
    if origins != set(EXPECTED_SCORE_ORIGINS):
        if any("2026-06" in origin for origin in origins):
            raise DynamicMacroHouseholdScaleError(f"{cohort} contains June rows")
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface must be Feb-May only"
        )
    if (
        joined.astype(str)
        .apply(lambda column: column.str.contains(r"2026-06", regex=True).any())
        .any()
    ):
        raise DynamicMacroHouseholdScaleError(f"{cohort} contains June rows")
    contract = _target_contract(normalized_spec, manifest, cohort)
    identity = joined[["candidate", "origin_month", "target_name"]]
    if identity.duplicated().any():
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface contains duplicate target rows"
        )
    expected = {
        (candidate, origin, name)
        for candidate in ("llm", "adaptive")
        for origin in EXPECTED_SCORE_ORIGINS
        for name in contract
    }
    actual = {
        (str(row.candidate), str(row.origin_month), str(row.target_name))
        for row in joined.itertuples(index=False)
    }
    if len(joined) != len(expected) or actual != expected:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface must contain exactly ten locked targets per candidate and origin"
        )
    metadata = joined[
        ["target_name", "series_id", "family", "transform", "default_scale"]
    ].copy()
    metadata["target_name"] = metadata["target_name"].astype(str)
    for name, target in contract.items():
        rows = metadata[metadata["target_name"].eq(name)]
        expected_values = (
            target["series_id"],
            target["family"],
            target["transform"],
            target["default_scale"],
        )
        for row in rows.itertuples(index=False):
            actual_values = (
                str(row.series_id),
                str(row.family),
                str(row.transform),
                float(row.default_scale),
            )
            if actual_values[:3] != expected_values[:3] or not math.isclose(
                actual_values[3], expected_values[3], rel_tol=0.0, abs_tol=1e-12
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} target metadata differs from its locked contract"
                )
    if set(metadata["family"].astype(str)) != set(REQUIRED_FAMILIES):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} joined score surface lacks the required five families"
        )
    numeric = _numeric_frame(
        joined,
        {
            "prediction",
            "target_value",
            "default_scale",
            "origin_visible_denominator_value",
        },
        cohort,
        "joined score surface",
    )
    if (numeric["default_scale"] <= 0).any():
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} target default scales are invalid"
        )
    joined["error"] = numeric["prediction"] - numeric["target_value"]
    joined["scaled_error"] = joined["error"] / numeric["default_scale"]
    joined["scaled_squared_error"] = joined["scaled_error"].pow(2)
    joined["absolute_scaled_error"] = joined["scaled_error"].abs()
    transformed = ~joined["transform"].astype(str).eq("level")
    target_direction = (
        numeric["target_value"].where(
            transformed,
            numeric["target_value"] - numeric["origin_visible_denominator_value"],
        )
    ).map(_sign)
    forecast_direction = (
        numeric["prediction"].where(
            transformed,
            numeric["prediction"] - numeric["origin_visible_denominator_value"],
        )
    ).map(_sign)
    joined["target_direction"] = target_direction
    joined["forecast_direction"] = forecast_direction
    joined["direction_correct"] = target_direction.eq(forecast_direction)


def _target_contract(
    normalized_spec: Mapping[str, Any], manifest: Mapping[str, Any], cohort: str
) -> dict[str, dict[str, Any]]:
    mappings = normalized_spec.get("target_mappings")
    if not isinstance(mappings, list):
        mappings = manifest.get("target_mappings")
    if not isinstance(mappings, list):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} normalized target contract is missing"
        )
    targets: dict[str, dict[str, Any]] = {}
    for row in mappings:
        if not isinstance(row, Mapping):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} normalized target contract is invalid"
            )
        name = str(row.get("target_name", ""))
        family = str(row.get("family", ""))
        series = str(row.get("series_id", ""))
        transform = str(row.get("transform", ""))
        economy_measure = str(row.get("economy_measure", ""))
        economy_transform = str(row.get("economy_transform", ""))
        scale = row.get("default_scale")
        if (
            not name
            or not series
            or family not in REQUIRED_FAMILIES
            or transform not in {"pct_change", "diff", "level"}
            or not economy_measure
            or economy_transform
            not in {
                "pct_change",
                "nominal_pct_change",
                "diff",
                "level",
                "annual_rate_to_monthly_pct",
            }
            or isinstance(scale, bool)
        ):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} normalized target contract is invalid"
            )
        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} normalized target contract is invalid"
            ) from exc
        if not math.isfinite(scale_value) or scale_value <= 0 or name in targets:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} normalized target contract is invalid"
            )
        targets[name] = {
            "series_id": series,
            "family": family,
            "transform": transform,
            "default_scale": scale_value,
            "economy_measure": economy_measure,
            "economy_transform": economy_transform,
        }
    if len(targets) != 10 or {row["family"] for row in targets.values()} != set(
        REQUIRED_FAMILIES
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} normalized target contract must lock ten targets across five families"
        )
    declared_target_count = manifest.get("target_count")
    if declared_target_count is not None and declared_target_count != len(targets):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest target inventory is invalid"
        )
    score_contract = manifest.get("score_origin_contract")
    declared = manifest.get("scored_target_row_count")
    if isinstance(score_contract, Mapping):
        scored_origins = tuple(
            str(value) for value in score_contract.get("scored_origins", [])
        )
        if (
            scored_origins != EXPECTED_SCORE_ORIGINS
            or score_contract.get("scored_origin_count") != len(EXPECTED_SCORE_ORIGINS)
            or score_contract.get("target_rows_per_scored_origin") != len(targets)
            or declared != len(targets) * len(EXPECTED_SCORE_ORIGINS)
        ):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} manifest target inventory is invalid"
            )
    elif declared is not None and declared != len(targets):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} manifest target inventory is invalid"
        )
    return targets


def _all_scores(
    target_scores: pd.DataFrame, macro_scores: Mapping[str, float], cohort: str
) -> dict[str, float]:
    names = target_scores["target_name"].astype(str).str.upper()
    if names.eq("ALL").any():
        all_frame = target_scores[names.eq("ALL")]
        if set(all_frame["candidate"].astype(str)) != {"llm", "adaptive"}:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} ALL score is not a matched pair"
            )
        return {
            str(row["candidate"]): float(row["target_score"])
            for _, row in all_frame.iterrows()
        }
    return {str(candidate): float(value) for candidate, value in macro_scores.items()}


def _mechanism_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return _mechanism_projection(value)


def _mechanism_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        lowered = key.lower()
        if (
            key in MECHANISM_EXCLUDED_KEYS
            or "batch" in lowered
            or "household" in lowered
            or "raw_record" in lowered
            or lowered.endswith("_path")
        ):
            continue
        if isinstance(child, Mapping):
            result[key] = _mechanism_projection(child)
        elif isinstance(child, list):
            result[key] = [
                _mechanism_projection(item) if isinstance(item, Mapping) else item
                for item in child
            ]
        else:
            result[key] = child
    return result


def _actual_lock_values(
    manifest: Mapping[str, Any], normalized_spec: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "provider": _first_value(manifest, normalized_spec, "provider"),
        "model": _first_value(manifest, normalized_spec, "model"),
        "source": _execution_source_tree_sha256(manifest),
        "prompt": _first_locked_value(manifest, PROMPT_LOCK_KEYS),
    }


def _expected_household_count(spec: Mapping[str, Any], cohort: str) -> int:
    lock = _cohort_lock(spec, cohort) or {}
    value = lock.get(
        "household_count", _mapping_value(spec.get("household_counts"), cohort)
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != (81 if cohort == "corrected81" else 200)
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} household count lock is invalid"
        )
    return value


def _expected_batch_counts(spec: Mapping[str, Any], cohort: str) -> dict[str, int]:
    lock = _cohort_lock(spec, cohort) or {}
    value = lock.get("raw_record_batch_counts", lock.get("batch_counts"))
    if value is None:
        value = spec.get("raw_record_batch_counts")
    if not isinstance(value, Mapping):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record batch count lock is missing"
        )
    if not value or any(
        isinstance(val, bool) or not isinstance(val, int) for val in value.values()
    ):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record batch count lock is invalid"
        )
    result = {str(key): int(val) for key, val in value.items()}
    if any(val < 1 for val in result.values()):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record batch count lock is invalid"
        )
    return result


def _cohort_lock(spec: Mapping[str, Any], cohort: str) -> Mapping[str, Any] | None:
    runs = spec.get("runs")
    if isinstance(runs, Mapping):
        for alias in COHORT_SPEC_ALIASES[cohort]:
            value = runs.get(alias)
            if isinstance(value, Mapping):
                return value
    for alias in COHORT_SPEC_ALIASES[cohort]:
        value = spec.get(alias)
        if isinstance(value, Mapping):
            return value
    return None


def _reject_confirmatory_labels(value: Any, label: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        negated = any(
            marker in lowered
            for marker in (
                "not_confirmatory",
                "non_confirmatory",
                "not confirmatory",
                "not-confirmatory",
            )
        )
        if "confirmatory" in lowered and not negated:
            raise DynamicMacroHouseholdScaleError(
                f"{label} contains a confirmatory label"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_confirmatory_labels(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_confirmatory_labels(child, label)


def _contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle.lower() in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_string(child, needle) for child in value.values())
    if isinstance(value, list):
        return any(_contains_string(child, needle) for child in value)
    return False


def _contains_false(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            (name == key and child is False) or _contains_false(child, key)
            for name, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_false(child, key) for child in value)
    return False


def _named_bools(value: Any, names: set[str]) -> list[bool]:
    if isinstance(value, Mapping):
        found = [
            child
            for name, child in value.items()
            if name in names and isinstance(child, bool)
        ]
        return found + [
            item for child in value.values() for item in _named_bools(child, names)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _named_bools(child, names)]
    return []


def _named_ints(value: Any, names: set[str]) -> list[int]:
    if isinstance(value, Mapping):
        found = [
            int(child)
            for name, child in value.items()
            if name in names and isinstance(child, int) and not isinstance(child, bool)
        ]
        return found + [
            item for child in value.values() for item in _named_ints(child, names)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _named_ints(child, names)]
    return []


def _first_int(value: Mapping[str, Any], *names: str) -> int | None:
    found = _named_ints(value, set(names))
    return found[0] if found else None


def _expected_value(value: Mapping[str, Any], key: str) -> Any:
    return value.get(key)


def _has_any(value: Mapping[str, Any], *keys: str) -> bool:
    return any(value.get(key) is not None for key in keys)


def _first_locked_value(value: Mapping[str, Any], keys: Iterable[str]) -> Any:
    keys = tuple(keys)
    for key in keys:
        if key in value:
            return value[key]
    for group in (
        "source",
        "source_lock",
        "execution_source",
        "prompt",
        "prompt_lock",
        "prompt_contract",
    ):
        child = value.get(group)
        if isinstance(child, Mapping):
            for key in keys:
                if key in child:
                    return child[key]
            if "sha256" in child and any("sha" in key for key in keys):
                return child["sha256"]
            if "version" in child and "prompt_version" in keys:
                return child["version"]
    return None


def _execution_source_tree_sha256(value: Mapping[str, Any]) -> str | None:
    direct = value.get("execution_source_tree_sha256")
    if _is_sha256(direct):
        return direct
    execution_source = value.get("execution_source")
    if isinstance(execution_source, Mapping):
        tree = execution_source.get("tree_sha256")
        if _is_sha256(tree):
            return tree
    return None


def _attempt_cache_path(
    root: Path, value: Any, raw_cache_path: Any, cohort: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accepted live-attempt cache path is invalid"
        )
    if raw_cache_path is None:
        path = (root / value).resolve()
    elif not isinstance(raw_cache_path, str) or not raw_cache_path:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} raw-record cache path is invalid"
        )
    else:
        raw_path = Path(raw_cache_path).expanduser()
        path = (
            raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        ).resolve()
        if path.name != value:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} raw-record cache path does not match its live attempt"
            )
    if root != path and root not in path.parents:
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} accepted live-attempt cache path escapes its run"
        )
    return path


def _validate_prompt_cards(
    cards: pd.DataFrame,
    raw_records: list[dict[str, Any]],
    household_ids: tuple[str, ...],
    cohort: str,
) -> None:
    required = {
        "candidate",
        "period_index",
        "batch_index",
        "batch_count",
        "household_type_ids",
        "household_type_ids_sha256",
        "prompt_payload_sha256",
        "prompt_payload",
    }
    if not required.issubset(cards.columns):
        raise DynamicMacroHouseholdScaleError(f"{cohort} prompt cards are incomplete")
    llm = cards[cards["candidate"].astype(str).eq("llm")].copy()
    if llm.empty:
        raise DynamicMacroHouseholdScaleError(f"{cohort} LLM prompt cards are missing")
    by_batch: dict[tuple[int, int], dict[str, Any]] = {}
    expected_ids = set(household_ids)
    for row in llm.to_dict(orient="records"):
        try:
            period = int(row["period_index"])
            batch = int(row["batch_index"])
            batch_count = int(row["batch_count"])
            ids = _json_string_list(row["household_type_ids"])
            payload = _json_object(row["prompt_payload"])
        except (SyntaxError, TypeError, ValueError) as exc:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} prompt card is malformed"
            ) from exc
        key = (period, batch)
        if (
            period not in range(5)
            or batch < 0
            or batch >= batch_count
            or key in by_batch
            or ids != sorted(ids)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(expected_ids)
        ):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM prompt-card batch identity is invalid"
            )
        if row.get("household_type_ids_sha256") != _canonical_sha(ids) or row.get(
            "prompt_payload_sha256"
        ) != _canonical_sha(payload):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM prompt-card hashes do not reproduce"
            )
        _reject_prompt_weight_leakage(payload, cohort)
        by_batch[key] = {
            "ids": ids,
            "batch_count": batch_count,
            "prompt_sha": row["prompt_payload_sha256"],
        }
    raw_by_batch = {
        (int(record["period_index"]), int(record.get("batch_index", 0))): record
        for record in raw_records
        if record.get("candidate") == "llm_belief"
    }
    if len(raw_by_batch) != sum(
        record.get("candidate") == "llm_belief" for record in raw_records
    ) or set(by_batch) != set(raw_by_batch):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} LLM prompt cards do not match raw-record batches"
        )
    for period in range(5):
        rows = [
            (batch, value)
            for (row_period, batch), value in by_batch.items()
            if row_period == period
        ]
        if not rows:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM prompt cards omit a period"
            )
        counts = {value["batch_count"] for _, value in rows}
        if len(counts) != 1 or sorted(batch for batch, _ in rows) != list(
            range(next(iter(counts)))
        ):
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM prompt-card batch layout is incomplete"
            )
        covered: set[str] = set()
        for batch, value in rows:
            record = raw_by_batch[(period, batch)]
            if (
                covered.intersection(value["ids"])
                or value["ids"] != record.get("household_type_ids")
                or value["prompt_sha"] != record.get("prompt_payload_sha256")
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} LLM prompt cards do not bind raw-record identities"
                )
            covered.update(value["ids"])
        if covered != expected_ids:
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} LLM prompt cards do not cover the cohort"
            )


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) and item for item in parsed
    ):
        raise ValueError("not a non-empty string list")
    return parsed


def _json_object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("not an object")
    return parsed


def _reject_prompt_weight_leakage(value: Any, cohort: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            normalized = _weight_alias_token(lowered)
            if (
                lowered in FORBIDDEN_PROMPT_KEYS
                or lowered.startswith(FORBIDDEN_PROMPT_PREFIXES)
                or _contains_weight_alias(normalized)
            ):
                raise DynamicMacroHouseholdScaleError(
                    f"{cohort} LLM prompt contains forbidden target or weight data"
                )
            _reject_prompt_weight_leakage(child, cohort)
    elif isinstance(value, list):
        for child in value:
            _reject_prompt_weight_leakage(child, cohort)
    elif isinstance(value, str) and _contains_weight_alias(_weight_alias_token(value)):
        raise DynamicMacroHouseholdScaleError(
            f"{cohort} LLM prompt contains forbidden target or weight data"
        )


def _weight_alias_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _contains_weight_alias(value: str) -> bool:
    aliases = (
        "populationweight",
        "popweight",
        "popwgt",
        "surveyweight",
        "surveywgt",
        "sampleweight",
        "samplewgt",
        "samplingweight",
        "selectionweight",
        "respondentweight",
        "personweight",
        "adjustedweight",
        "normalizedweight",
        "normalisedweight",
        "poststratificationweight",
        "rakingweight",
        "finalweight",
        "finalwgt",
    )
    return any(alias in value for alias in aliases)


def _numeric_frame(
    frame: pd.DataFrame, columns: Iterable[str], cohort: str, label: str
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not result[column].map(math.isfinite).all():
            raise DynamicMacroHouseholdScaleError(
                f"{cohort} {label} column {column} is invalid"
            )
    return result


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _first_value(first: Mapping[str, Any], second: Mapping[str, Any], key: str) -> Any:
    return first.get(key, second.get(key))


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _lookup_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DynamicMacroHouseholdScaleError(f"{label} must be a JSON object")
    return value


def _read_list(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DynamicMacroHouseholdScaleError(f"{label} must be a JSON list of objects")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DynamicMacroHouseholdScaleError(f"Cannot read {label}: {path}") from exc


def _canonical_obj(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
