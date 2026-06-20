from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd


FORECAST_CARD_PROMPT_VERSION = "spf_forecast_card_v3_vintage_survey"


@dataclass(frozen=True)
class ForecastCard:
    card_id: str
    variable: str
    variable_name: str
    units: str
    origin: str
    origin_year: int
    origin_quarter: int
    origin_index: int
    horizon: int
    target_realized: float
    spf_consensus_forecast: float
    asof_reference_value: float
    prior_spf_forecast: float
    rolling_signal_mean_4: float
    rolling_signal_mean_8: float
    recent_signal_change_4: float
    recent_signal_volatility_8: float
    evaluation_split: str
    regime_label: str
    contamination_label: str
    source_url: str
    variable_page_url: str
    prompt_payload: dict[str, Any]


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _round_float(value: Any, digits: int = 4) -> float | None:
    if not finite(value):
        return None
    return round(float(value), digits)


def _history_rows(history: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in history.iterrows():
        rows.append(
            {
                "origin": row["origin"],
                "spf_consensus_forecast": _round_float(row["spf_forecast"]),
                "official_iar_forecast": _round_float(row.get("official_iar_forecast")),
                "official_no_change_forecast": _round_float(row.get("official_no_change_forecast")),
                "official_dar_forecast": _round_float(row.get("official_dar_forecast")),
                "official_darm_forecast": _round_float(row.get("official_darm_forecast")),
            }
        )
    return rows


def _signal_summary(history: pd.DataFrame) -> dict[str, float | None]:
    spf = pd.to_numeric(history["spf_forecast"], errors="coerce").dropna()
    prior_spf = spf.iloc[-1] if len(spf) else np.nan
    mean4 = spf.tail(4).mean() if len(spf) else np.nan
    mean8 = spf.tail(8).mean() if len(spf) else np.nan
    change4 = spf.iloc[-1] - spf.iloc[-5] if len(spf) >= 5 else np.nan
    vol8 = spf.tail(8).std(ddof=0) if len(spf) >= 2 else np.nan
    return {
        "asof_reference_value": _round_float(prior_spf),
        "prior_spf_consensus_forecast": _round_float(prior_spf),
        "rolling_spf_consensus_mean_4": _round_float(mean4),
        "rolling_spf_consensus_mean_8": _round_float(mean8),
        "recent_spf_consensus_change_4": _round_float(change4),
        "recent_spf_consensus_volatility_8": _round_float(vol8),
    }


def _forecast_signal_summary(history: pd.DataFrame) -> list[dict[str, Any]]:
    benchmarks = [
        ("spf_consensus", "spf_forecast"),
        ("official_iar", "official_iar_forecast"),
        ("official_no_change", "official_no_change_forecast"),
        ("official_dar", "official_dar_forecast"),
        ("official_darm", "official_darm_forecast"),
    ]
    rows: list[dict[str, Any]] = []
    for label, column in benchmarks:
        if column not in history:
            continue
        valid = history[history[column].map(finite)].tail(24)
        if valid.empty:
            continue
        values = valid[column].astype(float)
        recent = valid.tail(8)
        recent_values = recent[column].astype(float)
        rows.append(
            {
                "source": label,
                "n_24": int(valid.shape[0]),
                "last_forecast": _round_float(values.iloc[-1]),
                "mean_24": _round_float(values.mean()),
                "std_24": _round_float(values.std(ddof=0)),
                "mean_8": _round_float(recent_values.mean()) if not recent.empty else None,
                "std_8": _round_float(recent_values.std(ddof=0)) if not recent.empty else None,
            }
        )
    return rows


def _regime_label(origin_year: int) -> str:
    if origin_year <= 2019:
        return "pre_covid_low_rate_expansion"
    if origin_year == 2020:
        return "covid_shock"
    if origin_year in {2021, 2022}:
        return "inflation_surge"
    if origin_year in {2023, 2024}:
        return "high_rate_disinflation"
    if origin_year == 2025:
        return "near_model_cutoff"
    return "post_model_cutoff_candidate"


def _evaluation_split(origin_year: int, holdout_start_year: int, holdout_end_year: int) -> str:
    if origin_year < holdout_start_year:
        return "tuning_pre_holdout"
    if holdout_start_year <= origin_year <= holdout_end_year:
        return f"holdout_{holdout_start_year}_{holdout_end_year}"
    return "future_or_unscored"


def _card_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _contamination_label(origin_year: int, *, model_cutoff_year: int = 2025) -> str:
    if origin_year <= model_cutoff_year:
        return "historical_pre_model_cutoff_confounded"
    return "post_model_cutoff_candidate"


def _cross_variable_context(
    spf_data: pd.DataFrame,
    *,
    origin_index: int,
    horizon: int,
    variables: Iterable[str],
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for variable in variables:
        history = spf_data[
            (spf_data["variable"] == variable)
            & (spf_data["horizon"] == horizon)
            & (spf_data["origin_index"] < origin_index)
            & (spf_data["spf_forecast"].map(finite))
        ].sort_values("origin_index")
        if history.empty:
            continue
        summary = _signal_summary(history)
        context.append(
            {
                "variable": variable,
                "last_origin": history.iloc[-1]["origin"],
                "asof_reference_value": summary["asof_reference_value"],
                "prior_spf_consensus_forecast": summary["prior_spf_consensus_forecast"],
                "rolling_spf_consensus_mean_4": summary["rolling_spf_consensus_mean_4"],
                "recent_spf_consensus_change_4": summary["recent_spf_consensus_change_4"],
                "recent_spf_consensus_volatility_8": summary["recent_spf_consensus_volatility_8"],
            }
        )
    return context


def _card_from_row(
    row: pd.Series,
    history: pd.DataFrame,
    *,
    spf_data: pd.DataFrame,
    variables: Iterable[str],
    holdout_start_year: int,
    holdout_end_year: int,
    history_quarters: int,
) -> ForecastCard | None:
    if history.empty:
        return None
    summary = _signal_summary(history)
    if summary["asof_reference_value"] is None:
        return None
    visible_history = history.tail(history_quarters)
    regime_label = _regime_label(int(row["origin_year"]))
    evaluation_split = _evaluation_split(int(row["origin_year"]), holdout_start_year, holdout_end_year)
    prompt_payload = {
        "prompt_version": FORECAST_CARD_PROMPT_VERSION,
        "task": "Forecast the requested U.S. macroeconomic variable from information available before this SPF survey origin.",
        "as_of_design": {
            "evaluation_split": evaluation_split,
            "regime_label": regime_label,
            "contamination_label": _contamination_label(int(row["origin_year"])),
            "current_origin_spf_consensus_hidden": True,
            "outcome_hidden": True,
            "history_uses_lagged_forecasts_only": True,
            "vintage_macro_context_supported": True,
            "household_survey_beliefs_supported": True,
        },
        "variable": row["variable"],
        "variable_name": row["variable_name"],
        "units": row["units"],
        "survey_origin": row["origin"],
        "forecast_horizon_quarters_ahead": int(row["horizon"]),
        "available_history": _history_rows(visible_history),
        "asof_forecast_signal_summary": summary,
        "historical_forecast_signal_summary": _forecast_signal_summary(history),
        "cross_variable_lagged_context": _cross_variable_context(
            spf_data,
            origin_index=int(row["origin_index"]),
            horizon=int(row["horizon"]),
            variables=variables,
        ),
        "required_response": {
            "point_forecast": "number in the stated units",
            "p10": "10th percentile forecast",
            "p50": "median forecast",
            "p90": "90th percentile forecast",
            "confidence": "0 to 1",
            "forecaster_draws": "8 simulated professional-forecaster point forecasts",
        },
    }
    card_id = _card_hash(prompt_payload)
    return ForecastCard(
        card_id=card_id,
        variable=str(row["variable"]),
        variable_name=str(row["variable_name"]),
        units=str(row["units"]),
        origin=str(row["origin"]),
        origin_year=int(row["origin_year"]),
        origin_quarter=int(row["origin_quarter"]),
        origin_index=int(row["origin_index"]),
        horizon=int(row["horizon"]),
        target_realized=float(row["realized"]),
        spf_consensus_forecast=float(row["spf_forecast"]),
        asof_reference_value=float(summary["asof_reference_value"]),
        prior_spf_forecast=float(summary["prior_spf_consensus_forecast"] or summary["asof_reference_value"]),
        rolling_signal_mean_4=float(summary["rolling_spf_consensus_mean_4"] or summary["asof_reference_value"]),
        rolling_signal_mean_8=float(summary["rolling_spf_consensus_mean_8"] or summary["asof_reference_value"]),
        recent_signal_change_4=float(summary["recent_spf_consensus_change_4"] or 0.0),
        recent_signal_volatility_8=float(summary["recent_spf_consensus_volatility_8"] or 0.0),
        evaluation_split=evaluation_split,
        regime_label=regime_label,
        contamination_label=_contamination_label(int(row["origin_year"])),
        source_url=str(row["source_url"]),
        variable_page_url=str(row["variable_page_url"]),
        prompt_payload=prompt_payload,
    )


def _eligible_rows(
    spf_data: pd.DataFrame,
    *,
    variables: Iterable[str],
    horizons: Iterable[int],
    holdout_start_year: int,
    holdout_end_year: int,
) -> pd.DataFrame:
    variables_set = {str(variable).upper() for variable in variables}
    horizons_set = {int(horizon) for horizon in horizons}
    mask = (
        spf_data["variable"].isin(variables_set)
        & spf_data["horizon"].isin(horizons_set)
        & spf_data["origin_year"].between(holdout_start_year, holdout_end_year)
        & spf_data["realized"].map(finite)
        & spf_data["spf_forecast"].map(finite)
    )
    return spf_data[mask].sort_values(["origin_index", "variable", "horizon"]).copy()


def select_pilot_rows(eligible: pd.DataFrame, card_count: int) -> pd.DataFrame:
    if card_count <= 0 or eligible.shape[0] <= card_count:
        return eligible
    variables = list(dict.fromkeys(str(value) for value in eligible["variable"].tolist()))
    per_variable = max(1, card_count // max(1, len(variables)))
    selected_indices: list[int] = []
    for variable in variables:
        group = eligible[eligible["variable"] == variable].sort_values("origin_index")
        if group.empty:
            continue
        take = min(per_variable, len(group))
        regime_groups = [
            sub_group.sort_values("origin_index")
            for _, sub_group in group.assign(regime=group["origin_year"].map(_regime_label)).groupby("regime", sort=False)
        ]
        per_regime = max(1, take // max(1, len(regime_groups)))
        for sub_group in regime_groups:
            sub_take = min(per_regime, len(sub_group))
            positions = np.linspace(0, len(sub_group) - 1, sub_take).round().astype(int)
            for pos in positions:
                idx = int(sub_group.iloc[pos].name)
                if idx not in selected_indices:
                    selected_indices.append(idx)
        if sum(1 for idx in selected_indices if eligible.loc[idx, "variable"] == variable) < take:
            for idx in group.index:
                idx = int(idx)
                if idx not in selected_indices:
                    selected_indices.append(idx)
                variable_count = sum(1 for selected in selected_indices if eligible.loc[selected, "variable"] == variable)
                if variable_count >= take:
                    break
    if len(selected_indices) < card_count:
        for idx in eligible.index:
            idx = int(idx)
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= card_count:
                break
    return eligible.loc[selected_indices[:card_count]].sort_values(["origin_index", "variable", "horizon"])


def build_forecast_cards(
    spf_data: pd.DataFrame,
    *,
    variables: Iterable[str],
    horizons: Iterable[int],
    holdout_start_year: int = 2015,
    holdout_end_year: int = 2024,
    history_quarters: int = 24,
    card_count: int = 12,
) -> list[ForecastCard]:
    eligible = _eligible_rows(
        spf_data,
        variables=variables,
        horizons=horizons,
        holdout_start_year=holdout_start_year,
        holdout_end_year=holdout_end_year,
    )
    selected = select_pilot_rows(eligible, card_count)
    cards: list[ForecastCard] = []
    for _, row in selected.iterrows():
        history = spf_data[
            (spf_data["variable"] == row["variable"])
            & (spf_data["horizon"] == row["horizon"])
            & (spf_data["origin_index"] < row["origin_index"])
            & (spf_data["spf_forecast"].map(finite))
        ].sort_values("origin_index")
        card = _card_from_row(
            row,
            history,
            spf_data=spf_data,
            variables=variables,
            holdout_start_year=holdout_start_year,
            holdout_end_year=holdout_end_year,
            history_quarters=history_quarters,
        )
        if card is not None:
            cards.append(card)
    return cards


def build_forecast_cards_from_rows(
    spf_data: pd.DataFrame,
    selected_rows: pd.DataFrame,
    *,
    variables: Iterable[str],
    holdout_start_year: int,
    holdout_end_year: int,
    history_quarters: int = 24,
) -> list[ForecastCard]:
    cards: list[ForecastCard] = []
    selected = selected_rows.sort_values(["origin_index", "variable", "horizon"])
    for _, row in selected.iterrows():
        history = spf_data[
            (spf_data["variable"] == row["variable"])
            & (spf_data["horizon"] == row["horizon"])
            & (spf_data["origin_index"] < row["origin_index"])
            & (spf_data["spf_forecast"].map(finite))
        ].sort_values("origin_index")
        card = _card_from_row(
            row,
            history,
            spf_data=spf_data,
            variables=variables,
            holdout_start_year=holdout_start_year,
            holdout_end_year=holdout_end_year,
            history_quarters=history_quarters,
        )
        if card is not None:
            cards.append(card)
    return cards


def cards_to_frame(cards: Iterable[ForecastCard]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        rows.append(
            {
                "card_id": card.card_id,
                "variable": card.variable,
                "variable_name": card.variable_name,
                "units": card.units,
                "origin": card.origin,
                "origin_year": card.origin_year,
                "origin_quarter": card.origin_quarter,
                "origin_index": card.origin_index,
                "horizon": card.horizon,
                "target_realized": card.target_realized,
                "spf_consensus_forecast": card.spf_consensus_forecast,
                "asof_reference_value": card.asof_reference_value,
                "prior_spf_forecast": card.prior_spf_forecast,
                "rolling_signal_mean_4": card.rolling_signal_mean_4,
                "rolling_signal_mean_8": card.rolling_signal_mean_8,
                "recent_signal_change_4": card.recent_signal_change_4,
                "recent_signal_volatility_8": card.recent_signal_volatility_8,
                "evaluation_split": card.evaluation_split,
                "regime_label": card.regime_label,
                "contamination_label": card.contamination_label,
                "source_url": card.source_url,
                "variable_page_url": card.variable_page_url,
                "prompt_payload_json": json.dumps(card.prompt_payload, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def enrich_forecast_cards(
    cards: Iterable[ForecastCard],
    *,
    vintage_context_by_card: dict[str, dict[str, Any]] | None = None,
    survey_context_by_card: dict[str, dict[str, Any]] | None = None,
) -> list[ForecastCard]:
    enriched: list[ForecastCard] = []
    for card in cards:
        prompt_payload = json.loads(json.dumps(card.prompt_payload))
        prompt_payload["prompt_version"] = FORECAST_CARD_PROMPT_VERSION
        if vintage_context_by_card is not None:
            prompt_payload["vintage_macro_context"] = vintage_context_by_card.get(
                card.card_id,
                {"status": "missing", "series": []},
            )
        if survey_context_by_card is not None:
            prompt_payload["household_survey_belief_context"] = survey_context_by_card.get(
                card.card_id,
                {"status": "missing", "latest_observations": []},
            )
        prompt_payload["as_of_design"]["vintage_macro_context_included"] = vintage_context_by_card is not None
        prompt_payload["as_of_design"]["household_survey_beliefs_included"] = survey_context_by_card is not None
        enriched.append(replace(card, card_id=_card_hash(prompt_payload), prompt_payload=prompt_payload))
    return enriched


def assert_no_prompt_target_leakage(cards: Iterable[ForecastCard]) -> None:
    for card in cards:
        prompt = json.dumps(card.prompt_payload, sort_keys=True)
        forbidden_labels = [
            "target_realized",
            "current_spf_consensus_forecast",
            "spf_consensus_forecast_current_card",
            "forecast_error",
        ]
        for label in forbidden_labels:
            if label in prompt:
                raise ValueError(f"Forecast card {card.card_id} leaks {label}")
