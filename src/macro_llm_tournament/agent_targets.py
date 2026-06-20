from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .forecast_cards import ForecastCard
from .fred_vintage import approximate_spf_as_of_date


HOUSEHOLD_INFLATION_TARGETS = {
    "median_expected_price_change_next_12_months",
    "median_expected_inflation",
    "median_point_prediction_inflation",
}


def build_agent_belief_target_rows(cards: Iterable[ForecastCard], survey_targets: pd.DataFrame) -> pd.DataFrame:
    cards = list(cards)
    if survey_targets.empty:
        return pd.DataFrame()
    targets = survey_targets.copy()
    targets["date"] = pd.to_datetime(targets["date"], errors="coerce")
    targets = targets.dropna(subset=["date", "value"])
    targets = targets[
        (targets["horizon_months"].astype(str) == "12")
        & targets["target_name"].isin(HOUSEHOLD_INFLATION_TARGETS)
    ].copy()
    cards_by_origin: dict[tuple[str, int], list[ForecastCard]] = {}
    for card in cards:
        cards_by_origin.setdefault((card.origin, int(card.horizon)), []).append(card)
    rows: list[dict[str, Any]] = []
    for (origin, horizon), origin_cards in sorted(cards_by_origin.items()):
        as_of = pd.Timestamp(approximate_spf_as_of_date(origin))
        target_end = as_of + pd.DateOffset(months=3 * horizon)
        variables = sorted({card.variable for card in origin_cards})
        for (survey_source, target_name), group in targets.groupby(["survey_source", "target_name"], dropna=False):
            eligible = group[(group["date"] > as_of) & (group["date"] <= target_end)].sort_values("date")
            if eligible.empty:
                continue
            target = eligible.iloc[-1]
            rows.append(
                {
                    "origin": origin,
                    "horizon": horizon,
                    "as_of_date": as_of.date().isoformat(),
                    "target_window_end": target_end.date().isoformat(),
                    "survey_source": survey_source,
                    "target_name": target_name,
                    "target_date": target["date"].date().isoformat(),
                    "target_value": float(target["value"]),
                    "units": target["units"],
                    "source_url": target["source_url"],
                    "card_count": len(origin_cards),
                    "variables": ",".join(variables),
                }
            )
    return pd.DataFrame(rows)


def score_agent_belief_targets(aggregates: pd.DataFrame, belief_targets: pd.DataFrame) -> pd.DataFrame:
    if aggregates.empty or belief_targets.empty:
        return pd.DataFrame()
    predictions = aggregates[
        [
            "source",
            "origin",
            "horizon",
            "variable",
            "aggregate_expected_inflation_1y",
            "aggregate_confidence",
            "aggregate_uncertainty",
        ]
    ].copy()
    origin_predictions = (
        predictions.groupby(["source", "origin", "horizon"], dropna=False)
        .agg(
            aggregate_expected_inflation_1y=("aggregate_expected_inflation_1y", "mean"),
            aggregate_confidence=("aggregate_confidence", "mean"),
            aggregate_uncertainty=("aggregate_uncertainty", "mean"),
            predicted_card_count=("variable", "size"),
            predicted_variables=("variable", lambda values: ",".join(sorted({str(value) for value in values}))),
        )
        .reset_index()
    )
    joined = origin_predictions.merge(belief_targets, on=["origin", "horizon"], how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["error"] = joined["aggregate_expected_inflation_1y"].astype(float) - joined["target_value"].astype(float)
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "survey_source", "target_name"], dropna=False):
        source, survey_source, target_name = keys
        error = group["error"].astype(float)
        rows.append(
            {
                "source": source,
                "survey_source": survey_source,
                "target_name": target_name,
                "n": int(group.shape[0]),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "mean_prediction": float(group["aggregate_expected_inflation_1y"].mean()),
                "mean_target": float(group["target_value"].mean()),
                "mean_confidence": float(group["aggregate_confidence"].mean()),
                "mean_uncertainty": float(group["aggregate_uncertainty"].mean()),
                "mean_predicted_card_count": float(group["predicted_card_count"].mean()),
            }
        )
    return pd.DataFrame(rows)
