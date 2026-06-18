from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .forecast_cards import ForecastCard, finite


CONTROL_SOURCES = [
    "spf_consensus",
    "official_iar",
    "official_no_change",
    "official_dar",
    "official_darm",
    "no_change",
    "rolling_mean_4",
    "rolling_mean_8",
    "ar1",
    "ar2",
    "recursive_least_squares",
    "constant_gain",
    "extrapolative",
    "diagnostic",
]


def _series_for(spf_data: pd.DataFrame, variable: str, horizon: int) -> pd.DataFrame:
    frame = spf_data[
        (spf_data["variable"] == variable)
        & (spf_data["horizon"] == horizon)
        & (spf_data["spf_forecast"].map(finite))
    ].copy()
    return frame.sort_values("origin_index").reset_index(drop=True)


def _prior_signal_values(series: pd.DataFrame, origin_index: int) -> np.ndarray:
    values = series[series["origin_index"] < origin_index]["spf_forecast"].astype(float).to_numpy()
    return values[np.isfinite(values)]


def _ols_predict(y: np.ndarray, lags: int) -> float:
    if len(y) <= lags + 2:
        return float(y[-1]) if len(y) else float("nan")
    rows: list[list[float]] = []
    targets: list[float] = []
    for idx in range(lags, len(y)):
        rows.append([1.0, *[float(y[idx - lag]) for lag in range(1, lags + 1)]])
        targets.append(float(y[idx]))
    x = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    try:
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        latest = np.asarray([1.0, *[float(y[-lag]) for lag in range(1, lags + 1)]], dtype=float)
        return float(latest @ beta)
    except np.linalg.LinAlgError:
        return float(y[-1])


def _recursive_ls_predict(y: np.ndarray) -> float:
    if len(y) < 10:
        return _ols_predict(y, 1)
    rows: list[list[float]] = []
    targets: list[float] = []
    for idx in range(4, len(y)):
        lag1 = float(y[idx - 1])
        lag2 = float(y[idx - 2])
        mean4 = float(np.mean(y[idx - 4 : idx]))
        trend = float(y[idx - 1] - y[idx - 4])
        rows.append([1.0, lag1, lag2, mean4, trend])
        targets.append(float(y[idx]))
    try:
        beta = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), rcond=None)[0]
        latest = np.asarray([1.0, float(y[-1]), float(y[-2]), float(np.mean(y[-4:])), float(y[-1] - y[-4])])
        return float(latest @ beta)
    except np.linalg.LinAlgError:
        return _ols_predict(y, 2)


def _constant_gain_path(y: np.ndarray, gain: float) -> np.ndarray:
    if len(y) == 0:
        return np.asarray([], dtype=float)
    forecasts = [float(y[0])]
    state = float(y[0])
    for value in y[:-1]:
        state = state + gain * (float(value) - state)
        forecasts.append(float(state))
    return np.asarray(forecasts, dtype=float)


def _tune_grid(
    y: np.ndarray,
    origin_indices: np.ndarray,
    *,
    tune_end_index: int,
    grid: Iterable[float],
    predictor,
) -> float:
    best_param = float(next(iter(grid)))
    best_rmse = float("inf")
    grid_values = list(grid)
    for param in grid_values:
        errors: list[float] = []
        for pos in range(8, len(y)):
            if origin_indices[pos] > tune_end_index:
                continue
            pred = predictor(y[:pos], float(param))
            if np.isfinite(pred):
                errors.append(float(y[pos]) - float(pred))
        if errors:
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            if rmse < best_rmse:
                best_rmse = rmse
                best_param = float(param)
    return best_param


def _constant_gain_predict(y: np.ndarray, origin_indices: np.ndarray, tune_end_index: int) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 19)
    gain = _tune_grid(
        y,
        origin_indices,
        tune_end_index=tune_end_index,
        grid=grid,
        predictor=lambda hist, param: _constant_gain_path(hist, param)[-1],
    )
    path = _constant_gain_path(y, gain)
    return (float(path[-1]) if len(path) else float("nan"), gain)


def _extrapolative_predict(hist: np.ndarray, gamma: float) -> float:
    if len(hist) < 5:
        return float(hist[-1]) if len(hist) else float("nan")
    return float(hist[-1] + gamma * (hist[-1] - hist[-5]))


def _diagnostic_predict(hist: np.ndarray, theta: float) -> float:
    if len(hist) < 8:
        return float(hist[-1]) if len(hist) else float("nan")
    return float(hist[-1] + theta * (hist[-1] - np.mean(hist[-8:])))


def _tuned_recent_rule(
    y: np.ndarray,
    origin_indices: np.ndarray,
    tune_end_index: int,
    *,
    rule: str,
) -> tuple[float, float]:
    if rule == "extrapolative":
        grid = np.linspace(0.0, 1.5, 16)
        func = _extrapolative_predict
    elif rule == "diagnostic":
        grid = np.linspace(0.0, 2.0, 21)
        func = _diagnostic_predict
    else:
        raise ValueError(rule)
    param = _tune_grid(y, origin_indices, tune_end_index=tune_end_index, grid=grid, predictor=func)
    return func(y, param), param


def _official_row(spf_data: pd.DataFrame, card: ForecastCard) -> pd.Series:
    match = spf_data[
        (spf_data["variable"] == card.variable)
        & (spf_data["horizon"] == card.horizon)
        & (spf_data["origin_index"] == card.origin_index)
    ]
    if match.empty:
        raise ValueError(f"No SPF row for card {card.card_id}")
    return match.iloc[0]


def build_control_forecasts(
    spf_data: pd.DataFrame,
    cards: Iterable[ForecastCard],
    *,
    tune_end_year: int = 2014,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tune_end_index = tune_end_year * 4 + 4
    for card in cards:
        series = _series_for(spf_data, card.variable, card.horizon)
        y = _prior_signal_values(series, card.origin_index)
        idx = series[series["origin_index"] < card.origin_index]["origin_index"].astype(int).to_numpy()
        official = _official_row(spf_data, card)
        official_values = {
            "spf_consensus": official.get("spf_forecast"),
            "official_iar": official.get("official_iar_forecast"),
            "official_no_change": official.get("official_no_change_forecast"),
            "official_dar": official.get("official_dar_forecast"),
            "official_darm": official.get("official_darm_forecast"),
        }
        local_values: dict[str, tuple[float, dict[str, Any]]] = {
            "no_change": (float(y[-1]) if len(y) else float("nan"), {}),
            "rolling_mean_4": (float(np.mean(y[-4:])) if len(y) else float("nan"), {}),
            "rolling_mean_8": (float(np.mean(y[-8:])) if len(y) else float("nan"), {}),
            "ar1": (_ols_predict(y, 1), {}),
            "ar2": (_ols_predict(y, 2), {}),
            "recursive_least_squares": (_recursive_ls_predict(y), {}),
        }
        cg_pred, gain = _constant_gain_predict(y, idx, tune_end_index)
        ex_pred, gamma = _tuned_recent_rule(y, idx, tune_end_index, rule="extrapolative")
        diag_pred, theta = _tuned_recent_rule(y, idx, tune_end_index, rule="diagnostic")
        local_values.update(
            {
                "constant_gain": (cg_pred, {"gain": gain}),
                "extrapolative": (ex_pred, {"gamma": gamma}),
                "diagnostic": (diag_pred, {"theta": theta}),
            }
        )
        for source, value in official_values.items():
            if finite(value):
                rows.append(_forecast_row(card, source, float(value), model="official_spf_error_file", params={}))
        for source, (value, params) in local_values.items():
            if finite(value):
                rows.append(_forecast_row(card, source, float(value), model="local_control", params=params))
    return pd.DataFrame(rows)


def _forecast_row(
    card: ForecastCard,
    source: str,
    point_forecast: float,
    *,
    model: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "source": source,
        "provider": "control",
        "model": model,
        "variable": card.variable,
        "origin": card.origin,
        "origin_index": card.origin_index,
        "horizon": card.horizon,
        "point_forecast": float(point_forecast),
        "p10": np.nan,
        "p50": np.nan,
        "p90": np.nan,
        "confidence": np.nan,
        "panel_mean": np.nan,
        "panel_std": np.nan,
        "params_json": json_dumps(params),
        "cache_hit": True,
    }


def json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, sort_keys=True)
