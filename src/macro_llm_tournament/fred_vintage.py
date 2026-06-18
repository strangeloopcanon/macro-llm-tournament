from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .env import load_secret_env
from .forecast_data import quarter_index
from .llm_common import LLMUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "fred_vintage"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_DOC_URL = "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"


@dataclass(frozen=True)
class VintageSeriesSpec:
    series_id: str
    label: str
    units: str
    frequency: str
    transform_hint: str


VINTAGE_SERIES: tuple[VintageSeriesSpec, ...] = (
    VintageSeriesSpec("CPIAUCSL", "CPI level", "index 1982-1984=100", "monthly", "inflation pressure"),
    VintageSeriesSpec("UNRATE", "unemployment rate", "percent", "monthly", "labor market slack"),
    VintageSeriesSpec("GDPC1", "real GDP", "billions chained 2017 dollars", "quarterly", "real activity"),
    VintageSeriesSpec("PCECC96", "real personal consumption", "billions chained 2017 dollars", "monthly", "demand"),
    VintageSeriesSpec("FEDFUNDS", "effective federal funds rate", "percent", "monthly", "policy rate"),
    VintageSeriesSpec("TB3MS", "3-month Treasury bill", "percent", "monthly", "short rate"),
    VintageSeriesSpec("DGS10", "10-year Treasury yield", "percent", "daily", "long rate"),
)


def approximate_spf_as_of_date(origin: str) -> str:
    year_text, quarter_text = origin.replace(" ", "").split(":Q")
    year = int(year_text)
    quarter = int(quarter_text)
    month = {1: 2, 2: 5, 3: 8, 4: 11}[quarter]
    return date(year, month, 15).isoformat()


def observation_start_for_origin(origin: str, quarters_back: int = 12) -> str:
    idx = quarter_index(origin) - int(quarters_back)
    year = idx // 4
    quarter = idx - year * 4
    if quarter == 0:
        year -= 1
        quarter = 4
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    return date(year, month, 1).isoformat()


def build_vintage_context_for_cards(
    cards: Iterable[Any],
    *,
    work_dir: Path = WORK_ROOT,
    refresh: bool = False,
    mode: str = "best_effort",
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    if mode not in {"off", "best_effort", "require"}:
        raise ValueError(f"Unknown vintage context mode: {mode}")
    cards = list(cards)
    if mode == "off":
        return {}, pd.DataFrame(), {"status": "off", "series_count": 0, "card_count": len(cards)}

    load_secret_env()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        status = {
            "status": "missing_api_key",
            "required_env": "FRED_API_KEY",
            "card_count": len(cards),
            "series_count": len(VINTAGE_SERIES),
            "source_url": FRED_DOC_URL,
        }
        if mode == "require":
            raise LLMUnavailable("FRED_API_KEY is required for ALFRED/FRED vintage context.")
        contexts = {
            card.card_id: {
                "status": "missing_api_key",
                "source": "FRED/ALFRED real-time observations API",
                "source_url": FRED_DOC_URL,
                "required_env": "FRED_API_KEY",
                "as_of_date": approximate_spf_as_of_date(card.origin),
                "series": [],
            }
            for card in cards
        }
        return contexts, pd.DataFrame(), status

    work_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    cache_hits = 0
    fetches = 0
    cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for card in cards:
        as_of_date = approximate_spf_as_of_date(card.origin)
        observation_start = observation_start_for_origin(card.origin)
        series_context: list[dict[str, Any]] = []
        for spec in VINTAGE_SERIES:
            key = (spec.series_id, as_of_date, observation_start, as_of_date)
            if key in cache:
                frame = cache[key]
            else:
                frame, was_cache_hit = fetch_vintage_observations(
                    spec,
                    as_of_date=as_of_date,
                    observation_start=observation_start,
                    observation_end=as_of_date,
                    api_key=api_key,
                    work_dir=work_dir,
                    refresh=refresh,
                )
                cache[key] = frame
                cache_hits += int(was_cache_hit)
                fetches += int(not was_cache_hit)
            series_context.append(summarize_vintage_series(frame, spec))
            for _, row in frame.iterrows():
                raw_rows.append(
                    {
                        "card_id": card.card_id,
                        "origin": card.origin,
                        "as_of_date": as_of_date,
                        "series_id": spec.series_id,
                        "label": spec.label,
                        "observation_date": row["date"],
                        "value": row["value"],
                        "realtime_start": row.get("realtime_start"),
                        "realtime_end": row.get("realtime_end"),
                    }
                )
        contexts[card.card_id] = {
            "status": "ok",
            "source": "FRED/ALFRED real-time observations API",
            "source_url": FRED_DOC_URL,
            "as_of_date": as_of_date,
            "observation_start": observation_start,
            "series": series_context,
        }
    return (
        contexts,
        pd.DataFrame(raw_rows),
        {
            "status": "ok",
            "card_count": len(cards),
            "series_count": len(VINTAGE_SERIES),
            "api_cache_hits": cache_hits,
            "api_fetches": fetches,
            "source_url": FRED_DOC_URL,
        },
    )


def fetch_vintage_observations(
    spec: VintageSeriesSpec,
    *,
    as_of_date: str,
    observation_start: str,
    observation_end: str,
    api_key: str,
    work_dir: Path = WORK_ROOT,
    refresh: bool = False,
) -> tuple[pd.DataFrame, bool]:
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_name = _cache_name(spec.series_id, as_of_date, observation_start, observation_end)
    path = work_dir / f"{cache_name}.json"
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _observations_to_frame(payload.get("observations", [])), True
    params = {
        "series_id": spec.series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": as_of_date,
        "realtime_end": as_of_date,
        "observation_start": observation_start,
        "observation_end": observation_end,
        "sort_order": "asc",
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return _observations_to_frame(payload.get("observations", [])), False


def summarize_vintage_series(frame: pd.DataFrame, spec: VintageSeriesSpec) -> dict[str, Any]:
    valid = frame[frame["value"].map(np.isfinite)].sort_values("date")
    if valid.empty:
        return {
            "series_id": spec.series_id,
            "label": spec.label,
            "status": "empty",
            "units": spec.units,
            "frequency": spec.frequency,
            "transform_hint": spec.transform_hint,
        }
    values = valid["value"].astype(float).to_numpy()
    dates = valid["date"].astype(str).to_numpy()
    last = float(values[-1])
    previous = float(values[-2]) if len(values) >= 2 else np.nan
    previous_4 = float(values[-5]) if len(values) >= 5 else np.nan
    pct_change_1 = 100.0 * (last / previous - 1.0) if np.isfinite(previous) and previous != 0 else np.nan
    pct_change_4 = 100.0 * (last / previous_4 - 1.0) if np.isfinite(previous_4) and previous_4 != 0 else np.nan
    return {
        "series_id": spec.series_id,
        "label": spec.label,
        "status": "ok",
        "units": spec.units,
        "frequency": spec.frequency,
        "transform_hint": spec.transform_hint,
        "last_observation_date": str(dates[-1]),
        "last_value": _round_or_none(last),
        "previous_observation_date": str(dates[-2]) if len(dates) >= 2 else None,
        "previous_value": _round_or_none(previous),
        "one_period_pct_change": _round_or_none(pct_change_1),
        "four_period_pct_change": _round_or_none(pct_change_4),
        "last_4_observations": [
            {"date": str(row["date"]), "value": _round_or_none(row["value"])}
            for _, row in valid.tail(4).iterrows()
        ],
    }


def _observations_to_frame(observations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in observations:
        value = pd.to_numeric(row.get("value"), errors="coerce")
        rows.append(
            {
                "date": row.get("date"),
                "value": float(value) if np.isfinite(value) else np.nan,
                "realtime_start": row.get("realtime_start"),
                "realtime_end": row.get("realtime_end"),
            }
        )
    return pd.DataFrame(rows, columns=["date", "value", "realtime_start", "realtime_end"])


def _cache_name(series_id: str, as_of_date: str, observation_start: str, observation_end: str) -> str:
    payload = json.dumps(
        {
            "series_id": series_id,
            "as_of_date": as_of_date,
            "observation_start": observation_start,
            "observation_end": observation_end,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)
