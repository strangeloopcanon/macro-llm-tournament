from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .fred_vintage import approximate_spf_as_of_date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "survey_beliefs"
MICHIGAN_MICH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MICH"
MICHIGAN_MICH_PAGE_URL = "https://fred.stlouisfed.org/series/MICH"
SCE_CHART_DATA_URL = (
    "https://www.newyorkfed.org/medialibrary/interactives/sce/sce/downloads/data/"
    "frbny-sce-data.xlsx?sc_lang=en"
)
SCE_PAGE_URL = "https://www.newyorkfed.org/microeconomics/sce"


def load_survey_belief_targets(
    *,
    work_dir: Path = WORK_ROOT,
    refresh: bool = False,
    mode: str = "best_effort",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mode not in {"off", "best_effort", "require"}:
        raise ValueError(f"Unknown survey belief mode: {mode}")
    if mode == "off":
        return pd.DataFrame(), {"status": "off", "target_count": 0}
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    try:
        frames.append(load_michigan_inflation_expectations(work_dir=work_dir, refresh=refresh))
    except Exception as exc:  # pragma: no cover - exercised only on network/source failure
        errors.append(f"michigan_mich: {exc}")
    try:
        frames.append(load_sce_chart_targets(work_dir=work_dir, refresh=refresh))
    except Exception as exc:  # pragma: no cover - exercised only on network/source failure
        errors.append(f"sce_chart_data: {exc}")
    if errors and mode == "require":
        raise RuntimeError("; ".join(errors))
    frame = pd.concat([part for part in frames if not part.empty], ignore_index=True) if frames else pd.DataFrame()
    status = "ok" if not frame.empty and not errors else ("partial" if not frame.empty else "unavailable")
    return frame, {"status": status, "target_count": int(frame.shape[0]), "errors": errors}


def load_michigan_inflation_expectations(*, work_dir: Path = WORK_ROOT, refresh: bool = False) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "michigan_mich.csv"
    if not path.exists() or refresh:
        response = requests.get(MICHIGAN_MICH_URL, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    raw = pd.read_csv(StringIO(path.read_text(encoding="utf-8")))
    raw["date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["MICH"].replace(".", np.nan), errors="coerce")
    frame = raw.dropna(subset=["date", "value"]).copy()
    frame["survey_source"] = "michigan_survey_of_consumers"
    frame["target_name"] = "median_expected_price_change_next_12_months"
    frame["horizon_months"] = 12
    frame["units"] = "percent"
    frame["source_url"] = MICHIGAN_MICH_PAGE_URL
    return frame[
        ["survey_source", "target_name", "date", "horizon_months", "value", "units", "source_url"]
    ].reset_index(drop=True)


def load_sce_chart_targets(*, work_dir: Path = WORK_ROOT, refresh: bool = False) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "frbny_sce_chart_data.xlsx"
    if not path.exists() or refresh:
        response = requests.get(SCE_CHART_DATA_URL, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
    inflation = pd.read_excel(BytesIO(path.read_bytes()), sheet_name="Inflation expectations", header=3)
    uncertainty = pd.read_excel(BytesIO(path.read_bytes()), sheet_name="Inflation uncertainty", header=3)
    frames = [
        _sce_long_frame(
            inflation,
            {
                "Median one-year ahead expected inflation rate": ("median_expected_inflation", 12),
                "Median three-year ahead expected inflation rate": ("median_expected_inflation", 36),
                "25th Percentile one-year ahead expected inflation rate": ("p25_expected_inflation", 12),
                "75th Percentile one-year ahead expected inflation rate": ("p75_expected_inflation", 12),
                "Median point prediction one-year ahead inflation rate": ("median_point_prediction_inflation", 12),
            },
        ),
        _sce_long_frame(
            uncertainty,
            {
                "Median one-year ahead uncertainty": ("median_inflation_uncertainty", 12),
                "Median three-year ahead uncertainty": ("median_inflation_uncertainty", 36),
            },
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def survey_context_by_card(cards: Iterable[Any], targets: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    cards = list(cards)
    if targets.empty:
        contexts = {
            card.card_id: {
                "status": "unavailable",
                "source": "Michigan/SCE household belief surveys",
                "latest_observations": [],
            }
            for card in cards
        }
        return contexts, pd.DataFrame()
    target_frame = targets.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    for card in cards:
        as_of_date = pd.Timestamp(approximate_spf_as_of_date(card.origin))
        latest_observations: list[dict[str, Any]] = []
        for (source, target_name, horizon), group in target_frame.groupby(
            ["survey_source", "target_name", "horizon_months"], dropna=False
        ):
            eligible = group[group["date"] <= as_of_date].sort_values("date")
            if eligible.empty:
                continue
            latest = eligible.iloc[-1]
            row = {
                "card_id": card.card_id,
                "origin": card.origin,
                "as_of_date": as_of_date.date().isoformat(),
                "variable": card.variable,
                "survey_source": source,
                "target_name": target_name,
                "horizon_months": int(horizon),
                "latest_date": latest["date"].date().isoformat(),
                "value": float(latest["value"]),
                "units": latest["units"],
                "source_url": latest["source_url"],
            }
            rows.append(row)
            latest_observations.append(
                {
                    "survey_source": source,
                    "target_name": target_name,
                    "horizon_months": int(horizon),
                    "latest_date": row["latest_date"],
                    "value": _round_or_none(row["value"]),
                    "units": row["units"],
                }
            )
        contexts[card.card_id] = {
            "status": "ok" if latest_observations else "empty_as_of_date",
            "source": "Michigan/SCE household belief surveys",
            "as_of_date": as_of_date.date().isoformat(),
            "source_urls": sorted({str(row["source_url"]) for row in rows if row["card_id"] == card.card_id}),
            "latest_observations": latest_observations,
            "usage_note": (
                "Survey beliefs are as-of household expectations and diagnostics; they are not the same "
                "target as one-quarter SPF realized outcomes."
            ),
        }
    return contexts, pd.DataFrame(rows)


def _sce_long_frame(raw: pd.DataFrame, columns: dict[str, tuple[str, int]]) -> pd.DataFrame:
    date_col = raw.columns[0]
    frame = raw.rename(columns={date_col: "yyyymm"}).copy()
    frame["yyyymm"] = pd.to_numeric(frame["yyyymm"], errors="coerce")
    frame = frame.dropna(subset=["yyyymm"])
    frame["yyyymm"] = frame["yyyymm"].astype(int).astype(str)
    frame["date"] = pd.to_datetime(frame["yyyymm"] + "01", format="%Y%m%d", errors="coerce")
    rows: list[dict[str, Any]] = []
    for column, (target_name, horizon_months) in columns.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        for idx, value in values.dropna().items():
            rows.append(
                {
                    "survey_source": "ny_fed_sce",
                    "target_name": target_name,
                    "date": frame.loc[idx, "date"],
                    "horizon_months": int(horizon_months),
                    "value": float(value),
                    "units": "percent",
                    "source_url": SCE_PAGE_URL,
                }
            )
    return pd.DataFrame(rows)


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)
