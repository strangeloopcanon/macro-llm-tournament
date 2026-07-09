"""Build prior-update extension inputs with monthly as-of vintage environment cards (v2).

The v1 December 2024 extension reused the quarterly SPF origin card (as-of 2024-11-15),
which fed a monthly survey wave a stale information set and drove the persistence-gate
failure documented in reports/prior_update_environment_v2_prereg.md. This module applies
the pre-registered v2 rule: each SCE wave gets an ALFRED vintage card as-of the 15th of
its own survey month, with real vintage UMCSENT and MICH replacing the synthetic
sentiment and aggregate-inflation-expectation formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .env import load_secret_env
from .fred_vintage import VintageSeriesSpec, fetch_vintage_observations
from .llm_common import LLMUnavailable
from .prepare_sce_microdata import SCE_REAL_TARGET_FIELDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work"
DEFAULT_BASE_EXTENSION = WORK_ROOT / "persona_beliefs" / "sce_2024_12_prior_update_extension_gpt55_input.csv"
DEFAULT_MICRODATA = WORK_ROOT / "persona_beliefs" / "sce_real_microdata.csv"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "persona_beliefs"
MONTHLY_VINTAGE_CACHE = WORK_ROOT / "fred_vintage_panel" / "monthly_vintage_cache"
ENVIRONMENT_PROVENANCE_V2 = "fred_alfred_monthly_vintage_context_by_sce_wave_v2"

MONTHLY_VINTAGE_SERIES: tuple[VintageSeriesSpec, ...] = (
    VintageSeriesSpec("CPIAUCSL", "CPI level", "index 1982-1984=100", "monthly", "inflation pressure"),
    VintageSeriesSpec("UNRATE", "unemployment rate", "percent", "monthly", "labor market slack"),
    VintageSeriesSpec("FEDFUNDS", "effective federal funds rate", "percent", "monthly", "policy rate"),
    VintageSeriesSpec("TB3MS", "3-month Treasury bill", "percent", "monthly", "short rate"),
    VintageSeriesSpec("GDPC1", "real GDP", "billions chained 2017 dollars", "quarterly", "real activity"),
    VintageSeriesSpec("PCECC96", "real personal consumption", "billions chained 2017 dollars", "monthly", "demand"),
    VintageSeriesSpec("MICH", "Michigan median 1y expected inflation", "percent", "monthly", "household inflation expectations"),
    VintageSeriesSpec("UMCSENT", "Michigan consumer sentiment", "index 1966=100", "monthly", "household sentiment"),
)

ENVIRONMENT_COLUMNS = (
    "observed_inflation_1y",
    "observed_unemployment_rate",
    "observed_real_income_growth",
    "policy_rate",
    "sentiment_index",
    "news_inflation_pressure",
    "news_labor_pressure",
    "credit_tightness",
    "aggregate_expected_inflation_1y",
    "aggregate_expected_unemployment_rate",
    "aggregate_expected_real_income_growth",
)

PROFILE_COLUMNS = (
    "age_group",
    "income_group",
    "education_group",
    "gender",
    "region",
    "employment_status",
    "homeownership",
    "liquid_wealth_group",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 prior-update extension inputs with monthly vintage environment cards.")
    parser.add_argument("--base-extension-csv", type=Path, default=DEFAULT_BASE_EXTENSION)
    parser.add_argument("--microdata-csv", type=Path, default=DEFAULT_MICRODATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-vintage", action="store_true", help="Refetch ALFRED vintages instead of using the local cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_secret_env()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise LLMUnavailable("FRED_API_KEY is required to build monthly vintage environment cards.")
    result = build_v2_extension_inputs(
        base_extension_csv=args.base_extension_csv,
        microdata_csv=args.microdata_csv,
        output_dir=args.output_dir,
        api_key=api_key,
        refresh=bool(args.refresh_vintage),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "environments"}, indent=2, sort_keys=True))
    return 0


def monthly_as_of_date(survey_date: str) -> str:
    ts = pd.Timestamp(survey_date)
    return ts.replace(day=15).date().isoformat()


def fetch_monthly_vintage(
    as_of_date: str,
    *,
    api_key: str,
    refresh: bool = False,
    cache_dir: Path = MONTHLY_VINTAGE_CACHE,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    observation_start = (pd.Timestamp(as_of_date) - pd.DateOffset(years=3)).date().isoformat()
    for spec in MONTHLY_VINTAGE_SERIES:
        frame, _cached = fetch_vintage_observations(
            spec,
            as_of_date=as_of_date,
            observation_start=observation_start,
            observation_end=as_of_date,
            api_key=api_key,
            work_dir=cache_dir,
            refresh=refresh,
        )
        frame = frame.copy()
        frame["series_id"] = spec.series_id
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["observation_date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    return merged.dropna(subset=["observation_date", "value"]).copy()


def _latest(vintage: pd.DataFrame, series_id: str, fallback: float | None = None) -> float | None:
    series = vintage[vintage["series_id"].eq(series_id)].sort_values("observation_date")
    if series.empty:
        return fallback
    return float(series.iloc[-1]["value"])


def _yoy_growth(vintage: pd.DataFrame, series_id: str, fallback: float | None = None) -> float | None:
    series = vintage[vintage["series_id"].eq(series_id)].sort_values("observation_date")
    if series.shape[0] < 2:
        return fallback
    latest = series.iloc[-1]
    prior_date = pd.Timestamp(latest["observation_date"]) - pd.DateOffset(months=12)
    prior = series[series["observation_date"].le(prior_date)]
    if prior.empty:
        return fallback
    prior_value = float(prior.iloc[-1]["value"])
    if prior_value <= 0:
        return fallback
    return float(100.0 * (float(latest["value"]) / prior_value - 1.0))


def build_monthly_environment(vintage: pd.DataFrame, *, as_of_date: str) -> dict[str, Any]:
    inflation_yoy = _yoy_growth(vintage, "CPIAUCSL")
    if inflation_yoy is None:
        raise ValueError(f"Monthly vintage as-of {as_of_date} has no usable CPI observations")
    unemployment = _latest(vintage, "UNRATE")
    if unemployment is None:
        raise ValueError(f"Monthly vintage as-of {as_of_date} has no usable UNRATE observations")
    policy = _latest(vintage, "FEDFUNDS", fallback=_latest(vintage, "TB3MS", fallback=3.0))
    real_growth = _yoy_growth(vintage, "GDPC1", fallback=_yoy_growth(vintage, "PCECC96", fallback=1.0))
    michigan_expected_inflation = _latest(vintage, "MICH")
    sentiment = _latest(vintage, "UMCSENT")
    if michigan_expected_inflation is None:
        raise ValueError(f"Monthly vintage as-of {as_of_date} has no usable MICH observations")
    if sentiment is None:
        raise ValueError(f"Monthly vintage as-of {as_of_date} has no usable UMCSENT observations")
    return {
        "as_of_date": as_of_date,
        "observed_inflation_1y": round(float(inflation_yoy), 4),
        "observed_unemployment_rate": round(float(unemployment), 4),
        "observed_real_income_growth": round(float(real_growth), 4),
        "policy_rate": round(float(policy), 4),
        "sentiment_index": round(float(sentiment), 4),
        "news_inflation_pressure": round(float((inflation_yoy - 2.5) / 3.0), 4),
        "news_labor_pressure": round(float((unemployment - 4.5) / 2.5), 4),
        "credit_tightness": round(float(np.clip(0.22 + policy / 12.0 + max(unemployment - 4.5, 0.0) / 20.0, 0.05, 0.95)), 4),
        "aggregate_expected_inflation_1y": round(float(michigan_expected_inflation), 4),
        "aggregate_expected_unemployment_rate": round(float(np.clip(unemployment + 0.25 * (unemployment - 4.5), 0.0, 30.0)), 4),
        "aggregate_expected_real_income_growth": round(float(np.clip(0.40 * real_growth - 0.15 * max(unemployment - 4.5, 0.0), -8.0, 8.0)), 4),
        "environment_provenance": ENVIRONMENT_PROVENANCE_V2,
    }


def apply_environment(frame: pd.DataFrame, environment: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column in ENVIRONMENT_COLUMNS:
        out[column] = environment[column]
    out["environment_provenance"] = ENVIRONMENT_PROVENANCE_V2
    return out


def build_december_v2_input(base_extension: pd.DataFrame, environment: dict[str, Any]) -> pd.DataFrame:
    return apply_environment(base_extension, environment)


def build_january_v2_input(
    base_extension: pd.DataFrame,
    microdata: pd.DataFrame,
    environment: dict[str, Any],
    *,
    january_wave: str = "2025-01-01",
    period_id: str = "sce_2025_01",
    period_index: int = 3,
) -> pd.DataFrame:
    id_map = base_extension.set_index("sce_raw_userid")["respondent_id"].to_dict()
    december_actuals = base_extension.set_index("sce_raw_userid")
    january = microdata[microdata["survey_date"].astype(str).eq(january_wave)].copy()
    january = january[january["sce_raw_userid"].isin(id_map)].copy()
    target_columns = [f"actual_{target}" for target in SCE_REAL_TARGET_FIELDS]
    january = january.dropna(subset=target_columns)
    if january.empty:
        raise ValueError("No continuing respondents with complete January 2025 answers")
    rows: list[dict[str, Any]] = []
    for _, respondent in january.sort_values("sce_raw_userid").iterrows():
        userid = respondent["sce_raw_userid"]
        respondent_id = str(id_map[userid])
        december_row = december_actuals.loc[userid]
        out: dict[str, Any] = {
            "respondent_id": respondent_id,
            "survey_source": "ny_fed_sce_microdata",
            "survey_date": january_wave,
            "weight": float(respondent.get("weight", 1.0)),
        }
        for column in PROFILE_COLUMNS:
            out[column] = respondent.get(column, "unknown")
        for column in target_columns:
            out[column] = float(respondent[column])
        for column in ("sce_nominal_income_growth", "sce_question_unemployment_higher_prob", "sce_raw_userid", "sce_raw_date"):
            if column in respondent:
                out[column] = respondent[column]
        out.update(
            {
                "period_id": period_id,
                "period_index": int(period_index),
                "panel_row_id": f"{respondent_id}__{period_id}",
                "persona_panel_kind": str(december_row["persona_panel_kind"]),
                "target_provenance": str(december_row["target_provenance"]),
                "environment_provenance": ENVIRONMENT_PROVENANCE_V2,
            }
        )
        for target in SCE_REAL_TARGET_FIELDS:
            out[f"prior_{target}"] = float(december_row[f"actual_{target}"])
        rows.append(out)
    frame = pd.DataFrame(rows)
    weights = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["weight"] = weights / weights.sum() if float(weights.sum()) > 0 else 1.0 / max(1, frame.shape[0])
    frame = apply_environment(frame, environment)
    return frame[list(base_extension.columns)]


def build_v2_extension_inputs(
    *,
    base_extension_csv: Path,
    microdata_csv: Path,
    output_dir: Path,
    api_key: str,
    refresh: bool = False,
) -> dict[str, Any]:
    if not base_extension_csv.exists():
        raise FileNotFoundError(f"Base extension CSV not found: {base_extension_csv}")
    if not microdata_csv.exists():
        raise FileNotFoundError(f"SCE microdata CSV not found: {microdata_csv}")
    base_extension = pd.read_csv(base_extension_csv)
    microdata = pd.read_csv(microdata_csv)

    december_wave = str(base_extension["survey_date"].iloc[0])
    december_as_of = monthly_as_of_date(december_wave)
    january_as_of = monthly_as_of_date("2025-01-01")

    december_vintage = fetch_monthly_vintage(december_as_of, api_key=api_key, refresh=refresh)
    january_vintage = fetch_monthly_vintage(january_as_of, api_key=api_key, refresh=refresh)
    december_environment = build_monthly_environment(december_vintage, as_of_date=december_as_of)
    january_environment = build_monthly_environment(january_vintage, as_of_date=january_as_of)

    december_input = build_december_v2_input(base_extension, december_environment)
    january_input = build_january_v2_input(base_extension, microdata, january_environment)

    output_dir.mkdir(parents=True, exist_ok=True)
    december_path = output_dir / "sce_2024_12_prior_update_extension_v2_input.csv"
    january_path = output_dir / "sce_2025_01_prior_update_extension_v2_input.csv"
    december_input.to_csv(december_path, index=False)
    january_input.to_csv(january_path, index=False)

    manifest = {
        "schema_version": "prior_update_extension_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": "reports/prior_update_environment_v2_prereg.md",
        "environment_provenance": ENVIRONMENT_PROVENANCE_V2,
        "monthly_as_of_rule": "15th of each SCE survey month",
        "environments": {"sce_2024_12": december_environment, "sce_2025_01": january_environment},
        "series": [spec.series_id for spec in MONTHLY_VINTAGE_SERIES],
        "inputs": {
            "base_extension_csv": {"path": str(base_extension_csv), "sha256": _file_sha256(base_extension_csv)},
            "microdata_csv": {"path": str(microdata_csv), "sha256": _file_sha256(microdata_csv)},
        },
        "outputs": {
            "december_v2_input": {
                "path": str(december_path),
                "sha256": _file_sha256(december_path),
                "rows": int(december_input.shape[0]),
                "claim_scope": "diagnostic_information_set_fix",
            },
            "january_v2_input": {
                "path": str(january_path),
                "sha256": _file_sha256(january_path),
                "rows": int(january_input.shape[0]),
                "claim_scope": "fresh_prior_update_gate_leg",
            },
        },
    }
    manifest_path = output_dir / "prior_update_extension_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "december_input": str(december_path),
        "january_input": str(january_path),
        "manifest": str(manifest_path),
        "december_rows": int(december_input.shape[0]),
        "january_rows": int(january_input.shape[0]),
        "environments": manifest["environments"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
