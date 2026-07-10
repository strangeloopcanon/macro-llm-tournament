"""Build and validate frozen rolling-origin ALFRED vintage bundles.

The bundle deliberately separates information that was visible at an origin from
the first-release realization and its latest revision.  It is a small,
file-oriented interchange format intended for reproducible forecast evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Protocol

import requests

from .env import load_secret_env


BUNDLE_SCHEMA_VERSION = "frozen_rolling_origin_vintage_bundle_v4"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
FRED_DOCUMENTATION_URL = "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"
DEFAULT_RELEASE_LAG_DAYS = 180
DEFAULT_AS_OF_DAY = 15
HISTORY_MONTHS = 24
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "work" / "frozen_vintage_bundle_cache"


@dataclass(frozen=True)
class TargetSpec:
    target_name: str
    series_id: str
    label: str
    family: str
    transform: str
    default_scale: float
    units: str
    frequency: str = "monthly"


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("pce_growth_pct", "PCE", "personal consumption expenditures growth", "demand", "pct_change", 0.8, "percent"),
    TargetSpec("real_pce_growth_pct", "PCEC96", "real personal consumption expenditures growth", "demand", "pct_change", 0.8, "percent"),
    TargetSpec("retail_sales_growth_pct", "RSAFS", "retail sales growth", "demand", "pct_change", 1.0, "percent"),
    TargetSpec("personal_saving_rate_change", "PSAVERT", "personal saving rate change", "balance_sheet", "diff", 1.5, "percentage points"),
    TargetSpec("revolving_credit_growth_pct", "REVOLSL", "revolving consumer credit growth", "balance_sheet", "pct_change", 2.0, "percent"),
    TargetSpec("payroll_growth_pct", "PAYEMS", "nonfarm payroll growth", "labor", "pct_change", 0.3, "percent"),
    TargetSpec("unemployment_rate_level", "UNRATE", "unemployment rate", "labor", "level", 1.0, "percent"),
    TargetSpec("pce_price_growth_pct", "PCEPI", "PCE price index growth", "prices", "pct_change", 0.3, "percent"),
    TargetSpec("real_disposable_income_growth_pct", "DSPIC96", "real disposable income growth", "income_policy", "pct_change", 0.8, "percent"),
    TargetSpec("fed_funds_rate_level", "FEDFUNDS", "effective federal funds rate", "income_policy", "level", 1.0, "percent"),
)
CONTEXT_SERIES: tuple[str, ...] = ("CPIAUCSL", "UMCSENT")
MODEL_CUTOFFS: dict[str, str] = {
    "gpt-5-codex": "2024-09-30",
    "gpt-5.4": "2025-08-31",
    "gpt-5.5": "2025-12-01",
}

ORIGIN_COLUMNS = ("origin_month", "as_of_date")
HISTORY_COLUMNS = (
    "origin_month",
    "as_of_date",
    "series_id",
    "series_role",
    "observation_date",
    "value",
    "realtime_start",
    "realtime_end",
)
TARGET_COLUMNS = (
    "origin_month",
    "as_of_date",
    "target_name",
    "series_id",
    "family",
    "transform",
    "default_scale",
    "target_observation_date",
    "first_release_as_of_date",
    "release_detection_method",
    "first_release_value",
    "origin_visible_denominator_date",
    "origin_visible_denominator_value",
    "first_release_denominator_date",
    "first_release_denominator_value",
    "target_value",
)
REVISION_AUDIT_COLUMNS = (
    "origin_month",
    "as_of_date",
    "target_name",
    "series_id",
    "target_observation_date",
    "first_release_value",
    "latest_revision_value",
    "latest_minus_first_release",
)
TARGET_CONTAMINATION_COLUMNS = (
    "origin_month",
    "as_of_date",
    "target_name",
    "series_id",
    "target_observation_date",
    "first_release_as_of_date",
    "model",
    "model_cutoff_date",
    "contamination_label",
    "origin_information_label",
)
SOURCE_REQUEST_COLUMNS = (
    "origin_month",
    "as_of_date",
    "series_id",
    "request_kind",
    "realtime_start",
    "realtime_end",
    "observation_start",
    "observation_end",
    "release_detection_method",
)
PAYLOAD_FILES: dict[str, tuple[str, ...]] = {
    "origins.csv": ORIGIN_COLUMNS,
    "history.csv": HISTORY_COLUMNS,
    "targets.csv": TARGET_COLUMNS,
    "revision_audit.csv": REVISION_AUDIT_COLUMNS,
    "target_contamination.csv": TARGET_CONTAMINATION_COLUMNS,
    "source_requests.csv": SOURCE_REQUEST_COLUMNS,
}


class FrozenVintageBundleError(ValueError):
    """Raised when a bundle cannot be trusted."""


@dataclass(frozen=True)
class FrozenVintageBundle:
    root: Path
    manifest: dict[str, Any]
    origins: tuple[dict[str, str], ...]
    history: tuple[dict[str, str], ...]
    targets: tuple[dict[str, str], ...]
    revision_audit: tuple[dict[str, str], ...]
    target_contamination: tuple[dict[str, str], ...]
    source_requests: tuple[dict[str, str], ...]


class ObservationClient(Protocol):
    def observations(
        self,
        series_id: str,
        *,
        observation_start: str,
        observation_end: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def vintage_dates(self, series_id: str, *, realtime_start: str, realtime_end: str) -> list[str]: ...


class AlfredClient:
    """Small cached client that never writes credentials into bundle artifacts."""

    def __init__(self, api_key: str, cache_dir: Path, *, refresh: bool = False) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.refresh = refresh
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _request_json(self, url: str, params: dict[str, str], *, series_id: str, endpoint: str) -> dict[str, Any]:
        for attempt in range(6):
            wait = 0.55 - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            response = requests.get(url, params=params, timeout=45)
            self._last_request_at = time.monotonic()
            if response.ok:
                return dict(response.json())
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(16.0, 1.0 * 2**attempt)
                time.sleep(delay)
                continue
            raise FrozenVintageBundleError(
                f"FRED {endpoint} request failed with HTTP {response.status_code} for series {series_id}"
            )
        raise FrozenVintageBundleError(
            f"FRED {endpoint} request exhausted retries for series {series_id}"
        )

    def observations(
        self,
        series_id: str,
        *,
        observation_start: str,
        observation_end: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = _canonical_json_sha256(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
            }
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not self.refresh:
            return list(json.loads(cache_path.read_text(encoding="utf-8")).get("observations", []))
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "sort_order": "asc",
        }
        if realtime_start is not None:
            params["realtime_start"] = realtime_start
        if realtime_end is not None:
            params["realtime_end"] = realtime_end
        payload = self._request_json(FRED_OBSERVATIONS_URL, params, series_id=series_id, endpoint="observations")
        cache_path.write_text(_canonical_json(payload), encoding="utf-8")
        return list(payload.get("observations", []))

    def vintage_dates(self, series_id: str, *, realtime_start: str, realtime_end: str) -> list[str]:
        cache_key = _canonical_json_sha256(
            {
                "endpoint": "vintagedates",
                "series_id": series_id,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
            }
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not self.refresh:
            return [str(value) for value in json.loads(cache_path.read_text(encoding="utf-8")).get("vintage_dates", [])]
        payload = self._request_json(
            FRED_VINTAGE_DATES_URL,
            {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
            },
            series_id=series_id,
            endpoint="vintage-dates",
        )
        cache_path.write_text(_canonical_json(payload), encoding="utf-8")
        return [str(value) for value in payload.get("vintage_dates", [])]


class FixtureAlfredClient:
    """Deterministic ALFRED-shaped data with non-zero latest revisions."""

    def observations(
        self,
        series_id: str,
        *,
        observation_start: str,
        observation_end: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> list[dict[str, Any]]:
        start = _parse_iso_date(observation_start)
        end = _parse_iso_date(observation_end)
        rows: list[dict[str, Any]] = []
        current = date(start.year, start.month, 1)
        finish = date(end.year, end.month, 1)
        while current <= finish:
            if realtime_start is not None and current + timedelta(days=20) > _parse_iso_date(realtime_start):
                current = _next_month(current)
                continue
            value = self._value(series_id, current)
            if realtime_start is None:
                value += self._revision(series_id, current)
            rows.append(
                {
                    "date": current.isoformat(),
                    "value": _format_number(value),
                    "realtime_start": realtime_start or "9999-12-31",
                    "realtime_end": realtime_end or "9999-12-31",
                }
            )
            current = _next_month(current)
        return rows

    def vintage_dates(self, series_id: str, *, realtime_start: str, realtime_end: str) -> list[str]:
        start = _parse_iso_date(realtime_start)
        end = _parse_iso_date(realtime_end)
        dates: list[str] = []
        current = date(start.year, start.month, 1)
        while current <= end:
            release = current + timedelta(days=20)
            revision = current + timedelta(days=40)
            for vintage in (release, revision):
                if start <= vintage <= end:
                    dates.append(vintage.isoformat())
            current = _next_month(current)
        return sorted(set(dates))

    @staticmethod
    def _value(series_id: str, observation_date: date) -> float:
        index = observation_date.year * 12 + observation_date.month
        salt = (sum(ord(char) for char in series_id) % 17) + 3
        if series_id == "PSAVERT":
            return 4.0 + (index % 12) * 0.08 + salt * 0.01
        return 80.0 + salt * 3.0 + index * (0.03 + salt / 1000.0)

    @staticmethod
    def _revision(series_id: str, observation_date: date) -> float:
        salt = (sum(ord(char) for char in series_id) % 5) + 1
        return salt * 0.01 + (observation_date.month % 3) * 0.001


def parse_monthly_origins(value: str) -> list[str]:
    """Parse ``YYYY-MM-DD:YYYY-MM-DD`` into inclusive, month-start origins."""
    if value.count(":") != 1:
        raise FrozenVintageBundleError("--origins must be START:END")
    start_text, end_text = value.split(":", 1)
    start, end = _parse_iso_date(start_text), _parse_iso_date(end_text)
    if start.day != 1 or end.day != 1:
        raise FrozenVintageBundleError("Monthly origins must use the first day of each month")
    if start > end:
        raise FrozenVintageBundleError("Origin range start must not be after end")
    origins: list[str] = []
    current = start
    while current <= end:
        origins.append(current.isoformat())
        current = _next_month(current)
    return origins


def contamination_label(model: str, target_observation_date: str, first_release_as_of_date: str) -> str:
    """Classify target availability relative to a model cutoff, not its origin slot."""
    if model not in MODEL_CUTOFFS:
        raise FrozenVintageBundleError(f"Unknown model for contamination registry: {model}")
    cutoff = MODEL_CUTOFFS[model]
    if first_release_as_of_date <= cutoff:
        return "potential_training_contamination"
    if target_observation_date <= cutoff:
        return "pre_cutoff_observation_post_cutoff_release"
    return "post_cutoff_holdout"


def origin_information_label(model: str, as_of_date: str) -> str:
    if model not in MODEL_CUTOFFS:
        raise FrozenVintageBundleError(f"Unknown model for contamination registry: {model}")
    return "origin_as_of_pre_cutoff" if as_of_date <= MODEL_CUTOFFS[model] else "origin_as_of_post_cutoff"


def build_fixture_bundle(
    output_dir: Path | str,
    origins: Iterable[str],
    *,
    release_lag_days: int = DEFAULT_RELEASE_LAG_DAYS,
    as_of_day: int = DEFAULT_AS_OF_DAY,
) -> dict[str, Any]:
    return build_frozen_vintage_bundle(
        output_dir,
        origins,
        mode="fixture",
        release_lag_days=release_lag_days,
        as_of_day=as_of_day,
    )


def build_frozen_vintage_bundle(
    output_dir: Path | str,
    origins: Iterable[str],
    *,
    mode: str = "fred",
    refresh: bool = False,
    release_lag_days: int = DEFAULT_RELEASE_LAG_DAYS,
    as_of_day: int = DEFAULT_AS_OF_DAY,
    client: ObservationClient | None = None,
) -> dict[str, Any]:
    """Write a frozen bundle and return its validated manifest.

    ``mode='fred'`` reads ``FRED_API_KEY`` from the ignored local environment.
    The key is used only in request parameters and is never serialized.
    """
    if mode not in {"fixture", "fred"}:
        raise FrozenVintageBundleError("mode must be fixture or fred")
    if int(release_lag_days) < 1:
        raise FrozenVintageBundleError("release_lag_days must be positive")
    build_as_of_date = date.today()
    origin_records = _origin_records(origins, as_of_day=as_of_day)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FrozenVintageBundleError(
            f"Frozen bundle output must be absent or empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if client is None:
        if mode == "fixture":
            client = FixtureAlfredClient()
        else:
            load_secret_env()
            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                raise FrozenVintageBundleError("FRED_API_KEY is required for --mode fred")
            client = AlfredClient(api_key, DEFAULT_CACHE_DIR, refresh=refresh)

    history_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for origin_record in origin_records:
        origin_month = origin_record["origin_month"]
        as_of_date = origin_record["as_of_date"]
        origin = _parse_iso_date(origin_month)
        history_start = _month_offset(origin, -HISTORY_MONTHS).isoformat()
        history_by_series: dict[str, list[dict[str, Any]]] = {}
        for series_id, role in _all_series():
            observations = _valid_observations(
                client.observations(
                    series_id,
                    observation_start=history_start,
                    observation_end=as_of_date,
                    realtime_start=as_of_date,
                    realtime_end=as_of_date,
                )
            )
            history_by_series[series_id] = observations
            request_rows.append(
                _request_row(origin_month, as_of_date, series_id, "origin_history", as_of_date, as_of_date, history_start, as_of_date)
            )
            for observation in observations:
                history_rows.append(
                    {
                        "origin_month": origin_month,
                        "as_of_date": as_of_date,
                        "series_id": series_id,
                        "series_role": role,
                        "observation_date": observation["date"],
                        "value": observation["value"],
                        "realtime_start": observation.get("realtime_start", as_of_date),
                        "realtime_end": observation.get("realtime_end", as_of_date),
                    }
                )
        for spec in TARGET_SPECS:
            visible = history_by_series[spec.series_id]
            if not visible:
                raise FrozenVintageBundleError(f"No origin-visible denominator for {spec.series_id} at {as_of_date}")
            origin_visible_denominator = visible[-1]
            target_observation = origin
            target_text = target_observation.isoformat()
            transform_denominator = _month_offset(target_observation, -1)
            transform_denominator_text = transform_denominator.isoformat()
            horizon_date = min(target_observation + timedelta(days=int(release_lag_days)), build_as_of_date)
            horizon_text = horizon_date.isoformat()
            first_release, first_release_text, detection_method = _detect_first_release(
                client,
                spec.series_id,
                target_text,
                horizon_text,
            )
            request_rows.append(
                _request_row(
                    origin_month,
                    as_of_date,
                    spec.series_id,
                    "vintage_date_search",
                    target_text,
                    horizon_text,
                    "",
                    "",
                    "vintage_dates",
                )
            )
            first_release_window = _valid_observations(
                client.observations(
                    spec.series_id,
                    observation_start=transform_denominator_text,
                    observation_end=target_text,
                    realtime_start=first_release_text,
                    realtime_end=first_release_text,
                )
            )
            first_by_date = {row["date"]: row for row in first_release_window}
            if target_text not in first_by_date or transform_denominator_text not in first_by_date:
                raise FrozenVintageBundleError(
                    f"First-release vintage lacks a matched numerator/denominator for {spec.series_id} {target_text}"
                )
            request_rows.append(
                _request_row(
                    origin_month,
                    as_of_date,
                    spec.series_id,
                    "first_release_target",
                    first_release_text,
                    first_release_text,
                    transform_denominator_text,
                    target_text,
                    detection_method,
                )
            )
            first_value = float(first_by_date[target_text]["value"])
            first_denominator_value = float(first_by_date[transform_denominator_text]["value"])
            origin_visible_denominator_value = float(origin_visible_denominator["value"])
            target_rows.append(
                {
                    "origin_month": origin_month,
                    "as_of_date": as_of_date,
                    "target_name": spec.target_name,
                    "series_id": spec.series_id,
                    "family": spec.family,
                    "transform": spec.transform,
                    "default_scale": spec.default_scale,
                    "target_observation_date": target_text,
                    "first_release_as_of_date": first_release_text,
                    "release_detection_method": detection_method,
                    "first_release_value": first_value,
                    "origin_visible_denominator_date": origin_visible_denominator["date"],
                    "origin_visible_denominator_value": origin_visible_denominator_value,
                    "first_release_denominator_date": transform_denominator_text,
                    "first_release_denominator_value": first_denominator_value,
                    "target_value": _apply_transform(spec.transform, first_value, first_denominator_value),
                }
            )
            latest = _valid_observations(
                client.observations(
                    spec.series_id,
                    observation_start=target_text,
                    observation_end=target_text,
                )
            )
            request_rows.append(
                _request_row(
                    origin_month,
                    as_of_date,
                    spec.series_id,
                    "latest_revision_audit",
                    "",
                    "",
                    target_text,
                    target_text,
                )
            )
            if not latest:
                raise FrozenVintageBundleError(f"No latest revision audit value for {spec.series_id} {target_text}")
            latest_value = float(latest[-1]["value"])
            audit_rows.append(
                {
                    "origin_month": origin_month,
                    "as_of_date": as_of_date,
                    "target_name": spec.target_name,
                    "series_id": spec.series_id,
                    "target_observation_date": target_text,
                    "first_release_value": first_value,
                    "latest_revision_value": latest_value,
                    "latest_minus_first_release": latest_value - first_value,
                }
            )

    contamination_rows = [
        {
            "origin_month": target["origin_month"],
            "as_of_date": target["as_of_date"],
            "target_name": target["target_name"],
            "series_id": target["series_id"],
            "target_observation_date": target["target_observation_date"],
            "first_release_as_of_date": target["first_release_as_of_date"],
            "model": model,
            "model_cutoff_date": cutoff,
            "contamination_label": contamination_label(
                model,
                target["target_observation_date"],
                target["first_release_as_of_date"],
            ),
            "origin_information_label": origin_information_label(model, target["as_of_date"]),
        }
        for target in target_rows
        for model, cutoff in MODEL_CUTOFFS.items()
    ]
    rows_by_file = {
        "origins.csv": origin_records,
        "history.csv": history_rows,
        "targets.csv": target_rows,
        "revision_audit.csv": audit_rows,
        "target_contamination.csv": contamination_rows,
        "source_requests.csv": request_rows,
    }
    payload_sha256: dict[str, str] = {}
    for file_name, columns in PAYLOAD_FILES.items():
        path = root / file_name
        _write_canonical_csv(path, columns, rows_by_file[file_name])
        payload_sha256[file_name] = _file_sha256(path)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "format": "canonical_json_csv",
        "mode": mode,
        "release_lag_days": int(release_lag_days),
        "release_lag_days_semantics": "maximum first-release vintage-date search horizon and fallback snapshot lag",
        "build_as_of_date": build_as_of_date.isoformat(),
        "as_of_day": int(as_of_day),
        "history_months": HISTORY_MONTHS,
        "fred_observations_url": FRED_OBSERVATIONS_URL,
        "fred_vintage_dates_url": FRED_VINTAGE_DATES_URL,
        "fred_documentation_url": FRED_DOCUMENTATION_URL,
        "origins": origin_records,
        "target_observation_semantics": {
            "rule": "common_origin_observation_month",
            "publication_lag_note": "Every target at an origin refers to the same calendar observation month, equal to origin_month, regardless of series-specific publication lag.",
            "transform_denominator": "Percentage-change and difference targets use the previous calendar month and the target month from the same earliest first-release vintage; the latest origin-visible value is retained separately for audit and level-direction scoring.",
        },
        "first_release_detection": "earliest usable fred/series/vintagedates entry within release_lag_days, otherwise horizon fallback",
        "vintage_dates_semantics": "FRED vintage dates represent releases or revisions; each candidate is checked for target-observation availability.",
        "target_specs": [asdict(spec) for spec in TARGET_SPECS],
        "target_set": [spec.target_name for spec in TARGET_SPECS],
        "context_series": list(CONTEXT_SERIES),
        "model_cutoffs": MODEL_CUTOFFS,
        "payload_files": list(PAYLOAD_FILES),
        "payload_sha256": payload_sha256,
    }
    manifest["bundle_sha256"] = _bundle_sha256(manifest)
    _write_canonical_json(root / "manifest.json", manifest)
    validate_frozen_vintage_bundle(root)
    return manifest


def validate_frozen_vintage_bundle(root: Path | str) -> dict[str, Any]:
    """Validate all format and integrity contracts, raising on any mismatch."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FrozenVintageBundleError(f"Missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrozenVintageBundleError("manifest.json is not valid JSON") from exc
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise FrozenVintageBundleError("Unsupported or mismatched bundle schema")
    expected_specs = [asdict(spec) for spec in TARGET_SPECS]
    if manifest.get("target_specs") != expected_specs or manifest.get("target_set") != [spec.target_name for spec in TARGET_SPECS]:
        raise FrozenVintageBundleError("Bundle target set does not match the required target catalogue")
    if manifest.get("model_cutoffs") != MODEL_CUTOFFS:
        raise FrozenVintageBundleError("Bundle model cutoff registry does not match the required registry")
    if not isinstance(manifest.get("release_lag_days"), int) or manifest["release_lag_days"] < 1:
        raise FrozenVintageBundleError("Bundle release_lag_days must be a positive integer")
    build_as_of_date = str(manifest.get("build_as_of_date", ""))
    _parse_iso_date(build_as_of_date)
    if manifest.get("payload_files") != list(PAYLOAD_FILES):
        raise FrozenVintageBundleError("Bundle payload file set does not match the canonical format")
    hashes = manifest.get("payload_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(PAYLOAD_FILES):
        raise FrozenVintageBundleError("Bundle payload hashes are incomplete")
    payload_rows: dict[str, list[dict[str, str]]] = {}
    for file_name, columns in PAYLOAD_FILES.items():
        path = root / file_name
        if not path.exists() or hashes[file_name] != _file_sha256(path):
            raise FrozenVintageBundleError(f"Hash mismatch for {file_name}")
        rows = _read_canonical_csv(path, columns)
        payload_rows[file_name] = rows
        if not rows and file_name in {"origins.csv", "history.csv", "targets.csv"}:
            raise FrozenVintageBundleError(f"Required payload is empty: {file_name}")
    if manifest.get("bundle_sha256") != _bundle_sha256(manifest):
        raise FrozenVintageBundleError("Bundle hash mismatch")
    as_of_day = manifest.get("as_of_day")
    if not isinstance(as_of_day, int) or not 1 <= as_of_day <= 28:
        raise FrozenVintageBundleError("Bundle as_of_day must be an integer from 1 through 28")
    origins = payload_rows["origins.csv"]
    expected_origins = _origin_records([row["origin_month"] for row in origins], as_of_day=as_of_day)
    if origins != expected_origins or origins != manifest.get("origins"):
        raise FrozenVintageBundleError("Bundle origins do not match canonical origin_month and as_of_date records")
    semantics = manifest.get("target_observation_semantics")
    if not isinstance(semantics, dict) or semantics.get("rule") != "common_origin_observation_month":
        raise FrozenVintageBundleError("Bundle does not declare common-observation-month target semantics")
    targets = payload_rows["targets.csv"]
    expected_pairs = {(row["origin_month"], spec.target_name) for row in origins for spec in TARGET_SPECS}
    actual_pairs = {(row["origin_month"], row["target_name"]) for row in targets}
    if len(targets) != len(actual_pairs):
        raise FrozenVintageBundleError("Target rows contain duplicate origin-target keys")
    if actual_pairs != expected_pairs:
        raise FrozenVintageBundleError("Target rows do not exactly cover the required target set for every origin")
    target_by_key = {(row["origin_month"], row["target_name"]): row for row in targets}
    _validate_numeric_columns(payload_rows["history.csv"], "history.csv", ("value",))
    _validate_numeric_columns(
        targets,
        "targets.csv",
        (
            "default_scale",
            "first_release_value",
            "origin_visible_denominator_value",
            "first_release_denominator_value",
            "target_value",
        ),
    )
    specs_by_name = {spec.target_name: spec for spec in TARGET_SPECS}
    origins_by_month = {row["origin_month"]: row for row in origins}
    for target in targets:
        spec = specs_by_name[target["target_name"]]
        if (
            target["as_of_date"] != origins_by_month[target["origin_month"]]["as_of_date"]
            or target["series_id"] != spec.series_id
            or target["family"] != spec.family
            or target["transform"] != spec.transform
            or target["default_scale"] != _format_number(spec.default_scale)
            or target["release_detection_method"] not in {"vintage_dates", "release_lag_fallback"}
            or target["target_observation_date"] != target["origin_month"]
            or target["first_release_denominator_date"]
            != _month_offset(_parse_iso_date(target["target_observation_date"]), -1).isoformat()
        ):
            raise FrozenVintageBundleError("Target row metadata does not match the required target specification")
    audits = payload_rows["revision_audit.csv"]
    audit_pairs = {(row["origin_month"], row["target_name"]) for row in audits}
    if len(audits) != len(audit_pairs) or audit_pairs != expected_pairs:
        raise FrozenVintageBundleError("Revision audit rows do not exactly cover unique origin-target keys")
    _validate_numeric_columns(audits, "revision_audit.csv", ("first_release_value", "latest_revision_value", "latest_minus_first_release"))
    for audit in audits:
        target = target_by_key[(audit["origin_month"], audit["target_name"])]
        if any(audit[column] != target[column] for column in ("as_of_date", "series_id", "target_observation_date", "first_release_value")):
            raise FrozenVintageBundleError("Revision audit does not match its frozen target row")
    contamination = payload_rows["target_contamination.csv"]
    expected_contamination = {(origin, target_name, model) for origin, target_name in expected_pairs for model in MODEL_CUTOFFS}
    contamination_keys = {(row["origin_month"], row["target_name"], row["model"]) for row in contamination}
    if len(contamination) != len(contamination_keys) or contamination_keys != expected_contamination:
        raise FrozenVintageBundleError("Target contamination rows do not exactly cover origin-target-model keys")
    for row in contamination:
        target = target_by_key[(row["origin_month"], row["target_name"])]
        if row["model_cutoff_date"] != MODEL_CUTOFFS.get(row["model"]):
            raise FrozenVintageBundleError("Target contamination has an unknown or mismatched model cutoff")
        if any(row[column] != target[column] for column in ("as_of_date", "series_id", "target_observation_date", "first_release_as_of_date")):
            raise FrozenVintageBundleError("Target contamination does not match its target row")
        if row["contamination_label"] != contamination_label(row["model"], row["target_observation_date"], row["first_release_as_of_date"]):
            raise FrozenVintageBundleError("Target contamination label is inconsistent with target release timing")
        if row["origin_information_label"] != origin_information_label(row["model"], row["as_of_date"]):
            raise FrozenVintageBundleError("Target contamination origin information label is inconsistent")
    _validate_source_request_provenance(
        payload_rows["source_requests.csv"],
        origins,
        targets,
        release_lag_days=int(manifest["release_lag_days"]),
        build_as_of_date=build_as_of_date,
    )
    return manifest


def load_frozen_vintage_bundle(root: Path | str) -> FrozenVintageBundle:
    """Fail closed, then load a validated bundle into immutable row collections."""
    root = Path(root)
    manifest = validate_frozen_vintage_bundle(root)
    return FrozenVintageBundle(
        root=root,
        manifest=manifest,
        origins=tuple(_read_canonical_csv(root / "origins.csv", ORIGIN_COLUMNS)),
        history=tuple(_read_canonical_csv(root / "history.csv", HISTORY_COLUMNS)),
        targets=tuple(_read_canonical_csv(root / "targets.csv", TARGET_COLUMNS)),
        revision_audit=tuple(_read_canonical_csv(root / "revision_audit.csv", REVISION_AUDIT_COLUMNS)),
        target_contamination=tuple(_read_canonical_csv(root / "target_contamination.csv", TARGET_CONTAMINATION_COLUMNS)),
        source_requests=tuple(_read_canonical_csv(root / "source_requests.csv", SOURCE_REQUEST_COLUMNS)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen rolling-origin ALFRED vintage bundle.")
    parser.add_argument("--origins", required=True, help="Inclusive monthly range START:END; dates must be YYYY-MM-01.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("fixture", "fred"), default="fixture")
    parser.add_argument("--refresh", action="store_true", help="Bypass the local ALFRED response cache in fred mode.")
    parser.add_argument("--release-lag-days", type=int, default=DEFAULT_RELEASE_LAG_DAYS)
    parser.add_argument("--as-of-day", type=int, default=DEFAULT_AS_OF_DAY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_frozen_vintage_bundle(
            args.output_dir,
            parse_monthly_origins(args.origins),
            mode=args.mode,
            refresh=args.refresh,
            release_lag_days=args.release_lag_days,
            as_of_day=args.as_of_day,
        )
    except FrozenVintageBundleError as exc:
        raise SystemExit(str(exc)) from exc
    print(_canonical_json({"bundle_sha256": manifest["bundle_sha256"], "output_dir": str(args.output_dir)}))
    return 0


def _all_series() -> tuple[tuple[str, str], ...]:
    return tuple((spec.series_id, "target") for spec in TARGET_SPECS) + tuple(
        (series_id, "context") for series_id in CONTEXT_SERIES
    )


def _normalize_origins(origins: Iterable[str]) -> list[str]:
    normalized = [str(origin) for origin in origins]
    if not normalized:
        raise FrozenVintageBundleError("At least one origin is required")
    if normalized != sorted(set(normalized)):
        raise FrozenVintageBundleError("Origins must be unique and strictly ascending")
    for origin in normalized:
        parsed = _parse_iso_date(origin)
        if parsed.day != 1 or parsed.isoformat() != origin:
            raise FrozenVintageBundleError("Origins must be canonical YYYY-MM-01 dates")
    return normalized


def _origin_records(origins: Iterable[str], *, as_of_day: int) -> list[dict[str, str]]:
    if not isinstance(as_of_day, int) or not 1 <= as_of_day <= 28:
        raise FrozenVintageBundleError("as_of_day must be an integer from 1 through 28")
    return [
        {"origin_month": origin, "as_of_date": date(_parse_iso_date(origin).year, _parse_iso_date(origin).month, as_of_day).isoformat()}
        for origin in _normalize_origins(origins)
    ]


def _valid_observations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and row.get("date"):
            observations.append({**row, "date": str(row["date"]), "value": value})
    return sorted(observations, key=lambda row: str(row["date"]))


def _detect_first_release(
    client: ObservationClient,
    series_id: str,
    target_observation_date: str,
    horizon_date: str,
) -> tuple[list[dict[str, Any]], str, str]:
    vintage_dates = sorted(
        {
            vintage_date
            for vintage_date in client.vintage_dates(
                series_id,
                realtime_start=target_observation_date,
                realtime_end=horizon_date,
            )
            if target_observation_date <= vintage_date <= horizon_date
        }
    )
    for vintage_date in vintage_dates:
        observations = _valid_observations(
            client.observations(
                series_id,
                observation_start=target_observation_date,
                observation_end=target_observation_date,
                realtime_start=vintage_date,
                realtime_end=vintage_date,
            )
        )
        if observations:
            return observations, vintage_date, "vintage_dates"
    fallback = _valid_observations(
        client.observations(
            series_id,
            observation_start=target_observation_date,
            observation_end=target_observation_date,
            realtime_start=horizon_date,
            realtime_end=horizon_date,
        )
    )
    if fallback:
        return fallback, horizon_date, "release_lag_fallback"
    raise FrozenVintageBundleError(
        f"No first-release observation for {series_id} {target_observation_date} within horizon {horizon_date}"
    )


def _validate_numeric_columns(rows: Iterable[dict[str, str]], file_name: str, columns: Iterable[str]) -> None:
    for row in rows:
        for column in columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise FrozenVintageBundleError(f"Non-numeric {column} in {file_name}") from exc
            if not math.isfinite(value):
                raise FrozenVintageBundleError(f"Non-finite {column} in {file_name}")


def _validate_source_request_provenance(
    requests_rows: list[dict[str, str]],
    origins: list[dict[str, str]],
    targets: list[dict[str, str]],
    *,
    release_lag_days: int,
    build_as_of_date: str,
) -> None:
    request_keys = {
        (
            row["origin_month"],
            row["target_name"] if "target_name" in row else row["series_id"],
            row["request_kind"],
            row["realtime_start"],
        )
        for row in requests_rows
    }
    if len(request_keys) != len(requests_rows):
        raise FrozenVintageBundleError("Source request provenance contains duplicate request rows")
    expected_history = {
        (origin["origin_month"], series_id, "origin_history", origin["as_of_date"])
        for origin in origins
        for series_id, _ in _all_series()
    }
    actual_history = {
        (row["origin_month"], row["series_id"], row["request_kind"], row["realtime_start"])
        for row in requests_rows
        if row["request_kind"] == "origin_history"
    }
    if actual_history != expected_history:
        raise FrozenVintageBundleError("Source request provenance is missing or altering origin history requests")
    history_start_by_origin = {
        origin["origin_month"]: _month_offset(_parse_iso_date(origin["origin_month"]), -HISTORY_MONTHS).isoformat()
        for origin in origins
    }
    for row in requests_rows:
        if row["request_kind"] == "origin_history":
            origin = next(item for item in origins if item["origin_month"] == row["origin_month"])
            if (
                row["as_of_date"] != origin["as_of_date"]
                or row["realtime_start"] != origin["as_of_date"]
                or row["realtime_end"] != origin["as_of_date"]
                or row["observation_start"] != history_start_by_origin[row["origin_month"]]
                or row["observation_end"] != origin["as_of_date"]
            ):
                raise FrozenVintageBundleError("Origin history request is not frozen exactly at as_of_date")
    expected_target_requests = {
        (target["origin_month"], target["series_id"], kind)
        for target in targets
        for kind in ("vintage_date_search", "first_release_target", "latest_revision_audit")
    }
    actual_target_requests = {
        (row["origin_month"], row["series_id"], row["request_kind"])
        for row in requests_rows
        if row["request_kind"] != "origin_history"
    }
    if actual_target_requests != expected_target_requests:
        raise FrozenVintageBundleError("Source request provenance does not exactly cover target release and audit requests")
    request_by_target_kind = {(row["origin_month"], row["series_id"], row["request_kind"]): row for row in requests_rows}
    for target in targets:
        key = (target["origin_month"], target["series_id"])
        search = request_by_target_kind[(*key, "vintage_date_search")]
        first = request_by_target_kind[(*key, "first_release_target")]
        latest = request_by_target_kind[(*key, "latest_revision_audit")]
        horizon = min(
            _parse_iso_date(target["target_observation_date"]) + timedelta(days=release_lag_days),
            _parse_iso_date(build_as_of_date),
        ).isoformat()
        if (
            search["as_of_date"] != target["as_of_date"]
            or search["realtime_start"] != target["target_observation_date"]
            or search["realtime_end"] != horizon
            or search["release_detection_method"] != "vintage_dates"
            or first["realtime_start"] != target["first_release_as_of_date"]
            or first["realtime_end"] != target["first_release_as_of_date"]
            or first["observation_start"] != target["first_release_denominator_date"]
            or first["observation_end"] != target["target_observation_date"]
            or first["release_detection_method"] != target["release_detection_method"]
            or latest["realtime_start"]
            or latest["realtime_end"]
            or latest["observation_start"] != target["target_observation_date"]
            or latest["observation_end"] != target["target_observation_date"]
        ):
            raise FrozenVintageBundleError("Source request provenance conflicts with target release records")


def _apply_transform(transform: str, numerator: float, denominator: float) -> float:
    if transform == "pct_change":
        if denominator == 0:
            raise FrozenVintageBundleError("Cannot calculate pct_change with a zero origin-visible denominator")
        return 100.0 * (numerator / denominator - 1.0)
    if transform == "diff":
        return numerator - denominator
    if transform == "level":
        return numerator
    raise FrozenVintageBundleError(f"Unsupported target transform: {transform}")


def _request_row(
    origin_month: str,
    as_of_date: str,
    series_id: str,
    request_kind: str,
    realtime_start: str,
    realtime_end: str,
    observation_start: str,
    observation_end: str,
    release_detection_method: str = "",
) -> dict[str, str]:
    return {
        "origin_month": origin_month,
        "as_of_date": as_of_date,
        "series_id": series_id,
        "request_kind": request_kind,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "observation_start": observation_start,
        "observation_end": observation_end,
        "release_detection_method": release_detection_method,
    }


def _write_canonical_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    ordered_rows = sorted(rows, key=lambda row: tuple(str(row.get(column, "")) for column in columns))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _read_canonical_csv(path: Path, columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise FrozenVintageBundleError(f"Unexpected columns in {path.name}")
        rows = list(reader)
    if rows != sorted(rows, key=lambda row: tuple(row[column] for column in columns)):
        raise FrozenVintageBundleError(f"Rows in {path.name} are not canonically ordered")
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FrozenVintageBundleError("Canonical CSV cannot contain non-finite values")
        return _format_number(value)
    return str(value)


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bundle_sha256(manifest: dict[str, Any]) -> str:
    signed = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    return _canonical_json_sha256(signed)


def _write_canonical_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FrozenVintageBundleError(f"Invalid ISO date: {value}") from exc


def _next_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def _month_offset(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


if __name__ == "__main__":
    raise SystemExit(main())
