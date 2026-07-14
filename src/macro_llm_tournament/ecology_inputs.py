"""Origin-safe inputs for the household-first ecology."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .env import load_secret_env
from .frozen_vintage_bundle import (
    AlfredClient,
    BUNDLE_SCHEMA_VERSION,
    CONTEXT_SERIES,
    DEFAULT_CACHE_DIR,
    HISTORY_COLUMNS,
    ORIGIN_COLUMNS,
    TARGET_SPECS,
    _bundle_sha256,
    _file_sha256,
    _read_canonical_csv,
)


ORIGIN_SNAPSHOT_SCHEMA_VERSION = "household_ecology_origin_snapshot_v1"


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def load_origin_information(bundle_dir: Path, origin_month: str) -> tuple[dict[str, Any], str]:
    if bundle_dir.is_file():
        payload = json.loads(bundle_dir.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ORIGIN_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("origin snapshot schema mismatch")
        supplied_hash = payload.get("snapshot_sha256")
        expected_hash = _canonical_sha256({key: value for key, value in payload.items() if key != "snapshot_sha256"})
        if supplied_hash != expected_hash:
            raise ValueError("origin snapshot hash mismatch")
        information = payload["origin_information"]
        if information.get("origin_month") != origin_month:
            raise ValueError("origin snapshot month mismatch")
        return dict(information), str(supplied_hash)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("origin-input bundle schema mismatch")
    if manifest.get("bundle_sha256") != _bundle_sha256(manifest):
        raise ValueError("origin-input bundle manifest hash mismatch")
    hashes = manifest.get("payload_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("origin-input bundle payload hashes are missing")
    for name in ("origins.csv", "history.csv"):
        path = bundle_dir / name
        if not path.exists() or hashes.get(name) != _file_sha256(path):
            raise ValueError(f"origin-input bundle hash mismatch for {name}")
    origin_rows = _read_canonical_csv(bundle_dir / "origins.csv", ORIGIN_COLUMNS)
    history_rows = _read_canonical_csv(bundle_dir / "history.csv", HISTORY_COLUMNS)
    if origin_rows != manifest.get("origins"):
        raise ValueError("origin-input rows do not match the manifest")
    origins = [row for row in origin_rows if row["origin_month"] == origin_month]
    if len(origins) != 1:
        raise ValueError(f"origin {origin_month} is absent or duplicated in frozen bundle")
    origin = origins[0]
    history = [row for row in history_rows if row["origin_month"] == origin_month]
    if not history:
        raise ValueError(f"origin {origin_month} has no origin-visible history")
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(history, key=lambda item: (item["series_id"], item["observation_date"])):
        by_series.setdefault(row["series_id"], []).append(
            {"observation_date": row["observation_date"], "value": float(row["value"])}
        )
    latest = {series: rows[-1] for series, rows in by_series.items()}
    information = {
        "origin_month": origin["origin_month"],
        "as_of_date": origin["as_of_date"],
        "origin_visible_macro_context": latest,
        "origin_visible_macro_history": by_series,
        "public_events": [],
    }
    if any(
        row["as_of_date"] != origin["as_of_date"]
        or row["observation_date"] > origin["as_of_date"]
        or not math.isfinite(float(row["value"]))
        for row in history
    ):
        raise ValueError("origin-input history violates its information cutoff")
    return information, str(manifest["bundle_sha256"])


def build_origin_snapshot(
    output_path: Path,
    *,
    origin_month: str,
    as_of_date: str,
    refresh: bool = False,
) -> dict[str, Any]:
    origin = pd.Timestamp(origin_month)
    as_of = pd.Timestamp(as_of_date)
    if origin.day != 1 or origin > as_of or as_of.date() > date.today():
        raise ValueError("origin must be month-start, no later than as-of, and as-of cannot be future")
    if output_path.exists():
        raise ValueError(f"origin snapshot already exists: {output_path}")
    load_secret_env()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY is required to build an origin snapshot")
    client = AlfredClient(api_key, DEFAULT_CACHE_DIR, refresh=refresh)
    history_start = (origin - pd.offsets.MonthBegin(24)).date().isoformat()
    series = list(dict.fromkeys([spec.series_id for spec in TARGET_SPECS] + list(CONTEXT_SERIES)))
    by_series: dict[str, list[dict[str, Any]]] = {}
    requests: list[dict[str, str]] = []
    for series_id in series:
        observations = client.observations(
            series_id,
            observation_start=history_start,
            observation_end=as_of_date,
            realtime_start=as_of_date,
            realtime_end=as_of_date,
        )
        rows: list[dict[str, Any]] = []
        for observation in observations:
            try:
                value = float(observation["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows.append({"observation_date": str(observation["date"]), "value": value})
        if not rows:
            raise ValueError(f"origin snapshot has no visible observations for {series_id}")
        by_series[series_id] = rows
        requests.append(
            {
                "series_id": series_id,
                "observation_start": history_start,
                "observation_end": as_of_date,
                "realtime_start": as_of_date,
                "realtime_end": as_of_date,
            }
        )
    information = {
        "origin_month": origin.date().isoformat(),
        "as_of_date": as_of.date().isoformat(),
        "origin_visible_macro_context": {series_id: rows[-1] for series_id, rows in by_series.items()},
        "origin_visible_macro_history": by_series,
        "public_events": [],
    }
    payload = {
        "schema_version": ORIGIN_SNAPSHOT_SCHEMA_VERSION,
        "origin_information": information,
        "source": "FRED/ALFRED series observations with realtime_start=realtime_end=as_of_date",
        "source_requests": requests,
    }
    payload["snapshot_sha256"] = _canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an origin-only ALFRED snapshot without outcomes")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build_origin_snapshot(args.output, origin_month=args.origin, as_of_date=args.as_of, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
