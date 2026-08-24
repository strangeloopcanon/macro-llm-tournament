"""Augment the frozen dense ALFRED stream with required missing series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "PCEC96"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-events",
        type=Path,
        default=Path("work/fred_dense_events/dense_macro_events.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("work/fred_dense_events/api_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/fred_dense_events_event_jepa_v2"),
    )
    parser.add_argument("--observation-start", default="1990-01-01")
    parser.add_argument("--through", default="2026-06-23")
    parser.add_argument("--refresh", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        raise ValueError("FRED_API_KEY is required")

    payloads = {
        output_type: _fetch_payload(
            api_key=api_key,
            cache_dir=args.cache_dir,
            observation_start=args.observation_start,
            through=args.through,
            output_type=output_type,
            refresh=bool(args.refresh),
        )
        for output_type in (3, 4)
    }
    added = _events_from_payloads(payloads[3], payloads[4])
    base_rows: list[dict[str, Any]] = []
    with args.base_events.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("series_id")) != SERIES_ID:
                base_rows.append(row)
    rows = sorted(
        [*base_rows, *added],
        key=lambda row: (
            str(row["vintage_date"]),
            str(row["series_id"]),
            str(row["observation_date"]),
            str(row["event_kind"]),
            str(row["event_id"]),
        ),
    )
    events_path = output / "dense_macro_events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "macro_event_stream_augmentation_v1",
        "base_events": str(args.base_events.resolve()),
        "base_events_sha256": _sha256(args.base_events),
        "series_added": SERIES_ID,
        "observation_start": args.observation_start,
        "through": args.through,
        "source_url": FRED_OBSERVATIONS_URL,
        "output_types": [3, 4],
        "added_event_count": len(added),
        "total_event_count": len(rows),
        "events_sha256": _sha256(events_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _fetch_payload(
    *,
    api_key: str,
    cache_dir: Path,
    observation_start: str,
    through: str,
    output_type: int,
    refresh: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{SERIES_ID}_output_type_{output_type}_event_jepa.json"
    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": SERIES_ID,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": through,
            "realtime_start": observation_start,
            "realtime_end": through,
            "output_type": str(output_type),
            "sort_order": "asc",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _events_from_payloads(
    revisions_payload: dict[str, Any],
    initial_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observation in initial_payload.get("observations", []):
        value = _finite_value(observation.get("value"))
        if value is None:
            continue
        rows.append(
            _event_row(
                observation_date=str(observation["date"]),
                vintage_date=str(observation["realtime_start"]),
                value=value,
                event_kind="initial_release",
                output_type=4,
            )
        )
    prefix = f"{SERIES_ID}_"
    for observation in revisions_payload.get("observations", []):
        observation_date = str(observation["date"])
        for key, raw_value in observation.items():
            if not str(key).startswith(prefix):
                continue
            value = _finite_value(raw_value)
            if value is None:
                continue
            vintage_token = str(key)[len(prefix) :]
            if len(vintage_token) != 8 or not vintage_token.isdigit():
                raise ValueError(f"invalid ALFRED vintage column: {key}")
            vintage_date = (
                f"{vintage_token[:4]}-{vintage_token[4:6]}-{vintage_token[6:]}"
            )
            rows.append(
                _event_row(
                    observation_date=observation_date,
                    vintage_date=vintage_date,
                    value=value,
                    event_kind="revision",
                    output_type=3,
                )
            )
    return rows


def _event_row(
    *,
    observation_date: str,
    vintage_date: str,
    value: float,
    event_kind: str,
    output_type: int,
) -> dict[str, Any]:
    identity = f"{SERIES_ID}|{observation_date}|{vintage_date}|{event_kind}|{value}"
    event_id = f"fred:{SERIES_ID}:{vintage_date}:{observation_date}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    return {
        "event_id": event_id,
        "event_kind": event_kind,
        "event_type": f"macro_series_{event_kind}",
        "frequency_hint": "monthly",
        "observation_date": observation_date,
        "provenance": {
            "output_type": output_type,
            "source_docs": "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
            "source_url": FRED_OBSERVATIONS_URL,
            "target_hidden": False,
        },
        "realtime_end": vintage_date,
        "realtime_start": vintage_date,
        "series_category": "consumption",
        "series_id": SERIES_ID,
        "series_label": "Real personal consumption expenditures",
        "source": "fred_alfred",
        "text": (
            f"{SERIES_ID} {event_kind.replace('_', ' ')} on {vintage_date}: "
            f"observation {observation_date} value {value}."
        ),
        "ts": f"{vintage_date}T00:00:00Z",
        "value": value,
        "vintage_date": vintage_date,
    }


def _finite_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
