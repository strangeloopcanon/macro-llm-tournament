"""Build a timestamped provenance event stream for every data asset in the project.

Scans raw downloads, derived datasets, packaged target catalogs, and run
manifests, then writes one JSON event per line to
``data_provenance/data_events.jsonl`` (tracked in git), sorted by timestamp.
Each event records where the timestamp came from (``ts_basis``) so later
consumers can distinguish exact capture times from file-mtime reconstructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .agent_common import PROJECT_ROOT

PROVENANCE_SCHEMA_VERSION = "data_provenance_v1"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_provenance" / "data_events.jsonl"

RAW_DOWNLOAD_DIRS: dict[str, dict[str, str]] = {
    "work/persona_beliefs/sce_raw": {
        "dataset": "frbny_sce_core_public_microdata",
        "source": "Federal Reserve Bank of New York, Survey of Consumer Expectations",
        "url": "https://www.newyorkfed.org/microeconomics/sce",
        "notes": "Core SCE public microdata workbooks; respondent belief panel.",
    },
    "work/spending_survey_raw": {
        "dataset": "frbny_sce_household_spending_survey",
        "source": "Federal Reserve Bank of New York, SCE Household Spending Survey",
        "url": "https://www.newyorkfed.org/microeconomics/sce/household-spending",
        "notes": (
            "Household spending module microdata (userid, wave date, reported and expected "
            "spending changes, windfall allocation questions), chart data, questionnaire, "
            "glossary, and the January 2023 series note. Fielded every four months to core "
            "SCE panelists; public microdata released with an eighteen-month lag."
        ),
    },
}

DERIVED_DATA_DIRS: dict[str, dict[str, str]] = {
    "work/persona_beliefs": {
        "dataset": "sce_processed_panels",
        "source": "derived_in_repo",
        "notes": "Normalized SCE panels produced by prepare_sce_microdata / prepare_persona_holdouts.",
    },
}

PACKAGED_DATA_FILES: dict[str, dict[str, str]] = {
    "src/macro_llm_tournament/data/public_behavior_targets.csv": {
        "dataset": "public_behavior_target_catalog",
        "source": "curated_from_literature",
        "notes": (
            "Behavior target bands with per-row citation columns (tax rebate, 2008 stimulus, "
            "2020 EIP, lottery, UI exhaustion, 2021 CTC families)."
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _file_event(
    path: Path,
    *,
    event_type: str,
    dataset: str,
    source: str,
    ts_basis: str,
    recorded_at: str,
    url: str | None = None,
    notes: str | None = None,
    ts_utc: str | None = None,
) -> dict[str, Any]:
    relative = str(path.relative_to(PROJECT_ROOT))
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "ts_utc": ts_utc or _mtime_utc(path),
        "ts_basis": ts_basis,
        "recorded_at_utc": recorded_at,
        "event_type": event_type,
        "dataset": dataset,
        "source": source,
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "url": url,
        "notes": notes,
    }


def _iter_download_events(recorded_at: str) -> Iterator[dict[str, Any]]:
    for rel_dir, meta in RAW_DOWNLOAD_DIRS.items():
        directory = PROJECT_ROOT / rel_dir
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            yield _file_event(
                path,
                event_type="raw_download",
                dataset=meta["dataset"],
                source=meta["source"],
                ts_basis="file_mtime",
                recorded_at=recorded_at,
                url=meta.get("url"),
                notes=meta.get("notes"),
            )


def _iter_derived_events(recorded_at: str) -> Iterator[dict[str, Any]]:
    for rel_dir, meta in DERIVED_DATA_DIRS.items():
        directory = PROJECT_ROOT / rel_dir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            yield _file_event(
                path,
                event_type="derived_dataset",
                dataset=meta["dataset"],
                source=meta["source"],
                ts_basis="file_mtime",
                recorded_at=recorded_at,
                notes=meta.get("notes"),
            )
    for rel_file, meta in PACKAGED_DATA_FILES.items():
        path = PROJECT_ROOT / rel_file
        if path.exists():
            yield _file_event(
                path,
                event_type="packaged_dataset",
                dataset=meta["dataset"],
                source=meta["source"],
                ts_basis="file_mtime",
                recorded_at=recorded_at,
                notes=meta.get("notes"),
            )


def _iter_manifest_events(recorded_at: str) -> Iterator[dict[str, Any]]:
    data_manifest = PROJECT_ROOT / "work" / "DATA_MANIFEST.json"
    if data_manifest.exists():
        payload = json.loads(data_manifest.read_text(encoding="utf-8"))
        created = str(payload.get("created_utc") or _mtime_utc(data_manifest))
        for row in payload.get("datasets", []):
            yield {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "ts_utc": created,
                "ts_basis": "data_manifest_created_utc",
                "recorded_at_utc": recorded_at,
                "event_type": "raw_download",
                "dataset": str(row.get("group") or "data_manifest_entry"),
                "source": "work/DATA_MANIFEST.json",
                "path": row.get("path"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
                "url": row.get("url"),
                "notes": f"status={row.get('status')} name={row.get('name')}",
            }


def _iter_run_events(recorded_at: str) -> Iterator[dict[str, Any]]:
    outputs = PROJECT_ROOT / "outputs"
    if not outputs.exists():
        return
    for manifest_path in sorted(outputs.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        raw_ts = str(manifest.get("timestamp_utc") or "")
        try:
            ts = datetime.strptime(raw_ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
            basis = "run_manifest_timestamp"
        except ValueError:
            ts = _mtime_utc(manifest_path)
            basis = "file_mtime"
        yield {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "ts_utc": ts,
            "ts_basis": basis,
            "recorded_at_utc": recorded_at,
            "event_type": "experiment_run",
            "dataset": manifest_path.parent.name,
            "source": str(manifest.get("schema_version") or "run_manifest"),
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "bytes": None,
            "sha256": None,
            "url": None,
            "notes": json.dumps(
                {
                    key: manifest.get(key)
                    for key in ("provider", "model", "status", "live_call_count", "verdict")
                    if key in manifest
                },
                sort_keys=True,
            ),
        }


def build_events() -> list[dict[str, Any]]:
    recorded_at = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, Any]] = []
    events.extend(_iter_manifest_events(recorded_at))
    events.extend(_iter_download_events(recorded_at))
    events.extend(_iter_derived_events(recorded_at))
    events.extend(_iter_run_events(recorded_at))
    events.sort(key=lambda event: (str(event.get("ts_utc")), str(event.get("path"))))
    return events


def write_events(output_path: Path) -> int:
    events = build_events()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    return len(events)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the data provenance event stream.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    count = write_events(Path(args.output))
    print(json.dumps({"events_written": count, "output": args.output}))
    return 0


if __name__ == "__main__":
    main()
