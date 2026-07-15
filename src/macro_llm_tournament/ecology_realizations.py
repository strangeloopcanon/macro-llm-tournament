"""Append realized outcomes to a frozen ecology forecast run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import pandas as pd

from .ecology import SCHEMA_VERSION, _artifact_sha256, _file_sha256, _write_json


REALIZATION_SCHEMA_VERSION = "household_first_rolling_microeconomy_realization_append_v3"
RETROSPECTIVE_LABEL = "retrospective_realized_outcomes_append"
REALIZATION_OUTPUT_DIRECTORY = "realization_append"
REALIZATION_FILE_NAMES = (
    "realized_outcomes.csv",
    "forecast_errors.csv",
    "canonical_realizations.csv",
    "realization_manifest.json",
)
REALIZATION_METRICS = (
    "consumption_growth_pct",
    "revolving_credit_growth_pct",
    "employment_rate_pct",
    "price_growth_pct",
)
REALIZATION_COLUMNS = (
    "target_month",
    "metric",
    "value",
    "source",
    "source_url",
    "vintage_date",
    "release_date",
)
SCENARIOS = ("median",)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--realizations-csv", type=Path, required=True)
    return parser


def append_realizations(run_dir: Path, realizations_csv: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    realizations_csv = realizations_csv.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing manifest.json in run directory: {run_dir}")
    _refuse_existing_outputs(run_dir)
    lock_path, lock_fd = _acquire_append_lock(run_dir)
    try:
        _refuse_existing_outputs(run_dir)
        manifest = _load_manifest(manifest_path)
        _validate_manifest(manifest)
        _verify_original_artifacts(run_dir, manifest)
        forecast_paths = _load_forecast_paths(run_dir / "macro_forecast_paths.csv")
        realization_rows = _load_realization_rows(
            realizations_csv,
            expected_target=str(manifest["target_month"]),
            forecast_cutoff=str(manifest["as_of_date"]),
        )
        realization_row = {
            "target_month": str(manifest["target_month"]),
            **{row["metric"]: row["value"] for row in realization_rows},
        }
        realized_outcomes = pd.DataFrame(
            [realization_row],
            columns=["target_month", *REALIZATION_METRICS],
        )
        forecast_errors = _build_forecast_errors(
            forecast_paths=forecast_paths,
            realization_row=realization_row,
            target_month=str(manifest["target_month"]),
        )

        return _stage_and_publish_append(
            run_dir=run_dir,
            realized_outcomes=realized_outcomes,
            forecast_errors=forecast_errors,
            realization_rows=realization_rows,
            source_manifest_path=manifest_path,
            realizations_csv=realizations_csv,
            forecast_cutoff=str(manifest["as_of_date"]),
        )
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _refuse_existing_outputs(run_dir: Path) -> None:
    existing = [
        name
        for name in (REALIZATION_OUTPUT_DIRECTORY, *REALIZATION_FILE_NAMES)
        if (run_dir / name).exists()
    ]
    if existing:
        raise ValueError(f"Realization outputs already exist: {', '.join(existing)}")


def _acquire_append_lock(run_dir: Path) -> tuple[Path, int]:
    lock_path = run_dir / ".realization_append.lock"
    try:
        return lock_path, os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "A realization append is already in progress or an interrupted append requires inspection"
        ) from exc


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return payload


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("forecast_frozen_before_realization") is not True:
        raise ValueError("manifest forecast_frozen_before_realization must be true")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest artifacts must be a non-empty object")
    if not isinstance(manifest.get("target_month"), str) or not manifest["target_month"]:
        raise ValueError("manifest target_month must be present")
    _parse_iso_date(manifest.get("as_of_date"), "manifest as_of_date")


def _verify_original_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest["artifacts"]
    for name, expected_hash in sorted(artifacts.items()):
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError("manifest artifacts entries must map file names to sha256 strings")
        artifact_path = run_dir / name
        if not artifact_path.exists():
            raise ValueError(f"Manifest artifact is missing: {name}")
        actual_hash = _artifact_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Manifest artifact hash mismatch for {name}")


def _load_forecast_paths(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Missing macro_forecast_paths.csv: {path}")
    forecast_paths = pd.read_csv(path)
    required_columns = {"scenario", *REALIZATION_METRICS}
    if not required_columns.issubset(set(forecast_paths.columns)):
        missing = sorted(required_columns.difference(forecast_paths.columns))
        raise ValueError(f"macro_forecast_paths.csv is missing required columns: {missing}")
    scenarios = tuple(forecast_paths["scenario"].astype(str))
    if scenarios != SCENARIOS:
        raise ValueError("macro_forecast_paths.csv must contain exactly one median row")
    return forecast_paths.loc[:, ["scenario", *REALIZATION_METRICS]].copy()


def _load_realization_rows(
    path: Path,
    *,
    expected_target: str,
    forecast_cutoff: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Missing realizations CSV: {path}")
    frame = pd.read_csv(path)
    columns = tuple(frame.columns)
    if columns != REALIZATION_COLUMNS:
        raise ValueError(
            "realizations CSV must contain exactly these columns in order: "
            + ", ".join(REALIZATION_COLUMNS)
        )
    if len(frame) != len(REALIZATION_METRICS):
        raise ValueError(f"realizations CSV must contain exactly {len(REALIZATION_METRICS)} rows")
    cutoff_date = _parse_iso_date(forecast_cutoff, "manifest as_of_date")
    normalized_by_metric: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        target_month = str(row["target_month"])
        if target_month != expected_target:
            raise ValueError(
                f"realizations CSV target_month {target_month!r} does not match manifest target {expected_target!r}"
            )
        metric = str(row["metric"])
        if metric not in REALIZATION_METRICS:
            raise ValueError(f"realizations CSV metric {metric!r} is not supported")
        if metric in normalized_by_metric:
            raise ValueError(f"realizations CSV contains duplicate metric {metric!r}")
        source = _required_text(row["source"], "source", metric)
        source_url = _required_text(row["source_url"], "source_url", metric)
        if not source_url.startswith(("https://", "http://")):
            raise ValueError(f"realizations CSV source_url for {metric} must be an http(s) URL")
        vintage_date = _parse_iso_date(row["vintage_date"], f"vintage_date for {metric}")
        release_date = _parse_iso_date(row["release_date"], f"release_date for {metric}")
        if release_date <= cutoff_date:
            raise ValueError(
                f"realizations CSV release_date for {metric} must be after forecast cutoff {forecast_cutoff}"
            )
        if vintage_date < release_date:
            raise ValueError(f"realizations CSV vintage_date for {metric} cannot precede release_date")
        value = row["value"]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"realizations CSV value for {metric} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"realizations CSV value for {metric} must be finite")
        normalized_by_metric[metric] = {
            "target_month": target_month,
            "metric": metric,
            "value": number,
            "source": source,
            "source_url": source_url,
            "vintage_date": vintage_date.isoformat(),
            "release_date": release_date.isoformat(),
        }
    missing = sorted(set(REALIZATION_METRICS).difference(normalized_by_metric))
    if missing:
        raise ValueError(f"realizations CSV is missing required metrics: {', '.join(missing)}")
    return [normalized_by_metric[metric] for metric in REALIZATION_METRICS]


def _required_text(value: Any, field: str, metric: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"realizations CSV {field} for {metric} must be non-empty")
    return value.strip()


def _parse_iso_date(value: Any, field: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO calendar date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO calendar date") from exc


def _stage_and_publish_append(
    *,
    run_dir: Path,
    realized_outcomes: pd.DataFrame,
    forecast_errors: pd.DataFrame,
    realization_rows: list[dict[str, Any]],
    source_manifest_path: Path,
    realizations_csv: Path,
    forecast_cutoff: str,
) -> dict[str, Any]:
    """Publish a complete append as one directory rename on the run filesystem."""

    staging_path: Path | None = None
    try:
        staging_path = Path(tempfile.mkdtemp(prefix=".realization_append-", suffix=".staging", dir=run_dir))
        realized_path = staging_path / "realized_outcomes.csv"
        errors_path = staging_path / "forecast_errors.csv"
        canonical_path = staging_path / "canonical_realizations.csv"
        realization_manifest_path = staging_path / "realization_manifest.json"
        realized_outcomes.to_csv(realized_path, index=False)
        forecast_errors.to_csv(errors_path, index=False)
        pd.DataFrame(realization_rows, columns=list(REALIZATION_COLUMNS)).to_csv(canonical_path, index=False)
        for path in (realized_path, errors_path, canonical_path):
            _fsync_file(path)
        realization_manifest = {
            "schema_version": REALIZATION_SCHEMA_VERSION,
            "retrospective_label": RETROSPECTIVE_LABEL,
            "target_month": realization_rows[0]["target_month"],
            "forecast_cutoff_as_of_date": forecast_cutoff,
            "error_definition": "realized_minus_forecast",
            "source_forecast_manifest_sha256": _file_sha256(source_manifest_path),
            "realizations_input_sha256": _file_sha256(realizations_csv),
            "canonical_realization_input_sha256": _file_sha256(canonical_path),
            "normalized_realization_rows": realization_rows,
            "output_artifacts_sha256": {
                "realized_outcomes.csv": _artifact_sha256(realized_path),
                "forecast_errors.csv": _artifact_sha256(errors_path),
                "canonical_realizations.csv": _artifact_sha256(canonical_path),
            },
        }
        _write_json(realization_manifest_path, realization_manifest)
        _fsync_file(realization_manifest_path)
        _fsync_directory(staging_path)
        os.replace(staging_path, run_dir / REALIZATION_OUTPUT_DIRECTORY)
        staging_path = None
        _fsync_directory(run_dir)
        return realization_manifest
    finally:
        if staging_path is not None:
            shutil.rmtree(staging_path, ignore_errors=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _build_forecast_errors(
    forecast_paths: pd.DataFrame,
    realization_row: dict[str, Any],
    target_month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        match = forecast_paths.loc[forecast_paths["scenario"].astype(str) == scenario]
        if len(match) != 1:
            raise ValueError(f"macro_forecast_paths.csv must contain exactly one {scenario} row")
        forecast_row = match.iloc[0]
        error_row: dict[str, Any] = {"target_month": target_month, "scenario": scenario}
        for metric in REALIZATION_METRICS:
            forecast_value = float(forecast_row[metric])
            realized_value = float(realization_row[metric])
            error_row[f"forecast_{metric}"] = forecast_value
            error_row[f"realized_{metric}"] = realized_value
            error_row[f"error_{metric}"] = realized_value - forecast_value
        rows.append(error_row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    append_realizations(run_dir=args.run_dir, realizations_csv=args.realizations_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
