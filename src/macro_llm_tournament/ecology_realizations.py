"""Append realized outcomes to a frozen ecology forecast run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .ecology import SCHEMA_VERSION, _artifact_sha256, _file_sha256, _write_json


REALIZATION_SCHEMA_VERSION = "household_first_rolling_microeconomy_realization_append_v1"
RETROSPECTIVE_LABEL = "retrospective_realized_outcomes_append"
REALIZATION_FILE_NAMES = (
    "realized_outcomes.csv",
    "forecast_errors.csv",
    "realization_manifest.json",
)
REALIZATION_METRICS = (
    "consumption_growth_pct",
    "saving_rate_pct",
    "revolving_credit_growth_pct",
    "employment_rate_pct",
    "price_growth_pct",
)
REALIZATION_COLUMNS = ("target_month",) + REALIZATION_METRICS
SCENARIOS = ("downside", "median", "upside")


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
    manifest = _load_manifest(manifest_path)
    _validate_manifest(manifest)
    _verify_original_artifacts(run_dir, manifest)
    forecast_paths = _load_forecast_paths(run_dir / "macro_forecast_paths.csv")
    realization_row = _load_realization_row(realizations_csv, expected_target=str(manifest["target_month"]))
    realized_outcomes = pd.DataFrame([realization_row], columns=list(REALIZATION_COLUMNS))
    forecast_errors = _build_forecast_errors(
        forecast_paths=forecast_paths,
        realization_row=realization_row,
        target_month=str(manifest["target_month"]),
    )

    realized_path = run_dir / "realized_outcomes.csv"
    errors_path = run_dir / "forecast_errors.csv"
    realization_manifest_path = run_dir / "realization_manifest.json"

    realized_outcomes.to_csv(realized_path, index=False)
    forecast_errors.to_csv(errors_path, index=False)

    realization_manifest = {
        "schema_version": REALIZATION_SCHEMA_VERSION,
        "retrospective_label": RETROSPECTIVE_LABEL,
        "target_month": str(manifest["target_month"]),
        "error_definition": "realized_minus_forecast",
        "source_forecast_manifest_sha256": _file_sha256(manifest_path),
        "realizations_input_sha256": _file_sha256(realizations_csv),
        "output_artifacts_sha256": {
            "realized_outcomes.csv": _artifact_sha256(realized_path),
            "forecast_errors.csv": _artifact_sha256(errors_path),
        },
    }
    _write_json(realization_manifest_path, realization_manifest)
    return realization_manifest


def _refuse_existing_outputs(run_dir: Path) -> None:
    existing = [name for name in REALIZATION_FILE_NAMES if (run_dir / name).exists()]
    if existing:
        raise ValueError(f"Realization outputs already exist: {', '.join(existing)}")


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
    if set(scenarios) != set(SCENARIOS) or len(forecast_paths) != len(SCENARIOS):
        raise ValueError("macro_forecast_paths.csv must contain exactly downside, median, and upside rows")
    return forecast_paths.loc[:, ["scenario", *REALIZATION_METRICS]].copy()


def _load_realization_row(path: Path, expected_target: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing realizations CSV: {path}")
    frame = pd.read_csv(path)
    columns = tuple(frame.columns)
    if columns != REALIZATION_COLUMNS:
        raise ValueError(
            "realizations CSV must contain exactly these columns in order: "
            + ", ".join(REALIZATION_COLUMNS)
        )
    if len(frame) != 1:
        raise ValueError("realizations CSV must contain exactly one row")
    row = frame.iloc[0]
    target_month = str(row["target_month"])
    if target_month != expected_target:
        raise ValueError(
            f"realizations CSV target_month {target_month!r} does not match manifest target {expected_target!r}"
        )
    normalized: dict[str, Any] = {"target_month": target_month}
    for metric in REALIZATION_METRICS:
        value = row[metric]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"realizations CSV field {metric} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"realizations CSV field {metric} must be finite")
        normalized[metric] = number
    return normalized


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
