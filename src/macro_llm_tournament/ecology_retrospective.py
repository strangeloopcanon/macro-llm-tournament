"""Run the household ecology recursively over historical rolling origins."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ecology import _artifact_sha256, _file_sha256, _write_json, run as run_ecology


SCHEMA_VERSION = "household_ecology_retrospective_v1"
MAPPING_SCHEMA_VERSION = "household_ecology_macro_mapping_v1"
METRIC_MAPPINGS: dict[str, dict[str, str]] = {
    "consumption_growth_pct": {
        "target_name": "pce_growth_pct",
        "series_id": "PCE",
        "actual_field": "target_value",
        "actual_transform": "identity",
        "mapping_quality": "closest_aggregate_proxy",
        "note": "Ecology nominal household consumption growth mapped to nominal PCE growth.",
    },
    "saving_rate_pct": {
        "target_name": "personal_saving_rate_change",
        "series_id": "PSAVERT",
        "actual_field": "first_release_value",
        "actual_transform": "identity",
        "source_value_semantics": "first_release_series_level_not_derived_change_target",
        "mapping_quality": "directional_proxy",
        "note": "Ecology saving-rate level mapped to the raw first-release PSAVERT series level stored on the change-target row; target_value is deliberately not used.",
    },
    "revolving_credit_growth_pct": {
        "target_name": "revolving_credit_growth_pct",
        "series_id": "REVOLSL",
        "actual_field": "target_value",
        "actual_transform": "identity",
        "mapping_quality": "directional_proxy",
        "note": "Ecology revolving-debt growth mapped to aggregate revolving consumer-credit growth.",
    },
    "employment_rate_pct": {
        "target_name": "unemployment_rate_level",
        "series_id": "UNRATE",
        "actual_field": "first_release_value",
        "actual_transform": "one_hundred_minus",
        "mapping_quality": "directional_proxy",
        "note": "Ecology employed-household share mapped to 100 minus the first-release unemployment rate.",
    },
    "price_growth_pct": {
        "target_name": "pce_price_growth_pct",
        "series_id": "PCEPI",
        "actual_field": "target_value",
        "actual_transform": "identity",
        "mapping_quality": "directional_proxy",
        "note": "Ecology unit-price growth mapped to PCE price-index growth.",
    },
}
SCENARIOS = ("downside", "median", "upside")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", required=True, help="Inclusive month-start range START:END")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--households", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "replay", "live"), required=True)
    parser.add_argument("--provider", default="codex_cli", choices=("codex_cli",))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--household-count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _parse_origins(value: str) -> list[str]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("--origins must be START:END")
    start, end = (pd.Timestamp(piece) for piece in pieces)
    if start.day != 1 or end.day != 1 or end < start:
        raise ValueError("origin bounds must be ascending month starts")
    return [stamp.date().isoformat() for stamp in pd.date_range(start, end, freq="MS")]


def _validate_origins(bundle: Path, origins: list[str]) -> None:
    origin_path = bundle / "origins.csv"
    if not origin_path.exists():
        raise ValueError(f"bundle is missing origins.csv: {bundle}")
    available = set(pd.read_csv(origin_path)["origin_month"].astype(str))
    missing = [origin for origin in origins if origin not in available]
    if missing:
        raise ValueError(f"origins absent from bundle: {', '.join(missing)}")


def _mapped_actual(row: pd.Series, mapping: dict[str, str]) -> float:
    value = float(row[mapping["actual_field"]])
    if mapping["actual_transform"] == "one_hundred_minus":
        value = 100.0 - value
    if not math.isfinite(value):
        raise ValueError("mapped actual must be finite")
    return value


def _realization_rows(targets_path: Path, target_months: list[str]) -> pd.DataFrame:
    targets = pd.read_csv(targets_path)
    required = {
        "target_name",
        "series_id",
        "target_observation_date",
        "first_release_as_of_date",
        "first_release_value",
        "target_value",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"targets CSV missing fields: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        for metric, mapping in METRIC_MAPPINGS.items():
            match = targets.loc[
                targets["target_observation_date"].astype(str).eq(target_month)
                & targets["target_name"].astype(str).eq(mapping["target_name"])
                & targets["series_id"].astype(str).eq(mapping["series_id"])
            ]
            if len(match) != 1:
                raise ValueError(
                    f"expected one first-release target for {target_month} {mapping['target_name']}; "
                    f"found {len(match)}"
                )
            source = match.iloc[0]
            rows.append(
                {
                    "target_month": target_month,
                    "metric": metric,
                    "actual": _mapped_actual(source, mapping),
                    "target_name": mapping["target_name"],
                    "series_id": mapping["series_id"],
                    "source_target_origin": str(source.get("origin_month", "")),
                    "first_release_as_of_date": str(source["first_release_as_of_date"]),
                    "mapping_quality": mapping["mapping_quality"],
                    "mapping_note": mapping["note"],
                }
            )
    return pd.DataFrame(rows)


def _joined_rows(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["origin_month", "target_month", "as_of_date", "scenario"]
    long = forecasts.melt(
        id_vars=id_columns,
        value_vars=list(METRIC_MAPPINGS),
        var_name="metric",
        value_name="prediction",
    )
    joined = long.merge(actuals, on=["target_month", "metric"], how="left", validate="many_to_one")
    if joined["actual"].isna().any():
        raise ValueError("forecast-to-actual join produced missing outcomes")
    joined["error"] = joined["actual"] - joined["prediction"]
    joined["absolute_error"] = joined["error"].abs()
    return joined.sort_values(["target_month", "metric", "scenario"]).reset_index(drop=True)


def _score_rows(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, metric), group in joined.groupby(["scenario", "metric"], sort=True):
        error = group["prediction"].to_numpy(dtype=float) - group["actual"].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(actual, prediction)[0, 1])
            if len(group) > 1 and np.std(actual) > 0 and np.std(prediction) > 0
            else float("nan")
        )
        direction_match = np.sign(prediction) == np.sign(actual)
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "correlation": correlation,
                "direction_accuracy": float(np.mean(direction_match)),
                "mapping_quality": str(group["mapping_quality"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    origins = _parse_origins(args.origins)
    _validate_origins(args.bundle, origins)
    if args.mode != "live" and args.max_live_calls != 0:
        raise ValueError("non-live retrospective runs must use --max-live-calls 0")
    if args.mode == "live" and args.max_live_calls < args.household_count * len(origins):
        raise ValueError("live-call cap must cover at least one call per household and origin")

    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    remaining_calls = args.max_live_calls
    per_origin_cap = math.ceil(args.max_live_calls / len(origins)) if args.mode == "live" else 0
    forecast_frames: list[pd.DataFrame] = []
    child_manifests: list[dict[str, Any]] = []
    prior_state: Path | None = None
    try:
        for origin in origins:
            run_dir = staging / "runs" / origin
            child_args = argparse.Namespace(
                origin=origin,
                mode=args.mode,
                provider=args.provider,
                model=args.model,
                max_live_calls=min(per_origin_cap, remaining_calls),
                workers=args.workers,
                household_count=args.household_count,
                households=args.households,
                history=args.history,
                bundle=args.bundle,
                state_json=prior_state,
                expected_replay_sha256=None,
                cache_dir=args.cache_dir,
                output_dir=run_dir,
            )
            manifest = run_ecology(child_args)
            remaining_calls -= int(manifest["live_call_count"])
            child_manifests.append(manifest)
            frame = pd.read_csv(run_dir / "macro_forecast_paths.csv")
            frame.insert(0, "as_of_date", manifest["as_of_date"])
            frame.insert(0, "target_month", manifest["target_month"])
            frame.insert(0, "origin_month", manifest["origin_month"])
            forecast_frames.append(frame)
            prior_state = run_dir / "median_next_state.json"

        forecasts = pd.concat(forecast_frames, ignore_index=True)
        target_months = list(dict.fromkeys(forecasts["target_month"].astype(str)))
        actuals = _realization_rows(args.targets, target_months)
        joined = _joined_rows(forecasts, actuals)
        scores = _score_rows(joined)

        forecasts.to_csv(staging / "one_step_forecasts_by_origin.csv", index=False)
        actuals.to_csv(staging / "realized_outcomes_by_target.csv", index=False)
        joined.to_csv(staging / "predicted_vs_actual.csv", index=False)
        scores.to_csv(staging / "retrospective_scores.csv", index=False)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "evaluation_status": "retrospective_diagnostic_not_confirmatory",
            "state_policy": "median_recursive_spine",
            "outcomes_loaded_after_all_forecasts": True,
            "origin_months": origins,
            "target_months": target_months,
            "mode": args.mode,
            "provider": args.provider,
            "model": args.model,
            "household_count": args.household_count,
            "live_call_count": sum(int(row["live_call_count"]) for row in child_manifests),
            "cache_hit_count": sum(int(row["cache_hit_count"]) for row in child_manifests),
            "accounting_passed": all(bool(row["accounting_passed"]) for row in child_manifests),
            "mapping_contract": METRIC_MAPPINGS,
            "inputs_sha256": {
                "bundle": _artifact_sha256(args.bundle),
                "targets": _file_sha256(args.targets),
                "households": _file_sha256(args.households),
                "history": _file_sha256(args.history),
            },
            "child_run_manifest_sha256": {
                row["origin_month"]: _file_sha256(staging / "runs" / row["origin_month"] / "manifest.json")
                for row in child_manifests
            },
            "artifacts": {},
        }
        report = [
            "# Household Ecology Retrospective",
            "",
            "This diagnostic runs the current household ecology recursively over historical origin vintages. "
            "The median simulated state is carried into the next origin; realized outcomes are loaded only after every forecast is complete.",
            "",
            "It is not confirmatory evidence. The dates are historical and the model may know them. Its purpose is to reveal sign, scale, and mapping failures before prospective outcomes arrive.",
            "",
            "## Mapping",
            "",
            "| Ecology output | Realized proxy | Quality |",
            "| --- | --- | --- |",
        ]
        for metric, mapping in METRIC_MAPPINGS.items():
            report.append(f"| `{metric}` | `{mapping['series_id']}` / `{mapping['target_name']}` | `{mapping['mapping_quality']}` |")
        median_scores = scores.loc[scores["scenario"].eq("median")]
        report.extend(["", "## Median-Path Scores", "", "| Metric | MAE | RMSE | Direction |", "| --- | ---: | ---: | ---: |"])
        for row in median_scores.itertuples(index=False):
            report.append(f"| `{row.metric}` | {row.mae:.3f} | {row.rmse:.3f} | {row.direction_accuracy:.1%} |")
        report.extend(["", f"Accounting passed across all child runs: **{manifest['accounting_passed']}**."])
        (staging / "retrospective_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        for path in sorted(staging.iterdir()):
            if path.name != "manifest.json":
                manifest["artifacts"][path.name] = _artifact_sha256(path)
        _write_json(staging / "manifest.json", manifest)
        staging.rename(output)
        return manifest
    except Exception:
        # Keep accepted responses in the dedicated cache, but never expose a partial result bundle.
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
