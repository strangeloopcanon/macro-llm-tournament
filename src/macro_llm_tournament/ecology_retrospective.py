"""Run the household ecology over historical rolling one-month origins."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ecology import (
    _artifact_sha256,
    _file_sha256,
    _source_sha256,
    _write_json,
    run as run_ecology,
)
from .ecology_macro import MACRO_PREDICTION_SCHEMA_VERSION, MACRO_TARGET_CONTRACT


SCHEMA_VERSION = "household_ecology_retrospective_v8"
MAPPING_SCHEMA_VERSION = "household_ecology_macro_mapping_v5"
METRIC_MAPPINGS: dict[str, dict[str, str]] = MACRO_TARGET_CONTRACT
SCENARIOS = ("median",)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument(
        "--prospective-run",
        type=Path,
        help="Optional unscored ecology run to show as a separate prospective marker.",
    )
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
    value = float(row[mapping.get("actual_field", "target_value")])
    transform = mapping.get("actual_transform", "identity")
    if transform == "one_hundred_minus":
        value = 100.0 - value
    elif transform == "negative_first_difference":
        value = -(value - float(row["first_release_denominator_value"]))
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
        "first_release_denominator_value",
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


def _prospective_rows(run_dir: Path | None) -> pd.DataFrame:
    columns = [
        "target_month",
        "metric",
        "scenario",
        "prediction",
        "context_prediction",
    ]
    if run_dir is None:
        return pd.DataFrame(columns=columns)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = pd.read_csv(run_dir / "macro_forecast_paths.csv")
    if len(paths) != 1 or set(paths["scenario"].astype(str)) != set(SCENARIOS):
        raise ValueError("macro_forecast_paths.csv must contain exactly one median point path")
    rows = []
    source = paths.iloc[0]
    for metric, mapping in METRIC_MAPPINGS.items():
        rows.append(
            {
                "target_month": str(manifest["target_month"]),
                "metric": metric,
                "scenario": str(source["scenario"]),
                "prediction": float(source[metric]),
                "context_prediction": float(source[mapping["visible_baseline_field"]]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _chart_subtitle(joined: pd.DataFrame, model: str) -> str:
    origins = pd.to_datetime(joined["origin_month"])
    return (
        f"{model} rolling origins {origins.min():%b %Y}-{origins.max():%b %Y}; every origin is "
        "re-anchored to its SCF financial state, latest origin-safe SCE history, "
        "and origin-visible public information."
    )


def _write_chart(
    joined: pd.DataFrame,
    prospective: pd.DataFrame,
    output: Path,
    *,
    model: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    metrics = list(METRIC_MAPPINGS)
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.5), sharex=True)
    actual_color = "#2463A6"
    prediction_color = "#B23A2B"
    for axis, metric in zip(axes.flat, metrics, strict=True):
        mapping = METRIC_MAPPINGS[metric]
        subset = joined.loc[joined["metric"].eq(metric)].copy()
        pivot = subset.pivot(index="target_month", columns="scenario", values="prediction")
        dates = pd.to_datetime(pivot.index)
        actual = (
            subset.drop_duplicates("target_month")
            .set_index("target_month")
            .loc[pivot.index, "actual"]
            .to_numpy(dtype=float)
        )
        prediction = pivot["median"].to_numpy(dtype=float)
        axis.plot(
            dates,
            np.sign(prediction) if mapping["score_mode"] == "direction_only" else prediction,
            color=prediction_color,
            marker="s",
            linewidth=1.8,
            label="LLM economy",
        )
        axis.plot(
            dates,
            np.sign(actual) if mapping["score_mode"] == "direction_only" else actual,
            color=actual_color,
            marker="o",
            linewidth=2.1,
            label="First-release actual",
        )
        context = (
            subset.drop_duplicates("target_month")
            .set_index("target_month")
            .loc[pivot.index, "context_prediction"]
            .to_numpy(dtype=float)
        )
        axis.plot(
            dates,
            np.sign(context) if mapping["score_mode"] == "direction_only" else context,
            color="#6B6B6B",
            linestyle="--",
            linewidth=1.5,
            marker="^",
            label="Origin-visible baseline",
        )
        marker = prospective.loc[
            prospective["metric"].eq(metric) & prospective["scenario"].eq("median")
        ]
        if not marker.empty:
            axis.scatter(
                pd.to_datetime(marker["target_month"]),
                np.sign(marker["prediction"])
                if mapping["score_mode"] == "direction_only"
                else marker["prediction"],
                marker="D",
                s=54,
                facecolors="white",
                edgecolors=prediction_color,
                linewidths=1.8,
                zorder=4,
                label="Frozen, unscored",
            )
            axis.scatter(
                pd.to_datetime(marker["target_month"]),
                np.sign(marker["context_prediction"])
                if mapping["score_mode"] == "direction_only"
                else marker["context_prediction"],
                marker="D",
                s=42,
                facecolors="white",
                edgecolors="#6B6B6B",
                linewidths=1.4,
                zorder=4,
            )
        if metric != "unemployment_rate_level":
            axis.axhline(0.0, color="#888888", linewidth=0.7)
        axis.set_title(mapping["chart_title"], fontsize=10.5)
        axis.set_ylabel(mapping["unit_label"], fontsize=8.5, labelpad=7)
        if mapping["score_mode"] == "direction_only":
            axis.set_yticks([-1.0, 0.0, 1.0], ["contract", "zero", "expand"])
            axis.set_ylim(-1.25, 1.25)
        else:
            axis.margins(y=0.16)
        axis.text(
            0.01,
            0.98,
            mapping["mapping_quality"].replace("_", " "),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#666666",
        )
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1, :]:
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(
        "LLM Household Economy: One-Month Predictions vs First-Release Reality",
        x=0.05,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.971,
        _chart_subtitle(joined, model),
        ha="left",
        fontsize=10,
        color="#4A4A4A",
    )
    handles: list[Any] = []
    legend_labels: list[str] = []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.958),
        frameon=False,
        ncol=4,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.07, hspace=0.25, wspace=0.22)
    fig.text(
        0.05,
        0.012,
        f"Sources: rolling {model} household ecology; ALFRED/FRED first releases. Every panel uses the same target month; proxy strength is stated inside each panel.",
        ha="left",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _joined_rows(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["origin_month", "target_month", "as_of_date", "state_provenance", "scenario"]
    rows: list[pd.DataFrame] = []
    for metric, mapping in METRIC_MAPPINGS.items():
        required = {metric, mapping["visible_baseline_field"]}
        missing = required.difference(forecasts.columns)
        if missing:
            raise ValueError(
                f"forecast paths missing macro-contract fields for {metric}: "
                f"{', '.join(sorted(missing))}"
            )
        block = forecasts.loc[:, id_columns].copy()
        block["metric"] = metric
        block["prediction"] = forecasts[metric].to_numpy(dtype=float)
        block["context_prediction"] = forecasts[
            mapping["visible_baseline_field"]
        ].to_numpy(dtype=float)
        rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    joined = long.merge(actuals, on=["target_month", "metric"], how="left", validate="many_to_one")
    if joined["actual"].isna().any():
        raise ValueError("forecast-to-actual join produced missing outcomes")
    if (
        pd.to_datetime(joined["first_release_as_of_date"])
        <= pd.to_datetime(joined["as_of_date"])
    ).any():
        raise ValueError("realization was available at or before its forecast cutoff")
    joined["error"] = joined["actual"] - joined["prediction"]
    joined["absolute_error"] = joined["error"].abs()
    return joined.sort_values(["target_month", "metric", "scenario"]).reset_index(drop=True)


def _score_rows(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, metric), group in joined.groupby(["scenario", "metric"], sort=True):
        eligible = group.loc[group["state_provenance"].eq("fixed_survey_scf_anchor")]
        if eligible.empty:
            raise ValueError("retrospective score has no fixed SCE-SCF anchor observations")
        error = eligible["prediction"].to_numpy(dtype=float) - eligible["actual"].to_numpy(dtype=float)
        actual = eligible["actual"].to_numpy(dtype=float)
        prediction = eligible["prediction"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(actual, prediction)[0, 1])
            if len(eligible) > 1 and np.std(actual) > 0 and np.std(prediction) > 0
            else float("nan")
        )
        score_mode = METRIC_MAPPINGS[str(metric)]["score_mode"]
        if METRIC_MAPPINGS[str(metric)].get("direction_mode") == "change_from_visible_baseline":
            baseline = eligible["context_prediction"].to_numpy(dtype=float)
            direction_prediction = prediction - baseline
            direction_actual = actual - baseline
        else:
            direction_prediction = prediction
            direction_actual = actual
        direction_mask = (
            (np.abs(direction_prediction) > 1e-12)
            | (np.abs(direction_actual) > 1e-12)
        )
        direction_match = (
            np.sign(direction_prediction[direction_mask])
            == np.sign(direction_actual[direction_mask])
            if score_mode in {"full", "direction_only"}
            else np.asarray([], dtype=bool)
        )
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "n": len(eligible),
                "mae": float(np.mean(np.abs(error))) if score_mode == "full" else float("nan"),
                "rmse": (
                    float(np.sqrt(np.mean(np.square(error))))
                    if score_mode == "full"
                    else float("nan")
                ),
                "correlation": correlation if score_mode == "full" else float("nan"),
                "mean_bias": (
                    float(np.mean(error)) if score_mode == "full" else float("nan")
                ),
                "demeaned_rmse": (
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    (prediction - np.mean(prediction))
                                    - (actual - np.mean(actual))
                                )
                            )
                        )
                    )
                    if score_mode == "full"
                    else float("nan")
                ),
                "standard_deviation_ratio": (
                    float(np.std(prediction) / np.std(actual))
                    if score_mode == "full" and np.std(actual) > 0
                    else float("nan")
                ),
                "direction_accuracy": (
                    float(np.mean(direction_match))
                    if score_mode in {"full", "direction_only"} and direction_match.size
                    else float("nan")
                ),
                "direction_n": int(direction_match.size),
                "mapping_quality": str(group["mapping_quality"].iloc[0]),
                "score_mode": score_mode,
            }
        )
    return pd.DataFrame(rows)


def _origin_visible_context_score_rows(joined: pd.DataFrame) -> pd.DataFrame:
    eligible = joined.loc[
        joined["state_provenance"].eq("fixed_survey_scf_anchor")
    ].drop_duplicates(["target_month", "metric"])
    if "context_prediction" not in eligible.columns:
        return pd.DataFrame()
    context = eligible.copy()
    context["scenario"] = "origin_visible_baseline"
    context["prediction"] = context["context_prediction"]
    return _score_rows(context)


def _git_worktree_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    origins = _parse_origins(args.origins)
    _validate_origins(args.bundle, origins)
    if args.mode != "live" and args.max_live_calls != 0:
        raise ValueError("non-live retrospective runs must use --max-live-calls 0")
    if args.mode == "live" and args.max_live_calls < args.household_count * len(origins):
        raise ValueError("live-call cap must cover at least one call per household and origin")
    # Validate the optional chart marker before any paid child run starts.
    prospective = _prospective_rows(getattr(args, "prospective_run", None))

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
                expected_replay_sha256=None,
                cache_dir=args.cache_dir,
                output_dir=run_dir,
            )
            manifest = run_ecology(child_args)
            remaining_calls -= int(manifest["live_call_count"])
            child_manifests.append(manifest)
            frame = pd.read_csv(run_dir / "macro_forecast_paths.csv")
            if len(frame) != 1 or set(frame["scenario"].astype(str)) != set(SCENARIOS):
                raise ValueError("macro_forecast_paths.csv must contain exactly one median point path")
            frame.insert(0, "state_provenance", "fixed_survey_scf_anchor")
            frame.insert(0, "as_of_date", manifest["as_of_date"])
            frame.insert(0, "target_month", manifest["target_month"])
            frame.insert(0, "origin_month", manifest["origin_month"])
            forecast_frames.append(frame)

        forecasts = pd.concat(forecast_frames, ignore_index=True)
        target_months = list(dict.fromkeys(forecasts["target_month"].astype(str)))
        actuals = _realization_rows(args.targets, target_months)
        joined = _joined_rows(forecasts, actuals)
        scores = pd.concat(
            [_score_rows(joined), _origin_visible_context_score_rows(joined)],
            ignore_index=True,
        )

        forecasts.to_csv(staging / "one_step_forecasts_by_origin.csv", index=False)
        actuals.to_csv(staging / "realized_outcomes_by_target.csv", index=False)
        joined.to_csv(staging / "predicted_vs_actual.csv", index=False)
        scores.to_csv(staging / "retrospective_scores.csv", index=False)
        _write_chart(
            joined,
            prospective,
            staging / "predicted_vs_actual.png",
            model=args.model,
        )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "macro_prediction_schema_version": MACRO_PREDICTION_SCHEMA_VERSION,
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "evaluation_status": "retrospective_diagnostic_not_confirmatory",
            "outcomes_loaded_after_all_forecasts": True,
            "forecast_process_opened_realization_files": False,
            "score_eligibility": "all_fixed_survey_scf_anchor_rows",
            "initialization_transition": "fixed_survey_scf_anchor_reinitialized_at_each_rolling_origin",
            "forecast_semantics": {
                metric: mapping["note"] for metric, mapping in METRIC_MAPPINGS.items()
            },
            "origin_months": origins,
            "target_months": target_months,
            "mode": args.mode,
            "provider": args.provider,
            "model": args.model,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "git_worktree_dirty": _git_worktree_dirty(),
            "source_sha256": _source_sha256(),
            "source_binding_authority": "source_sha256",
            "household_count": args.household_count,
            "accepted_household_response_count": sum(
                int(row["accepted_household_response_count"]) for row in child_manifests
            ),
            "provider_response_count": sum(
                int(row["provider_response_count"]) for row in child_manifests
            ),
            "live_call_count": sum(int(row["live_call_count"]) for row in child_manifests),
            "fresh_accepted_response_count": sum(
                int(row["fresh_accepted_response_count"]) for row in child_manifests
            ),
            "failed_provider_attempt_count": sum(
                int(row["failed_provider_attempt_count"]) for row in child_manifests
            ),
            "cache_hit_count": sum(int(row["cache_hit_count"]) for row in child_manifests),
            "accepted_call_journal_coverage": {
                "eligible_response_count": sum(
                    int(row["accepted_call_journal_coverage"]["eligible_response_count"])
                    for row in child_manifests
                ),
                "matched_response_count": sum(
                    int(row["accepted_call_journal_coverage"]["matched_response_count"])
                    for row in child_manifests
                ),
                "missing_response_count": sum(
                    int(row["accepted_call_journal_coverage"]["missing_response_count"])
                    for row in child_manifests
                ),
                "malformed_journal_count": sum(
                    int(row["accepted_call_journal_coverage"]["malformed_journal_count"])
                    for row in child_manifests
                ),
            },
            "replay_verified": (
                args.mode == "replay"
                and all(bool(row["replay_verified"]) for row in child_manifests)
            ),
            "codex_tool_isolation_version": child_manifests[0]["codex_tool_isolation_version"],
            "codex_instruction_context_version": child_manifests[0][
                "codex_instruction_context_version"
            ],
            "local_text_file_tools_available_to_model": (
                False if args.mode in {"live", "replay"} else None
            ),
            "accounting_passed": all(bool(row["accounting_passed"]) for row in child_manifests),
            "mapping_contract": METRIC_MAPPINGS,
            "context_comparator": {
                "scenario": "origin_visible_baseline",
                "metrics": list(METRIC_MAPPINGS),
                "source": "the declared one-month change, or latest level for unemployment, visible at each forecast origin",
                "execution_role": "context_only_not_pre_applied",
            },
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
            "child_replay_verified_against_live_reference": {
                row["origin_month"]: row["replay_verified"] for row in child_manifests
            },
            "artifacts": {},
        }
        journal_coverage = manifest["accepted_call_journal_coverage"]
        journal_coverage["match_rate"] = (
            journal_coverage["matched_response_count"]
            / journal_coverage["eligible_response_count"]
            if journal_coverage["eligible_response_count"]
            else None
        )
        report = [
            "# Household Ecology Retrospective",
            "",
            "This diagnostic runs the current household ecology as rolling one-month estimates over historical origin vintages. "
            "Every origin is scored in rolling mode. Each forecast restarts from the same fixed SCE-SCF household anchor; only information available at that origin updates. Simulated balances and forecast errors never enter the next origin. "
            "The forecast process does not open realization values. Realized outcomes are loaded only after every forecast is complete.",
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
            mae = f"{row.mae:.3f}" if math.isfinite(row.mae) else "not scored"
            rmse = f"{row.rmse:.3f}" if math.isfinite(row.rmse) else "not scored"
            direction = (
                f"{row.direction_accuracy:.1%} (n={row.direction_n})"
                if math.isfinite(row.direction_accuracy)
                else "not scored"
            )
            report.append(f"| `{row.metric}` | {mae} | {rmse} | {direction} |")
        report.extend(
            [
                "",
                "## Origin-Visible Baselines",
                "",
                "Each origin-visible one-month change, plus the latest unemployment-rate level, is scored separately. These values are context supplied to households; they are not executed by the economy.",
                "",
                "| Metric | LLM RMSE | Visible-baseline RMSE | LLM direction | Baseline direction |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        visible_scores = scores.loc[
            scores["scenario"].eq("origin_visible_baseline")
        ].set_index("metric")
        llm_scores = median_scores.set_index("metric")
        for metric in METRIC_MAPPINGS:
            llm = llm_scores.loc[metric]
            visible = visible_scores.loc[metric]
            llm_rmse = f"{llm.rmse:.3f}" if math.isfinite(llm.rmse) else "not scored"
            visible_rmse = (
                f"{visible.rmse:.3f}" if math.isfinite(visible.rmse) else "not scored"
            )
            llm_direction = (
                f"{llm.direction_accuracy:.1%}"
                if math.isfinite(llm.direction_accuracy)
                else "not scored"
            )
            visible_direction = (
                f"{visible.direction_accuracy:.1%}"
                if math.isfinite(visible.direction_accuracy)
                else "not scored"
            )
            report.append(
                f"| `{metric}` | {llm_rmse} | {visible_rmse} | {llm_direction} | {visible_direction} |"
            )
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
