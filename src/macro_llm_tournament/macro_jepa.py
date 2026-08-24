"""Train and evaluate an origin-safe macro StructuredStateJEPA companion model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ecology_macro import MACRO_TARGET_CONTRACT, fisher_real_growth
from .ecology_retrospective import _realization_rows
from .macro_jepa_data import (
    EVENT_ACTION_CATEGORICAL_COLUMNS,
    EVENT_ACTION_COLUMNS,
    SERIES_VALUE_NORMALIZATION,
    STATE_SERIES,
    TARGET_SPECS,
    apply_origin_information_overrides,
    build_event_state_panel,
    build_origin_state_panel,
    event_observation_columns,
    first_release_targets,
    load_macro_events,
    origin_visible_baselines,
    validate_canonical_series_units,
)


SCHEMA_VERSION = "macro_event_residual_jepa_v3"
READOUT_CUTOFF = "2024-01-15"
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
RESIDUAL_GAIN_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTEXT_LENGTH = 16
BACKTEST_ORIGINS = tuple(
    month.date().isoformat()
    for month in pd.date_range("2024-01-01", "2025-12-01", freq="MS")
)
EVALUATION_ORIGINS = (
    "2026-01-01",
    "2026-02-01",
    "2026-03-01",
    "2026-04-01",
)
DIRECT_METRICS = tuple(TARGET_SPECS)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VEI_RUNTIME_BASE_COMMIT = "682a3691bd5796951e8a7e7f5c1b78c35b2d2828"
VEI_RUNTIME_FILE_SHA256 = {
    "pipeline/train/model.py": "0fae66850c172348091d14a526ceb4fbc641a36b2a8bea8720cd062cba97e4c4",
    "pipeline/train/schema.py": "e0c49307686f3a4d453b54fac4283ef95ae9872d677c4006c7b9c23885decd51",
    "pipeline/train/storage.py": "52608d682f0cda4a320eff638d3a448800acfc8549480ae8b77798fa8ed3463c",
    "pipeline/train/training.py": "7a2be984b987b3ab5afb2fbea5c0b6bcc63b6238411c1f00a9dbfc5f68e8aabc",
}
DEFAULT_VEI_RUNTIME_PATCH = (
    PROJECT_ROOT / "vendor/vei_macro_jepa_runtime_682a369.patch"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(
            "work/fred_dense_events_event_jepa_v2/dense_macro_events.jsonl"
        ),
    )
    parser.add_argument(
        "--comparison-panel",
        type=Path,
        default=Path(
            "outputs/household_ecology_observability_v24/observed_comparison_panel.csv"
        ),
    )
    parser.add_argument(
        "--frozen-targets",
        type=Path,
        default=Path(
            "work/dynamic_macro/frozen_2026_01_2026_05_common_month_v1/targets.csv"
        ),
    )
    parser.add_argument(
        "--retrospective-run",
        type=Path,
        default=Path("outputs/household_ecology_retrospective_2026_01_04_v24"),
    )
    parser.add_argument(
        "--vei-root",
        type=Path,
        default=PROJECT_ROOT.parent / "vei",
    )
    parser.add_argument(
        "--vei-runtime-patch",
        type=Path,
        default=DEFAULT_VEI_RUNTIME_PATCH,
        help="Tracked patch reproducing the exact locked VEI training runtime.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("work/macro_jepa_v3")
    )
    parser.add_argument(
        "--report-figure",
        type=Path,
        default=Path("reports/current_macro_jepa_comparison.png"),
    )
    parser.add_argument(
        "--report-evidence-json",
        type=Path,
        default=None,
        help="Optional tracked evidence bundle for scores and provenance.",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--refresh-comparison-only",
        action="store_true",
        help="replace the household/actual/baseline overlay without retraining JEPA",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    figure = output / "macro_jepa_comparison.png"
    if args.plot_only:
        write_chart(output / "comparison_predictions.csv", figure)
        return {"figure": str(figure)}
    if args.refresh_comparison_only:
        return _refresh_comparison(args, output=output, figure=figure)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    vei_runtime = _verify_vei_runtime(
        root=args.vei_root,
        reproduction_patch=args.vei_runtime_patch,
    )
    _add_vei_to_path(args.vei_root)
    from pipeline.train.schema import ModelConfig, TrainConfig
    from pipeline.train.storage import (
        collate_step_batches,
        finalize_processed_dataset,
        load_processed_dataset,
        rows_to_step_batch,
    )
    from pipeline.train.training import (
        evaluate_model,
        load_trained_model,
        train_model,
    )

    selected_series = set(STATE_SERIES) | {
        series_id for series_id, _ in TARGET_SPECS.values()
    }
    events = load_macro_events(args.events, series_ids=selected_series)
    targets = first_release_targets(events)
    event_start_month = (
        pd.Timestamp(min(event.vintage_date for event in events))
        .to_period("M")
        .to_timestamp()
        .date()
        .isoformat()
    )
    overrides = _origin_overrides(args.retrospective_run)
    exact_as_of_dates = {
        month: str(json.loads(path.read_text(encoding="utf-8"))["as_of_date"])
        for month, path in overrides.items()
    }
    anchors = build_origin_state_panel(
        events,
        start_month=event_start_month,
        end_month="2026-04-01",
        as_of_day=15,
        as_of_date_overrides=exact_as_of_dates,
    )
    anchors = apply_origin_information_overrides(anchors, overrides)
    validate_canonical_series_units(anchors)
    baselines = origin_visible_baselines(anchors)
    _audit_origin_visible_baselines(baselines, args.comparison_panel)
    panel = build_event_state_panel(
        events,
        anchors=anchors,
        start_date=event_start_month,
        end_date=str(anchors["as_of_date"].max()),
    )
    observations = event_observation_columns()

    panel.to_csv(output / "origin_safe_event_states.csv", index=False)
    anchors.to_csv(output / "monthly_anchor_states.csv", index=False)
    baselines.to_csv(output / "origin_visible_baselines.csv", index=False)
    targets.to_csv(output / "first_release_targets.csv", index=False)
    _write_json(
        output / "extraction_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state_start": str(panel["as_of_date"].min()),
            "state_end": str(panel["as_of_date"].max()),
            "row_count": len(panel),
            "release_day_count": int(panel["row_kind"].eq("release_day").sum()),
            "forecast_anchor_count": int(
                panel["row_kind"].eq("monthly_forecast_anchor").sum()
            ),
            "split_counts": {
                str(key): int(value)
                for key, value in panel["split"].value_counts().items()
            },
            "event_count": len(events),
            "state_series": list(STATE_SERIES),
            "series_value_normalization": SERIES_VALUE_NORMALIZATION,
            "target_specs": {
                key: {"series_id": value[0], "transform": value[1]}
                for key, value in TARGET_SPECS.items()
            },
            "origin_information_overrides": {
                month: {"path": str(path), "sha256": _file_sha256(path)}
                for month, path in overrides.items()
            },
            "readout_cutoff": READOUT_CUTOFF,
        },
    )

    dataset = finalize_processed_dataset(
        raw_steps=panel,
        output_dir=output / "dataset",
        dataset_kind="timeseries",
        observation_numeric_columns=observations,
        observation_categorical_columns=["row_kind"],
        action_numeric_columns=list(EVENT_ACTION_COLUMNS),
        action_categorical_columns=list(EVENT_ACTION_CATEGORICAL_COLUMNS),
        auxiliary_numeric_targets=[],
        metadata_columns=[
            "origin_month",
            "as_of_date",
            "source_event_count",
            "source_event_ids_sha256",
        ],
        notes={
            "claim_scope": "event_time_residual_macro_companion_not_hank_hybrid",
            "readout_cutoff": READOUT_CUTOFF,
            "event_alignment": "pre_release_state_plus_release_bundle_to_next_state",
        },
        context_length=CONTEXT_LENGTH,
        prediction_horizon=1,
    )
    model_config = ModelConfig(
        encoder_type="flat",
        d_state=96,
        d_action=64,
        depth=2,
        heads=4,
        dropout=0.05,
        sigreg_lambda=0.005,
        sigreg_knots=17,
        sigreg_num_proj=32,
        contrastive_lambda=0.15,
        contrastive_temperature=0.1,
        target_ema_decay=0.99,
        time_encoding="relative_fourier",
        time_fourier_bands=4,
        auxiliary_loss_weight=0.0,
        context_length=CONTEXT_LENGTH,
    )
    train_config = TrainConfig(
        batch_size=128,
        epochs=int(args.epochs),
        lr=8e-4,
        weight_decay=1e-4,
        seed=20260715,
        device=str(args.device),
        rollout_horizon=1,
        rollout_loss_weight=0.0,
        warmup_epochs=3,
        cosine_lr=True,
        early_stopping_patience=6,
        early_stopping_min_delta=1e-4,
        action_sensitivity_weight=0.0,
        action_sensitivity_margin=0.01,
        grad_clip_norm=1.0,
    )
    _write_json(
        output / "locked_training_spec.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_config": model_config.model_dump(mode="json"),
            "train_config": train_config.model_dump(mode="json"),
            "context_length": CONTEXT_LENGTH,
            "readout_alpha_grid": list(ALPHA_GRID),
            "residual_gain_grid": list(RESIDUAL_GAIN_GRID),
            "readout_cutoff": READOUT_CUTOFF,
            "primary_metric": "consumption_growth_pct",
            "readout_train_period": "1995-01-01:2021-12-31",
            "readout_validation_period": "2022-01-01:2023-12-31",
            "blocked_backtest_origins": list(BACKTEST_ORIGINS),
            "evaluation_origins": list(EVALUATION_ORIGINS),
            "prospective_jepa_status": (
                "not_emitted_until_a_complete_origin_state_is_available"
            ),
            "evaluation_actual_source": str(args.frozen_targets.resolve()),
        },
    )
    artifacts = train_model(
        dataset_root=dataset.root,
        output_dir=output / "model",
        model_config=model_config,
        train_config=train_config,
    )
    evaluation = evaluate_model(
        dataset_root=dataset.root,
        checkpoint_path=artifacts.model_path,
        device=str(args.device),
    )
    _write_json(output / "jepa_evaluation.json", evaluation.model_dump(mode="json"))

    prepared = load_processed_dataset(dataset.root)
    model, schema = load_trained_model(artifacts.model_path)
    model = model.to(str(args.device)).eval()
    latent_frame, state_frame = _prediction_features(
        prepared,
        model,
        schema,
        rows_to_step_batch=rows_to_step_batch,
        collate_step_batches=collate_step_batches,
        device=str(args.device),
    )
    jepa_predictions, jepa_readouts = _fit_readouts(
        latent_frame,
        targets,
        baselines=baselines,
        feature_prefix="latent__",
        output_dir=output / "readouts" / "jepa",
    )
    state_predictions, state_readouts = _fit_readouts(
        state_frame,
        targets,
        baselines=baselines,
        feature_prefix="state__",
        output_dir=output / "readouts" / "macro_state_ridge",
    )
    constant_predictions, constant_metadata = _historical_constant_predictions(
        latent_frame, targets
    )
    _write_json(
        output / "readout_selection.json",
        {
            "cutoff": READOUT_CUTOFF,
            "alpha_grid": list(ALPHA_GRID),
            "residual_gain_grid": list(RESIDUAL_GAIN_GRID),
            "jepa": jepa_readouts,
            "macro_state_ridge": state_readouts,
            "historical_constants": constant_metadata,
        },
    )

    frozen_actuals = _realization_rows(
        args.frozen_targets,
        ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"],
    )
    comparison = _comparison_table(
        comparison_panel=args.comparison_panel,
        targets=targets,
        frozen_actuals=frozen_actuals,
        jepa_predictions=jepa_predictions,
        state_predictions=state_predictions,
        constant_predictions=constant_predictions,
    )
    comparison.to_csv(output / "comparison_predictions.csv", index=False)
    scores = _score_predictions(
        comparison,
        sources=(
            "llm_household_economy",
            "origin_visible_baseline",
            "event_jepa_residual",
            "macro_state_ridge",
            "historical_mean",
            "historical_median",
        ),
    )
    scores.to_csv(output / "score_table.csv", index=False)
    backtest = _backtest_table(
        targets=targets,
        baselines=baselines,
        jepa_predictions=jepa_predictions,
        state_predictions=state_predictions,
        constant_predictions=constant_predictions,
    )
    backtest.to_csv(output / "blocked_backtest_predictions.csv", index=False)
    backtest_scores = _score_predictions(
        backtest,
        sources=(
            "origin_visible_baseline",
            "event_jepa_residual",
            "macro_state_ridge",
            "historical_mean",
            "historical_median",
        ),
        evaluation_status="blocked_backtest_2024_2025",
    )
    backtest_scores.to_csv(output / "blocked_backtest_score_table.csv", index=False)
    _plot_with_system_python(output, figure)
    args.report_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, args.report_figure)

    report = _report_text(
        scores,
        backtest_scores,
        evaluation.model_dump(mode="json"),
        jepa_readouts=jepa_readouts,
    )
    (output / "macro_jepa_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "Event-time residual StructuredStateJEPA companion benchmark. The design "
            "was locked before scoring 2024-2026. It does not use household-ecology "
            "outputs as training features and is not an LLM-HANK-JEPA hybrid."
        ),
        "events_sha256": _file_sha256(args.events),
        "frozen_targets_sha256": _file_sha256(args.frozen_targets),
        "comparison_panel_sha256": _file_sha256(args.comparison_panel),
        "vei_root": str(args.vei_root.resolve()),
        "vei_git_commit": _git_commit(args.vei_root),
        "vei_runtime": vei_runtime,
        "model_checkpoint": str(Path(artifacts.model_path).resolve()),
        "model_checkpoint_sha256": _file_sha256(Path(artifacts.model_path)),
        "readout_cutoff": READOUT_CUTOFF,
        "self_supervised_train_split_end": "2021-12-31",
        "self_supervised_validation_split": "2022-01-01:2023-12-31",
        "readout_selection_split": "2022-01-01:2023-12-31",
        "blocked_backtest": "2024-01-01:2025-12-01",
        "retrospective_evaluation": "2026-01-01:2026-04-01",
        "prospective_jepa_status": (
            "not_emitted_until_a_complete_origin_state_is_available"
        ),
        "primary_score": _primary_score_dict(scores),
        "blocked_backtest_primary_score": _primary_score_dict(backtest_scores),
        "artifacts": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][str(path.relative_to(output))] = _file_sha256(path)
    _write_json(output / "manifest.json", manifest)
    _write_tracked_evidence(
        output=output,
        destination=args.report_evidence_json,
        manifest=manifest,
    )
    return manifest


def _household_overlay_table(
    existing_comparison: pd.DataFrame, comparison_panel: pd.DataFrame
) -> pd.DataFrame:
    source_names = {
        "first_release_actual": "first_release_actual",
        "llm_household_economy": "llm_household_economy",
        "origin_visible_baseline": "origin_visible_baseline",
    }
    required = {
        "origin_month",
        "target_month",
        "metric",
        "evaluation_status",
        "series_role",
        "value",
    }
    missing = sorted(required - set(comparison_panel.columns))
    if missing:
        raise ValueError(
            "comparison panel is missing overlay columns: " + ", ".join(missing)
        )
    replacement = comparison_panel[list(required)].copy()
    replacement["source"] = replacement.pop("series_role").map(source_names)
    if replacement["source"].isna().any():
        unknown = sorted(
            comparison_panel.loc[replacement["source"].isna(), "series_role"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"comparison panel has unsupported series roles: {unknown}")
    kept = existing_comparison.loc[
        ~existing_comparison["source"].isin(source_names.values())
    ].copy()
    result = pd.concat([kept, replacement], ignore_index=True)
    keys = ["origin_month", "target_month", "metric", "source"]
    if result.duplicated(keys).any():
        raise ValueError("refreshed comparison contains duplicate source predictions")
    return result.sort_values(["metric", "target_month", "source"]).reset_index(
        drop=True
    )


def _refresh_comparison(
    args: argparse.Namespace, *, output: Path, figure: Path
) -> dict[str, Any]:
    required = (
        "comparison_predictions.csv",
        "blocked_backtest_score_table.csv",
        "jepa_evaluation.json",
        "readout_selection.json",
        "manifest.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(
            "cannot refresh an incomplete locked JEPA run: " + ", ".join(missing)
        )
    existing = pd.read_csv(output / "comparison_predictions.csv")
    panel = pd.read_csv(args.comparison_panel)
    comparison = _household_overlay_table(existing, panel)
    comparison.to_csv(output / "comparison_predictions.csv", index=False)
    sources = tuple(
        source
        for source in (
            "llm_household_economy",
            "origin_visible_baseline",
            "event_jepa_residual",
            "macro_state_ridge",
            "historical_mean",
            "historical_median",
        )
        if source in set(comparison["source"])
    )
    scores = _score_predictions(comparison, sources=sources)
    scores.to_csv(output / "score_table.csv", index=False)
    backtest_scores = pd.read_csv(output / "blocked_backtest_score_table.csv")
    evaluation = json.loads((output / "jepa_evaluation.json").read_text(encoding="utf-8"))
    readouts = json.loads((output / "readout_selection.json").read_text(encoding="utf-8"))
    report = _report_text(
        scores,
        backtest_scores,
        evaluation,
        jepa_readouts=readouts["jepa"],
    )
    (output / "macro_jepa_report.md").write_text(report, encoding="utf-8")
    write_chart(output / "comparison_predictions.csv", figure)
    args.report_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, args.report_figure)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["vei_runtime"] = _verify_vei_runtime(
        root=args.vei_root,
        reproduction_patch=args.vei_runtime_patch,
    )
    manifest["comparison_panel_sha256"] = _file_sha256(args.comparison_panel)
    manifest["household_overlay_refresh"] = {
        "comparison_panel": str(args.comparison_panel.resolve()),
        "policy": "replace_household_actual_and_visible_baseline_only_no_jepa_retraining",
    }
    manifest["primary_score"] = _primary_score_dict(scores)
    manifest["artifacts"] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][str(path.relative_to(output))] = _file_sha256(path)
    _write_json(output / "manifest.json", manifest)
    _write_tracked_evidence(
        output=output,
        destination=args.report_evidence_json,
        manifest=manifest,
    )
    return manifest


def _write_tracked_evidence(
    *,
    output: Path,
    destination: Path | None,
    manifest: dict[str, Any],
) -> None:
    if destination is None:
        return
    extraction = json.loads(
        (output / "extraction_manifest.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (output / "locked_training_spec.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (output / "jepa_evaluation.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (output / "readout_selection.json").read_text(encoding="utf-8")
    )
    portable_manifest = dict(manifest)
    checkpoint = Path(str(portable_manifest["model_checkpoint"]))
    try:
        portable_manifest["model_checkpoint"] = str(
            checkpoint.resolve().relative_to(PROJECT_ROOT)
        )
    except ValueError:
        portable_manifest["model_checkpoint"] = checkpoint.name
    portable_manifest["vei_root"] = "../vei"
    payload: dict[str, Any] = {
        "evidence_schema_version": "macro_jepa_tracked_evidence_v1",
        "run_manifest": portable_manifest,
        "extraction_manifest": extraction,
        "locked_training_spec": training,
        "jepa_evaluation": evaluation,
        "primary_metric_readout_selection": {
            "metric": "consumption_growth_pct",
            "jepa": selection["jepa"]["consumption_growth_pct"],
            "macro_state_ridge": selection["macro_state_ridge"][
                "consumption_growth_pct"
            ],
            "historical_constants": selection["historical_constants"][
                "consumption_growth_pct"
            ],
        },
        "generator_source_sha256": {
            name: _file_sha256(PROJECT_ROOT / "src/macro_llm_tournament" / name)
            for name in ("macro_jepa.py", "macro_jepa_data.py", "macro_jepa_events.py")
        },
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    _write_json(destination, payload)


def _origin_overrides(retrospective: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for month in pd.date_range("2026-01-01", "2026-04-01", freq="MS"):
        origin = month.date().isoformat()
        path = retrospective / "runs" / origin / "normalized_origin_information.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[origin] = path.resolve()
    return result


def _audit_origin_visible_baselines(
    baselines: pd.DataFrame,
    comparison_panel: Path,
) -> None:
    """Require current reconstructed anchors to match the banked chart inputs."""

    existing = pd.read_csv(comparison_panel)
    expected = existing.loc[
        existing["series_role"].eq("origin_visible_baseline")
        & existing["origin_month"].astype(str).isin(EVALUATION_ORIGINS),
        ["origin_month", "target_month", "metric", "value"],
    ].rename(columns={"value": "expected"})
    observed = baselines.loc[
        baselines["origin_month"].astype(str).isin(EVALUATION_ORIGINS),
        ["origin_month", "target_month", "metric", "baseline_prediction"],
    ]
    audit = expected.merge(
        observed,
        on=["origin_month", "target_month", "metric"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not audit["_merge"].eq("both").all():
        raise ValueError("origin-visible baseline audit has coverage mismatches")
    max_error = float(
        np.max(np.abs(audit["expected"] - audit["baseline_prediction"]))
    )
    if max_error > 1e-9:
        raise ValueError(
            f"origin-visible baseline audit mismatch: max error {max_error:.12f}"
        )


def _prediction_features(
    prepared: Any,
    model: Any,
    schema: Any,
    *,
    rows_to_step_batch: Any,
    collate_step_batches: Any,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    frame = prepared.frame.sort_values(["episode_id", "step_idx"]).reset_index(drop=True)
    state_columns = (
        schema.observation_numeric_columns + schema.observation_mask_columns
    )
    latent_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for end in range(CONTEXT_LENGTH - 1, len(frame)):
            origin_value = frame.iloc[end]["origin_month"]
            if pd.isna(origin_value) or not str(origin_value).strip():
                continue
            rows = frame.iloc[end - CONTEXT_LENGTH + 1 : end + 1]
            batch = collate_step_batches([rows_to_step_batch(rows, schema)]).to(device)
            z_ctx = model.encode_steps(batch)
            a_ctx = model.encode_actions(batch)
            time_signal = (
                None
                if model.config.time_feature_index is None
                else batch.observation_numeric[
                    :, :, model.config.time_feature_index
                ]
            )
            predicted = model.predict_next(
                z_ctx, a_ctx, time_signal=time_signal
            )[:, -1]
            origin_month = str(origin_value)
            common = {
                "origin_month": origin_month,
                "target_month": _month_offset(origin_month, 1),
                "split": str(frame.iloc[end]["split"]),
            }
            latent_rows.append(
                {
                    **common,
                    **{
                        f"latent__{index}": float(value)
                        for index, value in enumerate(
                            predicted.detach().cpu().numpy()[0]
                        )
                    },
                }
            )
            state_rows.append(
                {
                    **common,
                    **{
                        f"state__{column}": float(frame.iloc[end][column])
                        for column in state_columns
                    },
                }
            )
    return pd.DataFrame(latent_rows), pd.DataFrame(state_rows)


def _fit_readouts(
    feature_frame: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    baselines: pd.DataFrame,
    feature_prefix: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from joblib import dump
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = [
        column for column in feature_frame.columns if column.startswith(feature_prefix)
    ]
    prediction_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for metric in DIRECT_METRICS:
        target_rows = targets.loc[targets["metric"].eq(metric), [
            "target_month",
            "target_value",
            "first_release_date",
        ]]
        baseline_rows = baselines.loc[
            baselines["metric"].eq(metric),
            ["origin_month", "target_month", "baseline_prediction"],
        ]
        joined = feature_frame.merge(target_rows, on="target_month", how="left").merge(
            baseline_rows, on=["origin_month", "target_month"], how="left"
        )
        eligible = joined.loc[
            joined["target_value"].notna()
            & joined["baseline_prediction"].notna()
            & joined["first_release_date"].astype(str).le(READOUT_CUTOFF)
        ].copy()
        eligible["residual_target"] = (
            eligible["target_value"] - eligible["baseline_prediction"]
        )
        train = eligible.loc[eligible["split"].eq("train")]
        validation = eligible.loc[eligible["split"].eq("val")]
        if len(train) < 24 or len(validation) < 6:
            raise ValueError(
                f"insufficient readout rows for {metric}: train={len(train)} val={len(validation)}"
            )
        candidates: list[dict[str, float]] = []
        for alpha in ALPHA_GRID:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(train[feature_columns], train["residual_target"])
            residual_prediction = candidate.predict(validation[feature_columns])
            for residual_gain in RESIDUAL_GAIN_GRID:
                prediction = (
                    validation["baseline_prediction"].to_numpy(dtype=float)
                    + residual_gain * residual_prediction
                )
                candidates.append(
                    {
                        "alpha": float(alpha),
                        "residual_gain": float(residual_gain),
                        "validation_rmse": _rmse(
                            validation["target_value"].to_numpy(dtype=float),
                            prediction,
                        ),
                    }
                )
        chosen = min(
            candidates,
            key=lambda row: (
                row["validation_rmse"],
                row["residual_gain"],
                row["alpha"],
            ),
        )
        fit_rows = pd.concat([train, validation], ignore_index=True)
        model = make_pipeline(StandardScaler(), Ridge(alpha=chosen["alpha"]))
        model.fit(fit_rows[feature_columns], fit_rows["residual_target"])
        dump(model, output_dir / f"{metric}.joblib")
        prediction_frame = feature_frame.merge(
            baseline_rows, on=["origin_month", "target_month"], how="inner"
        )
        all_predictions = (
            prediction_frame["baseline_prediction"].to_numpy(dtype=float)
            + chosen["residual_gain"]
            * model.predict(prediction_frame[feature_columns])
        )
        for row, prediction in zip(
            prediction_frame[["origin_month", "target_month", "split"]].to_dict(
                "records"
            ),
            all_predictions,
            strict=True,
        ):
            prediction_rows.append(
                {
                    **row,
                    "metric": metric,
                    "prediction": float(prediction),
                }
            )
        metadata[metric] = {
            "chosen_alpha": chosen["alpha"],
            "chosen_residual_gain": chosen["residual_gain"],
            "validation_rmse": chosen["validation_rmse"],
            "baseline_validation_rmse": _rmse(
                validation["target_value"].to_numpy(dtype=float),
                validation["baseline_prediction"].to_numpy(dtype=float),
            ),
            "train_count": len(train),
            "validation_count": len(validation),
            "fit_count": len(fit_rows),
            "candidate_scores": candidates,
            "selection_boundary_hit": bool(
                chosen["alpha"] in {min(ALPHA_GRID), max(ALPHA_GRID)}
                or chosen["residual_gain"]
                in {min(RESIDUAL_GAIN_GRID), max(RESIDUAL_GAIN_GRID)}
            ),
        }
    result = pd.DataFrame(prediction_rows)
    result = _add_real_consumption_predictions(result)
    return result.sort_values(["target_month", "metric"]).reset_index(drop=True), metadata


def _add_real_consumption_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (origin, target, split), group in frame.groupby(
        ["origin_month", "target_month", "split"], sort=False
    ):
        values = group.set_index("metric")["prediction"]
        if (
            "real_consumption_growth_pct" not in values.index
            and {"consumption_growth_pct", "price_growth_pct"}.issubset(values.index)
        ):
            rows.append(
                {
                    "origin_month": origin,
                    "target_month": target,
                    "split": split,
                    "metric": "real_consumption_growth_pct",
                    "prediction": fisher_real_growth(
                        float(values["consumption_growth_pct"]),
                        float(values["price_growth_pct"]),
                    ),
                }
            )
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def _historical_constant_predictions(
    feature_frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit frozen mean/median comparators on labels available by the lock date."""

    origins = feature_frame[["origin_month", "target_month", "split"]].drop_duplicates()
    eligible_target_months = set(origins["target_month"].astype(str))
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for metric in DIRECT_METRICS:
        eligible = targets.loc[
            (
                targets["metric"].eq(metric)
                & targets["target_value"].notna()
                & targets["first_release_date"].astype(str).le(READOUT_CUTOFF)
                & targets["target_month"].astype(str).isin(eligible_target_months)
            ),
            "target_value",
        ].astype(float)
        if len(eligible) < 24:
            raise ValueError(f"insufficient historical constant rows for {metric}")
        values = {
            "historical_mean": float(eligible.mean()),
            "historical_median": float(eligible.median()),
        }
        metadata[metric] = {"fit_count": len(eligible), **values}
        for source, value in values.items():
            for origin in origins.to_dict("records"):
                rows.append(
                    {
                        **origin,
                        "metric": metric,
                        "prediction": value,
                        "source": source,
                    }
                )
    frame = pd.DataFrame(rows)
    real_rows: list[pd.DataFrame] = []
    for source, subset in frame.groupby("source", sort=False):
        enriched = _add_real_consumption_predictions(subset.drop(columns="source"))
        enriched["source"] = source
        real_rows.append(enriched)
    return (
        pd.concat(real_rows, ignore_index=True).sort_values(
            ["target_month", "metric", "source"]
        ).reset_index(drop=True),
        metadata,
    )


def _comparison_table(
    *,
    comparison_panel: Path,
    targets: pd.DataFrame,
    frozen_actuals: pd.DataFrame,
    jepa_predictions: pd.DataFrame,
    state_predictions: pd.DataFrame,
    constant_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    existing = pd.read_csv(comparison_panel)
    source_names = {
        "first_release_actual": "first_release_actual",
        "llm_household_economy": "llm_household_economy",
        "origin_visible_baseline": "origin_visible_baseline",
    }
    rows = existing[[
        "origin_month",
        "target_month",
        "metric",
        "evaluation_status",
        "series_role",
        "value",
    ]].copy()
    rows["source"] = rows["series_role"].map(source_names)
    rows = rows.drop(columns="series_role")

    actuals = targets[["target_month", "metric", "target_value"]].rename(
        columns={"target_value": "reconstructed_actual"}
    )
    audit = rows.loc[rows["source"].eq("first_release_actual")].merge(
        actuals, on=["target_month", "metric"], how="left"
    )
    frozen_audit = rows.loc[rows["source"].eq("first_release_actual")].merge(
        frozen_actuals[["target_month", "metric", "actual"]],
        on=["target_month", "metric"],
        how="left",
    )
    if frozen_audit["actual"].isna().any():
        raise ValueError("frozen target catalogue does not cover every scored actual")
    frozen_error = float(np.max(np.abs(frozen_audit["value"] - frozen_audit["actual"])))
    if frozen_error > 1e-9:
        raise ValueError(
            f"frozen target catalogue mismatch: max error {frozen_error:.12f}"
        )
    matched_audit = audit.loc[audit["reconstructed_actual"].notna()]
    if matched_audit.empty:
        raise ValueError("no comparison actuals can be audited against the event stream")
    max_error = float(
        np.max(
            np.abs(
                matched_audit["value"] - matched_audit["reconstructed_actual"]
            )
        )
    )
    if max_error > 1e-6:
        raise ValueError(
            f"first-release target reconstruction mismatch: max error {max_error:.9f}"
        )
    missing_audit = audit.loc[audit["reconstructed_actual"].isna()]
    latest_target_month = str(audit["target_month"].max())
    allowed_missing = missing_audit["metric"].eq(
        "real_consumption_growth_pct"
    ) | missing_audit["target_month"].astype(str).eq(latest_target_month)
    if not allowed_missing.all() or len(matched_audit) / len(audit) < 0.70:
        raise ValueError(
            "event-stream first-release audit has unexpected coverage gaps"
        )

    wanted_origins = set(
        rows.loc[
            rows["source"].eq("first_release_actual")
            & rows["evaluation_status"].eq(
                "retrospective_diagnostic_not_confirmatory"
            ),
            "origin_month",
        ].astype(str)
    )
    for source, predictions in (
        ("event_jepa_residual", jepa_predictions),
        ("macro_state_ridge", state_predictions),
    ):
        subset = predictions.loc[
            predictions["origin_month"].astype(str).isin(wanted_origins)
        ].copy()
        subset["source"] = source
        subset["value"] = subset.pop("prediction")
        subset["evaluation_status"] = "retrospective_diagnostic_not_confirmatory"
        rows = pd.concat(
            [
                rows,
                subset[
                    [
                        "origin_month",
                        "target_month",
                        "metric",
                        "evaluation_status",
                        "value",
                        "source",
                    ]
                ],
            ],
            ignore_index=True,
        )
    if constant_predictions is not None and not constant_predictions.empty:
        subset = constant_predictions.loc[
            constant_predictions["origin_month"].astype(str).isin(wanted_origins)
        ].copy()
        subset["value"] = subset.pop("prediction")
        subset["evaluation_status"] = "retrospective_diagnostic_not_confirmatory"
        rows = pd.concat(
            [
                rows,
                subset[
                    [
                        "origin_month",
                        "target_month",
                        "metric",
                        "evaluation_status",
                        "value",
                        "source",
                    ]
                ],
            ],
            ignore_index=True,
        )
    if rows.duplicated(["origin_month", "target_month", "metric", "source"]).any():
        raise ValueError("comparison table contains duplicate source predictions")
    return rows.sort_values(["metric", "target_month", "source"]).reset_index(drop=True)


def _backtest_table(
    *,
    targets: pd.DataFrame,
    baselines: pd.DataFrame,
    jepa_predictions: pd.DataFrame,
    state_predictions: pd.DataFrame,
    constant_predictions: pd.DataFrame,
) -> pd.DataFrame:
    wanted = set(BACKTEST_ORIGINS)
    actual = targets[["target_month", "metric", "target_value"]].copy()
    actual["origin_month"] = actual["target_month"].map(
        lambda value: _month_offset(str(value), -1)
    )
    actual = actual.loc[actual["origin_month"].isin(wanted)].rename(
        columns={"target_value": "value"}
    )
    actual["source"] = "first_release_actual"
    frames = [
        actual[["origin_month", "target_month", "metric", "value", "source"]]
    ]
    baseline = baselines.loc[baselines["origin_month"].isin(wanted)].rename(
        columns={"baseline_prediction": "value"}
    )
    baseline["source"] = "origin_visible_baseline"
    frames.append(
        baseline[["origin_month", "target_month", "metric", "value", "source"]]
    )
    for source, predictions in (
        ("event_jepa_residual", jepa_predictions),
        ("macro_state_ridge", state_predictions),
    ):
        frame = predictions.loc[predictions["origin_month"].isin(wanted)].rename(
            columns={"prediction": "value"}
        )
        frame["source"] = source
        frames.append(
            frame[["origin_month", "target_month", "metric", "value", "source"]]
        )
    constants = constant_predictions.loc[
        constant_predictions["origin_month"].isin(wanted)
    ].rename(columns={"prediction": "value"})
    frames.append(
        constants[["origin_month", "target_month", "metric", "value", "source"]]
    )
    result = pd.concat(frames, ignore_index=True)
    result["evaluation_status"] = "blocked_backtest_2024_2025"
    if result.duplicated(["origin_month", "target_month", "metric", "source"]).any():
        raise ValueError("blocked backtest table contains duplicate predictions")
    return result.sort_values(["metric", "target_month", "source"]).reset_index(
        drop=True
    )


def _score_predictions(
    comparison: pd.DataFrame,
    *,
    sources: Sequence[str],
    evaluation_status: str = "retrospective_diagnostic_not_confirmatory",
) -> pd.DataFrame:
    retrospective = comparison.loc[
        comparison["evaluation_status"].eq(evaluation_status)
    ]
    rows: list[dict[str, Any]] = []
    for metric, group in retrospective.groupby("metric"):
        actual = group.loc[group["source"].eq("first_release_actual"), [
            "target_month",
            "value",
        ]].rename(columns={"value": "actual"})
        visible_baseline = group.loc[
            group["source"].eq("origin_visible_baseline"),
            ["target_month", "value"],
        ].rename(columns={"value": "visible_baseline"})
        score_mode = MACRO_TARGET_CONTRACT[str(metric)]["score_mode"]
        for source in sources:
            predicted = group.loc[group["source"].eq(source), [
                "target_month",
                "value",
            ]].rename(columns={"value": "prediction"})
            if predicted.empty:
                continue
            joined = actual.merge(predicted, on="target_month", how="inner").merge(
                visible_baseline, on="target_month", how="left"
            )
            if len(joined) != len(actual):
                raise ValueError(f"incomplete score rows for {metric} {source}")
            observed = joined["actual"].to_numpy(dtype=float)
            forecast = joined["prediction"].to_numpy(dtype=float)
            scored_actual = np.sign(observed) if score_mode == "direction_only" else observed
            scored_forecast = np.sign(forecast) if score_mode == "direction_only" else forecast
            if (
                MACRO_TARGET_CONTRACT[str(metric)].get("direction_mode")
                == "change_from_visible_baseline"
            ):
                baseline_values = joined["visible_baseline"].to_numpy(dtype=float)
                direction_actual = np.sign(observed - baseline_values)
                direction_forecast = np.sign(forecast - baseline_values)
            else:
                direction_actual = np.sign(observed)
                direction_forecast = np.sign(forecast)
            rows.append(
                {
                    "metric": metric,
                    "source": source,
                    "score_mode": score_mode,
                    "n": len(joined),
                    "rmse": _rmse(scored_actual, scored_forecast),
                    "mae": float(np.mean(np.abs(scored_actual - scored_forecast))),
                    "direction_accuracy": float(
                        np.mean(direction_actual == direction_forecast)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "rmse", "source"]).reset_index(drop=True)


def write_chart(comparison_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    comparison = pd.read_csv(comparison_path)
    metrics = list(MACRO_TARGET_CONTRACT)
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.5), sharex=True)
    styles = {
        "llm_household_economy": {
            "label": "LLM household economy",
            "color": "#B23A2B",
            "marker": "s",
            "linestyle": "-",
            "linewidth": 1.8,
        },
        "event_jepa_residual": {
            "label": "Event-JEPA residual forecast",
            "color": "#3B7D5A",
            "marker": "D",
            "linestyle": "-",
            "linewidth": 1.8,
        },
        "first_release_actual": {
            "label": "First-release actual",
            "color": "#2463A6",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.1,
        },
        "origin_visible_baseline": {
            "label": "Origin-visible lag-1 baseline",
            "color": "#6B6B6B",
            "marker": "^",
            "linestyle": "--",
            "linewidth": 1.5,
        },
    }
    for axis, metric in zip(axes.flat, metrics, strict=True):
        mapping = MACRO_TARGET_CONTRACT[metric]
        subset = comparison.loc[comparison["metric"].eq(metric)].copy()
        for source in styles:
            retrospective = subset.loc[
                subset["source"].eq(source)
                & subset["evaluation_status"].eq(
                    "retrospective_diagnostic_not_confirmatory"
                )
            ].sort_values("target_month")
            if retrospective.empty:
                continue
            values = retrospective["value"].to_numpy(dtype=float)
            if mapping["score_mode"] == "direction_only":
                values = np.sign(values)
            axis.plot(
                pd.to_datetime(retrospective["target_month"]),
                values,
                **styles[source],
            )
            prospective = subset.loc[
                subset["source"].eq(source)
                & subset["evaluation_status"].eq("prospective_frozen_unscored")
            ]
            if not prospective.empty:
                prospective_values = prospective["value"].to_numpy(dtype=float)
                if mapping["score_mode"] == "direction_only":
                    prospective_values = np.sign(prospective_values)
                axis.scatter(
                    pd.to_datetime(prospective["target_month"]),
                    prospective_values,
                    marker=styles[source]["marker"],
                    s=54,
                    facecolors="white",
                    edgecolors=styles[source]["color"],
                    linewidths=1.8,
                    zorder=4,
                    label="_nolegend_",
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
        "Household Economy vs Event-JEPA: One-Month Forecasts",
        x=0.05,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.971,
        "Event-JEPA uses target months through 2023 and labels released by Jan 15, 2024; four Jan-Apr 2026 household comparisons shown (n=4). Aug household/baseline markers are prospective and unscored.",
        ha="left",
        fontsize=9.5,
        color="#4A4A4A",
    )
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels and not label.startswith("_"):
                handles.append(handle)
                labels.append(label)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#6B6B6B",
            markeredgewidth=1.5,
            markersize=6,
        )
    )
    labels.append("Prospective, unscored")
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.958),
        frameon=False,
        ncol=5,
        fontsize=8.7,
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.91, bottom=0.085, hspace=0.25, wspace=0.22
    )
    fig.text(
        0.05,
        0.025,
        "Sources: rolling gpt-5.5 household ecology; 4,000+ origin-safe ALFRED release days; frozen first-release FRED targets. Nominal PCE is primary; other panels retain declared proxy mappings.",
        ha="left",
        fontsize=8.1,
        color="#555555",
    )
    fig.text(
        0.05,
        0.010,
        "Event-JEPA = next-release latent plus a validation-locked ridge correction to the origin-visible baseline. Direct state residual and historical constants are scored but omitted here. No JEPA Aug point: full July 10 state unavailable.",
        ha="left",
        fontsize=8.1,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_with_system_python(output: Path, figure: Path) -> None:
    command = [
        "python3",
        "-m",
        "macro_llm_tournament.macro_jepa",
        "--plot-only",
        "--output-dir",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    if not figure.is_file():
        raise RuntimeError(f"plot command did not create {figure}")


def _report_text(
    scores: pd.DataFrame,
    backtest_scores: pd.DataFrame,
    evaluation: dict[str, Any],
    *,
    jepa_readouts: dict[str, Any],
) -> str:
    primary = scores.loc[scores["metric"].eq("consumption_growth_pct")].sort_values(
        "rmse"
    )
    backtest_primary = backtest_scores.loc[
        backtest_scores["metric"].eq("consumption_growth_pct")
    ].sort_values("rmse")
    lines = [
        "# Event-Time Residual StructuredStateJEPA Comparison",
        "",
        "This run replaces the monthly-snapshot shortcut with a release-time world model. Identical ALFRED initial/revision twins and later no-op revisions are canonicalized; each remaining release day is an atomic transition, with exact monthly forecast anchors inserted into the same stream.",
        "",
        f"The model trains only on pre-2022 release transitions and selects residual readouts using target months through 2023 whose first releases arrived by {READOUT_CUTOFF}. Each readout predicts only the correction to the origin-visible lag-1 baseline. The 2024-2025 block is therefore an untouched 24-origin diagnostic for this architecture; no 2024-2026 target month enters training or selection.",
        "",
        "## Blocked 2024-2025 Result",
        "",
        "| Source | PCE RMSE | Direction accuracy | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in backtest_primary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.rmse:.4f} | {row.direction_accuracy:.3f} | {row.n} |"
        )
    lines.extend(
        [
        "",
        "## Current 2026 Household Comparison",
        "",
        "Nominal PCE remains primary. Four January-April 2026 origins have both household-economy paths and realized targets. The household economy and lag-1 baseline retain their frozen August markers; Event-JEPA does not emit an August point because the dense archive lacks a complete July 10 state.",
        "",
        "| Source | PCE RMSE | Direction accuracy |",
        "| --- | ---: | ---: |",
        ]
    )
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.rmse:.4f} | {row.direction_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Model Audit",
            "",
            f"- Event-JEPA held-out latent MSE: `{float(evaluation['latent_mse']):.6f}`.",
            f"- Naive held-out observation MSE: `{float(evaluation['naive_observation_mse']):.6f}`.",
            "- REVOLSL and TOTALSL are canonicalized to billions of dollars before state construction; exact origin-card overrides pass through the same conversion and a fail-closed unit audit.",
            f"- Readout labels were unavailable to model training and locked to first releases dated no later than `{READOUT_CUTOFF}`.",
            "- Readout regularization was selected on 2022-2023 only, then locked before the 2024-2026 scores.",
            f"- The primary PCE readout selected alpha `{float(jepa_readouts['consumption_growth_pct']['chosen_alpha']):g}` and residual gain `{float(jepa_readouts['consumption_growth_pct']['chosen_residual_gain']):g}` on validation; boundary hit: `{bool(jepa_readouts['consumption_growth_pct']['selection_boundary_hit'])}`.",
            f"- On readout validation, the origin-visible PCE baseline RMSE was `{float(jepa_readouts['consumption_growth_pct']['baseline_validation_rmse']):.4f}` and the selected residual readout RMSE was `{float(jepa_readouts['consumption_growth_pct']['validation_rmse']):.4f}`.",
            "- Every graph actual exactly matches the frozen vintage target catalogue. The augmented dense event archive includes direct monthly PCEC96 vintages and independently reproduces every graph row it covers; later evaluation-only labels never enter JEPA or readout fitting.",
            "- The action channels describe observed policy, credit, sentiment, and release updates. They are not randomized interventions and do not support causal policy claims.",
            "- Frozen historical mean and median forecasts are scored alongside the direct macro-state ridge. All three are omitted from the figure to keep it readable.",
            "- The 24-origin block is the main architecture diagnostic. The four-origin household overlay remains too small for an independent household-versus-JEPA claim.",
            "",
            "## Next Honest Hybrid",
            "",
            "A true LLM-HANK-JEPA still needs historical household states and actions at many rolling origins. That backfill should add real SCE belief distributions, LLM belief updates, policy schedules, and resulting consumption/saving/credit intentions to these event-time states without changing this locked macro-only result.",
            "",
        ]
    )
    return "\n".join(lines)


def _primary_score_dict(scores: pd.DataFrame) -> dict[str, Any]:
    rows = scores.loc[scores["metric"].eq("consumption_growth_pct")]
    return {
        str(row.source): {
            "rmse": float(row.rmse),
            "direction_accuracy": float(row.direction_accuracy),
            "n": int(row.n),
        }
        for row in rows.itertuples(index=False)
    }


def _add_vei_to_path(root: Path) -> None:
    resolved = root.expanduser().resolve()
    if not (resolved / "pipeline" / "train" / "training.py").is_file():
        raise FileNotFoundError(f"VEI training package not found: {resolved}")
    sys.path.insert(0, str(resolved))


def _rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def _month_offset(value: str, months: int) -> str:
    return (
        pd.Timestamp(value) + pd.DateOffset(months=months)
    ).to_period("M").to_timestamp().date().isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _verify_vei_runtime(
    *, root: Path, reproduction_patch: Path
) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    commit = _git_commit(resolved)
    if commit != VEI_RUNTIME_BASE_COMMIT:
        raise ValueError(
            "VEI runtime base commit changed; lock a new runtime before retraining: "
            f"expected {VEI_RUNTIME_BASE_COMMIT}, got {commit}"
        )
    actual = {
        relative: _file_sha256(resolved / relative)
        for relative in VEI_RUNTIME_FILE_SHA256
    }
    mismatched = {
        relative: {"expected": VEI_RUNTIME_FILE_SHA256[relative], "actual": digest}
        for relative, digest in actual.items()
        if digest != VEI_RUNTIME_FILE_SHA256[relative]
    }
    if mismatched:
        raise ValueError(f"VEI runtime source hash mismatch: {mismatched}")
    patch = reproduction_patch.expanduser().resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"VEI runtime reproduction patch not found: {patch}")
    dirty_rows = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *VEI_RUNTIME_FILE_SHA256,
        ],
        cwd=resolved,
        text=True,
    ).splitlines()
    return {
        "base_commit": commit,
        "dependency_files_dirty": bool(dirty_rows),
        "dependency_file_sha256": actual,
        "reproduction_patch": str(
            patch.relative_to(PROJECT_ROOT)
            if patch.is_relative_to(PROJECT_ROOT)
            else patch.name
        ),
        "reproduction_patch_sha256": _file_sha256(patch),
    }


def main(argv: Sequence[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
