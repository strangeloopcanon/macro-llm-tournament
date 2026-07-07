from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, markdown_table, round_or_none
from .forecast_llm import ForecastLLMClient
from .llm_common import LLMUnavailable
from .persona_belief_panel import (
    DEMOGRAPHIC_DIMENSIONS,
    EVIDENCE_THRESHOLDS,
    TARGET_SPECS,
    _anonymize_csv_respondent_ids,
    _select_static_respondents,
    build_group_means,
    classify_persona_evidence,
    load_persona_respondents,
    score_common_core,
    score_distribution_distance,
    score_regression_gradient_match,
    score_variance_flattening,
)
from .prepare_persona_holdouts import (
    DEFAULT_FORECAST_ORIGINS,
    DEFAULT_FRED_VINTAGE_CONTEXT,
    DEFAULT_SURVEY_BELIEFS,
    prepare_real_sce_holdouts,
)


CAMPAIGN_VERSION = "persona_elicitation_campaign_v1"
DEFAULT_TARGET_FIELDS = (
    "expected_inflation_1y",
    "expected_unemployment_higher_prob",
    "expected_real_income_growth",
)
DEFAULT_MODELS = ("gpt-5.5", "gpt-5.4")
DEFAULT_SAMPLE_STRATA = ("income_group", "age_group", "education_group")
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "persona_elicitation_campaign"
DEFAULT_WORK_DIR = Path("work/persona_beliefs/persona_elicitation_campaign")
DEFAULT_DECEMBER_RUN = OUTPUT_ROOT / "persona_belief_panel_sce_real_design_gpt55_gpt54_500"
DEFAULT_SCE_MICRODATA = Path("work/persona_beliefs/sce_real_microdata.csv")
ARM1_THRESHOLDS = {
    "median_within_variance_ratio_improvement_min": 3.0,
    "max_weighted_ks_drop_min": 0.05,
    "regression_sign_rate_degrade_max": 0.05,
    "group_mean_mae_growth_max": 0.20,
}
ARM3_THRESHOLDS = EVIDENCE_THRESHOLDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SCE persona elicitation campaign.")
    parser.add_argument("--mode", choices=("prepare", "arm0", "arm1", "arm2", "arm3", "all"), default="prepare")
    parser.add_argument("--provider", choices=("codex_cli",), default="codex_cli")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--execute-live", action="store_true", help="Required for arms that spend codex_cli calls.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--sce-microdata", type=Path, default=DEFAULT_SCE_MICRODATA)
    parser.add_argument("--december-run", type=Path, default=DEFAULT_DECEMBER_RUN)
    parser.add_argument("--target-fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--arm1-seed", type=int, default=20260707)
    parser.add_argument("--arm3-seed", type=int, default=20260708)
    parser.add_argument("--draw-seed", type=int, default=20260707)
    parser.add_argument("--arm1-live-cap", type=int, default=440)
    parser.add_argument("--arm2-live-cap", type=int, default=6)
    parser.add_argument("--arm3-live-cap", type=int, default=440)
    parser.add_argument("--json-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    target_fields = tuple(part.strip() for part in args.target_fields.split(",") if part.strip())
    _validate_args(args, models=models, target_fields=target_fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = _read_optional_json(args.output_dir / "campaign_manifest.json") or {}
    manifest = _merge_campaign_manifest(
        existing_manifest,
        _campaign_manifest(args, models=models, target_fields=target_fields),
    )
    _write_json(args.output_dir / "campaign_manifest.json", manifest)

    if args.mode in {"prepare", "all", "arm1", "arm2", "arm3"}:
        prepared = prepare_campaign_inputs(
            args.work_dir,
            sce_microdata_path=args.sce_microdata,
            target_fields=target_fields,
            sample_size=args.sample_size,
            arm1_seed=args.arm1_seed,
            arm3_seed=args.arm3_seed,
        )
        manifest["prepared_inputs"] = prepared
        _write_json(args.output_dir / "campaign_manifest.json", manifest)

    if args.mode in {"arm0", "all"}:
        manifest["arm0"] = run_arm0_distributional_rescore(
            args.december_run,
            args.output_dir / "arm0_distributional_rescore",
            target_fields=target_fields,
            draw_seed=args.draw_seed,
        )
        _write_json(args.output_dir / "campaign_manifest.json", manifest)

    if args.mode in {"arm1", "all"}:
        _require_live(args, "Arm 1")
        validation_csv = Path(manifest["prepared_inputs"]["validation_static_csv"])
        manifest["arm1"] = run_arm1_point_vs_backstory(
            validation_csv,
            args.output_dir / "arm1_point_vs_backstory",
            provider=args.provider,
            models=models,
            target_fields=target_fields,
            sample_size=args.sample_size,
            sample_seed=args.arm1_seed,
            live_cap=args.arm1_live_cap,
            json_attempts=args.json_attempts,
        )
        _write_json(args.output_dir / "campaign_manifest.json", manifest)

    if args.mode in {"arm2", "all"}:
        _require_live(args, "Arm 2")
        validation_csv = Path(manifest["prepared_inputs"]["validation_static_csv"])
        manifest["arm2"] = run_arm2_no_hints_probe(
            validation_csv,
            args.output_dir / "arm2_no_hints_probe",
            provider=args.provider,
            models=models,
            target_fields=target_fields,
            live_cap=args.arm2_live_cap,
            json_attempts=args.json_attempts,
        )
        _write_json(args.output_dir / "campaign_manifest.json", manifest)

    if args.mode in {"arm3", "all"}:
        arm1 = manifest.get("arm1") or _read_optional_json(args.output_dir / "arm1_point_vs_backstory" / "arm1_manifest.json")
        passing_models = _arm1_passing_models(arm1)
        if not passing_models:
            skipped = {
                "status": "skipped",
                "reason": "Arm 1 did not return backstory_recovers_spread for any model.",
                "conditional_rule": "run only if Arm 1 passes for at least one model",
            }
            manifest["arm3"] = skipped
            _write_json(args.output_dir / "arm3_priors_backstory_confirmatory" / "arm3_manifest.json", skipped)
            _write_json(args.output_dir / "campaign_manifest.json", manifest)
        else:
            _require_live(args, "Arm 3")
            sealed_panel_csv = Path(manifest["prepared_inputs"]["sealed_panel_csv"])
            manifest["arm3"] = run_arm3_priors_backstory_confirmatory(
                sealed_panel_csv,
                args.output_dir / "arm3_priors_backstory_confirmatory",
                provider=args.provider,
                models=tuple(passing_models),
                target_fields=target_fields,
                sample_size=args.sample_size,
                sample_seed=args.arm3_seed,
                live_cap=args.arm3_live_cap,
                json_attempts=args.json_attempts,
            )
            _write_json(args.output_dir / "campaign_manifest.json", manifest)

    build_campaign_report(args.output_dir, manifest)
    print(args.output_dir)
    return 0


def _validate_args(args: argparse.Namespace, *, models: tuple[str, ...], target_fields: tuple[str, ...]) -> None:
    if args.provider != "codex_cli":
        raise SystemExit("persona elicitation campaign only supports --provider codex_cli")
    if not models:
        raise SystemExit("--models must contain at least one model")
    unknown_targets = sorted(set(target_fields) - set(TARGET_SPECS))
    if unknown_targets:
        raise SystemExit(f"Unknown target fields: {', '.join(unknown_targets)}")
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if args.mode in {"arm1", "arm2", "arm3", "all"} and not args.execute_live:
        raise SystemExit("--execute-live is required for live campaign arms")


def prepare_campaign_inputs(
    work_dir: Path,
    *,
    sce_microdata_path: Path,
    target_fields: tuple[str, ...] = DEFAULT_TARGET_FIELDS,
    sample_size: int = 100,
    arm1_seed: int = 20260707,
    arm3_seed: int = 20260708,
) -> dict[str, Any]:
    if not sce_microdata_path.exists():
        raise FileNotFoundError(f"SCE microdata file not found: {sce_microdata_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(sce_microdata_path)
    raw["survey_date"] = pd.to_datetime(raw["survey_date"], errors="coerce").dt.date.astype(str)

    validation = raw[raw["survey_date"].eq("2024-11-01")].copy()
    if validation.empty:
        raise ValueError("No November 2024 validation rows found in SCE microdata")
    validation_path = work_dir / "sce_validation_2024_11_static.csv"
    validation.to_csv(validation_path, index=False)

    validation_norm = load_persona_respondents(
        source="csv",
        respondent_csv=validation_path,
        respondent_count=0,
        survey_date="2024-11-01",
        target_fields=target_fields,
        survey_schema="normalized",
    )
    validation_sample, validation_selection = _select_static_respondents(
        _anonymize_csv_respondent_ids(validation_norm),
        respondent_limit=0,
        sample_size=sample_size,
        sample_seed=arm1_seed,
        sample_strata=DEFAULT_SAMPLE_STRATA,
    )
    validation_sample_path = work_dir / "sce_validation_2024_11_static_sample_manifest_only.csv"
    validation_sample.to_csv(validation_sample_path, index=False)

    sealed_dir = work_dir / "sealed_2025_06"
    sealed = prepare_real_sce_holdouts(
        sce_microdata_path=sce_microdata_path,
        survey_beliefs_path=DEFAULT_SURVEY_BELIEFS,
        forecast_origins_path=DEFAULT_FORECAST_ORIGINS,
        fred_vintage_context_path=DEFAULT_FRED_VINTAGE_CONTEXT,
        output_dir=sealed_dir,
        static_output="sce_sealed_2025_06_static.csv",
        panel_output="sce_sealed_2025_05_2025_06_panel.csv",
        period_count=2,
        start_as_of="2025-05-01",
    )
    sealed_panel = pd.read_csv(sealed["panel_csv"])
    complete = _complete_panel_respondents(sealed_panel, ("sce_2025_05", "sce_2025_06"))
    sealed_complete_path = sealed_dir / "sce_sealed_2025_05_2025_06_complete_panel.csv"
    complete.to_csv(sealed_complete_path, index=False)

    return {
        "schema_version": "persona_elicitation_prepared_inputs_v1",
        "validation_static_csv": str(validation_path),
        "validation_static_rows": int(validation.shape[0]),
        "validation_sample_manifest_csv": str(validation_sample_path),
        "validation_sample_size": int(validation_sample.shape[0]),
        "validation_selection": validation_selection,
        "sealed_panel_csv": str(sealed_complete_path),
        "sealed_panel_rows": int(complete.shape[0]),
        "sealed_panel_respondents": int(complete["respondent_id"].nunique()) if not complete.empty else 0,
        "sealed_period_ids": sorted(complete["period_id"].astype(str).unique()) if not complete.empty else [],
        "sealed_build_rule": "converter_only_no_distribution_summary_before_arm3",
        "sealed_manifest": sealed["manifest"],
        "arm3_seed": int(arm3_seed),
    }


def run_arm0_distributional_rescore(
    december_run: Path,
    output_dir: Path,
    *,
    target_fields: tuple[str, ...] = DEFAULT_TARGET_FIELDS,
    draw_seed: int = 20260707,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    respondents = pd.read_csv(december_run / "persona_respondents.csv")
    predictions = pd.read_csv(december_run / "persona_belief_predictions.csv")
    drawn = draw_distributional_predictions(predictions, target_fields=target_fields, seed=draw_seed)
    variance = score_variance_flattening(respondents, drawn, target_fields=target_fields)
    distribution = score_distribution_distance(respondents, drawn, target_fields=target_fields)
    interval = score_interval_calibration(respondents, predictions, target_fields=target_fields)
    variance.to_csv(output_dir / "arm0_distributional_variance_scores.csv", index=False)
    distribution.to_csv(output_dir / "arm0_distributional_distribution_scores.csv", index=False)
    interval.to_csv(output_dir / "arm0_interval_calibration.csv", index=False)
    drawn.to_csv(output_dir / "arm0_drawn_predictions.csv", index=False)
    manifest = {
        "schema_version": CAMPAIGN_VERSION,
        "arm": "arm0_distributional_rescore",
        "status": "ok",
        "label": "exploratory_spent_wave_reanalysis",
        "source_run": str(december_run),
        "draw_seed": int(draw_seed),
        "target_fields": list(target_fields),
        "median_within_variance_ratio": _finite_median(variance, "within_variance_ratio"),
        "max_weighted_ks_stat": _finite_max(distribution, "ks_stat"),
        "median_distribution_std_ratio": _finite_median(distribution, "std_ratio"),
        "mean_interval_coverage": _finite_mean(interval, "coverage_p10_p90"),
        "interval_target": 0.80,
        "outputs": [
            "arm0_distributional_variance_scores.csv",
            "arm0_distributional_distribution_scores.csv",
            "arm0_interval_calibration.csv",
            "arm0_drawn_predictions.csv",
        ],
    }
    _write_json(output_dir / "arm0_manifest.json", manifest)
    (output_dir / "arm0_report.md").write_text(_arm0_report(manifest, interval), encoding="utf-8")
    return manifest


def run_arm1_point_vs_backstory(
    validation_csv: Path,
    output_dir: Path,
    *,
    provider: str,
    models: tuple[str, ...],
    target_fields: tuple[str, ...],
    sample_size: int,
    sample_seed: int,
    live_cap: int,
    json_attempts: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_mode_cap = int(np.ceil(live_cap / 2.0))
    mode_dirs: dict[str, Path] = {}
    for elicitation in ("point", "backstory"):
        mode_dir = output_dir / elicitation
        mode_dirs[elicitation] = mode_dir
        _run_persona_panel_subprocess(
            mode_dir,
            provider=provider,
            models=models,
            respondent_csv=validation_csv,
            target_fields=target_fields,
            sample_size=sample_size,
            sample_seed=sample_seed,
            live_cap=per_mode_cap,
            elicitation=elicitation,
            json_attempts=json_attempts,
        )

    point = _load_panel_outputs(mode_dirs["point"])
    backstory = _load_panel_outputs(mode_dirs["backstory"])
    respondents = point["respondents"]
    backstory_drawn = draw_distributional_predictions(backstory["predictions"], target_fields=target_fields, seed=sample_seed)
    draw_variance = score_variance_flattening(respondents, backstory_drawn, target_fields=target_fields)
    draw_distribution = score_distribution_distance(respondents, backstory_drawn, target_fields=target_fields)
    draw_variance.to_csv(output_dir / "arm1_backstory_draw_variance_scores.csv", index=False)
    draw_distribution.to_csv(output_dir / "arm1_backstory_draw_distribution_scores.csv", index=False)

    scoreboard = build_arm1_scoreboard(point, backstory)
    verdicts = classify_arm1_backstory(scoreboard)
    scoreboard.to_csv(output_dir / "arm1_scoreboard.csv", index=False)
    verdicts.to_csv(output_dir / "arm1_verdicts.csv", index=False)
    manifest = {
        "schema_version": CAMPAIGN_VERSION,
        "arm": "arm1_point_vs_backstory",
        "status": "ok",
        "provider": provider,
        "models": list(models),
        "sample_size": int(sample_size),
        "sample_seed": int(sample_seed),
        "live_cap": int(live_cap),
        "per_mode_live_cap": per_mode_cap,
        "thresholds": ARM1_THRESHOLDS,
        "verdicts": verdicts.to_dict(orient="records"),
        "passing_models": _arm1_passing_models({"verdicts": verdicts.to_dict(orient="records")}),
        "outputs": ["point", "backstory", "arm1_scoreboard.csv", "arm1_verdicts.csv"],
    }
    _write_json(output_dir / "arm1_manifest.json", manifest)
    (output_dir / "arm1_report.md").write_text(_arm1_report(manifest, scoreboard), encoding="utf-8")
    return manifest


def run_arm2_no_hints_probe(
    validation_csv: Path,
    output_dir: Path,
    *,
    provider: str,
    models: tuple[str, ...],
    target_fields: tuple[str, ...],
    live_cap: int,
    json_attempts: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    respondents = load_persona_respondents(
        source="csv",
        respondent_csv=validation_csv,
        respondent_count=0,
        survey_date="2024-11-01",
        target_fields=target_fields,
        survey_schema="normalized",
    )
    required_calls = len(models) * len(target_fields)
    if live_cap < required_calls:
        raise SystemExit(f"Arm 2 live cap {live_cap} is below required call count {required_calls}")
    cache_dir = output_dir / "fresh_no_hints_cache"
    execution_cwd = output_dir / "provider_cwd"
    execution_cwd.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    live_used = 0
    old_attempts = os.environ.get("LLM_JSON_ATTEMPTS")
    os.environ["LLM_JSON_ATTEMPTS"] = str(json_attempts)
    try:
        for model in models:
            client = ForecastLLMClient(
                provider,
                model,
                cache_dir,
                mode="live",
                max_live_calls=live_cap - live_used,
                execution_cwd=execution_cwd,
            )
            for target in target_fields:
                prompt_payload = no_hints_prompt_payload(target)
                prompt = json.dumps(prompt_payload, indent=2, sort_keys=True)
                cache_name = f"persona_no_hints_{model}_{target}"
                data = client.json_call(prompt, cache_name, instructions=_no_hints_instructions())
                payload = normalize_no_hints_payload(data.get("payload", data), target)
                score = score_no_hints_distribution(respondents, payload, target)
                rows.append({"model": model, "source": f"llm_{provider}_{model}", "target_name": target, **payload, **score})
                raw_records.append({"model": model, "target_name": target, "payload": data.get("payload", data), "cache_hit": data.get("cache_hit", False)})
            live_used += client.live_call_count
    finally:
        if old_attempts is None:
            os.environ.pop("LLM_JSON_ATTEMPTS", None)
        else:
            os.environ["LLM_JSON_ATTEMPTS"] = old_attempts
    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "arm2_no_hints_scores.csv", index=False)
    _write_json(output_dir / "arm2_no_hints_raw_records.json", raw_records)
    manifest = {
        "schema_version": CAMPAIGN_VERSION,
        "arm": "arm2_no_hints_probe",
        "status": "ok",
        "provider": provider,
        "models": list(models),
        "required_call_count": required_calls,
        "live_call_count": int(live_used),
        "live_cap": int(live_cap),
        "json_attempts": int(json_attempts),
        "outputs": ["arm2_no_hints_scores.csv", "arm2_no_hints_raw_records.json"],
    }
    _write_json(output_dir / "arm2_manifest.json", manifest)
    (output_dir / "arm2_report.md").write_text(_arm2_report(scores), encoding="utf-8")
    return manifest


def run_arm3_priors_backstory_confirmatory(
    sealed_panel_csv: Path,
    output_dir: Path,
    *,
    provider: str,
    models: tuple[str, ...],
    target_fields: tuple[str, ...],
    sample_size: int,
    sample_seed: int,
    live_cap: int,
    json_attempts: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_persona_ecology_subprocess(
        output_dir / "ecology",
        provider=provider,
        models=models,
        respondent_csv=sealed_panel_csv,
        target_fields=target_fields,
        sample_size=sample_size,
        sample_seed=sample_seed,
        live_cap=live_cap,
        json_attempts=json_attempts,
    )
    eco = output_dir / "ecology"
    panel = pd.read_csv(eco / "persona_ecology_panel.csv")
    predictions = pd.read_csv(eco / "persona_ecology_predictions.csv")
    scoring_panel = panel[panel["period_id"].astype(str).eq("sce_2025_06")].copy()
    scoring_predictions = predictions[predictions["period_id"].astype(str).eq("sce_2025_06")].copy()
    drawn = draw_distributional_predictions(scoring_predictions, target_fields=target_fields, seed=sample_seed)
    regression = score_regression_gradient_match(scoring_panel, scoring_predictions, target_fields=target_fields)
    variance = score_variance_flattening(scoring_panel, drawn, target_fields=target_fields)
    distribution = score_distribution_distance(scoring_panel, drawn, target_fields=target_fields)
    common_core = score_common_core(scoring_predictions, target_fields=target_fields)
    evidence = classify_persona_evidence(regression, variance, common_core, distribution)
    regression.to_csv(output_dir / "arm3_regression_scores.csv", index=False)
    variance.to_csv(output_dir / "arm3_distributional_variance_scores.csv", index=False)
    distribution.to_csv(output_dir / "arm3_distributional_distribution_scores.csv", index=False)
    common_core.to_csv(output_dir / "arm3_common_core.csv", index=False)
    drawn.to_csv(output_dir / "arm3_drawn_predictions.csv", index=False)
    manifest = {
        "schema_version": CAMPAIGN_VERSION,
        "arm": "arm3_priors_backstory_confirmatory",
        "status": "ok",
        "provider": provider,
        "models": list(models),
        "sample_size": int(sample_size),
        "sample_seed": int(sample_seed),
        "live_cap": int(live_cap),
        "json_attempts": int(json_attempts),
        "thresholds": ARM3_THRESHOLDS,
        "spread_metrics_scored_on": "distributional_draws_from_p10_p50_p90",
        "level_metrics_scored_on": "point_predictions",
        "evidence": evidence,
        "common_core_note": "Original gate requires at least two model sources; one passing Arm 1 model leaves common-core unmeasured.",
        "outputs": [
            "ecology",
            "arm3_regression_scores.csv",
            "arm3_distributional_variance_scores.csv",
            "arm3_distributional_distribution_scores.csv",
            "arm3_common_core.csv",
            "arm3_drawn_predictions.csv",
        ],
    }
    _write_json(output_dir / "arm3_manifest.json", manifest)
    (output_dir / "arm3_report.md").write_text(_arm3_report(manifest), encoding="utf-8")
    return manifest


def draw_distributional_predictions(
    predictions: pd.DataFrame,
    *,
    target_fields: tuple[str, ...] = DEFAULT_TARGET_FIELDS,
    seed: int = 20260707,
) -> pd.DataFrame:
    out = predictions.copy()
    rngs = {source: np.random.default_rng(seed + idx * 1009) for idx, source in enumerate(sorted(out["source"].astype(str).unique()))}
    draws: list[float] = []
    for _, row in out.iterrows():
        target = str(row["target_name"])
        spec = TARGET_SPECS[target]
        rng = rngs[str(row["source"])]
        u = float(rng.random())
        quantiles = np.array([0.0, 0.10, 0.50, 0.90, 1.0], dtype=float)
        values = np.clip(
            np.array(
            [
                float(spec["lower"]),
                float(row["p10"]),
                float(row.get("p50", row["prediction"])),
                float(row["p90"]),
                float(spec["upper"]),
            ],
            dtype=float,
            ),
            float(spec["lower"]),
            float(spec["upper"]),
        )
        values = np.maximum.accumulate(values)
        draws.append(float(np.interp(u, quantiles, values)))
    out["point_prediction"] = out["prediction"]
    out["prediction"] = draws
    out["distributional_draw_seed"] = int(seed)
    return out[out["target_name"].isin(target_fields)].copy()


def score_interval_calibration(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: tuple[str, ...] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = predictions.merge(
        respondents[["respondent_id", "weight", *[f"actual_{target}" for target in target_fields]]],
        on="respondent_id",
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (source, target), group in joined.groupby(["source", "target_name"], dropna=False):
        actual = pd.to_numeric(group[f"actual_{target}"], errors="coerce")
        inside = actual.ge(pd.to_numeric(group["p10"], errors="coerce")) & actual.le(pd.to_numeric(group["p90"], errors="coerce"))
        rows.append(
            {
                "source": source,
                "target_name": target,
                "n": int(group.shape[0]),
                "coverage_p10_p90": _weighted_mean(inside.astype(float), group["weight"]),
                "mean_interval_width": _weighted_mean(pd.to_numeric(group["p90"], errors="coerce") - pd.to_numeric(group["p10"], errors="coerce"), group["weight"]),
                "target_coverage": 0.80,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_name", "source"]).reset_index(drop=True)


def build_arm1_scoreboard(point: dict[str, pd.DataFrame], backstory: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = sorted(set(point["predictions"]["source"].astype(str)).intersection(backstory["predictions"]["source"].astype(str)))
    for source in sources:
        point_metrics = _panel_metric_summary(point, source)
        backstory_metrics = _panel_metric_summary(backstory, source)
        rows.append(
            {
                "source": source,
                "model": str(backstory["predictions"].loc[backstory["predictions"]["source"].astype(str).eq(source), "model"].iloc[0]),
                **{f"point_{key}": value for key, value in point_metrics.items()},
                **{f"backstory_{key}": value for key, value in backstory_metrics.items()},
                "within_variance_ratio_improvement": _safe_ratio(
                    backstory_metrics["median_within_variance_ratio"],
                    point_metrics["median_within_variance_ratio"],
                ),
                "ks_drop": point_metrics["max_weighted_ks_stat"] - backstory_metrics["max_weighted_ks_stat"],
                "sign_rate_degrade": point_metrics["regression_sign_rate"] - backstory_metrics["regression_sign_rate"],
                "group_mean_mae_growth": _safe_ratio(
                    backstory_metrics["group_mean_mae"],
                    point_metrics["group_mean_mae"],
                )
                - 1.0,
            }
        )
    return pd.DataFrame(rows)


def classify_arm1_backstory(scoreboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in scoreboard.iterrows():
        spread = bool(row["within_variance_ratio_improvement"] >= ARM1_THRESHOLDS["median_within_variance_ratio_improvement_min"])
        ks = bool(row["ks_drop"] >= ARM1_THRESHOLDS["max_weighted_ks_drop_min"])
        signs = bool(row["sign_rate_degrade"] <= ARM1_THRESHOLDS["regression_sign_rate_degrade_max"])
        levels = bool(row["group_mean_mae_growth"] < ARM1_THRESHOLDS["group_mean_mae_growth_max"])
        if spread and ks and signs and levels:
            verdict = "backstory_recovers_spread"
        elif not signs or not levels:
            verdict = "backstory_caricature"
        else:
            verdict = "backstory_fails"
        rows.append(
            {
                "source": row["source"],
                "model": row["model"],
                "verdict": verdict,
                "spread_clear": spread,
                "ks_clear": ks,
                "caricature_guard_clear": signs,
                "levels_guard_clear": levels,
            }
        )
    return pd.DataFrame(rows)


def no_hints_prompt_payload(target: str) -> dict[str, Any]:
    spec = TARGET_SPECS[target]
    return {
        "task": (
            "As of November 2024, predict the distribution of answers that U.S. household survey respondents "
            f"gave to this question: {spec['label']}."
        ),
        "target_name": target,
        "units": spec["units"],
        "bounds": {"lower": spec["lower"], "upper": spec["upper"]},
        "no_hints_rule": "No respondent profile, no true distribution, and no answer-shape hints are supplied.",
        "required_response": {
            "target_name": target,
            "d10": "10th percentile",
            "d20": "20th percentile",
            "d30": "30th percentile",
            "d40": "40th percentile",
            "d50": "50th percentile",
            "d60": "60th percentile",
            "d70": "70th percentile",
            "d80": "80th percentile",
            "d90": "90th percentile",
            "multiple_of_5_share": "share of answers that are exact multiples of 5, 0 to 1",
            "reason": "short explanation",
        },
    }


def normalize_no_hints_payload(payload: dict[str, Any], target: str) -> dict[str, Any]:
    spec = TARGET_SPECS[target]
    deciles = {}
    last = float(spec["lower"])
    for idx in range(10, 100, 10):
        key = f"d{idx}"
        value = float(np.clip(float(payload.get(key, last)), spec["lower"], spec["upper"]))
        last = max(last, value)
        deciles[key] = last
    return {
        "target_name": target,
        **deciles,
        "multiple_of_5_share": float(np.clip(float(payload.get("multiple_of_5_share", np.nan)), 0.0, 1.0)),
        "reason": str(payload.get("reason", ""))[:500],
    }


def score_no_hints_distribution(respondents: pd.DataFrame, payload: dict[str, Any], target: str) -> dict[str, float]:
    actual = pd.to_numeric(respondents[f"actual_{target}"], errors="coerce")
    weights = pd.to_numeric(respondents["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    probs = np.arange(0.1, 1.0, 0.1)
    predicted_deciles = np.array([payload[f"d{int(p * 100)}"] for p in probs], dtype=float)
    actual_deciles = _weighted_quantile(actual, weights, probs)
    actual_cdf_at_predicted = np.array([_weighted_cdf_scalar(actual, weights, value) for value in predicted_deciles], dtype=float)
    rounded = actual.dropna().astype(float).map(lambda value: abs(value / 5.0 - round(value / 5.0)) < 1e-9)
    return {
        "decile_mae": float(np.nanmean(np.abs(predicted_deciles - actual_deciles))),
        "decile_ks_proxy": float(np.nanmax(np.abs(actual_cdf_at_predicted - probs))),
        "actual_multiple_of_5_share": _weighted_mean(rounded.astype(float), weights.loc[rounded.index]),
        "rounding_share_error": abs(float(payload["multiple_of_5_share"]) - _weighted_mean(rounded.astype(float), weights.loc[rounded.index])),
    }


def _run_persona_panel_subprocess(
    output_dir: Path,
    *,
    provider: str,
    models: tuple[str, ...],
    respondent_csv: Path,
    target_fields: tuple[str, ...],
    sample_size: int,
    sample_seed: int,
    live_cap: int,
    elicitation: str,
    json_attempts: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "macro_llm_tournament.persona_belief_panel",
        "--provider",
        provider,
        "--models",
        ",".join(models),
        "--belief-mode",
        "live",
        "--fresh-cache",
        "--max-live-calls",
        str(live_cap),
        "--respondent-source",
        "csv",
        "--survey-schema",
        "normalized",
        "--respondent-csv",
        str(respondent_csv),
        "--respondent-sample-size",
        str(sample_size),
        "--respondent-sample-seed",
        str(sample_seed),
        "--respondent-sample-strata",
        ",".join(DEFAULT_SAMPLE_STRATA),
        "--elicitation-mode",
        elicitation,
        "--target-fields",
        ",".join(target_fields),
        "--output-dir",
        str(output_dir),
    ]
    _run_subprocess(cmd, output_dir=output_dir, json_attempts=json_attempts)


def _run_persona_ecology_subprocess(
    output_dir: Path,
    *,
    provider: str,
    models: tuple[str, ...],
    respondent_csv: Path,
    target_fields: tuple[str, ...],
    sample_size: int,
    sample_seed: int,
    live_cap: int,
    json_attempts: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "macro_llm_tournament.persona_ecology",
        "--provider",
        provider,
        "--models",
        ",".join(models),
        "--ecology-mode",
        "live",
        "--fresh-cache",
        "--max-live-calls",
        str(live_cap),
        "--respondent-source",
        "csv",
        "--survey-schema",
        "normalized",
        "--respondent-csv",
        str(respondent_csv),
        "--period-ids",
        "sce_2025_05,sce_2025_06",
        "--require-complete-periods",
        "--respondent-sample-size",
        str(sample_size),
        "--respondent-sample-seed",
        str(sample_seed),
        "--respondent-sample-strata",
        ",".join(DEFAULT_SAMPLE_STRATA),
        "--prior-mode",
        "empirical",
        "--feedback-mode",
        "none",
        "--date-mode",
        "relative",
        "--elicitation-mode",
        "backstory",
        "--target-fields",
        ",".join(target_fields),
        "--output-dir",
        str(output_dir),
    ]
    _run_subprocess(cmd, output_dir=output_dir, json_attempts=json_attempts)


def _run_subprocess(cmd: list[str], *, output_dir: Path, json_attempts: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["LLM_JSON_ATTEMPTS"] = str(json_attempts)
    result = subprocess.run(cmd, cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False)
    (output_dir / "subprocess_command.txt").write_text(shlex.join(cmd), encoding="utf-8")
    (output_dir / "subprocess_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "subprocess_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"Subprocess failed ({result.returncode}): {shlex.join(cmd)}\n{result.stderr[-2000:]}")


def _load_panel_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "respondents": pd.read_csv(output_dir / "persona_respondents.csv"),
        "predictions": pd.read_csv(output_dir / "persona_belief_predictions.csv"),
        "regression": pd.read_csv(output_dir / "persona_belief_regression_scores.csv"),
        "variance": pd.read_csv(output_dir / "persona_belief_variance_scores.csv"),
        "distribution": pd.read_csv(output_dir / "persona_belief_distribution_scores.csv"),
        "group_means": pd.read_csv(output_dir / "persona_belief_group_means.csv"),
    }


def _panel_metric_summary(outputs: dict[str, pd.DataFrame], source: str) -> dict[str, float]:
    regression = outputs["regression"]
    variance = outputs["variance"]
    distribution = outputs["distribution"]
    group_means = outputs["group_means"]
    source_reg = regression[regression["source"].astype(str).eq(source)].copy()
    if "scoreable" in source_reg:
        source_reg = source_reg[source_reg["scoreable"].astype(bool)].copy()
    source_var = variance[variance["source"].astype(str).eq(source)].copy()
    source_dist = distribution[distribution["source"].astype(str).eq(source)].copy()
    source_group = group_means[group_means["source"].astype(str).eq(source)].copy()
    source_group["abs_error"] = (source_group["simulated_mean"].astype(float) - source_group["survey_mean"].astype(float)).abs()
    return {
        "regression_sign_rate": float(source_reg["sign_match"].mean()) if not source_reg.empty else np.nan,
        "median_within_variance_ratio": _finite_median(source_var, "within_variance_ratio"),
        "max_weighted_ks_stat": _finite_max(source_dist, "ks_stat"),
        "median_distribution_std_ratio": _finite_median(source_dist, "std_ratio"),
        "group_mean_mae": float(source_group["abs_error"].mean()) if not source_group.empty else np.nan,
    }


def _complete_panel_respondents(panel: pd.DataFrame, period_ids: tuple[str, ...]) -> pd.DataFrame:
    subset = panel[panel["period_id"].astype(str).isin(period_ids)].copy()
    counts = subset.groupby("respondent_id")["period_id"].nunique()
    complete_ids = set(counts[counts == len(period_ids)].index.astype(str))
    return subset[subset["respondent_id"].astype(str).isin(complete_ids)].reset_index(drop=True)


def _arm1_passing_models(arm1: dict[str, Any] | None) -> list[str]:
    if not arm1:
        return []
    rows = arm1.get("verdicts", [])
    return [str(row["model"]) for row in rows if row.get("verdict") == "backstory_recovers_spread"]


def _campaign_manifest(args: argparse.Namespace, *, models: tuple[str, ...], target_fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "provider": args.provider,
        "models": list(models),
        "target_fields": list(target_fields),
        "execute_live": bool(args.execute_live),
        "call_caps": {"arm1": int(args.arm1_live_cap), "arm2": int(args.arm2_live_cap), "arm3": int(args.arm3_live_cap)},
        "seeds": {"arm1": int(args.arm1_seed), "arm3": int(args.arm3_seed), "draw": int(args.draw_seed)},
        "arm1_thresholds": ARM1_THRESHOLDS,
        "arm3_thresholds": ARM3_THRESHOLDS,
        "wave_roles": {
            "2024-12": "spent_reanalysis_only",
            "2024-10_2024-11": "validation",
            "2025-06": "sealed_confirmatory",
        },
    }


def _merge_campaign_manifest(existing: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key in ("prepared_inputs", "arm0", "arm1", "arm2", "arm3"):
        if key in existing:
            merged[key] = existing[key]
    return merged


def build_campaign_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Persona Elicitation Campaign",
        "",
        "## Bottom Line",
        _campaign_bottom_line(manifest),
        "",
        "## Arms",
    ]
    for arm in ("arm0", "arm1", "arm2", "arm3"):
        if arm in manifest:
            lines.extend(["", f"### {arm.upper()}", "```json", json.dumps(manifest[arm], indent=2, sort_keys=True), "```"])
    (output_dir / "persona_elicitation_campaign_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _campaign_bottom_line(manifest: dict[str, Any]) -> str:
    arm3 = manifest.get("arm3")
    if isinstance(arm3, dict) and arm3.get("status") == "ok":
        verdict = arm3.get("evidence", {}).get("evidence_verdict", "unknown")
        return f"Arm 3 completed with verdict `{verdict}`. Read this as the one-shot sealed-wave persona result."
    if isinstance(arm3, dict) and arm3.get("status") == "skipped":
        return "Arm 3 was skipped because Arm 1 did not pass; the persona layer remains unvalidated beyond prior-update signal."
    if "arm1" in manifest:
        passing = manifest["arm1"].get("passing_models", [])
        return f"Arm 1 completed. Passing backstory models: `{', '.join(passing) if passing else 'none'}`."
    if "arm0" in manifest:
        return "Arm 0 completed as a zero-call diagnostic rescore of the spent December wave."
    return "Inputs are prepared; no live arm has run yet."


def _arm0_report(manifest: dict[str, Any], interval: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Arm 0 Distributional Rescore",
            "",
            "Exploratory reanalysis of the spent December 2024 run. No new model calls.",
            "",
            f"- Median within-variance ratio on draws: `{round_or_none(manifest['median_within_variance_ratio'])}`",
            f"- Max weighted KS on draws: `{round_or_none(manifest['max_weighted_ks_stat'])}`",
            f"- Mean p10-p90 interval coverage: `{round_or_none(manifest['mean_interval_coverage'])}`",
            "",
            markdown_table(interval) if not interval.empty else "",
            "",
        ]
    )


def _arm1_report(manifest: dict[str, Any], scoreboard: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Arm 1 Point vs Backstory",
            "",
            "Backstory passes only if it recovers spread without worsening signs or group levels beyond the pre-registered guards.",
            "",
            markdown_table(scoreboard) if not scoreboard.empty else "",
            "",
            "```json",
            json.dumps(manifest.get("verdicts", []), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _arm2_report(scores: pd.DataFrame) -> str:
    return "\n".join(["# Arm 2 No-Hints Probe", "", markdown_table(scores) if not scores.empty else "", ""])


def _arm3_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Arm 3 Priors + Backstory Confirmatory",
            "",
            f"Evidence verdict: `{manifest.get('evidence', {}).get('evidence_verdict')}`",
            "",
            "```json",
            json.dumps(manifest.get("evidence", {}), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _no_hints_instructions() -> str:
    return "Return only valid JSON. Do not browse, inspect files, run commands, or cite hidden survey data."


def _require_live(args: argparse.Namespace, arm_label: str) -> None:
    if not args.execute_live:
        raise SystemExit(f"{arm_label} requires --execute-live")


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _weighted_mean(values: Iterable[Any], weights: Iterable[Any]) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    values = values[mask].astype(float)
    weights = weights[mask].astype(float)
    total = float(weights.sum())
    if total <= 0.0:
        return float(values.mean())
    return float((values * weights).sum() / total)


def _weighted_quantile(values: Iterable[Any], weights: Iterable[Any], quantiles: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.full_like(quantiles, np.nan, dtype=float)
    clean = pd.DataFrame({"value": values[mask].astype(float), "weight": weights[mask].astype(float)}).sort_values("value")
    cumulative = clean["weight"].cumsum() / float(clean["weight"].sum())
    return np.interp(quantiles, cumulative.to_numpy(), clean["value"].to_numpy())


def _weighted_cdf_scalar(values: Iterable[Any], weights: Iterable[Any], point: float) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    clean_values = values[mask].astype(float)
    clean_weights = weights[mask].astype(float)
    return float(clean_weights[clean_values <= point].sum() / clean_weights.sum())


def _finite_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _finite_max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


if __name__ == "__main__":
    raise SystemExit(main())
