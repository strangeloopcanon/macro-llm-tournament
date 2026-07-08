from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, bounded_float, cache_key, markdown_table, round_or_none
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable
from .persona_belief_panel import (
    DEFAULT_TARGET_FIELDS,
    DEMOGRAPHIC_DIMENSIONS,
    RESPONDENT_COLUMNS,
    TARGET_SPECS,
    build_fixture_respondent_panel,
    classify_persona_evidence,
    score_common_core,
    score_distribution_distance,
    score_gradient_match,
    score_regression_gradient_match,
    score_variance_flattening,
)


PERSONA_ECOLOGY_VERSION = "persona_belief_ecology_v1"
PERSONA_ECOLOGY_PROMPT_VERSION = "persona_belief_ecology_v1"
PERSONA_ECOLOGY_BACKSTORY_PROMPT_VERSION = "persona_belief_ecology_backstory_v1"
DEFAULT_ECOLOGY_SAMPLE_STRATA = ("income_group", "age_group", "education_group")
ECOLOGY_ELICITATION_MODES = ("point", "backstory")
PRIOR_UPDATE_THRESHOLDS = {
    "primary_source_mean_delta_correlation_min": 0.10,
    "primary_source_mean_direction_accuracy_min": 0.55,
    "primary_source_median_update_amplitude_ratio_min": 0.25,
    "primary_source_median_update_amplitude_ratio_max": 2.00,
    "primary_source_mean_rmse_improvement_vs_persistence_min": 0.02,
}
ACTION_COLUMNS = (
    "consumption_change_pct",
    "liquid_buffer_change_pct",
    "borrowing_desire_index",
    "portfolio_rebalance_to_liquid_pct",
    "job_search_intensity_index",
)
OPTIONAL_BEHAVIOR_TARGET_COLUMNS = tuple(f"actual_{column}" for column in ACTION_COLUMNS)
ENVIRONMENT_COLUMNS = (
    "observed_inflation_1y",
    "observed_unemployment_rate",
    "observed_real_income_growth",
    "policy_rate",
    "sentiment_index",
    "news_inflation_pressure",
    "news_labor_pressure",
    "credit_tightness",
    "aggregate_expected_inflation_1y",
    "aggregate_expected_unemployment_rate",
    "aggregate_expected_real_income_growth",
)
INITIAL_PRIORS = {
    "expected_inflation_1y": 3.0,
    "expected_unemployment_rate": 4.5,
    "expected_unemployment_higher_prob": 35.0,
    "expected_real_income_growth": 1.0,
}
SURVEY_SCHEMAS = ("normalized", "sce", "michigan")
PRIOR_MODES = ("simulated", "empirical")
FEEDBACK_MODES = ("closed_loop", "none")
DATE_MODES = ("actual", "relative")


ALIASES: dict[str, tuple[str, ...]] = {
    "respondent_id": ("respondent_id", "caseid", "case_id", "userid", "user_id", "id", "hhid", "pid"),
    "period_id": ("period_id", "wave", "yyyymm", "survey_month", "month", "date", "survey_date"),
    "period_index": ("period_index", "wave_index", "period", "t"),
    "survey_date": ("survey_date", "date", "interview_date", "yyyymm", "survey_month"),
    "weight": ("weight", "wt", "sample_weight", "final_weight", "pweight", "hhweight", "weight_final"),
    "age": ("age", "respondent_age", "rage"),
    "age_group": ("age_group", "agecat", "agegrp", "age_bucket"),
    "income": ("income", "hhinc", "household_income", "income_numeric"),
    "income_group": ("income_group", "income_quartile", "income_bucket", "income_cat", "hhinc_cat"),
    "education": ("education", "educ", "education_level", "educ_cat"),
    "education_group": ("education_group", "educ_group", "education_bucket"),
    "gender": ("gender", "sex", "respondent_gender"),
    "female": ("female", "is_female"),
    "region": ("region", "census_region", "region4"),
    "employment_status": ("employment_status", "employment", "labor_status", "work_status"),
    "homeownership": ("homeownership", "housing_tenure", "tenure", "home_owner", "homeowner"),
    "liquid_wealth": ("liquid_wealth", "liquid_assets", "checking_savings", "cash_on_hand"),
    "liquid_wealth_group": ("liquid_wealth_group", "liquid_assets_group", "liquidity_group"),
    "actual_expected_inflation_1y": (
        "actual_expected_inflation_1y",
        "expected_inflation_1y",
        "inflation_expectation_1y",
        "inflation_1y",
        "q9_mean",
        "q9mean",
        "q8_mean",
        "px1",
        "px1_mean",
    ),
    "actual_expected_unemployment_rate": (
        "actual_expected_unemployment_rate",
        "expected_unemployment_rate",
        "unemployment_expectation",
        "unemp_expectation",
        "unemployment_1y",
        "jobloss_mean",
        "unemp_mean",
    ),
    "actual_expected_unemployment_higher_prob": (
        "actual_expected_unemployment_higher_prob",
        "expected_unemployment_higher_prob",
        "unemployment_higher_probability",
        "unemployment_higher_prob",
        "q4new",
        "q4_new",
    ),
    "actual_expected_real_income_growth": (
        "actual_expected_real_income_growth",
        "expected_real_income_growth",
        "real_income_growth",
        "income_growth_expectation",
        "earnings_growth",
        "earnings_mean",
    ),
    "observed_inflation_1y": ("observed_inflation_1y", "current_inflation", "inflation_observed", "cpi_yoy"),
    "observed_unemployment_rate": ("observed_unemployment_rate", "unemployment_rate", "unrate"),
    "observed_real_income_growth": ("observed_real_income_growth", "real_income_growth_observed", "real_wage_growth"),
    "policy_rate": ("policy_rate", "fed_funds_rate", "ffr", "tbill_rate", "short_rate"),
    "sentiment_index": ("sentiment_index", "consumer_sentiment", "sentiment"),
    "news_inflation_pressure": ("news_inflation_pressure", "inflation_news", "inflation_treatment"),
    "news_labor_pressure": ("news_labor_pressure", "labor_news", "unemployment_news", "labor_treatment"),
    "credit_tightness": ("credit_tightness", "credit_conditions", "credit_tightening"),
}


@dataclass(frozen=True)
class EcologyCard:
    panel_row_id: str
    respondent_id: str
    period_id: str
    period_index: int
    survey_source: str
    survey_date: str
    weight: float
    profile: dict[str, Any]
    empirical_priors: dict[str, float]
    environment: dict[str, float]
    targets: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a respondent-seeded macro belief ecology.")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5,gpt-5.4")
    parser.add_argument("--ecology-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--respondent-source", choices=["fixture", "csv"], default="fixture")
    parser.add_argument("--respondent-csv", default=None)
    parser.add_argument("--survey-schema", choices=SURVEY_SCHEMAS, default="normalized")
    parser.add_argument("--preserve-csv-respondent-ids", action="store_true")
    parser.add_argument("--respondent-count", type=int, default=60)
    parser.add_argument("--respondent-limit", type=int, default=0)
    parser.add_argument("--respondent-sample-size", type=int, default=0)
    parser.add_argument("--respondent-sample-seed", type=int, default=0)
    parser.add_argument("--respondent-sample-strata", default=",".join(DEFAULT_ECOLOGY_SAMPLE_STRATA))
    parser.add_argument("--period-count", type=int, default=4)
    parser.add_argument("--period-ids", default="")
    parser.add_argument("--require-complete-periods", action="store_true")
    parser.add_argument("--survey-start", default="2026-01")
    parser.add_argument("--target-fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default="simulated")
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default="closed_loop")
    parser.add_argument("--date-mode", choices=DATE_MODES, default="actual")
    parser.add_argument("--elicitation-mode", choices=ECOLOGY_ELICITATION_MODES, default="point")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    target_fields = [part.strip() for part in args.target_fields.split(",") if part.strip()]
    unknown_targets = sorted(set(target_fields) - set(TARGET_SPECS))
    if unknown_targets:
        raise SystemExit(f"Unknown target fields: {', '.join(unknown_targets)}")
    if not models:
        raise SystemExit("--models must contain at least one model")
    if args.respondent_limit > 0 and args.respondent_sample_size > 0:
        raise SystemExit("--respondent-limit and --respondent-sample-size cannot be combined")
    if args.ecology_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --ecology-mode live is used")
    if args.ecology_mode == "live" and not args.fresh_cache:
        raise SystemExit("--fresh-cache is required when --ecology-mode live is used")
    if args.respondent_source == "csv":
        if not args.respondent_csv:
            raise SystemExit("--respondent-csv is required when --respondent-source csv")
        if not Path(args.respondent_csv).exists():
            raise SystemExit(f"--respondent-csv does not exist: {args.respondent_csv}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"persona_ecology_{timestamp}"
    cache_dir = output_dir / "fresh_persona_ecology_cache" if args.fresh_cache else WORK_ROOT / "persona_ecology_cache"
    respondent_csv = Path(args.respondent_csv) if args.respondent_csv else None
    panel = load_ecology_panel(
        source=args.respondent_source,
        respondent_csv=respondent_csv,
        survey_schema=args.survey_schema,
        respondent_count=args.respondent_count,
        period_count=args.period_count,
        survey_start=args.survey_start,
        target_fields=target_fields,
    )
    panel = _anonymize_csv_panel_ids(panel) if args.respondent_source == "csv" and not args.preserve_csv_respondent_ids else panel
    pre_selection_panel = panel.copy()
    period_ids = _parse_period_ids(args.period_ids)
    sample_strata = _parse_sample_strata(args.respondent_sample_strata)
    panel, selection_manifest = _select_ecology_panel(
        panel,
        respondent_limit=args.respondent_limit,
        respondent_sample_size=args.respondent_sample_size,
        respondent_sample_seed=args.respondent_sample_seed,
        sample_strata=sample_strata,
        period_ids=period_ids,
        require_complete_periods=args.require_complete_periods,
    )
    behavior_target_source = _behavior_target_source(panel, respondent_source=args.respondent_source)
    cards = build_ecology_cards(panel, target_fields=target_fields)
    required_calls = int(len(cards) * len(models))
    if args.ecology_mode == "live" and args.fresh_cache and args.max_live_calls < required_calls:
        raise SystemExit(
            "--max-live-calls must be at least "
            f"{required_calls} for a fresh live ecology run with {len(cards)} panel rows and {len(models)} models"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    provider_execution_cwd = _isolated_provider_cwd(
        output_dir,
        provider=args.provider,
        source=args.respondent_source,
        mode=args.ecology_mode,
    )
    respondent_input = _respondent_input_manifest(
        source=args.respondent_source,
        respondent_csv=respondent_csv,
        normalized=panel,
        pre_selection_normalized=pre_selection_panel,
        respondent_limit=args.respondent_limit,
        selection_manifest=selection_manifest,
        anonymized=args.respondent_source == "csv" and not args.preserve_csv_respondent_ids,
    )
    primary_update_source = f"llm_{args.provider}_{models[0]}".replace("/", "_")
    manifest: dict[str, Any] = {
        "schema_version": PERSONA_ECOLOGY_VERSION,
        "prompt_version": PERSONA_ECOLOGY_PROMPT_VERSION,
        "timestamp_utc": timestamp,
        "argv": _sanitized_argv(),
        "run_command": shlex.join(_sanitized_argv()),
        "git": _git_metadata(),
        "provider": args.provider,
        "models": models,
        "ecology_mode": args.ecology_mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "respondent_source": args.respondent_source,
        "respondent_input": respondent_input,
        "survey_schema": args.survey_schema,
        "respondent_count": int(panel["respondent_id"].nunique()),
        "respondent_limit": int(args.respondent_limit),
        "respondent_sample_size": int(args.respondent_sample_size),
        "respondent_sample_seed": int(args.respondent_sample_seed),
        "respondent_sample_strata": list(sample_strata),
        "preserve_csv_respondent_ids": bool(args.preserve_csv_respondent_ids),
        "period_ids_filter": list(period_ids),
        "require_complete_periods": bool(args.require_complete_periods),
        "panel_row_count": int(panel.shape[0]),
        "period_count": int(panel["period_index"].nunique()),
        "target_fields": target_fields,
        "prior_mode": args.prior_mode,
        "elicitation_mode": args.elicitation_mode,
        "primary_update_source": primary_update_source,
        "pre_registered_prior_update_thresholds": PRIOR_UPDATE_THRESHOLDS,
        "split_roles": _split_roles_manifest(respondent_input, period_ids=period_ids, prior_mode=args.prior_mode),
        "feedback_mode": args.feedback_mode,
        "date_mode": args.date_mode,
        "behavior_target_source": behavior_target_source,
        "required_call_count": required_calls,
        "provider_execution_cwd": _safe_relative(provider_execution_cwd) if provider_execution_cwd else None,
        "shared_cache_allowed": bool(not args.fresh_cache),
        "status": "running",
    }
    _write_json(output_dir / "manifest.json", manifest)

    try:
        all_predictions: list[pd.DataFrame] = []
        all_actions: list[pd.DataFrame] = []
        all_environments: list[pd.DataFrame] = []
        raw_records: list[dict[str, Any]] = []
        prompt_rows: list[dict[str, Any]] = []
        live_calls = 0
        cache_hits = 0
        model_run_modes: dict[str, dict[str, Any]] = {}
        for model in models:
            client_mode, client_cap = _client_mode_and_cap(args.ecology_mode, args.max_live_calls, live_calls)
            model_run_modes[model] = {"mode": client_mode, "max_live_calls": int(client_cap)}
            client = PersonaEcologyClient(
                args.provider,
                model,
                cache_dir,
                mode=client_mode,
                max_live_calls=client_cap,
                execution_cwd=provider_execution_cwd,
                elicitation=args.elicitation_mode,
            )
            predictions, actions, environments, prompts = run_persona_ecology(
                cards,
                client,
                target_fields=target_fields,
                prior_mode=args.prior_mode,
                feedback_mode=args.feedback_mode,
                date_mode=args.date_mode,
                elicitation=args.elicitation_mode,
            )
            all_predictions.append(predictions)
            all_actions.append(actions)
            all_environments.append(environments)
            prompt_rows.extend(prompts)
            raw_records.extend(client.raw_records)
            live_calls += client.live_call_count
            cache_hits += client.cache_hit_count

        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
        environments = pd.concat(all_environments, ignore_index=True) if all_environments else pd.DataFrame()
        scoring_panel, scoring_predictions = _scoring_frames(panel, predictions)
        regression_scores = score_regression_gradient_match(scoring_panel, scoring_predictions, target_fields=target_fields)
        gradient_scores = score_gradient_match(scoring_panel, scoring_predictions, target_fields=target_fields)
        variance_scores = score_variance_flattening(scoring_panel, scoring_predictions, target_fields=target_fields)
        distribution_scores = score_distribution_distance(scoring_panel, scoring_predictions, target_fields=target_fields)
        common_core = score_common_core(scoring_predictions, target_fields=target_fields)
        temporal_scores = score_temporal_dynamics(panel, predictions, target_fields=target_fields)
        period_scores = score_period_levels(panel, predictions, target_fields=target_fields)
        update_scores = score_period_updates(panel, predictions, target_fields=target_fields)
        prior_update_scores = score_prior_update_dynamics(panel, predictions, target_fields=target_fields)
        behavior_scores = score_behavior_actions(panel, actions)
        action_period_summary = summarize_period_actions(actions)
        ablation_predictions = build_module_ablations(panel, target_fields=target_fields)
        ablation_scores = score_module_ablations(panel, ablation_predictions, target_fields=target_fields)
        static_evidence = classify_persona_evidence(regression_scores, variance_scores, common_core, distribution_scores)
        ecology_evidence = classify_ecology_evidence(
            static_evidence,
            temporal_scores,
            behavior_scores,
            environments,
            manifest,
        )
        prior_update_evidence = classify_prior_update_evidence(prior_update_scores, manifest)

        panel.to_csv(output_dir / "persona_ecology_panel.csv", index=False)
        pd.DataFrame(prompt_rows).to_json(output_dir / "persona_ecology_prompt_cards.jsonl", orient="records", lines=True)
        predictions.to_csv(output_dir / "persona_ecology_predictions.csv", index=False)
        actions.to_csv(output_dir / "persona_ecology_actions.csv", index=False)
        environments.to_csv(output_dir / "persona_ecology_environment_history.csv", index=False)
        regression_scores.to_csv(output_dir / "persona_ecology_regression_scores.csv", index=False)
        gradient_scores.to_csv(output_dir / "persona_ecology_gradient_scores.csv", index=False)
        variance_scores.to_csv(output_dir / "persona_ecology_variance_scores.csv", index=False)
        distribution_scores.to_csv(output_dir / "persona_ecology_distribution_scores.csv", index=False)
        common_core.to_csv(output_dir / "persona_ecology_common_core.csv", index=False)
        temporal_scores.to_csv(output_dir / "persona_ecology_temporal_scores.csv", index=False)
        period_scores.to_csv(output_dir / "persona_ecology_period_scores.csv", index=False)
        update_scores.to_csv(output_dir / "persona_ecology_update_scores.csv", index=False)
        prior_update_scores.to_csv(output_dir / "persona_ecology_prior_update_scores.csv", index=False)
        behavior_scores.to_csv(output_dir / "persona_ecology_behavior_scores.csv", index=False)
        action_period_summary.to_csv(output_dir / "persona_ecology_action_period_summary.csv", index=False)
        ablation_predictions.to_csv(output_dir / "persona_ecology_module_ablations.csv", index=False)
        ablation_scores.to_csv(output_dir / "persona_ecology_module_ablation_scores.csv", index=False)
        _write_json(output_dir / "persona_ecology_raw_records.json", raw_records)

        manifest.update(
            {
                "status": "ok",
                "model_run_modes": model_run_modes,
                "prediction_rows": int(predictions.shape[0]),
                "action_rows": int(actions.shape[0]),
                "environment_rows": int(environments.shape[0]),
                "live_call_count": int(live_calls),
                "cache_hit_count": int(cache_hits),
                "fixture_generated_count": int((predictions["call_source"] == "fixture").sum()) if "call_source" in predictions else 0,
                "static_evidence": static_evidence,
                "ecology_evidence": ecology_evidence,
                "prior_update_evidence": prior_update_evidence,
                "behavior_targets_available": bool(_available_behavior_targets(panel)),
                "behavior_target_source": behavior_target_source,
                "cache_dir": _safe_relative(cache_dir),
                "outputs": [
                    "persona_ecology_panel.csv",
                    "persona_ecology_prompt_cards.jsonl",
                    "persona_ecology_predictions.csv",
                    "persona_ecology_actions.csv",
                    "persona_ecology_environment_history.csv",
                    "persona_ecology_regression_scores.csv",
                    "persona_ecology_gradient_scores.csv",
                    "persona_ecology_variance_scores.csv",
                    "persona_ecology_distribution_scores.csv",
                    "persona_ecology_common_core.csv",
                    "persona_ecology_temporal_scores.csv",
                    "persona_ecology_period_scores.csv",
                    "persona_ecology_update_scores.csv",
                    "persona_ecology_prior_update_scores.csv",
                    "persona_ecology_behavior_scores.csv",
                    "persona_ecology_action_period_summary.csv",
                    "persona_ecology_module_ablations.csv",
                    "persona_ecology_module_ablation_scores.csv",
                    "persona_ecology_raw_records.json",
                    "persona_ecology_report.md",
                ],
            }
        )
        report = build_persona_ecology_report(
            manifest,
            regression_scores,
            variance_scores,
            distribution_scores,
            common_core,
            temporal_scores,
            behavior_scores,
            ablation_scores,
            environments,
            period_scores=period_scores,
            update_scores=update_scores,
            prior_update_scores=prior_update_scores,
            action_period_summary=action_period_summary,
        )
        (output_dir / "persona_ecology_report.md").write_text(report, encoding="utf-8")
        _write_json(output_dir / "manifest.json", manifest)
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        _write_json(output_dir / "manifest.json", manifest)
        raise


class PersonaEcologyClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str,
        max_live_calls: int,
        execution_cwd: Path | None = None,
        elicitation: str = "point",
    ):
        if elicitation not in ECOLOGY_ELICITATION_MODES:
            raise ValueError(f"Unsupported ecology elicitation mode: {elicitation}")
        self.provider = provider
        self.model = model
        self.mode = mode
        self.elicitation = elicitation
        self._client = ForecastLLMClient(
            provider,
            model,
            cache_dir,
            mode=mode,
            max_live_calls=max_live_calls,
            execution_cwd=execution_cwd,
        )
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._client.cache_hit_count

    def ecology_card(
        self,
        card: EcologyCard,
        *,
        target_fields: Iterable[str],
        prior_beliefs: dict[str, float],
        environment: dict[str, float],
        date_mode: str = "actual",
    ) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_persona_ecology_payload(
                    card,
                    self.model,
                    target_fields=target_fields,
                    prior_beliefs=prior_beliefs,
                    environment=environment,
                    date_mode=date_mode,
                    elicitation=self.elicitation,
                ),
                "cache_hit": False,
                "cache_path": None,
                "call_source": "fixture",
            }
        else:
            prompt = persona_ecology_prompt(
                card,
                target_fields=target_fields,
                prior_beliefs=prior_beliefs,
                environment=environment,
                date_mode=date_mode,
                elicitation=self.elicitation,
            )
            cache_name = f"persona_ecology_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"
            data = self._client.json_call(prompt, cache_name, instructions=_persona_ecology_instructions())
        call_source = data.get("call_source") or ("cache" if data.get("cache_hit") else ("live" if self.mode == "live" else self.mode))
        normalized = normalize_persona_ecology_payload(
            card,
            data,
            provider=self.provider,
            model=self.model,
            target_fields=target_fields,
            date_mode=date_mode,
        )
        normalized["call_source"] = call_source
        self.raw_records.append(
            {
                "panel_row_id": card.panel_row_id,
                "respondent_id": card.respondent_id,
                "period_id": card.period_id,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "call_source": call_source,
                "payload": data.get("payload", data),
            }
        )
        return normalized


def load_ecology_panel(
    *,
    source: str,
    respondent_csv: Path | None,
    survey_schema: str,
    respondent_count: int,
    period_count: int,
    survey_start: str,
    target_fields: Iterable[str],
) -> pd.DataFrame:
    if source == "fixture":
        return build_fixture_ecology_panel(
            respondent_count=respondent_count,
            period_count=period_count,
            survey_start=survey_start,
            target_fields=target_fields,
        )
    if respondent_csv is None:
        raise ValueError("--respondent-csv is required when --respondent-source csv")
    frame = pd.read_csv(respondent_csv)
    return normalize_ecology_panel(frame, survey_schema=survey_schema, target_fields=target_fields)


def normalize_ecology_panel(
    frame: pd.DataFrame,
    *,
    survey_schema: str = "normalized",
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if survey_schema not in SURVEY_SCHEMAS:
        raise ValueError(f"Unsupported survey schema: {survey_schema}")
    out = _standardize_aliases(frame)
    if "respondent_id" not in out:
        out["respondent_id"] = [f"respondent_{idx + 1:05d}" for idx in range(out.shape[0])]
    if "period_id" not in out:
        if "survey_date" in out:
            out["period_id"] = out["survey_date"].astype(str)
        else:
            out["period_id"] = "period_000"
    if "survey_date" not in out:
        out["survey_date"] = out["period_id"].astype(str)
    if "period_index" not in out:
        period_map = {period: idx for idx, period in enumerate(sorted(out["period_id"].astype(str).unique()))}
        out["period_index"] = out["period_id"].astype(str).map(period_map)
    if "weight" not in out:
        out["weight"] = 1.0
    if "survey_source" not in out:
        out["survey_source"] = f"{survey_schema}_respondent_csv"

    out["age_group"] = _coalesce_text(out, "age_group", _age_group_from_numeric(out.get("age")))
    out["income_group"] = _coalesce_text(out, "income_group", _income_group_from_numeric(out.get("income")))
    out["education_group"] = _coalesce_text(out, "education_group", _education_group_from_raw(out.get("education")))
    out["gender"] = _coalesce_text(out, "gender", _gender_from_raw(out.get("female")))
    out["region"] = _coalesce_text(out, "region", pd.Series("unknown", index=out.index))
    out["employment_status"] = _coalesce_text(out, "employment_status", pd.Series("unknown", index=out.index))
    out["homeownership"] = _coalesce_text(out, "homeownership", pd.Series("unknown", index=out.index))
    out["liquid_wealth_group"] = _coalesce_text(out, "liquid_wealth_group", _income_group_from_numeric(out.get("liquid_wealth")))

    for target in target_fields:
        actual_column = f"actual_{target}"
        if actual_column not in out:
            if target in out:
                out[actual_column] = out[target]
            else:
                raise ValueError(f"Respondent panel missing `{actual_column}`")
        out[actual_column] = pd.to_numeric(out[actual_column], errors="coerce")
    out = out.dropna(subset=[f"actual_{target}" for target in target_fields]).copy()
    if out.empty:
        raise ValueError("Respondent panel has no rows with scoreable target responses")

    out["period_index"] = pd.to_numeric(out["period_index"], errors="coerce").fillna(0).astype(int)
    out["period_id"] = out["period_id"].astype(str)
    out["respondent_id"] = out["respondent_id"].astype(str)
    out["panel_row_id"] = out.get("panel_row_id", out["respondent_id"] + "__" + out["period_id"]).astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = _normalize_period_weights(out)

    for target in target_fields:
        prior_column = f"prior_{target}"
        if prior_column not in out:
            out[prior_column] = np.nan
        out[prior_column] = pd.to_numeric(out[prior_column], errors="coerce")
    out = _fill_missing_priors(out, target_fields=target_fields)

    for column in ENVIRONMENT_COLUMNS:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = _fill_missing_environment(out)

    for column in OPTIONAL_BEHAVIOR_TARGET_COLUMNS:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("persona_panel_kind", "target_provenance", "environment_provenance"):
        if column not in out:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)

    for column in RESPONDENT_COLUMNS:
        if column not in out:
            out[column] = "unknown" if column != "weight" else 1.0
    for column in RESPONDENT_COLUMNS:
        if column != "weight":
            out[column] = out[column].fillna("unknown").astype(str)

    keep = [
        "panel_row_id",
        "period_id",
        "period_index",
        *RESPONDENT_COLUMNS,
        *[f"prior_{target}" for target in target_fields],
        *[f"actual_{target}" for target in target_fields],
        *ENVIRONMENT_COLUMNS,
        *[column for column in OPTIONAL_BEHAVIOR_TARGET_COLUMNS if column in out],
        "persona_panel_kind",
        "target_provenance",
        "environment_provenance",
    ]
    out = out[keep].sort_values(["period_index", "respondent_id"]).reset_index(drop=True)
    if out["panel_row_id"].duplicated().any():
        duplicates = sorted(out.loc[out["panel_row_id"].duplicated(), "panel_row_id"].astype(str).unique())[:5]
        raise ValueError(f"Ecology panel has duplicate panel_row_id values: {', '.join(duplicates)}")
    return out


def build_fixture_ecology_panel(
    *,
    respondent_count: int = 60,
    period_count: int = 4,
    survey_start: str = "2026-01",
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if period_count <= 0:
        raise ValueError("period_count must be positive")
    base = build_fixture_respondent_panel(
        respondent_count=respondent_count,
        survey_date=f"{survey_start}-01" if len(survey_start) == 7 else survey_start,
        target_fields=target_fields,
    )
    period_dates = pd.date_range(f"{survey_start}-01" if len(survey_start) == 7 else survey_start, periods=period_count, freq="MS")
    prior_by_respondent: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for period_index, survey_date in enumerate(period_dates):
        environment = _fixture_environment(period_index)
        for row_index, (_, respondent) in enumerate(base.iterrows()):
            respondent_id = str(respondent["respondent_id"])
            profile = {column: respondent[column] for column in RESPONDENT_COLUMNS if column not in {"respondent_id", "survey_source", "survey_date", "weight"}}
            prior = prior_by_respondent.get(respondent_id) or {
                target: _profile_anchor(profile, target) for target in target_fields
            }
            actuals = {
                target: _fixture_actual_ecology_belief(profile, prior[target], environment, target, row_index, period_index)
                for target in target_fields
            }
            action_targets = _fixture_actual_actions(profile, actuals, environment)
            period_id = f"p{period_index:02d}"
            out = respondent.to_dict()
            out.update(
                {
                    "panel_row_id": f"{respondent_id}__{period_id}",
                    "period_id": period_id,
                    "period_index": period_index,
                    "survey_date": survey_date.date().isoformat(),
                    "survey_source": "fixture_dynamic_micro_survey",
                    "weight": 1.0 / max(1, respondent_count),
                    **{f"prior_{target}": prior[target] for target in target_fields},
                    **{f"actual_{target}": actuals[target] for target in target_fields},
                    **environment,
                    **{f"actual_{column}": value for column, value in action_targets.items()},
                }
            )
            rows.append(out)
            prior_by_respondent[respondent_id] = actuals
    return normalize_ecology_panel(pd.DataFrame(rows), survey_schema="normalized", target_fields=target_fields)


def build_ecology_cards(panel: pd.DataFrame, *, target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS) -> list[EcologyCard]:
    cards: list[EcologyCard] = []
    for _, row in panel.sort_values(["period_index", "respondent_id"]).iterrows():
        profile = {column: row[column] for column in RESPONDENT_COLUMNS if column not in {"respondent_id", "survey_source", "survey_date", "weight"}}
        priors = {target: float(row[f"prior_{target}"]) for target in target_fields}
        targets = {target: float(row[f"actual_{target}"]) for target in target_fields}
        environment = {column: float(row[column]) for column in ENVIRONMENT_COLUMNS}
        cards.append(
            EcologyCard(
                panel_row_id=str(row["panel_row_id"]),
                respondent_id=str(row["respondent_id"]),
                period_id=str(row["period_id"]),
                period_index=int(row["period_index"]),
                survey_source=str(row["survey_source"]),
                survey_date=str(row["survey_date"]),
                weight=float(row["weight"]),
                profile=profile,
                empirical_priors=priors,
                environment=environment,
                targets=targets,
            )
        )
    return cards


def persona_ecology_prompt(
    card: EcologyCard,
    *,
    target_fields: Iterable[str],
    prior_beliefs: dict[str, float],
    environment: dict[str, float],
    date_mode: str = "actual",
    elicitation: str = "point",
) -> str:
    payload = persona_ecology_prompt_payload(
        card,
        target_fields=target_fields,
        prior_beliefs=prior_beliefs,
        environment=environment,
        date_mode=date_mode,
        elicitation=elicitation,
    )
    return json.dumps(payload, indent=2, sort_keys=True)


def persona_ecology_prompt_payload(
    card: EcologyCard,
    *,
    target_fields: Iterable[str],
    prior_beliefs: dict[str, float],
    environment: dict[str, float],
    date_mode: str = "actual",
    elicitation: str = "point",
) -> dict[str, Any]:
    if elicitation not in ECOLOGY_ELICITATION_MODES:
        raise ValueError(f"Unsupported ecology elicitation mode: {elicitation}")
    prompt_identity = _prompt_identity(card, date_mode=date_mode)
    record_kind = "synthetic fixture respondent" if "fixture" in card.survey_source else "survey respondent"
    payload = {
        "prompt_version": PERSONA_ECOLOGY_BACKSTORY_PROMPT_VERSION if elicitation == "backstory" else PERSONA_ECOLOGY_PROMPT_VERSION,
        "task": (
            f"Simulate this {record_kind} as a persistent household agent in a macroeconomic survey panel."
        ),
        "paper_spine": {
            "profile_module": "Use stable personal characteristics only as one input, not as a stereotype.",
            "prior_expectations_module": "Start from the supplied prior expectations and update rather than overwrite them.",
            "external_information_module": "Use only the supplied as-of environment and public aggregate feedback signal.",
            "belief_update_module": "Report how much weight you placed on prior beliefs, profile, environment, and peer/aggregate signal.",
            "behavior_module": "Choose desired behavior consistent with the updated beliefs and household constraints.",
        },
        "as_of_rule": (
            "Do not use realized future values, hidden held-out survey responses, files, tools, or outside data. "
            "The held-out current survey answers are not shown and are used only for scoring."
        ),
        "respondent_record_kind": record_kind,
        "date_mode": date_mode,
        "panel_row_id": prompt_identity["panel_row_id"],
        "respondent_id": prompt_identity["respondent_id"],
        "survey_source": _prompt_survey_source(card, date_mode=date_mode),
        "period_id": prompt_identity["period_id"],
        "survey_date": prompt_identity["survey_date"],
        "respondent_profile": card.profile,
        "prior_beliefs": {target: round_or_none(prior_beliefs[target]) for target in target_fields},
        "current_environment": {key: round_or_none(value) for key, value in environment.items()},
        "required_response": {
            "panel_row_id": prompt_identity["panel_row_id"],
            "respondent_id": prompt_identity["respondent_id"],
            "beliefs": {
                target: {
                    "value": f"numeric {TARGET_SPECS[target]['units']} value",
                    "p10": "10th percentile subjective belief",
                    "p50": "median subjective belief",
                    "p90": "90th percentile subjective belief",
                }
                for target in target_fields
            },
            "actions": {
                "consumption_change_pct": "desired percent change in near-term consumption, -20 to 20",
                "liquid_buffer_change_pct": "desired percent change in liquid buffer, -25 to 25",
                "borrowing_desire_index": "positive means borrow more, negative means repay debt, -5 to 5",
                "portfolio_rebalance_to_liquid_pct": "positive moves illiquid wealth to liquid, -15 to 15",
                "job_search_intensity_index": "-3 to 6",
            },
            "module_weights": {
                "profile_weight": "0 to 1",
                "prior_weight": "0 to 1",
                "environment_weight": "0 to 1",
                "aggregate_feedback_weight": "0 to 1",
            },
            "confidence": "0 to 1",
            "uncertainty": "0 to 1.5",
            "reason": "short explanation based only on supplied profile, priors, and environment",
        },
    }
    if elicitation == "backstory":
        payload["task"] = (
            f"First imagine one specific plausible individual who matches this {record_kind} profile and already held "
            "the supplied prior beliefs. Then update that individual's beliefs and behavior from the current environment."
        )
        payload["persona_rule"] = (
            "Invent one concrete individual consistent with the profile and prior beliefs: household situation, work, "
            "recent experiences with prices, income, job security, and liquidity. Different respondents with the same "
            "profile can still be different people. Use the imagined life to preserve respondent-level heterogeneity, "
            "but do not infer or copy hidden survey answers."
        )
        payload["required_response"]["persona_sketch"] = "2-3 sentence sketch of the specific imagined individual"
        payload["required_response"]["reason"] = "short explanation grounded in the imagined individual, priors, and environment"
    return payload


def _prompt_identity(card: EcologyCard, *, date_mode: str) -> dict[str, str]:
    if date_mode not in DATE_MODES:
        raise ValueError(f"Unsupported date_mode: {date_mode}")
    if date_mode == "actual":
        return {
            "panel_row_id": card.panel_row_id,
            "respondent_id": card.respondent_id,
            "period_id": card.period_id,
            "survey_date": card.survey_date,
        }
    period_label = f"period_{card.period_index}"
    return {
        "panel_row_id": f"{card.respondent_id}__{period_label}",
        "respondent_id": card.respondent_id,
        "period_id": period_label,
        "survey_date": period_label,
    }


def _prompt_survey_source(card: EcologyCard, *, date_mode: str) -> str:
    if date_mode not in DATE_MODES:
        raise ValueError(f"Unsupported date_mode: {date_mode}")
    if date_mode == "actual":
        return card.survey_source
    if "fixture" in card.survey_source:
        return "synthetic_fixture_panel"
    return "survey_panel"


def run_persona_ecology(
    cards: Iterable[EcologyCard],
    client: PersonaEcologyClient,
    *,
    target_fields: Iterable[str],
    prior_mode: str = "simulated",
    feedback_mode: str = "closed_loop",
    date_mode: str = "actual",
    elicitation: str = "point",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"Unsupported prior mode: {prior_mode}")
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"Unsupported feedback mode: {feedback_mode}")
    if date_mode not in DATE_MODES:
        raise ValueError(f"Unsupported date_mode: {date_mode}")
    if elicitation not in ECOLOGY_ELICITATION_MODES:
        raise ValueError(f"Unsupported ecology elicitation mode: {elicitation}")
    ordered_cards = sorted(cards, key=lambda card: (card.period_index, card.respondent_id))
    source = f"llm_{client.provider}_{client.model}".replace("/", "_")
    simulated_priors: dict[str, dict[str, float]] = {}
    previous_aggregate: dict[str, Any] | None = None
    prediction_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    periods = sorted({card.period_index for card in ordered_cards})
    for period_index in periods:
        period_cards = [card for card in ordered_cards if card.period_index == period_index]
        period_predictions: list[dict[str, Any]] = []
        period_actions: list[dict[str, Any]] = []
        for card in period_cards:
            prior_beliefs = _prior_for_card(card, simulated_priors, prior_mode=prior_mode, target_fields=target_fields)
            environment = _environment_for_card(card.environment, previous_aggregate, feedback_mode=feedback_mode)
            prompt_identity = _prompt_identity(card, date_mode=date_mode)
            prompt_rows.append(
                {
                    "panel_row_id": prompt_identity["panel_row_id"],
                    "respondent_id": prompt_identity["respondent_id"],
                    "period_id": prompt_identity["period_id"],
                    "survey_date": prompt_identity["survey_date"],
                    "date_mode": date_mode,
                    "source": source,
                    "prompt_payload": persona_ecology_prompt_payload(
                        card,
                        target_fields=target_fields,
                        prior_beliefs=prior_beliefs,
                        environment=environment,
                        date_mode=date_mode,
                        elicitation=elicitation,
                    ),
                    "prompt_text": persona_ecology_prompt(
                        card,
                        target_fields=target_fields,
                        prior_beliefs=prior_beliefs,
                        environment=environment,
                        date_mode=date_mode,
                        elicitation=elicitation,
                    ),
                }
            )
            payload = client.ecology_card(
                card,
                target_fields=target_fields,
                prior_beliefs=prior_beliefs,
                environment=environment,
                date_mode=date_mode,
            )
            simulated_priors[card.respondent_id] = {
                target: float(payload["beliefs"][target]["value"]) for target in target_fields
            }
            action_row = _action_row(card, payload, source=source, environment=environment, prior_beliefs=prior_beliefs)
            action_rows.append(action_row)
            period_actions.append(action_row)
            for target in target_fields:
                belief = payload["beliefs"][target]
                prediction_row = {
                    "schema_version": PERSONA_ECOLOGY_VERSION,
                    "panel_row_id": card.panel_row_id,
                    "respondent_id": card.respondent_id,
                    "period_id": card.period_id,
                    "period_index": card.period_index,
                    "survey_source": card.survey_source,
                    "survey_date": card.survey_date,
                    "source": source,
                    "provider": client.provider,
                    "model": client.model,
                    "target_name": target,
                    "prior_prediction": prior_beliefs[target],
                    "prediction": belief["value"],
                    "p10": belief["p10"],
                    "p50": belief["p50"],
                    "p90": belief["p90"],
                    "confidence": payload["confidence"],
                    "uncertainty": payload["uncertainty"],
                    "profile_weight": payload["module_weights"]["profile_weight"],
                    "prior_weight": payload["module_weights"]["prior_weight"],
                    "environment_weight": payload["module_weights"]["environment_weight"],
                    "aggregate_feedback_weight": payload["module_weights"]["aggregate_feedback_weight"],
                    "reason": payload["reason"],
                    "persona_sketch": payload.get("persona_sketch", ""),
                    "cache_hit": payload["cache_hit"],
                    "cache_path": payload["cache_path"],
                    "call_source": payload["call_source"],
                }
                prediction_rows.append(prediction_row)
                period_predictions.append(prediction_row)
        previous_aggregate = aggregate_period_feedback(
            period_cards,
            period_predictions,
            period_actions,
            source=source,
            feedback_mode=feedback_mode,
            previous_aggregate=previous_aggregate,
            target_fields=target_fields,
        )
        environment_rows.append(previous_aggregate)

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(action_rows),
        pd.DataFrame(environment_rows),
        prompt_rows,
    )


def normalize_persona_ecology_payload(
    card: EcologyCard,
    data: dict[str, Any],
    *,
    provider: str,
    model: str,
    target_fields: Iterable[str],
    date_mode: str = "actual",
) -> dict[str, Any]:
    payload = data.get("payload", data)
    prompt_identity = _prompt_identity(card, date_mode=date_mode)
    panel_row_id = str(payload.get("panel_row_id", card.panel_row_id))
    respondent_id = str(payload.get("respondent_id", card.respondent_id))
    if panel_row_id != prompt_identity["panel_row_id"]:
        raise LLMUnavailable(
            f"Ecology payload panel_row_id mismatch: expected {prompt_identity['panel_row_id']}, got {panel_row_id}"
        )
    if respondent_id != prompt_identity["respondent_id"]:
        raise LLMUnavailable(
            f"Ecology payload respondent_id mismatch: expected {prompt_identity['respondent_id']}, got {respondent_id}"
        )
    beliefs = payload.get("beliefs")
    actions = payload.get("actions")
    module_weights = payload.get("module_weights")
    if not isinstance(beliefs, dict):
        raise LLMUnavailable(f"Ecology payload for {card.panel_row_id} is missing beliefs object")
    if not isinstance(actions, dict):
        raise LLMUnavailable(f"Ecology payload for {card.panel_row_id} is missing actions object")
    if not isinstance(module_weights, dict):
        raise LLMUnavailable(f"Ecology payload for {card.panel_row_id} is missing module_weights object")
    normalized: dict[str, Any] = {
        "panel_row_id": card.panel_row_id,
        "respondent_id": card.respondent_id,
        "provider": provider,
        "model": model,
        "beliefs": {},
        "actions": {
            "consumption_change_pct": bounded_float(actions, "consumption_change_pct", -20.0, 20.0),
            "liquid_buffer_change_pct": bounded_float(actions, "liquid_buffer_change_pct", -25.0, 25.0),
            "borrowing_desire_index": bounded_float(actions, "borrowing_desire_index", -5.0, 5.0),
            "portfolio_rebalance_to_liquid_pct": bounded_float(actions, "portfolio_rebalance_to_liquid_pct", -15.0, 15.0),
            "job_search_intensity_index": bounded_float(actions, "job_search_intensity_index", -3.0, 6.0),
        },
        "module_weights": {
            "profile_weight": bounded_float(module_weights, "profile_weight", 0.0, 1.0),
            "prior_weight": bounded_float(module_weights, "prior_weight", 0.0, 1.0),
            "environment_weight": bounded_float(module_weights, "environment_weight", 0.0, 1.0),
            "aggregate_feedback_weight": bounded_float(module_weights, "aggregate_feedback_weight", 0.0, 1.0),
        },
        "confidence": bounded_float(payload, "confidence", 0.0, 1.0),
        "uncertainty": bounded_float(payload, "uncertainty", 0.0, 1.5),
        "reason": str(payload.get("reason", ""))[:500],
        "persona_sketch": str(payload.get("persona_sketch", ""))[:500],
        "cache_hit": bool(data.get("cache_hit", False)),
        "cache_path": data.get("cache_path"),
    }
    missing = [target for target in target_fields if target not in beliefs]
    if missing:
        raise LLMUnavailable(f"Ecology payload for {card.panel_row_id} missing beliefs: {', '.join(missing)}")
    for target in target_fields:
        spec = TARGET_SPECS[target]
        raw = beliefs[target]
        if not isinstance(raw, dict):
            raise LLMUnavailable(f"Ecology belief `{target}` for {card.panel_row_id} must be an object")
        value = bounded_float(raw, "value", spec["lower"], spec["upper"])
        p10 = bounded_float(raw, "p10", spec["lower"], spec["upper"])
        p50 = bounded_float(raw, "p50", spec["lower"], spec["upper"])
        p90 = bounded_float(raw, "p90", spec["lower"], spec["upper"])
        lo, mid, hi = sorted([p10, p50, p90])
        normalized["beliefs"][target] = {"value": value, "p10": lo, "p50": mid, "p90": hi}
    return normalized


def fixture_persona_ecology_payload(
    card: EcologyCard,
    model: str,
    *,
    target_fields: Iterable[str],
    prior_beliefs: dict[str, float],
    environment: dict[str, float],
    date_mode: str = "actual",
    elicitation: str = "point",
) -> dict[str, Any]:
    if elicitation not in ECOLOGY_ELICITATION_MODES:
        raise ValueError(f"Unsupported ecology elicitation mode: {elicitation}")
    beliefs: dict[str, dict[str, float]] = {}
    module_weights = _fixture_module_weights(card.profile, environment)
    for target in target_fields:
        prior = float(prior_beliefs[target])
        profile_anchor = _profile_anchor(card.profile, target)
        environment_anchor = _environment_anchor(environment, target)
        model_shift = _model_shift(model, card.respondent_id, target)
        point = (
            module_weights["prior_weight"] * prior
            + module_weights["profile_weight"] * profile_anchor
            + module_weights["environment_weight"] * environment_anchor
            + module_weights["aggregate_feedback_weight"] * _aggregate_anchor(environment, target)
            + model_shift
        )
        width = _fixture_ecology_uncertainty(card.profile, environment, target)
        spec = TARGET_SPECS[target]
        value = float(np.clip(point, spec["lower"], spec["upper"]))
        beliefs[target] = {
            "value": value,
            "p10": float(np.clip(value - width, spec["lower"], spec["upper"])),
            "p50": value,
            "p90": float(np.clip(value + width, spec["lower"], spec["upper"])),
        }
    actions = _fixture_actions(card.profile, {target: beliefs[target]["value"] for target in target_fields}, environment)
    prompt_identity = _prompt_identity(card, date_mode=date_mode)
    payload = {
        "prompt_version": PERSONA_ECOLOGY_BACKSTORY_PROMPT_VERSION if elicitation == "backstory" else PERSONA_ECOLOGY_PROMPT_VERSION,
        "panel_row_id": prompt_identity["panel_row_id"],
        "respondent_id": prompt_identity["respondent_id"],
        "beliefs": beliefs,
        "actions": actions,
        "module_weights": module_weights,
        "confidence": float(np.clip(0.78 - 0.10 * actions["job_search_intensity_index"] / 6.0, 0.25, 0.92)),
        "uncertainty": float(np.clip(max(_fixture_ecology_uncertainty(card.profile, environment, target) for target in target_fields) / 4.0, 0.10, 1.25)),
        "reason": "deterministic fixture ecology update from profile, priors, environment, and aggregate feedback",
    }
    if elicitation == "backstory":
        payload["persona_sketch"] = (
            f"deterministic fixture individual for {card.respondent_id}: "
            f"{card.profile.get('employment_status', 'unknown')} household with prior beliefs carried into this period"
        )
    return payload


def aggregate_period_feedback(
    period_cards: list[EcologyCard],
    prediction_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    *,
    source: str,
    feedback_mode: str,
    previous_aggregate: dict[str, Any] | None,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> dict[str, Any]:
    if not period_cards:
        raise ValueError("period_cards must not be empty")
    period_id = period_cards[0].period_id
    period_index = period_cards[0].period_index
    weights = {card.panel_row_id: card.weight for card in period_cards}
    predictions = pd.DataFrame(prediction_rows)
    actions = pd.DataFrame(action_rows)
    row: dict[str, Any] = {
        "schema_version": PERSONA_ECOLOGY_VERSION,
        "source": source,
        "period_id": period_id,
        "period_index": int(period_index),
        "feedback_mode": feedback_mode,
        "respondent_count": len(period_cards),
        "previous_period_id": previous_aggregate.get("period_id") if previous_aggregate else None,
    }
    for column in ENVIRONMENT_COLUMNS:
        row[f"mean_seen_{column}"] = _weighted_card_environment(period_cards, column)
    for target in target_fields:
        group = predictions[predictions["target_name"] == target] if not predictions.empty else pd.DataFrame()
        if group.empty:
            continue
        group_weights = group["panel_row_id"].map(weights).fillna(0.0)
        row[f"aggregate_{target}"] = _weighted_mean(group["prediction"], group_weights)
    for column in ACTION_COLUMNS:
        row[f"aggregate_{column}"] = _weighted_mean(actions[column], actions["weight"]) if not actions.empty else np.nan
    consumption = float(row.get("aggregate_consumption_change_pct", 0.0))
    borrowing = float(row.get("aggregate_borrowing_desire_index", 0.0))
    liquid_buffer = float(row.get("aggregate_liquid_buffer_change_pct", 0.0))
    job_search = float(row.get("aggregate_job_search_intensity_index", 0.0))
    portfolio_liquid = float(row.get("aggregate_portfolio_rebalance_to_liquid_pct", 0.0))
    row["aggregate_demand_pressure"] = float(np.clip(consumption + 0.35 * borrowing - 0.15 * liquid_buffer, -20.0, 20.0))
    row["aggregate_credit_pressure"] = float(np.clip(borrowing + 0.10 * portfolio_liquid, -5.0, 5.0))
    row["aggregate_labor_pressure"] = float(np.clip(job_search - 0.06 * consumption, -3.0, 6.0))
    row["feedback_inflation_impulse"] = float(np.clip(0.035 * row["aggregate_demand_pressure"], -0.75, 0.75))
    row["feedback_unemployment_impulse"] = float(np.clip(0.040 * row["aggregate_labor_pressure"] - 0.015 * row["aggregate_demand_pressure"], -0.75, 0.75))
    row["feedback_credit_tightness_impulse"] = float(np.clip(0.050 * row["aggregate_credit_pressure"], -0.20, 0.20))
    return row


def build_module_ablations(
    panel: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        profile = {column: row[column] for column in RESPONDENT_COLUMNS if column not in {"respondent_id", "survey_source", "survey_date", "weight"}}
        environment = {column: float(row[column]) for column in ENVIRONMENT_COLUMNS}
        for target in target_fields:
            prior = float(row[f"prior_{target}"])
            profile_anchor = _profile_anchor(profile, target)
            environment_anchor = _environment_anchor(environment, target)
            controls = {
                "ablation_prior_only": prior,
                "ablation_profile_only": profile_anchor,
                "ablation_environment_only": environment_anchor,
                "ablation_profile_prior_no_environment": 0.75 * prior + 0.25 * profile_anchor,
                "ablation_rule_full_no_llm": 0.60 * prior + 0.18 * profile_anchor + 0.22 * environment_anchor,
            }
            for source, prediction in controls.items():
                spec = TARGET_SPECS[target]
                rows.append(
                    {
                        "schema_version": PERSONA_ECOLOGY_VERSION,
                        "panel_row_id": row["panel_row_id"],
                        "respondent_id": row["respondent_id"],
                        "period_id": row["period_id"],
                        "period_index": int(row["period_index"]),
                        "source": source,
                        "target_name": target,
                        "prediction": float(np.clip(prediction, spec["lower"], spec["upper"])),
                    }
                )
    return pd.DataFrame(rows)


def score_module_ablations(
    panel: pd.DataFrame,
    ablation_predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if ablation_predictions.empty:
        return pd.DataFrame()
    joined = ablation_predictions.merge(
        panel[["panel_row_id", "weight", *[f"actual_{target}" for target in target_fields]]],
        on="panel_row_id",
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name"], dropna=False):
        source, target = keys
        rows.append(_score_prediction_group(group, source=str(source), target_name=str(target), actual_column=f"actual_{target}"))
    for source, group in joined.groupby("source", dropna=False):
        pieces = []
        for target in target_fields:
            target_group = group[group["target_name"] == target].copy()
            if target_group.empty:
                continue
            target_group["actual"] = target_group[f"actual_{target}"]
            pieces.append(target_group)
        if pieces:
            rows.append(_score_prediction_group(pd.concat(pieces, ignore_index=True), source=str(source), target_name="ALL", actual_column="actual"))
    return pd.DataFrame(rows).sort_values(["target_name", "rmse", "source"]).reset_index(drop=True)


def score_temporal_dynamics(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if predictions.empty or panel["period_index"].nunique() < 2:
        return pd.DataFrame()
    joined = predictions.merge(
        panel[["panel_row_id", "respondent_id", "period_index", "weight", *[f"actual_{target}" for target in target_fields]]],
        on=["panel_row_id", "respondent_id", "period_index"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name"], dropna=False):
        source, target = keys
        target_group = group.sort_values(["respondent_id", "period_index"]).copy()
        target_group["actual"] = target_group[f"actual_{target}"]
        target_group["delta_actual"] = target_group.groupby("respondent_id")["actual"].diff()
        target_group["delta_prediction"] = target_group.groupby("respondent_id")["prediction"].diff()
        clean = target_group.dropna(subset=["delta_actual", "delta_prediction"])
        if clean.empty:
            continue
        errors = clean["delta_prediction"] - clean["delta_actual"]
        rows.append(
            {
                "source": source,
                "target_name": target,
                "n_deltas": int(clean.shape[0]),
                "delta_rmse": float(np.sqrt(_weighted_mean(errors**2, clean["weight"]))),
                "delta_mae": _weighted_mean(errors.abs(), clean["weight"]),
                "delta_bias": _weighted_mean(errors, clean["weight"]),
                "delta_correlation": _safe_corr(clean["delta_prediction"], clean["delta_actual"]),
                "mean_abs_simulated_update": _weighted_mean(clean["delta_prediction"].abs(), clean["weight"]),
                "mean_abs_actual_update": _weighted_mean(clean["delta_actual"].abs(), clean["weight"]),
                "update_amplitude_ratio": _weighted_mean(clean["delta_prediction"].abs(), clean["weight"])
                / max(_weighted_mean(clean["delta_actual"].abs(), clean["weight"]), 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_name", "delta_rmse", "source"]).reset_index(drop=True)


def score_period_levels(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    joined = predictions.merge(
        panel[["panel_row_id", "respondent_id", "period_id", "period_index", "weight", *[f"actual_{target}" for target in target_fields]]],
        on=["panel_row_id", "respondent_id", "period_id", "period_index"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name", "period_index", "period_id"], dropna=False):
        source, target, period_index, period_id = keys
        actual_column = f"actual_{target}"
        clean = group.dropna(subset=["prediction", actual_column]).copy()
        if clean.empty:
            continue
        errors = clean["prediction"].astype(float) - clean[actual_column].astype(float)
        actual_std = _weighted_std(clean[actual_column], clean["weight"])
        prediction_std = _weighted_std(clean["prediction"], clean["weight"])
        rows.append(
            {
                "source": source,
                "target_name": target,
                "period_index": int(period_index),
                "period_id": str(period_id),
                "n": int(clean.shape[0]),
                "actual_mean": _weighted_mean(clean[actual_column], clean["weight"]),
                "prediction_mean": _weighted_mean(clean["prediction"], clean["weight"]),
                "bias": _weighted_mean(errors, clean["weight"]),
                "mae": _weighted_mean(errors.abs(), clean["weight"]),
                "rmse": float(np.sqrt(_weighted_mean(errors**2, clean["weight"]))),
                "correlation": _safe_corr(clean["prediction"], clean[actual_column]),
                "actual_std": actual_std,
                "prediction_std": prediction_std,
                "std_ratio": prediction_std / actual_std if np.isfinite(actual_std) and actual_std > 1e-9 else np.nan,
                "mean_p10": _weighted_mean(clean["p10"], clean["weight"]) if "p10" in clean else np.nan,
                "mean_p90": _weighted_mean(clean["p90"], clean["weight"]) if "p90" in clean else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_name", "period_index", "rmse", "source"]).reset_index(drop=True)


def score_period_updates(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if predictions.empty or panel["period_index"].nunique() < 2:
        return pd.DataFrame()
    joined = predictions.merge(
        panel[["panel_row_id", "respondent_id", "period_id", "period_index", "weight", *[f"actual_{target}" for target in target_fields]]],
        on=["panel_row_id", "respondent_id", "period_id", "period_index"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name"], dropna=False):
        source, target = keys
        actual_column = f"actual_{target}"
        ordered = group.sort_values(["respondent_id", "period_index"]).copy()
        ordered["actual"] = ordered[actual_column]
        ordered["previous_period_id"] = ordered.groupby("respondent_id")["period_id"].shift(1)
        ordered["delta_actual"] = ordered.groupby("respondent_id")["actual"].diff()
        ordered["delta_prediction"] = ordered.groupby("respondent_id")["prediction"].diff()
        clean = ordered.dropna(subset=["delta_actual", "delta_prediction", "previous_period_id"]).copy()
        for period_keys, period_group in clean.groupby(["period_index", "previous_period_id", "period_id"], dropna=False):
            period_index, previous_period_id, period_id = period_keys
            errors = period_group["delta_prediction"].astype(float) - period_group["delta_actual"].astype(float)
            actual_abs = _weighted_mean(period_group["delta_actual"].abs(), period_group["weight"])
            prediction_abs = _weighted_mean(period_group["delta_prediction"].abs(), period_group["weight"])
            sign_match = (
                np.sign(period_group["delta_prediction"].astype(float)).eq(np.sign(period_group["delta_actual"].astype(float)))
            )
            rows.append(
                {
                    "source": source,
                    "target_name": target,
                    "period_index": int(period_index),
                    "from_period_id": str(previous_period_id),
                    "to_period_id": str(period_id),
                    "n_deltas": int(period_group.shape[0]),
                    "actual_delta_mean": _weighted_mean(period_group["delta_actual"], period_group["weight"]),
                    "prediction_delta_mean": _weighted_mean(period_group["delta_prediction"], period_group["weight"]),
                    "delta_bias": _weighted_mean(errors, period_group["weight"]),
                    "delta_mae": _weighted_mean(errors.abs(), period_group["weight"]),
                    "delta_rmse": float(np.sqrt(_weighted_mean(errors**2, period_group["weight"]))),
                    "delta_correlation": _safe_corr(period_group["delta_prediction"], period_group["delta_actual"]),
                    "direction_accuracy": _weighted_mean(sign_match.astype(float), period_group["weight"]),
                    "mean_abs_actual_update": actual_abs,
                    "mean_abs_simulated_update": prediction_abs,
                    "update_amplitude_ratio": prediction_abs / max(actual_abs, 1e-9),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_name", "period_index", "delta_rmse", "source"]).reset_index(drop=True)


def score_prior_update_dynamics(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    columns = [
        "panel_row_id",
        "respondent_id",
        "period_id",
        "period_index",
        "weight",
        *[f"prior_{target}" for target in target_fields],
        *[f"actual_{target}" for target in target_fields],
    ]
    joined = predictions.merge(
        panel[columns],
        on=["panel_row_id", "respondent_id", "period_id", "period_index"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name"], dropna=False):
        source, target = keys
        prior_column = f"prior_{target}"
        actual_column = f"actual_{target}"
        clean = group.dropna(subset=["prediction", "prior_prediction", prior_column, actual_column]).copy()
        if clean.empty:
            continue
        clean["prior"] = pd.to_numeric(clean["prior_prediction"], errors="coerce")
        clean["actual_update"] = pd.to_numeric(clean[actual_column], errors="coerce") - clean["prior"]
        clean["predicted_update"] = pd.to_numeric(clean["prediction"], errors="coerce") - clean["prior"]
        clean = clean.dropna(subset=["actual_update", "predicted_update", "weight"])
        if clean.empty:
            continue
        errors = clean["predicted_update"] - clean["actual_update"]
        persistence_errors = -clean["actual_update"]
        actual_abs = _weighted_mean(clean["actual_update"].abs(), clean["weight"])
        predicted_abs = _weighted_mean(clean["predicted_update"].abs(), clean["weight"])
        rmse = float(np.sqrt(_weighted_mean(errors**2, clean["weight"])))
        persistence_rmse = float(np.sqrt(_weighted_mean(persistence_errors**2, clean["weight"])))
        nonzero = clean["actual_update"].abs().gt(1e-9)
        direction_accuracy = _weighted_mean(
            np.sign(clean.loc[nonzero, "predicted_update"]).eq(np.sign(clean.loc[nonzero, "actual_update"])).astype(float),
            clean.loc[nonzero, "weight"],
        ) if nonzero.any() else np.nan
        rows.append(
            {
                "source": source,
                "target_name": target,
                "n_updates": int(clean.shape[0]),
                "actual_update_mean": _weighted_mean(clean["actual_update"], clean["weight"]),
                "predicted_update_mean": _weighted_mean(clean["predicted_update"], clean["weight"]),
                "update_bias": _weighted_mean(errors, clean["weight"]),
                "update_mae": _weighted_mean(errors.abs(), clean["weight"]),
                "update_rmse": rmse,
                "persistence_rmse": persistence_rmse,
                "rmse_improvement_vs_persistence": (persistence_rmse - rmse) / max(persistence_rmse, 1e-9),
                "update_correlation": _safe_corr(clean["predicted_update"], clean["actual_update"]),
                "direction_accuracy": direction_accuracy,
                "mean_abs_actual_update": actual_abs,
                "mean_abs_predicted_update": predicted_abs,
                "update_amplitude_ratio": predicted_abs / max(actual_abs, 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_name", "update_rmse", "source"]).reset_index(drop=True)


def score_behavior_actions(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    available = _available_behavior_targets(panel)
    if actions.empty or not available:
        return pd.DataFrame()
    joined = actions.merge(
        panel[["panel_row_id", *available]],
        on="panel_row_id",
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source"], dropna=False):
        source = str(keys[0] if isinstance(keys, tuple) else keys)
        all_pieces = []
        for actual_column in available:
            action_column = actual_column.replace("actual_", "", 1)
            clean = group.dropna(subset=[actual_column, action_column])
            if clean.empty:
                continue
            action_errors = clean[action_column] - clean[actual_column]
            rows.append(
                {
                    "source": source,
                    "action_name": action_column,
                    "n": int(clean.shape[0]),
                    "rmse": float(np.sqrt(_weighted_mean(action_errors**2, clean["weight"]))),
                    "mae": _weighted_mean(action_errors.abs(), clean["weight"]),
                    "bias": _weighted_mean(action_errors, clean["weight"]),
                    "correlation": _safe_corr(clean[action_column], clean[actual_column]),
                }
            )
            piece = clean[[action_column, actual_column, "weight"]].rename(
                columns={action_column: "prediction", actual_column: "actual"}
            )
            all_pieces.append(piece)
        if all_pieces:
            all_clean = pd.concat(all_pieces, ignore_index=True)
            errors = all_clean["prediction"] - all_clean["actual"]
            rows.append(
                {
                    "source": source,
                    "action_name": "ALL",
                    "n": int(all_clean.shape[0]),
                    "rmse": float(np.sqrt(_weighted_mean(errors**2, all_clean["weight"]))),
                    "mae": _weighted_mean(errors.abs(), all_clean["weight"]),
                    "bias": _weighted_mean(errors, all_clean["weight"]),
                    "correlation": _safe_corr(all_clean["prediction"], all_clean["actual"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["action_name", "rmse", "source"]).reset_index(drop=True)


def summarize_period_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in actions.groupby(["source", "period_index", "period_id"], dropna=False):
        source, period_index, period_id = keys
        row: dict[str, Any] = {
            "source": source,
            "period_index": int(period_index),
            "period_id": str(period_id),
            "n": int(group.shape[0]),
            "weight_sum": float(pd.to_numeric(group["weight"], errors="coerce").fillna(0.0).sum()),
        }
        for column in ACTION_COLUMNS:
            row[f"mean_{column}"] = _weighted_mean(group[column], group["weight"])
            row[f"std_{column}"] = _weighted_std(group[column], group["weight"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["period_index", "source"]).reset_index(drop=True)


def classify_ecology_evidence(
    static_evidence: dict[str, Any],
    temporal_scores: pd.DataFrame,
    behavior_scores: pd.DataFrame,
    environments: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    period_count = int(manifest.get("period_count", 0))
    dynamic_tested = period_count >= 2 and not temporal_scores.empty
    behavior_tested = not behavior_scores.empty
    behavior_target_source = str(manifest.get("behavior_target_source", "unavailable"))
    external_behavior_tested = behavior_tested and behavior_target_source == "external_csv_targets"
    closed_loop_tested = manifest.get("feedback_mode") == "closed_loop" and not environments.empty
    mean_delta_corr = float(temporal_scores["delta_correlation"].dropna().mean()) if dynamic_tested else np.nan
    best_behavior_rmse = float(
        behavior_scores[behavior_scores["action_name"] == "ALL"]["rmse"].min()
    ) if behavior_tested and "action_name" in behavior_scores else np.nan
    feedback_moved = bool(
        closed_loop_tested
        and "aggregate_demand_pressure" in environments
        and float(environments["aggregate_demand_pressure"].abs().mean()) > 1e-9
    )
    if manifest.get("respondent_source") == "fixture":
        verdict = "fixture_ecology_harness_ready"
        branch = "synthetic_panel_exercises_dynamic_stack"
    elif not dynamic_tested:
        verdict = "incomplete_dynamic_panel"
        branch = "need_at_least_two_periods"
    elif not external_behavior_tested:
        verdict = "belief_ecology_ready_behavior_unscored"
        branch = "dynamic_beliefs_measured_but_external_behavior_targets_missing"
    elif mean_delta_corr >= 0.20 and feedback_moved:
        verdict = "clears_dynamic_behavior_setup_gate"
        branch = "dynamic_beliefs_behavior_and_feedback_measured"
    else:
        verdict = "dynamic_gate_not_cleared"
        branch = "temporal_or_behavior_fit_weak"
    return {
        "evidence_verdict": verdict,
        "decision_tree_branch": branch,
        "static_evidence_verdict": static_evidence.get("evidence_verdict"),
        "dynamic_tested": dynamic_tested,
        "behavior_tested": behavior_tested,
        "external_behavior_tested": external_behavior_tested,
        "behavior_target_source": behavior_target_source,
        "closed_loop_tested": closed_loop_tested,
        "feedback_moved": feedback_moved,
        "mean_delta_correlation": mean_delta_corr,
        "best_behavior_rmse": best_behavior_rmse,
    }


def classify_prior_update_evidence(prior_update_scores: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    primary_source = str(manifest.get("primary_update_source") or "")
    thresholds = dict(PRIOR_UPDATE_THRESHOLDS)
    if str(manifest.get("ecology_mode")) == "fixture":
        return {
            "evidence_verdict": "prior_update_fixture_ready",
            "decision_tree_branch": "fixture_scores_exercise_prior_update_harness_only",
            "primary_update_source": primary_source,
            "thresholds": thresholds,
        }
    if str(manifest.get("prior_mode")) != "empirical":
        return {
            "evidence_verdict": "prior_update_unscored",
            "decision_tree_branch": "prior_mode_not_empirical",
            "primary_update_source": primary_source,
            "thresholds": thresholds,
        }
    if prior_update_scores.empty:
        return {
            "evidence_verdict": "prior_update_incomplete",
            "decision_tree_branch": "no_prior_update_scores",
            "primary_update_source": primary_source,
            "thresholds": thresholds,
        }
    primary = prior_update_scores[prior_update_scores["source"].astype(str) == primary_source].copy()
    if primary.empty:
        return {
            "evidence_verdict": "prior_update_incomplete",
            "decision_tree_branch": "primary_source_missing",
            "primary_update_source": primary_source,
            "thresholds": thresholds,
        }
    mean_corr = float(primary["update_correlation"].dropna().mean()) if primary["update_correlation"].notna().any() else np.nan
    mean_direction = float(primary["direction_accuracy"].dropna().mean()) if primary["direction_accuracy"].notna().any() else np.nan
    median_amplitude = (
        float(primary["update_amplitude_ratio"].replace([np.inf, -np.inf], np.nan).dropna().median())
        if primary["update_amplitude_ratio"].replace([np.inf, -np.inf], np.nan).notna().any()
        else np.nan
    )
    mean_rmse_gain = (
        float(primary["rmse_improvement_vs_persistence"].dropna().mean())
        if primary["rmse_improvement_vs_persistence"].notna().any()
        else np.nan
    )
    corr_clear = bool(np.isfinite(mean_corr) and mean_corr >= thresholds["primary_source_mean_delta_correlation_min"])
    direction_clear = bool(
        np.isfinite(mean_direction)
        and mean_direction >= thresholds["primary_source_mean_direction_accuracy_min"]
    )
    amplitude_clear = bool(
        np.isfinite(median_amplitude)
        and thresholds["primary_source_median_update_amplitude_ratio_min"]
        <= median_amplitude
        <= thresholds["primary_source_median_update_amplitude_ratio_max"]
    )
    persistence_clear = bool(
        np.isfinite(mean_rmse_gain)
        and mean_rmse_gain >= thresholds["primary_source_mean_rmse_improvement_vs_persistence_min"]
    )
    if corr_clear and direction_clear and amplitude_clear and persistence_clear:
        verdict = "clears_prior_update_gate"
        branch = "primary_source_updates_beat_persistence_with_direction_and_amplitude"
    else:
        verdict = "prior_update_gate_not_cleared"
        branch = "correlation_direction_amplitude_or_persistence_gain_weak"
    return {
        "evidence_verdict": verdict,
        "decision_tree_branch": branch,
        "primary_update_source": primary_source,
        "thresholds": thresholds,
        "primary_source_mean_update_correlation": mean_corr,
        "primary_source_mean_direction_accuracy": mean_direction,
        "primary_source_median_update_amplitude_ratio": median_amplitude,
        "primary_source_mean_rmse_improvement_vs_persistence": mean_rmse_gain,
        "correlation_clear": corr_clear,
        "direction_clear": direction_clear,
        "amplitude_clear": amplitude_clear,
        "persistence_clear": persistence_clear,
    }


def build_persona_ecology_report(
    manifest: dict[str, Any],
    regression_scores: pd.DataFrame,
    variance_scores: pd.DataFrame,
    distribution_scores: pd.DataFrame,
    common_core: pd.DataFrame,
    temporal_scores: pd.DataFrame,
    behavior_scores: pd.DataFrame,
    ablation_scores: pd.DataFrame,
    environments: pd.DataFrame,
    *,
    period_scores: pd.DataFrame | None = None,
    update_scores: pd.DataFrame | None = None,
    prior_update_scores: pd.DataFrame | None = None,
    action_period_summary: pd.DataFrame | None = None,
) -> str:
    period_scores = period_scores if period_scores is not None else pd.DataFrame()
    update_scores = update_scores if update_scores is not None else pd.DataFrame()
    prior_update_scores = prior_update_scores if prior_update_scores is not None else pd.DataFrame()
    action_period_summary = action_period_summary if action_period_summary is not None else pd.DataFrame()
    ecology = manifest.get("ecology_evidence", {})
    prior_update = manifest.get("prior_update_evidence", {})
    verdict = ecology.get("evidence_verdict", "unknown")
    if verdict == "fixture_ecology_harness_ready":
        bottom_line = (
            "The fixture run exercises the full respondent-seeded ecology: profile, priors, environment, "
            "belief updates, behavior responses, aggregate feedback, temporal scoring, and module ablations. "
            "It is harness readiness, not empirical evidence."
        )
    elif verdict == "clears_dynamic_behavior_setup_gate":
        bottom_line = (
            "The data-backed panel clears the dynamic setup gate: beliefs evolve over multiple periods, "
            "behavior targets are scored, and closed-loop aggregate feedback is active."
        )
    elif verdict == "belief_ecology_ready_behavior_unscored":
        bottom_line = (
            "The dynamic belief ecology ran on data, but behavior targets were unavailable. "
            "The next gate is direct behavior microdata."
        )
    else:
        bottom_line = f"The ecology ran with verdict `{verdict}` on branch `{ecology.get('decision_tree_branch', 'unknown')}`."
    lines = [
        "# Persona Belief Ecology",
        "",
        "## Bottom Line",
        bottom_line,
        "",
        "## Run Setup",
        f"- Provider: `{manifest.get('provider')}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Ecology mode: `{manifest.get('ecology_mode')}`",
        f"- Respondents: `{manifest.get('respondent_count')}`",
        f"- Panel rows: `{manifest.get('panel_row_count')}` across `{manifest.get('period_count')}` periods",
        f"- Prior mode: `{manifest.get('prior_mode')}`",
        f"- Feedback mode: `{manifest.get('feedback_mode')}`",
        f"- Date mode: `{manifest.get('date_mode', 'actual')}`",
        f"- Live calls used: `{manifest.get('live_call_count', 0)}` of cap `{manifest.get('max_live_calls', 0)}`",
        f"- Evidence verdict: `{verdict}`",
        f"- Dynamic tested: `{ecology.get('dynamic_tested')}`",
        f"- Behavior tested: `{ecology.get('behavior_tested')}`",
        f"- Behavior target source: `{ecology.get('behavior_target_source', manifest.get('behavior_target_source'))}`",
        f"- External behavior tested: `{ecology.get('external_behavior_tested')}`",
        f"- Feedback moved: `{ecology.get('feedback_moved')}`",
        f"- Prior-update verdict: `{prior_update.get('evidence_verdict')}`",
        f"- Primary update source: `{manifest.get('primary_update_source')}`",
        f"- Pre-registered prior-update thresholds: `{json.dumps(manifest.get('pre_registered_prior_update_thresholds', {}), sort_keys=True)}`",
        "",
        "## Spine",
        (
            "This runner follows the survey-agent spine from the Alex/paper discussion: stable respondent "
            "profile, prior expectations, dynamic external information, explicit belief-update weights, "
            "behavior choices, and aggregate feedback into the next period. Prompts never include held-out "
            "current responses."
        ),
        "",
        "## Heterogeneity Regression Scores",
        markdown_table(regression_scores.head(48)),
        "",
        "## Variance / Flattening",
        markdown_table(variance_scores.head(48)),
        "",
        "## Distribution Distance",
        markdown_table(distribution_scores.head(48)),
        "",
        "## Common Core",
        markdown_table(common_core),
        "",
        "## Temporal Dynamics",
        markdown_table(temporal_scores.head(48)),
        "",
        "## Period-Level Scores",
        markdown_table(period_scores.head(72)),
        "",
        "## Period Update Scores",
        markdown_table(update_scores.head(72)),
        "",
        "## Prior-Conditioned Update Scores",
        (
            "These rows score the economically relevant update: predicted current belief minus the supplied "
            "empirical prior, compared with actual current survey response minus that same prior. Persistence "
            "is the baseline where the respondent does not update from the prior."
        ),
        markdown_table(prior_update_scores.head(72)),
        "",
        "## Behavior Scores",
        (
            "Fixture behavior scores are synthetic self-consistency checks. They become behavior evidence only "
            "when `behavior_target_source` is `external_csv_targets`."
        ),
        markdown_table(behavior_scores.head(48)),
        "",
        "## Action Period Summary",
        markdown_table(action_period_summary.head(72)),
        "",
        "## Module Ablations",
        markdown_table(ablation_scores.head(48)),
        "",
        "## Environment Feedback",
        markdown_table(environments.head(24)),
        "",
        "## Manifest",
        "```json",
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def _action_row(
    card: EcologyCard,
    payload: dict[str, Any],
    *,
    source: str,
    environment: dict[str, float],
    prior_beliefs: dict[str, float],
) -> dict[str, Any]:
    actions = payload["actions"]
    row = {
        "schema_version": PERSONA_ECOLOGY_VERSION,
        "panel_row_id": card.panel_row_id,
        "respondent_id": card.respondent_id,
        "period_id": card.period_id,
        "period_index": card.period_index,
        "survey_date": card.survey_date,
        "source": source,
        "weight": card.weight,
        "confidence": payload["confidence"],
        "uncertainty": payload["uncertainty"],
        "call_source": payload["call_source"],
        "cache_hit": payload["cache_hit"],
        **{f"prior_{target}": prior_beliefs[target] for target in DEFAULT_TARGET_FIELDS if target in prior_beliefs},
        **{f"seen_{key}": value for key, value in environment.items()},
    }
    row.update(actions)
    row.update(payload["module_weights"])
    return row


def _scoring_frames(panel: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_panel = panel.copy()
    scoring_panel["original_respondent_id"] = scoring_panel["respondent_id"]
    scoring_panel["respondent_id"] = scoring_panel["panel_row_id"]
    scoring_predictions = predictions.copy()
    scoring_predictions["original_respondent_id"] = scoring_predictions["respondent_id"]
    scoring_predictions["respondent_id"] = scoring_predictions["panel_row_id"]
    return scoring_panel, scoring_predictions


def _available_behavior_targets(panel: pd.DataFrame) -> list[str]:
    return [
        column
        for column in OPTIONAL_BEHAVIOR_TARGET_COLUMNS
        if column in panel and panel[column].notna().any()
    ]


def _limit_ecology_respondents(frame: pd.DataFrame, respondent_limit: int) -> pd.DataFrame:
    respondent_count = int(frame["respondent_id"].nunique()) if "respondent_id" in frame else 0
    if respondent_limit <= 0 or respondent_count <= respondent_limit:
        return frame.reset_index(drop=True)
    keep = sorted(frame["respondent_id"].astype(str).unique())[:respondent_limit]
    out = frame[frame["respondent_id"].astype(str).isin(keep)].copy()
    return _normalize_period_weights(out).reset_index(drop=True)


def _parse_period_ids(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_sample_strata(value: str) -> tuple[str, ...]:
    strata = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not strata:
        raise ValueError("--respondent-sample-strata must contain at least one column")
    return strata


def _select_ecology_panel(
    frame: pd.DataFrame,
    *,
    respondent_limit: int,
    respondent_sample_size: int,
    respondent_sample_seed: int,
    sample_strata: tuple[str, ...],
    period_ids: tuple[str, ...],
    require_complete_periods: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = frame.copy()
    if period_ids:
        selected = selected[selected["period_id"].astype(str).isin(period_ids)].copy()
        if selected.empty:
            raise ValueError(f"No panel rows match --period-ids: {', '.join(period_ids)}")
    if require_complete_periods:
        required = set(period_ids) if period_ids else set(selected["period_id"].astype(str).unique())
        counts = selected.groupby("respondent_id")["period_id"].apply(lambda series: set(series.astype(str)))
        keep = counts[counts.map(lambda values: required.issubset(values))].index
        selected = selected[selected["respondent_id"].isin(keep)].copy()
        if selected.empty:
            raise ValueError("No respondents have complete coverage for the selected periods")
    pre_sample = selected.copy()
    if respondent_sample_size > 0:
        selected, sample_manifest = _stratified_ecology_sample(
            selected,
            sample_size=respondent_sample_size,
            seed=respondent_sample_seed,
            strata=sample_strata,
        )
    else:
        selected = _limit_ecology_respondents(selected, respondent_limit)
        sample_manifest = {
            "sampling_strategy": "head_limit" if respondent_limit > 0 and selected["respondent_id"].nunique() < pre_sample["respondent_id"].nunique() else "none",
            "respondent_limit": int(respondent_limit),
            "sample_size_requested": 0,
            "sample_size_actual": int(selected["respondent_id"].nunique()),
            "sample_seed": None,
            "sample_strata": list(sample_strata),
            "stratum_count": int(_respondent_strata_frame(selected, sample_strata).shape[0]) if not selected.empty else 0,
            "stratum_counts": _ecology_stratum_count_records(pre_sample, selected, sample_strata) if not selected.empty else [],
        }
    manifest = {
        **sample_manifest,
        "period_ids_filter": list(period_ids),
        "require_complete_periods": bool(require_complete_periods),
        "pre_selection_row_count": int(frame.shape[0]),
        "pre_selection_respondent_count": int(frame["respondent_id"].nunique()),
        "post_period_filter_row_count": int(pre_sample.shape[0]),
        "post_period_filter_respondent_count": int(pre_sample["respondent_id"].nunique()),
        "selected_row_count": int(selected.shape[0]),
        "selected_respondent_count": int(selected["respondent_id"].nunique()),
        "selected_period_count": int(selected["period_index"].nunique()),
        "weights_retained": True,
        "weights_renormalized_after_selection": True,
    }
    return _normalize_period_weights(selected).reset_index(drop=True), manifest


def _stratified_ecology_sample(
    frame: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    strata: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    missing = [column for column in strata if column not in frame]
    if missing:
        raise ValueError(f"Cannot stratify ecology respondents; missing columns: {', '.join(missing)}")
    respondent_frame = _respondent_strata_frame(frame, strata)
    respondent_count = int(respondent_frame.shape[0])
    if sample_size >= respondent_count:
        out = _normalize_period_weights(frame.copy())
        return out.reset_index(drop=True), {
            "sampling_strategy": "all_respondents",
            "sample_size_requested": int(sample_size),
            "sample_size_actual": respondent_count,
            "sample_seed": int(seed),
            "sample_strata": list(strata),
            "stratum_count": int(respondent_frame.groupby(list(strata), dropna=False).ngroups),
            "stratum_counts": _ecology_stratum_count_records(frame, out, strata),
        }
    groups = list(respondent_frame.groupby(list(strata), dropna=False, sort=True))
    if sample_size < len(groups):
        raise ValueError(
            f"Stratified sample size {sample_size} is smaller than the {len(groups)} non-empty strata; "
            "increase --respondent-sample-size or reduce --respondent-sample-strata"
        )
    allocation = _proportional_stratum_allocation(groups, sample_size=sample_size, population_size=respondent_count)
    rng = np.random.default_rng(seed)
    chosen_ids: list[str] = []
    for (_, group), count in zip(groups, allocation):
        if count <= 0:
            continue
        chosen_ids.extend(rng.choice(group["respondent_id"].astype(str).to_numpy(), size=int(count), replace=False).tolist())
    selected = frame[frame["respondent_id"].astype(str).isin(chosen_ids)].copy()
    selected = _normalize_period_weights(selected)
    return selected.reset_index(drop=True), {
        "sampling_strategy": "stratified_respondents_without_replacement",
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(selected["respondent_id"].nunique()),
        "sample_seed": int(seed),
        "sample_strata": list(strata),
        "stratum_count": int(len(groups)),
        "stratum_counts": _ecology_stratum_count_records(frame, selected, strata),
    }


def _respondent_strata_frame(frame: pd.DataFrame, strata: tuple[str, ...]) -> pd.DataFrame:
    columns = ["respondent_id", *strata]
    return frame.sort_values(["respondent_id", "period_index"]).drop_duplicates("respondent_id")[columns].copy()


def _proportional_stratum_allocation(
    groups: list[tuple[Any, pd.DataFrame]],
    *,
    sample_size: int,
    population_size: int,
) -> list[int]:
    exact = [group.shape[0] * sample_size / population_size for _, group in groups]
    allocation = [max(1, min(group.shape[0], int(np.floor(value)))) for value, (_, group) in zip(exact, groups)]
    remaining = sample_size - sum(allocation)
    order = sorted(
        range(len(groups)),
        key=lambda idx: (exact[idx] - np.floor(exact[idx]), groups[idx][1].shape[0], str(groups[idx][0])),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for idx in order:
            capacity = groups[idx][1].shape[0] - allocation[idx]
            if capacity <= 0:
                continue
            allocation[idx] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    while remaining < 0:
        progressed = False
        for idx in reversed(order):
            if allocation[idx] <= 1:
                continue
            allocation[idx] -= 1
            remaining += 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    if sum(allocation) != sample_size:
        raise ValueError(f"Could not allocate exact stratified sample size {sample_size}; allocated {sum(allocation)}")
    return allocation


def _ecology_stratum_count_records(population: pd.DataFrame, sampled: pd.DataFrame, strata: tuple[str, ...]) -> list[dict[str, Any]]:
    population_respondents = _respondent_strata_frame(population, strata)
    sampled_respondents = _respondent_strata_frame(sampled, strata) if not sampled.empty else pd.DataFrame(columns=["respondent_id", *strata])
    population_counts = population_respondents.groupby(list(strata), dropna=False).agg(
        population_respondent_count=("respondent_id", "size"),
    )
    sampled_counts = sampled_respondents.groupby(list(strata), dropna=False).agg(
        sampled_respondent_count=("respondent_id", "size"),
    )
    joined = population_counts.join(sampled_counts, how="left").fillna({"sampled_respondent_count": 0})
    records: list[dict[str, Any]] = []
    for _, row in joined.reset_index().iterrows():
        record = {column: row[column] for column in strata}
        record.update(
            {
                "population_respondent_count": int(row["population_respondent_count"]),
                "sampled_respondent_count": int(row["sampled_respondent_count"]),
            }
        )
        records.append(record)
    return records


def _anonymize_csv_panel_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ids = list(dict.fromkeys(out["respondent_id"].astype(str)))
    mapping = {respondent_id: f"respondent_{idx + 1:05d}" for idx, respondent_id in enumerate(ids)}
    out["respondent_id"] = out["respondent_id"].astype(str).map(mapping)
    out["panel_row_id"] = out["respondent_id"] + "__" + out["period_id"].astype(str)
    return out


def _respondent_input_manifest(
    *,
    source: str,
    respondent_csv: Path | None,
    normalized: pd.DataFrame,
    pre_selection_normalized: pd.DataFrame,
    respondent_limit: int,
    selection_manifest: dict[str, Any],
    anonymized: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": source,
        "respondent_limit": int(respondent_limit),
        "respondent_ids_anonymized": bool(anonymized),
        "pre_selection_normalized_row_count": int(pre_selection_normalized.shape[0]),
        "pre_selection_normalized_respondent_count": int(pre_selection_normalized["respondent_id"].nunique())
        if "respondent_id" in pre_selection_normalized
        else 0,
        "normalized_row_count": int(normalized.shape[0]),
        "normalized_respondent_count": int(normalized["respondent_id"].nunique()) if "respondent_id" in normalized else 0,
        "normalized_period_count": int(normalized["period_index"].nunique()) if "period_index" in normalized else 0,
        "normalized_weight_sum": round_or_none(normalized["weight"].sum()) if "weight" in normalized else None,
        "behavior_target_columns": _available_behavior_targets(normalized),
        "selection": selection_manifest,
    }
    if respondent_csv is not None:
        raw = pd.read_csv(respondent_csv)
        panel_kind = _raw_unique_value(raw, "persona_panel_kind")
        target_provenance = _raw_unique_value(raw, "target_provenance")
        environment_provenance = _raw_unique_value(raw, "environment_provenance")
        manifest.update(
            {
                "path": _safe_relative(respondent_csv),
                "sha256": _file_sha256(respondent_csv),
                "byte_size": int(respondent_csv.stat().st_size),
                "raw_row_count": int(raw.shape[0]),
                "raw_column_count": int(raw.shape[1]),
                "panel_kind": panel_kind,
                "target_provenance": target_provenance,
                "environment_provenance": environment_provenance,
                "synthetic_enriched": bool(
                    "synthetic" in str(panel_kind).lower() or "synthetic" in str(target_provenance).lower()
                ),
            }
        )
    return manifest


def _split_roles_manifest(respondent_input: dict[str, Any], *, period_ids: tuple[str, ...], prior_mode: str) -> dict[str, Any]:
    panel_kind = str(respondent_input.get("panel_kind") or "").lower()
    if panel_kind == "real_sce_microdata_v1" and prior_mode == "empirical":
        return {
            "current_run_surface": ",".join(period_ids) if period_ids else "selected_real_sce_panel",
            "current_run_role": "prior_conditioned_update_validation",
            "test_wave_status": "december_2024_static_wave_spent_not_used_here",
            "calibration_reserved_surface": "october_november_2024_panel_rows",
            "holdout_reuse_rule": "declare thresholds_before_live_calls_and_do_not_tune_on_update_results",
        }
    return {
        "current_run_surface": ",".join(period_ids) if period_ids else "unspecified",
        "current_run_role": "development_or_fixture",
        "test_wave_status": None,
        "holdout_reuse_rule": "not_applicable",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _behavior_target_source(panel: pd.DataFrame, *, respondent_source: str) -> str:
    if not _available_behavior_targets(panel):
        return "unavailable"
    if "target_provenance" in panel:
        values = " ".join(str(value).lower() for value in panel["target_provenance"].dropna().unique())
        if "synthetic" in values:
            return "synthetic_enriched_targets"
    if "persona_panel_kind" in panel:
        values = " ".join(str(value).lower() for value in panel["persona_panel_kind"].dropna().unique())
        if "synthetic" in values:
            return "synthetic_enriched_targets"
    if respondent_source == "fixture":
        return "synthetic_fixture_targets"
    return "external_csv_targets"


def _isolated_provider_cwd(output_dir: Path, *, provider: str, source: str, mode: str) -> Path | None:
    if source != "csv" or mode != "live" or provider not in {"codex_cli", "gemini_cli", "antigravity_cli"}:
        return None
    root = Path(tempfile.gettempdir()) / "macro_llm_tournament_provider_cwd"
    path = root / cache_key({"output_dir": str(output_dir.resolve()), "provider": provider})
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raw_unique_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = [str(value) for value in frame[column].dropna().unique()]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "mixed:" + ",".join(values[:5])


def _standardize_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    normalized = {_normalize_name(column): column for column in out.columns}
    for canonical, aliases in ALIASES.items():
        if canonical in out:
            continue
        source = next((normalized[_normalize_name(alias)] for alias in aliases if _normalize_name(alias) in normalized), None)
        if source is not None:
            out[canonical] = out[source]
    return out


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _coalesce_text(frame: pd.DataFrame, column: str, fallback: pd.Series) -> pd.Series:
    if fallback.empty:
        fallback = pd.Series("unknown", index=frame.index)
    if column not in frame:
        return fallback.fillna("unknown").astype(str)
    series = frame[column]
    return series.where(series.notna(), fallback).fillna("unknown").astype(str).map(_normalize_label)


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not text or text in {"nan", "none"}:
        return "unknown"
    replacements = {
        "college": "college_plus",
        "ba_plus": "college_plus",
        "bachelor_or_more": "college_plus",
        "high_school": "high_school_or_less",
        "hs_or_less": "high_school_or_less",
        "not_college": "high_school_or_less",
        "owner": "owner",
        "own": "owner",
        "renter": "renter",
        "rent": "renter",
    }
    return replacements.get(text, text)


def _age_group_from_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([numeric < 35, numeric < 55, numeric >= 55], ["18_34", "35_54", "55_plus"], default="unknown"),
        index=series.index,
    )


def _income_group_from_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 3:
        return pd.Series("unknown", index=series.index)
    q33 = float(numeric.quantile(0.33))
    q67 = float(numeric.quantile(0.67))
    return pd.Series(
        np.select([numeric <= q33, numeric >= q67], ["low", "high"], default="middle"),
        index=series.index,
    )


def _education_group_from_raw(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    text = series.astype(str).str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                text.str.contains("college|bachelor|ba|graduate|post", regex=True) | (numeric >= 16),
                text.str.contains("some|associate", regex=True) | ((numeric > 12) & (numeric < 16)),
                text.str.contains("high|hs|less", regex=True) | (numeric <= 12),
            ],
            ["college_plus", "some_college", "high_school_or_less"],
            default="unknown",
        ),
        index=series.index,
    )


def _gender_from_raw(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    text = series.astype(str).str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [text.str.startswith("f") | numeric.eq(1), text.str.startswith("m") | numeric.eq(0)],
            ["female", "male"],
            default="unknown",
        ),
        index=series.index,
    )


def _normalize_period_weights(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    totals = out.groupby("period_index")["weight"].transform("sum")
    bad = totals <= 0
    if bad.any():
        out.loc[bad, "weight"] = 1.0
        totals = out.groupby("period_index")["weight"].transform("sum")
    out["weight"] = out["weight"] / totals
    return out


def _fill_missing_priors(frame: pd.DataFrame, *, target_fields: Iterable[str]) -> pd.DataFrame:
    out = frame.sort_values(["respondent_id", "period_index"]).copy()
    for target in target_fields:
        prior_column = f"prior_{target}"
        actual_column = f"actual_{target}"
        lag = out.groupby("respondent_id")[actual_column].shift(1)
        out[prior_column] = out[prior_column].where(out[prior_column].notna(), lag)
        missing = out[prior_column].isna()
        if missing.any():
            out.loc[missing, prior_column] = out.loc[missing].apply(
                lambda row: _profile_anchor(
                    {
                        column: row[column]
                        for column in RESPONDENT_COLUMNS
                        if column not in {"respondent_id", "survey_source", "survey_date", "weight"}
                    },
                    target,
                ),
                axis=1,
            )
    return out


def _fill_missing_environment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for period_index in sorted(out["period_index"].unique()):
        fixture = _fixture_environment(int(period_index))
        mask = out["period_index"].eq(period_index)
        for column, value in fixture.items():
            out.loc[mask, column] = out.loc[mask, column].fillna(value)
    return out


def _fixture_environment(period_index: int) -> dict[str, float]:
    inflation = 3.0 + 0.65 * np.sin(period_index / 1.7) + (0.9 if period_index == 1 else 0.0)
    unemployment = 4.5 + 0.30 * period_index + (0.8 if period_index == 2 else 0.0)
    real_income = 1.3 - 0.25 * period_index + (0.5 if period_index == 0 else 0.0)
    policy_rate = 3.2 + 0.35 * period_index
    sentiment = 86.0 - 3.5 * period_index - (4.0 if period_index == 2 else 0.0)
    return {
        "observed_inflation_1y": float(inflation),
        "observed_unemployment_rate": float(unemployment),
        "observed_real_income_growth": float(real_income),
        "policy_rate": float(policy_rate),
        "sentiment_index": float(sentiment),
        "news_inflation_pressure": float((inflation - 3.0) / 2.0),
        "news_labor_pressure": float((unemployment - 4.5) / 2.5),
        "credit_tightness": float(np.clip(0.35 + 0.06 * period_index, 0.0, 1.0)),
        "aggregate_expected_inflation_1y": float(inflation + 0.2),
        "aggregate_expected_unemployment_rate": float(unemployment + 0.1),
        "aggregate_expected_real_income_growth": float(real_income - 0.1),
    }


def _prior_for_card(
    card: EcologyCard,
    simulated_priors: dict[str, dict[str, float]],
    *,
    prior_mode: str,
    target_fields: Iterable[str],
) -> dict[str, float]:
    if prior_mode == "simulated" and card.respondent_id in simulated_priors:
        return {target: float(simulated_priors[card.respondent_id][target]) for target in target_fields}
    return {target: float(card.empirical_priors[target]) for target in target_fields}


def _environment_for_card(
    base_environment: dict[str, float],
    previous_aggregate: dict[str, Any] | None,
    *,
    feedback_mode: str,
) -> dict[str, float]:
    environment = dict(base_environment)
    if feedback_mode == "none" or previous_aggregate is None:
        return environment
    environment["aggregate_expected_inflation_1y"] = float(
        _finite_or_default(
            previous_aggregate.get("aggregate_expected_inflation_1y"),
            environment["aggregate_expected_inflation_1y"],
        )
    )
    environment["aggregate_expected_unemployment_rate"] = float(
        _finite_or_default(
            previous_aggregate.get("aggregate_expected_unemployment_rate"),
            environment["aggregate_expected_unemployment_rate"],
        )
    )
    environment["aggregate_expected_real_income_growth"] = float(
        _finite_or_default(
            previous_aggregate.get("aggregate_expected_real_income_growth"),
            environment["aggregate_expected_real_income_growth"],
        )
    )
    environment["observed_inflation_1y"] = float(
        environment["observed_inflation_1y"] + _finite_or_default(previous_aggregate.get("feedback_inflation_impulse"), 0.0)
    )
    environment["observed_unemployment_rate"] = float(
        environment["observed_unemployment_rate"] + _finite_or_default(previous_aggregate.get("feedback_unemployment_impulse"), 0.0)
    )
    environment["credit_tightness"] = float(
        np.clip(
            environment["credit_tightness"] + _finite_or_default(previous_aggregate.get("feedback_credit_tightness_impulse"), 0.0),
            0.0,
            1.0,
        )
    )
    environment["sentiment_index"] = float(
        np.clip(
            environment["sentiment_index"] + 0.20 * _finite_or_default(previous_aggregate.get("aggregate_consumption_change_pct"), 0.0),
            0.0,
            120.0,
        )
    )
    return environment


def _profile_anchor(profile: dict[str, Any], target: str) -> float:
    income = str(profile.get("income_group", "unknown"))
    education = str(profile.get("education_group", "unknown"))
    age = str(profile.get("age_group", "unknown"))
    gender = str(profile.get("gender", "unknown"))
    employment = str(profile.get("employment_status", "unknown"))
    liquid = str(profile.get("liquid_wealth_group", "unknown"))
    if target == "expected_inflation_1y":
        value = 3.0
        value += {"low": 1.0, "middle": 0.35, "high": -0.25}.get(income, 0.0)
        value += {"high_school_or_less": 0.45, "some_college": 0.15, "college_plus": -0.20}.get(education, 0.0)
        value += {"55_plus": 0.25, "35_54": 0.10, "18_34": -0.10}.get(age, 0.0)
        value += 0.10 if gender == "female" else 0.0
        value += 0.30 if liquid == "low" else 0.0
        return value
    if target == "expected_unemployment_rate":
        value = 4.8
        value += {"low": 0.70, "middle": 0.20, "high": -0.35}.get(income, 0.0)
        value += {"unemployed": 2.0, "not_in_labor_force": 0.6, "retired": 0.4}.get(employment, 0.0)
        value += 0.25 if liquid == "low" else 0.0
        return value
    if target == "expected_unemployment_higher_prob":
        value = 35.0
        value += {"low": 5.0, "middle": 1.0, "high": -2.5}.get(income, 0.0)
        value += {"unemployed": 12.0, "not_in_labor_force": 3.0, "retired": 1.5}.get(employment, 0.0)
        value += 2.0 if liquid == "low" else 0.0
        return value
    if target == "expected_real_income_growth":
        value = 1.0
        value += {"low": -0.45, "middle": 0.10, "high": 0.55}.get(income, 0.0)
        value += {"college_plus": 0.30, "some_college": 0.05, "high_school_or_less": -0.20}.get(education, 0.0)
        value += {"unemployed": -1.2, "retired": -0.4}.get(employment, 0.0)
        return value
    return INITIAL_PRIORS.get(target, 0.0)


def _environment_anchor(environment: dict[str, float], target: str) -> float:
    if target == "expected_inflation_1y":
        return float(
            0.62 * environment["observed_inflation_1y"]
            + 0.28 * environment["aggregate_expected_inflation_1y"]
            + 0.45 * environment["news_inflation_pressure"]
        )
    if target == "expected_unemployment_rate":
        return float(
            0.68 * environment["observed_unemployment_rate"]
            + 0.24 * environment["aggregate_expected_unemployment_rate"]
            + 0.70 * environment["news_labor_pressure"]
        )
    if target == "expected_unemployment_higher_prob":
        unemployment_anchor = float(
            35.0
            + 4.0 * (environment["observed_unemployment_rate"] - 4.5)
            + 9.0 * environment["news_labor_pressure"]
        )
        return float(np.clip(unemployment_anchor, 0.0, 100.0))
    if target == "expected_real_income_growth":
        sentiment_gap = (environment["sentiment_index"] - 80.0) / 20.0
        return float(
            0.72 * environment["observed_real_income_growth"]
            + 0.20 * environment["aggregate_expected_real_income_growth"]
            + 0.30 * sentiment_gap
            - 0.20 * environment["credit_tightness"]
        )
    return INITIAL_PRIORS.get(target, 0.0)


def _aggregate_anchor(environment: dict[str, float], target: str) -> float:
    mapping = {
        "expected_inflation_1y": "aggregate_expected_inflation_1y",
        "expected_unemployment_rate": "aggregate_expected_unemployment_rate",
        "expected_unemployment_higher_prob": "aggregate_expected_unemployment_higher_prob",
        "expected_real_income_growth": "aggregate_expected_real_income_growth",
    }
    aggregate = mapping.get(target, "")
    if target == "expected_unemployment_higher_prob" and aggregate not in environment:
        return float(np.clip(35.0 + 4.0 * (environment.get("aggregate_expected_unemployment_rate", 4.5) - 4.5), 0.0, 100.0))
    return float(environment.get(aggregate, INITIAL_PRIORS.get(target, 0.0)))


def _fixture_actual_ecology_belief(
    profile: dict[str, Any],
    prior: float,
    environment: dict[str, float],
    target: str,
    row_index: int,
    period_index: int,
) -> float:
    idio = 0.18 * np.sin(row_index * 1.7 + period_index)
    profile_anchor = _profile_anchor(profile, target)
    environment_anchor = _environment_anchor(environment, target)
    spec = TARGET_SPECS[target]
    value = 0.52 * prior + 0.23 * profile_anchor + 0.25 * environment_anchor + idio
    return float(np.clip(value, spec["lower"], spec["upper"]))


def _fixture_module_weights(profile: dict[str, Any], environment: dict[str, float]) -> dict[str, float]:
    uncertainty = abs(environment["news_inflation_pressure"]) + abs(environment["news_labor_pressure"]) + environment["credit_tightness"]
    prior_weight = float(np.clip(0.58 - 0.04 * uncertainty, 0.42, 0.68))
    environment_weight = float(np.clip(0.22 + 0.04 * uncertainty, 0.16, 0.34))
    aggregate_weight = float(np.clip(0.06 + 0.03 * environment["credit_tightness"], 0.04, 0.12))
    profile_weight = max(0.05, 1.0 - prior_weight - environment_weight - aggregate_weight)
    total = prior_weight + environment_weight + aggregate_weight + profile_weight
    return {
        "profile_weight": profile_weight / total,
        "prior_weight": prior_weight / total,
        "environment_weight": environment_weight / total,
        "aggregate_feedback_weight": aggregate_weight / total,
    }


def _fixture_ecology_uncertainty(profile: dict[str, Any], environment: dict[str, float], target: str) -> float:
    base = 1.0
    base += 0.35 if str(profile.get("liquid_wealth_group")) == "low" else 0.0
    base += 0.20 * abs(float(environment.get("news_inflation_pressure", 0.0)))
    base += 0.20 * abs(float(environment.get("news_labor_pressure", 0.0)))
    base += 0.25 * float(environment.get("credit_tightness", 0.0))
    if target == "expected_unemployment_rate":
        base += 0.35
    if target == "expected_unemployment_higher_prob":
        base += 1.20
    if target == "expected_real_income_growth":
        base += 0.55
    return float(np.clip(base, 0.35, 5.0))


def _model_shift(model: str, respondent_id: str, target: str) -> float:
    seed = sum(ord(char) for char in f"{model}:{respondent_id}:{target}")
    deterministic = ((seed % 17) - 8) / 100.0
    if "5.4" in model:
        return -0.04 + deterministic
    if "5.5" in model:
        return 0.05 + deterministic
    if "gemini" in model.lower():
        return 0.02 - deterministic
    return deterministic


def _fixture_actions(profile: dict[str, Any], beliefs: dict[str, float], environment: dict[str, float]) -> dict[str, float]:
    income_group = str(profile.get("income_group", "unknown"))
    liquid_group = str(profile.get("liquid_wealth_group", "unknown"))
    employment = str(profile.get("employment_status", "unknown"))
    liquidity = {"low": 1.35, "middle": 0.85, "high": 0.45}.get(liquid_group, 0.8)
    income_buffer = {"low": 1.25, "middle": 0.90, "high": 0.55}.get(income_group, 0.9)
    inflation_gap = float(beliefs.get("expected_inflation_1y", 3.0)) - 3.0
    unemployment_gap = float(beliefs.get("expected_unemployment_rate", 4.5)) - 4.5
    income_growth = float(beliefs.get("expected_real_income_growth", 1.0))
    credit = float(environment.get("credit_tightness", 0.35))
    consumption = 0.75 * income_growth - 0.42 * liquidity * max(inflation_gap, 0.0) - 0.55 * income_buffer * max(unemployment_gap, 0.0)
    buffer = 0.70 * liquidity * max(inflation_gap, 0.0) + 0.85 * liquidity * max(unemployment_gap, 0.0) - 0.15 * income_growth
    borrowing = 0.30 * income_growth - 0.70 * credit - 0.35 * liquidity * max(unemployment_gap, 0.0)
    portfolio = 0.30 * max(inflation_gap, 0.0) + 0.40 * credit - 0.20 * income_growth
    job = 0.70 * max(unemployment_gap, 0.0) + (1.0 if employment == "unemployed" else 0.0) - 0.20 * income_growth
    return {
        "consumption_change_pct": float(np.clip(consumption, -20.0, 20.0)),
        "liquid_buffer_change_pct": float(np.clip(buffer, -25.0, 25.0)),
        "borrowing_desire_index": float(np.clip(borrowing, -5.0, 5.0)),
        "portfolio_rebalance_to_liquid_pct": float(np.clip(portfolio, -15.0, 15.0)),
        "job_search_intensity_index": float(np.clip(job, -3.0, 6.0)),
    }


def _fixture_actual_actions(profile: dict[str, Any], actuals: dict[str, float], environment: dict[str, float]) -> dict[str, float]:
    actions = _fixture_actions(profile, actuals, environment)
    return {column: float(value * 1.04) for column, value in actions.items()}


def _weighted_card_environment(cards: list[EcologyCard], column: str) -> float:
    weights = np.array([card.weight for card in cards], dtype=float)
    values = np.array([card.environment[column] for card in cards], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return float(values.mean())
    return float(np.sum(values * weights) / total)


def _score_prediction_group(group: pd.DataFrame, *, source: str, target_name: str, actual_column: str) -> dict[str, Any]:
    clean = group.dropna(subset=[actual_column, "prediction"]).copy()
    errors = clean["prediction"].astype(float) - clean[actual_column].astype(float)
    return {
        "source": source,
        "target_name": target_name,
        "n": int(clean.shape[0]),
        "rmse": float(np.sqrt(_weighted_mean(errors**2, clean["weight"]))) if not clean.empty else np.nan,
        "mae": _weighted_mean(errors.abs(), clean["weight"]) if not clean.empty else np.nan,
        "bias": _weighted_mean(errors, clean["weight"]) if not clean.empty else np.nan,
        "correlation": _safe_corr(clean["prediction"], clean[actual_column]) if not clean.empty else np.nan,
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce")
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = clean_values.notna() & clean_weights.notna()
    if not mask.any():
        return np.nan
    clean_values = clean_values[mask].astype(float)
    clean_weights = clean_weights[mask].astype(float)
    total = float(clean_weights.sum())
    if total <= 0:
        return float(clean_values.mean())
    return float((clean_values * clean_weights).sum() / total)


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce")
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = clean_values.notna() & clean_weights.notna()
    if not mask.any():
        return np.nan
    clean_values = clean_values[mask].astype(float)
    clean_weights = clean_weights[mask].astype(float)
    total = float(clean_weights.sum())
    if total <= 0:
        return float(clean_values.std(ddof=0))
    mean = float((clean_values * clean_weights).sum() / total)
    variance = float(((clean_values - mean) ** 2 * clean_weights).sum() / total)
    return float(np.sqrt(max(0.0, variance)))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    clean = pd.DataFrame({"left": left, "right": right}).dropna()
    if clean.shape[0] < 3:
        return np.nan
    corr = clean["left"].corr(clean["right"])
    return float(corr) if np.isfinite(corr) else np.nan


def _finite_or_default(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _client_mode_and_cap(mode: str, max_live_calls: int, live_calls_used: int) -> tuple[str, int]:
    if mode != "live":
        return mode, 0
    remaining = max(0, int(max_live_calls) - int(live_calls_used))
    if remaining <= 0:
        return "replay", 0
    return "live", remaining


def _persona_ecology_instructions() -> str:
    return """
Return only valid JSON. You are simulating a persistent household survey agent.
Use only the supplied respondent profile, prior beliefs, current environment, and aggregate feedback.
Do not browse, inspect files, run commands, cite hidden current survey answers, or cite realized future values.
""".strip()


def _safe_relative(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return str(resolved.relative_to(cwd)) if resolved.is_relative_to(cwd) else resolved.name


def _sanitized_argv() -> list[str]:
    raw = list(sys.argv)
    if not raw:
        return ["python3", "-m", "macro_llm_tournament.persona_ecology"]
    module_path = Path(raw[0])
    if module_path.name == "persona_ecology.py":
        return ["python3", "-m", "macro_llm_tournament.persona_ecology", *raw[1:]]
    return raw


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def run(command: list[str]) -> str | None:
        try:
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False, timeout=10)
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
