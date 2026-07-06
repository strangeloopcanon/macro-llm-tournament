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

from .agent_common import OUTPUT_ROOT, WORK_ROOT, cache_key, markdown_table, round_or_none
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable


PERSONA_BELIEF_PANEL_VERSION = "persona_belief_panel_v1"
PERSONA_BELIEF_PROMPT_VERSION = "persona_belief_panel_v1"
SURVEY_SCHEMAS = ("normalized", "sce")
DEFAULT_TARGET_FIELDS = (
    "expected_inflation_1y",
    "expected_unemployment_rate",
    "expected_real_income_growth",
)
TARGET_SPECS: dict[str, dict[str, Any]] = {
    "expected_inflation_1y": {
        "label": "Expected inflation over the next 12 months",
        "units": "percent",
        "lower": -5.0,
        "upper": 20.0,
    },
    "expected_unemployment_rate": {
        "label": "Expected U.S. unemployment rate in 12 months",
        "units": "percent",
        "lower": 0.0,
        "upper": 35.0,
    },
    "expected_real_income_growth": {
        "label": "Expected own household real income growth over the next 12 months",
        "units": "percent",
        "lower": -20.0,
        "upper": 20.0,
    },
}
DEMOGRAPHIC_DIMENSIONS: dict[str, dict[str, Any]] = {
    "income_group": {
        "low": "low",
        "high": "high",
        "label": "low_minus_high_income",
    },
    "education_group": {
        "low": "high_school_or_less",
        "high": "college_plus",
        "label": "less_educated_minus_college",
    },
    "age_group": {
        "low": "55_plus",
        "high": "18_34",
        "label": "older_minus_younger",
    },
    "gender": {
        "low": "female",
        "high": "male",
        "label": "female_minus_male",
    },
}
EVIDENCE_THRESHOLDS = {
    "regression_sign_rate_min": 0.75,
    "median_within_variance_ratio_min": 0.5,
    "max_distribution_ks_stat_max": 0.35,
    "median_distribution_std_ratio_min": 0.45,
    "median_distribution_std_ratio_max": 1.80,
    "common_core_correlation_max": 0.95,
}
RESPONDENT_COLUMNS = [
    "respondent_id",
    "survey_source",
    "survey_date",
    "weight",
    "age_group",
    "income_group",
    "education_group",
    "gender",
    "region",
    "employment_status",
    "homeownership",
    "liquid_wealth_group",
]
SCE_ALIASES: dict[str, tuple[str, ...]] = {
    "respondent_id": ("respondent_id", "caseid", "case_id", "userid", "user_id", "id", "hhid", "pid"),
    "survey_date": ("survey_date", "date", "interview_date", "yyyymm", "survey_month", "wave"),
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
    "actual_expected_real_income_growth": (
        "actual_expected_real_income_growth",
        "expected_real_income_growth",
        "real_income_growth",
        "income_growth_expectation",
        "earnings_growth",
        "earnings_mean",
    ),
}


@dataclass(frozen=True)
class PersonaCard:
    respondent_id: str
    survey_source: str
    survey_date: str
    weight: float
    profile: dict[str, Any]
    targets: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a data-grounded persona belief panel.")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5")
    parser.add_argument("--belief-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--respondent-source", choices=["fixture", "csv"], default="fixture")
    parser.add_argument("--respondent-csv", default=None)
    parser.add_argument("--survey-schema", choices=SURVEY_SCHEMAS, default="normalized")
    parser.add_argument("--respondent-count", type=int, default=54)
    parser.add_argument("--respondent-limit", type=int, default=0)
    parser.add_argument("--survey-date", default="2026-01-01")
    parser.add_argument("--target-fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    target_fields = [part.strip() for part in args.target_fields.split(",") if part.strip()]
    unknown_targets = sorted(set(target_fields) - set(TARGET_SPECS))
    if unknown_targets:
        raise SystemExit(f"Unknown target fields: {', '.join(unknown_targets)}")
    if args.belief_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --belief-mode live is used")
    if args.belief_mode == "live" and not args.fresh_cache and not args.cache_dir:
        raise SystemExit("--fresh-cache or --cache-dir is required when --belief-mode live is used")
    if args.belief_mode == "live" and args.fresh_cache and args.cache_dir:
        raise SystemExit("--fresh-cache and --cache-dir cannot be combined; use one fresh run or one explicit resume cache")
    if not models:
        raise SystemExit("--models must contain at least one model")
    if args.respondent_source == "csv":
        if not args.respondent_csv:
            raise SystemExit("--respondent-csv is required when --respondent-source csv")
        if not Path(args.respondent_csv).exists():
            raise SystemExit(f"--respondent-csv does not exist: {args.respondent_csv}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"persona_belief_panel_{timestamp}"
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "fresh_persona_belief_cache"
        if args.fresh_cache
        else WORK_ROOT / "persona_belief_cache"
    )
    respondent_csv = Path(args.respondent_csv) if args.respondent_csv else None
    respondents = load_persona_respondents(
        source=args.respondent_source,
        respondent_csv=respondent_csv,
        respondent_count=args.respondent_count,
        survey_date=args.survey_date,
        target_fields=target_fields,
        survey_schema=args.survey_schema,
    )
    respondents = _anonymize_csv_respondent_ids(respondents) if args.respondent_source == "csv" else respondents
    respondents = _limit_static_respondents(respondents, args.respondent_limit)
    required_calls = int(respondents.shape[0] * len(models))
    if args.belief_mode == "live" and args.fresh_cache and args.max_live_calls < required_calls:
        raise SystemExit(
            "--max-live-calls must be at least "
            f"{required_calls} for a fresh live persona panel with {respondents.shape[0]} respondents and {len(models)} models"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    provider_execution_cwd = _isolated_provider_cwd(output_dir, provider=args.provider, source=args.respondent_source, mode=args.belief_mode)
    respondent_input = _respondent_input_manifest(
        source=args.respondent_source,
        respondent_csv=respondent_csv,
        normalized=respondents,
        respondent_limit=args.respondent_limit,
        anonymized=args.respondent_source == "csv",
    )
    manifest: dict[str, Any] = {
        "schema_version": PERSONA_BELIEF_PANEL_VERSION,
        "timestamp_utc": timestamp,
        "argv": _sanitized_argv(),
        "run_command": shlex.join(_sanitized_argv()),
        "git": _git_metadata(),
        "provider": args.provider,
        "models": models,
        "belief_mode": args.belief_mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "explicit_cache_dir": bool(args.cache_dir),
        "cache_dir": _safe_relative(cache_dir),
        "respondent_source": args.respondent_source,
        "survey_schema": args.survey_schema,
        "respondent_input": respondent_input,
        "respondent_count": int(respondents.shape[0]),
        "respondent_limit": int(args.respondent_limit),
        "target_fields": target_fields,
        "required_call_count": required_calls,
        "provider_execution_cwd": _safe_relative(provider_execution_cwd) if provider_execution_cwd else None,
        "shared_cache_allowed": bool(not args.fresh_cache and not args.cache_dir),
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    try:
        cards = build_persona_cards(respondents, target_fields=target_fields)
        all_predictions: list[pd.DataFrame] = []
        raw_records: list[dict[str, Any]] = []
        prompt_rows = [persona_prompt_record(card, target_fields=target_fields) for card in cards]
        live_calls = 0
        cache_hits = 0
        model_run_modes: dict[str, dict[str, Any]] = {}
        for model in models:
            client_mode, client_cap = _client_mode_and_cap(args.belief_mode, args.max_live_calls, live_calls)
            model_run_modes[model] = {"mode": client_mode, "max_live_calls": int(client_cap)}
            client = PersonaBeliefClient(
                args.provider,
                model,
                cache_dir,
                mode=client_mode,
                max_live_calls=client_cap,
                execution_cwd=provider_execution_cwd,
            )
            predictions = run_persona_beliefs(cards, client, target_fields=target_fields)
            all_predictions.append(predictions)
            raw_records.extend(client.raw_records)
            live_calls += client.live_call_count
            cache_hits += client.cache_hit_count
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        regression_scores = score_regression_gradient_match(respondents, predictions, target_fields=target_fields)
        gradient_scores = score_gradient_match(respondents, predictions, target_fields=target_fields)
        variance_scores = score_variance_flattening(respondents, predictions, target_fields=target_fields)
        distribution_scores = score_distribution_distance(respondents, predictions, target_fields=target_fields)
        common_core = score_common_core(predictions, target_fields=target_fields)
        group_means = build_group_means(respondents, predictions, target_fields=target_fields)
        evidence = classify_persona_evidence(regression_scores, variance_scores, common_core, distribution_scores)

        respondents.to_csv(output_dir / "persona_respondents.csv", index=False)
        pd.DataFrame(prompt_rows).to_json(output_dir / "persona_prompt_cards.jsonl", orient="records", lines=True)
        predictions.to_csv(output_dir / "persona_belief_predictions.csv", index=False)
        regression_scores.to_csv(output_dir / "persona_belief_regression_scores.csv", index=False)
        gradient_scores.to_csv(output_dir / "persona_belief_gradient_scores.csv", index=False)
        variance_scores.to_csv(output_dir / "persona_belief_variance_scores.csv", index=False)
        distribution_scores.to_csv(output_dir / "persona_belief_distribution_scores.csv", index=False)
        common_core.to_csv(output_dir / "persona_belief_common_core.csv", index=False)
        group_means.to_csv(output_dir / "persona_belief_group_means.csv", index=False)
        (output_dir / "persona_belief_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True), encoding="utf-8")

        manifest.update(
            {
                "status": "ok",
                "model_run_modes": model_run_modes,
                "prediction_rows": int(predictions.shape[0]),
                "regression_score_rows": int(regression_scores.shape[0]),
                "gradient_score_rows": int(gradient_scores.shape[0]),
                "variance_score_rows": int(variance_scores.shape[0]),
                "distribution_score_rows": int(distribution_scores.shape[0]),
                "common_core_rows": int(common_core.shape[0]),
                "live_call_count": int(live_calls),
                "cache_hit_count": int(cache_hits),
                "fixture_generated_count": int((predictions["call_source"] == "fixture").sum()) if "call_source" in predictions else 0,
                "evidence": evidence,
                "cache_dir": _safe_relative(cache_dir),
                "outputs": [
                    "persona_respondents.csv",
                    "persona_prompt_cards.jsonl",
                    "persona_belief_predictions.csv",
                    "persona_belief_regression_scores.csv",
                    "persona_belief_gradient_scores.csv",
                    "persona_belief_variance_scores.csv",
                    "persona_belief_distribution_scores.csv",
                    "persona_belief_common_core.csv",
                    "persona_belief_group_means.csv",
                    "persona_belief_raw_records.json",
                    "persona_belief_panel_report.md",
                ],
            }
        )
        report = build_persona_belief_report(manifest, regression_scores, gradient_scores, variance_scores, distribution_scores, common_core)
        (output_dir / "persona_belief_panel_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


class PersonaBeliefClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str,
        max_live_calls: int,
        execution_cwd: Path | None = None,
    ):
        self.provider = provider
        self.model = model
        self.mode = mode
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

    def belief_card(self, card: PersonaCard, *, target_fields: Iterable[str]) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_persona_belief_payload(card, self.model, target_fields=target_fields),
                "cache_hit": False,
                "cache_path": None,
                "call_source": "fixture",
            }
        else:
            prompt = persona_belief_prompt(card, target_fields=target_fields)
            cache_name = f"persona_belief_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"
            data = self._client.json_call(prompt, cache_name, instructions=_persona_belief_instructions())
        call_source = data.get("call_source") or ("cache" if data.get("cache_hit") else ("live" if self.mode == "live" else self.mode))
        normalized = normalize_persona_belief_payload(card, data, provider=self.provider, model=self.model, target_fields=target_fields)
        normalized["call_source"] = call_source
        self.raw_records.append(
            {
                "respondent_id": card.respondent_id,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "call_source": call_source,
                "payload": data.get("payload", data),
            }
        )
        return normalized


def load_persona_respondents(
    *,
    source: str,
    respondent_csv: Path | None,
    respondent_count: int,
    survey_date: str,
    target_fields: Iterable[str],
    survey_schema: str = "normalized",
) -> pd.DataFrame:
    if source == "fixture":
        return build_fixture_respondent_panel(respondent_count=respondent_count, survey_date=survey_date, target_fields=target_fields)
    if respondent_csv is None:
        raise ValueError("--respondent-csv is required when --respondent-source csv")
    frame = pd.read_csv(respondent_csv)
    if survey_schema == "sce":
        return normalize_sce_respondent_panel(frame, target_fields=target_fields)
    if survey_schema != "normalized":
        raise ValueError(f"Unsupported survey schema: {survey_schema}")
    return normalize_respondent_panel(frame, target_fields=target_fields)


def normalize_sce_respondent_panel(
    frame: pd.DataFrame,
    *,
    target_fields: Iterable[str],
    require_unique_respondents: bool = True,
) -> pd.DataFrame:
    out = _standardize_sce_aliases(frame)
    if "respondent_id" not in out:
        out["respondent_id"] = [f"sce_respondent_{idx + 1:05d}" for idx in range(out.shape[0])]
    if "survey_date" not in out:
        out["survey_date"] = "unknown"
    out["survey_date"] = _normalize_survey_dates(out["survey_date"])
    if "weight" not in out:
        out["weight"] = 1.0
    if "survey_source" not in out:
        out["survey_source"] = "ny_fed_sce_microdata"

    out["age_group"] = _coalesce_text(out, "age_group", _age_group_from_numeric(out.get("age")))
    out["income_group"] = _coalesce_text(out, "income_group", _income_group_from_numeric(out.get("income")))
    out["education_group"] = _coalesce_text(out, "education_group", _education_group_from_raw(out.get("education")))
    out["gender"] = _coalesce_text(out, "gender", _gender_from_raw(out.get("female")))
    out["region"] = _coalesce_text(out, "region", pd.Series("unknown", index=out.index))
    out["employment_status"] = _coalesce_text(out, "employment_status", pd.Series("unknown", index=out.index))
    out["homeownership"] = _coalesce_text(out, "homeownership", pd.Series("unknown", index=out.index))
    out["liquid_wealth_group"] = _coalesce_text(out, "liquid_wealth_group", _income_group_from_numeric(out.get("liquid_wealth")))

    return normalize_respondent_panel(out, target_fields=target_fields, require_unique_respondents=require_unique_respondents)


def _standardize_sce_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    normalized = {_normalize_name(column): column for column in out.columns}
    for canonical, aliases in SCE_ALIASES.items():
        if canonical in out:
            continue
        source = next((normalized[_normalize_name(alias)] for alias in aliases if _normalize_name(alias) in normalized), None)
        if source is not None:
            out[canonical] = out[source]
    return out


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize_survey_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    yyyymm = text.str.fullmatch(r"\d{6}")
    normalized = pd.Series(pd.NA, index=series.index, dtype="object")
    if yyyymm.any():
        normalized.loc[yyyymm] = text.loc[yyyymm].str.slice(0, 4) + "-" + text.loc[yyyymm].str.slice(4, 6) + "-01"
    if (~yyyymm).any():
        parsed = pd.to_datetime(text.loc[~yyyymm], errors="coerce")
        normalized.loc[~yyyymm] = parsed.dt.date.astype(str).where(parsed.notna(), "unknown")
    return normalized.fillna("unknown").astype(str)


def _coalesce_text(frame: pd.DataFrame, column: str, fallback: pd.Series) -> pd.Series:
    if fallback.empty:
        fallback = pd.Series("unknown", index=frame.index)
    if column not in frame:
        return fallback.reindex(frame.index).fillna("unknown").astype(str).map(_normalize_label)
    series = frame[column]
    return series.where(series.notna(), fallback.reindex(frame.index)).fillna("unknown").astype(str).map(_normalize_label)


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not text or text in {"nan", "none", "<na>"}:
        return "unknown"
    replacements = {
        "college": "college_plus",
        "ba_plus": "college_plus",
        "bachelor": "college_plus",
        "bachelor_or_more": "college_plus",
        "graduate": "college_plus",
        "postgraduate": "college_plus",
        "high_school": "high_school_or_less",
        "high_school_or_lower": "high_school_or_less",
        "hs_or_less": "high_school_or_less",
        "not_college": "high_school_or_less",
        "less_than_high_school": "high_school_or_less",
        "some_college_or_associate": "some_college",
        "associate": "some_college",
        "owner": "owner",
        "own": "owner",
        "homeowner": "owner",
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


def normalize_respondent_panel(
    frame: pd.DataFrame,
    *,
    target_fields: Iterable[str],
    require_unique_respondents: bool = True,
) -> pd.DataFrame:
    out = frame.copy()
    if "respondent_id" not in out:
        out["respondent_id"] = [f"respondent_{idx + 1:05d}" for idx in range(out.shape[0])]
    if "weight" not in out:
        out["weight"] = 1.0
    if "survey_source" not in out:
        out["survey_source"] = "respondent_csv"
    if "survey_date" not in out:
        out["survey_date"] = "unknown"
    for column in RESPONDENT_COLUMNS:
        if column not in out:
            out[column] = "unknown" if column != "weight" else 1.0
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(out["weight"].sum()) <= 0.0:
        out["weight"] = 1.0
    out["weight"] = out["weight"] / float(out["weight"].sum())
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
    post_drop_weight = float(out["weight"].sum())
    if post_drop_weight <= 0.0:
        out["weight"] = 1.0
        post_drop_weight = float(out["weight"].sum())
    out["weight"] = out["weight"] / post_drop_weight
    for column in RESPONDENT_COLUMNS:
        if column != "weight":
            out[column] = out[column].fillna("unknown").astype(str)
    if require_unique_respondents and out["respondent_id"].duplicated().any():
        duplicates = sorted(out.loc[out["respondent_id"].duplicated(), "respondent_id"].astype(str).unique())[:5]
        raise ValueError(f"Respondent panel has duplicate respondent_id values: {', '.join(duplicates)}")
    return out.reset_index(drop=True)


def build_fixture_respondent_panel(
    *,
    respondent_count: int = 54,
    survey_date: str = "2026-01-01",
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    if respondent_count <= 0:
        raise ValueError("respondent_count must be positive")
    income_groups = ["low", "middle", "high"]
    education_groups = ["high_school_or_less", "some_college", "college_plus"]
    age_groups = ["18_34", "35_54", "55_plus"]
    genders = ["female", "male"]
    combinations = [
        (income, education, age, gender)
        for income in income_groups
        for education in education_groups
        for age in age_groups
        for gender in genders
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(1, respondent_count + 1):
        income, education, age, gender = combinations[(idx - 1) % len(combinations)]
        employment = _fixture_employment(idx, income, age)
        row = {
            "respondent_id": f"fixture_resp_{idx:03d}",
            "survey_source": "fixture_micro_survey",
            "survey_date": survey_date,
            "weight": 1.0,
            "age_group": age,
            "income_group": income,
            "education_group": education,
            "gender": gender,
            "region": ["northeast", "midwest", "south", "west"][idx % 4],
            "employment_status": employment,
            "homeownership": "owner" if income in {"middle", "high"} and idx % 3 else "renter",
            "liquid_wealth_group": "low" if income == "low" or employment == "unemployed" else ("high" if income == "high" else "middle"),
        }
        row.update(_fixture_actual_beliefs(row, idx))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["weight"] = 1.0 / max(1, frame.shape[0])
    keep_targets = [f"actual_{target}" for target in target_fields]
    return frame[RESPONDENT_COLUMNS + keep_targets].reset_index(drop=True)


def build_persona_cards(respondents: pd.DataFrame, *, target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS) -> list[PersonaCard]:
    cards: list[PersonaCard] = []
    for _, row in respondents.iterrows():
        profile = {column: row[column] for column in RESPONDENT_COLUMNS if column not in {"respondent_id", "survey_source", "survey_date", "weight"}}
        targets = {target: float(row[f"actual_{target}"]) for target in target_fields}
        cards.append(
            PersonaCard(
                respondent_id=str(row["respondent_id"]),
                survey_source=str(row["survey_source"]),
                survey_date=str(row["survey_date"]),
                weight=float(row["weight"]),
                profile=profile,
                targets=targets,
            )
        )
    return cards


def persona_prompt_record(card: PersonaCard, *, target_fields: Iterable[str]) -> dict[str, Any]:
    prompt_text = persona_belief_prompt(card, target_fields=target_fields)
    return {
        "respondent_id": card.respondent_id,
        "survey_source": card.survey_source,
        "survey_date": card.survey_date,
        "weight": card.weight,
        "prompt_payload": json.loads(prompt_text),
        "prompt_text": prompt_text,
    }


def persona_prompt_payload(card: PersonaCard) -> dict[str, Any]:
    source = card.survey_source.lower()
    record_kind = (
        "synthetic fixture respondent"
        if "fixture" in source
        else ("synthetic enriched respondent" if "synthetic" in source else "survey respondent")
    )
    return {
        "prompt_version": PERSONA_BELIEF_PROMPT_VERSION,
        "task": f"Simulate this {record_kind}'s subjective macroeconomic beliefs.",
        "as_of_rule": (
            "Use only the respondent profile and survey date. Do not use realized future values, "
            "published survey response targets, files, or tools."
        ),
        "respondent_record_kind": record_kind,
        "survey_source": card.survey_source,
        "survey_date": card.survey_date,
        "respondent_profile": card.profile,
    }


def persona_belief_prompt(card: PersonaCard, *, target_fields: Iterable[str]) -> str:
    payload = persona_prompt_payload(card)
    payload["required_response"] = {
        "respondent_id": card.respondent_id,
        "beliefs": {
            target: {
                "value": f"numeric {TARGET_SPECS[target]['units']} value",
                "p10": "10th percentile subjective belief",
                "p50": "median subjective belief",
                "p90": "90th percentile subjective belief",
            }
            for target in target_fields
        },
        "confidence": "0 to 1",
        "uncertainty": "0 to 1.5",
        "reason": "short explanation based only on supplied profile",
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def run_persona_beliefs(cards: Iterable[PersonaCard], client: PersonaBeliefClient, *, target_fields: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = f"llm_{client.provider}_{client.model}".replace("/", "_")
    for card in cards:
        payload = client.belief_card(card, target_fields=target_fields)
        for target in target_fields:
            belief = payload["beliefs"][target]
            rows.append(
                {
                    "schema_version": PERSONA_BELIEF_PANEL_VERSION,
                    "respondent_id": card.respondent_id,
                    "survey_source": card.survey_source,
                    "survey_date": card.survey_date,
                    "source": source,
                    "provider": client.provider,
                    "model": client.model,
                    "target_name": target,
                    "prediction": belief["value"],
                    "p10": belief["p10"],
                    "p50": belief["p50"],
                    "p90": belief["p90"],
                    "confidence": payload["confidence"],
                    "uncertainty": payload["uncertainty"],
                    "reason": payload["reason"],
                    "cache_hit": payload["cache_hit"],
                    "cache_path": payload["cache_path"],
                    "call_source": payload["call_source"],
                }
            )
    return pd.DataFrame(rows)


def normalize_persona_belief_payload(
    card: PersonaCard,
    data: dict[str, Any],
    *,
    provider: str,
    model: str,
    target_fields: Iterable[str],
) -> dict[str, Any]:
    payload = data.get("payload", data)
    respondent_id = str(payload.get("respondent_id", card.respondent_id))
    if respondent_id != card.respondent_id:
        raise LLMUnavailable(f"Persona payload respondent_id mismatch: expected {card.respondent_id}, got {respondent_id}")
    beliefs = payload.get("beliefs")
    if not isinstance(beliefs, dict):
        raise LLMUnavailable(f"Persona payload for {card.respondent_id} is missing beliefs object")
    normalized: dict[str, Any] = {
        "respondent_id": card.respondent_id,
        "provider": provider,
        "model": model,
        "beliefs": {},
        "confidence": _bounded_float(payload, "confidence", 0.0, 1.0, default=0.5),
        "uncertainty": _bounded_float(payload, "uncertainty", 0.0, 1.5, default=0.5),
        "reason": str(payload.get("reason", ""))[:300],
        "cache_hit": bool(data.get("cache_hit", False)),
        "cache_path": data.get("cache_path"),
    }
    missing = [target for target in target_fields if target not in beliefs]
    if missing:
        raise LLMUnavailable(f"Persona payload for {card.respondent_id} missing beliefs: {', '.join(missing)}")
    for target in target_fields:
        spec = TARGET_SPECS[target]
        raw = beliefs[target]
        if not isinstance(raw, dict):
            raise LLMUnavailable(f"Persona belief `{target}` for {card.respondent_id} must be an object")
        value = _bounded_float(raw, "value", spec["lower"], spec["upper"])
        p10 = _bounded_float(raw, "p10", spec["lower"], spec["upper"], default=value)
        p50 = _bounded_float(raw, "p50", spec["lower"], spec["upper"], default=value)
        p90 = _bounded_float(raw, "p90", spec["lower"], spec["upper"], default=value)
        lo, mid, hi = sorted([p10, p50, p90])
        normalized["beliefs"][target] = {"value": value, "p10": lo, "p50": mid, "p90": hi}
    return normalized


def fixture_persona_belief_payload(card: PersonaCard, model: str, *, target_fields: Iterable[str]) -> dict[str, Any]:
    beliefs: dict[str, dict[str, float]] = {}
    for target in target_fields:
        point = _fixture_predicted_belief(card.profile, card.respondent_id, model, target)
        width = _fixture_uncertainty(card.profile, target)
        spec = TARGET_SPECS[target]
        value = float(np.clip(point, spec["lower"], spec["upper"]))
        beliefs[target] = {
            "value": value,
            "p10": float(np.clip(value - width, spec["lower"], spec["upper"])),
            "p50": value,
            "p90": float(np.clip(value + width, spec["lower"], spec["upper"])),
        }
    return {
        "prompt_version": PERSONA_BELIEF_PROMPT_VERSION,
        "respondent_id": card.respondent_id,
        "beliefs": beliefs,
        "confidence": 0.62,
        "uncertainty": 0.55,
        "reason": "deterministic fixture persona belief from respondent profile",
    }


def score_regression_gradient_match(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = _joined_predictions(respondents, predictions)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_group in joined.groupby("source", dropna=False):
        for target in target_fields:
            target_group = source_group[source_group["target_name"] == target]
            if target_group.empty:
                continue
            survey_coefficients = _weighted_demographic_regression(target_group, f"actual_{target}")
            simulated_coefficients = _weighted_demographic_regression(target_group, "prediction")
            for dimension, spec in DEMOGRAPHIC_DIMENSIONS.items():
                low_group = str(spec["low"])
                high_group = str(spec["high"])
                contrast_key = f"{dimension}={low_group}"
                has_contrast = (
                    target_group[dimension].astype(str).eq(low_group).any()
                    and target_group[dimension].astype(str).eq(high_group).any()
                )
                survey_coefficient = survey_coefficients.get(contrast_key, np.nan) if has_contrast else np.nan
                simulated_coefficient = simulated_coefficients.get(contrast_key, np.nan) if has_contrast else np.nan
                sign_match = _sign_match(survey_coefficient, simulated_coefficient)
                rows.append(
                    {
                        "source": source,
                        "target_name": target,
                        "dimension": dimension,
                        "contrast": spec["label"],
                        "reference_group": high_group,
                        "scoreable": bool(has_contrast),
                        "survey_coefficient": survey_coefficient,
                        "simulated_coefficient": simulated_coefficient,
                        "sign_match": sign_match,
                        "magnitude_error": abs(simulated_coefficient - survey_coefficient)
                        if np.isfinite(survey_coefficient) and np.isfinite(simulated_coefficient)
                        else np.nan,
                        "magnitude_ratio": simulated_coefficient / survey_coefficient if abs(survey_coefficient) > 1e-9 else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_name", "dimension", "source"]).reset_index(drop=True)


def score_gradient_match(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = _joined_predictions(respondents, predictions)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_group in joined.groupby("source", dropna=False):
        for target in target_fields:
            target_group = source_group[source_group["target_name"] == target]
            if target_group.empty:
                continue
            for dimension, spec in DEMOGRAPHIC_DIMENSIONS.items():
                survey_gradient = _two_group_gradient(target_group, dimension, spec["low"], spec["high"], f"actual_{target}")
                simulated_gradient = _two_group_gradient(target_group, dimension, spec["low"], spec["high"], "prediction")
                scoreable = bool(np.isfinite(survey_gradient) and np.isfinite(simulated_gradient))
                sign_match = _sign_match(survey_gradient, simulated_gradient)
                rows.append(
                    {
                        "source": source,
                        "target_name": target,
                        "dimension": dimension,
                        "contrast": spec["label"],
                        "scoreable": scoreable,
                        "survey_gradient": survey_gradient,
                        "simulated_gradient": simulated_gradient,
                        "sign_match": sign_match,
                        "magnitude_error": abs(simulated_gradient - survey_gradient),
                        "magnitude_ratio": simulated_gradient / survey_gradient if abs(survey_gradient) > 1e-9 else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_name", "dimension", "source"]).reset_index(drop=True)


def score_variance_flattening(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = _joined_predictions(respondents, predictions)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_group in joined.groupby("source", dropna=False):
        for target in target_fields:
            target_group = source_group[source_group["target_name"] == target]
            if target_group.empty:
                continue
            for dimension in DEMOGRAPHIC_DIMENSIONS:
                survey_within = _within_group_variance(target_group, dimension, f"actual_{target}")
                simulated_within = _within_group_variance(target_group, dimension, "prediction")
                survey_total = _weighted_var(target_group[f"actual_{target}"], target_group["weight"])
                simulated_total = _weighted_var(target_group["prediction"], target_group["weight"])
                rows.append(
                    {
                        "source": source,
                        "target_name": target,
                        "dimension": dimension,
                        "survey_total_variance": survey_total,
                        "simulated_total_variance": simulated_total,
                        "total_variance_ratio": simulated_total / survey_total if survey_total > 1e-12 else np.nan,
                        "survey_within_variance": survey_within,
                        "simulated_within_variance": simulated_within,
                        "within_variance_ratio": simulated_within / survey_within if survey_within > 1e-12 else np.nan,
                        "flattening_flag": bool(simulated_within / survey_within < 0.5) if survey_within > 1e-12 else False,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_name", "dimension", "source"]).reset_index(drop=True)


def score_distribution_distance(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = _joined_predictions(respondents, predictions)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_group in joined.groupby("source", dropna=False):
        for target in target_fields:
            group = source_group[source_group["target_name"] == target]
            if group.empty:
                continue
            actual = group[f"actual_{target}"].astype(float)
            predicted = group["prediction"].astype(float)
            weights = group["weight"].astype(float)
            survey_std = _weighted_std(actual, weights)
            simulated_std = _weighted_std(predicted, weights)
            rows.append(
                {
                    "source": source,
                    "target_name": target,
                    "n": int(group.shape[0]),
                    "survey_mean": _weighted_mean(actual, weights),
                    "simulated_mean": _weighted_mean(predicted, weights),
                    "mean_error": _weighted_mean(predicted, weights) - _weighted_mean(actual, weights),
                    "survey_std": survey_std,
                    "simulated_std": simulated_std,
                    "std_ratio": simulated_std / survey_std if survey_std > 1e-12 else np.nan,
                    "wasserstein_1": _weighted_wasserstein(actual, predicted, weights),
                    "ks_stat": _weighted_ks_stat(actual, predicted, weights),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_name", "source"]).reset_index(drop=True)


def score_common_core(
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in target_fields:
        target_predictions = predictions[predictions["target_name"] == target]
        if target_predictions.empty:
            continue
        pivot = target_predictions.pivot_table(index="respondent_id", columns="source", values="prediction", aggfunc="mean")
        sources = list(pivot.columns)
        correlations: list[float] = []
        for i, left in enumerate(sources):
            for right in sources[i + 1 :]:
                pair = pivot[[left, right]].dropna()
                if pair.shape[0] < 3:
                    continue
                corr = pair[left].corr(pair[right])
                if np.isfinite(corr):
                    correlations.append(float(corr))
        rows.append(
            {
                "target_name": target,
                "source_count": len(sources),
                "pair_count": len(correlations),
                "common_core_tested": bool(correlations),
                "mean_pairwise_correlation": float(np.mean(correlations)) if correlations else np.nan,
                "max_pairwise_correlation": float(np.max(correlations)) if correlations else np.nan,
                "common_core_flag": bool(correlations and np.mean(correlations) > 0.95),
            }
        )
    return pd.DataFrame(rows)


def build_group_means(
    respondents: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_fields: Iterable[str] = DEFAULT_TARGET_FIELDS,
) -> pd.DataFrame:
    joined = _joined_predictions(respondents, predictions)
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, source_group in joined.groupby("source", dropna=False):
        for target in target_fields:
            target_group = source_group[source_group["target_name"] == target]
            for dimension in DEMOGRAPHIC_DIMENSIONS:
                for group_name, group in target_group.groupby(dimension, dropna=False):
                    rows.append(
                        {
                            "source": source,
                            "target_name": target,
                            "dimension": dimension,
                            "group": group_name,
                            "n": int(group.shape[0]),
                            "weight": float(group["weight"].sum()),
                            "survey_mean": _weighted_mean(group[f"actual_{target}"], group["weight"]),
                            "simulated_mean": _weighted_mean(group["prediction"], group["weight"]),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["target_name", "dimension", "source", "group"]).reset_index(drop=True)


def classify_persona_evidence(
    regression_scores: pd.DataFrame,
    variance_scores: pd.DataFrame,
    common_core: pd.DataFrame,
    distribution_scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if regression_scores.empty:
        scoreable_regressions = pd.DataFrame()
    elif "scoreable" in regression_scores:
        scoreable_regressions = regression_scores[regression_scores["scoreable"].astype(bool)].copy()
    else:
        scoreable_regressions = regression_scores.copy()
    sign_rate = float(scoreable_regressions["sign_match"].mean()) if not scoreable_regressions.empty else np.nan
    scoreable_contrast_count = int(scoreable_regressions.shape[0])
    skipped_contrast_count = int(regression_scores.shape[0] - scoreable_contrast_count) if not regression_scores.empty else 0
    median_within_ratio = float(variance_scores["within_variance_ratio"].median()) if not variance_scores.empty else np.nan
    common_core_max = float(common_core["mean_pairwise_correlation"].max()) if not common_core.empty else np.nan
    distribution_scores = distribution_scores if distribution_scores is not None else pd.DataFrame()
    max_ks_stat = float(distribution_scores["ks_stat"].max()) if not distribution_scores.empty else np.nan
    median_std_ratio = float(distribution_scores["std_ratio"].median()) if not distribution_scores.empty else np.nan
    common_core_tested = bool(not common_core.empty and common_core["pair_count"].fillna(0).min() >= 1)
    gradients_clear = bool(np.isfinite(sign_rate) and sign_rate >= EVIDENCE_THRESHOLDS["regression_sign_rate_min"])
    spread_clear = bool(
        np.isfinite(median_within_ratio)
        and median_within_ratio >= EVIDENCE_THRESHOLDS["median_within_variance_ratio_min"]
    )
    distribution_clear = bool(
        np.isfinite(max_ks_stat)
        and np.isfinite(median_std_ratio)
        and max_ks_stat <= EVIDENCE_THRESHOLDS["max_distribution_ks_stat_max"]
        and EVIDENCE_THRESHOLDS["median_distribution_std_ratio_min"]
        <= median_std_ratio
        <= EVIDENCE_THRESHOLDS["median_distribution_std_ratio_max"]
    )
    common_core_clear = bool(
        common_core_tested
        and np.isfinite(common_core_max)
        and common_core_max <= EVIDENCE_THRESHOLDS["common_core_correlation_max"]
    )

    if not common_core_tested:
        verdict = "incomplete_common_core_test"
        branch = "common_core_unmeasured"
    elif gradients_clear and spread_clear and distribution_clear and common_core_clear:
        verdict = "clears_heterogeneity_gate"
        branch = "proceed_to_responses"
    elif gradients_clear and spread_clear and (not distribution_clear) and (not common_core_clear):
        verdict = "partial_distribution_and_common_core_failure"
        branch = "gradients_and_spread_right_but_distribution_or_common_core_fail"
    elif gradients_clear and spread_clear and not distribution_clear:
        verdict = "partial_distribution_failure"
        branch = "gradients_and_spread_right_but_distribution_distance_high"
    elif gradients_clear and (not spread_clear) and (not common_core_clear):
        verdict = "partial_flattening_and_common_core_failure"
        branch = "gradients_right_but_spread_collapsed_and_common_core_high"
    elif gradients_clear and not spread_clear:
        verdict = "partial_flattening_failure"
        branch = "gradients_right_but_spread_collapsed"
    elif gradients_clear and not common_core_clear:
        verdict = "partial_common_core_failure"
        branch = "gradients_right_but_common_core_high"
    else:
        verdict = "null_gradient_failure"
        branch = "gradients_flat_or_wrong"

    return {
        "evidence_verdict": verdict,
        "decision_tree_branch": branch,
        "thresholds": EVIDENCE_THRESHOLDS,
        "regression_sign_rate": sign_rate,
        "scoreable_contrast_count": scoreable_contrast_count,
        "skipped_contrast_count": skipped_contrast_count,
        "median_within_variance_ratio": median_within_ratio,
        "max_distribution_ks_stat": max_ks_stat,
        "median_distribution_std_ratio": median_std_ratio,
        "max_mean_pairwise_source_correlation": common_core_max,
        "common_core_tested": common_core_tested,
        "gradients_clear": gradients_clear,
        "spread_clear": spread_clear,
        "distribution_clear": distribution_clear,
        "common_core_clear": common_core_clear,
    }


def build_persona_belief_report(
    manifest: dict[str, Any],
    regression_scores: pd.DataFrame,
    gradient_scores: pd.DataFrame,
    variance_scores: pd.DataFrame,
    distribution_scores: pd.DataFrame,
    common_core: pd.DataFrame,
) -> str:
    evidence = manifest.get("evidence", {})
    sign_rate = float(regression_scores["sign_match"].mean()) if not regression_scores.empty else np.nan
    median_within_ratio = float(variance_scores["within_variance_ratio"].median()) if not variance_scores.empty else np.nan
    max_ks_stat = float(distribution_scores["ks_stat"].max()) if not distribution_scores.empty else np.nan
    median_std_ratio = float(distribution_scores["std_ratio"].median()) if not distribution_scores.empty else np.nan
    common_core_max = float(common_core["mean_pairwise_correlation"].max()) if not common_core.empty else np.nan
    respondent_source = str(manifest.get("respondent_source", ""))
    respondent_input = manifest.get("respondent_input", {})
    if respondent_source == "fixture":
        source_label = "synthetic fixture respondent panel"
    elif respondent_input.get("synthetic_enriched"):
        source_label = "synthetic enriched respondent panel"
    else:
        source_label = "data-grounded respondent panel"
    if evidence.get("evidence_verdict") == "clears_heterogeneity_gate":
        bottom_line = (
            f"The {source_label} clears the heterogeneity gate: adjusted gradients mostly match, "
            "within-group spread is not collapsed, distribution distance is inside threshold, "
            "and the mosaic/common-core check is measured and below threshold."
        )
    elif evidence.get("evidence_verdict") == "incomplete_common_core_test":
        bottom_line = (
            f"The {source_label} ran, but it cannot clear the heterogeneity gate because the common-core "
            "test is unmeasured; use at least two model sources before making a mosaic claim."
        )
    else:
        bottom_line = (
            f"The {source_label} ran, but it does not clear the heterogeneity gate. "
            "The decision-tree branch is "
            f"`{evidence.get('decision_tree_branch', 'unknown')}`."
        )
    lines = [
        "# Persona Belief Panel",
        "",
        "## Bottom Line",
        bottom_line,
        "",
        "## Run Setup",
        f"- Provider: `{manifest.get('provider')}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        f"- Belief mode: `{manifest.get('belief_mode')}`",
        f"- Respondents: `{manifest.get('respondent_count')}` from `{manifest.get('respondent_source')}`",
        f"- Panel kind: `{respondent_input.get('panel_kind')}`",
        f"- Target provenance: `{respondent_input.get('target_provenance')}`",
        f"- Live calls used: `{manifest.get('live_call_count', 0)}` of cap `{manifest.get('max_live_calls', 0)}`",
        f"- Cache hits: `{manifest.get('cache_hit_count', 0)}`",
        f"- Evidence verdict: `{evidence.get('evidence_verdict')}`",
        f"- Decision-tree branch: `{evidence.get('decision_tree_branch')}`",
        f"- Adjusted gradient sign-match rate: `{round_or_none(sign_rate)}`",
        f"- Median within-variance ratio: `{round_or_none(median_within_ratio)}`",
        f"- Max weighted KS statistic: `{round_or_none(max_ks_stat)}`",
        f"- Median distribution std ratio: `{round_or_none(median_std_ratio)}`",
        f"- Max mean pairwise source correlation: `{round_or_none(common_core_max)}`",
        "",
        "## Regression Gradient Match",
        markdown_table(regression_scores.head(48)),
        "",
        "## Contrast Diagnostics",
        markdown_table(gradient_scores.head(48)),
        "",
        "## Flattening / Variance",
        markdown_table(variance_scores.head(48)),
        "",
        "## Distribution Distance",
        markdown_table(distribution_scores.head(48)),
        "",
        "## Common Core",
        markdown_table(common_core),
        "",
        "## What This Gate Means",
        (
            "This gate asks whether persona respondents reproduce the cross-sectional structure "
            "of household beliefs. Point-forecast accuracy is deliberately not the target. The decisive "
            "evidence is adjusted demographic-gradient match, within-cell spread, full-distribution distance, "
            "and whether multiple model sources collapse onto the same common core. Fixture runs validate the "
            "runner and scoring logic only; csv-backed runs with held-out real respondent responses are the "
            "empirical evidence gate."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _persona_belief_instructions() -> str:
    return """
Return only valid JSON. You are simulating a survey respondent's subjective beliefs.
Use only the supplied respondent profile and survey date. Do not browse, inspect files,
run commands, cite realized future values, or cite hidden survey responses.
""".strip()


def _joined_predictions(respondents: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    columns = RESPONDENT_COLUMNS + [column for column in respondents.columns if column.startswith("actual_")]
    return predictions.merge(respondents[columns], on="respondent_id", how="inner", validate="many_to_one")


def _weighted_demographic_regression(frame: pd.DataFrame, outcome_column: str) -> dict[str, float]:
    outcome = pd.to_numeric(frame[outcome_column], errors="coerce")
    weights = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    columns = [np.ones(frame.shape[0], dtype=float)]
    names = ["intercept"]
    for dimension, spec in DEMOGRAPHIC_DIMENSIONS.items():
        values = frame[dimension].fillna("unknown").astype(str)
        reference = str(spec["high"])
        levels = sorted(level for level in values.unique() if level != reference)
        for level in levels:
            columns.append(values.eq(level).astype(float).to_numpy())
            names.append(f"{dimension}={level}")
    design = np.column_stack(columns)
    mask = outcome.notna().to_numpy() & np.isfinite(design).all(axis=1) & weights.notna().to_numpy()
    if not mask.any():
        return {name: np.nan for name in names}
    y = outcome.to_numpy(dtype=float)[mask]
    x = design[mask]
    w = weights.to_numpy(dtype=float)[mask]
    if float(w.sum()) <= 0.0:
        w = np.ones_like(w)
    root_w = np.sqrt(w)
    try:
        coefficients, *_ = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)
    except np.linalg.LinAlgError:
        return {name: np.nan for name in names}
    return {name: float(value) for name, value in zip(names, coefficients, strict=True)}


def _two_group_gradient(frame: pd.DataFrame, dimension: str, low_group: str, high_group: str, column: str) -> float:
    low = frame[frame[dimension].astype(str) == low_group]
    high = frame[frame[dimension].astype(str) == high_group]
    if low.empty or high.empty:
        return np.nan
    return _weighted_mean(low[column], low["weight"]) - _weighted_mean(high[column], high["weight"])


def _within_group_variance(frame: pd.DataFrame, dimension: str, column: str) -> float:
    total_weight = float(frame["weight"].sum())
    if total_weight <= 0.0:
        return np.nan
    pieces = []
    for _, group in frame.groupby(dimension, dropna=False):
        group_weight = float(group["weight"].sum())
        if group_weight <= 0.0:
            continue
        pieces.append((group_weight / total_weight) * _weighted_var(group[column], group["weight"]))
    return float(np.sum(pieces)) if pieces else np.nan


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


def _weighted_var(values: Iterable[Any], weights: Iterable[Any]) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    values = values[mask].astype(float)
    weights = weights[mask].astype(float)
    mean = _weighted_mean(values, weights)
    total = float(weights.sum())
    if total <= 0.0:
        return float(values.var(ddof=0))
    return float((((values - mean) ** 2) * weights).sum() / total)


def _weighted_std(values: Iterable[Any], weights: Iterable[Any]) -> float:
    variance = _weighted_var(values, weights)
    return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan


def _weighted_quantile(values: Iterable[Any], weights: Iterable[Any], quantiles: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.full_like(quantiles, np.nan, dtype=float)
    clean = pd.DataFrame({"value": values[mask].astype(float), "weight": weights[mask].astype(float)}).sort_values("value")
    cumulative = clean["weight"].cumsum()
    cumulative = cumulative / float(cumulative.iloc[-1])
    return np.interp(quantiles, cumulative.to_numpy(), clean["value"].to_numpy())


def _weighted_wasserstein(actual: Iterable[Any], predicted: Iterable[Any], weights: Iterable[Any]) -> float:
    quantiles = np.linspace(0.01, 0.99, 99)
    actual_q = _weighted_quantile(actual, weights, quantiles)
    predicted_q = _weighted_quantile(predicted, weights, quantiles)
    return float(np.nanmean(np.abs(actual_q - predicted_q)))


def _weighted_ks_stat(actual: Iterable[Any], predicted: Iterable[Any], weights: Iterable[Any]) -> float:
    actual_frame = _weighted_distribution_frame(actual, weights)
    predicted_frame = _weighted_distribution_frame(predicted, weights)
    if actual_frame.empty or predicted_frame.empty:
        return np.nan
    values = np.sort(np.unique(np.concatenate([actual_frame["value"].to_numpy(), predicted_frame["value"].to_numpy()])))
    actual_cdf = _weighted_cdf_at(actual_frame, values)
    predicted_cdf = _weighted_cdf_at(predicted_frame, values)
    return float(np.max(np.abs(actual_cdf - predicted_cdf)))


def _weighted_distribution_frame(values: Iterable[Any], weights: Iterable[Any]) -> pd.DataFrame:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return pd.DataFrame(columns=["value", "weight", "cdf"])
    frame = pd.DataFrame({"value": values[mask].astype(float), "weight": weights[mask].astype(float)}).sort_values("value")
    frame["cdf"] = frame["weight"].cumsum() / float(frame["weight"].sum())
    return frame


def _weighted_cdf_at(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    sorted_values = frame["value"].to_numpy()
    cdf = frame["cdf"].to_numpy()
    positions = np.searchsorted(sorted_values, values, side="right") - 1
    out = np.zeros(values.shape[0], dtype=float)
    valid = positions >= 0
    out[valid] = cdf[positions[valid]]
    return out


def _ks_stat(actual: Iterable[Any], predicted: Iterable[Any]) -> float:
    actual_values = np.sort(pd.to_numeric(pd.Series(actual), errors="coerce").dropna().astype(float).to_numpy())
    predicted_values = np.sort(pd.to_numeric(pd.Series(predicted), errors="coerce").dropna().astype(float).to_numpy())
    if len(actual_values) == 0 or len(predicted_values) == 0:
        return np.nan
    values = np.sort(np.unique(np.concatenate([actual_values, predicted_values])))
    actual_cdf = np.searchsorted(actual_values, values, side="right") / len(actual_values)
    predicted_cdf = np.searchsorted(predicted_values, values, side="right") / len(predicted_values)
    return float(np.max(np.abs(actual_cdf - predicted_cdf)))


def _sign_match(left: float, right: float) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    if abs(left) < 1e-9 and abs(right) < 1e-9:
        return True
    return bool(np.sign(left) == np.sign(right))


def _bounded_float(mapping: dict[str, Any], key: str, lower: float, upper: float, *, default: float | None = None) -> float:
    value = mapping.get(key, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise LLMUnavailable(f"Persona payload field `{key}` must be numeric")
        numeric = float(default)
    if not np.isfinite(numeric):
        if default is None:
            raise LLMUnavailable(f"Persona payload field `{key}` must be finite")
        numeric = float(default)
    return float(np.clip(numeric, lower, upper))


def _fixture_actual_beliefs(row: dict[str, Any], idx: int) -> dict[str, float]:
    income_effect = {"low": 0.95, "middle": 0.15, "high": -0.45}[row["income_group"]]
    education_effect = {"high_school_or_less": 0.55, "some_college": 0.15, "college_plus": -0.25}[row["education_group"]]
    age_effect = {"18_34": -0.15, "35_54": 0.05, "55_plus": 0.25}[row["age_group"]]
    gender_effect = 0.25 if row["gender"] == "female" else -0.05
    employment_effect = 1.2 if row["employment_status"] == "unemployed" else 0.0
    wave = 0.55 * np.sin(idx * 1.7)
    inflation = 3.0 + income_effect + education_effect + age_effect + gender_effect + 0.45 * employment_effect + wave
    unemployment = 4.8 + 0.75 * income_effect + 0.35 * education_effect + employment_effect + 0.35 * np.cos(idx * 1.3)
    income_growth = 1.4 - 0.35 * income_effect - 0.15 * education_effect - 0.85 * employment_effect + 0.45 * np.sin(idx * 0.9)
    return {
        "actual_expected_inflation_1y": float(np.clip(inflation, -5.0, 20.0)),
        "actual_expected_unemployment_rate": float(np.clip(unemployment, 0.0, 35.0)),
        "actual_expected_real_income_growth": float(np.clip(income_growth, -20.0, 20.0)),
    }


def _fixture_predicted_belief(profile: dict[str, Any], respondent_id: str, model: str, target: str) -> float:
    idx = int("".join(ch for ch in respondent_id if ch.isdigit()) or "1")
    income = str(profile.get("income_group", "middle"))
    education = str(profile.get("education_group", "some_college"))
    age = str(profile.get("age_group", "35_54"))
    gender = str(profile.get("gender", "unknown"))
    employment = str(profile.get("employment_status", "employed"))
    income_effect = {"low": 0.78, "middle": 0.10, "high": -0.35}.get(income, 0.0)
    education_effect = {"high_school_or_less": 0.42, "some_college": 0.10, "college_plus": -0.18}.get(education, 0.0)
    age_effect = {"18_34": -0.10, "35_54": 0.02, "55_plus": 0.18}.get(age, 0.0)
    gender_effect = 0.18 if gender == "female" else -0.03
    employment_effect = 0.9 if employment == "unemployed" else 0.0
    model_shift = _model_shift(model)
    jitter = 0.28 * np.sin(idx * 1.13 + model_shift)
    if target == "expected_inflation_1y":
        return 3.05 + income_effect + education_effect + age_effect + gender_effect + 0.35 * employment_effect + jitter
    if target == "expected_unemployment_rate":
        return 4.7 + 0.58 * income_effect + 0.25 * education_effect + 0.85 * employment_effect + 0.18 * np.cos(idx + model_shift)
    if target == "expected_real_income_growth":
        return 1.45 - 0.28 * income_effect - 0.10 * education_effect - 0.70 * employment_effect + 0.22 * np.sin(idx * 0.8 + model_shift)
    raise ValueError(f"Unknown target: {target}")


def _fixture_uncertainty(profile: dict[str, Any], target: str) -> float:
    low_liquid = str(profile.get("liquid_wealth_group")) == "low"
    unemployed = str(profile.get("employment_status")) == "unemployed"
    base = 0.9 if target != "expected_real_income_growth" else 1.2
    return base + (0.35 if low_liquid else 0.0) + (0.4 if unemployed else 0.0)


def _fixture_employment(idx: int, income: str, age: str) -> str:
    if income == "low" and idx % 5 == 0:
        return "unemployed"
    if age == "55_plus" and idx % 4 == 0:
        return "retired"
    return "employed"


def _limit_static_respondents(frame: pd.DataFrame, respondent_limit: int) -> pd.DataFrame:
    if respondent_limit <= 0 or frame.shape[0] <= respondent_limit:
        return frame.reset_index(drop=True)
    keep = sorted(frame["respondent_id"].astype(str).unique())[:respondent_limit]
    out = frame[frame["respondent_id"].astype(str).isin(keep)].copy()
    total = float(out["weight"].sum())
    out["weight"] = 1.0 / max(1, out.shape[0]) if total <= 0 else out["weight"] / total
    return out.reset_index(drop=True)


def _anonymize_csv_respondent_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ids = list(dict.fromkeys(out["respondent_id"].astype(str)))
    mapping = {respondent_id: f"respondent_{idx + 1:05d}" for idx, respondent_id in enumerate(ids)}
    out["respondent_id"] = out["respondent_id"].astype(str).map(mapping)
    return out


def _respondent_input_manifest(
    *,
    source: str,
    respondent_csv: Path | None,
    normalized: pd.DataFrame,
    respondent_limit: int,
    anonymized: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": source,
        "respondent_limit": int(respondent_limit),
        "respondent_ids_anonymized": bool(anonymized),
        "normalized_row_count": int(normalized.shape[0]),
        "normalized_respondent_count": int(normalized["respondent_id"].nunique()) if "respondent_id" in normalized else 0,
        "normalized_weight_sum": round_or_none(normalized["weight"].sum()) if "weight" in normalized else None,
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


def _raw_unique_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = [str(value) for value in frame[column].dropna().unique()]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "mixed:" + ",".join(values[:5])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isolated_provider_cwd(output_dir: Path, *, provider: str, source: str, mode: str) -> Path | None:
    if source != "csv" or mode != "live" or provider not in {"codex_cli", "gemini_cli", "antigravity_cli"}:
        return None
    root = Path(tempfile.gettempdir()) / "macro_llm_tournament_provider_cwd"
    path = root / cache_key({"output_dir": str(output_dir.resolve()), "provider": provider})
    path.mkdir(parents=True, exist_ok=True)
    return path


def _client_mode_and_cap(requested_mode: str, max_live_calls: int, live_calls_used: int) -> tuple[str, int]:
    if requested_mode != "live":
        return requested_mode, max_live_calls
    remaining = max(0, int(max_live_calls) - int(live_calls_used))
    if remaining <= 0:
        return "replay", 0
    return "live", remaining


def _model_shift(model: str) -> float:
    return (sum(ord(ch) for ch in model) % 17) / 17.0


def _git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "").strip())
        return result.stdout.strip()

    try:
        status = run_git(["status", "--short"])
        return {
            "available": True,
            "branch": run_git(["branch", "--show-current"]),
            "commit": run_git(["rev-parse", "HEAD"]),
            "dirty": bool(status),
        }
    except Exception as exc:  # pragma: no cover - only exercised outside git checkouts
        return {"available": False, "error": str(exc)}


def _sanitized_argv() -> list[str]:
    if sys.argv and Path(sys.argv[0]).resolve() == Path(__file__).resolve():
        return ["python3", "-m", "macro_llm_tournament.persona_belief_panel", *sys.argv[1:]]
    if not sys.argv:
        return ["python3", "-m", "macro_llm_tournament.persona_belief_panel"]
    return [Path(sys.argv[0]).name, *sys.argv[1:]]


def _safe_relative(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return str(resolved.relative_to(cwd)) if resolved.is_relative_to(cwd) else resolved.name


if __name__ == "__main__":
    raise SystemExit(main())
