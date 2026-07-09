"""Prepare a deterministic, leakage-audited SCE panel for dynamic macro runs.

The input is the normalized SCE respondent file produced by
``prepare_sce_microdata``.  This module deliberately keeps the raw respondent
identifier inside the preparation process only: output panel identifiers are
stable pseudonyms salted by the sample seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "dynamic_macro_sce_panel_v1"
MODEL_CUTOFFS: dict[str, str] = {
    "gpt-5-codex": "2024-09-30",
    "gpt-5.4": "2025-08-31",
    "gpt-5.5": "2025-12-01",
}
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_SAMPLE_SIZE = 81
DEFAULT_SAMPLE_SEED = 20250709
DEFAULT_PUBLICATION_LAG_MONTHS = 9
INITIAL_WAVE_POSITIONS = ("first", "last")
DEFAULT_SPLITS: dict[str, str] = {
    "development": "2023-01..2023-06",
    "validation": "2024-01..2024-06",
    "sealed_test": "2025-04..2025-09",
}

BELIEF_COLUMNS = (
    "actual_expected_inflation_1y",
    "actual_expected_unemployment_higher_prob",
    "actual_expected_real_income_growth",
)
STRATIFICATION_COLUMNS = ("income_group", "age_group", "education_group")
REQUIRED_COLUMNS = ("respondent_id", "survey_date", "weight", *STRATIFICATION_COLUMNS, *BELIEF_COLUMNS)
OPTIONAL_DEMOGRAPHIC_COLUMNS = (
    "gender",
    "region",
    "employment_status",
    "homeownership",
    "liquid_wealth_group",
)
PRIOR_COLUMNS = (
    "prior_expected_inflation_1y",
    "prior_expected_unemployment_higher_prob",
    "prior_expected_real_income_growth",
)
CURRENT_TO_PRIOR = dict(zip(BELIEF_COLUMNS, PRIOR_COLUMNS))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "work" / "persona_beliefs" / "sce_real_microdata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "work" / "persona_beliefs" / "dynamic_macro_panel_gpt55"


class DynamicMacroPanelError(ValueError):
    """Raised when a dynamic macro input cannot be prepared without guessing."""


def _month_start(value: Any, *, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DynamicMacroPanelError(f"{field} contains an invalid date: {value!r}")
    return pd.Timestamp(parsed).to_period("M").to_timestamp()


def _month_text(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-01")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _month_number(value: pd.Timestamp) -> int:
    return value.year * 12 + value.month


def _require_nonblank(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    bad: list[str] = []
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            bad.append(column)
    if bad:
        raise DynamicMacroPanelError(f"{label} contains missing or blank fields: {', '.join(sorted(bad))}")


def validate_normalized_sce(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize the normalized SCE respondent grain.

    Dates are represented as month starts. Respondents may enter, leave, or skip
    waves; the source itself may not omit a month globally.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("SCE input must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DynamicMacroPanelError(f"normalized SCE input missing required fields: {', '.join(missing)}")
    out = frame.copy()
    _require_nonblank(out, ("respondent_id",), label="respondent_id")
    _require_nonblank(out, STRATIFICATION_COLUMNS, label="stratification fields")
    out["respondent_id"] = out["respondent_id"].astype(str).str.strip()
    out["_survey_month"] = pd.to_datetime(out["survey_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if out["_survey_month"].isna().any():
        raise DynamicMacroPanelError("survey_date contains invalid dates")
    out["survey_date"] = out["_survey_month"].map(_month_text)

    if out.duplicated(["respondent_id", "_survey_month"]).any():
        duplicate = out.loc[out.duplicated(["respondent_id", "_survey_month"], keep=False), ["respondent_id", "survey_date"]]
        example = duplicate.iloc[0].to_dict()
        raise DynamicMacroPanelError(
            "duplicate respondent_id+survey_date grain; example=" + json.dumps(example, sort_keys=True)
        )

    numeric_columns = ("weight", *BELIEF_COLUMNS)
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        values = out[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DynamicMacroPanelError(f"{column} contains missing, non-numeric, or non-finite values")
    if (out["weight"] <= 0).any():
        raise DynamicMacroPanelError("weights must be finite and strictly positive")
    if not math.isfinite(float(out["weight"].sum())) or float(out["weight"].sum()) <= 0:
        raise DynamicMacroPanelError("weights must sum to a positive finite value")

    available = sorted(out["_survey_month"].unique())
    expected = _month_range(available[0], available[-1])
    if available != expected:
        missing_months = sorted(set(expected) - set(available))
        raise DynamicMacroPanelError(
            "input survey months are not globally consecutive; missing="
            + ", ".join(_month_text(month) for month in missing_months[:12])
        )
    return out.sort_values(["_survey_month", "respondent_id"]).reset_index(drop=True)


def parse_period_range(value: str | Sequence[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a closed monthly range such as ``2020-01..2023-12``."""

    if not isinstance(value, str):
        if len(value) != 2:
            raise DynamicMacroPanelError("period ranges must contain exactly start and end months")
        start_text, end_text = str(value[0]), str(value[1])
    else:
        separator = next((candidate for candidate in ("..", ":", "/") if candidate in value), None)
        if separator is None:
            start_text = end_text = value
        else:
            start_text, end_text = value.split(separator, 1)
    start = _month_start(start_text.strip(), field="period start")
    end = _month_start(end_text.strip(), field="period end")
    if start > end:
        raise DynamicMacroPanelError(f"period range starts after it ends: {value!r}")
    return start, end


def parse_split_spec(spec: str) -> tuple[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if "=" not in spec:
        raise DynamicMacroPanelError("split must use ROLE=YYYY-MM..YYYY-MM syntax")
    role, period = spec.split("=", 1)
    role = role.strip()
    if not role:
        raise DynamicMacroPanelError("split role cannot be blank")
    return role, parse_period_range(period.strip())


def normalize_split_windows(splits: Mapping[str, str | Sequence[str]] | None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    raw = DEFAULT_SPLITS if splits is None else splits
    if not raw:
        raise DynamicMacroPanelError("at least one split window is required")
    normalized: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for role, value in raw.items():
        role = str(role).strip()
        if not role:
            raise DynamicMacroPanelError("split role cannot be blank")
        if role in normalized:
            raise DynamicMacroPanelError(f"duplicate split role: {role}")
        normalized[role] = parse_period_range(value)

    ordered = sorted(normalized.items(), key=lambda item: (item[1][0], item[1][1], item[0]))
    previous_role: str | None = None
    previous_end: pd.Timestamp | None = None
    for role, (start, end) in ordered:
        if previous_end is not None:
            if start <= previous_end:
                raise DynamicMacroPanelError(f"split windows overlap: {previous_role} and {role}")
        previous_role, previous_end = role, end
    return normalized


def _role_seed(sample_seed: int, role: str) -> int:
    return int(_stable_digest("split-seed", sample_seed, role)[:15], 16) % (2**31 - 1)


def _initial_sample_seed(sample_seed: int, role: str) -> int:
    return int(_stable_digest("initial-wave-seed", sample_seed, role)[:15], 16) % (2**31 - 1)


def _select_initial_role(
    windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]], initial_role: str | None
) -> str:
    if initial_role is not None:
        if initial_role not in windows:
            raise DynamicMacroPanelError(f"initial role is not one of the split roles: {initial_role}")
        return initial_role
    if "sealed_test" in windows:
        return "sealed_test"
    return max(windows, key=lambda role: (windows[role][1], windows[role][0], role))


def contamination_label(
    model: str,
    survey_event_date: str | pd.Timestamp,
    *,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
) -> str:
    """Classify contamination using public availability, never event date."""

    if model not in MODEL_CUTOFFS:
        raise DynamicMacroPanelError(f"unknown model for contamination registry: {model}")
    if publication_lag_months < 0:
        raise DynamicMacroPanelError("publication lag must be non-negative")
    event = _month_start(survey_event_date, field="survey event date")
    available = event + pd.DateOffset(months=int(publication_lag_months))
    cutoff = pd.Timestamp(MODEL_CUTOFFS[model])
    return "potential_training_contamination" if available.normalize() <= cutoff else "post_cutoff_holdout"


def _availability_date(event: pd.Timestamp, publication_lag_months: int) -> pd.Timestamp:
    return (event + pd.DateOffset(months=publication_lag_months)).normalize()


def _stable_digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _anonymous_id(raw_id: str, sample_seed: int, role: str) -> str:
    return "respondent_" + _stable_digest("dynamic-macro-panel-v1", sample_seed, role, raw_id)[:16]


def _allocate_strata(strata_counts: Mapping[tuple[str, ...], int], sample_size: int, seed: int) -> dict[tuple[str, ...], int]:
    if sample_size < len(strata_counts):
        raise DynamicMacroPanelError(
            f"sample size {sample_size} is insufficient for {len(strata_counts)} non-empty strata"
        )
    total = sum(strata_counts.values())
    allocations = {key: 1 for key in strata_counts}
    ideals = {key: sample_size * count / total for key, count in strata_counts.items()}
    while sum(allocations.values()) < sample_size:
        candidates = [key for key, count in strata_counts.items() if allocations[key] < count]
        if not candidates:
            raise DynamicMacroPanelError("insufficient respondents to fill the requested stratified sample")
        key = max(
            candidates,
            key=lambda candidate: (
                ideals[candidate] - allocations[candidate],
                _stable_digest("stratum", seed, *candidate),
            ),
        )
        allocations[key] += 1
    return allocations


def _stratified_wave_sample(
    snapshot: pd.DataFrame, *, sample_size: int, sample_seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible_count = int(snapshot["respondent_id"].nunique())
    if eligible_count < sample_size:
        raise DynamicMacroPanelError(
            f"insufficient eligible first-wave respondents: need {sample_size}, found {eligible_count}"
        )
    strata_counts_series = snapshot.groupby(list(STRATIFICATION_COLUMNS), dropna=False)["respondent_id"].nunique()
    strata_counts = {tuple(str(key) for key in index): int(count) for index, count in strata_counts_series.items()}
    if not strata_counts:
        raise DynamicMacroPanelError("no non-empty stratification strata are available")
    allocations = _allocate_strata(strata_counts, sample_size, sample_seed)
    ranked = snapshot.copy()
    ranked["_sample_rank"] = ranked.apply(
        lambda row: _stable_digest(
            "respondent", sample_seed, row["respondent_id"], *[row[column] for column in STRATIFICATION_COLUMNS]
        ),
        axis=1,
    )
    selected_parts: list[pd.DataFrame] = []
    for key in sorted(allocations):
        key_values = pd.Series(key, index=STRATIFICATION_COLUMNS)
        key_mask = ranked[list(STRATIFICATION_COLUMNS)].astype(str).eq(key_values, axis="columns").all(axis=1)
        selected_parts.append(ranked[key_mask].sort_values(["_sample_rank", "respondent_id"]).head(allocations[key]))
    selected = pd.concat(selected_parts, ignore_index=True).drop(columns=["_sample_rank"])
    metadata = {
        "eligible_respondents": eligible_count,
        "sampled_respondents": sample_size,
        "strata_count": len(strata_counts),
        "strata_population_counts": {"|".join(key): value for key, value in sorted(strata_counts.items())},
        "strata_sample_counts": {"|".join(key): allocations[key] for key in sorted(allocations)},
    }
    return selected.sort_values("respondent_id").reset_index(drop=True), metadata


def stratified_complete_panel_sample(
    frame: pd.DataFrame,
    *,
    months: Sequence[pd.Timestamp],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic complete respondent cohort for one split window."""

    if sample_size <= 0:
        raise DynamicMacroPanelError("sample size must be positive")
    requested_months = list(months)
    if not requested_months:
        raise DynamicMacroPanelError("complete-panel sampling requires at least one month")
    expected_months = _month_range(requested_months[0], requested_months[-1])
    if requested_months != expected_months:
        raise DynamicMacroPanelError("complete-panel sampling requires one consecutive monthly window")
    source = frame[frame["_survey_month"].isin(set(requested_months))].copy()
    expected_count = len(requested_months)
    counts = source.groupby("respondent_id")["_survey_month"].nunique()
    complete_ids = counts[counts == expected_count].index.astype(str).tolist()
    if len(complete_ids) < sample_size:
        raise DynamicMacroPanelError(
            f"insufficient complete-panel respondents: need {sample_size}, found {len(complete_ids)}"
        )
    complete = source[source["respondent_id"].isin(complete_ids)].copy()
    snapshot = complete[complete["_survey_month"].eq(requested_months[0])].copy()
    selected_snapshot, wave_metadata = _stratified_wave_sample(
        snapshot, sample_size=sample_size, sample_seed=sample_seed
    )
    selected_ids = selected_snapshot["respondent_id"].tolist()
    selected = complete[complete["respondent_id"].isin(selected_ids)].copy()
    if selected["respondent_id"].nunique() != sample_size or selected.shape[0] != sample_size * expected_count:
        raise DynamicMacroPanelError("stratified sample did not produce the requested complete panel")
    metadata = {
        "complete_panel_mode": "longitudinal",
        "eligible_complete_panel_respondents": len(complete_ids),
        "sampled_respondents": sample_size,
        "sampled_panel_rows": int(selected.shape[0]),
        "unique_sampled_respondents": int(selected["respondent_id"].nunique()),
        "strata_count": wave_metadata["strata_count"],
        "strata_population_counts": wave_metadata["strata_population_counts"],
        "strata_sample_counts": wave_metadata["strata_sample_counts"],
    }
    return selected.sort_values(["_survey_month", "respondent_id"]).reset_index(drop=True), metadata


def _normalized_group(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value).strip().lower()
    return text if text and text != "nan" else fallback


def _group_factor(group: str, values: Mapping[str, float], default: float) -> float:
    for key, value in values.items():
        if key in group:
            return value
    return default


def _build_initial_households(initial: pd.DataFrame) -> pd.DataFrame:
    if initial.empty:
        raise DynamicMacroPanelError("initial household sample is empty")
    if initial["_survey_month"].nunique() != 1:
        raise DynamicMacroPanelError("initial household sample must contain exactly one survey wave")
    first = (
        initial.sort_values(["respondent_id", "_survey_month"])
        .groupby("respondent_id", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for row in first.to_dict("records"):
        income = _normalized_group(row.get("income_group"))
        liquidity_source = _normalized_group(row.get("liquid_wealth_group"))
        liquidity = "high" if "high" in liquidity_source else "low"
        age = _normalized_group(row.get("age_group"))
        employment = _normalized_group(row.get("employment_status"))
        income_value = _group_factor(income, {"low": 36_000.0, "middle": 72_000.0, "high": 135_000.0}, 72_000.0)
        consumption_ratio = _group_factor(income, {"low": 0.94, "middle": 0.82, "high": 0.68}, 0.82)
        liquid_months = 5.0 if liquidity == "high" else 0.8
        annual_income = income_value
        baseline_consumption = annual_income * consumption_ratio
        job_risk = "high" if any(token in employment for token in ("unemployed", "retired", "out of", "not working")) else "low"
        baseline_job_loss = float(np.clip(float(row["actual_expected_unemployment_higher_prob"]) / 4.0, 0.5, 20.0))
        rows.append(
            {
                "type_id": row["respondent_id"],
                "label": f"{income} income, {liquidity} liquid assets, {job_risk} job risk, {age}",
                "population_weight": float(row["weight"]),
                "source_split_role": row["_split_role"],
                "source_survey_event_date": row["survey_event_date"],
                "source_estimated_public_availability_date": row["estimated_public_availability_date"],
                "source_contamination_label": row["contamination_label"],
                "age_bucket": "older" if any(token in age for token in ("55", "65", "older", "retired")) else "prime",
                "income_group": income,
                "liquidity_group": liquidity,
                "job_loss_risk_type": job_risk,
                "employment_status": employment,
                "annual_income": annual_income,
                "baseline_consumption_annual": baseline_consumption,
                "liquid_assets": liquid_months * baseline_consumption / 12.0,
                "debt": annual_income * _group_factor(income, {"low": 0.16, "middle": 0.13, "high": 0.09}, 0.13),
                "debt_service_burden": _group_factor(income, {"low": 0.16, "middle": 0.13, "high": 0.09}, 0.13),
                "base_mpc": _group_factor(income, {"low": 0.80, "middle": 0.55, "high": 0.30}, 0.55),
                "base_saving_rate": _group_factor(income, {"low": 0.06, "middle": 0.12, "high": 0.20}, 0.12),
                "rate_sensitivity": _group_factor(income, {"low": 0.42, "middle": 0.55, "high": 0.70}, 0.55),
                "income_sensitivity": _group_factor(income, {"low": 0.82, "middle": 0.58, "high": 0.36}, 0.58),
                "precautionary_sensitivity": 0.68 if job_risk == "high" else (0.62 if liquidity == "low" else 0.38),
                "baseline_job_loss_probability": baseline_job_loss,
                "unemployment_higher_probability_1y": float(row["actual_expected_unemployment_higher_prob"]),
                "target_buffer_months": (6.2 if job_risk == "high" else 4.8) if liquidity == "high" else (2.7 if job_risk == "high" else 1.5),
                "inflation_expectation_1y": float(row["actual_expected_inflation_1y"]),
                "income_growth_expectation_1y": float(row["actual_expected_real_income_growth"]),
                "confidence_index": 50.0,
                "attention_weight_prices": 0.55,
                "attention_weight_jobs": 0.78 if job_risk == "high" else 0.46,
                "attention_weight_rates": 0.68 if liquidity == "high" else 0.36,
                "income_volatility": 0.18 if job_risk == "high" else 0.12,
                "subsistence_floor_share": 0.56 if income == "low" else (0.48 if income == "middle" else 0.38),
            }
        )
    return pd.DataFrame(rows).sort_values("type_id").reset_index(drop=True)


def _prepare_frames(
    source: pd.DataFrame,
    *,
    model: str,
    splits: Mapping[str, str | Sequence[str]] | None,
    sample_size: int,
    sample_seed: int,
    publication_lag_months: int,
    initial_role: str | None,
    initial_wave_position: str,
    source_path: Path | None,
) -> dict[str, Any]:
    if model not in MODEL_CUTOFFS:
        raise DynamicMacroPanelError(f"unknown model: {model}")
    if publication_lag_months < 0:
        raise DynamicMacroPanelError("publication lag must be non-negative")
    if initial_wave_position not in INITIAL_WAVE_POSITIONS:
        raise DynamicMacroPanelError(
            f"initial wave position must be one of: {', '.join(INITIAL_WAVE_POSITIONS)}"
        )
    normalized = validate_normalized_sce(source)
    windows = normalize_split_windows(splits)
    ordered_roles = sorted(windows, key=lambda role: (windows[role][0], windows[role][1], role))
    requested_months = sorted(
        {month for role in ordered_roles for month in _month_range(*windows[role])}
    )
    available = set(normalized["_survey_month"])
    missing_months = [month for month in requested_months if month not in available]
    if missing_months:
        raise DynamicMacroPanelError(
            "input does not cover every requested split month: "
            + ", ".join(_month_text(month) for month in missing_months[:12])
        )
    source = normalized.sort_values(["respondent_id", "_survey_month"]).copy()
    source["_raw_respondent_id"] = source["respondent_id"]
    source["_prior_survey_month"] = source.groupby("respondent_id", sort=False)["_survey_month"].shift(1)
    for current, prior in CURRENT_TO_PRIOR.items():
        source[prior] = source.groupby("respondent_id", sort=False)[current].shift(1)
    selected_initial_role = _select_initial_role(windows, initial_role)
    role_frames: list[pd.DataFrame] = []
    split_manifest: dict[str, Any] = {}
    split_seeds: dict[str, int] = {}
    for role in ordered_roles:
        start, end = windows[role]
        role_months = _month_range(start, end)
        role_seed = _role_seed(sample_seed, role)
        split_seeds[role] = role_seed
        sampled, sample_metadata = stratified_complete_panel_sample(
            source,
            months=role_months,
            sample_size=sample_size,
            sample_seed=role_seed,
        )
        sampled["respondent_id"] = sampled["_raw_respondent_id"].map(
            lambda value, seed=role_seed, split_role=role: _anonymous_id(value, seed, split_role)
        )
        identity_map = sampled[["_raw_respondent_id", "respondent_id"]].drop_duplicates()
        if identity_map["respondent_id"].duplicated().any():
            raise DynamicMacroPanelError(f"anonymization produced a respondent identifier collision in {role}")
        sampled["_split_role"] = role
        sampled["split_role"] = role
        period_index_by_month = {month: index for index, month in enumerate(role_months)}
        sampled["_period_index"] = sampled["_survey_month"].map(period_index_by_month)
        sampled["period_index"] = sampled["_period_index"]
        sampled["period_id"] = sampled["_survey_month"].map(lambda value: f"sce_{value.strftime('%Y_%m')}")
        sampled["survey_event_date"] = sampled["_survey_month"].map(_month_text)
        sampled["estimated_public_availability_date"] = sampled["_survey_month"].map(
            lambda value: _month_text(_availability_date(value, publication_lag_months))
        )
        sampled["model"] = model
        sampled["model_cutoff_date"] = MODEL_CUTOFFS[model]
        sampled["contamination_label"] = sampled.apply(
            lambda row: contamination_label(model, row["survey_event_date"], publication_lag_months=publication_lag_months), axis=1
        )
        sampled["panel_row_id"] = sampled["respondent_id"] + "__" + role + "__" + sampled["period_id"]
        for column in OPTIONAL_DEMOGRAPHIC_COLUMNS:
            if column not in sampled:
                sampled[column] = "unknown"
        role_frames.append(sampled.sort_values(["_period_index", "respondent_id"]).reset_index(drop=True))
        split_manifest[role] = {
            "start_date": _month_text(start),
            "end_date": _month_text(end),
            "month_count": len(role_months),
            "row_count": int(sampled.shape[0]),
            "respondent_count": int(sampled["respondent_id"].nunique()),
            "sample_seed": role_seed,
            "complete_panel_mode": sample_metadata["complete_panel_mode"],
            "eligible_complete_panel_respondents": sample_metadata["eligible_complete_panel_respondents"],
            "strata_population_counts": sample_metadata["strata_population_counts"],
            "strata_sample_counts": sample_metadata["strata_sample_counts"],
            "contamination_counts": {str(key): int(value) for key, value in sampled["contamination_label"].value_counts().sort_index().items()},
        }
    sampled_panel = pd.concat(role_frames, ignore_index=True)
    initial_start, initial_end = windows[selected_initial_role]
    if initial_wave_position == "last":
        initial_wave = initial_end
        role_panel = sampled_panel[sampled_panel["_split_role"].eq(selected_initial_role)]
        initial_rows = role_panel[role_panel["_survey_month"].eq(initial_wave)].copy()
        initial_sample_metadata = {
            "mode": "complete_cohort_last_wave",
            "sample_seed": split_seeds[selected_initial_role],
            "eligible_respondents": split_manifest[selected_initial_role]["eligible_complete_panel_respondents"],
            "sampled_respondents": int(initial_rows["respondent_id"].nunique()),
            "strata_population_counts": split_manifest[selected_initial_role]["strata_population_counts"],
            "strata_sample_counts": split_manifest[selected_initial_role]["strata_sample_counts"],
        }
    else:
        initial_wave = initial_start
        expected_prior_wave = initial_wave - pd.offsets.MonthBegin(1)
        first_wave = source[source["_survey_month"].eq(initial_wave)].copy()
        finite_priors = np.isfinite(first_wave[list(PRIOR_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
        eligible = first_wave[
            first_wave["_prior_survey_month"].eq(expected_prior_wave) & finite_priors
        ].copy()
        initial_seed = _initial_sample_seed(sample_seed, selected_initial_role)
        initial_rows, initial_sample_metadata = _stratified_wave_sample(
            eligible,
            sample_size=sample_size,
            sample_seed=initial_seed,
        )
        initial_rows["respondent_id"] = initial_rows["_raw_respondent_id"].map(
            lambda value: _anonymous_id(value, initial_seed, f"{selected_initial_role}_initial_first")
        )
        initial_rows["_split_role"] = selected_initial_role
        initial_rows["split_role"] = selected_initial_role
        initial_rows["survey_event_date"] = initial_rows["_survey_month"].map(_month_text)
        initial_rows["estimated_public_availability_date"] = initial_rows["_survey_month"].map(
            lambda value: _month_text(_availability_date(value, publication_lag_months))
        )
        initial_rows["contamination_label"] = initial_rows.apply(
            lambda row: contamination_label(
                model, row["survey_event_date"], publication_lag_months=publication_lag_months
            ),
            axis=1,
        )
        for column in OPTIONAL_DEMOGRAPHIC_COLUMNS:
            if column not in initial_rows:
                initial_rows[column] = "unknown"
        initial_sample_metadata = {
            "mode": "as_of_first_wave_with_immediate_prior",
            "sample_seed": initial_seed,
            "required_prior_wave": _month_text(expected_prior_wave),
            **initial_sample_metadata,
        }
    households = _build_initial_households(initial_rows)
    initial_availability_date = _availability_date(initial_wave, publication_lag_months)
    output_columns = [
        "respondent_id",
        "panel_row_id",
        "split_role",
        "period_id",
        "period_index",
        "survey_date",
        "survey_event_date",
        "estimated_public_availability_date",
        "model",
        "model_cutoff_date",
        "contamination_label",
        "weight",
        *STRATIFICATION_COLUMNS,
        *OPTIONAL_DEMOGRAPHIC_COLUMNS,
        *BELIEF_COLUMNS,
        *PRIOR_COLUMNS,
    ]
    panel = sampled_panel[output_columns].copy()
    role_order = {role: index for index, role in enumerate(ordered_roles)}
    panel["_role_order"] = panel["split_role"].map(role_order)
    panel = panel.sort_values(["_role_order", "period_index", "respondent_id"]).drop(columns=["_role_order"]).reset_index(drop=True)
    contamination_by_model: dict[str, Any] = {}
    for candidate, cutoff in MODEL_CUTOFFS.items():
        labels = panel.apply(
            lambda row: contamination_label(candidate, row["survey_event_date"], publication_lag_months=publication_lag_months), axis=1
        )
        contamination_by_model[candidate] = {
            "cutoff_date": cutoff,
            "row_counts": {str(key): int(value) for key, value in labels.value_counts().sort_index().items()},
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(source_path) if source_path is not None else None,
            "row_count": int(normalized.shape[0]),
            "respondent_count": int(normalized["respondent_id"].nunique()),
        },
        "model": model,
        "model_cutoffs": MODEL_CUTOFFS,
        "publication_assumptions": {
            "core_sce_respondent_microdata_publication_lag_months": publication_lag_months,
            "availability_date_definition": "survey_event_date plus publication lag in calendar months",
            "contamination_basis": "estimated_public_availability_date compared with model_cutoff_date",
            "survey_event_date_is_not_contamination_basis": True,
        },
        "sample": {
            "requested_respondents_per_split": sample_size,
            "sample_seed": int(sample_seed),
            "split_seeds": split_seeds,
        },
        "initial_role": selected_initial_role,
        "initial_wave_position": initial_wave_position,
        "initial_wave": _month_text(initial_wave),
        "initial_estimated_public_availability_date": _month_text(initial_availability_date),
        "initial_sample": initial_sample_metadata,
        "date_coverage": {
            "start_date": _month_text(requested_months[0]),
            "end_date": _month_text(requested_months[-1]),
            "month_count": len(requested_months),
            "split_windows_may_be_nonconsecutive": True,
        },
        "split_roles": split_manifest,
        "contamination_by_model": contamination_by_model,
        "counts": {
            "input_rows": int(normalized.shape[0]),
            "input_respondents": int(normalized["respondent_id"].nunique()),
            "split_count": len(ordered_roles),
            "sampled_respondents_per_split": int(sample_size),
            "unique_sampled_respondents": int(panel["respondent_id"].nunique()),
            "panel_rows": int(panel.shape[0]),
            "initial_households_rows": int(households.shape[0]),
        },
        "outputs": {},
    }
    return {"panel": panel, "initial_households": households, "manifest": manifest}


def prepare_dynamic_macro_panel(
    frame: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    splits: Mapping[str, str | Sequence[str]] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
    initial_role: str | None = None,
    initial_wave_position: str = "last",
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare panel and demand-household frames without writing files."""

    return _prepare_frames(
        frame,
        model=model,
        splits=splits,
        sample_size=sample_size,
        sample_seed=sample_seed,
        publication_lag_months=publication_lag_months,
        initial_role=initial_role,
        initial_wave_position=initial_wave_position,
        source_path=source_path,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dynamic_macro_panel(
    input_csv: Path | str = DEFAULT_INPUT,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    panel_csv: Path | str | None = None,
    initial_households_csv: Path | str | None = None,
    manifest_path: Path | str | None = None,
    model: str = DEFAULT_MODEL,
    splits: Mapping[str, str | Sequence[str]] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
    initial_role: str | None = None,
    initial_wave_position: str = "last",
) -> dict[str, Any]:
    """Read normalized SCE data, write the three preparation artifacts, and hash them."""

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"normalized SCE input not found: {input_path}")
    output_root = Path(output_dir)
    panel_path = Path(panel_csv) if panel_csv is not None else output_root / "sce_dynamic_macro_panel.csv"
    households_path = Path(initial_households_csv) if initial_households_csv is not None else output_root / "initial_households.csv"
    manifest_file = Path(manifest_path) if manifest_path is not None else output_root / "manifest.json"
    result = prepare_dynamic_macro_panel(
        pd.read_csv(input_path),
        model=model,
        splits=splits,
        sample_size=sample_size,
        sample_seed=sample_seed,
        publication_lag_months=publication_lag_months,
        initial_role=initial_role,
        initial_wave_position=initial_wave_position,
        source_path=input_path,
    )
    for path in (panel_path, households_path, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    result["panel"].to_csv(panel_path, index=False)
    result["initial_households"].to_csv(households_path, index=False)
    manifest = result["manifest"]
    manifest["input"]["sha256"] = _sha256(input_path)
    manifest["outputs"] = {
        "panel_csv": {"path": str(panel_path), "sha256": _sha256(panel_path), "row_count": int(result["panel"].shape[0])},
        "initial_households_csv": {"path": str(households_path), "sha256": _sha256(households_path), "row_count": int(result["initial_households"].shape[0])},
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **result,
        "panel_csv": panel_path,
        "initial_households_csv": households_path,
        "manifest_path": manifest_file,
    }


def _parse_cli_splits(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    parsed: dict[str, str] = {}
    for value in values:
        role, period = parse_split_spec(value)
        if role in parsed:
            raise DynamicMacroPanelError(f"duplicate split role: {role}")
        parsed[role] = period
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a leakage-audited dynamic macro SCE panel.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-csv", type=Path, default=None)
    parser.add_argument("--initial-households-csv", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", choices=sorted(MODEL_CUTOFFS), default=DEFAULT_MODEL)
    parser.add_argument("--split", action="append", default=None, metavar="ROLE=YYYY-MM..YYYY-MM")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--publication-lag-months", type=int, default=DEFAULT_PUBLICATION_LAG_MONTHS)
    parser.add_argument("--initial-role", default=None)
    parser.add_argument("--initial-wave-position", choices=INITIAL_WAVE_POSITIONS, default="last")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_dynamic_macro_panel(
            args.input_csv,
            output_dir=args.output_dir,
            panel_csv=args.panel_csv,
            initial_households_csv=args.initial_households_csv,
            manifest_path=args.manifest,
            model=args.model,
            splits=_parse_cli_splits(args.split),
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            publication_lag_months=args.publication_lag_months,
            initial_role=args.initial_role,
            initial_wave_position=args.initial_wave_position,
        )
    except (DynamicMacroPanelError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(result["panel_csv"])
    print(result["initial_households_csv"])
    print(result["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
