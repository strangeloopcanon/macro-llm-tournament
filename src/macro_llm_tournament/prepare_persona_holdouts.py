from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import round_or_none
from .persona_belief_panel import DEFAULT_TARGET_FIELDS, build_fixture_respondent_panel, normalize_sce_respondent_panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "persona_beliefs"
DEFAULT_SURVEY_BELIEFS = WORK_ROOT / "survey_beliefs" / "survey_belief_targets.csv"
DEFAULT_FORECAST_ORIGINS = WORK_ROOT / "fred_vintage_panel" / "forecast_origins_for_vintage_context.csv"
DEFAULT_FRED_VINTAGE_CONTEXT = WORK_ROOT / "fred_vintage_panel" / "fred_vintage_context.csv"
PANEL_KIND = "synthetic_enriched_sce_vintage_v1"
REAL_SCE_PANEL_KIND = "real_sce_microdata_v1"
TARGET_PROVENANCE = "synthetic_persona_targets_anchored_to_public_sce_michigan_aggregates"
REAL_SCE_TARGET_PROVENANCE = "public_ny_fed_sce_microdata_responses"
ENVIRONMENT_PROVENANCE = "fred_alfred_vintage_context_by_spf_origin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare enriched synthetic persona holdout CSVs.")
    parser.add_argument("--survey-beliefs", type=Path, default=DEFAULT_SURVEY_BELIEFS)
    parser.add_argument("--forecast-origins", type=Path, default=DEFAULT_FORECAST_ORIGINS)
    parser.add_argument("--fred-vintage-context", type=Path, default=DEFAULT_FRED_VINTAGE_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--static-output", default="sce_micro_holdout.csv")
    parser.add_argument("--panel-output", default="sce_panel_holdout.csv")
    parser.add_argument("--respondent-count", type=int, default=54)
    parser.add_argument("--period-count", type=int, default=3)
    parser.add_argument("--start-as-of", default="2024-10-01")
    parser.add_argument("--panel-kind", default=PANEL_KIND)
    parser.add_argument("--sce-microdata", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = prepare_persona_holdouts(
        survey_beliefs_path=args.survey_beliefs,
        forecast_origins_path=args.forecast_origins,
        fred_vintage_context_path=args.fred_vintage_context,
        output_dir=args.output_dir,
        static_output=args.static_output,
        panel_output=args.panel_output,
        respondent_count=args.respondent_count,
        period_count=args.period_count,
        start_as_of=args.start_as_of,
        panel_kind=args.panel_kind,
        sce_microdata_path=args.sce_microdata,
    )
    print(outputs["static_csv"])
    print(outputs["panel_csv"])
    return 0


def prepare_persona_holdouts(
    *,
    survey_beliefs_path: Path,
    forecast_origins_path: Path,
    fred_vintage_context_path: Path,
    output_dir: Path,
    static_output: str,
    panel_output: str,
    respondent_count: int,
    period_count: int,
    start_as_of: str,
    panel_kind: str = PANEL_KIND,
    sce_microdata_path: Path | None = None,
) -> dict[str, Any]:
    if respondent_count <= 0:
        raise ValueError("respondent_count must be positive")
    if period_count <= 0:
        raise ValueError("period_count must be positive")
    if sce_microdata_path is not None:
        return prepare_real_sce_holdouts(
            sce_microdata_path=sce_microdata_path,
            survey_beliefs_path=survey_beliefs_path,
            forecast_origins_path=forecast_origins_path,
            fred_vintage_context_path=fred_vintage_context_path,
            output_dir=output_dir,
            static_output=static_output,
            panel_output=panel_output,
            period_count=period_count,
            start_as_of=start_as_of,
        )
    survey = _load_survey_targets(survey_beliefs_path)
    origins = _select_origins(forecast_origins_path, start_as_of=start_as_of, period_count=period_count)
    vintage = _load_vintage_context(fred_vintage_context_path)
    environments = build_period_environments(origins, vintage, survey)
    base_profiles = _base_profiles(respondent_count)
    panel = build_enriched_persona_panel(base_profiles, environments, panel_kind=panel_kind)
    static = panel[panel["period_index"] == int(panel["period_index"].max())].copy()
    static["respondent_id"] = static["respondent_id"].str.replace("__.*$", "", regex=True)
    static = static.drop(columns=["panel_row_id", "period_id", "period_index"], errors="ignore")
    static["weight"] = static["weight"] / float(static["weight"].sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / static_output
    panel_path = output_dir / panel_output
    manifest_path = output_dir / "persona_holdout_manifest.json"
    static.to_csv(static_path, index=False)
    panel.to_csv(panel_path, index=False)
    manifest = {
        "schema_version": "persona_holdout_preparation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "panel_kind": panel_kind,
        "target_provenance": TARGET_PROVENANCE,
        "environment_provenance": ENVIRONMENT_PROVENANCE,
        "respondent_count": int(base_profiles.shape[0]),
        "period_count": int(environments.shape[0]),
        "static_rows": int(static.shape[0]),
        "panel_rows": int(panel.shape[0]),
        "start_as_of": start_as_of,
        "origins": environments[["origin", "as_of_date", "survey_date"]].to_dict(orient="records"),
        "aggregate_expected_inflation_1y": [
            round_or_none(value) for value in environments["aggregate_expected_inflation_1y"].tolist()
        ],
        "source_files": {
            "survey_beliefs": _safe_relative(survey_beliefs_path),
            "forecast_origins": _safe_relative(forecast_origins_path),
            "fred_vintage_context": _safe_relative(fred_vintage_context_path),
        },
        "outputs": {
            "static_csv": _safe_relative(static_path),
            "panel_csv": _safe_relative(panel_path),
        },
        "usage_note": (
            "These are synthetic persona targets anchored to public aggregate survey beliefs and as-of vintage macro "
            "context. They are suitable for live ecology wiring and emergent-feedback canaries, not for claiming "
            "respondent-level empirical microdata evidence."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "static_csv": str(static_path),
        "panel_csv": str(panel_path),
        "manifest": str(manifest_path),
        "static_rows": int(static.shape[0]),
        "panel_rows": int(panel.shape[0]),
    }


def prepare_real_sce_holdouts(
    *,
    sce_microdata_path: Path,
    survey_beliefs_path: Path,
    forecast_origins_path: Path,
    fred_vintage_context_path: Path,
    output_dir: Path,
    static_output: str,
    panel_output: str,
    period_count: int,
    start_as_of: str,
) -> dict[str, Any]:
    if not sce_microdata_path.exists():
        raise FileNotFoundError(f"SCE microdata file not found: {sce_microdata_path}")
    if period_count <= 0:
        raise ValueError("period_count must be positive")
    survey = _load_survey_targets(survey_beliefs_path)
    origins = _load_forecast_origins(forecast_origins_path)
    vintage = _load_vintage_context(fred_vintage_context_path)
    raw = pd.read_csv(sce_microdata_path)
    normalized = normalize_sce_respondent_panel(
        raw,
        target_fields=DEFAULT_TARGET_FIELDS,
        require_unique_respondents=False,
    )
    normalized["_survey_timestamp"] = pd.to_datetime(normalized["survey_date"], errors="coerce")
    normalized = normalized.dropna(subset=["_survey_timestamp"]).sort_values(["_survey_timestamp", "respondent_id"])
    normalized = normalized[normalized["_survey_timestamp"].ge(pd.Timestamp(start_as_of))].copy()
    if normalized.empty:
        raise ValueError(f"SCE microdata has no scoreable rows at or after {start_as_of}")

    wave_dates = sorted(normalized["_survey_timestamp"].dt.normalize().unique())[:period_count]
    if not wave_dates:
        raise ValueError("SCE microdata has no usable survey waves")
    matched_origins = _match_origins_to_wave_dates(origins, wave_dates)
    environments = build_period_environments(matched_origins, vintage, survey)

    panel_rows: list[dict[str, Any]] = []
    prior_by_respondent: dict[str, dict[str, float]] = {}
    for period_index, wave_date in enumerate(wave_dates):
        wave_ts = pd.Timestamp(wave_date)
        period_id = f"sce_{wave_ts.strftime('%Y_%m')}"
        wave = normalized[normalized["_survey_timestamp"].dt.normalize().eq(wave_ts.normalize())].copy()
        env = environments.iloc[period_index].to_dict()
        env["survey_date"] = wave_ts.date().isoformat()
        env["period_id"] = period_id
        env["period_index"] = int(period_index)
        period_rows: list[dict[str, Any]] = []
        used_panel_row_ids: set[str] = set()
        for row_index, (_, respondent) in enumerate(wave.reset_index(drop=True).iterrows()):
            respondent_id = str(respondent["respondent_id"])
            out = {
                column: respondent[column]
                for column in normalized.columns
                if column != "_survey_timestamp"
            }
            out.update(
                {
                    "respondent_id": respondent_id,
                    "survey_source": "ny_fed_sce_microdata",
                    "survey_date": env["survey_date"],
                    "period_id": period_id,
                    "period_index": int(period_index),
                    "panel_row_id": f"{respondent_id}__{period_id}",
                    "persona_panel_kind": REAL_SCE_PANEL_KIND,
                    "target_provenance": REAL_SCE_TARGET_PROVENANCE,
                    "environment_provenance": ENVIRONMENT_PROVENANCE,
                }
            )
            if out["panel_row_id"] in used_panel_row_ids:
                out["panel_row_id"] = f"{respondent_id}__{period_id}__row_{row_index + 1:04d}"
            used_panel_row_ids.add(str(out["panel_row_id"]))
            for column in [
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
            ]:
                out[column] = env[column]
            prior = prior_by_respondent.get(respondent_id)
            for target in DEFAULT_TARGET_FIELDS:
                if prior is not None:
                    out[f"prior_{target}"] = prior[target]
                else:
                    out[f"prior_{target}"] = _initial_prior(target, out, pd.Series(env))
            prior_by_respondent[respondent_id] = {
                target: float(out[f"actual_{target}"])
                for target in DEFAULT_TARGET_FIELDS
            }
            period_rows.append(out)
        period_frame = pd.DataFrame(period_rows)
        period_frame["weight"] = pd.to_numeric(period_frame["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(period_frame["weight"].sum()) <= 0.0:
            period_frame["weight"] = 1.0
        period_frame["weight"] = period_frame["weight"] / float(period_frame["weight"].sum())
        panel_rows.extend(period_frame.to_dict(orient="records"))

    panel = pd.DataFrame(panel_rows)
    static = panel[panel["period_index"] == int(panel["period_index"].max())].copy()
    static = static.drop(columns=["panel_row_id", "period_id", "period_index"], errors="ignore")
    static["weight"] = pd.to_numeric(static["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    static["weight"] = 1.0 / max(1, static.shape[0]) if float(static["weight"].sum()) <= 0.0 else static["weight"] / float(static["weight"].sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / static_output
    panel_path = output_dir / panel_output
    manifest_path = output_dir / "persona_holdout_manifest.json"
    static.to_csv(static_path, index=False)
    panel.to_csv(panel_path, index=False)
    manifest = {
        "schema_version": "persona_holdout_preparation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "panel_kind": REAL_SCE_PANEL_KIND,
        "target_provenance": REAL_SCE_TARGET_PROVENANCE,
        "environment_provenance": ENVIRONMENT_PROVENANCE,
        "respondent_count": int(panel["respondent_id"].nunique()),
        "period_count": int(panel["period_id"].nunique()),
        "static_rows": int(static.shape[0]),
        "panel_rows": int(panel.shape[0]),
        "start_as_of": start_as_of,
        "origins": environments[["origin", "as_of_date", "survey_date"]].to_dict(orient="records"),
        "source_files": {
            "sce_microdata": {
                "path": _safe_relative(sce_microdata_path),
                "sha256": _file_sha256(sce_microdata_path),
                "raw_rows": int(raw.shape[0]),
                "raw_columns": int(raw.shape[1]),
            },
            "survey_beliefs": _safe_relative(survey_beliefs_path),
            "forecast_origins": _safe_relative(forecast_origins_path),
            "fred_vintage_context": _safe_relative(fred_vintage_context_path),
        },
        "outputs": {
            "static_csv": _safe_relative(static_path),
            "panel_csv": _safe_relative(panel_path),
        },
        "usage_note": (
            "These rows use public SCE respondent microdata responses as held-out targets. Prompt cards must use "
            "profile, survey date, priors, and vintage environment only; actual_* columns are scoring labels."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "static_csv": str(static_path),
        "panel_csv": str(panel_path),
        "manifest": str(manifest_path),
        "static_rows": int(static.shape[0]),
        "panel_rows": int(panel.shape[0]),
    }


def build_period_environments(origins: pd.DataFrame, vintage: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_index, origin in origins.reset_index(drop=True).iterrows():
        origin_name = str(origin["origin"])
        as_of = pd.Timestamp(origin["as_of_date"])
        origin_vintage = vintage[vintage["origin"].astype(str).eq(origin_name)].copy()
        belief = _survey_beliefs_as_of(survey, as_of)
        inflation_yoy = _series_growth(origin_vintage, "CPIAUCSL", periods=12, fallback=belief["median_inflation"])
        unemployment = _latest_series(origin_vintage, "UNRATE", fallback=4.5)
        policy = _latest_series(origin_vintage, "FEDFUNDS", fallback=_latest_series(origin_vintage, "TB3MS", fallback=3.0))
        real_growth = _series_growth(
            origin_vintage,
            "GDPC1",
            periods=4,
            fallback=_series_growth(origin_vintage, "PCECC96", periods=12, fallback=1.0),
        )
        aggregate_expected_unemployment = float(np.clip(unemployment + 0.25 * (unemployment - 4.5), 0.0, 30.0))
        aggregate_expected_income = float(np.clip(0.40 * real_growth - 0.15 * max(unemployment - 4.5, 0.0), -8.0, 8.0))
        sentiment = float(np.clip(94.0 - 4.5 * max(belief["median_inflation"] - 3.0, 0.0) - 2.2 * max(unemployment - 4.5, 0.0), 45.0, 115.0))
        rows.append(
            {
                "origin": origin_name,
                "as_of_date": as_of.date().isoformat(),
                "period_id": origin_name,
                "period_index": int(period_index),
                "survey_date": as_of.date().isoformat(),
                "observed_inflation_1y": round(float(inflation_yoy), 4),
                "observed_unemployment_rate": round(float(unemployment), 4),
                "observed_real_income_growth": round(float(real_growth), 4),
                "policy_rate": round(float(policy), 4),
                "sentiment_index": round(sentiment, 4),
                "news_inflation_pressure": round(float((inflation_yoy - 2.5) / 3.0), 4),
                "news_labor_pressure": round(float((unemployment - 4.5) / 2.5), 4),
                "credit_tightness": round(float(np.clip(0.22 + policy / 12.0 + max(unemployment - 4.5, 0.0) / 20.0, 0.05, 0.95)), 4),
                "aggregate_expected_inflation_1y": round(float(belief["median_inflation"]), 4),
                "aggregate_expected_unemployment_rate": round(aggregate_expected_unemployment, 4),
                "aggregate_expected_real_income_growth": round(aggregate_expected_income, 4),
                "inflation_iqr": round(float(belief["inflation_iqr"]), 4),
                "inflation_uncertainty": round(float(belief["inflation_uncertainty"]), 4),
                "environment_provenance": ENVIRONMENT_PROVENANCE,
            }
        )
    return pd.DataFrame(rows)


def build_enriched_persona_panel(
    base_profiles: pd.DataFrame,
    environments: pd.DataFrame,
    *,
    panel_kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous: dict[str, dict[str, float]] = {}
    for _, env in environments.iterrows():
        period_index = int(env["period_index"])
        period_rows: list[dict[str, Any]] = []
        for respondent_position, profile in base_profiles.reset_index(drop=True).iterrows():
            respondent_id = f"synthetic_resp_{respondent_position + 1:04d}"
            row = {column: profile[column] for column in base_profiles.columns if column.startswith("actual_") is False}
            row["respondent_id"] = respondent_id
            row["survey_source"] = panel_kind
            row["survey_date"] = env["survey_date"]
            row["period_id"] = env["period_id"]
            row["period_index"] = period_index
            row["panel_row_id"] = f"{respondent_id}__{env['period_id']}"
            row["persona_panel_kind"] = panel_kind
            row["target_provenance"] = TARGET_PROVENANCE
            row["environment_provenance"] = ENVIRONMENT_PROVENANCE
            for column in [
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
            ]:
                row[column] = env[column]
            actuals = _synthetic_targets(row, env, respondent_position)
            prior = previous.get(respondent_id, {})
            for target, value in actuals.items():
                row[f"actual_{target}"] = value
                row[f"prior_{target}"] = prior.get(target, _initial_prior(target, row, env))
            previous[respondent_id] = actuals
            period_rows.append(row)
        period_frame = pd.DataFrame(period_rows)
        period_frame["weight"] = period_frame["weight"] / float(period_frame["weight"].sum())
        rows.extend(period_frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _base_profiles(respondent_count: int) -> pd.DataFrame:
    base = build_fixture_respondent_panel(respondent_count=respondent_count, survey_date="synthetic_enriched")
    base = base.drop(columns=[column for column in base.columns if column.startswith("actual_")], errors="ignore")
    base["respondent_id"] = [f"synthetic_resp_{idx + 1:04d}" for idx in range(base.shape[0])]
    base["survey_source"] = PANEL_KIND
    return base


def _synthetic_targets(profile: dict[str, Any], env: pd.Series, respondent_position: int) -> dict[str, float]:
    idio = np.sin((respondent_position + 1) * 1.618 + int(env["period_index"]) * 0.73)
    inflation_spread = max(float(env.get("inflation_iqr", 1.5)) / 1.35, 0.35)
    inflation = (
        float(env["aggregate_expected_inflation_1y"])
        + _profile_effect(profile, "expected_inflation_1y")
        + 0.28 * idio * inflation_spread
    )
    unemployment = (
        float(env["aggregate_expected_unemployment_rate"])
        + _profile_effect(profile, "expected_unemployment_rate")
        + 0.18 * np.cos((respondent_position + 3) * 1.11 + int(env["period_index"]))
    )
    income = (
        float(env["aggregate_expected_real_income_growth"])
        + _profile_effect(profile, "expected_real_income_growth")
        + 0.22 * np.sin((respondent_position + 5) * 0.91 - int(env["period_index"]))
    )
    return {
        "expected_inflation_1y": round(float(np.clip(inflation, -5.0, 20.0)), 4),
        "expected_unemployment_rate": round(float(np.clip(unemployment, 0.0, 35.0)), 4),
        "expected_real_income_growth": round(float(np.clip(income, -20.0, 20.0)), 4),
    }


def _profile_effect(profile: dict[str, Any], target: str) -> float:
    income = str(profile.get("income_group", "middle"))
    education = str(profile.get("education_group", "some_college"))
    age = str(profile.get("age_group", "35_54"))
    gender = str(profile.get("gender", "unknown"))
    employment = str(profile.get("employment_status", "employed"))
    liquid = str(profile.get("liquid_wealth_group", "middle"))
    if target == "expected_inflation_1y":
        return (
            {"low": 0.65, "middle": 0.10, "high": -0.25}.get(income, 0.0)
            + {"high_school_or_less": 0.28, "some_college": 0.08, "college_plus": -0.12}.get(education, 0.0)
            + {"18_34": -0.05, "35_54": 0.05, "55_plus": 0.18}.get(age, 0.0)
            + (0.10 if gender == "female" else 0.0)
            + (0.18 if liquid == "low" else 0.0)
        )
    if target == "expected_unemployment_rate":
        return (
            {"low": 0.55, "middle": 0.08, "high": -0.22}.get(income, 0.0)
            + {"unemployed": 1.45, "not_in_labor_force": 0.35, "retired": 0.15}.get(employment, 0.0)
            + (0.20 if liquid == "low" else 0.0)
        )
    if target == "expected_real_income_growth":
        return (
            {"low": -0.42, "middle": 0.02, "high": 0.38}.get(income, 0.0)
            + {"college_plus": 0.20, "some_college": 0.03, "high_school_or_less": -0.16}.get(education, 0.0)
            + {"unemployed": -1.10, "retired": -0.35}.get(employment, 0.0)
            - (0.15 if liquid == "low" else 0.0)
        )
    return 0.0


def _initial_prior(target: str, profile: dict[str, Any], env: pd.Series) -> float:
    aggregate = {
        "expected_inflation_1y": "aggregate_expected_inflation_1y",
        "expected_unemployment_rate": "aggregate_expected_unemployment_rate",
        "expected_real_income_growth": "aggregate_expected_real_income_growth",
    }[target]
    return round(float(env[aggregate]) + 0.5 * _profile_effect(profile, target), 4)


def _load_survey_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Survey belief target file not found: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).copy()
    if frame.empty:
        raise ValueError(f"Survey belief target file has no usable rows: {path}")
    return frame


def _select_origins(path: Path, *, start_as_of: str, period_count: int) -> pd.DataFrame:
    origins = _load_forecast_origins(path)
    filtered = origins[origins["as_of_date"] >= pd.Timestamp(start_as_of)].copy()
    if filtered.empty:
        raise ValueError(f"No forecast origins at or after {start_as_of}")
    return filtered.head(period_count).reset_index(drop=True)


def _load_forecast_origins(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Forecast origins file not found: {path}")
    origins = pd.read_csv(path)
    origins["as_of_date"] = pd.to_datetime(origins["as_of_date"], errors="coerce")
    origins = origins.dropna(subset=["origin", "as_of_date"]).sort_values("as_of_date")
    if origins.empty:
        raise ValueError(f"Forecast origins file has no usable rows: {path}")
    return origins.reset_index(drop=True)


def _match_origins_to_wave_dates(origins: pd.DataFrame, wave_dates: list[Any]) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for wave_date in wave_dates:
        wave_ts = pd.Timestamp(wave_date)
        prior = origins[origins["as_of_date"].le(wave_ts)].sort_values("as_of_date")
        if prior.empty:
            prior = origins[origins["as_of_date"].ge(wave_ts)].sort_values("as_of_date")
        if prior.empty:
            raise ValueError(f"No forecast origin can be matched to SCE wave {wave_ts.date().isoformat()}")
        rows.append(prior.iloc[-1] if prior["as_of_date"].le(wave_ts).any() else prior.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_vintage_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FRED vintage context file not found: {path}")
    frame = pd.read_csv(path)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["origin", "series_id", "observation_date", "value"]).copy()


def _survey_beliefs_as_of(survey: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    def latest(target_name: str, horizon: int = 12) -> float | None:
        eligible = survey[
            survey["date"].le(as_of)
            & survey["target_name"].eq(target_name)
            & survey["horizon_months"].astype(int).eq(horizon)
        ].sort_values("date")
        if eligible.empty:
            return None
        return float(eligible.iloc[-1]["value"])

    median = (
        latest("median_expected_inflation")
        or latest("median_point_prediction_inflation")
        or latest("median_expected_price_change_next_12_months")
        or 3.0
    )
    p25 = latest("p25_expected_inflation") or (median - 1.0)
    p75 = latest("p75_expected_inflation") or (median + 1.0)
    uncertainty = latest("median_inflation_uncertainty") or max(0.8, (p75 - p25) / 1.35)
    return {
        "median_inflation": float(median),
        "inflation_iqr": float(max(0.2, p75 - p25)),
        "inflation_uncertainty": float(max(0.1, uncertainty)),
    }


def _latest_series(vintage: pd.DataFrame, series_id: str, *, fallback: float) -> float:
    series = vintage[vintage["series_id"].eq(series_id)].sort_values("observation_date")
    if series.empty:
        return float(fallback)
    return float(series.iloc[-1]["value"])


def _series_growth(vintage: pd.DataFrame, series_id: str, *, periods: int, fallback: float) -> float:
    series = vintage[vintage["series_id"].eq(series_id)].sort_values("observation_date")
    if series.shape[0] < 2:
        return float(fallback)
    latest = series.iloc[-1]
    offset_months = 12 if periods == 12 else 12
    prior_date = pd.Timestamp(latest["observation_date"]) - pd.DateOffset(months=offset_months)
    prior = series[series["observation_date"].le(prior_date)]
    if prior.empty:
        return float(fallback)
    prior_value = float(prior.iloc[-1]["value"])
    latest_value = float(latest["value"])
    if prior_value <= 0:
        return float(fallback)
    return float(100.0 * (latest_value / prior_value - 1.0))


def _safe_relative(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return str(resolved.relative_to(cwd)) if resolved.is_relative_to(cwd) else resolved.name


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
