"""Build and maintain a fixed, leakage-safe household cohort.

The cohort is selected once from the April 2025 wave and its immediately
preceding March response.  Later waves are matched to that fixed registry;
they can never change cohort membership or rewrite a simulated state.
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

from .prepare_dynamic_macro_panel import (
    BELIEF_COLUMNS,
    CURRENT_TO_PRIOR,
    DEFAULT_PUBLICATION_LAG_MONTHS,
    DEFAULT_SAMPLE_SEED,
    OPTIONAL_DEMOGRAPHIC_COLUMNS,
    PERSONAL_JOB_LOSS_COLUMN,
    PRIOR_COLUMNS,
    PRIOR_PERSONAL_JOB_LOSS_COLUMN,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    STRATIFICATION_COLUMNS,
    _availability_date,
    _build_initial_households,
    _month_start,
    _month_text,
    _stratified_wave_sample,
)


PERSISTENT_HOUSEHOLD_SCHEMA_VERSION = "persistent_household_cohort_v2"
DEFAULT_COHORT_EVENT_DATE = "2025-04-01"
DEFAULT_MASTER_COHORT_SIZE = 200
DEFAULT_CORE_COHORT_SIZE = 81
PRIVATE_REGISTRY_COLUMNS = (
    "household_id",
    "raw_respondent_id",
    "included_in_master_200",
    "included_in_core_81",
)
HISTORY_COLUMNS = (
    "household_id",
    "event_date",
    "public_availability_date",
    "source_name",
    "observation_status",
    "responded",
    "attrition_status",
    "death_status",
    "replay_required_from_event_date",
    "survey_weight",
    *STRATIFICATION_COLUMNS,
    *OPTIONAL_DEMOGRAPHIC_COLUMNS,
    *BELIEF_COLUMNS,
    *PRIOR_COLUMNS,
    PERSONAL_JOB_LOSS_COLUMN,
)


class PersistentHouseholdError(ValueError):
    """Raised when the fixed cohort or its append-only history is malformed."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def derived_initial_seed(
    sample_seed: int = DEFAULT_SAMPLE_SEED, *, initial_role: str = "sealed_test"
) -> int:
    """Return the documented deterministic seed used for the April cohort draw."""

    return (
        int(_digest("initial-wave-seed", int(sample_seed), initial_role)[:15], 16)
        % (2**31 - 1)
    )


def stable_household_id(raw_respondent_id: str, *, sample_seed: int = DEFAULT_SAMPLE_SEED) -> str:
    """Return an identity independent of cohort role, size, and input ordering."""

    # ``sample_seed`` remains accepted for backwards-compatible callers, but
    # deliberately does not influence public identity.  It only controls the
    # cohort draw; changing a draw must not rename an already-known household.
    del sample_seed
    return "household_" + _digest("persistent-household-identity-v1", str(raw_respondent_id))[:20]


def _cohort_stratum(frame: pd.DataFrame) -> pd.Series:
    return frame[list(STRATIFICATION_COLUMNS)].astype(str).agg("|".join, axis=1)


def _weight_metrics(weights: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(weights, errors="coerce").astype(float)
    if clean.empty or not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any():
        raise PersistentHouseholdError("cohort weights must be positive finite values")
    total = float(clean.sum())
    return {
        "population_weight_sum": total,
        "effective_sample_size": float(total**2 / float((clean**2).sum())),
        "max_population_weight": float(clean.max()),
    }


def _add_cohort_weights(selected: pd.DataFrame, population: pd.DataFrame, *, suffix: str) -> pd.DataFrame:
    """Apply survey_weight * N_h / n_h and normalize it inside one cohort."""

    out = selected.copy()
    population_counts = _cohort_stratum(population).value_counts().sort_index()
    sample_counts = _cohort_stratum(out).value_counts().sort_index()
    out["cohort_stratum"] = _cohort_stratum(out)
    out[f"stratum_population_count_{suffix}"] = out["cohort_stratum"].map(population_counts).astype(int)
    out[f"stratum_sample_count_{suffix}"] = out["cohort_stratum"].map(sample_counts).astype(int)
    out[f"inclusion_probability_{suffix}"] = (
        out[f"stratum_sample_count_{suffix}"] / out[f"stratum_population_count_{suffix}"]
    )
    out[f"selection_weight_{suffix}"] = (
        out["survey_weight"]
        * out[f"stratum_population_count_{suffix}"]
        / out[f"stratum_sample_count_{suffix}"]
    )
    selection_total = float(out[f"selection_weight_{suffix}"].sum())
    if not math.isfinite(selection_total) or selection_total <= 0:
        raise PersistentHouseholdError("selection weights must sum to a positive finite value")
    out[f"population_weight_{suffix}"] = out[f"selection_weight_{suffix}"] / selection_total
    return out


def _cohort_source_at_event_and_prior(
    frame: pd.DataFrame,
    *,
    event_month: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalize only the event wave and its immediate predecessor.

    This intentionally does not call the whole-panel validator: a malformed
    later response must not change whether a household was eligible in April.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("SCE input must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise PersistentHouseholdError(f"normalized SCE input missing required fields: {', '.join(missing)}")
    source = frame.copy()
    source["respondent_id"] = source["respondent_id"].astype(str).str.strip()
    source["_survey_month"] = pd.to_datetime(source["survey_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    prior_month = event_month - pd.offsets.MonthBegin(1)
    selected = source[source["_survey_month"].isin({event_month, prior_month})].copy()
    if selected["_survey_month"].isna().any():
        raise PersistentHouseholdError("survey_date contains invalid event or immediate-prior dates")
    if selected["respondent_id"].eq("").any():
        raise PersistentHouseholdError("respondent_id contains blank event or immediate-prior values")
    if selected.duplicated(["respondent_id", "_survey_month"]).any():
        raise PersistentHouseholdError("duplicate respondent_id+survey_date grain in event or immediate-prior wave")
    numeric_columns = ["weight", *BELIEF_COLUMNS]
    if PERSONAL_JOB_LOSS_COLUMN in selected:
        numeric_columns.append(PERSONAL_JOB_LOSS_COLUMN)
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    april = selected[selected["_survey_month"].eq(event_month)].copy()
    march = selected[selected["_survey_month"].eq(prior_month)].copy()
    if april.empty:
        raise PersistentHouseholdError(f"no respondents are observed at cohort event date {_month_text(event_month)}")
    if march.empty:
        raise PersistentHouseholdError(f"no respondents are observed at immediate prior date {_month_text(prior_month)}")
    for column in STRATIFICATION_COLUMNS:
        if april[column].isna().any() or april[column].astype(str).str.strip().eq("").any():
            raise PersistentHouseholdError(f"{column} contains blank event-wave values")
    if not np.isfinite(april[["weight", *BELIEF_COLUMNS]].to_numpy(dtype=float)).all() or (april["weight"] <= 0).any():
        raise PersistentHouseholdError("event-wave survey weights and beliefs must be finite, with positive weights")
    return april, march


def eligible_april_2025_with_march_prior(
    frame: pd.DataFrame,
    *,
    cohort_event_date: str | pd.Timestamp = DEFAULT_COHORT_EVENT_DATE,
) -> pd.DataFrame:
    """Return April respondents observed in the immediately preceding March wave.

    The result deliberately does not inspect any later wave, completion status,
    or outcome.  The March values become the April prior-belief fields.
    """

    event_month = _month_start(cohort_event_date, field="cohort event date")
    april, march = _cohort_source_at_event_and_prior(frame, event_month=event_month)
    prior_source_columns = [
        "respondent_id",
        "weight",
        *STRATIFICATION_COLUMNS,
        *[
            column
            for column in OPTIONAL_DEMOGRAPHIC_COLUMNS
            if column in march.columns and column not in STRATIFICATION_COLUMNS
        ],
        *BELIEF_COLUMNS,
        *([PERSONAL_JOB_LOSS_COLUMN] if PERSONAL_JOB_LOSS_COLUMN in march else []),
    ]
    prior_rename = {
        "weight": "prior_survey_weight",
        **CURRENT_TO_PRIOR,
        **(
            {PERSONAL_JOB_LOSS_COLUMN: PRIOR_PERSONAL_JOB_LOSS_COLUMN}
            if PERSONAL_JOB_LOSS_COLUMN in prior_source_columns
            else {}
        ),
        **{
            column: f"prior_{column}"
            for column in prior_source_columns
            if column
            not in {
                "respondent_id",
                "weight",
                *BELIEF_COLUMNS,
                PERSONAL_JOB_LOSS_COLUMN,
            }
        },
    }
    prior = march[prior_source_columns].rename(columns=prior_rename)
    eligible = april.merge(prior, on="respondent_id", how="inner", validate="one_to_one")
    finite = np.isfinite(eligible[list(PRIOR_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    eligible = eligible.loc[finite].copy()
    if eligible.empty:
        raise PersistentHouseholdError("no April respondents have a finite immediate March prior")
    return eligible.sort_values("respondent_id").reset_index(drop=True)


def _public_registry(master: pd.DataFrame, core_ids: set[str]) -> pd.DataFrame:
    registry = master[
        [
            "household_id",
            "survey_weight",
            "cohort_stratum",
            *STRATIFICATION_COLUMNS,
            "stratum_population_count_200",
            "stratum_sample_count_200",
            "inclusion_probability_200",
            "selection_weight_200",
            "population_weight_200",
        ]
    ].copy()
    registry["included_in_master_200"] = True
    registry["included_in_core_81"] = registry["household_id"].isin(core_ids)
    registry["selection_rule"] = "april_2025_observed_with_immediate_march_2025_prior"
    registry["identity_schema_version"] = PERSISTENT_HOUSEHOLD_SCHEMA_VERSION
    return registry.sort_values("household_id").reset_index(drop=True)


def _initial_states(selected: pd.DataFrame, *, cohort_name: str, publication_lag_months: int) -> pd.DataFrame:
    event_month = pd.Timestamp(selected["_survey_month"].iloc[0])
    prepared = selected.copy()
    prepared["respondent_id"] = prepared["household_id"]
    prepared["weight"] = prepared["population_weight"]
    prepared["_split_role"] = cohort_name
    prepared["survey_event_date"] = _month_text(event_month)
    prepared["estimated_public_availability_date"] = _month_text(
        _availability_date(event_month, publication_lag_months)
    )
    prepared["contamination_label"] = "source_availability_recorded_separately"
    for column in OPTIONAL_DEMOGRAPHIC_COLUMNS:
        if column not in prepared:
            prepared[column] = "unknown"
    states = _build_initial_households(prepared)
    included = selected[
        [
            "household_id",
            "survey_weight",
            "cohort_stratum",
            "inclusion_probability",
            "selection_weight",
        ]
    ].rename(columns={"household_id": "type_id"})
    states = states.merge(included, on="type_id", how="left", validate="one_to_one")
    states["cohort_name"] = cohort_name
    states["balance_sheet_source"] = (
        "coarse_synthetic_mapping_from_income_and_liquidity_groups_not_observed_balances"
    )
    states["liquid_wealth_measurement"] = (
        "coarse_group_proxy_derived_from_income_category_in_normalized_SCE_input"
    )
    return states.sort_values("type_id").reset_index(drop=True)


def _initial_observed_history(
    master: pd.DataFrame,
    *,
    event_month: pd.Timestamp,
    publication_lag_months: int,
    source_name: str,
) -> pd.DataFrame:
    """Materialize both the March prior and April cohort observation."""

    common = {
        "source_name": source_name,
        "observation_status": "observed",
        "responded": True,
        "attrition_status": "responding",
        "death_status": "alive_no_death_observation",
    }
    april = master.copy()
    april["event_date"] = _month_text(event_month)
    april["public_availability_date"] = _month_text(
        _availability_date(event_month, publication_lag_months)
    )
    april["replay_required_from_event_date"] = _month_text(event_month)
    april = april.assign(**common)

    prior_month = event_month - pd.offsets.MonthBegin(1)
    march = pd.DataFrame({"household_id": master["household_id"]})
    march["survey_weight"] = master["prior_survey_weight"]
    for column in (*STRATIFICATION_COLUMNS, *OPTIONAL_DEMOGRAPHIC_COLUMNS):
        prior_column = f"prior_{column}"
        if prior_column in master:
            march[column] = master[prior_column]
        elif column in master:
            march[column] = master[column]
    for current, prior_column in CURRENT_TO_PRIOR.items():
        march[current] = master[prior_column]
    if PRIOR_PERSONAL_JOB_LOSS_COLUMN in master:
        march[PERSONAL_JOB_LOSS_COLUMN] = master[PRIOR_PERSONAL_JOB_LOSS_COLUMN]
    march["event_date"] = _month_text(prior_month)
    march["public_availability_date"] = _month_text(
        _availability_date(prior_month, publication_lag_months)
    )
    march["replay_required_from_event_date"] = _month_text(prior_month)
    march = march.assign(**common)

    return (
        pd.concat([_history_columns(march), _history_columns(april)], ignore_index=True)
        .sort_values(["event_date", "household_id"])
        .reset_index(drop=True)
    )


def prepare_household_scale_cohorts(
    frame: pd.DataFrame,
    *,
    cohort_event_date: str | pd.Timestamp = DEFAULT_COHORT_EVENT_DATE,
    master_sample_size: int = DEFAULT_MASTER_COHORT_SIZE,
    core_sample_size: int = DEFAULT_CORE_COHORT_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
    source_name: str = "SCE respondent microdata",
) -> dict[str, Any]:
    """Prepare a fixed 200-person registry and its nested 81-person core."""

    if master_sample_size <= 0 or core_sample_size <= 0 or core_sample_size > master_sample_size:
        raise PersistentHouseholdError("cohort sizes must be positive and core_sample_size must not exceed master_sample_size")
    if publication_lag_months < 0:
        raise PersistentHouseholdError("publication lag must be non-negative")
    eligible = eligible_april_2025_with_march_prior(frame, cohort_event_date=cohort_event_date)
    if eligible["respondent_id"].nunique() < master_sample_size:
        raise PersistentHouseholdError(
            f"insufficient April-with-March-prior eligible respondents: need {master_sample_size}, found {eligible['respondent_id'].nunique()}"
        )
    initial_seed = derived_initial_seed(sample_seed)
    # The historical 81-person sample is the same-seed direct draw from the
    # eligible population.  The 200-person draw uses the same ranked strata,
    # which preserves all 81 members while adding 119 respondents.
    core, core_metadata = _stratified_wave_sample(
        eligible, sample_size=core_sample_size, sample_seed=initial_seed
    )
    master, master_metadata = _stratified_wave_sample(
        eligible, sample_size=master_sample_size, sample_seed=initial_seed
    )
    master = master.rename(columns={"weight": "survey_weight"}).copy()
    master["household_id"] = master["respondent_id"].map(lambda raw: stable_household_id(raw, sample_seed=sample_seed))
    if master["household_id"].duplicated().any():
        raise PersistentHouseholdError("stable household identity collision")
    master = _add_cohort_weights(master, eligible, suffix="200")

    if not set(core["respondent_id"].astype(str)).issubset(
        set(master["respondent_id"].astype(str))
    ):
        raise PersistentHouseholdError(
            "same-seed stratified master cohort does not preserve the historical core"
        )
    core = core.rename(columns={"weight": "survey_weight"}).copy()
    core["household_id"] = core["respondent_id"].map(stable_household_id)
    core = _add_cohort_weights(core, eligible, suffix="81")
    core_ids = set(core["household_id"])
    registry = _public_registry(master, core_ids)
    core_weighted = core[
        [
            "household_id",
            "stratum_population_count_81",
            "stratum_sample_count_81",
            "inclusion_probability_81",
            "selection_weight_81",
            "population_weight_81",
        ]
    ]
    registry = registry.merge(core_weighted, on="household_id", how="left", validate="one_to_one")
    private_registry = master[["household_id", "respondent_id"]].rename(columns={"respondent_id": "raw_respondent_id"})
    private_registry["included_in_master_200"] = True
    private_registry["included_in_core_81"] = private_registry["household_id"].isin(core_ids)
    private_registry = private_registry.loc[:, PRIVATE_REGISTRY_COLUMNS].sort_values("household_id").reset_index(drop=True)

    event_month = _month_start(cohort_event_date, field="cohort event date")
    public_date = _availability_date(event_month, publication_lag_months)
    history = _initial_observed_history(
        master,
        event_month=event_month,
        publication_lag_months=publication_lag_months,
        source_name=source_name,
    )
    provenance = registry[["household_id", "included_in_master_200", "included_in_core_81"]].copy()
    provenance["source_name"] = source_name
    provenance["source_event_date"] = _month_text(event_month)
    provenance["source_public_availability_date"] = _month_text(public_date)
    provenance["selection_eligibility"] = "observed_april_2025_and_immediate_march_2025_prior"
    provenance["selection_did_not_use_future_completion"] = True
    schedule = pd.DataFrame(
        [
            {
                "source_name": source_name,
                "event_date": _month_text(month),
                "public_availability_date": _month_text(
                    _availability_date(month, publication_lag_months)
                ),
                "selected_households": int(master.shape[0]),
                "core_households": int(core.shape[0]),
                "information_rule": "admit observations only on or after public_availability_date",
            }
            for month in (event_month - pd.offsets.MonthBegin(1), event_month)
        ]
    )

    master_states = master.copy()
    master_states["population_weight"] = master_states["population_weight_200"]
    master_states["selection_weight"] = master_states["selection_weight_200"]
    master_states["inclusion_probability"] = master_states["inclusion_probability_200"]
    core_states = core.copy()
    core_states["population_weight"] = core_states["population_weight_81"]
    core_states["selection_weight"] = core_states["selection_weight_81"]
    core_states["inclusion_probability"] = core_states["inclusion_probability_81"]
    initial_200 = _initial_states(master_states, cohort_name="master_200", publication_lag_months=publication_lag_months)
    initial_81 = _initial_states(core_states, cohort_name="core_81", publication_lag_months=publication_lag_months)
    metrics = {
        "master_200": _weight_metrics(master_states["population_weight"]),
        "core_81": _weight_metrics(core_states["population_weight"]),
    }
    manifest = {
        "schema_version": PERSISTENT_HOUSEHOLD_SCHEMA_VERSION,
        "cohort_event_date": _month_text(event_month),
        "immediate_prior_event_date": _month_text(event_month - pd.offsets.MonthBegin(1)),
        "sample_seed": int(sample_seed),
        "derived_initial_seed": initial_seed,
        "derived_core_seed": initial_seed,
        "selection_rule": "same-seed deterministic income x age x education ranking; historical core_81 is nested in master_200",
        "future_completion_used_for_selection": False,
        "master_sample": master_metadata,
        "core_sample": core_metadata,
        "weight_formula": "survey_weight * N_h / n_h; population_weight normalizes selection_weight within cohort",
        "weight_population": "all April-2025 respondents with an immediate finite March-2025 prior",
        "core_membership_sha256": _digest(*sorted(core["household_id"].astype(str))),
        "master_membership_sha256": _digest(*sorted(master["household_id"].astype(str))),
        "balance_sheet_provenance": "coarse synthetic mappings from survey groups; balances are not respondent-observed",
        "weight_metrics": metrics,
        "counts": {"eligible": int(eligible.shape[0]), "master_200": int(master.shape[0]), "core_81": int(core.shape[0])},
    }
    return {
        "identity_registry": registry.sort_values("household_id").reset_index(drop=True),
        "private_registry": private_registry,
        "source_provenance": provenance.sort_values("household_id").reset_index(drop=True),
        "observed_history": history.sort_values(["event_date", "household_id"]).reset_index(drop=True),
        "public_information_schedule": schedule,
        "initial_households_200": initial_200,
        "initial_households_81": initial_81,
        "manifest": manifest,
    }


def _history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out.loc[:, HISTORY_COLUMNS]


def append_selected_observed_history(
    history: pd.DataFrame,
    observations: pd.DataFrame,
    private_registry: pd.DataFrame,
    *,
    event_date: str | pd.Timestamp,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
    source_name: str = "SCE respondent microdata",
) -> dict[str, Any]:
    """Append a new wave while preserving selected nonrespondents.

    This helper never rewrites a simulated state.  Its return value explicitly
    tells the caller to replay from the new event date after it has persisted
    the returned history.
    """

    event_month = _month_start(event_date, field="event date")
    if publication_lag_months < 0:
        raise PersistentHouseholdError("publication lag must be non-negative")
    required_private = set(PRIVATE_REGISTRY_COLUMNS)
    if missing := sorted(required_private - set(private_registry.columns)):
        raise PersistentHouseholdError(f"private registry missing fields: {', '.join(missing)}")
    if private_registry["raw_respondent_id"].duplicated().any() or private_registry["household_id"].duplicated().any():
        raise PersistentHouseholdError("private registry must map raw IDs one-to-one to household IDs")
    if "respondent_id" not in observations:
        raise PersistentHouseholdError("observations must include respondent_id for private-registry matching")
    observed = observations.copy()
    observed["respondent_id"] = observed["respondent_id"].astype(str).str.strip()
    if "survey_date" in observed:
        observed_months = (
            pd.to_datetime(observed["survey_date"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        if observed_months.isna().any() or not observed_months.eq(event_month).all():
            raise PersistentHouseholdError(
                "observation survey_date does not match the append event date"
            )
    if observed["respondent_id"].duplicated().any():
        raise PersistentHouseholdError("duplicate observation for raw respondent at one event date")
    selected = private_registry[private_registry["included_in_master_200"].astype(bool)].copy()
    raw_to_household = dict(zip(selected["raw_respondent_id"], selected["household_id"]))
    unselected_observation_count = int(
        (~observed["respondent_id"].isin(raw_to_household)).sum()
    )
    observed = observed[observed["respondent_id"].isin(raw_to_household)].copy()
    history = _history_columns(history)
    event_text = _month_text(event_month)
    if history["event_date"].astype(str).eq(event_text).any():
        raise PersistentHouseholdError("duplicate or overwriting observation event is not allowed in append-only history")
    observed["household_id"] = observed["respondent_id"].map(raw_to_household)
    if "survey_weight" not in observed and "weight" in observed:
        observed["survey_weight"] = pd.to_numeric(observed["weight"], errors="coerce")
    observed["event_date"] = event_text
    observed["public_availability_date"] = _month_text(_availability_date(event_month, publication_lag_months))
    observed["source_name"] = source_name
    observed["observation_status"] = "observed"
    observed["responded"] = True
    observed["attrition_status"] = "responding"
    observed["death_status"] = "alive_no_death_observation"
    observed["replay_required_from_event_date"] = event_text
    response_ids = set(observed["household_id"])
    nonrespondents = selected.loc[~selected["household_id"].isin(response_ids), ["household_id"]].copy()
    nonrespondents["event_date"] = event_text
    nonrespondents["public_availability_date"] = _month_text(_availability_date(event_month, publication_lag_months))
    nonrespondents["source_name"] = source_name
    nonrespondents["observation_status"] = "nonresponse"
    nonrespondents["responded"] = False
    nonrespondents["attrition_status"] = "survey_nonresponse_not_economic_exit"
    nonrespondents["death_status"] = "alive_no_death_observation"
    nonrespondents["replay_required_from_event_date"] = event_text
    appended = pd.concat([_history_columns(observed), _history_columns(nonrespondents)], ignore_index=True)
    updated = pd.concat([history, appended], ignore_index=True).sort_values(["event_date", "household_id"]).reset_index(drop=True)
    replay_from = event_text if not observed.empty else None
    return {
        "observed_history": updated,
        "appended_history": appended.sort_values("household_id").reset_index(drop=True),
        "replay_required_from_event_date": replay_from,
        "simulated_state_overwritten": False,
        "matched_observation_count": int(observed.shape[0]),
        "unselected_observation_count": unselected_observation_count,
    }


append_household_observations = append_selected_observed_history


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_household_scale_cohorts(
    input_csv: Path | str,
    *,
    output_dir: Path | str,
    private_output_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write public artifacts plus a separate raw-ID registry."""

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"normalized SCE input not found: {input_path}")
    result = prepare_household_scale_cohorts(pd.read_csv(input_path), **kwargs)
    output_root = Path(output_dir)
    private_root = Path(private_output_dir) if private_output_dir is not None else output_root / "private"
    output_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "identity_registry_csv": (output_root / "household_identity_registry.csv", result["identity_registry"]),
        "source_provenance_csv": (output_root / "household_source_provenance.csv", result["source_provenance"]),
        "observed_history_csv": (output_root / "selected_observed_history.csv", result["observed_history"]),
        "public_information_schedule_csv": (output_root / "public_information_schedule.csv", result["public_information_schedule"]),
        "initial_households_200_csv": (output_root / "initial_households_200.csv", result["initial_households_200"]),
        "initial_households_81_csv": (output_root / "initial_households_81.csv", result["initial_households_81"]),
        "private_registry_csv": (private_root / "household_private_registry.csv", result["private_registry"]),
    }
    for path, artifact in artifacts.values():
        artifact.to_csv(path, index=False)
    manifest = dict(result["manifest"])
    manifest["input"] = {"path": str(input_path), "sha256": _sha256(input_path)}
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path), "row_count": int(artifact.shape[0])}
        for name, (path, artifact) in artifacts.items()
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, **{name: path for name, (path, _) in artifacts.items()}, "manifest_path": manifest_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a persistent 200/81 April-2025 household cohort.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, default=None)
    parser.add_argument("--cohort-event-date", default=DEFAULT_COHORT_EVENT_DATE)
    parser.add_argument("--master-sample-size", type=int, default=DEFAULT_MASTER_COHORT_SIZE)
    parser.add_argument("--core-sample-size", type=int, default=DEFAULT_CORE_COHORT_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_household_scale_cohorts(
            args.input_csv,
            output_dir=args.output_dir,
            private_output_dir=args.private_output_dir,
            cohort_event_date=args.cohort_event_date,
            master_sample_size=args.master_sample_size,
            core_sample_size=args.core_sample_size,
            sample_seed=args.sample_seed,
        )
    except (PersistentHouseholdError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(result["identity_registry_csv"])
    print(result["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
