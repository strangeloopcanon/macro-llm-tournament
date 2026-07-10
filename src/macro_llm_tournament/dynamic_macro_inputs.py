"""Frozen-bundle, household, and behavior-profile preparation for macro runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .demand_economy import (
    build_hybrid_behavior_policy_profile,
    load_behavior_policy_profile,
    load_empirical_bridge_profile,
    load_state_behavior_policy_profile,
    normalize_demand_households,
)
from .dynamic_macro_common import (
    BundleView,
    DynamicMacroError,
    FORBIDDEN_PROMPT_PREFIXES,
    FORBIDDEN_PROMPT_KEYS,
    _first_column,
    _first_finite,
    _finite_or,
    _group,
    _monthly_rate_to_annual_pct,
    _optional_signal,
    _sha256_json,
)
from .frozen_vintage_bundle import load_frozen_vintage_bundle


CONTAMINATION_LABELS = (
    "post_cutoff_holdout",
    "potential_training_contamination",
    "pre_cutoff_observation_post_cutoff_release",
)
CONTAMINATION_POLICIES = (
    "unavailable_at_cutoff",
    "strict_post_cutoff_event",
    "all",
)
REQUIRED_FAMILIES = frozenset(
    {"demand", "balance_sheet", "labor", "prices", "income_policy"}
)


@dataclass(frozen=True)
class TargetMapping:
    series_id: str
    economy_measure: str
    economy_transform: str
    description: str


MAPPING_BY_SERIES: dict[str, TargetMapping] = {
    "PCE": TargetMapping(
        "PCE",
        "aggregate_consumption",
        "nominal_pct_change",
        "Monthly real consumption growth compounded with the model monthly price rate.",
    ),
    "PCEC96": TargetMapping(
        "PCEC96",
        "aggregate_consumption",
        "pct_change",
        "Monthly percentage change in aggregate real consumption.",
    ),
    "RSAFS": TargetMapping(
        "RSAFS",
        "aggregate_consumption",
        "pct_change",
        "Monthly percentage change in aggregate consumption demand.",
    ),
    "PSAVERT": TargetMapping(
        "PSAVERT",
        "saving_rate_pct",
        "diff",
        "Monthly change in aggregate saving as a percent of aggregate income.",
    ),
    "REVOLSL": TargetMapping(
        "REVOLSL",
        "aggregate_debt",
        "pct_change",
        "Monthly percentage change in the household debt stock.",
    ),
    "PAYEMS": TargetMapping(
        "PAYEMS",
        "next_employment_rate",
        "pct_change",
        "Monthly percentage change in post-transition model employment.",
    ),
    "UNRATE": TargetMapping(
        "UNRATE",
        "next_unemployment_rate_pct",
        "level",
        "One minus post-transition model employment, expressed in percent.",
    ),
    "PCEPI": TargetMapping(
        "PCEPI",
        "next_inflation_rate",
        "annual_rate_to_monthly_pct",
        "Post-transition annualized inflation converted to an exact monthly percentage rate.",
    ),
    "DSPIC96": TargetMapping(
        "DSPIC96",
        "next_aggregate_income",
        "pct_change",
        "Monthly percentage change in post-transition aggregate real labor income.",
    ),
    "FEDFUNDS": TargetMapping(
        "FEDFUNDS",
        "next_policy_rate",
        "level",
        "Post-transition model policy-rate level in percent.",
    ),
}


def load_bundle_view(bundle_dir: Path) -> BundleView:
    loaded = load_frozen_vintage_bundle(bundle_dir)
    manifest = dict(loaded.manifest)
    origins = tuple(_normalize_origin(row) for row in loaded.origins)
    assert_contiguous_origins(origins)
    history = tuple(_normalize_history(row) for row in loaded.history)
    raw_specs = manifest.get("target_specs") or []
    specs_by_name = {
        str(row["target_name"]): dict(row) for row in raw_specs if isinstance(row, dict)
    }
    targets = tuple(_normalize_target(row, specs_by_name) for row in loaded.targets)
    target_specs = tuple(_target_specs_from_rows(targets))
    target_contamination = tuple(
        _normalize_contamination(row) for row in loaded.target_contamination
    )
    families = {str(row["family"]) for row in target_specs}
    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        raise DynamicMacroError(
            f"Frozen bundle does not cover broad target families: {', '.join(sorted(missing_families))}"
        )
    unsupported = sorted(
        {str(row["series_id"]) for row in target_specs} - set(MAPPING_BY_SERIES)
    )
    if unsupported:
        raise DynamicMacroError(
            f"Incomplete target coverage; no economy mapping for: {', '.join(unsupported)}"
        )
    expected = {
        (row["origin_month"], spec["target_name"])
        for row in origins
        for spec in target_specs
    }
    actual = {(row["origin_month"], row["target_name"]) for row in targets}
    if len(targets) != len(actual) or actual != expected:
        raise DynamicMacroError("Incomplete target coverage across frozen origins")
    history_origins = {row["origin_month"] for row in history}
    missing_history = [
        row["origin_month"]
        for row in origins
        if row["origin_month"] not in history_origins
    ]
    if missing_history:
        raise DynamicMacroError(
            f"Missing periods in origin-visible history: {', '.join(missing_history)}"
        )
    bundle_sha = str(manifest.get("bundle_sha256", ""))
    if not bundle_sha:
        raise DynamicMacroError("Validated bundle manifest has no bundle_sha256")
    return BundleView(
        bundle_sha,
        origins,
        history,
        targets,
        target_specs,
        target_contamination,
    )


def filter_bundle_targets(
    bundle: BundleView,
    *,
    model: str,
    policy: str,
) -> tuple[BundleView, dict[str, Any]]:
    if policy not in CONTAMINATION_POLICIES:
        raise DynamicMacroError(f"Unsupported contamination policy: {policy}")
    model_rows = [row for row in bundle.target_contamination if row["model"] == model]
    expected_keys = {
        (row["origin_month"], row["target_name"]) for row in bundle.targets
    }
    actual_keys = {(row["origin_month"], row["target_name"]) for row in model_rows}
    if len(model_rows) != len(actual_keys) or actual_keys != expected_keys:
        raise DynamicMacroError(
            f"Target-contamination coverage is incomplete for requested model {model}"
        )
    contamination_by_key = {
        (row["origin_month"], row["target_name"]): row for row in model_rows
    }
    labels_by_key = {
        key: row["contamination_label"] for key, row in contamination_by_key.items()
    }
    unavailable_labels = {
        "post_cutoff_holdout",
        "pre_cutoff_observation_post_cutoff_release",
    }
    release_after_cutoff_by_key = {
        key: date.fromisoformat(row["first_release_as_of_date"])
        > date.fromisoformat(row["model_cutoff_date"])
        for key, row in contamination_by_key.items()
    }
    for key, release_after_cutoff in release_after_cutoff_by_key.items():
        label_marks_unavailable = labels_by_key[key] in unavailable_labels
        if release_after_cutoff != label_marks_unavailable:
            raise DynamicMacroError(
                "Target-contamination label conflicts with first-release availability"
            )

    def selected_by_policy(key: tuple[str, str]) -> bool:
        if policy == "all":
            return True
        if policy == "unavailable_at_cutoff":
            return release_after_cutoff_by_key[key]
        return labels_by_key[key] == "post_cutoff_holdout"

    selected = tuple(
        row
        for row in bundle.targets
        if selected_by_policy((row["origin_month"], row["target_name"]))
    )
    if not selected:
        raise DynamicMacroError(
            f"Contamination policy {policy} selects no frozen targets for model {model}"
        )
    selected_families = {row["family"] for row in selected}
    missing_families = REQUIRED_FAMILIES - selected_families
    if missing_families:
        raise DynamicMacroError(
            "Contamination-filtered targets do not cover broad families: "
            + ", ".join(sorted(missing_families))
        )
    label_counts = pd.Series(
        list(labels_by_key.values()), dtype="object"
    ).value_counts()
    selected_key_set = {(row["origin_month"], row["target_name"]) for row in selected}
    selected_label_counts = pd.Series(
        [label for key, label in labels_by_key.items() if key in selected_key_set],
        dtype="object",
    ).value_counts()
    excluded_label_counts = pd.Series(
        [label for key, label in labels_by_key.items() if key not in selected_key_set],
        dtype="object",
    ).value_counts()
    selected_keys = sorted(
        (row["origin_month"], row["target_name"]) for row in selected
    )
    selected_counts_by_origin = pd.Series(
        [row["origin_month"] for row in selected], dtype="object"
    ).value_counts()
    targets_per_complete_origin = len(bundle.target_specs)
    complete_origins = sorted(
        str(origin)
        for origin, count in selected_counts_by_origin.items()
        if int(count) == targets_per_complete_origin
    )
    partial_origins = sorted(
        str(origin)
        for origin, count in selected_counts_by_origin.items()
        if 0 < int(count) < targets_per_complete_origin
    )
    empty_origins = sorted(
        origin["origin_month"]
        for origin in bundle.origins
        if origin["origin_month"]
        not in set(selected_counts_by_origin.index.astype(str))
    )
    coverage = {
        "model": model,
        "policy": policy,
        "available_label_counts": {
            str(label): int(count) for label, count in label_counts.sort_index().items()
        },
        "selected_label_counts": {
            str(label): int(count)
            for label, count in selected_label_counts.sort_index().items()
        },
        "excluded_label_counts": {
            str(label): int(count)
            for label, count in excluded_label_counts.sort_index().items()
        },
        "catalogue_rows": len(bundle.targets),
        "first_release_after_cutoff_rows": int(
            sum(release_after_cutoff_by_key.values())
        ),
        "selected_rows": len(selected),
        "excluded_rows": len(bundle.targets) - len(selected),
        "selected_origin_count": len({row["origin_month"] for row in selected}),
        "selected_target_count": len({row["target_name"] for row in selected}),
        "selected_family_count": len(selected_families),
        "targets_per_complete_origin": targets_per_complete_origin,
        "complete_origin_count": len(complete_origins),
        "partial_origin_count": len(partial_origins),
        "empty_origin_count": len(empty_origins),
        "complete_origins": complete_origins,
        "partial_origins": partial_origins,
        "empty_origins": empty_origins,
        "selected_pairs_sha256": _sha256_json(selected_keys),
    }
    return (
        BundleView(
            bundle.bundle_sha256,
            bundle.origins,
            bundle.history,
            selected,
            bundle.target_specs,
            bundle.target_contamination,
        ),
        coverage,
    )


def select_score_origins(
    bundle: BundleView,
    *,
    start: str | None,
    end: str | None,
) -> tuple[BundleView, dict[str, Any]]:
    available = [str(row["origin_month"]) for row in bundle.origins]
    if not available:
        raise DynamicMacroError("Cannot select score origins from an empty bundle")
    score_start = str(start or available[0])
    score_end = str(end or available[-1])
    if score_start not in available or score_end not in available:
        raise DynamicMacroError(
            f"Score-origin range {score_start}:{score_end} must use bundle origins"
        )
    start_index = available.index(score_start)
    end_index = available.index(score_end)
    if start_index > end_index:
        raise DynamicMacroError("--score-origin-start must not follow --score-origin-end")
    selected = available[start_index : end_index + 1]
    selected_set = set(selected)
    targets = tuple(
        row for row in bundle.targets if str(row["origin_month"]) in selected_set
    )
    contamination = tuple(
        row
        for row in bundle.target_contamination
        if str(row["origin_month"]) in selected_set
    )
    expected_per_origin = len(bundle.target_specs)
    counts = {
        origin: sum(str(row["origin_month"]) == origin for row in targets)
        for origin in selected
    }
    if not targets or any(count != expected_per_origin for count in counts.values()):
        raise DynamicMacroError("Score-origin range does not retain the complete target contract")
    contract = {
        "score_origin_start": score_start,
        "score_origin_end": score_end,
        "scored_origins": selected,
        "scored_origin_count": len(selected),
        "warmup_origins": available[:start_index],
        "trailing_unscored_origins": available[end_index + 1 :],
        "target_rows_per_scored_origin": expected_per_origin,
        "rule": "all_target_families_share_the_origin_observation_month",
    }
    return (
        BundleView(
            bundle.bundle_sha256,
            bundle.origins,
            bundle.history,
            targets,
            bundle.target_specs,
            contamination,
        ),
        contract,
    )


def _normalize_origin(row: Mapping[str, Any]) -> dict[str, str]:
    origin = str(row.get("origin_month") or row.get("origin_date") or "")
    as_of = str(row.get("as_of_date") or row.get("asof_date") or origin)
    if not origin or not as_of:
        raise DynamicMacroError("Bundle origin row is missing origin_month/as_of_date")
    return {"origin_month": origin, "as_of_date": as_of}


def _normalize_history(row: Mapping[str, Any]) -> dict[str, str]:
    origin = _normalize_origin(row)
    normalized: dict[str, Any] = {
        **origin,
        "series_id": str(row.get("series_id", "")),
        "observation_date": str(row.get("observation_date", "")),
        "value": str(row.get("value", "")),
    }
    try:
        value = float(normalized["value"])
    except ValueError as exc:
        raise DynamicMacroError(
            "Origin-visible history has a non-numeric value"
        ) from exc
    if (
        not normalized["series_id"]
        or not normalized["observation_date"]
        or not math.isfinite(value)
    ):
        raise DynamicMacroError("Origin-visible history row is incomplete")
    if date.fromisoformat(normalized["observation_date"]) > date.fromisoformat(
        origin["as_of_date"]
    ):
        raise DynamicMacroError(
            "History contains an observation beyond its origin as-of date"
        )
    return normalized


def _normalize_target(
    row: Mapping[str, Any], specs_by_name: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    origin = _normalize_origin(row)
    name = str(row.get("target_name", ""))
    spec = specs_by_name.get(name, {})
    normalized: dict[str, Any] = {
        **origin,
        "target_name": name,
        "series_id": str(row.get("series_id") or spec.get("series_id") or ""),
        "family": str(row.get("family") or spec.get("family") or ""),
        "transform": str(row.get("transform") or spec.get("transform") or ""),
        "default_scale": float(
            row.get("default_scale") or spec.get("default_scale") or np.nan
        ),
        "target_value": float(row.get("target_value", np.nan)),
        "origin_visible_denominator_value": float(
            row.get("origin_visible_denominator_value", np.nan)
        ),
        "target_observation_date": str(
            row.get("target_observation_date") or row.get("target_month") or ""
        ),
    }
    if (
        not normalized["target_name"]
        or not normalized["series_id"]
        or not normalized["family"]
        or normalized["transform"] not in {"pct_change", "diff", "level"}
        or not math.isfinite(normalized["default_scale"])
        or normalized["default_scale"] <= 0.0
        or not math.isfinite(normalized["target_value"])
    ):
        raise DynamicMacroError(
            f"Frozen target row is incomplete or invalid: {name or '<unnamed>'}"
        )
    return normalized


def _normalize_contamination(row: Mapping[str, Any]) -> dict[str, str]:
    origin = _normalize_origin(row)
    normalized = {
        **origin,
        "target_name": str(row.get("target_name", "")),
        "model": str(row.get("model", "")),
        "model_cutoff_date": str(row.get("model_cutoff_date", "")),
        "target_observation_date": str(row.get("target_observation_date", "")),
        "first_release_as_of_date": str(row.get("first_release_as_of_date", "")),
        "contamination_label": str(row.get("contamination_label", "")),
    }
    if (
        not normalized["target_name"]
        or not normalized["model"]
        or normalized["contamination_label"] not in CONTAMINATION_LABELS
    ):
        raise DynamicMacroError("Frozen target-contamination row is incomplete")
    try:
        for field in (
            "model_cutoff_date",
            "target_observation_date",
            "first_release_as_of_date",
        ):
            date.fromisoformat(normalized[field])
    except ValueError as exc:
        raise DynamicMacroError(
            "Frozen target-contamination row has an invalid availability date"
        ) from exc
    return normalized


def _target_specs_from_rows(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in targets:
        spec = {
            key: row[key]
            for key in (
                "target_name",
                "series_id",
                "family",
                "transform",
                "default_scale",
            )
        }
        previous = by_name.setdefault(str(row["target_name"]), spec)
        if previous != spec:
            raise DynamicMacroError(
                f"Target metadata changes across origins: {row['target_name']}"
            )
    return [by_name[name] for name in sorted(by_name)]


def assert_contiguous_origins(origins: tuple[dict[str, str], ...]) -> None:
    if not origins:
        raise DynamicMacroError("Frozen bundle has no origins")
    values = [date.fromisoformat(row["origin_month"]) for row in origins]
    if values != sorted(set(values)):
        raise DynamicMacroError("Frozen origins must be unique and sorted")
    for previous, current in zip(values, values[1:]):
        expected = date(
            previous.year + (previous.month == 12),
            1 if previous.month == 12 else previous.month + 1,
            1,
        )
        if current != expected:
            raise DynamicMacroError(
                f"Missing periods between frozen origins {previous} and {current}"
            )


def build_period_overrides(
    bundle: BundleView,
    *,
    policy_state_mode: str = "recursive",
) -> dict[int, dict[str, Any]]:
    if policy_state_mode not in {"recursive", "origin_visible"}:
        raise DynamicMacroError(f"Unsupported policy state mode: {policy_state_mode}")
    rows_by_origin: dict[str, list[dict[str, str]]] = {}
    for row in bundle.history:
        rows_by_origin.setdefault(row["origin_month"], []).append(row)
    overrides: dict[int, dict[str, Any]] = {}
    for period_index, origin in enumerate(bundle.origins):
        raw_by_series: dict[str, list[dict[str, Any]]] = {}
        latest: dict[str, dict[str, Any]] = {}
        origin_rows = sorted(
            rows_by_origin[origin["origin_month"]],
            key=lambda item: (item["series_id"], item["observation_date"]),
        )
        for row in origin_rows:
            raw_by_series.setdefault(row["series_id"], []).append(
                {
                    "observation_date": row["observation_date"],
                    "value": float(row["value"]),
                }
            )
            latest[row["series_id"]] = {
                "observation_date": row["observation_date"],
                "value": float(row["value"]),
            }
        if not latest:
            raise DynamicMacroError(
                f"Missing periods for origin {origin['origin_month']}"
            )
        raw_history = {key: raw_by_series[key] for key in sorted(raw_by_series)}
        period_override = {
            "origin_month": origin["origin_month"],
            "as_of_date": origin["as_of_date"],
            "origin_visible_macro_context": {
                key: latest[key] for key in sorted(latest)
            },
            "origin_visible_macro_history": raw_history,
            "observed_signal_summary": observed_signal_summary(raw_history),
        }
        if policy_state_mode == "origin_visible":
            policy_state = latest.get("FEDFUNDS")
            if policy_state is None:
                raise DynamicMacroError(
                    f"Origin {origin['origin_month']} lacks origin-visible FEDFUNDS state"
                )
            if date.fromisoformat(
                str(policy_state["observation_date"])
            ) >= date.fromisoformat(origin["origin_month"]):
                raise DynamicMacroError(
                    "Origin-visible policy state must predate the scored observation month"
                )
            period_override["origin_visible_state_assimilation"] = {
                "policy_rate": {
                    "series_id": "FEDFUNDS",
                    "observation_date": policy_state["observation_date"],
                    "value": float(policy_state["value"]),
                }
            }
        overrides[period_index] = period_override
    assert_no_prompt_target_leakage(
        [{"prompt_payload": value} for value in overrides.values()]
    )
    return overrides


def anchor_household_flows(
    households: pd.DataFrame,
    period_overrides: Mapping[int, dict[str, Any]],
    *,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mode not in {"origin_saving_rate", "none"}:
        raise DynamicMacroError(f"Unsupported household flow anchor: {mode}")
    if not period_overrides or min(period_overrides) != 0:
        raise DynamicMacroError("Household flow anchor requires first-origin period zero")
    out = households.copy()
    weights = out["population_weight"].astype(float)
    income = float((weights * out["annual_income"].astype(float)).sum())
    consumption = float((weights * out["baseline_consumption_annual"].astype(float)).sum())
    if income <= 0.0 or consumption <= 0.0:
        raise DynamicMacroError("Household flow anchor requires positive aggregate income and consumption")
    pre_saving_rate = 100.0 * (income - consumption) / income
    if mode == "none":
        return out, {
            "mode": mode,
            "origin_visible_saving_rate_pct": None,
            "pre_anchor_saving_rate_pct": pre_saving_rate,
            "post_anchor_saving_rate_pct": pre_saving_rate,
            "consumption_scale": 1.0,
        }

    summary = period_overrides[0].get("observed_signal_summary")
    series = summary.get("series") if isinstance(summary, dict) else None
    psavert = series.get("PSAVERT") if isinstance(series, dict) else None
    target_rate = _optional_signal(psavert.get("latest_value") if isinstance(psavert, dict) else None, np.nan)
    if not math.isfinite(target_rate) or not -10.0 <= target_rate <= 50.0:
        raise DynamicMacroError("First origin lacks a plausible origin-visible PSAVERT level")
    target_consumption = income * (1.0 - target_rate / 100.0)
    if target_consumption <= 0.0:
        raise DynamicMacroError("Origin-visible saving rate implies non-positive aggregate consumption")
    scale = target_consumption / consumption
    out["baseline_consumption_annual"] = out["baseline_consumption_annual"].astype(float) * scale
    out["liquid_assets"] = out["liquid_assets"].astype(float) * scale
    out["base_saving_rate"] = (
        1.0 - out["baseline_consumption_annual"].astype(float) / out["annual_income"].astype(float)
    ).clip(lower=-0.10, upper=0.55)
    anchored_consumption = float((weights * out["baseline_consumption_annual"].astype(float)).sum())
    post_saving_rate = 100.0 * (income - anchored_consumption) / income
    if not math.isclose(post_saving_rate, target_rate, rel_tol=0.0, abs_tol=1e-9):
        raise DynamicMacroError("Household flow anchor failed to reproduce the origin-visible saving rate")
    return out, {
        "mode": mode,
        "origin_visible_saving_rate_pct": target_rate,
        "pre_anchor_saving_rate_pct": pre_saving_rate,
        "post_anchor_saving_rate_pct": post_saving_rate,
        "consumption_scale": scale,
        "liquid_buffer_preservation": "liquid assets scaled with baseline consumption",
    }


def build_initial_environment_anchor(
    period_overrides: Mapping[int, dict[str, Any]],
) -> dict[str, float]:
    if not period_overrides or min(period_overrides) != 0:
        raise DynamicMacroError(
            "Initial environment anchor requires first-origin period zero"
        )
    summary = period_overrides[0].get("observed_signal_summary")
    derived = summary.get("derived") if isinstance(summary, dict) else None
    if not isinstance(derived, dict):
        raise DynamicMacroError(
            "First origin has no observed signal summary for anchoring"
        )

    def required(field: str) -> float:
        value = _optional_signal(derived.get(field), np.nan)
        if not math.isfinite(value):
            raise DynamicMacroError(
                f"First-origin observed signal summary is missing anchor field {field}"
            )
        return value

    unemployment_rate = required("unemployment_rate_pct")
    employment_rate = 1.0 - unemployment_rate / 100.0
    if not 0.0 <= employment_rate <= 1.0:
        raise DynamicMacroError(
            "First-origin unemployment signal cannot produce a valid employment rate"
        )
    return {
        "employment_rate": employment_rate,
        "inflation_rate": required("inflation_annualized_pct"),
        "policy_rate": required("policy_rate_pct"),
        "output_gap_pct": 0.0,
    }


def observed_signal_summary(
    history_by_series: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    series: dict[str, dict[str, Any]] = {}
    for series_id, observations in sorted(history_by_series.items()):
        if not observations:
            continue
        latest = observations[-1]
        previous = observations[-2] if len(observations) >= 2 else None
        latest_value = float(latest["value"])
        previous_value = float(previous["value"]) if previous is not None else None
        series[series_id] = {
            "latest_observation_date": str(latest["observation_date"]),
            "latest_value": latest_value,
            "previous_observation_date": (
                str(previous["observation_date"]) if previous is not None else None
            ),
            "previous_value": previous_value,
            "change": (
                latest_value - previous_value if previous_value is not None else None
            ),
            "pct_change": (
                100.0 * (latest_value / previous_value - 1.0)
                if previous_value is not None and abs(previous_value) > 1e-12
                else None
            ),
        }

    def value(series_id: str, field: str) -> float | None:
        raw = series.get(series_id, {}).get(field)
        return float(raw) if raw is not None and math.isfinite(float(raw)) else None

    inflation_monthly = value("PCEPI", "pct_change")
    if inflation_monthly is None:
        inflation_monthly = value("CPIAUCSL", "pct_change")
    income_monthly = value("DSPIC96", "pct_change")
    derived = {
        "inflation_annualized_pct": (
            _monthly_rate_to_annual_pct(inflation_monthly)
            if inflation_monthly is not None
            else None
        ),
        "unemployment_rate_pct": value("UNRATE", "latest_value"),
        "unemployment_rate_change_pp": value("UNRATE", "change"),
        "policy_rate_pct": value("FEDFUNDS", "latest_value"),
        "payroll_growth_pct": value("PAYEMS", "pct_change"),
        "real_income_growth_annualized_pct": (
            _monthly_rate_to_annual_pct(income_monthly)
            if income_monthly is not None
            else None
        ),
        "real_demand_growth_pct": value("PCEC96", "pct_change"),
        "sentiment_level": value("UMCSENT", "latest_value"),
        "sentiment_change": value("UMCSENT", "change"),
    }
    return {
        "derivation": "latest two origin-visible observations only",
        "series": series,
        "derived": derived,
    }


def load_households(args: argparse.Namespace) -> pd.DataFrame:
    if args.households_csv:
        return normalize_demand_households(pd.read_csv(Path(args.households_csv)))
    panel = pd.read_csv(Path(args.household_panel))
    required_demand = {
        "type_id",
        "annual_income",
        "baseline_consumption_annual",
        "liquid_assets",
        "debt",
    }
    if required_demand.issubset(panel.columns):
        return normalize_demand_households(panel)
    return prepare_sce_households(panel)


def household_input_provenance(
    args: argparse.Namespace,
    households: pd.DataFrame,
    *,
    temporal_coverage: dict[str, Any],
) -> dict[str, Any]:
    source_kind = "households_csv" if args.households_csv else "household_panel"
    source_path = Path(args.households_csv or args.household_panel)
    if not source_path.is_file():
        raise DynamicMacroError(f"Household input file does not exist: {source_path}")
    return {
        "source_kind": source_kind,
        "raw_input_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "normalized_household_state_sha256": canonical_household_state_sha256(
            households
        ),
        "normalized_household_count": int(len(households)),
        "temporal_coverage": temporal_coverage,
    }


def validate_household_temporal_availability(
    args: argparse.Namespace,
    households: pd.DataFrame,
    *,
    first_origin: Mapping[str, str],
) -> dict[str, Any]:
    availability_column = "source_estimated_public_availability_date"
    event_candidates = ("source_survey_event_date", "survey_event_date", "survey_date")
    first_as_of = pd.Timestamp(first_origin["as_of_date"]).normalize()
    if availability_column not in households:
        if args.mode in {"live", "replay"}:
            raise DynamicMacroError(
                f"Real household input requires {availability_column} in {args.mode} mode"
            )
        return {
            "status": "fixture_omitted",
            "first_macro_origin_month": first_origin["origin_month"],
            "first_macro_as_of_date": first_origin["as_of_date"],
            "source_event_date_min": None,
            "source_event_date_max": None,
            "source_availability_date_min": None,
            "source_availability_date_max": None,
        }

    availability = pd.to_datetime(
        households[availability_column],
        errors="coerce",
    ).dt.normalize()
    if availability.isna().any():
        raise DynamicMacroError(
            f"Household input has invalid or missing {availability_column}"
        )
    latest_availability = availability.max()
    if latest_availability > first_as_of:
        raise DynamicMacroError(
            "Household data was not publicly available by the first macro origin: "
            f"latest availability {latest_availability.date()} exceeds {first_as_of.date()}"
        )
    event_column = next(
        (column for column in event_candidates if column in households),
        None,
    )
    event_dates = None
    if event_column is not None:
        event_dates = pd.to_datetime(
            households[event_column], errors="coerce"
        ).dt.normalize()
        if event_dates.isna().any():
            raise DynamicMacroError(
                f"Household input has invalid or missing {event_column}"
            )
    return {
        "status": "available_by_first_macro_origin",
        "first_macro_origin_month": first_origin["origin_month"],
        "first_macro_as_of_date": first_origin["as_of_date"],
        "source_event_date_column": event_column,
        "source_event_date_min": (
            event_dates.min().date().isoformat() if event_dates is not None else None
        ),
        "source_event_date_max": (
            event_dates.max().date().isoformat() if event_dates is not None else None
        ),
        "source_availability_date_column": availability_column,
        "source_availability_date_min": availability.min().date().isoformat(),
        "source_availability_date_max": latest_availability.date().isoformat(),
    }


def canonical_household_state_sha256(households: pd.DataFrame) -> str:
    ordered = households.sort_values("type_id").reset_index(drop=True)
    ordered = ordered.reindex(sorted(ordered.columns), axis="columns")
    payload = json.loads(
        ordered.to_json(
            orient="records",
            double_precision=15,
            date_format="iso",
        )
    )
    return _sha256_json(payload)


def canonical_behavior_profile_sha256(
    profile: dict[str, Any] | None,
) -> str | None:
    if profile is None:
        return None
    path_keys = {
        "path",
        "profile_json",
        "policy_profile_json",
        "profile_path",
        "source_path",
        "raw_records_path",
        "raw_records_json",
        "empirical_bridge_json",
        "empirical_bridge_path",
        "state_schedule_json",
        "state_profile_path",
        "cache_path",
    }

    def strip_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): strip_paths(nested)
                for key, nested in value.items()
                if str(key) not in path_keys and not str(key).endswith("_path")
            }
        if isinstance(value, list):
            return [strip_paths(item) for item in value]
        if isinstance(value, tuple):
            return [strip_paths(item) for item in value]
        return value

    return _sha256_json(strip_paths(profile))


def prepare_sce_households(panel: pd.DataFrame) -> pd.DataFrame:
    id_column = _first_column(panel, "respondent_id", "type_id", "household_id")
    weight_column = _first_column(panel, "weight", "population_weight", "survey_weight")
    if id_column is None or weight_column is None:
        raise DynamicMacroError(
            "Prepared SCE household panel must contain respondent_id and weight columns"
        )
    base = panel.copy()
    if "period_index" in base:
        base["period_index"] = pd.to_numeric(base["period_index"], errors="raise")
        base = base[base["period_index"].eq(base["period_index"].min())]
    base = base.drop_duplicates(id_column)
    event_source_column = _first_column(
        base,
        "source_survey_event_date",
        "survey_event_date",
        "survey_date",
    )
    availability_source_column = _first_column(
        base,
        "source_estimated_public_availability_date",
        "estimated_public_availability_date",
    )
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        income_group = _group(
            row.get("income_group"), ("low", "middle", "high"), "middle"
        )
        liquid_group = _group(
            row.get("liquid_wealth_group", row.get("liquidity_group")),
            ("low", "middle", "high"),
            "middle",
        )
        employment = str(row.get("employment_status", "unknown"))
        annual_income = _finite_or(
            row.get("annual_income"),
            {"low": 38000.0, "middle": 76000.0, "high": 135000.0}[income_group],
        )
        consumption_ratio = {"low": 0.94, "middle": 0.82, "high": 0.68}[income_group]
        baseline_consumption = _finite_or(
            row.get("baseline_consumption_annual"), annual_income * consumption_ratio
        )
        liquidity_months = {
            "low": {"low": 0.6, "middle": 0.9, "high": 1.4}[income_group],
            "middle": {"low": 1.8, "middle": 2.8, "high": 4.2}[income_group],
            "high": {"low": 3.8, "middle": 5.2, "high": 8.5}[income_group],
        }[liquid_group]
        unemp_higher = _first_finite(
            row,
            (
                "prior_expected_unemployment_higher_prob",
                "actual_expected_unemployment_higher_prob",
                "unemployment_higher_probability_1y",
            ),
            35.0,
        )
        job_loss = float(np.clip(0.24 * unemp_higher, 1.0, 24.0))
        job_risk = (
            "high"
            if job_loss >= 9.0 or employment.lower() in {"unemployed", "not_employed"}
            else "low"
        )
        inflation = _first_finite(
            row,
            (
                "prior_expected_inflation_1y",
                "actual_expected_inflation_1y",
                "inflation_expectation_1y",
            ),
            3.0,
        )
        income_growth = _first_finite(
            row,
            (
                "prior_expected_real_income_growth",
                "actual_expected_real_income_growth",
                "income_growth_expectation_1y",
            ),
            1.0,
        )
        low_liquid = liquid_group == "low"
        debt_service = {"low": 0.16, "middle": 0.13, "high": 0.09}[income_group]
        base_mpc = (0.72 if low_liquid else 0.24) + {
            "low": 0.09,
            "middle": 0.0,
            "high": -0.08,
        }[income_group]
        base_mpc += 0.04 if job_risk == "high" else -0.02
        confidence = float(
            np.clip(
                63.0
                - 0.36 * unemp_higher
                + 1.2 * income_growth
                - 0.45 * max(0.0, inflation - 2.0),
                0.0,
                100.0,
            )
        )
        household = {
            "type_id": str(row[id_column]),
            "label": f"SCE respondent {row[id_column]}",
            "population_weight": float(row[weight_column]),
            "age_bucket": (
                "older"
                if str(row.get("age_group", "")).lower()
                in {"55_plus", "55+", "older", "retired"}
                else "prime"
            ),
            "income_group": income_group,
            "liquidity_group": "low" if low_liquid else "high",
            "job_loss_risk_type": job_risk,
            "employment_status": employment,
            "annual_income": annual_income,
            "baseline_consumption_annual": baseline_consumption,
            "liquid_assets": _finite_or(
                row.get("liquid_assets"),
                liquidity_months * baseline_consumption / 12.0,
            ),
            "debt": _finite_or(
                row.get("debt"),
                annual_income * debt_service * (1.3 if low_liquid else 0.8),
            ),
            "debt_service_burden": debt_service,
            "base_mpc": float(np.clip(base_mpc, 0.08, 0.92)),
            "base_saving_rate": float(
                np.clip(
                    0.12
                    + (0.08 if not low_liquid else -0.03)
                    + (0.05 if income_group == "high" else 0.0),
                    0.02,
                    0.35,
                )
            ),
            "rate_sensitivity": 0.39 if low_liquid else 0.62,
            "income_sensitivity": {"low": 0.82, "middle": 0.58, "high": 0.36}[
                income_group
            ],
            "precautionary_sensitivity": 0.34
            + (0.28 if low_liquid else 0.08)
            + (0.16 if job_risk == "high" else 0.0),
            "baseline_job_loss_probability": job_loss,
            "unemployment_higher_probability_1y": unemp_higher,
            "target_buffer_months": 1.6 if low_liquid else 5.2,
            "inflation_expectation_1y": inflation,
            "income_growth_expectation_1y": income_growth,
            "confidence_index": confidence,
            "attention_weight_prices": (
                0.68
                if income_group == "low"
                else 0.56 if income_group == "middle" else 0.46
            ),
            "attention_weight_jobs": 0.75 if job_risk == "high" else 0.48,
            "attention_weight_rates": (
                0.66 if not low_liquid or income_group == "high" else 0.40
            ),
            "income_volatility": 0.10
            + (0.08 if job_risk == "high" else 0.02)
            + (0.04 if income_group == "low" else 0.0),
            "subsistence_floor_share": (
                0.56
                if income_group == "low"
                else 0.48 if income_group == "middle" else 0.38
            ),
        }
        if event_source_column is not None:
            household["source_survey_event_date"] = row[event_source_column]
        if availability_source_column is not None:
            household["source_estimated_public_availability_date"] = row[
                availability_source_column
            ]
        rows.append(household)
    if not rows:
        raise DynamicMacroError("Prepared SCE household panel has no usable households")
    return normalize_demand_households(pd.DataFrame(rows))


def load_behavior_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    mode = args.behavior_policy_mode
    generic = (
        Path(args.behavior_policy_profile) if args.behavior_policy_profile else None
    )
    if mode == "fixed_kernel":
        return None
    if mode == "schedule":
        path = (
            Path(args.behavior_policy_raw_records_json)
            if args.behavior_policy_raw_records_json
            else generic
        )
        if path is None:
            raise DynamicMacroError(
                "schedule mode requires --behavior-policy-raw-records-json or --behavior-policy-profile"
            )
        return load_behavior_policy_profile(path)
    if mode == "state_schedule":
        path = (
            Path(args.behavior_policy_state_profile_json)
            if args.behavior_policy_state_profile_json
            else generic
        )
        if path is None:
            raise DynamicMacroError(
                "state_schedule mode requires --behavior-policy-state-profile-json or --behavior-policy-profile"
            )
        return load_state_behavior_policy_profile(path)
    if mode == "empirical_bridge":
        path = (
            Path(args.empirical_bridge_json) if args.empirical_bridge_json else generic
        )
        if path is None:
            raise DynamicMacroError(
                "empirical_bridge mode requires --empirical-bridge-json or --behavior-policy-profile"
            )
        return load_empirical_bridge_profile(path)
    state_path = (
        Path(args.behavior_policy_state_profile_json)
        if args.behavior_policy_state_profile_json
        else None
    )
    bridge_path = (
        Path(args.empirical_bridge_json) if args.empirical_bridge_json else None
    )
    if state_path is None or bridge_path is None:
        raise DynamicMacroError(
            "hybrid policy mode requires state-profile and empirical-bridge paths"
        )
    return build_hybrid_behavior_policy_profile(
        load_empirical_bridge_profile(bridge_path),
        load_state_behavior_policy_profile(state_path),
        state_weight=float(args.hybrid_state_weight),
    )


def assert_no_prompt_target_leakage(prompt_rows: Iterable[dict[str, Any]]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            keys = tuple(map(str, value.keys()))
            leaked = set(FORBIDDEN_PROMPT_KEYS.intersection(keys))
            leaked.update(
                key
                for key in keys
                if key.startswith(FORBIDDEN_PROMPT_PREFIXES)
            )
            if leaked:
                raise DynamicMacroError(
                    f"Prompt target leakage detected: {', '.join(sorted(leaked))}"
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for row in prompt_rows:
        walk(row.get("prompt_payload", row))
