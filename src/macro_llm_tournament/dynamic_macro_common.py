"""Shared contracts and deterministic helpers for the dynamic macro runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "target_name",
        "target_value",
        "target_observation_date",
        "first_release_value",
        "first_release_as_of_date",
        "latest_revision_value",
        "latest_minus_first_release",
        "revision_audit",
        "target_contamination",
        "target_month",
        "target_realized",
        "realized",
        "forecast_error",
        "first_release",
        "latest_revision",
        "outcome",
        "origin_visible_denominator_value",
        "first_release_denominator_date",
        "first_release_denominator_value",
    }
)


class DynamicMacroError(ValueError):
    """Raised when a reproducibility, identity, or scoring contract fails."""


@dataclass(frozen=True)
class BundleView:
    bundle_sha256: str
    origins: tuple[dict[str, str], ...]
    history: tuple[dict[str, str], ...]
    targets: tuple[dict[str, Any], ...]
    target_specs: tuple[dict[str, Any], ...]
    target_contamination: tuple[dict[str, str], ...] = ()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _first_column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _group(value: Any, allowed: tuple[str, ...], default: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in allowed else default


def _finite_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _first_finite(row: pd.Series, columns: Iterable[str], default: float) -> float:
    for column in columns:
        if column in row.index:
            value = _finite_or(row[column], np.nan)
            if math.isfinite(value):
                return value
    return float(default)


def _monthly_rate_to_annual_pct(rate: float) -> float:
    if rate <= -100.0:
        raise DynamicMacroError(
            "Monthly growth rate cannot be at or below -100 percent"
        )
    return 100.0 * ((1.0 + rate / 100.0) ** 12.0 - 1.0)


def _optional_signal(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)
