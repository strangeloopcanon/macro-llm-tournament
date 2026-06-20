from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .llm_common import LLMUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
WORK_ROOT = PROJECT_ROOT / "work"
LLM_CACHE_ROOT = WORK_ROOT / "llm_cache"
AGENT_LLM_CACHE_ROOT = WORK_ROOT / "agent_llm_cache"

AGENT_ECONOMY_VERSION = "forecast_first_typed_agent_economy_v1"
AGENT_LLM_PROMPT_VERSION = "forecast_first_typed_agents_v1"
ADJUSTMENT_COST_RATE = 0.02
ACCOUNTING_TOLERANCE = 1e-6


def finite_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def bounded_float(mapping: dict[str, Any], key: str, lower: float, upper: float, *, default: float | None = None) -> float:
    fallback = np.nan if default is None else float(default)
    value = finite_float(mapping.get(key), default=fallback)
    if not np.isfinite(value):
        if default is None:
            raise LLMUnavailable(f"Agent payload field {key} must be numeric")
        return float(default)
    return float(np.clip(value, lower, upper))


def round_or_none(value: Any, digits: int = 4) -> float | None:
    value = finite_float(value, default=np.nan)
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def round_numeric_columns(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].astype(float).round(digits)
    return out


def weighted_sum(rows: list[dict[str, Any]], column: str) -> float:
    return float(sum(float(row.get(column, 0.0)) * float(row.get("population_weight", 0.0)) for row in rows))


def pct_change(value: float, baseline: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return 0.0
    return float(100.0 * (float(value) / float(baseline) - 1.0))


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
    if not match:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    return json.loads(match.group(1))


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def max_abs(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    return float(frame[column].abs().max())


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    clean = clean.fillna("").astype(str)
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)
