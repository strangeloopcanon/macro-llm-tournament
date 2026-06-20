from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .agent_common import AGENT_ECONOMY_VERSION, WORK_ROOT, round_numeric_columns


@dataclass(frozen=True)
class AgentTypeDefinition:
    type_id: str
    label: str
    population_weight: float
    annual_income: float
    liquid_assets: float
    illiquid_assets: float
    debt: float
    consumption_proxy_annual: float
    credit_limit_proxy: float
    liquidity_sensitivity: float
    rate_sensitivity: float
    unemployment_sensitivity: float
    portfolio_sensitivity: float
    source: str = "fallback"


DEFAULT_AGENT_TYPES: tuple[AgentTypeDefinition, ...] = (
    AgentTypeDefinition("liquid_poor_renter", "Liquid-poor renter", 0.22, 54000, 1800, 12000, 9000, 42000, 9000, 1.45, 0.65, 1.25, 0.30),
    AgentTypeDefinition("wealthy_htm_homeowner", "Wealthy hand-to-mouth homeowner", 0.10, 110000, 4500, 420000, 280000, 72000, 18000, 1.20, 1.05, 0.85, 0.60),
    AgentTypeDefinition("leveraged_homeowner", "Leveraged homeowner", 0.13, 92000, 9000, 310000, 230000, 65000, 22000, 1.10, 1.35, 0.75, 0.45),
    AgentTypeDefinition("middle_income_buffer", "Middle-income buffered household", 0.24, 82000, 26000, 160000, 85000, 58000, 26000, 0.80, 0.75, 0.65, 0.30),
    AgentTypeDefinition("retiree_liquid_assets", "Retiree with liquid assets", 0.12, 52000, 70000, 360000, 25000, 43000, 12000, 0.55, 0.90, 0.25, 0.70),
    AgentTypeDefinition("high_income_illiquid_rich", "High-income illiquid-wealth household", 0.09, 260000, 65000, 1250000, 280000, 130000, 45000, 0.40, 0.80, 0.35, 0.90),
    AgentTypeDefinition("unemployed_low_liquid", "Unemployed low-liquid household", 0.06, 26000, 900, 6000, 12000, 24000, 5000, 1.70, 0.50, 1.75, 0.10),
    AgentTypeDefinition("business_owner_top_wealth", "Business owner / top-wealth household", 0.04, 310000, 85000, 2100000, 420000, 150000, 65000, 0.70, 1.15, 0.60, 1.10),
)


def build_household_type_cells(*, work_dir: Path = WORK_ROOT / "scf", wave: int = 2022) -> tuple[pd.DataFrame, dict[str, Any]]:
    fallback = _default_type_cells()
    path = work_dir / str(wave) / f"scfp{wave}excel.zip"
    if not path.exists():
        return fallback, {"status": "fallback_missing_scf_extract", "wave": wave, "expected_path": str(path)}
    try:
        scf = _read_scf_public_extract(path)
        derived = _derive_scf_type_cells(scf, wave=wave)
        if derived.empty:
            return fallback, {"status": "fallback_empty_scf_cells", "wave": wave, "source_path": str(path)}
        return derived, {
            "status": "ok",
            "wave": wave,
            "source_path": str(path),
            "raw_rows": int(scf.shape[0]),
            "derived_type_count": int(derived.shape[0]),
            "population_weight_sum": float(derived["population_weight"].sum()),
        }
    except Exception as exc:
        return fallback, {"status": "fallback_scf_parse_error", "wave": wave, "source_path": str(path), "error": str(exc)}


def credit_limit_proxy(row: dict[str, Any]) -> float:
    return float(
        max(
            0.0,
            0.30 * float(row["annual_income"])
            + 0.20 * float(row["liquid_assets"])
            + 0.03 * float(row["illiquid_assets"])
            - 0.08 * float(row["debt"]),
        )
    )


def _read_scf_public_extract(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {path}")
        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    return frame


def _derive_scf_type_cells(scf: pd.DataFrame, *, wave: int) -> pd.DataFrame:
    frame = scf.copy()
    for column in [
        "WGT",
        "INCOME",
        "LIQ",
        "ASSET",
        "DEBT",
        "NETWORTH",
        "HOUSES",
        "BUS",
        "AGE",
        "WAGEINC",
        "FOODHOME",
        "FOODAWAY",
        "FOODDELV",
        "RENT",
    ]:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["annual_income"] = frame["INCOME"].clip(lower=0.0)
    frame["liquid_assets"] = frame["LIQ"].clip(lower=0.0)
    frame["illiquid_assets"] = (frame["ASSET"] - frame["LIQ"]).clip(lower=0.0)
    frame["debt"] = frame["DEBT"].clip(lower=0.0)
    observed_consumption_proxy = frame[["FOODHOME", "FOODAWAY", "FOODDELV", "RENT"]].sum(axis=1).clip(lower=0.0)
    imputed_consumption = (0.58 * frame["annual_income"]).clip(lower=12000.0, upper=180000.0)
    frame["consumption_proxy_annual"] = np.maximum(observed_consumption_proxy, imputed_consumption)
    income_floor = frame["annual_income"].replace(0.0, np.nan).median()
    if not np.isfinite(income_floor) or income_floor <= 0.0:
        income_floor = 50000.0
    frame["income_for_ratios"] = frame["annual_income"].where(frame["annual_income"] > 0.0, income_floor)
    frame["liquid_buffer_months"] = 12.0 * frame["liquid_assets"] / frame["income_for_ratios"].clip(lower=1.0)
    frame["debt_to_asset"] = (frame["debt"] / frame["ASSET"].clip(lower=1000.0)).clip(lower=0.0, upper=5.0)
    frame["is_homeowner"] = frame["HOUSES"] > 0.0
    frame["is_business_owner"] = frame["BUS"] > 0.0
    frame["is_retiree"] = frame["AGE"] >= 65.0
    frame["is_low_liquid"] = frame["liquid_buffer_months"] < 1.0
    frame["networth_rank"] = frame["NETWORTH"].rank(pct=True)
    frame["income_rank"] = frame["annual_income"].rank(pct=True)
    masks = _scf_type_masks(frame)
    assigned = pd.Series(False, index=frame.index)
    rows: list[dict[str, Any]] = []
    total_weight = float(frame["WGT"].clip(lower=0.0).sum())
    if total_weight <= 0:
        frame["WGT"] = 1.0
        total_weight = float(frame.shape[0])
    for default in DEFAULT_AGENT_TYPES:
        mask = masks[default.type_id] & ~assigned
        if not mask.any() and default.type_id == "middle_income_buffer":
            mask = ~assigned
        if not mask.any():
            rows.append(_definition_to_row(default, source=f"fallback_empty_scf_{wave}"))
            continue
        assigned |= mask
        subset = frame[mask].copy()
        weights = subset["WGT"].clip(lower=0.0)
        weight_sum = float(weights.sum())
        population_weight = weight_sum / total_weight
        row = {
            "schema_version": AGENT_ECONOMY_VERSION,
            "type_id": default.type_id,
            "label": default.label,
            "population_weight": population_weight,
            "annual_income": _weighted_mean(subset["annual_income"], weights),
            "liquid_assets": _weighted_mean(subset["liquid_assets"], weights),
            "illiquid_assets": _weighted_mean(subset["illiquid_assets"], weights),
            "debt": _weighted_mean(subset["debt"], weights),
            "consumption_proxy_annual": _weighted_mean(subset["consumption_proxy_annual"], weights),
            "credit_limit_proxy": np.nan,
            "liquidity_sensitivity": default.liquidity_sensitivity,
            "rate_sensitivity": default.rate_sensitivity,
            "unemployment_sensitivity": default.unemployment_sensitivity,
            "portfolio_sensitivity": default.portfolio_sensitivity,
            "liquid_buffer_months": _weighted_mean(subset["liquid_buffer_months"], weights),
            "debt_to_asset": _weighted_mean(subset["debt_to_asset"], weights),
            "homeowner_share": _weighted_mean(subset["is_homeowner"].astype(float), weights),
            "source": f"scf_{wave}_public_extract",
            "raw_household_rows": int(subset.shape[0]),
        }
        row["credit_limit_proxy"] = credit_limit_proxy(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["population_weight"] = out["population_weight"].clip(lower=0.0)
    if out["population_weight"].sum() <= 0:
        return _default_type_cells()
    out["population_weight"] = out["population_weight"] / out["population_weight"].sum()
    return round_numeric_columns(out)


def _scf_type_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "liquid_poor_renter": (~frame["is_homeowner"]) & frame["is_low_liquid"] & (~frame["is_retiree"]),
        "wealthy_htm_homeowner": frame["is_homeowner"] & frame["is_low_liquid"] & (frame["networth_rank"] >= 0.55),
        "leveraged_homeowner": frame["is_homeowner"] & (frame["debt_to_asset"] >= 0.45) & (~frame["is_low_liquid"]),
        "middle_income_buffer": (frame["income_rank"].between(0.30, 0.75)) & (frame["liquid_buffer_months"].between(1.0, 8.0)),
        "retiree_liquid_assets": frame["is_retiree"] & (frame["liquid_buffer_months"] >= 3.0),
        "high_income_illiquid_rich": (frame["income_rank"] >= 0.85) & (frame["illiquid_assets"] > frame["liquid_assets"]),
        "unemployed_low_liquid": (frame["WAGEINC"] <= 0.0) & frame["is_low_liquid"] & (~frame["is_retiree"]),
        "business_owner_top_wealth": frame["is_business_owner"] & (frame["networth_rank"] >= 0.80),
    }


def _default_type_cells() -> pd.DataFrame:
    frame = pd.DataFrame([_definition_to_row(definition) for definition in DEFAULT_AGENT_TYPES])
    frame["population_weight"] = frame["population_weight"] / frame["population_weight"].sum()
    return round_numeric_columns(frame)


def _definition_to_row(definition: AgentTypeDefinition, *, source: str | None = None) -> dict[str, Any]:
    row = {
        "schema_version": AGENT_ECONOMY_VERSION,
        "type_id": definition.type_id,
        "label": definition.label,
        "population_weight": definition.population_weight,
        "annual_income": definition.annual_income,
        "liquid_assets": definition.liquid_assets,
        "illiquid_assets": definition.illiquid_assets,
        "debt": definition.debt,
        "consumption_proxy_annual": definition.consumption_proxy_annual,
        "credit_limit_proxy": definition.credit_limit_proxy,
        "liquidity_sensitivity": definition.liquidity_sensitivity,
        "rate_sensitivity": definition.rate_sensitivity,
        "unemployment_sensitivity": definition.unemployment_sensitivity,
        "portfolio_sensitivity": definition.portfolio_sensitivity,
        "liquid_buffer_months": 12.0 * definition.liquid_assets / max(definition.annual_income, 1.0),
        "debt_to_asset": definition.debt / max(definition.liquid_assets + definition.illiquid_assets, 1.0),
        "homeowner_share": 1.0 if "homeowner" in definition.type_id or definition.type_id in {"high_income_illiquid_rich", "business_owner_top_wealth"} else 0.0,
        "source": source or definition.source,
        "raw_household_rows": 0,
    }
    row["credit_limit_proxy"] = max(float(row["credit_limit_proxy"]), credit_limit_proxy(row))
    return row


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    total = float(weights.sum())
    if total <= 0.0:
        return float(numeric.mean()) if len(numeric) else 0.0
    return float((numeric * weights).sum() / total)
