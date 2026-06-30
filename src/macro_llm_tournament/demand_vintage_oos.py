from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, cache_key, markdown_table
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable


DEMAND_VINTAGE_OOS_VERSION = "demand_vintage_oos_v1"
PROMPT_VERSION = "demand_vintage_oos_card_v1"
DEFAULT_VINTAGE_PANEL_DIR = WORK_ROOT / "fred_vintage_panel"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "demand_vintage_oos_fixture"
DEFAULT_FORECAST_CACHE_DIR = WORK_ROOT / "demand_vintage_oos_cache"
OOS_MODES = ("fixture", "panel")
FORECAST_MODES = ("fixture", "replay", "live")


@dataclass(frozen=True)
class VintageTargetSpec:
    target_name: str
    series_id: str
    label: str
    transform: str
    default_scale: float


TARGET_SPECS: tuple[VintageTargetSpec, ...] = (
    VintageTargetSpec("real_consumption_growth_pct", "PCECC96", "real consumption growth", "pct_change", 0.8),
    VintageTargetSpec("saving_rate_level", "PSAVERT", "personal saving rate", "level", 2.0),
    VintageTargetSpec("inflation_growth_pct", "CPIAUCSL", "consumer price inflation", "pct_change", 0.6),
    VintageTargetSpec("unemployment_rate_level", "UNRATE", "unemployment rate", "level", 1.2),
    VintageTargetSpec("policy_rate_level", "FEDFUNDS", "policy rate", "level", 1.0),
    VintageTargetSpec("sentiment_growth_pct", "UMCSENT", "consumer sentiment growth", "pct_change", 4.0),
    VintageTargetSpec("output_growth_pct", "GDPC1", "real output growth", "pct_change", 0.8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build date-free vintage OOS demand cards, model forecasts, targets, and scores.")
    parser.add_argument("--vintage-panel-dir", default=str(DEFAULT_VINTAGE_PANEL_DIR))
    parser.add_argument("--mode", choices=OOS_MODES, default="panel")
    parser.add_argument("--forecast-mode", choices=FORECAST_MODES, default="fixture")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--splits", default="", help="Optional comma-separated split filter such as val,test. Empty means all splits.")
    parser.add_argument("--max-origins", type=int, default=0, help="Optional cap on origins; 0 means all available origins.")
    parser.add_argument("--history-periods", type=int, default=8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_models(args.models)
    if args.forecast_mode in {"replay", "live"} and not models:
        raise SystemExit("--models must contain at least one model when --forecast-mode uses an LLM client")
    if args.forecast_mode == "live":
        if args.max_live_calls <= 0:
            raise SystemExit("--max-live-calls must be positive when --forecast-mode live is used")
        if not args.fresh_cache:
            raise SystemExit("--fresh-cache is required when --forecast-mode live is used")
    output_dir = Path(args.output_dir)
    cache_dir = _resolve_forecast_cache_dir(args, output_dir)
    origins, context, data_status = load_vintage_panel(Path(args.vintage_panel_dir), mode=args.mode)
    try:
        result = build_demand_vintage_oos(
            origins,
            context,
            mode=args.mode,
            forecast_mode=args.forecast_mode,
            provider=args.provider,
            models=models,
            max_live_calls=args.max_live_calls,
            cache_dir=cache_dir,
            fresh_cache=args.fresh_cache,
            splits=_parse_splits(args.splits),
            max_origins=args.max_origins,
            history_periods=args.history_periods,
            data_status=data_status,
            vintage_panel_dir=Path(args.vintage_panel_dir),
        )
    except LLMUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    write_demand_vintage_oos_outputs(result, output_dir)
    print(f"Wrote demand vintage OOS run to {args.output_dir}")
    print(json.dumps({"verdict": result["manifest"]["verdict"], "passed": result["manifest"]["passed"]}, indent=2, sort_keys=True))
    return 0 if result["manifest"]["passed"] else 1


def load_vintage_panel(vintage_panel_dir: Path, *, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if mode == "fixture":
        origins, context = fixture_vintage_panel()
        return origins, context, {"status": "fixture", "source": "deterministic fixture"}
    origins_path = vintage_panel_dir / "forecast_origins_for_vintage_context.csv"
    context_path = vintage_panel_dir / "fred_vintage_context.csv"
    if not origins_path.exists() or not context_path.exists():
        raise FileNotFoundError(f"Vintage panel files not found under {vintage_panel_dir}")
    origins = pd.read_csv(origins_path)
    context = pd.read_csv(context_path)
    return origins, context, {"status": "panel", "source": str(vintage_panel_dir)}


def build_demand_vintage_oos(
    origins: pd.DataFrame,
    context: pd.DataFrame,
    *,
    mode: str,
    forecast_mode: str = "fixture",
    provider: str = "codex_cli",
    models: list[str] | None = None,
    max_live_calls: int = 0,
    cache_dir: Path | None = None,
    fresh_cache: bool = False,
    splits: list[str] | None = None,
    max_origins: int = 0,
    history_periods: int = 8,
    data_status: dict[str, Any] | None = None,
    vintage_panel_dir: Path | None = None,
) -> dict[str, Any]:
    if mode not in OOS_MODES:
        raise ValueError(f"Unsupported vintage OOS mode: {mode}")
    if forecast_mode not in FORECAST_MODES:
        raise ValueError(f"Unsupported forecast mode: {forecast_mode}")
    models = list(models or ["gpt-5.5"])
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_FORECAST_CACHE_DIR
    origins = _normalize_origins(origins)
    context = _normalize_context(context)
    origins = _filter_origins_by_splits(origins, splits or [])
    if max_origins and max_origins > 0:
        origins = _balanced_origin_sample(origins, max_origins)
    cards, targets = build_vintage_cards_and_targets(origins, context, history_periods=history_periods)
    if forecast_mode == "live" and fresh_cache and _cache_json_count(cache_dir) > 0:
        raise LLMUnavailable(
            "--fresh-cache live vintage OOS runs require an empty cache directory; "
            f"found existing cached JSON under {cache_dir}"
        )
    if forecast_mode == "live":
        required_live_calls = int(cards.shape[0]) * len(models)
        if int(max_live_calls) < required_live_calls:
            raise LLMUnavailable(
                "--max-live-calls must be at least "
                f"{required_live_calls} for a fresh live vintage OOS run with {len(models)} model(s) and {cards.shape[0]} card(s)"
            )
    forecasts = build_vintage_forecasts(cards, targets, include_llm_fixture=(forecast_mode == "fixture"))
    raw_records: list[dict[str, Any]] = []
    live_call_count = 0
    cache_hit_count = 0
    if forecast_mode in {"replay", "live"}:
        model_forecasts, raw_records, live_call_count, cache_hit_count = build_vintage_model_forecasts(
            cards,
            provider=provider,
            models=models,
            forecast_mode=forecast_mode,
            max_live_calls=max_live_calls,
            cache_dir=cache_dir,
        )
        forecasts = pd.concat([forecasts, model_forecasts], ignore_index=True) if not model_forecasts.empty else forecasts
    scores, joined = score_vintage_forecasts(forecasts, targets)
    summary = summarize_vintage_scores(scores)
    leakage = audit_card_leakage(cards)
    verdict = vintage_oos_verdict(cards, targets, scores, leakage, mode=mode, forecast_mode=forecast_mode)
    manifest = {
        "schema_version": DEMAND_VINTAGE_OOS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "forecast_mode": forecast_mode,
        "status": "ok",
        "verdict": verdict,
        "passed": verdict in {"demand_vintage_oos_fixture_ready", "demand_vintage_oos_scored"},
        "vintage_panel_dir": str(vintage_panel_dir) if vintage_panel_dir is not None else None,
        "data_status": data_status or {},
        "provider": provider if forecast_mode in {"replay", "live"} else None,
        "models": models if forecast_mode in {"replay", "live"} else [],
        "max_live_calls": int(max_live_calls),
        "live_call_count": int(live_call_count),
        "cache_hit_count": int(cache_hit_count),
        "cache_dir": str(cache_dir) if forecast_mode in {"replay", "live"} else None,
        "fresh_cache": bool(fresh_cache),
        "splits": sorted(origins["split"].dropna().astype(str).unique().tolist()) if "split" in origins else [],
        "origin_count": int(origins.shape[0]),
        "scored_origin_count": int(cards["origin_id"].nunique()) if "origin_id" in cards else 0,
        "card_count": int(cards.shape[0]),
        "target_rows": int(targets.shape[0]),
        "forecast_rows": int(forecasts.shape[0]),
        "score_rows": int(scores.shape[0]),
        "raw_record_count": int(len(raw_records)),
        "leakage_issue_count": int(leakage.shape[0]),
        "target_specs": [spec.target_name for spec in TARGET_SPECS],
        "cards_sha256": frame_sha256(cards),
        "targets_sha256": frame_sha256(targets),
        "outputs": [
            "demand_vintage_oos_cards.csv",
            "demand_vintage_oos_targets.csv",
            "demand_vintage_oos_forecasts.csv",
            "demand_vintage_oos_scores.csv",
            "demand_vintage_oos_joined_errors.csv",
            "demand_vintage_oos_summary.csv",
            "demand_vintage_oos_leakage_audit.csv",
            "demand_vintage_oos_raw_records.json",
            "demand_vintage_oos_report.md",
            "manifest.json",
        ],
    }
    report = build_vintage_oos_report(manifest, summary, leakage)
    return {
        "manifest": manifest,
        "cards": cards,
        "targets": targets,
        "forecasts": forecasts,
        "scores": scores,
        "joined": joined,
        "summary": summary,
        "leakage": leakage,
        "raw_records": raw_records,
        "report": report,
    }


def build_vintage_cards_and_targets(origins: pd.DataFrame, context: pd.DataFrame, *, history_periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_values = _final_values(context)
    card_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for origin_index, origin in origins.reset_index(drop=True).iterrows():
        origin_key = _origin_key(origin, origin_index)
        origin_context = context[context["origin"].astype(str) == str(origin["origin"])].copy()
        if origin_context.empty:
            continue
        for spec in TARGET_SPECS:
            series_context = origin_context[origin_context["series_id"].astype(str) == spec.series_id].sort_values("observation_date")
            final_series = final_values[final_values["series_id"].astype(str) == spec.series_id].sort_values("observation_date")
            if series_context.empty or final_series.empty:
                continue
            history = _history_rows(series_context, history_periods)
            if history.empty:
                continue
            current_observation = pd.Timestamp(history["observation_date"].iloc[-1])
            target_candidates = final_series[final_series["observation_date"] > current_observation]
            if target_candidates.empty:
                continue
            target_row = target_candidates.iloc[0]
            current_value = float(history["value"].iloc[-1])
            final_current_value = _final_current_value(final_series, current_observation)
            target_denominator_value = final_current_value if spec.transform == "pct_change" else current_value
            target_value = _target_value(spec, current_value=target_denominator_value, target_raw=float(target_row["value"]))
            if not np.isfinite(target_value):
                continue
            card_id = f"{origin_key}_{spec.target_name}"
            prompt_payload = {
                "prompt_version": PROMPT_VERSION,
                "card_id": card_id,
                "origin_id": origin_key,
                "target_name": spec.target_name,
                "target_transform": spec.transform,
                "target_period": "next_observed_period",
                "contamination_control": "No calendar dates, named historical episodes, hidden outcome values, or future observation labels are supplied.",
                "history": _relative_history(history),
                "required_response": {
                    "forecast_value": "numeric forecast for target_name at target_period",
                    "confidence": "0 to 1",
                    "reason": "short explanation",
                },
            }
            card_rows.append(
                {
                    "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                    "card_id": card_id,
                    "origin_id": origin_key,
                    "split": str(origin["split"]),
                    "target_name": spec.target_name,
                    "series_id": spec.series_id,
                    "transform": spec.transform,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_payload_json": json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")),
                    "history_json": json.dumps(_relative_history(history), sort_keys=True, separators=(",", ":")),
                }
            )
            target_rows.append(
                {
                    "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                    "card_id": card_id,
                    "origin": str(origin["origin"]),
                    "origin_id": origin_key,
                    "as_of_date": origin["as_of_date"].date().isoformat(),
                    "split": str(origin["split"]),
                    "target_name": spec.target_name,
                    "series_id": spec.series_id,
                    "transform": spec.transform,
                    "current_observation_date": current_observation.date().isoformat(),
                    "target_observation_date": pd.Timestamp(target_row["observation_date"]).date().isoformat(),
                    "current_value": current_value,
                    "target_current_value": target_denominator_value,
                    "target_raw_value": float(target_row["value"]),
                    "target_value": target_value,
                    "default_scale": float(spec.default_scale),
                    "target_available": True,
                }
            )
    return pd.DataFrame(card_rows), pd.DataFrame(target_rows)


def _origin_key(origin: pd.Series, fallback_index: int) -> str:
    sequence = origin.get("_origin_sequence", fallback_index)
    if pd.isna(sequence):
        sequence = fallback_index
    return f"vintage_origin_{int(sequence):04d}"


def build_vintage_forecasts(cards: pd.DataFrame, targets: pd.DataFrame, *, include_llm_fixture: bool = True) -> pd.DataFrame:
    if cards.empty or targets.empty:
        return pd.DataFrame()
    target_by_card = targets.set_index("card_id")
    rows: list[dict[str, Any]] = []
    for _, card in cards.iterrows():
        target = target_by_card.loc[card["card_id"]]
        history = json.loads(card["history_json"])
        values = [float(item["value"]) for item in history if np.isfinite(float(item["value"]))]
        if not values:
            continue
        no_change = _no_change_forecast(str(target["transform"]), values[-1])
        rolling_mean = _rolling_mean_forecast(str(target["transform"]), values)
        rolling_trend = _rolling_trend_forecast(str(target["transform"]), values)
        llm_fixture = 0.20 * no_change + 0.35 * rolling_mean + 0.45 * rolling_trend
        forecast_specs = [
            ("no_change", no_change, "no_change"),
            ("rolling_mean", rolling_mean, "rolling_mean"),
            ("rolling_trend", rolling_trend, "rolling_trend"),
        ]
        if include_llm_fixture:
            forecast_specs.append(("llm_belief_fixture", llm_fixture, "llm_belief"))
        for source, forecast, variant in forecast_specs:
            rows.append(
                {
                    "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                    "card_id": card["card_id"],
                    "origin_id": card["origin_id"],
                    "split": card["split"],
                    "target_name": card["target_name"],
                    "source": source,
                    "variant": variant,
                    "forecast_value": float(forecast),
                    "confidence": 0.80 if source == "llm_belief_fixture" else 0.70,
                    "reason": f"{source} fixture forecast from date-free vintage history",
                }
            )
    return pd.DataFrame(rows)


def build_vintage_model_forecasts(
    cards: pd.DataFrame,
    *,
    provider: str,
    models: list[str],
    forecast_mode: str,
    max_live_calls: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    live_used = 0
    cache_hits = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_prompt_payloads = _legacy_prompt_payloads_by_card(cards)
    for model in models:
        client = ForecastLLMClient(
            provider,
            model,
            cache_dir,
            mode=forecast_mode,
            max_live_calls=max(0, int(max_live_calls) - live_used),
        )
        for _, card in cards.iterrows():
            prompt_payload = json.loads(str(card["prompt_payload_json"]))
            cache_name = vintage_forecast_cache_name(provider, model, prompt_payload)
            legacy_cache_name = None
            legacy_prompt_hash = None
            legacy_cache_hit = False
            try:
                raw = client.json_call(vintage_forecast_prompt(prompt_payload), cache_name, instructions=vintage_forecast_instructions())
            except LLMUnavailable:
                legacy_prompt_payload = legacy_prompt_payloads.get(str(card["card_id"]))
                if forecast_mode != "replay" or legacy_prompt_payload is None:
                    raise
                legacy_cache_name = vintage_forecast_cache_name(provider, model, legacy_prompt_payload)
                raw = client.json_call(
                    vintage_forecast_prompt(legacy_prompt_payload),
                    legacy_cache_name,
                    instructions=vintage_forecast_instructions(),
                )
                legacy_prompt_hash = hashlib.sha256(
                    json.dumps(legacy_prompt_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                ).hexdigest()
                legacy_cache_hit = True
            payload = normalize_vintage_forecast_payload(raw.get("payload"), prompt_payload=prompt_payload)
            payload_hash = hashlib.sha256(
                json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            prompt_hash = hashlib.sha256(
                json.dumps(prompt_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            source = f"llm_{provider}_{model}".replace("/", "_").replace(":", "_")
            rows.append(
                {
                    "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                    "card_id": card["card_id"],
                    "origin_id": card["origin_id"],
                    "split": card["split"],
                    "target_name": card["target_name"],
                    "source": source,
                    "variant": "llm_belief",
                    "forecast_value": float(payload["forecast_value"]),
                    "confidence": float(payload["confidence"]),
                    "reason": str(payload["reason"]),
                }
            )
            raw_records.append(
                {
                    "schema_version": DEMAND_VINTAGE_OOS_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "forecast_mode": forecast_mode,
                    "cache_name": cache_name,
                    "legacy_cache_name": legacy_cache_name,
                    "legacy_cache_hit": legacy_cache_hit,
                    "cache_hit": bool(raw.get("cache_hit", False)),
                    "cache_path": raw.get("cache_path"),
                    "card_id": card["card_id"],
                    "origin_id": card["origin_id"],
                    "split": card["split"],
                    "target_name": card["target_name"],
                    "prompt_payload_sha256": prompt_hash,
                    "legacy_prompt_payload_sha256": legacy_prompt_hash,
                    "payload_sha256": payload_hash,
                    "payload": payload,
                }
            )
        live_used += client.live_call_count
        cache_hits += client.cache_hit_count
    return pd.DataFrame(rows), raw_records, int(live_used), int(cache_hits)


def _legacy_prompt_payloads_by_card(cards: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if cards.empty or "origin_id" not in cards or "split" not in cards:
        return {}
    origin_rows = cards[["split", "origin_id"]].drop_duplicates().copy()
    origin_rows["_origin_number"] = origin_rows["origin_id"].map(_origin_number)
    origin_rows = origin_rows.dropna(subset=["_origin_number"]).sort_values(["split", "_origin_number"])
    legacy_by_origin: dict[tuple[str, str], str] = {}
    for split, group in origin_rows.groupby("split", sort=True):
        for local_index, (_, row) in enumerate(group.iterrows()):
            legacy_by_origin[(str(split), str(row["origin_id"]))] = f"vintage_origin_{local_index:04d}"
    payloads: dict[str, dict[str, Any]] = {}
    for _, card in cards.iterrows():
        legacy_origin_id = legacy_by_origin.get((str(card["split"]), str(card["origin_id"])))
        if not legacy_origin_id or legacy_origin_id == str(card["origin_id"]):
            continue
        prompt_payload = json.loads(str(card["prompt_payload_json"]))
        legacy_payload = dict(prompt_payload)
        legacy_payload["origin_id"] = legacy_origin_id
        legacy_payload["card_id"] = f"{legacy_origin_id}_{card['target_name']}"
        payloads[str(card["card_id"])] = legacy_payload
    return payloads


def _origin_number(origin_id: Any) -> float:
    match = re.search(r"(\d+)$", str(origin_id))
    return float(match.group(1)) if match else np.nan


def vintage_forecast_cache_name(provider: str, model: str, prompt_payload: dict[str, Any]) -> str:
    key = cache_key(
        {
            "kind": "demand_vintage_oos_forecast",
            "provider": provider,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "prompt_payload": prompt_payload,
        }
    )
    return f"demand_vintage_oos_forecast_{key}"


def vintage_forecast_prompt(prompt_payload: dict[str, Any]) -> str:
    return f"""
You are forecasting one date-free macro demand card.
Use only the card JSON below. Do not infer a calendar date, named crisis, or hidden realized outcome.

Card JSON:
{json.dumps(prompt_payload, indent=2, sort_keys=True)}

Return exactly this JSON object:
{{
  "forecast_value": 0.0,
  "confidence": 0.0,
  "reason": "short reason based only on the supplied relative history"
}}
""".strip()


def vintage_forecast_instructions() -> str:
    return """
Return only valid JSON. Use only the supplied date-free card. Do not browse,
inspect files, run commands, name calendar episodes, or cite realized outcomes.
forecast_value must be numeric in the target units; confidence must be between 0 and 1.
""".strip()


def normalize_vintage_forecast_payload(payload: Any, *, prompt_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise LLMUnavailable("Vintage OOS forecast payload must be a JSON object")
    allowed = {"prompt_version", "forecast_value", "confidence", "reason"}
    extra = sorted(set(payload) - allowed)
    if extra:
        raise LLMUnavailable(f"Vintage OOS forecast payload has unexpected field(s): {', '.join(extra)}")
    if "prompt_version" in payload and str(payload["prompt_version"]) != str(prompt_payload.get("prompt_version")):
        raise LLMUnavailable("Vintage OOS forecast payload prompt_version does not match the card")
    forecast_value = _finite_payload_float(payload.get("forecast_value"), "forecast_value")
    if abs(forecast_value) > 10000.0:
        raise LLMUnavailable("Vintage OOS forecast_value is outside the allowed +/-10000 bound")
    confidence = _finite_payload_float(payload.get("confidence"), "confidence")
    if confidence < 0.0 or confidence > 1.0:
        raise LLMUnavailable("Vintage OOS confidence must be between 0 and 1")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise LLMUnavailable("Vintage OOS forecast payload reason must be non-empty")
    return {
        "forecast_value": float(forecast_value),
        "confidence": float(confidence),
        "reason": reason[:500],
    }


def score_vintage_forecasts(forecasts: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if forecasts.empty or targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    joined = forecasts.merge(
        targets[
            [
                "card_id",
                "origin",
                "as_of_date",
                "target_observation_date",
                "target_value",
                "default_scale",
                "target_available",
            ]
        ],
        on="card_id",
        how="inner",
    )
    joined["error"] = pd.to_numeric(joined["forecast_value"], errors="coerce") - pd.to_numeric(joined["target_value"], errors="coerce")
    joined["abs_error"] = joined["error"].abs()
    joined["squared_error"] = joined["error"] ** 2
    joined["normalized_abs_error"] = joined["abs_error"] / pd.to_numeric(joined["default_scale"], errors="coerce").replace(0.0, np.nan)
    rows = []
    for (source, variant, split, target_name), group in joined.groupby(["source", "variant", "split", "target_name"], sort=True):
        rows.append(
            {
                "source": source,
                "variant": variant,
                "split": split,
                "target_name": target_name,
                "n": int(group.shape[0]),
                "mae": float(group["abs_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "weighted_normalized_abs_error": float(group["normalized_abs_error"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "source", "target_name"]).reset_index(drop=True), joined


def summarize_vintage_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    rows = []
    for (source, variant, split), group in scores.groupby(["source", "variant", "split"], sort=True):
        weights = pd.to_numeric(group["n"], errors="coerce").fillna(0.0).clip(lower=0.0)
        values = pd.to_numeric(group["weighted_normalized_abs_error"], errors="coerce")
        denominator = float(weights.sum())
        rows.append(
            {
                "source": source,
                "variant": variant,
                "split": split,
                "target_family_count": int(group["target_name"].nunique()),
                "n": int(group["n"].sum()),
                "weighted_normalized_abs_error": float((weights * values).sum() / denominator) if denominator > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "source"]).reset_index(drop=True)


def audit_card_leakage(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["card_id", "issue", "snippet"]
    if cards.empty:
        return pd.DataFrame(columns=columns)
    forbidden_patterns = [
        ("calendar_date", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")),
        ("target_value_key", re.compile(r"target_value|target_raw_value|realized", re.I)),
        ("target_observation_date", re.compile(r"target_observation_date|as_of_date", re.I)),
        ("historical_episode", re.compile(r"great recession|covid|pandemic|2008|2020", re.I)),
    ]
    for _, card in cards.iterrows():
        payload = str(card["prompt_payload_json"])
        for issue, pattern in forbidden_patterns:
            match = pattern.search(payload)
            if match:
                rows.append({"card_id": card["card_id"], "issue": issue, "snippet": payload[max(0, match.start() - 20) : match.end() + 20]})
    return pd.DataFrame(rows, columns=columns)


def vintage_oos_verdict(
    cards: pd.DataFrame,
    targets: pd.DataFrame,
    scores: pd.DataFrame,
    leakage: pd.DataFrame,
    *,
    mode: str,
    forecast_mode: str,
) -> str:
    if not leakage.empty:
        return "demand_vintage_oos_leakage_failed"
    if cards.empty or targets.empty or scores.empty:
        return "demand_vintage_oos_needs_work"
    if mode == "fixture" or forecast_mode == "fixture":
        return "demand_vintage_oos_fixture_ready"
    return "demand_vintage_oos_scored"


def write_demand_vintage_oos_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["cards"].to_csv(output_dir / "demand_vintage_oos_cards.csv", index=False)
    result["targets"].to_csv(output_dir / "demand_vintage_oos_targets.csv", index=False)
    result["forecasts"].to_csv(output_dir / "demand_vintage_oos_forecasts.csv", index=False)
    result["scores"].to_csv(output_dir / "demand_vintage_oos_scores.csv", index=False)
    result["joined"].to_csv(output_dir / "demand_vintage_oos_joined_errors.csv", index=False)
    result["summary"].to_csv(output_dir / "demand_vintage_oos_summary.csv", index=False)
    result["leakage"].to_csv(output_dir / "demand_vintage_oos_leakage_audit.csv", index=False)
    (output_dir / "demand_vintage_oos_raw_records.json").write_text(
        json.dumps(_jsonable(result.get("raw_records", [])), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "demand_vintage_oos_report.md").write_text(result["report"], encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(result["manifest"]), indent=2, sort_keys=True), encoding="utf-8")


def build_vintage_oos_report(manifest: dict[str, Any], summary: pd.DataFrame, leakage: pd.DataFrame) -> str:
    lines = [
        "# Demand Vintage OOS",
        "",
        "## Bottom Line",
        _vintage_bottom_line(manifest),
        "",
        "## Score Summary",
        markdown_table(summary),
        "",
        "## Leakage Audit",
        markdown_table(leakage),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def frame_sha256(frame: pd.DataFrame) -> str:
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fixture_vintage_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = []
    context_rows = []
    dates = pd.date_range("2018-01-01", periods=36, freq="QS")
    series_ids = [spec.series_id for spec in TARGET_SPECS]
    for idx in range(18):
        split = "train" if idx < 10 else "val" if idx < 13 else "test"
        as_of = dates[idx + 8] + pd.DateOffset(days=45)
        origin = f"period_{idx}"
        origins.append({"origin": origin, "as_of_date": as_of.date().isoformat(), "split": split})
        for series_id in series_ids:
            base = 100.0 + 2.5 * series_ids.index(series_id)
            for obs_idx, obs_date in enumerate(dates[: idx + 9]):
                value = base + 0.55 * obs_idx + 1.5 * np.sin((obs_idx + series_ids.index(series_id)) / 3.0)
                if series_id in {"UNRATE", "FEDFUNDS", "PSAVERT"}:
                    value = 4.0 + 0.08 * obs_idx + 0.3 * np.sin(obs_idx / 4.0 + series_ids.index(series_id))
                context_rows.append(
                    {
                        "origin": origin,
                        "as_of_date": as_of.date().isoformat(),
                        "series_id": series_id,
                        "label": series_id,
                        "observation_date": obs_date.date().isoformat(),
                        "value": float(value),
                        "realtime_start": as_of.date().isoformat(),
                        "realtime_end": as_of.date().isoformat(),
                    }
                )
    return pd.DataFrame(origins), pd.DataFrame(context_rows)


def _normalize_origins(origins: pd.DataFrame) -> pd.DataFrame:
    required = {"origin", "as_of_date", "split"}
    missing = required - set(origins.columns)
    if missing:
        raise ValueError(f"Vintage origins missing columns: {', '.join(sorted(missing))}")
    out = origins.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    out["split"] = out["split"].fillna("train").astype(str)
    out = out.sort_values(["as_of_date", "origin"]).reset_index(drop=True)
    out["_origin_sequence"] = np.arange(out.shape[0], dtype=int)
    return out


def _filter_origins_by_splits(origins: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    wanted = {str(split).strip() for split in splits if str(split).strip()}
    if not wanted:
        return origins
    available = set(origins["split"].dropna().astype(str))
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(f"Vintage origins do not contain requested split(s): {', '.join(missing)}")
    filtered = origins[origins["split"].astype(str).isin(wanted)].copy()
    if filtered.empty:
        raise ValueError(f"Vintage split filter produced no origins: {', '.join(sorted(wanted))}")
    return filtered.sort_values(["as_of_date", "origin"]).reset_index(drop=True)


def _normalize_context(context: pd.DataFrame) -> pd.DataFrame:
    required = {"origin", "as_of_date", "series_id", "observation_date", "value", "realtime_start", "realtime_end"}
    missing = required - set(context.columns)
    if missing:
        raise ValueError(f"Vintage context missing columns: {', '.join(sorted(missing))}")
    out = context.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    out["observation_date"] = pd.to_datetime(out["observation_date"])
    out["realtime_start"] = pd.to_datetime(out["realtime_start"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value", "observation_date"])
    return out.sort_values(["origin", "series_id", "observation_date"]).reset_index(drop=True)


def _balanced_origin_sample(origins: pd.DataFrame, max_origins: int) -> pd.DataFrame:
    if origins.shape[0] <= max_origins:
        return origins
    pieces = []
    per_split = max(1, int(np.ceil(max_origins / max(1, origins["split"].nunique()))))
    for _, group in origins.groupby("split", sort=True):
        pieces.append(group.tail(per_split))
    sampled = pd.concat(pieces, ignore_index=True).sort_values(["as_of_date", "origin"]).tail(max_origins)
    return sampled.reset_index(drop=True)


def _final_values(context: pd.DataFrame) -> pd.DataFrame:
    return (
        context.sort_values(["series_id", "observation_date", "realtime_start"])
        .drop_duplicates(["series_id", "observation_date"], keep="last")
        .reset_index(drop=True)
    )


def _history_rows(series_context: pd.DataFrame, history_periods: int) -> pd.DataFrame:
    return (
        series_context.sort_values("observation_date")
        .drop_duplicates(["series_id", "observation_date"], keep="last")
        .tail(max(2, history_periods))
        .reset_index(drop=True)
    )


def _final_current_value(final_series: pd.DataFrame, current_observation: pd.Timestamp) -> float:
    rows = final_series[final_series["observation_date"] == current_observation]
    if rows.empty:
        return np.nan
    return float(rows.sort_values("observation_date").iloc[-1]["value"])


def _relative_history(history: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    n = history.shape[0]
    for idx, row in history.reset_index(drop=True).iterrows():
        rows.append(
            {
                "relative_period": int(idx - n + 1),
                "series_id": str(row["series_id"]),
                "value": round(float(row["value"]), 6),
            }
        )
    return rows


def _target_value(spec: VintageTargetSpec, *, current_value: float, target_raw: float) -> float:
    if spec.transform == "level":
        return float(target_raw)
    if abs(current_value) < 1e-12:
        return np.nan
    return float(100.0 * (target_raw / current_value - 1.0))


def _no_change_forecast(transform: str, current_value: float) -> float:
    return 0.0 if transform == "pct_change" else float(current_value)


def _rolling_mean_forecast(transform: str, values: list[float]) -> float:
    if transform == "pct_change":
        changes = _pct_changes(values)
        return float(np.mean(changes[-4:])) if changes else 0.0
    return float(np.mean(values[-4:]))


def _rolling_trend_forecast(transform: str, values: list[float]) -> float:
    if transform == "pct_change":
        changes = _pct_changes(values)
        return float(0.65 * changes[-1] + 0.35 * np.mean(changes[-4:])) if changes else 0.0
    if len(values) < 2:
        return float(values[-1])
    diffs = np.diff(values[-4:])
    return float(values[-1] + np.mean(diffs))


def _pct_changes(values: list[float]) -> list[float]:
    out = []
    for prev, cur in zip(values[:-1], values[1:]):
        if abs(prev) > 1e-12:
            out.append(100.0 * (cur / prev - 1.0))
    return out


def _parse_models(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_splits(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _resolve_forecast_cache_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir)
    if args.fresh_cache:
        return output_dir / "fresh_demand_vintage_oos_cache"
    return DEFAULT_FORECAST_CACHE_DIR


def _finite_payload_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMUnavailable(f"Vintage OOS forecast payload field {field_name} must be numeric") from exc
    if not np.isfinite(numeric):
        raise LLMUnavailable(f"Vintage OOS forecast payload field {field_name} must be finite")
    return float(numeric)


def _cache_json_count(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for path in cache_dir.rglob("*.json") if path.is_file())


def _vintage_bottom_line(manifest: dict[str, Any]) -> str:
    verdict = manifest.get("verdict")
    if verdict == "demand_vintage_oos_fixture_ready":
        return f"Verdict: `{verdict}`. Date-free card, hidden-target, baseline, scoring, and leakage-audit wiring is ready."
    if verdict == "demand_vintage_oos_scored":
        return f"Verdict: `{verdict}`. Scored vintage OOS artifacts are available for the macro performance gate."
    return f"Verdict: `{verdict}`. The vintage OOS runner needs work before it can feed empirical performance readiness."


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
        item = float(value)
        return item if np.isfinite(item) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
