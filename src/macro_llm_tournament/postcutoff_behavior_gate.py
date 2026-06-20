from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .agent_common import OUTPUT_ROOT, WORK_ROOT, cache_key, markdown_table, round_or_none
from .agent_llm import AgentLLMClient
from .agent_types import build_household_type_cells
from .env import load_secret_env
from .llm_common import LLMUnavailable


POSTCUTOFF_BEHAVIOR_VERSION = "postcutoff_household_behavior_proxy_v1"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_DOC_URL = "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"
DEFAULT_MODEL_CUTOFF_DATE = "2025-12-01"


@dataclass(frozen=True)
class ProxyTargetSpec:
    target_name: str
    series_id: str
    label: str
    units: str
    transform: str
    lower: float
    upper: float
    default_scale: float


@dataclass(frozen=True)
class ContextSeriesSpec:
    series_id: str
    label: str
    units: str


@dataclass(frozen=True)
class PostcutoffBehaviorCard:
    card_id: str
    period_id: str
    as_of_date: str
    target_month: str
    contamination_label: str
    prompt_payload: dict[str, Any]
    signal_history_by_target: dict[str, list[float]]


TARGET_SPECS: tuple[ProxyTargetSpec, ...] = (
    ProxyTargetSpec("pce_mom_pct", "PCE", "Nominal personal consumption expenditures MoM growth", "percent", "pct_change", -8.0, 8.0, 0.35),
    ProxyTargetSpec("real_pce_mom_pct", "PCECC96", "Real personal consumption expenditures MoM growth", "percent", "pct_change", -8.0, 8.0, 0.30),
    ProxyTargetSpec("retail_sales_mom_pct", "RSAFS", "Advance retail and food services sales MoM growth", "percent", "pct_change", -12.0, 12.0, 0.60),
    ProxyTargetSpec("personal_saving_rate_pct", "PSAVERT", "Personal saving rate", "percent", "level", 0.0, 25.0, 0.75),
    ProxyTargetSpec("revolving_credit_mom_pct", "REVOLSL", "Revolving consumer credit MoM growth", "percent", "pct_change", -6.0, 6.0, 0.35),
)


CONTEXT_SERIES: tuple[ContextSeriesSpec, ...] = (
    ContextSeriesSpec("PCE", "nominal consumption", "billions of dollars"),
    ContextSeriesSpec("PCECC96", "real consumption", "billions chained dollars"),
    ContextSeriesSpec("RSAFS", "retail sales", "millions of dollars"),
    ContextSeriesSpec("PSAVERT", "personal saving rate", "percent"),
    ContextSeriesSpec("REVOLSL", "revolving consumer credit", "millions of dollars"),
    ContextSeriesSpec("CPIAUCSL", "consumer price index", "index"),
    ContextSeriesSpec("UNRATE", "unemployment rate", "percent"),
    ContextSeriesSpec("FEDFUNDS", "effective federal funds rate", "percent"),
    ContextSeriesSpec("UMCSENT", "consumer sentiment", "index"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-cutoff household behavior proxy gate.")
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--agent-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--data-mode", choices=["fixture", "fred"], default="fred")
    parser.add_argument("--refresh-fred", action="store_true")
    parser.add_argument("--cutoff-date", default=DEFAULT_MODEL_CUTOFF_DATE)
    parser.add_argument("--asof-start", default="2025-12-15")
    parser.add_argument("--asof-end", default="2026-04-15")
    parser.add_argument("--history-months", type=int, default=18)
    parser.add_argument("--scoreable-only", action="store_true")
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.agent_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --agent-mode live is used")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"postcutoff_behavior_gate_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fresh_postcutoff_behavior_cache" if args.fresh_cache else WORK_ROOT / "postcutoff_behavior_llm_cache"

    manifest: dict[str, Any] = {
        "schema_version": POSTCUTOFF_BEHAVIOR_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "agent_mode": args.agent_mode,
        "max_live_calls": int(args.max_live_calls),
        "fresh_cache": bool(args.fresh_cache),
        "data_mode": args.data_mode,
        "cutoff_date": args.cutoff_date,
        "asof_start": args.asof_start,
        "asof_end": args.asof_end,
        "history_months": int(args.history_months),
        "scoreable_only": bool(args.scoreable_only),
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        frames, data_status = load_proxy_data(
            data_mode=args.data_mode,
            refresh=args.refresh_fred,
            cutoff_date=args.cutoff_date,
            asof_end=args.asof_end,
        )
        type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        cards, targets, raw_context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date=args.cutoff_date,
            asof_start=args.asof_start,
            asof_end=args.asof_end,
            history_months=args.history_months,
            scoreable_only=args.scoreable_only,
        )
        if not cards:
            raise ValueError("No post-cutoff behavior cards were built.")
        client = PostcutoffBehaviorLLMClient(args.provider, args.model, cache_dir, mode=args.agent_mode, max_live_calls=args.max_live_calls)
        llm_actions = run_postcutoff_behavior_agents(cards, type_cells, llm_client=client)
        llm_forecasts = aggregate_agent_proxy_forecasts(cards, llm_actions, source=f"llm_{args.provider}_{args.model}")
        controls = build_proxy_control_forecasts(cards)
        forecasts = pd.concat([llm_forecasts, controls], ignore_index=True)
        scores, joined = score_proxy_forecasts(forecasts, targets)

        cards_to_frame(cards).to_csv(output_dir / "postcutoff_behavior_cards.csv", index=False)
        raw_context.to_csv(output_dir / "postcutoff_behavior_context.csv", index=False)
        targets.to_csv(output_dir / "postcutoff_behavior_targets.csv", index=False)
        llm_actions.to_csv(output_dir / "postcutoff_behavior_agent_actions.csv", index=False)
        forecasts.to_csv(output_dir / "postcutoff_behavior_forecasts.csv", index=False)
        joined.to_csv(output_dir / "postcutoff_behavior_joined_errors.csv", index=False)
        scores.to_csv(output_dir / "postcutoff_behavior_scores.csv", index=False)
        type_cells.to_csv(output_dir / "household_type_cells.csv", index=False)
        (output_dir / "postcutoff_behavior_raw_records.json").write_text(json.dumps(client.raw_records, indent=2, sort_keys=True), encoding="utf-8")

        manifest.update(
            {
                "status": "ok",
                "data_status": data_status,
                "household_type_status": type_status,
                "household_type_count": int(type_cells.shape[0]),
                "card_count": int(len(cards)),
                "target_rows": int(targets.shape[0]),
                "scoreable_target_rows": int(targets["target_available"].sum()) if not targets.empty else 0,
                "frozen_unscored_target_rows": int((~targets["target_available"]).sum()) if not targets.empty else 0,
                "forecast_rows": int(forecasts.shape[0]),
                "score_rows": int(scores.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "cache_dir": str(cache_dir.relative_to(Path.cwd()) if cache_dir.is_relative_to(Path.cwd()) else cache_dir),
                "target_names": [spec.target_name for spec in TARGET_SPECS],
                "source_url": FRED_DOC_URL,
                "outputs": [
                    "postcutoff_behavior_cards.csv",
                    "postcutoff_behavior_context.csv",
                    "postcutoff_behavior_targets.csv",
                    "postcutoff_behavior_agent_actions.csv",
                    "postcutoff_behavior_forecasts.csv",
                    "postcutoff_behavior_joined_errors.csv",
                    "postcutoff_behavior_scores.csv",
                    "postcutoff_behavior_report.md",
                    "postcutoff_behavior_raw_records.json",
                ],
            }
        )
        report = build_postcutoff_behavior_report(manifest, scores, targets)
        (output_dir / "postcutoff_behavior_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


class PostcutoffBehaviorLLMClient:
    def __init__(self, provider: str, model: str, cache_dir: Path, *, mode: str, max_live_calls: int):
        self.provider = provider
        self.model = model
        self.mode = mode
        self._client = AgentLLMClient(provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls)
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._client.cache_hit_count

    def behavior_panel(self, card: PostcutoffBehaviorCard, type_cells: pd.DataFrame) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_postcutoff_behavior_payload(card, type_cells),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = postcutoff_behavior_prompt(card, type_cells)
            data = self._client._codex_call(prompt, f"postcutoff_behavior_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}")
        normalized = normalize_postcutoff_behavior_payload(card, type_cells, data)
        self.raw_records.append(
            {
                "card_id": card.card_id,
                "period_id": card.period_id,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized


def load_proxy_data(*, data_mode: str, refresh: bool, cutoff_date: str, asof_end: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if data_mode == "fixture":
        frames = fixture_fred_proxy_frames()
        return frames, {"status": "fixture", "series_count": len(frames), "source": "deterministic fixture"}
    load_secret_env()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise LLMUnavailable("FRED_API_KEY is required for post-cutoff behavior proxy data.")
    series_ids = sorted({spec.series_id for spec in CONTEXT_SERIES} | {spec.series_id for spec in TARGET_SPECS})
    observation_start = (pd.Timestamp(cutoff_date) - pd.DateOffset(months=36)).date().isoformat()
    observation_end = (pd.Timestamp(asof_end) + pd.DateOffset(months=3)).date().isoformat()
    work_dir = WORK_ROOT / "postcutoff_behavior_fred"
    frames = {
        series_id: fetch_fred_observations(
            series_id,
            api_key=api_key,
            work_dir=work_dir,
            observation_start=observation_start,
            observation_end=observation_end,
            realtime_date=None,
            refresh=refresh,
        )
        for series_id in series_ids
    }
    return {
        key: value for key, value in frames.items()
    }, {
        "status": "ok",
        "series_count": len(frames),
        "series_ids": series_ids,
        "observation_start": observation_start,
        "observation_end": observation_end,
        "source": "FRED current observations API",
        "source_url": FRED_DOC_URL,
    }


def fetch_fred_observations(
    series_id: str,
    *,
    api_key: str,
    work_dir: Path,
    observation_start: str,
    observation_end: str,
    realtime_date: str | None,
    refresh: bool = False,
) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "series_id": series_id,
        "observation_start": observation_start,
        "observation_end": observation_end,
        "realtime_date": realtime_date,
    }
    path = work_dir / f"{series_id}_{cache_key(cache_payload)}.json"
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return observations_to_frame(payload.get("observations", []))
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "observation_end": observation_end,
        "sort_order": "asc",
    }
    if realtime_date:
        params["realtime_start"] = realtime_date
        params["realtime_end"] = realtime_date
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return observations_to_frame(payload.get("observations", []))


def observations_to_frame(observations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in observations:
        value = pd.to_numeric(row.get("value"), errors="coerce")
        rows.append(
            {
                "date": pd.Timestamp(row.get("date")),
                "value": float(value) if np.isfinite(value) else np.nan,
                "realtime_start": row.get("realtime_start"),
                "realtime_end": row.get("realtime_end"),
            }
        )
    return pd.DataFrame(rows, columns=["date", "value", "realtime_start", "realtime_end"]).dropna(subset=["date"]).sort_values("date")


def build_postcutoff_behavior_cards(
    frames: dict[str, pd.DataFrame],
    *,
    cutoff_date: str,
    asof_start: str,
    asof_end: str,
    history_months: int,
    scoreable_only: bool,
) -> tuple[list[PostcutoffBehaviorCard], pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff_date)
    asof_dates = month_anchor_dates(asof_start, asof_end)
    cards: list[PostcutoffBehaviorCard] = []
    target_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for period_index, asof in enumerate(asof_dates, start=1):
        target_month = (asof + pd.DateOffset(months=1)).replace(day=1)
        signal_history_by_target: dict[str, list[float]] = {}
        signal_prompt_rows: list[dict[str, Any]] = []
        for spec in TARGET_SPECS:
            history = target_signal_history(frames[spec.series_id], spec)
            available = history[history["date"] <= asof].tail(history_months).copy()
            values = [float(value) for value in available["signal_value"].dropna().tolist()]
            signal_history_by_target[spec.target_name] = values
            for relative_period, (_, row) in enumerate(available.tail(8).iterrows(), start=-available.tail(8).shape[0] + 1):
                signal_prompt_rows.append(
                    {
                        "target_name": spec.target_name,
                        "relative_period": int(relative_period),
                        "value": round_or_none(row["signal_value"]),
                        "units": spec.units,
                    }
                )
            target_value = target_value_for_month(frames[spec.series_id], spec, target_month)
            scale = history_scale(values, spec.default_scale)
            last_signal = values[-1] if values else np.nan
            target_rows.append(
                {
                    "period_id": f"blind_post_cutoff_{period_index:02d}",
                    "as_of_date": asof.date().isoformat(),
                    "target_month": target_month.date().isoformat(),
                    "target_name": spec.target_name,
                    "target_label": spec.label,
                    "target_value": target_value,
                    "target_available": bool(np.isfinite(target_value)),
                    "last_signal": last_signal,
                    "history_scale": scale,
                    "series_id": spec.series_id,
                    "units": spec.units,
                    "contamination_label": "post_model_cutoff_clean" if target_month > cutoff else "pre_cutoff_or_boundary",
                    "source_url": f"https://fred.stlouisfed.org/series/{spec.series_id}",
                }
            )
        macro_prompt_rows = []
        for spec in CONTEXT_SERIES:
            series = frames[spec.series_id]
            visible = series[(series["date"] <= asof) & series["value"].map(np.isfinite)].tail(6)
            for relative_period, (_, row) in enumerate(visible.iterrows(), start=-visible.shape[0] + 1):
                macro_prompt_rows.append(
                    {
                        "series_label": spec.label,
                        "series_id": spec.series_id,
                        "relative_period": int(relative_period),
                        "value": round_or_none(row["value"]),
                        "units": spec.units,
                    }
                )
                context_rows.append(
                    {
                        "period_id": f"blind_post_cutoff_{period_index:02d}",
                        "as_of_date": asof.date().isoformat(),
                        "series_id": spec.series_id,
                        "series_label": spec.label,
                        "observation_date": row["date"].date().isoformat(),
                        "relative_period": int(relative_period),
                        "value": float(row["value"]),
                        "units": spec.units,
                    }
                )
        prompt_payload = {
            "prompt_version": POSTCUTOFF_BEHAVIOR_VERSION,
            "task": "Forecast next-month U.S. household behavior proxies from only the supplied post-cutoff as-of data.",
            "contamination_proof_design": {
                "target_month_hidden": True,
                "calendar_date_hidden": True,
                "event_label_hidden": True,
                "realized_targets_hidden": True,
                "all_targets_after_model_cutoff": True,
            },
            "period_id": f"blind_post_cutoff_{period_index:02d}",
            "horizon": "next calendar month",
            "available_behavior_signal_history": signal_prompt_rows,
            "available_macro_context": macro_prompt_rows,
            "required_response": {
                "household_actions": [
                    {
                        "type_id": "one supplied type_id",
                        **{spec.target_name: f"{spec.lower} to {spec.upper}, {spec.units}" for spec in TARGET_SPECS},
                        "consumption_change_pct": "-10 to 10 desired next-month consumption change",
                        "liquid_saving_change_pct": "-10 to 10 desired next-month liquid saving change",
                        "debt_balance_change_pct": "-10 to 10 desired next-month revolving debt balance change",
                        "confidence": "0 to 1",
                        "reason": "short reason",
                    }
                ]
            },
        }
        card_id = f"pcbeh_{cache_key(prompt_payload)}"
        card = PostcutoffBehaviorCard(
            card_id=card_id,
            period_id=f"blind_post_cutoff_{period_index:02d}",
            as_of_date=asof.date().isoformat(),
            target_month=target_month.date().isoformat(),
            contamination_label="post_model_cutoff_clean" if target_month > cutoff else "pre_cutoff_or_boundary",
            prompt_payload=prompt_payload,
            signal_history_by_target=signal_history_by_target,
        )
        cards.append(card)
    targets = pd.DataFrame(target_rows)
    if scoreable_only:
        scoreable_periods = set(targets[targets["target_available"]]["period_id"])
        cards = [card for card in cards if card.period_id in scoreable_periods]
        targets = targets[targets["period_id"].isin(scoreable_periods)].copy()
    targets["card_id"] = targets["period_id"].map({card.period_id: card.card_id for card in cards})
    return cards, targets.reset_index(drop=True), pd.DataFrame(context_rows)


def month_anchor_dates(asof_start: str, asof_end: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(asof_start)
    end = pd.Timestamp(asof_end)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current = current + pd.DateOffset(months=1)
    return dates


def target_signal_history(frame: pd.DataFrame, spec: ProxyTargetSpec) -> pd.DataFrame:
    clean = frame[frame["value"].map(np.isfinite)].sort_values("date").copy()
    if clean.empty:
        return pd.DataFrame(columns=["date", "signal_value"])
    if spec.transform == "level":
        clean["signal_value"] = clean["value"].astype(float)
    elif spec.transform == "pct_change":
        clean["signal_value"] = 100.0 * clean["value"].astype(float).pct_change()
    else:
        raise ValueError(f"Unsupported target transform: {spec.transform}")
    return clean[["date", "signal_value"]].dropna().reset_index(drop=True)


def target_value_for_month(frame: pd.DataFrame, spec: ProxyTargetSpec, target_month: pd.Timestamp) -> float:
    history = target_signal_history(frame, spec)
    if history.empty:
        return float("nan")
    match = history[history["date"] == target_month]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["signal_value"])


def history_scale(values: list[float], default_scale: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size < 3:
        return float(default_scale)
    scale = float(np.std(finite[-12:], ddof=0))
    return max(scale, float(default_scale))


def postcutoff_behavior_prompt(card: PostcutoffBehaviorCard, type_cells: pd.DataFrame) -> str:
    payload = dict(card.prompt_payload)
    payload["household_type_cells"] = [
        {
            "type_id": row["type_id"],
            "label": row["label"],
            "population_weight": round_or_none(row["population_weight"]),
            "annual_income": round_or_none(row["annual_income"]),
            "liquid_assets": round_or_none(row["liquid_assets"]),
            "illiquid_assets": round_or_none(row["illiquid_assets"]),
            "debt": round_or_none(row["debt"]),
            "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
        }
        for _, row in type_cells.iterrows()
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_postcutoff_behavior_payload(card: PostcutoffBehaviorCard, type_cells: pd.DataFrame) -> dict[str, Any]:
    last = {spec.target_name: last_or_default(card.signal_history_by_target.get(spec.target_name, []), 0.0) for spec in TARGET_SPECS}
    actions = []
    for _, row in type_cells.iterrows():
        liquidity = float(row.get("liquidity_sensitivity", 1.0))
        buffer_months = float(row.get("liquid_buffer_months", 2.0))
        out = {
            "type_id": str(row["type_id"]),
            "consumption_change_pct": float(np.clip(0.10 * liquidity - 0.01 * buffer_months, -10.0, 10.0)),
            "liquid_saving_change_pct": float(np.clip(0.02 * buffer_months, -10.0, 10.0)),
            "debt_balance_change_pct": float(np.clip(last["revolving_credit_mom_pct"], -10.0, 10.0)),
            "confidence": 0.55,
            "reason": "deterministic no-change fixture for post-cutoff behavior proxy gate",
        }
        for spec in TARGET_SPECS:
            out[spec.target_name] = float(np.clip(last[spec.target_name], spec.lower, spec.upper))
        actions.append(out)
    return {"prompt_version": POSTCUTOFF_BEHAVIOR_VERSION, "household_actions": actions}


def normalize_postcutoff_behavior_payload(card: PostcutoffBehaviorCard, type_cells: pd.DataFrame, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    actions = payload.get("household_actions")
    if not isinstance(actions, list):
        raise LLMUnavailable(f"Post-cutoff behavior payload for {card.card_id} is missing household_actions list")
    expected_ids = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        type_id = str(action.get("type_id", ""))
        if type_id not in expected_ids:
            continue
        row = {
            "consumption_change_pct": bounded_number(action, "consumption_change_pct", -10.0, 10.0),
            "liquid_saving_change_pct": bounded_number(action, "liquid_saving_change_pct", -10.0, 10.0),
            "debt_balance_change_pct": bounded_number(action, "debt_balance_change_pct", -10.0, 10.0),
            "confidence": bounded_number(action, "confidence", 0.0, 1.0),
            "reason": str(action.get("reason", ""))[:300],
        }
        for spec in TARGET_SPECS:
            row[spec.target_name] = bounded_number(action, spec.target_name, spec.lower, spec.upper)
        by_type[type_id] = row
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Post-cutoff behavior payload for {card.card_id} is missing type ids: {', '.join(missing)}")
    return {"household_by_type": by_type}


def run_postcutoff_behavior_agents(cards: Iterable[PostcutoffBehaviorCard], type_cells: pd.DataFrame, *, llm_client: PostcutoffBehaviorLLMClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        payload = llm_client.behavior_panel(card, type_cells)
        for _, type_cell in type_cells.iterrows():
            response = payload["household_by_type"][str(type_cell["type_id"])]
            rows.append(
                {
                    "card_id": card.card_id,
                    "period_id": card.period_id,
                    "as_of_date": card.as_of_date,
                    "target_month": card.target_month,
                    "source": f"llm_{llm_client.provider}_{llm_client.model}",
                    "type_id": str(type_cell["type_id"]),
                    "population_weight": float(type_cell["population_weight"]),
                    "confidence": float(response["confidence"]),
                    "consumption_change_pct": float(response["consumption_change_pct"]),
                    "liquid_saving_change_pct": float(response["liquid_saving_change_pct"]),
                    "debt_balance_change_pct": float(response["debt_balance_change_pct"]),
                    "reason": str(response.get("reason", "")),
                    **{spec.target_name: float(response[spec.target_name]) for spec in TARGET_SPECS},
                }
            )
    return pd.DataFrame(rows)


def aggregate_agent_proxy_forecasts(cards: Iterable[PostcutoffBehaviorCard], actions: pd.DataFrame, *, source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    card_by_id = {card.card_id: card for card in cards}
    for card_id, group in actions.groupby("card_id", dropna=False):
        card = card_by_id[str(card_id)]
        weights = group["population_weight"].astype(float).clip(lower=0.0)
        total_weight = float(weights.sum()) or 1.0
        for spec in TARGET_SPECS:
            prediction = float((group[spec.target_name].astype(float) * weights).sum() / total_weight)
            rows.append(
                {
                    "card_id": card.card_id,
                    "period_id": card.period_id,
                    "target_name": spec.target_name,
                    "source": source,
                    "prediction": prediction,
                    "method": "weighted_typed_household_llm",
                }
            )
    return pd.DataFrame(rows)


def build_proxy_control_forecasts(cards: Iterable[PostcutoffBehaviorCard]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controls = {
        "no_change": no_change_forecast,
        "rolling_mean_3": lambda values: rolling_mean_forecast(values, 3),
        "rolling_mean_6": lambda values: rolling_mean_forecast(values, 6),
        "linear_trend_6": lambda values: linear_trend_forecast(values, 6),
    }
    for card in cards:
        for spec in TARGET_SPECS:
            values = card.signal_history_by_target.get(spec.target_name, [])
            for source, fn in controls.items():
                rows.append(
                    {
                        "card_id": card.card_id,
                        "period_id": card.period_id,
                        "target_name": spec.target_name,
                        "source": source,
                        "prediction": float(np.clip(fn(values), spec.lower, spec.upper)),
                        "method": source,
                    }
                )
    return pd.DataFrame(rows)


def score_proxy_forecasts(forecasts: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if forecasts.empty or targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    target_cols = [
        "card_id",
        "period_id",
        "target_month",
        "target_name",
        "target_value",
        "target_available",
        "last_signal",
        "history_scale",
        "contamination_label",
        "source_url",
    ]
    joined = forecasts.merge(targets[target_cols], on=["card_id", "period_id", "target_name"], how="inner")
    joined = joined[joined["target_available"]].copy()
    if joined.empty:
        return pd.DataFrame(), joined
    joined["error"] = joined["prediction"].astype(float) - joined["target_value"].astype(float)
    joined["abs_error"] = joined["error"].abs()
    joined["scaled_error"] = joined["error"] / joined["history_scale"].astype(float).clip(lower=1e-9)
    joined["abs_scaled_error"] = joined["scaled_error"].abs()
    joined["direction_correct"] = np.sign(joined["prediction"] - joined["last_signal"]) == np.sign(joined["target_value"] - joined["last_signal"])
    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(["source", "target_name"], dropna=False):
        source, target_name = keys
        rows.append(score_group(group, source=str(source), target_name=str(target_name)))
    for source, group in joined.groupby("source", dropna=False):
        rows.append(score_group(group, source=str(source), target_name="ALL"))
    scores = pd.DataFrame(rows).sort_values(["target_name", "rmse_scaled", "source"]).reset_index(drop=True)
    return scores, joined


def score_group(group: pd.DataFrame, *, source: str, target_name: str) -> dict[str, Any]:
    error = group["error"].astype(float)
    scaled = group["scaled_error"].astype(float)
    return {
        "source": source,
        "target_name": target_name,
        "n": int(group.shape[0]),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rmse_scaled": float(np.sqrt(np.mean(np.square(scaled)))),
        "mae_scaled": float(np.mean(np.abs(scaled))),
        "direction_accuracy": float(group["direction_correct"].mean()),
        "mean_prediction": float(group["prediction"].mean()),
        "mean_target": float(group["target_value"].mean()),
    }


def build_postcutoff_behavior_report(manifest: dict[str, Any], scores: pd.DataFrame, targets: pd.DataFrame) -> str:
    lines = [
        "# Post-Cutoff Household Behavior Proxy Gate",
        "",
        "## Bottom Line",
        postcutoff_behavior_bottom_line(scores),
        "",
        "## Contamination Design",
        "- Target months are after the configured model cutoff.",
        "- Prompt payloads hide target months, calendar dates, event labels, and realized targets.",
        "- Missing future target rows are frozen rather than imputed.",
        "- This is an aggregate proxy gate using public FRED series, not a replacement for micro household bank-account data.",
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- Agent mode: `{manifest.get('agent_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Cards: `{manifest.get('card_count')}`",
        f"- Scoreable targets: `{manifest.get('scoreable_target_rows')}`",
        f"- Frozen unscored targets: `{manifest.get('frozen_unscored_target_rows')}`",
        f"- Data source: `{manifest.get('data_status', {}).get('source', 'unknown')}`",
        "",
        "## Scoreboard",
        markdown_table(scores[scores["target_name"] == "ALL"].sort_values(["rmse_scaled", "source"]) if not scores.empty else scores),
        "",
        "## By Target",
        markdown_table(scores[scores["target_name"] != "ALL"].sort_values(["target_name", "rmse_scaled", "source"]) if not scores.empty else scores),
        "",
        "## Target Availability",
        markdown_table(targets.groupby(["target_month", "target_available"], dropna=False).size().reset_index(name="rows") if not targets.empty else targets),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def postcutoff_behavior_bottom_line(scores: pd.DataFrame) -> str:
    if scores.empty:
        return "No scoreable post-cutoff behavior targets are available yet."
    overall = scores[scores["target_name"] == "ALL"].sort_values("rmse_scaled")
    if overall.empty:
        return "No overall score row was produced."
    best = overall.iloc[0]
    return f"Best overall source is `{best['source']}` with scaled RMSE `{float(best['rmse_scaled']):.4f}` across `{int(best['n'])}` post-cutoff behavior proxy targets."


def cards_to_frame(cards: Iterable[PostcutoffBehaviorCard]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "card_id": card.card_id,
                "period_id": card.period_id,
                "as_of_date": card.as_of_date,
                "target_month": card.target_month,
                "contamination_label": card.contamination_label,
                "prompt_payload": json.dumps(card.prompt_payload, sort_keys=True),
            }
            for card in cards
        ]
    )


def bounded_number(mapping: dict[str, Any], key: str, lower: float, upper: float) -> float:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError):
        raise LLMUnavailable(f"Post-cutoff behavior payload field {key} must be numeric") from None
    if not np.isfinite(value):
        raise LLMUnavailable(f"Post-cutoff behavior payload field {key} must be finite")
    return float(np.clip(value, lower, upper))


def no_change_forecast(values: list[float]) -> float:
    return last_or_default(values, 0.0)


def rolling_mean_forecast(values: list[float], window: int) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 0.0
    return float(np.mean(finite[-window:]))


def linear_trend_forecast(values: list[float], window: int) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size < 3:
        return no_change_forecast(values)
    y = finite[-window:]
    x = np.arange(y.size, dtype=float)
    beta = np.polyfit(x, y, deg=1)
    return float(beta[0] * y.size + beta[1])


def last_or_default(values: list[float], default: float) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(finite[-1]) if finite else float(default)


def fixture_fred_proxy_frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", "2026-05-01", freq="MS")
    frames: dict[str, pd.DataFrame] = {}
    base = np.arange(len(dates), dtype=float)
    series_values = {
        "PCE": 19000.0 + 75.0 * base + 20.0 * np.sin(base / 3.0),
        "PCECC96": 15500.0 + 30.0 * base + 10.0 * np.cos(base / 4.0),
        "RSAFS": 690000.0 + 1800.0 * base + 2500.0 * np.sin(base / 2.0),
        "PSAVERT": 4.5 - 0.05 * base + 0.25 * np.sin(base / 5.0),
        "REVOLSL": 1250000.0 + 4200.0 * base + 4000.0 * np.cos(base / 6.0),
        "CPIAUCSL": 305.0 + 0.7 * base,
        "UNRATE": 3.8 + 0.02 * base,
        "FEDFUNDS": 5.3 - 0.03 * base,
        "UMCSENT": 68.0 + 0.5 * np.sin(base / 2.0),
    }
    for series_id, values in series_values.items():
        frames[series_id] = pd.DataFrame({"date": dates, "value": values, "realtime_start": None, "realtime_end": None})
    return frames


if __name__ == "__main__":
    raise SystemExit(main())
