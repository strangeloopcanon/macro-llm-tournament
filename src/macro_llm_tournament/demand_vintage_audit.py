from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, markdown_table
from .forecast_audit import (
    DirectRecallClient,
    build_belief_structure_audit,
    build_paired_loss_tests,
    build_qualitative_recall_targets,
    build_source_scores,
    build_theil_u,
    fixture_qualitative_recall_payload,
    normalize_qualitative_recall_payload,
    qualitative_recall_cache_name,
    qualitative_recall_call,
    qualitative_recall_prompt,
    run_direct_recall,
    score_direct_recall,
    score_qualitative_recall,
)
from .forecast_llm import SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable


DEMAND_VINTAGE_AUDIT_VERSION = "demand_vintage_audit_v1"
DEFAULT_RUN_DIR = OUTPUT_ROOT / "belief_calibration_gpt55_gpt54_val_to_test_consistent_targets"
TARGET_LABELS = {
    "real_consumption_growth_pct": "Real consumption growth",
    "saving_rate_level": "Personal saving rate",
    "inflation_growth_pct": "Consumer price inflation",
    "unemployment_rate_level": "Unemployment rate",
    "policy_rate_level": "Policy rate",
    "sentiment_growth_pct": "Consumer sentiment growth",
    "output_growth_pct": "Real output growth",
}
MODEL_CUTOFFS = {
    "gpt-5-codex-user-supplied": "2024-09-30",
    "gpt-5.4-user-supplied": "2025-08-31",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the demand-vintage OOS split for recall and belief dynamics.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5,gpt-5.4")
    parser.add_argument("--recall-mode", choices=["off", "fixture", "replay", "live"], default="off")
    parser.add_argument("--qualitative-recall-mode", choices=["off", "fixture", "replay", "live"], default="off")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--recall-batch-size", type=int, default=49)
    parser.add_argument("--qualitative-recall-batch-size", type=int, default=49)
    parser.add_argument("--qualitative-recall-min-batch-size", type=int, default=12)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--provider-cwd", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [part.strip() for part in str(args.models).split(",") if part.strip()]
    if not models:
        raise SystemExit("--models must contain at least one model")
    if (args.recall_mode == "live" or args.qualitative_recall_mode == "live") and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive for live recall modes")
    if (args.recall_mode == "live" or args.qualitative_recall_mode == "live") and not args.fresh_cache:
        raise SystemExit("--fresh-cache is required for live recall modes")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"demand_vintage_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cards_raw, targets, forecasts, joined_raw = load_demand_vintage_run(run_dir)
    cards = build_audit_cards(cards_raw, targets, forecasts)
    joined = build_audit_joined(cards, joined_raw)
    cutoff_status = build_cutoff_status(cards)
    source_scores = build_source_scores(joined)
    theil = build_theil_u(joined)
    paired = build_paired_loss_tests(joined, baseline="no_change")
    belief_detail, belief_summary = build_belief_structure_audit(joined)
    qualitative_targets = build_qualitative_recall_targets(cards, joined)

    required_live_calls = estimate_live_calls(
        card_count=int(cards.shape[0]),
        qualitative_count=int(qualitative_targets.shape[0]),
        model_count=len(models),
        recall_mode=args.recall_mode,
        qualitative_recall_mode=args.qualitative_recall_mode,
        recall_batch_size=args.recall_batch_size,
        qualitative_batch_size=args.qualitative_recall_batch_size,
    )
    adaptive_live_call_ceiling = estimate_live_calls(
        card_count=int(cards.shape[0]),
        qualitative_count=int(qualitative_targets.shape[0]),
        model_count=len(models),
        recall_mode=args.recall_mode,
        qualitative_recall_mode=args.qualitative_recall_mode,
        recall_batch_size=args.recall_batch_size,
        qualitative_batch_size=min(int(args.qualitative_recall_batch_size), int(args.qualitative_recall_min_batch_size)),
    )
    if required_live_calls and int(args.max_live_calls) < required_live_calls:
        raise SystemExit(f"--max-live-calls must be at least {required_live_calls} for this fresh audit")

    cache_root = (
        Path(args.cache_root)
        if args.cache_root
        else output_dir / "fresh_demand_vintage_audit_cache"
        if args.fresh_cache
        else WORK_ROOT / "demand_vintage_audit_cache"
    )
    provider_cwd = Path(args.provider_cwd) if args.provider_cwd else output_dir / "provider_cwd"
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": DEMAND_VINTAGE_AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "provider": args.provider,
        "models": models,
        "recall_mode": args.recall_mode,
        "qualitative_recall_mode": args.qualitative_recall_mode,
        "max_live_calls": int(args.max_live_calls),
        "required_live_calls": int(required_live_calls),
        "required_live_calls_lower_bound": int(required_live_calls),
        "adaptive_live_call_ceiling": int(adaptive_live_call_ceiling),
        "recall_batch_size": int(args.recall_batch_size),
        "qualitative_recall_batch_size": int(args.qualitative_recall_batch_size),
        "qualitative_recall_min_batch_size": int(args.qualitative_recall_min_batch_size),
        "cache_root": str(cache_root),
        "provider_cwd": str(provider_cwd),
        "fresh_cache": bool(args.fresh_cache),
        "card_count": int(cards.shape[0]),
        "forecast_rows": int(forecasts.shape[0]),
        "joined_rows": int(joined.shape[0]),
        "qualitative_recall_target_rows": int(qualitative_targets.shape[0]),
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    try:
        (
            recall_predictions,
            recall_scores,
            direct_raw,
            qualitative_predictions,
            qualitative_scores,
            qualitative_raw,
            live_calls,
            cache_hits,
        ) = run_model_recalls(
            cards,
            qualitative_targets,
            provider=args.provider,
            models=models,
            recall_mode=args.recall_mode,
            qualitative_recall_mode=args.qualitative_recall_mode,
            max_live_calls=int(args.max_live_calls),
            recall_batch_size=int(args.recall_batch_size),
            qualitative_batch_size=int(args.qualitative_recall_batch_size),
            qualitative_min_batch_size=int(args.qualitative_recall_min_batch_size),
            fresh_cache=bool(args.fresh_cache),
            cache_root=cache_root,
            provider_cwd=provider_cwd,
            output_dir=output_dir,
        )
        cards.to_csv(output_dir / "demand_vintage_audit_cards.csv", index=False)
        joined.to_csv(output_dir / "demand_vintage_audit_joined.csv", index=False)
        source_scores.to_csv(output_dir / "audit_source_scores.csv", index=False)
        theil.to_csv(output_dir / "audit_theils_u.csv", index=False)
        paired.to_csv(output_dir / "audit_paired_loss_tests.csv", index=False)
        belief_detail.to_csv(output_dir / "audit_belief_structure.csv", index=False)
        belief_summary.to_csv(output_dir / "audit_belief_structure_summary.csv", index=False)
        cutoff_status.to_csv(output_dir / "cutoff_status.csv", index=False)
        qualitative_targets.to_csv(output_dir / "qualitative_recall_targets.csv", index=False)
        recall_predictions.to_csv(output_dir / "direct_recall_predictions.csv", index=False)
        recall_scores.to_csv(output_dir / "direct_recall_scores.csv", index=False)
        qualitative_predictions.to_csv(output_dir / "qualitative_recall_predictions.csv", index=False)
        qualitative_scores.to_csv(output_dir / "qualitative_recall_scores.csv", index=False)
        (output_dir / "direct_recall_raw_records.json").write_text(
            json.dumps(direct_raw, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "qualitative_recall_raw_records.json").write_text(
            json.dumps(qualitative_raw, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        manifest.update(
            {
                "status": "ok",
                "live_call_count": int(live_calls),
                "cache_hit_count": int(cache_hits),
                "direct_recall_prediction_rows": int(recall_predictions.shape[0]),
                "direct_recall_score_rows": int(recall_scores.shape[0]),
                "qualitative_recall_prediction_rows": int(qualitative_predictions.shape[0]),
                "qualitative_recall_score_rows": int(qualitative_scores.shape[0]),
                "belief_structure_summary_rows": int(belief_summary.shape[0]),
                "outputs": [
                    "demand_vintage_audit_cards.csv",
                    "demand_vintage_audit_joined.csv",
                    "audit_source_scores.csv",
                    "audit_theils_u.csv",
                    "audit_paired_loss_tests.csv",
                    "audit_belief_structure.csv",
                    "audit_belief_structure_summary.csv",
                    "cutoff_status.csv",
                    "direct_recall_predictions.csv",
                    "direct_recall_scores.csv",
                    "direct_recall_raw_records.json",
                    "qualitative_recall_targets.csv",
                    "qualitative_recall_predictions.csv",
                    "qualitative_recall_scores.csv",
                    "qualitative_recall_raw_records.json",
                    "demand_vintage_audit_report.md",
                    "manifest.json",
                ],
            }
        )
        report = build_report(manifest, cutoff_status, recall_scores, qualitative_scores, belief_summary, theil, paired)
        (output_dir / "demand_vintage_audit_report.md").write_text(report, encoding="utf-8")
        manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)}"})
        manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def load_demand_vintage_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "demand_vintage_oos_cards.csv": "cards",
        "demand_vintage_oos_targets.csv": "targets",
        "demand_vintage_oos_forecasts.csv": "forecasts",
        "demand_vintage_oos_joined_errors.csv": "joined",
    }
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Demand vintage run missing required file(s): {', '.join(missing)}")
    return (
        pd.read_csv(run_dir / "demand_vintage_oos_cards.csv"),
        pd.read_csv(run_dir / "demand_vintage_oos_targets.csv"),
        pd.read_csv(run_dir / "demand_vintage_oos_forecasts.csv"),
        pd.read_csv(run_dir / "demand_vintage_oos_joined_errors.csv"),
    )


def build_audit_cards(cards: pd.DataFrame, targets: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    merged = cards.merge(targets, on=["card_id", "origin_id", "split", "target_name", "series_id", "transform"], how="inner")
    no_change = forecasts[forecasts["source"].astype(str) == "no_change"][["card_id", "forecast_value"]].rename(
        columns={"forecast_value": "asof_reference_value"}
    )
    merged = merged.merge(no_change, on="card_id", how="left")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        history = json.loads(str(row["history_json"]))
        history_values = [float(item["value"]) for item in history if np.isfinite(float(item["value"]))]
        mean4, recent_change = history_features(str(row["transform"]), history_values)
        target_date = pd.Timestamp(row["target_observation_date"])
        rows.append(
            {
                "card_id": row["card_id"],
                "origin_id": row["origin_id"],
                "split": row["split"],
                "variable": row["target_name"],
                "variable_name": TARGET_LABELS.get(str(row["target_name"]), str(row["target_name"])),
                "origin": target_date.date().isoformat(),
                "forecast_origin": row["origin"],
                "horizon": 1,
                "units": "percent" if row["transform"] == "pct_change" else "level",
                "target_realized": float(row["target_value"]),
                "asof_reference_value": float(row["asof_reference_value"]),
                "prior_spf_forecast": float(row["asof_reference_value"]),
                "rolling_signal_mean_4": mean4,
                "recent_signal_change_4": recent_change,
                "target_observation_date": target_date.date().isoformat(),
                "as_of_date": row["as_of_date"],
                "regime_label": regime_label(target_date),
                "contamination_label": "post_model_cutoff_2024_09_30"
                if target_date > pd.Timestamp("2024-09-30")
                else "pre_model_cutoff_2024_09_30",
            }
        )
    return pd.DataFrame(rows).sort_values(["origin_id", "variable"]).reset_index(drop=True)


def build_audit_joined(cards: pd.DataFrame, joined_raw: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "card_id",
        "origin_id",
        "variable",
        "variable_name",
        "origin",
        "forecast_origin",
        "horizon",
        "target_realized",
        "asof_reference_value",
        "prior_spf_forecast",
        "rolling_signal_mean_4",
        "recent_signal_change_4",
        "regime_label",
        "contamination_label",
    ]
    joined = joined_raw.merge(cards[base_columns], on=["card_id", "origin_id"], how="inner")
    joined = joined.rename(columns={"forecast_value": "point_forecast"})
    joined["provider"] = np.where(joined["source"].astype(str).str.startswith("llm_"), "llm", "control")
    joined["model"] = joined["source"]
    joined["p10"] = np.nan
    joined["p50"] = np.nan
    joined["p90"] = np.nan
    joined["panel_mean"] = np.nan
    joined["panel_std"] = np.nan
    joined["error"] = pd.to_numeric(joined["point_forecast"], errors="coerce") - pd.to_numeric(joined["target_realized"], errors="coerce")
    joined["target_change"] = pd.to_numeric(joined["target_realized"], errors="coerce") - pd.to_numeric(
        joined["asof_reference_value"], errors="coerce"
    )
    joined["forecast_change"] = pd.to_numeric(joined["point_forecast"], errors="coerce") - pd.to_numeric(
        joined["asof_reference_value"], errors="coerce"
    )
    joined["direction_correct"] = np.sign(joined["target_change"]) == np.sign(joined["forecast_change"])
    joined["squared_error"] = joined["error"].astype(float) ** 2
    joined["abs_error"] = joined["error"].abs()
    return joined.reset_index(drop=True)


def history_features(transform: str, values: list[float]) -> tuple[float, float]:
    if not values:
        return np.nan, np.nan
    if transform == "pct_change":
        signal = []
        for prev, cur in zip(values[:-1], values[1:]):
            if abs(prev) > 1e-12:
                signal.append(100.0 * (cur / prev - 1.0))
    else:
        signal = list(values)
    if not signal:
        return 0.0, 0.0
    mean4 = float(np.mean(signal[-4:]))
    recent = float(signal[-1] - signal[-min(4, len(signal))]) if len(signal) > 1 else 0.0
    return mean4, recent


def regime_label(target_date: pd.Timestamp) -> str:
    if pd.Timestamp("2020-03-01") <= target_date <= pd.Timestamp("2020-06-30"):
        return "covid_shock"
    if pd.Timestamp("2021-04-01") <= target_date <= pd.Timestamp("2022-12-31"):
        return "inflation_surge"
    return "normal"


def build_cutoff_status(cards: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(cards["target_observation_date"], errors="coerce")
    rows = []
    for label, cutoff in MODEL_CUTOFFS.items():
        cutoff_date = pd.Timestamp(cutoff)
        post = dates > cutoff_date
        rows.append(
            {
                "cutoff_label": label,
                "cutoff_date": cutoff_date.date().isoformat(),
                "card_count": int(cards.shape[0]),
                "post_cutoff_card_count": int(post.sum()),
                "post_cutoff_share": float(post.mean()) if len(post) else np.nan,
                "post_cutoff_origin_count": int(cards.loc[post, "origin_id"].nunique()),
                "target_date_min": dates.min().date().isoformat() if dates.notna().any() else "",
                "target_date_max": dates.max().date().isoformat() if dates.notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def estimate_live_calls(
    *,
    card_count: int,
    qualitative_count: int,
    model_count: int,
    recall_mode: str,
    qualitative_recall_mode: str,
    recall_batch_size: int,
    qualitative_batch_size: int,
) -> int:
    direct = math.ceil(card_count / max(1, int(recall_batch_size))) if recall_mode == "live" else 0
    qualitative = math.ceil(qualitative_count / max(1, int(qualitative_batch_size))) if qualitative_recall_mode == "live" else 0
    return int((direct + qualitative) * model_count)


def run_model_recalls(
    cards: pd.DataFrame,
    qualitative_targets: pd.DataFrame,
    *,
    provider: str,
    models: list[str],
    recall_mode: str,
    qualitative_recall_mode: str,
    max_live_calls: int,
    recall_batch_size: int,
    qualitative_batch_size: int,
    qualitative_min_batch_size: int,
    fresh_cache: bool,
    cache_root: Path | None,
    provider_cwd: Path | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame, list[dict[str, Any]], int, int]:
    direct_predictions: list[pd.DataFrame] = []
    direct_scores: list[pd.DataFrame] = []
    qualitative_predictions: list[pd.DataFrame] = []
    qualitative_scores: list[pd.DataFrame] = []
    direct_raw: list[dict[str, Any]] = []
    qualitative_raw: list[dict[str, Any]] = []
    live_used = 0
    cache_hits = 0
    resolved_cache_root = cache_root or (
        output_dir / "fresh_demand_vintage_audit_cache" if fresh_cache else WORK_ROOT / "demand_vintage_audit_cache"
    )
    resolved_provider_cwd = provider_cwd or output_dir / "provider_cwd"
    resolved_provider_cwd.mkdir(parents=True, exist_ok=True)
    for model in models:
        direct_client = DirectRecallClient(
            provider,
            model,
            resolved_cache_root,
            mode=recall_mode,
            max_live_calls=max(0, max_live_calls - live_used),
            execution_cwd=resolved_provider_cwd,
        )
        if recall_mode != "off":
            predictions, raw, direct_client = run_direct_recall(cards, direct_client, batch_size=recall_batch_size)
            predictions["provider"] = provider
            predictions["model"] = model
            scores = score_direct_recall(cards, predictions)
            scores["provider"] = provider
            scores["model"] = model
            direct_predictions.append(predictions)
            direct_scores.append(scores)
            direct_raw.extend([{**record, "model": model, "provider": provider} for record in raw])
            live_used += direct_client.live_call_count
            cache_hits += direct_client.cache_hit_count
        qualitative_client = DirectRecallClient(
            provider,
            model,
            resolved_cache_root,
            mode=qualitative_recall_mode,
            max_live_calls=max(0, max_live_calls - live_used),
            execution_cwd=resolved_provider_cwd,
        )
        if qualitative_recall_mode != "off":
            predictions, raw, qualitative_client = run_qualitative_recall_adaptive(
                qualitative_targets,
                qualitative_client,
                batch_size=qualitative_batch_size,
                min_batch_size=qualitative_min_batch_size,
            )
            predictions["provider"] = provider
            predictions["model"] = model
            scores = score_qualitative_recall(qualitative_targets, predictions)
            scores["provider"] = provider
            scores["model"] = model
            qualitative_predictions.append(predictions)
            qualitative_scores.append(scores)
            qualitative_raw.extend([{**record, "model": model, "provider": provider} for record in raw])
            live_used += qualitative_client.live_call_count
            cache_hits += qualitative_client.cache_hit_count
    return (
        pd.concat(direct_predictions, ignore_index=True) if direct_predictions else pd.DataFrame(),
        pd.concat(direct_scores, ignore_index=True) if direct_scores else pd.DataFrame(),
        direct_raw,
        pd.concat(qualitative_predictions, ignore_index=True) if qualitative_predictions else pd.DataFrame(),
        pd.concat(qualitative_scores, ignore_index=True) if qualitative_scores else pd.DataFrame(),
        qualitative_raw,
        live_used,
        cache_hits,
    )


def run_qualitative_recall_adaptive(
    targets: pd.DataFrame,
    client: DirectRecallClient,
    *,
    batch_size: int,
    min_batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], DirectRecallClient]:
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    min_batch_size = max(1, int(min_batch_size))
    for start in range(0, targets.shape[0], batch_size):
        batch = targets.iloc[start : start + batch_size].copy()
        batch_rows, batch_raw, client = _run_qualitative_recall_batch_adaptive(
            batch,
            client,
            start=start,
            min_batch_size=min_batch_size,
        )
        rows.extend(batch_rows)
        raw_records.extend(batch_raw)
    return pd.DataFrame(rows), raw_records, client


def _run_qualitative_recall_batch_adaptive(
    batch: pd.DataFrame,
    client: DirectRecallClient,
    *,
    start: int,
    min_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], DirectRecallClient]:
    if batch.empty:
        return [], [], client
    if client.mode in {"live", "replay"} and batch.shape[0] > min_batch_size:
        prompt = qualitative_recall_prompt(batch)
        cache_name = qualitative_recall_cache_name(client, prompt)
        if not _client_cache_path(client, cache_name).exists():
            return _split_qualitative_recall_batch(batch, client, start=start, min_batch_size=min_batch_size)
    try:
        rows, raw_record, client = _run_qualitative_recall_batch_once(batch, client, start=start)
        return rows, [raw_record], client
    except LLMUnavailable:
        if client.mode not in {"live", "replay"} or batch.shape[0] <= min_batch_size:
            raise
        return _split_qualitative_recall_batch(batch, client, start=start, min_batch_size=min_batch_size)


def _split_qualitative_recall_batch(
    batch: pd.DataFrame,
    client: DirectRecallClient,
    *,
    start: int,
    min_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], DirectRecallClient]:
    midpoint = max(1, batch.shape[0] // 2)
    left = batch.iloc[:midpoint].copy()
    right = batch.iloc[midpoint:].copy()
    left_rows, left_raw, client = _run_qualitative_recall_batch_adaptive(
        left,
        client,
        start=start,
        min_batch_size=min_batch_size,
    )
    right_rows, right_raw, client = _run_qualitative_recall_batch_adaptive(
        right,
        client,
        start=start + midpoint,
        min_batch_size=min_batch_size,
    )
    split_record = {
        "batch_start": int(start),
        "batch_size": int(batch.shape[0]),
        "provider": client.provider,
        "model": client.model,
        "cache_hit": False,
        "cache_path": None,
        "split_batch": True,
        "child_batch_sizes": [int(left.shape[0]), int(right.shape[0])],
        "payload": {
            "reason": f"{client.mode} qualitative recall batch split because no exact parent-batch cache existed"
        },
    }
    return left_rows + right_rows, [split_record, *left_raw, *right_raw], client


def _run_qualitative_recall_batch_once(
    batch: pd.DataFrame,
    client: DirectRecallClient,
    *,
    start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], DirectRecallClient]:
    if client.mode == "fixture":
        payload = fixture_qualitative_recall_payload(batch)
        data = {
            "provider": client.provider,
            "model": client.model,
            "payload": payload,
            "cache_hit": True,
            "cache_path": None,
        }
    else:
        prompt = qualitative_recall_prompt(batch)
        data, client = qualitative_recall_call(client, prompt, qualitative_recall_cache_name(client, prompt))
    rows = normalize_qualitative_recall_payload(batch, data)
    raw_record = {
        "batch_start": int(start),
        "batch_size": int(batch.shape[0]),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "cache_hit": bool(data.get("cache_hit", False)),
        "cache_path": data.get("cache_path"),
        "split_batch": False,
        "payload": data.get("payload", data),
    }
    return rows, raw_record, client


def _client_cache_path(client: DirectRecallClient, cache_name: str) -> Path:
    return client.cache_dir / client.provider / f"{cache_name}.json"


def build_report(
    manifest: dict[str, Any],
    cutoff_status: pd.DataFrame,
    recall_scores: pd.DataFrame,
    qualitative_scores: pd.DataFrame,
    belief_summary: pd.DataFrame,
    theil: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    recall_all = recall_scores[recall_scores["variable"].astype(str) == "ALL"].copy() if not recall_scores.empty else pd.DataFrame()
    qualitative_all = (
        qualitative_scores[
            (qualitative_scores["group_type"].astype(str) == "ALL") & (qualitative_scores["group"].astype(str) == "ALL")
        ].copy()
        if not qualitative_scores.empty
        else pd.DataFrame()
    )
    belief_all = belief_summary[belief_summary["variable"].astype(str) == "ALL"].copy() if not belief_summary.empty else pd.DataFrame()
    theil_all = theil[theil["variable"].astype(str) == "ALL"].copy() if not theil.empty else pd.DataFrame()
    paired_all = paired[paired["variable"].astype(str) == "ALL"].copy() if not paired.empty else pd.DataFrame()
    lines = [
        "# Demand Vintage Audit",
        "",
        "## Bottom Line",
        audit_bottom_line(manifest, cutoff_status, recall_all, qualitative_all),
        "",
        "## Cutoff Status",
        markdown_table(cutoff_status),
        "",
        "## Direct Recall",
        markdown_table(recall_all),
        "",
        "## Qualitative Recall",
        markdown_table(qualitative_all),
        "",
        "## Belief Structure",
        markdown_table(
            belief_all[
                [
                    "source",
                    "n",
                    "underreaction_slope_error_on_revision",
                    "extrapolation_slope_forecast_change_on_recent_signal",
                    "mean_panel_std",
                    "confidence_abs_error_corr",
                    "surprise_rmse_gap_high_minus_low_spf_error",
                ]
            ]
            if not belief_all.empty
            else belief_all
        ),
        "",
        "## Theil's U Vs No-Change",
        markdown_table(theil_all.sort_values("theils_u_vs_no_change") if not theil_all.empty else theil_all),
        "",
        "## Paired Loss Difference Vs No-Change",
        markdown_table(paired_all.sort_values("mean_abs_loss_diff_vs_baseline") if not paired_all.empty else paired_all),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def audit_bottom_line(
    manifest: dict[str, Any],
    cutoff_status: pd.DataFrame,
    recall_all: pd.DataFrame,
    qualitative_all: pd.DataFrame,
) -> str:
    parts = [
        f"Audited `{manifest.get('card_count')}` demand-vintage cards for `{', '.join(manifest.get('models', []))}` via `{manifest.get('provider')}`."
    ]
    if not cutoff_status.empty:
        strict = cutoff_status[cutoff_status["cutoff_date"].astype(str) == "2025-08-31"]
        codex = cutoff_status[cutoff_status["cutoff_date"].astype(str) == "2024-09-30"]
        if not strict.empty:
            row = strict.iloc[0]
            parts.append(f"Against the 2025-08-31 cutoff, `{int(row['post_cutoff_card_count'])}/{int(row['card_count'])}` cards are post-cutoff.")
        if not codex.empty:
            row = codex.iloc[0]
            parts.append(f"Against the 2024-09-30 cutoff, `{int(row['post_cutoff_card_count'])}/{int(row['card_count'])}` cards are post-cutoff.")
    if not recall_all.empty:
        fragments = []
        for _, row in recall_all.iterrows():
            fragments.append(f"{row['model']}: coverage {float(row['coverage']):.2%}")
        parts.append("Direct realized-value recall: " + "; ".join(fragments) + ".")
    else:
        parts.append("Direct realized-value recall was not run.")
    if not qualitative_all.empty:
        fragments = []
        for _, row in qualitative_all.iterrows():
            fragments.append(
                f"{row['model']}: card accuracy {float(row['card_mean_accuracy']):.2%} vs base {float(row['card_mean_base_rate']):.2%}"
            )
        parts.append("Qualitative path recall: " + "; ".join(fragments) + ".")
    else:
        parts.append("Qualitative path recall was not run.")
    return " ".join(parts)


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
