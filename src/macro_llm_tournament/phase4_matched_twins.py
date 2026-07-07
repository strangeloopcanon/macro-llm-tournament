from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, WORK_ROOT, markdown_table
from .demand_economy import (
    DemandEconomyClient,
    DemandScenario,
    build_fixture_demand_households,
    normalize_demand_households,
    run_demand_economy,
)
from .postcutoff_behavior_gate import (
    TARGET_SPECS,
    build_postcutoff_behavior_cards,
    load_proxy_data,
    score_proxy_forecasts,
)


PHASE4_VERSION = "phase4_prior_update_matched_twins_v1"
DEFAULT_SCENARIO = DemandScenario(
    "baseline",
    "No exogenous shock; households react only to endogenous feedback.",
)
COMPARISON_VARIABLES = (
    "aggregate_consumption",
    "output_gap_pct",
    "employment_rate",
    "inflation_rate",
    "policy_rate",
    "aggregate_job_loss_belief",
    "aggregate_confidence_index",
    "aggregate_liquid_buffer_months",
)


@dataclass(frozen=True)
class OutputMappingSpec:
    target_name: str
    series_id: str
    target_label: str
    target_units: str
    target_transform: str
    economy_variable: str
    economy_transform: str
    period_alignment: str
    lower: float
    upper: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 prior-update matched-twin economy gate.")
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=("fixture", "replay", "live"), default="fixture")
    parser.add_argument("--data-mode", choices=("fixture", "fred"), default="fixture")
    parser.add_argument("--refresh-fred", action="store_true")
    parser.add_argument("--cutoff-date", default="2025-12-01")
    parser.add_argument("--asof-start", default="2025-12-15")
    parser.add_argument("--asof-end", default="2026-04-15")
    parser.add_argument("--history-months", type=int, default=18)
    parser.add_argument("--scoreable-only", action="store_true")
    parser.add_argument("--household-source", choices=("fixture", "csv"), default="fixture")
    parser.add_argument("--household-csv", default=None)
    parser.add_argument("--household-count", type=int, default=24)
    parser.add_argument("--period-count", type=int, default=12)
    parser.add_argument("--feedback-mode", choices=("closed_loop", "none"), default="closed_loop")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"phase4_matched_twins_{timestamp_slug()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(args)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        validate_args(args)
        mapping = default_output_mapping()
        mapping_payload = normalized_mapping_payload(mapping)
        mapping_sha = mapping_sha256(mapping_payload)

        frames, data_status = load_proxy_data(
            data_mode=args.data_mode,
            refresh=args.refresh_fred,
            cutoff_date=args.cutoff_date,
            asof_end=args.asof_end,
        )
        cards, targets, context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date=args.cutoff_date,
            asof_start=args.asof_start,
            asof_end=args.asof_end,
            history_months=args.history_months,
            scoreable_only=args.scoreable_only,
        )
        if not cards:
            raise ValueError("No post-cutoff proxy cards were built.")

        households = load_households(args)
        period_count = max(int(args.period_count), len(cards) + 1, 2)
        cache_dir = output_dir / "fresh_phase4_matched_twins_cache" if args.fresh_cache else WORK_ROOT / "phase4_matched_twins_cache"
        scenarios = [DEFAULT_SCENARIO]
        twins = [
            DemandEconomyClient(args.provider, args.model, cache_dir, mode=args.mode, variant="llm_belief", max_live_calls=args.max_live_calls),
            DemandEconomyClient(args.provider, "adaptive", cache_dir, mode="fixture", variant="adaptive", max_live_calls=0),
        ]

        all_initial: list[pd.DataFrame] = []
        all_beliefs: list[pd.DataFrame] = []
        all_decisions: list[pd.DataFrame] = []
        all_periods: list[pd.DataFrame] = []
        all_accounting: list[pd.DataFrame] = []
        raw_records: list[dict[str, Any]] = []
        live_used = 0
        cache_hits = 0
        for client in twins:
            initial, beliefs, decisions, periods, accounting, _prompt_rows = run_demand_economy(
                households,
                scenarios,
                client,
                period_count=period_count,
                feedback_mode=args.feedback_mode,
            )
            all_initial.append(initial)
            all_beliefs.append(beliefs)
            all_decisions.append(decisions)
            all_periods.append(periods)
            all_accounting.append(accounting)
            raw_records.extend(client.raw_records)
            live_used += client.live_call_count
            cache_hits += client.cache_hit_count

        initial_frame = pd.concat(all_initial, ignore_index=True)
        beliefs_frame = pd.concat(all_beliefs, ignore_index=True)
        decisions_frame = pd.concat(all_decisions, ignore_index=True)
        periods_frame = pd.concat(all_periods, ignore_index=True)
        accounting_frame = pd.concat(all_accounting, ignore_index=True)
        path_comparison = build_path_comparison(periods_frame)
        proxy_forecasts = economy_proxy_forecasts(periods_frame, cards, mapping)
        proxy_scores, joined = score_proxy_forecasts(proxy_forecasts, targets)
        verdict = classify_phase4_fixture(periods_frame, accounting_frame, proxy_scores)

        write_outputs(
            output_dir,
            mapping_payload=mapping_payload,
            cards=cards_to_frame(cards),
            context=context,
            targets=targets,
            households=households,
            initial=initial_frame,
            beliefs=beliefs_frame,
            decisions=decisions_frame,
            periods=periods_frame,
            accounting=accounting_frame,
            comparison=path_comparison,
            forecasts=proxy_forecasts,
            joined=joined,
            scores=proxy_scores,
            raw_records=raw_records,
        )
        manifest.update(
            {
                "status": "ok",
                "verdict": verdict["verdict"],
                "claim_scope": verdict["claim_scope"],
                "passed": bool(verdict["passed"]),
                "mapping_sha256": mapping_sha,
                "mapping_spec_version": PHASE4_VERSION,
                "data_status": data_status,
                "card_count": int(len(cards)),
                "target_rows": int(targets.shape[0]),
                "scoreable_target_rows": int(targets["target_available"].sum()) if not targets.empty else 0,
                "period_count_effective": int(period_count),
                "household_count": int(households.shape[0]),
                "sources": sorted(periods_frame["source"].astype(str).unique()),
                "live_call_count": int(live_used),
                "cache_hit_count": int(cache_hits),
                "max_accounting_abs_residual": max_accounting_abs_residual(accounting_frame),
                "score_rows": int(proxy_scores.shape[0]),
                "winner_source": phase4_winner(proxy_scores),
                "outputs": output_filenames(),
            }
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "phase4_matched_twins_report.md").write_text(build_report(manifest, proxy_scores, path_comparison), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


def validate_args(args: argparse.Namespace) -> None:
    if args.mode != "fixture":
        raise ValueError("Phase 4 live/replay is blocked until real household prior-state inputs and a locked replay cache are supplied.")
    if args.max_live_calls != 0:
        raise ValueError("Phase 4 fixture mode must use --max-live-calls 0.")
    if args.household_source == "csv":
        if not args.household_csv:
            raise ValueError("--household-csv is required when --household-source csv")
        if not Path(args.household_csv).exists():
            raise ValueError(f"--household-csv does not exist: {args.household_csv}")


def load_households(args: argparse.Namespace) -> pd.DataFrame:
    if args.household_source == "csv":
        return normalize_demand_households(pd.read_csv(Path(args.household_csv)))
    return build_fixture_demand_households(args.household_count)


def default_output_mapping() -> list[OutputMappingSpec]:
    by_name = {spec.target_name: spec for spec in TARGET_SPECS}
    rows = [
        ("pce_mom_pct", "aggregate_consumption", "pct_change", "Nominal PCE proxy maps to aggregate consumption growth."),
        ("real_pce_mom_pct", "output", "pct_change", "Real PCE proxy maps to real output growth in the one-good economy."),
        ("retail_sales_mom_pct", "aggregate_consumption", "pct_change", "Retail proxy maps to aggregate consumption growth; this is deliberately locked before scoring."),
        ("personal_saving_rate_pct", "aggregate_saving", "saving_rate", "Saving-rate proxy maps to aggregate saving divided by aggregate income."),
        ("revolving_credit_mom_pct", "aggregate_debt", "pct_change", "Revolving-credit proxy maps to aggregate household debt growth."),
    ]
    mapping: list[OutputMappingSpec] = []
    for target_name, variable, transform, note in rows:
        target = by_name[target_name]
        mapping.append(
            OutputMappingSpec(
                target_name=target.target_name,
                series_id=target.series_id,
                target_label=target.label,
                target_units=target.units,
                target_transform=target.transform,
                economy_variable=variable,
                economy_transform=transform,
                period_alignment="card_i_scores_economy_period_i_plus_1_against_next_month_target",
                lower=target.lower,
                upper=target.upper,
                note=note,
            )
        )
    return mapping


def normalized_mapping_payload(mapping: Iterable[OutputMappingSpec]) -> dict[str, Any]:
    rows = [spec.__dict__ for spec in mapping]
    return {
        "schema_version": PHASE4_VERSION,
        "leakage_rule": "This mapping is written before forecasts are scored and keyed only by card_id, period_id, and target_name.",
        "rows": sorted(rows, key=lambda row: row["target_name"]),
    }


def mapping_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def economy_proxy_forecasts(
    periods: pd.DataFrame,
    cards: Iterable[Any],
    mapping: Iterable[OutputMappingSpec],
) -> pd.DataFrame:
    period_by_source = {source: group.sort_values("period_index").reset_index(drop=True) for source, group in periods.groupby("source", dropna=False)}
    rows: list[dict[str, Any]] = []
    cards_list = list(cards)
    for source, path in period_by_source.items():
        for card_index, card in enumerate(cards_list):
            target_period = card_index + 1
            if target_period >= path.shape[0]:
                continue
            for spec in mapping:
                prediction = mapped_period_value(path, target_period, spec)
                rows.append(
                    {
                        "card_id": card.card_id,
                        "period_id": card.period_id,
                        "target_name": spec.target_name,
                        "source": str(source),
                        "prediction": float(np.clip(prediction, spec.lower, spec.upper)),
                        "method": f"phase4_locked_{spec.economy_transform}_{spec.economy_variable}",
                    }
                )
    return pd.DataFrame(rows)


def mapped_period_value(path: pd.DataFrame, target_period: int, spec: OutputMappingSpec) -> float:
    current = path.iloc[target_period]
    previous = path.iloc[target_period - 1]
    if spec.economy_transform == "pct_change":
        base = float(previous[spec.economy_variable])
        if abs(base) <= 1e-9:
            return 0.0
        return 100.0 * (float(current[spec.economy_variable]) / base - 1.0)
    if spec.economy_transform == "saving_rate":
        income = float(current["aggregate_income"])
        if abs(income) <= 1e-9:
            return 0.0
        return 100.0 * float(current["aggregate_saving"]) / income
    raise ValueError(f"Unsupported economy transform: {spec.economy_transform}")


def build_path_comparison(periods: pd.DataFrame) -> pd.DataFrame:
    adaptive = periods[periods["variant"].astype(str).eq("adaptive")].copy()
    llm = periods[periods["variant"].astype(str).eq("llm_belief")].copy()
    rows: list[dict[str, Any]] = []
    for _, llm_row in llm.iterrows():
        match = adaptive[
            (adaptive["scenario_id"].astype(str) == str(llm_row["scenario_id"]))
            & (adaptive["period_index"].astype(int) == int(llm_row["period_index"]))
        ]
        if match.empty:
            continue
        adaptive_row = match.iloc[0]
        for variable in COMPARISON_VARIABLES:
            llm_value = float(llm_row[variable])
            adaptive_value = float(adaptive_row[variable])
            rows.append(
                {
                    "scenario_id": str(llm_row["scenario_id"]),
                    "period_id": str(llm_row["period_id"]),
                    "period_index": int(llm_row["period_index"]),
                    "llm_source": str(llm_row["source"]),
                    "adaptive_source": str(adaptive_row["source"]),
                    "variable": variable,
                    "llm_value": llm_value,
                    "adaptive_value": adaptive_value,
                    "delta": llm_value - adaptive_value,
                    "abs_delta": abs(llm_value - adaptive_value),
                }
            )
    return pd.DataFrame(rows)


def classify_phase4_fixture(periods: pd.DataFrame, accounting: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Any]:
    variants = set(periods["variant"].astype(str)) if not periods.empty else set()
    has_twins = {"llm_belief", "adaptive"}.issubset(variants)
    accounting_ok = max_accounting_abs_residual(accounting) <= ACCOUNTING_TOLERANCE
    has_scores = not scores.empty and "ALL" in set(scores["target_name"].astype(str))
    passed = bool(has_twins and accounting_ok and has_scores)
    return {
        "passed": passed,
        "verdict": "phase4_matched_twin_fixture_ready" if passed else "phase4_matched_twin_fixture_needs_work",
        "claim_scope": (
            "Fixture readiness only: the runner locks the proxy mapping and compares matched LLM-updater and adaptive twins, "
            "but does not establish empirical macro validity without real household state and live/replay belief updates."
        ),
    }


def phase4_winner(scores: pd.DataFrame) -> str | None:
    if scores.empty:
        return None
    overall = scores[scores["target_name"].astype(str).eq("ALL")].sort_values(["rmse_scaled", "source"])
    if overall.empty:
        return None
    return str(overall.iloc[0]["source"])


def max_accounting_abs_residual(accounting: pd.DataFrame) -> float:
    if accounting.empty or "abs_residual" not in accounting:
        return float("inf")
    return float(pd.to_numeric(accounting["abs_residual"], errors="coerce").fillna(float("inf")).max())


def cards_to_frame(cards: Iterable[Any]) -> pd.DataFrame:
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


def write_outputs(
    output_dir: Path,
    *,
    mapping_payload: dict[str, Any],
    cards: pd.DataFrame,
    context: pd.DataFrame,
    targets: pd.DataFrame,
    households: pd.DataFrame,
    initial: pd.DataFrame,
    beliefs: pd.DataFrame,
    decisions: pd.DataFrame,
    periods: pd.DataFrame,
    accounting: pd.DataFrame,
    comparison: pd.DataFrame,
    forecasts: pd.DataFrame,
    joined: pd.DataFrame,
    scores: pd.DataFrame,
    raw_records: list[dict[str, Any]],
) -> None:
    (output_dir / "phase4_output_mapping.json").write_text(json.dumps(mapping_payload, indent=2, sort_keys=True), encoding="utf-8")
    cards.to_csv(output_dir / "phase4_proxy_cards.csv", index=False)
    context.to_csv(output_dir / "phase4_proxy_context.csv", index=False)
    targets.to_csv(output_dir / "phase4_proxy_targets.csv", index=False)
    households.to_csv(output_dir / "phase4_households.csv", index=False)
    initial.to_csv(output_dir / "phase4_initial_state.csv", index=False)
    beliefs.to_csv(output_dir / "phase4_beliefs.csv", index=False)
    decisions.to_csv(output_dir / "phase4_household_decisions.csv", index=False)
    periods.to_csv(output_dir / "phase4_twin_periods.csv", index=False)
    accounting.to_csv(output_dir / "phase4_twin_accounting.csv", index=False)
    comparison.to_csv(output_dir / "phase4_twin_path_comparison.csv", index=False)
    forecasts.to_csv(output_dir / "phase4_proxy_forecasts.csv", index=False)
    joined.to_csv(output_dir / "phase4_proxy_joined_errors.csv", index=False)
    scores.to_csv(output_dir / "phase4_proxy_scores.csv", index=False)
    (output_dir / "phase4_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True), encoding="utf-8")


def build_report(manifest: dict[str, Any], scores: pd.DataFrame, comparison: pd.DataFrame) -> str:
    overall = scores[scores["target_name"].astype(str).eq("ALL")].sort_values(["rmse_scaled", "source"]) if not scores.empty else scores
    lines = [
        "# Phase 4 Prior-Update Matched Twins",
        "",
        "## Bottom Line",
        phase4_bottom_line(manifest),
        "",
        "## What This Tests",
        "This runner compares the same accounting-constrained demand economy under two belief-updating rules: an LLM belief-updater fixture and an adaptive-expectations twin. Real household heterogeneity and live prior-update caches are not used in fixture mode, so this is a readiness gate, not an empirical macro-validity result.",
        "",
        "## Locked Output Mapping",
        f"- Mapping SHA-256: `{manifest.get('mapping_sha256')}`",
        "- Mapping is written to `phase4_output_mapping.json` before proxy forecasts are scored.",
        "- Forecasts join to targets only on `card_id`, `period_id`, and `target_name`.",
        "",
        "## Proxy Scoreboard",
        markdown_table(overall),
        "",
        "## Path Differences",
        markdown_table(comparison.groupby("variable", dropna=False)["abs_delta"].mean().reset_index(name="mean_abs_delta") if not comparison.empty else comparison),
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def phase4_bottom_line(manifest: dict[str, Any]) -> str:
    verdict = manifest.get("verdict", "unknown")
    winner = manifest.get("winner_source")
    if manifest.get("passed"):
        return (
            f"`{verdict}`. The fixture runner locks the post-cutoff proxy mapping, runs matched LLM-updater and "
            f"adaptive twins from the same initial state, preserves accounting, and emits comparable proxy scores. "
            f"Best fixture source by scaled RMSE: `{winner}`. This is readiness, not empirical validity."
        )
    return f"`{verdict}`. The matched-twin fixture did not clear readiness checks."


def output_filenames() -> list[str]:
    return [
        "phase4_output_mapping.json",
        "phase4_proxy_cards.csv",
        "phase4_proxy_context.csv",
        "phase4_proxy_targets.csv",
        "phase4_households.csv",
        "phase4_initial_state.csv",
        "phase4_beliefs.csv",
        "phase4_household_decisions.csv",
        "phase4_twin_periods.csv",
        "phase4_twin_accounting.csv",
        "phase4_twin_path_comparison.csv",
        "phase4_proxy_forecasts.csv",
        "phase4_proxy_joined_errors.csv",
        "phase4_proxy_scores.csv",
        "phase4_raw_records.json",
        "phase4_matched_twins_report.md",
        "manifest.json",
    ]


def base_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": PHASE4_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "data_mode": args.data_mode,
        "cutoff_date": args.cutoff_date,
        "asof_start": args.asof_start,
        "asof_end": args.asof_end,
        "history_months": int(args.history_months),
        "scoreable_only": bool(args.scoreable_only),
        "household_source": args.household_source,
        "household_count_requested": int(args.household_count),
        "period_count_requested": int(args.period_count),
        "feedback_mode": args.feedback_mode,
        "fresh_cache": bool(args.fresh_cache),
        "max_live_calls": int(args.max_live_calls),
        "status": "running",
        "live_mode_blocked_until": "real household prior-state panel, replay cache contract, and empirical mapping approval are supplied",
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
