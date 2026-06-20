from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from .forecast_agent_panel import build_forecast_agent_panel
from .forecast_cards import (
    FORECAST_CARD_PROMPT_VERSION,
    assert_no_prompt_target_leakage,
    build_forecast_cards_from_rows,
    cards_to_frame,
    enrich_forecast_cards,
)
from .forecast_controls import build_control_forecasts
from .forecast_data import PROJECT_ROOT, SPF_VARIABLES, load_spf_error_data, parse_variable_list, quarter_index
from .forecast_llm import ForecastLLMClient, run_llm_forecasts
from .forecast_scoring import score_forecast_slices, score_forecasts, verdict_from_scores
from .fred_vintage import WORK_ROOT as FRED_VINTAGE_WORK_ROOT
from .fred_vintage import approximate_spf_as_of_date, build_vintage_context_for_cards
from .llm_common import LLMUnavailable
from .survey_beliefs import WORK_ROOT as SURVEY_BELIEF_WORK_ROOT
from .survey_beliefs import load_survey_belief_targets, survey_context_by_card


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LLM_CACHE_ROOT = PROJECT_ROOT / "work" / "llm_cache"
SPF_DETAIL_ROOT = PROJECT_ROOT / "work" / "spf_detail"
FRED_CURRENT_ROOT = PROJECT_ROOT / "work" / "fred_current"
POSTCUTOFF_DEFAULT = "2025-12-01"


@dataclass(frozen=True)
class DetailForecastSpec:
    variable: str
    relative_path: str
    step1_column: str
    realization_series: str
    realization_method: str
    units: str


DETAIL_FORECAST_SPECS: dict[str, DetailForecastSpec] = {
    "CPI": DetailForecastSpec(
        variable="CPI",
        relative_path="CPI/Mean_CPI_Level.xlsx",
        step1_column="CPI2",
        realization_series="CPIAUCSL",
        realization_method="quarterly_average_annualized_pct_change",
        units="annualized percentage points",
    ),
    "RGDP": DetailForecastSpec(
        variable="RGDP",
        relative_path="RGDP/Mean_RGDP_Growth.xlsx",
        step1_column="DRGDP2",
        realization_series="GDPC1",
        realization_method="quarterly_level_annualized_pct_change",
        units="annualized percentage points",
    ),
    "UNEMP": DetailForecastSpec(
        variable="UNEMP",
        relative_path="UNEMP/Mean_UNEMP_Level.xlsx",
        step1_column="UNEMP2",
        realization_series="UNRATE",
        realization_method="quarterly_average_level",
        units="percentage points",
    ),
    "TBILL": DetailForecastSpec(
        variable="TBILL",
        relative_path="TBILL/Mean_TBILL_Level.xlsx",
        step1_column="TBILL2",
        realization_series="TB3MS",
        realization_method="quarterly_average_level",
        units="percentage points",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-cutoff SPF forecasts with FRED proxy scoring where possible.")
    parser.add_argument("--provider", choices=["codex_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--llm-mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--variables", default="CPI,RGDP,UNEMP,TBILL")
    parser.add_argument("--cutoff-date", default=POSTCUTOFF_DEFAULT)
    parser.add_argument("--history-quarters", type=int, default=24)
    parser.add_argument("--scoreable-only", action="store_true")
    parser.add_argument("--replay-cache-miss-policy", choices=["fail", "freeze"], default="fail")
    parser.add_argument("--previous-run-dir", default=None)
    parser.add_argument("--vintage-context", choices=["off", "best_effort", "require"], default="require")
    parser.add_argument("--refresh-fred-vintage", action="store_true")
    parser.add_argument("--belief-targets", choices=["off", "best_effort", "require"], default="best_effort")
    parser.add_argument("--refresh-belief-targets", action="store_true")
    parser.add_argument("--typed-agent-panel", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.llm_mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --llm-mode live is used")

    variables = parse_variable_list(args.variables)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"spf_postcutoff_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "llm_mode": args.llm_mode,
        "max_live_calls": int(args.max_live_calls),
        "variables": variables,
        "horizons": [1],
        "cutoff_date": args.cutoff_date,
        "history_quarters": int(args.history_quarters),
        "card_prompt_version": FORECAST_CARD_PROMPT_VERSION,
        "vintage_context_mode": args.vintage_context,
        "belief_targets_mode": args.belief_targets,
        "typed_agent_panel": bool(args.typed_agent_panel),
        "scoreable_only": bool(args.scoreable_only),
        "replay_cache_miss_policy": args.replay_cache_miss_policy,
        "previous_run_dir": args.previous_run_dir,
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    try:
        spf_data, selected_rows, freeze_rows, data_status = build_postcutoff_selection(
            variables,
            cutoff_date=args.cutoff_date,
            scoreable_only=args.scoreable_only,
        )
        if selected_rows.empty:
            raise ValueError("No post-cutoff SPF detail rows available for the requested variables.")
        cards = build_forecast_cards_from_rows(
            spf_data,
            selected_rows,
            variables=variables,
            holdout_start_year=selected_rows["origin_year"].min(),
            holdout_end_year=selected_rows["origin_year"].max(),
            history_quarters=args.history_quarters,
        )
        if not cards:
            raise ValueError("No post-cutoff forecast cards built.")

        base_card_ids = [card.card_id for card in cards]
        vintage_contexts, vintage_rows, vintage_status = build_vintage_context_for_cards(
            cards,
            work_dir=FRED_VINTAGE_WORK_ROOT,
            refresh=args.refresh_fred_vintage,
            mode=args.vintage_context,
        )
        survey_targets, survey_status = load_survey_belief_targets(
            work_dir=SURVEY_BELIEF_WORK_ROOT,
            refresh=args.refresh_belief_targets,
            mode=args.belief_targets,
        )
        survey_contexts, survey_by_card = survey_context_by_card(cards, survey_targets)
        cards = enrich_forecast_cards(
            cards,
            vintage_context_by_card=vintage_contexts if args.vintage_context != "off" else None,
            survey_context_by_card=survey_contexts if args.belief_targets != "off" else None,
        )
        id_map = {old: new.card_id for old, new in zip(base_card_ids, cards)}
        if not vintage_rows.empty and "card_id" in vintage_rows:
            vintage_rows["card_id"] = vintage_rows["card_id"].map(id_map).fillna(vintage_rows["card_id"])
        if not survey_by_card.empty and "card_id" in survey_by_card:
            survey_by_card["card_id"] = survey_by_card["card_id"].map(id_map).fillna(survey_by_card["card_id"])
        assert_no_prompt_target_leakage(cards)

        candidate_cards = list(cards)
        client = ForecastLLMClient(
            args.provider,
            args.model,
            LLM_CACHE_ROOT,
            mode=args.llm_mode,
            max_live_calls=args.max_live_calls,
        )
        replay_cache_miss_rows = pd.DataFrame()
        if args.llm_mode == "replay" and args.replay_cache_miss_policy == "freeze":
            cards, replay_cache_miss_cards = split_cards_by_replay_cache(candidate_cards, client)
            replay_cache_miss_rows = cards_to_frame(replay_cache_miss_cards)
            if not replay_cache_miss_rows.empty:
                replay_cache_miss_rows["freeze_reason"] = "replay_cache_miss"
            if not cards:
                raise ValueError("Replay freeze policy found no cached post-cutoff cards to forecast.")

        candidate_cards_frame = cards_to_frame(candidate_cards)
        cards_frame = cards_to_frame(cards)
        candidate_cards_frame.to_csv(output_dir / "postcutoff_candidate_cards.csv", index=False)
        cards_frame.to_csv(output_dir / "postcutoff_forecast_cards.csv", index=False)
        replay_cache_miss_rows.to_csv(output_dir / "postcutoff_replay_cache_freeze_cards.csv", index=False)
        selected_rows.to_csv(output_dir / "postcutoff_selected_rows.csv", index=False)
        freeze_rows.to_csv(output_dir / "postcutoff_freeze_rows.csv", index=False)
        spf_data.to_csv(output_dir / "postcutoff_spf_input_rows.csv", index=False)
        vintage_rows.to_csv(output_dir / "fred_vintage_context.csv", index=False)
        (output_dir / "fred_vintage_status.json").write_text(json.dumps(vintage_status, indent=2, sort_keys=True))
        survey_targets.to_csv(output_dir / "survey_belief_targets.csv", index=False)
        survey_by_card.to_csv(output_dir / "survey_belief_targets_by_card.csv", index=False)
        (output_dir / "survey_belief_status.json").write_text(json.dumps(survey_status, indent=2, sort_keys=True))

        control_forecasts = build_control_forecasts(spf_data, cards, tune_end_year=2024)
        control_forecasts.to_csv(output_dir / "control_forecasts.csv", index=False)

        llm_forecasts, raw_records = run_llm_forecasts(client, cards)
        try:
            recall_probe = client.recall_probe()
        except LLMUnavailable as exc:
            if args.llm_mode == "replay" and args.replay_cache_miss_policy == "freeze":
                recall_probe = {"status": "skipped_replay_cache_miss", "error": str(exc)}
            else:
                raise
        llm_forecasts.to_json(output_dir / "llm_forecasts.jsonl", orient="records", lines=True)
        (output_dir / "llm_raw_records.json").write_text(json.dumps(raw_records, indent=2, sort_keys=True))
        (output_dir / "recall_probe.json").write_text(json.dumps(recall_probe, indent=2, sort_keys=True))

        all_forecasts = pd.concat([control_forecasts, llm_forecasts], ignore_index=True)
        all_forecasts.to_csv(output_dir / "all_forecasts.csv", index=False)
        agent_panel_rows = pd.DataFrame()
        agent_aggregate_rows = pd.DataFrame()
        if args.typed_agent_panel:
            agent_panel_rows, agent_aggregate_rows = build_forecast_agent_panel(cards, all_forecasts)
            agent_panel_rows.to_csv(output_dir / "forecast_agent_panel.csv", index=False)
            agent_aggregate_rows.to_csv(output_dir / "forecast_agent_aggregates.csv", index=False)

        scored_cards = cards_frame[cards_frame["target_realized"].map(is_finite)].copy()
        scored_ids = set(scored_cards["card_id"].astype(str))
        scored_forecasts = all_forecasts[all_forecasts["card_id"].astype(str).isin(scored_ids)].copy()
        if scored_cards.empty:
            scores = pd.DataFrame()
            behavior = pd.DataFrame()
            joined = pd.DataFrame()
            slice_scores = pd.DataFrame()
            verdict = {"status": "no_scoreable_cards"}
        else:
            scores, behavior, joined = score_forecasts(scored_cards, scored_forecasts)
            slice_scores = score_forecast_slices(scored_cards, joined)
            verdict = verdict_from_scores(scores)
        scores.to_csv(output_dir / "forecast_scores.csv", index=False)
        behavior.to_csv(output_dir / "behavioral_coefficients.csv", index=False)
        joined.to_csv(output_dir / "forecast_joined_errors.csv", index=False)
        slice_scores.to_csv(output_dir / "forecast_slice_scores.csv", index=False)

        manifest.update(
            {
                "status": "ok",
                "card_count": int(len(cards)),
                "candidate_card_count": int(len(candidate_cards)),
                "forecasted_card_count": int(len(cards)),
                "scoreable_card_count": int(scored_cards.shape[0]),
                "frozen_unscored_card_count": int(cards_frame.shape[0] - scored_cards.shape[0] + replay_cache_miss_rows.shape[0]),
                "replayed_card_count": int(len(cards)) if args.llm_mode == "replay" else 0,
                "uncached_frozen_card_count": int(replay_cache_miss_rows.shape[0]),
                "newly_scoreable_card_count": int(count_newly_scoreable_rows(selected_rows, args.previous_run_dir)),
                "card_regime_counts": cards_frame["regime_label"].value_counts().sort_index().to_dict(),
                "card_contamination_counts": cards_frame["contamination_label"].value_counts().sort_index().to_dict(),
                "vintage_context_status": vintage_status,
                "survey_belief_status": survey_status,
                "survey_belief_target_rows": int(survey_targets.shape[0]),
                "survey_belief_card_rows": int(survey_by_card.shape[0]),
                "typed_agent_panel_rows": int(agent_panel_rows.shape[0]),
                "typed_agent_aggregate_rows": int(agent_aggregate_rows.shape[0]),
                "forecast_rows": int(all_forecasts.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "verdict": verdict,
                "data_status": data_status,
                "replay_cache_miss_freeze_rows": int(replay_cache_miss_rows.shape[0]),
            }
        )
        report = build_postcutoff_report(manifest, scores=scores, slice_scores=slice_scores)
        (output_dir / "postcutoff_tournament_report.md").write_text(report, encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except (Exception, LLMUnavailable) as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


def build_postcutoff_selection(
    variables: Iterable[str],
    *,
    cutoff_date: str,
    scoreable_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variables = parse_variable_list(variables)
    official = load_spf_error_data(variables)
    official_h1 = official[official["horizon"] == 1].copy()
    detail = load_detail_mean_forecasts(variables)
    realized = build_fred_realization_frame(variables)
    detail = detail.merge(realized, on=["variable", "origin"], how="left")
    detail["realized"] = detail["fred_realized"]
    detail["official_iar_forecast"] = np.nan
    detail["official_no_change_forecast"] = np.nan
    detail["official_dar_forecast"] = np.nan
    detail["official_darm_forecast"] = np.nan
    detail["source_url"] = detail["source_file"].map(lambda value: f"local:{value}")
    detail["variable_page_url"] = detail["variable"].map(lambda variable: SPF_VARIABLES[variable].page_url)

    official_keys = set(
        tuple(row)
        for row in official_h1[official_h1["spf_forecast"].map(is_finite)][["variable", "origin_index", "horizon"]].to_numpy()
    )
    detail_extra = detail[
        ~detail[["variable", "origin_index", "horizon"]].apply(lambda row: tuple(row) in official_keys, axis=1)
    ].copy()
    spf_data = combine_official_and_detail_rows(official_h1, detail_extra[official_h1.columns])

    cutoff = date.fromisoformat(cutoff_date)
    selected = detail[detail["as_of_date"].map(lambda value: date.fromisoformat(value) > cutoff)].copy()
    selected["scoreable"] = selected["realization_complete"].fillna(False).astype(bool) & selected["realized"].map(is_finite)
    freeze_rows = selected[~selected["scoreable"]].copy()
    if scoreable_only:
        selected = selected[selected["scoreable"]].copy()
    selected = selected.sort_values(["origin_index", "variable"]).reset_index(drop=True)
    status = {
        "source": "Philadelphia Fed SPF detail mean workbooks plus FRED current graph CSV proxy realizations",
        "cutoff_date": cutoff_date,
        "postcutoff_rows": int(detail[detail["as_of_date"].map(lambda value: date.fromisoformat(value) > cutoff)].shape[0]),
        "selected_rows": int(selected.shape[0]),
        "scoreable_rows": int(selected["scoreable"].sum()) if not selected.empty else 0,
        "freeze_rows": int(freeze_rows.shape[0]),
        "realization_note": "FRED proxy realizations are not official Philadelphia Fed SPF error-stat rows.",
    }
    return spf_data, selected, freeze_rows, status


def split_cards_by_replay_cache(
    cards: Iterable[Any],
    client: ForecastLLMClient,
) -> tuple[list[Any], list[Any]]:
    cached: list[Any] = []
    missing: list[Any] = []
    for card in cards:
        path = client.cache_path(client.forecast_cache_name(card))
        if path.exists():
            cached.append(card)
        else:
            missing.append(card)
    return cached, missing


def count_newly_scoreable_rows(selected_rows: pd.DataFrame, previous_run_dir: str | None) -> int:
    if not previous_run_dir:
        return 0
    previous_path = Path(previous_run_dir) / "postcutoff_freeze_rows.csv"
    if not previous_path.exists() or selected_rows.empty:
        return 0
    previous = pd.read_csv(previous_path)
    if previous.empty:
        return 0
    key_columns = ["variable", "origin_index", "horizon"]
    if any(column not in previous for column in key_columns) or any(column not in selected_rows for column in key_columns):
        return 0
    if "scoreable" not in selected_rows:
        return 0
    scoreable = selected_rows[selected_rows["scoreable"].fillna(False).astype(bool)].copy()
    previous_keys = set(tuple(row) for row in previous[key_columns].to_numpy())
    return int(sum(tuple(row) in previous_keys for row in scoreable[key_columns].to_numpy()))


def combine_official_and_detail_rows(official_h1: pd.DataFrame, detail_extra: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([official_h1, detail_extra], ignore_index=True)
    combined["_has_spf_forecast"] = combined["spf_forecast"].map(is_finite)
    combined["_has_realized"] = combined["realized"].map(is_finite)
    combined = combined.sort_values(
        ["variable", "horizon", "origin_index", "_has_spf_forecast", "_has_realized"],
        ascending=[True, True, True, False, False],
    )
    combined = combined.drop_duplicates(["variable", "horizon", "origin_index"], keep="first")
    combined = combined.drop(columns=["_has_spf_forecast", "_has_realized"])
    return combined.sort_values(["variable", "horizon", "origin_index"]).reset_index(drop=True)


def load_detail_mean_forecasts(variables: Iterable[str], *, work_dir: Path = SPF_DETAIL_ROOT) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in parse_variable_list(variables):
        spec = DETAIL_FORECAST_SPECS[variable]
        path = work_dir / spec.relative_path
        frame = read_simple_xlsx(path)
        for _, row in frame.iterrows():
            year = to_int(row.get("YEAR"))
            quarter = to_int(row.get("QUARTER"))
            forecast = to_float(row.get(spec.step1_column))
            if year is None or quarter is None or not is_finite(forecast):
                continue
            origin = f"{year}:Q{quarter}"
            rows.append(
                {
                    "variable": variable,
                    "variable_name": SPF_VARIABLES[variable].name,
                    "units": spec.units,
                    "origin": origin,
                    "origin_year": year,
                    "origin_quarter": quarter,
                    "origin_index": quarter_index(origin),
                    "horizon": 1,
                    "spf_forecast": float(forecast),
                    "detail_step1_column": spec.step1_column,
                    "as_of_date": approximate_spf_as_of_date(origin),
                    "source_file": str(path),
                    "source_file_name": path.name,
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "origin_index"]).reset_index(drop=True)


def read_simple_xlsx(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[Any]] = []
        max_col = 0
        for row in root.findall(".//a:sheetData/a:row", ns):
            values: dict[int, Any] = {}
            for cell in row.findall("a:c", ns):
                idx = excel_column_index(cell.attrib.get("r", ""))
                max_col = max(max_col, idx)
                raw_value = cell.findtext("a:v", namespaces=ns)
                if raw_value is None:
                    value = None
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(raw_value)]
                else:
                    value = to_float(raw_value)
                values[idx] = value
            rows.append([values.get(idx) for idx in range(max_col + 1)])
    if not rows:
        return pd.DataFrame()
    header = [str(value) if value is not None else "" for value in rows[0]]
    records = []
    for row in rows[1:]:
        records.append({header[idx]: row[idx] if idx < len(row) else None for idx in range(len(header))})
    return pd.DataFrame(records)


def excel_column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


def build_fred_realization_frame(variables: Iterable[str], *, fred_dir: Path = FRED_CURRENT_ROOT) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in parse_variable_list(variables):
        spec = DETAIL_FORECAST_SPECS[variable]
        series = read_fred_graph_csv(fred_dir / f"{spec.realization_series}.csv")
        realized_rows = realize_variable(variable, spec, series)
        rows.extend(realized_rows)
    return pd.DataFrame(rows)


def read_fred_graph_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame = frame.rename(columns={date_col: "date", value_col: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


def realize_variable(variable: str, spec: DetailForecastSpec, series: pd.DataFrame) -> list[dict[str, Any]]:
    frame = series.copy()
    frame["quarter"] = frame["date"].dt.to_period("Q")
    rows: list[dict[str, Any]] = []
    if spec.realization_method == "quarterly_level_annualized_pct_change":
        quarterly = frame.groupby("quarter")["value"].last().sort_index()
        for quarter, value in quarterly.items():
            prior = quarter - 1
            realized = annualized_pct_change(value, quarterly.get(prior, np.nan))
            rows.append(realization_row(variable, quarter, realized, True, spec))
    elif spec.realization_method == "quarterly_average_annualized_pct_change":
        counts = frame.groupby("quarter")["value"].count()
        quarterly = frame.groupby("quarter")["value"].mean().sort_index()
        for quarter, value in quarterly.items():
            prior = quarter - 1
            complete = int(counts.get(quarter, 0)) >= 3 and int(counts.get(prior, 0)) >= 3
            realized = annualized_pct_change(value, quarterly.get(prior, np.nan)) if complete else np.nan
            rows.append(realization_row(variable, quarter, realized, complete, spec))
    elif spec.realization_method == "quarterly_average_level":
        counts = frame.groupby("quarter")["value"].count()
        quarterly = frame.groupby("quarter")["value"].mean().sort_index()
        for quarter, value in quarterly.items():
            complete = int(counts.get(quarter, 0)) >= 3
            rows.append(realization_row(variable, quarter, float(value) if complete else np.nan, complete, spec))
    else:
        raise ValueError(f"Unknown realization method: {spec.realization_method}")
    return rows


def realization_row(
    variable: str,
    quarter: pd.Period,
    realized: float,
    complete: bool,
    spec: DetailForecastSpec,
) -> dict[str, Any]:
    origin = f"{quarter.year}:Q{quarter.quarter}"
    return {
        "variable": variable,
        "origin": origin,
        "fred_realized": float(realized) if is_finite(realized) else np.nan,
        "realization_complete": bool(complete and is_finite(realized)),
        "realization_source": spec.realization_series,
        "realization_method": spec.realization_method,
    }


def annualized_pct_change(value: float, prior: float) -> float:
    if not is_finite(value) or not is_finite(prior) or float(prior) == 0.0:
        return np.nan
    return float((float(value) / float(prior)) ** 4 * 100.0 - 100.0)


def build_postcutoff_report(manifest: dict[str, Any], *, scores: pd.DataFrame, slice_scores: pd.DataFrame) -> str:
    verdict = manifest.get("verdict", {})
    lines = [
        "# Post-Cutoff SPF Forecast Tournament",
        "",
        "## Bottom Line",
        postcutoff_verdict_sentence(verdict),
        "",
        "## Run Setup",
        f"- Provider/model: `{manifest.get('provider')}` / `{manifest.get('model')}`",
        f"- LLM mode: `{manifest.get('llm_mode')}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Cache hits: `{manifest.get('cache_hit_count')}`",
        f"- Forecast cards: `{manifest.get('card_count')}`",
        f"- Candidate cards before replay freeze: `{manifest.get('candidate_card_count', manifest.get('card_count'))}`",
        f"- Scoreable cards now: `{manifest.get('scoreable_card_count')}`",
        f"- Frozen unscored cards: `{manifest.get('frozen_unscored_card_count')}`",
        f"- Uncached replay-frozen cards: `{manifest.get('uncached_frozen_card_count', 0)}`",
        f"- Newly scoreable from previous run: `{manifest.get('newly_scoreable_card_count', 0)}`",
        f"- Cutoff date: `{manifest.get('cutoff_date')}`",
        f"- Variables: `{', '.join(manifest.get('variables', []))}`",
        f"- Vintage macro context: `{status_label(manifest.get('vintage_context_status'))}`",
        f"- Survey belief targets: `{status_label(manifest.get('survey_belief_status'))}`",
        f"- Contamination labels: `{json.dumps(manifest.get('card_contamination_counts', {}), sort_keys=True)}`",
        "",
        "## Scoreable Leaderboard",
        markdown_table(scores[scores["variable"] == "ALL"].sort_values("rmse").head(12))
        if not scores.empty
        else "_No scoreable rows yet._",
        "",
        "## Variable Breakdown",
        variable_breakdown(scores),
        "",
        "## LLM Regime Slices",
        regime_breakdown(slice_scores),
        "",
        "## Caveats",
        "- These cards are post-cutoff relative to the supplied GPT-5.5 cutoff date.",
        "- Current scoreable rows use FRED current-vintage proxy realizations, not official SPF forecast-error rows.",
        "- Frozen rows have forecasts saved now and should be rescored when official SPF/FRED realizations become complete.",
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def postcutoff_verdict_sentence(verdict: dict[str, Any]) -> str:
    if verdict.get("status") != "ok":
        return f"Verdict unavailable: `{verdict.get('status')}`."
    llm = verdict.get("llm_source")
    if verdict.get("beats_all_primary_behavioral_controls") and verdict.get("beats_spf_consensus"):
        return (
            f"`{llm}` beats SPF consensus and the primary behavioral controls on the currently scoreable "
            f"post-cutoff proxy rows, with RMSE `{verdict.get('llm_rmse'):.4f}`."
        )
    if verdict.get("beats_all_primary_behavioral_controls"):
        return (
            f"`{llm}` beats the primary behavioral controls on currently scoreable post-cutoff proxy rows, "
            f"but does not beat SPF consensus."
        )
    return f"`{llm}` does not clear the primary behavioral-control gate on currently scoreable post-cutoff proxy rows."


def variable_breakdown(scores: pd.DataFrame) -> str:
    if scores.empty:
        return "_No scoreable rows yet._"
    rows = []
    llm_rows = scores[scores["source"].astype(str).str.startswith("llm_")]
    if llm_rows.empty:
        return "_No LLM score rows._"
    llm_source = llm_rows.iloc[0]["source"]
    for variable in sorted(value for value in scores["variable"].dropna().unique() if value != "ALL"):
        var_scores = scores[scores["variable"] == variable]
        llm = var_scores[var_scores["source"] == llm_source]
        spf = var_scores[var_scores["source"] == "spf_consensus"]
        if llm.empty:
            continue
        rows.append(
            {
                "variable": variable,
                "llm_rmse": float(llm.iloc[0]["rmse"]),
                "spf_consensus_rmse": float(spf.iloc[0]["rmse"]) if not spf.empty else np.nan,
                "n": int(llm.iloc[0]["n"]),
                "winner": "LLM" if not spf.empty and float(llm.iloc[0]["rmse"]) < float(spf.iloc[0]["rmse"]) else "SPF",
            }
        )
    return markdown_table(pd.DataFrame(rows))


def regime_breakdown(slice_scores: pd.DataFrame) -> str:
    if slice_scores.empty:
        return "_No scoreable regime rows yet._"
    llm = slice_scores[
        (slice_scores["slice"] == "regime") & (slice_scores["source"].astype(str).str.startswith("llm_"))
    ]
    if llm.empty:
        return "_No LLM regime rows._"
    return markdown_table(llm[["regime_label", "n", "rmse", "mae", "bias", "direction_accuracy"]])


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    clean = clean.fillna("").astype(str)
    lines = [
        "| " + " | ".join(clean.columns) + " |",
        "| " + " | ".join(["---"] * len(clean.columns)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in clean.columns) + " |")
    return "\n".join(lines)


def status_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "unknown"))
    return str(value or "unknown")


def is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def to_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if math.isfinite(numeric) else np.nan


def to_int(value: Any) -> int | None:
    numeric = to_float(value)
    if not is_finite(numeric):
        return None
    return int(numeric)


if __name__ == "__main__":
    raise SystemExit(main())
