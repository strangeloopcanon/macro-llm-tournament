"""Origin-safe macro state and first-release target construction for JEPA runs."""

from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_SPECS: dict[str, tuple[str, str]] = {
    "consumption_growth_pct": ("PCE", "pct_change"),
    "real_consumption_growth_pct": ("PCEC96", "pct_change"),
    "price_growth_pct": ("PCEPI", "pct_change"),
    "real_disposable_income_growth_pct": ("DSPIC96", "pct_change"),
    "payroll_growth_pct": ("PAYEMS", "pct_change"),
    "unemployment_rate_level": ("UNRATE", "level"),
    "personal_saving_rate_change_pp": ("PSAVERT", "diff"),
    "revolving_credit_growth_pct": ("REVOLSL", "pct_change"),
    "retail_sales_growth_pct": ("RSAFS", "pct_change"),
}

STATE_SERIES = (
    "PCE",
    "PCEC96",
    "PCEPI",
    "DSPIC96",
    "PAYEMS",
    "UNRATE",
    "PSAVERT",
    "REVOLSL",
    "RSAFS",
    "CPIAUCSL",
    "CPILFESL",
    "INDPRO",
    "UMCSENT",
    "FEDFUNDS",
    "M2SL",
    "TOTALSL",
    "BUSLOANS",
    "ICSA",
    "CCSA",
    "CES0500000003",
    "AWHMAN",
    "HOUST",
    "PERMIT",
)

# These series are expressed as level changes in the origin cards. Every other
# state-series change is a percent change, including the UMCSENT index.
DIFFERENCE_SERIES = {"UNRATE", "PSAVERT", "FEDFUNDS"}

ACTION_COLUMNS = (
    "policy_rate_change_1m",
    "money_growth_1m",
    "consumer_credit_growth_1m",
    "business_credit_growth_1m",
    "sentiment_change_1m",
    "claims_growth_1m",
    "release_event_count",
    "revision_event_count",
    "mean_absolute_revision_pct",
)

EVENT_ACTION_COLUMNS = (
    "event_initial_count_log1p",
    "event_revision_count_log1p",
    "event_distinct_series_count_log1p",
    "event_mean_absolute_revision_pct_log1p",
    "event_max_absolute_revision_pct_log1p",
    *(f"event_count_log1p__{series_id}" for series_id in STATE_SERIES),
    *(f"event_update__{series_id}" for series_id in STATE_SERIES),
)

EVENT_ACTION_CATEGORICAL_COLUMNS = (
    "event_kind_group",
    "dominant_series",
)

ORIGIN_BASELINE_FIELDS: dict[str, str] = {
    "consumption_growth_pct": "series__PCE__change_1m",
    "real_consumption_growth_pct": "series__PCEC96__change_1m",
    "price_growth_pct": "series__PCEPI__change_1m",
    "real_disposable_income_growth_pct": "series__DSPIC96__change_1m",
    "payroll_growth_pct": "series__PAYEMS__change_1m",
    "unemployment_rate_level": "series__UNRATE__level",
    "personal_saving_rate_change_pp": "series__PSAVERT__change_1m",
    "revolving_credit_growth_pct": "series__REVOLSL__change_1m",
    "retail_sales_growth_pct": "series__RSAFS__change_1m",
}

# FRED changed these two historical credit series from billions to millions
# without a unit field in the archived event records. Keep one canonical level
# scale so raw-state models never see an artificial 1,000x transition.
SERIES_VALUE_NORMALIZATION: dict[str, dict[str, float | str]] = {
    "REVOLSL": {
        "canonical_unit": "billions_of_dollars",
        "million_value_threshold": 100_000.0,
        "million_to_billion_scale": 0.001,
    },
    "TOTALSL": {
        "canonical_unit": "billions_of_dollars",
        "million_value_threshold": 100_000.0,
        "million_to_billion_scale": 0.001,
    },
}


@dataclass(frozen=True)
class MacroEvent:
    series_id: str
    observation_date: str
    vintage_date: str
    event_kind: str
    value: float
    event_id: str


def load_macro_events(
    path: Path | str, *, series_ids: Iterable[str] | None = None
) -> list[MacroEvent]:
    """Load canonical ALFRED events, retaining only selected finite rows."""

    selected = None if series_ids is None else set(series_ids)
    events: list[MacroEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            series_id = str(row.get("series_id", ""))
            if selected is not None and series_id not in selected:
                continue
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid macro event value at line {line_number}"
                ) from exc
            if not math.isfinite(value):
                continue
            event = MacroEvent(
                series_id=series_id,
                observation_date=_date_text(row.get("observation_date")),
                vintage_date=_date_text(
                    row.get("vintage_date") or row.get("realtime_start")
                ),
                event_kind=str(row.get("event_kind", "")),
                value=_canonical_series_value(series_id, value),
                event_id=str(row.get("event_id", f"line:{line_number}")),
            )
            if event.event_kind not in {"initial_release", "revision"}:
                raise ValueError(f"unsupported macro event kind: {event.event_kind}")
            events.append(event)
    if not events:
        raise ValueError(f"no usable macro events in {path}")
    canonical: dict[tuple[str, str, str], MacroEvent] = {}
    for event in events:
        key = (event.series_id, event.observation_date, event.vintage_date)
        prior = canonical.get(key)
        if prior is None:
            canonical[key] = event
            continue
        if not math.isclose(event.value, prior.value, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"conflicting same-vintage macro events: {key}")
        if event.event_kind == "initial_release":
            canonical[key] = event

    ordered_canonical = sorted(
        canonical.values(),
        key=lambda row: (
            row.vintage_date,
            row.series_id,
            row.observation_date,
            row.event_kind == "revision",
            row.event_id,
        ),
    )
    histories: dict[tuple[str, str], list[MacroEvent]] = defaultdict(list)
    for event in ordered_canonical:
        histories[(event.series_id, event.observation_date)].append(event)
    changed_events: list[MacroEvent] = []
    for history in histories.values():
        prior_value: float | None = None
        for event in history:
            if prior_value is not None and math.isclose(
                event.value, prior_value, rel_tol=0.0, abs_tol=0.0
            ):
                continue
            changed_events.append(
                MacroEvent(
                    series_id=event.series_id,
                    observation_date=event.observation_date,
                    vintage_date=event.vintage_date,
                    event_kind=(
                        "initial_release" if prior_value is None else "revision"
                    ),
                    value=event.value,
                    event_id=event.event_id,
                )
            )
            prior_value = event.value
    return sorted(
        changed_events,
        key=lambda row: (
            row.vintage_date,
            row.series_id,
            row.observation_date,
            row.event_kind == "revision",
            row.event_id,
        ),
    )


def _canonical_series_value(series_id: str, value: float) -> float:
    rule = SERIES_VALUE_NORMALIZATION.get(series_id)
    if rule is None:
        return value
    threshold = float(rule["million_value_threshold"])
    scale = float(rule["million_to_billion_scale"])
    return value * scale if abs(value) >= threshold else value


def build_event_state_panel(
    events: Sequence[MacroEvent],
    *,
    anchors: pd.DataFrame,
    start_date: str,
    end_date: str,
    state_series: Sequence[str] = STATE_SERIES,
) -> pd.DataFrame:
    """Build one causal state per release day plus exact monthly forecast anchors."""

    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError("event panel start_date must not exceed end_date")
    required_anchor_columns = {"origin_month", "as_of_date"}
    missing = required_anchor_columns - set(anchors.columns)
    if missing:
        raise ValueError(f"anchor panel missing columns: {sorted(missing)}")
    if anchors["as_of_date"].astype(str).duplicated().any():
        raise ValueError("anchor panel contains duplicate as-of dates")

    selected = set(state_series)
    ordered = [
        event
        for event in events
        if event.series_id in selected and event.vintage_date <= end
    ]
    grouped_events: dict[str, list[MacroEvent]] = defaultdict(list)
    for event in ordered:
        grouped_events[event.vintage_date].append(event)
    anchor_by_date = {
        str(row.as_of_date): row._asdict()
        for row in anchors.itertuples(index=False)
    }
    timeline_dates = sorted(
        date_text
        for date_text in set(grouped_events) | set(anchor_by_date)
        if date_text <= end
    )
    if not timeline_dates:
        raise ValueError("event panel contains no rows in the requested interval")

    latest: dict[tuple[str, str], MacroEvent] = {}
    observations_by_series: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for as_of in timeline_dates:
        day_events = grouped_events.get(as_of, [])
        before = {
            series_id: _latest_series_event(
                series_id, latest, observations_by_series
            )
            for series_id in state_series
        }
        event_row = None
        if day_events and as_of >= start:
            event_row = _event_base_row(
                as_of=as_of,
                latest=latest,
                observations_by_series=observations_by_series,
                state_series=state_series,
            )
        revision_pct: list[float] = []
        series_counts: dict[str, int] = defaultdict(int)
        for event in day_events:
            prior = latest.get((event.series_id, event.observation_date))
            if event.event_kind == "revision" and prior is not None and prior.value != 0:
                revision_pct.append(abs(100.0 * (event.value / prior.value - 1.0)))
            latest[(event.series_id, event.observation_date)] = event
            observations_by_series[event.series_id].add(event.observation_date)
            series_counts[event.series_id] += 1

        if event_row is not None:
            row = event_row
            initial_count = sum(
                event.event_kind == "initial_release" for event in day_events
            )
            revision_count = len(day_events) - initial_count
            row.update(
                {
                    "row_kind": "release_day",
                    "origin_month": "",
                    "is_forecast_anchor": 0.0,
                    "action_name": "macro_release_day",
                    "event_kind_group": (
                        "mixed"
                        if initial_count and revision_count
                        else "initial_release"
                        if initial_count
                        else "revision"
                    ),
                    "dominant_series": min(
                        series_counts,
                        key=lambda series_id: (-series_counts[series_id], series_id),
                    ),
                    "source_event_count": len(day_events),
                    "source_event_ids_sha256": hashlib.sha256(
                        "\n".join(sorted(event.event_id for event in day_events)).encode()
                    ).hexdigest(),
                    "event_initial_count_log1p": math.log1p(initial_count),
                    "event_revision_count_log1p": math.log1p(revision_count),
                    "event_distinct_series_count_log1p": math.log1p(
                        len(series_counts)
                    ),
                    "event_mean_absolute_revision_pct_log1p": math.log1p(
                        float(np.mean(revision_pct)) if revision_pct else 0.0
                    ),
                    "event_max_absolute_revision_pct_log1p": math.log1p(
                        max(revision_pct) if revision_pct else 0.0
                    ),
                }
            )
            for series_id in state_series:
                after = _latest_series_event(
                    series_id, latest, observations_by_series
                )
                row[f"event_count_log1p__{series_id}"] = math.log1p(
                    series_counts.get(series_id, 0)
                )
                row[f"event_update__{series_id}"] = _event_level_change(
                    series_id, before.get(series_id), after
                )
            rows.append(row)

        anchor = anchor_by_date.get(as_of)
        if anchor is not None and as_of >= start:
            row = {
                "as_of_date": as_of,
                "row_kind": "monthly_forecast_anchor",
                "origin_month": str(anchor["origin_month"]),
                "is_forecast_anchor": 1.0,
                "action_name": "monthly_forecast_anchor",
                "event_kind_group": "anchor",
                "dominant_series": "anchor",
                "source_event_count": 0,
                "source_event_ids_sha256": hashlib.sha256(b"").hexdigest(),
            }
            for column in state_feature_columns(state_series):
                row[column] = anchor.get(column, np.nan)
            for column in EVENT_ACTION_COLUMNS:
                row[column] = 0.0
            rows.append(row)

    result = pd.DataFrame(rows)
    result["timestamp"] = result.apply(
        lambda row: (
            f"{row['as_of_date']}T23:59:59Z"
            if row["row_kind"] == "monthly_forecast_anchor"
            else f"{row['as_of_date']}T12:00:00Z"
        ),
        axis=1,
    )
    result = result.sort_values(["timestamp", "row_kind"]).reset_index(drop=True)
    timestamps = pd.to_datetime(result["timestamp"], utc=True)
    result["episode_id"] = "us_macro_release_stream"
    result["step_idx"] = np.arange(len(result), dtype=int)
    result["delta_t_s"] = timestamps.diff().dt.total_seconds().fillna(0.0)
    result["log_delta_t_s"] = np.log1p(result["delta_t_s"].clip(lower=0.0))
    result["done"] = False
    result.loc[result.index[-1], "done"] = True
    result["split"] = timestamps.map(_split_for_date)
    result["month_sin"] = np.sin(2.0 * np.pi * timestamps.dt.month / 12.0)
    result["month_cos"] = np.cos(2.0 * np.pi * timestamps.dt.month / 12.0)
    result["day_sin"] = np.sin(2.0 * np.pi * timestamps.dt.day / 31.0)
    result["day_cos"] = np.cos(2.0 * np.pi * timestamps.dt.day / 31.0)
    result["linear_time"] = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds()
        / (365.25 * 24.0 * 60.0 * 60.0)
    )
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("event panel timestamps are not monotonic")
    return result


def origin_visible_baselines(anchor_panel: pd.DataFrame) -> pd.DataFrame:
    """Return the lag-1/level baseline available at every monthly anchor."""

    rows: list[dict[str, Any]] = []
    for anchor in anchor_panel.itertuples(index=False):
        origin_month = str(anchor.origin_month)
        target_month = _month_text(pd.Timestamp(origin_month) + pd.DateOffset(months=1))
        values = anchor._asdict()
        for metric, column in ORIGIN_BASELINE_FIELDS.items():
            value = float(values[column])
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "origin_month": origin_month,
                    "target_month": target_month,
                    "metric": metric,
                    "baseline_prediction": value,
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["origin_month", "metric"]).any():
        raise ValueError("origin-visible baselines contain duplicates")
    return result.sort_values(["origin_month", "metric"]).reset_index(drop=True)


def state_feature_columns(
    state_series: Sequence[str] = STATE_SERIES,
) -> list[str]:
    return [
        f"series__{series_id}__{suffix}"
        for series_id in state_series
        for suffix in (
            "level",
            "change_1m",
            "change_3m",
            "observation_age_days",
            "release_age_days",
        )
    ]


def event_observation_columns(
    state_series: Sequence[str] = STATE_SERIES,
) -> list[str]:
    return state_feature_columns(state_series) + [
        "is_forecast_anchor",
        "log_delta_t_s",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "linear_time",
    ]


def _event_base_row(
    *,
    as_of: str,
    latest: Mapping[tuple[str, str], MacroEvent],
    observations_by_series: Mapping[str, set[str]],
    state_series: Sequence[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"as_of_date": as_of}
    for series_id in state_series:
        available = [
            latest[(series_id, observation)]
            for observation in sorted(observations_by_series.get(series_id, set()))
            if (series_id, observation) in latest
        ]
        prefix = f"series__{series_id}"
        if not available:
            for suffix in (
                "level",
                "change_1m",
                "change_3m",
                "observation_age_days",
                "release_age_days",
            ):
                row[f"{prefix}__{suffix}"] = np.nan
            continue
        current = available[-1]
        row[f"{prefix}__level"] = current.value
        row[f"{prefix}__change_1m"] = _state_change(series_id, available, 1)
        row[f"{prefix}__change_3m"] = _state_change(series_id, available, 3)
        row[f"{prefix}__observation_age_days"] = float(
            (pd.Timestamp(as_of) - pd.Timestamp(current.observation_date)).days
        )
        row[f"{prefix}__release_age_days"] = float(
            (pd.Timestamp(as_of) - pd.Timestamp(current.vintage_date)).days
        )
    return row


def _latest_series_event(
    series_id: str,
    latest: Mapping[tuple[str, str], MacroEvent],
    observations_by_series: Mapping[str, set[str]],
) -> MacroEvent | None:
    observations = sorted(observations_by_series.get(series_id, set()))
    if not observations:
        return None
    return latest.get((series_id, observations[-1]))


def _event_level_change(
    series_id: str,
    before: MacroEvent | None,
    after: MacroEvent | None,
) -> float:
    if before is None or after is None or before.value == after.value:
        return 0.0
    if series_id in DIFFERENCE_SERIES:
        return float(after.value - before.value)
    if before.value == 0:
        return 0.0
    return float(100.0 * (after.value / before.value - 1.0))


def _split_for_date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp <= pd.Timestamp("2021-12-31T23:59:59Z"):
        return "train"
    if timestamp <= pd.Timestamp("2023-12-31T23:59:59Z"):
        return "val"
    return "test"


def first_release_targets(events: Sequence[MacroEvent]) -> pd.DataFrame:
    """Reconstruct same-vintage first-release target transforms."""

    grouped: dict[tuple[str, str], list[MacroEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.series_id, event.observation_date)].append(event)

    rows: list[dict[str, Any]] = []
    for metric, (series_id, transform) in TARGET_SPECS.items():
        observations = sorted(
            observation
            for candidate_series, observation in grouped
            if candidate_series == series_id
        )
        for observation_date in observations:
            initial = [
                row
                for row in grouped[(series_id, observation_date)]
                if row.event_kind == "initial_release"
            ]
            if not initial:
                continue
            release = min(initial, key=lambda row: (row.vintage_date, row.event_id))
            denominator_date = _month_offset(observation_date, -1)
            denominator_candidates = [
                row
                for row in grouped.get((series_id, denominator_date), [])
                if row.vintage_date <= release.vintage_date
            ]
            if transform != "level" and not denominator_candidates:
                continue
            denominator = (
                None
                if transform == "level"
                else max(
                    denominator_candidates,
                    key=lambda row: (
                        row.vintage_date,
                        row.event_kind == "revision",
                        row.event_id,
                    ),
                )
            )
            rows.append(
                {
                    "target_month": observation_date,
                    "metric": metric,
                    "series_id": series_id,
                    "transform": transform,
                    "first_release_date": release.vintage_date,
                    "first_release_value": release.value,
                    "denominator_date": denominator_date if denominator else "",
                    "first_release_denominator_value": (
                        np.nan if denominator is None else denominator.value
                    ),
                    "target_value": _target_transform(
                        numerator=release.value,
                        denominator=None if denominator is None else denominator.value,
                        transform=transform,
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("first-release target reconstruction produced no rows")
    if result.duplicated(["target_month", "metric"]).any():
        raise ValueError("first-release targets contain duplicate month+metric rows")
    return result.sort_values(["target_month", "metric"]).reset_index(drop=True)


def build_origin_state_panel(
    events: Sequence[MacroEvent],
    *,
    start_month: str,
    end_month: str,
    as_of_day: int = 15,
    as_of_date_overrides: Mapping[str, str] | None = None,
    state_series: Sequence[str] = STATE_SERIES,
) -> pd.DataFrame:
    """Build one origin-safe macro state per month from release-time events."""

    if not 1 <= int(as_of_day) <= 28:
        raise ValueError("as_of_day must be between 1 and 28")
    origins = pd.date_range(_month_text(start_month), _month_text(end_month), freq="MS")
    selected = set(state_series)
    ordered = [row for row in events if row.series_id in selected]
    latest: dict[tuple[str, str], MacroEvent] = {}
    observations_by_series: dict[str, set[str]] = defaultdict(set)
    event_index = 0
    previous_as_of: str | None = None
    rows: list[dict[str, Any]] = []
    exact_as_of_dates = dict(as_of_date_overrides or {})

    for step_index, origin in enumerate(origins):
        origin_month = origin.date().isoformat()
        as_of = exact_as_of_dates.get(
            origin_month, date(origin.year, origin.month, int(as_of_day)).isoformat()
        )
        if pd.Timestamp(as_of).to_period("M") != origin.to_period("M"):
            raise ValueError(
                f"as-of override must remain inside origin month: {origin_month} {as_of}"
            )
        interval_events: list[MacroEvent] = []
        revision_pct: list[float] = []
        while event_index < len(ordered) and ordered[event_index].vintage_date <= as_of:
            event = ordered[event_index]
            prior = latest.get((event.series_id, event.observation_date))
            if (
                event.event_kind == "revision"
                and prior is not None
                and prior.value != 0
                and (previous_as_of is None or event.vintage_date > previous_as_of)
            ):
                revision_pct.append(abs(100.0 * (event.value / prior.value - 1.0)))
            latest[(event.series_id, event.observation_date)] = event
            observations_by_series[event.series_id].add(event.observation_date)
            if previous_as_of is None or event.vintage_date > previous_as_of:
                interval_events.append(event)
            event_index += 1

        row: dict[str, Any] = {
            "episode_id": "monthly_us_macro",
            "step_idx": step_index,
            "timestamp": f"{as_of}T00:00:00Z",
            "delta_t_s": (
                0.0
                if previous_as_of is None
                else float(
                    (pd.Timestamp(as_of) - pd.Timestamp(previous_as_of)).total_seconds()
                )
            ),
            "done": False,
            "split": _split_for_month(origin),
            "action_name": "origin_visible_macro_information_update",
            "origin_month": origin_month,
            "as_of_date": as_of,
            "release_event_count": float(
                sum(event.event_kind == "initial_release" for event in interval_events)
            ),
            "revision_event_count": float(
                sum(event.event_kind == "revision" for event in interval_events)
            ),
            "mean_absolute_revision_pct": (
                float(np.mean(revision_pct)) if revision_pct else 0.0
            ),
        }

        for series_id in state_series:
            available = [
                latest[(series_id, observation)]
                for observation in sorted(observations_by_series.get(series_id, set()))
                if (series_id, observation) in latest
            ]
            prefix = f"series__{series_id}"
            if not available:
                for suffix in (
                    "level",
                    "change_1m",
                    "change_3m",
                    "observation_age_days",
                    "release_age_days",
                ):
                    row[f"{prefix}__{suffix}"] = np.nan
                continue
            current = available[-1]
            row[f"{prefix}__level"] = current.value
            row[f"{prefix}__change_1m"] = _state_change(series_id, available, 1)
            row[f"{prefix}__change_3m"] = _state_change(series_id, available, 3)
            row[f"{prefix}__observation_age_days"] = float(
                (pd.Timestamp(as_of) - pd.Timestamp(current.observation_date)).days
            )
            row[f"{prefix}__release_age_days"] = float(
                (pd.Timestamp(as_of) - pd.Timestamp(current.vintage_date)).days
            )

        row.update(_action_values(row))
        rows.append(row)
        previous_as_of = as_of

    if rows:
        rows[-1]["done"] = True
    return pd.DataFrame(rows)


def apply_origin_information_overrides(
    panel: pd.DataFrame,
    origin_information_paths: Mapping[str, Path | str],
) -> pd.DataFrame:
    """Replace reconstructed origin fields with the exact cards used by LLM-HANK."""

    result = panel.copy()
    for origin_month, path in origin_information_paths.items():
        matches = result.index[result["origin_month"].astype(str).eq(str(origin_month))]
        if len(matches) != 1:
            raise ValueError(
                f"origin override does not match exactly one panel row: {origin_month}"
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("origin_month")) != str(origin_month):
            raise ValueError(f"origin information month mismatch: {origin_month}")
        row_index = matches[0]
        result.loc[row_index, "as_of_date"] = str(payload["as_of_date"])
        result.loc[row_index, "timestamp"] = f"{payload['as_of_date']}T00:00:00Z"
        series = payload["compact_macro_information"]["series"]
        for series_id, values in series.items():
            if series_id not in STATE_SERIES or not bool(values.get("available")):
                continue
            expected_change_method = (
                "level_delta" if series_id in DIFFERENCE_SERIES else "percent_change"
            )
            if values.get("change_method") != expected_change_method:
                raise ValueError(
                    f"origin information change method mismatch for {series_id}: "
                    f"expected {expected_change_method}, got {values.get('change_method')}"
                )
            prefix = f"series__{series_id}"
            result.loc[row_index, f"{prefix}__level"] = _canonical_series_value(
                series_id, float(values["latest_value"])
            )
            result.loc[row_index, f"{prefix}__observation_age_days"] = float(
                values["staleness_days"]
            )
            release_date = values.get("latest_release_date")
            result.loc[row_index, f"{prefix}__release_age_days"] = (
                float(
                    (
                        pd.Timestamp(payload["as_of_date"])
                        - pd.Timestamp(release_date)
                    ).days
                )
                if release_date
                else np.nan
            )
            changes = values.get("changes", {})
            if "1m" in changes:
                result.loc[row_index, f"{prefix}__change_1m"] = float(
                    changes["1m"]["value"]
                )
            if "3m" in changes:
                result.loc[row_index, f"{prefix}__change_3m"] = float(
                    changes["3m"]["value"]
                )
        for column, value in _action_values(result.loc[row_index].to_dict()).items():
            result.loc[row_index, column] = value
    return result


def validate_canonical_series_units(panel: pd.DataFrame) -> None:
    """Fail if a normalized state series still contains its old raw unit."""

    for series_id, rule in SERIES_VALUE_NORMALIZATION.items():
        column = f"series__{series_id}__level"
        if column not in panel.columns:
            continue
        values = pd.to_numeric(panel[column], errors="coerce").dropna().abs()
        threshold = float(rule["million_value_threshold"])
        if not values.empty and bool(values.ge(threshold).any()):
            raise ValueError(
                f"noncanonical level unit remains for {series_id}: {values.max()}"
            )


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    months = pd.to_datetime(result["origin_month"])
    result["month_sin"] = np.sin(2.0 * np.pi * months.dt.month / 12.0)
    result["month_cos"] = np.cos(2.0 * np.pi * months.dt.month / 12.0)
    result["linear_time"] = np.arange(len(result), dtype=float) / 12.0
    return result


def observation_columns(panel: pd.DataFrame) -> list[str]:
    return sorted(column for column in panel.columns if column.startswith("series__")) + [
        "month_sin",
        "month_cos",
        "linear_time",
    ]


def _action_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "policy_rate_change_1m": _finite_or_zero(
            row.get("series__FEDFUNDS__change_1m")
        ),
        "money_growth_1m": _finite_or_zero(row.get("series__M2SL__change_1m")),
        "consumer_credit_growth_1m": _finite_or_zero(
            row.get("series__TOTALSL__change_1m")
        ),
        "business_credit_growth_1m": _finite_or_zero(
            row.get("series__BUSLOANS__change_1m")
        ),
        "sentiment_change_1m": _finite_or_zero(
            row.get("series__UMCSENT__change_1m")
        ),
        "claims_growth_1m": _finite_or_zero(row.get("series__ICSA__change_1m")),
    }


def _state_change(series_id: str, observations: Sequence[MacroEvent], lag: int) -> float:
    if len(observations) <= 1:
        return float("nan")
    current = observations[-1].value
    comparison_date = pd.Timestamp(observations[-1].observation_date) - pd.DateOffset(
        months=lag
    )
    candidates = [
        observation
        for observation in observations[:-1]
        if pd.Timestamp(observation.observation_date) <= comparison_date
    ]
    if not candidates:
        return float("nan")
    previous = candidates[-1].value
    if series_id in DIFFERENCE_SERIES:
        return current - previous
    if previous == 0:
        return float("nan")
    return 100.0 * (current / previous - 1.0)


def _target_transform(*, numerator: float, denominator: float | None, transform: str) -> float:
    if transform == "level":
        return float(numerator)
    if denominator is None:
        raise ValueError("non-level target requires a denominator")
    if transform == "diff":
        return float(numerator - denominator)
    if transform == "pct_change":
        if denominator == 0:
            raise ValueError("percentage target denominator cannot be zero")
        return float(100.0 * (numerator / denominator - 1.0))
    raise ValueError(f"unsupported target transform: {transform}")


def _split_for_month(value: pd.Timestamp) -> str:
    if value <= pd.Timestamp("2023-12-01"):
        return "train"
    if value <= pd.Timestamp("2025-12-01"):
        return "val"
    return "test"


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _month_text(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"invalid month: {value!r}")
    return parsed.to_period("M").to_timestamp().date().isoformat()


def _date_text(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value!r}")
    return parsed.date().isoformat()


def _month_offset(value: str, months: int) -> str:
    return (
        pd.Timestamp(value) + pd.DateOffset(months=months)
    ).to_period("M").to_timestamp().date().isoformat()
