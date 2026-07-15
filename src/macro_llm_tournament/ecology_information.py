"""Origin-safe public macro information cards for the household ecology.

The ecology runner uses this module to replace the raw frozen-origin structure
with a compact, validated presentation payload for household cards.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


MACRO_INFORMATION_CARD_SCHEMA_VERSION = "ecology_macro_information_card_v1"

# These are the complete public-series universe of the frozen ecology bundle.
# A caller may pass a strict subset to build_macro_information_card, but never
# add an arbitrary series at the presentation boundary.
ALLOWED_PUBLIC_SERIES: dict[str, dict[str, str]] = {
    "CPIAUCSL": {"label": "consumer price index", "units": "index 1982-84=100", "kind": "level"},
    "DSPIC96": {"label": "real disposable personal income", "units": "billions of chained dollars", "kind": "level"},
    "FEDFUNDS": {"label": "effective federal funds rate", "units": "percent", "kind": "rate"},
    "PAYEMS": {"label": "all employees, total nonfarm", "units": "thousands of persons", "kind": "level"},
    "PCE": {"label": "personal consumption expenditures", "units": "billions of dollars", "kind": "level"},
    "PCEPI": {"label": "personal consumption expenditures price index", "units": "index", "kind": "level"},
    "PCEC96": {"label": "real personal consumption expenditures", "units": "billions of chained dollars", "kind": "level"},
    "PSAVERT": {"label": "personal saving rate", "units": "percent", "kind": "rate"},
    "REVOLSL": {"label": "revolving consumer credit", "units": "millions of dollars", "kind": "level"},
    "RSAFS": {"label": "advance retail sales", "units": "millions of dollars", "kind": "level"},
    "UMCSENT": {"label": "consumer sentiment", "units": "index", "kind": "level"},
    "UNRATE": {"label": "civilian unemployment rate", "units": "percent", "kind": "rate"},
}

_OBSERVATION_FIELDS = frozenset({"observation_date", "value", "release_date", "public_availability_date"})
_PUBLIC_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "event_date",
        "event_type",
        "public_availability_date",
        "release_date",
        "title",
        "summary",
        "source",
        "url",
    }
)
_POLICY_DECLARATION_FIELDS = frozenset({"declared_spread_bps", "declared_pass_through_fraction"})
_FORBIDDEN_PREFIXES = ("actual_", "target_")


def canonical_sha256(value: Any) -> str:
    """Return a deterministic hash for a JSON-compatible, finite payload."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_macro_information_card(
    origin_information: Mapping[str, Any],
    *,
    allowed_series: Sequence[str] | None = None,
    policy_declarations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact public-information card from an ecology origin payload.

    Each configured public series is represented, including unavailable ones.
    Rates use percentage-point deltas; all other series use percentage changes.
    No realized ``actual_*`` or ``target_*`` field is allowed at this boundary.
    """

    _reject_leakage_fields(origin_information)
    origin_month = _parse_date(origin_information.get("origin_month"), "origin_month")
    if origin_month.day != 1:
        raise ValueError("origin_month must be a month-start date")
    as_of = _parse_date(origin_information.get("as_of_date"), "as_of_date")
    if origin_month > as_of:
        raise ValueError("origin_month cannot be later than as_of_date")
    selected = _selected_series(allowed_series)
    history_series = tuple(sorted(set(selected) | {"FEDFUNDS"}))
    histories = _normalize_histories(origin_information.get("origin_visible_macro_history"), as_of, history_series)
    _validate_context(origin_information.get("origin_visible_macro_context", {}), histories, as_of, history_series)
    events = _normalize_public_events(origin_information.get("public_events", []), as_of)
    declarations = _normalize_policy_declarations(policy_declarations)

    series = {
        series_id: _series_summary(series_id, histories.get(series_id, ()), as_of)
        for series_id in selected
    }
    payload = {
        "schema_version": MACRO_INFORMATION_CARD_SCHEMA_VERSION,
        "origin_month": origin_month.isoformat(),
        "as_of_date": as_of.isoformat(),
        "series": series,
        "public_events": events,
        "policy": build_policy_payload(histories.get("FEDFUNDS", ()), as_of, declarations),
    }
    return payload | {"card_sha256": canonical_sha256(payload)}


def build_policy_payload(
    fed_funds_history: Sequence[Mapping[str, Any]],
    as_of_date: str | date,
    declarations: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """Describe visible Fed funds policy without inferring bank spreads or pass-through."""

    as_of = _parse_date(as_of_date, "as_of_date")
    rows = _normalize_observations("FEDFUNDS", fed_funds_history, as_of)
    declared = _normalize_policy_declarations(declarations)
    if not rows:
        return {
            "series_id": "FEDFUNDS",
            "available": False,
            "current_visible_rate": None,
            "current_observation_date": None,
            "previous_visible_rate": None,
            "previous_observation_date": None,
            "basis_point_change": None,
            **declared,
        }
    current = rows[-1]
    previous = rows[-2] if len(rows) > 1 else None
    return {
        "series_id": "FEDFUNDS",
        "available": True,
        "current_visible_rate": current["value"],
        "current_observation_date": current["observation_date"],
        "previous_visible_rate": previous["value"] if previous else None,
        "previous_observation_date": previous["observation_date"] if previous else None,
        "basis_point_change": (current["value"] - previous["value"]) * 100.0 if previous else None,
        **declared,
    }


def _selected_series(allowed_series: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(sorted(ALLOWED_PUBLIC_SERIES if allowed_series is None else set(allowed_series)))
    if not selected:
        raise ValueError("allowed_series cannot be empty")
    unknown = set(selected) - ALLOWED_PUBLIC_SERIES.keys()
    if unknown:
        raise ValueError(f"unapproved public series: {sorted(unknown)}")
    return selected


def _normalize_histories(
    value: Any, as_of: date, selected: Sequence[str]
) -> dict[str, tuple[dict[str, Any], ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("origin_visible_macro_history must be a mapping")
    unknown = set(value) - ALLOWED_PUBLIC_SERIES.keys()
    if unknown:
        raise ValueError(f"unapproved public series: {sorted(unknown)}")
    return {
        series_id: _normalize_observations(series_id, value.get(series_id, ()), as_of)
        for series_id in selected
    }


def _normalize_observations(
    series_id: str, value: Any, as_of: date
) -> tuple[dict[str, Any], ...]:
    if series_id not in ALLOWED_PUBLIC_SERIES:
        raise ValueError(f"unapproved public series: {series_id}")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{series_id} history must be a sequence")
    rows: list[dict[str, Any]] = []
    dates: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError(f"{series_id} observation must be a mapping")
        _reject_leakage_fields(row)
        extras = set(row) - _OBSERVATION_FIELDS
        if extras:
            raise ValueError(f"{series_id} observation has unapproved fields: {sorted(extras)}")
        observation_date = _parse_date(row.get("observation_date"), f"{series_id}.observation_date")
        if observation_date > as_of:
            raise ValueError(f"{series_id} has a future observation")
        release_date = _optional_visible_date(row, as_of, series_id)
        value_float = _finite_float(row.get("value"), f"{series_id}.value")
        key = observation_date.isoformat()
        if key in dates:
            raise ValueError(f"{series_id} contains duplicate observation dates")
        dates.add(key)
        rows.append(
            {
                "observation_date": key,
                "release_date": release_date.isoformat() if release_date else None,
                "value": value_float,
            }
        )
    return tuple(sorted(rows, key=lambda row: row["observation_date"]))


def _validate_context(value: Any, histories: Mapping[str, Sequence[Mapping[str, Any]]], as_of: date, selected: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("origin_visible_macro_context must be a mapping")
    unknown = set(value) - ALLOWED_PUBLIC_SERIES.keys()
    if unknown:
        raise ValueError(f"unapproved public series: {sorted(unknown)}")
    for series_id, row in value.items():
        if series_id not in selected:
            continue
        normalized = _normalize_observations(series_id, [row], as_of)
        history = histories.get(series_id, ())
        if history and normalized[0]["observation_date"] != history[-1]["observation_date"]:
            raise ValueError(f"{series_id} context does not match the latest visible history")
        if history and normalized[0]["value"] != history[-1]["value"]:
            raise ValueError(f"{series_id} context does not match the latest visible history")


def _series_summary(series_id: str, history: Sequence[Mapping[str, Any]], as_of: date) -> dict[str, Any]:
    metadata = ALLOWED_PUBLIC_SERIES[series_id]
    method = "level_delta" if metadata["kind"] == "rate" else "percent_change"
    if not history:
        return {
            "label": metadata["label"],
            "units": metadata["units"],
            "kind": metadata["kind"],
            "available": False,
            "latest_value": None,
            "latest_observation_date": None,
            "latest_release_date": None,
            "latest_visible_date": None,
            "staleness_days": None,
            "change_method": method,
            "changes": {f"{months}m": None for months in (1, 3, 12)},
        }
    latest = history[-1]
    observation = _parse_date(latest["observation_date"], f"{series_id}.observation_date")
    release = _parse_date(latest["release_date"], f"{series_id}.release_date") if latest["release_date"] else None
    visible = release or observation
    return {
        "label": metadata["label"],
        "units": metadata["units"],
        "kind": metadata["kind"],
        "available": True,
        "latest_value": latest["value"],
        "latest_observation_date": latest["observation_date"],
        "latest_release_date": latest["release_date"],
        "latest_visible_date": visible.isoformat(),
        "staleness_days": (as_of - observation).days,
        "change_method": method,
        "changes": {f"{months}m": _change(history, latest, months, metadata["kind"]) for months in (1, 3, 12)},
    }


def _change(history: Sequence[Mapping[str, Any]], latest: Mapping[str, Any], months: int, kind: str) -> dict[str, Any] | None:
    latest_date = _parse_date(latest["observation_date"], "latest observation_date")
    cutoff = _subtract_months(latest_date, months)
    candidates = [row for row in history if _parse_date(row["observation_date"], "observation_date") <= cutoff]
    if not candidates:
        return None
    base = candidates[-1]
    if kind == "rate":
        value = latest["value"] - base["value"]
    elif base["value"] == 0:
        return None
    else:
        value = 100.0 * (latest["value"] / base["value"] - 1.0)
    return {"value": value, "base_observation_date": base["observation_date"]}


def _normalize_public_events(value: Any, as_of: date) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("public_events must be a sequence")
    events: list[dict[str, Any]] = []
    for event in value:
        if not isinstance(event, Mapping):
            raise ValueError("public event must be a mapping")
        _reject_leakage_fields(event)
        extras = set(event) - _PUBLIC_EVENT_FIELDS
        if extras:
            raise ValueError(f"public event has unapproved fields: {sorted(extras)}")
        if not event.get("event_date") or not event.get("title") or not event.get("event_type"):
            raise ValueError("public event requires event_date, event_type, and title")
        normalized = {key: event[key] for key in sorted(event)}
        _parse_date(normalized["event_date"], "public event event_date")
        for key in ("public_availability_date", "release_date"):
            if normalized.get(key) is not None and _parse_date(normalized[key], f"public event {key}") > as_of:
                raise ValueError("public event was not available by as_of_date")
        events.append(normalized)
    return sorted(events, key=lambda event: (str(event["event_date"]), str(event.get("event_id", "")), str(event["title"])))


def _normalize_policy_declarations(value: Mapping[str, Any] | None) -> dict[str, float | None]:
    if value is None:
        return {field: None for field in sorted(_POLICY_DECLARATION_FIELDS)}
    if not isinstance(value, Mapping):
        raise ValueError("policy_declarations must be a mapping")
    _reject_leakage_fields(value)
    extras = set(value) - _POLICY_DECLARATION_FIELDS
    if extras:
        raise ValueError(f"policy declarations have unapproved fields: {sorted(extras)}")
    spread = value.get("declared_spread_bps")
    pass_through = value.get("declared_pass_through_fraction")
    if pass_through is not None:
        pass_through = _finite_float(pass_through, "declared_pass_through_fraction")
        if not 0.0 <= pass_through <= 1.0:
            raise ValueError("declared_pass_through_fraction must be between zero and one")
    return {
        "declared_pass_through_fraction": pass_through,
        "declared_spread_bps": _finite_float(spread, "declared_spread_bps") if spread is not None else None,
    }


def _optional_visible_date(row: Mapping[str, Any], as_of: date, series_id: str) -> date | None:
    dates = [row[key] for key in ("release_date", "public_availability_date") if row.get(key) is not None]
    if len(dates) == 2 and dates[0] != dates[1]:
        raise ValueError(f"{series_id} has conflicting release dates")
    if not dates:
        return None
    release_date = _parse_date(dates[0], f"{series_id}.release_date")
    if release_date > as_of:
        raise ValueError(f"{series_id} has a future release")
    return release_date


def _reject_leakage_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if name.startswith(_FORBIDDEN_PREFIXES):
                raise ValueError(f"origin information contains forbidden leakage field: {key}")
            _reject_leakage_fields(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_leakage_fields(item)


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _subtract_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 - months
    year, month_index = divmod(index, 12)
    month = month_index + 1
    # Observation rows are normally month starts, but clamping makes the helper
    # well-defined for the occasional non-month-start public release date.
    month_ends = (31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return date(year, month, min(value.day, month_ends[month - 1]))
