"""Materialize an origin-aware, public selected-household history.

The raw SCE respondent identifier is used only while matching a normalized
wave to the separately supplied private registry.  It is never carried into
the public history or its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .persistent_households import (
    HISTORY_COLUMNS,
    PRIVATE_REGISTRY_COLUMNS,
    PersistentHouseholdError,
    append_selected_observed_history,
)
from .prepare_dynamic_macro_panel import (
    DEFAULT_PUBLICATION_LAG_MONTHS,
    DynamicMacroPanelError,
    _month_start,
    _month_text,
    validate_normalized_sce,
)


ECOLOGY_HISTORY_SCHEMA_VERSION = "ecology_history_materialization_v1"
SELECTED_HOUSEHOLD_COUNT = 200


class EcologyHistoryError(ValueError):
    """Raised when a public history cannot be materialized without guessing."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise EcologyHistoryError(f"{label} is not a readable nonempty CSV: {path}") from exc
    if frame.empty:
        raise EcologyHistoryError(f"{label} must contain at least one row")
    return frame


def _require_exact_columns(frame: pd.DataFrame, expected: Sequence[str], *, label: str) -> None:
    actual = list(frame.columns)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise EcologyHistoryError(f"{label} columns do not match the public schema: {'; '.join(details)}")


def _require_nonblank(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    bad = [
        column
        for column in columns
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()
    ]
    if bad:
        raise EcologyHistoryError(f"{label} contains missing or blank fields: {', '.join(sorted(bad))}")


def _parse_boolean(series: pd.Series, *, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise EcologyHistoryError(f"{field} contains missing boolean values")
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False}
    if (~values.isin(allowed)).any():
        raise EcologyHistoryError(f"{field} must contain only true or false")
    return values.map(allowed).astype(bool)


def _validate_private_registry(frame: pd.DataFrame) -> pd.DataFrame:
    _require_exact_columns(frame, PRIVATE_REGISTRY_COLUMNS, label="private registry")
    _require_nonblank(frame, ("household_id", "raw_respondent_id"), label="private registry")
    registry = frame.copy()
    registry["household_id"] = registry["household_id"].astype(str).str.strip()
    registry["raw_respondent_id"] = registry["raw_respondent_id"].astype(str).str.strip()
    for field in ("included_in_master_200", "included_in_core_81"):
        registry[field] = _parse_boolean(registry[field], field=field)
    if registry["household_id"].duplicated().any() or registry["raw_respondent_id"].duplicated().any():
        raise EcologyHistoryError("private registry must map raw IDs one-to-one to public household IDs")
    if len(registry) != SELECTED_HOUSEHOLD_COUNT or not registry["included_in_master_200"].all():
        raise EcologyHistoryError(
            f"private registry must contain exactly {SELECTED_HOUSEHOLD_COUNT} selected master households"
        )
    return registry.sort_values("household_id").reset_index(drop=True)


def _validate_public_history(
    frame: pd.DataFrame,
    *,
    selected_household_ids: set[str],
    label: str,
) -> pd.DataFrame:
    _require_exact_columns(frame, HISTORY_COLUMNS, label=label)
    _require_nonblank(
        frame,
        (
            "household_id",
            "event_date",
            "public_availability_date",
            "source_name",
            "observation_status",
            "responded",
            "attrition_status",
            "death_status",
            "replay_required_from_event_date",
        ),
        label=label,
    )
    history = frame.copy()
    history["household_id"] = history["household_id"].astype(str).str.strip()
    history["responded"] = _parse_boolean(history["responded"], field=f"{label} responded")
    if (~history["observation_status"].isin({"observed", "nonresponse"})).any():
        raise EcologyHistoryError(f"{label} has invalid observation_status values")
    response_status_matches = (
        history["responded"].eq(history["observation_status"].eq("observed"))
    )
    if not response_status_matches.all():
        raise EcologyHistoryError(f"{label} response flags disagree with observation_status")
    event_months = pd.to_datetime(history["event_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    availability_months = (
        pd.to_datetime(history["public_availability_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    replay_months = (
        pd.to_datetime(history["replay_required_from_event_date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if event_months.isna().any() or availability_months.isna().any() or replay_months.isna().any():
        raise EcologyHistoryError(f"{label} contains invalid event, availability, or replay dates")
    if (event_months > availability_months).any():
        raise EcologyHistoryError(f"{label} cannot be public before its survey event")
    if not history["event_date"].astype(str).eq(event_months.map(_month_text)).all():
        raise EcologyHistoryError(f"{label} event_date values must be canonical month starts")
    if not history["public_availability_date"].astype(str).eq(availability_months.map(_month_text)).all():
        raise EcologyHistoryError(f"{label} public_availability_date values must be canonical month starts")
    if not history["replay_required_from_event_date"].astype(str).eq(replay_months.map(_month_text)).all():
        raise EcologyHistoryError(
            f"{label} replay_required_from_event_date values must be canonical month starts"
        )
    if not replay_months.eq(event_months).all():
        raise EcologyHistoryError(f"{label} replay dates must match event dates")
    history["event_date"] = event_months.map(_month_text)
    history["public_availability_date"] = availability_months.map(_month_text)
    history["replay_required_from_event_date"] = replay_months.map(_month_text)
    if history.duplicated(["event_date", "household_id"]).any():
        raise EcologyHistoryError(f"{label} has duplicate household_id+event_date rows")
    for event_date, wave in history.groupby("event_date", sort=True):
        ids = set(wave["household_id"])
        if len(wave) != SELECTED_HOUSEHOLD_COUNT or ids != selected_household_ids:
            raise EcologyHistoryError(
                f"{label} event {event_date} does not contain exactly the selected {SELECTED_HOUSEHOLD_COUNT} households"
            )
    return history.loc[:, HISTORY_COLUMNS].sort_values(["event_date", "household_id"]).reset_index(drop=True)


def _validate_base_history(frame: pd.DataFrame, *, selected_household_ids: set[str]) -> pd.DataFrame:
    return _validate_public_history(
        frame,
        selected_household_ids=selected_household_ids,
        label="base selected history",
    )


def validate_materialized_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete public-history contract at every runtime boundary."""

    if frame.empty:
        raise EcologyHistoryError("materialized public history must contain at least one row")
    if "household_id" not in frame:
        raise EcologyHistoryError("materialized public history is missing household_id")
    selected_household_ids = set(frame["household_id"].dropna().astype(str).str.strip())
    if len(selected_household_ids) != SELECTED_HOUSEHOLD_COUNT:
        raise EcologyHistoryError(
            f"materialized public history must contain exactly {SELECTED_HOUSEHOLD_COUNT} households"
        )
    return _validate_public_history(
        frame,
        selected_household_ids=selected_household_ids,
        label="materialized public history",
    )


def _assert_no_private_identifiers(frame: pd.DataFrame, raw_identifiers: Sequence[str], *, label: str) -> None:
    text_values = [str(value) for value in frame.to_numpy().ravel()]
    for identifier in raw_identifiers:
        if any(identifier in value for value in text_values):
            raise EcologyHistoryError(f"{label} contains a private respondent identifier")


def _validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return validate_normalized_sce(frame)
    except DynamicMacroPanelError as exc:
        raise EcologyHistoryError(f"normalized SCE microdata is malformed: {exc}") from exc


def _assert_empty_output(path: Path, *, label: str) -> None:
    if path.exists():
        if not path.is_file():
            raise EcologyHistoryError(f"{label} must be a file path, not an existing directory: {path}")
        if path.stat().st_size > 0:
            raise EcologyHistoryError(f"refusing to overwrite nonempty {label}: {path}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _history_csv_text(history: pd.DataFrame) -> str:
    return history.loc[:, HISTORY_COLUMNS].to_csv(index=False, lineterminator="\n")


def materialize_ecology_history(
    *,
    base_history_csv: Path | str,
    normalized_sce_csv: Path | str,
    private_registry_csv: Path | str,
    through_event_month: str | pd.Timestamp,
    output_history_csv: Path | str,
    output_manifest_json: Path | str,
    publication_lag_months: int = DEFAULT_PUBLICATION_LAG_MONTHS,
    source_name: str = "SCE respondent microdata",
) -> dict[str, Any]:
    """Append every normalized SCE wave after the base event through one month.

    Output paths must be absent or empty.  The caller must deliberately remove
    a previous materialization before replacing it.
    """

    if publication_lag_months < 0:
        raise EcologyHistoryError("publication lag must be non-negative")
    if not str(source_name).strip():
        raise EcologyHistoryError("source name cannot be blank")
    output_history_path = Path(output_history_csv)
    output_manifest_path = Path(output_manifest_json)
    if output_history_path == output_manifest_path:
        raise EcologyHistoryError("history CSV and manifest JSON must use different output paths")
    _assert_empty_output(output_history_path, label="public history output")
    _assert_empty_output(output_manifest_path, label="manifest output")

    base_path = Path(base_history_csv)
    source_path = Path(normalized_sce_csv)
    registry_path = Path(private_registry_csv)
    registry = _validate_private_registry(_read_csv(registry_path, label="private registry"))
    selected_household_ids = set(registry["household_id"])
    base_history = _validate_base_history(
        _read_csv(base_path, label="base selected history"), selected_household_ids=selected_household_ids
    )
    _assert_no_private_identifiers(
        base_history,
        registry["raw_respondent_id"].tolist(),
        label="base selected history",
    )
    _assert_no_private_identifiers(
        pd.DataFrame({"source_name": [str(source_name).strip()]}),
        registry["raw_respondent_id"].tolist(),
        label="source name",
    )
    normalized_source = _validate_source(_read_csv(source_path, label="normalized SCE microdata"))

    base_max = pd.to_datetime(base_history["event_date"]).max().to_period("M").to_timestamp()
    through = _month_start(through_event_month, field="through event month")
    if through < base_max:
        raise EcologyHistoryError("through event month cannot precede the base history maximum event month")
    requested_months = list(pd.date_range(base_max + pd.offsets.MonthBegin(1), through, freq="MS"))
    available_months = set(normalized_source["_survey_month"])
    missing_months = [month for month in requested_months if month not in available_months]
    if missing_months:
        raise EcologyHistoryError(
            "normalized SCE microdata is missing requested append waves: "
            + ", ".join(_month_text(month) for month in missing_months)
        )

    history = base_history
    appended_waves: list[dict[str, Any]] = []
    for month in requested_months:
        observations = normalized_source.loc[normalized_source["_survey_month"].eq(month)].drop(
            columns="_survey_month"
        )
        try:
            update = append_selected_observed_history(
                history,
                observations,
                registry,
                event_date=month,
                publication_lag_months=publication_lag_months,
                source_name=str(source_name).strip(),
            )
        except PersistentHouseholdError as exc:
            raise EcologyHistoryError(f"could not append SCE wave {_month_text(month)}: {exc}") from exc
        appended = update["appended_history"]
        if len(appended) != SELECTED_HOUSEHOLD_COUNT or set(appended["household_id"]) != selected_household_ids:
            raise EcologyHistoryError(f"append did not preserve all selected households for {_month_text(month)}")
        history = update["observed_history"]
        appended_waves.append(
            {
                "event_date": _month_text(month),
                "public_availability_date": str(appended["public_availability_date"].iloc[0]),
                "matched_observation_count": int(update["matched_observation_count"]),
                "unselected_observation_count": int(update["unselected_observation_count"]),
                "nonresponse_count": int((~appended["responded"].astype(bool)).sum()),
            }
        )

    history = history.loc[:, HISTORY_COLUMNS].sort_values(["event_date", "household_id"]).reset_index(drop=True)
    for event_date, wave in history.groupby("event_date", sort=True):
        if len(wave) != SELECTED_HOUSEHOLD_COUNT or set(wave["household_id"]) != selected_household_ids:
            raise EcologyHistoryError(f"materialized event {event_date} does not preserve all selected households")
    history_text = _history_csv_text(history)
    history_hash = hashlib.sha256(history_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": ECOLOGY_HISTORY_SCHEMA_VERSION,
        "through_event_month": _month_text(through),
        "base_max_event_date": _month_text(base_max),
        "publication_lag_months": int(publication_lag_months),
        "source_name": str(source_name).strip(),
        "selected_household_count": SELECTED_HOUSEHOLD_COUNT,
        "input_provenance": {
            "base_history": {"sha256": _sha256(base_path), "row_count": int(base_history.shape[0])},
            "normalized_sce_microdata": {"sha256": _sha256(source_path), "row_count": int(normalized_source.shape[0])},
            "private_registry": {"sha256": _sha256(registry_path), "row_count": int(registry.shape[0])},
        },
        "appended_waves": appended_waves,
        "public_history": {
            "sha256": history_hash,
            "row_count": int(history.shape[0]),
            "event_count": int(history["event_date"].nunique()),
            "columns": list(HISTORY_COLUMNS),
        },
        "privacy": "The public history and manifest contain stable household IDs only; respondent identifiers are not materialized.",
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(output_history_path, history_text)
    _atomic_write_text(output_manifest_path, manifest_text)
    return {
        "output_history_csv": output_history_path,
        "output_manifest_json": output_manifest_path,
        "history_sha256": history_hash,
        "appended_wave_count": len(appended_waves),
        "selected_household_count": SELECTED_HOUSEHOLD_COUNT,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize an origin-aware public selected-household history.")
    parser.add_argument("--base-history-csv", type=Path, required=True)
    parser.add_argument("--normalized-sce-csv", type=Path, required=True)
    parser.add_argument("--private-registry-csv", type=Path, required=True)
    parser.add_argument("--through-event-month", required=True, help="Inclusive YYYY-MM event month.")
    parser.add_argument("--output-history-csv", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--publication-lag-months", type=int, default=DEFAULT_PUBLICATION_LAG_MONTHS)
    parser.add_argument("--source-name", default="SCE respondent microdata")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = materialize_ecology_history(
            base_history_csv=args.base_history_csv,
            normalized_sce_csv=args.normalized_sce_csv,
            private_registry_csv=args.private_registry_csv,
            through_event_month=args.through_event_month,
            output_history_csv=args.output_history_csv,
            output_manifest_json=args.output_manifest_json,
            publication_lag_months=args.publication_lag_months,
            source_name=args.source_name,
        )
    except (EcologyHistoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result["output_history_csv"])
    print(result["output_manifest_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
