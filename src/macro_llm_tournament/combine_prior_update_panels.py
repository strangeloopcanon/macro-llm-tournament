"""Stitch banked prior-update persona-ecology runs into a combined multi-period panel.

The first October-November-December combined panel was assembled ad hoc; this module
makes the stitching reproducible for the v2 December re-run and the January 2025 leg
described in reports/prior_update_environment_v2_prereg.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

COMBINED_SCHEMA_VERSION = "persona_belief_ecology_combined_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine a banked prior-update run with extension legs into one panel.")
    parser.add_argument("--bank-dir", type=Path, required=True, help="Multi-period banked run (e.g. the Oct-Nov 100-respondent bank).")
    parser.add_argument(
        "--extension-dirs",
        required=True,
        help="Comma-separated single-period extension run dirs, in period order.",
    )
    parser.add_argument("--primary-source", default="llm_codex_cli_gpt-5.5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note", default="", help="Verdict/claim-scope note recorded in the combined manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = combine_prior_update_panels(
        bank_dir=args.bank_dir,
        extension_dirs=[Path(item) for item in args.extension_dirs.split(",") if item],
        primary_source=args.primary_source,
        output_dir=args.output_dir,
        note=args.note,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    panel_path = run_dir / "persona_ecology_panel.csv"
    predictions_path = run_dir / "persona_ecology_predictions.csv"
    for path in (manifest_path, panel_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"Run artifact missing: {path}")
    return {
        "dir": run_dir,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "panel": pd.read_csv(panel_path),
        "predictions": pd.read_csv(predictions_path),
        "hashes": {
            "manifest_sha256": _file_sha256(manifest_path),
            "panel_sha256": _file_sha256(panel_path),
            "predictions_sha256": _file_sha256(predictions_path),
        },
    }


def combine_prior_update_panels(
    *,
    bank_dir: Path,
    extension_dirs: list[Path],
    primary_source: str,
    output_dir: Path,
    note: str = "",
) -> dict[str, Any]:
    if not extension_dirs:
        raise ValueError("At least one extension dir is required")
    bank = _load_run(bank_dir)
    extensions = [_load_run(run_dir) for run_dir in extension_dirs]

    respondent_ids = set(extensions[0]["panel"]["respondent_id"].astype(str))
    for extension in extensions[1:]:
        ids = set(extension["panel"]["respondent_id"].astype(str))
        respondent_ids &= ids
    if not respondent_ids:
        raise ValueError("Extension runs have no common respondents")

    panels: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []

    bank_panel = bank["panel"][bank["panel"]["respondent_id"].astype(str).isin(respondent_ids)].copy()
    bank_predictions = bank["predictions"][
        bank["predictions"]["respondent_id"].astype(str).isin(respondent_ids)
        & bank["predictions"]["source"].astype(str).eq(primary_source)
    ].copy()
    if bank_panel.empty or bank_predictions.empty:
        raise ValueError("Bank run has no rows for the requested respondents/source")
    panels.append(bank_panel)
    predictions.append(bank_predictions)

    for extension in extensions:
        extension_panel = extension["panel"][extension["panel"]["respondent_id"].astype(str).isin(respondent_ids)].copy()
        extension_predictions = extension["predictions"][
            extension["predictions"]["respondent_id"].astype(str).isin(respondent_ids)
            & extension["predictions"]["source"].astype(str).eq(primary_source)
        ].copy()
        if extension_panel.empty or extension_predictions.empty:
            raise ValueError(f"Extension run has no usable rows: {extension['dir']}")
        panels.append(extension_panel)
        predictions.append(extension_predictions)

    combined_panel = pd.concat(panels, ignore_index=True).sort_values(["period_index", "respondent_id"]).reset_index(drop=True)
    combined_predictions = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["period_index", "respondent_id", "target_name"])
        .reset_index(drop=True)
    )

    _validate_combined(combined_panel, combined_predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "persona_ecology_panel.csv"
    predictions_path = output_dir / "persona_ecology_predictions.csv"
    combined_panel.to_csv(panel_path, index=False)
    combined_predictions.to_csv(predictions_path, index=False)

    manifest = {
        "schema_version": COMBINED_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "provider": bank["manifest"].get("provider"),
        "models": [primary_source.split("_")[-1]],
        "primary_update_source": primary_source,
        "prior_mode": bank["manifest"].get("prior_mode"),
        "feedback_mode": bank["manifest"].get("feedback_mode"),
        "date_mode": bank["manifest"].get("date_mode"),
        "respondent_count": int(combined_panel["respondent_id"].nunique()),
        "period_count": int(combined_panel["period_id"].nunique()),
        "period_ids_filter": sorted(combined_panel["period_id"].astype(str).unique()),
        "panel_row_count": int(combined_panel.shape[0]),
        "prediction_row_count": int(combined_predictions.shape[0]),
        "leakage_rule": (
            "Combined only from prompt outputs and panel rows; no actual target columns are used by "
            "Phase 4 demand-economy inputs."
        ),
        "prior_update_evidence": {"note": note} if note else {},
        "upstream_prior_update_evidence": {
            "bank": bank["manifest"].get("prior_update_evidence"),
            **{
                f"extension_{index}": extension["manifest"].get("prior_update_evidence")
                for index, extension in enumerate(extensions)
            },
        },
        "source_artifacts": {
            "bank": {"path": str(bank_dir), **bank["hashes"]},
            **{
                f"extension_{index}": {"path": str(extension["dir"]), **extension["hashes"]}
                for index, extension in enumerate(extensions)
            },
        },
        "status": "ok",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "respondent_count": manifest["respondent_count"],
        "period_count": manifest["period_count"],
        "panel_row_count": manifest["panel_row_count"],
        "prediction_row_count": manifest["prediction_row_count"],
        "period_ids": manifest["period_ids_filter"],
    }


def _validate_combined(panel: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if panel["panel_row_id"].duplicated().any():
        raise ValueError("Combined panel has duplicate panel_row_id values")
    respondent_sets = panel.groupby("period_index")["respondent_id"].apply(lambda ids: frozenset(ids.astype(str)))
    if respondent_sets.nunique() != 1:
        raise ValueError("Combined panel periods do not share an identical respondent set")
    target_count = predictions["target_name"].nunique()
    expected_predictions = panel.shape[0] * target_count
    if predictions.shape[0] != expected_predictions:
        raise ValueError(
            f"Combined predictions row count {predictions.shape[0]} does not match panel rows {panel.shape[0]} x {target_count} targets"
        )
    if predictions.groupby(["respondent_id", "period_index", "target_name"]).size().max() != 1:
        raise ValueError("Combined predictions contain duplicate respondent/period/target rows")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
