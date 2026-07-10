"""Pair selection and report artifact writing for household-scale comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .dynamic_macro_household_scale_validation import (
    COHORTS,
    EXPECTED_ORIGINS,
    EXPECTED_SCORE_ORIGINS,
    SURFACE_COLUMNS,
    DynamicMacroHouseholdScaleError,
    _sha256_file,
)


SCHEMA_VERSION = "dynamic_macro_household_scale_comparison_v1"
CLAIM_SCOPE = "developmental_retrospective_only"
OUTPUT_FILES = (
    "manifest.json",
    "comparison.csv",
    "family_scores.csv",
    "origin_scores.csv",
    "promotion_receipt.json",
    "report.md",
)


def _validate_run_pair(
    runs: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any]
) -> None:
    first = runs[COHORTS[0]]["joined"]
    second = runs[COHORTS[1]]["joined"]
    for column in SURFACE_COLUMNS:
        if column not in first.columns or column not in second.columns:
            raise DynamicMacroHouseholdScaleError(
                f"Score surface lacks locked column {column}"
            )
    left = _surface_frame(first)
    right = _surface_frame(second)
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, check_exact=False, rtol=0.0, atol=1e-12
        )
    except AssertionError as exc:
        raise DynamicMacroHouseholdScaleError(
            "Corrected 81 and 200 Jan-May score surfaces differ"
        ) from exc
    if (
        runs[COHORTS[0]]["mechanism_fingerprint"]
        != runs[COHORTS[1]]["mechanism_fingerprint"]
    ):
        raise DynamicMacroHouseholdScaleError("Corrected 81 and 200 mechanisms differ")
    if runs[COHORTS[0]]["lock_values"] != runs[COHORTS[1]]["lock_values"]:
        raise DynamicMacroHouseholdScaleError(
            "Corrected 81 and 200 source/prompt/model locks differ"
        )
    expected_origins = tuple(spec.get("origins", EXPECTED_ORIGINS))
    if (
        expected_origins != EXPECTED_ORIGINS
        or tuple(spec.get("score_origins", EXPECTED_SCORE_ORIGINS))
        != EXPECTED_SCORE_ORIGINS
    ):
        raise DynamicMacroHouseholdScaleError(
            "Comparison spec must lock January-May with February-May scored"
        )


def _build_result(
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_path: Path,
    spec_path: Path,
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    score81 = runs["corrected81"]["all_scores"]["llm"]
    score200 = runs["corrected200"]["all_scores"]["llm"]
    relative_gap = (
        math.inf
        if score81 == 0 and score200 != 0
        else (
            0.0 if score81 == score200 == 0 else abs(score200 - score81) / abs(score81)
        )
    )
    tolerance = float(spec.get("relative_gap_tolerance", 0.01))
    winner = (
        "corrected200"
        if score200 <= score81 or relative_gap <= tolerance
        else "corrected81"
    )
    comparison_rows = [_historical_comparison_row(evidence)]
    comparison_rows.extend(
        _comparison_row(runs[cohort], cohort, score81, score200, relative_gap, winner)
        for cohort in COHORTS
    )
    comparison = pd.DataFrame(comparison_rows)
    family = pd.concat(
        [_tag_scores(runs[c]["family_scores"], c) for c in COHORTS], ignore_index=True
    )
    origin = pd.concat(
        [_tag_scores(runs[c]["origin_scores"], c) for c in COHORTS], ignore_index=True
    )
    family = family.sort_values(
        ["cohort", "candidate", "family"], kind="mergesort"
    ).reset_index(drop=True)
    origin = origin.sort_values(
        ["cohort", "candidate", "origin_month"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_scope": CLAIM_SCOPE,
        "output_dir": "",
        "winner": winner,
        "decision_rule": {
            "metric": "absolute_llm_ALL_rmse_scaled",
            "lower_is_better": True,
            "relative_gap_tolerance": tolerance,
            "near_tie_preference": "corrected200",
            "adaptive_role": "diagnostic_only_not_a_selection_veto",
        },
        "spec_sha256": _sha256_file(spec_path),
        "historical_evidence_sha256": _sha256_file(evidence_path),
        "origins": list(EXPECTED_ORIGINS),
        "score_origins": list(EXPECTED_SCORE_ORIGINS),
        "runs": {
            cohort: {
                key: value
                for key, value in runs[cohort].items()
                if key
                in {
                    "run_dir",
                    "manifest_sha256",
                    "normalized_spec_sha256",
                    "output_hashes",
                    "household_count",
                    "raw_record_batch_counts",
                    "call_counts",
                    "effective_weighted_sample_size",
                    "max_normalized_household_weight",
                    "direction_accuracy",
                    "max_accounting_abs_residual",
                    "macro_scores",
                    "all_scores",
                    "mechanism_fingerprint",
                    "lock_values",
                }
            }
            for cohort in COHORTS
        },
        "recomputed_scores": {cohort: runs[cohort]["all_scores"] for cohort in COHORTS},
        "relative_llm_gap_corrected200_vs_corrected81": relative_gap,
        "output_contract": list(OUTPUT_FILES),
    }
    return {
        "manifest": manifest,
        "comparison": comparison,
        "family": family,
        "origin": origin,
        "receipt": {
            "winner": winner,
            "llm_all_rmse_scaled": {c: runs[c]["all_scores"]["llm"] for c in COHORTS},
            "relative_gap": relative_gap,
        },
    }


def _write_result(output_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison = result["comparison"]
    family = result["family"]
    origin = result["origin"]
    manifest = dict(result["manifest"])
    manifest["output_dir"] = str(output_dir)
    _write_csv(output_dir / "comparison.csv", comparison)
    _write_csv(output_dir / "family_scores.csv", family)
    _write_csv(output_dir / "origin_scores.csv", origin)
    report = _build_report(manifest, comparison, family, origin)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest["outputs"] = {
        name: _sha256_file(output_dir / name)
        for name in (
            "comparison.csv",
            "family_scores.csv",
            "origin_scores.csv",
            "report.md",
        )
    }
    receipt = dict(result["receipt"])
    receipt.update(
        {
            "schema_version": "dynamic_macro_household_scale_promotion_receipt_v1",
            "status": "complete",
            "claim_scope": CLAIM_SCOPE,
            "artifact_hashes": dict(manifest["outputs"]),
        }
    )
    _write_json(output_dir / "promotion_receipt.json", receipt)
    manifest["outputs"]["promotion_receipt.json"] = _sha256_file(
        output_dir / "promotion_receipt.json"
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _comparison_row(
    run: Mapping[str, Any],
    cohort: str,
    score81: float,
    score200: float,
    relative_gap: float,
    winner: str,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "result_role": "current_corrected_development_comparison",
        "eligible_for_promotion": True,
        "household_count": run["household_count"],
        "effective_weighted_sample_size": run["effective_weighted_sample_size"],
        "max_normalized_household_weight": run["max_normalized_household_weight"],
        "raw_record_batch_count": sum(run["raw_record_batch_counts"].values()),
        "llm_ALL_rmse_scaled": run["all_scores"]["llm"],
        "adaptive_ALL_rmse_scaled": run["all_scores"]["adaptive"],
        "llm_macro_score_recomputed": run["macro_scores"]["llm"],
        "adaptive_macro_score_recomputed": run["macro_scores"]["adaptive"],
        "llm_minus_adaptive_recomputed": run["macro_scores"]["llm"]
        - run["macro_scores"]["adaptive"],
        "llm_direction_accuracy": run["direction_accuracy"]["llm"],
        "adaptive_direction_accuracy": run["direction_accuracy"]["adaptive"],
        "max_accounting_abs_residual": run["max_accounting_abs_residual"],
        "live_call_count": run["call_counts"]["live"],
        "accepted_live_call_count": run["call_counts"]["accepted"],
        "failed_live_call_count": run["call_counts"]["failed"],
        "semantic_retry_count": run["call_counts"]["semantic_retries"],
        "replayed_record_count": run["call_counts"]["replayed"],
        "cache_hit_count": run["call_counts"]["cache_hits"],
        "relative_gap_corrected200_vs_corrected81": relative_gap,
        "selected": cohort == winner,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
    }


def _historical_comparison_row(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = evidence["historical_uncorrected81_context"]
    return {
        "cohort": "historical_uncorrected81",
        "result_role": "historical_context_only",
        "eligible_for_promotion": False,
        "household_count": int(context["household_count"]),
        "effective_weighted_sample_size": float(
            context["effective_weighted_sample_size"]
        ),
        "max_normalized_household_weight": float(
            context["max_normalized_household_weight"]
        ),
        "raw_record_batch_count": int(context["replayed_record_count"]),
        "llm_ALL_rmse_scaled": float(context["llm_macro_score"]),
        "adaptive_ALL_rmse_scaled": float(context["adaptive_macro_score"]),
        "llm_macro_score_recomputed": float(context["llm_macro_score"]),
        "adaptive_macro_score_recomputed": float(context["adaptive_macro_score"]),
        "llm_minus_adaptive_recomputed": float(context["llm_macro_score"])
        - float(context["adaptive_macro_score"]),
        "llm_direction_accuracy": float(context["llm_direction_accuracy"]),
        "adaptive_direction_accuracy": float(context["adaptive_direction_accuracy"]),
        "max_accounting_abs_residual": float(context["max_accounting_abs_residual"]),
        "live_call_count": int(context["live_call_count"]),
        "accepted_live_call_count": 0,
        "failed_live_call_count": 0,
        "semantic_retry_count": int(context["semantic_retry_count"]),
        "replayed_record_count": int(context["replayed_record_count"]),
        "cache_hit_count": int(context["cache_hit_count"]),
        "relative_gap_corrected200_vs_corrected81": math.nan,
        "selected": False,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
    }


def _tag_scores(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "cohort", cohort)
    return result


def _surface_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[list(SURFACE_COLUMNS)].copy()
    return result.sort_values(list(SURFACE_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _build_report(
    manifest: Mapping[str, Any],
    comparison: pd.DataFrame,
    family: pd.DataFrame,
    origin: pd.DataFrame,
) -> str:
    rows = [
        "# Corrected Household-Scale Comparison",
        "",
        f"Claim scope: `{manifest['claim_scope']}`.",
        "",
        f"Winner: `{manifest['winner']}`. The rule minimizes absolute LLM `ALL.rmse_scaled`; a relative gap at or below 1% prefers corrected200.",
        "",
        "Adaptive scores are diagnostic only and do not veto promotion.",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Integrity",
        "",
        "- Both runs reproduce MacroScore from `joined_errors.csv`.",
        "- The Jan-May surface and mechanisms match; only cohort-specific household and batch fields differ.",
        "- June rows and accounting violations were rejected by the validator.",
        "- Historical uncorrected81 evidence is context only and is not eligible for promotion.",
        "",
        f"Family rows: `{len(family)}`; origin rows: `{len(origin)}`.",
        "",
    ]
    return "\n".join(rows)
