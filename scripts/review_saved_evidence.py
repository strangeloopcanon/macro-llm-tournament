"""Verify and summarize the recovered run without training or model calls.

Run from the repository root with PYTHONPATH=src.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from macro_llm_tournament.ecology_realizations import _verify_original_artifacts
from macro_llm_tournament.macro_jepa import _score_predictions

def review_portable_reports(reports: Path) -> None:
    panel = pd.read_csv(reports / "macro_forecast_panel.csv")
    for filename, status in (
        ("jepa_heldout_scores.csv", "blocked_backtest_2024_2025"),
        ("jepa_2026_scores.csv", "retrospective_diagnostic_not_confirmatory"),
    ):
        saved = pd.read_csv(reports / filename)
        recomputed = _score_predictions(panel, sources=saved.source.unique().tolist(), evaluation_status=status)
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False, rtol=1e-10, atol=1e-10)
    scores = pd.read_csv(reports / "august_2026_scores.csv")
    actuals = pd.read_csv(reports / "august_2026_realizations.csv").set_index("metric")["value"]
    for row in scores.itertuples(index=False):
        prediction = panel.loc[(panel.target_month == row.target_month) & (panel.metric == row.metric) & (panel.source == row.source), "value"]
        if len(prediction) != 1:
            raise ValueError("Missing or duplicate August chart prediction")
        np.testing.assert_allclose([row.forecast, row.actual, row.error, row.absolute_error],
                                   [prediction.iloc[0], actuals[row.metric], row.actual - row.forecast, abs(row.actual - row.forecast)])
    print("Portable reports reproduce all 99 JEPA score rows and four August targets. Raw checkpoint/input verification requires restoring the Drive archives.")


def main() -> None:
    root = Path("work/macro_jepa_v3")
    reports = Path("reports")
    if not root.is_dir():
        review_portable_reports(reports)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        with (root / relative).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Recovered artifact differs: {relative}")
    for predictions, scores, status, destination in (
        ("comparison_predictions.csv", "score_table.csv", "retrospective_diagnostic_not_confirmatory", "jepa_2026_scores.csv"),
        ("blocked_backtest_predictions.csv", "blocked_backtest_score_table.csv", "blocked_backtest_2024_2025", "jepa_heldout_scores.csv"),
    ):
        saved = pd.read_csv(root / scores)
        recomputed = _score_predictions(
            pd.read_csv(root / predictions), sources=saved.source.unique().tolist(),
            evaluation_status=status,
        )
        joined = saved.merge(recomputed, on=["metric", "source", "score_mode"], suffixes=("_saved", "_recomputed"), validate="one_to_one")
        if len(joined) != len(saved) or len(recomputed) != len(saved):
            raise ValueError("Score coverage differs")
        for column in ("n", "rmse", "mae", "direction_accuracy"):
            np.testing.assert_allclose(joined[column + "_saved"], joined[column + "_recomputed"], rtol=1e-10, atol=1e-10)
        recomputed.to_csv(reports / destination, index=False)

    run = Path("outputs/household_ecology_200_july_v24_current")
    _verify_original_artifacts(run, json.loads((run / "manifest.json").read_text()))
    append = run / "realization_append"
    receipt = json.loads((append / "realization_manifest.json").read_text())
    for relative, expected in receipt["output_artifacts_sha256"].items():
        with (append / relative).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Realization artifact differs: {relative}")
    errors = pd.read_csv(append / "forecast_errors.csv").iloc[0]
    rows = []
    for record in receipt["normalized_realization_rows"]:
        metric = record["metric"]
        for source, column in (("llm_household_economy", "forecast_"), ("origin_visible_baseline", "visible_baseline_")):
            prediction = float(errors[column + metric])
            actual = float(record["value"])
            rows.append(dict(target_month=record["target_month"], metric=metric, source=source, forecast=prediction, actual=actual, error=actual - prediction, absolute_error=abs(actual - prediction)))
    scores = pd.DataFrame(rows)
    scores.to_csv(reports / "august_2026_scores.csv", index=False)

    frame = pd.read_csv(root / "comparison_predictions.csv")
    nominal = frame.loc[frame.metric.eq("consumption_growth_pct")]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    styles = {
        "llm_household_economy": ("LLM households", "#B23A2B", "s", "-"),
        "event_jepa_residual": ("Event-JEPA", "#3B7D5A", "D", "-"),
        "origin_visible_baseline": ("Frozen baseline", "#6B6B6B", "^", "--"),
        "first_release_actual": ("First-release actual", "#2463A6", "o", "-"),
    }
    for source, (label, color, marker, line) in styles.items():
        history = nominal.loc[nominal.source.eq(source) & nominal.evaluation_status.eq("retrospective_diagnostic_not_confirmatory")].sort_values("target_month")
        axis.plot(pd.to_datetime(history.target_month), history.value, label=label, color=color, marker=marker, linestyle=line)
        if source == "event_jepa_residual":
            continue
        value = 0.9 if source == "first_release_actual" else float(scores.loc[scores.metric.eq("consumption_growth_pct") & scores.source.eq(source), "forecast"].iloc[0])
        axis.scatter(pd.Timestamp("2026-08-01"), value, color=color, marker=marker, s=65)
        axis.annotate(f"{value:+.3f}%", (pd.Timestamp("2026-08-01"), value), xytext=(12, -3), textcoords="offset points", color=color)
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel("Monthly nominal spending growth (%)")
    axis.set_title("August spending rose; the frozen household forecast predicted a fall", loc="left", pad=20)
    axis.set_xlim(pd.Timestamp("2026-01-20"), pd.Timestamp("2026-09-10"))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.text(0.08, 0.025, "Feb–May: retrospective diagnostics. August: previously frozen forecasts; BEA first release, Sept 30 (rounded to 0.1%).\nNo saved JEPA August forecast. Lines stop in May; no forecasts are invented for June or July.", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.86, bottom=0.25)
    fig.savefig(reports / "august_2026_consumption.png", dpi=180)
    plt.close(fig)
    print(f"Verified {len(manifest['artifacts'])} recovered artifacts, both JEPA score tables, and frozen household artifacts; scored {len(scores) // 2} August targets.")


if __name__ == "__main__":
    main()
