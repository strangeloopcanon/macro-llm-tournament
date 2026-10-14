#!/usr/bin/env python3
"""Build the portable macro forecast panel and render its scorecards."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "reports/macro_forecast_panel.csv"
METRICS = [
    ("consumption_growth_pct", "Consumption growth", "% m/m"),
    ("real_consumption_growth_pct", "Real consumption growth", "% m/m"),
    ("retail_sales_growth_pct", "Retail sales growth", "% m/m"),
    ("real_disposable_income_growth_pct", "Real disposable income growth", "% m/m"),
    ("payroll_growth_pct", "Payroll growth", "% m/m"),
    ("unemployment_rate_level", "Unemployment rate", "%"),
    ("price_growth_pct", "Price growth", "% m/m"),
    ("personal_saving_rate_change_pp", "Personal saving rate change", "percentage points"),
    ("revolving_credit_growth_pct", "Revolving credit growth", "direction only"),
]
SERIES = {
    "llm_household_economy": ("Household forecast", "#c43c39", "o"),
    "event_jepa_residual": ("JEPA forecast", "#27834a", "s"),
    "first_release_actual": ("First-release actual", "#2864a5", "D"),
    "origin_visible_baseline": ("Visible baseline", "#707780", "^"),
}


def build_panel() -> pd.DataFrame:
    sources = [
        (ROOT / "work/macro_jepa_v3/blocked_backtest_predictions.csv", "blocked"),
        (ROOT / "work/macro_jepa_v3/comparison_predictions.csv", "comparison"),
    ]
    frames = []
    for path, _ in sources:
        df = pd.read_csv(path)
        frames.append(df[["origin_month", "target_month", "metric", "value", "source", "evaluation_status"]])
    panel = pd.concat(frames, ignore_index=True)
    aug = pd.read_csv(ROOT / "reports/august_2026_realizations.csv")
    aug_rows = pd.DataFrame({
        "origin_month": "",
        "target_month": aug.target_month,
        "metric": aug.metric,
        "value": aug.value,
        "source": "first_release_actual",
        "evaluation_status": "frozen_august_2026_realization",
    })
    panel = pd.concat([panel, aug_rows], ignore_index=True)
    # The August forecasts have since become scoreable for the four observed measures.
    observed = set(zip(aug.target_month, aug.metric))
    is_aug_forecast = (panel.target_month == "2026-08-01") & panel.source.isin(
        ["llm_household_economy", "origin_visible_baseline"]
    )
    panel.loc[is_aug_forecast, "evaluation_status"] = panel.loc[is_aug_forecast].apply(
        lambda r: "scored_frozen" if (r.target_month, r.metric) in observed else "prospective_frozen_unscored",
        axis=1,
    )
    panel["value"] = pd.to_numeric(panel.value, errors="raise")
    keys = ["target_month", "metric", "source"]
    duplicated = panel[panel.duplicated(keys, keep=False)]
    if not duplicated.empty:
        raise ValueError(f"duplicate forecast keys:\n{duplicated[keys].to_string(index=False)}")
    panel = panel.sort_values(["target_month", "metric", "source"]).reset_index(drop=True)
    return panel


def _series(ax, data: pd.DataFrame, metric: str, source: str, start: str | None = None):
    d = data[(data.metric == metric) & (data.source == source)].copy()
    if start:
        d = d[d.target_month >= start]
    if d.empty:
        return
    d["date"] = pd.to_datetime(d.target_month)
    # Insert explicit NaN months so matplotlib never bridges unavailable intervals.
    d = d.set_index("date").sort_index()
    full = pd.date_range(d.index.min(), d.index.max(), freq="MS")
    d = d.reindex(full)
    label, color, marker = SERIES[source]
    status = d.evaluation_status.fillna("")
    scored = status.isin(["blocked_backtest_2024_2025", "retrospective_diagnostic_not_confirmatory", "scored_frozen", "frozen_august_2026_realization"])
    values = np.sign(d.value) if metric == "revolving_credit_growth_pct" else d.value
    ax.plot(d.index, values.where(scored), color=color, marker=marker, markersize=3.5, linewidth=1.15,
            label=label, zorder=3)
    hollow = status.isin(["prospective_frozen_unscored"])
    ax.scatter(d.index[hollow], values[hollow], marker=marker, s=27, facecolors="white", edgecolors=color,
               linewidths=1.2, zorder=5)


def render(panel: pd.DataFrame, suffix: str, start: str | None):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10.5 if not start else 10.2), sharex=True)
    fig.patch.set_facecolor("#fbfbf8")
    for ax, (metric, title, unit) in zip(axes.flat, METRICS):
        ax.set_facecolor("#fbfbf8")
        for source in SERIES:
            _series(ax, panel, metric, source, start)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=7)
        ax.text(1, 1.015, unit, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#59616a")
        ax.grid(axis="y", color="#dfe2e2", linewidth=.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b8bdc0")
        ax.tick_params(axis="y", labelsize=8, length=0, pad=3)
        ax.tick_params(axis="x", labelsize=8, length=0, pad=5)
        if metric == "revolving_credit_growth_pct":
            ax.set_yticks([-1, 0, 1], ["Contract", "No change", "Expand"])
            ax.set_ylim(-1.25, 1.25)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%YQ"))
    # Use explicit quarter labels; retain monthly observations without aggregation.
    for ax in axes[-1, :]:
        ticks = ax.get_xticks()
        labels = []
        for tick in ticks:
            dt = mdates.num2date(tick)
            q = (dt.month - 1) // 3 + 1
            labels.append(f"{dt.year}Q{q}")
        ax.set_xticks(ticks, labels, rotation=35, ha="right", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), ncol=4, frameon=False,
               fontsize=9, handlelength=2.4, columnspacing=2)
    title = "Monthly macro forecasts and first-release outcomes"
    if start:
        title = "2026 macro forecasts and first-release outcomes"
    fig.suptitle(title, x=.055, y=.995, ha="left", fontsize=17, fontweight="bold", color="#17232c")
    foot = ("Monthly values; quarter labels are navigation only. Jan 2024–Dec 2025 origins: held-out blocked backtest.\n"
            "2026 retrospective values are development evidence; August forecasts are frozen and paired with first releases.\n"
            "Hollow August markers are unscored; no JEPA August forecast exists. No uncertainty bands were saved.\n"
            "Household outputs retain their declared proxy mappings; nominal spending is primary. Credit shows direction only.")
    fig.text(.055, .012, foot, ha="left", va="bottom", fontsize=8, color="#59616a", linespacing=1.35)
    fig.subplots_adjust(left=.055, right=.99, top=.89, bottom=.15 if not start else .17, hspace=.52, wspace=.18)
    out = ROOT / f"reports/macro_forecast_scorecard{suffix}"
    fig.savefig(out.with_suffix(".png"), dpi=190, facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-panel", action="store_true", help="rebuild portable panel from saved source CSVs")
    args = parser.parse_args()
    if args.build_panel:
        build_panel().to_csv(PANEL, index=False)
    panel = pd.read_csv(PANEL)
    render(panel, "", None)
    render(panel, "_2026", "2026-01-01")
    print(f"panel_rows={len(panel)} metrics={panel.metric.nunique()} sources={panel.source.nunique()}")
    print(panel.groupby(["source", "evaluation_status"]).size().to_string())


if __name__ == "__main__":
    main()
