# Empirical Bridge v4 Specification

Status: LOCKED before v4 fitting. Empirical bridge v3 is preserved as a
rejected specification. This revision fixes the v3 units error by changing the
outcome from nominal expected spending growth to respondent-level real expected
spending growth. The split definitions, estimator, bounds, and downstream
Phase 4 use below require a new version bump before they can change.

## What v3 Taught Us

v3 joined the SCE Household Spending Survey to the core SCE belief panel and
fit the planned Mundlak bridge, but it was rejected by its own gates. The
load-bearing failure was the inflation coefficient: expected inflation raised
nominal expected spending growth by about one-for-one. That is not a real
consumption response; it is expected price pass-through. v4 therefore deflates
the spending-growth outcome instead of relaxing the real-consumption bounds.

The public spending workbook still lacks native person weights. The chart
reproduction check remains reported, but v4 treats the known weight-substitution
mismatch as a provenance warning unless max absolute chart error exceeds
`0.35` percentage points. The v3 `0.10` percentage-point tolerance remains the
ideal standard for a fully weighted public extract, not a gate for this derived
bridge.

## Data

- Joined panel: `work/empirical_bridge/spending_belief_panel.csv`.
- Coverage report: `work/empirical_bridge/spending_belief_panel_coverage.json`.
- Source files and hashes remain recorded in `data_provenance/data_events.jsonl`.
- No Phase 4 score surface is consumed unless the v4 bridge artifact is
  `status: accepted`.

## Outcome

Primary fitted outcome:

```text
expected_real_total_spending_growth_pct =
    100 * ((1 + expected_total_spending_growth_pct / 100)
           / (1 + actual_expected_inflation_1y / 100) - 1)
```

The nominal outcome is retained in the artifact for provenance. Winsor bounds
are computed from FIT waves only on the real outcome and then applied unchanged
to internal-check and validation waves.

## Regressors

Same as v3, from the respondent's most recent core SCE wave within three
calendar months before the spending wave:

- `actual_expected_inflation_1y`
- `actual_expected_real_income_growth`
- `sce_question_unemployment_higher_prob`

Controls remain `income_group`, `liquid_wealth_group`, and `age_group`.

## Estimator

Same as v3: Mundlak / correlated-random-effects regression. Each belief
regressor enters as a wave mean (between component) plus within-wave deviations
interacted with `liquid_wealth_group x income_group`.

The Phase 4 executor uses only the between coefficients as the aggregate
belief-to-real-consumption-growth bridge. Within-cell slopes are reported and
used for sign/gradient diagnostics.

## Splits

- FIT: `202004, 202008, 202012, 202104, 202108, 202112, 202204, 202208,
  202304, 202308`.
- INTERNAL CHECK: `202212`.
- VALIDATION: `202312, 202404, 202408`.
- CONFIRMATORY: spending waves released after this lock. None are spent by
  v4 fitting.

## Fail-Closed Bounds

Bounds are on between coefficients, measured as percentage points of annual
real expected spending growth per percentage point of belief change:

- Expected inflation: `[-0.30, +0.30]`.
- Expected real income growth: `[0.00, +0.50]`.
- Unemployment-higher probability: `[-0.40, 0.00]` per 10 percentage points,
  equivalently `[-0.04, 0.00]` per percentage point.

The fit is unconstrained. If any coefficient violates its bound, or if the
within-cell liquidity-gradient audit fails, v4 is rejected. No refit, clipping,
or coefficient tuning is allowed inside v4.

## Phase 4 Use

If accepted, `behavior_policy_mode=empirical_bridge` loads
`work/empirical_bridge/empirical_bridge_v4.json`. The bridge maps belief
changes to the consumption-growth margin only. Deterministic code still owns
budget feasibility, saving/debt closure, accounting, and market clearing. Both
matched twins use the same bridge; only their belief-updating source differs.
The SCE spending outcome is annual expected growth; the current demand economy
uses quarterly household income and consumption states, so the executor divides
the annual bridge deviation by `4` before applying it to one simulated period.
Both the annual and per-period deviations are emitted in household decisions.

Run strict one-card and hold-last five-card Phase 4 replays with mapping schema
v2, `scoring_label: retrospective`, zero live calls, and explicit manifest
fields for the bridge version and hash.

## Success Criteria

The first v4 attempt is an improvement only if all of the following hold:

1. The v4 artifact is accepted by its fail-closed bridge gates.
2. The strict LLM-updater scaled RMSE is below the current state-schedule
   strict baseline `5.0843`.
3. The hold-last LLM-updater scaled RMSE is below the current state-schedule
   hold-last baseline `3.2132`.
4. The LLM-minus-adaptive gap shrinks versus the current state-schedule gaps:
   strict `3.7372`, hold-last `1.4842`.
5. The adaptive twin does not degrade by more than 10 percent versus its own
   state-schedule runs.
6. Accounting residuals pass in both replays.

A win over adaptive is claimed only if the v4 LLM-updater scaled RMSE is
strictly below the v4 adaptive scaled RMSE.

## Live-Call Budget

Zero. v4 uses already-banked belief and behavior outputs.
