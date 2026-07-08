# Empirical Bridge v3 Specification (revision 2, post red-team)

Status: LOCKED 2026-07-08. Revision 1 was red-teamed by an independent
Opus 4.8 review before any fitting or scoring; all four blockers and the
should-fix items were resolved by this revision. The split definitions,
estimator, constraints, and success criteria below may not change without a
version bump to v4. No fitting, validation, or scoring ran before this lock.

## Problem

The Phase 4 matched-twin economy loses to the adaptive-expectations twin
because the hand-authored belief-to-behavior bridge transmits belief
pessimism into a consumption contraction (~ -2% MoM) that real data does not
show. The empirical literature says the transmission is weak and
category-split (Coibion-Georgarakos-Gorodnichenko-van Rooij RCT: modest
positive nondurable response to higher expected inflation, negative durable
response; Bank of Canada SWP 2024-44: ~74% report no spending change;
D'Acunto-Hoang-Weber VAT experiment: +8pp durable readiness). The fix:
estimate the bridge from matched respondent-level data instead of authoring
it, with literature bounds enforced fail-closed.

## Data

- Spending outcomes: `work/spending_survey_raw/sce-household-spending-microdata.xlsx`
  (sha256 in `data_provenance/data_events.jsonl`), SCE Household Spending
  Survey, 30 waves 2014-12 to 2024-08, ~1,000 respondents/wave, fielded
  April/August/December. Question-code mapping comes from
  `sce_household-spending-questionnaire.pdf` + glossary and must be verified
  by reproducing the published chart-data wave medians within +/-0.1pp
  (absolute) for expected and reported total spending growth.
- Beliefs: `work/persona_beliefs/sce_real_microdata.csv` (core SCE panel,
  waves 2020-01 to 2025-08). Join key: `sce_raw_userid` = spending `userid`.
  Verified overlap: >= 950 of ~970 spending respondents per 2023-2024 wave.

## Variables (exact, no discretion)

Outcome (primary): expected year-ahead total household spending growth from
the spending survey (the forward-looking percent question), winsorized at
the 2nd/98th percentile bounds computed on FIT waves only and applied
unchanged to all later waves. This is forward-aligned with belief
measurement. Realized next-wave reported spending change is computed and
reported as a descriptive robustness column, never fitted on.

Regressors, from the respondent's most recent core wave dated within 3
calendar months before the spending wave (rows with no such wave are
dropped; the lag distribution is reported):

- `actual_expected_inflation_1y` (percent)
- `actual_expected_real_income_growth` (percent)
- `sce_question_unemployment_higher_prob` (0-100 probability, raw SCE scale)

Controls: `income_group`, `liquid_wealth_group`, `age_group`. SCE person
`weight` is used in the fit, in cell targets, and in economy aggregation —
the same weighting everywhere.

## Estimator (resolves the wave-FE identification blocker)

Mundlak / correlated-random-effects regression: each belief regressor enters
twice — as its wave mean (between component) and as its within-wave
deviation (within component). No wave fixed effects.

- The BETWEEN coefficients are the aggregate transmission the economy uses.
- The WITHIN coefficients (interacted with `liquid_wealth_group` x
  `income_group` cells) supply cross-sectional heterogeneity.
- The between-vs-within gap is reported for every regressor.

## Dynamics (resolves the runaway-contraction blocker)

The bridge maps belief CHANGES, not levels, into consumption-growth
deviations: at each simulated period, response = between_coef x (mean belief
update this period), so constant beliefs imply zero growth deviation and the
path is stationary. A test must assert that a constant belief input produces
cumulative consumption drift < 0.01% over 12 periods.

## Fit / validation / confirmatory splits (resolves the double-use blocker)

- FIT: spending waves 2020-04 through 2023-08 (the earliest waves with core
  belief coverage), EXCLUDING 2022-12.
- INTERNAL CHECK: 2022-12 (mid-sample temporal holdout, one-shot, reported).
- VALIDATION: 2023-12, 2024-04, 2024-08 — strictly one-shot, read-only.
  Pass criteria fixed now: every VALIDATION between-coefficient sign matches
  FIT, and magnitudes lie within [0.5x, 2.0x] of FIT; weighted cell-level
  out-of-sample RMSE <= 1.5x in-sample. The `--validate` CLI must refuse to
  run unless a locked fit artifact already exists.
- SELECTION SURFACE: cross-sectional cell targets are built from FIT waves
  only. VALIDATION cells are scored once with the locked bridge and then
  frozen.
- CONFIRMATORY (sealed): any spending wave released after this lock
  (2024-12 onward under the 18-month lag). Never fitted or selected on.
- RESERVED FAMILY: `behavior_holdout_spending_windfall_v1` (the 10%-income
  windfall allocation moments). Built now, flagged unscoreable by default,
  spent only as a confirmatory family for a locked mechanism.

## Constraints (fail-closed, on BETWEEN coefficients, per pp of belief change
to pp of annual spending growth)

1. Expected inflation: [-0.10, +0.30].
2. Expected real income growth: [0.00, +0.50].
3. Unemployment-higher probability (per 10pp): [-0.40, 0.00].

Procedure: the fit is UNCONSTRAINED and reported as estimated. If any
between coefficient falls outside its bound or a within-cell slope pattern
reverses the liquidity gradient sign from the behavior-gate evidence, v3 is
REJECTED and reported as a negative. No clipping, no refitting within v3.

## Execution-scale parity (resolves the units mismatch)

Fitted regressors map to economy inputs with identical units:

| fitted regressor | economy input | scale |
| --- | --- | --- |
| `actual_expected_inflation_1y` | household `inflation_expectation_1y` | percent, 1:1 |
| `actual_expected_real_income_growth` | household expected real income growth | percent, 1:1 |
| `sce_question_unemployment_higher_prob` | raw unemployment-higher prob (BEFORE the 0.24 job-loss adapter) | 0-100, 1:1 |

An executor test must assert parity by feeding a known belief vector through
both the fit transform and the executor transform. Both twins' belief inputs
are clipped to the fitted support (min/max of FIT regressor values),
identically, and the share of clipped inputs per twin is reported in the
manifest.

## Phase 4 integration

- New `behavior_policy_mode` value `empirical_bridge` in the matched-twins
  runner (mutually exclusive with existing modes; no separate flag).
- The bridge moves only the consumption-growth margin; saving/debt margins
  keep the existing accounting closure. Applied identically to both twins;
  only belief inputs differ.
- Manifest fields: `bridge_spec_version: empirical_bridge_v3`,
  `empirical_bridge_sha256`, clipped-input shares, and the FIT/VALIDATION
  wave lists.
- Rerun strict one-card and hold-last five-card FRED replays, mapping v2
  unchanged, zero live LLM calls.

## Success criteria (fixed numerics, resolves gameability)

Baselines from the state-schedule run: strict LLM `5.0843` / adaptive
`1.3471` (gap `3.7372`); hold-last LLM `3.2132` / adaptive `1.7290`
(gap `1.4842`). v3 is an improvement only if ALL hold:

1. v3 LLM-updater scaled RMSE < `5.0843` (strict) AND < `3.2132` (hold-last).
2. v3 gap (LLM minus adaptive, both under v3) < `3.7372` (strict) AND
   < `1.4842` (hold-last).
3. v3 adaptive scaled RMSE does not degrade by more than 10% versus its own
   run under the previous executor (guard against winning by breaking the
   control).
4. Accounting residuals pass in both runs.

Anything else is reported as a negative. A win over adaptive is not claimed
unless the v3 LLM RMSE is strictly below the v3 adaptive RMSE.

## Live-call budget

Zero. Everything is deterministic given already-banked LLM outputs.

## Deliverables

1. `src/macro_llm_tournament/prepare_spending_survey.py`: parse microdata,
   map question codes, verify against chart data, join to core panel, emit
   `work/empirical_bridge/spending_belief_panel.csv` + coverage report.
2. `src/macro_llm_tournament/empirical_bridge.py`: Mundlak fit on FIT waves,
   internal-check and validation scoring, fail-closed bound enforcement,
   serialization to `work/empirical_bridge/empirical_bridge_v3.json` with
   sha256.
3. Phase 4 `behavior_policy_mode=empirical_bridge` integration + parity and
   stationarity tests.
4. FIT-wave cross-sectional cell targets; VALIDATION scored once and frozen.
5. Reserved `behavior_holdout_spending_windfall_v1` rows (unscoreable by
   default).
6. Tests: chart-data reproduction within +/-0.1pp; join integrity;
   winsor-bounds-from-FIT-only; constraint fail-closed; twin parity;
   constant-belief stationarity; zero-live-call guarantee.
7. Rerun `data_provenance` after new derived files exist.
