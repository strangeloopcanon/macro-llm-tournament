# Empirical Bridge v5 Stabilized Spec

## Purpose

v5 is a stabilization revision after the v4 diagnostics showed weak between-wave identification. It is not a quiet refit and it does not spend a new confirmatory surface. The goal is to ask whether a locked shrinkage estimator makes the real-data belief-to-spending bridge less brittle before Phase 4 scoring is revisited.

## Locked Estimator

The outcome remains respondent-level real expected spending growth:

```text
100 * ((1 + expected_total_spending_growth_pct / 100)
       / (1 + actual_expected_inflation_1y / 100) - 1)
```

The panel, FIT waves, validation waves, regressors, controls, winsor bounds, chart-quality check, coefficient bounds, and liquidity-gradient check are the same bridge surface used by v4.

The estimator changes from OLS to standardized ridge regression:

- Alpha grid: `0.1, 0.3, 1, 3, 10, 30, 100, 300`.
- Selection surface: FIT waves only.
- Cross-validation: leave one FIT spending wave out.
- Selection rule: choose the largest alpha within one standard error of the minimum weighted RMSE.
- Downstream Phase 4 scores are not used to choose alpha or bridge coefficients.

## Acceptance Rules

v5 is accepted only if the hard v4 gates still pass:

- real outcome transform present,
- coefficient bounds pass,
- public-chart reproduction stays within the locked v4 tolerance,
- liquidity-gradient sanity check passes,
- no Phase 4 scoring is used during fitting.

The validation diagnostic remains diagnostic, not an automatic blocker: it reports cell RMSE generalization and validation-wave coefficient sign/magnitude stability. A validation failure means the bridge is exploratory, not confirmatory.

## Recorded Result

The fitted artifact is `work/empirical_bridge/empirical_bridge_v5_stabilized.json`.

- Status: `accepted`.
- Estimator: `ridge_cv`.
- Selected alpha: `3.0`.
- FIT CV minimum alpha: `0.3`.
- Selection: largest alpha within the one-standard-error threshold.

Between-wave coefficients:

| Belief regressor | v5 coefficient |
| --- | ---: |
| Expected inflation | `+0.0781` |
| Expected real income growth | `+0.2101` |
| Unemployment-higher probability | `-0.0173` per pp |

The validation diagnostic still fails:

- Validation cell RMSE ratio: `1.5852`, above the `1.5` diagnostic band.
- Validation inflation coefficient flips sign.
- Validation real-income and unemployment coefficients are much larger than the FIT coefficients.

So v5 is usable as a stabilized exploratory bridge, but not as a validated final mechanism.

## Phase 4 Use

Any Phase 4 run using v5 must pass the bridge path explicitly with:

```text
--empirical-bridge-json work/empirical_bridge/empirical_bridge_v5_stabilized.json
```

All v5 Phase 4 runs in the current report are labeled retrospective. Confirmatory scoring under this bridge requires a future or otherwise unspent score surface, with the bridge and mapping locked before scoring.
