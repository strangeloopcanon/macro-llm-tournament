# Household-First Rolling Microeconomy

## Bottom Line

This is the first complete **200-household** forecast from the new ecology. Each household was elicited separately; deterministic institutions then reconciled their choices into production, employment, inventories, credit, and settlement.

The median path predicts consumption growth of **-4.08%**. Households themselves intended **-2.95%**; the additional contraction comes from mandatory debt service and binding household resources, not an arbitrary aggregate gain.

This is a frozen forecast, not an accuracy claim. August 2026 outcomes were not used and are not yet scored. A separate four-origin retrospective diagnostic now shows that the same mechanism is severely miscalibrated in sign and scale.

## Architecture

```text
survey-seeded household state + own history + as-of public information
                              |
                    one isolated LLM call
                              |
                 beliefs and intended choices
                              |
          deterministic budgets, credit, and production
                              |
       household actions -> employer -> macro state -> next origin
```

## Run Facts

- Forecast origin: `2026-07-01`; information cutoff: `2026-07-10`.
- One-month-ahead target: `2026-08-01`; status: `prospective_frozen`.
- Provider/model: `codex_cli` / `gpt-5.5`.
- Accepted household responses: `200`; provider-created records represented: `200`.
- Replay cache hits in this execution: `200`; fresh calls: `0`.
- Accounting: **PASS**; maximum residual `4.19e-09`.
- Origin snapshot hash: `7ccc2bbf1d378faec01cbb021a3af82de5af85094a0cd195da6bb0312a610981`.
- Executable source hash: `0cb0d80abba203f5322cf19d5f747630bb7ff34aee41bc943f0156afbb043aca`.
- Replay equivalence: `f935182686559b8140e7f03bfc1755f9bda05ded4e0517167a74ba6a81f1f674`; verified against an expected hash: `True`.

## Forecast Paths

| Scenario | Consumption growth | Saving rate | Revolving-credit growth | Employment rate | Price growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| downside | -8.14% | -2.53% | -6.03% | 90.96% | -0.99% |
| median | -4.08% | -1.33% | -6.37% | 90.70% | -1.35% |
| upside | -0.73% | 0.48% | -6.63% | 90.70% | -1.93% |

## Household Signal

Population-weighted median beliefs imply inflation of **4.89%**, income growth of **-1.01%**, and a **9.99%** one-year job-loss probability.

The median economy executes `$1,016,637` of consumption, `$169,080` of debt payments, and `$2,750` of new borrowing across the 200 simulated household units.

The cross-section is not a repeated representative household: low-liquidity agents plan larger consumption cuts, beliefs vary materially, and balance-sheet constraints alter feasible actions household by household.

## What This Establishes

The system is now a branchable, recursive microeconomy rather than an aggregate demand identity. Production comes from labor and capacity; sales clear against production plus inventories; employment and vacancies are explicit; loans, payments, deposits, and defaults have counterparties; and every scenario emits the exact state used by the next origin.

The cohort is initialized from March-April 2025 SCE observations, the latest common two-wave panel used here, while the public macro card is current to the forecast cutoff. Household balance sheets are coarse survey mappings rather than contemporaneous measured accounts.

## Retrospective Diagnostic

The current mechanism was run recursively from January through April 2026, producing one-month-ahead forecasts for February through May. Each origin used 200 isolated GPT-5.5 household calls through Codex CLI. These origins are after GPT-5.5's December 2025 cutoff. The median simulated state, not realized outcomes, was carried into the next origin; first-release outcomes were loaded only after all four forecasts finished.

| Metric | February prediction | May prediction | Median-path RMSE | Direction accuracy |
| --- | ---: | ---: | ---: | ---: |
| Nominal consumption growth | -4.05% | -8.62% | 7.10 pp | 0% |
| PCE price growth | -1.34% | -5.68% | 4.33 pp | 0% |
| Revolving-credit growth | -6.51% | -9.29% | 7.47 pp | 25% |
| Saving-rate proxy | -1.60% | 1.78% | 2.89 pp | 75% |
| Employment-rate proxy | 90.70% | 90.70% | 4.97 pp | 100% |

The result is unambiguous: the ecology predicts contraction and deflation in every historical month, while first-release nominal PCE and prices rise. Recursive feedback amplifies the error over time. Accounting passes across all four runs, so feasibility is not the problem; the household-intention-to-macro transition and its feedback scale are.

Consumption is the closest aggregate mapping. Credit, saving, employment, and prices remain directional proxies: saving uses the raw first-release PSAVERT level, employment uses `100 - UNRATE`, and the ecology's unit price and debt stock do not exactly equal PCEPI and REVOLSL.

The full diagnostic contract is under `outputs/household_ecology_retrospective_2026_01_04/`. It is retrospective developmental evidence, not confirmation. The frozen August forecast remains untouched.

## Current Verdict

The system is a real, accounting-safe simulated microeconomy, but it is not yet a credible macro predictor. The next development task is to identify and correct the contractionary feedback and scale error on these historical origins. More institutional complexity should wait until that simpler failure is understood.
