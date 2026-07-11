# Household-First Rolling Microeconomy

## Bottom Line

This is the first complete **200-household** forecast from the new ecology. Each household was elicited separately; deterministic institutions then reconciled their choices into production, employment, inventories, credit, and settlement.

The median path predicts consumption growth of **-4.08%**. Households themselves intended **-2.95%**; the additional contraction comes from mandatory debt service and binding household resources, not an arbitrary aggregate gain.

This is a frozen forecast, not an accuracy claim. August 2026 outcomes were not used and are not yet scored.

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

It does not yet establish predictive accuracy. Employment is still governed by one aggregate employer, and one untouched forecast origin cannot validate dynamics. The next evidence comes from appending realized August 2026 outcomes and repeating the same frozen procedure over several new months.
