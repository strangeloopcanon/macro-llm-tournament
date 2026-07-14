# Household-First Rolling Microeconomy

## Bottom Line

The project now runs a recursive, accounting-constrained economy built from 200
anonymized SCE household types. Each household receives its own state, own survey
history, and only the public information available at the forecast origin. GPT-5.5
returns beliefs and intended choices; deterministic code enforces household budgets,
labor transitions, production, inventories, credit, and settlement.

The mechanics are now trustworthy. The predictions are not yet good. On the three
valid historical recursive months, the median path predicts falling nominal
consumption while first-release PCE rises. Consumption RMSE is **5.48 percentage
points**, with the wrong sign in all three months. The chained consumption index
falls from 100 to **86.40** while the first-release path rises to **102.14**.

The separate July 2026 origin is still frozen and unscored for August. Its median
path predicts **-3.77%** consumption growth.

## Architecture

```text
survey-seeded household state + own history + as-of public information
                              |
              one tool-isolated LLM elicitation
                              |
                 beliefs and intended choices
                              |
       population-weighted deterministic institutions
                              |
 households -> employer -> wages/output/prices -> next household state
      |                                             |
      +------------- credit intermediary -----------+
```

Survey weights determine each type's population mass using
`200 * population_weight / sum(population_weight)`. Every household-to-institution
flow uses that mass. Annual job-loss probabilities become constant monthly hazards;
fractional hiring preserves population mass, and new hires receive the employer's
wage offer.

## Current Forecast

| Item | Value |
| --- | --- |
| Forecast origin | `2026-07-01` |
| Information cutoff | `2026-07-10` |
| Target month | `2026-08-01` |
| Provider/model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted |
| Prompt | `household_ecology_monthly_v7` |
| Replay | 200 hits, 0 calls, immutable-reference match |
| Accounting | PASS; maximum residual `2.79e-09` |
| Source hash | `7962c63dc10d995b25099812957dadff71553eb08b1b2c8b3d607d9f2b7bbc2f` |
| Economy hash | `50e4bfad16aef2a7be8b699c4628d41595553d0e57b183c4309b74c9742e7931` |

| Scenario | Consumption growth | Saving rate | Credit growth | Employment rate | Price growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downside | -9.76% | 18.11% | -5.62% | 90.70% | -1.11% |
| Median | -3.77% | 18.23% | -5.65% | 90.70% | -1.27% |
| Upside | 1.50% | 18.67% | -5.66% | 90.70% | -1.50% |

The population-weighted mean of household p50 responses implies **5.16%**
inflation, **-0.80%** income growth, and a **9.12%** one-year job-loss
probability. Households intend -3.64% median consumption growth; feasibility and
goods clearing move the executed path to -3.77%.

## Historical Diagnostic

The historical run uses January-April 2026 rolling origins and February-May
one-month targets. The first transition is survey-seeded, so it is displayed but
excluded from scoring and compounding. Each later origin inherits the exact median
state produced by the preceding origin. Public macro information updates at each
origin, but simulated household and institutional state is not re-anchored to the
realized outcome.

| Metric | March prediction | May prediction | Score on March-May |
| --- | ---: | ---: | ---: |
| Nominal consumption growth | -4.43% | -4.27% | RMSE 5.48 pp; 0/3 signs |
| PCE price growth | -2.90% | -6.00% | 0/3 signs; direction only |
| Revolving-credit growth | -5.52% | -10.16% | 1/3 signs; direction only |
| Employment-rate change | -0.81 pp | -0.85 pp | 0/3 signs; direction only |
| Saving-rate proxy | 20.54% | 23.32% | descriptive only |

Consumption is the closest aggregate mapping. Credit, employment, and prices are
sign proxies, so their magnitudes are not presented as empirical estimates. Saving
is not scored because the economy omits taxes and transfers.

## Integrity Audit

- Every Codex call ran with an instruction-free `CODEX_HOME` in a fresh directory.
  Shell, local files, web, browser, apps, memory, tool search, and subagents were
  disabled.
- Cache identity binds provider, model, prompt, household card, and execution
  context. Replay is checked against an immutable live-response reference.
- Realization files are opened only after all child forecasts finish. Every first
  release used for scoring occurred after its forecast cutoff.
- A bottom-up audit verified 800 target-free household cards, 4,200 exact recursive
  state-field links, every accounting row, and all first-release dates.
- Macro rows recomputed from household decisions and survey weights agree with the
  emitted CSV to `1.34e-13`. RMSE and the cumulative index reproduce from the
  long-form comparison rows.
- Full test suite: 55 tests passing after the final integrity changes.

## Verdict

This is now a real, inspectable simulated microeconomy and the chart accurately
shows what it predicted. It does **not** yet predict the historical macro path well.
The remaining problem is behavioral: household plans and the recursive feedback
remain too contractionary. The August outcome is the first untouched check of the
current procedure.

Artifacts:

- Current run: `outputs/household_ecology_200_july_v7_current/`
- Historical diagnostic: `outputs/household_ecology_retrospective_2026_01_04_v7/`
