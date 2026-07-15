# LLM Household Economy

This project builds a macro economy from 200 persistent, anonymized households.
Real New York Fed Survey of Consumer Expectations histories supply individual
belief priors. Public 2022 SCF records supply matched family income, liquidity,
spending, and debt states. GPT-5.5 then writes each household's next-month
beliefs and conditional dollar policy.

The result is an **LLM household economy**. Code enforces budgets, credit limits,
goods-market settlement, and stock-flow accounting; it does not invent another
representative household or overwrite the LLM choices.

```text
200 SCE histories + matched SCF financial states + as-of public information
                                  |
                     one isolated GPT-5.5 call each
                                  |
         household beliefs + employed/not-employed dollar policies
                                  |
           code-enforced budgets, credit, production, settlement
                                  |
                  population-weighted household demand
                                  |
       output + inventory -> employment + wages -> family income
                                  |
            same LLM households decide again in period two
```

The producer loop is intentionally small and mechanical. Period-one demand sets
expected sales; the producer closes 35% of its inventory gap, 25% of its labor
gap, and 10% of the resulting employment-rate change into wages, capped at 2%
per month. There are no LLM firms or banks.

## Current Result

The economy runs, closes its accounts, carries household stocks forward, changes
producer employment and wages, and elicits a fresh second decision from every
household. That is a genuine two-period dynamic economy.

It does not yet forecast macro consumption well. On four retrospective monthly
origins, it ranks stronger and weaker months reasonably well (`r = 0.81`) but
keeps consumption growth near zero. RMSE is **0.70 percentage points**, sign
accuracy is **1 of 4**, and an origin-visible nominal-spending drift remains much
better at **0.24 points**.

| Origin | Target | LLM household economy | First-release PCE | Routine drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | -0.24% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.11% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | -0.02% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | -0.02% | +0.71% | +0.48% |

The frozen July-to-August forecast is **+0.03%**. August outcomes were unavailable
and excluded. The first recursive period then moves producer employment to
`1.00015`, wages to `1.00001`, and household consumption by **-0.06%** relative
to period one. This recursive result is an unscored mechanism experiment.

## Deposit Correction

Earlier prompts asked the LLM for consumption, debt, borrowing, and a separate
deposit contribution. That overdetermined the household budget. Deposits are now
the cash residual after income, fixed outflows, spending, debt service, and
borrowing.

The SCF income anchor is gross while its spending proxy omits taxes and some
recurring obligations. The state therefore records a declared fixed-outflow
calibration: at least 10% of gross family income, or more when the household's
existing saving-rate field requires it. This lowers the aggregate gross-income
cash residual from roughly 17% to about **7%**. It is an internal budget measure,
not the national personal saving rate.

## Run It

```bash
make ecology-fixture
make ecology-current-replay
make ecology-retrospective-replay
make ecology-feedback-replay
make ecology-observability
```

Fresh calls use Codex CLI only. `make ecology-live-200` elicits the first period;
`make ecology-feedback-live` elicits the second. Every call runs in a fresh empty
directory without shell, local-file, web, browser, app, memory, plugin, or
subagent access.

Every run emits prompt cards, accepted responses, household decisions, firm and
credit ledgers, accounting audits, source hashes, event hashes, and a manifest.
The feedback run additionally emits household and firm state transitions plus a
two-period macro path.

## Evidence Boundary

- January-April 2026 is retrospective development evidence and may be in model
  knowledge.
- The July 2026 origin is frozen for August and remains unscored.
- The recursive second period is a mechanism test, not a forecast score.
- Financial states are SCE-conditioned public SCF matches, not linked household
  accounts.
- The current bottleneck is behavioral amplitude. Better budget accounting and a
  real feedback loop do not make the households respond strongly enough to
  ordinary nominal growth.

See [CURRENT.md](CURRENT.md) for the milestone,
[reports/current_ecology_report.md](reports/current_ecology_report.md) for the
sendable result, and [research_history.md](research_history.md) for the experiment
trail.

```bash
make check
make test
```

The previous weighted-demand economy is recoverable at Git tag
`macro-v1-weighted-demand`.
