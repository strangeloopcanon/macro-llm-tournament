# Household-First Rolling Microeconomy

## Bottom Line

The project now runs 200 separate LLM households as a rolling demand economy. Real
SCE histories supply beliefs and demographics; deterministic SCE-conditioned SCF
matches supply coherent income, liquidity, spending, and debt states. GPT-5.5 writes
each household's conditional monthly policy. Code, not the model, enforces budgets,
credit limits, production feasibility, and accounting.

This redesign fixed the old economy's contractionary failure. On four historical
one-month origins, the consumption sign is correct **4/4** times and RMSE falls from
**5.48** to **0.47 percentage points**. The result is not yet strong forecasting
performance: a simple origin-visible nominal-spending drift scores **0.24 points**.
The households move in the right direction but understate ordinary nominal growth.

The July-to-August 2026 forecast is frozen at **+0.28%** median nominal consumption
growth. August outcomes were unavailable and excluded.

## The Economy

```text
200 SCE histories + 200 matched SCF financial states + as-of public data
                                  |
                     one isolated GPT-5.5 call each
                                  |
       beliefs + employed/not-employed household dollar policies
                                  |
       deterministic budgets, credit, production, and settlement
                                  |
              household consumption, saving, and debt paths
                                  |
              population-weighted macro demand forecast
```

For the active forecasting test, respondent employment and wages are fixed. This
prevents an uncalibrated labor-matching model from creating the answer. Production
follows expected sales with gradual inventory adjustment. Firms and banks are not
LLM agents.

The rolling chart uses independent one-month forecasts. Each origin restarts from
the same fixed SCE-SCF household anchor and receives newly available public data.
No simulated balance, forecast error, or realized target carries into the next
origin.

## Historical Results

| Origin | Target | LLM consumption | First-release PCE | Routine drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | +0.13% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.16% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | +0.30% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | +0.31% | +0.71% | +0.48% |

| Diagnostic | Result |
| --- | ---: |
| Consumption direction | 4/4 |
| Consumption RMSE | 0.468 pp |
| Routine-drift RMSE | 0.243 pp |
| Saving-rate direction | 3/4 |
| Revolving-credit direction | 1/4 |
| Compounded LLM consumption index | 100.90 |
| Compounded first-release PCE index | 102.63 |
| Compounded routine-drift index | 101.75 |

The credit miss is systematic: households pay down revolving balances while the
aggregate series grew in three of four months. Saving changes are directionally more
credible. Those two mappings are sign proxies only; consumption is the only
magnitude-scored aggregate.

## Frozen August Forecast

| Item | Value |
| --- | --- |
| Origin / cutoff | `2026-07-01` / `2026-07-10` |
| Target | `2026-08-01` |
| Provider / model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted |
| Median consumption growth | +0.28% |
| Downside / upside | +0.28% / +0.28% |
| Median saving rate | 16.73% |
| Median revolving-credit growth | -2.06% |
| Accounting | PASS; max residual `1.12e-08` |
| Replay | PASS; 200 hits, 0 calls, immutable reference matched |
| Source hash | `e1979a7dd789d5dca322aa5bcb141a8957ab94935750dd99eb2370126369a694` |
| Replay-equivalence hash | `d233bb4f1708772e95f4a155223d7be7c01aac4ea3ef34b7fd79646d485ca24d` |

The three scenario paths coincide because employment and wages are fixed and the
household dollar policy is a point policy. They are not a predictive interval.

## Integrity

- All 800 historical and 200 prospective responses were created through Codex CLI.
- Each model call ran without shell, local files, web, browser, apps, memory,
  plugins, or subagents.
- Prompt cards contain no `actual_*` target fields and no prior simulated state.
- Realizations load only after all historical forecasts finish.
- SCF income components reconcile to annual household income for all 200 states.
- SCF family earnings are held fixed at the household boundary; no respondent wage
  share is invented. Dynamic labor is blocked for these states.
- All five runs pass household, credit, employer, and stock-flow accounting.
- The historical dates are retrospective and may be in model knowledge; this is
  development evidence, not a clean holdout.

## What We Learned

The household economy was not failing because it lacked firms, banks, or a richer
general-equilibrium loop. It was failing because the behavioral object was wrong:
households were asked for abstract percentage cuts, were given coarse financial
templates, and one respondent's employment status stood in for the whole household.

Once households received coherent continuous states and wrote conditional dollar
policies, the mechanical collapse disappeared and every consumption sign became
correct. The remaining gap is narrower: GPT households preserve spending levels and
react to conditions, but they do not fully carry normal nominal trend into next
month's dollar plan. That is the next behavior problem to study. It is too early to
add LLM firms or banks.

Artifacts:

- Prospective run: `outputs/household_ecology_200_july_v18_current/`
- Historical diagnostic: `outputs/household_ecology_retrospective_2026_01_04_v18/`
