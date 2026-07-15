# The Household Economy Finds A Timing Pattern, Not Yet The Level

The 200-household economy now produces its forecast from household decisions alone.
That correction overturns the previous 4/4 directional result: across four historical
months, the anchor-free forecasts have RMSE `0.782` percentage points and get one
direction right. Yet they rank the months well. Correlation with first-release PCE is
`0.683`, and RMSE after removing the common level bias is `0.126` points.

So we have a real simulated household economy and an intriguing relative pattern over
four retrospective proxy observations. We do not yet have a good absolute macro
forecast. The households are systematically too pessimistic by `0.771` points and
generate a little over half the observed month-to-month variation.

## What The Economy Does

```text
200 anonymized SCE histories + matched SCF household states
                              +
                  public information known at the origin
                              |
                              v
                one isolated GPT-5.5 call per household
                              |
                              v
        beliefs + conditional household policies in nominal dollars
                              |
                              v
              household budgets, credit, and settlement
                              |
                              v
                    population-weighted demand
                              |
                              v
     producer output + inventory -> employment + wages -> family income
                              |
                              v
               fresh period-two household decisions
```

Each historical origin gets the latest SCE observations that would have been public
under a nine-month release lag. The household card includes observed aggregate
spending growth as dated context, but the engine does not add it to the household's
spending before the model acts. GPT-5.5 returns signed dollar changes in committed
spending, discretionary spending, one-off purchases, debt repayment, and borrowing.
Code then applies cash, credit, goods, and counterparty constraints.

## Historical Result

| Origin | Target | LLM household economy | First-release PCE | Origin-visible drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | -0.24% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.01% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | -0.08% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | -0.17% | +0.71% | +0.48% |

| Diagnostic | Result |
| --- | ---: |
| Consumption RMSE | 0.782 pp |
| Mean forecast bias | -0.771 pp |
| Demeaned RMSE | 0.126 pp |
| Consumption correlation | 0.683 |
| Forecast/actual standard-deviation ratio | 0.563 |
| Consumption direction | 1/4 |
| Origin-visible drift RMSE | 0.243 pp |
| Revolving-credit direction | 1/4 |
| Settlement audit | PASS |

![Economic observability surface](current_ecology_observability_surface.png)

The level and timing results are different facts. The households reduce spending in
three months when nominal PCE rose in all four, so their point forecasts are poor. But
February is their weakest month and March their strongest, matching the ranking of the
real outcomes. The model seems to recognize changes in relative demand pressure while
turning that information into excessively cautious household budgets.

The simple origin-visible drift remains much better in levels. It is shown as context,
not as part of the LLM economy.

## Why The Previous Result Changed

V22 began each household at a spending level that already incorporated the latest
aggregate PCE growth. The LLM then adjusted that pre-grown level. Because household
adjustments were negative in every historical month, the reported positive forecasts
came from the aggregate anchor, not from bottom-up household behavior.

V23 removes the anchor from execution. Households start from recurring committed and
discretionary dollar levels; aggregate consumption growth is information they may use,
not a number the engine silently carries forward. This is the first period-by-period
chart that cleanly answers whether the households themselves generate the forecast.

## Household State And Monthly Transition

SCE histories now update by origin rather than freezing every run at the same two
waves. The materializer includes an observation only once its event date and public
availability date are both safe for that origin. Missing later answers do not erase a
household's latest valid field-level prior.

The monthly transition carries settled deposits, debt, committed spending, and
discretionary spending. One-off purchases remain one-off. Producer wage and income
feedback updates the next household state, and the same policy schema is used in
rolling and recursive runs.

## Current Forecast And Recursive Trace

The frozen July origin predicts **-0.13%** nominal consumption growth for August.
Its weighted household actions add about `$10.80` to committed spending, subtract
`$30.34` from discretionary spending, add no one-off purchase, repay `$19.49` of
extra debt, and borrow `$45.08` per represented household. August is unscored.

Starting from that economy, a fresh period-two call for every household produces a
further **0.31% fall** in consumption. Producer output, employment, and wages barely
move because the aggregate demand change is small. This is a verified state transition,
not a September forecast or an estimate of the causal value of feedback.

## Integrity Record

- Final published runs replay 200 current, 800 historical, and 200 period-two banked
  Codex CLI responses with no fresh calls.
- Prompt cards contain no realized targets. Historical cards expose only household
  observations and public information available at each origin.
- The household-history manifest and public history file are hash-bound into every
  run; raw SCE respondent identifiers never enter public artifacts.
- Recurring spending, deposits, debt, inventory, employment, wages, and income cross
  the monthly boundary through one canonical transition.
- Household budgets, goods inventory, bank stocks, and named counterparty flows
  reconcile at numerical tolerance. A firm balance sheet and full external sector are
  outside this version.

## Where This Leaves The Project

The architecture is now honest enough to diagnose. Survey data supplies household
heterogeneity, the LLM writes state-dependent policies, settlement creates aggregate
demand, and demand feeds back into income before the next decision. The four-point
timing pattern is enough to justify improving this object, not enough to call it a
durable signal.

The next work is narrower than adding more agents. We need a better household spending
state and a better elicitation of ordinary nominal inertia. The current SCF-conditioned
"typical month" is not the same thing as each household's actual previous-month
expenditure, and the model responds to uncertainty with broad discretionary cuts. Those
two effects can explain the level shift without changing the economy's structure.

January-April remains retrospective development evidence. The July forecast stays
frozen until the August realization arrives.
