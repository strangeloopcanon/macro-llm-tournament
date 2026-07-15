# We Built the Dynamic Household Economy. Its Behavior Is Still Too Compressed.

Two hundred LLM households now choose spending, debt repayment, and borrowing
from real survey histories and matched financial states. Their demand moves a
producer's output, inventory, employment, wages, and family income; the same
households then decide again. The economy is dynamic and accounting-safe.

The macro result remains weak. Across four retrospective months, the households
rank stronger and weaker consumption months better than before, but their
aggregate stays too close to zero. The producer feedback loop cannot amplify a
signal the households barely emit.

## The Economy

```text
200 anonymized SCE histories + SCE-conditioned SCF household states
                              +
                  public information as of the origin
                              |
                              v
                one isolated GPT-5.5 call per household
                              |
                              v
        beliefs + conditional household policies in nominal dollars
                              |
                              v
             code-enforced budgets, credit, and settlement
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

The firm is not another role-playing model. It is the smallest transparent loop
needed for demand to change production and income: 35% inventory-gap closure,
25% labor-gap closure, and a small wage response capped at 2% per month. Banks
remain mechanical.

## Historical Diagnostic

| Origin | Target | LLM household economy | First-release PCE | Routine drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | -0.24% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.11% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | -0.02% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | -0.02% | +0.71% | +0.48% |

| Diagnostic | Result |
| --- | ---: |
| Consumption direction | 1/4 |
| Consumption RMSE | 0.701 pp |
| Routine-drift RMSE | 0.243 pp |
| Consumption correlation | 0.809 |
| Revolving-credit direction | 1/4 |
| Accounting | PASS |

![Diagnostic economic observability surface](current_ecology_observability_surface.png)

The new information is the combination of correlation and scale. The households
move March above February, April, and May in roughly the right ordering, but the
whole path is compressed around zero. This is no longer a story about a broken
accounting engine hiding good intentions. The intentions themselves are small.

## What Changed About Deposits

The previous prompt asked for both household actions and a deposit contribution.
That was redundant: once income, spending, debt, and borrowing are known, deposits
are the residual. The explicit deposit choice has been removed.

The matched SCF state also mixed gross family income with a spending proxy that
omits taxes and some recurring obligations. The new budget records those as a
fixed outflow, with a declared 10% gross-income floor and the existing household
saving-rate field used when it implies more. The weighted gross-income residual
falls from roughly 17% to about **7%**. It is still an internal cash measure, not
the national personal saving rate.

## The Two-Period Result

The frozen July origin predicts **+0.03%** nominal consumption growth for August.
Starting from that period-one economy:

| Dynamic quantity | Period one | Period two |
| --- | ---: | ---: |
| Consumption | $3.110m | $3.109m |
| Output | 3.110m units | 3.112m units |
| Inventory | 248,774 units | 252,472 units |
| Producer employment index | 1.00000 | 1.00015 |
| Producer wage index | 1.00000 | 1.00001 |

Period-two household spending changes **-0.06%**. The producer raises output
slightly in response to prior demand, but households buy a little less and
inventories accumulate. Every stock and counterparty flow reconciles. This is an
unscored mechanism result, not a September forecast claim.

## Integrity Record

- The v21 campaign elicited 1,000 fresh first-period Codex CLI responses: 800
  historical and 200 prospective, with zero provider failures.
- The feedback experiment elicited 200 additional period-two responses. The
  accepted calls were banked before a control-flow bug stopped settlement; after
  the bug fix, the published run used those exact cached responses and replayed
  with zero calls.
- Prompt cards contain no realized target values. The simulated period-two firm
  state is separately labelled and never presented as public news.
- Period-two opening deposits, debt, and inventory equal period-one closing
  stocks. Respondent employment status is unchanged; aggregate producer
  employment scales family wage income instead.
- All published runs pass accounting and immutable replay checks.

## So What

We now have the object we wanted: a bottom-up household economy whose actions
create demand, whose producer responds, and whose new income state feeds back
into household decisions. Adding that loop did not solve forecasting. It located
the remaining problem more sharply.

The LLM households update and differentiate across months, but they do not turn
that information into enough nominal spending movement. The next work belongs at
the belief-to-action interface, not in a larger ecology of firms and banks.

The January-April results are retrospective development evidence. August remains
the untouched prospective score.
