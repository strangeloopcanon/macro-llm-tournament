# Current Project Surface

The active system is a 200-household LLM economy with one closed producer-income
feedback step.

```text
real household priors -> LLM beliefs and dollar policies -> code-enforced
budgets and settlement -> aggregate demand -> producer output/inventory ->
employment/wages -> updated family income -> fresh LLM household decisions
```

Rolling forecasts still re-anchor to observed SCE-SCF state at every historical
origin. The separate two-period experiment carries simulated deposits, debt,
inventory, employment, wages, and family income forward once. This keeps forecast
evaluation and recursive simulation distinct while using the same household
behavior.

## What Works

- 200 distinct households receive private histories, financial states, and only
  origin-visible information.
- Household choices are expressed in nominal dollars for committed spending,
  discretionary spending, debt repayment, and borrowing.
- Deposits are a residual rather than a second, inconsistent choice.
- The producer pays family wage income, plans output from prior demand, carries
  inventory, and adjusts employment and wages gradually.
- The same households decide again from carried balances and updated income.
- All household, producer, credit, inventory, deposit, and debt identities pass.
- Live and replay executions are hash-bound and equivalent.

## What The Numbers Say

The fresh v21 retrospective forecasts are `-0.24%, +0.11%, -0.02%, -0.02%`
against first-release PCE growth of `+0.48%, +0.90%, +0.51%, +0.71%`.

- Consumption RMSE: **0.70 pp**.
- Direction: **1/4**.
- Correlation: **0.81**.
- Routine-drift RMSE: **0.24 pp**.
- Revolving-credit direction: **1/4**.
- Accounting: **PASS** in all runs.

The correlation improvement means the model contains some information about
relative month strength. The level and sign failures mean it is still too
conservative to be a useful macro forecast.

The July origin is frozen for August at **+0.03%**. In the two-period mechanism
run, period-one demand produces a `+0.015%` employment-index change and a
`+0.0015%` wage-index change; fresh period-two household spending then changes
**-0.06%**. The loop works, but weak household demand naturally produces weak
feedback.

## Deposit State

The old “deposit intention” was removed because it did not constrain consumption
and overdetermined the budget. Taxes and omitted recurring outflows are now an
explicit fixed household flow. The weighted gross-income residual is about 7%,
down from about 17%; the remaining deposit change is a cash residual, not an LLM
saving instruction or a national saving-rate forecast.

## Next Model-Building Problem

The economy no longer lacks a feedback loop. It lacks enough realistic household
response amplitude. The next development should improve how supplied beliefs,
recent spending drift, and household state change committed and discretionary
spending, using the spent historical origins. The frozen August result must remain
untouched until its realization arrives.

Current local evidence:

- `outputs/household_ecology_200_july_v21_current/`
- `outputs/household_ecology_retrospective_2026_01_04_v21/`
- `outputs/household_ecology_feedback_200_july_v1/`
- `outputs/household_ecology_observability_v2/`
