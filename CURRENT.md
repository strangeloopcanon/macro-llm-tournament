# Current Project Surface

The active system is a 200-household LLM economy with one aggregate
producer-income feedback step.

```text
real household priors -> LLM beliefs and dollar policies -> enforced budgets
and settlement -> aggregate demand -> producer output/inventory -> aggregate
employment/wages -> updated family income -> fresh LLM household decisions
```

Rolling forecasts re-anchor to observed SCE-SCF state at every origin. The
separate two-period experiment carries simulated deposits, debt, inventory,
aggregate employment, wages, and family income forward once. Forecast evaluation
and recursive simulation therefore answer different questions with the same
household behavior.

## What Works

- 200 households receive private histories, matched financial states, and only
  information available at the origin.
- Households choose committed spending, discretionary spending, debt repayment,
  and borrowing in nominal dollars.
- Deposits are a cash residual, not a second household choice.
- Total saving is split between liquid-buffer accumulation and non-deposit saving;
  matched deficit households remain deficit households.
- The producer carries inventory and realizes aggregate output, employment, wages,
  and family wage income before the next household decision.
- Household budgets, goods inventory, bank stocks, and named counterparty flows
  reconcile at numerical tolerance.
- Live and replay executions are hash-bound, including feedback parent artifacts
  and source inputs.

## Latest Numbers

The v22 historical forecasts are `+0.15%, +0.26%, +0.11%, +0.12%` against
first-release PCE growth of `+0.48%, +0.90%, +0.51%, +0.71%`.

- Consumption RMSE: **0.508 pp**.
- Direction: **4/4**.
- Correlation: **0.759**.
- Origin-visible drift RMSE: **0.243 pp**.
- Revolving-credit direction: **1/4**.
- Settlement audit: **PASS** in every run.

The July origin is frozen for August at **+0.19%**. In the two-period mechanism
run, settled producer employment rises `0.0013%`, the average wage rises
`0.00013%`, and fresh period-two household spending rises **0.23%**. Respondent
job labels do not change; the producer loop updates aggregate family labor income.

## What This Means

The natural-household redesign fixed the sign problem. The model now recognizes
which way nominal consumption is moving at all four historical origins. It still
compresses the magnitude by roughly half and loses to the origin-visible drift
anchor. Revolving-credit behavior is also still poor.

The economy itself is now simulatable for short counterfactual paths: household
choices create demand, the producer changes output and labor income, and households
respond to the resulting state. The period-two result is an unscored mechanism
trace, not evidence that this feedback improves forecasts; no matched no-feedback
household-call arm has been run.

## Next Model-Building Problem

The next iteration should improve state-dependent spending amplitude without
adding LLM firms or banks. The promising surface is the household policy itself:
how recent nominal spending, income, liquidity, and supplied beliefs change
committed versus discretionary dollars. Development stays on the spent historical
origins. The frozen August result remains untouched until its realization arrives.

Current evidence:

- `outputs/household_ecology_200_july_v22_current/`
- `outputs/household_ecology_retrospective_2026_01_04_v22/`
- `outputs/household_ecology_feedback_200_july_v2/`
- `outputs/household_ecology_observability_v3/`
