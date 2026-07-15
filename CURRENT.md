# Current Project Surface

The active system is an anchor-free, 200-household LLM economy with a one-step
producer and income feedback loop.

```text
real household priors -> LLM beliefs and dollar policies -> household settlement
-> aggregate demand -> producer output/inventory -> employment/wages -> family
income -> fresh LLM household decisions
```

Rolling forecasts re-anchor financial states at every origin and use the latest SCE
observations available under a nine-month publication lag. The two-period experiment
instead carries simulated deposits, debt, recurring spending, inventory, employment,
wages, and family income forward once.

## What Works

- All 200 households receive their own observed history, matched financial state,
  and only public information available at the origin.
- Households choose signed dollar changes in committed and discretionary spending,
  one-off purchases, debt repayment, and borrowing under employed and job-loss
  branches.
- Aggregate PCE growth is context, not an executable spending anchor.
- Recurring spending, deposits, debt, and producer state survive the monthly
  transition; one-off purchases do not become recurring consumption.
- Household budgets, goods inventory, bank stocks, and named counterparty flows
  reconcile at numerical tolerance.
- Live and replay executions are hash-bound to the materialized household history,
  source inputs, prompts, payloads, parent artifacts, and result.

## Latest Numbers

The v23 historical forecasts are `-0.24%, +0.01%, -0.08%, -0.17%` against
first-release PCE growth of `+0.48%, +0.90%, +0.51%, +0.71%`.

- Full consumption RMSE: **0.782 pp**.
- Mean bias: **-0.771 pp**.
- Demeaned RMSE: **0.126 pp**.
- Correlation: **0.683**.
- Forecast/actual standard-deviation ratio: **0.563**.
- Direction: **1/4**.
- Origin-visible drift RMSE: **0.243 pp**.
- Settlement audit: **PASS** in every final run.

The July origin is frozen for August at **-0.13%**. In the two-period mechanism
trace, fresh period-two household spending falls **0.31%**; producer employment and
wages remain nearly flat because the demand change is small.

## What This Means

V23 answers a harder question than v22. Once the observed PCE drift is removed from
the executable baseline, the households do not predict the absolute monthly growth
rate well. They are about 0.75 points too pessimistic on average and produce a little
over half the observed variation.

But the cross-month ordering survives. A `0.683` correlation and `0.126`-point
demeaned RMSE across four observations say the households react more strongly in the
months when real consumption later grows more strongly. With four retrospective proxy
observations, that is a hypothesis worth testing, not evidence of a durable signal.

The economy is simulatable: household policies create demand, demand changes the
producer state and family income, and households respond again. The next model
problem is the level of ordinary nominal spending, especially the recurring baseline
and the model's systematic tendency to cut discretionary purchases. Adding LLM firms
or banks would not fix that.

## Active Evidence

- `outputs/household_ecology_200_july_v23_current/`
- `outputs/household_ecology_retrospective_2026_01_04_v23/`
- `outputs/household_ecology_feedback_200_july_v23/`
- `outputs/household_ecology_observability_v23/`
