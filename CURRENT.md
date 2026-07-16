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

## Latest Numbers: The Primary Comparison

The v23 historical forecasts are `-0.24%, +0.01%, -0.08%, -0.17%` against
first-release PCE growth of `+0.48%, +0.90%, +0.51%, +0.71%`.

- Full consumption RMSE: **0.782 pp**.
- Mean bias: **-0.771 pp**.
- Demeaned RMSE: **0.126 pp**.
- Correlation: **0.683**.
- Forecast/actual standard-deviation ratio: **0.563**.
- Direction: **1/4**.
- Origin-visible drift RMSE: **0.243 pp**.
- Origin-visible drift demeaned RMSE: **0.113 pp**.
- Origin-visible drift correlation: **0.974**.
- Origin-visible drift direction: **4/4**.
- Settlement audit: **PASS** in every final run.

The July origin is frozen for August at **-0.13%**. In the two-period mechanism
trace, fresh period-two household spending falls **0.31%**; producer employment and
wages remain nearly flat because the demand change is small.

## Nine-Variable Panel

`outputs/household_ecology_observability_v23/` now binds a predicted-versus-
first-release-actual panel for all nine declared macro outputs. The four historical
origins are retrospective development evidence (`n=4` for every variable); the July
row is frozen for August and unscored.

| Variable | Realized comparator | What the comparison is |
| --- | --- | --- |
| Nominal consumption growth | nominal PCE | the only closest aggregate comparison; the main result |
| Price growth | PCEPI | household-belief/deflation proxy |
| Real consumption growth | real PCE | belief/deflation proxy |
| Real disposable income growth | real disposable income | belief/deflation proxy |
| Payroll growth | PAYEMS | target-month producer-plan proxy using origin inventory |
| Unemployment-rate level | UNRATE | target-month producer-plan proxy using origin inventory |
| Saving-rate change | PSAVERT | gross household budget-residual proxy, not a national saving identity |
| Revolving-credit growth | revolving credit | direction-only proxy |
| Retail-sales growth | retail sales | declared demand proxy |

The panel makes the model's cross-variable surface inspectable. It does not validate
the belief/deflation mappings, the target-month producer plan, a national-accounts saving
equation, or a retail-sales model. Credit has no level-error score. The consumption
result still controls the conclusion: the household layer loses to PCE drift already
visible in its own prompt.

## What This Means

V23 answers a harder question than v22. Once the observed PCE drift is removed from
the executable baseline, the households do not predict the absolute monthly growth
rate well. They are about 0.75 points too pessimistic on average and produce a little
over half the observed variation.

The cross-month ordering does not establish incremental signal. The PCE drift already
shown in every household prompt has `0.974` correlation, `0.113`-point demeaned RMSE,
and 4/4 direction, beating the household economy on every scored consumption
diagnostic. The household layer currently turns its visible context into worse point
forecasts.

The economy is simulatable: household policies create demand, demand changes the
producer state and family income, and households respond again. The next model
problem is the level of ordinary nominal spending, especially the recurring baseline
and the model's systematic tendency to cut discretionary purchases. Adding LLM firms
or banks would not fix that.

All cached payloads replay exactly. Retained accepted-call journals cover 200/200
current payloads, 309/800 historical payloads, and 200/200 period-two payloads. The
491 historical records without journals remain useful development artifacts, but their
original provider attempts are not independently evidenced by the retained workspace.

## Active Evidence

- `outputs/household_ecology_200_july_v23_current/`
- `outputs/household_ecology_retrospective_2026_01_04_v23/`
- `outputs/household_ecology_feedback_200_july_v23/`
- `outputs/household_ecology_observability_v23/`
