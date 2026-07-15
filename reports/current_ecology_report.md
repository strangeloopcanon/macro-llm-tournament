# The Household Economy Moves in the Right Direction, but by Too Little

The current system turns 200 real survey histories into 200 separate LLM
households, executes their next-month policies inside an accounting-constrained
economy, and aggregates the result. All four historical consumption forecasts
have the right sign.

It has not yet produced a competitive macro forecast. Consumption RMSE is **0.61
percentage points**, versus **0.24** for a simple origin-visible nominal-spending
drift. The household aggregate rises each month, but consistently by too little.

The July-to-August 2026 forecast is frozen at **+0.08%**. August outcomes were not
available to the model and have not yet been scored.

## What Runs

```text
200 anonymized SCE household histories
              +
200 deterministic SCE-conditioned SCF financial states
              +
public information available at the forecast date
              |
              v
one tool-isolated GPT-5.5 call per household
              |
              v
beliefs + employed/not-employed dollar policies
              |
              v
deterministic budgets, credit, production, inventories, settlement
              |
              v
population-weighted consumption and balance-sheet paths
```

The LLM does not directly set the macro total. It decides what each household
expects and intends to do: committed spending, discretionary spending, deposit
changes, debt repayment, and borrowing. Code then enforces the household budget,
credit limit, minimum debt payment, goods-market clearing, and counterparty
accounts.

Real data supplies the heterogeneity that earlier experiments showed LLM personas
could not invent. The personal job-loss prior comes from same-wave SCE `Q13new`,
the respondent's chance of losing the current job over 12 months; unanswered rows
stay missing. The separate `Q4new` answer, the chance that U.S. unemployment
rises, remains an aggregate belief and is never converted into personal risk.

## Historical Diagnostic

Each origin is an independent rolling one-month forecast. Every run starts from
the same fixed SCE-SCF household anchor and receives only public information
available at that origin. Realizations are loaded after all four forecasts finish.
No simulated balance, forecast error, or future observation carries forward.

| Origin | Target | LLM household economy | First-release PCE | Routine drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | +0.01% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.02% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | +0.06% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | +0.16% | +0.71% | +0.48% |

| Diagnostic | Result |
| --- | ---: |
| Consumption direction | 4/4 |
| Consumption RMSE | 0.613 pp |
| Routine-drift RMSE | 0.243 pp |
| Consumption correlation | 0.103 |
| Revolving-credit direction | 1/4 |
| Compounded LLM consumption index | 100.25 |
| Compounded first-release PCE index | 102.63 |
| Compounded routine-drift index | 101.75 |

![Diagnostic economic observability surface](current_ecology_observability_surface.png)

The sign result is real; the weak correlation is equally important.
The model produces positive nominal growth every month, but does not rank stronger
and weaker months correctly in this four-month sample. Credit is also wrong in
three of four months: households pay revolving balances down while the aggregate
series usually rises.

## What The Expanded Surface Adds

The top row preserves the two observed comparisons. The remaining rows expose the
internal causal chain rather than adding more scores:

- population-weighted inflation beliefs rise from 4.16% to 5.04% across the shown
  origins, income-growth beliefs stay negative at -1.07% to -1.61%, and
  personal next-month job-loss risk stays near 0.93% to 0.97%;
- intended deposit additions equal 13.4% to 18.8% of baseline monthly consumption,
  while extra debt payments remain below 0.38% and new borrowing below 0.14%;
- intended consumption growth is only 0.01% to 0.16%, and deterministic execution
  leaves it almost unchanged; and
- the resulting revolving-debt stock falls 2.0% to 2.8% per month.

This identifies the current amplitude problem more sharply. The accounting engine
is not clipping a healthy demand signal. The household policies themselves combine
negative income expectations with unusually large intended deposit additions, so
weak demand enters before production and settlement.

The bottom-right panel adds the smallest defensible firm-side structure. Expected
sales inherit household-demand growth; target output closes 35% of the inventory
gap; fixed productivity maps output into required labor; planned employment closes
25% of that requirement; and a declared index combines demand and inventory into
price pressure. These are unscored deterministic mechanics. They do not feed wages,
employment, prices, or income back into another household decision, and they do not
improve any historical score.

## Frozen August Forecast

| Item | Value |
| --- | --- |
| Origin / cutoff | `2026-07-01` / `2026-07-10` |
| Target | `2026-08-01` |
| Provider / model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted |
| Point consumption growth | +0.08% |
| Revolving-credit growth | -2.77% |
| Employment rate | 66.20%, held fixed |
| Gross-income residual rate | 16.89%, internal accounting only |
| Accounting | PASS; max residual `1.30e-08` |
| Replay | PASS; 200 hits, 0 calls, immutable reference matched |
| Source commit | `d42e3b2b7d02f7eda28d9b42bda49311fd1d255d` |
| Source hash | `4aa98380ba94a269a7699bcdb5ee6745be2ae532d5fc061ef3d1a1cb7ed22897` |
| Replay-equivalence hash | `55ae0b8904519669a3bfb1eb7ab17958c7f18c7a5baad2fe7904fa9a807b20e2` |

The population-weighted household p50 beliefs are 5.04% inflation, -1.61%
income growth, and 0.93% next-month job-loss risk. Population-weighted intended
consumption is $3.113 million before feasibility and $3.112 million after it, so the +0.08%
result comes from household policies rather than material rationing.

## How To Read The Mapping

The model's `consumption_growth_pct` is the change from a numerically fixed,
synthetic SCE-SCF estimate of each household's recent typical spending to its
executed target-month spending. The aggregate is interpreted as month-over-month
nominal PCE growth. This is the closest available macro proxy, not observed growth
for the same households.

The internal income remainder uses synthetic gross SCF-family income. It is not
disposable personal income and is not scored against the national personal saving
rate. Revolving debt is retained as a sign-only proxy for aggregate revolving
consumer credit.

## Integrity Record

- The corrected campaign used 1,000 fresh Codex CLI calls: 200 prospective and
  800 historical. There were zero failed provider attempts.
- Replays use 1,000 cache hits and zero calls. The prospective immutable reference
  and all four historical child references match exactly.
- Calls ran in fresh empty directories without shell, local-file, web, browser,
  app, memory, plugin, or subagent access.
- Prompt cards contain no `actual_*` targets and no simulated prior state.
- Time-varying SCE answers preserve wave-specific missingness; no future `Q13new`
  value is used to complete an earlier household history.
- All five economies pass household, employer, credit, and stock-flow accounting.
- Manifests bind the runs to clean Git commit `d42e3b2` and executable source hash
  `4aa98380...`.
- The historical dates are retrospective and may be in model knowledge. They are
  development evidence. August is the active prospective test.

## What We Learned

The current result does not establish which individual redesign choice caused the
sign correction; there is no component-by-component ablation. It does establish
that the full corrected architecture can keep demand positive while remaining
accounting-safe. Earlier versions asked for abstract percentage cuts, used coarse
financial templates, or mishandled personal-risk information, and are retired.

In the corrected architecture, household policies still fail to carry enough
ordinary nominal growth into the next month. That amplitude miss is large: the
four LLM predictions compound to 100.25 while first-release PCE compounds to
102.63. The model also pays revolving balances down in every origin, matching the
aggregate credit direction only once.

That is the next model-building problem. Firms and banks should remain mechanical
until repeated prospective errors show that a missing institutional response,
rather than household elicitation, is responsible.

The expanded surface strengthens that sequencing: the household policy is now the
visibly dominant bottleneck. A recursive firm-labor loop would require an identified
allocation of SCF family earnings to respondent labor income; inventing that split
would make the economy look fuller while weakening its empirical basis.

## Current Evidence

- Prospective run: `outputs/household_ecology_200_july_v20_current/`
- Historical diagnostic: `outputs/household_ecology_retrospective_2026_01_04_v20/`
- Economic observability surface: `outputs/household_ecology_observability_v1/`
- Retired research record: `research_history.md`
