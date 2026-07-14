# Household-First Macro Ecology

This project asks whether LLM-updated household beliefs can produce useful macro
forecasts and counterfactuals inside an explicit, accounting-constrained economy.
Real survey data supplies household heterogeneity. Each household gets one tool-isolated
LLM elicitation to report its beliefs and intended choices; deterministic code then
enforces budgets, credit, production, employment, inventories, and settlement.
Survey weights determine each household type's population mass, and annual job-loss
risk is converted to a monthly hazard. The aggregate path emerges from those
household decisions.

The household panel comes from the Federal Reserve Bank of New York's Survey of
Consumer Expectations (SCE).

```text
survey-seeded SCE household state + own history + as-of public information
                              |
              one tool-isolated LLM elicitation
                              |
                 beliefs and intended choices
                              |
          deterministic budget and credit feasibility
                              |
 household actions ---> aggregate employer ---> jobs, wages, output, prices
         |                      |                         |
         +------ credit intermediary <------------------+
                                |
                 next household and macro state
```

The first version deliberately has one aggregate employer and one credit
intermediary. Firms and banks are not additional role-play agents. This keeps the
economy interpretable while allowing demand, production, labor, prices, borrowing,
debt repayment, defaults, and inventories to interact.

## Current Result

The corrected full run gave 200 GPT-5.5 households only information available by
July 10, 2026, and froze their path for August before August outcomes were known.
Every prompt now receives the exact engine state it will execute and says whether
that state is survey-seeded or inherited. The median path has household intentions
at **-3.64%** consumption growth and feasible consumption at **-3.77%**, with
lower-liquidity households cutting more.

### Run facts

| Item | Value |
| --- | --- |
| Forecast origin | `2026-07-01` |
| Information cutoff | `2026-07-10` |
| One-month-ahead target | `2026-08-01` |
| Provider and model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted, one tool-isolated response per household |
| Reproducibility check | 200 replay hits, 0 fresh calls |
| Accounting | **PASS**, maximum residual `2.79e-09` |
| Replay | **PASS**, exact economy hash reproduced |

### Forecast paths

| Scenario | Consumption growth | Saving rate | Revolving-credit growth | Employment rate | Price growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downside | -9.76% | 18.11% | -5.62% | 90.70% | -1.11% |
| Median | -3.77% | 18.23% | -5.65% | 90.70% | -1.27% |
| Upside | 1.50% | 18.67% | -5.66% | 90.70% | -1.50% |

The population-weighted mean of household p50 responses implies **5.16%**
inflation, **-0.80%** income growth, and a **9.12%** one-year job-loss
probability. In the median economy, 200 household types representing 200
population-equivalent units execute `$915,697` of consumption, `$134,377` of
debt payments, and `$11,922` of new borrowing.

### Evidence boundary

| This run shows | This run does not yet show |
| --- | --- |
| A branchable, recursive household microeconomy runs end to end. | That its August forecast is accurate. |
| Household heterogeneity affects feasible consumption and balance sheets. | That one origin validates the economy's dynamics. |
| Production, inventories, employment, credit, and settlement have explicit counterparties. | That one aggregate employer is sufficient. |
| All three scenarios satisfy household, employer, credit, and stock-flow accounting. | That coarse survey-mapped balance sheets equal measured household accounts. |

The August outcome surface was unavailable and excluded when the forecast was
frozen. The persistent cohort is initialized from March-April 2025 SCE
observations; July 2026 public macro information is current to the forecast
cutoff, but the household survey state is not contemporaneous. See the
[canonical report](reports/current_ecology_report.md) for hashes and the full
result statement.

## Run It

Run the deterministic 12-household fixture in a clean clone. Its households,
history, and hash-validated origin snapshot are fully synthetic tracked inputs;
they contain no realized targets, private identifiers, or secrets:

```bash
make ecology-fixture
```

The 200-household replay is a separate private-local workflow. It defaults to
ignored `work/` household/history inputs, origin snapshot, and accepted response
cache, and makes no new model calls:

```bash
make ecology-current-replay
```

For a new untouched month, set `ORIGIN`, `AS_OF`, `ORIGIN_SNAPSHOT`,
`ECOLOGY_HOUSEHOLDS`, `ECOLOGY_HISTORY`, `ECOLOGY_CACHE`, and
`CURRENT_RUN_DIR`, build the origin snapshot, then run `make ecology-live-200`.
`ECOLOGY_BUNDLE` names the private frozen bundle used by the live canary. Live
household calls go only through Codex CLI.

Live household elicitation runs Codex in a fresh empty directory with shell,
local text-file, web, browser, app, memory, and multi-agent access disabled. The model
therefore receives the household card in the prompt but cannot inspect the repo or
the realization files. Cache records are bound to the provider, model, full prompt,
card, and isolation version.

Every run writes the normalized origin information, private household cards,
responses, feasible decisions, employer and credit ledgers, downside/median/upside
macro paths, accounting audit, event hashes, manifest, and a short report. Realized
targets are not loaded into prompts or scored during forecast creation.

When native outcomes become available, run:

```bash
make ecology-realize REALIZATIONS_CSV=path/to/realizations.csv
```

This writes a separate retrospective score bundle. The input row is keyed to the
one-month-ahead `target_month`; the command verifies the frozen artifacts first
and never rewrites the forecast manifest.

To diagnose period-by-period scale and direction before the prospective August
outcome arrives, run the current ecology recursively over the four available
post-cutoff historical origins:

```bash
make ecology-retrospective-live
```

This is explicitly retrospective, not confirmatory. It carries the median
simulated state forward after an unscored survey-seeded initialization transition,
opens only origin/history inputs during forecasting, loads first-release outcomes
only after all forecasts finish, and writes both long-form predicted-versus-actual rows and a chart. It uses
a separate cache and reads the frozen July-to-August run only to add its unscored
August marker.

Once the response cache exists, `make ecology-retrospective-replay` reconstructs
the same diagnostic with zero model calls.

The current retrospective result is a negative: on the three valid recursive
months, nominal-consumption RMSE is **5.48 percentage points** and the sign is
wrong in all three. The chained prediction falls from 100 to **86.40** while the
first-release PCE path rises to **102.14**. This is useful diagnosis, not evidence
that the economy is already an accurate macro predictor.

## Current Boundary

- 200 persistent anonymized SCE households are available locally.
- Their survey histories are append-only and availability dated.
- Balance sheets remain coarse mappings from survey groups, not measured SCF-linked
  household accounts.
- The employer and credit intermediary are intentionally aggregate.
- The present target is repeated one-month-ahead forecasting. Sectoral firms or
  additional institutions are justified only by stable errors across new origins.

See [CURRENT.md](CURRENT.md) for the exact current milestone and
[reports/current_ecology_report.md](reports/current_ecology_report.md) for the
canonical result, and [research_history.md](research_history.md) for the compact
record of retired work.

The checked-in current report is source- and hash-bound evidence from that run.
It does not bundle, reconstruct, or stand in for the private 200-household raw
inputs or accepted response cache.

## Verification

```bash
make check
make test
```

The previous weighted-demand economy and its tracked history remain available at
the Git tag `macro-v1-weighted-demand`.
