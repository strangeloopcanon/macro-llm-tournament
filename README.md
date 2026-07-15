# Household-First Macro Ecology

This project asks whether LLM-updated household beliefs can produce useful macro
forecasts and counterfactuals inside an explicit, accounting-constrained economy.
Real survey data supplies household heterogeneity. Each household gets one tool-isolated
LLM elicitation to report its beliefs and intended choices; deterministic code then
enforces budgets, credit, production feasibility, inventories, and settlement.
Survey weights determine each household type's population mass. The current forecasting
experiment holds wages and respondent employment fixed so the aggregate demand path
comes from household choices rather than an uncalibrated labor market.

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
 household actions ---> aggregate demand ---> production and inventories
         |
         +------ deterministic credit and settlement
```

Firms and banks are not role-play agents in the current version. Production follows
expected sales with gradual inventory adjustment; credit and settlement are mechanical.
That keeps the behavioral test focused on the 200 households.

## Current Result

The corrected full run gave 200 GPT-5.5 households only information available by
July 10, 2026, and froze their path for August before August outcomes were known.
The median path predicts **+0.28%** nominal consumption growth. A separate four-origin
historical diagnostic gets the consumption direction right in **4/4** months, with
RMSE **0.47 percentage points**. That is a large improvement over the retired ecology,
but a simple origin-visible nominal-spending drift remains better at **0.24 points**.

### Run facts

| Item | Value |
| --- | --- |
| Forecast origin | `2026-07-01` |
| Information cutoff | `2026-07-10` |
| One-month-ahead target | `2026-08-01` |
| Provider and model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted, one tool-isolated response per household |
| Reproducibility check | 200 replay hits, 0 fresh calls, immutable live reference matched |
| Accounting | **PASS**, maximum residual `1.12e-08` |
| Replay | **PASS**, exact replay-equivalence hash reproduced |

### Forecast paths

| Scenario | Consumption growth | Saving rate | Revolving-credit growth | Employment rate | Price growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downside | 0.28% | 16.73% | -2.06% | 66.20% | 0.00% |
| Median | 0.28% | 16.73% | -2.06% | 66.20% | 0.00% |
| Upside | 0.28% | 16.73% | -2.06% | 66.20% | 0.00% |

The population-weighted mean of household p50 responses implies **5.01%**
inflation, **-0.64%** income growth, and a **0.74%** next-month job-loss
probability. In the median economy, 200 household types representing 200
population-equivalent units execute `$3,118,296` of consumption, `$22,247` of
debt payments, and `$2,244` of new borrowing. Scenario paths coincide in this
fixed-labor diagnostic; they are not an uncertainty interval.

### Evidence boundary

| This run shows | This run does not yet show |
| --- | --- |
| A 200-household demand economy runs end to end. | That its August forecast is accurate. |
| Household heterogeneity affects feasible consumption and balance sheets. | That one origin validates the economy's dynamics. |
| Production, inventories, credit, and settlement have explicit counterparties. | That firms or banks need LLM decision agents. |
| All three scenarios satisfy household, employer, credit, and stock-flow accounting. | That SCE-conditioned SCF matches are linked household accounts. |

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
outcome arrives, run the current ecology independently over the four available
post-cutoff historical origins:

```bash
make ecology-retrospective-live
```

This is explicitly retrospective, not confirmatory. Every origin restarts from the
same fixed SCE-SCF household anchor, receives only origin-visible public information,
and is scored as a one-month forecast. Simulated balances and errors never enter the
next origin. Realizations load only after all forecasts finish.

Once the response cache exists, `make ecology-retrospective-replay` reconstructs
the same diagnostic with zero model calls.

The current retrospective result is promising but not yet competitive. Nominal
consumption has the right sign in **4/4** months and RMSE **0.47 percentage points**.
The compounded LLM path reaches **100.90** versus first-release PCE at **102.63**.
The origin-visible routine anchor reaches **101.75** and has lower RMSE (**0.24**),
so the LLM households still underreact to normal nominal spending growth.

## Current Boundary

- 200 persistent anonymized SCE households are available locally.
- Their survey histories are append-only and availability dated.
- Financial states are deterministic SCE-conditioned matches to SCF 2022 households,
  not linked or contemporaneous household accounts.
- Wages and respondent employment are fixed in the active forecast diagnostic.
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
