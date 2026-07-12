# Household-First Macro Ecology

This repository builds a monthly microeconomy from real household survey states.
Each household gets one isolated LLM call. The LLM reports beliefs and intentions;
deterministic code enforces budgets, credit, production, employment, inventories,
and settlement. The aggregate is the forecast.

```text
survey-seeded SCE household state + own history + as-of public information
                              |
                    one isolated LLM call
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

The first full run freezes an August 2026 target path from a July 2026 origin and
200 separately elicited GPT-5.5 households. The median path has household
intentions at **-2.95%** consumption growth and feasible consumption at
**-4.08%**, with lower-liquidity households cutting more.

### Run facts

| Item | Value |
| --- | --- |
| Forecast origin | `2026-07-01` |
| Information cutoff | `2026-07-10` |
| One-month-ahead target | `2026-08-01` |
| Provider and model | `codex_cli` / `gpt-5.5` |
| Household responses | 200 accepted, one isolated response per household |
| Final execution | 200 replay hits, 0 fresh calls |
| Accounting | **PASS**, maximum residual `4.19e-09` |
| Replay | **PASS**, exact economy hash reproduced |

### Forecast paths

| Scenario | Consumption growth | Saving rate | Revolving-credit growth | Employment rate | Price growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downside | -8.14% | -2.53% | -6.03% | 90.96% | -0.99% |
| Median | -4.08% | -1.33% | -6.37% | 90.70% | -1.35% |
| Upside | -0.73% | 0.48% | -6.63% | 90.70% | -1.93% |

Population-weighted household responses imply median beliefs of **4.89%**
inflation, **-1.01%** income growth, and a **9.99%** one-year job-loss
probability. In the median economy, the 200 simulated household units execute
`$1,016,637` of consumption, `$169,080` of debt payments, and `$2,750` of new
borrowing.

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

Every run writes the normalized origin information, private household cards,
responses, feasible decisions, employer and credit ledgers, downside/median/upside
macro paths, accounting audit, event hashes, manifest, and a short report. Realized
targets are not loaded into prompts or scored during forecast creation.

When native outcomes become available, `make ecology-realize
REALIZATIONS_CSV=...` writes a separate retrospective score bundle. The input row
is keyed to the one-month-ahead `target_month`. The command verifies the frozen
artifacts first and never rewrites the forecast manifest.

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

The previous weighted-demand economy is preserved by the Git tag
`macro-v1-weighted-demand` and a hashed local archive under
`~/Downloads/llm-hank-docs/archive/v1/`.
