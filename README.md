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
200 separately elicited GPT-5.5 households. The median path has household intentions at -2.95% consumption
growth and feasible consumption at -4.08%, with lower-liquidity households cutting
more. All three paths pass the household, employer, credit, and stock-flow audits.

That is evidence that the ecology runs and produces heterogeneous, internally
consistent macro paths. It is not evidence that the forecast is accurate; the
August outcome surface was unavailable and excluded when the forecast was frozen.
The persistent cohort is initialized from March-April 2025 SCE observations;
July 2026 public macro information is current to the forecast cutoff, but the
household survey state is not contemporaneous.

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
