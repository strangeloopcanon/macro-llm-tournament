# Household-First Macro Ecology

This project builds a macro forecast from the bottom up. Two hundred persistent,
anonymized households come from the New York Fed Survey of Consumer Expectations
(SCE). Each receives its own survey history, an SCE-conditioned financial state,
and public information available at the forecast date. GPT-5.5 writes that
household's next-month beliefs and dollar policy. Deterministic code enforces
budgets, credit limits, production feasibility, inventories, and settlement.

```text
200 SCE household histories + matched SCF financial states + as-of public data
                                   |
                      one isolated GPT-5.5 call each
                                   |
           beliefs + employed/not-employed household dollar policies
                                   |
           deterministic budgets, credit, production, and settlement
                                   |
                 population-weighted macro demand forecast
                                   |
       unscored next-period firm response shadow (diagnostic only)
```

Firms and banks are not LLM agents in the current version. Production follows
expected sales with gradual inventory adjustment; credit and settlement are
mechanical. Respondent employment and wages are fixed so the active test isolates
household demand. A separate observability layer projects how that demand would
move next-period sales, output, required labor, partial employment adjustment,
and price pressure. It is a transparent mechanical shadow, not a closed feedback
loop or an empirical forecast.

## Current Result

The corrected ecology gets the direction of nominal consumption growth right in
all four historical one-month forecasts. Its RMSE is **0.61 percentage points**.
The current design does not reproduce the earlier contraction, but it is not yet
a good forecasting result: an origin-visible
nominal-spending drift scores **0.24 points**.

| Origin | Target | LLM household economy | First-release PCE | Routine drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | +0.01% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.02% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | +0.06% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | +0.16% | +0.71% | +0.48% |

The July 2026 origin is frozen for August at **+0.08%** nominal consumption
growth. August outcomes were unavailable and excluded. The replay reproduces all
200 accepted live responses with zero calls and matches the immutable live
reference.

The consumption mapping is explicit and load-bearing: executed household spending
is measured against a numerically fixed synthetic SCE-SCF recent-typical spending
anchor and interpreted as month-over-month nominal PCE growth. It is not linked
household-level spending growth. The internal gross-income residual is not mapped
to the national personal saving rate.

## Run It

The tracked 12-household fixture contains synthetic inputs and needs no private
data or model calls:

```bash
make ecology-fixture
```

The private-local 200-household replays use ignored SCE/SCF inputs and accepted
Codex CLI response caches:

```bash
make ecology-current-replay
make ecology-retrospective-replay
make ecology-observability
```

For a new month, set `ORIGIN`, `AS_OF`, `ORIGIN_SNAPSHOT`,
`ECOLOGY_HOUSEHOLDS`, `ECOLOGY_HISTORY`, `ECOLOGY_CACHE`, and
`CURRENT_RUN_DIR`, then run `make origin-snapshot` and
`make ecology-live-200`. Live household calls go only through Codex CLI in fresh
empty directories with shell, local files, web, browser, apps, memory, plugins,
and subagents disabled.

Every forecast writes its normalized origin card, household cards and responses,
feasible household decisions, employer and credit ledgers, one median macro path,
an accounting audit, event hashes, a manifest, and a report. Realizations are
loaded only after forecasts finish. Append a prospective realization with:

```bash
make ecology-realize REALIZATIONS_CSV=path/to/provenance_rich_realizations.csv
```

The realization file is long-form with one row per metric and the columns
`target_month,metric,value,source,source_url,vintage_date,release_date`. Release
dates must be after the frozen forecast cutoff. The append is published atomically
with the canonical input and hashes under the run's `realization_append/` folder.

`make ecology-observability` combines the banked historical and prospective runs
into two tidy diagnostic panels and a six-panel figure. It keeps observed outcomes,
LLM beliefs and intentions, deterministic execution, and the firm shadow in
separate source classes.

## Evidence Boundary

- The historical four-month run is diagnostic. Those dates may be in model
  knowledge.
- The prospective August run is frozen but not yet scored.
- The 200-household cohort is anchored in March-April 2025 SCE data; public macro
  information is current at each origin.
- Financial states are deterministic matches to public 2022 SCF households, not
  linked or contemporaneous accounts.
- Personal job-loss priors use same-wave SCE `Q13new`, the respondent's reported
  chance of losing the current job over 12 months. Missing answers stay missing.
  SCE `Q4new`, the chance that aggregate U.S. unemployment rises, remains a
  separate belief.
- The next problem is behavioral amplitude: households preserve spending and get
  its sign right, but underreact to ordinary nominal growth. The full surface
  shows the proximate mechanism: households intend deposit additions equal to
  roughly 13% to 19% of baseline monthly consumption while expecting income
  contraction.

See [CURRENT.md](CURRENT.md) for the exact milestone,
[reports/current_ecology_report.md](reports/current_ecology_report.md) for the
canonical result, and [research_history.md](research_history.md) for retired work.

## Verification

```bash
make check
make test
```

The previous weighted-demand economy is recoverable at Git tag
`macro-v1-weighted-demand`.
