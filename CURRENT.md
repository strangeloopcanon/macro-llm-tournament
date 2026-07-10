# Current Project Surface

This file is the clean entry point for the current state of the repo.

## Current Recursive Development Winner

The current economy starts with 200 persistent, population-weighted real SCE
household states. GPT-5.5 updates their beliefs from each month's as-of
information in two fixed batches of 100. Empirical bridge v4 converts belief
changes into spending pressure; deterministic code executes household choices,
enforces household and aggregate accounting, and carries the resulting state
into the next origin. An adaptive-expectations version of the same economy is
scored beside it as a strong dumb benchmark.

The current promotion comes from a pre-registered corrected-weight comparison,
not another mechanism search. Both cohorts use the same January-May 2026
development bundle, prompts, model, behavior bridge, feedback, policy-state
assimilation, and score mapping:

| Cohort | LLM MacroScore | Adaptive | LLM minus adaptive | Effective sample size | Largest household weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical uncorrected 81 | `0.549789` | `0.548667` | `+0.001122` | `55.34` | `4.72%` |
| Corrected 81 | **`0.546550`** | `0.548858` | `-0.002308` | `46.53` | `6.41%` |
| Corrected 200 | `0.548996` | `0.548379` | `+0.000617` | `123.58` | `2.15%` |

Lower is better. Increasing the ecology from 81 to 200 did not improve the
absolute LLM score: the corrected 81 path is better by `0.45%`. That difference
is inside the locked 1% near-tie band, so the pre-registered rule promotes 200
households for substantially better population coverage. The 200-household LLM
path trails adaptive by `0.11%`; its origin-block interval
`[-0.00655, 0.00757]` crosses zero. This remains developmental evidence, not a
confirmed predictive win.

This is developmental model selection, not confirmation. The compact tracked
evidence is:

```text
reports/dynamic_macro_household_scale_v1_evidence.json
```

The local run is:

```text
outputs/dynamic_macro_household_scale_incumbent_replay_v1/
```

Reproduce the selected economy from the banked five-period records with zero
provider calls:

```bash
make dynamic-macro-incumbent-replay
```

That writes `outputs/dynamic_macro_household_scale_incumbent_replay_v1/` and
must reproduce the promoted 200-household scores, two-batch belief records, and
accounting contract with zero provider calls.

The canonical reader-facing report is:

```text
reports/macro_simulation_report.md
```

The historical `make macro-incumbent-v1` replay remains available for the older
Phase 4 tournament. It is not the current recursive winner.

## Locked June Confirmation

The promoted cohort, adjusted population weights, provider, model, two-batch
layout, prompts, empirical bridge, replay records, target contract, executable
source, and input hashes are frozen in:

```text
configs/dynamic_macro/confirmatory_june_2026_v1.json
```

`make dynamic-macro-confirmatory-june` is a one-shot command. It currently
fails before writing a receipt because the complete June first-release bundle
does not yet exist. The full 10-target June surface is expected to become
available with the [BEA June Personal Income and Outlays release on July 30,
2026](https://www.bea.gov/news/schedule/). Once a valid bundle exists, any
started run spends the surface; failed runs cannot be renamed and retried. The
five development periods replay in 10 banked batches, and June permits exactly
two accepted GPT-5.5 batches plus two schema retries per batch.

These targets depend on ignored local `work/` and `outputs/` artifacts. They are
runnable in this research workspace, not from a clean clone. The historical
uncorrected 81 path remains evidence only and is not eligible for promotion.

## Held-Out February Diagnostic

The `2026-02-15` score date has been spent and must not be rerun:

```bash
make macro-confirmatory-v1
```

The result is negative: both LLM-economy candidates lose to their adaptive
twins. A July 2026 integrity review found that this run used revised current
FRED observations rather than frozen February vintages, held December 2024
beliefs forward, and scored four available targets with one observation each.
The spec, lock code, registry, and result also entered git together. It is
therefore a one-shot held-out current-vintage diagnostic, not independently
pre-registered confirmatory evidence. The compact tracked evidence bundle is:

```text
reports/macro_confirmatory_fred_2026_02_v1_evidence.json
```

The spent-surface registry is:

```text
reports/macro_tournament_confirmatory_registry.json
```

Future confirmatory runs must use frozen, hashed vintage inputs; declare a
complete target contract; reserve the score date atomically before scoring; and
record a clean pre-result commit.

## Search Lane

Use tournament targets only when deliberately searching for a better economy:

```bash
make macro-tournament-dev
```

These are exploratory development surfaces. The current mechanism search is
closed; do not tune another candidate on January-May before the June lock runs.

## Historical Evidence

Older behavior, persona, bridge, playground, validity, and Phase 4 targets are intentionally retained. They are not the active path, but they explain what failed, what was ruled out, and how the current incumbent was selected.

Do not delete old code or run artifacts merely because they are no longer the current route. Archive bulky or superseded outputs when needed; keep anything referenced by the report, compact evidence bundles, tests, or provenance logs.
