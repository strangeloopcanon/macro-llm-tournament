# Current Project Surface

This file is the clean entry point for the current state of the repo.

## Current Recursive Development Winner

The current economy starts with 81 real SCE household states, asks GPT-5.5 to
update their beliefs from each month's as-of information, converts those
beliefs into spending through empirical bridge v4, enforces household and
aggregate accounting, and carries the resulting state into the next origin. An
adaptive-expectations version of the same economy is scored beside it as a
strong dumb benchmark.

The locked January-May 2026 development tournament uses January as warmup and
scores February-May against 10 common-month first-release targets per origin.
The winner is
`empirical_v4_moderate_policy_full_100_smooth_085`: LLM MacroScore `0.549789`
versus adaptive `0.548667`. Lower is better, so adaptive still leads by `0.2%`;
the bootstrap interval crosses zero. The winning mechanism improves the base
LLM economy by `2.61%`, mainly by assimilating the observed policy-rate state
at each rolling origin and smoothing the next policy-rate transition.

This is developmental model selection, not confirmation. The compact tracked
evidence is:

```text
reports/dynamic_macro_development_v1_evidence.json
```

The local run is:

```text
outputs/dynamic_macro_policy_partial_gpt55_live_v4/
```

Reproduce the selected economy from the banked five-period records with zero
provider calls:

```bash
make dynamic-macro-incumbent-replay
```

That writes `outputs/dynamic_macro_incumbent_replay/` and must reproduce the
winner's scores and accounting contract.

The canonical reader-facing report is:

```text
reports/macro_simulation_report.md
```

The historical `make macro-incumbent-v1` replay remains available for the older
Phase 4 tournament. It is not the current recursive winner.

## Locked June Confirmation

The winner, provider, model, household panel, empirical bridge, replay records,
target contract, and input hashes are frozen in:

```text
configs/dynamic_macro/confirmatory_june_2026_v1.json
```

`make dynamic-macro-confirmatory-june` is a one-shot command. It currently
fails before writing a receipt because the complete June first-release bundle
does not yet exist. The full 10-target June surface is expected to become
available with the [BEA June Personal Income and Outlays release on July 30,
2026](https://www.bea.gov/news/schedule/). Once a valid bundle exists, any started run spends
the surface; failed runs cannot be renamed and retried.

These targets depend on ignored local `work/` and `outputs/` artifacts. They are
runnable in this research workspace, not from a clean clone.

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
