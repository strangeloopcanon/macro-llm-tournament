# Current Project Surface

This file is the clean entry point for the current state of the repo.

## Replayable Development Incumbent

Run:

```bash
make macro-incumbent-v1
```

That replays the v1 development incumbent from:

```text
configs/macro_tournament/incumbent_v1.json
```

It writes:

```text
outputs/macro_incumbent_v1/
```

The compact tracked evidence bundle is:

```text
reports/macro_incumbent_v1_evidence.json
```

The canonical reader-facing report is:

```text
reports/macro_simulation_report.md
```

This target depends on ignored local `work/` and `outputs/` artifacts. It is
runnable in the research workspace, not from a clean clone. The later v3
development search selected a different candidate after the December panel was
rebuilt; no candidate is currently promoted as a validated best economy.

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

These are exploratory development surfaces. They should not be described as confirmatory results.

## Historical Evidence

Older behavior, persona, bridge, playground, validity, and Phase 4 targets are intentionally retained. They are not the active path, but they explain what failed, what was ruled out, and how the current incumbent was selected.

Do not delete old code or run artifacts merely because they are no longer the current route. Archive bulky or superseded outputs when needed; keep anything referenced by the report, compact evidence bundles, tests, or provenance logs.
