# Current Project Surface

This file is the clean entry point for the current state of the repo.

## Current Best Economy

Run:

```bash
make macro-incumbent-v1
```

That replays the promoted current-best economy from:

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

## Confirmatory Surface

The first locked fresh FRED confirmatory score has been spent:

```bash
make macro-confirmatory-v1
```

The rerun guard now blocks that target as already spent. The result is negative:
both locked LLM-economy candidates lose to the adaptive diagnostic on the
`2026-02-15` as-of surface. The compact tracked evidence bundle is:

```text
reports/macro_confirmatory_fred_2026_02_v1_evidence.json
```

The spent-surface registry is:

```text
reports/macro_tournament_confirmatory_registry.json
```

## Search Lane

Use tournament targets only when deliberately searching for a better economy:

```bash
make macro-tournament-dev
```

These are exploratory development surfaces. They should not be described as confirmatory results.

## Historical Evidence

Older behavior, persona, bridge, playground, validity, and Phase 4 targets are intentionally retained. They are not the active path, but they explain what failed, what was ruled out, and how the current incumbent was selected.

Do not delete old code or run artifacts merely because they are no longer the current route. Archive bulky or superseded outputs when needed; keep anything referenced by the report, compact evidence bundles, tests, or provenance logs.
