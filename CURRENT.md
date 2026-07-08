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

## Search Lane

Use tournament targets only when deliberately searching for a better economy:

```bash
make macro-tournament-dev
```

These are exploratory development surfaces. They should not be described as confirmatory results.

## Historical Evidence

Older behavior, persona, bridge, playground, validity, and Phase 4 targets are intentionally retained. They are not the active path, but they explain what failed, what was ruled out, and how the current incumbent was selected.

Do not delete old code or run artifacts merely because they are no longer the current route. Archive bulky or superseded outputs when needed; keep anything referenced by the report, compact evidence bundles, tests, or provenance logs.
