# Current Project Surface

The active system is a 200-household LLM demand economy with a one-step producer,
income, and household feedback loop.

```text
real household histories + repaired financial states + dated public information
                                  |
                                  v
                    GPT-5.5 household decisions
                                  |
                                  v
               code-enforced budgets and settlement
                                  |
                                  v
            demand -> output -> employment -> wages/income
                                  |
                                  v
                    next household decision
```

## Current Verdict

The simulator works, but the repaired household economy forecasts poorly.

The earlier v23 population was not trustworthy as a 200-household aggregate. One
survey household had been paired with a $21.9 million SCF donor and contributed
31.3% of baseline spending; the effective spending sample was about nine households.
V24 keeps each SCE household's own income, restricts SCF donors to one-half through
twice that income, and scales donor flows and stocks to the SCE income anchor.

The repair succeeds mechanically:

- largest household spending contribution: **2.36%**;
- top-five contribution: **8.87%**;
- effective spending household count: **140.4**;
- all 200 donor matches stayed within the income bound;
- no demographic match had to be relaxed.

It does not improve prediction. The four v24 nominal-consumption forecasts are
`-1.08%, -1.00%, -1.06%, -1.05%`, against first-release PCE growth of
`+0.48%, +0.90%, +0.51%, +0.71%`.

| Diagnostic | Repaired household economy | Visible-data baseline |
| --- | ---: | ---: |
| Nominal PCE RMSE | 1.703 pp | 0.243 pp |
| Mean bias | -1.696 pp | -0.215 pp |
| Direction | 0/4 | 4/4 |

About 78-81% of represented households cut spending at each historical origin.
Only about 0.7% are materially constrained by settlement, so the pessimism comes
mainly from the decisions authored by the model, not from the accounting code.

The frozen July-to-August forecast is **-1.12%**, against BEA's first-release
**+0.9%**. Its absolute error is **2.021 pp**, versus **0.187 pp** for the saved
baseline. Four BEA targets are scored; five other macro targets remain unscored. In the
unscored recursive run, period-two spending falls another **0.80%**. The dynamic
system executes and replays exactly, but its current household behavior is too
cautious.

## One-Page Scorecard

[`reports/current_macro_jepa_comparison.png`](reports/current_macro_jepa_comparison.png)
is the complete nine-panel scorecard. It plots first-release actual outcomes, the
repaired LLM household economy, the locked Event-JEPA predictions, the origin-visible
lag-one baseline, and frozen unscored points where available.

That image is the August 23 historical snapshot. The
[August score chart](reports/august_2026_consumption.png) now shows the observed PCE
outcome. No saved JEPA August point exists. See the
[recovery report](reports/recovery_and_august_score.md) for verified scores and sources.

The Event-JEPA remains a separate macro-only diagnostic. A fresh unit audit found that
REVOLSL and TOTALSL changed from billions to millions inside the archived history and
that the exact 2026 cards could reintroduce the raw unit. V3 canonicalizes both inputs
and fails closed if the old unit remains. On the four 2026 nominal-PCE targets it now
has RMSE **0.256 pp**, close to the visible-data baseline's **0.243 pp**. On its more
informative locked 2024-2025 block it scores **0.458 pp**, still behind lag-one's
**0.331 pp** and the historical median's **0.234 pp**.

JEPA did win the saved 24-origin payroll comparison: **0.054 pp**, versus **0.060 pp**
for lag-one. This is a target-specific gain; the overall result is mixed.

## What Works

- Household histories and financial states are origin-safe and hash-bound.
- Every LLM call is isolated and goes through `codex_cli`.
- Household budgets, credit, goods, bank stocks, and counterparties reconcile.
- Producer output, inventory, employment, wages, and household income feed into the
  next period.
- Live and replay runs are equivalent and source-bound.
- Reporting rejects any attempt to combine runs with a different household file.

## What Does Not Work Yet

The household action layer turns ordinary public information into broad spending
cuts. Correcting the household population makes that failure clearer. A further audit
also found that SCF's composite revolving-payment measure had been treated as the
minimum due on card debt. V5 preserves that observed measure but uses the simulation's
actual 3%/$25 card contract. The same 200 households and donors are retained; 109
payment fields change. This input is ready, but it has not been used for a new LLM run.

The next step is an offline purchase-response dataset and baseline study. The restored
SCE spending supplement contains expected purchases and later self-reported purchases.
Resolve category mapping, weights, publication dates, and test splits before adding
model calls. The current ecology cache has dollar policies, not matched purchase
probabilities. See the household response inventory before another simulation run.

## Saved Evidence

All archived `work/` files were restored on October 14 and verified against the
saved archive receipt. The JEPA checkpoint and all 37 manifest artifacts match.
The [household response inventory](reports/household_response_readiness.md) identifies
real purchase targets and the remaining model-output and weighting gaps.

The raw files were then returned to Strange Lab's `vei-shared` Drive. Full downloaded
byte comparison passed before local `work/` and `outputs/` bulk was removed. Their
restore pointers remain. The paths below are inside the cloud archives, except for
the small repository reports. Restore metadata accompanies each private archive.

- `work/persona_beliefs/persistent_household_scale_v5/` (corrected input, not yet live-scored)
- `outputs/household_ecology_200_july_v24_current/`
- `outputs/household_ecology_retrospective_2026_01_04_v24/`
- `outputs/household_ecology_feedback_200_july_v24/`
- `outputs/household_ecology_observability_v24/`
- `work/macro_jepa_v3/`
- `reports/current_macro_jepa_comparison.png`
- `reports/current_macro_jepa_evidence.json`
