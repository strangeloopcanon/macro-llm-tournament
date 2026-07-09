# Pre-registration: Prior-Update Environment v2 (Monthly Vintage Cards)

Status: LOCKED before any live elicitation. Written 2026-07-08 (UTC-7), after the December 2024 prior-update extension failed its locked persistence gate and before any re-elicitation.

## Problem

The December 2024 prior-update extension (`outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_dec2024_extension/`) failed the locked prior-update gate: mean RMSE improvement versus persistence was `-0.0084` against the `+0.02` threshold. The failure is concentrated in one target: inflation scored `-0.1598` while income (`+0.0748`) and unemployment risk (`+0.0599`) both passed.

## Root-cause diagnosis (from banked artifacts only, no new calls)

Real respondents raised year-ahead inflation expectations by `+0.87`pp on average between the November and December 2024 waves. The model lowered them slightly (`-0.12`pp mean, bias `-0.99`).

The model's behavior was rational for the information it was shown. The December environment card was built from the quarterly SPF origin `2024:Q4`, as-of `2024-11-15`. That card showed CPI YoY at `2.5763` (down from `2.9236` on the previous card), near-zero inflation news pressure (`0.0254`), and a synthetic sentiment index of `94.0`. It contained none of the information that moved real households between mid-November and December 2024: the November CPI reacceleration (released 2024-12-11), the post-election tariff news flow, and the December rise in published household inflation expectations. The card also uses synthetic formulas for sentiment and aggregate expectations even though real vintage series (`UMCSENT`, `MICH`) already exist in the repo's vintage panel.

In short: a monthly survey panel was fed quarterly-stale environment cards, and the December wave is the first leg where the stale card and the true information set pointed in opposite directions.

## Pre-registered fix (general rule, not December-specific)

1. **Monthly as-of vintage cards.** Each SCE wave gets an environment card built from an ALFRED vintage as-of the 15th of its own survey month (December 2024 wave: as-of `2024-12-15`; January 2025 wave: as-of `2025-01-15`). SCE fields respondents throughout the month, so a mid-month vintage is a fair average information set. This rule applies uniformly to every wave prepared under v2; it is not tuned per wave.
2. **Real household-salient series replace synthetic formulas.** `sentiment_index` is taken from vintage `UMCSENT` (latest observation as-of the card date) instead of the synthetic formula. `aggregate_expected_inflation_1y` is taken from vintage `MICH` (University of Michigan median 1-year expected inflation) instead of the SPF-derived survey median. All other card fields keep their existing formulas, computed from the monthly vintage.
3. **Environment provenance label** becomes `fred_alfred_monthly_vintage_context_by_sce_wave_v2` so v2 rows can never be confused with v1 rows.
4. **Everything else is unchanged**: prompt template, prompt version semantics, `prior-mode empirical`, `feedback-mode none`, `date-mode relative`, target fields, provider `codex_cli`, model `gpt-5.5`, and the locked gate thresholds:
   - mean update correlation >= `0.10`
   - mean direction accuracy >= `0.55`
   - median amplitude ratio in `[0.25, 2.00]`
   - mean RMSE improvement versus persistence >= `+0.02`

## Pre-registered runs and claim scopes

- **Run A (diagnostic, not confirmatory): December 2024 re-elicitation.** Same 81 respondents, same priors (their real November answers), v2 environment card. Claim scope is `diagnostic_information_set_fix`: the December update surface has already been observed once, so a pass here can only support the mechanism claim ("the failure was the information set, not the updater"), never a fresh validation claim.
- **Run B (fresh gate test): January 2025 leg.** The subset of the 81 respondents with complete January 2025 answers (67 respondents at preparation time), priors = their real December answers, v2 environment card. This wave's updates have never been elicited or scored. The locked gate is evaluated once on this leg. If it clears, the prior-conditioned updater claim extends to a third consecutive update leg under the corrected information rule. If it fails, we report that and do not re-run it.

Budget: at most `81 + 67 = 148` live GPT-5.5 calls via `codex_cli`, plus retry headroom within each runner's `--max-live-calls` cap. No other live calls.

## Pre-registered decision rules

1. The gate thresholds above are unchanged from `PRIOR_UPDATE_THRESHOLDS` and are not renegotiable after seeing results.
2. **Combined panel rebuild rule:** if Run A (December v2) clears the gate, the stitched October-November-December panel used by the aligned Phase 4 surface is rebuilt from the v2 December leg, and downstream Phase 4/tournament replays on that surface are re-run and reported with the caveat replaced by the diagnostic-pass status. If Run A fails again, the aligned surface keeps its "exploratory, ungated December leg" caveat and the v1 December leg remains the surface of record.
3. **January extension rule:** if Run B clears the gate, an October-through-January stitched panel may be built as an additional retrospective alignment surface. If Run B fails, no January surface is built.
4. Whatever happens, both runs are reported (pass or fail) in the canonical report with their claim scopes, and no third elicitation of either wave is permitted under this pre-registration.
5. The next newly scoreable FRED month remains reserved for confirmatory Phase 4 scoring and is not touched by any of this.

## Honesty note

We are re-running a wave whose failure we have already seen, after diagnosing the cause. That is why Run A is permanently capped at diagnostic status. The only leg that can carry fresh evidential weight is Run B (January 2025), whose respondent answers have never been seen by any elicitation in this project, and whose gate is being declared here before the run.
