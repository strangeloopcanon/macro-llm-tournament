# Macro LLM Simulation: Current Result

## Bottom Line

The belief engine result survived the harder audit. On the 147-card date-free vintage macro split, raw GPT-5.5 and GPT-5.4 still beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate. GPT-5.5 remains the best source on the split.

The stronger statement is now this: the models are not just remembering exact realized values. In the live recall audit, both models gave zero usable exact realized-value recalls across the 147 cards. They do show some memory of broad macro regimes, but the qualitative recall edge is small. That leaves the belief forecast result standing as a real signal, while keeping the contamination claim narrow.

The agent-behavior claim is still not done, but the diagnostic is better now. The behavior economy passes accounting and mechanism checks, and the live synthetic-SCE persona panel clears demographic-gradient and spread checks. The behavior gate now separates the original selection targets from the lottery-windfall holdout, instead of pooling them into one headline row. On the selection split, raw GPT-5.5 still loses to the liquidity rule. On the 5-target lottery holdout, raw GPT-5.5 beats the best rule baseline. The 50 percent liquidity-prior blend fails the holdout. A new primitive-to-action path is interpretable and passes its sign audits, but selection calibration overfits: it slightly beats the rule on the selection split and then fails the lottery holdout.

## What Was Run

- Vintage OOS forecast split: 147 hidden-target, date-free macro cards.
- Models: GPT-5.5 and GPT-5.4 through Codex live calls.
- Baselines: no-change, rolling mean, rolling trend, AR(2), recursive least squares.
- Recall audit: exact realized-value recall plus qualitative path recall on the same 147 cards.
- Belief-structure audit: underreaction, extrapolation, direction, confidence, and error structure.
- Persona/SCE panel: 54 synthetic-enriched SCE-style respondents, 2 models, 108 total model responses: 72 cache hits from the prior live panel plus 36 new Codex live calls.
- Behavior economy: HANK-lite demand economy with deterministic accounting and bounded belief inputs, plus a live GPT-5.5 behavior gate scored against rule baselines. The current behavior gate scores raw allocation prompts, fixed blend ablations, a fixed primitive-control kernel, and a selection-calibrated primitive-to-action path on the same six scenarios. Raw allocation prompts were replayed from cache; the v2 primitive schema used six fresh Codex live calls, then regenerated from cache for the final report.

The test targets run from `2020-01-01` to `2025-02-01`. Against the user-supplied GPT-5.4 cutoff of `2025-08-31`, `0/147` cards are post-cutoff. Against the Codex GPT-5 cutoff of `2024-09-30`, `14/147` cards are post-cutoff. So this is a hidden-target, date-free vintage test, not a clean post-cutoff test for the frontier models.

## Forecast Performance

| Source | Test WNAE | Status |
| --- | ---: | --- |
| GPT-5.5 raw | 0.8945 | Best overall |
| GPT-5.4 raw | 0.9064 | Pass |
| No change | 1.1260 | Best deterministic baseline |
| Rolling trend | 1.2668 | Baseline |
| Rolling mean | 1.2790 | Baseline |
| GPT-5.4 calibrated | 1.3933 | Fails |
| GPT-5.5 calibrated | 1.8851 | Fails |
| AR(2) | 2.4307 | Baseline |
| Recursive least squares | 4.2816 | Baseline |

GPT-5.5 lowers mean origin-level loss by `0.2315` versus the best deterministic baseline, a `20.56%` improvement. The origin-cluster bootstrap interval is `[0.0688, 0.4153]`, with `99.82%` of bootstrap draws positive.

In raw error terms, GPT-5.5 has RMSE `3.4974` and MAE `1.5805`; GPT-5.4 has RMSE `3.6081` and MAE `1.6591`; no-change has RMSE `3.6756` and MAE `1.8013`. Direction accuracy is `69.39%` for both GPT models versus `6.80%` for no-change.

The validation-fitted residual calibration is not working. It makes held-out forecasts worse. The empirical signal is in the raw belief forecasts.

## Where It Works

The cleanest family-level win is real consumption growth. GPT-5.5 beats no-change by `29.96%`, with bootstrap mean loss reduction `[0.2786, 1.1281]` and a DM-style p-value of `0.0033`.

Output, policy rates, and unemployment move in the right direction, but less decisively. GPT-5.5 improves output by `21.88%` versus no-change, policy rate by `33.82%` versus rolling trend, and unemployment by `20.00%` versus no-change; those bootstrap intervals still cross zero.

Inflation remains the obvious miss. Rolling trend beats GPT-5.5 on inflation growth (`0.2626` versus `0.2904`). Saving and sentiment are weak wins at best.

## Recall Audit

Exact realized-value recall was zero. Both GPT-5.5 and GPT-5.4 returned no usable numeric realized values across the 147-card live audit. That is the important contamination check: the models did not appear to be retrieving the hidden target table directly.

Qualitative path recall is not zero, but it is modest:

| Model | Direction Accuracy | Level Accuracy | Turbulence Accuracy | Mean Card Accuracy | Base Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 | 0.5986 | 0.5578 | 0.7619 | 0.6395 | 0.6213 |
| GPT-5.4 | 0.5714 | 0.5306 | 0.7211 | 0.6077 | 0.6213 |

This says the models know something about broad macro eras, especially turbulence, but not enough for the result to collapse into simple remembered outcomes. GPT-5.5 is only `+0.0181` above the qualitative majority baseline on mean card accuracy; GPT-5.4 is `-0.0136` below it.

## Belief Structure

The raw GPT forecasts do not look like a trivial trend extrapolator. GPT-5.5 has the closest-to-zero underreaction slope among the scored sources (`-0.0279`), while GPT-5.4 is `-0.3226`; rolling mean and rolling trend are around `-1.13` and `-1.15`.

The GPT forecasts also avoid the worst extrapolation behavior. The slope of forecast change on the recent signal is `-0.1940` for GPT-5.5 and `-0.2764` for GPT-5.4, versus `-1.0000` for rolling mean and `-1.4819` for AR(2).

Confidence carries some signal: higher-confidence GPT forecasts have lower absolute errors than lower-confidence forecasts. The confidence-error correlation is negative for both raw GPT models (`-0.5453` for GPT-5.5 and `-0.5646` for GPT-5.4). The system should use that information, but not treat the current calibration layer as solved.

## Persona/SCE Cross-Section

The expanded live persona panel ran on 54 synthetic-enriched SCE-style respondents across GPT-5.5 and GPT-5.4. It reused 72 cached responses from the prior run and added 36 fresh Codex live calls, giving 108 total model responses. The panel is anchored to public aggregate survey beliefs and vintage macro context. It is useful for wiring and scoring the ecology; it is not real respondent-level microdata.

The expanded panel fixes the old coverage problem. The previous 36-row run had no high-income/high-liquidity group, so six income contrasts were skipped. The current run scores the full synthetic grid: `24/24` demographic contrasts are scoreable, with zero skipped contrasts.

The good part: demographic gradients and spread now clear cleanly. The scoreable sign rate is `24/24 = 100.00%`, the median within-variance ratio is `1.1025`, and the maximum cross-model common-core correlation is `0.8912`, below the `0.95` failure threshold.

The bad part: distribution shape still fails. The panel verdict remains `partial_distribution_failure`. The maximum KS statistic is `0.7407`, above the `0.35` threshold. Unemployment expectations are the main problem: the synthetic target mean is `4.4262`, while GPT-5.4 predicts `5.3389` and GPT-5.5 predicts `5.2241`. Inflation is also shifted upward: the synthetic target mean is `3.6045`, while GPT-5.4 predicts `3.8889` and GPT-5.5 predicts `4.1630`. GPT-5.4 is closer on real income growth; GPT-5.5 undershoots it.

So the persona layer now has useful structure, but it is not calibrated enough to be the behavioral engine's empirical input layer.

## Behavior Economy

The HANK-lite behavior economy still passes its lab gate. Bounded beliefs feed into deterministic code that owns consumption, saving, debt repayment, liquidity, aggregation, inflation, policy feedback, and accounting.

The behavior layer passes mechanism checks: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and household budget identities. That means the macro sandbox is playable and accounting-safe.

The behavior gate now emits split-separated baseline comparisons against the existing rule controls: liquidity rule, flat 30 percent rule, and permanent-income rule. The old pooled aggregate row is gone. An `ALL` row now means all targets within one evaluation split, never selection plus holdout.

That change matters. On the 11 original selection-split aggregate targets, raw GPT-5.5 loses to the liquidity rule: range RMSE `0.0762` versus `0.0560`. On the 6 cell-level EIP MPC targets, raw GPT-5.5 also loses: `0.0387` versus the liquidity rule at `0.0272`. The earlier pooled aggregate win was therefore not a general behavior win; it was the lottery holdout swamping the selection split.

The lottery holdout is still encouraging, but it is narrow. On the 5 Fagereng-Holm-Natvik lottery-windfall targets, raw GPT-5.5 has range RMSE `0.0844`, beating the best rule baseline, the flat 30 percent rule, at `0.2608`. The pre-specified 50 percent liquidity-prior blend fails the same holdout: `0.3857` versus `0.2608`. So the blend is demoted, and the useful behavioral signal is in raw GPT-5.5's out-of-domain response to lottery shock size and liquidity heterogeneity.

The primitive-to-action run was added to make that signal usable inside a simulated economy. GPT-5.5 now emits bounded primitives: perceived job-loss risk, expected income growth, precautionary saving motive, liquidity stress, debt-repayment urgency, durable purchase pull-forward, clipped shock size, log transfer-to-income shock size, and confidence. The deterministic kernel maps those primitives into spending, debt repayment, and liquid saving. The v2 primitive schema fixes the old shock-size bottleneck by adding `shock_size_log_income_ratio`, so lottery-scale windfalls no longer all collapse to the same clipped value.

The primitive path clears its sign audit. Liquidity stress raises MPC, precaution raises liquid saving, job risk raises liquid saving, debt urgency raises repayment, larger windfalls lower MPC, low-liquidity EIP cells spend more than high-liquidity cells, and small lottery windfalls have higher MPC than large lottery windfalls.

The calibrated primitive kernel is not the answer yet. It is fit only on `behavior_selection_v1`, using aggregate selection targets as the objective, and then locked before the holdout is rescored. It improves the selection aggregate surface from `0.5588` to `0.0539`, narrowly beating the liquidity rule's `0.0560`. It also improves the cell EIP surface to `0.0152`, but that is still selection-split evidence, not holdout evidence. On the lottery holdout it fails badly: `0.8510` versus the flat rule at `0.2608`. The fixed log-shock primitive control does better on the holdout (`0.1641`), but because that control was introduced after seeing the holdout problem, it is a diagnostic clue, not a clean confirmatory result.

It does not yet mean the simulated economy predicts real macro behavior better than a strong empirical model. The missing bridge is still behavior validation: the next calibrated behavior mechanism has to be pre-specified, scored separately from raw LLM behavior, and tested on targets it did not select against.

## Current Claim

The sendable claim is:

> On a 147-card held-out, date-free vintage macro split, raw GPT-5.5 and GPT-5.4 belief forecasts beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate. The win survives origin-cluster bootstrapping. Live recall probes show zero exact realized-value recall and only modest qualitative path recall. The behavior economy is accounting-safe and mechanism-complete. The behavior evidence is split: raw GPT-5.5 loses to the liquidity rule on the original selection targets and the cell-level EIP MPC surface, but beats rule baselines on a 5-target lottery-windfall holdout. The fixed 50 percent liquidity-prior blend fails that holdout. The primitive-to-action path is now interpretable and sign-correct, but the selection-calibrated kernel overfits and fails the lottery holdout. The persona/behavior layer still needs real microdata and a behavior kernel that generalizes before we can claim broad macro-agent predictive validity.

The next claim is not ready:

> LLM-based macro agents predict broad real-world macro behavior better than strong empirical baselines.

## Next Gate

The next work should not add more agent theater. It should make the belief-to-behavior layer harder to fool.

1. Replace the synthetic persona panel with real respondent-level microdata, or treat the current 54-row panel strictly as a wiring fixture. The synthetic coverage issue is fixed; the empirical data issue is not.
2. Calibrate the persona belief layer on validation only, targeting obvious level shifts and distribution shape without chasing a synthetic KS threshold. The immediate misses are unemployment level/shape and inflation compression.
3. Retire the fixed 50 percent liquidity-prior blend as the leading mechanism. Keep it only as an ablation.
4. Keep split-separated scoreboards as the default. Do not quote pooled selection-plus-holdout behavior rows.
5. Keep the primitive-to-action architecture, but treat `primitive_to_action_policy_selection_calibrated_v1` as a failed first calibrated kernel. The next kernel should preserve the raw model's lottery shock-size/liquidity signal without fitting away holdout generalization.
6. Keep the current forecast audit fixed in the report: AR(2), RLS, bootstrap intervals, DM-style tests, exact recall, qualitative recall, belief-structure audit, and cutoff status.

That is the path from "the belief engine contains signal" to "the simulated economy produces useful macro behavior."
