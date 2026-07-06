# Macro LLM Simulation: Current Result

## Bottom Line

The belief engine result survived the harder audit. On the 147-card date-free vintage macro split, raw GPT-5.5 and GPT-5.4 still beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate. GPT-5.5 remains the best source on the split.

The stronger statement is now this: the models are not just remembering exact realized values. In the live recall audit, both models gave zero usable exact realized-value recalls across the 147 cards. They do show some memory of broad macro regimes, but the qualitative recall edge is small. That leaves the belief forecast result standing as a real signal, while keeping the contamination claim narrow.

The agent-behavior claim is still not done. The behavior economy passes accounting and mechanism checks, and the live synthetic-SCE persona panel now clears demographic-gradient and spread checks. But the persona distribution still misses shape and level, especially unemployment expectations. That is the next gate.

## What Was Run

- Vintage OOS forecast split: 147 hidden-target, date-free macro cards.
- Models: GPT-5.5 and GPT-5.4 through Codex live calls.
- Baselines: no-change, rolling mean, rolling trend, AR(2), recursive least squares.
- Recall audit: exact realized-value recall plus qualitative path recall on the same 147 cards.
- Belief-structure audit: underreaction, extrapolation, direction, confidence, and error structure.
- Persona/SCE panel: 36 synthetic-enriched SCE-style respondents, 2 models, 72 live Codex calls, replay-rescored from cache.
- Behavior economy: HANK-lite demand economy with deterministic accounting and bounded belief inputs.

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

The full live persona panel ran on 36 synthetic-enriched SCE-style respondents, with 72 Codex live calls across GPT-5.5 and GPT-5.4. The panel is anchored to public aggregate survey beliefs and vintage macro context. It is useful for wiring and scoring the ecology; it is not real respondent-level microdata.

After fixing the scorer so missing income contrasts are skipped rather than counted as failures, the persona panel verdict is `partial_distribution_failure`.

The good part: demographic gradients and spread mostly work. The scoreable sign rate is `16/18 = 88.89%`, the median within-variance ratio is `1.1626`, and the maximum cross-model common-core correlation is `0.8501`, below the `0.95` failure threshold. Six income contrasts are skipped because this synthetic holdout has `low` and `middle` income groups but no `high` income group.

The bad part: distribution shape still fails. The maximum KS statistic is `0.7500`, above the `0.35` threshold. Unemployment expectations are the main problem: the synthetic target mean is `4.6813`, while GPT-5.4 predicts `5.5472` and GPT-5.5 predicts `5.4778`. Inflation is compressed and shifted upward as well. GPT-5.4 is closer on real income growth; GPT-5.5 undershoots it.

So the persona layer now has useful structure, but it is not calibrated enough to be the behavioral engine's empirical input layer.

## Behavior Economy

The HANK-lite behavior economy still passes its lab gate. Bounded beliefs feed into deterministic code that owns consumption, saving, debt repayment, liquidity, aggregation, inflation, policy feedback, and accounting.

The behavior layer passes mechanism checks: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and household budget identities. That means the macro sandbox is playable and accounting-safe.

It does not yet mean the simulated economy predicts real macro behavior better than a strong empirical model. The missing bridge is still behavior validation: MPCs, saving, debt repayment, liquidity shifts, and response dynamics against external targets.

## Current Claim

The sendable claim is:

> On a 147-card held-out, date-free vintage macro split, raw GPT-5.5 and GPT-5.4 belief forecasts beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate. The win survives origin-cluster bootstrapping. Live recall probes show zero exact realized-value recall and only modest qualitative path recall. The behavior economy is accounting-safe and mechanism-complete, but the persona/behavior layer still needs distribution calibration before we can claim broad macro-agent predictive validity.

The next claim is not ready:

> LLM-based macro agents predict broad real-world macro behavior better than strong empirical baselines.

## Next Gate

The next work should not add more agent theater. It should make the belief-to-behavior layer harder to fool.

1. Rebuild the persona holdout so all scored demographic and liquidity contrasts are actually present, including high-income and high-liquidity groups.
2. Calibrate the persona belief layer on validation only, targeting distribution shape as well as gradient signs. The immediate misses are unemployment level/shape and inflation compression.
3. Feed the calibrated belief distributions into the behavior economy and score real behavior targets: transfer MPC by liquidity, debt repayment, saving, and liquidity/portfolio shifts.
4. Keep the current forecast audit fixed in the report: AR(2), RLS, bootstrap intervals, DM-style tests, exact recall, qualitative recall, belief-structure audit, and cutoff status.

That is the path from "the belief engine contains signal" to "the simulated economy produces useful macro behavior."
