# Macro LLM Simulation: Current Result

## Bottom Line

The project now has a playable, date-free macro simulation engine and a live out-of-sample forecast test.

The new result is positive but bounded: live GPT models beat simple deterministic baselines on held-out, date-free vintage macro cards, but the absolute forecast errors are still too large for the stricter empirical-ready gate. In plain English: there is real predictive signal in the LLM belief layer, but it is not yet a strong macro-prediction engine.

## What The Ecology Does

The economy is intentionally small: one good, no capital, no full asset market, and no external sector. That keeps the loop inspectable.

Each period works like this:

1. A branch starts from a date-free state such as `period_0`.
2. Firm actors provide bounded output, hiring, price-pressure, and credit-tightening reactions.
3. Policy/narrative actors provide bounded rate, transfer, confidence, job-risk, and dispersion modifiers.
4. Household actors provide beliefs only: inflation, income growth, job-loss risk, confidence, and precaution.
5. Deterministic structural code converts those beliefs into consumption, saving, debt repayment, and liquidity changes.
6. Aggregate demand updates output, employment, income, inflation, the policy rate, and next-period household states.
7. A critic actor reads the result and flags accounting breaks or implausible sign patterns, but cannot mutate the state.

This is closer to a controllable macro laboratory than a persona exercise. The LLM-shaped part is the belief/reaction layer; the economy itself is bounded by code.

## Current Run

The current state combines three layers:

- Playable macro playground fixture: 12 household cells, 20 periods, 5 branches.
- Demand-economy performance fixture: 24 household cells, closed-loop scenarios, mechanism scoring.
- Live vintage OOS forecast test: 147 held-out date-free cards from the FRED vintage panel test split.

The live OOS test used `openai_responses` with `gpt-5.5` and `gpt-5.4`. Each model made 147 live calls. The combined replay artifact then scored both models with zero additional live calls.

## Results

### 1. Playable macro engine: pass

The macro playground returned `macro_playground_fixture_ready`.

| Gate | Result |
| --- | --- |
| Actor role coverage | Pass: firm, policy/narrative, household-belief, and critic payloads emitted each period |
| Schema closure | Pass: actor payloads normalize through fail-closed schemas |
| Accounting | Pass: max absolute residual `5.82e-11` |
| Branch divergence | Pass: branches measurably diverge from baseline |
| Rate hike | Pass: consumption and output contract over the early response window |
| Transfer shock | Pass: transfer spending, debt repayment, and liquid saving exhaust the transfer |
| Liquidity gradient | Pass: low-liquidity MPC exceeds high-liquidity MPC; gradient `0.461` |
| Job-risk shock | Pass: precautionary saving rises and consumption falls before income moves |
| Firm channel | Pass: firm price/hiring effects enter only through bounded channels |
| Critic flags | Pass: no blocking critic flags |

### 2. Lab performance gate: pass

The performance gate returned `macro_lab_performance_ready`.

| Variant | Lab Targets | Blocking Failures | Weighted Loss | Score |
| --- | ---: | ---: | ---: | ---: |
| LLM belief fixture | 16/16 pass | 0 | 0.0000 | 1.0000 |
| Adaptive baseline | 13/14 pass | 1 | 0.0043 | 0.9957 |
| Representative baseline | 10/14 pass | 4 | 0.0424 | 0.9576 |
| Naive persona fixture | 9/14 pass | 5 | 0.2341 | 0.7659 |

The useful comparison is that the belief-module architecture clears the mechanism surface while the direct persona baseline fails expected checks. The architecture is doing the right job: beliefs go through constrained economics instead of directly choosing whatever consumption path sounds plausible.

### 3. Live date-free vintage OOS: relative signal, not full pass

The live OOS runner returned `demand_vintage_oos_scored`.

| Check | Result |
| --- | --- |
| Held-out test cards | 147 |
| Target families | 7 |
| GPT-5.5 live calls | 147 |
| GPT-5.4 live calls | 147 |
| Combined replay cache hits | 294 |
| Leakage issues | 0 |

Weighted normalized absolute error on the held-out test split:

| Source | Test Error | Improvement vs best baseline |
| --- | ---: | ---: |
| GPT-5.4 LLM belief | 2.8429 | 9.88% |
| GPT-5.5 LLM belief | 2.9096 | 7.77% |
| Rolling trend | 3.1546 | baseline |
| Rolling mean | 3.2177 | baseline |
| No change | 3.2716 | baseline |

The live models beat all simple baselines overall. GPT-5.4 is best overall; GPT-5.5 wins more individual target families but by smaller margins.

GPT-5.5 beats the best baseline on 6 of 7 target families, losing only inflation. GPT-5.4 beats the best baseline on 5 of 7 target families, losing inflation and sentiment.

| Target Family | GPT-5.5 vs Best Baseline | GPT-5.4 vs Best Baseline |
| --- | ---: | ---: |
| Policy rate level | +33.82% | +42.51% |
| Unemployment rate level | +20.00% | +44.62% |
| Saving rate level | +9.58% | +10.54% |
| Output growth | +5.01% | +9.81% |
| Real consumption growth | +2.11% | +3.53% |
| Sentiment growth | +2.20% | -9.69% |
| Inflation growth | -2.69% | -13.59% |

The macro performance gate still reports `empirical_ready: false` because the absolute OOS error target is not met. The best LLM error is `2.8429`, while the current empirical-ready target requires weighted normalized absolute error at or below `1.0`.

## What We Can Say Now

We can now say that the LLM belief layer contains useful out-of-sample macro signal under date-free, hidden-target conditions. That is new.

We cannot yet say that the full simulated economy is empirically predictive in the strong sense. The live forecast layer beats baselines, but the misses are large in volatile consumption/output periods. The system has crossed from "playable sandbox" to "promising predictive signal"; it has not crossed to "strong macro validity."

## Current Limit

The main failure mode is underreaction to large real consumption and output swings. The date-free test split includes big macro moves, and the live models mostly produce conservative forecasts. That helps against naive baselines but does not get close enough to the realized path.

This is actually useful diagnostically: the next improvement should not be more personas. It should be better belief dynamics around regime shifts, rebound risk, uncertainty, and asymmetric responses after sharp changes.

## Next Test

The next gate should add a calibrated belief-dynamics module before the household behavior layer:

- expose recent level changes, volatility, and drawdown/rebound features more explicitly in the date-free cards;
- ask models for distributions, not only point forecasts;
- calibrate on validation split only, then lock prompts/transforms before test scoring;
- feed calibrated beliefs into the demand economy and test whether behavior predictions improve, not just raw macro forecasts.

Success means the live/replay LLM layer beats deterministic baselines and brings absolute OOS error materially closer to the empirical-ready threshold.
