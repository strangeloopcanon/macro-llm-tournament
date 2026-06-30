# Macro LLM Simulation: Current Result

## Bottom Line

The current repository now has a playable, date-free macro simulation engine with quantitative gates. Households emit bounded beliefs, firms and policy/narrative actors emit bounded reactions, and deterministic code enforces budgets, consumption, saving, debt repayment, liquidity, aggregation, feedback, and accounting. The current fixture run clears the lab performance gate and the branchable playground QA gate.

The claim supported by the current run is: the macro ecology is ready for controlled experiments and live/cached model comparisons. The empirical macro-prediction claim is gated behind a non-fixture vintage out-of-sample run.

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

The current fixture run used:

- 24 fixture household cells in the demand-economy performance run.
- 12 household cells, 20 periods, and 5 branches in the playable macro playground.
- Branches: baseline, transfer shock, rate hike, job-risk shock, and belief-feedback.
- 119 date-free vintage OOS fixture cards with hidden targets and a leakage audit.
- Zero live model calls.

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

### 2. Macro performance gate: pass

The performance gate returned `macro_lab_performance_ready`.

| Variant | Lab Targets | Blocking Failures | Weighted Loss | Score |
| --- | ---: | ---: | ---: | ---: |
| LLM belief fixture | 16/16 pass | 0 | 0.0000 | 1.0000 |
| Adaptive baseline | 13/14 pass | 1 | 0.0043 | 0.9957 |
| Representative baseline | 10/14 pass | 4 | 0.0424 | 0.9576 |
| Naive persona fixture | 9/14 pass | 5 | 0.2341 | 0.7659 |

The LLM belief fixture clears the lab target catalog: accounting, transfer MPC, low/high-liquidity MPCs, liquidity and income gradients, monetary-shock signs, job-risk precaution, belief-feedback amplification, belief dispersion, and IRF shape.

The useful comparison is not that the fixture label is magic. It is that the belief-module architecture clears the target surface while the direct persona baseline fails the expected mechanism checks. The architecture is doing the right job: beliefs go through constrained economics instead of directly choosing whatever consumption path sounds plausible.

### 3. Date-free vintage OOS fixture: pass as a runner, not as model evidence

The vintage OOS runner returned `demand_vintage_oos_fixture_ready`.

| Check | Result |
| --- | --- |
| Cards | 119 |
| Hidden target rows | 119 |
| Forecast score rows | 84 |
| Leakage issues | 0 |

On the fixture test split, weighted normalized absolute error was:

| Source | Test Error |
| --- | ---: |
| LLM belief fixture | 0.0791 |
| Rolling trend | 0.0968 |
| Rolling mean | 0.1894 |
| No change | 0.1990 |

This validates the date-free card, hidden-target, scoring, and leakage-audit machinery. It is still a fixture, so it does not count as evidence that a live LLM predicts macro outcomes out of sample.

## What We Can Say Now

The project has moved from "can we make macro agents look plausible?" to "can we run a bounded macro ecology and score it?" The answer to the second question is now yes.

The system can run counterfactual branches, preserve accounting, expose actor contributions, and score whether mechanisms behave in the right direction. It can test whether belief updates, firm reactions, policy narratives, and household heterogeneity add value over rule-only baselines.

## Current Limit

The current run is fixture-only. The manifest correctly reports `empirical_ready: false`.

The next result to chase is a non-fixture vintage OOS run: same date-free cards, hidden targets, and baseline comparisons, but with actual model belief payloads instead of fixture forecasts. That is the gate that turns this from a validated macro sandbox into empirical predictive evidence.

## Next Test

Run the vintage OOS path with live or replayed model belief calls, then feed those artifacts into the performance gate. The pass condition is `macro_empirical_oos_ready`: the lab gate still passes, the vintage OOS artifact is non-fixture, the LLM-belief rows have no blocking gaps or failures, and the result beats simple baselines on the held-out target surface.
