# Live LLM Households in a HANK-Lite Demand Economy

## Summary

We built and ran a small, date-free macroeconomic ecology in which live LLM household agents update beliefs over time and those beliefs feed into an accounting-constrained consumption economy. The main result is that the GPT-5.5 belief-module economy clears the full dynamic validation gate: it matches survey-style belief seeds better than the structural baselines, preserves HANK-style consumption heterogeneity, responds in the right direction to transfer, monetary, and job-risk shocks, generates belief-feedback amplification, and closes accounting every period.

The result is an early but substantive behavioral macro result. The useful claim now extends beyond macro forecasting: in this setup, an LLM belief module can sit inside a constrained economic system and produce plausible aggregate behavior across periods.

## What Was Built

The test economy is a deliberately simple demand-driven HANK-lite environment:

- Twelve household cells, weighted by population share, with different income, liquidity, job-risk, age, asset, and baseline MPC profiles.
- A live LLM belief module that updates household beliefs each period about inflation, income growth, job loss risk, confidence, and precautionary saving.
- A deterministic structural layer that turns those beliefs into budget-constrained consumption and saving decisions.
- An aggregate feedback loop where consumption affects demand, demand affects output, output affects employment and income, inflation adjusts sluggishly, and a Taylor-rule policy rate responds to inflation and output.
- A date-free prompt design: households see abstract current conditions and their prior state rather than calendar dates, named crises, or realized historical paths.

This matters because the LLM supplies beliefs while the code owns budgets, accounting, aggregation, feedback, and validation.

## Validation Design

The live GPT-5.5 run used:

- 12 household cells
- 20 periods per scenario
- 5 scenarios: baseline, transfer shock, rate hike, job-risk shock, and belief-feedback amplification
- 4 comparison variants: representative agent, adaptive heterogeneous baseline, live LLM belief module, and naive direct persona baseline
- 38 live model calls, with the remaining scenario-period payloads served from the run cache

The validation checks cover:

- baseline stability
- aggregate transfer MPC
- liquidity and income MPC gradients
- monetary tightening response
- precautionary saving under perceived job-risk shocks
- endogenous belief-feedback amplification
- belief dispersion
- per-period budget and goods-market accounting
- survey-style belief seed fit

## Results

The full live run returns `hank_lite_belief_lab_ready`.

The LLM belief module passes all 19 of its validation metrics. Across the ablation surface, all 54 required metrics pass. The maximum accounting residual is `5.82e-11`, effectively numerical zero.

| Metric | Live GPT-5.5 belief module |
| --- | ---: |
| Survey-seed normalized belief MAE | 0.029 |
| Transfer impact MPC | 0.479 |
| Four-period transfer MPC | 0.251 |
| Low-liquidity impact MPC | 0.788 |
| High-liquidity impact MPC | 0.184 |
| Liquidity MPC gradient | 0.604 |
| Income MPC gradient | 0.427 |
| Mean six-period consumption response to rate hike | -367.1 |
| Mean six-period output-gap response to rate hike | -2.83 |
| Late inflation response to rate hike | -0.161 |
| Consumption response to job-risk shock | -506.9 |
| Income movement on impact in pure job-risk shock | 0.0 |
| Belief-feedback amplification ratio | 2.68x |
| Maximum accounting residual | 5.82e-11 |

## Baseline Comparison

| Variant | Belief MAE | Transfer MPC | Liquidity gradient | Income gradient | Feedback amplification | Rate-hike consumption response | Job-risk consumption response |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Representative agent | 0.364 | 0.452 | 0.011 | 0.042 | 0.99x | -225.2 | -518.7 |
| Adaptive heterogeneous baseline | 0.106 | 0.454 | 0.552 | 0.405 | 0.97x | -252.7 | -511.1 |
| Live GPT-5.5 belief module | 0.029 | 0.479 | 0.604 | 0.427 | 2.68x | -367.1 | -506.9 |
| Naive direct persona baseline | fixture seed echo | 0.778 | -0.144 | -0.704 | 0.00x | -105.8 | -311.3 |

The representative agent is smooth and stable, but loses the cross-sectional heterogeneity that drives liquidity-sensitive MPCs. The adaptive baseline captures some heterogeneity, but its feedback amplification stays below threshold. The naive direct persona baseline is included as a stress test: it can echo seed beliefs and make direct consume/save choices, but it fails important gradient and feedback checks.

The live LLM belief module is the only variant in this run that combines low belief error, HANK-style MPC gradients, monetary and job-risk responses, and endogenous amplification.

## Interpretation

The strongest conclusion is architectural. LLMs work better here as belief-updating modules inside a constrained economy than as unconstrained roleplaying households. The LLM supplies forward-looking household beliefs; the structural layer enforces feasibility and macro accounting. That combination is what produces plausible behavior.

The result also narrows the original research question. The first empirical claim was that LLMs improve macro belief forecasts. This run adds a behavior result: in a controlled, date-free HANK-lite economy, live LLM belief updates can improve downstream macro dynamics relative to representative, adaptive, and naive persona baselines.

## Scope

This is a first dynamic gate. The evidence is strongest for the mechanism: survey-seeded household belief updating, abstract period-by-period information, constrained consumption behavior, feedback, and accounting closure. The next validation work is to scale the ecology across more household cells, more seeds, more model families, and richer external household behavior targets.

## Next Work

The next step is replication and empirical thickening:

1. Repeat the same full gate across GPT-5.4, GPT-5.5, and Gemini where provider access is stable.
2. Expand from 12 cells to a larger survey-derived household surface.
3. Replace fixture-style target moments with direct micro behavior targets where available: MPC by liquidity, debt repayment, saving, liquidity shifts, job search, and portfolio rebalancing.
4. Run robustness checks across prompt versions, random seeds, and alternative shock sizes.
5. Keep the setup date-free so the test remains about belief formation and economic feedback rather than remembered history.

The current result earns the next scale-up. It shows that a live LLM belief ecology can clear a serious macro-behavior gate while staying inside accounting and contamination controls.
