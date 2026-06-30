# Macro LLM Simulation: Current Result

## Bottom Line

The current run clears the first empirical OOS gate.

After fixing the vintage scoring target, GPT-5.5 and GPT-5.4 both beat deterministic baselines on held-out, date-free macro cards and clear the current absolute weighted-error threshold. The behavior economy also passes the lab gate after feeding in a bounded belief-dynamics profile. The win is the belief layer plus the behavior engine, not the fitted calibration model: the validation-fitted forecast calibration overfits and makes held-out forecasts worse.

The clean claim is:

> Under corrected, date-free, hidden-target scoring, GPT belief forecasts beat no-change, rolling-mean, and rolling-trend controls on the held-out vintage test split. Those beliefs can be fed into the HANK-lite demand economy without breaking accounting or the mechanism tests.

## What Changed

The important change was not another persona variant. It was the scoring and calibration setup.

The old vintage target construction mixed vintages for percent-change targets: it divided a final next-period value by the vintage current value. That accidentally put benchmark revisions into the growth target. The scorer now uses the final current value and final next value for percent-change targets, while the prompt still sees only the as-of vintage history. That makes the hidden outcome a realized growth target rather than a revision artifact.

On top of that corrected scorer, we added a validation-only belief-dynamics calibration step. It learns regime-shift, rebound, and uncertainty features from the validation split, applies the locked transform to the test split, and emits a bounded behavior profile for the demand economy.

## Current Evidence

The current result combines three artifacts:

| Layer | Artifact | Verdict |
| --- | --- | --- |
| Corrected validation vintage run | `demand_vintage_oos_replay_gpt55_gpt54_val_panel_consistent_targets` | `demand_vintage_oos_scored` |
| Corrected held-out test plus calibration report | `belief_calibration_gpt55_gpt54_val_to_test_consistent_targets` | `belief_calibration_evaluated` |
| Combined macro performance gate | `macro_performance_gate_calibrated_consistent_oos` | `macro_empirical_oos_ready` |

The validation run used 20 validation origins, 140 date-free cards, and the already-collected GPT-5.5/GPT-5.4 cache. The held-out test run used 21 scored test origins and 147 date-free cards. The test replay used 294 cache hits and made zero new live calls. The leakage audit found zero issues.

## Held-Out OOS Results

Weighted normalized absolute error on the corrected held-out test split:

| Source | Test WNAE | Gate |
| --- | ---: | --- |
| GPT-5.5 | 0.8945 | Pass |
| GPT-5.4 | 0.9064 | Pass |
| No change | 1.1260 | Fail |
| Rolling trend | 1.2668 | Fail |
| Rolling mean | 1.2790 | Fail |

GPT-5.5 is the best model in this run. It improves on the best deterministic baseline by 20.56%. The paired origin-cluster bootstrap is strongly positive: mean loss reduction `0.2315`, 95% interval `[0.0688, 0.4153]`, positive share `99.82%`.

GPT-5.4 is close behind. It improves on the best deterministic baseline by 19.51%, with mean loss reduction `0.2197`, 95% interval `[0.0077, 0.4879]`, and positive share `97.97%`.

The gains are strongest where the target is closer to beliefs about policy, labor, and inflation. The models still miss large real activity swings: output and consumption remain above WNAE 1.0 by target family even though the combined score clears the overall gate.

| Source | Inflation | Output | Policy Rate | Consumption | Saving | Sentiment | Unemployment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 | 0.2904 | 1.4596 | 0.0652 | 1.5179 | 1.3476 | 1.3747 | 0.2063 |
| GPT-5.4 | 0.3174 | 1.3133 | 0.0567 | 1.6393 | 1.3333 | 1.5417 | 0.1429 |
| No change | 0.5625 | 1.8684 | 0.1300 | 2.1671 | 1.4905 | 1.4057 | 0.2579 |

## Calibration Result

The validation-fitted residual calibration does not improve held-out forecasts.

| Source | Test WNAE |
| --- | ---: |
| GPT-5.5 raw | 0.8945 |
| GPT-5.4 raw | 0.9064 |
| GPT-5.4 calibrated | 1.2254 |
| GPT-5.5 calibrated | 1.7201 |

That result matters. We now know the raw LLM belief signal is strong enough to clear the first empirical OOS threshold, but the simple residual/ridge calibration is too eager to extrapolate validation patterns into the held-out period. The calibration artifact is still useful because it produces a bounded behavior-economy profile, but it should not be treated as the winning forecast source.

## Behavior Economy

The calibrated belief profile was fed into the HANK-lite demand economy as a bounded postprocessor on household beliefs. Households still emit beliefs only; deterministic code owns consumption, saving, debt repayment, liquidity, aggregation, sticky inflation, policy feedback, and accounting.

The behavior run passed the full lab surface:

| Check | Result |
| --- | --- |
| Verdict | `hank_lite_belief_lab_ready` |
| Required metrics | 54 of 54 passed |
| Accounting residual | `5.82e-11` |
| Transfer MPC shape | Passed |
| Liquidity and income MPC gradients | Passed |
| Rate-hike response | Consumption/output contract; inflation softens |
| Job-risk response | Consumption falls before income moves |
| Belief feedback and dispersion | Passed for the LLM belief source |

The combined macro gate then returned `macro_empirical_oos_ready`: the lab behavior source passes with no blocking failures, and the held-out OOS belief source beats the best deterministic baseline with WNAE below 1.0.

## What We Have Learned

The belief-engine claim is now real in the narrow empirical sense that matters first. Date-free GPT belief forecasts contain enough signal to beat simple macro baselines on held-out vintage targets and clear the current absolute error gate.

The behavior-economy claim is also stronger than before. The engine can take LLM-shaped beliefs, enforce feasibility through code, and produce macro-consistent responses across transfers, rate hikes, job-risk shocks, and endogenous belief feedback.

The calibration story is weaker. Simple validation residual calibration is not the way through; it overfits. The next calibration attempt should be more conservative, probably distributional, and explicitly guarded against regime-transfer failures.

## Scope

This is still a first empirical gate, not a finished macro model. The sample is 147 held-out vintage cards across seven target families, and the behavior economy is a one-good HANK-lite demand economy without capital, a full banking sector, an asset market, or an external sector.

That scope is enough for the current claim: the system now produces empirical belief signal and behavior-consistent macro dynamics. The next question is whether that signal can be made stronger on hard real-activity periods without overfitting.

## Next Move

Freeze the corrected target construction and current OOS scorer.

Then improve belief dynamics around exactly the failure modes visible here:

- regime shifts and rebounds after sharp drawdowns;
- uncertainty and asymmetric underreaction in output and consumption;
- distributional forecasts rather than only point forecasts;
- calibration rules that can decline to move a forecast when validation evidence is not robust.

The goal for the next run is simple: keep the OOS WNAE below 1.0, improve output and consumption target-family errors, and preserve the behavior-economy lab pass.
