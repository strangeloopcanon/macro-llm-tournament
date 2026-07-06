# Macro LLM Simulation: Current Result

## Bottom Line

After adding AR(2) and recursive least-squares controls, the raw GPT belief forecasts still clear the current vintage OOS gate.

On 147 held-out, date-free macro cards, GPT-5.5 has weighted normalized absolute error (WNAE) of `0.8945`. GPT-5.4 is close at `0.9064`. The best deterministic baseline is still no-change at `1.1260`; rolling trend is `1.2668`, rolling mean is `1.2790`, AR(2) is `2.4307`, and recursive least squares is `4.2816`.

The aggregate paired comparison remains positive. GPT-5.5 lowers mean origin-level loss by `0.2315` versus the best deterministic baseline, a `20.56%` improvement. The origin-cluster bootstrap interval is `[0.0688, 0.4153]`, with `99.82%` of bootstrap draws positive.

That is the claim. It is a date-free, hidden-target result on vintage macro cards, not a clean post-cutoff result for the frontier models. The test targets run from `2020-01-01` to `2025-02-01`; using the cutoffs we have been working with, `0/147` target rows are after `2025-08-31`, and only `14/147` are after `2024-09-30`.

## What Changed

Alex was right about the report: the earlier version compared GPT only against no-change, rolling mean, and rolling trend. That was too weak. The scorer and macro gate now include AR(2) and recursive least-squares baselines, and the gate emits a family-level paired comparison with block-bootstrap intervals plus a DM-style z test.

The stronger controls do not overturn the aggregate result on this split. They do change the shape of the evidence: the win is not uniform across target families, and the fitted calibration layer is clearly worse than the raw forecasts.

## Headline OOS Table

| Source | Test WNAE | Status |
| --- | ---: | --- |
| GPT-5.5 raw | 0.8945 | Pass |
| GPT-5.4 raw | 0.9064 | Pass |
| No change | 1.1260 | Best deterministic baseline |
| Rolling trend | 1.2668 | Baseline |
| Rolling mean | 1.2790 | Baseline |
| GPT-5.4 calibrated | 1.3933 | Fails |
| GPT-5.5 calibrated | 1.8851 | Fails |
| AR(2) | 2.4307 | Baseline |
| Recursive least squares | 4.2816 | Baseline |

The validation-fitted residual calibration should not be treated as a successful model. It overfits the validation period and damages held-out forecasts. The empirical signal is in the raw belief forecasts.

## Where It Works

The best family-level evidence is real consumption growth. GPT-5.5 beats no-change by `29.96%`, with a bootstrap interval `[0.2786, 1.1281]` for mean loss reduction and a DM-style p-value of `0.0033`.

Output, policy rates, and unemployment also move in the right direction, but the evidence is less decisive. GPT-5.5 improves output by `21.88%` versus no-change, policy rate by `33.82%` versus rolling trend, and unemployment by `20.00%` versus no-change; the bootstrap intervals still cross zero for those families.

Inflation is the visible miss. Rolling trend beats GPT-5.5 on inflation growth (`0.2626` versus `0.2904`). Saving and sentiment are weak wins at best: GPT-5.5 improves saving by `4.99%` versus AR(2) and sentiment by `2.20%` versus no-change, with wide intervals.

## Behavior Economy

The HANK-lite behavior economy still passes its lab gate. The current run uses bounded LLM-shaped beliefs while deterministic code owns consumption, saving, debt repayment, liquidity, aggregation, inflation, policy feedback, and accounting.

The behavior layer passes the accounting and mechanism checks: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and household budget identities. This means the beliefs can be inserted into a feasible macro engine without breaking the sandbox.

It does not yet mean the full simulated economy predicts real macro behavior better than a strong empirical model. The behavior engine is ready for harder empirical validation; it is not the empirical win by itself.

## What Is Still Missing

Three audit surfaces should come next before expanding the claim.

First, run the direct and qualitative recall probes on the 147-card split and report the cutoff status alongside the score. The code has recall machinery, but this report does not include a completed live recall audit for these vintage cards.

Second, bring the belief-structure audit back into the canonical result: underreaction, extrapolation, disagreement or dispersion, calibration, and surprise response. The code has that machinery in the forecast audit path; the current macro gate does not yet consume it.

Third, add the SCE/persona cross-section moments as a separate evidence layer. The repo has synthetic SCE/persona holdouts and scoring code, but synthetic canaries should not be promoted as respondent-level empirical evidence.

## Current Claim

The cleaned-up claim is narrow and worth keeping:

> On a 147-card held-out, date-free vintage macro split, raw GPT-5.5 and GPT-5.4 belief forecasts beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate. The win survives origin-cluster bootstrapping. The same belief profile can be fed into the HANK-lite behavior economy while preserving accounting and expected macro mechanisms.

The next claim is not ready yet:

> LLM-based macro agents predict broad real-world macro behavior better than strong empirical baselines.

To get there, the next run should add the recall audit, belief-structure audit, and SCE/persona cross-section moments to the canonical report, then test whether the belief-driven behavior economy improves real behavior targets rather than only passing mechanism checks.
