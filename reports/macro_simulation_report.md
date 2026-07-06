# Macro LLM Simulation: Results

## Bottom Line

We tested whether frontier LLMs can act as macroeconomic belief engines: forecast hidden macro outcomes, survive contamination audits, and drive household behavior inside an accounting-constrained simulated economy. Three results came out of it.

First, the forecast result is real. On a 147-card, date-free vintage macro split with hidden targets, raw GPT-5.5 and GPT-5.4 beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate, and the win survives origin-cluster bootstrapping. Live recall probes show the models are not retrieving remembered outcomes: exact realized-value recall was zero across all 147 cards, and the qualitative recall edge over a majority baseline is near zero.

Second, the behavior result splits by domain, and the split is the finding. On stimulus-style targets the mechanisms were designed around, raw GPT-5.5 loses to a hand-tuned liquidity rule. On a held-out lottery-windfall family the rules never saw, every rule baseline breaks and raw GPT-5.5 wins. LLM behavioral priors generalize out of domain; calibrated rules do not.

Third, we cannot yet carry that signal into an interpretable simulated economy. A pre-registered blend mechanism failed its holdout test. A primitive-to-action kernel — the model emits beliefs and stresses, deterministic code turns them into spending, saving, and debt repayment — passes every sign audit but destroys the out-of-domain signal, and calibrating it on selection targets overfits. The interpretable layer is the open problem, and we know precisely where it fails.

The claim this evidence supports: **LLMs contain genuine, contamination-audited macro belief signal, and that signal generalizes to behavioral domains where hand-tuned rules break. Turning it into a validated simulated economy still requires a behavior kernel that preserves the signal and a persona layer scored on real microdata.**

## The Forecast Result

The tournament builds as-of prompt cards from vintage FRED/ALFRED macro data. Cards are date-free: prompts use relative periods and hide calendar dates, named crises, and realized values. Each source forecasts the same hidden targets and is scored on weighted normalized absolute error (WNAE).

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

GPT-5.5 lowers mean origin-level loss by `0.2315` versus the best deterministic baseline, a `20.56%` improvement. The origin-cluster bootstrap interval is `[0.0688, 0.4153]`, with `99.82%` of draws positive. Direction accuracy is `69.39%` for both GPT models versus `6.80%` for no-change.

The cleanest family-level win is real consumption growth: GPT-5.5 beats no-change by `29.96%`, with bootstrap mean loss reduction `[0.2786, 1.1281]` and a DM-style p-value of `0.0033`. Output, policy rates, and unemployment improve by `20-34%` against their best baselines, though those intervals cross zero. Inflation is the miss: rolling trend beats GPT-5.5 on inflation growth (`0.2626` versus `0.2904`).

A validation-fitted residual calibration makes held-out forecasts worse (see the table), so the empirical signal is in the raw belief forecasts. Confidence does carry information — the confidence-error correlation is `-0.5453` for GPT-5.5 and `-0.5646` for GPT-5.4 — which a future calibration layer should use.

### Contamination

The test targets run from `2020-01-01` to `2025-02-01`. Against the Codex GPT-5 cutoff of `2024-09-30`, 14 of 147 cards are post-cutoff, so this is a hidden-target vintage test, not a post-cutoff test; a separate post-cutoff gate accrues scoreable rows as public data arrives. The contamination defense is the recall audit, run live on the same 147 cards:

- **Exact recall: zero.** Neither model returned a usable numeric realized value for any card when asked directly, without tools, whether it remembered the outcome.
- **Qualitative recall: near the base rate.** Asked about path direction, level versus normal, and crisis/calm status, GPT-5.5 scores `+0.0181` above the qualitative majority baseline on mean card accuracy; GPT-5.4 is `-0.0136` below it. The models remember broad macro eras, especially turbulence (`0.76` accuracy), but not the outcomes being scored.

The forecasts also do not look like a disguised trend extrapolator. GPT-5.5's underreaction slope is `-0.0279` — closest to zero among scored sources — against `-1.13` to `-1.15` for the rolling baselines, and its extrapolation slope is `-0.1940` against `-1.0000` for rolling mean and `-1.4819` for AR(2).

## The Behavior Result

The behavior gate scores household responses to transfer scenarios against published moments — MPC by liquidity, debt repayment, liquid saving — from the tax-rebate, 2008 stimulus, and 2020 EIP literature, with three rule baselines: a liquidity rule, a flat 30 percent rule, and a permanent-income rule. Targets are split into a selection set (11 aggregate plus 6 cell-level targets used during development) and a pre-registered holdout: 5 lottery-windfall targets from Fagereng-Holm-Natvik that no mechanism was selected against. Scoreboards and verdicts never pool the two splits.

On the selection split, raw GPT-5.5 loses to the liquidity rule: range RMSE `0.0762` versus `0.0560` on aggregate targets, `0.0387` versus `0.0272` on cell-level EIP MPC. In-domain, a rule tuned to the literature is still the better predictor.

On the lottery holdout, the ordering inverts. Raw GPT-5.5 scores `0.0844`. The best rule, the flat 30 percent rule, scores `0.2608`. The liquidity rule — the in-domain champion — scores `1.1231`, predicting spending shares above one and a liquidity gradient of `5.0` against a target of `2.0`. The model tracks what the rules cannot: lottery MPC falls with prize size, and raw GPT-5.5 carries that shock-size gradient (`0.0843` on the shock-size family, best of all sources).

Two mechanisms tried to make that signal usable, and both failed informatively:

- **The fixed 50 percent liquidity-prior blend**, pre-specified after it won on selection cell targets, fails the holdout at `0.3857` versus the flat rule's `0.2608`. Demoted to an ablation.
- **The primitive-to-action kernel** has GPT-5.5 emit bounded primitives only — perceived job-loss risk, expected income growth, precautionary motive, liquidity stress, debt-repayment urgency, durable pull-forward, log shock size, confidence — and the payload fails closed if the model outputs allocation shares. A deterministic kernel maps primitives to actions. It passes all sign audits: liquidity stress raises MPC, job risk raises saving, debt urgency raises repayment, larger windfalls lower MPC. Calibrated on the selection split only and locked, it narrowly beats the liquidity rule there (`0.0539` versus `0.0560`) and then fails the holdout at `0.8510`. Sign-correct, interpretable, and wrong out of domain.

The diagnosis is that the kernel's functional form, not the model, is the bottleneck: the signal exists in raw GPT-5.5's outputs and dies at the primitive interface. The lottery holdout has now been scored against three mechanisms and is frozen; the next kernel gets developed on the selection split and tested once against a new holdout family.

Underneath the gate, the HANK-lite demand economy passes its full lab surface: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and per-period accounting identities, with the GPT-5.5 belief module clearing all 19 validation metrics in the live 12-cell run. The sandbox is playable and accounting-safe; what it awaits is a behavior layer worth putting inside it.

## The Persona Layer

The persona panel asks whether data-grounded personas reproduce the cross-sectional structure of household beliefs. The live panel covers 54 synthetic-enriched SCE-style respondents across GPT-5.5 and GPT-5.4 — 108 model responses — anchored to public aggregate survey beliefs and vintage macro context.

Structure passes cleanly: all 24 demographic contrasts score with the correct sign, the median within-variance ratio is `1.1025` (no stereotype flattening), and the maximum cross-model common-core correlation is `0.8912`, below the `0.95` collapse threshold. Distribution shape fails: maximum KS statistic `0.7407` against a `0.35` threshold, driven by unemployment expectations (target mean `4.43`, models predict `5.22-5.34`) and upward-shifted inflation.

The scope note that governs this section: the targets are synthetic, anchored to public aggregates. The panel validates the wiring and scoring machinery end to end; empirical persona claims wait on real respondent-level microdata (SCE or Michigan), which the harness already accepts.

## What Comes Next

1. **Real microdata for the persona layer.** This is the gating input for everything unproven. The 500-respondent, two-model design run is specified and costs about 1,000 calls once the data exists.
2. **A new behavior holdout family** — UI-exhaustion spending drops (Ganong-Noel) are the natural candidate, an income-loss shock unlike anything the current mechanisms were shaped by. The lottery holdout is spent and stays frozen.
3. **A behavior kernel that preserves the signal.** Either a richer primitive interface, or reframe the interpretable layer as distillation: match raw GPT-5.5 within tolerance rather than beat baselines directly.
4. **Let the post-cutoff forecast gate accrue.** Frozen post-cutoff cards get rescored as public data arrives, converting the forecast claim from recall-audit-defended to genuinely post-cutoff over time.

The forecast audit — baselines, bootstrap intervals, DM-style tests, both recall probes, belief-structure audit, cutoff status — stays fixed in this report as the evidence base.
