# Macro LLM Simulation: Results

## Bottom Line

We tested whether frontier LLMs can act as macroeconomic belief engines: forecast hidden macro outcomes, survive contamination audits, and drive household behavior inside an accounting-constrained simulated economy. Three results came out of it.

First, the forecast result is real. On a 147-card, date-free vintage macro split with hidden targets, raw GPT-5.5 and GPT-5.4 beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate, and the win survives origin-cluster bootstrapping. Live recall probes show the models are not retrieving remembered outcomes: exact realized-value recall was zero across all 147 cards, and the qualitative recall edge over a majority baseline is near zero.

Second, the behavior result now splits by shock type, and that split is the finding. On stimulus-style targets the mechanisms were designed around, raw GPT-5.5 loses to a hand-tuned liquidity rule. On a held-out lottery-windfall family, the rules break and raw GPT-5.5 wins. On a new UI-exhaustion income-loss holdout, raw GPT-5.5 loses to a simple flat rule. The behavioral signal is real but not generic: it transfers to windfall-size reasoning, not yet to predictable income-loss dynamics.

Third, we cannot yet carry that signal into an interpretable simulated economy. A pre-registered blend mechanism failed its holdout test. A primitive-to-action kernel — the model emits beliefs and stresses, deterministic code turns them into spending, saving, and debt repayment — passes every sign audit but destroys the out-of-domain signal, and calibrating it on selection targets overfits. The interpretable layer is the open problem, and we know precisely where it fails.

The claim this evidence supports: **LLMs contain genuine, contamination-audited macro belief signal, and raw GPT-5.5 contains useful household-behavior priors in at least one out-of-domain shock family. Turning that into a validated simulated economy still requires a behavior architecture that preserves the signal across shock types and a persona layer scored on real microdata.**

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

The behavior gate scores household responses against published moments with three rule baselines: a liquidity rule, a flat rule, and a permanent-income rule. Targets are split explicitly, and scoreboards never pool splits:

- `behavior_selection_v1`: 11 aggregate plus 6 cell-level stimulus/rebate targets used during development.
- `behavior_holdout_v1`: 5 lottery-windfall targets from Fagereng-Holm-Natvik. This holdout is spent and frozen.
- `behavior_holdout_ui_v1`: 3 UI-exhaustion path targets from Ganong-Noel: about a 6% nondurable-spending drop at unemployment onset, less than 1% monthly drift during UI receipt, and about a 12% drop at predictable benefit exhaustion. The checking/liquidity gradient remains an unscored gap until a directly scoreable public number is pinned down.

On the selection split, raw GPT-5.5 loses to the liquidity rule: range RMSE `0.0762` versus `0.0560` on aggregate targets, `0.0387` versus `0.0272` on cell-level EIP MPC. In-domain, a rule tuned to the literature is still the better predictor.

On the lottery holdout, the ordering inverts. Raw GPT-5.5 scores `0.0844`. The best rule, the flat 30 percent rule, scores `0.2608`. The liquidity rule — the in-domain champion — scores `1.1231`, predicting spending shares above one and a liquidity gradient of `5.0` against a target of `2.0`. The model tracks what the rules cannot: lottery MPC falls with prize size, and raw GPT-5.5 carries that shock-size gradient (`0.0843` on the shock-size family, best of all sources).

On the UI-exhaustion holdout, the result flips back against the model. Raw GPT-5.5 scores `0.0397`; the best rule, the flat UI spending-drop rule, scores `0.0289`. The residual-over-liquidity ablation scores `0.0353`, and the primitive path scores `0.0311`; both lose. This is a clean negative result for the broad “LLM behavior generalizes out of domain” claim. The lottery win is not enough: income-gain windfalls and predictable income-loss exhaustion are different mechanisms.

Two mechanisms tried to make that signal usable, and both failed informatively:

- **The fixed 50 percent liquidity-prior blend**, pre-specified after it won on selection cell targets, fails the holdout at `0.3857` versus the flat rule's `0.2608`. Demoted to an ablation.
- **The primitive-to-action kernel** has GPT-5.5 emit bounded primitives only — perceived job-loss risk, expected income growth, precautionary motive, liquidity stress, debt-repayment urgency, durable pull-forward, log shock size, confidence — and the payload fails closed if the model outputs allocation shares. A deterministic kernel maps primitives to actions. It passes all sign audits: liquidity stress raises MPC, job risk raises saving, debt urgency raises repayment, larger windfalls lower MPC. Calibrated on the selection split only and locked, it narrowly beats the liquidity rule there (`0.0539` versus `0.0560`) and then fails the holdout at `0.8510`. Sign-correct, interpretable, and wrong out of domain.

The diagnosis is sharper now. The raw model has windfall signal that the rules miss, but it does not automatically solve predictable income-loss behavior. The primitive kernel still destroys the lottery signal and only nearly matches the flat rule on UI. The next architecture has to preserve behavior across shock types, not just pass sign audits or fit the selection split.

Underneath the gate, the HANK-lite demand economy passes its full lab surface: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and per-period accounting identities, with the GPT-5.5 belief module clearing all 19 validation metrics in the live 12-cell run. The sandbox is playable and accounting-safe; what it awaits is a behavior layer worth putting inside it.

## The Persona Layer

The persona panel asks whether data-grounded personas reproduce the cross-sectional structure of household beliefs. The live panel covers 54 synthetic-enriched SCE-style respondents across GPT-5.5 and GPT-5.4 — 108 model responses — anchored to public aggregate survey beliefs and vintage macro context.

Structure passes cleanly: all 24 demographic contrasts score with the correct sign, the median within-variance ratio is `1.1025` (no stereotype flattening), and the maximum cross-model common-core correlation is `0.8912`, below the `0.95` collapse threshold. Distribution shape fails: maximum KS statistic `0.7407` against a `0.35` threshold, driven by unemployment expectations (target mean `4.43`, models predict `5.22-5.34`) and upward-shifted inflation.

The scope note that governs this section: the targets are synthetic, anchored to public aggregates. The panel validates the wiring and scoring machinery end to end; empirical persona claims wait on real respondent-level microdata (SCE or Michigan), which the harness already accepts.

## What Comes Next

1. **Real microdata for the persona layer.** This is the gating input for everything unproven. The 500-respondent, two-model design run is specified and costs about 1,000 calls once the data exists.
2. **Architecture fidelity, developed only on the selection split.** Compare constrained-raw, constrained-choice, and primitive-kernel behavior paths. The winner should be the most interpretable architecture within a pre-declared tolerance of raw GPT-5.5 on the UI holdout.
3. **A behavior kernel that preserves signal across shock types.** The lottery and UI results say the hard problem is not “make a plausible rule.” It is preserving the model's useful windfall reasoning while adding a mechanism for predictable income-loss responses.
4. **Let the post-cutoff forecast gate accrue.** Frozen post-cutoff cards get rescored as public data arrives, converting the forecast claim from recall-audit-defended to genuinely post-cutoff over time.

The forecast audit — baselines, bootstrap intervals, DM-style tests, both recall probes, belief-structure audit, cutoff status — stays fixed in this report as the evidence base.
