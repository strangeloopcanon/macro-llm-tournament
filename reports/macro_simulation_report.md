# Macro LLM Simulation: Results

## Bottom Line

We tested whether frontier LLMs can act as macroeconomic belief engines: forecast hidden macro outcomes, survive contamination audits, and drive household behavior inside an accounting-constrained simulated economy. Five results came out of it.

First, the forecast result is real. On a 147-card, date-free vintage macro split with hidden targets, raw GPT-5.5 and GPT-5.4 beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines in aggregate, and the win survives origin-cluster bootstrapping. Live recall probes show the models are not retrieving remembered outcomes: exact realized-value recall was zero across all 147 cards, and the qualitative recall edge over a majority baseline is near zero.

Second, the behavior result now splits by shock type, and that split is the finding. On stimulus-style targets the mechanisms were designed around, raw GPT-5.5 loses to a hand-tuned liquidity rule. On a held-out lottery-windfall family, the rules break and raw GPT-5.5 wins. On a new UI-exhaustion income-loss holdout, raw GPT-5.5 loses to a simple flat rule. The behavioral signal is real but not generic: it transfers to windfall-size reasoning, not yet to predictable income-loss dynamics.

Third, the real-data profile-only persona gate failed. The approved 500-respondent SCE design run was live, stratified, leak-checked, and pre-registered. It did not reproduce real household belief heterogeneity. The models got some inflation and income-gradient signs right, but they compressed individual dispersion by roughly one to two orders of magnitude and got the unemployment-higher-probability gradients mostly wrong.

Fourth, prior-conditioned updating works in a narrower but more economically relevant sense. On 100 repeated SCE respondents across October-November 2024, date-blind prompts gave each agent its own prior beliefs and current environment, then scored the update from prior to current belief. GPT-5.5 cleared the locked update gate: mean update correlation `0.3578`, direction accuracy `0.5769`, median update-amplitude ratio `0.3234`, and RMSE improvement versus persistence `0.0437`. This does not fix the cross-sectional heterogeneity failure; it says that when the model is given a person's previous beliefs, it can partly model how beliefs move.

Fifth, we cannot yet carry the belief signal into a validated simulated economy. A pre-registered blend mechanism failed its holdout test. A primitive-to-action kernel — the model emits beliefs and stresses, deterministic code turns them into spending, saving, and debt repayment — passes every sign audit but destroys the out-of-domain signal, and calibrating it on selection targets overfits. The interpretable behavior layer remains open. The persona layer is no longer simply an empirical miss: profile-only personas failed, while prior-conditioned update agents now have a first positive gate.

The claim this evidence supports: **LLMs contain genuine, contamination-audited macro belief signal; raw GPT-5.5 contains useful household-behavior priors in one out-of-domain windfall family; and prior-conditioned SCE agents show real belief-updating signal. But profile-only personas do not reproduce real SCE household belief heterogeneity, and the project has not yet achieved a validated LLM-based simulated economy.**

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

The architecture-fidelity run then tested three locked behavior architectures on the UI holdout, without rescoring the spent lottery holdout:

- **A. Constrained raw:** raw GPT-5.5 allocation/drop shares, UI RMSE `0.0397`.
- **B. Constrained choice:** the model chooses a named policy family and bounded parameters, then code executes it. The first pass did not fairly evaluate this architecture: one bounded policy could still drive high-liquidity EIP spending to zero, making the low/high liquidity ratio divide by a near-zero denominator and exploding selection RMSE to `248,350,850.7`. That is a bounds-enforcement bug, now fixed in code for future selection-split development. It is not a holdout result to interpret.
- **C. Primitive v3:** the model emits richer primitives — including income-change attention, predictable-drop attention, windfall permanence, and spending commitments — and deterministic code maps them into behavior. It is promising but unconfirmed: UI RMSE `0.0200`, beating both raw GPT-5.5 and the flat rule (`0.0289`), but selection-split RMSE is `0.3401`, far worse than raw GPT-5.5 (`0.0762`) and the liquidity rule (`0.0560`).

Two mechanisms tried to make that signal usable, and both failed informatively:

- **The fixed 50 percent liquidity-prior blend**, pre-specified after it won on selection cell targets, fails the holdout at `0.3857` versus the flat rule's `0.2608`. Demoted to an ablation.
- **The primitive-to-action kernel** has GPT-5.5 emit bounded primitives only — perceived job-loss risk, expected income growth, precautionary motive, liquidity stress, debt-repayment urgency, durable pull-forward, log shock size, confidence — and the payload fails closed if the model outputs allocation shares. A deterministic kernel maps primitives to actions. It passes all sign audits: liquidity stress raises MPC, job risk raises saving, debt urgency raises repayment, larger windfalls lower MPC. Calibrated on the selection split only and locked, it narrowly beats the liquidity rule there (`0.0539` versus `0.0560`) and then fails the holdout at `0.8510`. Sign-correct, interpretable, and wrong out of domain.

The corrected Phase 2 rule is stricter than the original one: an architecture must be within 25 percent of raw GPT-5.5 on the UI holdout and within 25 percent of the best observed source on the selection split. Under that rule, no behavior architecture passes. The diagnosis is sharper now. The raw model has windfall signal that the rules miss, but it does not automatically solve predictable income-loss behavior. Primitive v3 may encode useful income-loss primitives, but it is not credible as the economy engine while it is wrong on the stimulus literature. The Phase 4 primary should therefore be architecture A, constrained raw, with primitive v3 reported only as a secondary variant.

Underneath the gate, the HANK-lite demand economy passes its full lab surface: transfer MPC gradients, rate-hike contraction, job-risk precaution, belief feedback, and per-period accounting identities, with the GPT-5.5 belief module clearing all 19 validation metrics in the live 12-cell run. The sandbox is playable and accounting-safe; what it awaits is a behavior layer worth putting inside it.

## The Persona Layer

The persona panel asks whether data-grounded personas reproduce the cross-sectional structure of household beliefs. The live panel covers 54 synthetic-enriched SCE-style respondents across GPT-5.5 and GPT-5.4 — 108 model responses — anchored to public aggregate survey beliefs and vintage macro context.

Structure passes cleanly: all 24 demographic contrasts score with the correct sign, the median within-variance ratio is `1.1025` (no stereotype flattening), and the maximum cross-model common-core correlation is `0.8912`, below the `0.95` collapse threshold. Distribution shape fails: maximum KS statistic `0.7407` against a `0.35` threshold, driven by unemployment expectations (target mean `4.43`, models predict `5.22-5.34`) and upward-shifted inflation.

The scope note that governs this section: the targets are synthetic, anchored to public aggregates. The panel validates the wiring and scoring machinery end to end; empirical persona claims wait on real respondent-level microdata (SCE or Michigan).

That real-data path has now been exercised on public SCE microdata. The converter reads the NY Fed public-use SCE workbooks, uses `frbny-sce-public-microdata-latest.xlsx` as the target panel, and uses all three raw workbooks for demographic backfill. It writes `work/persona_beliefs/sce_real_microdata.csv`: 78,996 scoreable monthly responses from 10,617 respondents, January 2020 through August 2025. Inflation is `Q9_cent50`; unemployment is scored as the actual SCE question, `Q4new`, the percent chance that U.S. unemployment will be higher in 12 months; real income growth is derived as nominal expected household income growth (`Q25v2part2`) minus the respondent's inflation expectation.

The holdout-prep CLI aligns real SCE waves to the vintage FRED/SPF environment, writes static and panel holdouts, normalizes weights by period, and records `real_sce_microdata_v1` provenance. The current real holdout files contain 951 static respondents for December 2024 and 3,022 panel rows across October-December 2024.

The mandatory 4-call canary also ran: two respondents, GPT-5.5 and GPT-5.4, via `openai_responses`, with zero cache hits. Prompt-card inspection found no leakage of `actual_*` target responses, raw question codes, or SCE target columns. The canary was too small to score as evidence, but it correctly flagged the hard problem: both models overpredicted the SCE unemployment-higher probability for the two sampled respondents by large margins.

The approved design run is now complete. It sampled 500 of the 951 December 2024 respondents, stratified across `income_group x age_group x education_group`, with seed `20260707`; every non-empty stratum is represented. It ran GPT-5.5 and GPT-5.4 through `openai_responses`: 1,000 live calls, zero cache hits, 3,000 prediction rows. The December 2024 wave is the profile-only test surface and is now spent.

The result is a clear empirical failure for the current persona layer:

- Evidence verdict: `null_gradient_failure`, branch `gradients_flat_or_wrong`.
- Regression sign-match rate: `0.625`, below the pre-registered `0.75` threshold.
- Median within-variance ratio: `0.0129`, far below the `0.5` threshold.
- Max weighted KS statistic: `0.5475`, above the `0.35` threshold.
- Median distribution std ratio: `0.1190`, below the `0.45` lower bound.
- Common-core check passes: max mean pairwise source correlation `0.8719`, below the `0.95` collapse threshold.

The failure mode is informative. Inflation gradients mostly line up, and real-income gradients partly line up. The unemployment-higher-probability target breaks the persona story: raw survey mean is `35.84`; GPT-5.5 predicts `45.16` and GPT-5.4 predicts `53.36`. More importantly, the models invert several real gradients: older, less-educated, female, and low-income respondents in the sampled SCE wave report lower unemployment-higher probabilities than their reference groups, while the model generally pushes those groups higher. Across all targets, the simulated distributions are far too narrow: inflation simulated standard deviation is `0.60-0.70` versus survey `5.84`; real-income-growth standard deviation is `1.13-1.18` versus survey `9.97`; unemployment-higher-probability standard deviation is `6.75-8.45` versus survey `26.41`.

This is not a prompt-leakage or data-plumbing failure. It is the result the empirical gate was designed to find: profile-only LLM personas do not currently reproduce real SCE belief heterogeneity well enough to serve as validated household belief agents.

The follow-up prior-conditioned update gate changes the persona story, but only in the narrower direction that matters for an economy. It sampled 100 repeated respondents from the October-November 2024 SCE panel, stratified across `income_group x age_group x education_group`, with seed `20260707`. Prompts were date-blind (`period_0`, `period_1`), used `prior-mode empirical`, and included each respondent's own previous beliefs plus the as-of macro environment. They did not include held-out current responses, raw SCE question codes, December 2024 rows, or `actual_*` target columns.

That run used the Codex CLI provider, not the direct API: GPT-5.5 and GPT-5.4, 200 panel rows, 400 total model records. The first invocation reached 396 records and failed on one empty Codex response; the successful final invocation reused those 396 cache records and made the remaining 4 live calls. The cache now contains the full 400-record Codex run.

The locked primary update gate clears for GPT-5.5:

- Evidence verdict: `clears_prior_update_gate`.
- Mean update correlation: `0.3578`, above the `0.10` threshold.
- Mean direction accuracy: `0.5769`, above the `0.55` threshold.
- Median update-amplitude ratio: `0.3234`, inside the `[0.25, 2.00]` band.
- Mean RMSE improvement versus persistence: `0.0437`, above the `0.02` threshold.

GPT-5.4 also clears directionally: mean update correlation `0.3976`, direction accuracy `0.5677`, median update-amplitude ratio `0.4140`, and mean RMSE improvement versus persistence `0.0575`. The positive result is not a broad distributional pass. In the same run, both models still compress within-group variance: mean within-variance ratio is `0.1677` for GPT-5.5 and `0.1266` for GPT-5.4. GPT-5.5's distribution distance is better than the December profile-only run but still misses the unemployment and real-income shapes: max KS `0.3691` against the old `0.35` distribution threshold. The stricter period-to-period delta table is also weak: simulated October-to-November deltas are negatively correlated with actual respondent deltas. So the result is specific: empirical priors help the model choose useful prior-to-current belief updates; they do not yet make a full household-belief simulator.

## What Comes Next

1. **Use prior-conditioned agents, not profile-only agents, for the next ecology.** Profile-only personas failed. Prior-conditioned agents cleared a first update gate. The next economy run should treat households as persistent belief states, not demographic sketches.
2. **Test the update mechanism once on a fresh panel holdout before calling it validated.** October-November 2024 has now been used. A future prior-conditioned architecture should be locked on validation data and tested once on a newly declared SCE or Michigan panel wave.
3. **Do not promote behavior yet.** The strongest result remains the contamination-audited macro belief engine. The behavior result is windfall-scoped, the income-loss result is negative, and no behavior architecture clears both in-domain and holdout bars.
4. **Let the post-cutoff forecast gate accrue.** Frozen post-cutoff cards get rescored as public data arrives, converting the forecast claim from recall-audit-defended to genuinely post-cutoff over time. The repo has a zero-live-call `postcutoff-replay-refresh` Make target for that loop.

The forecast audit — baselines, bootstrap intervals, DM-style tests, both recall probes, belief-structure audit, cutoff status — stays fixed in this report as the evidence base.
