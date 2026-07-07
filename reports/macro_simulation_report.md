# Macro LLM Simulation: What the Evidence Now Says

## Executive Finding

The project started with a broad question: can LLM agents simulate the macroeconomy well enough to produce useful forecasts and counterfactuals?

The answer is now narrower, cleaner, and more useful. Frontier LLMs are good macro belief engines. They forecast hidden macro outcomes better than strong statistical baselines, survive direct recall audits, and update beliefs in the right direction when given a person's prior beliefs. They are not, at least in the current evidence, reliable generators of individual household heterogeneity from demographics or backstories. And the project has not yet shown a validated LLM-based simulated economy.

The short version:

**LLMs are belief updaters, not belief generators. Individual heterogeneity has to come from data; once it is supplied, the model can partly move it in the right direction.**

That is not the original maximal claim, but it is a real result. It says where LLMs add signal, where they fail, and what an honest macro-agent architecture should look like next.

## Results at a Glance

| Question | Result | Evidence |
| --- | --- | --- |
| Can GPT-5.5 and GPT-5.4 forecast hidden macro outcomes? | Yes. | On 147 date-free vintage macro cards, raw GPT-5.5 WNAE is `0.8945`; GPT-5.4 is `0.9064`; the best deterministic baseline, no-change, is `1.1260`. GPT-5.5 improves over no-change by `20.56%`, with an origin-cluster bootstrap interval of `[0.0688, 0.4153]`. |
| Is the forecast result explained by remembered realized values? | No evidence of that. | Live recall probes found zero exact realized-value recall across all 147 cards. Qualitative recall was near the majority baseline: GPT-5.5 was `+0.0181`; GPT-5.4 was `-0.0136`. |
| Does raw GPT-5.5 predict household behavior better than simple rules? | Only in one out-of-domain family. | It loses to a tuned liquidity rule on stimulus targets, wins on lottery windfalls, then loses again on UI-exhaustion income-loss targets. |
| Can interpretable kernels preserve the behavior signal? | Not yet. | Blend and primitive-to-action mechanisms either fail holdout tests or overfit selection targets. Sign-correct primitives are not enough. |
| Can profile-only LLM personas reproduce real SCE household belief heterogeneity? | No. | In a 500-respondent December 2024 SCE run, the evidence verdict is `null_gradient_failure`; median within-variance ratio is `0.0129`; max weighted KS is `0.5475`. |
| Can richer backstory prompts fix that failure? | No. | In the November 2024 validation campaign, both GPT-5.5 and GPT-5.4 receive `backstory_caricature`; Arm 3 is skipped by design. |
| Can prior-conditioned agents update beliefs? | Yes, modestly. | On repeated October-November 2024 SCE respondents, GPT-5.5 clears the locked prior-update gate: update correlation `0.3578`, direction accuracy `0.5769`, amplitude ratio `0.3234`, and RMSE improvement versus persistence `0.0437`. |
| Is the economy ready as a validated macro simulator? | No. | The accounting-safe demand sandbox works, and Phase 4 now runs an exploratory matched-twin replay, but the LLM-updater economy loses to the adaptive-expectations twin on the first FRED proxy pass. |

## Finding 1: The Aggregate Belief Engine Works

The forecast tournament builds as-of prompt cards from vintage FRED/ALFRED macro data. Cards are date-free: prompts use relative periods and hide calendar dates, named crises, and realized target values. Every source forecasts the same hidden targets and is scored on weighted normalized absolute error (WNAE).

| Source | Test WNAE | Interpretation |
| --- | ---: | --- |
| GPT-5.5 raw | 0.8945 | Best overall |
| GPT-5.4 raw | 0.9064 | Pass |
| No change | 1.1260 | Best deterministic baseline |
| Rolling trend | 1.2668 | Baseline |
| Rolling mean | 1.2790 | Baseline |
| GPT-5.4 calibrated | 1.3933 | Calibration hurts |
| GPT-5.5 calibrated | 1.8851 | Calibration hurts |
| AR(2) | 2.4307 | Baseline |
| Recursive least squares | 4.2816 | Baseline |

GPT-5.5 lowers mean origin-level loss by `0.2315` versus the best deterministic baseline, a `20.56%` improvement. The origin-cluster bootstrap interval is `[0.0688, 0.4153]`, and `99.82%` of draws are positive. Direction accuracy is `69.39%` for both GPT models versus `6.80%` for no-change.

The cleanest family-level win is real consumption growth: GPT-5.5 beats no-change by `29.96%`, with bootstrap mean loss reduction `[0.2786, 1.1281]` and a DM-style p-value of `0.0033`. Output, policy rates, and unemployment improve by `20-34%` against their best baselines, though those intervals cross zero. Inflation is the miss: rolling trend beats GPT-5.5 on inflation growth (`0.2626` versus `0.2904`).

The raw model is the result. A validation-fitted residual calibration makes held-out forecasts worse, so the current evidence does not support post-hoc calibration as a free improvement. Confidence does carry information: the confidence-error correlation is `-0.5453` for GPT-5.5 and `-0.5646` for GPT-5.4.

### Contamination audit

The test targets run from `2020-01-01` to `2025-02-01`. Against the Codex GPT-5 cutoff of `2024-09-30`, 14 of 147 cards are post-cutoff, so the main forecast tournament is a hidden-target vintage test, not a purely post-cutoff test. The contamination defense is the recall audit, run live on the same 147 cards.

- Exact recall is zero. Neither model returned a usable numeric realized value for any card when asked directly, without tools, whether it remembered the outcome.
- Qualitative recall is near the base rate. Asked about path direction, level versus normal, and crisis/calm status, GPT-5.5 scores `+0.0181` above the qualitative majority baseline on mean card accuracy; GPT-5.4 scores `-0.0136` below it.
- The forecasts do not look like a disguised trend extrapolator. GPT-5.5's underreaction slope is `-0.0279`, closest to zero among scored sources, against `-1.13` to `-1.15` for rolling baselines. Its extrapolation slope is `-0.1940`, versus `-1.0000` for rolling mean and `-1.4819` for AR(2).

## Finding 2: Behavior Signal Exists, But It Is Shock-Specific

The behavior gate asks whether model-generated household responses match published behavioral moments better than simple rules. The rules are deliberately strong: a liquidity rule, a flat rule, and a permanent-income rule.

The result is split, and the split is the point.

On stimulus-style targets, raw GPT-5.5 loses to the liquidity rule. Aggregate range RMSE is `0.0762` for GPT-5.5 versus `0.0560` for the liquidity rule. On cell-level EIP MPC, GPT-5.5 is `0.0387`; the liquidity rule is `0.0272`.

On the lottery-windfall holdout, the ordering reverses. Raw GPT-5.5 scores `0.0844`. The best simple rule, the flat 30 percent rule, scores `0.2608`. The in-domain liquidity rule breaks badly at `1.1231`, including a low/high liquidity gradient of `5.0` against a target of `2.0`. GPT-5.5 carries the shock-size intuition the rules miss: lottery MPC falls with prize size.

On the UI-exhaustion income-loss holdout, GPT-5.5 loses again. Raw GPT-5.5 scores `0.0397`; the best rule, a flat UI spending-drop rule, scores `0.0289`. A residual-over-liquidity ablation scores `0.0353`, and the primitive path scores `0.0311`; both lose.

So the behavior result is not "LLM agents predict household behavior." It is more specific: **raw GPT-5.5 contains behavioral priors that transfer to windfall-size reasoning, but not yet to predictable income-loss dynamics.**

## Finding 3: Interpretable Behavior Kernels Failed Productively

The project then tried to make the behavior signal usable inside an economy. That required moving away from raw allocation shares and toward interpretable mechanisms.

Two routes failed:

- The fixed 50 percent liquidity-prior blend won on selected cell targets, then failed the lottery holdout at `0.3857` versus the flat rule's `0.2608`. That demotes the blend to an ablation.
- The primitive-to-action kernel made GPT-5.5 emit bounded primitives only: perceived job-loss risk, expected income growth, precautionary motive, liquidity stress, debt-repayment urgency, durable pull-forward, log shock size, and confidence. Deterministic code then mapped those primitives to spending, saving, debt repayment, and liquidity. It passed every sign audit, narrowly beat the liquidity rule on the selection split (`0.0539` versus `0.0560`), and then failed the holdout at `0.8510`.

That is an important negative. A mechanism can have the right signs and still be wrong out of domain. The primitive interface preserved interpretability but destroyed the raw model's useful windfall signal.

The later architecture-fidelity run tested three behavior architectures on the UI holdout:

| Architecture | Result |
| --- | --- |
| Constrained raw | UI RMSE `0.0397`; credible as a baseline, but not a win. |
| Constrained choice | The first pass exposed a bounds bug that exploded selection RMSE to `248,350,850.7`. That was fixed for future selection-split work; the result is not interpretable as a holdout score. |
| Primitive v3 | UI RMSE `0.0200`, beating the flat rule (`0.0289`), but selection-split RMSE is `0.3401`, far worse than raw GPT-5.5 and the liquidity rule. Promising, but not credible as the economy engine. |

The corrected rule is now stricter: an architecture must stay close to raw GPT-5.5 on the holdout and stay close to the best observed source on the selection split. Under that rule, no behavior architecture passes.

## Finding 4: Profile-Only Personas Do Not Reproduce Real Household Beliefs

The persona layer asks whether LLMs can reproduce the cross-sectional structure of household beliefs. The synthetic wiring test passed enough to prove the machinery worked, but the real SCE test is the one that matters.

The real-data run used public New York Fed Survey of Consumer Expectations microdata. It sampled 500 December 2024 respondents from 951 available rows, stratified across `income_group x age_group x education_group`, with seed `20260707`. It ran GPT-5.5 and GPT-5.4 on profile-only prompts: 1,000 live calls and 3,000 prediction rows. The prompt-card audit found no target leakage.

The result is a clean empirical failure:

- Evidence verdict: `null_gradient_failure`.
- Regression sign-match rate: `0.625`, below the `0.75` threshold.
- Median within-variance ratio: `0.0129`, far below the `0.50` threshold.
- Max weighted KS statistic: `0.5475`, above the `0.35` threshold.
- Median distribution std ratio: `0.1190`, below the `0.45` lower band.
- Common-core check passes: max mean pairwise source correlation `0.8719`, below the `0.95` collapse threshold.

The failure mode is not random noise. Inflation gradients mostly line up, and real-income gradients partly line up. The unemployment-higher-probability target breaks the persona story. The raw survey mean is `35.84`; GPT-5.5 predicts `45.16`; GPT-5.4 predicts `53.36`. The models also invert several real gradients: older, less-educated, female, and low-income respondents in the sampled SCE wave report lower unemployment-higher probabilities than their reference groups, while the model generally pushes those groups higher.

The deeper failure is dispersion. Simulated distributions are far too narrow:

| Target | Survey standard deviation | Simulated standard deviation |
| --- | ---: | ---: |
| Inflation expectations | 5.84 | 0.60-0.70 |
| Real-income-growth expectations | 9.97 | 1.13-1.18 |
| Unemployment-higher probability | 26.41 | 6.75-8.45 |

This is not a data-plumbing failure. It is the result the gate was built to detect: profile-only LLM personas do not currently reproduce real SCE belief heterogeneity.

## Finding 5: Backstory Prompts Do Not Rescue Personas

The elicitation campaign tested whether richer prompting could rescue the profile-only failure. It used a locked November 2024 validation wave and ran through `codex_cli` only, with high reasoning effort.

Arm 0 rescored the spent December run using draws from each model's p10-p90 bands instead of point answers. This half-rescued the spread story, but only half. Median within-variance ratio rose to `0.4821`, near the `0.50` threshold. But p10-p90 interval coverage was only `0.3876` against an 80 percent target, and max weighted KS stayed at `0.3944` against a `0.35` threshold. The missing spread is partly in the bands, but the bands still miss too many real respondents.

Arm 1 ran the live point-vs-backstory test: the same 100 stratified November respondents, GPT-5.5 and GPT-5.4, point prompts versus backstory prompts. Neither model passed.

| Model | Arm 1 verdict | What happened |
| --- | --- | --- |
| GPT-5.5 | `backstory_caricature` | Spread improved by `2.40x`, below the `3.0x` threshold; KS worsened; group-mean error grew by `34.3%`. Gradient signs did not degrade, but the levels guard failed. |
| GPT-5.4 | `backstory_caricature` | Spread improved by `2.71x`, still below threshold; KS worsened; gradient signs degraded; group-mean error grew by `35.4%`. |

The caricature guard mattered. Backstories did add movement, but they moved group means in the wrong way without adding enough real individual dispersion.

Arm 2 asked each model, cold and without respondent hints, for the unconditional distribution of real SCE answers. This split the diagnosis cleanly. GPT-5.5's inflation deciles were good for this task, with decile MAE `0.8569`; GPT-5.4's inflation decile MAE was `0.8013`. Real-income-growth decile MAE was `0.9371` for GPT-5.5 and `1.5222` for GPT-5.4. But unemployment-higher-probability deciles were bad for both, with decile MAE `15.0`.

So the models know rough unconditional shapes for some belief distributions. They cannot place a specific respondent inside those distributions from demographics or invented life texture.

Arm 3, the sealed June 2025 priors-plus-backstory confirmation, did not run. That was the locked rule: Arm 3 fires only if Arm 1 passes for at least one model. No confirmatory calls were spent on the sealed wave.

## Finding 6: Prior-Conditioned Updating Is the Surviving Persona Result

The strongest persona result is not generation. It is updating.

The prior-conditioned update gate sampled 100 repeated SCE respondents from the October-November 2024 panel, stratified across `income_group x age_group x education_group`, with seed `20260707`. Prompts were date-blind, used `prior-mode empirical`, and included each respondent's own previous beliefs plus the as-of macro environment. They did not include held-out current responses, raw SCE question codes, December 2024 rows, or `actual_*` target columns.

GPT-5.5 clears the locked primary update gate:

- Mean update correlation: `0.3578`, above the `0.10` threshold.
- Mean direction accuracy: `0.5769`, above the `0.55` threshold.
- Median update-amplitude ratio: `0.3234`, inside the `[0.25, 2.00]` band.
- Mean RMSE improvement versus persistence: `0.0437`, above the `0.02` threshold.

GPT-5.4 clears directionally too: update correlation `0.3976`, direction accuracy `0.5677`, amplitude ratio `0.4140`, and RMSE improvement versus persistence `0.0575`.

The result is modest and underreactive. It does not fix the cross-sectional dispersion problem. Both models still compress within-group variance: mean within-variance ratio is `0.1677` for GPT-5.5 and `0.1266` for GPT-5.4. GPT-5.5's max KS is `0.3691`, still just above the old `0.35` distribution threshold. The stricter period-to-period delta table is weak.

But this is the architecture lesson. Give the model a person's prior belief state, and it can partly model how that belief moves. Ask it to conjure the person from demographics or a backstory, and it collapses the distribution.

## What This Means for the Macro Economy

The accounting-safe demand economy is useful, but the full macro-agent claim is not yet achieved.

The sandbox can run transfer shocks, rate hikes, job-risk shocks, belief feedback, and per-period accounting checks. It passes its lab validation surface. The problem is upstream: the household belief and behavior layers do not yet supply a validated empirical engine.

The right economy architecture now looks different from the early "personas simulate households" idea:

1. Start households with empirical belief states or latent states inferred from panel data.
2. Let LLMs update those states from new information, rather than inventing the states from profiles.
3. Keep deterministic code responsible for feasibility, budgets, accounting, and market clearing.
4. Treat behavior mechanisms as separately validated modules, not as free-form LLM allocation guesses.
5. Compare the resulting economy to an adaptive-expectations twin: the same demand economy with LLM belief updates swapped out for a standard adaptive baseline.

That final comparison now has a fixture harness and an exploratory replay. `outputs/phase4_matched_twins_fixture/` locks the output-to-proxy mapping, runs the LLM-belief and adaptive-expectations twins from the same initial state, preserves accounting, and emits comparable post-cutoff proxy scores. Its verdict is `phase4_matched_twin_fixture_ready`, with max accounting residual `2.91e-11` and zero live calls.

The first real-SCE prior-update replay uses the banked Codex ecology run in `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100/`, filters to `llm_codex_cli_gpt-5.5`, and feeds those prior-conditioned belief updates into the deterministic demand economy. The Phase 4 output mapping is now schema v2: `personal_saving_rate_pct` is scored as month-over-month change in the saving-rate proxy, not the saving-rate level, and that transform is applied identically to both twins and to the FRED target before scoring.

The existing strict one-card FRED proxy replay has been rescored under v2 and labeled retrospective. Adaptive still wins: scaled RMSE `0.5860` versus `4.7165` for the LLM-updater path. The LLM-updater path has better direction accuracy (`1.0000` versus `0.6000`), but it remains too pessimistic on consumption growth. The five-card `hold_last` ablation, also rescored retrospectively under v2, says the same thing: adaptive scaled RMSE `1.0745`; LLM-updater scaled RMSE `2.9051`. Accounting passes in both runs. Confirmatory scoring under mapping v2 begins with the next newly scoreable data month. This is a useful negative: the validated belief updater is not yet enough to improve the macro proxy economy.

## What We Can Claim Now

The evidence supports three positive claims:

1. **Frontier LLMs contain audited macro belief signal.** GPT-5.5 and GPT-5.4 beat strong empirical baselines on a hidden-target vintage macro tournament, and live recall probes do not find realized-value recall.
2. **Raw GPT-5.5 contains behavior signal in one out-of-domain windfall family.** It generalizes where tuned rules break, but that result does not transfer to predictable income-loss dynamics.
3. **Prior-conditioned agents can partly update real household beliefs.** The model is useful when it operates on supplied respondent state.

The evidence also supports three negative claims:

1. **Profile-only personas fail on real SCE microdata.**
2. **Backstory elicitation fails by caricature rather than rescuing heterogeneity.**
3. **No behavior architecture currently clears both in-domain credibility and out-of-domain holdout tests.**
4. **The first Phase 4 matched-twin replay does not beat adaptive expectations.**

The full claim remains open:

**We have not yet shown that an LLM-based simulated economy predicts real macro behavior better than strong empirical alternatives.**

## Recommended Next Work

The next phase should not chase richer personas. It should build from the result that survived.

1. Build a longer, same-horizon prior-update panel before spending more Phase 4 score surface.
2. Revisit the bridge from SCE beliefs to demand-economy primitives, especially the unemployment-higher-probability to personal job-risk mapping and the excessive consumption-growth contraction in the LLM-updater path.
3. Keep the Phase 4 v2 output mapping locked unless there is a pre-registered replacement.
4. Run the next matched-twin comparison only when the household panel horizon, belief replay horizon, and proxy scoring horizon are aligned.
5. Keep the post-cutoff forecast gate running in the background as new public data becomes scoreable.

Everything else is lower priority. The forecast evidence is already strong. The backstory route is closed. The behavior route needs new mechanisms or new data before another holdout is spent.

## Methods Appendix

### Forecast tournament

The forecast tournament uses 147 date-free vintage macro cards built from as-of FRED/ALFRED data. Prompts hide realized values, calendar crisis labels, and target dates. Sources are scored on WNAE and compared to no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines. Family-level comparisons use bootstrap and DM-style checks.

### Recall audit

The recall audit asks the live model, without tools, whether it remembers the realized value or qualitative path for each card. Exact realized-value recall is scored separately from broad qualitative era recall.

### Behavior gate

The behavior gate scores model and rule-generated household actions against published public moments. Splits are kept separate:

- `behavior_selection_v1`: stimulus/rebate targets used during development.
- `behavior_holdout_v1`: lottery windfall targets. This holdout is spent and frozen.
- `behavior_holdout_ui_v1`: UI-exhaustion income-loss targets.

### Persona gates

The real persona gates use public SCE microdata. Inflation is the SCE density-median one-year inflation expectation. Unemployment is scored as the actual SCE question: percent chance U.S. unemployment will be higher in 12 months. Real income growth is derived as nominal expected household income growth minus the respondent's inflation expectation.

The December 2024 profile-only wave is spent. The November 2024 backstory validation wave is spent. The sealed June 2025 priors-plus-backstory confirmation was not spent because the pre-registered Arm 1 condition failed.

### Phase 4 matched twins

The Phase 4 fixture compares two versions of the same deterministic demand economy: an LLM-belief fixture and an adaptive-expectations twin. It writes the locked proxy mapping, cards, targets, household states, twin paths, accounting, forecasts, joined errors, scores, manifest, and `phase4_matched_twins_report.md`.

The Phase 4 replay adapter consumes persona-ecology predictions rather than raw prompt payloads. It joins on the normalized CSV outputs, not prompt-facing relative IDs; derives household states from demographics, weights, and prior beliefs; and excludes `actual_*` labels from the demand-economy input. The strict Codex replay artifact is `outputs/phase4_matched_twins_prior_update_codex_replay_fred_onecard/`. The extrapolating ablation is `outputs/phase4_matched_twins_prior_update_codex_replay_fred_holdlast_5cards/`. Both current replay artifacts use mapping schema v2 and `scoring_label: retrospective`; the first confirmatory v2 score is reserved for the next newly scoreable data month.

### Reproducibility notes

The canonical report in the repository is `reports/macro_simulation_report.md`, with the sendable copy exported to `Downloads/macro_simulation_report.md`. The live elicitation campaign artifacts are under `outputs/persona_elicitation_campaign/`. The Phase 4 fixture artifacts are under `outputs/phase4_matched_twins_fixture/`. The latest full test run passed `141` tests.
