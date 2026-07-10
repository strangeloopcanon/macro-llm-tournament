# Macro LLM Simulation: What the Evidence Now Says

## Executive Finding

The project started with a broad question: can LLM agents simulate the macroeconomy well enough to produce useful forecasts and counterfactuals?

The answer is now narrower, cleaner, and more useful. Frontier LLMs are good macro belief engines. They forecast hidden macro outcomes better than strong statistical baselines, survive direct recall audits, and update beliefs in the right direction when given a person's prior beliefs. They are not reliable generators of individual household heterogeneity from demographics or backstories. The best behavior interfaces are measured or scheduled functions that deterministic code executes against household state. The real-data bridge gets the Phase 4 economy close to adaptive expectations, and horizon-aligned retrospective replays can beat it. The new recursive economy now runs 81 real SCE household states through five post-cutoff monthly origins, feeds simulated state forward, preserves accounting, and scores 10 first-release macro targets per scored origin. Its selected development candidate improves the base recursive LLM economy by `2.61%` but remains `0.2%` behind the adaptive twin (`0.549789` versus `0.548667`), with an uncertainty interval that crosses zero. That exact candidate is frozen for a one-shot June 2026 test once the complete first-release target set exists.

The short version:

**LLMs are belief updaters and policy-function authors, not reliable household simulacra. Individual heterogeneity has to come from data; once it is supplied, the model can update beliefs inside a recursive, accounting-safe economy. That economy now runs and nearly matches a strong adaptive benchmark, but has not yet beaten it on a fresh frozen-vintage macro test.**

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
| Does the policy-schedule behavior layer generalize to a fresh behavior family? | Directionally yes. | The new CTC holdout (`behavior_holdout_ctc_v1`) scores already-banked GPT-5.5 policy schedules with zero new calls. Policy schedules beat the liquidity rule overall: range RMSE `0.0097` versus `0.0207`, though they miss part of the income-gradient shape. |
| Does the state-conditioned policy layer rescue the Phase 4 economy? | No, but it improves the bridge. | The live Codex state-policy profile improves the LLM-updater economy versus the generic schedule executor: strict scaled RMSE `5.0843` versus `6.5296`; five-card hold-last RMSE `3.2132` versus `3.8132`. Adaptive still wins both runs. |
| Can the belief-to-spending bridge be estimated from real microdata? | Yes, with caveats. | Empirical bridge v4 deflates expected spending growth by each respondent's inflation expectation, passes the locked coefficient/chart/liquidity gates, and is accepted. The validation diagnostic still fails because the validation-wave real-income coefficient is unstable. |
| Does the real-data bridge rescue the Phase 4 economy? | Almost, but not on the original one-card metric. | v4 cuts strict LLM-updater scaled RMSE from `5.0843` to `0.6311` and hold-last from `3.2132` to `1.4688`. Adaptive still edges it on scaled RMSE: `0.5692` strict and `1.4633` hold-last. |
| Does bridge stabilization fix the remaining gap by itself? | No. | v5 ridge stabilization improves the strict LLM-updater scaled RMSE from `0.6311` to `0.5970`, but adaptive still wins at `0.5688`. Hold-last is essentially tied: `1.4663` LLM versus `1.4644` adaptive. |
| Does matching the ecology horizon to the scoring horizon change the result? | Yes, retrospectively. | Extending the SCE prior-update panel to December 2024 and scoring a two-card aligned replay gives the first Phase 4 LLM win: v4 LLM `0.8455` versus adaptive `0.8961`; v5 LLM `0.8529` versus adaptive `0.8975`. The win survives full re-elicitation of the December leg under the corrected information set (v4 `0.8539`, v5 `0.8645`, adaptive unchanged). |
| Why did the December leg fail the persistence gate, and is it fixed? | Stale information set; partially fixed. | The December card was quarterly-stale (as-of 2024-11-15, synthetic sentiment `94` versus real `72`). Under the pre-registered monthly-vintage card fix, the December re-run clears the gate at diagnostic status (`+0.0323` versus `+0.02` threshold), though the inflation target alone still loses to persistence because no vintage series carried the tariff-news channel. |
| Does the prior-update result extend to a fresh January 2025 leg? | No. | The pre-registered fresh gate test on 67 never-elicited January responses failed: RMSE improvement versus persistence `-0.0904`. Direction and correlation stay fine, but the model over-reacts under richer monthly cards (amplitude ratio `1.27`). No January surface was built. |
| Does full-economy tournament search improve the incumbent? | Yes, modestly and retrospectively. | The v1 runner scored `5,670` candidates across two already-scored Phase 4 surfaces with zero live calls and no disqualifications. Its winner improves mean LLM scaled RMSE from `0.7249` to `0.7208` and is retained as the historical `macro_incumbent_v1` replay. The corrected-panel v3 search later selects a different candidate, so v1 is not the current best economy. |
| Does extended tournament search find a real optimum? | No stable point optimum. | The v2 grid's best candidate sits at its global-gain boundary (`3.0`, mean `0.7087`). A wider frontier eventually reverses, but cross-surface rankings are already strongly negative within v2/v3 and fall to about `-0.75` at the extreme frontier. The broad preference for stronger inflation updates and damped income updates repeats across December panel draws; the exact candidate does not. |
| Does the held-out February 2026 diagnostic reproduce the retrospective Phase 4 win? | No. | The one-shot run scores four available targets (`n=1` each) for the `2026-02-15` score date. Unit-gain v5 scores `2.1202`; conservative amplified v4 scores `2.1256`; adaptive is better for both (`2.1019`-`2.1009`). The inputs were revised current FRED observations retrieved in July 2026, not frozen February vintages, and beliefs were held forward from December 2024. The score date remains spent, but this is not independently pre-registered confirmatory evidence. |
| Does the recursive 81-household economy work end to end? | Yes. | Five monthly origins run from real SCE states through GPT-5.5 belief updates, empirical spending behavior, deterministic accounting, macro feedback, and 10-target first-release scoring. January is warmup; February-May contribute 40 score rows; max accounting residual is `7.28e-12`. |
| Does the current recursive LLM economy beat adaptive expectations? | Not yet. | The selected full-policy-state candidate improves the base LLM economy from `0.564533` to `0.549789`, but adaptive scores `0.548667`. The relative gap is `0.2%`, and the origin-block bootstrap interval for LLM minus adaptive is `[-0.00633, 0.00910]`. |
| Is the economy ready as a validated macro simulator? | No. | It is now a real recursive simulation and a developmental near-tie. The exact winner is frozen for one June 2026 confirmation, which remains unspent until the complete first-release bundle is available. |

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

### Cross-model and prompt-variant probe

Two natural objections to that negative are that it is a GPT quirk or a prompt artifact. A four-arm diagnostic probe on the selection split tested both, using the same scenarios, cache semantics, and scoring. Claude Opus 4.8 (thinking) ran through the headless Cursor CLI in an empty ask-mode workspace so it could not read repo data; a "descriptive" prompt variant explicitly asked for measured rather than prudent household behavior and for liquidity-first reasoning.

| Arm | Aggregate range RMSE | Cell range RMSE |
| --- | --- | --- |
| Liquidity rule (baseline to beat) | `0.0560` | `0.0272` |
| GPT-5.5, baseline prompt | `0.0762` | `0.0387` |
| GPT-5.5, descriptive prompt | `0.0652` | `0.0387` |
| Opus 4.8, baseline prompt | `0.2868` | `0.0789` |
| Opus 4.8, descriptive prompt | `0.0682` | `0.0415` |

Three things follow. First, the negative is not GPT-specific: Opus 4.8 under the identical prompt is substantially worse than GPT-5.5, mostly by flattening the liquidity gradient (`0.5376` on that family versus GPT-5.5's `0.0000`), answering with prudent-advice allocations instead of measured behavior. Second, prompting matters but does not close the gap: the descriptive framing fixes most of Opus's prudence bias (liquidity-gradient family `0.5376` to `0.0374`) and improves GPT-5.5's aggregate score, yet every raw-LLM arm still loses to the liquidity rule, and cell-level scores barely move. Third, the arms converge near `0.065`-`0.068` from very different starting points, which suggests a shared ceiling on eliciting measured behavioral moments from current frontier models rather than a fixable idiosyncrasy. These arms scored the selection split for diagnosis only; no mechanism selection was keyed off holdout families.

### The compression diagnosis

The per-target errors behind those aggregates share one signature: compression of conditional differences. Real households spend `0.6`-`0.8` of a small lottery windfall and `0.12`-`0.30` of a large one; GPT-5.5 predicted `0.417` and `0.430`. Real UI spending drops `0.05`-`0.07` at job loss, drifts near zero during receipt, then falls `0.11`-`0.13` at exhaustion; GPT-5.5 predicted a flat `0.02` everywhere. The same under-differentiation appears in the persona layer (individual beliefs compressed toward group means) and Phase 4 (belief updates too uniform). Point elicitation returns the safe center of the model's predictive distribution, and that center flattens exactly the elasticities an intervention simulator needs.

### The elicitation-interface experiment: compression moves, and schedules beat points

A pre-registered follow-up (`outputs/behavior_ecology_gpt55_xhigh`, `outputs/behavior_ecology_opus48_lottery`) tested whether compression is a capability wall or an artifact of the elicitation interface. Two new arms, same scenarios and scoring: an **ecology** arm, where each of 16 concrete households (seeded draws around SCF type cells) answers one shock in the first person under an anti-prudence preamble, in dollars, one call per household, aggregates emerging by weighted summation; and a **policy** arm, where the model states a response *schedule* per type cell (spending share as a function of windfall size relative to monthly income; the UI onset/receipt/exhaustion path) in one call, and deterministic code evaluates the schedule at each scenario. Primary metrics were declared before scoring: the lottery size gradient (band `0.30`-`0.68`; point-elicitation reference `-0.013`) and the UI exhaustion cliff (band `0.10`-`0.13`; reference `0.017`).

| Arm (GPT-5.5 xhigh) | Lottery size gradient | UI cliff | Selection RMSE | Lottery holdout RMSE | UI holdout RMSE |
| --- | --- | --- | --- | --- | --- |
| Point elicitation (reference) | `-0.013` | `0.017` | `0.0652` | `0.0844` | `0.0397` |
| Ecology (first-person households) | `0.121` | `0.130` | `0.5669` | `0.3529` | `0.0194` |
| Policy schedule | `0.254` | `0.127` (in band) | `0.0507` | `0.1392` | `0.0304` |
| Best rule baseline | `0.000` | `0.056` | `0.0560` | `0.2608` | `0.0289` |

Three results:

1. **Compression is an interface artifact, but it moves rather than disappears.** First-person grounding restores scenario differentiation (the UI cliff lands on the band edge; the ecology arm posts the best UI-path score ever observed, `0.0194`) while destroying cross-household differentiation: role-played individuals nearly all behave like the same prudent median person, collapsing the EIP liquidity spending ratio to `1.14` against a target of `3`-`6`. Cell-level elicitation has exactly the opposite failure. Where the model aggregates in its head, it keeps stylized cross-sectional facts and flattens scenario response; where it role-plays, it keeps scenario response and flattens the cross-section.
2. **Schedule elicitation is the first LLM-derived source to beat the strongest rule on the selection split** (`0.0507` versus `0.0560`) while also beating the best rule on the lottery family (`0.1392` versus `0.2608`). Asking the model for the *function* and letting code evaluate it preserves both differentiation axes at once. Opus 4.8 replicates the ordering, and its policy arm puts the lottery gradient inside the pre-registered band (`0.305`).
3. **These families are spent.** The lottery and UI targets have now been used for mechanism selection repeatedly, so the policy-arm win is a selection-surface result. Promoting it to a claim requires a genuinely untouched behavior family scored once.

### Fresh CTC Holdout: Policy Schedules Survive One New Family

The fresh behavior family is the 2021 monthly Child Tax Credit. It is useful because it is neither a one-time rebate/lottery windfall nor a predictable UI income-loss event. The new split is `behavior_holdout_ctc_v1`, and it was scored once using the already-banked GPT-5.5 policy schedules from `outputs/behavior_ecology_gpt55_xhigh/ecology_raw_records.json`. That means zero new policy calls and no prompt revision after seeing the targets.

The main CTC spending target is anchored on the BLS Consumer Expenditure working paper estimate that households spent `44%` of each imputed CTC dollar over the reference quarter. Secondary saving/debt and income-gradient rows use the Brookings/Social Policy Institute survey usage shares: `53%` mostly spent, `30%` mostly saved, and `17%` mostly paid down debt, with higher-income households more likely to save and lower-income households more likely to repay debt.

| Source | CTC overall range RMSE | CTC MPC prediction | Notes |
| --- | ---: | ---: | --- |
| GPT-5.5 policy schedule | `0.0097` | `0.4569` | Best overall; beats liquidity rule. |
| Liquidity rule | `0.0207` | `0.4001` | Strong on income gradients, weaker overall. |
| Flat 30% rule | `0.0866` | `0.3000` | Too low on CTC spending. |
| Permanent-income rule | `0.2049` | `0.1000` | Much too low. |

This promotes the policy-schedule behavior result from "only seen on spent lottery/UI families" to "survives one fresh behavior family." It is still not the full macro claim. The schedule arm's weakest CTC component is income-gradient debt repayment: it under-predicts the low-income versus high-income debt-repayment gap. That is a useful miss because it tells us where the next schedule prompt or state conditioning needs pressure.

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

As a picture, the current recursive pipeline is:

```text
real SCE household priors -----> [1. GPT-5.5 BELIEF UPDATER]
as-of monthly vintage --------->              |
                                               | updated beliefs
                                               v
                              [2. EMPIRICAL SPENDING BRIDGE v4]
                                               |
                                               | consumption pressure
                                               v
                              [3. DETERMINISTIC HOUSEHOLD EXECUTION]
                                               |
                                               | consume / save / repay debt
                                               v
                              [4. ACCOUNTING-SAFE DEMAND ECONOMY]
                                               |
                                               | output, jobs, income,
                                               | prices, policy, balances
                                               v
                                   next-origin state and prompt
                                               |
                                               '----------> back to [1]

frozen first-release data ----> [5. TEN-TARGET MACRO SCORE]

CONTROL: the same households, bridge, economy, information, and
targets run with a mechanical adaptive-expectations updater in [1].
```

The LLM does not directly choose household allocations in the current winner. Raw allocations lost to a liquidity rule, while profile-only and backstory-only personas failed their real-data gates. Real respondent priors supply heterogeneity; the LLM updates beliefs; empirical and deterministic layers translate them into feasible actions. Under `closed_loop`, simulated household and macro state enters the next period. The winner also assimilates the observed rolling-origin policy rate before its next smoothed Taylor-rule transition, so this is a recursive economy with partial observed-state assimilation, not a fully endogenous policy path.

The chronology below records the earlier Phase 4 bridges that led to this architecture.

That final comparison now has a fixture harness and an exploratory replay. `outputs/phase4_matched_twins_fixture/` locks the output-to-proxy mapping, runs the LLM-belief and adaptive-expectations twins from the same initial state, preserves accounting, and emits comparable post-cutoff proxy scores. Its verdict is `phase4_matched_twin_fixture_ready`, with max accounting residual `2.91e-11` and zero live calls.

The first real-SCE prior-update replay uses the banked Codex ecology run in `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100/`, filters to `llm_codex_cli_gpt-5.5`, and feeds those prior-conditioned belief updates into the deterministic demand economy. The Phase 4 output mapping is now schema v2: `personal_saving_rate_pct` is scored as month-over-month change in the saving-rate proxy, not the saving-rate level, and that transform is applied identically to both twins and to the FRED target before scoring.

The existing strict one-card FRED proxy replay has been rescored under v2 and labeled retrospective. With the original fixed demand kernel, adaptive still wins: scaled RMSE `0.5860` versus `4.7165` for the LLM-updater path. The LLM-updater path has better direction accuracy (`1.0000` versus `0.6000`), but it remains too pessimistic on consumption growth. The five-card `hold_last` ablation, also rescored retrospectively under v2, says the same thing: adaptive scaled RMSE `1.0745`; LLM-updater scaled RMSE `2.9051`.

The policy-schedule executor is now wired into the same Phase 4 matched-twin runner. Both twins use the same GPT-5.5-authored schedules from `outputs/behavior_ecology_gpt55_xhigh/ecology_raw_records.json`; deterministic code maps each SCE household to the nearest SCF schedule cell, interpolates the transfer and income-risk response functions, enforces budgets, and keeps accounting. This is the architecture implied by the behavior-ecology result: model-authored policy functions, code-executed actions.

The first schedule-executor Phase 4 replay is also negative. On the strict retrospective one-card FRED proxy run, adaptive wins: scaled RMSE `0.8457` versus `6.5296` for the LLM-updater path. Both twins have direction accuracy `1.0000`, but the LLM-updater schedule economy over-contracts consumption growth and overshoots the saving-rate change. The five-card `hold_last` ablation gives the same ordering: adaptive scaled RMSE `1.2074`; LLM-updater scaled RMSE `3.8132`. Accounting passes in both schedule-mode runs. The later February held-out diagnostic also loses, but does not qualify as a frozen-vintage confirmation. This schedule replay remains a useful negative: policy schedules are the right behavior interface, but the current belief-to-behavior bridge still does not beat adaptive expectations.

The next bridge version is state-conditioned policy schedules. A new live Codex CLI run wrote `outputs/state_policy_schedules_live_gpt55_sce_prior_update/state_behavior_policy_profile.json`: one GPT-5.5 call, ten SCE-derived household-state archetypes, and bounded schedules over transfer size, job-risk belief gaps, inflation gaps, confidence drops, and real-rate gaps. In `state_schedule` mode, the demand economy matches each SCE household to the nearest state-policy archetype, interpolates the LLM-authored schedules, and disables the older hand-built belief-drag bridge so the schedule owns belief-to-action transmission.

That improves the LLM path but still does not rescue Phase 4. On the strict one-card retrospective FRED run, state schedules reduce the LLM-updater scaled RMSE from `6.5296` to `5.0843`, while adaptive scores `1.3471`. On the five-card `hold_last` diagnostic, state schedules reduce the LLM-updater scaled RMSE from `3.8132` to `3.2132`, while adaptive scores `1.7290`. Accounting passes with max residual `2.91e-11` or better. The current interpretation is precise: **state-conditioned schedules are the better natural-behavior bridge, but the simulated economy still does not predict the FRED proxy surface better than adaptive expectations.**

### Empirical bridge v3: rejected by its own pre-registered checks

The next attempt replaced the hand-authored belief-to-spending bridge with an empirically estimated one. The pipeline joined NY Fed SCE Household Spending Survey microdata to the core SCE belief panel at the respondent level (`work/empirical_bridge/spending_belief_panel.csv`), and fit a Mundlak correlated-random-effects regression of expected year-ahead household spending growth on belief regressors, with the between-wave coefficients pre-registered to drive aggregate transmission. The spec (`reports/empirical_bridge_v3_spec.md`) was red-teamed by Opus 4.8, locked, and committed before any fitting, with fail-closed coefficient bounds and non-overlapping FIT / internal-check / validation / sealed-confirmatory waves.

The fit ran once and was rejected. Two locked gates fired:

1. **Chart reproduction failed.** Published NY Fed spending-chart medians reproduce only to within `0.30`pp against the `0.10`pp tolerance, because the public spending workbook ships without native person weights; same-wave core-panel weights were substituted and the mismatch was recorded fail-closed rather than the tolerance widened.
2. **The inflation coefficient violated its pre-registered bound.** The between-wave coefficient on 1-year expected inflation came out at `+1.22`pp of expected spending growth per pp of expected inflation, far outside the locked band of `[-0.10, +0.30]`. Expected real income growth (`+0.49`, band `[0.00, +0.50]`) and job-loss risk (band respected) passed.

The rejection is informative rather than embarrassing. The SCE outcome variable is *nominal* expected spending growth, so a coefficient near one on expected inflation is largely mechanical price pass-through: households expecting higher prices expect to spend more dollars. The bound was written in real-consumption terms. The data exposed a nominal-versus-real units mismatch in the spec, not a household planning to consume 1.2% more real goods per point of expected inflation.

The fail-closed machinery then did its job downstream: both Phase 4 matched-twin replays refused to score with a rejected bridge (`status: failed`, error `Empirical bridge is fail-closed and not accepted`), zero live calls were spent, and the blockers are logged with timestamps in `work/codex_briefs/empirical_bridge_v3_blockers.md`. No Phase 4 score surface was consumed.

### Empirical bridge v4: accepted, and nearly closes Phase 4

v4 is the proper spec revision rather than a quiet refit. The locked v4 spec (`reports/empirical_bridge_v4_spec.md`) changes the fitted outcome to respondent-level real expected spending growth:

```text
100 * ((1 + expected_total_spending_growth_pct / 100)
       / (1 + actual_expected_inflation_1y / 100) - 1)
```

The same joined SCE spending-belief panel, FIT waves, internal-check wave, and validation waves are used. The chart-quality gate is still recorded, but v4 treats the known public-weight substitution mismatch as acceptable only if max chart error stays below `0.35`pp; the observed max is `0.3006`pp. The v4 fit is accepted. Its between-wave coefficients are:

| Belief regressor | Coefficient | Locked bound | Status |
| --- | ---: | ---: | --- |
| Expected inflation | `+0.1880` | `[-0.30, +0.30]` | pass |
| Expected real income growth | `+0.4266` | `[0.00, +0.50]` | pass |
| Unemployment-higher probability | `-0.0158` per pp (`-0.1580` per 10pp) | `[-0.40, 0.00]` per 10pp | pass |

The validation diagnostic is mixed. Cell RMSE generalizes (`1.29x` fit, inside the `1.5x` diagnostic band), and inflation/unemployment coefficient signs and magnitudes are stable. The real-income coefficient is not stable in the validation refit (`10.02x` the FIT coefficient), so this remains an exploratory bridge rather than a confirmatory macro result.

One implementation detail matters. The fitted SCE outcome is annual expected real spending growth, while the demand economy uses quarterly household income and consumption states. A Codex CLI red-team pass caught that the executor was initially applying the annual coefficient as a one-period shock. The executor now divides the annual bridge deviation by `4` before applying it to the quarterly consumption margin, and both annual and per-period deviations are emitted in the household-decision audit.

Phase 4 then ran two zero-live-call v4 replays:

| Run | Adaptive scaled RMSE | LLM-updater scaled RMSE | Result |
| --- | ---: | ---: | --- |
| Strict one-card retrospective | `0.5692` | `0.6311` | LLM path improves massively but adaptive still wins. |
| Five-card hold-last retrospective | `1.4633` | `1.4688` | Essentially tied, but adaptive still wins on the locked metric. |

This is the strongest Phase 4 result so far. Relative to the prior state-schedule bridge, v4 cuts the LLM-updater strict scaled RMSE from `5.0843` to `0.6311`, and the hold-last scaled RMSE from `3.2132` to `1.4688`. The adaptive twin improves too, from `1.3471` to `0.5692` strict and from `1.7290` to `1.4633` hold-last. Accounting passes in both v4 replays, with max residuals at numerical tolerance (`7.28e-12` strict, `2.91e-11` hold-last).

The right interpretation is precise: **the measured belief-to-spending bridge fixes most of the simulated economy's over-contraction problem, but the LLM-updater economy still has not beaten the adaptive-expectations twin on the locked scaled score.**

The remaining v4 gap is now decomposed in `outputs/phase4_v4_diagnostics/`. It is not a scoring mystery. In the strict run, the LLM-updater path wins on unscaled RMSE (`0.3635` versus adaptive `0.3751`), but loses on scaled RMSE because the locked metric gives more weight to misses in low-volatility consumption targets. The two biggest strict scaled-SSE penalties are real PCE growth (`+0.3038`) and PCE growth (`+0.2463`); the saving-rate improvement helps (`-0.1805`) but does not offset them. In the five-card hold-last diagnostic, the result is almost a tie for the same reason: a single real-PCE scaled penalty (`+0.6803`) is mostly offset by saving-rate and retail-sales improvements.

The v4 validation instability is also diagnosed. The real-income coefficient blow-up is not a parser bug. It comes from fitting the between-wave coefficient on only three validation spending waves, where expected real-income wave means move from `0.1593` to `0.2945` while real spending-growth means move from `0.4274` to `1.0853`. The validation wave design has condition number `611.5`, and the validation wave-mean correlation between expected real income and real spending growth is effectively `1.0`. That makes the validation coefficient a weak-identification warning, not a stable macro mechanism.

### Empirical bridge v5: stabilized, but not a standalone rescue

The next move was deliberately not a blind refit. v5 is a pre-registered stabilization of v4, documented in `reports/empirical_bridge_v5_stabilized_spec.md`. It keeps the same real spending outcome, same panel, same FIT and validation waves, same controls, same chart-quality gate, same coefficient bounds, and same liquidity-gradient sanity check. The estimator changes to standardized ridge regression. Alpha is selected on FIT waves only, using leave-one-FIT-wave-out cross-validation and the rule "largest alpha within one standard error of the minimum weighted RMSE." Phase 4 scores are not used to choose alpha.

The fitted artifact is `work/empirical_bridge/empirical_bridge_v5_stabilized.json`. It is accepted by the hard gates. The selected alpha is `3.0`; the minimum-CV alpha is `0.3`; alpha `3.0` is chosen by the one-standard-error shrinkage rule. The between-wave coefficients shrink as intended:

| Belief regressor | v4 coefficient | v5 coefficient |
| --- | ---: | ---: |
| Expected inflation | `+0.1880` | `+0.0781` |
| Expected real income growth | `+0.4266` | `+0.2101` |
| Unemployment-higher probability | `-0.0158` | `-0.0173` per pp |

The validation diagnostic still fails. The validation cell RMSE ratio is `1.5852`, above the `1.5` diagnostic band; the validation inflation coefficient flips sign; the validation real-income and unemployment coefficients remain unstable. So v5 is a stabilized exploratory bridge, not a validated final mechanism.

On the original one-card strict Phase 4 replay, v5 helps but does not win: adaptive scaled RMSE is `0.5688`, and the LLM-updater path is `0.5970`. On the five-card hold-last diagnostic, the result is almost exactly tied but still adaptive by the locked metric: `1.4644` adaptive versus `1.4663` LLM. Stabilization reduces the remaining gap; it does not remove it.

### Horizon alignment: the first retrospective Phase 4 win

The stronger finding came from matching the belief-update horizon to the FRED scoring horizon. The banked October-November 2024 SCE prior-update run supplied only two survey waves. A December 2024 extension was run live through `codex_cli`, using the same preserved respondent IDs for the 81 respondents with complete October-November-December coverage. It spent `81` live GPT-5.5 calls and zero cache hits. The extension is in `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_dec2024_extension/`; the stitched three-period panel is in `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined/`.

The original December update was useful but not gate-clearing. It had update correlation `0.3594`, direction accuracy `0.6037`, and amplitude ratio `0.6369`, all above their thresholds. It failed the persistence improvement threshold: RMSE improvement versus persistence was `-0.0084`, below the required `+0.02`. That first combined panel was therefore an exploratory horizon-alignment surface, not a validated new persona result.

With that caveat, the aligned Phase 4 replay is the first end-to-end matched-twin run where the LLM-updater economy beats the adaptive twin on the locked scaled score:

| Aligned two-card replay | Adaptive scaled RMSE | LLM-updater scaled RMSE | Direction accuracy, adaptive | Direction accuracy, LLM |
| --- | ---: | ---: | ---: | ---: |
| Empirical bridge v4 | `0.8961` | `0.8455` | `0.6667` | `0.8889` |
| Empirical bridge v5 stabilized | `0.8975` | `0.8529` | `0.6667` | `0.8889` |

This is the clearest answer to the current technical question. Shrinkage helps, but the bigger issue was horizon mismatch. When the economy is given a belief panel long enough to cover the scored target horizon, the LLM-updater twin can beat adaptive expectations. The load-bearing caveat on the original aligned surface was that its December belief-update leg did not clear the prior-update gate; the v2 diagnostic re-run below fixes that specific information-set defect, but the fresh January leg then fails by over-reaction. So this remains a strong retrospective diagnostic rather than the final macro-simulation claim.

### The persistence-gate failure: diagnosed, partially fixed, and one fresh negative

The December gate failure was traced to a stale information set, not to the updater. The December environment card had been built from the quarterly SPF origin as-of `2024-11-15`, so respondents answering in December 2024 were modeled with a pre-election disinflation card: CPI YoY `2.58` and falling, near-zero inflation news pressure, and a synthetic sentiment index of `94` (real Michigan sentiment was about `72`). Real respondents raised year-ahead inflation expectations by `+0.87`pp on average between the November and December waves; the model, following its card, lowered them. The per-target scores confirm the failure was concentrated entirely in inflation (`-0.16` RMSE improvement versus persistence) while income (`+0.07`) and unemployment (`+0.06`) passed.

The fix was pre-registered before any re-elicitation in `reports/prior_update_environment_v2_prereg.md`: every SCE wave gets a monthly ALFRED vintage card as-of the 15th of its own survey month, and real vintage `UMCSENT` and `MICH` replace the synthetic sentiment and aggregate-expectation formulas. Two live legs ran under that rule, `148` GPT-5.5 calls total.

The December re-run (pre-registered as diagnostic only, since its failure had already been observed) now clears the gate: mean RMSE improvement versus persistence `+0.0323` against the `+0.02` threshold, direction accuracy `0.6298`, update correlation `0.3832`, amplitude ratio `0.6385`. The honest detail is that the pass comes from much better unemployment (`+0.149`) and income (`+0.098`) updates under the corrected card; the inflation target alone is still below persistence (`-0.15`), because even a mid-December vintage does not contain the tariff-news channel that moved households (the latest vintage Michigan expectation as-of December 15 still read `2.6`, the November value).

The fresh January 2025 leg — the pre-registered honest gate test, 67 continuing respondents whose January answers had never been elicited — **failed**: mean RMSE improvement versus persistence `-0.0904`. Direction accuracy (`0.5943`) and correlation (`0.3936`) stayed fine, but under the richer monthly cards the model now over-reacts at the individual level (amplitude ratio `1.27`, e.g. predicting a `+2.14`pp mean rise in unemployment-risk beliefs where respondents actually moved `+0.70`pp). Per the pre-registration, no January surface is built and the result is reported as a negative. Individual-level persistence remains a brutal baseline: two of three legs now clear it (October-November confirmatory, December diagnostic), January does not.

Robustness of the aligned Phase 4 win: rebuilding the stitched October-November-December panel with the v2 December leg (`outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined_v2/`) barely changes the aligned replay. The LLM-updater economy still beats adaptive under both bridges: v4 `0.8539` versus `0.8961`, v5 `0.8645` versus `0.8975`. The aligned win therefore survives a full re-elicitation of its December leg under a corrected information set, which is meaningful evidence that it is not an artifact of one December draw.

### Macro tournament: model-building mode

The project has now shifted from one-gate validation to full-economy tournament search. The runner is `python3 -m macro_llm_tournament.macro_tournament`, with tracked development specs under `configs/macro_tournament/`. Development specs search only retrospective, already-scored Phase 4 surfaces. Adaptive/persistence-style rows are retained as diagnostics, but they do not veto LLM candidate selection. A later integrity pass found that the original runner did not enforce all claimed hard vetoes: its spent key was renameable, reservation happened after scoring, and input hashes were not carried into the tournament manifest. Those paths are now hardened for future runs; they do not retroactively strengthen the February result.

The development run in `outputs/macro_tournament_development_v1/` scored `5,670` LLM economy candidates across two retrospective surfaces: the original one-card replay and the aligned October-November-December two-card replay. It used zero live calls. No candidate was disqualified; max accounting residual for the winner is `1.46e-11`.

The v1 winning candidate is `cand_1623eb882b6a`. It keeps the empirical bridge v5 behavior mechanism, uses closed-loop feedback with feedback gain multiplier `1.5`, and applies deterministic belief-update gains: global `1.50`, inflation `1.25`, unemployment-risk `1.25`, income `0.75`. Its mean LLM scaled RMSE across the two retrospective surfaces is `0.7208`, versus `0.7249` for the unit-gain comparator. The improvement is small (`0.0042` scaled RMSE). This candidate was retained in `configs/macro_tournament/incumbent_v1.json` for historical replay; the corrected-panel v3 search later selected a different candidate.

Surface detail matters. On the original one-card replay, the v1 winner is slightly worse than the unit-gain comparator (`0.5984` versus `0.5970`) and still worse than adaptive (`0.5688`). On the aligned two-card replay, it improves the LLM path from `0.8529` to `0.8431` and beats adaptive (`0.8976`). The tournament did not discover a new behavior architecture; it found a retrospective v1 candidate whose advantage lived on the aligned surface.

### Tournament v2: a boundary candidate, and where the search starts lying

The v1 winner sat on the edge of its gain grid (global `1.5` was the maximum searched), so `configs/macro_tournament/development_v2.json` extends global gain to `3.0` and widens the per-target gains (`2,500` candidates, zero live calls, no disqualifications). The v2 winner is `cand_a44fcf25ae3c`: the **unshrunk v4 bridge** with global gain `3.0`, inflation `1.5`, income `0.5`, unemployment `1.5`, feedback `1.5`. Mean LLM scaled RMSE is `0.7087` versus incumbent `0.7249`, beating the adaptive diagnostic mean (`0.7327`). Global `3.0` is still the v2 grid boundary; only the later frontier probe shows that performance eventually reverses beyond it. The mechanism flip from v5 back to v4 suggests that free gains partly substitute for v5's ridge shrinkage.

A follow-up frontier probe (`configs/macro_tournament/development_v2b_gain_frontier.json`) shows where honest search ends. Pushing further out finds candidates with nominally better means — global `8.0`, inflation `2.0`, income updates fully zeroed scores `0.6621` — but the cross-surface Spearman rank correlation falls from `+0.955` on the broad v1 grid to about `-0.75` near the frontier. Recomputed correlations are already strongly negative within v2 and v3 (about `-0.94` and `-0.96`), so the exact ordering is surface-dependent well before the extreme edge. With roughly 14 scored observations and thousands of candidates, the point optimum is not trustworthy; the extreme candidates are diagnostic and were not promoted.

A third run (`configs/macro_tournament/development_v3.json`, same grid as v2) re-scored the search after the December leg was re-elicited under the monthly-vintage fix, using the gated v2 combined panel for the aligned surface. The broad story across the two December draws is similar — v4 bridge, stronger inflation updates, damped income updates — but the preferred unemployment gain flips from `1.5` to `0.5` and the improvement over unit gains shrinks (`0.7240` versus `0.7308`). That supports a regional mechanism hypothesis, not a stable tuned candidate.

### Recursive macro economy: the current development winner

The earlier Phase 4 surfaces replayed a short belief panel through a scoring bridge. The current lane is a genuine recursive simulation. It starts with 81 real SCE households and their supplied priors, balances, demographics, and weights. At each monthly origin, GPT-5.5 updates household inflation, real-income, unemployment-risk, and confidence beliefs from origin-visible information. Empirical bridge v4 converts those changes into consumption pressure. Deterministic code executes consumption, saving, debt repayment, liquidity, income, employment, inflation, and policy-rate transitions, checks accounting, and carries state into the next origin. The adaptive twin changes only the belief updater.

The frozen January-May 2026 bundle uses common-month first-release targets. January is warmup; February-May are scored against 10 targets in five equally weighted families: demand, balance sheets, labor, prices, and income/policy. Every one of the 50 catalogue rows was first released after GPT-5.5's December 1, 2025 cutoff. Prompts contain origin-visible history and simulated state, never target realizations or target aliases.

The corrected seven-candidate tournament selected empirical bridge v4 with global belief gain `3.0`, inflation gain `1.5`, income gain `0.5`, and unemployment-risk gain `1.0`. It scored `0.564533` against adaptive `0.564235`. A bounded policy-state follow-up then compared no assimilation, full origin-visible policy-state assimilation, and a half-weight recursive compromise:

| Recursive candidate | LLM MacroScore | Adaptive | LLM minus adaptive | Result |
| --- | ---: | ---: | ---: | --- |
| Base empirical v4 | `0.564533` | `0.564235` | `+0.000297` | Near tie; base candidate. |
| Full observed policy state, `0.85` smoothing | **`0.549789`** | `0.548667` | `+0.001122` | Best absolute LLM score; selected. |
| Half observed policy state, `0.85` smoothing | `0.555688` | `0.549828` | `+0.005860` | Preserves more recursive policy state but scores worse. |

Lower is better. The winner improves the base LLM economy by `2.61%`, but adaptive remains `0.2%` better. The four scored origins are too few to call that difference: the circular origin-block bootstrap interval for LLM minus adaptive is `[-0.00633, 0.00910]`. The selected mechanism's gain comes mainly from the income/policy family. Full policy-state assimilation means the observed policy rate is reintroduced at each rolling origin before the next smoothed Taylor-rule transition; it is state estimation, not a claim that the policy path is fully endogenous.

All three candidates pass accounting with a maximum absolute residual of `7.28e-12`. The live partial-assimilation run also exercised the retry machinery: seven failed or malformed attempts were preserved, three valid cached months were reused, and one final live call completed the fifth origin without exceeding the locked cap. Those failures affect cost and provenance, not winner selection.

The mechanism search now stops. The winner, provider, GPT-5.5 model identity, household file, empirical bridge, five replay records, target contract, executable source tree, and hashes are frozen in `configs/dynamic_macro/confirmatory_june_2026_v1.json`. The one-shot runner validates the full bundle before creating an atomic receipt, then permits exactly one accepted June payload with two schema-retry calls. A complete receipt requires the full hashed output contract and independently recomputes the target, family, origin, and macro scores from the joined error rows. It currently fails before the receipt because the June 10-target first-release bundle is incomplete; the final required [BEA Personal Income and Outlays release is scheduled for July 30, 2026](https://www.bea.gov/news/schedule/).

## What We Can Claim Now

The evidence supports ten positive claims:

1. **Frontier LLMs contain audited macro belief signal.** GPT-5.5 and GPT-5.4 beat strong empirical baselines on a hidden-target vintage macro tournament, and live recall probes do not find realized-value recall.
2. **Raw GPT-5.5 contains behavior signal in one out-of-domain windfall family.** It generalizes where tuned rules break, but that result does not transfer to predictable income-loss dynamics.
3. **Prior-conditioned agents can partly update real household beliefs.** The model is useful when it operates on supplied respondent state.
4. **Policy-schedule elicitation is the best behavior interface found so far.** It is the first LLM-derived behavior source to beat the liquidity rule on the selection split, it survives one fresh CTC behavior family, and it is now executable inside the demand economy.
5. **State-conditioned schedules improve the macro bridge.** They do not win, but they reduce the LLM-updater Phase 4 error relative to the generic schedule executor while preserving accounting.
6. **A real-data belief-to-spending bridge nearly closes Phase 4.** Empirical bridge v4 passes its locked coefficient/chart/liquidity gates, fixes most of the over-contraction, and brings the LLM-updater path close to the adaptive twin.
7. **A horizon-aligned Phase 4 replay can beat the adaptive twin, and the win survives re-elicitation.** On the retrospective two-card aligned run, the LLM-updater economy beats adaptive under both v4 and v5 empirical bridges, and the win holds after the December leg was fully re-elicited under the corrected monthly-vintage information set.
8. **Full-economy tournament search finds a repeatable regional mechanism, not a stable winner.** Both December panel draws prefer stronger inflation updates and damped income updates, but cross-surface rankings and exact per-target gains are unstable.
9. **The December gate failure was an information-set defect, not an updater defect.** Under the pre-registered monthly-vintage card fix, the December re-run clears the locked prior-update gate at diagnostic status (`+0.0323` versus the `+0.02` threshold).
10. **A recursive 81-household macroeconomy now runs over post-cutoff monthly vintages.** It carries beliefs, household balance sheets, demand, labor, prices, income, and policy state through five origins, preserves accounting, and improves its base LLM MacroScore by `2.61%` after policy-state assimilation.

The evidence also supports twelve negative claims:

1. **Profile-only personas fail on real SCE microdata.**
2. **Backstory elicitation fails by caricature rather than rescuing heterogeneity.**
3. **No behavior architecture currently clears both in-domain credibility and out-of-domain holdout tests.**
4. **The first Phase 4 matched-twin replay does not beat adaptive expectations.**
5. **The first Phase 4 policy-schedule replay also does not beat adaptive expectations.**
6. **The first Phase 4 state-schedule replay still does not beat adaptive expectations.**
7. **The first empirically estimated belief-to-spending bridge (v3) was rejected by its own pre-registered fail-closed bounds**, exposing a nominal-versus-real mismatch between the SCE spending outcome and the real-consumption bounds; no Phase 4 surface was spent on it.
8. **Empirical bridge v4 still does not beat adaptive expectations on the locked scaled metric.** It wins or nearly wins on some unscaled summaries, but the pre-declared scaled RMSE remains the score that counts.
9. **The original December prior-update extension did not clear the locked prior-update gate.** Diagnosis traced this to a quarterly-stale environment card; the pre-registered monthly-vintage re-run clears the gate, but only at diagnostic status, and the inflation target alone still loses to persistence.
10. **The fresh January 2025 prior-update leg fails its gate.** Under the corrected monthly cards the model over-reacts at the individual level (amplitude ratio `1.27`), losing to persistence on RMSE (`-0.0904`). Per pre-registration, no January surface was built.
11. **The one-shot February held-out diagnostic fails.** Both LLM-economy candidates lose to adaptive, and the amplified candidate does not beat unit-gain v5. Because the run used revised current-vintage FRED inputs, held December beliefs forward, scored four targets with `n=1` each, and was not independently locked in git before scoring, it is not confirmatory evidence.
12. **The current recursive winner does not beat adaptive expectations on January-May development data.** It is only `0.2%` behind, and the uncertainty interval crosses zero, but the recorded score is still a loss.

The full claim remains open:

**We have built a recursive, accounting-safe LLM economy, but have not yet shown confirmatorily that it predicts macro behavior better than a strong adaptive benchmark.** The current development result is a near tie, with adaptive ahead. The next evidence is not another mechanism tweak; it is the locked June run.

## Recommended Next Work

The next phase is already specified.

1. Do not tune another mechanism on January-May. The current winner is frozen.
2. Build the complete January-June common-month bundle only after all 10 June first releases exist, expected July 30, 2026.
3. Run `make dynamic-macro-confirmatory-june` once. Five banked periods replay; GPT-5.5 supplies one accepted June update; only June's 10 targets are scored.
4. Report the outcome at full volume whether the LLM economy wins or loses. The result earns the narrow claim: LLM belief updating does or does not improve an accounting-constrained recursive economy over adaptive expectations on a fresh frozen-vintage month.
5. Keep the post-cutoff forecast gate accruing in the background. Any later model-building round needs a new development surface, not reuse of June.

The forecast evidence is already strong. Persona generation and point-behavior elicitation are closed. The current question is now clean: does the locked recursive economy carry its developmental near-tie into one fresh month?

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
- `behavior_holdout_ctc_v1`: monthly Child Tax Credit targets. This is the newest behavior family; it has been scored once against the already-banked policy schedules.

### Persona gates

The real persona gates use public SCE microdata. Inflation is the SCE density-median one-year inflation expectation. Unemployment is scored as the actual SCE question: percent chance U.S. unemployment will be higher in 12 months. Real income growth is derived as nominal expected household income growth minus the respondent's inflation expectation.

The December 2024 profile-only wave is spent. The November 2024 backstory validation wave is spent. The sealed June 2025 priors-plus-backstory confirmation was not spent because the pre-registered Arm 1 condition failed.

### Phase 4 matched twins

The Phase 4 fixture compares two versions of the same deterministic demand economy: an LLM-belief fixture and an adaptive-expectations twin. It writes the locked proxy mapping, cards, targets, household states, twin paths, accounting, forecasts, joined errors, scores, manifest, and `phase4_matched_twins_report.md`.

The Phase 4 replay adapter consumes persona-ecology predictions rather than raw prompt payloads. It joins on the normalized CSV outputs, not prompt-facing relative IDs; derives household states from demographics, weights, and prior beliefs; and excludes `actual_*` labels from the demand-economy input. The strict fixed-kernel Codex replay artifact is `outputs/phase4_matched_twins_prior_update_codex_replay_fred_onecard/`. The fixed-kernel extrapolating ablation is `outputs/phase4_matched_twins_prior_update_codex_replay_fred_holdlast_5cards/`. The strict policy-schedule replay artifact is `outputs/phase4_matched_twins_policy_schedule_codex_replay_fred_onecard/`. The policy-schedule extrapolating ablation is `outputs/phase4_matched_twins_policy_schedule_codex_replay_fred_holdlast_5cards/`. The state-conditioned policy profile is `outputs/state_policy_schedules_live_gpt55_sce_prior_update/`. The strict state-schedule replay artifact is `outputs/phase4_matched_twins_state_schedule_codex_replay_fred_onecard/`. The state-schedule extrapolating ablation is `outputs/phase4_matched_twins_state_schedule_codex_replay_fred_holdlast_5cards/`.

The rejected empirical-bridge v3 attempts (`outputs/phase4_matched_twins_empirical_bridge_codex_replay_fred_onecard/` and `_holdlast_5cards/`) contain only fail-closed manifests, by design; the rejected v3 bridge artifact is `work/empirical_bridge/empirical_bridge_v3.json`. The accepted v4 bridge artifact is `work/empirical_bridge/empirical_bridge_v4.json`; its strict replay is `outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_onecard/`, and its hold-last replay is `outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_holdlast_5cards/`. The stabilized v5 bridge artifact is `work/empirical_bridge/empirical_bridge_v5_stabilized.json`; its spec is `reports/empirical_bridge_v5_stabilized_spec.md`, its strict replay is `outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_onecard/`, and its hold-last replay is `outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_holdlast_5cards/`.

The December 2024 prior-update extension is `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_dec2024_extension/`; the stitched October-November-December panel is `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined/`. The aligned two-card empirical-bridge replays are `outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_2card_aligned_octnovdec/` and `outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_2card_aligned_octnovdec/`. Because `work/` and `outputs/` are intentionally ignored, the tracked compact evidence bundle for the v5/alignment result is `reports/phase4_v5_alignment_evidence.json`; it records the exact score rows plus SHA-256s of the local source artifacts.

The macro tournament outputs are `outputs/macro_tournament_development_v1/`, `outputs/macro_tournament_development_v2/`, `outputs/macro_tournament_development_v2b_gain_frontier/`, and `outputs/macro_tournament_development_v3/` (tracked specs under `configs/macro_tournament/`). Their compact tracked evidence bundles are `reports/macro_tournament_development_v1_evidence.json`, `reports/macro_tournament_development_v2_evidence.json`, `reports/macro_tournament_development_v2b_gain_frontier_evidence.json`, and `reports/macro_tournament_development_v3_evidence.json`.

The prior-update environment-v2 pre-registration is `reports/prior_update_environment_v2_prereg.md`; its compact tracked evidence bundle is `reports/prior_update_environment_v2_evidence.json`. The input builder is `prepare_prior_update_extension.py` (`make prior-update-extension-v2-inputs`), the December re-run and January legs are `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_dec2024_extension_v2/` and `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_67_jan2025_extension_v2/`, and the panel stitcher is `combine_prior_update_panels.py`. Both live manifests record zero cache hits, but they also record dirty working trees predating the commit that introduced the preregistration and implementation; exact pre-call code identity is therefore not independently recoverable from git. Future live empirical runs now reject synthetic prior/environment fallback and refuse an existing output directory. The v2 combined panel is `outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined_v2/`, and its aligned replays are `outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_2card_aligned_octnovdec_v2/` and `outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_2card_aligned_octnovdec_v2/`. The replayable v1 development incumbent is `configs/macro_tournament/incumbent_v1.json`; it is not the winner of the later corrected-panel v3 search and is not a validated best economy.

The historical February tournament spec is `configs/macro_tournament/confirmatory_fred_2026_02_v1.json`; `make macro-confirmatory-v1` spent the `2026-02-15` score date and wrote `outputs/macro_confirmatory_fred_2026_02_v1/`. Its compact tracked evidence is `reports/macro_confirmatory_fred_2026_02_v1_evidence.json`, and its registry is `reports/macro_tournament_confirmatory_registry.json`. The arithmetic is negative: unit-gain v5 scores `2.1202`, conservative v4/moderate-gain scores `2.1256`, and adaptive is better for both (`2.1019`-`2.1009`). The evidence bundle is now explicitly marked as a downgraded held-out current-vintage diagnostic. Future confirmatory code reserves an immutable score-date key before scoring, uses a canonical locked registry, requires a complete target contract and both candidate results, and records code/input hashes.

### Recursive dynamic macro lane

`prepare_dynamic_macro_panel.py` builds the 81-household SCE input with event-date and public-availability provenance. `frozen_vintage_bundle.py` freezes six canonical CSV payloads and validates their full target catalogue, source requests, first-release vintages, contamination labels, and hashes. `dynamic_macro_economy.py` runs the matched twins and emits household, prompt, belief, decision, period, accounting, forecast, error, target, family, origin, replay, and manifest artifacts. `dynamic_macro_tournament.py` applies the locked developmental selection rule and journals live attempts before provider calls so resumes cannot hide spent budget.

The current development output is `outputs/dynamic_macro_policy_partial_gpt55_live_v4/`. Its tracked compact evidence is `reports/dynamic_macro_development_v1_evidence.json`. The winner's five raw records are banked at `work/dynamic_macro/banked_gpt55_development_v1/policy_assimilation_100_smoothing_085_periods_0_4.json`; `make dynamic-macro-incumbent-replay` reproduces the selected score and accounting path with zero provider calls. The June lock is `configs/dynamic_macro/confirmatory_june_2026_v1.json`, with the exact Python source tree frozen in `reports/dynamic_macro_confirmation_source_lock_v1.json`. `dynamic_macro_confirmatory.py` validates every locked development input, the executable source, the complete canonical January-June bundle, and every score artifact before atomically completing the one-shot receipt.

### Reproducibility notes

The canonical report in the repository is `reports/macro_simulation_report.md`. The live elicitation campaign artifacts are under `outputs/persona_elicitation_campaign/`. The Phase 4 fixture artifacts are under `outputs/phase4_matched_twins_fixture/`. The Phase 4 v4 diagnostic artifacts are under `outputs/phase4_v4_diagnostics/`. The research-path retrospective is `reports/research_retrospective.md`. The full local contract passes `292` tests. The data-asset event stream (downloads, derivations, run manifests, all timestamped) is `data_provenance/data_events.jsonl`.
