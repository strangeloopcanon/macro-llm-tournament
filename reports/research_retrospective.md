# Macro LLM Simulation: Research Retrospective

**Status:** Historical experiment log through the first aligned Phase 4 runs.
For the current evidence verdict, use
[`macro_simulation_report.md`](macro_simulation_report.md). A July 2026
integrity review downgraded the February score from confirmatory to a one-shot
held-out current-vintage diagnostic.

This is the project log behind the current report. It tracks what we tested, what broke, what survived, and why the architecture moved.

## The Current Shape

The surviving architecture is not "LLM households role-play the economy."

It is:

1. Real data supplies household heterogeneity.
2. LLMs update beliefs from prior state and new information.
3. LLMs can author conditional behavior schedules.
4. Deterministic code executes those schedules, enforces budgets, clears accounts, and aggregates.
5. The final comparator is an adaptive-expectations twin inside the same economy.

That arrangement is narrower than the original ambition, but every link in it comes from a test rather than a preference.

## What We Tried

### 1. Aggregate Macro Forecasting

Question: can frontier models forecast hidden macro outcomes from vintage, date-free cards?

Result: yes. GPT-5.5 and GPT-5.4 beat no-change, rolling mean, rolling trend, AR(2), and recursive least-squares baselines on the 147-card vintage split. Recall probes found zero exact realized-value recall.

Status: keep. This is the strongest positive result.

### 2. Belief-Structure Audits

Question: is the forecast win just a score, or does the belief layer have useful dynamics?

Result: the audit added underreaction, extrapolation, dispersion, calibration, and surprise-response diagnostics. The model is less trend-mechanical than the deterministic baselines, and confidence carries error information.

Status: keep. This turns "it forecasts well" into "its belief dynamics are measurable."

### 3. Typed Agent Economy

Question: can we build an accounting-safe toy economy around household, firm, bank, and policy actors?

Result: yes as a lab. The economy clears accounting, produces impulse responses, and supports transfer, rate, job-risk, and feedback shocks.

Status: keep as infrastructure, not proof of macro validity.

### 4. Direct Household Behavior Elicitation

Question: can raw GPT-5.5 actions match published household-response moments?

Result: mixed. Raw GPT-5.5 loses to the liquidity rule on stimulus targets, wins on lottery windfalls, then loses on UI-exhaustion income-loss targets.

Status: partial. The model has behavioral priors, but direct point elicitation compresses the responses that matter.

### 5. Blends and Primitive Kernels

Question: can we recover the behavior signal by combining the LLM with simple rules or interpretable primitives?

Result: no. The 50 percent liquidity blend looked promising on selected targets and failed the lottery holdout. Primitive kernels passed sign audits and then overfit or collapsed out of domain.

Status: closed as mainline. Useful negative: sign-correct mechanisms are not enough.

### 6. Real SCE Persona Beliefs

Question: can profile-only personas reproduce real household belief heterogeneity?

Result: no. The December 2024 SCE profile-only run failed by near-total flattening. The model returned too little within-group variance and partly wrong gradients.

Status: closed. Demographic sketches do not contain enough information to place people inside belief distributions.

### 7. Backstory and Distributional Elicitation

Question: can richer backstories or distributional answers rescue persona heterogeneity?

Result: no. Backstory prompts failed by caricature. Stated p10-p90 intervals widened the answer but still under-covered real respondent dispersion.

Status: closed. Narrative detail invents texture but not the missing idiosyncratic state.

### 8. Prior-Conditioned Belief Updating

Question: can an LLM update a real respondent's prior belief when the prior is supplied?

Result: yes, modestly. GPT-5.5 clears the prior-update gate on repeated SCE respondents: the update is directionally useful but underreactive.

Status: keep. This is the persona result that survives.

### 9. Behavior-Ecology Experiment

Question: is behavior compression a capability wall or an elicitation-interface artifact?

Result: it is an interface artifact, but the failure moves. First-person household ecology restores scenario dynamics but collapses cross-household structure. Policy schedules preserve more structure: the GPT-5.5 policy arm is the first LLM-derived source to beat the liquidity rule on the selection split, and Opus 4.8 replicates the schedule ordering on lottery gradients.

Status: keep as the behavior interface. Not confirmatory because lottery and UI families are spent.

### 10. Phase 4 Fixed-Kernel Matched Twins

Question: do prior-conditioned LLM beliefs improve an accounting-safe economy over an adaptive-expectations twin?

Result: no on the first retrospective FRED proxy run. Adaptive wins under the fixed demand kernel.

Status: keep as a negative. The belief updater alone is not enough.

### 11. Phase 4 Policy-Schedule Matched Twins

Question: does the new policy-schedule behavior executor rescue Phase 4?

Result: no on the first retrospective FRED proxy run. The schedule executor is now wired, uses the same GPT-5.5-authored behavior schedules for both twins, and preserves accounting. But adaptive still wins. The LLM-updater schedule economy over-contracts consumption growth and overshoots the saving-rate change.

Status: keep as the new mainline architecture and a negative first macro replay.

### 12. Fresh CTC Behavior Holdout

Question: do the already-banked GPT-5.5 policy schedules generalize to one fresh behavior family?

Result: yes, directionally. The new `behavior_holdout_ctc_v1` split scores a monthly Child Tax Credit style payment. It uses a BLS Consumer Expenditure spending target and Brookings/Social Policy Institute saving/debt/income-gradient usage moments. The run replays the existing GPT-5.5 policy schedules from `outputs/behavior_ecology_gpt55_xhigh/ecology_raw_records.json`, so it spends zero new policy calls. Policy schedules beat the liquidity rule overall on CTC range RMSE (`0.0097` versus `0.0207`) and predict spending near the BLS target (`0.4569` versus target `0.44`). The miss is income-gradient debt repayment: the policy schedule under-predicts how much more low-income households use CTC money for debt repayment.

Status: keep as a positive behavior-interface result with caveat. It promotes schedules beyond spent lottery/UI families, but it does not validate the macro economy.

### 13. State-Conditioned Policy Schedules

Question: can a more natural household bridge improve Phase 4 by making behavior policy depend on actual SCE household state and belief gaps?

Result: yes as an improvement, no as a win. A one-call live Codex GPT-5.5 run created ten SCE-derived household-state archetype schedules in `outputs/state_policy_schedules_live_gpt55_sce_prior_update/`. The demand economy now has `state_schedule` mode: it matches each household to a state-policy archetype, interpolates transfer and belief-stress schedules, disables the older hand-built belief-drag bridge, and keeps deterministic budget/accounting execution.

The strict one-card Phase 4 replay improves the LLM-updater scaled RMSE from `6.5296` under generic schedules to `5.0843` under state schedules, but adaptive still scores `1.3471`. The five-card hold-last diagnostic improves the LLM-updater from `3.8132` to `3.2132`, but adaptive still scores `1.7290`.

Status: keep as the current best bridge and a negative macro result. The architecture is closer to the north star, but it still does not beat the adaptive twin.

### 14. Real-Data Empirical Bridge

Question: can the belief-to-spending link be measured from SCE spending and belief microdata instead of hand-written?

Result: yes, with an important revision. v3 was rejected by its own pre-registered checks because the SCE spending outcome was nominal while the coefficient bounds were written in real-consumption terms. v4 fixed the units by deflating expected spending growth with each respondent's inflation expectation. v4 passed the hard gates and nearly closed Phase 4: strict scaled RMSE fell to `0.6311` for the LLM updater versus `0.5692` for adaptive, and hold-last fell to `1.4688` versus `1.4633` for adaptive. It still did not beat the adaptive twin.

Status: keep as the main empirical bridge, with caveat. The v4 validation diagnostic remains weakly identified, especially on real-income variation.

### 15. Stabilized Bridge and Horizon Alignment

Question: was the remaining Phase 4 gap caused by an unstable bridge estimator, or by a mismatch between the belief panel horizon and the FRED scoring horizon?

Result: both mattered, but horizon alignment mattered more. v5 locked a ridge estimator before Phase 4 scoring: FIT-wave leave-one-wave-out CV, alpha grid `0.1` through `300`, and the largest alpha within one standard error of the minimum RMSE. It selected alpha `3.0`, shrank coefficients, and improved the original strict LLM replay from `0.6311` to `0.5970`, but adaptive still won at `0.5688`. The v5 validation diagnostic still failed.

Then the SCE prior-update panel was extended to December 2024 for the 81 respondents with complete October-November-December coverage. That extension spent `81` live GPT-5.5 calls through `codex_cli`. It had useful direction, correlation, and amplitude, but failed the persistence RMSE gate. On the stitched three-period panel, the aligned two-card Phase 4 replay produced the first retrospective end-to-end win: v4 LLM `0.8455` versus adaptive `0.8961`; v5 LLM `0.8529` versus adaptive `0.8975`.

Status: keep as a promising diagnostic, not final validation. The aligned economy can beat adaptive, but the added December belief-update leg is not gate-clearing, so the next real confirmation needs a longer aligned panel that passes its own prior-update gate before scoring.

## The Lessons

1. LLMs contain audited aggregate macro belief signal.
2. LLMs do not generate individual household heterogeneity from profiles.
3. LLMs can update supplied household belief state.
4. Direct action elicitation compresses behavior.
5. First-person role-play restores dynamics but collapses cross-sectional heterogeneity.
6. Policy schedules are the best behavior interface so far, and the fresh CTC holdout gives that statement one new-family check.
7. State-conditioned schedules improve the macro bridge, but the measured spending bridge improves it more.
8. Deterministic execution remains non-negotiable: budgets, feasibility, accounting, and aggregation belong in code.
9. Bridge stabilization helps but does not solve the problem alone.
10. Horizon alignment is load-bearing. The first aligned retrospective Phase 4 replay beats the adaptive twin, but its added belief-update leg does not clear persistence.
11. The macro-simulation claim is unvalidated. We have a retrospective end-to-end positive and a negative held-out diagnostic, not a valid real-time-vintage confirmation.

## What Should Happen Next

The next real test needs new evidence, not more reuse of spent holdouts.

1. Treat v5 as the current stabilized bridge and stop tuning it against Phase 4.
2. Build a longer same-horizon SCE prior-update panel that clears the prior-update gate.
3. Run the next Phase 4 comparison as confirmatory only after the panel horizon, belief replay horizon, behavior executor, bridge, target mapping, complete target set, and frozen input hashes are committed before scoring.
4. Freeze CTC, lottery, and UI for mechanism evaluation; any new behavior promotion needs a new family.
5. Keep the Phase 4 v2 mapping locked and use frozen ALFRED/release-aware inputs for the next newly scoreable month.
