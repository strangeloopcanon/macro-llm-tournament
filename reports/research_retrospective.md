# Macro LLM Simulation: Research Retrospective

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

## The Lessons

1. LLMs contain audited aggregate macro belief signal.
2. LLMs do not generate individual household heterogeneity from profiles.
3. LLMs can update supplied household belief state.
4. Direct action elicitation compresses behavior.
5. First-person role-play restores dynamics but collapses cross-sectional heterogeneity.
6. Policy schedules are the best behavior interface so far.
7. Deterministic execution remains non-negotiable: budgets, feasibility, accounting, and aggregation belong in code.
8. The macro-simulation claim is still open. The current LLM-updater economy does not beat the adaptive twin.

## What Should Happen Next

The next real test needs new evidence, not more reuse of spent holdouts.

1. Build a fresh behavior holdout family and score the policy-schedule arm once.
2. Build a longer same-horizon SCE prior-update panel.
3. Improve the bridge from updated beliefs to schedule execution on development dynamics only.
4. Keep the Phase 4 v2 FRED mapping locked until the next newly scoreable month.
5. Run the next Phase 4 comparison as confirmatory only after the panel horizon, belief replay horizon, behavior executor, and target mapping are all fixed in the manifest.

