# Research History

This project began as an aggregate forecast tournament and progressed through
typed agents, persona panels, behavior gates, accounting-constrained demand models,
matched twins, and mechanism tournaments.

The durable findings are:

- aggregate LLM forecasts contained audited macro belief signal;
- demographic or backstory personas flattened real household heterogeneity;
- prior-conditioned belief updating worked better than generating beliefs from a
  profile alone;
- raw allocation answers were unstable, while code-enforced feasibility and
  LLM-authored conditional policies were more useful;
- the earlier recursive 200-household economy ran and balanced, but its output was
  still effectively demand-defined, labor and credit barely moved, and large belief
  gains plus clipping obscured interpretation;
- comparison scores were useful diagnostics but became a distraction from building
  the economy itself.

The active architecture therefore keeps the parts that survived: survey-seeded
household state supplies heterogeneity, the LLM updates beliefs and states intentions,
and code-enforced institutions execute and settle actions. Historical implementation
and evidence are preserved at Git tag `macro-v1-weighted-demand`; they are not part
of the active code surface.

## July 2026 Ecology Audit

The first period-by-period chart exposed both implementation errors and a substantive
model failure. The implementation audit fixed stale recursive prompt state, mismatched
wage units, unweighted institutional clearing, thresholded job loss, annual-risk versus
monthly-hazard confusion, stale new-hire wages, target-file access, and circular replay
verification. It also excluded the survey-seeded first transition from scoring and
made weak proxy mappings direction-only.

After those corrections, the retired diagnostic recorded consumption RMSE of 5.48
percentage points and three wrong signs across the valid recursive transitions; its
simulated consumption index fell to 86.40 while first-release PCE rose to 102.14.
That figure is retained as contemporaneous research history. The old raw run bundle is
not part of the current hash-bound evidence surface and should not be used as a formal
cross-version benchmark.

## July 2026 Natural-Household Redesign

The next iteration changed the behavioral object instead of adding institutions.
Continuous, correlated SCF financial states replaced three coarse templates. Household
cards gained dated one-, three-, and twelve-month public changes. GPT households stopped
returning abstract percentage cuts and instead wrote conditional dollar policies for
committed spending, discretionary spending, deposits, debt, and borrowing. The active
diagnostic froze wages and respondent employment so household demand could be tested
without a speculative labor model.

Successive full runs exposed two further structural errors. Respondent unemployment had
been treated as unemployment of the entire household, and independently clipped SCF
income components could exceed total household income. The final financial-state schema
holds SCF family earnings fixed at the household boundary, invents no respondent wage
share, and reconciles every component to annual income.
Rolling forecasts also stopped carrying simulated deposits and debt; each origin now
restarts from the same SCE-SCF anchor with newly visible public information.

The first four-origin natural-household diagnostic predicted positive consumption
growth in all four months and recorded RMSE of 0.47 percentage points. A later integrity
pass found that aggregate unemployment expectations had been divided by four and
mislabeled as personal job-loss risk, that a gross-income budget residual had been
reported as personal saving, and that a point policy had been written out as three
identical scenarios.

The next v19 run separated SCE `Q13new` personal job-loss priors from the aggregate
`Q4new` unemployment outlook, removed the national saving-rate comparison, and emitted
one point path. A final contrary review then found that `Q13new` had accidentally been
included in demographic backfilling: missing March-April answers could be filled from
later 2025 waves. It also found that the prompt described job-loss execution while the
rolling engine held employment fixed. The v19 arithmetic was reproducible, but the run
is invalid as evidence and is preserved only in the local archive.

The v20 campaign preserved wave-specific missingness, described the fixed-labor
executor truthfully, and reran all 1,000 household calls from fresh cache identities.
It retained 4/4 consumption signs with RMSE 0.61 percentage points, versus 0.24 for the
origin-visible routine-drift anchor. Correlation is 0.10 and credit direction is 1/4.
The prospective July-to-August forecast was frozen at +0.08%. It was superseded by
the v21 prompt and budget contract described below.

## Full Observability And Firm Shadow

The v20 response bank was next expanded into a six-layer diagnostic surface rather
than another forecasting variant. It records observed proxy comparisons, weighted
beliefs, household dollar policies, feasible household outcomes, firm and credit
accounts, and a one-step firm-response shadow in separate source classes. The panel
shows that the weak demand amplitude is already present in household intentions:
deposit additions equal 13% to 19% of baseline monthly consumption, while intended
consumption growth remains below 0.2%. Feasibility changes little.

An audit rejected switching the real household states into the engine's old dynamic
labor mode. The matched SCF data identifies family earnings, not the respondent's
wage share; assigning all family earnings to the respondent or hiring at the current
zero wage would be fabricated structure. The retained firm shadow therefore stops at
sales, inventory-adjusted output, labor requirement, partial employment adjustment,
and price pressure. It is a diagnostic projection, not a scored forecast or a closed
feedback loop.

## Deposit Residual And Two-Period Producer Loop

The observability pass also overturned its own deposit diagnosis. Counterfactual
execution showed that the LLM's explicit deposit contribution did not constrain
consumption for 198 of 200 households; it mainly changed extra debt repayment. Asking
for consumption, debt, borrowing, and deposits separately had overdetermined the
budget. V21 removed the deposit field and made deposits the cash residual.

The large remaining residual came from a different mismatch: SCF gross family income
was being compared with a spending proxy that excludes taxes and some recurring
obligations. V21 records those omitted outflows explicitly, using a declared 10% of
gross-income floor and the existing household saving-rate field when it implies more.
The weighted gross-income residual fell from roughly 17% to about 7%.

The prompt also gained an origin-visible next-month spending anchor and clearer
instructions to preserve ordinary nominal inertia. Fresh calls did not improve the
macro score. The four forecasts became -0.24%, +0.11%, -0.02%, and -0.02% against
positive first-release PCE growth in every month. RMSE rose to 0.70 points and sign
accuracy fell to 1/4, although correlation improved to 0.81. The model ranks the
months better while remaining too compressed around zero.

Finally, the producer shadow was wired into a two-period runner without adding an
LLM firm. Prior demand changed planned output, inventory, an aggregate employment
index, wages, and family wage income before all 200 households made fresh choices.
The first unscored trace reported employment up 0.015%, wages up 0.0015%, and
period-two consumption down 0.06%.

A major-change integrity review then found that v21's aggregate headcount remained a
planning target while settlement still reported unchanged realized employment. It
also found that the 10% fixed-outflow floor overrode most household saving-rate
contracts, that feedback parent artifacts were not all hash-checked, and that the
observability report still described the older non-recursive shadow. Those findings
make v21 a useful development record, not the current result.

## V22 Natural Budgets And Realized Producer Feedback

V22 replaced the outflow floor with an explicit saving allocation. The household's
existing saving rate supplies a total-saving target, capped by cash capacity. Liquid
deposits close any buffer gap over twelve months; taxes, recurring obligations, and
remaining saving sit outside deposits. Matched households whose expenditure exceeds
income retain an explicit deficit. The 200-household baseline identity reconciles to
floating-point tolerance. Half the weighted cohort has no liquid-saving target.

The household prompt was rerun through Codex CLI for 200 prospective and 800
historical household-origin pairs. The historical consumption forecasts became
+0.15%, +0.26%, +0.11%, and +0.12%. Direction improved from 1/4 to 4/4 and RMSE
fell from 0.701 to 0.508 percentage points. The simple origin-visible drift still
wins at 0.243 points, and revolving-credit direction remains 1/4. The architecture
now gets the sign right but still compresses the amplitude.

The producer runner now realizes aggregate employment in settlement and derives its
average wage from the family wage bill per producer worker. The period-two parent
binding covers replay equivalence, every consumed parent artifact, and both source
input files. An explicit feedback input fails closed when its schema, accounting,
replay, household count, parent binding, or dynamic artifact is wrong. The historical
firm shadow and recursive mechanism use distinct metrics.

In the corrected unscored trace, settled employment rises 0.0013%, the average wage
rises 0.00013%, and fresh period-two consumption rises 0.23%. That establishes a
working two-period state transition. It is not a causal feedback estimate because no
matched no-feedback household-call arm was run.
