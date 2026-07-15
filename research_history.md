# Research History

This project began as an aggregate forecast tournament and progressed through
typed agents, persona panels, behavior gates, accounting-constrained demand models,
matched twins, and mechanism tournaments.

The durable findings are:

- aggregate LLM forecasts contained audited macro belief signal;
- demographic or backstory personas flattened real household heterogeneity;
- prior-conditioned belief updating worked better than generating beliefs from a
  profile alone;
- raw allocation answers were unstable, while deterministic feasibility and
  LLM-authored conditional policies were more useful;
- the earlier recursive 200-household economy ran and balanced, but its output was
  still effectively demand-defined, labor and credit barely moved, and large belief
  gains plus clipping obscured interpretation;
- comparison scores were useful diagnostics but became a distraction from building
  the economy itself.

The active architecture therefore keeps the parts that survived: survey-seeded
household state supplies heterogeneity, the LLM updates beliefs and states intentions,
and deterministic institutions execute and settle actions. Historical implementation
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

The active v20 campaign preserves wave-specific missingness, describes the fixed-labor
executor truthfully, and reruns all 1,000 household calls from fresh cache identities.
It retains 4/4 consumption signs with RMSE 0.61 percentage points, versus 0.24 for the
origin-visible routine-drift anchor. Correlation is 0.10 and credit direction is 1/4.
The prospective July-to-August forecast is frozen at +0.08%. This v20 result is the
active evidence surface.
