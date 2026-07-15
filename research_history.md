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

After those corrections, the negative survived. On the three valid March-May 2026
recursive transitions, consumption RMSE was 5.48 percentage points and all three signs
were wrong; the simulated consumption index fell to 86.40 while first-release PCE rose
to 102.14. This localizes the remaining problem to contractionary household behavior
and feedback rather than plotting, aggregation, accounting, or replay mechanics.

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

The final four-origin diagnostic predicts positive consumption growth in all four months
and lowers RMSE to 0.47 percentage points. It still loses to an origin-visible nominal
spending drift at 0.24 points. The redesign therefore removed the artificial collapse
and recovered direction, but not enough ordinary nominal-growth amplitude. Credit
paydown remains too strong. The prospective July-to-August forecast is frozen at +0.28%.
