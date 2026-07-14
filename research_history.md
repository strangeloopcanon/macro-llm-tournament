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
