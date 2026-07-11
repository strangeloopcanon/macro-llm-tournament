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
