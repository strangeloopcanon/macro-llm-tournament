# Current Project Surface

The active system is the household-first rolling microeconomy in
`macro_llm_tournament.ecology`.

It starts from the fixed 200-household SCE cohort, admits only household records
available by the forecast origin, and gives each household a separate prompt. A
response contains uncertainty bands for inflation, income, job loss, consumption,
hours, and job search, plus buffer, debt-payment, and borrowing intentions.
Deterministic code converts those intentions into feasible actions and clears them
through production, employment, inventories, and credit.

The first full July 2026 origin run is complete for an August 2026 target. It contains 200 accepted GPT-5.5
household responses elicited through Codex CLI from an origin-only ALFRED snapshot.
The final executable replays those responses with zero calls and produces distinct
downside, median, and upside paths. The median path has intended consumption growth
of -2.95% and feasible consumption growth of -4.08%; accounting passes with a
maximum numerical residual below 1e-6.

The public information card is current to July 10, 2026. The persistent household
cohort is initialized from March-April 2025 SCE observations, so the run should be
read as a current environment acting on a survey-seeded population, not as a
contemporaneous July household survey.

This establishes a runnable, recursive economy and a genuinely frozen forecast.
It does not establish predictive validity. The next gates are:

1. append the August native outcomes once released, without rewriting the forecast;
2. run August and subsequent origins under the same frozen procedure;
3. compare path errors and household calibration over several untouched months;
4. add sectoral employers or richer institutions only if repeated errors identify
   a mechanism the current aggregate employer or credit intermediary cannot express.

The old adaptive-twin tournament is retired from the active path. Its code and
results remain recoverable at Git tag `macro-v1-weighted-demand` and in the local
hashed archive under `~/Downloads/llm-hank-docs/archive/v1/`.
