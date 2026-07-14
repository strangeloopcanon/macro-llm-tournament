# Current Project Surface

The active system is the household-first rolling microeconomy in
`macro_llm_tournament.ecology`.

It starts from the fixed 200-household SCE cohort, admits only household records
available by the forecast origin, and gives each household a separate prompt. A
response contains uncertainty bands for inflation, income, job loss, consumption,
hours, and job search, plus buffer, debt-payment, and borrowing intentions.
Deterministic code converts those intentions into feasible actions and clears them
through production, employment, inventories, and credit. Survey weights enter every
household-to-institution aggregate. Annual job-loss probabilities are converted to
constant monthly hazards before the labor transition.

The corrected July 2026 origin run is complete for an August 2026 target. It contains 200 accepted GPT-5.5
household responses elicited through Codex CLI from an origin-only ALFRED snapshot.
Each call ran without shell, local text-file, web, browser, app, memory, or multi-agent
tools in a fresh empty directory.
The final executable replays those responses with zero calls and produces distinct
downside, median, and upside paths. The median path has intended consumption growth
of -3.64% and feasible consumption growth of -3.77%; accounting passes with a
maximum numerical residual below 1e-6.

The public information card is current to July 10, 2026. The persistent household
cohort is initialized from March-April 2025 SCE observations, so the run should be
read as a current environment acting on a survey-seeded population, not as a
contemporaneous July household survey.

This establishes a runnable, recursive economy and a genuinely frozen forecast.
A separate retrospective diagnostic runs the exact mechanism over four post-cutoff
2026 origins, with 800 accepted tool-isolated household responses. The first origin
is a stale survey-seeded initialization transition and is shown but not scored;
each later origin carries the preceding median simulated state forward. On the
three valid recursive transitions, consumption RMSE is 5.48 percentage points
with the wrong sign in all three months. Price signs are wrong in all three months,
credit signs in two of three, and employment signs are wrong in all three. The
compounded March-May consumption index falls from 100 to 86.40 while the
first-release actual rises to 102.14. Accounting still passes.

The next gates are therefore:

1. append the August native outcomes once released, without rewriting the forecast;
2. diagnose the remaining contractionary behavior and feedback on retrospective
   development origins without changing the corrected frozen August forecast;
3. rerun the historical path and require credible signs and magnitudes before adding
   firms, banks, or other institutional complexity;
4. run August and subsequent origins under a newly locked procedure only after that
   development gate is met.

The old adaptive-twin tournament is retired from the active path. Its code and
results remain recoverable at Git tag `macro-v1-weighted-demand` and in the local
hashed archive under `~/Downloads/llm-hank-docs/archive/v1/`.
