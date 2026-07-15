# Current Project Surface

The active system is a 200-household rolling demand economy. Real SCE histories
supply household heterogeneity and personal belief priors. Deterministic matches
to the 2022 SCF supply income, liquidity, spending, and debt states. GPT-5.5 writes
paired state-contingent next-month dollar policies per household; code applies
them to the fixed observed employment shares and closes the accounts.

Each forecast origin starts again from the same fixed SCE-SCF household anchor and
receives newly available public information. It does not carry simulated balances,
forecast errors, or later observations into the next origin. This is the sensible
rolling procedure for period-by-period forecasts; free-running counterfactuals are
not part of the active evidence path.

## What We Know

Across January-April 2026 origins, the ecology forecasts nominal consumption
growth of **+0.01%, +0.02%, +0.06%, and +0.16%**. First-release PCE growth is
**+0.48%, +0.90%, +0.51%, and +0.71%**.

- Direction: **4/4 correct**.
- RMSE: **0.61 percentage points**.
- Origin-visible routine-drift RMSE: **0.24 points**.
- Revolving-credit direction: **1/4 correct**.
- Accounting: **PASS** in all four runs.

The result is narrower than “the simulated economy forecasts well.” It says the
household-policy design produces positive demand at all four diagnostic origins.
It remains too conservative and is not yet competitive with
a simple current-information anchor.

## Frozen Forecast

The July 1 origin, using information through July 10, is frozen for August 2026.
Its point forecast is **+0.08%** nominal consumption growth. The run used 200 fresh
Codex CLI calls with zero failures; its replay uses 200 cache hits and zero calls,
matches the immutable live reference, and passes accounting with maximum residual
`1.30e-08`.

## Next Work

1. Append the August first release without changing the frozen forecast.
2. Continue the same rolling forecast at each new origin.
3. Improve the household policy's response amplitude on spent historical origins,
   while keeping the genuine SCE personal job-loss prior and fixed accounting
   contract.
4. Add LLM firms or banks only if repeated errors identify a missing institutional
   mechanism rather than a household elicitation problem.

The canonical evidence is in
`outputs/household_ecology_200_july_v20_current/` and
`outputs/household_ecology_retrospective_2026_01_04_v20/`. Earlier versions are
superseded and belong in the local archive.
