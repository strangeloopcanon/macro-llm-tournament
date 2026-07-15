# Current Project Surface

The active system is the household-first rolling microeconomy in
`macro_llm_tournament.ecology`.

It contains 200 persistent anonymized SCE household types. Each household receives
its own survey history, an SCE-conditioned SCF 2022 financial state, and a compact
public-information card containing only data visible at the forecast origin. GPT-5.5
returns beliefs and conditional dollar policies for committed spending,
discretionary spending, deposits, debt repayment, and borrowing. Deterministic code
enforces the budget and accounting identities.

The active diagnostic isolates household demand. Respondent employment and wages are
held fixed, production follows expected sales with gradual inventory adjustment, and
firms and banks are not LLM agents. Every rolling origin restarts from the same fixed
SCE-SCF anchor; only the origin-visible information changes. Simulated cash, debt, and
forecast errors never enter the next rolling forecast.

## What The Historical Run Shows

Across January-April 2026 origins and February-May first-release outcomes, the
median consumption forecasts are **+0.13%, +0.16%, +0.30%, and +0.31%**. Actual
nominal PCE growth is **+0.48%, +0.90%, +0.51%, and +0.71%**.

- Consumption direction: **4/4 correct**.
- Consumption RMSE: **0.47 percentage points**.
- Retired ecology RMSE: **5.48 points**, with 0/3 signs correct.
- Origin-visible routine-drift RMSE: **0.24 points**.
- Saving-rate direction: **3/4 correct**.
- Revolving-credit direction: **1/4 correct**.
- Accounting: **PASS** in all four runs; maximum residual `1.68e-08`.

The improvement is real, but the LLM path remains too conservative. It compounds to
**100.90** from a base of 100, versus **102.63** for first-release PCE and **101.75**
for the routine drift anchor. The remaining problem is ordinary nominal-spending
amplitude, not a collapse caused by labor feedback or a broken household budget.

## Frozen Forecast

The July 1 origin, using information through July 10, is frozen for August 2026.
Its median consumption forecast is **+0.28%**. The run contains 200 accepted Codex
CLI responses, replays with 200 cache hits and zero calls, matches its immutable live
reference, and passes accounting with maximum residual `1.12e-08`.

This is not an accuracy claim until the August first release exists. The household
survey anchor is from March-April 2025, while public macro information is current to
the forecast cutoff.

## Next Work

1. Append the August realization without changing the frozen forecast.
2. Continue the same rolling procedure at each new origin.
3. Diagnose why household policies omit part of routine nominal drift; do not add
   LLM firms or banks until repeated errors show an institutional mechanism is missing.
4. Keep the historical chart as development evidence, not confirmatory evidence.

The retired adaptive-twin and macro-tournament work remains recoverable at Git tag
`macro-v1-weighted-demand` and in the local hashed archive under
`~/Downloads/llm-hank-docs/archive/v1/`.
