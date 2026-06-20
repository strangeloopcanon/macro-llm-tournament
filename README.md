# LLM macro forecast tournament

Research code for running as-of macro forecast tournaments against public Survey of Professional Forecasters data and simple statistical controls.

The repository contains the reusable harness only. Generated reports, model outputs, caches, and research notes are local artifacts and are not part of the public source tree.

## What is in the repo

- `src/macro_llm_tournament/forecast_tournament.py` builds and scores SPF-style forecast tournaments.
- `src/macro_llm_tournament/forecast_audit.py` audits completed tournament runs for reviewer checks: direct realized-value recall, surprise splits, Theil's U, and paired loss gaps.
- `src/macro_llm_tournament/forecast_cards.py` creates as-of prompt cards with hidden realized outcomes and hidden same-card SPF consensus.
- `src/macro_llm_tournament/forecast_controls.py` implements no-change, rolling mean, AR, recursive least-squares, constant-gain, extrapolative, diagnostic, and official SPF benchmark controls.
- `src/macro_llm_tournament/fred_vintage.py` adds FRED/ALFRED real-time macro context when `FRED_API_KEY` is available.
- `src/macro_llm_tournament/survey_beliefs.py` loads household belief context from NY Fed SCE chart data and Michigan/FRED inflation expectations.
- `src/macro_llm_tournament/forecast_agent_panel.py` maps forecasts into a typed household-panel scaffold without spending extra LLM calls.
- `src/macro_llm_tournament/agent_economy.py` runs the forecast-first typed agent economy CLI.
- `src/macro_llm_tournament/agent_llm.py`, `agent_runtime.py`, `agent_types.py`, `agent_targets.py`, and `agent_report.py` hold the LLM-agent schema, accounting runtime, SCF-style type cells, origin-level household-belief scoring, and report rendering.
- `src/macro_llm_tournament/behavior_gate.py` scores typed household agents against public stimulus-response targets for MPC, liquidity gradients, debt repayment, and liquid saving.
- `src/macro_llm_tournament/postcutoff_behavior_gate.py` runs the contamination-clean post-cutoff household behavior proxy gate using public FRED spending, saving, and revolving-credit series.

## Quick start

Create an environment and install the base dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run a zero-cost fixture forecast tournament:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.forecast_tournament \
  --llm-mode fixture \
  --card-count 8 \
  --vintage-context best_effort \
  --belief-targets best_effort \
  --typed-agent-panel \
  --output-dir outputs/spf_fixture
```

This downloads public SPF/SCE/Michigan data as needed, writes local outputs under `outputs/`, and makes no live LLM calls.

After installing the package, the same runner is available as `macro-llm-forecast-tournament`.

Download the public local data bundle:

```bash
make data
```

This writes ignored files under `work/`, including SPF workbooks, survey-belief files, public FRED/ALFRED graph CSVs, and SCF public files.

After the local data bundle exists, run a zero-cost post-cutoff fixture tournament:

```bash
make postcutoff-fixture
```

This builds post-cutoff SPF cards from official SPF mean forecast files. Rows with complete FRED proxy realizations are scored immediately; incomplete rows are frozen for later rescoring.

Run the zero-cost typed agent economy fixture:

```bash
make agent-fixture
```

This derives household type cells from the local public SCF extract when available, runs the packed typed-agent fixture schema for households/firms/banks, and emits persistent agent state, desired actions, feasible actions, aggregate outcomes, accounting diagnostics, origin-level household-belief target scores, and a report under `outputs/`.

Run the zero-cost household behavior target fixture:

```bash
make behavior-fixture
```

This scores SCF-style household types against public stimulus-response targets from the tax-rebate, 2008 stimulus, and 2020 EIP literature. The target gate covers aggregate MPC/spending, liquidity gradients, debt repayment, and liquid saving.

Run the zero-cost post-cutoff household behavior proxy fixture:

```bash
make postcutoff-behavior-fixture
```

This exercises the contamination-clean behavior runner without using live data or LLM calls.

Run the zero-cost forecast audit fixture:

```bash
make audit-fixture
```

This reruns the SPF fixture if needed, then emits reviewer-style audit files under `outputs/forecast_audit_fixture`.

## Live LLM runs

Live runs are deliberately capped. A live run fails unless `--max-live-calls` is positive.

The post-cutoff gate is the main contamination-clean screen. It uses SPF detail forecast files after the model cutoff and scores rows when FRED proxy realizations are complete:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_tournament \
  --provider codex_cli \
  --model gpt-5.5 \
  --llm-mode live \
  --max-live-calls 9 \
  --vintage-context require \
  --belief-targets best_effort \
  --typed-agent-panel \
  --output-dir outputs/spf_postcutoff_gpt55_2026q1q2
```

Use replay mode to rescore frozen post-cutoff cards after new public data arrives without spending live calls:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_tournament \
  --provider codex_cli \
  --model gpt-5.5 \
  --llm-mode replay \
  --max-live-calls 0 \
  --vintage-context require \
  --belief-targets best_effort \
  --typed-agent-panel \
  --output-dir outputs/spf_postcutoff_gpt55_replay
```

The historical tournament remains useful for scale, controls, and debugging, but its holdout is pre-cutoff:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.forecast_tournament \
  --provider codex_cli \
  --model gpt-5.5 \
  --llm-mode live \
  --max-live-calls 25 \
  --card-count 24 \
  --vintage-context require \
  --belief-targets best_effort \
  --typed-agent-panel \
  --output-dir outputs/spf_vintage_survey_agent_gpt55
```

Requirements for that run:

- `codex` CLI available on `PATH`, or set `CODEX_CLI_BIN`.
- `FRED_API_KEY` set in `.env` or the process environment.

Copy `.env.example` to `.env` and fill in only the keys you need.

Audit any completed forecast tournament:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.forecast_audit \
  --run-dir outputs/spf_vintage_survey_agent_gpt55 \
  --recall-mode live \
  --max-live-calls 1 \
  --recall-batch-size 64 \
  --fresh-cache \
  --output-dir outputs/spf_vintage_survey_agent_gpt55_audit
```

The direct recall probe is deliberately separate from the forecast call. It asks the model, without tools or files, whether it remembers the realized value for each card. Use `--recall-mode fixture` for zero-cost CI and `--recall-mode replay` when a prior recall cache should be reused.

Run a fresh-cache typed agent economy pilot after the fixture passes:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.agent_economy \
  --provider codex_cli \
  --model gpt-5.5 \
  --llm-mode live \
  --max-live-calls 9 \
  --fresh-forecast-cache \
  --agent-mode live \
  --max-agent-live-calls 8 \
  --fresh-agent-cache \
  --belief-sources llm \
  --card-count 8 \
  --vintage-context require \
  --belief-targets best_effort \
  --output-dir outputs/agent_economy_gpt55_fresh
```

`--fresh-forecast-cache` and `--fresh-agent-cache` write model responses under the run output directory instead of the shared local cache, so a live pilot does not accidentally replay prior calls. `--llm-mode` controls the macro forecast calls. `--agent-mode` controls whether typed households, firms, and banks are rule-based, fixture, replayed, or live LLM agents. Even when agents are live LLMs, code still enforces budgets, credit limits, portfolio conservation, and aggregation.

Agent state advances once per SPF origin, not once per variable card. That prevents CPI, GDP, and rate cards from the same survey date from becoming artificial time steps. Household belief target scores are also origin-level, so a single future SCE or Michigan observation is counted once per origin.

Run a fresh-cache household behavior target pilot:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.behavior_gate \
  --provider codex_cli \
  --model gpt-5.5 \
  --behavior-mode live \
  --max-live-calls 3 \
  --fresh-cache \
  --output-dir outputs/behavior_gate_gpt55_fresh
```

The behavior gate uses one packed LLM call per event scenario. The deterministic layer then aggregates household-type allocations and scores them against published spending, debt-repayment, saving, and liquidity-gradient moments.

Run the contamination-clean post-cutoff household behavior proxy gate:

```bash
PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_behavior_gate \
  --provider codex_cli \
  --model gpt-5.5 \
  --data-mode fred \
  --agent-mode live \
  --max-live-calls 4 \
  --fresh-cache \
  --scoreable-only \
  --output-dir outputs/postcutoff_behavior_gate_gpt55_fresh
```

This gate hides calendar dates, event labels, target months, and realized targets from the prompt. It scores only target months after the configured model cutoff and freezes target rows whose public data are not complete yet. It uses aggregate FRED behavior proxies, so it complements rather than replaces direct micro household behavior data.

## Scope

This is an experimental research harness. The working hypothesis is that LLMs are most useful as structured belief engines, then household/firm/bank behavior should be constrained by accounting rather than left to free-form simulation. Public source files are limited to runners, controls, tests, and setup instructions. Research interpretations and generated summaries should be kept outside the public repository until they are ready for release.

## Data sources

- Philadelphia Fed Survey of Professional Forecasters: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
- Philadelphia Fed SPF data files: https://www.philadelphiafed.org/surveys-and-data/data-files
- FRED observations API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- FRED real-time periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- NY Fed Survey of Consumer Expectations: https://www.newyorkfed.org/microeconomics/sce
- Michigan inflation expectations on FRED: https://fred.stlouisfed.org/series/MICH
- Parker, Souleles, Johnson, and McClelland 2008 ESP replication archive: https://www.openicpsr.org/openicpsr/project/116117/version/V1/view
- CFPB/Pagel et al. liquidity and EIP response note: https://files.consumerfinance.gov/f/documents/cfpb_pagel_income-liquidity-and-the-consumption-response-to-the-2020-economic-_H7TYltp.pdf
- NY Fed Liberty Street Economics EIP use summary: https://libertystreeteconomics.newyorkfed.org/2020/10/how-have-households-used-their-stimulus-payments-and-how-would-they-spend-the-next/
- FRED Personal Consumption Expenditures: https://fred.stlouisfed.org/series/PCE
- FRED Personal Saving Rate: https://fred.stlouisfed.org/series/PSAVERT
- FRED Revolving Consumer Credit: https://fred.stlouisfed.org/series/REVOLSL
- FRED Advance Retail Sales: https://fred.stlouisfed.org/series/RSAFS

## License

MIT.
