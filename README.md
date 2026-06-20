# LLM macro forecast tournament

Research code for running as-of macro forecast tournaments against public Survey of Professional Forecasters data and simple statistical controls.

The repository contains the reusable harness only. Generated reports, model outputs, caches, and research notes are local artifacts and are not part of the public source tree.

## What is in the repo

- `src/macro_llm_tournament/forecast_tournament.py` builds and scores SPF-style forecast tournaments.
- `src/macro_llm_tournament/forecast_cards.py` creates as-of prompt cards with hidden realized outcomes and hidden same-card SPF consensus.
- `src/macro_llm_tournament/forecast_controls.py` implements no-change, rolling mean, AR, recursive least-squares, constant-gain, extrapolative, diagnostic, and official SPF benchmark controls.
- `src/macro_llm_tournament/fred_vintage.py` adds FRED/ALFRED real-time macro context when `FRED_API_KEY` is available.
- `src/macro_llm_tournament/survey_beliefs.py` loads household belief context from NY Fed SCE chart data and Michigan/FRED inflation expectations.
- `src/macro_llm_tournament/forecast_agent_panel.py` maps forecasts into a typed household-panel scaffold without spending extra LLM calls.

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

## Live LLM runs

Live runs are deliberately capped. A live run fails unless `--max-live-calls` is positive.

The current hard-gate command uses true real-time FRED/ALFRED context:

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

The post-cutoff live gate uses the same call cap discipline:

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

## Scope

This is an experimental research harness. Public source files are limited to the runner, controls, tests, and setup instructions. Research interpretations and generated summaries should be kept outside the public repository until they are ready for release.

## Data sources

- Philadelphia Fed Survey of Professional Forecasters: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
- Philadelphia Fed SPF data files: https://www.philadelphiafed.org/surveys-and-data/data-files
- FRED observations API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- FRED real-time periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- NY Fed Survey of Consumer Expectations: https://www.newyorkfed.org/microeconomics/sce
- Michigan inflation expectations on FRED: https://fred.stlouisfed.org/series/MICH

## License

MIT.
