.PHONY: all check test ecology-fixture ecology-live-canary ecology-live-200 ecology-current-replay ecology-realize ecology-retrospective-live household-cohort vintage-bundle origin-snapshot

ORIGIN ?= 2026-07-01
AS_OF ?= 2026-07-10
MODEL ?= gpt-5.5
ORIGIN_SNAPSHOT ?= work/ecology_origins/$(AS_OF).json
ECOLOGY_CACHE ?= work/ecology_cache_200_july_v1
ECOLOGY_HOUSEHOLDS ?= work/persona_beliefs/persistent_household_scale_v1/initial_households_200.csv
ECOLOGY_HISTORY ?= work/persona_beliefs/persistent_household_scale_v1/selected_observed_history.csv
ECOLOGY_BUNDLE ?= work/dynamic_macro/frozen_2026_01_2026_05_common_month_v1
ECOLOGY_FIXTURE_DIR := examples/ecology_fixture
CURRENT_RUN_DIR ?= outputs/household_ecology_200_july_replay_current
REPLAY_REFERENCE_DIR ?= outputs/.household_ecology_replay_reference
RETROSPECTIVE_DIR ?= outputs/household_ecology_retrospective_2026_01_04
RETROSPECTIVE_CACHE ?= work/ecology_cache_retrospective_2026_01_04

all: check test

check:
	PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q src tests
	@for file in $$(git ls-files '*.json'); do python3 -m json.tool "$$file" >/dev/null || exit 1; done
	@python3 -c 'import json, pathlib, subprocess; [json.loads(line) for name in subprocess.check_output(["git", "ls-files", "*.jsonl"], text=True).splitlines() for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines() if line.strip()]'
	git diff --check

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v

ecology-fixture:
	rm -rf outputs/household_ecology_fixture_v1
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology \
		--origin 2026-05-01 \
		--mode fixture \
		--provider codex_cli \
		--model gpt-5.5 \
		--household-count 12 \
		--households $(ECOLOGY_FIXTURE_DIR)/households.csv \
		--history $(ECOLOGY_FIXTURE_DIR)/history.csv \
		--bundle $(ECOLOGY_FIXTURE_DIR)/origin_snapshot.json \
		--workers 4 \
		--max-live-calls 0 \
		--output-dir outputs/household_ecology_fixture_v1

ecology-live-canary:
	rm -rf outputs/household_ecology_canary_v1
	CODEX_CLI_REASONING_EFFORT=high CODEX_CLI_TIMEOUT_SECONDS=600 \
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology \
		--origin 2026-05-01 \
		--mode live \
		--provider codex_cli \
		--model gpt-5.5 \
		--household-count 12 \
		--households $(ECOLOGY_HOUSEHOLDS) \
		--history $(ECOLOGY_HISTORY) \
		--bundle $(ECOLOGY_BUNDLE) \
		--workers 4 \
		--max-live-calls 14 \
		--output-dir outputs/household_ecology_canary_v1

ecology-live-200:
	rm -rf $(CURRENT_RUN_DIR)
	CODEX_CLI_REASONING_EFFORT=high CODEX_CLI_TIMEOUT_SECONDS=600 \
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology \
		--origin $(ORIGIN) \
		--bundle $(ORIGIN_SNAPSHOT) \
		--households $(ECOLOGY_HOUSEHOLDS) \
		--history $(ECOLOGY_HISTORY) \
		--mode live \
		--provider codex_cli \
		--model $(MODEL) \
		--household-count 200 \
		--workers 4 \
		--max-live-calls 210 \
		--cache-dir $(ECOLOGY_CACHE) \
		--output-dir $(CURRENT_RUN_DIR)

ecology-current-replay:
	rm -rf $(CURRENT_RUN_DIR) $(REPLAY_REFERENCE_DIR)
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology \
		--origin $(ORIGIN) \
		--bundle $(ORIGIN_SNAPSHOT) \
		--households $(ECOLOGY_HOUSEHOLDS) \
		--history $(ECOLOGY_HISTORY) \
		--mode replay \
		--provider codex_cli \
		--model $(MODEL) \
		--household-count 200 \
		--workers 4 \
		--max-live-calls 0 \
		--cache-dir $(ECOLOGY_CACHE) \
		--output-dir $(REPLAY_REFERENCE_DIR)
	@expected=$$(python3 -c 'import json; print(json.load(open("$(REPLAY_REFERENCE_DIR)/manifest.json"))["replay_equivalence_sha256"])'); \
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology \
		--origin $(ORIGIN) \
		--bundle $(ORIGIN_SNAPSHOT) \
		--households $(ECOLOGY_HOUSEHOLDS) \
		--history $(ECOLOGY_HISTORY) \
		--mode replay \
		--provider codex_cli \
		--model $(MODEL) \
		--household-count 200 \
		--workers 4 \
		--max-live-calls 0 \
		--expected-replay-sha256 $$expected \
		--cache-dir $(ECOLOGY_CACHE) \
		--output-dir $(CURRENT_RUN_DIR)
	rm -rf $(REPLAY_REFERENCE_DIR)

ecology-realize:
	@test -n "$(REALIZATIONS_CSV)" || (echo "REALIZATIONS_CSV is required" >&2; exit 2)
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology_realizations \
		--run-dir $(CURRENT_RUN_DIR) \
		--realizations-csv $(REALIZATIONS_CSV)

ecology-retrospective-live:
	rm -rf $(RETROSPECTIVE_DIR)
	CODEX_CLI_REASONING_EFFORT=high CODEX_CLI_TIMEOUT_SECONDS=600 \
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology_retrospective \
		--origins 2026-01-01:2026-04-01 \
		--bundle work/dynamic_macro/frozen_2026_01_2026_05_common_month_v1 \
		--targets work/dynamic_macro/frozen_2026_01_2026_06_v1/targets.csv \
		--households $(ECOLOGY_HOUSEHOLDS) \
		--history $(ECOLOGY_HISTORY) \
		--mode live \
		--provider codex_cli \
		--model $(MODEL) \
		--household-count 200 \
		--workers 8 \
		--max-live-calls 840 \
		--cache-dir $(RETROSPECTIVE_CACHE) \
		--output-dir $(RETROSPECTIVE_DIR)

household-cohort:
	PYTHONPATH=src python3 -m macro_llm_tournament.persistent_households \
		--input-csv work/persona_beliefs/sce_real_microdata.csv \
		--output-dir work/persona_beliefs/persistent_household_scale_v1 \
		--private-output-dir work/persona_beliefs/persistent_household_scale_v1_private \
		--cohort-event-date 2025-04-01 \
		--master-sample-size 200 \
		--core-sample-size 81 \
		--sample-seed 20250709

vintage-bundle:
	PYTHONPATH=src python3 -m macro_llm_tournament.frozen_vintage_bundle \
		--origins 2026-01-01:2026-05-01 \
		--mode fred \
		--output-dir work/dynamic_macro/frozen_2026_01_2026_05_common_month_v1

origin-snapshot:
	PYTHONPATH=src python3 -m macro_llm_tournament.ecology_inputs \
		--origin $(ORIGIN) \
		--as-of $(AS_OF) \
		--output $(ORIGIN_SNAPSHOT)
