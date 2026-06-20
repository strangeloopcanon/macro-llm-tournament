.PHONY: test fixture data postcutoff-fixture

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v

fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.forecast_tournament \
		--llm-mode fixture \
		--card-count 8 \
		--vintage-context best_effort \
		--belief-targets best_effort \
		--typed-agent-panel \
		--output-dir outputs/spf_fixture

data:
	PYTHONPATH=src python3 -m macro_llm_tournament.download_data

postcutoff-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_tournament \
		--llm-mode fixture \
		--max-live-calls 0 \
		--vintage-context best_effort \
		--belief-targets best_effort \
		--typed-agent-panel \
		--output-dir outputs/spf_postcutoff_fixture
