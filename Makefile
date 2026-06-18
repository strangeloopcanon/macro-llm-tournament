.PHONY: test fixture

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
