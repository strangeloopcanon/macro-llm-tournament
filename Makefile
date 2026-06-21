.PHONY: test fixture data postcutoff-fixture agent-fixture agent-counterfactual-fixture behavior-fixture postcutoff-behavior-fixture audit-fixture

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

agent-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.agent_economy \
		--llm-mode fixture \
		--max-live-calls 0 \
		--agent-mode fixture \
		--max-agent-live-calls 0 \
		--card-count 8 \
		--vintage-context best_effort \
		--belief-targets best_effort \
		--output-dir outputs/agent_economy_fixture

agent-counterfactual-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.agent_economy \
		--llm-mode fixture \
		--max-live-calls 0 \
		--agent-mode fixture \
		--max-agent-live-calls 0 \
		--card-count 8 \
		--vintage-context best_effort \
		--belief-targets best_effort \
		--belief-sources llm \
		--household-policy residual_over_liquidity \
		--feedback-mode closed_loop \
		--counterfactual-shocks rate_hike,growth_slump,credit_crunch \
		--output-dir outputs/agent_economy_counterfactual_fixture

behavior-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.behavior_gate \
		--behavior-mode fixture \
		--max-live-calls 0 \
		--output-dir outputs/behavior_gate_fixture

postcutoff-behavior-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_behavior_gate \
		--data-mode fixture \
		--agent-mode fixture \
		--max-live-calls 0 \
		--output-dir outputs/postcutoff_behavior_gate_fixture

audit-fixture: fixture
	PYTHONPATH=src python3 -m macro_llm_tournament.forecast_audit \
		--run-dir outputs/spf_fixture \
		--recall-mode fixture \
		--qualitative-recall-mode fixture \
		--output-dir outputs/forecast_audit_fixture
