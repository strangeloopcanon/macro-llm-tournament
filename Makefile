.PHONY: test fixture data postcutoff-fixture agent-fixture agent-counterfactual-fixture behavior-fixture persona-holdouts persona-belief-fixture persona-ecology-fixture demand-economy-fixture demand-economy-live-replay demand-vintage-oos-fixture macro-playground-fixture macro-performance-fixture macro-validity-scorecard postcutoff-behavior-fixture audit-fixture

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

persona-holdouts:
	PYTHONPATH=src python3 -m macro_llm_tournament.prepare_persona_holdouts \
		--respondent-count 36 \
		--period-count 3 \
		--start-as-of 2024-10-01 \
		--output-dir work/persona_beliefs

persona-belief-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.persona_belief_panel \
		--belief-mode fixture \
		--max-live-calls 0 \
		--models gpt-5.5,gpt-5.4 \
		--respondent-source fixture \
		--respondent-count 54 \
		--output-dir outputs/persona_belief_panel_fixture

persona-ecology-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.persona_ecology \
		--ecology-mode fixture \
		--max-live-calls 0 \
		--models gpt-5.5,gpt-5.4 \
		--respondent-source fixture \
		--respondent-count 60 \
		--period-count 4 \
		--prior-mode simulated \
		--feedback-mode closed_loop \
		--output-dir outputs/persona_ecology_fixture

persona-ecology-relative-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.persona_ecology \
		--ecology-mode fixture \
		--max-live-calls 0 \
		--models gpt-5.5 \
		--respondent-source fixture \
		--respondent-count 6 \
		--period-count 3 \
		--target-fields expected_inflation_1y \
		--prior-mode simulated \
		--feedback-mode closed_loop \
		--date-mode relative \
		--output-dir outputs/persona_ecology_fixture_relative_gate

demand-economy-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.demand_economy \
		--belief-mode fixture \
		--max-live-calls 0 \
		--models gpt-5.5 \
		--household-source fixture \
		--household-count 24 \
		--period-count 100 \
		--variants representative,adaptive,llm_belief,naive_persona \
		--feedback-mode closed_loop \
		--output-dir outputs/demand_economy_fixture

demand-economy-live-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.demand_economy \
		--provider codex_cli \
		--models gpt-5.5 \
		--belief-mode raw_replay \
		--raw-records-json outputs/demand_economy_live_gpt55_p20_12cell_full_v4/demand_raw_records.json \
		--max-live-calls 0 \
		--household-source fixture \
		--household-count 12 \
		--period-count 20 \
		--variants representative,adaptive,llm_belief,naive_persona \
		--fixture-variants naive_persona \
		--feedback-mode closed_loop \
		--scenarios baseline,transfer_shock,rate_hike,job_risk_shock,belief_feedback \
		--output-dir outputs/demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5

demand-vintage-oos-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.demand_vintage_oos \
		--mode fixture \
		--max-origins 18 \
		--history-periods 8 \
		--output-dir outputs/demand_vintage_oos_fixture

macro-playground-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_playground \
		--spec configs/macro_playground_fixture_spec.json \
		--mode fixture \
		--max-live-calls 0 \
		--output-dir outputs/macro_playground_fixture

macro-performance-fixture: demand-economy-fixture demand-vintage-oos-fixture
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_performance_gate \
		--mode fixture \
		--demand-run-dir outputs/demand_economy_fixture \
		--vintage-panel-dir work/fred_vintage_panel \
		--vintage-oos-dir outputs/demand_vintage_oos_fixture \
		--output-dir outputs/macro_performance_gate_fixture

macro-validity-scorecard: demand-economy-live-replay
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_validity \
		--demand-run-dir outputs/demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5 \
		--vintage-panel-dir work/fred_vintage_panel \
		--output-dir outputs/macro_validity_scorecard

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
