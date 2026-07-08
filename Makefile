DEMAND_ECONOMY_REPLAY_OUTPUT ?= outputs/demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5
POSTCUTOFF_REPLAY_OUTPUT ?= outputs/spf_postcutoff_replay_refresh

.PHONY: test fixture data postcutoff-fixture postcutoff-replay-refresh agent-fixture agent-counterfactual-fixture behavior-fixture behavior-architecture-fixture behavior-ecology-ctc-holdout-replay persona-holdouts persona-belief-fixture persona-ecology-fixture persona-ecology-relative-fixture persona-elicitation-prepare persona-elicitation-live demand-economy-fixture demand-economy-live-replay demand-vintage-oos-fixture state-policy-schedules-fixture state-policy-schedules-live macro-playground-fixture phase4-matched-twins-fixture phase4-prior-update-codex-replay phase4-prior-update-policy-schedule-replay phase4-prior-update-state-schedule-replay empirical-bridge-v4-fit empirical-bridge-v5-stabilized-fit phase4-prior-update-empirical-bridge-v4-replay phase4-prior-update-empirical-bridge-v4-holdlast-replay phase4-prior-update-empirical-bridge-v4-aligned-replay phase4-prior-update-empirical-bridge-v5-stabilized-replay phase4-prior-update-empirical-bridge-v5-stabilized-holdlast-replay phase4-prior-update-empirical-bridge-v5-stabilized-aligned-replay phase4-v4-diagnostics macro-performance-fixture macro-validity-scorecard postcutoff-behavior-fixture audit-fixture

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

postcutoff-replay-refresh:
	@if [ -e "$(POSTCUTOFF_REPLAY_OUTPUT)" ] && [ "$(ALLOW_REPLAY_OVERWRITE)" != "1" ]; then \
		echo "Refusing to overwrite $(POSTCUTOFF_REPLAY_OUTPUT). Set POSTCUTOFF_REPLAY_OUTPUT=outputs/spf_postcutoff_replay_$$(date -u +%Y%m%dT%H%M%SZ) or ALLOW_REPLAY_OVERWRITE=1."; \
		exit 2; \
	fi
	PYTHONPATH=src python3 -m macro_llm_tournament.postcutoff_tournament \
		--llm-mode replay \
		--max-live-calls 0 \
		--replay-cache-miss-policy freeze \
		--vintage-context best_effort \
		--belief-targets best_effort \
		--typed-agent-panel \
		--output-dir "$(POSTCUTOFF_REPLAY_OUTPUT)"

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

behavior-architecture-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.behavior_architecture_fidelity \
		--mode fixture \
		--max-live-calls 0 \
		--output-dir outputs/behavior_architecture_fidelity_fixture

behavior-ecology-ctc-holdout-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.behavior_ecology \
		--provider codex_cli \
		--model gpt-5.5 \
		--mode replay \
		--max-live-calls 0 \
		--arms policy \
		--scenario-ids ctc_2021_monthly_child_credit_style \
		--policy-raw-records-json outputs/behavior_ecology_gpt55_xhigh/ecology_raw_records.json \
		--output-dir outputs/behavior_ecology_ctc_holdout_policy_replay

persona-holdouts:
	PYTHONPATH=src python3 -m macro_llm_tournament.prepare_persona_holdouts \
		--respondent-count 54 \
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

persona-elicitation-prepare:
	PYTHONPATH=src python3 -m macro_llm_tournament.persona_elicitation_campaign \
		--mode prepare \
		--provider codex_cli \
		--output-dir outputs/persona_elicitation_campaign \
		--work-dir work/persona_beliefs/persona_elicitation_campaign

persona-elicitation-live:
	PYTHONPATH=src python3 -m macro_llm_tournament.persona_elicitation_campaign \
		--mode all \
		--provider codex_cli \
		--execute-live \
		--output-dir outputs/persona_elicitation_campaign \
		--work-dir work/persona_beliefs/persona_elicitation_campaign

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
	@if [ -e "$(DEMAND_ECONOMY_REPLAY_OUTPUT)" ] && [ "$(ALLOW_REPLAY_OVERWRITE)" != "1" ]; then \
		echo "Refusing to overwrite $(DEMAND_ECONOMY_REPLAY_OUTPUT). Set DEMAND_ECONOMY_REPLAY_OUTPUT=outputs/demand_economy_replay_$$(date -u +%Y%m%dT%H%M%SZ) or ALLOW_REPLAY_OVERWRITE=1."; \
		exit 2; \
	fi
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
		--output-dir "$(DEMAND_ECONOMY_REPLAY_OUTPUT)"

demand-vintage-oos-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.demand_vintage_oos \
		--mode fixture \
		--max-origins 18 \
		--history-periods 8 \
		--output-dir outputs/demand_vintage_oos_fixture

state-policy-schedules-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.state_policy_schedules \
		--mode fixture \
		--max-live-calls 0 \
		--household-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--output-dir outputs/state_policy_schedules_fixture

state-policy-schedules-live:
	CODEX_CLI_REASONING_EFFORT=high PYTHONPATH=src python3 -m macro_llm_tournament.state_policy_schedules \
		--provider codex_cli \
		--model gpt-5.5 \
		--mode live \
		--max-live-calls 1 \
		--household-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--output-dir outputs/state_policy_schedules_live_gpt55_sce_prior_update

macro-playground-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_playground \
		--spec configs/macro_playground_fixture_spec.json \
		--mode fixture \
		--max-live-calls 0 \
		--output-dir outputs/macro_playground_fixture

phase4-matched-twins-fixture:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode fixture \
		--data-mode fixture \
		--max-live-calls 0 \
		--household-source fixture \
		--household-count 24 \
		--period-count 12 \
		--feedback-mode closed_loop \
		--output-dir outputs/phase4_matched_twins_fixture

phase4-prior-update-codex-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2025-12-15 \
		--history-months 18 \
		--period-count 2 \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_prior_update_codex_replay_fred_onecard

phase4-prior-update-policy-schedule-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2025-12-15 \
		--history-months 18 \
		--period-count 2 \
		--behavior-policy-mode schedule \
		--behavior-policy-raw-records-json outputs/behavior_ecology_gpt55_xhigh/ecology_raw_records.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_policy_schedule_codex_replay_fred_onecard

phase4-prior-update-state-schedule-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2025-12-15 \
		--history-months 18 \
		--period-count 2 \
		--behavior-policy-mode state_schedule \
		--behavior-policy-state-profile-json outputs/state_policy_schedules_live_gpt55_sce_prior_update/state_behavior_policy_profile.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_state_schedule_codex_replay_fred_onecard

empirical-bridge-v4-fit:
	PYTHONPATH=src python3 -m macro_llm_tournament.empirical_bridge \
		--panel-csv work/empirical_bridge/spending_belief_panel.csv \
		--coverage-json work/empirical_bridge/spending_belief_panel_coverage.json \
		--output-json work/empirical_bridge/empirical_bridge_v4.json \
		--cell-targets-csv work/empirical_bridge/empirical_bridge_v4_cell_targets.csv \
		--validation-scores-csv work/empirical_bridge/empirical_bridge_v4_validation_scores.csv

empirical-bridge-v5-stabilized-fit:
	PYTHONPATH=src python3 -m macro_llm_tournament.empirical_bridge \
		--panel-csv work/empirical_bridge/spending_belief_panel.csv \
		--coverage-json work/empirical_bridge/spending_belief_panel_coverage.json \
		--output-json work/empirical_bridge/empirical_bridge_v5_stabilized.json \
		--cell-targets-csv work/empirical_bridge/empirical_bridge_v5_stabilized_cell_targets.csv \
		--validation-scores-csv work/empirical_bridge/empirical_bridge_v5_stabilized_validation_scores.csv \
		--estimator ridge_cv \
		--bridge-spec-version empirical_bridge_v5_stabilized

phase4-prior-update-empirical-bridge-v4-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2025-12-15 \
		--history-months 18 \
		--period-count 2 \
		--behavior-policy-mode empirical_bridge \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_onecard

phase4-prior-update-empirical-bridge-v4-holdlast-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2026-04-15 \
		--history-months 18 \
		--period-count 6 \
		--ecology-period-policy hold_last \
		--behavior-policy-mode empirical_bridge \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_holdlast_5cards

phase4-prior-update-empirical-bridge-v4-aligned-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2026-01-15 \
		--history-months 18 \
		--period-count 3 \
		--behavior-policy-mode empirical_bridge \
		--empirical-bridge-json work/empirical_bridge/empirical_bridge_v4.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v4_codex_replay_fred_2card_aligned_octnovdec

phase4-prior-update-empirical-bridge-v5-stabilized-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2025-12-15 \
		--history-months 18 \
		--period-count 2 \
		--behavior-policy-mode empirical_bridge \
		--empirical-bridge-json work/empirical_bridge/empirical_bridge_v5_stabilized.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_onecard

phase4-prior-update-empirical-bridge-v5-stabilized-holdlast-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_gpt54_100 \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2026-04-15 \
		--history-months 18 \
		--period-count 6 \
		--ecology-period-policy hold_last \
		--behavior-policy-mode empirical_bridge \
		--empirical-bridge-json work/empirical_bridge/empirical_bridge_v5_stabilized.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_holdlast_5cards

phase4-prior-update-empirical-bridge-v5-stabilized-aligned-replay:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_matched_twins \
		--mode replay \
		--belief-source persona_ecology_replay \
		--persona-ecology-dir outputs/persona_ecology_sce_prior_update_live_codex_gpt55_81_octnovdec_combined \
		--data-mode fred \
		--asof-start 2025-12-15 \
		--asof-end 2026-01-15 \
		--history-months 18 \
		--period-count 3 \
		--behavior-policy-mode empirical_bridge \
		--empirical-bridge-json work/empirical_bridge/empirical_bridge_v5_stabilized.json \
		--scoring-label retrospective \
		--max-live-calls 0 \
		--output-dir outputs/phase4_matched_twins_empirical_bridge_v5_stabilized_codex_replay_fred_2card_aligned_octnovdec

phase4-v4-diagnostics:
	PYTHONPATH=src python3 -m macro_llm_tournament.phase4_v4_diagnostics \
		--output-dir outputs/phase4_v4_diagnostics

macro-performance-fixture: demand-economy-fixture demand-vintage-oos-fixture
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_performance_gate \
		--mode fixture \
		--demand-run-dir outputs/demand_economy_fixture \
		--vintage-panel-dir work/fred_vintage_panel \
		--vintage-oos-dir outputs/demand_vintage_oos_fixture \
		--output-dir outputs/macro_performance_gate_fixture

macro-validity-scorecard:
	@test -d "$(DEMAND_ECONOMY_REPLAY_OUTPUT)" || { echo "Missing $(DEMAND_ECONOMY_REPLAY_OUTPUT). Run demand-economy-live-replay with a fresh DEMAND_ECONOMY_REPLAY_OUTPUT first."; exit 2; }
	PYTHONPATH=src python3 -m macro_llm_tournament.macro_validity \
		--demand-run-dir "$(DEMAND_ECONOMY_REPLAY_OUTPUT)" \
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
