# Macro Validity Scorecard

## Bottom Line
Verdict: `macro_behavior_dynamics_ready_but_vintage_oos_unscored`. Micro behavior is `pass`, impulse-response shape is `pass`, and vintage OOS readiness is `partial`. So the current demand economy has a real qualitative macro-dynamics score. Broader macro validity still depends on a scored date-free vintage OOS run.

## Scorecard
| gate | status | passed_count | scored_count | gap_count | failed_count | blocking_issue_count | interpretation | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| micro_behavior | pass | 12 | 12 | 2 | 0 | 0 | Micro behavior is scored against transfer-MPC bridges, HANK-style heterogeneity, precaution, and direct-action gaps. | Use this as the core consumption-behavior gate; labor and portfolio targets remain future-scope extensions. |
| impulse_response | pass | 10 | 10 | 0 | 0 | 0 | Impulse responses are scored as scenario-minus-baseline shape constraints. | Keep this as the qualitative IRF gate; add sourced magnitude bands before calling it empirical IRF validation. |
| vintage_oos | partial | 13 | 13 | 3 | 0 | 3 | Vintage context coverage is scored separately from hidden-outcome OOS performance. | Build the date-free demand-vintage card/target/scoring runner and replay this scorecard on those artifacts. |

## Micro Behavior Gate
| gate | source | variant | metric | value | target_low | target_high | status | passed | blocking | target_kind | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | aggregate_transfer_mpc_public_bridge | 0.3560 | 0.2660 | 0.9000 | pass | True | True | public_target_bridge | Existing transfer-shock impact MPC against public total-spending transfer-response bands. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | low_liquidity_mpc_public_bridge | 0.6199 | 0.4000 | 0.7000 | pass | True | True | public_target_bridge | Low-liquidity impact MPC against the public low-balance bridge band. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | high_liquidity_mpc_public_bridge | 0.1046 | 0.0500 | 0.1500 | pass | True | True | public_target_bridge | High-liquidity impact MPC against the public high-balance bridge band. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | low_high_liquidity_mpc_ratio_public_bridge | 5.9260 | 3.0000 | 6.0000 | pass | True | True | public_target_bridge | Low-liquidity to high-liquidity MPC ratio against the public liquidity-gradient bridge band. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | liquidity_mpc_gradient | 0.5153 | 0.1500 | 1.0000 | pass | True | True | hank_shape_constraint | Low-liquidity impact MPC minus high-liquidity impact MPC. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | income_mpc_gradient | 0.3054 | 0.0500 | 0.8000 | pass | True | True | hank_shape_constraint | Low-income impact MPC minus high-income impact MPC. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | job_risk_impact_consumption_delta | -506.8503 | -inf | -0.0000 | pass | True | True | behavior_mechanism_constraint | Consumption falls when perceived job risk rises before income changes. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | job_risk_impact_income_delta_abs | 0.0000 | 0.0000 | 0.0000 | pass | True | True | behavior_mechanism_constraint | Income is unchanged on impact in a pure perceived-risk shock. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | max_accounting_abs_residual | 0.0000 | 0.0000 | 0.0000 | pass | True | True | accounting_constraint | Household and goods-market identities hold to numerical tolerance. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | transfer_allocation_share_sum_residual | 0.0000 | 0.0000 | 0.0000 | pass | True | True | mechanism_accounting | Transfer allocation shares should exhaust each transfer dollar across consumption, debt repayment, and liquid saving. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | debt_repayment_action_target | 0.3284 | 0.3000 | 0.4000 | pass | True | True | public_direct_behavior_target | Demand-economy transfer mechanism allocates a public-target-consistent share to debt repayment. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | liquid_saving_action_target | 0.3397 | 0.3100 | 0.4100 | pass | True | True | public_direct_behavior_target | Demand-economy transfer mechanism allocates a public-target-consistent share to liquid saving. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | labor_response_action_target |  |  |  | gap | False | False | future_scope_gap | Demand-economy output does not yet score direct household labor-response targets. |
| micro_behavior | llm_belief_raw_replay_gpt-5.5 | llm_belief | portfolio_liquidity_shift_action_target |  |  |  | gap | False | False | future_scope_gap | Demand-economy output does not yet score direct portfolio or liquidity-shift targets. |

## Impulse-Response Shape Gate
| gate | source | variant | metric | value | target_low | target_high | status | passed | blocking | target_kind | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | rate_hike_output_negative_share_p1_p8 | 1.0000 | 0.7500 | inf | pass | True | True | empirical_shape | Rate-hike output-gap response is negative in most early periods. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | rate_hike_consumption_negative_share_p1_p8 | 1.0000 | 0.7500 | inf | pass | True | True | empirical_shape | Rate-hike consumption response is negative in most early periods. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | rate_hike_inflation_lagged_negative_share_p4_p12 | 1.0000 | 0.6000 | inf | pass | True | True | empirical_shape | Rate-hike inflation response turns negative over the lagged window. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | rate_hike_output_trough_period | 4.0000 | 2.0000 | 12.0000 | pass | True | True | empirical_shape | Rate-hike output trough occurs after impact and inside the medium-run window. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | rate_hike_output_sign_reversal_pre_p8 | 0.0000 | 0.0000 | 0.0000 | pass | True | True | empirical_shape | Rate-hike output response does not flip positive before period 8. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | transfer_consumption_peak_period | 1.0000 | 1.0000 | 2.0000 | pass | True | True | empirical_shape | Transfer consumption response peaks on impact or soon after. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | transfer_consumption_decay_ratio_p4_vs_p1 | 0.0253 | 0.0000 | 1.0000 | pass | True | True | empirical_shape | Transfer consumption response fades by period 4 relative to impact. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | job_risk_consumption_negative_share_p1_p4 | 1.0000 | 0.7500 | inf | pass | True | True | behavior_mechanism_constraint | Job-risk news lowers consumption before income mechanically changes. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | job_risk_income_impact_abs | 0.0000 | 0.0000 | 0.0000 | pass | True | True | behavior_mechanism_constraint | Job-risk shock does not move income mechanically on impact. |
| impulse_response | llm_belief_raw_replay_gpt-5.5 | llm_belief | belief_feedback_output_rms_ratio | 2.6773 | 1.0500 | inf | pass | True | True | model_sanity | Belief-feedback output movement divided by baseline movement. |

## Vintage OOS Readiness Gate
| gate | source | variant | metric | value | target_low | target_high | status | passed | blocking | target_kind | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vintage_oos | work/fred_vintage_panel | vintage_oos | fred_vintage_panel_present | 1.0000 | 1.0000 | 1.0000 | pass | True | True | readiness | As-of FRED/ALFRED vintage panel files are present. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_origin_count | 117.0000 | 24.0000 | inf | pass | True | True | readiness | Vintage context has enough origins for a first OOS scoring split. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_test_origin_count | 22.0000 | 8.0000 | inf | pass | True | True | readiness | Vintage context has a held-out split large enough for a first scored run. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_realtime_fields_present | 1.0000 | 1.0000 | 1.0000 | pass | True | True | leakage_control | Vintage rows retain as-of and real-time fields needed for prompt leakage audits. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_real_consumption | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers real consumption via one of PCE, PCECC96. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_saving_rate | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers saving rate via one of PSAVERT. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_retail_spending | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers retail spending via one of RSAFS, RSXFS. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_inflation | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers inflation via one of CPIAUCSL, CPILFESL, PCEPI, PCEPILFE. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_labor_market | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers labor market via one of PAYEMS, UNRATE. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_policy_rate | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers policy rate via one of FEDFUNDS, TB3MS. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_sentiment | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers sentiment via one of MICH, UMCSENT. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_income | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers income via one of DSPIC96. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | vintage_series_group_output | 1.0000 | 1.0000 | 1.0000 | pass | True | True | coverage | Vintage panel covers output via one of GDPC1, INDPRO. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | demand_vintage_oos_cards_available |  |  |  | gap | False | True | scored_oos_gap | Scored date-free demand-vintage OOS artifact is not present yet: demand_vintage_oos_cards.csv. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | demand_vintage_oos_targets_available |  |  |  | gap | False | True | scored_oos_gap | Scored date-free demand-vintage OOS artifact is not present yet: demand_vintage_oos_targets.csv. |
| vintage_oos | work/fred_vintage_panel | vintage_oos | demand_vintage_oos_scores_available |  |  |  | gap | False | True | scored_oos_gap | Scored date-free demand-vintage OOS artifact is not present yet: demand_vintage_oos_scores.csv. |

## What This Means
The current HANK-lite demand run has moved past pure mechanics. It now has a repeatable macro-validity surface: direct micro-behavior constraints, scenario-minus-baseline impulse responses, and vintage OOS readiness are checked in one place. The result should be read as a bridge scorecard. Passing the IRF shape gate says the abstract economy reacts in the right qualitative directions. It does not yet prove real-world vintage OOS macro accuracy, because the date-free vintage demand cards and hidden outcome scores are still missing.

## Next Gate
The next implementation target is the scored demand-vintage OOS runner: build hidden date-free cards from the vintage panel, map as-of macro states into the demand economy, withhold future demand outcomes, run the belief module, and compare against no-change, rolling/trend, and SPF-style controls.

## Manifest
```json
{
  "created_at_utc": "2026-06-29T22:22:13.807284+00:00",
  "demand_run_dir": "outputs/demand_economy_live_gpt55_p20_12cell_mechanism_replay_v5",
  "demand_run_evidence_verdict": "hank_lite_belief_lab_ready",
  "demand_run_verdict": "hank_lite_belief_lab_ready",
  "outputs": [
    "macro_validity_scorecard.csv",
    "macro_validity_micro_behavior_scores.csv",
    "macro_validity_irf_paths.csv",
    "macro_validity_irf_scores.csv",
    "macro_validity_vintage_readiness.csv",
    "macro_validity_report.md",
    "manifest.json"
  ],
  "overall_verdict": "macro_behavior_dynamics_ready_but_vintage_oos_unscored",
  "schema_version": "macro_validity_scorecard_v1",
  "scored_source": "llm_belief_raw_replay_gpt-5.5",
  "scored_variant": "llm_belief",
  "vintage_panel_dir": "work/fred_vintage_panel"
}
```
