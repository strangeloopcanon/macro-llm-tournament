import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from macro_llm_tournament.forecast_agent_panel import build_forecast_agent_panel
from macro_llm_tournament.agent_economy import (
    AgentLLMClient,
    add_counterfactual_forecasts,
    agent_prompt,
    build_agent_belief_target_rows,
    build_household_type_cells,
    fixture_agent_payload,
    normalize_agent_payload,
    run_agent_economy,
    score_agent_belief_targets,
)
from macro_llm_tournament.behavior_gate import (
    BEHAVIOR_SCENARIOS,
    BehaviorLLMClient,
    aggregate_behavior_actions,
    behavior_target_catalog,
    behavior_targets_frame,
    fixture_behavior_payload,
    join_cell_behavior_target_errors,
    normalize_behavior_payload,
    run_behavior_ablations,
    run_behavior_controls,
    run_behavior_gate,
    score_cell_behavior_targets,
    score_behavior_targets,
)
from macro_llm_tournament.demand_economy import (
    DemandEconomyClient,
    build_fixture_demand_households,
    classify_demand_economy_evidence,
    default_demand_scenarios,
    fixture_demand_payload,
    normalize_demand_payload,
    run_demand_economy,
    score_demand_economy_validation,
)
from macro_llm_tournament.forecast_audit import (
    DirectRecallClient,
    build_belief_structure_audit,
    build_qualitative_recall_targets,
    build_source_scores,
    build_surprise_audit,
    build_theil_u,
    direct_recall_call,
    direct_recall_cache_name,
    direct_recall_prompt,
    fixture_recall_payload,
    normalize_direct_recall_payload,
    normalize_qualitative_recall_payload,
    qualitative_recall_prompt,
    score_direct_recall,
    score_qualitative_recall,
)
from macro_llm_tournament.forecast_cards import (
    assert_no_prompt_target_leakage,
    build_forecast_cards,
    build_forecast_cards_from_rows,
    cards_to_frame,
    enrich_forecast_cards,
)
from macro_llm_tournament.forecast_controls import build_control_forecasts
from macro_llm_tournament.forecast_data import clean_numeric, quarter_index
from macro_llm_tournament.download_data import (
    DownloadResult,
    extract_scf_download_links,
    extract_spf_download_links,
    summarize,
)
from macro_llm_tournament.forecast_llm import ForecastLLMClient, normalize_forecast_payload, run_llm_forecasts, _extract_json
from macro_llm_tournament.forecast_scoring import score_forecast_slices, score_forecasts, verdict_from_scores
from macro_llm_tournament.fred_vintage import approximate_spf_as_of_date, build_vintage_context_for_cards
from macro_llm_tournament.llm_common import LLMUnavailable
from macro_llm_tournament.persona_belief_panel import (
    PersonaBeliefClient,
    build_fixture_respondent_panel,
    build_persona_cards,
    classify_persona_evidence,
    normalize_respondent_panel,
    persona_belief_prompt,
    run_persona_beliefs,
    score_common_core,
    score_distribution_distance,
    score_gradient_match,
    score_regression_gradient_match,
    score_variance_flattening,
    _client_mode_and_cap,
    _sanitized_argv,
)
import macro_llm_tournament.persona_belief_panel as persona_belief_panel_module
from macro_llm_tournament.persona_ecology import (
    EcologyCard,
    PersonaEcologyClient,
    build_ecology_cards,
    build_fixture_ecology_panel,
    build_module_ablations,
    classify_ecology_evidence,
    normalize_persona_ecology_payload,
    normalize_ecology_panel,
    persona_ecology_prompt,
    run_persona_ecology,
    score_behavior_actions,
    score_module_ablations,
    score_period_levels,
    score_period_updates,
    score_temporal_dynamics,
    summarize_period_actions,
)
from macro_llm_tournament.postcutoff_tournament import (
    DETAIL_FORECAST_SPECS,
    annualized_pct_change,
    combine_official_and_detail_rows,
    count_newly_scoreable_rows,
    read_simple_xlsx,
    realize_variable,
    split_cards_by_replay_cache,
)
from macro_llm_tournament.postcutoff_behavior_gate import (
    PostcutoffBehaviorLLMClient,
    TARGET_SPECS,
    aggregate_agent_proxy_forecasts,
    build_postcutoff_behavior_cards,
    build_proxy_control_forecasts,
    fixture_fred_proxy_frames,
    normalize_postcutoff_behavior_payload,
    postcutoff_behavior_prompt,
    run_postcutoff_behavior_agents,
    score_proxy_forecasts,
)
from macro_llm_tournament.survey_beliefs import survey_context_by_card


def spf_fixture() -> pd.DataFrame:
    rows = []
    for variable, base in [("CPI", 2.0), ("RGDP", 2.5), ("UNEMP", 5.5), ("TBILL", 1.0)]:
        for idx, origin_index in enumerate(range(2010 * 4 + 1, 2021 * 4 + 1)):
            year = origin_index // 4
            quarter = origin_index - year * 4
            if quarter == 0:
                year -= 1
                quarter = 4
            origin = f"{year}:Q{quarter}"
            spf_forecast = base + 0.03 * idx + (0.1 if variable == "RGDP" and idx % 5 == 0 else 0.0)
            realized = spf_forecast + 0.2 + (50.0 if variable == "CPI" and origin == "2017:Q4" else 0.0)
            rows.append(
                {
                    "variable": variable,
                    "variable_name": variable,
                    "units": "percentage points",
                    "origin": origin,
                    "origin_year": year,
                    "origin_quarter": quarter,
                    "origin_index": origin_index,
                    "horizon": 1,
                    "spf_forecast": spf_forecast,
                    "official_iar_forecast": spf_forecast + 0.02,
                    "official_no_change_forecast": spf_forecast - 0.03,
                    "official_dar_forecast": spf_forecast + 0.04,
                    "official_darm_forecast": spf_forecast + 0.01,
                    "realized": realized,
                    "source_url": "https://example.test/source.xls",
                    "variable_page_url": "https://example.test/page",
                }
            )
    return pd.DataFrame(rows)


class ForecastTournamentTests(unittest.TestCase):
    def test_spf_error_helpers_parse_quarters_and_missing_sentinel(self):
        self.assertEqual(quarter_index("2015:Q1"), 8061)
        self.assertTrue(np.isnan(clean_numeric(42)))
        self.assertAlmostEqual(clean_numeric("3.5"), 3.5)

    def test_forecast_cards_are_asof_and_do_not_prompt_final_realizations(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            history_quarters=8,
            card_count=4,
        )

        self.assertEqual(len(cards), 4)
        assert_no_prompt_target_leakage(cards)
        for card in cards:
            prompt = card.prompt_payload
            self.assertNotIn("target_realized", json.dumps(prompt))
            self.assertNotIn("forecast_error", json.dumps(prompt))
            self.assertTrue(prompt["as_of_design"]["outcome_hidden"])
            self.assertTrue(prompt["as_of_design"]["history_uses_lagged_forecasts_only"])
            self.assertIn("historical_forecast_signal_summary", prompt)
            self.assertEqual(prompt["forecast_target_period"], card.origin)
            self.assertEqual(prompt["spf_step_ahead"], card.horizon)
            self.assertIn("dated by the forecasted period", prompt["target_period_note"])
            self.assertNotIn("survey_origin", prompt)
            self.assertNotIn("forecast_horizon_quarters_ahead", prompt)
            self.assertNotIn("historical_benchmark_performance", prompt)
            for row in prompt["available_history"]:
                self.assertNotIn("realized", row)
                self.assertNotIn("forecast_error", row)
            history_origins = [row["origin"] for row in prompt["available_history"]]
            self.assertTrue(all(quarter_index(origin) < card.origin_index for origin in history_origins))
        card_frame = cards_to_frame(cards)
        self.assertIn("asof_reference_value", card_frame.columns)
        self.assertIn("rolling_signal_mean_4", card_frame.columns)

    def test_controls_use_prior_forecast_signals_not_final_realizations(self):
        data = spf_fixture()
        cards = build_forecast_cards(
            data,
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )
        controls = build_control_forecasts(data, cards, tune_end_year=2017)
        no_change = controls[controls["source"] == "no_change"].iloc[0]

        self.assertAlmostEqual(no_change["point_forecast"], cards[0].prior_spf_forecast)
        self.assertNotAlmostEqual(no_change["point_forecast"], 50.0, delta=1.0)

    def test_vintage_and_survey_context_enrichment_changes_card_identity_without_leaking_target(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            history_quarters=8,
            card_count=2,
        )
        self.assertEqual(approximate_spf_as_of_date("2018:Q3"), "2018-08-15")
        with patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=False):
            vintage_context, vintage_rows, vintage_status = build_vintage_context_for_cards(cards, mode="best_effort")
        targets = pd.DataFrame(
            [
                {
                    "survey_source": "ny_fed_sce",
                    "target_name": "median_expected_inflation",
                    "date": pd.Timestamp("2017-12-01"),
                    "horizon_months": 12,
                    "value": 3.0,
                    "units": "percent",
                    "source_url": "https://example.test/sce",
                }
            ]
        )
        survey_context, survey_rows = survey_context_by_card(cards, targets)
        enriched = enrich_forecast_cards(
            cards,
            vintage_context_by_card=vintage_context,
            survey_context_by_card=survey_context,
        )

        self.assertEqual(vintage_status["status"], "missing_api_key")
        self.assertTrue(vintage_rows.empty)
        self.assertFalse(survey_rows.empty)
        self.assertNotEqual(cards[0].card_id, enriched[0].card_id)
        prompt = enriched[0].prompt_payload
        self.assertEqual(prompt["vintage_macro_context"]["status"], "missing_api_key")
        self.assertEqual(prompt["household_survey_belief_context"]["status"], "ok")
        assert_no_prompt_target_leakage(enriched)

    def test_scoring_verdict_and_slices_use_honest_coverage_labels(self):
        data = spf_fixture()
        cards = build_forecast_cards(
            data,
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=4,
        )
        card_frame = cards_to_frame(cards)
        controls = build_control_forecasts(data, cards, tune_end_year=2017)
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, raw = run_llm_forecasts(llm_client, cards)
        scores, behavior, joined = score_forecasts(card_frame, pd.concat([controls, llm_forecasts], ignore_index=True))
        slice_scores = score_forecast_slices(card_frame, joined)
        verdict = verdict_from_scores(scores)

        self.assertGreater(len(raw), 0)
        self.assertIn("constant_gain", set(scores["source"]))
        self.assertIn("recursive_least_squares", set(scores["source"]))
        self.assertIn("underreaction_slope_error_on_revision", behavior.columns)
        self.assertEqual(verdict["status"], "ok")
        non_variable = slice_scores[slice_scores["slice"].isin(["regime", "evaluation_split", "contamination"])]
        self.assertEqual(set(non_variable["variable"]), {"ALL"})

    def test_forecast_audit_reports_surprise_theil_and_source_scores(self):
        data = spf_fixture()
        cards = build_forecast_cards(
            data,
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=6,
        )
        card_frame = cards_to_frame(cards)
        controls = build_control_forecasts(data, cards, tune_end_year=2017)
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        _scores, _behavior, joined = score_forecasts(card_frame, pd.concat([controls, llm_forecasts], ignore_index=True))

        source_scores = build_source_scores(joined)
        surprise = build_surprise_audit(joined, card_frame)
        theil = build_theil_u(joined)

        self.assertIn("ALL", set(source_scores["variable"]))
        self.assertIn("llm_minus_spf_gap", set(surprise["source"]))
        self.assertIn("surprise_bucket", set(surprise["bucket_type"]))
        self.assertIn("event_bucket", set(surprise["bucket_type"]))
        no_change = theil[(theil["source"] == "no_change") & (theil["variable"] == "ALL")].iloc[0]
        self.assertAlmostEqual(float(no_change["theils_u_vs_no_change"]), 1.0)

    def test_belief_structure_audit_emits_dynamics_calibration_and_surprise_metrics(self):
        data = spf_fixture()
        cards = build_forecast_cards(
            data,
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=6,
        )
        card_frame = cards_to_frame(cards)
        controls = build_control_forecasts(data, cards, tune_end_year=2017)
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        _scores, _behavior, joined = score_forecasts(card_frame, pd.concat([controls, llm_forecasts], ignore_index=True))

        detail, summary = build_belief_structure_audit(joined)

        self.assertFalse(detail.empty)
        self.assertFalse(summary.empty)
        self.assertIn("ALL", set(summary["variable"]))
        self.assertIn("underreaction", set(detail["metric_family"]))
        self.assertIn("extrapolation", set(detail["metric_family"]))
        self.assertIn("disagreement_dispersion", set(detail["metric_family"]))
        self.assertIn("interval_calibration", set(detail["metric_family"]))
        self.assertIn("confidence_calibration", set(detail["metric_family"]))
        self.assertIn("surprise_response", set(detail["metric_family"]))
        llm_overall = summary[(summary["variable"] == "ALL") & (summary["source"].str.startswith("llm_"))].iloc[0]
        self.assertGreater(int(llm_overall["interval_n"]), 0)
        self.assertTrue(np.isfinite(float(llm_overall["interval_coverage_p10_p90"])))

    def test_direct_recall_probe_is_card_specific_and_fails_closed(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=2,
        )
        card_frame = cards_to_frame(cards)
        prompt = direct_recall_prompt(card_frame)

        self.assertIn(cards[0].card_id, prompt)
        self.assertNotIn("target_realized", prompt)
        self.assertNotIn(str(cards[0].target_realized), prompt)
        self.assertIn("forecast_target_period", prompt)
        self.assertIn("spf_step_ahead", prompt)
        self.assertNotIn("forecast_horizon_quarters_ahead", prompt)

        fixture = {"payload": fixture_recall_payload(card_frame)}
        normalized = normalize_direct_recall_payload(card_frame, fixture)
        self.assertEqual(len(normalized), len(cards))
        self.assertTrue(all(np.isnan(row["recalled_realized"]) for row in normalized))

        exact = card_frame[["card_id"]].copy()
        exact["recalled_realized"] = card_frame["target_realized"]
        exact["confidence"] = 1.0
        exact["reason"] = "test oracle"
        exact["cache_hit"] = True
        exact["cache_path"] = ""
        scores = score_direct_recall(card_frame, exact)
        overall = scores[scores["variable"] == "ALL"].iloc[0]
        self.assertEqual(int(overall["n"]), len(cards))
        self.assertAlmostEqual(float(overall["coverage"]), 1.0)
        self.assertAlmostEqual(float(overall["rmse"]), 0.0)

        broken = fixture_recall_payload(card_frame)
        broken["items"] = broken["items"][:-1]
        with self.assertRaises(LLMUnavailable):
            normalize_direct_recall_payload(card_frame, {"payload": broken})

    def test_qualitative_recall_probe_scores_path_memory_without_leaking_values(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=6,
        )
        card_frame = cards_to_frame(cards)
        controls = build_control_forecasts(spf_fixture(), cards, tune_end_year=2017)
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        _scores, _behavior, joined = score_forecasts(card_frame, pd.concat([controls, llm_forecasts], ignore_index=True))
        targets = build_qualitative_recall_targets(card_frame, joined)
        prompt = qualitative_recall_prompt(targets)

        self.assertIn(cards[0].card_id, prompt)
        self.assertNotIn("target_realized", prompt)
        self.assertNotIn("truth_direction", prompt)
        self.assertNotIn(str(cards[0].target_realized), prompt)
        self.assertIn("forecast_target_period", prompt)
        self.assertIn("spf_step_ahead", prompt)
        self.assertNotIn("forecast_horizon_quarters_ahead", prompt)
        self.assertIn("gpt_edge_bucket", targets.columns)
        first_target = targets[targets["card_id"] == cards[0].card_id].iloc[0]
        expected_target_label = cards[0].origin.replace(":", " ")
        self.assertEqual(first_target["target_quarter_label"], expected_target_label)

        oracle = {
            "payload": {
                "items": [
                    {
                        "card_id": row["card_id"],
                        "direction": row["truth_direction"],
                        "level_vs_norm": row["truth_level_vs_norm"],
                        "turbulence": row["truth_turbulence"],
                        "confidence": 1.0,
                        "reason": "test oracle",
                    }
                    for _, row in targets.iterrows()
                ]
            }
        }
        predictions = pd.DataFrame(normalize_qualitative_recall_payload(targets, oracle))
        scores = score_qualitative_recall(targets, predictions)

        overall = scores[(scores["group_type"] == "ALL") & (scores["group"] == "ALL")].iloc[0]
        self.assertAlmostEqual(float(overall["card_mean_accuracy"]), 1.0)
        self.assertGreaterEqual(float(overall["card_mean_accuracy"]), float(overall["card_mean_base_rate"]))
        self.assertIn("event_bucket", set(scores["group_type"]))

        refusal = {"payload": {"error": "No reliable memory of these post-cutoff outcomes."}}
        refused = pd.DataFrame(normalize_qualitative_recall_payload(targets, refusal))
        self.assertEqual(len(refused), len(targets))
        self.assertEqual(set(refused["predicted_direction"]), {""})
        refused_scores = score_qualitative_recall(targets, refused)
        refused_overall = refused_scores[
            (refused_scores["group_type"] == "ALL") & (refused_scores["group"] == "ALL")
        ].iloc[0]
        self.assertAlmostEqual(float(refused_overall["card_mean_accuracy"]), 0.0)

    def test_invalid_llm_payload_fails_closed_instead_of_falling_back(self):
        card = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=1,
        )[0]

        with self.assertRaises(LLMUnavailable):
            normalize_forecast_payload(card, {"payload": {"p10": 1.0, "p50": 2.0, "p90": 3.0}}, source="llm_test")

    def test_forecast_first_agent_panel_maps_llm_forecasts_to_typed_household_rows(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=4,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)

        panel, aggregate = build_forecast_agent_panel(cards, llm_forecasts)

        self.assertEqual(panel.groupby("card_id")["type_id"].nunique().min(), 8)
        self.assertEqual(aggregate.shape[0], len(cards))
        self.assertIn("consumption_change_pct", panel.columns)
        self.assertIn("desired_liquid_buffer_change_pct", aggregate.columns)

    def test_agent_economy_derives_weighted_scf_type_cells_from_public_extract(self):
        with TemporaryDirectory() as tmp:
            scf_dir = Path(tmp) / "scf" / "2022"
            scf_dir.mkdir(parents=True)
            path = scf_dir / "scfp2022excel.zip"
            frame = pd.DataFrame(
                [
                    {
                        "WGT": 1.0,
                        "INCOME": 45000,
                        "LIQ": 500,
                        "ASSET": 8000,
                        "DEBT": 3000,
                        "NETWORTH": 5000,
                        "HOUSES": 0,
                        "BUS": 0,
                        "AGE": 35,
                        "WAGEINC": 40000,
                        "FOODHOME": 6000,
                        "FOODAWAY": 2000,
                        "FOODDELV": 500,
                        "RENT": 12000,
                    },
                    {
                        "WGT": 2.0,
                        "INCOME": 120000,
                        "LIQ": 3000,
                        "ASSET": 650000,
                        "DEBT": 320000,
                        "NETWORTH": 330000,
                        "HOUSES": 550000,
                        "BUS": 0,
                        "AGE": 42,
                        "WAGEINC": 110000,
                    },
                    {
                        "WGT": 1.0,
                        "INCOME": 300000,
                        "LIQ": 90000,
                        "ASSET": 2500000,
                        "DEBT": 350000,
                        "NETWORTH": 2150000,
                        "HOUSES": 900000,
                        "BUS": 800000,
                        "AGE": 55,
                        "WAGEINC": 100000,
                    },
                ]
            )
            with ZipFile(path, "w") as archive:
                archive.writestr("SCFP2022.csv", frame.to_csv(index=False))

            cells, status = build_household_type_cells(work_dir=Path(tmp) / "scf", wave=2022)

        self.assertEqual(status["status"], "ok")
        self.assertEqual(cells["type_id"].nunique(), 8)
        self.assertAlmostEqual(float(cells["population_weight"].sum()), 1.0)
        self.assertTrue((cells["credit_limit_proxy"] >= 0).all())
        self.assertIn("scf_2022_public_extract", set(cells["source"]))

    def test_agent_economy_persistent_actions_pass_accounting_and_emit_aggregates(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=4,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        controls = build_control_forecasts(spf_fixture(), cards, tune_end_year=2017)
        forecasts = pd.concat([controls, llm_forecasts], ignore_index=True)

        state, desired, feasible, aggregates, diagnostics, scores = run_agent_economy(
            cards,
            forecasts,
            build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0],
            source_filters=["llm", "constant_gain"],
        )

        self.assertFalse(state.empty)
        self.assertFalse(desired.empty)
        self.assertFalse(feasible.empty)
        self.assertFalse(aggregates.empty)
        self.assertFalse(scores.empty)
        self.assertTrue(diagnostics["passes_accounting"].all())
        self.assertLessEqual(float(diagnostics["max_abs_cash_residual"].max()), 1e-6)
        self.assertLessEqual(float(diagnostics["max_abs_networth_residual"].max()), 1e-6)
        self.assertIn("firm_hiring_index", aggregates.columns)
        self.assertIn("credit_rationing_ratio", feasible.columns)

    def test_agent_economy_fixture_llm_agents_drive_household_firm_and_bank_rows(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=4,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        agent_client = AgentLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")

        _state, desired, feasible, aggregates, diagnostics, _scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            agent_client=agent_client,
        )

        self.assertEqual(len(agent_client.raw_records), len(cards))
        self.assertEqual(set(desired["agent_behavior_mode"]), {"llm_agent"})
        self.assertTrue(diagnostics["passes_accounting"].all())
        self.assertTrue(feasible["agent_bank_credit_multiplier"].map(np.isfinite).all())
        self.assertTrue(aggregates["firm_hiring_index"].map(np.isfinite).all())
        self.assertEqual(desired.groupby("card_id")["type_id"].nunique().min(), type_cells.shape[0])

    def test_agent_feasibility_keeps_liquid_assets_nonnegative_under_aggressive_payload(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["TBILL"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]

        class AggressiveAgent:
            raw_records: list[dict] = []

            def agent_panel(self, _card, _forecast, cells, _prior_states):
                return {
                    "household_by_type": {
                        str(row["type_id"]): {
                            "consumption_change_pct": 20.0,
                            "liquid_buffer_change_pct": 0.0,
                            "borrowing_desire_index": -5.0,
                            "portfolio_rebalance_to_liquid_pct": -15.0,
                            "job_search_intensity_index": 0.0,
                            "expected_inflation_1y": 2.5,
                            "expected_real_income_growth": 1.5,
                            "expected_unemployment_rate": 4.5,
                            "expected_short_rate": 4.0,
                            "confidence": 0.7,
                            "uncertainty": 0.4,
                        }
                        for _, row in cells.iterrows()
                    },
                    "firm": {"hiring_index": 0.0, "price_pressure_index": 0.0, "confidence": 0.6},
                    "bank": {"credit_supply_multiplier": 1.0, "credit_tightening_index": 0.0, "confidence": 0.6},
                }

        _state, _desired, feasible, _aggregates, diagnostics, _scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            agent_client=AggressiveAgent(),
        )

        self.assertGreaterEqual(float(feasible["liquid_assets_after"].min()), -1e-6)
        self.assertGreaterEqual(float(feasible["illiquid_assets_after"].min()), -1e-6)
        self.assertGreaterEqual(float(feasible["debt_after"].min()), -1e-6)
        self.assertTrue(feasible["passes_balance_sheet_floors"].all())
        self.assertTrue(diagnostics["passes_accounting"].all())
        self.assertGreaterEqual(float(diagnostics["min_liquid_assets_after"].min()), -1e-6)

    def test_agent_residual_policy_preserves_liquidity_group_means(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["RGDP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]

        class SkewedAgent:
            raw_records: list[dict] = []

            def agent_panel(self, _card, _forecast, cells, _prior_states):
                offsets = {
                    "liquid_poor_renter": 8.0,
                    "wealthy_htm_homeowner": -2.0,
                    "unemployed_low_liquid": 5.0,
                    "retiree_liquid_assets": -4.0,
                    "high_income_illiquid_rich": 3.0,
                    "business_owner_top_wealth": -1.0,
                }
                return {
                    "household_by_type": {
                        str(row["type_id"]): {
                            "consumption_change_pct": offsets.get(str(row["type_id"]), 1.5),
                            "liquid_buffer_change_pct": -0.5 * offsets.get(str(row["type_id"]), 1.5),
                            "borrowing_desire_index": 0.10 * offsets.get(str(row["type_id"]), 1.5),
                            "portfolio_rebalance_to_liquid_pct": 0.20 * offsets.get(str(row["type_id"]), 1.5),
                            "job_search_intensity_index": 0.05 * offsets.get(str(row["type_id"]), 1.5),
                            "expected_inflation_1y": 2.5,
                            "expected_real_income_growth": 2.0,
                            "expected_unemployment_rate": 4.0,
                            "expected_short_rate": 3.5,
                            "confidence": 0.7,
                            "uncertainty": 0.4,
                        }
                        for _, row in cells.iterrows()
                    },
                    "firm": {"hiring_index": 1.0, "price_pressure_index": 0.5, "confidence": 0.6},
                    "bank": {"credit_supply_multiplier": 1.0, "credit_tightening_index": 0.0, "confidence": 0.6},
                }

        _state, rule_desired, _rule_feasible, _rule_aggregates, _rule_diagnostics, _rule_scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            household_policy="rule",
        )
        _state, hybrid_desired, _hybrid_feasible, _hybrid_aggregates, _hybrid_diagnostics, _hybrid_scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            agent_client=SkewedAgent(),
            household_policy="residual_over_liquidity",
        )

        for group_name, rule_group in rule_desired.groupby("liquidity_group"):
            hybrid_group = hybrid_desired[hybrid_desired["liquidity_group"] == group_name]
            rule_mean = np.average(rule_group["desired_consumption_change_pct"], weights=rule_group["population_weight"])
            hybrid_mean = np.average(hybrid_group["desired_consumption_change_pct"], weights=hybrid_group["population_weight"])
            self.assertAlmostEqual(float(rule_mean), float(hybrid_mean), places=6)
        merged = rule_desired.merge(
            hybrid_desired,
            on=["card_id", "source", "type_id"],
            suffixes=("_rule", "_hybrid"),
        )
        self.assertGreater(
            float((merged["desired_consumption_change_pct_rule"] - merged["desired_consumption_change_pct_hybrid"]).abs().max()),
            0.01,
        )
        self.assertEqual(set(hybrid_desired["agent_behavior_mode"]), {"llm_residual_over_liquidity"})

    def test_agent_closed_loop_feedback_changes_next_origin_state(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["RGDP", "UNEMP"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2019,
            card_count=4,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]

        _state, none_desired, none_feasible, _none_aggregates, _none_diagnostics, _none_scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            feedback_mode="none",
        )
        _state, closed_desired, closed_feasible, _closed_aggregates, _closed_diagnostics, _closed_scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
            feedback_mode="closed_loop",
        )

        self.assertIn("event_income_feedback_pct", closed_feasible.columns)
        self.assertTrue(closed_feasible["event_income_feedback_pct"].abs().gt(0).any())
        later_origin = closed_desired["origin_index"] > closed_desired["origin_index"].min()
        self.assertGreater(
            float((closed_desired.loc[later_origin, "annual_income"] - none_desired.loc[later_origin, "annual_income"]).abs().max()),
            0.0,
        )
        state_history = pd.DataFrame(closed_desired.attrs["state_history_records"])
        state_final = pd.DataFrame(closed_desired.attrs["state_final_records"])
        self.assertFalse(state_history.empty)
        self.assertFalse(state_final.empty)
        self.assertIn("mean_income_feedback_pct", state_history.columns)
        self.assertIn("mean_credit_limit_feedback_multiplier", state_final.columns)
        self.assertEqual(set(none_feasible["feedback_mode"]), {"none"})
        self.assertEqual(set(closed_feasible["feedback_mode"]), {"closed_loop"})

    def test_counterfactual_forecasts_append_named_shock_sources(self):
        forecasts = pd.DataFrame(
            [
                {"source": "llm_codex_cli_gpt-5.5", "variable": "TBILL", "point_forecast": 3.0, "p10": 2.5, "p50": 3.0, "p90": 3.5},
                {"source": "llm_codex_cli_gpt-5.5", "variable": "RGDP", "point_forecast": 2.0, "p10": 1.5, "p50": 2.0, "p90": 2.5},
            ]
        )

        expanded, scenarios = add_counterfactual_forecasts(forecasts, ["rate_hike"])

        self.assertEqual(expanded.shape[0], 4)
        self.assertIn("llm_codex_cli_gpt-5.5__cf_rate_hike", set(expanded["source"]))
        tbill = expanded[(expanded["source"].str.endswith("__cf_rate_hike")) & (expanded["variable"] == "TBILL")].iloc[0]
        rgdp = expanded[(expanded["source"].str.endswith("__cf_rate_hike")) & (expanded["variable"] == "RGDP")].iloc[0]
        self.assertAlmostEqual(float(tbill["point_forecast"]), 4.0)
        self.assertAlmostEqual(float(rgdp["point_forecast"]), 1.6)
        self.assertEqual(set(scenarios["counterfactual_shock"]), {"rate_hike"})

    def test_agent_belief_target_scoring_uses_future_survey_observations(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )
        survey_targets = pd.DataFrame(
            [
                {
                    "survey_source": "michigan_survey_of_consumers",
                    "target_name": "median_expected_price_change_next_12_months",
                    "date": pd.Timestamp("2018-02-01"),
                    "horizon_months": 12,
                    "value": 2.9,
                    "units": "percent",
                    "source_url": "https://example.test/mich",
                },
                {
                    "survey_source": "michigan_survey_of_consumers",
                    "target_name": "median_expected_price_change_next_12_months",
                    "date": pd.Timestamp("2018-04-01"),
                    "horizon_months": 12,
                    "value": 3.1,
                    "units": "percent",
                    "source_url": "https://example.test/mich",
                },
            ]
        )
        targets = build_agent_belief_target_rows(cards, survey_targets)
        aggregates = pd.DataFrame(
            [
                {
                    "card_id": cards[0].card_id,
                    "source": "llm_codex_cli_gpt-5.5",
                    "origin": cards[0].origin,
                    "horizon": cards[0].horizon,
                    "variable": cards[0].variable,
                    "aggregate_expected_inflation_1y": 3.4,
                    "aggregate_confidence": 0.7,
                    "aggregate_uncertainty": 0.4,
                }
            ]
        )

        scores = score_agent_belief_targets(aggregates, targets)

        self.assertFalse(targets.empty)
        self.assertEqual(targets.iloc[0]["target_date"], "2018-04-01")
        self.assertEqual(scores.iloc[0]["n"], 1)
        self.assertAlmostEqual(scores.iloc[0]["mae"], 0.3)

    def test_agent_belief_targets_are_origin_level_not_repeated_per_variable(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP", "TBILL"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=3,
        )
        survey_targets = pd.DataFrame(
            [
                {
                    "survey_source": "ny_fed_sce",
                    "target_name": "median_expected_inflation",
                    "date": pd.Timestamp("2018-04-01"),
                    "horizon_months": 12,
                    "value": 3.1,
                    "units": "percent",
                    "source_url": "https://example.test/sce",
                }
            ]
        )

        targets = build_agent_belief_target_rows(cards, survey_targets)

        self.assertEqual(targets.shape[0], 1)
        self.assertEqual(targets.iloc[0]["card_count"], 3)
        self.assertEqual(targets.iloc[0]["variables"], "CPI,RGDP,TBILL")

    def test_household_behavior_gate_scores_spending_debt_and_liquidity_targets(self):
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        client = BehaviorLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0)

        actions = run_behavior_gate(BEHAVIOR_SCENARIOS, type_cells, llm_client=client)
        controls = run_behavior_controls(BEHAVIOR_SCENARIOS, type_cells)
        ablations = run_behavior_ablations(pd.concat([actions, controls], ignore_index=True))
        all_actions = pd.concat([actions, controls, ablations], ignore_index=True)
        aggregates = aggregate_behavior_actions(all_actions)
        scores = score_behavior_targets(aggregates, behavior_targets_frame())
        cell_targets = behavior_targets_frame(target_scope="cell")
        cell_joined = join_cell_behavior_target_errors(all_actions, cell_targets)
        cell_scores = score_cell_behavior_targets(all_actions, cell_targets)

        self.assertEqual(actions.shape[0], len(BEHAVIOR_SCENARIOS) * type_cells.shape[0])
        self.assertEqual(len(client.raw_records), len(BEHAVIOR_SCENARIOS))
        self.assertEqual(all_actions.groupby(["scenario_id", "source"])["type_id"].nunique().min(), type_cells.shape[0])
        use_sum = all_actions["total_spending_share"] + all_actions["debt_repayment_share"] + all_actions["liquid_saving_share"]
        self.assertLessEqual(float(use_sum.max()), 1.000001)
        self.assertIn("llm_codex_cli_gpt-5.5__liquidity_prior_75", set(ablations["source"]))
        self.assertIn("llm_codex_cli_gpt-5.5__liquidity_prior_50", set(ablations["source"]))
        self.assertIn("llm_codex_cli_gpt-5.5__residual_over_liquidity", set(ablations["source"]))
        self.assertEqual(ablations.groupby(["scenario_id", "source"])["type_id"].nunique().min(), type_cells.shape[0])
        self.assertFalse(scores.empty)
        self.assertIn("debt_saving", set(scores["target_family"]))
        self.assertIn("directional_debt_saving", set(scores["target_family"]))
        self.assertIn("liquidity_gradient", set(scores["target_family"]))
        self.assertIn("low_minus_high_debt_repayment_share", aggregates.columns)
        self.assertIn("high_minus_low_liquid_saving_share", aggregates.columns)
        self.assertFalse(cell_targets.empty)
        self.assertFalse(cell_joined.empty)
        self.assertFalse(cell_scores.empty)
        self.assertIn("cell_mpc_by_liquidity", set(cell_scores["target_family"]))
        self.assertEqual(set(cell_joined["target_scope"]), {"cell"})

    def test_household_behavior_target_range_scoring_rewards_inside_interval(self):
        targets = behavior_targets_frame()
        prediction_columns = sorted(set(targets["prediction_column"]))
        rows = []
        for scenario_id, group in targets.groupby("scenario_id"):
            row = {"scenario_id": scenario_id, "source": "exact", "n_types": 8}
            row.update({column: 0.0 for column in prediction_columns})
            for _, target in group.iterrows():
                row[str(target["prediction_column"])] = float(target["target_value"])
            rows.append(row)
        exact = pd.DataFrame(rows)
        bad = exact.copy()
        bad["source"] = "bad"
        for column in prediction_columns:
            bad[column] = 0.0

        scores = score_behavior_targets(pd.concat([exact, bad], ignore_index=True), targets)

        exact_all = scores[(scores["source"] == "exact") & (scores["target_family"] == "ALL")].iloc[0]
        bad_all = scores[(scores["source"] == "bad") & (scores["target_family"] == "ALL")].iloc[0]
        self.assertAlmostEqual(float(exact_all["rmse_range"]), 0.0)
        self.assertGreater(float(bad_all["rmse_range"]), 0.0)

    def test_behavior_target_catalog_scores_only_verified_public_targets(self):
        catalog = behavior_target_catalog(include_unscored=True)
        scored = behavior_targets_frame()
        cell_scored = behavior_targets_frame(target_scope="cell")

        self.assertGreater(catalog.shape[0], scored.shape[0])
        self.assertEqual(set(scored["source_status"]), {"verified_public"})
        self.assertTrue(scored["scored"].all())
        self.assertEqual(set(scored["target_scope"]), {"aggregate"})
        self.assertEqual(set(cell_scored["target_scope"]), {"cell"})
        self.assertIn("eip_2020_low_gt_high_debt_share", set(scored["target_id"]))
        self.assertIn("eip_2020_high_gt_low_saving_share", set(scored["target_id"]))
        self.assertTrue(cell_scored["type_id"].astype(str).str.len().gt(0).all())
        self.assertIn("unscored_gap", set(catalog["source_status"]))
        self.assertIn("response_variable", scored.columns)

    def test_cell_behavior_target_scoring_rewards_inside_interval(self):
        targets = behavior_targets_frame(target_scope="cell")
        actions = pd.DataFrame(
            [
                {
                    "scenario_id": row["scenario_id"],
                    "source": "exact",
                    "type_id": row["type_id"],
                    "population_weight": 1.0,
                    "total_spending_share": float(row["target_value"]),
                }
                for _, row in targets.iterrows()
            ]
        )
        bad = actions.copy()
        bad["source"] = "bad"
        bad["total_spending_share"] = 0.0

        scores = score_cell_behavior_targets(pd.concat([actions, bad], ignore_index=True), targets)

        exact_all = scores[(scores["source"] == "exact") & (scores["target_family"] == "ALL")].iloc[0]
        bad_all = scores[(scores["source"] == "bad") & (scores["target_family"] == "ALL")].iloc[0]
        self.assertAlmostEqual(float(exact_all["rmse_range"]), 0.0)
        self.assertGreater(float(bad_all["rmse_range"]), 0.0)

    def test_household_behavior_payload_fails_closed_when_type_missing(self):
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        scenario = BEHAVIOR_SCENARIOS[0]
        payload = fixture_behavior_payload(scenario, type_cells)
        payload["household_actions"] = payload["household_actions"][:-1]

        with self.assertRaises(LLMUnavailable):
            normalize_behavior_payload(scenario, type_cells, {"payload": payload})

    def test_demand_economy_fixture_clears_dynamic_behavior_validation(self):
        with TemporaryDirectory() as temp_dir:
            households = build_fixture_demand_households(24)
            client = DemandEconomyClient("codex_cli", "gpt-5.5", Path(temp_dir), mode="fixture", max_live_calls=0)

            initial, beliefs, decisions, periods, accounting, prompts = run_demand_economy(
                households,
                default_demand_scenarios(),
                client,
                period_count=100,
                feedback_mode="closed_loop",
            )
            validation = score_demand_economy_validation(periods, decisions, beliefs, accounting)
            verdict = classify_demand_economy_evidence(validation, mode="fixture")
            metrics = validation.set_index("metric")["value"].to_dict()

            self.assertEqual(initial["type_id"].nunique(), 24)
            self.assertEqual(periods.groupby("scenario_id")["period_index"].nunique().min(), 100)
            self.assertEqual(decisions.groupby(["scenario_id", "period_index"])["type_id"].nunique().min(), 24)
            self.assertEqual(beliefs.groupby(["scenario_id", "period_index"])["type_id"].nunique().min(), 24)
            self.assertEqual(len(prompts), len(default_demand_scenarios()) * 100)
            self.assertEqual(verdict["evidence_verdict"], "hank_lite_metrics_pass_but_ablation_incomplete")
            self.assertTrue(validation.loc[validation["required"].astype(bool), "passed"].all())
            self.assertLessEqual(float(accounting["abs_residual"].max()), 1e-6)
            self.assertGreater(float(metrics["transfer_impact_mpc"]), 0.2)
            self.assertGreater(float(metrics["liquidity_mpc_gradient"]), 0.2)
            self.assertLess(float(metrics["rate_hike_mean_consumption_delta_6p"]), 0.0)
            self.assertGreater(float(metrics["belief_feedback_amplification_ratio"]), 1.1)

    def test_demand_economy_fixture_subsample_keeps_cross_sectional_cells(self):
        households = build_fixture_demand_households(6)

        self.assertEqual(households["type_id"].nunique(), 6)
        self.assertGreaterEqual(households["income_group"].nunique(), 3)
        self.assertGreaterEqual(households["liquidity_group"].nunique(), 2)
        self.assertAlmostEqual(float(households["population_weight"].sum()), 1.0)

    def test_demand_economy_prompt_is_date_free_and_relative_period_only(self):
        with TemporaryDirectory() as temp_dir:
            households = build_fixture_demand_households(3)
            client = DemandEconomyClient("codex_cli", "gpt-5.5", Path(temp_dir), mode="fixture", max_live_calls=0)

            _initial, _beliefs, _decisions, _periods, _accounting, prompts = run_demand_economy(
                households,
                [default_demand_scenarios()[0]],
                client,
                period_count=4,
                feedback_mode="closed_loop",
            )
        prompt_text = json.dumps(prompts[0]["prompt_payload"], sort_keys=True)

        self.assertIn("period_0", prompt_text)
        self.assertNotIn("survey_date", prompt_text)
        self.assertNotIn("2026-", prompt_text)
        self.assertNotIn("2008", prompt_text)
        self.assertNotIn("actual_", prompt_text)
        self.assertNotIn("job_risk_shock", prompt_text)
        self.assertNotIn("desired_consumption", prompt_text)
        self.assertNotIn("desired_saving", prompt_text)
        self.assertIn("survey-style anchors", prompt_text)
        self.assertIn("Do not collapse household cells", prompt_text)
        self.assertIn("5 means normal precaution", prompt_text)
        self.assertIn("recently absorbed liquid buffer improvement", prompt_text)
        self.assertIn("elevated macro-feedback regime", prompt_text)

        with TemporaryDirectory() as temp_dir:
            client = DemandEconomyClient("codex_cli", "gpt-5.5", Path(temp_dir), mode="fixture", max_live_calls=0)
            _initial, _beliefs, _decisions, _periods, _accounting, shock_prompts = run_demand_economy(
                households,
                [default_demand_scenarios()[1]],
                client,
                period_count=2,
                feedback_mode="closed_loop",
            )
        transfer_p0 = json.dumps(shock_prompts[0]["prompt_payload"], sort_keys=True)
        transfer_p1 = json.dumps(shock_prompts[1]["prompt_payload"], sort_keys=True)
        self.assertNotIn("transfer_shock", transfer_p0)
        self.assertNotIn("job_risk_shock", transfer_p0)
        self.assertNotIn("One-period lump-sum", transfer_p0)
        self.assertIn('"active_current_shocks": ["none"]', transfer_p0)
        self.assertIn("lump_sum_transfer_now", transfer_p1)

        with TemporaryDirectory() as temp_dir:
            client = DemandEconomyClient("codex_cli", "gpt-5.5", Path(temp_dir), mode="fixture", max_live_calls=0)
            _initial, _beliefs, _decisions, _periods, _accounting, feedback_prompts = run_demand_economy(
                households,
                [default_demand_scenarios()[4]],
                client,
                period_count=2,
                feedback_mode="closed_loop",
            )
        feedback_p0 = json.dumps(feedback_prompts[0]["prompt_payload"], sort_keys=True)
        self.assertNotIn("belief_feedback", feedback_p0)
        self.assertIn("elevated_belief_dispersion_regime_now", feedback_p0)
        self.assertIn("elevated_macro_feedback_regime_now", feedback_p0)

    def test_demand_economy_payload_fails_closed_when_type_missing(self):
        households = build_fixture_demand_households(4)
        initial = households.assign(
            source="fixture_gpt-5.5",
            labor_income=households["annual_income"] / 4.0,
            baseline_consumption=households["baseline_consumption_annual"] / 4.0,
            liquid_buffer_months=1.0,
            job_loss_probability=households["baseline_job_loss_probability"],
        ).to_dict(orient="records")
        scenario = default_demand_scenarios()[0]
        period_state = {
            "period_id": "period_0",
            "period_index": 0,
            "output_gap_pct": 0.0,
            "employment_rate": 0.955,
            "inflation_rate": 2.0,
            "policy_rate": 3.0,
            "transfer_per_household": 0.0,
            "policy_rate_shock_pp": 0.0,
            "job_risk_shock_pp": 0.0,
            "aggregate_job_loss_belief": 5.0,
            "aggregate_confidence_index": 50.0,
            "aggregate_liquid_buffer_months": 3.0,
        }
        payload = fixture_demand_payload(scenario, period_state, initial)
        payload["beliefs"] = payload["beliefs"][:-1]

        with self.assertRaises(LLMUnavailable):
            normalize_demand_payload(initial, {"payload": payload})

    def test_demand_economy_cli_fixture_writes_report(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_economy",
                    "--belief-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--models",
                    "gpt-5.5",
                    "--household-count",
                    "24",
                    "--period-count",
                    "100",
                    "--variants",
                    "representative,adaptive,llm_belief,naive_persona",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            report = (Path(temp_dir) / "demand_economy_report.md").read_text(encoding="utf-8")
            validation = pd.read_csv(Path(temp_dir) / "demand_validation_scores.csv")
            belief_targets = pd.read_csv(Path(temp_dir) / "demand_belief_target_scores.csv")
            ablations = pd.read_csv(Path(temp_dir) / "demand_ablation_table.csv")
            beliefs = pd.read_csv(Path(temp_dir) / "demand_beliefs.csv")
            prompts = (Path(temp_dir) / "demand_prompt_cards.jsonl").read_text(encoding="utf-8")
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["evidence"]["evidence_verdict"], "fixture_hank_lite_belief_lab_ready")
            self.assertTrue(manifest["evidence"]["full_lab_passed"])
            self.assertFalse(manifest["evidence"]["canary_passed"])
            self.assertEqual(manifest["verdict"], "fixture_hank_lite_belief_lab_ready")
            self.assertTrue(manifest["full_lab_passed"])
            self.assertIn("HANK-lite macro lab", report)
            self.assertIn("Baseline Comparison", report)
            self.assertIn("mechanical fixture seed echo", report)
            self.assertIn("Survey-Seed Belief Target Scores", report)
            self.assertTrue(validation.loc[validation["required"].astype(bool), "passed"].all())
            self.assertIn("ALL", set(belief_targets["belief_variable"]))
            self.assertEqual(set(ablations["variant"]), {"representative", "adaptive", "llm_belief", "naive_persona"})
            self.assertEqual(beliefs.groupby(["source", "scenario_id", "period_index"])["type_id"].nunique().min(), 24)
            self.assertIn("period_0", prompts)
            self.assertNotIn("2026-", prompts)
            self.assertNotIn("survey_date", prompts)
            self.assertNotIn("actual_", prompts)

    def test_demand_economy_live_mode_can_force_fixture_baseline_variants(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.demand_economy",
                    "--belief-mode",
                    "live",
                    "--max-live-calls",
                    "0",
                    "--models",
                    "gpt-5.5",
                    "--household-count",
                    "3",
                    "--period-count",
                    "8",
                    "--variants",
                    "naive_persona",
                    "--fixture-variants",
                    "naive_persona",
                    "--scenarios",
                    "baseline",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            beliefs = pd.read_csv(Path(temp_dir) / "demand_beliefs.csv")
            self.assertEqual(manifest["belief_mode"], "live")
            self.assertEqual(manifest["fixture_variants"], ["naive_persona"])
            self.assertEqual(manifest["live_call_count"], 0)
            self.assertEqual(set(beliefs["source"]), {"naive_persona_fixture_gpt-5.5"})
            self.assertEqual(manifest["verdict"], manifest["evidence"]["evidence_verdict"])

    def test_persona_belief_prompt_hides_heldout_responses(self):
        respondents = build_fixture_respondent_panel(respondent_count=6, survey_date="2026-01-01")
        card = build_persona_cards(respondents)[0]

        prompt = persona_belief_prompt(card, target_fields=["expected_inflation_1y"])

        self.assertIn("respondent_profile", prompt)
        self.assertIn("synthetic fixture respondent", prompt)
        self.assertNotIn("real survey respondent", prompt)
        self.assertIn("expected_inflation_1y", prompt)
        self.assertNotIn("actual_expected_inflation_1y", prompt)
        self.assertNotIn("actual_expected_unemployment_rate", prompt)
        self.assertNotIn("actual_expected_real_income_growth", prompt)
        self.assertNotIn(str(round(card.targets["expected_inflation_1y"], 4)), prompt)

    def test_persona_respondent_panel_respects_requested_count_and_scoreable_weights(self):
        fixture = build_fixture_respondent_panel(respondent_count=60, survey_date="2026-01-01")

        self.assertEqual(fixture.shape[0], 60)
        self.assertAlmostEqual(float(fixture["weight"].sum()), 1.0)

        raw = pd.DataFrame(
            [
                {
                    "respondent_id": "a",
                    "weight": 3,
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "high_school_or_less",
                    "gender": "female",
                    "actual_expected_inflation_1y": 4.2,
                },
                {
                    "respondent_id": "b",
                    "weight": 1,
                    "age_group": "55_plus",
                    "income_group": "high",
                    "education_group": "college_plus",
                    "gender": "male",
                    "actual_expected_inflation_1y": np.nan,
                },
            ]
        )
        normalized = normalize_respondent_panel(raw, target_fields=["expected_inflation_1y"])

        self.assertEqual(list(normalized["respondent_id"]), ["a"])
        self.assertAlmostEqual(float(normalized["weight"].sum()), 1.0)

        zero_retained = raw.copy()
        zero_retained.loc[zero_retained["respondent_id"] == "a", "weight"] = 0
        normalized_zero = normalize_respondent_panel(zero_retained, target_fields=["expected_inflation_1y"])

        self.assertEqual(list(normalized_zero["respondent_id"]), ["a"])
        self.assertAlmostEqual(float(normalized_zero["weight"].sum()), 1.0)

    def test_persona_regression_gradient_controls_for_confounded_demographics(self):
        respondents = pd.DataFrame(
            [
                {
                    "respondent_id": "low_hs",
                    "weight": 49,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "high_school_or_less",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "renter",
                    "liquid_wealth_group": "low",
                    "actual_expected_inflation_1y": 10.0,
                },
                {
                    "respondent_id": "low_college",
                    "weight": 1,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "college_plus",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "renter",
                    "liquid_wealth_group": "low",
                    "actual_expected_inflation_1y": 0.0,
                },
                {
                    "respondent_id": "high_hs",
                    "weight": 1,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "high",
                    "education_group": "high_school_or_less",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "owner",
                    "liquid_wealth_group": "high",
                    "actual_expected_inflation_1y": 18.0,
                },
                {
                    "respondent_id": "high_college",
                    "weight": 49,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "high",
                    "education_group": "college_plus",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "owner",
                    "liquid_wealth_group": "high",
                    "actual_expected_inflation_1y": 8.0,
                },
            ]
        )
        respondents["weight"] = respondents["weight"] / respondents["weight"].sum()
        predictions = pd.DataFrame(
            [
                {
                    "respondent_id": row["respondent_id"],
                    "source": "sim",
                    "target_name": "expected_inflation_1y",
                    "prediction": row["actual_expected_inflation_1y"],
                }
                for _, row in respondents.iterrows()
            ]
        )

        contrasts = score_gradient_match(respondents, predictions, target_fields=["expected_inflation_1y"])
        regressions = score_regression_gradient_match(respondents, predictions, target_fields=["expected_inflation_1y"])

        income_contrast = contrasts[contrasts["dimension"] == "income_group"].iloc[0]
        income_regression = regressions[regressions["dimension"] == "income_group"].iloc[0]
        self.assertGreater(float(income_contrast["survey_gradient"]), 0.0)
        self.assertLess(float(income_regression["survey_coefficient"]), 0.0)
        self.assertTrue(bool(income_regression["sign_match"]))

    def test_persona_distribution_ks_uses_survey_weights(self):
        respondents = pd.DataFrame(
            [
                {
                    "respondent_id": "mass",
                    "weight": 0.98,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "college_plus",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "renter",
                    "liquid_wealth_group": "low",
                    "actual_expected_inflation_1y": 0.0,
                },
                {
                    "respondent_id": "tail_a",
                    "weight": 0.01,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "college_plus",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "renter",
                    "liquid_wealth_group": "low",
                    "actual_expected_inflation_1y": 100.0,
                },
                {
                    "respondent_id": "tail_b",
                    "weight": 0.01,
                    "survey_source": "unit",
                    "survey_date": "2026-01-01",
                    "age_group": "18_34",
                    "income_group": "low",
                    "education_group": "college_plus",
                    "gender": "male",
                    "region": "x",
                    "employment_status": "employed",
                    "homeownership": "renter",
                    "liquid_wealth_group": "low",
                    "actual_expected_inflation_1y": 100.0,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {"respondent_id": "mass", "source": "sim", "target_name": "expected_inflation_1y", "prediction": 0.0},
                {"respondent_id": "tail_a", "source": "sim", "target_name": "expected_inflation_1y", "prediction": 0.0},
                {"respondent_id": "tail_b", "source": "sim", "target_name": "expected_inflation_1y", "prediction": 100.0},
            ]
        )

        distances = score_distribution_distance(respondents, predictions, target_fields=["expected_inflation_1y"])

        self.assertLess(float(distances.iloc[0]["ks_stat"]), 0.02)

    def test_persona_common_core_must_be_measured_to_clear_gate(self):
        respondents = build_fixture_respondent_panel(respondent_count=12, survey_date="2026-01-01")
        cards = build_persona_cards(respondents)
        predictions = run_persona_beliefs(
            cards,
            PersonaBeliefClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0),
            target_fields=["expected_inflation_1y"],
        )
        common_core = score_common_core(predictions, target_fields=["expected_inflation_1y"])
        verdict = classify_persona_evidence(
            pd.DataFrame({"sign_match": [True, True, True, True]}),
            pd.DataFrame({"within_variance_ratio": [1.0, 1.1]}),
            common_core,
        )

        self.assertEqual(int(common_core.iloc[0]["pair_count"]), 0)
        self.assertFalse(bool(common_core.iloc[0]["common_core_tested"]))
        self.assertEqual(verdict["evidence_verdict"], "incomplete_common_core_test")

    def test_persona_live_call_cap_is_global_across_models(self):
        self.assertEqual(_client_mode_and_cap("live", 10, 0), ("live", 10))
        self.assertEqual(_client_mode_and_cap("live", 10, 7), ("live", 3))
        self.assertEqual(_client_mode_and_cap("live", 10, 10), ("replay", 0))
        self.assertEqual(_client_mode_and_cap("fixture", 0, 0), ("fixture", 0))

    def test_persona_run_command_metadata_hides_local_module_path(self):
        old_argv = sys.argv
        try:
            sys.argv = [str(Path(persona_belief_panel_module.__file__).resolve()), "--belief-mode", "fixture"]

            argv = _sanitized_argv()

            self.assertEqual(argv[:3], ["python3", "-m", "macro_llm_tournament.persona_belief_panel"])
            self.assertNotIn(str(Path.home()), " ".join(argv))
        finally:
            sys.argv = old_argv

    def test_persona_belief_panel_fixture_scores_distributional_structure(self):
        respondents = build_fixture_respondent_panel(respondent_count=54, survey_date="2026-01-01")
        cards = build_persona_cards(respondents)
        predictions = pd.concat(
            [
                run_persona_beliefs(
                    cards,
                    PersonaBeliefClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0),
                    target_fields=["expected_inflation_1y", "expected_unemployment_rate", "expected_real_income_growth"],
                ),
                run_persona_beliefs(
                    cards,
                    PersonaBeliefClient("codex_cli", "gpt-5.4", Path("/tmp/unused"), mode="fixture", max_live_calls=0),
                    target_fields=["expected_inflation_1y", "expected_unemployment_rate", "expected_real_income_growth"],
                ),
            ],
            ignore_index=True,
        )

        regressions = score_regression_gradient_match(respondents, predictions)
        gradients = score_gradient_match(respondents, predictions)
        variances = score_variance_flattening(respondents, predictions)
        distances = score_distribution_distance(respondents, predictions)
        common_core = score_common_core(predictions)
        verdict = classify_persona_evidence(regressions, variances, common_core, distances)

        self.assertEqual(predictions["respondent_id"].nunique(), respondents.shape[0])
        self.assertEqual(predictions["source"].nunique(), 2)
        self.assertEqual(set(predictions["call_source"].unique()), {"fixture"})
        self.assertFalse(predictions["cache_hit"].any())
        self.assertFalse(regressions.empty)
        self.assertFalse(gradients.empty)
        self.assertFalse(variances.empty)
        self.assertFalse(distances.empty)
        self.assertFalse(common_core.empty)
        inflation_income = gradients[
            (gradients["target_name"] == "expected_inflation_1y")
            & (gradients["dimension"] == "income_group")
        ]
        self.assertTrue(inflation_income["sign_match"].all())
        self.assertGreater(float(inflation_income["survey_gradient"].mean()), 0.0)
        self.assertGreater(float(inflation_income["simulated_gradient"].mean()), 0.0)
        self.assertTrue(variances["within_variance_ratio"].map(np.isfinite).all())
        self.assertTrue(distances["wasserstein_1"].ge(0).all())
        self.assertTrue(common_core["pair_count"].ge(1).all())
        self.assertEqual(verdict["evidence_verdict"], "partial_flattening_and_common_core_failure")
        self.assertIn("distribution_clear", verdict)

    def test_persona_variance_score_detects_flattened_within_group_spread(self):
        respondents = build_fixture_respondent_panel(respondent_count=54, survey_date="2026-01-01")
        rows = []
        for _, respondent in respondents.iterrows():
            group = respondents[respondents["income_group"] == respondent["income_group"]]
            group_mean = float(group["actual_expected_inflation_1y"].mean())
            rows.append(
                {
                    "respondent_id": respondent["respondent_id"],
                    "source": "flat_income_cell",
                    "target_name": "expected_inflation_1y",
                    "prediction": group_mean,
                }
            )

        variances = score_variance_flattening(
            respondents,
            pd.DataFrame(rows),
            target_fields=["expected_inflation_1y"],
        )

        income_row = variances[variances["dimension"] == "income_group"].iloc[0]
        self.assertLess(float(income_row["within_variance_ratio"]), 0.05)
        self.assertTrue(bool(income_row["flattening_flag"]))

    def test_persona_ecology_prompt_hides_current_targets_and_exposes_modules(self):
        panel = build_fixture_ecology_panel(respondent_count=6, period_count=2, survey_start="2026-01")
        card = build_ecology_cards(panel)[0]
        prompt = persona_ecology_prompt(
            card,
            target_fields=["expected_inflation_1y"],
            prior_beliefs={"expected_inflation_1y": card.empirical_priors["expected_inflation_1y"]},
            environment=card.environment,
        )

        self.assertIn("profile_module", prompt)
        self.assertIn("prior_expectations_module", prompt)
        self.assertIn("external_information_module", prompt)
        self.assertIn("behavior_module", prompt)
        self.assertIn("prior_beliefs", prompt)
        self.assertIn("current_environment", prompt)
        self.assertNotIn("actual_expected_inflation_1y", prompt)
        self.assertNotIn(str(round(card.targets["expected_inflation_1y"], 4)), prompt)

    def test_persona_ecology_relative_date_mode_hides_calendar_and_maps_display_id(self):
        panel = build_fixture_ecology_panel(respondent_count=3, period_count=2, survey_start="2026-01")
        base = build_ecology_cards(panel, target_fields=["expected_inflation_1y"])[0]
        card = EcologyCard(
            panel_row_id=f"{base.respondent_id}__2024:Q4",
            respondent_id=base.respondent_id,
            period_id="2024:Q4",
            period_index=0,
            survey_source=base.survey_source,
            survey_date="2024-11-15",
            weight=base.weight,
            profile=base.profile,
            empirical_priors=base.empirical_priors,
            environment=base.environment,
            targets=base.targets,
        )
        prompt = persona_ecology_prompt(
            card,
            target_fields=["expected_inflation_1y"],
            prior_beliefs={"expected_inflation_1y": card.empirical_priors["expected_inflation_1y"]},
            environment=card.environment,
            date_mode="relative",
        )
        display_panel_row_id = f"{card.respondent_id}__period_0"
        payload = {
            "panel_row_id": display_panel_row_id,
            "respondent_id": card.respondent_id,
            "beliefs": {
                "expected_inflation_1y": {
                    "value": 3.1,
                    "p10": 2.0,
                    "p50": 3.1,
                    "p90": 4.2,
                }
            },
            "actions": {
                "consumption_change_pct": 0.0,
                "liquid_buffer_change_pct": 1.0,
                "borrowing_desire_index": -0.5,
                "portfolio_rebalance_to_liquid_pct": 0.0,
                "job_search_intensity_index": 0.0,
            },
            "module_weights": {
                "profile_weight": 0.2,
                "prior_weight": 0.5,
                "environment_weight": 0.25,
                "aggregate_feedback_weight": 0.05,
            },
            "confidence": 0.6,
            "uncertainty": 0.4,
            "reason": "relative-period response",
        }

        normalized = normalize_persona_ecology_payload(
            card,
            {"payload": payload},
            provider="codex_cli",
            model="gpt-5.5",
            target_fields=["expected_inflation_1y"],
            date_mode="relative",
        )

        self.assertIn("period_0", prompt)
        self.assertIn(display_panel_row_id, prompt)
        self.assertNotIn("2024:Q4", prompt)
        self.assertNotIn("2024-11-15", prompt)
        self.assertNotIn(card.panel_row_id, prompt)
        self.assertEqual(normalized["panel_row_id"], card.panel_row_id)
        self.assertEqual(normalized["respondent_id"], card.respondent_id)

    def test_persona_ecology_payload_fails_closed_when_required_behavior_fields_missing(self):
        panel = build_fixture_ecology_panel(respondent_count=3, period_count=2, survey_start="2026-01")
        card = build_ecology_cards(panel, target_fields=["expected_inflation_1y"])[0]
        valid_payload = {
            "panel_row_id": card.panel_row_id,
            "respondent_id": card.respondent_id,
            "beliefs": {
                "expected_inflation_1y": {
                    "value": 3.1,
                    "p10": 2.0,
                    "p50": 3.1,
                    "p90": 4.2,
                }
            },
            "actions": {
                "consumption_change_pct": 0.0,
                "liquid_buffer_change_pct": 1.0,
                "borrowing_desire_index": -0.5,
                "portfolio_rebalance_to_liquid_pct": 0.0,
                "job_search_intensity_index": 0.0,
            },
            "module_weights": {
                "profile_weight": 0.2,
                "prior_weight": 0.5,
                "environment_weight": 0.25,
                "aggregate_feedback_weight": 0.05,
            },
            "confidence": 0.6,
            "uncertainty": 0.4,
            "reason": "schema-complete response",
        }
        missing_action = json.loads(json.dumps(valid_payload))
        del missing_action["actions"]["consumption_change_pct"]
        missing_module_weights = json.loads(json.dumps(valid_payload))
        del missing_module_weights["module_weights"]
        missing_confidence = json.loads(json.dumps(valid_payload))
        del missing_confidence["confidence"]
        missing_quantile = json.loads(json.dumps(valid_payload))
        del missing_quantile["beliefs"]["expected_inflation_1y"]["p10"]

        for payload in (missing_action, missing_module_weights, missing_confidence, missing_quantile):
            with self.assertRaises(LLMUnavailable):
                normalize_persona_ecology_payload(
                    card,
                    {"payload": payload},
                    provider="codex_cli",
                    model="gpt-5.5",
                    target_fields=["expected_inflation_1y"],
                )

    def test_persona_ecology_relative_date_mode_hides_survey_source_calendar(self):
        panel = build_fixture_ecology_panel(respondent_count=3, period_count=2, survey_start="2026-01")
        base = build_ecology_cards(panel, target_fields=["expected_inflation_1y"])[0]
        card = EcologyCard(
            panel_row_id=f"{base.respondent_id}__2024:Q4",
            respondent_id=base.respondent_id,
            period_id="2024:Q4",
            period_index=0,
            survey_source="sce_2024_q4_public_panel",
            survey_date="2024-11-15",
            weight=base.weight,
            profile=base.profile,
            empirical_priors=base.empirical_priors,
            environment=base.environment,
            targets=base.targets,
        )

        prompt = persona_ecology_prompt(
            card,
            target_fields=["expected_inflation_1y"],
            prior_beliefs={"expected_inflation_1y": card.empirical_priors["expected_inflation_1y"]},
            environment=card.environment,
            date_mode="relative",
        )

        self.assertIn('"survey_source": "survey_panel"', prompt)
        self.assertNotIn("sce_2024", prompt)
        self.assertNotIn("2024", prompt)
        self.assertNotIn("q4", prompt.lower())

    def test_persona_ecology_normalizes_sce_style_aliases_and_lagged_priors(self):
        raw = pd.DataFrame(
            [
                {
                    "userid": "u1",
                    "yyyymm": "2026-01",
                    "weight_final": 2.0,
                    "age": 30,
                    "hhinc": 40000,
                    "educ": 12,
                    "female": 1,
                    "q9_mean": 4.0,
                    "unemp_mean": 6.0,
                    "earnings_mean": 1.0,
                },
                {
                    "userid": "u1",
                    "yyyymm": "2026-02",
                    "weight_final": 2.0,
                    "age": 30,
                    "hhinc": 40000,
                    "educ": 12,
                    "female": 1,
                    "q9_mean": 5.5,
                    "unemp_mean": 6.5,
                    "earnings_mean": 0.5,
                },
                {
                    "userid": "u2",
                    "yyyymm": "2026-01",
                    "weight_final": 1.0,
                    "age": 62,
                    "hhinc": 180000,
                    "educ": 18,
                    "female": 0,
                    "q9_mean": 2.0,
                    "unemp_mean": 4.0,
                    "earnings_mean": 2.0,
                },
                {
                    "userid": "u2",
                    "yyyymm": "2026-02",
                    "weight_final": 1.0,
                    "age": 62,
                    "hhinc": 180000,
                    "educ": 18,
                    "female": 0,
                    "q9_mean": 2.5,
                    "unemp_mean": 4.5,
                    "earnings_mean": 1.5,
                },
            ]
        )

        panel = normalize_ecology_panel(raw, survey_schema="sce")
        second_u1 = panel[(panel["respondent_id"] == "u1") & (panel["period_index"] == 1)].iloc[0]

        self.assertEqual(panel.shape[0], 4)
        self.assertAlmostEqual(float(panel[panel["period_index"] == 0]["weight"].sum()), 1.0)
        self.assertEqual(second_u1["age_group"], "18_34")
        self.assertEqual(second_u1["education_group"], "high_school_or_less")
        self.assertAlmostEqual(float(second_u1["prior_expected_inflation_1y"]), 4.0)
        self.assertAlmostEqual(float(second_u1["actual_expected_inflation_1y"]), 5.5)

    def test_persona_ecology_fixture_runs_dynamic_feedback_and_scores(self):
        panel = build_fixture_ecology_panel(respondent_count=12, period_count=3, survey_start="2026-01")
        cards = build_ecology_cards(panel)
        client = PersonaEcologyClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0)

        predictions, actions, environments, prompts = run_persona_ecology(
            cards,
            client,
            target_fields=["expected_inflation_1y", "expected_unemployment_rate", "expected_real_income_growth"],
            prior_mode="simulated",
            feedback_mode="closed_loop",
        )
        temporal = score_temporal_dynamics(panel, predictions)
        behavior = score_behavior_actions(panel, actions)
        ablations = build_module_ablations(panel)
        ablation_scores = score_module_ablations(panel, ablations)
        scoring_panel = panel.copy()
        scoring_panel["respondent_id"] = scoring_panel["panel_row_id"]
        scoring_predictions = predictions.copy()
        scoring_predictions["respondent_id"] = scoring_predictions["panel_row_id"]
        regression = score_regression_gradient_match(scoring_panel, scoring_predictions)
        variance = score_variance_flattening(scoring_panel, scoring_predictions)
        common_core = score_common_core(scoring_predictions)
        distance = score_distribution_distance(scoring_panel, scoring_predictions)
        static = classify_persona_evidence(regression, variance, common_core, distance)
        verdict = classify_ecology_evidence(
            static,
            temporal,
            behavior,
            environments,
            {"period_count": 3, "feedback_mode": "closed_loop", "respondent_source": "fixture"},
        )

        self.assertEqual(predictions.shape[0], panel.shape[0] * 3)
        self.assertEqual(actions.shape[0], panel.shape[0])
        self.assertEqual(environments.shape[0], 3)
        self.assertEqual(len(prompts), panel.shape[0])
        self.assertFalse(temporal.empty)
        self.assertFalse(behavior.empty)
        self.assertFalse(ablation_scores.empty)
        self.assertIn("ablation_prior_only", set(ablations["source"]))
        self.assertTrue(environments["aggregate_demand_pressure"].abs().gt(0).any())
        self.assertEqual(verdict["evidence_verdict"], "fixture_ecology_harness_ready")

    def test_persona_ecology_one_target_closed_loop_keeps_prompt_environment_finite(self):
        panel = build_fixture_ecology_panel(
            respondent_count=4,
            period_count=3,
            survey_start="2026-01",
            target_fields=["expected_inflation_1y"],
        )
        cards = build_ecology_cards(panel, target_fields=["expected_inflation_1y"])
        client = PersonaEcologyClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0)

        predictions, actions, environments, prompts = run_persona_ecology(
            cards,
            client,
            target_fields=["expected_inflation_1y"],
            prior_mode="simulated",
            feedback_mode="closed_loop",
            date_mode="relative",
        )
        prompt_environments = [row["prompt_payload"]["current_environment"] for row in prompts]
        period_scores = score_period_levels(panel, predictions, target_fields=["expected_inflation_1y"])
        update_scores = score_period_updates(panel, predictions, target_fields=["expected_inflation_1y"])
        action_summary = summarize_period_actions(actions)

        self.assertFalse(predictions.empty)
        self.assertFalse(environments.empty)
        self.assertFalse(period_scores.empty)
        self.assertFalse(update_scores.empty)
        self.assertFalse(action_summary.empty)
        for environment in prompt_environments:
            self.assertFalse(any(value is None for value in environment.values()))
        self.assertNotIn("aggregate_expected_unemployment_rate", environments.columns)
        self.assertIn("mean_seen_aggregate_expected_unemployment_rate", environments.columns)

    def test_persona_ecology_simulated_prior_feeds_previous_model_belief(self):
        panel = build_fixture_ecology_panel(respondent_count=3, period_count=2, survey_start="2026-01")
        cards = build_ecology_cards(panel, target_fields=["expected_inflation_1y"])
        client = PersonaEcologyClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0)

        predictions, actions, _environments, _prompts = run_persona_ecology(
            cards,
            client,
            target_fields=["expected_inflation_1y"],
            prior_mode="simulated",
            feedback_mode="none",
        )

        first = predictions[
            (predictions["respondent_id"] == "fixture_resp_001") & (predictions["period_index"] == 0)
        ].iloc[0]
        second = predictions[
            (predictions["respondent_id"] == "fixture_resp_001") & (predictions["period_index"] == 1)
        ].iloc[0]
        empirical_second = panel[
            (panel["respondent_id"] == "fixture_resp_001") & (panel["period_index"] == 1)
        ].iloc[0]

        self.assertAlmostEqual(float(second["prior_prediction"]), float(first["prediction"]))
        self.assertNotAlmostEqual(float(second["prior_prediction"]), float(empirical_second["prior_expected_inflation_1y"]))
        self.assertEqual(actions["source"].nunique(), 1)

    def test_persona_ecology_cli_fixture_writes_report(self):
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.persona_ecology",
                    "--ecology-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--models",
                    "gpt-5.5,gpt-5.4",
                    "--respondent-source",
                    "fixture",
                    "--respondent-count",
                    "8",
                    "--period-count",
                    "2",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            manifest_text = (Path(temp_dir) / "manifest.json").read_text(encoding="utf-8")
            report = (Path(temp_dir) / "persona_ecology_report.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["ecology_evidence"]["evidence_verdict"], "fixture_ecology_harness_ready")
            self.assertIn("profile, priors, environment", report)
            self.assertIn("persona_ecology_period_scores.csv", manifest["outputs"])
            self.assertIn("persona_ecology_update_scores.csv", manifest["outputs"])
            self.assertIn("persona_ecology_action_period_summary.csv", manifest["outputs"])
            self.assertNotIn("NaN", manifest_text)
            self.assertTrue((Path(temp_dir) / "persona_ecology_period_scores.csv").exists())
            self.assertTrue((Path(temp_dir) / "persona_ecology_update_scores.csv").exists())
            self.assertTrue((Path(temp_dir) / "persona_ecology_action_period_summary.csv").exists())

    def test_persona_live_modes_require_fresh_cache_before_calls(self):
        with TemporaryDirectory() as temp_dir:
            belief = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.persona_belief_panel",
                    "--belief-mode",
                    "live",
                    "--max-live-calls",
                    "1",
                    "--models",
                    "gpt-5.5",
                    "--respondent-source",
                    "fixture",
                    "--respondent-count",
                    "1",
                    "--target-fields",
                    "expected_inflation_1y",
                    "--output-dir",
                    str(Path(temp_dir) / "belief"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )
            ecology = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.persona_ecology",
                    "--ecology-mode",
                    "live",
                    "--max-live-calls",
                    "1",
                    "--models",
                    "gpt-5.5",
                    "--respondent-source",
                    "fixture",
                    "--respondent-count",
                    "1",
                    "--period-count",
                    "1",
                    "--target-fields",
                    "expected_inflation_1y",
                    "--output-dir",
                    str(Path(temp_dir) / "ecology"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(belief.returncode, 0)
            self.assertNotEqual(ecology.returncode, 0)
            self.assertIn("--fresh-cache is required", belief.stderr)
            self.assertIn("--fresh-cache is required", ecology.stderr)

    def test_persona_ecology_csv_fixture_anonymizes_and_records_input_manifest(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "holdout.csv"
            pd.DataFrame(
                [
                    {
                        "respondent_id": "real_person_1",
                        "period_id": "2026-01",
                        "weight": 2.0,
                        "age_group": "35_54",
                        "income_group": "middle",
                        "education_group": "college_plus",
                        "gender": "female",
                        "region": "northeast",
                        "employment_status": "employed",
                        "homeownership": "owner",
                        "liquid_wealth_group": "middle",
                        "actual_expected_inflation_1y": 3.2,
                        "actual_consumption_change_pct": 0.4,
                    },
                    {
                        "respondent_id": "real_person_1",
                        "period_id": "2026-02",
                        "weight": 2.0,
                        "age_group": "35_54",
                        "income_group": "middle",
                        "education_group": "college_plus",
                        "gender": "female",
                        "region": "northeast",
                        "employment_status": "employed",
                        "homeownership": "owner",
                        "liquid_wealth_group": "middle",
                        "actual_expected_inflation_1y": 3.7,
                        "actual_consumption_change_pct": 0.2,
                    },
                    {
                        "respondent_id": "real_person_2",
                        "period_id": "2026-01",
                        "weight": 1.0,
                        "age_group": "55_plus",
                        "income_group": "high",
                        "education_group": "college_plus",
                        "gender": "male",
                        "region": "west",
                        "employment_status": "retired",
                        "homeownership": "owner",
                        "liquid_wealth_group": "high",
                        "actual_expected_inflation_1y": 2.4,
                        "actual_consumption_change_pct": -0.1,
                    },
                    {
                        "respondent_id": "real_person_2",
                        "period_id": "2026-02",
                        "weight": 1.0,
                        "age_group": "55_plus",
                        "income_group": "high",
                        "education_group": "college_plus",
                        "gender": "male",
                        "region": "west",
                        "employment_status": "retired",
                        "homeownership": "owner",
                        "liquid_wealth_group": "high",
                        "actual_expected_inflation_1y": 2.6,
                        "actual_consumption_change_pct": -0.2,
                    },
                ]
            ).to_csv(csv_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.persona_ecology",
                    "--ecology-mode",
                    "fixture",
                    "--max-live-calls",
                    "0",
                    "--models",
                    "gpt-5.5",
                    "--respondent-source",
                    "csv",
                    "--survey-schema",
                    "normalized",
                    "--respondent-csv",
                    str(csv_path),
                    "--respondent-limit",
                    "1",
                    "--target-fields",
                    "expected_inflation_1y",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(temp_dir) / "manifest.json").read_text(encoding="utf-8"))
            prompt_cards = (Path(temp_dir) / "persona_ecology_prompt_cards.jsonl").read_text(encoding="utf-8")
            panel = pd.read_csv(Path(temp_dir) / "persona_ecology_panel.csv")
            self.assertEqual(manifest["respondent_input"]["source"], "csv")
            self.assertEqual(manifest["respondent_input"]["raw_row_count"], 4)
            self.assertEqual(manifest["respondent_input"]["normalized_respondent_count"], 1)
            self.assertTrue(manifest["respondent_input"]["respondent_ids_anonymized"])
            self.assertEqual(manifest["behavior_target_source"], "external_csv_targets")
            self.assertNotIn("real_person", prompt_cards)
            self.assertEqual(set(panel["respondent_id"]), {"respondent_00001"})

    def test_prepare_persona_holdouts_uses_vintage_environment_and_marks_synthetic(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            survey_path = temp_path / "survey.csv"
            origins_path = temp_path / "origins.csv"
            vintage_path = temp_path / "vintage.csv"
            output_dir = temp_path / "out"
            pd.DataFrame(
                [
                    {
                        "survey_source": "ny_fed_sce",
                        "target_name": "median_expected_inflation",
                        "date": "2024-11-01",
                        "horizon_months": 12,
                        "value": 3.4,
                        "units": "percent",
                        "source_url": "local",
                    },
                    {
                        "survey_source": "ny_fed_sce",
                        "target_name": "p25_expected_inflation",
                        "date": "2024-11-01",
                        "horizon_months": 12,
                        "value": 2.1,
                        "units": "percent",
                        "source_url": "local",
                    },
                    {
                        "survey_source": "ny_fed_sce",
                        "target_name": "p75_expected_inflation",
                        "date": "2024-11-01",
                        "horizon_months": 12,
                        "value": 5.2,
                        "units": "percent",
                        "source_url": "local",
                    },
                    {
                        "survey_source": "ny_fed_sce",
                        "target_name": "median_inflation_uncertainty",
                        "date": "2024-11-01",
                        "horizon_months": 12,
                        "value": 2.2,
                        "units": "percent",
                        "source_url": "local",
                    },
                ]
            ).to_csv(survey_path, index=False)
            pd.DataFrame(
                [
                    {"origin": "2024:Q4", "as_of_date": "2024-11-15", "split": "test"},
                    {"origin": "2025:Q1", "as_of_date": "2025-02-15", "split": "test"},
                ]
            ).to_csv(origins_path, index=False)
            rows = []
            for origin, as_of, latest_cpi, latest_gdp in [
                ("2024:Q4", "2024-11-15", 309.0, 23000.0),
                ("2025:Q1", "2025-02-15", 311.0, 23100.0),
            ]:
                for date, cpi, gdp in [
                    ("2023-11-01", 300.0, 22500.0),
                    ("2024-11-01", latest_cpi, latest_gdp),
                ]:
                    rows.append(
                        {
                            "origin": origin,
                            "as_of_date": as_of,
                            "series_id": "CPIAUCSL",
                            "label": "CPI level",
                            "observation_date": date,
                            "value": cpi,
                            "realtime_start": as_of,
                            "realtime_end": as_of,
                        }
                    )
                    rows.append(
                        {
                            "origin": origin,
                            "as_of_date": as_of,
                            "series_id": "GDPC1",
                            "label": "Real GDP",
                            "observation_date": date,
                            "value": gdp,
                            "realtime_start": as_of,
                            "realtime_end": as_of,
                        }
                    )
                rows.extend(
                    [
                        {
                            "origin": origin,
                            "as_of_date": as_of,
                            "series_id": "UNRATE",
                            "label": "Unemployment",
                            "observation_date": "2024-11-01",
                            "value": 4.2,
                            "realtime_start": as_of,
                            "realtime_end": as_of,
                        },
                        {
                            "origin": origin,
                            "as_of_date": as_of,
                            "series_id": "FEDFUNDS",
                            "label": "Fed funds",
                            "observation_date": "2024-11-01",
                            "value": 4.8,
                            "realtime_start": as_of,
                            "realtime_end": as_of,
                        },
                    ]
                )
            pd.DataFrame(rows).to_csv(vintage_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.prepare_persona_holdouts",
                    "--survey-beliefs",
                    str(survey_path),
                    "--forecast-origins",
                    str(origins_path),
                    "--fred-vintage-context",
                    str(vintage_path),
                    "--output-dir",
                    str(output_dir),
                    "--respondent-count",
                    "4",
                    "--period-count",
                    "2",
                    "--start-as-of",
                    "2024-10-01",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            static = pd.read_csv(output_dir / "sce_micro_holdout.csv")
            panel = pd.read_csv(output_dir / "sce_panel_holdout.csv")
            manifest = json.loads((output_dir / "persona_holdout_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(static.shape[0], 4)
            self.assertEqual(panel.shape[0], 8)
            self.assertEqual(manifest["panel_kind"], "synthetic_enriched_sce_vintage_v1")
            self.assertTrue(panel["target_provenance"].str.contains("synthetic").all())
            self.assertIn("observed_inflation_1y", panel)
            self.assertEqual(panel["period_id"].nunique(), 2)

    def test_agent_economy_same_origin_cards_share_prior_state(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI", "RGDP", "TBILL"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2019,
            card_count=6,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]

        _state, desired, _feasible, _aggregates, _diagnostics, _scores = run_agent_economy(
            cards,
            llm_forecasts,
            type_cells,
            source_filters=["llm"],
        )

        first_origin = desired["origin"].iloc[0]
        same_origin = desired[
            (desired["origin"] == first_origin)
            & (desired["type_id"] == "liquid_poor_renter")
        ]
        self.assertGreaterEqual(same_origin["variable"].nunique(), 2)
        self.assertEqual(same_origin["prior_liquid_assets"].nunique(), 1)
        self.assertEqual(same_origin["prior_debt"].nunique(), 1)

    def test_agent_prompt_does_not_expose_realized_forecast_error_state(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        prior_states = [
            {
                "type_id": str(row["type_id"]),
                "expected_inflation_1y": 2.5,
                "expected_real_income_growth": 1.5,
                "expected_unemployment_rate": 4.5,
                "expected_short_rate": 3.0,
                "confidence": 0.5,
                "uncertainty": 0.5,
                "desired_liquid_buffer_months": 2.0,
                "credit_access": "normal",
                "recent_forecast_error": 999.0,
            }
            for _, row in type_cells.iterrows()
        ]

        prompt = agent_prompt(cards[0], llm_forecasts.iloc[0], type_cells, prior_states)

        self.assertNotIn("recent_forecast_error", prompt)
        self.assertNotIn("999", prompt)

    def test_agent_payload_requires_firm_and_bank_sections(self):
        card = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2018,
            card_count=1,
        )[0]
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, [card])
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        actions = [
            {
                "type_id": str(row["type_id"]),
                "consumption_change_pct": 0.0,
                "liquid_buffer_change_pct": 0.0,
                "borrowing_desire_index": 0.0,
                "portfolio_rebalance_to_liquid_pct": 0.0,
                "job_search_intensity_index": 0.0,
                "expected_inflation_1y": 2.5,
                "expected_real_income_growth": 1.5,
                "expected_unemployment_rate": 4.5,
                "expected_short_rate": 3.0,
                "confidence": 0.5,
                "uncertainty": 0.5,
            }
            for _, row in type_cells.iterrows()
        ]

        with self.assertRaises(LLMUnavailable):
            normalize_agent_payload(card, llm_forecasts.iloc[0], type_cells, {"payload": {"household_actions": actions}})

    def test_agent_live_cap_blocks_uncached_second_packed_agent_call(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=2,
        )
        llm_client = ForecastLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture")
        llm_forecasts, _raw = run_llm_forecasts(llm_client, cards)
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        prior_states = [
            {
                "type_id": str(row["type_id"]),
                "expected_inflation_1y": 2.5,
                "expected_real_income_growth": 1.5,
                "expected_unemployment_rate": 4.5,
                "expected_short_rate": 3.0,
                "confidence": 0.5,
                "uncertainty": 0.5,
                "desired_liquid_buffer_months": 2.0,
                "credit_access": "normal",
                "recent_forecast_error": 0.0,
                "liquid_assets": float(row["liquid_assets"]),
                "debt": float(row["debt"]),
            }
            for _, row in type_cells.iterrows()
        ]

        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"CODEX_CLI_BIN": "/tmp/codex"}):
            agent_client = AgentLLMClient("codex_cli", "gpt-5.5", Path(tmp), mode="live", max_live_calls=1)

            def fake_run(command, input, text, capture_output, cwd, timeout, check):
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text(json.dumps(fixture_agent_payload(cards[0], llm_forecasts.iloc[0], type_cells, prior_states)))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("macro_llm_tournament.agent_llm.subprocess.run", side_effect=fake_run):
                agent_client.agent_panel(cards[0], llm_forecasts.iloc[0], type_cells, prior_states)
                with self.assertRaises(LLMUnavailable):
                    agent_client.agent_panel(cards[1], llm_forecasts.iloc[1], type_cells, prior_states)

        self.assertEqual(agent_client.live_call_count, 1)

    def test_forecast_live_cap_blocks_uncached_second_call(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=2,
        )
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"CODEX_CLI_BIN": "/tmp/codex"}):
            client = ForecastLLMClient("codex_cli", "gpt-5.5", Path(tmp), mode="live", max_live_calls=1)

            def fake_run(command, input, text, capture_output, cwd, timeout, check):
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "point_forecast": 2.0,
                            "p10": 1.0,
                            "p50": 2.0,
                            "p90": 3.0,
                            "confidence": 0.5,
                            "forecaster_draws": [
                                {"forecaster_id": f"x{idx}", "forecast": 2.0 + idx * 0.01}
                                for idx in range(8)
                            ],
                        }
                    )
                )
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch("macro_llm_tournament.forecast_llm.subprocess.run", side_effect=fake_run):
                client.forecast_card(cards[0])
                with self.assertRaises(LLMUnavailable):
                    client.forecast_card(cards[1])

        self.assertEqual(client.live_call_count, 1)

    def test_openai_responses_forecast_provider_uses_cache_and_live_cap(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=2,
        )

        class FakeResponse:
            id = "resp_test"
            status = "completed"
            model = "gpt-5"
            output_text = json.dumps(
                {
                    "point_forecast": 2.0,
                    "p10": 1.0,
                    "p50": 2.0,
                    "p90": 3.0,
                    "confidence": 0.5,
                    "forecaster_draws": [
                        {"forecaster_id": f"x{idx}", "forecast": 2.0 + idx * 0.01}
                        for idx in range(8)
                    ],
                }
            )
            usage = {"total_tokens": 123}

        class FakeResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                self.last_kwargs = kwargs
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = fake_responses

        fake_responses = FakeResponses()
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
            client = ForecastLLMClient("openai_responses", "gpt-5", Path(tmp), mode="live", max_live_calls=1)
            with patch("macro_llm_tournament.forecast_llm.OpenAI", FakeOpenAI):
                first = client.forecast_card(cards[0])
                cached = client.forecast_card(cards[0])
                with self.assertRaises(LLMUnavailable):
                    client.forecast_card(cards[1])

        self.assertEqual(client.live_call_count, 1)
        self.assertEqual(client.cache_hit_count, 1)
        self.assertEqual(fake_responses.calls, 1)
        self.assertEqual(first["payload"]["point_forecast"], 2.0)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(fake_responses.last_kwargs["model"], "gpt-5")
        self.assertEqual(fake_responses.last_kwargs["reasoning"], {"effort": "low"})

    def test_openai_responses_retries_malformed_json_without_caching_bad_payload(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=1,
        )

        class FakeResponse:
            id = "resp_retry"
            status = "completed"
            model = "gpt-5"
            usage = {"total_tokens": 123}

            def __init__(self, output_text):
                self.output_text = output_text

        class FakeResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse('{"point_forecast": 2.0')
                return FakeResponse(
                    json.dumps(
                        {
                            "point_forecast": 2.0,
                            "p10": 1.0,
                            "p50": 2.0,
                            "p90": 3.0,
                            "confidence": 0.5,
                            "forecaster_draws": [
                                {"forecaster_id": f"x{idx}", "forecast": 2.0 + idx * 0.01}
                                for idx in range(8)
                            ],
                        }
                    )
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = fake_responses

        fake_responses = FakeResponses()
        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-test", "LLM_JSON_ATTEMPTS": "2"},
            clear=False,
        ):
            client = ForecastLLMClient("openai_responses", "gpt-5", Path(tmp), mode="live", max_live_calls=2)
            with patch("macro_llm_tournament.forecast_llm.OpenAI", FakeOpenAI):
                first = client.forecast_card(cards[0])
                cached = client.forecast_card(cards[0])

            cache_files = list((Path(tmp) / "openai_responses").glob("*.json"))

        self.assertEqual(fake_responses.calls, 2)
        self.assertEqual(client.live_call_count, 2)
        self.assertEqual(client.cache_hit_count, 1)
        self.assertEqual(len(cache_files), 1)
        self.assertEqual(first["payload"]["point_forecast"], 2.0)
        self.assertEqual(first["json_attempt"], 2)
        self.assertTrue(cached["cache_hit"])

    def test_forecast_json_parser_rejects_prose_wrapped_payloads(self):
        self.assertEqual(_extract_json('{"point_forecast": 2.0}'), {"point_forecast": 2.0})
        self.assertEqual(_extract_json('```json\n{"point_forecast": 2.0}\n```'), {"point_forecast": 2.0})
        with self.assertRaises(ValueError):
            _extract_json('Here is the JSON: {"point_forecast": 2.0}')
        with self.assertRaises(ValueError):
            _extract_json('{"point_forecast": 2.0}\nDone.')

    def test_openai_responses_direct_recall_provider_uses_shared_router(self):
        prompt = json.dumps(
            {
                "prompt_version": "direct_realization_recall_probe_v1",
                "items": [{"card_id": "card_a", "forecast_target_period": "2026:Q1", "spf_step_ahead": 1}],
            }
        )
        payload = {
            "items": [
                {
                    "card_id": "card_a",
                    "recalled_realized": None,
                    "confidence": 0.0,
                    "reason": "not recalled",
                }
            ]
        }

        class FakeResponse:
            id = "resp_recall"
            status = "completed"
            model = "gpt-5"
            output_text = json.dumps(payload)
            usage = {"total_tokens": 42}

        class FakeResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                self.last_kwargs = kwargs
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = fake_responses

        fake_responses = FakeResponses()
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
            client = DirectRecallClient("openai_responses", "gpt-5", Path(tmp), mode="live", max_live_calls=1)
            cache_name = direct_recall_cache_name(client, prompt)
            with patch("macro_llm_tournament.forecast_llm.OpenAI", FakeOpenAI):
                first, client = direct_recall_call(client, prompt, cache_name)
                cached, client = direct_recall_call(client, prompt, cache_name)

        self.assertEqual(client.live_call_count, 1)
        self.assertEqual(client.cache_hit_count, 1)
        self.assertEqual(fake_responses.calls, 1)
        self.assertEqual(first["payload"]["items"][0]["card_id"], "card_a")
        self.assertTrue(cached["cache_hit"])
        self.assertIn("direct recall contamination probe", fake_responses.last_kwargs["instructions"])

    def test_gemini_cli_forecast_provider_parses_wrapper_and_uses_api_key_auth(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=2,
        )
        response_payload = {
            "point_forecast": 2.0,
            "p10": 1.0,
            "p50": 2.0,
            "p90": 3.0,
            "confidence": 0.5,
            "forecaster_draws": [
                {"forecaster_id": f"x{idx}", "forecast": 2.0 + idx * 0.01}
                for idx in range(8)
            ],
        }
        wrapper = {
            "session_id": "gemini_session",
            "response": json.dumps(response_payload),
            "stats": {"models": {"gemini-3.1-pro-preview": {"api": {"totalRequests": 1}}}},
        }
        calls = []

        def fake_run(command, input, text, capture_output, cwd, timeout, check, env):
            calls.append({"command": command, "input": input, "env": env})
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(wrapper), stderr="")

        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "gemini-test-key", "GEMINI_CLI_BIN": "/tmp/gemini"},
            clear=False,
        ):
            client = ForecastLLMClient("gemini_cli", "gemini-3.1-pro-preview", Path(tmp), mode="live", max_live_calls=1)
            with patch("macro_llm_tournament.forecast_llm.subprocess.run", side_effect=fake_run):
                first = client.forecast_card(cards[0])
                cached = client.forecast_card(cards[0])
                with self.assertRaises(LLMUnavailable):
                    client.forecast_card(cards[1])

        self.assertEqual(client.live_call_count, 1)
        self.assertEqual(client.cache_hit_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["payload"]["point_forecast"], 2.0)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(calls[0]["command"][calls[0]["command"].index("--model") + 1], "gemini-3.1-pro-preview")
        self.assertEqual(calls[0]["env"]["GEMINI_DEFAULT_AUTH_TYPE"], "gemini-api-key")
        self.assertTrue(calls[0]["env"]["GEMINI_CLI_SYSTEM_SETTINGS_PATH"].endswith("system_settings.json"))
        self.assertIn("Return only valid JSON", calls[0]["input"])

    def test_antigravity_cli_forecast_provider_uses_agy_and_retries_malformed_json(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=1,
        )
        response_payload = {
            "point_forecast": 2.0,
            "p10": 1.0,
            "p50": 2.0,
            "p90": 3.0,
            "confidence": 0.5,
            "forecaster_draws": [
                {"forecaster_id": f"x{idx}", "forecast": 2.0 + idx * 0.01}
                for idx in range(8)
            ],
        }
        calls = []

        def fake_run(command, text, capture_output, cwd, timeout, check, env):
            calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "settings_model": json.loads(settings_path.read_text())["model"],
                }
            )
            log_path = Path(command[command.index("--log-file") + 1])
            log_path.write_text(
                'I0000 model.go: Propagating selected model override to backend: label="Gemini 3.1 Pro (High)"',
                encoding="utf-8",
            )
            if len(calls) == 1:
                return subprocess.CompletedProcess(args=command, returncode=0, stdout='{"point_forecast": 2.0', stderr="")
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(response_payload), stderr="")

        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "ANTIGRAVITY_CLI_BIN": "/tmp/agy",
                "LLM_JSON_ATTEMPTS": "2",
            },
            clear=False,
        ):
            execution_cwd = Path(tmp) / "isolated"
            settings_path = Path(tmp) / "antigravity" / "settings.json"
            lock_path = Path(tmp) / "antigravity" / "lock"
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps({"model": "Gemini 3.5 Flash (Medium)"}), encoding="utf-8")
            client = ForecastLLMClient(
                "antigravity_cli",
                "gemini-3.1-pro-preview",
                Path(tmp) / "cache",
                mode="live",
                max_live_calls=2,
                execution_cwd=execution_cwd,
            )
            with (
                patch("macro_llm_tournament.forecast_llm.ANTIGRAVITY_SETTINGS_PATH", settings_path),
                patch("macro_llm_tournament.forecast_llm.ANTIGRAVITY_SETTINGS_LOCK_PATH", lock_path),
                patch("macro_llm_tournament.forecast_llm.subprocess.run", side_effect=fake_run),
            ):
                first = client.forecast_card(cards[0])
                cached = client.forecast_card(cards[0])

            cache_files = list((Path(tmp) / "cache" / "antigravity_cli").glob("*.json"))
            restored_settings = json.loads(settings_path.read_text())

        self.assertEqual(client.live_call_count, 2)
        self.assertEqual(client.cache_hit_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cache_files), 1)
        self.assertEqual(first["payload"]["point_forecast"], 2.0)
        self.assertEqual(first["json_attempt"], 2)
        self.assertEqual(first["provider_model_arg"], "gemini-3.1-pro")
        self.assertEqual(first["provider_actual_label"], "Gemini 3.1 Pro (High)")
        self.assertTrue(cached["cache_hit"])
        self.assertNotIn("--model", calls[0]["command"])
        self.assertEqual(calls[0]["settings_model"], "Gemini 3.1 Pro (High)")
        self.assertEqual(restored_settings["model"], "Gemini 3.5 Flash (Medium)")
        self.assertIn("--dangerously-skip-permissions", calls[0]["command"])
        self.assertEqual(calls[0]["command"][calls[0]["command"].index("--sandbox") + 1], "read-only")
        self.assertEqual(calls[0]["cwd"], str(execution_cwd.resolve()))
        self.assertIn("Return only valid JSON", calls[0]["command"][calls[0]["command"].index("--print") + 1])

    def test_spf_download_link_extractor_preserves_xlsx_urls(self):
        page_url = "https://www.philadelphiafed.org/surveys-and-data/data-files/cpi"
        html = """
        <a href="/surveys-and-data/real-time-data-research/survey-of-professional-forecasters/data-files/Individual_CPI.xlsx?sc_lang=en&amp;hash=ABC">individual</a>
        <a href="/surveys-and-data/real-time-data-research/survey-of-professional-forecasters/data-files/Mean_CPI_Level.xls">mean</a>
        <a href="/assets/images/preview.xlsx">asset</a>
        <a href="/surveys-and-data/other/data-files/private.xlsx">other</a>
        """

        links = extract_spf_download_links(page_url, html)

        self.assertEqual(len(links), 2)
        self.assertTrue(any("Individual_CPI.xlsx?" in link for link in links))
        self.assertTrue(any(link.endswith("Mean_CPI_Level.xls") for link in links))
        self.assertFalse(any("assets/images" in link for link in links))

    def test_scf_download_link_extractor_keeps_curated_modern_files(self):
        page_url = "https://www.federalreserve.gov/econres/scf_2019.htm"
        html = """
        <a href="/econres/files/scfp2019excel.zip">summary excel</a>
        <a href="/econres/files/scfp2019.zip">summary sas</a>
        <a href="/econres/files/2019map.xlsx">map</a>
        <a href="/econres/files/codebk2019.txt">codebook</a>
        <a href="/econres/files/scf2019rw1.zip">replicate weights</a>
        <a href="/econres/files/unrelated.xlsx">unrelated</a>
        """

        links = extract_scf_download_links(page_url, html)

        self.assertEqual(len(links), 4)
        self.assertTrue(any(link.endswith("scfp2019excel.zip") for link in links))
        self.assertTrue(any(link.endswith("2019map.xlsx") for link in links))
        self.assertFalse(any("rw1" in link for link in links))

    def test_download_summary_tracks_skipped_without_counting_as_file(self):
        summary = summarize(
            [
                DownloadResult("fred", "a", "https://example.test/a", "work/a.csv", "downloaded", bytes=10),
                DownloadResult("fred", "b", "https://example.test/b", "work/b.csv", "skipped"),
                DownloadResult("spf", "c", "https://example.test/c", "work/c.csv", "error", error="boom"),
            ]
        )

        self.assertEqual(summary["files"], 1)
        self.assertEqual(summary["bytes"], 10)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["groups"]["fred"], {"files": 1, "bytes": 10, "errors": 0, "skipped": 1})
        self.assertEqual(summary["groups"]["spf"], {"files": 0, "bytes": 0, "errors": 1, "skipped": 0})

    def test_simple_xlsx_reader_handles_shared_strings_and_numeric_cells(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.xlsx"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "xl/sharedStrings.xml",
                    """<?xml version="1.0" encoding="UTF-8"?>
                    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                      <si><t>YEAR</t></si><si><t>QUARTER</t></si><si><t>CPI2</t></si>
                    </sst>""",
                )
                archive.writestr(
                    "xl/worksheets/sheet1.xml",
                    """<?xml version="1.0" encoding="UTF-8"?>
                    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                      <sheetData>
                        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
                        <row r="2"><c r="A2"><v>2026</v></c><c r="B2"><v>1</v></c><c r="C2"><v>2.5</v></c></row>
                      </sheetData>
                    </worksheet>""",
                )

            frame = read_simple_xlsx(path)

        self.assertEqual(frame.loc[0, "YEAR"], 2026.0)
        self.assertEqual(frame.loc[0, "QUARTER"], 1.0)
        self.assertEqual(frame.loc[0, "CPI2"], 2.5)

    def test_postcutoff_realization_requires_complete_monthly_quarter(self):
        series = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"]),
                "value": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        rows = pd.DataFrame(realize_variable("CPI", DETAIL_FORECAST_SPECS["CPI"], series))

        q1 = rows[rows["origin"] == "2026:Q1"].iloc[0]
        self.assertFalse(q1["realization_complete"])
        self.assertTrue(np.isnan(q1["fred_realized"]))
        self.assertAlmostEqual(annualized_pct_change(104.0, 100.0), 16.985856, places=6)

    def test_postcutoff_combination_prefers_detail_forecast_over_blank_official_placeholder(self):
        columns = [
            "variable",
            "variable_name",
            "units",
            "origin",
            "origin_year",
            "origin_quarter",
            "origin_index",
            "horizon",
            "spf_forecast",
            "official_iar_forecast",
            "official_no_change_forecast",
            "official_dar_forecast",
            "official_darm_forecast",
            "realized",
            "source_url",
            "variable_page_url",
        ]
        official = pd.DataFrame(
            [
                ["RGDP", "real GDP growth", "annualized percentage points", "2026:Q1", 2026, 1, 8105, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "official", "page"],
            ],
            columns=columns,
        )
        detail = pd.DataFrame(
            [
                ["RGDP", "real GDP growth", "annualized percentage points", "2026:Q1", 2026, 1, 8105, 1, 2.538, np.nan, np.nan, np.nan, np.nan, 1.621, "detail", "page"],
            ],
            columns=columns,
        )

        combined = combine_official_and_detail_rows(official, detail)

        self.assertEqual(combined.shape[0], 1)
        self.assertAlmostEqual(combined.iloc[0]["spf_forecast"], 2.538)
        self.assertEqual(combined.iloc[0]["source_url"], "detail")

    def test_postcutoff_replay_cache_freeze_helpers_keep_zero_live_calls(self):
        cards = build_forecast_cards(
            spf_fixture(),
            variables=["CPI"],
            horizons=[1],
            holdout_start_year=2018,
            holdout_end_year=2020,
            card_count=2,
        )
        with TemporaryDirectory() as tmp:
            client = ForecastLLMClient("codex_cli", "gpt-5.5", Path(tmp), mode="replay", max_live_calls=0)
            cached_path = client.cache_path(client.forecast_cache_name(cards[0]))
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.write_text("{}", encoding="utf-8")

            cached, missing = split_cards_by_replay_cache(cards, client)

        self.assertEqual(len(cached), 1)
        self.assertEqual(len(missing), 1)
        self.assertEqual(client.live_call_count, 0)

    def test_postcutoff_cards_use_exact_cutoff_contamination_label(self):
        spf_data = spf_fixture()
        selected = spf_data[
            (spf_data["variable"] == "CPI")
            & (spf_data["origin"] == "2020:Q1")
            & (spf_data["horizon"] == 1)
        ].copy()
        selected["contamination_label"] = "post_model_cutoff_candidate"

        cards = build_forecast_cards_from_rows(
            spf_data,
            selected,
            variables=["CPI"],
            holdout_start_year=2020,
            holdout_end_year=2020,
            history_quarters=8,
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].contamination_label, "post_model_cutoff_candidate")
        self.assertEqual(cards[0].prompt_payload["as_of_design"]["contamination_label"], "post_model_cutoff_candidate")

    def test_newly_scoreable_count_uses_previous_freeze_rows(self):
        selected = pd.DataFrame(
            [
                {"variable": "RGDP", "origin_index": 8105, "horizon": 1, "scoreable": True},
                {"variable": "CPI", "origin_index": 8105, "horizon": 1, "scoreable": False},
                {"variable": "UNEMP", "origin_index": 8105, "horizon": 1, "scoreable": True},
            ]
        )
        with TemporaryDirectory() as tmp:
            previous = Path(tmp)
            pd.DataFrame(
                [
                    {"variable": "RGDP", "origin_index": 8105, "horizon": 1},
                    {"variable": "CPI", "origin_index": 8105, "horizon": 1},
                ]
            ).to_csv(previous / "postcutoff_freeze_rows.csv", index=False)

            count = count_newly_scoreable_rows(selected, str(previous))

        self.assertEqual(count, 1)

    def test_postcutoff_behavior_proxy_cards_hide_target_dates_and_values(self):
        frames = fixture_fred_proxy_frames()
        cards, targets, context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date="2025-12-01",
            asof_start="2025-12-15",
            asof_end="2026-01-15",
            history_months=12,
            scoreable_only=False,
        )
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        prompt = postcutoff_behavior_prompt(cards[0], type_cells)

        self.assertEqual(len(cards), 2)
        self.assertFalse(targets.empty)
        self.assertFalse(context.empty)
        self.assertTrue(targets["target_available"].all())
        self.assertEqual(set(targets["contamination_label"]), {"post_model_cutoff_clean"})
        self.assertNotIn(cards[0].target_month, prompt)
        self.assertNotIn(cards[0].as_of_date, prompt)
        self.assertNotIn("target_value", prompt)
        self.assertIn("target_month_hidden", prompt)

    def test_postcutoff_behavior_proxy_fixture_run_scores_agents_and_controls(self):
        frames = fixture_fred_proxy_frames()
        cards, targets, _context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date="2025-12-01",
            asof_start="2025-12-15",
            asof_end="2026-02-15",
            history_months=12,
            scoreable_only=True,
        )
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        client = PostcutoffBehaviorLLMClient("codex_cli", "gpt-5.5", Path("/tmp/unused"), mode="fixture", max_live_calls=0)

        actions = run_postcutoff_behavior_agents(cards, type_cells, llm_client=client)
        llm_forecasts = aggregate_agent_proxy_forecasts(cards, actions, source="llm_codex_cli_gpt-5.5")
        controls = build_proxy_control_forecasts(cards)
        scores, joined = score_proxy_forecasts(pd.concat([llm_forecasts, controls], ignore_index=True), targets)

        self.assertEqual(len(client.raw_records), len(cards))
        self.assertEqual(actions.groupby("card_id")["type_id"].nunique().min(), type_cells.shape[0])
        self.assertEqual(llm_forecasts.groupby("card_id")["target_name"].nunique().min(), len(TARGET_SPECS))
        self.assertFalse(joined.empty)
        self.assertFalse(scores.empty)
        self.assertIn("ALL", set(scores["target_name"]))
        self.assertTrue(scores["rmse_scaled"].map(np.isfinite).all())

    def test_postcutoff_behavior_proxy_exact_forecasts_score_zero(self):
        frames = fixture_fred_proxy_frames()
        cards, targets, _context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date="2025-12-01",
            asof_start="2025-12-15",
            asof_end="2025-12-15",
            history_months=12,
            scoreable_only=True,
        )
        exact = targets[["card_id", "period_id", "target_name"]].copy()
        exact["source"] = "exact"
        exact["prediction"] = targets["target_value"].to_numpy()
        exact["method"] = "oracle_fixture"

        scores, joined = score_proxy_forecasts(exact, targets)

        self.assertEqual(len(cards), 1)
        self.assertFalse(joined.empty)
        overall = scores[(scores["source"] == "exact") & (scores["target_name"] == "ALL")].iloc[0]
        self.assertAlmostEqual(float(overall["rmse_scaled"]), 0.0)

    def test_postcutoff_behavior_payload_fails_closed_when_type_missing(self):
        frames = fixture_fred_proxy_frames()
        cards, _targets, _context = build_postcutoff_behavior_cards(
            frames,
            cutoff_date="2025-12-01",
            asof_start="2025-12-15",
            asof_end="2025-12-15",
            history_months=12,
            scoreable_only=True,
        )
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        payload = {
            "payload": {
                "household_actions": [
                    {
                        "type_id": str(row["type_id"]),
                        "consumption_change_pct": 0.0,
                        "liquid_saving_change_pct": 0.0,
                        "debt_balance_change_pct": 0.0,
                        "confidence": 0.5,
                        **{spec.target_name: 0.0 for spec in TARGET_SPECS},
                    }
                    for _, row in type_cells.iloc[:-1].iterrows()
                ]
            }
        }

        with self.assertRaises(LLMUnavailable):
            normalize_postcutoff_behavior_payload(cards[0], type_cells, payload)


if __name__ == "__main__":
    unittest.main()
