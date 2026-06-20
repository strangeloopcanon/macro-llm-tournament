import json
import subprocess
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
    behavior_targets_frame,
    fixture_behavior_payload,
    normalize_behavior_payload,
    run_behavior_controls,
    run_behavior_gate,
    score_behavior_targets,
)
from macro_llm_tournament.forecast_audit import (
    build_source_scores,
    build_surprise_audit,
    build_theil_u,
    direct_recall_prompt,
    fixture_recall_payload,
    normalize_direct_recall_payload,
    score_direct_recall,
)
from macro_llm_tournament.forecast_cards import (
    assert_no_prompt_target_leakage,
    build_forecast_cards,
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
from macro_llm_tournament.forecast_llm import ForecastLLMClient, normalize_forecast_payload, run_llm_forecasts
from macro_llm_tournament.forecast_scoring import score_forecast_slices, score_forecasts, verdict_from_scores
from macro_llm_tournament.fred_vintage import approximate_spf_as_of_date, build_vintage_context_for_cards
from macro_llm_tournament.llm_common import LLMUnavailable
from macro_llm_tournament.postcutoff_tournament import (
    DETAIL_FORECAST_SPECS,
    annualized_pct_change,
    combine_official_and_detail_rows,
    read_simple_xlsx,
    realize_variable,
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
        all_actions = pd.concat([actions, controls], ignore_index=True)
        aggregates = aggregate_behavior_actions(all_actions)
        scores = score_behavior_targets(aggregates, behavior_targets_frame())

        self.assertEqual(actions.shape[0], len(BEHAVIOR_SCENARIOS) * type_cells.shape[0])
        self.assertEqual(len(client.raw_records), len(BEHAVIOR_SCENARIOS))
        self.assertEqual(all_actions.groupby(["scenario_id", "source"])["type_id"].nunique().min(), type_cells.shape[0])
        use_sum = all_actions["total_spending_share"] + all_actions["debt_repayment_share"] + all_actions["liquid_saving_share"]
        self.assertLessEqual(float(use_sum.max()), 1.000001)
        self.assertFalse(scores.empty)
        self.assertIn("debt_saving", set(scores["target_family"]))
        self.assertIn("liquidity_gradient", set(scores["target_family"]))

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

    def test_household_behavior_payload_fails_closed_when_type_missing(self):
        type_cells = build_household_type_cells(work_dir=Path("/tmp/missing_scf"), wave=2022)[0]
        scenario = BEHAVIOR_SCENARIOS[0]
        payload = fixture_behavior_payload(scenario, type_cells)
        payload["household_actions"] = payload["household_actions"][:-1]

        with self.assertRaises(LLMUnavailable):
            normalize_behavior_payload(scenario, type_cells, {"payload": payload})

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
