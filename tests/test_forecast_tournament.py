import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from macro_llm_tournament.forecast_agent_panel import build_forecast_agent_panel
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


if __name__ == "__main__":
    unittest.main()
