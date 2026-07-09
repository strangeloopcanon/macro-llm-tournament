from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.combine_prior_update_panels import combine_prior_update_panels
from macro_llm_tournament.prepare_prior_update_extension import (
    ENVIRONMENT_COLUMNS,
    ENVIRONMENT_PROVENANCE_V2,
    build_december_v2_input,
    build_january_v2_input,
    build_monthly_environment,
    monthly_as_of_date,
)


def _synthetic_vintage() -> pd.DataFrame:
    rows = []
    months = pd.date_range("2023-11-01", "2024-11-01", freq="MS")
    for index, month in enumerate(months):
        rows.append({"series_id": "CPIAUCSL", "observation_date": month, "value": 310.0 + 0.5 * index})
        rows.append({"series_id": "UNRATE", "observation_date": month, "value": 4.2})
        rows.append({"series_id": "FEDFUNDS", "observation_date": month, "value": 4.6})
        rows.append({"series_id": "PCECC96", "observation_date": month, "value": 15800.0 + 20.0 * index})
        rows.append({"series_id": "MICH", "observation_date": month, "value": 2.9})
        rows.append({"series_id": "UMCSENT", "observation_date": month, "value": 74.0})
    for quarter in pd.date_range("2023-10-01", "2024-07-01", freq="QS"):
        rows.append({"series_id": "GDPC1", "observation_date": quarter, "value": 22500.0})
    return pd.DataFrame(rows)


def _base_extension() -> pd.DataFrame:
    rows = []
    for index in range(3):
        rows.append(
            {
                "respondent_id": f"respondent_{index:05d}",
                "survey_source": "ny_fed_sce_microdata",
                "survey_date": "2024-12-01",
                "weight": 1.0 / 3.0,
                "age_group": "35_54",
                "income_group": "middle",
                "education_group": "college_plus",
                "gender": "female",
                "region": "midwest",
                "employment_status": "employed",
                "homeownership": "owner",
                "liquid_wealth_group": "middle",
                "actual_expected_inflation_1y": 3.0 + index,
                "actual_expected_unemployment_higher_prob": 30.0 + index,
                "actual_expected_real_income_growth": 1.0 + index,
                "sce_nominal_income_growth": 3.0,
                "sce_question_unemployment_higher_prob": 30.0 + index,
                "sce_raw_userid": 1000 + index,
                "sce_raw_date": 202412,
                "period_id": "sce_2024_12",
                "period_index": 2,
                "panel_row_id": f"respondent_{index:05d}__sce_2024_12",
                "persona_panel_kind": "real_sce_microdata_v1",
                "target_provenance": "public_ny_fed_sce_microdata_responses",
                "environment_provenance": "fred_alfred_vintage_context_by_spf_origin",
                "observed_inflation_1y": 2.58,
                "observed_unemployment_rate": 4.1,
                "observed_real_income_growth": 2.66,
                "policy_rate": 4.83,
                "sentiment_index": 94.0,
                "news_inflation_pressure": 0.03,
                "news_labor_pressure": -0.16,
                "credit_tightness": 0.62,
                "aggregate_expected_inflation_1y": 2.97,
                "aggregate_expected_unemployment_rate": 4.0,
                "aggregate_expected_real_income_growth": 1.06,
                "prior_expected_inflation_1y": 2.5 + index,
                "prior_expected_unemployment_higher_prob": 28.0 + index,
                "prior_expected_real_income_growth": 0.5 + index,
            }
        )
    return pd.DataFrame(rows)


def _microdata_january(userids: list[int]) -> pd.DataFrame:
    rows = []
    for index, userid in enumerate(userids):
        rows.append(
            {
                "respondent_id": f"other_id_{index:05d}",
                "survey_source": "ny_fed_sce_microdata",
                "survey_date": "2025-01-01",
                "weight": 0.5,
                "age_group": "35_54",
                "income_group": "middle",
                "education_group": "college_plus",
                "gender": "female",
                "region": "midwest",
                "employment_status": "employed",
                "homeownership": "owner",
                "liquid_wealth_group": "middle",
                "actual_expected_inflation_1y": 3.4 + index,
                "actual_expected_unemployment_higher_prob": 33.0,
                "actual_expected_real_income_growth": 0.8,
                "sce_nominal_income_growth": 3.1,
                "sce_question_unemployment_higher_prob": 33.0,
                "sce_raw_userid": userid,
                "sce_raw_date": 202501,
            }
        )
    return pd.DataFrame(rows)


class MonthlyEnvironmentTests(unittest.TestCase):
    def test_monthly_as_of_is_mid_survey_month(self) -> None:
        self.assertEqual(monthly_as_of_date("2024-12-01"), "2024-12-15")
        self.assertEqual(monthly_as_of_date("2025-01-01"), "2025-01-15")

    def test_environment_uses_real_vintage_sentiment_and_michigan_expectations(self) -> None:
        environment = build_monthly_environment(_synthetic_vintage(), as_of_date="2024-12-15")
        self.assertEqual(environment["sentiment_index"], 74.0)
        self.assertEqual(environment["aggregate_expected_inflation_1y"], 2.9)
        self.assertEqual(environment["environment_provenance"], ENVIRONMENT_PROVENANCE_V2)
        self.assertAlmostEqual(environment["observed_unemployment_rate"], 4.2)
        expected_yoy = 100.0 * (316.0 / 310.0 - 1.0)
        self.assertAlmostEqual(environment["observed_inflation_1y"], round(expected_yoy, 4))

    def test_environment_fails_closed_without_michigan_series(self) -> None:
        vintage = _synthetic_vintage()
        vintage = vintage[vintage["series_id"] != "MICH"]
        with self.assertRaises(ValueError):
            build_monthly_environment(vintage, as_of_date="2024-12-15")


class ExtensionInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = build_monthly_environment(_synthetic_vintage(), as_of_date="2024-12-15")

    def test_december_v2_replaces_environment_and_keeps_priors(self) -> None:
        base = _base_extension()
        out = build_december_v2_input(base, self.environment)
        self.assertEqual(list(out.columns), list(base.columns))
        for column in ENVIRONMENT_COLUMNS:
            self.assertTrue(out[column].eq(self.environment[column]).all(), column)
        self.assertTrue(out["environment_provenance"].eq(ENVIRONMENT_PROVENANCE_V2).all())
        pd.testing.assert_series_equal(out["prior_expected_inflation_1y"], base["prior_expected_inflation_1y"])
        pd.testing.assert_series_equal(out["actual_expected_inflation_1y"], base["actual_expected_inflation_1y"])

    def test_january_v2_uses_december_actuals_as_priors_and_keeps_ids(self) -> None:
        base = _base_extension()
        microdata = _microdata_january([1000, 1002, 9999])
        out = build_january_v2_input(base, microdata, self.environment)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(list(out.columns), list(base.columns))
        self.assertEqual(set(out["respondent_id"]), {"respondent_00000", "respondent_00002"})
        self.assertTrue(out["period_id"].eq("sce_2025_01").all())
        self.assertTrue(out["period_index"].eq(3).all())
        row = out[out["respondent_id"].eq("respondent_00000")].iloc[0]
        self.assertAlmostEqual(float(row["prior_expected_inflation_1y"]), 3.0)
        self.assertAlmostEqual(float(row["prior_expected_unemployment_higher_prob"]), 30.0)
        self.assertAlmostEqual(float(row["actual_expected_inflation_1y"]), 3.4)
        self.assertAlmostEqual(float(out["weight"].sum()), 1.0)

    def test_january_v2_fails_closed_with_no_continuing_respondents(self) -> None:
        base = _base_extension()
        microdata = _microdata_january([7777])
        with self.assertRaises(ValueError):
            build_january_v2_input(base, microdata, self.environment)


class CombinePriorUpdatePanelsTests(unittest.TestCase):
    def test_combiner_filters_to_extension_respondents_and_primary_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank_dir = root / "bank"
            extension_dir = root / "extension"
            output_dir = root / "combined"
            _write_run(
                bank_dir,
                respondents=["respondent_a", "respondent_b", "respondent_c"],
                periods=[("sce_2024_10", 0), ("sce_2024_11", 1)],
                include_alt_source=True,
            )
            _write_run(
                extension_dir,
                respondents=["respondent_a", "respondent_b"],
                periods=[("sce_2024_12", 2)],
                include_alt_source=True,
            )

            result = combine_prior_update_panels(
                bank_dir=bank_dir,
                extension_dirs=[extension_dir],
                primary_source="llm_codex_cli_gpt-5.5",
                output_dir=output_dir,
                note="test note",
            )

            panel = pd.read_csv(output_dir / "persona_ecology_panel.csv")
            predictions = pd.read_csv(output_dir / "persona_ecology_predictions.csv")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(result["respondent_count"], 2)
            self.assertEqual(result["period_count"], 3)
            self.assertEqual(panel.shape[0], 6)
            self.assertEqual(predictions.shape[0], 12)
            self.assertEqual(set(panel["respondent_id"]), {"respondent_a", "respondent_b"})
            self.assertEqual(set(predictions["source"]), {"llm_codex_cli_gpt-5.5"})
            self.assertEqual(manifest["prior_update_evidence"], {"note": "test note"})
            self.assertIn("source_artifacts", manifest)


def _write_run(
    root: Path,
    *,
    respondents: list[str],
    periods: list[tuple[str, int]],
    include_alt_source: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "provider": "codex_cli",
                "prior_mode": "empirical",
                "feedback_mode": "none",
                "date_mode": "relative",
                "prior_update_evidence": {"evidence_verdict": "test"},
            }
        ),
        encoding="utf-8",
    )
    panel_rows = []
    prediction_rows = []
    sources = ["llm_codex_cli_gpt-5.5"]
    if include_alt_source:
        sources.append("llm_codex_cli_gpt-5.4")
    for period_id, period_index in periods:
        for respondent in respondents:
            panel_row_id = f"{respondent}__{period_id}"
            panel_rows.append(
                {
                    "panel_row_id": panel_row_id,
                    "respondent_id": respondent,
                    "period_id": period_id,
                    "period_index": period_index,
                    "weight": 1.0 / len(respondents),
                }
            )
            for source in sources:
                for target_name in ["expected_inflation_1y", "expected_real_income_growth"]:
                    prediction_rows.append(
                        {
                            "panel_row_id": panel_row_id,
                            "respondent_id": respondent,
                            "period_id": period_id,
                            "period_index": period_index,
                            "source": source,
                            "target_name": target_name,
                            "prediction": 1.0,
                        }
                    )
    pd.DataFrame(panel_rows).to_csv(root / "persona_ecology_panel.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(root / "persona_ecology_predictions.csv", index=False)


if __name__ == "__main__":
    unittest.main()
