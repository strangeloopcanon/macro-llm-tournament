from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from macro_llm_tournament.persistent_households import (
    PersistentHouseholdError,
    append_selected_observed_history,
    build_household_scale_cohorts,
    derived_initial_seed,
    eligible_april_2025_with_march_prior,
    prepare_household_scale_cohorts,
    stable_household_id,
)
from macro_llm_tournament.prepare_dynamic_macro_panel import _stratified_wave_sample


def _fixture(*, include_future: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    months = ["2025-03-01", "2025-04-01"]
    if include_future:
        months.append("2025-05-01")
    for index in range(12):
        income = "low" if index < 6 else "high"
        for month_index, month in enumerate(months):
            rows.append(
                {
                    "respondent_id": f"raw-{index:03d}",
                    "survey_date": month,
                    "weight": float(index + 1),
                    "income_group": income,
                    "age_group": "18_34" if index % 2 else "55_plus",
                    "education_group": "college" if index % 2 else "high_school",
                    "employment_status": "employed",
                    "liquid_wealth_group": "high" if index % 2 else "low",
                    "actual_expected_inflation_1y": float(index + month_index),
                    "actual_expected_unemployment_higher_prob": float(10 + index),
                    "actual_expected_real_income_growth": float(1 + month_index),
                }
            )
    return pd.DataFrame(rows)


class PersistentHouseholdTests(unittest.TestCase):
    def _cohorts(self, frame: pd.DataFrame | None = None) -> dict[str, object]:
        return prepare_household_scale_cohorts(
            _fixture() if frame is None else frame,
            master_sample_size=8,
            core_sample_size=4,
            sample_seed=20250709,
        )

    def test_weight_formula_normalization_and_metrics(self) -> None:
        result = self._cohorts()
        registry = result["identity_registry"]
        expected = (
            registry["survey_weight"]
            * registry["stratum_population_count_200"]
            / registry["stratum_sample_count_200"]
        )
        pd.testing.assert_series_equal(expected, registry["selection_weight_200"], check_names=False)
        self.assertAlmostEqual(float(registry["population_weight_200"].sum()), 1.0)
        self.assertAlmostEqual(float(result["initial_households_200"]["population_weight"].sum()), 1.0)
        metrics = result["manifest"]["weight_metrics"]["master_200"]
        weights = registry["population_weight_200"]
        self.assertAlmostEqual(metrics["effective_sample_size"], float(weights.sum() ** 2 / (weights**2).sum()))
        self.assertAlmostEqual(metrics["max_population_weight"], float(weights.max()))
        core = registry[registry["included_in_core_81"]]
        self.assertAlmostEqual(float(core["population_weight_81"].sum()), 1.0)
        expected_core = (
            core["survey_weight"]
            * core["stratum_population_count_81"]
            / core["stratum_sample_count_81"]
        )
        pd.testing.assert_series_equal(
            expected_core, core["selection_weight_81"], check_names=False
        )

    def test_core_is_the_existing_same_seed_direct_draw(self) -> None:
        source = _fixture()
        result = self._cohorts(source)
        self.assertEqual(derived_initial_seed(20250709), 1317289316)
        eligible = eligible_april_2025_with_march_prior(source)
        legacy, _ = _stratified_wave_sample(
            eligible,
            sample_size=4,
            sample_seed=derived_initial_seed(20250709),
        )
        expected = {stable_household_id(value) for value in legacy["respondent_id"]}
        self.assertEqual(set(result["initial_households_81"]["type_id"]), expected)

    def test_id_and_selection_are_input_order_stable_and_core_is_nested(self) -> None:
        first = self._cohorts()
        second = self._cohorts(_fixture().sample(frac=1, random_state=2025))
        pd.testing.assert_frame_equal(first["identity_registry"], second["identity_registry"])
        registry = first["identity_registry"]
        self.assertEqual(
            set(first["initial_households_81"]["type_id"]),
            set(registry.loc[registry["included_in_core_81"], "household_id"]),
        )
        self.assertTrue(set(first["initial_households_81"]["type_id"]).issubset(set(first["initial_households_200"]["type_id"])))
        self.assertEqual(stable_household_id("raw-001"), stable_household_id("raw-001"))

    def test_future_wave_completion_cannot_affect_april_selection(self) -> None:
        full = self._cohorts(_fixture(include_future=True))
        reduced_source = _fixture(include_future=True)
        reduced_source = reduced_source[
            ~(
                reduced_source["respondent_id"].isin({"raw-000", "raw-001", "raw-002", "raw-003"})
                & reduced_source["survey_date"].eq("2025-05-01")
            )
        ]
        reduced = self._cohorts(reduced_source)
        pd.testing.assert_frame_equal(full["identity_registry"], reduced["identity_registry"])
        pd.testing.assert_frame_equal(full["initial_households_200"], reduced["initial_households_200"])
        self.assertEqual(full["manifest"]["immediate_prior_event_date"], "2025-03-01")
        self.assertFalse(full["manifest"]["future_completion_used_for_selection"])

    def test_append_is_private_registry_matched_nonresponse_preserving_and_rejects_duplicates(self) -> None:
        result = self._cohorts()
        history = result["observed_history"]
        self.assertEqual(history.shape[0], 16)
        self.assertEqual(set(history["event_date"]), {"2025-03-01", "2025-04-01"})
        self.assertEqual(set(history["attrition_status"]), {"responding"})
        self.assertEqual(set(history["death_status"]), {"alive_no_death_observation"})
        private = result["private_registry"]
        raw_id = private.iloc[0]["raw_respondent_id"]
        update = append_selected_observed_history(
            result["observed_history"],
            pd.DataFrame(
                [
                    {
                        "respondent_id": raw_id,
                        "survey_date": "2025-05-01",
                        "weight": 2.5,
                        "actual_expected_inflation_1y": 3.0,
                        "actual_expected_unemployment_higher_prob": 20.0,
                        "actual_expected_real_income_growth": 1.0,
                    },
                    {
                        "respondent_id": "not-in-selected-cohort",
                        "survey_date": "2025-05-01",
                        "weight": 1.0,
                    },
                ]
            ),
            private,
            event_date="2025-05-01",
        )
        appended = update["appended_history"]
        self.assertEqual(appended.shape[0], 8)
        self.assertEqual(int(appended["responded"].sum()), 1)
        self.assertEqual(set(appended["observation_status"]), {"observed", "nonresponse"})
        nonresponse = appended[~appended["responded"]]
        self.assertEqual(
            set(nonresponse["attrition_status"]),
            {"survey_nonresponse_not_economic_exit"},
        )
        self.assertEqual(set(nonresponse["death_status"]), {"alive_no_death_observation"})
        self.assertEqual(update["replay_required_from_event_date"], "2025-05-01")
        self.assertFalse(update["simulated_state_overwritten"])
        self.assertEqual(update["matched_observation_count"], 1)
        self.assertEqual(update["unselected_observation_count"], 1)
        self.assertEqual(
            float(appended.loc[appended["responded"], "survey_weight"].iloc[0]),
            2.5,
        )
        with self.assertRaisesRegex(PersistentHouseholdError, "duplicate or overwriting"):
            append_selected_observed_history(
                update["observed_history"],
                pd.DataFrame([{"respondent_id": raw_id}]),
                private,
                event_date="2025-05-01",
            )

    def test_writer_separates_private_raw_ids_from_public_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_csv = root / "source.csv"
            _fixture().to_csv(input_csv, index=False)
            result = build_household_scale_cohorts(
                input_csv,
                output_dir=root / "public",
                private_output_dir=root / "private",
                master_sample_size=8,
                core_sample_size=4,
            )
            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for key, path in result.items()
                if key.endswith("_csv") and "private" not in key
            )
            self.assertNotIn("raw-", public_text)
            self.assertIn("raw-", result["private_registry_csv"].read_text(encoding="utf-8"))
            self.assertTrue(result["public_information_schedule_csv"].exists())
            states = pd.read_csv(result["initial_households_200_csv"])
            self.assertEqual(
                set(states["balance_sheet_source"]),
                {
                    "coarse_synthetic_mapping_from_income_and_liquidity_groups_not_observed_balances"
                },
            )


if __name__ == "__main__":
    unittest.main()
