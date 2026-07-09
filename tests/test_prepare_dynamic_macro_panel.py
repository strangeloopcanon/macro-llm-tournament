from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.demand_economy import normalize_demand_households
from macro_llm_tournament.prepare_dynamic_macro_panel import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SPLITS,
    MODEL_CUTOFFS,
    DynamicMacroPanelError,
    build_dynamic_macro_panel,
    contamination_label,
    normalize_split_windows,
    prepare_dynamic_macro_panel,
    stratified_complete_panel_sample,
    validate_normalized_sce,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture(months: int = 6, respondents: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    rows: list[dict[str, object]] = []
    strata = [("low", "18_34", "high_school"), ("middle", "35_54", "college"), ("high", "55_plus", "college_plus")]
    for respondent in range(respondents):
        income, age, education = strata[respondent % len(strata)]
        for index, date in enumerate(dates):
            rows.append(
                {
                    "respondent_id": f"raw-{respondent:03d}",
                    "survey_date": date.strftime("%Y-%m-%d"),
                    "weight": float(respondent + 1),
                    "income_group": income,
                    "age_group": age,
                    "education_group": education,
                    "gender": "female" if respondent % 2 else "male",
                    "employment_status": "employed",
                    "liquid_wealth_group": "high" if respondent % 2 else "low",
                    "actual_expected_inflation_1y": 2.0 + respondent + index / 10.0,
                    "actual_expected_unemployment_higher_prob": 20.0 + respondent,
                    "actual_expected_real_income_growth": 1.0 + index,
                }
            )
    return pd.DataFrame(rows)


class ValidationTests(unittest.TestCase):
    def test_required_registry_and_default_split_contract(self) -> None:
        self.assertEqual(DEFAULT_SAMPLE_SIZE, 81)
        self.assertEqual(
            MODEL_CUTOFFS,
            {"gpt-5-codex": "2024-09-30", "gpt-5.4": "2025-08-31", "gpt-5.5": "2025-12-01"},
        )
        windows = normalize_split_windows(DEFAULT_SPLITS)
        self.assertEqual(windows["development"], (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")))
        self.assertEqual(windows["validation"][1], pd.Timestamp("2024-06-01"))
        self.assertEqual(windows["sealed_test"][1], pd.Timestamp("2025-09-01"))

    def test_duplicate_grain_is_rejected(self) -> None:
        frame = _fixture().iloc[[0, 0]].copy()
        with self.assertRaisesRegex(DynamicMacroPanelError, "duplicate respondent_id.survey_date grain"):
            validate_normalized_sce(frame)

    def test_nonconsecutive_input_and_bad_weights_are_rejected(self) -> None:
        frame = _fixture()
        frame = frame[~((frame["respondent_id"] == "raw-000") & (frame["survey_date"] == "2020-03-01"))]
        validated = validate_normalized_sce(frame)
        self.assertEqual(validated.shape[0], frame.shape[0])
        global_gap = _fixture()
        global_gap = global_gap[global_gap["survey_date"] != "2020-03-01"]
        with self.assertRaisesRegex(DynamicMacroPanelError, "globally consecutive"):
            validate_normalized_sce(global_gap)
        sampled, metadata = stratified_complete_panel_sample(
            validated,
            months=list(pd.date_range("2020-01-01", "2020-06-01", freq="MS")),
            sample_size=6,
        )
        self.assertEqual(metadata["eligible_complete_panel_respondents"], 11)
        self.assertEqual(sampled.shape[0], 36)
        frame = _fixture()
        frame.loc[0, "weight"] = -1
        with self.assertRaisesRegex(DynamicMacroPanelError, "weights"):
            validate_normalized_sce(frame)

    def test_missing_belief_or_stratification_field_is_rejected(self) -> None:
        frame = _fixture()
        frame.loc[0, "actual_expected_real_income_growth"] = None
        with self.assertRaisesRegex(DynamicMacroPanelError, "actual_expected_real_income_growth"):
            validate_normalized_sce(frame)
        frame = _fixture().drop(columns=["education_group"])
        with self.assertRaisesRegex(DynamicMacroPanelError, "education_group"):
            validate_normalized_sce(frame)

    def test_split_overlap_is_rejected_but_gap_is_allowed(self) -> None:
        with self.assertRaisesRegex(DynamicMacroPanelError, "overlap"):
            normalize_split_windows({"development": "2020-01..2020-03", "validation": "2020-03..2020-05"})
        windows = normalize_split_windows({"development": "2020-01..2020-03", "validation": "2020-05..2020-06"})
        self.assertEqual(windows["validation"][0], pd.Timestamp("2020-05-01"))


class PreparationTests(unittest.TestCase):
    def test_availability_lag_drives_contamination_not_event_date(self) -> None:
        self.assertEqual(
            contamination_label("gpt-5-codex", "2024-01-01"),
            "post_cutoff_holdout",
        )
        self.assertEqual(
            contamination_label("gpt-5-codex", "2024-09-01"),
            "post_cutoff_holdout",
        )

    def test_panel_is_complete_anonymized_and_has_lagged_beliefs(self) -> None:
        result = prepare_dynamic_macro_panel(
            _fixture(months=6, respondents=12),
            splits={"development": "2020-01..2020-03", "sealed_test": "2020-04..2020-06"},
            sample_size=6,
            sample_seed=17,
            model="gpt-5.5",
        )
        panel = result["panel"]
        self.assertEqual(panel.shape[0], 36)
        self.assertEqual(panel.groupby("split_role")["respondent_id"].nunique().to_dict(), {"development": 6, "sealed_test": 6})
        self.assertEqual(panel["respondent_id"].nunique(), 12)
        self.assertTrue(panel["respondent_id"].str.startswith("respondent_").all())
        self.assertNotIn("raw-", " ".join(panel.to_csv(index=False).splitlines()))
        development_first = panel.query("split_role == 'development' and period_index == 0")
        sealed_first = panel.query("split_role == 'sealed_test' and period_index == 0")
        self.assertTrue(development_first["prior_expected_inflation_1y"].isna().all())
        self.assertTrue(sealed_first["prior_expected_inflation_1y"].notna().all())
        second = panel.sort_values(["respondent_id", "survey_date"]).groupby("respondent_id").nth(1)
        self.assertTrue((second["prior_expected_inflation_1y"] < second["actual_expected_inflation_1y"]).all())
        self.assertEqual(set(panel["split_role"]), {"development", "sealed_test"})
        self.assertTrue((panel["estimated_public_availability_date"] > panel["survey_event_date"]).all())
        self.assertIn("publication_assumptions", result["manifest"])

    def test_sampling_is_deterministic_and_stratified(self) -> None:
        kwargs = {
            "splits": {"development": "2020-01..2020-03", "validation": "2020-04..2020-06"},
            "sample_size": 9,
            "sample_seed": 101,
        }
        first = prepare_dynamic_macro_panel(_fixture(respondents=15), **kwargs)["panel"]
        second = prepare_dynamic_macro_panel(_fixture(respondents=15).sample(frac=1, random_state=4), **kwargs)["panel"]
        pd.testing.assert_frame_equal(first, second)
        sampled = first[first["survey_date"] == "2020-01-01"]
        self.assertEqual(set(sampled["income_group"]), {"low", "middle", "high"})
        self.assertEqual(set(sampled["age_group"]), {"18_34", "35_54", "55_plus"})
        self.assertEqual(set(sampled["education_group"]), {"high_school", "college", "college_plus"})

    def test_insufficient_complete_respondents_or_strata_fail_closed(self) -> None:
        with self.assertRaisesRegex(DynamicMacroPanelError, "insufficient complete-panel"):
            prepare_dynamic_macro_panel(
                _fixture(respondents=5),
                splits={"development": "2020-01..2020-06"},
                sample_size=6,
            )
        frame = _fixture(respondents=12)
        with self.assertRaisesRegex(DynamicMacroPanelError, "insufficient for 3"):
            prepare_dynamic_macro_panel(
                frame,
                splits={"development": "2020-01..2020-06"},
                sample_size=2,
            )

    def test_initial_households_are_accepted_by_demand_normalizer(self) -> None:
        result = prepare_dynamic_macro_panel(
            _fixture(months=6, respondents=9),
            splits={"development": "2020-01..2020-03", "sealed_test": "2020-04..2020-06"},
            sample_size=6,
        )
        households = result["initial_households"]
        normalized = normalize_demand_households(households)
        self.assertEqual(normalized.shape[0], 6)
        self.assertAlmostEqual(float(normalized["population_weight"].sum()), 1.0)
        self.assertEqual(normalized["type_id"].nunique(), 6)
        self.assertEqual(result["manifest"]["initial_role"], "sealed_test")
        self.assertEqual(result["manifest"]["initial_wave_position"], "last")
        self.assertEqual(result["manifest"]["initial_wave"], "2020-06-01")
        self.assertEqual(result["manifest"]["initial_estimated_public_availability_date"], "2021-03-01")
        self.assertTrue(households["source_split_role"].eq("sealed_test").all())
        self.assertTrue(households["source_survey_event_date"].eq("2020-06-01").all())
        self.assertTrue(households["source_estimated_public_availability_date"].eq("2021-03-01").all())
        self.assertTrue(households["source_contamination_label"].eq("potential_training_contamination").all())
        sealed_last_ids = set(
            result["panel"].query("split_role == 'sealed_test' and period_index == 2")["respondent_id"]
        )
        self.assertEqual(set(households["type_id"]), sealed_last_ids)
        explicit = prepare_dynamic_macro_panel(
            _fixture(months=6, respondents=9),
            splits={"development": "2020-01..2020-03", "sealed_test": "2020-04..2020-06"},
            sample_size=6,
            initial_role="development",
        )
        self.assertEqual(explicit["manifest"]["initial_role"], "development")
        self.assertEqual(explicit["manifest"]["initial_wave"], "2020-03-01")

        first_wave = prepare_dynamic_macro_panel(
            _fixture(months=6, respondents=9),
            splits={"development": "2020-01..2020-03", "sealed_test": "2020-04..2020-06"},
            sample_size=6,
            initial_wave_position="first",
        )
        first_households = first_wave["initial_households"]
        self.assertEqual(first_wave["manifest"]["initial_wave_position"], "first")
        self.assertEqual(first_wave["manifest"]["initial_wave"], "2020-04-01")
        self.assertEqual(first_wave["manifest"]["initial_estimated_public_availability_date"], "2021-01-01")
        self.assertEqual(first_wave["manifest"]["initial_sample"]["eligible_respondents"], 9)
        self.assertNotEqual(
            first_wave["manifest"]["initial_sample"]["sample_seed"],
            first_wave["manifest"]["sample"]["split_seeds"]["sealed_test"],
        )
        self.assertTrue(first_households["source_survey_event_date"].eq("2020-04-01").all())
        self.assertTrue(first_households["source_estimated_public_availability_date"].eq("2021-01-01").all())

    def test_first_wave_initial_sample_does_not_depend_on_future_completion(self) -> None:
        source = _fixture(months=6, respondents=18)
        kwargs = {
            "splits": {"sealed_test": "2020-04..2020-06"},
            "sample_size": 6,
            "sample_seed": 99,
            "initial_wave_position": "first",
        }
        full = prepare_dynamic_macro_panel(source, **kwargs)
        removed_future = source[
            ~(
                source["respondent_id"].isin({f"raw-{index:03d}" for index in range(6)})
                & source["survey_date"].isin({"2020-05-01", "2020-06-01"})
            )
        ]
        reduced = prepare_dynamic_macro_panel(removed_future, **kwargs)
        pd.testing.assert_frame_equal(full["initial_households"], reduced["initial_households"])
        self.assertNotEqual(
            full["manifest"]["split_roles"]["sealed_test"]["eligible_complete_panel_respondents"],
            reduced["manifest"]["split_roles"]["sealed_test"]["eligible_complete_panel_respondents"],
        )


class OutputAndCliTests(unittest.TestCase):
    def test_outputs_include_hashes_manifest_counts_and_are_reproducible(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sce_real_microdata.csv"
            _fixture(months=3, respondents=9).to_csv(input_path, index=False)
            one = build_dynamic_macro_panel(
                input_path,
                output_dir=root / "one",
                splits={"development": "2020-01..2020-03"},
                sample_size=6,
                sample_seed=44,
            )
            two = build_dynamic_macro_panel(
                input_path,
                output_dir=root / "two",
                splits={"development": "2020-01..2020-03"},
                sample_size=6,
                sample_seed=44,
            )
            manifest = json.loads(Path(one["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["panel_rows"], 18)
            self.assertEqual(manifest["model_cutoffs"], MODEL_CUTOFFS)
            self.assertEqual(manifest["publication_assumptions"]["core_sce_respondent_microdata_publication_lag_months"], 9)
            self.assertEqual(
                Path(one["panel_csv"]).read_bytes(), Path(two["panel_csv"]).read_bytes()
            )
            self.assertEqual(
                Path(one["initial_households_csv"]).read_bytes(), Path(two["initial_households_csv"]).read_bytes()
            )
            self.assertEqual(manifest["outputs"]["panel_csv"]["sha256"], __import__("hashlib").sha256(Path(one["panel_csv"]).read_bytes()).hexdigest())

    def test_cli_writes_requested_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "sce_real_microdata.csv"
            _fixture(months=3, respondents=9).to_csv(input_path, index=False)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.prepare_dynamic_macro_panel",
                    "--input-csv",
                    str(input_path),
                    "--output-dir",
                    str(root / "out"),
                    "--split",
                    "development=2020-01..2020-03",
                    "--sample-size",
                    "6",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "out" / "sce_dynamic_macro_panel.csv").exists())
            self.assertTrue((root / "out" / "initial_households.csv").exists())
            self.assertTrue((root / "out" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
