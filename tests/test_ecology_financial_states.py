from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from macro_llm_tournament.ecology_financial_states import (
    SCHEMA_VERSION,
    collapse_scf_implicates,
    generate_financial_states,
    write_financial_states,
)


def _scf_implicates() -> pd.DataFrame:
    records: list[dict[str, float]] = []
    donors = [
        # yy1, income, age, edcl, employed, owner, liquid deposits, revolving debt
        (101, 30_000.0, 29, 2, 1, 0, 150.0, 0.0),
        (202, 80_000.0, 45, 3, 1, 1, 9_000.0, 1_800.0),
        (303, 220_000.0, 64, 4, 0, 1, 70_000.0, 7_000.0),
    ]
    for yy1, income, age, edcl, lf, own, liq, ccbal in donors:
        for implicate in range(5):
            records.append(
                {
                    "y1": yy1 * 10 + implicate + 1,
                    "yy1": yy1,
                    "wgt": 100.0 + yy1,
                    "age": age,
                    "edcl": edcl,
                    "lf": lf,
                    "own": own,
                    "income": income + implicate * 100.0,
                    "wageinc": income * 0.75,
                    "bussefarminc": 12_000.0 if yy1 == 202 else 0.0,
                    "ssretinc": 1_200.0 if yy1 == 303 else 0.0,
                    "transfothinc": 600.0 if yy1 == 101 else 0.0,
                    "foodhome": 6_000.0,
                    "foodaway": 1_200.0,
                    "fooddelv": 600.0,
                    "rent": 1_100.0 if own == 0 else 0.0,
                    "mortpay": 1_500.0 if own else 0.0,
                    "liq": liq + implicate,
                    "ccbal": ccbal,
                    "noccbal": 1 if not ccbal else 0,
                    "revpay": 90.0 if ccbal else 0.0,
                    "saved": 1 if yy1 != 101 else 0,
                }
            )
    return pd.DataFrame(records)


def _households() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type_id": "household_z",
                "population_weight": 0.2,
                "income_group": "low",
                "age_bucket": "18_34",
                "employment_status": "employed",
                "homeownership": "renter",
                "cohort_stratum": "low|18_34|hs_or_less",
                "inflation_expectation_1y": 3.0,
            },
            {
                "type_id": "household_a",
                "population_weight": 0.8,
                "income_group": "high",
                "age_bucket": "55_plus",
                "employment_status": "unemployed",
                "homeownership": "owner",
                "cohort_stratum": "high|55_plus|college_plus",
                "inflation_expectation_1y": 4.0,
            },
        ]
    )


class EcologyFinancialStatesTests(unittest.TestCase):
    def test_collapse_selects_one_coherent_implicate_and_preserves_weight(self) -> None:
        donors = collapse_scf_implicates(_scf_implicates().sample(frac=1.0, random_state=3))
        low = donors.loc[donors["yy1"].eq(101)].iloc[0]
        self.assertEqual(len(donors), 3)
        self.assertEqual(low["scf_implicate_count"], 5)
        self.assertEqual(low["income"], 30_000.0)
        self.assertEqual(low["wgt"], 201.0)
        self.assertEqual(low["income_group"], "low")

    def test_generation_is_order_stable_and_retains_sce_columns(self) -> None:
        households = _households()
        forward = generate_financial_states(households, _scf_implicates(), seed=91)
        reversed_input = generate_financial_states(households.iloc[::-1], _scf_implicates().iloc[::-1], seed=91)
        assert_frame_equal(forward, reversed_input, check_like=False)
        self.assertEqual(forward["household_id"].tolist(), ["household_a", "household_z"])
        self.assertEqual(forward["population_weight"].tolist(), [0.8, 0.2])
        self.assertEqual(forward["inflation_expectation_1y"].tolist(), [4.0, 3.0])
        self.assertEqual(forward.loc[forward["household_id"].eq("household_z"), "annual_income_usd"].item(), 30_000.0)
        self.assertEqual(forward.loc[forward["household_id"].eq("household_a"), "annual_income_usd"].item(), 220_000.0)
        self.assertIn("scf_donor_employment_group", forward.columns)

    def test_business_income_is_earned_income_not_nonwage_income(self) -> None:
        household = _households().iloc[[0]].copy()
        household["income_group"] = "middle"
        household["age_bucket"] = "35_54"
        household["homeownership"] = "owner"
        household["cohort_stratum"] = "middle|35_54|some_college"
        middle = generate_financial_states(household, _scf_implicates(), seed=91).iloc[0]
        self.assertEqual(middle["monthly_business_income_usd"], 1_000.0)
        self.assertEqual(
            middle["monthly_earned_income_usd"],
            middle["monthly_wage_income_usd"] + 1_000.0,
        )

    def test_income_components_reconcile_to_total_household_income(self) -> None:
        output = generate_financial_states(_households(), _scf_implicates(), seed=91)
        components = (
            output["monthly_earned_income_usd"]
            + output["monthly_nonwage_income_usd"]
            + output["monthly_transfers_benefits_usd"]
        )
        assert_series_equal(
            components,
            output["annual_income_usd"] / 12.0,
            check_names=False,
        )
        self.assertTrue(output["income_components_reconcile_to_annual_income"].all())

    def test_financial_state_allows_observed_zero_debt_and_liquidity_dispersion(self) -> None:
        output = generate_financial_states(_households(), _scf_implicates(), seed=1)
        debt_free = output.loc[output["annual_income_usd"].eq(30_000.0)].iloc[0]
        indebted = output.loc[output["annual_income_usd"].eq(220_000.0)].iloc[0]
        self.assertEqual(debt_free["revolving_debt_usd"], 0.0)
        self.assertGreater(debt_free["revolving_credit_limit_usd"], 0.0)
        self.assertEqual(debt_free["revolving_credit_utilization"], 0.0)
        self.assertGreater(indebted["liquid_deposits_usd"], debt_free["liquid_deposits_usd"])
        self.assertGreater(indebted["revolving_credit_limit_usd"], indebted["revolving_debt_usd"])

    def test_relaxation_is_explicit_when_an_exact_cell_is_empty(self) -> None:
        household = _households().iloc[[0]].copy()
        household["income_group"] = "high"
        household["age_bucket"] = "18_34"
        output = generate_financial_states(household, _scf_implicates(), seed=5)
        row = output.iloc[0]
        self.assertIn("age_group", row["scf_match_relaxed_fields"])
        self.assertTrue(row["scf_match_rule"].startswith("relaxed_"))

    def test_writer_reads_a_small_synthetic_stata_zip_and_writes_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            households_path = root / "households.csv"
            dta_path = root / "rscfp2022.dta"
            zip_path = root / "scfp2022s.zip"
            output_path = root / "financial_states.csv"
            manifest_path = root / "financial_states.manifest.json"
            _households().to_csv(households_path, index=False)
            _scf_implicates().to_stata(dta_path, write_index=False)
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(dta_path, "rscfp2022.dta")
            manifest = write_financial_states(
                households_path, zip_path, output_path, manifest_path, seed=77
            )
            saved = pd.read_csv(output_path)
            on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved["financial_state_schema_version"].unique().tolist(), [SCHEMA_VERSION])
        self.assertEqual(manifest["manifest_sha256"], on_disk_manifest["manifest_sha256"])
        self.assertEqual(manifest["output_csv"]["row_count"], 2)
        self.assertFalse(manifest["matching"]["target_or_actual_data_used"])


if __name__ == "__main__":
    unittest.main()
