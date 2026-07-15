import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from macro_llm_tournament import ecology_observability
from macro_llm_tournament.ecology import _artifact_sha256


class EcologyObservabilityTests(unittest.TestCase):
    def test_firm_shadow_maps_demand_and_inventory_with_declared_signs(self) -> None:
        balanced = ecology_observability.firm_response_shadow(
            consumption_growth_pct=2.0,
            inventory_end_units=8.0,
            units_sold=100.0,
        )
        shortage = ecology_observability.firm_response_shadow(
            consumption_growth_pct=2.0,
            inventory_end_units=4.0,
            units_sold=100.0,
        )
        self.assertAlmostEqual(balanced["firm_expected_sales_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_target_output_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_required_labor_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_planned_employment_index"], 100.5)
        self.assertGreater(shortage["firm_target_output_index"], balanced["firm_target_output_index"])
        self.assertGreater(shortage["firm_price_pressure_pp"], balanced["firm_price_pressure_pp"])
        with self.assertRaisesRegex(ValueError, "units_sold must be positive"):
            ecology_observability.firm_response_shadow(
                consumption_growth_pct=2.0,
                inventory_end_units=8.0,
                units_sold=0.0,
            )
        with self.assertRaisesRegex(ValueError, "inventory_end_units must be nonnegative"):
            ecology_observability.firm_response_shadow(
                consumption_growth_pct=2.0,
                inventory_end_units=-1.0,
                units_sold=100.0,
            )

    def test_employment_branch_requires_a_json_boolean(self) -> None:
        employed = {"household": {"current_state": {"employed": True}}}
        not_employed = {"household": {"current_state": {"employed": False}}}
        malformed = {"household": {"current_state": {"employed": "false"}}}
        self.assertEqual(
            ecology_observability._employment_policy_name(employed),
            "employed_policy",
        )
        self.assertEqual(
            ecology_observability._employment_policy_name(not_employed),
            "not_employed_policy",
        )
        with self.assertRaisesRegex(ValueError, "must be a JSON boolean"):
            ecology_observability._employment_policy_name(malformed)

    def test_weighted_mean_requires_exact_household_coverage(self) -> None:
        self.assertAlmostEqual(
            ecology_observability._weighted_mean(
                {"a": 10.0, "b": 20.0}, {"a": 1.0, "b": 3.0}
            ),
            17.5,
        )
        with self.assertRaisesRegex(ValueError, "do not match households"):
            ecology_observability._weighted_mean({"a": 10.0}, {"a": 1.0, "b": 3.0})

    def test_observed_panel_keeps_prospective_rows_unscored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retrospective = root / "retrospective"
            prospective = root / "prospective"
            retrospective.mkdir()
            prospective.mkdir()
            joined = pd.DataFrame(
                [
                    {
                        "origin_month": "2026-01-01",
                        "target_month": "2026-02-01",
                        "metric": "consumption_growth_pct",
                        "prediction": 0.2,
                        "actual": 0.4,
                        "routine_nominal_spending_drift_pct": 0.3,
                        "mapping_quality": "closest_aggregate_proxy",
                        "mapping_note": "test mapping",
                    }
                ]
            )
            joined.to_csv(retrospective / "predicted_vs_actual.csv", index=False)
            (retrospective / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "predicted_vs_actual.csv": _artifact_sha256(
                                retrospective / "predicted_vs_actual.csv"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "scenario": "median",
                        "consumption_growth_pct": 0.5,
                        "routine_nominal_spending_drift_pct": 0.6,
                        "revolving_credit_growth_pct": -1.0,
                    }
                ]
            ).to_csv(prospective / "macro_forecast_paths.csv", index=False)
            (prospective / "manifest.json").write_text(
                json.dumps(
                    {
                        "origin_month": "2026-07-01",
                        "target_month": "2026-08-01",
                        "evaluation_status": "prospective_frozen",
                        "artifacts": {
                            "macro_forecast_paths.csv": _artifact_sha256(
                                prospective / "macro_forecast_paths.csv"
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            panel = ecology_observability._observed_panel(retrospective, prospective)
            frozen = panel.loc[panel["target_month"].eq("2026-08-01")]
            self.assertFalse(frozen.empty)
            self.assertTrue(
                frozen["evaluation_status"].eq("prospective_frozen_unscored").all()
            )
            self.assertFalse(frozen["series_role"].eq("first_release_actual").any())

    def test_source_artifact_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            manifest = {"artifacts": {"data.csv": _artifact_sha256(path)}}
            path.write_text("a\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                ecology_observability._validate_artifact(root, manifest, "data.csv")

    def test_child_status_is_not_an_accuracy_label(self) -> None:
        self.assertEqual(
            ecology_observability._evaluation_status(
                {"evaluation_status": "prospective_frozen"}
            ),
            "prospective_frozen",
        )
        with self.assertRaisesRegex(ValueError, "unsupported child evaluation status"):
            ecology_observability._evaluation_status(
                {"evaluation_status": "confirmatory_pass"}
            )


if __name__ == "__main__":
    unittest.main()
