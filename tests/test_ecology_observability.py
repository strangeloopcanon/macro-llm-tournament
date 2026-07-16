import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from macro_llm_tournament import ecology_observability
from macro_llm_tournament.ecology import _artifact_sha256
from macro_llm_tournament.ecology_macro import (
    MACRO_TARGET_CONTRACT,
    contemporaneous_firm_plan,
)


class EcologyObservabilityTests(unittest.TestCase):
    def test_firm_plan_maps_demand_and_origin_inventory_with_declared_signs(self) -> None:
        balanced = contemporaneous_firm_plan(
            consumption_growth_pct=2.0,
            inventory_start_units=8.0,
            baseline_sales_units=100.0,
        )
        shortage = contemporaneous_firm_plan(
            consumption_growth_pct=2.0,
            inventory_start_units=4.0,
            baseline_sales_units=100.0,
        )
        self.assertAlmostEqual(balanced["firm_expected_sales_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_target_output_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_required_labor_index"], 102.0)
        self.assertAlmostEqual(balanced["firm_planned_employment_index"], 100.5)
        self.assertGreater(shortage["firm_target_output_index"], balanced["firm_target_output_index"])
        self.assertGreater(shortage["firm_price_pressure_pp"], balanced["firm_price_pressure_pp"])
        with self.assertRaisesRegex(ValueError, "baseline_sales_units must be positive"):
            contemporaneous_firm_plan(
                consumption_growth_pct=2.0,
                inventory_start_units=8.0,
                baseline_sales_units=0.0,
            )
        with self.assertRaisesRegex(ValueError, "inventory_start_units must be nonnegative"):
            contemporaneous_firm_plan(
                consumption_growth_pct=2.0,
                inventory_start_units=-1.0,
                baseline_sales_units=100.0,
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

    def test_run_payload_reports_executed_liquid_cash_residual_without_deposit_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            cards = [
                {
                    "household": {
                        "household_id": "h1",
                        "current_state": {
                            "employed": True,
                            "monthly_consumption": 1_000.0,
                        },
                    }
                }
            ]
            responses = [
                {
                    "payload": {
                        "household_id": "h1",
                        "expected_inflation_pct": {"p50": 2.0},
                        "expected_income_growth_pct": {"p50": 1.0},
                        "job_loss_probability_pct": {"p50": 3.0},
                        "employed_policy": {
                            "committed_consumption_change_usd": 50.0,
                            "discretionary_consumption_change_usd": -25.0,
                            "one_off_purchase_usd": 10.0,
                            "extra_debt_payment_usd": 50.0,
                            "borrowing_intent_usd": 25.0,
                        },
                    }
                }
            ]
            (run_dir / "household_cards.json").write_text(json.dumps(cards), encoding="utf-8")
            (run_dir / "household_responses.json").write_text(
                json.dumps(responses), encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "household_id": "h1",
                        "baseline_consumption_usd": 1_000.0,
                        "desired_consumption_usd": 900.0,
                        "consumption_usd": 875.0,
                        "debt_payment_usd": 75.0,
                        "borrowing_usd": 25.0,
                        "deposit_balance_start_usd": 500.0,
                        "deposit_balance_end_usd": 375.0,
                        "revolving_debt_start_usd": 400.0,
                        "revolving_debt_end_usd": 450.0,
                    }
                ]
            ).to_csv(run_dir / "household_decisions.csv", index=False)
            macro_row = {
                "scenario": "median",
                "routine_nominal_spending_drift_pct": 1.0,
                "gross_income_residual_rate_pct": 2.0,
                "gross_income_residual_rate_change_pp": -0.5,
                "next_period_price_growth_pct": 0.3,
                "employment_rate_pct": 95.0,
                "household_nonemployment_share_pct": 5.0,
                "expected_inflation_annual_pct": 2.0,
                "expected_income_growth_annual_pct": 1.5,
                "output_units": 90.0,
                "units_sold": 90.0,
                "inventory_end_units": 9.0,
                "firm_expected_sales_index": 87.5,
                "firm_target_output_index": 87.15,
                "firm_required_labor_index": 87.15,
                "firm_planned_employment_index": 96.7875,
                "firm_price_pressure_pp": -1.0,
                "firm_inventory_share_pct": 10.0,
                "firm_inventory_gap_pp": -2.0,
            }
            for index, (metric, mapping) in enumerate(MACRO_TARGET_CONTRACT.items()):
                macro_row[metric] = float(index + 1)
                macro_row[mapping["visible_baseline_field"]] = float(index) / 10.0
            pd.DataFrame([macro_row]).to_csv(
                run_dir / "macro_forecast_paths.csv", index=False
            )
            (run_dir / "median_economy.json").write_text(
                json.dumps(
                    {
                        "employer": {"demand_pressure": 0.9, "profit_usd": 10.0},
                        "credit": {"rationing_ratio": 0.8, "profit_usd": 5.0},
                    }
                ),
                encoding="utf-8",
            )
            artifacts = {
                name: _artifact_sha256(run_dir / name)
                for name in (
                    "household_cards.json",
                    "household_responses.json",
                    "household_decisions.csv",
                    "macro_forecast_paths.csv",
                    "median_economy.json",
                )
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "accounting_passed": True,
                        "household_count": 1,
                        "origin_month": "2026-01-01",
                        "target_month": "2026-02-01",
                        "evaluation_status": "prospective_frozen",
                        "artifacts": artifacts,
                    }
                ),
                encoding="utf-8",
            )

            _, rows = ecology_observability._run_payload(run_dir, {"h1": 1.0})

        metrics = {row["metric"] for row in rows}
        self.assertNotIn("deposit_change_intent_usd", metrics)
        self.assertNotIn("deposit_change_intent_pct_of_baseline_consumption", metrics)
        executed_residual = next(
            row
            for row in rows
            if row["metric"] == "executed_liquid_deposit_residual_usd"
        )
        self.assertEqual(executed_residual["value"], -125.0)
        self.assertEqual(
            executed_residual["source_class"],
            "code_enforced_budgets_and_settlement",
        )
        self.assertIn("liquid cash residual", executed_residual["interpretation"])
        self.assertEqual(
            next(row for row in rows if row["layer"] == "intended_policy")["source_class"],
            "llm_household_intention",
        )
        self.assertEqual(
            next(row for row in rows if row["layer"] == "firm_plan")["source_class"],
            "household_demand_driven_firm_plan",
        )
        macro_execution = {
            row["metric"]: row["value"]
            for row in rows
            if row["layer"] == "macro_execution"
        }
        self.assertTrue(set(MACRO_TARGET_CONTRACT).issubset(macro_execution))
        self.assertEqual(macro_execution["price_growth_pct"], 3.0)
        self.assertEqual(macro_execution["next_period_price_growth_pct"], 0.3)

    def test_feedback_period_two_adds_unscored_household_and_firm_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feedback_dir = Path(temporary)
            period_one_dir = feedback_dir / "period-one"
            period_one_dir.mkdir()
            (period_one_dir / "manifest.json").write_text(
                json.dumps({"target_month": "2026-08-01"}), encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "period": 1,
                        "consumption_usd": 100.0,
                        "output_units": 100.0,
                        "producer_employment_index": 1.0,
                        "producer_wage_index": 1.0,
                        "consumption_growth_from_period_1_pct": 0.0,
                    },
                    {
                        "period": 2,
                        "consumption_usd": 101.0,
                        "output_units": 102.0,
                        "producer_employment_index": 1.005,
                        "producer_wage_index": 1.001,
                        "consumption_growth_from_period_1_pct": 1.0,
                    }
                ]
            ).to_csv(feedback_dir / "dynamic_macro_paths.csv", index=False)
            (feedback_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": ecology_observability.FEEDBACK_SCHEMA_VERSION,
                        "origin_month": "2026-07-01",
                        "period_2_target_month": "2026-09-01",
                        "period_1_replay_equivalence_sha256": "period-one-equivalence",
                        "household_count": 1,
                        "accounting_passed": True,
                        "replay_verified": True,
                        "period_1_run": str(period_one_dir),
                        "period_1_manifest_sha256": _artifact_sha256(
                            period_one_dir / "manifest.json"
                        ),
                        "artifacts": {
                            "dynamic_macro_paths.csv": _artifact_sha256(
                                feedback_dir / "dynamic_macro_paths.csv"
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows, manifest_sha256 = ecology_observability._feedback_period_two_rows(
                feedback_dir
            )

        self.assertEqual(len(rows), 4)
        self.assertIsNotNone(manifest_sha256)
        self.assertTrue(all(row["target_month"] == "2026-09-01" for row in rows))
        self.assertTrue(
            all(
                row["evaluation_status"]
                == "prospective_feedback_period_2_unscored"
                for row in rows
            )
        )
        self.assertEqual({row["source_class"] for row in rows if row["layer"] == "macro_execution"}, {"llm_household_economy"})
        self.assertEqual(
            {
                row["source_class"]
                for row in rows
                if row["layer"] == "recursive_firm_feedback"
            },
            {"mechanical_firm_feedback"},
        )

    def test_feedback_period_two_fails_closed_when_its_schema_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feedback_dir = Path(temporary)
            pd.DataFrame([{"period": "period_2"}]).to_csv(
                feedback_dir / "dynamic_macro_paths.csv", index=False
            )
            (feedback_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": ecology_observability.FEEDBACK_SCHEMA_VERSION,
                        "household_count": 1,
                        "accounting_passed": True,
                        "replay_verified": True,
                        "artifacts": {
                            "dynamic_macro_paths.csv": _artifact_sha256(
                                feedback_dir / "dynamic_macro_paths.csv"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing fields"):
                ecology_observability._feedback_period_two_rows(feedback_dir)

    def test_explicit_feedback_run_missing_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing dynamic_macro_paths"):
                ecology_observability._feedback_period_two_rows(Path(temporary))

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
                        "metric": metric,
                        "prediction": float(index + 1),
                        "context_prediction": float(index) / 10.0,
                        "actual": float(index + 2),
                        "mapping_quality": mapping["mapping_quality"],
                        "mapping_note": mapping["note"],
                    }
                    for index, (metric, mapping) in enumerate(MACRO_TARGET_CONTRACT.items())
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
            prospective_row = {"scenario": "median"}
            for index, (metric, mapping) in enumerate(MACRO_TARGET_CONTRACT.items()):
                prospective_row[metric] = float(index + 1)
                prospective_row[mapping["visible_baseline_field"]] = float(index) / 10.0
            pd.DataFrame([prospective_row]).to_csv(
                prospective / "macro_forecast_paths.csv", index=False
            )
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
            economy_rows = frozen.loc[frozen["series_role"].eq("llm_household_economy")]
            baseline_rows = frozen.loc[
                frozen["series_role"].eq("origin_visible_baseline")
            ]
            self.assertEqual(set(economy_rows["metric"]), set(MACRO_TARGET_CONTRACT))
            self.assertEqual(set(baseline_rows["metric"]), set(MACRO_TARGET_CONTRACT))
            self.assertTrue(
                economy_rows["source_class"].eq(
                    "llm_household_economy_code_enforced_budgets_and_settlement"
                ).all()
            )
            derivation_by_metric = economy_rows.set_index("metric")[
                "derivation_class"
            ].to_dict()
            self.assertEqual(
                derivation_by_metric,
                {
                    metric: mapping["mapping_quality"]
                    for metric, mapping in MACRO_TARGET_CONTRACT.items()
                },
            )

    def test_public_economy_label_names_code_enforced_settlement(self) -> None:
        self.assertEqual(
            ecology_observability.LLM_HOUSEHOLD_ECONOMY_SETTLEMENT_LABEL,
            "LLM household economy - code-enforced budgets and settlement",
        )

    def test_parser_accepts_optional_feedback_run(self) -> None:
        args = ecology_observability.build_arg_parser().parse_args(
            [
                "--retrospective-run",
                "retrospective",
                "--prospective-run",
                "prospective",
                "--feedback-run",
                "feedback",
                "--households",
                "households.csv",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.feedback_run, Path("feedback"))

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
