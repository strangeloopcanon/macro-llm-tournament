from __future__ import annotations

import inspect
import math
import unittest

from macro_llm_tournament import ecology_macro


def _card() -> dict[str, object]:
    change_series = {
        "PCE": 0.4,
        "PCEC96": 0.2,
        "PCEPI": 0.1,
        "DSPIC96": 0.3,
        "PAYEMS": 0.15,
        "PSAVERT": -0.2,
        "REVOLSL": 0.5,
        "RSAFS": 0.45,
    }
    series: dict[str, object] = {
        series_id: {
            "available": True,
            "latest_value": 100.0,
            "changes": {"1m": {"value": value}},
        }
        for series_id, value in change_series.items()
    }
    series["UNRATE"] = {
        "available": True,
        "latest_value": 4.0,
        "changes": {"1m": {"value": -0.1}},
    }
    return {"series": series}


class EcologyMacroTests(unittest.TestCase):
    def test_metadata_covers_the_nine_standardized_targets(self) -> None:
        targets = ecology_macro.STANDARDIZED_MACRO_TARGETS
        self.assertEqual(len(targets), 9)
        self.assertEqual(
            [(metric, data["series_id"], data["target_name"]) for metric, data in targets.items()],
            [
                ("consumption_growth_pct", "PCE", "pce_growth_pct"),
                ("real_consumption_growth_pct", "PCEC96", "real_pce_growth_pct"),
                ("price_growth_pct", "PCEPI", "pce_price_growth_pct"),
                ("real_disposable_income_growth_pct", "DSPIC96", "real_disposable_income_growth_pct"),
                ("payroll_growth_pct", "PAYEMS", "payroll_growth_pct"),
                ("unemployment_rate_level", "UNRATE", "unemployment_rate_level"),
                ("personal_saving_rate_change_pp", "PSAVERT", "personal_saving_rate_change"),
                ("revolving_credit_growth_pct", "REVOLSL", "revolving_credit_growth_pct"),
                ("retail_sales_growth_pct", "RSAFS", "retail_sales_growth_pct"),
            ],
        )
        required = {
            "mapping_quality",
            "score_mode",
            "unit",
            "visible_baseline_field",
            "visible_baseline_source",
            "visible_baseline_kind",
        }
        self.assertTrue(all(required.issubset(data) for data in targets.values()))
        self.assertEqual(targets["revolving_credit_growth_pct"]["score_mode"], "direction_only")
        self.assertTrue(
            all(
                data["visible_baseline_field"] == f"visible_baseline_{metric}"
                for metric, data in targets.items()
            )
        )

    def test_transforms_and_firm_plan_preserve_declared_formulae(self) -> None:
        self.assertAlmostEqual(
            ecology_macro.annual_to_monthly_growth(12.0),
            100.0 * (1.12 ** (1.0 / 12.0) - 1.0),
        )
        self.assertAlmostEqual(
            ecology_macro.fisher_real_growth(3.0, 1.0), 100.0 * (1.03 / 1.01 - 1.0)
        )
        self.assertAlmostEqual(
            ecology_macro.unemployment_level_from_payroll_growth(4.0, 1.0), 3.04
        )
        firm_plan = ecology_macro.contemporaneous_firm_plan(
            consumption_growth_pct=2.0,
            inventory_start_units=4.0,
            baseline_sales_units=100.0,
        )
        self.assertAlmostEqual(firm_plan["firm_expected_sales_index"], 102.0)
        self.assertAlmostEqual(firm_plan["firm_target_output_index"], 103.4)
        self.assertAlmostEqual(firm_plan["firm_required_labor_index"], 103.4)
        self.assertAlmostEqual(firm_plan["firm_planned_employment_index"], 100.85)
        self.assertAlmostEqual(firm_plan["firm_price_pressure_pp"], 0.7)
        self.assertAlmostEqual(firm_plan["firm_inventory_share_pct"], 4.0)
        self.assertAlmostEqual(firm_plan["firm_inventory_gap_pp"], 4.0)

    def test_visible_baseline_reader_uses_declared_card_locations(self) -> None:
        card = _card()
        self.assertAlmostEqual(
            ecology_macro.visible_baseline_from_card(card, "PCE", "1m_change"), 0.4
        )
        self.assertAlmostEqual(
            ecology_macro.visible_baseline_from_card(card, "UNRATE", "latest_level"), 4.0
        )

    def test_firm_plan_contract_excludes_target_month_settled_state(self) -> None:
        parameters = inspect.signature(
            ecology_macro.contemporaneous_firm_plan
        ).parameters
        self.assertIn("inventory_start_units", parameters)
        self.assertIn("baseline_sales_units", parameters)
        self.assertNotIn("inventory_end_units", parameters)
        self.assertNotIn("units_sold", parameters)

    def test_builder_returns_finite_predictions_baselines_and_shadow_fields(self) -> None:
        result = ecology_macro.build_standardized_macro_predictions(
            nominal_consumption_growth_pct=2.0,
            annual_nominal_household_income_expectation_pct=12.0,
            price_growth_pct=1.0,
            personal_saving_rate_change_pp=-0.4,
            revolving_credit_growth_pct=0.8,
            inventory_start_units=4.0,
            baseline_sales_units=100.0,
            compact_macro_information=_card(),
        )
        expected_fields = set(ecology_macro.STANDARDIZED_MACRO_TARGETS)
        expected_fields.update(
            metadata["visible_baseline_field"]
            for metadata in ecology_macro.STANDARDIZED_MACRO_TARGETS.values()
        )
        expected_fields.update(
            {
                "firm_expected_sales_index",
                "firm_target_output_index",
                "firm_required_labor_index",
                "firm_planned_employment_index",
                "firm_price_pressure_pp",
                "firm_inventory_share_pct",
                "firm_inventory_gap_pp",
            }
        )
        self.assertTrue(expected_fields.issubset(result))
        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertAlmostEqual(result["real_consumption_growth_pct"], 100.0 * (1.02 / 1.01 - 1.0))
        self.assertAlmostEqual(
            result["real_disposable_income_growth_pct"],
            100.0 * ((1.12 ** (1.0 / 12.0)) / 1.01 - 1.0),
        )
        self.assertAlmostEqual(result["payroll_growth_pct"], 0.85)
        self.assertAlmostEqual(result["unemployment_rate_level"], 3.184)
        self.assertEqual(result["retail_sales_growth_pct"], result["consumption_growth_pct"])
        self.assertEqual(result["visible_baseline_unemployment_rate_level"], 4.0)

    def test_malformed_or_nonfinite_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            ecology_macro.annual_to_monthly_growth(float("nan"))
        with self.assertRaisesRegex(ValueError, "greater than -100"):
            ecology_macro.fisher_real_growth(1.0, -100.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            ecology_macro.unemployment_level_from_payroll_growth(101.0, 0.1)
        with self.assertRaisesRegex(ValueError, "baseline_sales_units must be positive"):
            ecology_macro.contemporaneous_firm_plan(
                consumption_growth_pct=1.0,
                inventory_start_units=1.0,
                baseline_sales_units=0.0,
            )
        missing = _card()
        del missing["series"]["PCE"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "lacks PCE"):
            ecology_macro.build_standardized_macro_predictions(
                nominal_consumption_growth_pct=1.0,
                annual_nominal_household_income_expectation_pct=3.0,
                price_growth_pct=0.5,
                personal_saving_rate_change_pp=0.0,
                revolving_credit_growth_pct=0.0,
                inventory_start_units=8.0,
                baseline_sales_units=100.0,
                compact_macro_information=missing,
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            ecology_macro.build_standardized_macro_predictions(
                nominal_consumption_growth_pct=float("inf"),
                annual_nominal_household_income_expectation_pct=3.0,
                price_growth_pct=0.5,
                personal_saving_rate_change_pp=0.0,
                revolving_credit_growth_pct=0.0,
                inventory_start_units=8.0,
                baseline_sales_units=100.0,
                compact_macro_information=_card(),
            )


if __name__ == "__main__":
    unittest.main()
