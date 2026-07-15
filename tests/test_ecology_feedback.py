from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from macro_llm_tournament.ecology_feedback import (
    AggregateFirmPeriodOne,
    HouseholdFamilyIncome,
    apply_producer_income_feedback,
    build_simulated_environment_payload,
    compute_aggregate_firm_feedback,
)
from macro_llm_tournament.ecology import _artifact_sha256, _file_sha256
from macro_llm_tournament.ecology_feedback_runner import (
    _load_period_one,
    _period_two_states,
    _validate_source_inputs,
)
from macro_llm_tournament.ecology_households import HOUSEHOLD_PROMPT_VERSION


def _period_one(
    *,
    demand_units: float = 1_000.0,
    sales_units: float = 1_000.0,
    opening_inventory_units: float = 80.0,
    closing_inventory_units: float = 80.0,
) -> AggregateFirmPeriodOne:
    return AggregateFirmPeriodOne(
        demand_units=demand_units,
        sales_units=sales_units,
        opening_inventory_units=opening_inventory_units,
        closing_inventory_units=closing_inventory_units,
        base_output_units=1_000.0,
        productivity_index=1.0,
        producer_employment_index=1.0,
        producer_wage_index=1.0,
    )


class AggregateFirmFeedbackTests(unittest.TestCase):
    def test_period_one_artifacts_and_source_inputs_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "median_economy.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [{"scenario": "median", "household_id": "h1"}]
            ).to_csv(root / "household_decisions.csv", index=False)
            (root / "normalized_origin_information.json").write_text(
                json.dumps({"as_of_date": "2026-07-15"}), encoding="utf-8"
            )
            households = root / "households.csv"
            history = root / "history.csv"
            households.write_text("type_id\nh1\n", encoding="utf-8")
            history.write_text("household_id\nh1\n", encoding="utf-8")
            artifacts = {
                name: _artifact_sha256(root / name)
                for name in (
                    "median_economy.json",
                    "household_decisions.csv",
                    "normalized_origin_information.json",
                )
            }
            manifest = {
                "accounting_passed": True,
                "household_prompt_version": HOUSEHOLD_PROMPT_VERSION,
                "household_count": 1,
                "household_input_sha256": _file_sha256(households),
                "history_input_sha256": _file_sha256(history),
                "artifacts": artifacts,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            _load_period_one(root)
            self.assertEqual(
                _validate_source_inputs(manifest, households, history),
                (_file_sha256(households), _file_sha256(history)),
            )
            (root / "median_economy.json").write_text(
                '{"tampered": true}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                _load_period_one(root)
            wrong = root / "wrong.csv"
            wrong.write_text("type_id\nh2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "household input"):
                _validate_source_inputs(manifest, wrong, history)

    def test_positive_demand_lifts_output_labor_employment_and_wages(self) -> None:
        feedback = compute_aggregate_firm_feedback(
            _period_one(
                demand_units=1_200.0,
                sales_units=1_200.0,
                opening_inventory_units=280.0,
                closing_inventory_units=80.0,
            )
        )

        self.assertAlmostEqual(feedback.expected_sales_units, 1_200.0)
        self.assertAlmostEqual(feedback.target_inventory_units, 96.0)
        self.assertAlmostEqual(feedback.inventory_gap_units, 16.0)
        self.assertAlmostEqual(feedback.planned_output_units, 1_205.6)
        self.assertAlmostEqual(feedback.required_labor_index, 1.2056)
        self.assertAlmostEqual(feedback.producer_employment_index, 1.0514)
        self.assertAlmostEqual(feedback.wage_change_pct, 0.00514)
        self.assertAlmostEqual(feedback.producer_wage_index, 1.00514)

    def test_negative_demand_reduces_plan_and_respects_wage_floor(self) -> None:
        feedback = compute_aggregate_firm_feedback(
            _period_one(
                demand_units=0.0,
                sales_units=0.0,
                opening_inventory_units=1_000.0,
                closing_inventory_units=2_000.0,
            )
        )

        self.assertEqual(feedback.expected_sales_units, 0.0)
        self.assertEqual(feedback.target_inventory_units, 0.0)
        self.assertEqual(feedback.planned_output_units, 0.0)
        self.assertAlmostEqual(feedback.producer_employment_index, 0.75)
        self.assertAlmostEqual(feedback.wage_change_pct, -0.02)
        self.assertAlmostEqual(feedback.producer_wage_index, 0.98)

    def test_period_one_stock_continuity_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "stock continuity"):
            compute_aggregate_firm_feedback(
                _period_one(
                    demand_units=1_000.0,
                    sales_units=1_000.0,
                    opening_inventory_units=80.0,
                    closing_inventory_units=70.0,
                )
            )

    def test_income_reconciles_and_preserves_nonwage_components(self) -> None:
        feedback = compute_aggregate_firm_feedback(
            _period_one(
                demand_units=800.0,
                sales_units=800.0,
                opening_inventory_units=80.0,
                closing_inventory_units=280.0,
            )
        )
        income = HouseholdFamilyIncome(
            respondent_employment_share=0.0,
            family_wage_income_usd=1_000.0,
            business_income_usd=300.0,
            nonwage_income_usd=200.0,
            transfer_income_usd=100.0,
        )

        updated = apply_producer_income_feedback(income, feedback)

        self.assertAlmostEqual(
            updated.family_wage_income_usd,
            1_000.0 * feedback.employment_index_change * feedback.wage_index_change,
        )
        self.assertEqual(updated.business_income_usd, 300.0)
        self.assertEqual(updated.nonwage_income_usd, 200.0)
        self.assertEqual(updated.transfer_income_usd, 100.0)
        self.assertEqual(updated.respondent_employment_share, 0.0)
        self.assertAlmostEqual(
            updated.gross_income_usd,
            updated.family_wage_income_usd + 300.0 + 200.0 + 100.0,
        )

    def test_environment_payload_is_separately_labelled(self) -> None:
        feedback = compute_aggregate_firm_feedback(_period_one())

        payload = build_simulated_environment_payload(feedback)

        self.assertEqual(set(payload), {"simulated_environment"})
        environment = payload["simulated_environment"]
        self.assertEqual(environment["source"], "aggregate_firm_feedback")
        self.assertEqual(environment["period"], 2)
        self.assertEqual(
            environment["producer"]["planned_output_units"],
            feedback.planned_output_units,
        )

    def test_period_two_state_carries_stocks_and_updates_only_family_wages(self) -> None:
        source = pd.DataFrame([{
            "type_id": "h1",
            "employment_status": "employed",
            "monthly_wage_income_usd": 4_000.0,
            "monthly_business_income_usd": 500.0,
            "monthly_earned_income_usd": 4_500.0,
            "monthly_nonwage_income_usd": 200.0,
            "monthly_transfers_benefits_usd": 100.0,
            "baseline_committed_consumption_monthly_usd": 2_000.0,
            "baseline_discretionary_consumption_monthly_usd": 1_000.0,
            "liquid_deposits_usd": 5_000.0,
            "revolving_debt_usd": 2_000.0,
            "revolving_credit_limit_usd": 6_000.0,
            "population_weight": 1.0,
        }])
        decisions = pd.DataFrame([{
            "household_id": "h1",
            "deposit_balance_end_usd": 5_250.0,
            "revolving_debt_end_usd": 1_900.0,
            "consumption_usd": 3_100.0,
        }])
        feedback = compute_aggregate_firm_feedback(_period_one(demand_units=1_200.0, sales_units=1_200.0, opening_inventory_units=280.0, closing_inventory_units=80.0))
        states, transitions = _period_two_states(source, decisions, feedback)
        state = states[0]
        self.assertEqual(state.deposit_balance_usd, 5_250.0)
        self.assertEqual(state.revolving_debt_usd, 1_900.0)
        self.assertEqual(state.employment_share, 1.0)
        self.assertEqual(state.monthly_family_business_income_usd, 500.0)
        self.assertEqual(state.monthly_nonwage_income_usd, 200.0)
        self.assertEqual(state.monthly_transfer_income_usd, 100.0)
        self.assertAlmostEqual(
            state.monthly_household_earned_income_usd,
            state.monthly_family_wage_income_usd + 500.0,
        )
        self.assertEqual(
            transitions[0]["deposit_balance_period_1_close_period_2_open_usd"],
            5_250.0,
        )


if __name__ == "__main__":
    unittest.main()
