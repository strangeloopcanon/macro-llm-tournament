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
    _period_one_history_binding,
    _period_one_replay_binding,
    _settled_period_one,
    _validate_source_inputs,
)
from macro_llm_tournament.ecology_households import HOUSEHOLD_PROMPT_VERSION
from macro_llm_tournament.ecology import _state_from_row
from macro_llm_tournament.ecology_transition import transition_household_states
from macro_llm_tournament.persistent_households import HISTORY_COLUMNS


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


def _valid_materialized_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(200):
        row = {column: None for column in HISTORY_COLUMNS}
        row.update(
            {
                "household_id": f"household_{index:03d}",
                "event_date": "2025-09-01",
                "public_availability_date": "2026-06-01",
                "source_name": "SCE respondent microdata",
                "observation_status": "observed",
                "responded": True,
                "attrition_status": "responding",
                "death_status": "alive_no_death_observation",
                "replay_required_from_event_date": "2025-09-01",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, HISTORY_COLUMNS]


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
            _valid_materialized_history().to_csv(history, index=False)
            history_digest = _file_sha256(history)
            input_provenance = {
                name: {"sha256": "a" * 64, "row_count": 1}
                for name in ("base_history", "normalized_sce_microdata", "private_registry")
            }
            history_manifest = {
                "schema_version": "ecology_history_materialization_v1",
                "through_event_month": "2025-09-01",
                "publication_lag_months": 9,
                "selected_household_count": 200,
                "input_provenance": input_provenance,
                "public_history": {
                    "sha256": history_digest,
                    "row_count": 200,
                    "event_count": 1,
                    "columns": list(HISTORY_COLUMNS),
                },
            }
            (root / "history_manifest.json").write_text(
                json.dumps(history_manifest), encoding="utf-8"
            )
            history_binding = {
                "schema_version": "ecology_history_materialization_v1",
                "manifest_sha256": _file_sha256(root / "history_manifest.json"),
                "history_sha256": history_digest,
                "through_event_month": "2025-09-01",
                "publication_lag_months": 9,
                "input_provenance": input_provenance,
            }
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
                "replay_equivalence_sha256": "period-one-equivalence",
                "household_input_sha256": _file_sha256(households),
                "history_input_sha256": _file_sha256(history),
                "history_materialization": history_binding,
                "artifacts": artifacts,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            _load_period_one(root)
            binding = _period_one_replay_binding(manifest)
            replay_manifest = dict(manifest, mode="replay", created_at_utc="later")
            self.assertEqual(binding, _period_one_replay_binding(replay_manifest))
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

            missing_history_binding = dict(manifest)
            missing_history_binding.pop("history_materialization")
            (root / "manifest.json").write_text(
                json.dumps(missing_history_binding), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "history materialization provenance"):
                _load_period_one(root)

            malformed_history_binding = dict(manifest)
            malformed_history_binding["history_materialization"] = {
                **history_binding,
                "schema_version": "wrong",
            }
            with self.assertRaisesRegex(ValueError, "schema is unsupported"):
                _period_one_history_binding(malformed_history_binding)

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
            "committed_consumption_usd": 2_150.0,
            "discretionary_consumption_usd": 850.0,
            "one_off_purchase_usd": 100.0,
        }])
        feedback = compute_aggregate_firm_feedback(_period_one(demand_units=1_200.0, sales_units=1_200.0, opening_inventory_units=280.0, closing_inventory_units=80.0))
        initial_states = [_state_from_row(row) for _, row in source.iterrows()]
        states, transitions = transition_household_states(
            initial_states, decisions, feedback
        )
        state = states[0]
        self.assertEqual(state.deposit_balance_usd, 5_250.0)
        self.assertEqual(state.revolving_debt_usd, 1_900.0)
        self.assertEqual(state.baseline_monthly_consumption_usd, 3_000.0)
        self.assertEqual(state.baseline_committed_consumption_usd, 2_150.0)
        self.assertEqual(state.baseline_discretionary_consumption_usd, 850.0)
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

    def test_firm_feedback_uses_settled_not_desired_household_demand(self) -> None:
        source = pd.DataFrame([{
            "type_id": "h1",
            "employment_status": "employed",
            "baseline_consumption_annual": 12_000.0,
            "liquid_deposits_usd": 1_000.0,
            "population_weight": 1.0,
        }])
        states = [_state_from_row(row) for _, row in source.iterrows()]
        decisions = pd.DataFrame([{
            "household_id": "h1",
            "desired_consumption_usd": 1_500.0,
            "consumption_usd": 800.0,
        }])
        period_one = _settled_period_one(
            states,
            decisions,
            {
                "current_price_per_unit_usd": 2.0,
                "units_sold": 400.0,
                "inventory_start_units": 80.0,
                "inventory_end_units": 80.0,
                "output_units": 400.0,
            },
        )
        self.assertEqual(period_one.demand_units, 400.0)
        self.assertNotEqual(period_one.demand_units, 750.0)

    def test_transition_rejects_unreconciled_consumption_categories(self) -> None:
        source = pd.DataFrame([{
            "type_id": "h1",
            "employment_status": "employed",
            "baseline_consumption_annual": 12_000.0,
            "liquid_deposits_usd": 1_000.0,
            "population_weight": 1.0,
        }])
        states = [_state_from_row(row) for _, row in source.iterrows()]
        decisions = pd.DataFrame([{
            "household_id": "h1",
            "deposit_balance_end_usd": 1_000.0,
            "revolving_debt_end_usd": 0.0,
            "consumption_usd": 1_000.0,
            "committed_consumption_usd": 600.0,
            "discretionary_consumption_usd": 300.0,
            "one_off_purchase_usd": 50.0,
        }])
        with self.assertRaisesRegex(ValueError, "categories do not reconcile"):
            transition_household_states(
                states,
                decisions,
                compute_aggregate_firm_feedback(_period_one()),
            )


if __name__ == "__main__":
    unittest.main()
