import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from macro_llm_tournament.demand_economy import (
    DemandEconomyClient,
    STEADY_EMPLOYMENT_RATE,
    _quarterly_flow_for_frequency,
    _quarterly_persistence_for_frequency,
    _structural_consumption_policy,
    build_fixture_demand_households,
    default_demand_scenarios,
    run_demand_economy,
)
from macro_llm_tournament.empirical_bridge import BRIDGE_SPEC_VERSION


def _accepted_bridge_profile(coefficients: dict[str, float]) -> dict[str, object]:
    return {
        "schema_version": BRIDGE_SPEC_VERSION,
        "bridge_spec_version": BRIDGE_SPEC_VERSION,
        "status": "accepted",
        "between_coefficients": coefficients,
        "support": {
            "actual_expected_inflation_1y": {"min": -5.0, "max": 15.0},
            "actual_expected_real_income_growth": {"min": -12.0, "max": 12.0},
            "sce_question_unemployment_higher_prob": {"min": 0.0, "max": 100.0},
        },
    }


class DemandFrequencyTests(unittest.TestCase):
    def _run_fixture(
        self,
        *,
        periods_per_year: float | None = None,
        period_overrides: dict[int, dict[str, object]] | None = None,
        initial_environment_override: dict[str, object] | None = None,
        period_count: int = 2,
        household_count: int = 3,
        scenario_index: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
        households = build_fixture_demand_households(household_count)
        with TemporaryDirectory() as temp_dir:
            client = DemandEconomyClient("codex_cli", "gpt-5.5", Path(temp_dir), mode="fixture", max_live_calls=0)
            kwargs: dict[str, object] = {
                "period_count": period_count,
                "feedback_mode": "closed_loop",
            }
            if periods_per_year is not None:
                kwargs["periods_per_year"] = periods_per_year
            if period_overrides is not None:
                kwargs["period_overrides"] = period_overrides
            if initial_environment_override is not None:
                kwargs["initial_environment_override"] = initial_environment_override
            return run_demand_economy(
                households,
                [default_demand_scenarios()[scenario_index]],
                client,
                **kwargs,
            )

    def test_explicit_quarterly_periods_match_legacy_default(self):
        default_run = self._run_fixture(period_count=3)
        explicit_run = self._run_fixture(periods_per_year=4.0, period_count=3)

        for default_frame, explicit_frame in zip(default_run[:5], explicit_run[:5]):
            assert_frame_equal(default_frame, explicit_frame, check_dtype=False, check_like=False)

        default_prompts = [json.dumps(row, sort_keys=True) for row in default_run[5]]
        explicit_prompts = [json.dumps(row, sort_keys=True) for row in explicit_run[5]]
        self.assertEqual(default_prompts, explicit_prompts)

    def test_monthly_scaling_and_state_transition_use_twelfth_of_annual_flows(self):
        households = build_fixture_demand_households(1)
        annual_income = float(households.iloc[0]["annual_income"])
        annual_consumption = float(households.iloc[0]["baseline_consumption_annual"])

        initial, _beliefs, decisions, periods, _accounting, prompts = self._run_fixture(
            periods_per_year=12.0,
            period_count=2,
            household_count=1,
        )

        self.assertAlmostEqual(float(initial.loc[0, "periods_per_year"]), 12.0)
        self.assertAlmostEqual(float(initial.loc[0, "labor_income"]), annual_income / 12.0)
        self.assertAlmostEqual(float(initial.loc[0, "baseline_consumption"]), annual_consumption / 12.0)
        self.assertAlmostEqual(float(periods.loc[0, "periods_per_year"]), 12.0)

        period_one = decisions[decisions["period_index"] == 1].iloc[0]
        employment_factor = float(periods.loc[0, "employment_rate"]) / STEADY_EMPLOYMENT_RATE
        expected_period_one_income = annual_income / 12.0 * np.clip(
            employment_factor * (1.0 + 0.0015 * float(periods.loc[0, "output_gap_pct"])),
            0.70,
            1.25,
        )
        self.assertAlmostEqual(float(period_one["labor_income"]), expected_period_one_income)
        self.assertAlmostEqual(float(period_one["periods_per_year"]), 12.0)

        prompt_cell = prompts[1]["prompt_payload"]["household_cells"][0]
        self.assertAlmostEqual(float(prompt_cell["periods_per_year"]), 12.0)
        self.assertAlmostEqual(float(prompt_cell["period_labor_income"]), expected_period_one_income, places=4)
        self.assertAlmostEqual(float(prompt_cell["period_baseline_consumption"]), annual_consumption / 12.0)
        self.assertNotIn("quarterly_labor_income", prompt_cell)
        self.assertNotIn("quarterly_baseline_consumption", prompt_cell)

    def test_monthly_feedback_preserves_quarterly_persistence_and_long_run_effect(self):
        quarterly_persistence = 0.64
        monthly_persistence = _quarterly_persistence_for_frequency(quarterly_persistence, 12.0)
        self.assertAlmostEqual(monthly_persistence**3, quarterly_persistence)
        self.assertAlmostEqual(
            _quarterly_persistence_for_frequency(quarterly_persistence, 4.0),
            quarterly_persistence,
        )

        quarterly_flow = 0.024
        monthly_flow = _quarterly_flow_for_frequency(quarterly_flow, quarterly_persistence, 12.0)
        self.assertAlmostEqual(
            monthly_flow / (1.0 - monthly_persistence),
            quarterly_flow / (1.0 - quarterly_persistence),
        )
        self.assertAlmostEqual(
            _quarterly_flow_for_frequency(quarterly_flow, quarterly_persistence, 4.0),
            quarterly_flow,
        )

    def test_period_rows_emit_the_post_action_state_used_by_the_next_prompt(self):
        _initial, _beliefs, _decisions, periods, _accounting, prompts = self._run_fixture(
            periods_per_year=12.0,
            period_count=2,
            household_count=3,
        )
        next_environment = prompts[1]["prompt_payload"]["current_environment"]
        first_period = periods.iloc[0]
        for period_column, prompt_field in (
            ("next_output_gap_pct", "output_gap_pct"),
            ("next_employment_rate", "employment_rate"),
            ("next_inflation_rate", "inflation_rate"),
            ("next_policy_rate", "policy_rate"),
        ):
            self.assertAlmostEqual(float(first_period[period_column]), float(next_environment[prompt_field]), places=4)
        self.assertTrue(np.isfinite(float(first_period["next_aggregate_income"])))

    def test_monthly_buffer_month_math_uses_monthly_consumption(self):
        households = build_fixture_demand_households(1)
        monthly_consumption = float(households.iloc[0]["baseline_consumption_annual"]) / 12.0

        _initial, _beliefs, decisions, _periods, _accounting, prompts = self._run_fixture(
            periods_per_year=12.0,
            period_count=2,
            household_count=1,
        )

        period_zero = decisions[decisions["period_index"] == 0].iloc[0]
        expected_after = float(period_zero["liquid_assets_after"]) / monthly_consumption
        self.assertAlmostEqual(float(period_zero["liquid_buffer_months_after"]), expected_after)

        period_one_prompt = prompts[1]["prompt_payload"]["household_cells"][0]
        self.assertAlmostEqual(float(period_one_prompt["liquid_buffer_months"]), expected_after, places=4)

    def test_empirical_bridge_period_deviation_scales_with_monthly_frequency(self):
        profile = _accepted_bridge_profile(
            {
                "actual_expected_inflation_1y": -0.10,
                "actual_expected_real_income_growth": 0.00,
                "sce_question_unemployment_higher_prob": 0.00,
            }
        )
        static = pd.Series(
            {
                "baseline_job_loss_probability": 7.2,
                "unemployment_higher_probability_1y": 30.0,
                "confidence_index": 55.0,
                "income_growth_expectation_1y": 1.0,
                "inflation_expectation_1y": 3.0,
                "target_buffer_months": 3.0,
                "base_saving_rate": 0.10,
                "base_mpc": 0.30,
                "annual_income": 80000.0,
                "baseline_consumption_annual": 12000.0,
                "liquid_assets": 10000.0,
                "debt": 5000.0,
                "debt_service_burden": 0.12,
                "rate_sensitivity": 0.3,
                "income_sensitivity": 0.5,
                "precautionary_sensitivity": 0.4,
            }
        )
        state = {
            "baseline_consumption": 1000.0,
            "liquid_assets": 10000.0,
            "job_loss_probability": 7.2,
            "unemployment_higher_probability_1y": 30.0,
            "confidence_index": 55.0,
            "income_growth_expectation_1y": 1.0,
            "inflation_expectation_1y": 3.0,
            "labor_income": 80000.0 / 12.0,
            "debt": 5000.0,
            "income_group": "middle",
            "liquidity_group": "low",
            "job_loss_risk_type": "low",
        }
        belief = {
            "expected_inflation_next_period": 2.0,
            "expected_income_growth_next_period": 1.0,
            "perceived_job_loss_probability": 7.2,
            "expected_unemployment_higher_probability_next_period": 30.0,
            "confidence_index": 55.0,
            "precautionary_saving_score": 5.0,
        }
        period_state = {"transfer_per_household": 0.0, "policy_rate": 3.0, "period_index": 0, "policy_rate_shock_pp": 0.0}

        policy = _structural_consumption_policy(
            static,
            state,
            belief,
            period_state,
            representative_mpc=None,
            behavior_policy_profile=profile,
            periods_per_year=12.0,
        )

        self.assertAlmostEqual(policy["empirical_bridge_annual_growth_deviation_pp"], 0.10)
        self.assertAlmostEqual(policy["empirical_bridge_period_growth_deviation_pp"], 0.10 / 12.0)
        self.assertAlmostEqual(policy["empirical_bridge_consumption_delta"], 1000.0 * (0.10 / 12.0) / 100.0)
        self.assertAlmostEqual(policy["desired_consumption"], 1000.0 + 1000.0 * (0.10 / 12.0) / 100.0)

    def test_period_overrides_surface_supplied_exogenous_fields_in_prompt_and_period_rows(self):
        overrides = {
            0: {
                "commodity_price_shock_index": 1.25,
                "inflation_rate": 4.5,
                "policy_rate": 5.25,
            }
        }

        _initial, _beliefs, _decisions, periods, _accounting, prompts = self._run_fixture(
            period_overrides=overrides,
            period_count=1,
            household_count=1,
        )

        exogenous = prompts[0]["prompt_payload"]["current_exogenous_conditions"]["supplied_exogenous_conditions"]
        environment = prompts[0]["prompt_payload"]["current_environment"]

        self.assertEqual(exogenous, overrides[0])
        self.assertEqual(environment["supplied_exogenous_conditions"], overrides[0])
        self.assertAlmostEqual(float(environment["inflation_rate"]), 4.5)
        self.assertAlmostEqual(float(environment["policy_rate"]), 5.25)
        self.assertEqual(json.loads(periods.loc[0, "supplied_exogenous_conditions_json"]), overrides[0])

    def test_period_overrides_reject_protected_fields_and_nonfinite_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "protected period identity fields"):
            self._run_fixture(period_overrides={0: {"scenario_id": "mutated"}}, period_count=1, household_count=1)

        with self.assertRaisesRegex(ValueError, "nonfinite numeric value"):
            self._run_fixture(period_overrides={0: {"commodity_price_shock_index": float("nan")}}, period_count=1, household_count=1)

        with self.assertRaisesRegex(ValueError, "nonfinite numeric value"):
            self._run_fixture(period_overrides={0: {"commodity_price_shock_index": np.inf}}, period_count=1, household_count=1)

    def test_initial_environment_override_anchors_the_shared_recursive_state_once(self):
        anchor = {
            "employment_rate": 0.941,
            "inflation_rate": 3.7,
            "policy_rate": 4.6,
            "output_gap_pct": -0.4,
        }
        _initial, _beliefs, _decisions, periods, _accounting, prompts = self._run_fixture(
            initial_environment_override=anchor,
            period_count=2,
            household_count=2,
        )
        first_environment = prompts[0]["prompt_payload"]["current_environment"]
        for key, expected in anchor.items():
            self.assertAlmostEqual(float(first_environment[key]), expected)
        self.assertAlmostEqual(float(periods.iloc[0]["employment_rate"]), anchor["employment_rate"])
        self.assertNotEqual(float(prompts[1]["prompt_payload"]["current_environment"]["inflation_rate"]), anchor["inflation_rate"])

        with self.assertRaisesRegex(ValueError, "between zero and one"):
            self._run_fixture(
                initial_environment_override={"employment_rate": 1.2},
                period_count=1,
                household_count=1,
            )


if __name__ == "__main__":
    unittest.main()
