from __future__ import annotations

import unittest

from macro_llm_tournament.ecology_engine import (
    annual_probability_to_monthly,
    build_household_trajectory,
    run_monthly_ecology,
)
from macro_llm_tournament.ecology_models import (
    ACCOUNTING_TOLERANCE,
    CreditIntermediaryState,
    EmployerState,
    HouseholdResponse,
    HouseholdState,
    QuantileTriplet,
    household_response_schema,
)


def _response(
    *,
    inflation: tuple[float, float, float] = (2.0, 3.0, 4.5),
    income: tuple[float, float, float] = (-3.0, 1.0, 4.0),
    job_loss: tuple[float, float, float] = (5.0, 18.0, 35.0),
    consumption: tuple[float, float, float] = (-12.0, 2.0, 10.0),
    work_hours: tuple[float, float, float] = (120.0, 150.0, 170.0),
    search_hours: tuple[float, float, float] = (2.0, 5.0, 12.0),
    target_buffer_months: float = 1.8,
    buffer_contribution_intent_usd: float = 150.0,
    debt_payment_intent_usd: float = 50.0,
    borrowing_intent_usd: float = 0.0,
) -> HouseholdResponse:
    return HouseholdResponse(
        expected_inflation_pct=QuantileTriplet(*inflation),
        expected_income_growth_pct=QuantileTriplet(*income),
        job_loss_probability_pct=QuantileTriplet(*job_loss),
        planned_consumption_change_pct=QuantileTriplet(*consumption),
        planned_work_hours=QuantileTriplet(*work_hours),
        planned_job_search_hours=QuantileTriplet(*search_hours),
        target_buffer_months=target_buffer_months,
        buffer_contribution_intent_usd=buffer_contribution_intent_usd,
        debt_payment_intent_usd=debt_payment_intent_usd,
        borrowing_intent_usd=borrowing_intent_usd,
    )


class EcologyEngineTests(unittest.TestCase):
    def test_schema_exposes_required_quantile_fields(self) -> None:
        schema = household_response_schema()
        self.assertEqual(schema["schema_version"], "household_first_response_v1")
        self.assertIn("expected_inflation_pct", schema["required"])
        self.assertEqual(
            schema["trajectory_mapping"]["downside"]["inflation_pct"],
            "expected_inflation_pct.p90",
        )
        self.assertEqual(
            schema["trajectory_mapping"]["upside"]["planned_job_search_hours"],
            "planned_job_search_hours.p10",
        )

    def test_quantiles_must_be_monotone(self) -> None:
        response = _response(inflation=(3.0, 2.0, 4.0))
        with self.assertRaisesRegex(ValueError, "expected_inflation_pct must satisfy"):
            build_household_trajectory(response)

    def test_trajectory_mapping_is_downside_to_upside(self) -> None:
        trajectory = build_household_trajectory(_response())
        self.assertEqual(trajectory.downside.inflation_pct, 4.5)
        self.assertEqual(trajectory.downside.income_growth_pct, -3.0)
        self.assertEqual(trajectory.median.planned_work_hours, 150.0)
        self.assertEqual(trajectory.upside.planned_work_hours, 170.0)
        self.assertEqual(trajectory.upside.job_loss_probability_pct, 5.0)

    def test_engine_reconciles_households_bank_and_employer(self) -> None:
        households = [
            HouseholdState(
                household_id="h1",
                employer_id="firm",
                deposit_balance_usd=600.0,
                revolving_debt_usd=250.0,
                revolving_credit_limit_usd=1_000.0,
                hourly_wage_usd=24.0,
                baseline_monthly_hours=160.0,
                baseline_monthly_consumption_usd=900.0,
                layoff_threshold_pct=60.0,
            ),
            HouseholdState(
                household_id="h2",
                employer_id="firm",
                deposit_balance_usd=150.0,
                revolving_debt_usd=500.0,
                revolving_credit_limit_usd=900.0,
                hourly_wage_usd=20.0,
                baseline_monthly_hours=150.0,
                baseline_monthly_consumption_usd=850.0,
                layoff_threshold_pct=40.0,
            ),
        ]
        responses = {
            "h1": _response(
                consumption=(-8.0, 6.0, 12.0),
                work_hours=(145.0, 160.0, 172.0),
                search_hours=(1.0, 3.0, 6.0),
                debt_payment_intent_usd=120.0,
            ),
            "h2": _response(
                job_loss=(35.0, 55.0, 82.0),
                consumption=(-35.0, -10.0, 4.0),
                work_hours=(0.0, 40.0, 120.0),
                search_hours=(10.0, 24.0, 40.0),
                target_buffer_months=0.8,
                buffer_contribution_intent_usd=10.0,
                debt_payment_intent_usd=10.0,
                borrowing_intent_usd=260.0,
            ),
        }
        employer = EmployerState(
            employer_id="firm",
            productivity_per_hour=0.25,
            monthly_capacity_units=80.0,
            inventory_units=5.0,
            price_per_unit_usd=20.0,
            target_headcount=3,
            wage_offer_usd=22.0,
            target_inventory_units=20.0,
            fixed_cost_usd=500.0,
            variable_nonlabor_cost_per_unit_usd=3.0,
            vacancy_cost_per_opening_usd=50.0,
            inventory_carry_cost_per_unit_usd=0.5,
        )
        credit = CreditIntermediaryState(
            intermediary_id="bank",
            annual_interest_rate_pct=18.0,
            minimum_payment_rate_pct=4.0,
            minimum_payment_floor_usd=25.0,
            loss_given_default_pct=50.0,
            new_lending_budget_usd=180.0,
        )

        result = run_monthly_ecology(households, responses, employer, credit)

        self.assertEqual(result.schema_version, "household_first_monthly_ecology_v1")
        self.assertEqual(len(result.households), 2)
        self.assertLessEqual(result.max_abs_residual(), ACCOUNTING_TOLERANCE)
        self.assertLess(result.credit.rationing_ratio, 1.0)
        self.assertAlmostEqual(result.employer.employment_count, 2.0)
        self.assertAlmostEqual(result.employer.vacancies, 1.0)
        self.assertGreater(result.credit.borrowing_total_usd, 0.0)
        self.assertGreater(result.credit.interest_income_usd, 0.0)
        self.assertNotEqual(result.employer.output_units, result.aggregate_consumption_usd)
        self.assertNotEqual(result.employer.output_units, result.employer.units_sold)
        self.assertAlmostEqual(
            result.employer.inventory_start_units + result.employer.output_units,
            result.employer.units_sold + result.employer.inventory_end_units,
        )
        h1 = next(row for row in result.households if row.household_id == "h1")
        h2 = next(row for row in result.households if row.household_id == "h2")
        self.assertTrue(h1.realized_job_loss)
        self.assertTrue(h2.realized_job_loss)
        self.assertAlmostEqual(h1.job_loss_share, annual_probability_to_monthly(18.0))
        self.assertAlmostEqual(h2.job_loss_share, annual_probability_to_monthly(55.0))
        self.assertAlmostEqual(h2.actual_hours_worked, 40.0)
        self.assertGreaterEqual(h2.actual_job_search_hours, 12.0)
        self.assertGreater(h2.borrowing_requested_usd, h2.borrowing_usd)
        self.assertGreaterEqual(h1.deposit_balance_end_usd, 0.0)
        self.assertGreaterEqual(h2.deposit_balance_end_usd, 0.0)
        self.assertGreaterEqual(h1.revolving_debt_end_usd, 0.0)
        self.assertGreaterEqual(h2.revolving_debt_end_usd, 0.0)
        categories = {flow.category for flow in result.counterparties}
        self.assertTrue(
            {"wages", "consumption_spending", "revolving_borrowing", "debt_payment"}.issubset(
                categories
            )
        )
        self.assertEqual(
            result.credit.deposits_end_usd,
            sum(row.deposit_balance_end_usd for row in result.households),
        )
        self.assertEqual(
            result.credit.revolving_debt_end_usd,
            sum(row.revolving_debt_end_usd for row in result.households),
        )

    def test_credit_rationing_cannot_make_consumption_or_sales_negative(self) -> None:
        household = HouseholdState(
            household_id="h1",
            employer_id="firm",
            deposit_balance_usd=0.0,
            revolving_debt_usd=500.0,
            revolving_credit_limit_usd=2_000.0,
            hourly_wage_usd=0.0,
            baseline_monthly_hours=0.0,
            baseline_monthly_consumption_usd=500.0,
            layoff_threshold_pct=1.0,
        )
        response = _response(
            job_loss=(50.0, 80.0, 95.0),
            consumption=(-20.0, 0.0, 10.0),
            work_hours=(0.0, 0.0, 0.0),
            debt_payment_intent_usd=400.0,
            borrowing_intent_usd=1_000.0,
        )
        result = run_monthly_ecology(
            [household],
            {"h1": response},
            EmployerState(
                employer_id="firm",
                productivity_per_hour=1.0,
                monthly_capacity_units=1_000.0,
                inventory_units=500.0,
                price_per_unit_usd=1.0,
                target_headcount=0,
                wage_offer_usd=0.0,
            ),
            CreditIntermediaryState(
                intermediary_id="bank",
                annual_interest_rate_pct=20.0,
                minimum_payment_rate_pct=4.0,
                new_lending_budget_usd=0.0,
            ),
        )
        row = result.households[0]
        self.assertGreaterEqual(row.consumption_usd, 0.0)
        self.assertGreaterEqual(result.employer.revenue_usd, 0.0)
        self.assertGreaterEqual(result.employer.units_sold, 0.0)
        self.assertLessEqual(result.max_abs_residual(), ACCOUNTING_TOLERANCE)

    def test_vacancy_and_job_search_can_move_an_unemployed_household_into_work(self) -> None:
        household = HouseholdState(
            household_id="h1",
            employer_id="firm",
            deposit_balance_usd=500.0,
            revolving_debt_usd=0.0,
            revolving_credit_limit_usd=1_000.0,
            hourly_wage_usd=20.0,
            baseline_monthly_hours=0.0,
            baseline_monthly_consumption_usd=600.0,
        )
        response = _response(
            job_loss=(10.0, 20.0, 30.0),
            work_hours=(0.0, 0.0, 160.0),
            search_hours=(5.0, 30.0, 60.0),
        )
        result = run_monthly_ecology(
            [household],
            {"h1": response},
            EmployerState(
                employer_id="firm",
                productivity_per_hour=1.0,
                monthly_capacity_units=500.0,
                inventory_units=100.0,
                price_per_unit_usd=1.0,
                target_headcount=1,
                wage_offer_usd=25.0,
            ),
            CreditIntermediaryState(
                intermediary_id="bank",
                annual_interest_rate_pct=10.0,
                minimum_payment_rate_pct=3.0,
            ),
            scenario="upside",
        )
        self.assertGreater(result.households[0].actual_hours_worked, 0.0)
        self.assertEqual(result.employer.employment_count, 1)
        self.assertAlmostEqual(result.households[0].wage_income_usd, 25.0 * 160.0)

    def test_population_weights_drive_institutional_aggregation(self) -> None:
        households = [
            HouseholdState(
                household_id="large_type",
                employer_id="firm",
                deposit_balance_usd=0.0,
                revolving_debt_usd=0.0,
                revolving_credit_limit_usd=0.0,
                hourly_wage_usd=20.0,
                baseline_monthly_hours=160.0,
                baseline_monthly_consumption_usd=100.0,
                population_weight=0.9,
            ),
            HouseholdState(
                household_id="small_type",
                employer_id="firm",
                deposit_balance_usd=0.0,
                revolving_debt_usd=0.0,
                revolving_credit_limit_usd=0.0,
                hourly_wage_usd=20.0,
                baseline_monthly_hours=160.0,
                baseline_monthly_consumption_usd=1_000.0,
                population_weight=0.1,
            ),
        ]
        responses = {
            row.household_id: _response(
                job_loss=(0.0, 0.0, 0.0),
                consumption=(0.0, 0.0, 0.0),
                work_hours=(160.0, 160.0, 160.0),
                target_buffer_months=0.0,
                buffer_contribution_intent_usd=0.0,
            )
            for row in households
        }
        result = run_monthly_ecology(
            households,
            responses,
            EmployerState(
                employer_id="firm",
                productivity_per_hour=10.0,
                monthly_capacity_units=10_000.0,
                inventory_units=0.0,
                price_per_unit_usd=1.0,
                target_headcount=2.0,
                wage_offer_usd=20.0,
            ),
            CreditIntermediaryState(
                intermediary_id="bank",
                annual_interest_rate_pct=0.0,
                minimum_payment_rate_pct=0.0,
            ),
        )
        self.assertAlmostEqual(result.employer.revenue_usd, 380.0)
        self.assertAlmostEqual(result.aggregate_consumption_usd, 380.0)
        self.assertLessEqual(result.max_abs_residual(), ACCOUNTING_TOLERANCE)

    def test_job_loss_probability_enters_smoothly_without_threshold_cliff(self) -> None:
        household = HouseholdState(
            household_id="h1",
            employer_id="firm",
            deposit_balance_usd=0.0,
            revolving_debt_usd=0.0,
            revolving_credit_limit_usd=0.0,
            hourly_wage_usd=20.0,
            baseline_monthly_hours=160.0,
            baseline_monthly_consumption_usd=100.0,
        )
        employer = EmployerState(
            employer_id="firm",
            productivity_per_hour=10.0,
            monthly_capacity_units=2_000.0,
            inventory_units=0.0,
            price_per_unit_usd=1.0,
            target_headcount=0.0,
            wage_offer_usd=20.0,
        )
        credit = CreditIntermediaryState(
            intermediary_id="bank",
            annual_interest_rate_pct=0.0,
            minimum_payment_rate_pct=0.0,
        )
        results = []
        for probability in (49.9, 50.0):
            results.append(
                run_monthly_ecology(
                    [household],
                    {
                        "h1": _response(
                            job_loss=(probability, probability, probability),
                            consumption=(0.0, 0.0, 0.0),
                            work_hours=(160.0, 160.0, 160.0),
                            search_hours=(0.0, 0.0, 0.0),
                        )
                    },
                    employer,
                    credit,
                )
            )
        expected_share_difference = (
            (1.0 - 0.499) ** (1.0 / 12.0)
            - (1.0 - 0.5) ** (1.0 / 12.0)
        )
        self.assertAlmostEqual(
            results[0].households[0].employment_share_end
            - results[1].households[0].employment_share_end,
            expected_share_difference,
        )
        self.assertAlmostEqual(
            results[0].households[0].wage_income_usd
            - results[1].households[0].wage_income_usd,
            expected_share_difference * 160.0 * 20.0,
        )

    def test_discretionary_debt_goal_does_not_displace_affordable_consumption(self) -> None:
        household = HouseholdState(
            household_id="h1",
            employer_id="firm",
            deposit_balance_usd=1_000.0,
            revolving_debt_usd=5_000.0,
            revolving_credit_limit_usd=7_000.0,
            hourly_wage_usd=25.0,
            baseline_monthly_hours=160.0,
            baseline_monthly_consumption_usd=2_000.0,
        )
        response = _response(
            consumption=(-5.0, 0.0, 5.0),
            work_hours=(150.0, 160.0, 170.0),
            debt_payment_intent_usd=2_500.0,
            buffer_contribution_intent_usd=500.0,
        )
        result = run_monthly_ecology(
            [household],
            {"h1": response},
            EmployerState(
                employer_id="firm",
                productivity_per_hour=20.0,
                monthly_capacity_units=4_000.0,
                inventory_units=500.0,
                price_per_unit_usd=1.0,
                target_headcount=1,
                wage_offer_usd=25.0,
            ),
            CreditIntermediaryState(
                intermediary_id="bank",
                annual_interest_rate_pct=18.0,
                minimum_payment_rate_pct=3.0,
            ),
        )
        row = result.households[0]
        self.assertAlmostEqual(row.consumption_usd, row.desired_consumption_usd)
        self.assertGreaterEqual(row.debt_payment_usd, row.minimum_payment_due_usd)
        self.assertLess(row.debt_payment_usd, response.debt_payment_intent_usd)


if __name__ == "__main__":
    unittest.main()
