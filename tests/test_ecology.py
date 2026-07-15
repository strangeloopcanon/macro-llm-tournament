from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path

import pandas as pd

from macro_llm_tournament.ecology import (
    LIVE_REFERENCE_SCHEMA_VERSION,
    PROJECT_ROOT,
    _coverage_sample,
    _initial_credit,
    _initial_employer,
    _state_from_row,
    _live_reference_path,
    _read_live_reference,
    _weighted_macro,
    build_arg_parser,
    run,
)
from macro_llm_tournament.ecology_engine import run_monthly_ecology
from macro_llm_tournament.ecology_households import (
    HouseholdElicitor,
    LiveCallBudget,
    canonical_sha256,
    fixture_response,
    household_card,
    household_request_identity,
    normalize_household_payload,
)
from macro_llm_tournament.ecology_models import (
    CreditIntermediaryState,
    EmployerState,
    HouseholdPolicyBranch,
    HouseholdResponse,
    QuantileTriplet,
    household_response_schema,
)

FIXTURE_ROOT = PROJECT_ROOT / "examples/ecology_fixture"
FIXTURE_HOUSEHOLDS = FIXTURE_ROOT / "households.csv"
FIXTURE_HISTORY = FIXTURE_ROOT / "history.csv"
FIXTURE_BUNDLE = FIXTURE_ROOT / "origin_snapshot.json"


class HouseholdEcologyTests(unittest.TestCase):
    def test_unknown_employment_uses_scf_donor_state_and_earned_income(self) -> None:
        not_working = _state_from_row(
            pd.Series(
                {
                    "type_id": "h_not_working",
                    "annual_income_usd": 48_000.0,
                    "monthly_earned_income_usd": 0.0,
                    "employment_status": "unknown",
                    "scf_donor_employment_group": "not_employed",
                    "baseline_consumption_annual": 36_000.0,
                }
            )
        )
        self.assertEqual(not_working.employment_share, 0.0)
        self.assertEqual(not_working.baseline_monthly_hours, 0.0)
        working = _state_from_row(
            pd.Series(
                {
                    "type_id": "h_working",
                    "annual_income_usd": 48_000.0,
                    "monthly_wage_income_usd": 0.0,
                    "monthly_earned_income_usd": 3_200.0,
                    "employment_status": "unknown",
                    "scf_donor_employment_group": "employed",
                    "baseline_consumption_annual": 36_000.0,
                }
            )
        )
        self.assertEqual(working.employment_share, 1.0)
        self.assertAlmostEqual(working.hourly_wage_usd, 0.0)
        self.assertAlmostEqual(working.monthly_household_earned_income_usd, 3_200.0)

    def test_nonworking_respondent_keeps_other_household_earnings(self) -> None:
        state = _state_from_row(
            pd.Series(
                {
                    "type_id": "h_nonworking_respondent",
                    "annual_income_usd": 72_000.0,
                    "monthly_earned_income_usd": 5_000.0,
                    "monthly_nonwage_income_usd": 500.0,
                    "employment_status": "unemployed",
                    "baseline_consumption_annual": 48_000.0,
                }
            )
        )
        self.assertEqual(state.employment_share, 0.0)
        self.assertEqual(state.baseline_monthly_hours, 0.0)
        self.assertAlmostEqual(state.monthly_household_earned_income_usd, 5_000.0)
        self.assertAlmostEqual(state.monthly_nonwage_income_usd, 500.0)

    def test_cli_requires_all_input_paths(self) -> None:
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--origin", "2026-05-01",
                "--mode", "fixture",
                "--output-dir", "out",
            ])

    def test_canary_sample_is_stable_and_covers_state_categories(self) -> None:
        frame = pd.read_csv(FIXTURE_HOUSEHOLDS)
        first = _coverage_sample(frame, 12)
        second = _coverage_sample(frame.sample(frac=1.0, random_state=7), 12)
        self.assertEqual(first["type_id"].tolist(), second["type_id"].tolist())
        for column in ("income_group", "liquidity_group", "age_bucket", "employment_status"):
            self.assertEqual(set(first[column].astype(str)), set(frame[column].astype(str)))

    def test_card_is_private_and_origin_bounded(self) -> None:
        state = {
            "type_id": "h1",
            "annual_income": 60_000,
            "baseline_consumption_annual": 42_000,
            "liquid_assets": 5_000,
            "debt": 2_000,
        }
        origin = {
            "origin_month": "2026-05-01",
            "as_of_date": "2026-05-15",
            "origin_visible_macro_context": {},
            "origin_visible_macro_history": {},
        }
        card = household_card(state, origin=origin, own_history=[{"household_id": "h1", "event_date": "2025-04-01"}])
        text = json.dumps(card)
        self.assertIn("h1", text)
        self.assertNotIn("h2", text)
        self.assertNotIn("actual_", text)
        self.assertEqual(
            card["household"]["current_state"]["provenance"],
            "survey_seeded_initial_state",
        )

    def test_card_uses_supplied_fixed_household_state(self) -> None:
        card = household_card(
            {
                "type_id": "h1",
                "annual_income": 60_000,
                "baseline_consumption_annual": 42_000,
                "employment_status": "employed",
                "baseline_monthly_consumption_usd": 3_125.0,
                "baseline_monthly_hours": 0.0,
                "hourly_wage_usd": 28.0,
                "deposit_balance_usd": 4_200.0,
                "revolving_debt_usd": 1_800.0,
                "revolving_credit_limit_usd": 9_000.0,
                "state_provenance": "fixed_survey_scf_anchor",
            },
            origin={
                "origin_month": "2026-06-01",
                "as_of_date": "2026-06-15",
                "origin_visible_macro_context": {},
                "origin_visible_macro_history": {},
            },
            own_history=[],
        )
        current = card["household"]["current_state"]
        self.assertEqual(current["monthly_consumption"], 3_125.0)
        self.assertEqual(current["hours_worked"], 0.0)
        self.assertFalse(current["employed"])
        self.assertEqual(current["credit_limit"], 9_000.0)
        self.assertEqual(current["hourly_wage"], 28.0)
        self.assertEqual(current["annualized_wage_income"], 0.0)
        self.assertEqual(current["provenance"], "fixed_survey_scf_anchor")

    def test_card_uses_latest_available_observed_prior_without_spending_anchor(self) -> None:
        card = household_card(
            {
                "type_id": "h1",
                "employment_status": "employed",
                "annual_income": 60_000,
                "baseline_consumption_annual": 42_000,
                "inflation_expectation_1y": 9.0,
                "personal_job_loss_probability_1y": 7.0,
            },
            origin={
                "origin_month": "2026-04-01",
                "as_of_date": "2026-04-15",
                "routine_nominal_spending_drift_pct": 0.7,
                "origin_visible_macro_context": {},
                "origin_visible_macro_history": {},
            },
            own_history=[
                {
                    "event_date": "2025-04-01",
                    "public_availability_date": "2026-01-01",
                    "observation_status": "observed",
                    "responded": True,
                    "actual_expected_inflation_1y": 3.0,
                    "sce_personal_job_loss_probability_1y": 8.0,
                },
                {
                    "event_date": "2025-05-01",
                    "public_availability_date": "2026-02-01",
                    "observation_status": "observed",
                    "responded": True,
                    "actual_expected_inflation_1y": 4.0,
                    "sce_personal_job_loss_probability_1y": None,
                },
                {
                    "event_date": "2025-06-01",
                    "public_availability_date": "2026-03-01",
                    "observation_status": "nonresponse",
                    "responded": False,
                    "actual_expected_inflation_1y": None,
                },
            ],
        )

        self.assertEqual(card["household"]["previous_beliefs"]["source_event_date"], "2025-05-01")
        self.assertEqual(card["household"]["previous_beliefs"]["inflation_expectation_1y"], 4.0)
        self.assertEqual(
            card["household"]["previous_beliefs"]["personal_job_loss_probability_1y"],
            8.0,
        )
        self.assertEqual(
            card["household"]["previous_beliefs"]["field_source_event_dates"]
            ["personal_job_loss_probability_1y"],
            "2025-04-01",
        )
        current = card["household"]["current_state"]
        self.assertEqual(current["recent_visible_aggregate_consumption_growth_pct"], 0.7)
        self.assertNotIn("status_quo_next_month_committed_consumption", current)
        self.assertNotIn("status_quo_next_month_discretionary_consumption", current)

    def test_annual_income_matches_twelve_simulated_work_months(self) -> None:
        state = _state_from_row(pd.Series({
            "type_id": "h1",
            "annual_income": 72_000.0,
            "baseline_consumption_annual": 42_000.0,
            "employment_status": "employed",
            "liquid_assets": 1_000.0,
            "debt": 0.0,
        }))
        self.assertAlmostEqual(state.hourly_wage_usd * state.baseline_monthly_hours * 12.0, 72_000.0)

    def test_unknown_cohort_member_with_zero_observed_wage_starts_out_of_work(self) -> None:
        state = _state_from_row(pd.Series({
            "type_id": "sce_unknown_zero_wage",
            "employment_status": "unknown",
            "annual_income_usd": 48_000.0,
            "monthly_wage_income_usd": 0.0,
            "monthly_nonwage_income_usd": 2_100.0,
            "monthly_transfers_benefits_usd": 1_900.0,
            "baseline_committed_consumption_monthly_usd": 1_500.0,
            "baseline_discretionary_consumption_monthly_usd": 900.0,
            "liquid_deposits_usd": 1_200.0,
            "revolving_debt_usd": 800.0,
            "revolving_credit_limit_usd": 2_000.0,
        }))
        self.assertEqual(state.employment_share, 0.0)
        self.assertEqual(state.baseline_monthly_hours, 0.0)

    def test_unknown_cohort_member_with_observed_wage_starts_employed(self) -> None:
        state = _state_from_row(pd.Series({
            "type_id": "sce_unknown_positive_wage",
            "employment_status": "unknown",
            "annual_income_usd": 72_000.0,
            "monthly_wage_income_usd": 4_500.0,
            "monthly_nonwage_income_usd": 1_000.0,
            "monthly_transfers_benefits_usd": 500.0,
            "baseline_committed_consumption_monthly_usd": 2_000.0,
            "baseline_discretionary_consumption_monthly_usd": 1_500.0,
            "liquid_deposits_usd": 4_000.0,
            "revolving_debt_usd": 1_000.0,
            "revolving_credit_limit_usd": 4_000.0,
        }))
        self.assertEqual(state.employment_share, 1.0)
        self.assertEqual(state.baseline_monthly_hours, 160.0)
        self.assertAlmostEqual(state.hourly_wage_usd, 0.0)
        self.assertAlmostEqual(state.monthly_household_earned_income_usd, 4_500.0)

    def test_gross_income_residual_uses_all_modeled_income_components(self) -> None:
        state = _state_from_row(pd.Series({
            "type_id": "income_components",
            "employment_status": "employed",
            "annual_income_usd": 12_000.0,
            "monthly_wage_income_usd": 1_000.0,
            "monthly_nonwage_income_usd": 200.0,
            "monthly_transfers_benefits_usd": 300.0,
            "baseline_committed_consumption_monthly_usd": 700.0,
            "baseline_discretionary_consumption_monthly_usd": 100.0,
            "liquid_deposits_usd": 500.0,
            "revolving_debt_usd": 0.0,
            "revolving_credit_limit_usd": 0.0,
        }))
        response = HouseholdResponse(
            expected_inflation_pct=QuantileTriplet(2.0, 2.0, 2.0),
            expected_income_growth_pct=QuantileTriplet(0.0, 0.0, 0.0),
            job_loss_probability_pct=QuantileTriplet(0.0, 0.0, 0.0),
            planned_work_hours=QuantileTriplet(160.0, 160.0, 160.0),
            planned_job_search_hours=QuantileTriplet(0.0, 0.0, 0.0),
            employed_policy=HouseholdPolicyBranch(0.0, 0.0, 0.0, 0.0, 0.0),
            not_employed_policy=HouseholdPolicyBranch(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        result = run_monthly_ecology(
            [state],
            {state.household_id: response},
            EmployerState(
                employer_id="aggregate_employer",
                productivity_per_hour=5.0,
                monthly_capacity_units=2_000.0,
                inventory_units=500.0,
                price_per_unit_usd=1.0,
                target_headcount=1.0,
                wage_offer_usd=state.hourly_wage_usd,
            ),
            CreditIntermediaryState(
                intermediary_id="aggregate_credit_intermediary",
                annual_interest_rate_pct=0.0,
                minimum_payment_rate_pct=0.0,
            ),
            institution_mode="household_demand",
        )
        macro = _weighted_macro(
            result,
            [state],
            {"origin_visible_macro_context": {"FEDFUNDS": {"value": 4.0}}},
        )
        consumption = result.households[0].consumption_usd
        self.assertAlmostEqual(macro["wage_income_usd"], 1_000.0)
        self.assertAlmostEqual(macro["household_earned_income_usd"], 1_000.0)
        self.assertAlmostEqual(macro["nonwage_income_usd"], 200.0)
        self.assertAlmostEqual(macro["transfer_income_usd"], 300.0)
        self.assertAlmostEqual(macro["gross_household_income_usd"], 1_500.0)
        self.assertAlmostEqual(macro["gross_income_residual_usd"], 1_500.0 - consumption)
        self.assertAlmostEqual(
            macro["gross_income_residual_rate_pct"],
            100.0 * (1_500.0 - consumption) / 1_500.0,
        )
        baseline_rate = 100.0 * (1_500.0 - 800.0) / 1_500.0
        self.assertAlmostEqual(
            macro["baseline_gross_income_residual_rate_pct"], baseline_rate
        )
        self.assertAlmostEqual(
            macro["gross_income_residual_rate_change_pp"],
            macro["gross_income_residual_rate_pct"] - baseline_rate,
        )

    def test_declared_saving_rate_calibrates_omitted_fixed_outflow(self) -> None:
        state = _state_from_row(pd.Series({
            "type_id": "saving_anchor",
            "employment_status": "employed",
            "monthly_wage_income_usd": 8_000.0,
            "monthly_business_income_usd": 0.0,
            "monthly_earned_income_usd": 8_000.0,
            "monthly_nonwage_income_usd": 1_000.0,
            "monthly_transfers_benefits_usd": 0.0,
            "baseline_committed_consumption_monthly_usd": 4_000.0,
            "baseline_discretionary_consumption_monthly_usd": 2_000.0,
            "recurring_minimum_debt_payment_usd": 0.0,
            "base_saving_rate": 0.10,
            "target_buffer_months": 1.0,
            "liquid_deposits_usd": 0.0,
        }))
        self.assertAlmostEqual(state.monthly_baseline_total_saving_target_usd, 900.0)
        self.assertAlmostEqual(state.monthly_baseline_liquid_saving_target_usd, 500.0)
        self.assertAlmostEqual(state.monthly_omitted_fixed_outflow_usd, 2_500.0)
        self.assertAlmostEqual(
            8_000.0 + 1_000.0
            - state.monthly_omitted_fixed_outflow_usd
            - state.baseline_monthly_consumption_usd,
            state.monthly_baseline_liquid_saving_target_usd,
        )

    def test_declared_saving_is_split_between_liquid_and_nondeposit_uses(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "type_id": "buffer_shortfall",
                    "employment_status": "employed",
                    "monthly_earned_income_usd": 5_000.0,
                    "monthly_nonwage_income_usd": 0.0,
                    "monthly_transfers_benefits_usd": 0.0,
                    "baseline_committed_consumption_monthly_usd": 2_000.0,
                    "baseline_discretionary_consumption_monthly_usd": 1_000.0,
                    "recurring_minimum_debt_payment_usd": 100.0,
                    "base_saving_rate": 0.20,
                    "target_buffer_months": 2.0,
                    "liquid_deposits_usd": 0.0,
                },
                {
                    "type_id": "buffer_full",
                    "employment_status": "employed",
                    "monthly_earned_income_usd": 5_000.0,
                    "monthly_nonwage_income_usd": 0.0,
                    "monthly_transfers_benefits_usd": 0.0,
                    "baseline_committed_consumption_monthly_usd": 2_000.0,
                    "baseline_discretionary_consumption_monthly_usd": 1_000.0,
                    "recurring_minimum_debt_payment_usd": 100.0,
                    "base_saving_rate": 0.20,
                    "target_buffer_months": 2.0,
                    "liquid_deposits_usd": 10_000.0,
                },
            ]
        )
        liquid_targets = []
        total_targets = []
        for _, row in source.iterrows():
            state = _state_from_row(row)
            gross_income = (
                state.monthly_household_earned_income_usd
                + state.monthly_nonwage_income_usd
                + state.monthly_transfer_income_usd
            )
            liquid_residual = (
                gross_income
                - state.baseline_monthly_consumption_usd
                - state.minimum_debt_payment_usd
                - state.monthly_omitted_fixed_outflow_usd
            )
            self.assertAlmostEqual(
                liquid_residual,
                state.monthly_baseline_liquid_saving_target_usd
                - state.monthly_baseline_cash_deficit_usd,
            )
            self.assertLessEqual(
                state.monthly_baseline_liquid_saving_target_usd,
                state.monthly_baseline_total_saving_target_usd,
            )
            liquid_targets.append(state.monthly_baseline_liquid_saving_target_usd)
            total_targets.append(state.monthly_baseline_total_saving_target_usd)
        self.assertLess(sum(liquid_targets), sum(total_targets))

    def test_initial_institutions_use_population_mass(self) -> None:
        rows = []
        for household_id, income, consumption, weight in (
            ("large_type", 38_400.0, 1_200.0, 0.9),
            ("small_type", 192_000.0, 12_000.0, 0.1),
        ):
            rows.append(
                _state_from_row(
                    pd.Series(
                        {
                            "type_id": household_id,
                            "annual_income": income,
                            "baseline_consumption_annual": consumption * 12.0,
                            "employment_status": "employed",
                            "liquid_assets": 0.0,
                            "debt": 1_000.0,
                            "population_weight": weight,
                        }
                    )
                )
            )
        employer = _initial_employer(rows)
        credit = _initial_credit(
            rows,
            {"origin_visible_macro_context": {"FEDFUNDS": {"value": 4.0}}},
        )
        self.assertAlmostEqual(employer.monthly_capacity_units / 1.15, 4_560.0)
        self.assertAlmostEqual(employer.wage_offer_usd, 28.0)
        self.assertAlmostEqual(credit.new_lending_budget_usd, 1_424.0)

    def test_fixture_payload_round_trips_strict_schema(self) -> None:
        card = household_card(
            {
                "type_id": "h1",
                "employment_status": "employed",
                "annual_income": 60_000,
                "baseline_consumption_annual": 42_000,
                "liquid_assets": 2_000,
                "debt": 4_000,
            },
            origin={
                "origin_month": "2026-05-01",
                "as_of_date": "2026-05-15",
                "origin_visible_macro_context": {},
                "origin_visible_macro_history": {},
            },
            own_history=[],
        )
        response = normalize_household_payload(fixture_response(card), "h1")
        response.validate()
        malformed = fixture_response(card) | {"direct_consumption_usd": 1.0}
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            normalize_household_payload(malformed, "h1")
        self.assertEqual(
            set(household_response_schema()["required"]),
            set(fixture_response(card)),
        )

    def test_history_materialization_manifest_is_hash_bound(self) -> None:
        from macro_llm_tournament.ecology import _history_materialization_binding

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            history.write_text("household_id\nh1\n", encoding="utf-8")
            digest = hashlib.sha256(history.read_bytes()).hexdigest()
            provenance = {
                name: {"sha256": hashlib.sha256(name.encode()).hexdigest(), "row_count": 1}
                for name in ("base_history", "normalized_sce_microdata", "private_registry")
            }
            (root / "history_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ecology_history_materialization_v1",
                        "through_event_month": "2025-09-01",
                        "publication_lag_months": 9,
                        "input_provenance": provenance,
                        "public_history": {
                            "sha256": digest,
                            "row_count": 1,
                            "event_count": 1,
                            "columns": ["household_id"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _history_materialization_binding(history, required=True)["history_sha256"], digest
            )
            history.write_text("household_id\nh2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                _history_materialization_binding(history, required=True)

    def test_history_materialization_manifest_rejects_wrong_schema(self) -> None:
        from macro_llm_tournament.ecology import _history_materialization_binding

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            history.write_text("household_id\nh1\n", encoding="utf-8")
            (root / "history_manifest.json").write_text(
                json.dumps({"schema_version": "wrong", "public_history": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported schema"):
                _history_materialization_binding(history, required=True)

    def test_history_materialization_manifest_rejects_missing_provenance(self) -> None:
        from macro_llm_tournament.ecology import _history_materialization_binding

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            history.write_text("household_id\nh1\n", encoding="utf-8")
            (root / "history_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ecology_history_materialization_v1",
                        "through_event_month": "2025-09-01",
                        "publication_lag_months": 9,
                        "input_provenance": {},
                        "public_history": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete input provenance"):
                _history_materialization_binding(history, required=True)

    def test_nonfixture_history_requires_materialization_manifest(self) -> None:
        from macro_llm_tournament.ecology import _history_materialization_binding

        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.csv"
            history.write_text("household_id\nh1\n", encoding="utf-8")
            self.assertIsNone(
                _history_materialization_binding(history, required=False)
            )
            with self.assertRaisesRegex(ValueError, "require a history"):
                _history_materialization_binding(history, required=True)

    def test_malformed_cached_payload_is_evicted_and_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            card = household_card(
                {
                    "type_id": "h1",
                    "employment_status": "employed",
                    "annual_income": 60_000,
                    "baseline_consumption_annual": 42_000,
                    "liquid_assets": 2_000,
                    "debt": 4_000,
                },
                origin={
                    "origin_month": "2026-05-01",
                    "as_of_date": "2026-05-15",
                    "origin_visible_macro_context": {},
                    "origin_visible_macro_history": {},
                },
                own_history=[],
            )
            budget = LiveCallBudget(2, root / "journal")
            elicitor = HouseholdElicitor(
                "codex_cli", "gpt-5.5", root / "cache", "live", 2, root, budget
            )
            _, _, cache_name = household_request_identity("codex_cli", "gpt-5.5", card)
            cache_path = elicitor.client.cache_path(cache_name)
            cache_path.write_text("cached", encoding="utf-8")
            malformed = {"payload": fixture_response(card) | {"reason_codes": []}, "cache_hit": True, "cache_path": str(cache_path)}
            valid = {"payload": fixture_response(card), "cache_hit": False, "cache_path": str(cache_path)}
            with mock.patch.object(
                elicitor.client,
                "json_call",
                side_effect=[malformed, valid],
            ):
                result = elicitor.elicit(card)
            self.assertEqual(result["payload"], valid["payload"])
            self.assertEqual(budget.used, 1)

    def test_fixture_writes_complete_forecast_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            manifest = run(
                Namespace(
                    origin="2026-05-01",
                    mode="fixture",
                    provider="codex_cli",
                    model="gpt-5.5",
                    max_live_calls=0,
                    household_count=12,
                    households=FIXTURE_HOUSEHOLDS,
                    history=FIXTURE_HISTORY,
                    bundle=FIXTURE_BUNDLE,
                    cache_dir=Path(directory) / "cache",
                    output_dir=output,
                )
            )
            self.assertTrue(manifest["accounting_passed"])
            self.assertEqual(manifest["household_count"], 12)
            self.assertEqual(manifest["accepted_household_response_count"], 12)
            self.assertEqual(manifest["live_call_count"], 0)
            expected = {
                "normalized_origin_information.json",
                "household_cards.json",
                "household_responses.json",
                "household_decisions.csv",
                "median_economy.json",
                "macro_forecast_paths.csv",
                "accounting_audit.csv",
                "event_log.json",
                "manifest.json",
                "ecology_report.md",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})
            paths = pd.read_csv(output / "macro_forecast_paths.csv")
            self.assertEqual(paths["scenario"].tolist(), ["median"])
            self.assertTrue((paths["output_units"] >= paths["units_sold"]).all())
            self.assertEqual(len(pd.read_csv(output / "household_decisions.csv")), 12)
            cards = json.loads((output / "household_cards.json").read_text())
            for card in cards:
                current = card["household"]["current_state"]
                self.assertEqual(current["provenance"], "fixed_survey_scf_anchor")
                self.assertEqual(current["employed"], current["hours_worked"] > 0.0)
                self.assertAlmostEqual(
                    current["annualized_wage_income"],
                    (current["hourly_wage"] or 0.0) * current["hours_worked"] * 12.0,
                )
            self.assertEqual(manifest["state_policy"], "rolling_reanchored")
            self.assertEqual(manifest["source_binding_authority"], "source_sha256")

    def test_recursive_state_policy_is_rejected(self) -> None:
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--origin", "2026-05-01",
                    "--mode", "fixture",
                    "--households", str(FIXTURE_HOUSEHOLDS),
                    "--history", str(FIXTURE_HISTORY),
                    "--bundle", str(FIXTURE_BUNDLE),
                    "--state-policy", "recursive",
                    "--output-dir", "out",
                ]
            )

    def test_expected_replay_hash_is_checked_before_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir = root / "first"
            base = dict(
                origin="2026-05-01",
                mode="fixture",
                provider="codex_cli",
                model="gpt-5.5",
                max_live_calls=0,
                household_count=12,
                households=FIXTURE_HOUSEHOLDS,
                history=FIXTURE_HISTORY,
                bundle=FIXTURE_BUNDLE,
                cache_dir=root / "cache",
            )
            first = run(Namespace(**base, output_dir=first_dir))
            verified_dir = root / "verified"
            verified = run(
                Namespace(
                    **base,
                    expected_replay_sha256=first["replay_equivalence_sha256"],
                    output_dir=verified_dir,
                )
            )
            self.assertTrue(verified["replay_verified"])
            self.assertEqual(first["replay_equivalence_sha256"], verified["replay_equivalence_sha256"])
            failed_dir = root / "failed"
            with self.assertRaisesRegex(ValueError, "replay equivalence hash mismatch"):
                run(
                    Namespace(
                        **base,
                        expected_replay_sha256="0" * 64,
                        output_dir=failed_dir,
                    )
                )
            self.assertEqual([], list(failed_dir.iterdir()))

    def test_live_reference_is_request_set_bound_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, request_set = _live_reference_path(
                Path(directory),
                origin="2026-07-01",
                provider="codex_cli",
                model="gpt-5.5",
                card_sha256s=["a", "b"],
            )
            payload = {
                "schema_version": LIVE_REFERENCE_SCHEMA_VERSION,
                "request_set_sha256": request_set,
                "replay_equivalence_sha256": "1" * 64,
            }
            payload["reference_sha256"] = canonical_sha256(payload)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                _read_live_reference(path, request_set)["replay_equivalence_sha256"],
                "1" * 64,
            )
            payload["replay_equivalence_sha256"] = "2" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed or mismatched"):
                _read_live_reference(path, request_set)


if __name__ == "__main__":
    unittest.main()
