from __future__ import annotations

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
    build_arg_parser,
    run,
)
from macro_llm_tournament.ecology_households import (
    HouseholdElicitor,
    LiveCallBudget,
    canonical_sha256,
    fixture_response,
    household_card,
    household_request_identity,
    normalize_household_payload,
)
from macro_llm_tournament.ecology_models import household_response_schema
from macro_llm_tournament.ecology_inputs import ORIGIN_SNAPSHOT_SCHEMA_VERSION, _canonical_sha256

FIXTURE_ROOT = PROJECT_ROOT / "examples/ecology_fixture"
FIXTURE_HOUSEHOLDS = FIXTURE_ROOT / "households.csv"
FIXTURE_HISTORY = FIXTURE_ROOT / "history.csv"
FIXTURE_BUNDLE = FIXTURE_ROOT / "origin_snapshot.json"


class HouseholdEcologyTests(unittest.TestCase):
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

    def test_recursive_card_uses_immediately_previous_simulated_state(self) -> None:
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
        self.assertEqual(current["provenance"], "prior_simulated_month")

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
                "downside_economy.json",
                "median_economy.json",
                "upside_economy.json",
                "downside_next_state.json",
                "median_next_state.json",
                "upside_next_state.json",
                "next_state.json",
                "macro_forecast_paths.csv",
                "accounting_audit.csv",
                "event_log.json",
                "manifest.json",
                "ecology_report.md",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})
            paths = pd.read_csv(output / "macro_forecast_paths.csv")
            self.assertEqual(set(paths["scenario"]), {"downside", "median", "upside"})
            self.assertFalse(paths["output_units"].equals(paths["units_sold"]))
            decisions = pd.read_csv(output / "household_decisions.csv").query("scenario == 'median'")
            next_state = json.loads((output / "median_next_state.json").read_text())
            next_by_id = {row["household_id"]: row for row in next_state["households"]}
            for row in decisions.itertuples(index=False):
                self.assertAlmostEqual(
                    next_by_id[row.household_id]["baseline_monthly_consumption_usd"],
                    row.consumption_usd,
                )
                self.assertAlmostEqual(
                    next_by_id[row.household_id]["baseline_monthly_hours"],
                    row.actual_hours_worked,
                )
            recursive = Path(directory) / "recursive"
            with self.assertRaisesRegex(ValueError, "month continuity mismatch"):
                run(
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
                        state_json=output / "next_state.json",
                        cache_dir=Path(directory) / "cache",
                        output_dir=recursive,
                    )
                )
            origin_information = json.loads(
                (output / "normalized_origin_information.json").read_text(encoding="utf-8")
            )
            origin_information["origin_month"] = "2026-06-01"
            origin_information["as_of_date"] = "2026-06-15"
            snapshot = {
                "schema_version": ORIGIN_SNAPSHOT_SCHEMA_VERSION,
                "origin_information": origin_information,
                "source": "recursive test",
                "source_requests": [],
            }
            snapshot["snapshot_sha256"] = _canonical_sha256(snapshot)
            snapshot_path = Path(directory) / "june_origin.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            run(
                Namespace(
                    origin="2026-06-01",
                    mode="fixture",
                    provider="codex_cli",
                    model="gpt-5.5",
                    max_live_calls=0,
                    household_count=12,
                    households=FIXTURE_HOUSEHOLDS,
                    history=FIXTURE_HISTORY,
                    bundle=snapshot_path,
                    state_json=output / "next_state.json",
                    cache_dir=Path(directory) / "cache",
                    output_dir=recursive,
                )
            )
            recursive_cards = json.loads((recursive / "household_cards.json").read_text())
            self.assertTrue(
                all("prior_simulated_state" in card["public_information"] for card in recursive_cards)
            )
            self.assertTrue(json.loads((recursive / "manifest.json").read_text())["accounting_passed"])

            initial_cards = json.loads((output / "household_cards.json").read_text())
            for card in initial_cards:
                current = card["household"]["current_state"]
                self.assertEqual(current["provenance"], "survey_seeded_initial_state")
                self.assertEqual(current["employed"], current["hours_worked"] > 0.0)
                self.assertAlmostEqual(
                    current["annualized_wage_income"],
                    (current["hourly_wage"] or 0.0) * current["hours_worked"] * 12.0,
                )
            self.assertTrue(
                all(
                    card["household"]["current_state"]["provenance"]
                    == "prior_simulated_month"
                    for card in recursive_cards
                )
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
