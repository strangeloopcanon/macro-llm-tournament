from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from macro_llm_tournament.ecology import (
    DEFAULT_BUNDLE,
    DEFAULT_HISTORY,
    DEFAULT_HOUSEHOLDS,
    PROJECT_ROOT,
    _coverage_sample,
    run,
)
from macro_llm_tournament.ecology_households import (
    fixture_response,
    household_card,
    normalize_household_payload,
)
from macro_llm_tournament.ecology_models import household_response_schema
from macro_llm_tournament.ecology_inputs import ORIGIN_SNAPSHOT_SCHEMA_VERSION, _canonical_sha256


class HouseholdEcologyTests(unittest.TestCase):
    def test_canary_sample_is_stable_and_covers_state_categories(self) -> None:
        frame = pd.read_csv(DEFAULT_HOUSEHOLDS)
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
                    households=DEFAULT_HOUSEHOLDS,
                    history=DEFAULT_HISTORY,
                    bundle=DEFAULT_BUNDLE,
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
                        households=DEFAULT_HOUSEHOLDS,
                        history=DEFAULT_HISTORY,
                        bundle=DEFAULT_BUNDLE,
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
                    households=DEFAULT_HOUSEHOLDS,
                    history=DEFAULT_HISTORY,
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
                households=DEFAULT_HOUSEHOLDS,
                history=DEFAULT_HISTORY,
                bundle=DEFAULT_BUNDLE,
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


if __name__ == "__main__":
    unittest.main()
