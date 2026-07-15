from __future__ import annotations

import copy
import unittest

from macro_llm_tournament.ecology_information import (
    build_macro_information_card,
    build_policy_payload,
    canonical_sha256,
)


def _origin() -> dict[str, object]:
    return {
        "origin_month": "2026-07-01",
        "as_of_date": "2026-07-15",
        "origin_visible_macro_context": {
            "FEDFUNDS": {"observation_date": "2026-06-01", "value": 4.5, "release_date": "2026-07-01"},
            "PCE": {"observation_date": "2026-06-01", "value": 108.0},
        },
        "origin_visible_macro_history": {
            "FEDFUNDS": [
                {"observation_date": "2025-06-01", "value": 3.0},
                {"observation_date": "2026-03-01", "value": 4.0},
                {"observation_date": "2026-05-01", "value": 4.25},
                {"observation_date": "2026-06-01", "value": 4.5, "release_date": "2026-07-01"},
            ],
            "PCE": [
                {"observation_date": "2025-06-01", "value": 90.0},
                {"observation_date": "2026-03-01", "value": 100.0},
                {"observation_date": "2026-05-01", "value": 105.0},
                {"observation_date": "2026-06-01", "value": 108.0},
            ],
        },
        "public_events": [
            {
                "event_id": "fomc-july",
                "event_date": "2026-07-29",
                "event_type": "policy_meeting",
                "title": "FOMC meeting",
                "public_availability_date": "2026-06-01",
            }
        ],
    }


class EcologyInformationTests(unittest.TestCase):
    def test_card_transforms_rates_levels_events_and_policy(self) -> None:
        card = build_macro_information_card(
            _origin(),
            allowed_series=["FEDFUNDS", "PCE", "UNRATE"],
            policy_declarations={"declared_spread_bps": 325, "declared_pass_through_fraction": 0.75},
        )
        fed_funds = card["series"]["FEDFUNDS"]
        self.assertEqual(fed_funds["changes"]["1m"], {"value": 0.25, "base_observation_date": "2026-05-01"})
        self.assertEqual(fed_funds["changes"]["3m"], {"value": 0.5, "base_observation_date": "2026-03-01"})
        self.assertEqual(fed_funds["changes"]["12m"], {"value": 1.5, "base_observation_date": "2025-06-01"})
        self.assertEqual(fed_funds["latest_release_date"], "2026-07-01")
        self.assertEqual(fed_funds["staleness_days"], 44)
        self.assertAlmostEqual(card["series"]["PCE"]["changes"]["1m"]["value"], 108 / 105 * 100 - 100)
        self.assertEqual(card["series"]["UNRATE"]["available"], False)
        self.assertEqual(card["public_events"][0]["event_id"], "fomc-july")
        self.assertEqual(card["policy"]["current_visible_rate"], 4.5)
        self.assertEqual(card["policy"]["previous_visible_rate"], 4.25)
        self.assertEqual(card["policy"]["basis_point_change"], 25.0)
        self.assertEqual(card["policy"]["declared_spread_bps"], 325.0)
        self.assertEqual(card["policy"]["declared_pass_through_fraction"], 0.75)

    def test_stale_and_missing_series_are_explicit(self) -> None:
        origin = _origin()
        origin["origin_visible_macro_context"] = {}
        origin["origin_visible_macro_history"] = {
            "PCE": [{"observation_date": "2025-01-01", "value": 100.0}]
        }
        card = build_macro_information_card(origin, allowed_series=["PCE", "UNRATE"])
        self.assertEqual(card["series"]["PCE"]["staleness_days"], 560)
        self.assertIsNone(card["series"]["PCE"]["changes"]["1m"])
        self.assertFalse(card["series"]["UNRATE"]["available"])
        self.assertIsNone(card["policy"]["current_visible_rate"])

    def test_future_or_malformed_fed_funds_observations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "future observation"):
            build_policy_payload(
                [{"observation_date": "2026-07-16", "value": 4.5}], "2026-07-15"
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            build_policy_payload(
                [{"observation_date": "2026-06-01", "value": "not-a-number"}], "2026-07-15"
            )
        with self.assertRaisesRegex(ValueError, "future release"):
            build_policy_payload(
                [{"observation_date": "2026-06-01", "value": 4.5, "release_date": "2026-07-16"}],
                "2026-07-15",
            )

    def test_canonical_hash_is_order_independent_and_card_is_bound(self) -> None:
        self.assertEqual(canonical_sha256({"a": [1, 2], "b": 3}), canonical_sha256({"b": 3, "a": [1, 2]}))
        first = build_macro_information_card(_origin(), allowed_series=["PCE", "FEDFUNDS"])
        reordered = copy.deepcopy(_origin())
        reordered["origin_visible_macro_history"] = {
            "PCE": reordered["origin_visible_macro_history"]["PCE"],
            "FEDFUNDS": reordered["origin_visible_macro_history"]["FEDFUNDS"],
        }
        second = build_macro_information_card(reordered, allowed_series=["FEDFUNDS", "PCE"])
        self.assertEqual(first["card_sha256"], second["card_sha256"])
        self.assertNotEqual(first["card_sha256"], build_macro_information_card(_origin(), allowed_series=["PCE"])["card_sha256"])

    def test_rejects_leakage_and_unapproved_series(self) -> None:
        leaked = _origin()
        leaked["origin_visible_macro_history"]["PCE"][0]["actual_pce"] = 999.0
        with self.assertRaisesRegex(ValueError, "forbidden leakage"):
            build_macro_information_card(leaked, allowed_series=["PCE", "FEDFUNDS"])
        unapproved = _origin()
        unapproved["origin_visible_macro_history"]["PRIVATE"] = [{"observation_date": "2026-06-01", "value": 1.0}]
        with self.assertRaisesRegex(ValueError, "unapproved public series"):
            build_macro_information_card(unapproved, allowed_series=["PCE", "FEDFUNDS"])


if __name__ == "__main__":
    unittest.main()
