from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.macro_jepa import (
    _comparison_table,
    _household_overlay_table,
    _write_tracked_evidence,
)
from macro_llm_tournament.macro_jepa_data import (
    MacroEvent,
    apply_origin_information_overrides,
    build_event_state_panel,
    build_origin_state_panel,
    first_release_targets,
    load_macro_events,
    validate_canonical_series_units,
)
from macro_llm_tournament.macro_jepa_events import _events_from_payloads


class MacroJepaDataTests(unittest.TestCase):
    def test_tracked_evidence_contains_portable_scores_and_source_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            output.mkdir()
            fixtures = {
                "extraction_manifest.json": {"event_count": 10},
                "locked_training_spec.json": {"primary_metric": "consumption_growth_pct"},
                "jepa_evaluation.json": {"latent_mse": 0.1},
                "readout_selection.json": {
                    "jepa": {"consumption_growth_pct": {"chosen_alpha": 1.0}},
                    "macro_state_ridge": {
                        "consumption_growth_pct": {"chosen_alpha": 10.0}
                    },
                    "historical_constants": {
                        "consumption_growth_pct": {"historical_median": 0.3}
                    },
                },
            }
            for name, payload in fixtures.items():
                (output / name).write_text(json.dumps(payload), encoding="utf-8")
            destination = root / "evidence.json"
            _write_tracked_evidence(
                output=output,
                destination=destination,
                manifest={
                    "model_checkpoint": str(output / "model.pt"),
                    "vei_root": "/private/vei",
                    "primary_score": {"event_jepa_residual": {"rmse": 0.2}},
                },
            )
            evidence = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(
            evidence["evidence_schema_version"], "macro_jepa_tracked_evidence_v1"
        )
        self.assertEqual(evidence["run_manifest"]["vei_root"], "../vei")
        self.assertIn("macro_jepa.py", evidence["generator_source_sha256"])
        self.assertEqual(len(evidence["evidence_sha256"]), 64)

    def test_household_overlay_refresh_preserves_locked_jepa_rows(self) -> None:
        existing = pd.DataFrame(
            [
                {
                    "origin_month": "2026-01-01",
                    "target_month": "2026-02-01",
                    "metric": "consumption_growth_pct",
                    "evaluation_status": "retrospective_diagnostic_not_confirmatory",
                    "value": 9.0,
                    "source": "llm_household_economy",
                },
                {
                    "origin_month": "2026-01-01",
                    "target_month": "2026-02-01",
                    "metric": "consumption_growth_pct",
                    "evaluation_status": "retrospective_diagnostic_not_confirmatory",
                    "value": 0.25,
                    "source": "event_jepa_residual",
                },
            ]
        )
        replacement = pd.DataFrame(
            [
                {
                    "origin_month": "2026-01-01",
                    "target_month": "2026-02-01",
                    "metric": "consumption_growth_pct",
                    "evaluation_status": "retrospective_diagnostic_not_confirmatory",
                    "series_role": "llm_household_economy",
                    "value": 0.5,
                }
            ]
        )
        result = _household_overlay_table(existing, replacement)
        self.assertEqual(
            result.loc[
                result["source"].eq("llm_household_economy"), "value"
            ].item(),
            0.5,
        )
        self.assertEqual(
            result.loc[result["source"].eq("event_jepa_residual"), "value"].item(),
            0.25,
        )

    def test_comparison_rejects_missing_frozen_actual(self) -> None:
        with TemporaryDirectory() as temporary:
            comparison_path = Path(temporary) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "origin_month": "2026-01-01",
                        "target_month": "2026-02-01",
                        "metric": "consumption_growth_pct",
                        "evaluation_status": "retrospective_diagnostic_not_confirmatory",
                        "series_role": "first_release_actual",
                        "value": 1.0,
                    }
                ]
            ).to_csv(comparison_path, index=False)
            reconstructed = pd.DataFrame(
                [
                    {
                        "target_month": "2026-02-01",
                        "metric": "consumption_growth_pct",
                        "target_value": 1.0,
                    }
                ]
            )
            empty_predictions = pd.DataFrame(
                columns=["origin_month", "target_month", "split", "metric", "prediction"]
            )

            with self.assertRaisesRegex(
                ValueError, "frozen target catalogue does not cover every scored actual"
            ):
                _comparison_table(
                    comparison_panel=comparison_path,
                    targets=reconstructed,
                    frozen_actuals=pd.DataFrame(
                        columns=["target_month", "metric", "actual"]
                    ),
                    jepa_predictions=empty_predictions,
                    state_predictions=empty_predictions,
                )

    def test_first_release_transform_uses_same_vintage_denominator(self) -> None:
        events = [
            MacroEvent("PCE", "2026-01-01", "2026-02-10", "initial_release", 100.0, "a"),
            MacroEvent("PCE", "2026-01-01", "2026-03-01", "revision", 110.0, "b"),
            MacroEvent("PCE", "2026-02-01", "2026-03-15", "initial_release", 121.0, "c"),
        ]

        targets = first_release_targets(events)
        row = targets.loc[
            targets["target_month"].eq("2026-02-01")
            & targets["metric"].eq("consumption_growth_pct")
        ].iloc[0]

        self.assertAlmostEqual(float(row["first_release_denominator_value"]), 110.0)
        self.assertAlmostEqual(float(row["target_value"]), 10.0)

    def test_origin_state_excludes_events_released_after_as_of_date(self) -> None:
        events = [
            MacroEvent("PCE", "2025-12-01", "2026-01-10", "initial_release", 100.0, "a"),
            MacroEvent("PCE", "2026-01-01", "2026-01-20", "initial_release", 200.0, "b"),
        ]

        panel = build_origin_state_panel(
            events,
            start_month="2026-01-01",
            end_month="2026-02-01",
            as_of_day=15,
            state_series=("PCE",),
        )

        self.assertEqual(float(panel.iloc[0]["series__PCE__level"]), 100.0)
        self.assertEqual(float(panel.iloc[1]["series__PCE__level"]), 200.0)

    def test_exact_origin_cutoff_applies_before_state_construction(self) -> None:
        events = [
            MacroEvent("PCE", "2026-06-01", "2026-07-09", "initial_release", 100.0, "a"),
            MacroEvent("PCE", "2026-06-01", "2026-07-12", "revision", 999.0, "b"),
        ]

        panel = build_origin_state_panel(
            events,
            start_month="2026-07-01",
            end_month="2026-07-01",
            as_of_day=15,
            as_of_date_overrides={"2026-07-01": "2026-07-10"},
            state_series=("PCE",),
        )

        self.assertEqual(str(panel.iloc[0]["as_of_date"]), "2026-07-10")
        self.assertEqual(float(panel.iloc[0]["series__PCE__level"]), 100.0)
        self.assertEqual(float(panel.iloc[0]["revision_event_count"]), 0.0)

    def test_revision_action_uses_pre_revision_value(self) -> None:
        events = [
            MacroEvent("PCE", "2025-12-01", "2026-01-10", "initial_release", 100.0, "a"),
            MacroEvent("PCE", "2025-12-01", "2026-02-10", "revision", 110.0, "b"),
        ]

        panel = build_origin_state_panel(
            events,
            start_month="2026-01-01",
            end_month="2026-02-01",
            as_of_day=15,
            state_series=("PCE",),
        )

        self.assertEqual(float(panel.iloc[1]["revision_event_count"]), 1.0)
        self.assertAlmostEqual(
            float(panel.iloc[1]["mean_absolute_revision_pct"]), 10.0
        )

    def test_sentiment_change_uses_percent_change_units(self) -> None:
        events = [
            MacroEvent(
                "UMCSENT", "2025-12-01", "2026-01-10", "initial_release", 51.0, "a"
            ),
            MacroEvent(
                "UMCSENT", "2026-01-01", "2026-02-10", "initial_release", 52.9, "b"
            ),
        ]

        panel = build_origin_state_panel(
            events,
            start_month="2026-01-01",
            end_month="2026-02-01",
            as_of_day=15,
            state_series=("UMCSENT",),
        )

        expected = 100.0 * (52.9 / 51.0 - 1.0)
        self.assertAlmostEqual(
            float(panel.iloc[1]["series__UMCSENT__change_1m"]), expected
        )
        self.assertAlmostEqual(
            float(panel.iloc[1]["sentiment_change_1m"]), expected
        )

    def test_comparison_omits_generated_prospective_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            comparison_path = Path(temporary) / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "origin_month": "2026-01-01",
                        "target_month": "2026-02-01",
                        "metric": "consumption_growth_pct",
                        "evaluation_status": "retrospective_diagnostic_not_confirmatory",
                        "series_role": "first_release_actual",
                        "value": 1.0,
                    },
                    {
                        "origin_month": "2026-07-01",
                        "target_month": "2026-08-01",
                        "metric": "consumption_growth_pct",
                        "evaluation_status": "prospective_frozen_unscored",
                        "series_role": "llm_household_economy",
                        "value": 0.5,
                    },
                ]
            ).to_csv(comparison_path, index=False)
            reconstructed = pd.DataFrame(
                [
                    {
                        "target_month": "2026-02-01",
                        "metric": "consumption_growth_pct",
                        "target_value": 1.0,
                    }
                ]
            )
            predictions = pd.DataFrame(
                [
                    {
                        "origin_month": "2026-01-01",
                        "target_month": "2026-02-01",
                        "split": "test",
                        "metric": "consumption_growth_pct",
                        "prediction": 0.8,
                    },
                    {
                        "origin_month": "2026-07-01",
                        "target_month": "2026-08-01",
                        "split": "test",
                        "metric": "consumption_growth_pct",
                        "prediction": 0.9,
                    },
                ]
            )

            result = _comparison_table(
                comparison_panel=comparison_path,
                targets=reconstructed,
                frozen_actuals=pd.DataFrame(
                    [
                        {
                            "target_month": "2026-02-01",
                            "metric": "consumption_growth_pct",
                            "actual": 1.0,
                        }
                    ]
                ),
                jepa_predictions=predictions,
                state_predictions=predictions.iloc[0:0],
            )

        jepa = result.loc[result["source"].eq("event_jepa_residual")]
        self.assertEqual(jepa["origin_month"].tolist(), ["2026-01-01"])

    def test_event_loader_collapses_twins_and_noop_revisions(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            rows = [
                {
                    "series_id": "PCE",
                    "observation_date": "2020-01-01",
                    "vintage_date": "2020-02-01",
                    "event_kind": "initial_release",
                    "value": 100.0,
                    "event_id": "initial",
                },
                {
                    "series_id": "PCE",
                    "observation_date": "2020-01-01",
                    "vintage_date": "2020-02-01",
                    "event_kind": "revision",
                    "value": 100.0,
                    "event_id": "twin",
                },
                {
                    "series_id": "PCE",
                    "observation_date": "2020-01-01",
                    "vintage_date": "2020-03-01",
                    "event_kind": "revision",
                    "value": 100.0,
                    "event_id": "noop",
                },
                {
                    "series_id": "PCE",
                    "observation_date": "2020-01-01",
                    "vintage_date": "2020-04-01",
                    "event_kind": "revision",
                    "value": 101.0,
                    "event_id": "changed",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            events = load_macro_events(path)

        self.assertEqual([row.event_id for row in events], ["initial", "changed"])
        self.assertEqual(
            [row.event_kind for row in events], ["initial_release", "revision"]
        )

    def test_event_loader_normalizes_credit_series_unit_break(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            rows = [
                {
                    "series_id": "REVOLSL",
                    "observation_date": "2024-11-01",
                    "vintage_date": "2025-01-01",
                    "event_kind": "initial_release",
                    "value": 1_359.0,
                    "event_id": "billions",
                },
                {
                    "series_id": "REVOLSL",
                    "observation_date": "2024-12-01",
                    "vintage_date": "2025-02-01",
                    "event_kind": "initial_release",
                    "value": 1_382_000.0,
                    "event_id": "millions",
                },
                {
                    "series_id": "TOTALSL",
                    "observation_date": "2024-12-01",
                    "vintage_date": "2025-02-01",
                    "event_kind": "initial_release",
                    "value": 5_145_000.0,
                    "event_id": "total_millions",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            events = load_macro_events(path)

        values = {row.event_id: row.value for row in events}
        self.assertEqual(values["billions"], 1_359.0)
        self.assertEqual(values["millions"], 1_382.0)
        self.assertEqual(values["total_millions"], 5_145.0)

    def test_origin_override_uses_the_same_credit_unit_normalization(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "origin_month": "2026-01-01",
                    "as_of_date": "2026-01-15",
                    "series__REVOLSL__level": 1_300.0,
                    "series__REVOLSL__change_1m": 0.2,
                    "series__REVOLSL__change_3m": 0.4,
                }
            ]
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "origin.json"
            path.write_text(
                json.dumps(
                    {
                        "origin_month": "2026-01-01",
                        "as_of_date": "2026-01-10",
                        "compact_macro_information": {
                            "series": {
                                "REVOLSL": {
                                    "available": True,
                                    "latest_value": 1_313_920.25,
                                    "staleness_days": 40,
                                    "latest_release_date": "2026-01-08",
                                    "change_method": "percent_change",
                                    "changes": {
                                        "1m": {"value": 0.25},
                                        "3m": {"value": 0.75},
                                    },
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = apply_origin_information_overrides(
                panel, {"2026-01-01": path}
            )

        self.assertAlmostEqual(
            float(result.iloc[0]["series__REVOLSL__level"]), 1_313.92025
        )
        validate_canonical_series_units(result)

    def test_unit_audit_rejects_a_raw_million_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "noncanonical level unit"):
            validate_canonical_series_units(
                pd.DataFrame({"series__TOTALSL__level": [5_145_000.0]})
            )

    def test_event_panel_aligns_release_action_with_pre_release_state(self) -> None:
        events = [
            MacroEvent("PCE", "2019-12-01", "2020-01-10", "initial_release", 100.0, "a"),
            MacroEvent("PCE", "2020-01-01", "2020-02-10", "initial_release", 110.0, "b"),
        ]
        anchors = build_origin_state_panel(
            events,
            start_month="2020-01-01",
            end_month="2020-02-01",
            as_of_day=15,
            state_series=("PCE",),
        )

        panel = build_event_state_panel(
            events,
            anchors=anchors,
            start_date="2020-01-01",
            end_date="2020-02-15",
            state_series=("PCE",),
        )

        feb_release = panel.loc[
            panel["timestamp"].eq("2020-02-10T12:00:00Z")
        ].iloc[0]
        feb_anchor = panel.loc[
            panel["timestamp"].eq("2020-02-15T23:59:59Z")
        ].iloc[0]
        self.assertEqual(float(feb_release["series__PCE__level"]), 100.0)
        self.assertAlmostEqual(float(feb_release["event_update__PCE"]), 10.0)
        self.assertEqual(float(feb_anchor["series__PCE__level"]), 110.0)

    def test_pcec96_payload_conversion_preserves_vintages(self) -> None:
        initial = {
            "observations": [
                {
                    "date": "2020-01-01",
                    "realtime_start": "2020-02-01",
                    "value": "100.0",
                }
            ]
        }
        revisions = {
            "observations": [
                {
                    "date": "2020-01-01",
                    "PCEC96_20200201": "100.0",
                    "PCEC96_20200301": "101.0",
                }
            ]
        }

        rows = _events_from_payloads(revisions, initial)

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            sorted(row["vintage_date"] for row in rows),
            ["2020-02-01", "2020-02-01", "2020-03-01"],
        )


if __name__ == "__main__":
    unittest.main()
