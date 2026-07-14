from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from macro_llm_tournament import ecology_retrospective


class EcologyRetrospectiveTests(unittest.TestCase):
    def test_chart_subtitle_uses_actual_origins_and_model(self) -> None:
        joined = pd.DataFrame({"origin_month": ["2025-09-01", "2025-10-01"]})
        self.assertEqual(
            ecology_retrospective._chart_subtitle(joined, "gpt-test"),
            "gpt-test rolling origins Sep 2025-Oct 2025; first origin is survey-seeded and unscored; later origins are recursive. Shading is downside-upside.",
        )

    def test_direction_scoring_excludes_joint_zero_ties(self) -> None:
        joined = pd.DataFrame(
            {
                "scenario": ["median"] * 4,
                "metric": ["employment_rate_change_pp"] * 4,
                "prediction": [-0.2, 0.0, 0.0, 0.0],
                "actual": [-0.1, 0.1, 0.0, 0.0],
                "state_provenance": ["prior_simulated_month"] * 4,
                "mapping_quality": ["directional_proxy"] * 4,
            }
        )
        row = ecology_retrospective._score_rows(joined).iloc[0]
        self.assertEqual(row["direction_n"], 2)
        self.assertEqual(row["direction_accuracy"], 0.5)

    def test_cumulative_index_compounds_sequential_growth(self) -> None:
        values = ecology_retrospective._cumulative_growth_index([10.0, -10.0])
        for value, expected in zip(values, [100.0, 110.0, 99.0], strict=True):
            self.assertAlmostEqual(value, expected)

    def test_join_rejects_realization_available_by_forecast_cutoff(self) -> None:
        forecasts = pd.DataFrame(
            [
                {
                    "origin_month": "2026-01-01",
                    "target_month": "2026-02-01",
                    "as_of_date": "2026-01-15",
                    "state_provenance": "prior_simulated_month",
                    "scenario": scenario,
                    **{metric: 0.0 for metric in ecology_retrospective.METRIC_MAPPINGS},
                }
                for scenario in ecology_retrospective.SCENARIOS
            ]
        )
        actuals = pd.DataFrame(
            [
                {
                    "target_month": "2026-02-01",
                    "metric": metric,
                    "actual": 0.0,
                    "first_release_as_of_date": "2026-01-15",
                    "mapping_quality": mapping["mapping_quality"],
                    "mapping_note": mapping["note"],
                }
                for metric, mapping in ecology_retrospective.METRIC_MAPPINGS.items()
            ]
        )
        with self.assertRaisesRegex(ValueError, "available at or before"):
            ecology_retrospective._joined_rows(forecasts, actuals)

    def test_realization_mapping_uses_observation_month_and_declared_transforms(self) -> None:
        rows = []
        values = {
            "pce_growth_pct": (0.4, 100.4),
            "personal_saving_rate_change": (-0.2, 3.8),
            "revolving_credit_growth_pct": (0.7, 100.7),
            "unemployment_rate_level": (4.2, 4.2),
            "pce_price_growth_pct": (0.3, 100.3),
        }
        for target_name, (target_value, first_release) in values.items():
            mapping = next(row for row in ecology_retrospective.METRIC_MAPPINGS.values() if row["target_name"] == target_name)
            rows.append({
                "origin_month": "2026-03-01",
                "target_name": target_name,
                "series_id": mapping["series_id"],
                "target_observation_date": "2026-02-01",
                "first_release_as_of_date": "2026-04-01",
                "first_release_value": first_release,
                "first_release_denominator_value": 4.4 if target_name == "unemployment_rate_level" else first_release,
                "target_value": target_value,
            })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            actuals = ecology_retrospective._realization_rows(path, ["2026-02-01"])
        by_metric = actuals.set_index("metric")["actual"].to_dict()
        self.assertEqual(by_metric["consumption_growth_pct"], 0.4)
        self.assertEqual(by_metric["saving_rate_pct"], 3.8)
        self.assertAlmostEqual(by_metric["employment_rate_change_pp"], 0.2)

    def test_two_origin_runner_carries_median_state_and_joins_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            pd.DataFrame({"origin_month": ["2026-01-01", "2026-02-01"]}).to_csv(bundle / "origins.csv", index=False)
            (bundle / "payload.txt").write_text("bundle", encoding="utf-8")
            households = root / "households.csv"
            history = root / "history.csv"
            households.write_text("type_id\nhh_1\n", encoding="utf-8")
            history.write_text("household_id\nhh_1\n", encoding="utf-8")
            targets = root / "targets.csv"
            target_rows = []
            for month in ("2026-02-01", "2026-03-01"):
                for metric, mapping in ecology_retrospective.METRIC_MAPPINGS.items():
                    target_rows.append({
                        "origin_month": month,
                        "target_name": mapping["target_name"],
                        "series_id": mapping["series_id"],
                        "target_observation_date": month,
                        "first_release_as_of_date": "2026-04-01",
                        "first_release_value": 4.0 if metric == "saving_rate_pct" else 5.0,
                        "first_release_denominator_value": 4.1 if metric == "employment_rate_change_pp" else 5.0,
                        "target_value": 0.5,
                    })
            pd.DataFrame(target_rows).to_csv(targets, index=False)
            output = root / "output"
            seen_states: list[Path | None] = []

            def fake_run(args: argparse.Namespace) -> dict[str, object]:
                seen_states.append(args.state_json)
                args.output_dir.mkdir(parents=True)
                target = (pd.Timestamp(args.origin) + pd.offsets.MonthBegin(1)).date().isoformat()
                pd.DataFrame([
                    {
                        "scenario": scenario,
                        **{metric: float(index + 1) for index, metric in enumerate(ecology_retrospective.METRIC_MAPPINGS)},
                    }
                    for scenario in ecology_retrospective.SCENARIOS
                ]).to_csv(args.output_dir / "macro_forecast_paths.csv", index=False)
                (args.output_dir / "median_next_state.json").write_text(json.dumps({"next_origin_month": target}), encoding="utf-8")
                manifest = {
                    "origin_month": args.origin,
                    "target_month": target,
                    "as_of_date": args.origin,
                    "live_call_count": 0,
                    "cache_hit_count": 1,
                    "accepted_household_response_count": 1,
                    "provider_response_count": 0,
                    "fresh_accepted_response_count": 0,
                    "failed_provider_attempt_count": 0,
                    "codex_tool_isolation_version": None,
                    "codex_instruction_context_version": None,
                    "replay_verified": None,
                    "accounting_passed": True,
                }
                (args.output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return manifest

            args = argparse.Namespace(
                origins="2026-01-01:2026-02-01",
                bundle=bundle,
                targets=targets,
                households=households,
                history=history,
                mode="fixture",
                provider="codex_cli",
                model="gpt-5.5",
                household_count=1,
                workers=1,
                max_live_calls=0,
                cache_dir=root / "cache",
                output_dir=output,
            )
            with mock.patch.object(ecology_retrospective, "run_ecology", side_effect=fake_run):
                manifest = ecology_retrospective.run(args)

            self.assertIsNone(seen_states[0])
            self.assertEqual(
                seen_states[1],
                output.resolve().with_name(output.name + ".building") / "runs/2026-01-01/median_next_state.json",
            )
            self.assertEqual(len(pd.read_csv(output / "one_step_forecasts_by_origin.csv")), 6)
            self.assertEqual(len(pd.read_csv(output / "predicted_vs_actual.csv")), 30)
            self.assertTrue((output / "predicted_vs_actual.png").exists())
            self.assertEqual(manifest["state_policy"], "median_recursive_spine")
            self.assertEqual(manifest["score_eligibility"], "prior_simulated_month_only")
            self.assertFalse(manifest["forecast_process_opened_realization_files"])
            self.assertTrue(manifest["source_sha256"])
            scored = pd.read_csv(output / "retrospective_scores.csv")
            self.assertTrue(scored["n"].eq(1).all())

    def test_origin_range_must_be_ascending_month_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "ascending month starts"):
            ecology_retrospective._parse_origins("2026-03-01:2026-02-01")
        with self.assertRaisesRegex(ValueError, "ascending month starts"):
            ecology_retrospective._parse_origins("2026-01-15:2026-02-01")

    def test_incomplete_prospective_marker_fails_before_any_child_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            pd.DataFrame({"origin_month": ["2026-01-01"]}).to_csv(
                bundle / "origins.csv", index=False
            )
            incomplete = root / "incomplete"
            incomplete.mkdir()
            args = argparse.Namespace(
                origins="2026-01-01:2026-01-01",
                bundle=bundle,
                targets=root / "targets.csv",
                households=root / "households.csv",
                history=root / "history.csv",
                mode="live",
                provider="codex_cli",
                model="gpt-5.5",
                household_count=1,
                workers=1,
                max_live_calls=1,
                cache_dir=root / "cache",
                output_dir=root / "output",
                prospective_run=incomplete,
            )
            with mock.patch.object(ecology_retrospective, "run_ecology") as child:
                with self.assertRaises(FileNotFoundError):
                    ecology_retrospective.run(args)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
