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
                "target_value": target_value,
            })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            actuals = ecology_retrospective._realization_rows(path, ["2026-02-01"])
        by_metric = actuals.set_index("metric")["actual"].to_dict()
        self.assertEqual(by_metric["consumption_growth_pct"], 0.4)
        self.assertEqual(by_metric["saving_rate_pct"], 3.8)
        self.assertEqual(by_metric["employment_rate_pct"], 95.8)

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
            self.assertEqual(manifest["state_policy"], "median_recursive_spine")

    def test_origin_range_must_be_ascending_month_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "ascending month starts"):
            ecology_retrospective._parse_origins("2026-03-01:2026-02-01")
        with self.assertRaisesRegex(ValueError, "ascending month starts"):
            ecology_retrospective._parse_origins("2026-01-15:2026-02-01")


if __name__ == "__main__":
    unittest.main()
