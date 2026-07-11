from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from macro_llm_tournament.ecology import PROJECT_ROOT, run
from macro_llm_tournament.ecology_realizations import (
    REALIZATION_COLUMNS,
    REALIZATION_METRICS,
    RETROSPECTIVE_LABEL,
    append_realizations,
    main,
)

FIXTURE_ROOT = PROJECT_ROOT / "examples/ecology_fixture"


class EcologyRealizationsTests(unittest.TestCase):
    def test_append_cli_writes_realization_outputs_and_preserves_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            original_files = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            realizations_csv = root / "realizations.csv"
            pd.DataFrame(
                [
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": 1.25,
                        "saving_rate_pct": 9.5,
                        "revolving_credit_growth_pct": -1.0,
                        "employment_rate_pct": 91.25,
                        "price_growth_pct": 2.75,
                    }
                ],
                columns=list(REALIZATION_COLUMNS),
            ).to_csv(realizations_csv, index=False)

            self.assertEqual(0, main(["--run-dir", str(run_dir), "--realizations-csv", str(realizations_csv)]))

            for name, original_bytes in original_files.items():
                self.assertEqual(original_bytes, (run_dir / name).read_bytes(), name)
            self.assertEqual(
                set(original_files) | {"realized_outcomes.csv", "forecast_errors.csv", "realization_manifest.json"},
                {path.name for path in run_dir.iterdir()},
            )

            realized_outcomes = pd.read_csv(run_dir / "realized_outcomes.csv")
            self.assertEqual(list(REALIZATION_COLUMNS), realized_outcomes.columns.tolist())
            self.assertEqual("2026-06-01", realized_outcomes.iloc[0]["target_month"])
            self.assertAlmostEqual(2.75, float(realized_outcomes.iloc[0]["price_growth_pct"]))

            forecast_errors = pd.read_csv(run_dir / "forecast_errors.csv")
            self.assertEqual(["downside", "median", "upside"], forecast_errors["scenario"].tolist())
            self.assertEqual(["2026-06-01"] * 3, forecast_errors["target_month"].tolist())
            for metric in REALIZATION_METRICS:
                self.assertIn(f"forecast_{metric}", forecast_errors.columns)
                self.assertIn(f"realized_{metric}", forecast_errors.columns)
                self.assertIn(f"error_{metric}", forecast_errors.columns)
            median_row = forecast_errors.loc[forecast_errors["scenario"] == "median"].iloc[0]
            self.assertAlmostEqual(
                float(median_row["realized_price_growth_pct"]) - float(median_row["forecast_price_growth_pct"]),
                float(median_row["error_price_growth_pct"]),
            )

            realization_manifest = json.loads((run_dir / "realization_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(RETROSPECTIVE_LABEL, realization_manifest["retrospective_label"])
            self.assertEqual("realized_minus_forecast", realization_manifest["error_definition"])
            self.assertIn("source_forecast_manifest_sha256", realization_manifest)
            self.assertEqual(
                {"realized_outcomes.csv", "forecast_errors.csv"},
                set(realization_manifest["output_artifacts_sha256"]),
            )

    def test_refuses_to_append_when_outputs_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")
            (run_dir / "realized_outcomes.csv").write_text("already here\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Realization outputs already exist"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

            self.assertFalse((run_dir / "forecast_errors.csv").exists())
            self.assertFalse((run_dir / "realization_manifest.json").exists())

    def test_rejects_invalid_realization_csv_contracts(self) -> None:
        cases = [
            (
                "extra column",
                [
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": 1.0,
                        "saving_rate_pct": 2.0,
                        "revolving_credit_growth_pct": 3.0,
                        "employment_rate_pct": 4.0,
                        "price_growth_pct": 5.0,
                        "extra_metric": 6.0,
                    }
                ],
                ["target_month", *REALIZATION_METRICS, "extra_metric"],
                "exactly these columns",
            ),
            (
                "nonnumeric",
                [
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": "nope",
                        "saving_rate_pct": 2.0,
                        "revolving_credit_growth_pct": 3.0,
                        "employment_rate_pct": 4.0,
                        "price_growth_pct": 5.0,
                    }
                ],
                list(REALIZATION_COLUMNS),
                "must be numeric",
            ),
            (
                "nonfinite",
                [
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": float("inf"),
                        "saving_rate_pct": 2.0,
                        "revolving_credit_growth_pct": 3.0,
                        "employment_rate_pct": 4.0,
                        "price_growth_pct": 5.0,
                    }
                ],
                list(REALIZATION_COLUMNS),
                "must be finite",
            ),
            (
                "target mismatch",
                [
                    {
                        "target_month": "2026-05-01",
                        "consumption_growth_pct": 1.0,
                        "saving_rate_pct": 2.0,
                        "revolving_credit_growth_pct": 3.0,
                        "employment_rate_pct": 4.0,
                        "price_growth_pct": 5.0,
                    }
                ],
                list(REALIZATION_COLUMNS),
                "does not match manifest target",
            ),
            (
                "wrong row count",
                [
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": 1.0,
                        "saving_rate_pct": 2.0,
                        "revolving_credit_growth_pct": 3.0,
                        "employment_rate_pct": 4.0,
                        "price_growth_pct": 5.0,
                    },
                    {
                        "target_month": "2026-06-01",
                        "consumption_growth_pct": 1.1,
                        "saving_rate_pct": 2.1,
                        "revolving_credit_growth_pct": 3.1,
                        "employment_rate_pct": 4.1,
                        "price_growth_pct": 5.1,
                    },
                ],
                list(REALIZATION_COLUMNS),
                "exactly one row",
            ),
        ]
        for label, rows, columns, message in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    run_dir = self._build_fixture_run(root)
                    realizations_csv = root / "realizations.csv"
                    pd.DataFrame(rows, columns=columns).to_csv(realizations_csv, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

    def test_rejects_manifest_schema_and_hash_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "wrong_schema"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported manifest schema_version"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["forecast_frozen_before_realization"] = False
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forecast_frozen_before_realization must be true"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("target_month", None)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest target_month must be present"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")

            (run_dir / "macro_forecast_paths.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifest artifact hash mismatch for macro_forecast_paths.csv"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

    @staticmethod
    def _build_fixture_run(root: Path) -> Path:
        run_dir = root / "run"
        run(
            Namespace(
                origin="2026-05-01",
                mode="fixture",
                provider="codex_cli",
                model="gpt-5.5",
                max_live_calls=0,
                household_count=12,
                households=FIXTURE_ROOT / "households.csv",
                history=FIXTURE_ROOT / "history.csv",
                bundle=FIXTURE_ROOT / "origin_snapshot.json",
                cache_dir=root / "cache",
                output_dir=run_dir,
            )
        )
        return run_dir

    @staticmethod
    def _write_valid_realizations_csv(path: Path) -> Path:
        pd.DataFrame(
            [
                {
                    "target_month": "2026-06-01",
                    "consumption_growth_pct": 1.0,
                    "saving_rate_pct": 2.0,
                    "revolving_credit_growth_pct": 3.0,
                    "employment_rate_pct": 4.0,
                    "price_growth_pct": 5.0,
                }
            ],
            columns=list(REALIZATION_COLUMNS),
        ).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
