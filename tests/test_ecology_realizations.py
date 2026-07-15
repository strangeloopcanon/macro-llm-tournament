from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

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
                self._realization_rows(),
                columns=list(REALIZATION_COLUMNS),
            ).to_csv(realizations_csv, index=False)

            self.assertEqual(0, main(["--run-dir", str(run_dir), "--realizations-csv", str(realizations_csv)]))

            for name, original_bytes in original_files.items():
                self.assertEqual(original_bytes, (run_dir / name).read_bytes(), name)
            self.assertEqual(
                set(original_files) | {"realization_append"},
                {path.name for path in run_dir.iterdir()},
            )

            append_dir = run_dir / "realization_append"
            self.assertEqual(
                {"realized_outcomes.csv", "forecast_errors.csv", "canonical_realizations.csv", "realization_manifest.json"},
                {path.name for path in append_dir.iterdir()},
            )
            realized_outcomes = pd.read_csv(append_dir / "realized_outcomes.csv")
            self.assertEqual(["target_month", *REALIZATION_METRICS], realized_outcomes.columns.tolist())
            self.assertEqual("2026-06-01", realized_outcomes.iloc[0]["target_month"])
            self.assertAlmostEqual(2.75, float(realized_outcomes.iloc[0]["price_growth_pct"]))

            forecast_errors = pd.read_csv(append_dir / "forecast_errors.csv")
            self.assertEqual(["median"], forecast_errors["scenario"].tolist())
            self.assertEqual(["2026-06-01"], forecast_errors["target_month"].tolist())
            for metric in REALIZATION_METRICS:
                self.assertIn(f"forecast_{metric}", forecast_errors.columns)
                self.assertIn(f"realized_{metric}", forecast_errors.columns)
                self.assertIn(f"error_{metric}", forecast_errors.columns)
            median_row = forecast_errors.loc[forecast_errors["scenario"] == "median"].iloc[0]
            self.assertAlmostEqual(
                float(median_row["realized_price_growth_pct"]) - float(median_row["forecast_price_growth_pct"]),
                float(median_row["error_price_growth_pct"]),
            )

            realization_manifest = json.loads((append_dir / "realization_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(RETROSPECTIVE_LABEL, realization_manifest["retrospective_label"])
            self.assertEqual("realized_minus_forecast", realization_manifest["error_definition"])
            self.assertEqual("2026-05-15", realization_manifest["forecast_cutoff_as_of_date"])
            self.assertIn("source_forecast_manifest_sha256", realization_manifest)
            self.assertEqual(
                {"realized_outcomes.csv", "forecast_errors.csv", "canonical_realizations.csv"},
                set(realization_manifest["output_artifacts_sha256"]),
            )
            canonical = pd.read_csv(append_dir / "canonical_realizations.csv").to_dict(orient="records")
            self.assertEqual(self._realization_rows(), realization_manifest["normalized_realization_rows"])
            self.assertEqual("BEA", canonical[0]["source"])
            self.assertEqual(
                realization_manifest["canonical_realization_input_sha256"],
                self._sha256(append_dir / "canonical_realizations.csv"),
            )

    def test_refuses_to_append_when_outputs_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")
            (run_dir / "realization_append").mkdir()

            with self.assertRaisesRegex(ValueError, "Realization outputs already exist"):
                append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

            self.assertFalse((run_dir / "forecast_errors.csv").exists())
            self.assertFalse((run_dir / "realization_manifest.json").exists())

    def test_rejects_invalid_realization_csv_contracts(self) -> None:
        cases = [
            (
                "extra column",
                self._realization_rows(),
                [*REALIZATION_COLUMNS, "extra_metric"],
                "exactly these columns",
            ),
            (
                "nonnumeric",
                [self._realization_rows()[0] | {"value": "nope"}, *self._realization_rows()[1:]],
                list(REALIZATION_COLUMNS),
                "must be numeric",
            ),
            (
                "nonfinite",
                [self._realization_rows()[0] | {"value": float("inf")}, *self._realization_rows()[1:]],
                list(REALIZATION_COLUMNS),
                "must be finite",
            ),
            (
                "target mismatch",
                [self._realization_rows()[0] | {"target_month": "2026-05-01"}, *self._realization_rows()[1:]],
                list(REALIZATION_COLUMNS),
                "does not match manifest target",
            ),
            (
                "wrong row count",
                self._realization_rows()[:3],
                list(REALIZATION_COLUMNS),
                "exactly 4 rows",
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

    def test_rejects_realization_provenance_that_cannot_prove_post_cutoff_availability(self) -> None:
        cases = [
            ("release at cutoff", {"release_date": "2026-05-15"}, "must be after forecast cutoff"),
            ("missing source", {"source": ""}, "source .* must be non-empty"),
            ("non-url source reference", {"source_url": "BEA table 2.3.5"}, "must be an http\\(s\\) URL"),
            ("vintage predates release", {"vintage_date": "2026-05-15"}, "cannot precede release_date"),
        ]
        for label, replacement, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = self._build_fixture_run(root)
                rows = [self._realization_rows()[0] | replacement, *self._realization_rows()[1:]]
                realizations_csv = root / "realizations.csv"
                pd.DataFrame(rows, columns=list(REALIZATION_COLUMNS)).to_csv(realizations_csv, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

    def test_publish_failure_leaves_no_accepted_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._build_fixture_run(root)
            realizations_csv = self._write_valid_realizations_csv(root / "realizations.csv")

            with mock.patch("macro_llm_tournament.ecology_realizations.os.replace", side_effect=OSError("publish failed")):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    append_realizations(run_dir=run_dir, realizations_csv=realizations_csv)

            self.assertFalse((run_dir / "realization_append").exists())
            self.assertFalse((run_dir / ".realization_append.lock").exists())
            self.assertFalse(any(path.name.endswith(".staging") for path in run_dir.iterdir()))

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
        pd.DataFrame(EcologyRealizationsTests._realization_rows(), columns=list(REALIZATION_COLUMNS)).to_csv(path, index=False)
        return path

    @staticmethod
    def _realization_rows() -> list[dict[str, object]]:
        values = {
            "consumption_growth_pct": 1.25,
            "revolving_credit_growth_pct": -1.0,
            "employment_rate_pct": 91.25,
            "price_growth_pct": 2.75,
        }
        return [
            {
                "target_month": "2026-06-01",
                "metric": metric,
                "value": value,
                "source": "BEA",
                "source_url": f"https://example.test/{metric}",
                "vintage_date": "2026-06-16",
                "release_date": "2026-06-15",
            }
            for metric, value in values.items()
        ]

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
