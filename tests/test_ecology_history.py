from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.ecology_history import EcologyHistoryError, materialize_ecology_history
from macro_llm_tournament.persistent_households import HISTORY_COLUMNS, PRIVATE_REGISTRY_COLUMNS


SELECTED = 200


def _private_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [f"household_{index:03d}" for index in range(SELECTED)],
            "raw_respondent_id": [f"raw-private-{index:03d}" for index in range(SELECTED)],
            "included_in_master_200": True,
            "included_in_core_81": [index < 81 for index in range(SELECTED)],
        }
    ).loc[:, PRIVATE_REGISTRY_COLUMNS]


def _base_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(SELECTED):
        row = {column: None for column in HISTORY_COLUMNS}
        row.update(
            {
                "household_id": f"household_{index:03d}",
                "event_date": "2025-04-01",
                "public_availability_date": "2026-01-01",
                "source_name": "SCE respondent microdata",
                "observation_status": "observed",
                "responded": True,
                "attrition_status": "responding",
                "death_status": "alive_no_death_observation",
                "replay_required_from_event_date": "2025-04-01",
                "survey_weight": 1.0,
                "income_group": "middle",
                "age_group": "35_44",
                "education_group": "college",
                "actual_expected_inflation_1y": 3.0,
                "actual_expected_unemployment_higher_prob": 20.0,
                "actual_expected_real_income_growth": 1.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, HISTORY_COLUMNS]


def _source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    response_counts = {"2025-04-01": SELECTED, "2025-05-01": 150, "2025-06-01": 130}
    for month, count in response_counts.items():
        for index in range(count):
            rows.append(
                {
                    "respondent_id": f"raw-private-{index:03d}",
                    "survey_date": month,
                    "weight": 1.0,
                    "income_group": "middle",
                    "age_group": "35_44",
                    "education_group": "college",
                    "actual_expected_inflation_1y": 3.0,
                    "actual_expected_unemployment_higher_prob": 20.0,
                    "actual_expected_real_income_growth": 1.0,
                }
            )
    for month in response_counts:
        rows.append(
            {
                "respondent_id": f"raw-private-unselected-{month}",
                "survey_date": month,
                "weight": 1.0,
                "income_group": "middle",
                "age_group": "35_44",
                "education_group": "college",
                "actual_expected_inflation_1y": 3.0,
                "actual_expected_unemployment_higher_prob": 20.0,
                "actual_expected_real_income_growth": 1.0,
            }
        )
    return pd.DataFrame(rows)


class EcologyHistoryTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        base = root / "base_history.csv"
        source = root / "normalized_sce.csv"
        registry = root / "private_registry.csv"
        _base_history().to_csv(base, index=False)
        _source().to_csv(source, index=False)
        _private_registry().to_csv(registry, index=False)
        return base, source, registry

    def _materialize(self, root: Path, *, output_stem: str = "first") -> dict[str, object]:
        base, source, registry = self._write_inputs(root)
        return materialize_ecology_history(
            base_history_csv=base,
            normalized_sce_csv=source,
            private_registry_csv=registry,
            through_event_month="2025-06",
            output_history_csv=root / f"{output_stem}_history.csv",
            output_manifest_json=root / f"{output_stem}_manifest.json",
        )

    def test_materializes_multiple_waves_with_nonresponse_and_availability_dates(self) -> None:
        with TemporaryDirectory() as directory:
            result = self._materialize(Path(directory))
            history = pd.read_csv(result["output_history_csv"])
            self.assertEqual(history.shape[0], 600)
            self.assertEqual(set(history["event_date"]), {"2025-04-01", "2025-05-01", "2025-06-01"})
            for _, wave in history.groupby("event_date"):
                self.assertEqual(len(wave), SELECTED)
                self.assertEqual(wave["household_id"].nunique(), SELECTED)
            may = history[history["event_date"].eq("2025-05-01")]
            june = history[history["event_date"].eq("2025-06-01")]
            self.assertEqual(int(may["responded"].sum()), 150)
            self.assertEqual(int(june["responded"].sum()), 130)
            self.assertEqual(set(may.loc[~may["responded"], "observation_status"]), {"nonresponse"})
            self.assertEqual(
                set(may.loc[~may["responded"], "attrition_status"]),
                {"survey_nonresponse_not_economic_exit"},
            )
            self.assertEqual(set(may["public_availability_date"]), {"2026-02-01"})
            self.assertEqual(set(june["public_availability_date"]), {"2026-03-01"})

    def test_public_artifacts_never_leak_private_respondent_ids(self) -> None:
        with TemporaryDirectory() as directory:
            result = self._materialize(Path(directory))
            public_text = Path(result["output_history_csv"]).read_text(encoding="utf-8")
            manifest_text = Path(result["output_manifest_json"]).read_text(encoding="utf-8")
            self.assertNotIn("raw-private-", public_text)
            self.assertNotIn("raw-private-", manifest_text)
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["selected_household_count"], SELECTED)
            self.assertEqual(manifest["appended_waves"][0]["nonresponse_count"], 50)
            self.assertEqual(manifest["appended_waves"][1]["nonresponse_count"], 70)

    def test_materialization_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._materialize(root, output_stem="first")
            second = self._materialize(root, output_stem="second")
            self.assertEqual(
                Path(first["output_history_csv"]).read_bytes(),
                Path(second["output_history_csv"]).read_bytes(),
            )
            self.assertEqual(
                Path(first["output_manifest_json"]).read_bytes(),
                Path(second["output_manifest_json"]).read_bytes(),
            )

    def test_rejects_malformed_source_without_writing_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, registry = self._write_inputs(root)
            malformed = pd.read_csv(source).drop(columns="weight")
            malformed.to_csv(source, index=False)
            history_output = root / "history.csv"
            manifest_output = root / "manifest.json"
            with self.assertRaisesRegex(EcologyHistoryError, "malformed"):
                materialize_ecology_history(
                    base_history_csv=base,
                    normalized_sce_csv=source,
                    private_registry_csv=registry,
                    through_event_month="2025-06",
                    output_history_csv=history_output,
                    output_manifest_json=manifest_output,
                )
            self.assertFalse(history_output.exists())
            self.assertFalse(manifest_output.exists())

    def test_refuses_nonempty_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, registry = self._write_inputs(root)
            history_output = root / "history.csv"
            manifest_output = root / "manifest.json"
            history_output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(EcologyHistoryError, "refusing to overwrite"):
                materialize_ecology_history(
                    base_history_csv=base,
                    normalized_sce_csv=source,
                    private_registry_csv=registry,
                    through_event_month="2025-06",
                    output_history_csv=history_output,
                    output_manifest_json=manifest_output,
                )
            self.assertEqual(history_output.read_text(encoding="utf-8"), "keep")
            self.assertFalse(manifest_output.exists())

    def test_rejects_history_published_before_its_survey_event(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, source, registry = self._write_inputs(root)
            history = pd.read_csv(base)
            history["public_availability_date"] = "2025-03-01"
            history.to_csv(base, index=False)
            with self.assertRaisesRegex(EcologyHistoryError, "cannot be public"):
                materialize_ecology_history(
                    base_history_csv=base,
                    normalized_sce_csv=source,
                    private_registry_csv=registry,
                    through_event_month="2025-06",
                    output_history_csv=root / "history.csv",
                    output_manifest_json=root / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
