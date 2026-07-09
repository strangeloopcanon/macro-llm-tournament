import csv
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from macro_llm_tournament.frozen_vintage_bundle import (
    AlfredClient,
    BUNDLE_SCHEMA_VERSION,
    CONTEXT_SERIES,
    FRED_VINTAGE_DATES_URL,
    MODEL_CUTOFFS,
    TARGET_SPECS,
    FixtureAlfredClient,
    FrozenVintageBundleError,
    build_fixture_bundle,
    build_frozen_vintage_bundle,
    contamination_label,
    load_frozen_vintage_bundle,
    parse_monthly_origins,
    validate_frozen_vintage_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class RecordingFixtureClient:
    def __init__(self) -> None:
        self.delegate = FixtureAlfredClient()
        self.observation_requests: list[dict[str, str | None]] = []
        self.vintage_date_requests: list[dict[str, str]] = []

    def observations(self, series_id: str, **kwargs):
        self.observation_requests.append({"series_id": series_id, **kwargs})
        return self.delegate.observations(series_id, **kwargs)

    def vintage_dates(self, series_id: str, **kwargs):
        self.vintage_date_requests.append({"series_id": series_id, **kwargs})
        return self.delegate.vintage_dates(series_id, **kwargs)


class NoVintageDatesFixtureClient:
    def __init__(self) -> None:
        self.delegate = FixtureAlfredClient()

    def observations(self, series_id: str, **kwargs):
        return self.delegate.observations(series_id, **kwargs)

    def vintage_dates(self, series_id: str, **kwargs):
        return []


class FrozenVintageBundleTests(unittest.TestCase):
    def test_target_catalogue_covers_the_requested_macro_families(self) -> None:
        self.assertEqual(
            [(spec.series_id, spec.family, spec.transform) for spec in TARGET_SPECS],
            [
                ("PCE", "demand", "pct_change"),
                ("PCEC96", "demand", "pct_change"),
                ("RSAFS", "demand", "pct_change"),
                ("PSAVERT", "balance_sheet", "diff"),
                ("REVOLSL", "balance_sheet", "pct_change"),
                ("PAYEMS", "labor", "pct_change"),
                ("UNRATE", "labor", "level"),
                ("PCEPI", "prices", "pct_change"),
                ("DSPIC96", "income_policy", "pct_change"),
                ("FEDFUNDS", "income_policy", "level"),
            ],
        )
        self.assertEqual(CONTEXT_SERIES, ("CPIAUCSL", "UMCSENT"))
        self.assertEqual(
            MODEL_CUTOFFS,
            {"gpt-5-codex": "2024-09-30", "gpt-5.4": "2025-08-31", "gpt-5.5": "2025-12-01"},
        )

    def test_parse_monthly_origins_is_inclusive_and_strict(self) -> None:
        self.assertEqual(
            parse_monthly_origins("2024-01-01:2024-03-01"),
            ["2024-01-01", "2024-02-01", "2024-03-01"],
        )
        with self.assertRaisesRegex(FrozenVintageBundleError, "first day"):
            parse_monthly_origins("2024-01-02:2024-03-01")

    def test_fixture_bundle_freezes_history_at_as_of_date_and_uses_next_unreleased_target(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-09-01", "2024-10-01"])
            bundle = load_frozen_vintage_bundle(root)

            self.assertEqual(bundle.manifest["schema_version"], BUNDLE_SCHEMA_VERSION)
            self.assertEqual(bundle.origins[0], {"origin_month": "2024-09-01", "as_of_date": "2024-09-15"})
            self.assertEqual(
                bundle.manifest["target_observation_semantics"]["rule"],
                "next_monthly_observation_after_last_origin_visible_observation",
            )
            self.assertIn("same earliest first-release vintage", bundle.manifest["target_observation_semantics"]["transform_denominator"])
            self.assertEqual(len(bundle.targets), len(TARGET_SPECS) * 2)
            target = next(row for row in bundle.targets if row["origin_month"] == "2024-09-01" and row["series_id"] == "PCE")
            self.assertEqual(target["as_of_date"], "2024-09-15")
            self.assertEqual(target["origin_visible_denominator_date"], "2024-08-01")
            self.assertEqual(target["target_observation_date"], "2024-09-01")
            self.assertEqual(target["first_release_as_of_date"], "2024-09-21")
            self.assertEqual(target["release_detection_method"], "vintage_dates")
            expected = 100 * (float(target["first_release_value"]) / float(target["first_release_denominator_value"]) - 1)
            self.assertAlmostEqual(float(target["target_value"]), expected, places=9)
            self.assertNotIn("latest_revision_value", target)

            level_target = next(row for row in bundle.targets if row["series_id"] == "UNRATE")
            self.assertEqual(level_target["transform"], "level")
            self.assertEqual(level_target["target_value"], level_target["first_release_value"])

            audit = next(row for row in bundle.revision_audit if row["origin_month"] == "2024-09-01" and row["series_id"] == "PCE")
            self.assertNotEqual(audit["first_release_value"], audit["latest_revision_value"])

    def test_earliest_vintage_is_selected_before_later_configured_horizon(self) -> None:
        with TemporaryDirectory() as temp_dir:
            client = RecordingFixtureClient()
            build_frozen_vintage_bundle(
                Path(temp_dir),
                ["2024-09-01"],
                mode="fixture",
                client=client,
                release_lag_days=45,
            )
            bundle = load_frozen_vintage_bundle(temp_dir)
            target = next(row for row in bundle.targets if row["series_id"] == "PCE")
            self.assertEqual(target["first_release_as_of_date"], "2024-09-21")
            self.assertNotEqual(target["first_release_as_of_date"], "2024-10-16")
            self.assertEqual(target["release_detection_method"], "vintage_dates")
            self.assertTrue(client.vintage_date_requests)
            history_requests = [request for request in client.observation_requests if request["observation_end"] == "2024-09-15"]
            self.assertTrue(history_requests)
            self.assertTrue(
                all(request["realtime_start"] == "2024-09-15" and request["realtime_end"] == "2024-09-15" for request in history_requests)
            )

    def test_release_lag_is_only_a_fallback_horizon(self) -> None:
        with TemporaryDirectory() as temp_dir:
            build_frozen_vintage_bundle(
                Path(temp_dir),
                ["2024-09-01"],
                mode="fixture",
                client=NoVintageDatesFixtureClient(),
                release_lag_days=45,
            )
            bundle = load_frozen_vintage_bundle(temp_dir)
            target = next(row for row in bundle.targets if row["series_id"] == "PCE")
            self.assertEqual(target["release_detection_method"], "release_lag_fallback")
            self.assertEqual(target["first_release_as_of_date"], "2024-10-16")

    def test_alfred_client_uses_official_vintage_dates_endpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("macro_llm_tournament.frozen_vintage_bundle.requests.get") as get:
                get.return_value.json.return_value = {"vintage_dates": ["2024-09-21", "2024-10-11"]}
                client = AlfredClient("test-key", Path(temp_dir))
                self.assertEqual(
                    client.vintage_dates("PCE", realtime_start="2024-09-01", realtime_end="2024-10-16"),
                    ["2024-09-21", "2024-10-11"],
                )
                self.assertEqual(get.call_args.args[0], FRED_VINTAGE_DATES_URL)
                self.assertEqual(
                    get.call_args.kwargs["params"],
                    {
                        "series_id": "PCE",
                        "api_key": "test-key",
                        "file_type": "json",
                        "realtime_start": "2024-09-01",
                        "realtime_end": "2024-10-16",
                    },
                )

    def test_target_contamination_is_target_specific_and_has_exact_coverage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-09-01", "2024-10-01"])
            bundle = load_frozen_vintage_bundle(root)
            self.assertEqual(len(bundle.target_contamination), len(TARGET_SPECS) * 2 * len(MODEL_CUTOFFS))
            row = next(
                item
                for item in bundle.target_contamination
                if item["origin_month"] == "2024-09-01" and item["target_name"] == "pce_growth_pct" and item["model"] == "gpt-5-codex"
            )
            self.assertEqual(row["contamination_label"], "potential_training_contamination")
            self.assertEqual(row["origin_information_label"], "origin_as_of_pre_cutoff")
            self.assertEqual(
                contamination_label("gpt-5-codex", "2024-09-01", "2024-10-01"),
                "pre_cutoff_observation_post_cutoff_release",
            )

    def test_validation_fails_closed_for_hash_schema_numeric_and_provenance_mismatches(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-01-01"])
            target_path = root / "targets.csv"
            target_path.write_text(target_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(FrozenVintageBundleError, "Hash mismatch"):
                validate_frozen_vintage_bundle(root)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-01-01"])
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "wrong"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FrozenVintageBundleError, "schema"):
                validate_frozen_vintage_bundle(root)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-01-01"])
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["target_set"] = manifest["target_set"][:-1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FrozenVintageBundleError, "target set"):
                validate_frozen_vintage_bundle(root)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-01-01"])
            _rewrite_payload_and_manifest(root, "targets.csv", lambda text: text.replace(",0.8,", ",nan,", 1))
            with self.assertRaisesRegex(FrozenVintageBundleError, "Non-finite"):
                validate_frozen_vintage_bundle(root)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_fixture_bundle(root, ["2024-01-01"])
            _rewrite_payload_and_manifest(root, "source_requests.csv", lambda text: text.replace("origin_history,2024-01-15,2024-01-15", "origin_history,2024-01-01,2024-01-01", 1))
            with self.assertRaisesRegex(FrozenVintageBundleError, "Source request provenance"):
                validate_frozen_vintage_bundle(root)

    def test_cli_fixture_keeps_month_start_input_and_writes_explicit_as_of_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.frozen_vintage_bundle",
                    "--origins",
                    "2025-12-01:2026-01-01",
                    "--output-dir",
                    temp_dir,
                    "--mode",
                    "fixture",
                    "--refresh",
                    "--release-lag-days",
                    "30",
                    "--as-of-day",
                    "12",
                ],
                cwd=REPO_ROOT,
                env=_env(),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = validate_frozen_vintage_bundle(temp_dir)
            self.assertEqual(manifest["release_lag_days"], 30)
            with (Path(temp_dir) / "origins.csv").open(newline="", encoding="utf-8") as handle:
                origins = list(csv.DictReader(handle))
            self.assertEqual(origins[0], {"origin_month": "2025-12-01", "as_of_date": "2025-12-12"})


def _rewrite_payload_and_manifest(root: Path, file_name: str, rewrite) -> None:
    path = root / file_name
    path.write_text(rewrite(path.read_text(encoding="utf-8")), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"][file_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    signed = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    manifest["bundle_sha256"] = hashlib.sha256(
        json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
