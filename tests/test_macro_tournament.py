import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import macro_llm_tournament.macro_tournament as tournament_module
from macro_llm_tournament.macro_tournament import (
    CONFIRMATORY_LOCK_ERROR,
    RETROSPECTIVE_ONLY_ERROR,
    candidate_id_for,
    confirmatory_surface_keys,
    expand_candidates,
    filter_surface_targets_for_score_dates,
    is_promoted_incumbent,
    run_tournament,
    select_winner,
    validate_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MacroTournamentTests(unittest.TestCase):
    def test_candidate_grid_expands_with_stable_ids(self):
        spec = _tiny_spec_dict()
        spec["candidate_grid"]["belief_gain_global"] = [1.0]
        spec["candidate_grid"]["behavior_mechanisms"] = ["empirical_bridge_v5_stabilized", "empirical_bridge_state_schedule"]
        spec["candidate_grid"]["hybrid_state_weights"] = [0.25, 0.75]

        first = expand_candidates(spec)
        second = expand_candidates(json.loads(json.dumps(spec, sort_keys=True)))

        self.assertEqual([candidate.candidate_id for candidate in first], [candidate.candidate_id for candidate in second])
        self.assertEqual(len(first), 3)
        self.assertEqual(len({candidate.candidate_id for candidate in first}), 3)

    def test_explicit_candidate_list_expands_with_stable_supplied_ids(self):
        conservative = _candidate_payload(
            behavior_mechanism="empirical_bridge_v4",
            belief_gain_global=3.0,
            belief_gain_inflation=1.5,
            belief_gain_income=0.5,
            belief_gain_unemployment=0.5,
            feedback_gain_multiplier=1.5,
        )
        unit_gain = _candidate_payload()
        spec = {
            "schema_version": "macro_economy_tournament_v1",
            "run_id": "test_explicit_candidates",
            "scoring_label": "confirmatory",
            "candidate_list": [
                {**conservative, "candidate_id": candidate_id_for(conservative)},
                {**unit_gain, "candidate_id": candidate_id_for(unit_gain)},
            ],
        }

        candidates = expand_candidates(spec)

        self.assertEqual([candidate.candidate_id for candidate in candidates], [candidate_id_for(conservative), candidate_id_for(unit_gain)])
        self.assertEqual(candidates[0].payload["behavior_mechanism"], "empirical_bridge_v4")
        self.assertEqual(candidates[1].payload["belief_gain_global"], 1.0)

    def test_promoted_incumbent_spec_locks_current_winner(self):
        spec = json.loads((REPO_ROOT / "configs/macro_tournament/incumbent_v1.json").read_text(encoding="utf-8"))

        candidates = expand_candidates(spec)

        self.assertEqual(len(candidates), 1)
        payload = candidates[0].payload
        self.assertEqual(candidates[0].candidate_id, "cand_1623eb882b6a")
        self.assertEqual(payload["behavior_mechanism"], "empirical_bridge_v5_stabilized")
        self.assertEqual(payload["feedback_mode"], "closed_loop")
        self.assertEqual(payload["feedback_gain_multiplier"], 1.5)
        self.assertEqual(payload["belief_gain_global"], 1.5)
        self.assertEqual(payload["belief_gain_inflation"], 1.25)
        self.assertEqual(payload["belief_gain_income"], 0.75)
        self.assertEqual(payload["belief_gain_unemployment"], 1.25)
        self.assertTrue(is_promoted_incumbent(candidates[0].candidate_id, spec))

    def test_select_winner_ignores_adaptive_diagnostic_loss_or_win(self):
        table = pd.DataFrame(
            [
                {
                    "candidate_id": "worse_than_adaptive_but_best_llm",
                    "mean_llm_rmse_scaled": 0.40,
                    "mean_llm_direction_accuracy": 0.5,
                    "mean_adaptive_rmse_scaled": 0.10,
                    "empirical_bridge_clipped_inputs": 0,
                    "max_accounting_abs_residual": 0.0,
                    "simplicity_score": 1.0,
                    "disqualified": False,
                },
                {
                    "candidate_id": "worse_llm",
                    "mean_llm_rmse_scaled": 0.60,
                    "mean_llm_direction_accuracy": 1.0,
                    "mean_adaptive_rmse_scaled": 9.0,
                    "empirical_bridge_clipped_inputs": 0,
                    "max_accounting_abs_residual": 0.0,
                    "simplicity_score": 1.0,
                    "disqualified": False,
                },
            ]
        )

        winner = select_winner(table)

        self.assertEqual(winner["candidate_id"], "worse_than_adaptive_but_best_llm")

    def test_macro_tournament_rejects_unlocked_confirmatory_development_spec(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _tiny_spec_dict()
            spec["scoring_label"] = "confirmatory"
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_tournament",
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "replay",
                    "--max-live-calls",
                    "0",
                    "--output-dir",
                    str(root / "out"),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(CONFIRMATORY_LOCK_ERROR, result.stderr)

    def test_confirmatory_locked_spec_validates_and_spent_registry_blocks_rerun(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _locked_confirmatory_spec(root)
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)

            with patch(
                "macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY",
                Path(spec["confirmatory_spent_registry_path"]),
            ):
                validate_spec(spec, args=args)

            registry = {
                "schema_version": "macro_tournament_confirmatory_registry_v1",
                "spent_surface_keys": confirmatory_surface_keys(spec),
                "records": [],
            }
            Path(spec["confirmatory_spent_registry_path"]).write_text(json.dumps(registry), encoding="utf-8")

            with patch(
                "macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY",
                Path(spec["confirmatory_spent_registry_path"]),
            ):
                with self.assertRaisesRegex(ValueError, "surface already spent"):
                    validate_spec(spec, args=args)

    def test_confirmatory_score_key_cannot_be_bypassed_by_surface_rename_or_history_change(self):
        with TemporaryDirectory() as temp_dir:
            spec = _locked_confirmatory_spec(Path(temp_dir))
            renamed = json.loads(json.dumps(spec))
            renamed["surfaces"][0]["surface_id"] = "renamed_surface"
            renamed["surfaces"][0]["asof_start"] = "2026-01-15"

            self.assertEqual(confirmatory_surface_keys(spec), confirmatory_surface_keys(renamed))

    def test_confirmatory_spec_rejects_noncanonical_registry_path(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical = root / "canonical_registry.json"
            spec = _locked_confirmatory_spec(root)
            spec["confirmatory_spent_registry_path"] = str(root / "alternate_registry.json")
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)

            with patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", canonical):
                with self.assertRaisesRegex(ValueError, CONFIRMATORY_LOCK_ERROR):
                    validate_spec(spec, args=args)

    def test_confirmatory_spec_requires_declared_target_contract(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical = root / "registry.json"
            spec = _locked_confirmatory_spec(root)
            del spec["required_target_names"]
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)

            with patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", canonical):
                with self.assertRaisesRegex(ValueError, CONFIRMATORY_LOCK_ERROR):
                    validate_spec(spec, args=args)

    def test_confirmatory_spec_rejects_all_as_declared_target(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _locked_confirmatory_spec(root)
            spec["required_target_names"].append("ALL")
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)

            with patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", root / "registry.json"):
                with self.assertRaisesRegex(ValueError, CONFIRMATORY_LOCK_ERROR):
                    validate_spec(spec, args=args)

    def test_confirmatory_current_fred_execution_is_disabled_until_frozen_loader(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical = root / "registry.json"
            spec = _locked_confirmatory_spec(root)
            args = SimpleNamespace(mode="replay", max_live_calls=0)

            with patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", canonical):
                with self.assertRaisesRegex(ValueError, "frozen vintage inputs"):
                    validate_spec(spec, args=args)

    def test_confirmatory_target_contract_rejects_missing_target(self):
        validator = getattr(tournament_module, "validate_confirmatory_target_set", None)
        self.assertIsNotNone(validator)
        spec = {"required_target_names": ["pce_mom_pct", "real_pce_mom_pct"]}
        targets = pd.DataFrame(
            [
                {
                    "target_name": "pce_mom_pct",
                    "target_available": True,
                    "target_value": 0.5,
                    "as_of_date": "2026-02-15",
                },
                {
                    "target_name": "real_pce_mom_pct",
                    "target_available": False,
                    "target_value": float("nan"),
                    "as_of_date": "2026-02-15",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing required targets"):
            validator(targets, spec)

    def test_confirmatory_target_contract_rejects_undeclared_target(self):
        spec = {"required_target_names": ["pce_mom_pct", "real_pce_mom_pct"]}
        targets = _confirmatory_targets(spec["required_target_names"])
        targets.loc[len(targets)] = {
            "target_name": "undeclared_extra",
            "target_available": True,
            "target_value": 0.1,
            "as_of_date": "2026-02-15",
        }

        with self.assertRaisesRegex(ValueError, "undeclared=undeclared_extra"):
            tournament_module.validate_confirmatory_target_set(targets, spec)

    def test_confirmatory_scoring_inputs_drop_undeclared_forecast_before_all_score(self):
        spec = {"required_target_names": ["pce_mom_pct", "real_pce_mom_pct"]}
        targets = _confirmatory_targets(spec["required_target_names"])
        forecasts = pd.DataFrame(
            {"target_name": ["pce_mom_pct", "real_pce_mom_pct", "undeclared_extra"]}
        )

        locked_forecasts, locked_targets = tournament_module.locked_confirmatory_scoring_inputs(
            forecasts,
            targets,
            spec,
        )

        self.assertEqual(
            set(locked_forecasts["target_name"]),
            set(spec["required_target_names"]),
        )
        self.assertEqual(
            set(locked_targets["target_name"]),
            set(spec["required_target_names"]),
        )

    def test_confirmatory_score_rows_require_each_target_for_both_sources_and_candidates(self):
        validator = getattr(tournament_module, "validate_confirmatory_score_results", None)
        self.assertIsNotNone(validator)
        spec = {
            "required_target_names": ["pce_mom_pct", "real_pce_mom_pct"],
            "candidate_list": [{"candidate_id": "candidate_a"}, {"candidate_id": "candidate_b"}],
        }
        scores = pd.DataFrame(
            [
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "target_name": target_name,
                    "rmse_scaled": 1.0,
                    "n": 0,
                }
                for candidate_id in ["candidate_a", "candidate_b"]
                for source in ["adaptive", "llm"]
                for target_name in ["pce_mom_pct"]
            ]
        )

        with self.assertRaisesRegex(ValueError, "score target contract mismatch"):
            validator(scores, spec)

    def test_confirmatory_score_rows_reject_undeclared_target_and_validate_all_count(self):
        spec = {
            "required_target_names": ["pce_mom_pct", "real_pce_mom_pct"],
            "candidate_list": [{"candidate_id": "candidate_a"}, {"candidate_id": "candidate_b"}],
        }
        scores = _valid_confirmatory_score_rows(spec)
        tournament_module.validate_confirmatory_score_results(scores, spec)

        scores.loc[len(scores)] = {
            "candidate_id": "candidate_a",
            "source": "llm",
            "target_name": "undeclared_extra",
            "rmse_scaled": 0.1,
            "n": 1,
        }
        with self.assertRaisesRegex(ValueError, "score target contract mismatch"):
            tournament_module.validate_confirmatory_score_results(scores, spec)

        scores = _valid_confirmatory_score_rows(spec)
        scores.loc[
            (scores["candidate_id"] == "candidate_a")
            & (scores["source"] == "llm")
            & (scores["target_name"] == "ALL"),
            "n",
        ] = 3
        with self.assertRaisesRegex(ValueError, "incomplete score rows"):
            tournament_module.validate_confirmatory_score_results(scores, spec)

    def test_confirmatory_result_rejects_partial_candidate_comparison(self):
        validator = getattr(tournament_module, "validate_confirmatory_candidate_results", None)
        self.assertIsNotNone(validator)
        candidates = pd.DataFrame(
            [
                {"candidate_id": "candidate_a", "disqualified": False, "surface_count": 1},
                {"candidate_id": "candidate_b", "disqualified": True, "surface_count": 0},
            ]
        )
        spec = {"candidate_list": [{}, {}], "surfaces": [{}]}

        with self.assertRaisesRegex(ValueError, "incomplete candidate comparison"):
            validator(candidates, spec)

    def test_confirmatory_surface_is_reserved_before_candidate_scoring(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry_path = root / "registry.json"
            spec = _locked_confirmatory_spec(root)
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)
            reserved_during_score: list[bool] = []
            prepared = {
                "surface_id": "fred_confirmatory_2026_02",
                "surface_spec": spec["surfaces"][0],
                "scored_asof_dates": ["2026-02-15"],
                "mapping_sha256": "mapping",
                "input_hashes": {},
            }

            def fake_score(candidate, _surfaces, *, spec, behavior_profile_cache):
                del behavior_profile_cache
                registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
                reserved_during_score.append(
                    bool(set(registry.get("spent_surface_keys", [])) & set(confirmatory_surface_keys(spec)))
                )
                return {
                    "summary": {
                        **candidate.payload,
                        "is_incumbent": False,
                        "is_promoted_incumbent": False,
                        "surface_count": 1,
                        "mean_llm_rmse_scaled": 1.0,
                        "mean_llm_direction_accuracy": 0.5,
                        "mean_adaptive_rmse_scaled": 1.0,
                        "llm_minus_adaptive_rmse_scaled": 0.0,
                        "empirical_bridge_clipped_inputs": 0,
                        "max_accounting_abs_residual": 0.0,
                        "simplicity_score": 1.0,
                        "disqualified": False,
                        "disqualification_reason": "",
                    },
                    "scores": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "source": source,
                            "target_name": target_name,
                            "rmse_scaled": 1.0,
                            "n": (
                                len(spec["required_target_names"])
                                if target_name == "ALL"
                                else 1
                            ),
                        }
                        for source in ["adaptive", "llm"]
                        for target_name in [*spec["required_target_names"], "ALL"]
                    ],
                    "accounting": [],
                }

            with (
                patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", registry_path),
                patch("macro_llm_tournament.macro_tournament.prepare_surface", return_value=prepared),
                patch("macro_llm_tournament.macro_tournament.preload_behavior_profiles", return_value={}),
                patch("macro_llm_tournament.macro_tournament.score_candidate", side_effect=fake_score),
                patch("macro_llm_tournament.macro_tournament.build_report", return_value="report"),
            ):
                run_tournament(spec, args=args, output_dir=root / "out")

            self.assertEqual(reserved_during_score, [True, True])

    def test_confirmatory_reservation_completion_is_persisted(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _locked_confirmatory_spec(root)
            manifest = _confirmatory_manifest(spec)
            record = tournament_module.reserve_confirmatory_surfaces(
                spec,
                output_dir=root / "out",
                manifest=manifest,
            )
            result = _confirmatory_result(manifest, record)

            completed = tournament_module.complete_confirmatory_reservation(record, result)
            registry = tournament_module.load_confirmatory_registry(root / "registry.json")

            self.assertEqual(completed["status"], "completed")
            self.assertIn("completed_at_utc", completed)
            self.assertEqual(registry["records"][0]["status"], "completed")
            self.assertEqual(registry["records"][0]["winner_candidate_id"], "candidate_a")

    def test_confirmatory_reservation_records_scoring_and_output_failures(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _locked_confirmatory_spec(root)
            manifest = _confirmatory_manifest(spec)
            record = tournament_module.reserve_confirmatory_surfaces(
                spec,
                output_dir=root / "scoring_failure",
                manifest=manifest,
            )

            tournament_module.fail_confirmatory_reservation(
                record,
                status="failed_after_reservation",
                error="candidate scoring failed",
            )
            registry = tournament_module.load_confirmatory_registry(root / "registry.json")
            self.assertEqual(registry["records"][0]["status"], "failed_after_reservation")
            self.assertEqual(registry["records"][0]["error"], "candidate scoring failed")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output_failure"
            output_dir.mkdir()
            spec = _locked_confirmatory_spec(root)
            manifest = _confirmatory_manifest(spec)
            record = tournament_module.reserve_confirmatory_surfaces(
                spec,
                output_dir=output_dir,
                manifest=manifest,
            )
            result = _confirmatory_result(manifest, record)

            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    tournament_module.write_tournament_outputs(output_dir, result)

            registry = tournament_module.load_confirmatory_registry(root / "registry.json")
            self.assertEqual(registry["records"][0]["status"], "output_incomplete")
            self.assertEqual(registry["records"][0]["error"], "disk full")

    def test_run_tournament_marks_reserved_surface_failed_when_scoring_raises(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry_path = root / "registry.json"
            spec = _locked_confirmatory_spec(root)
            args = SimpleNamespace(mode="replay", max_live_calls=0, allow_test_unfrozen_confirmatory=True)
            prepared = {
                "surface_id": "fred_confirmatory_2026_02",
                "surface_spec": spec["surfaces"][0],
                "scored_asof_dates": ["2026-02-15"],
                "mapping_sha256": "mapping",
                "input_hashes": {},
            }

            with (
                patch("macro_llm_tournament.macro_tournament.DEFAULT_CONFIRMATORY_REGISTRY", registry_path),
                patch("macro_llm_tournament.macro_tournament.prepare_surface", return_value=prepared),
                patch("macro_llm_tournament.macro_tournament.preload_behavior_profiles", return_value={}),
                patch(
                    "macro_llm_tournament.macro_tournament.score_candidate",
                    side_effect=RuntimeError("scoring exploded"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "scoring exploded"):
                    run_tournament(spec, args=args, output_dir=root / "out")

            registry = tournament_module.load_confirmatory_registry(registry_path)
            self.assertEqual(registry["records"][0]["status"], "failed_after_reservation")
            self.assertEqual(registry["records"][0]["error"], "scoring exploded")

    def test_confirmatory_registry_rejects_malformed_and_duplicate_reservations(self):
        with TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed confirmatory registry"):
                tournament_module.load_confirmatory_registry(registry_path)

            duplicate = {
                "schema_version": "macro_tournament_confirmatory_registry_v1",
                "spent_surface_keys": [],
                "records": [
                    {"reservation_id": "same", "status": "reserved_before_scoring"},
                    {"reservation_id": "same", "status": "reserved_before_scoring"},
                ],
            }
            registry_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate reservation_id"):
                tournament_module.load_confirmatory_registry(registry_path)

    def test_confirmatory_score_date_reservation_is_atomic_under_concurrency(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = _locked_confirmatory_spec(root)
            manifest = _confirmatory_manifest(spec)
            barrier = Barrier(2)

            def reserve_once(index: int):
                barrier.wait()
                return tournament_module.reserve_confirmatory_surfaces(
                    spec,
                    output_dir=root / f"out_{index}",
                    manifest=manifest,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(reserve_once, index) for index in range(2)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except ValueError as exc:
                        outcomes.append(exc)

            successes = [value for value in outcomes if isinstance(value, dict)]
            failures = [value for value in outcomes if isinstance(value, ValueError)]
            registry = tournament_module.load_confirmatory_registry(root / "registry.json")

            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertIn("surface already spent", str(failures[0]))
            self.assertEqual(len(registry["records"]), 1)
            self.assertEqual(
                set(registry["spent_surface_keys"]),
                set(confirmatory_surface_keys(spec)),
            )

    def test_score_asof_date_filter_keeps_alignment_history_out_of_scoring(self):
        targets = pd.DataFrame(
            [
                {"period_id": "blind_post_cutoff_01", "as_of_date": "2025-12-15", "target_name": "pce_mom_pct"},
                {"period_id": "blind_post_cutoff_02", "as_of_date": "2026-01-15", "target_name": "pce_mom_pct"},
                {"period_id": "blind_post_cutoff_03", "as_of_date": "2026-02-15", "target_name": "pce_mom_pct"},
            ]
        )
        surface = {"surface_id": "fresh", "score_asof_dates": ["2026-02-15"]}

        filtered = filter_surface_targets_for_score_dates(targets, surface)

        self.assertEqual(filtered["period_id"].tolist(), ["blind_post_cutoff_03"])

    def test_macro_tournament_fixture_cli_writes_output_contract(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ecology_dir = root / "ecology"
            bridge_path = root / "bridge.json"
            output_dir = root / "out"
            _write_minimal_ecology_dir(ecology_dir)
            _write_accepted_bridge(bridge_path)
            spec = _tiny_spec_dict(ecology_dir=ecology_dir, bridge_path=bridge_path)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "macro_llm_tournament.macro_tournament",
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "replay",
                    "--max-live-calls",
                    "0",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            candidates = pd.read_csv(output_dir / "macro_tournament_candidates.csv")
            scores = pd.read_csv(output_dir / "macro_tournament_scores.csv")
            accounting = pd.read_csv(output_dir / "macro_tournament_accounting.csv")
            winner = json.loads((output_dir / "winner_manifest.json").read_text(encoding="utf-8"))
            report = (output_dir / "macro_tournament_report.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["verdict"], "macro_tournament_development_scored")
            self.assertEqual(manifest["scoring_label"], "retrospective")
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(candidates.shape[0], 2)
            self.assertFalse(scores.empty)
            self.assertFalse(accounting.empty)
            self.assertIn("candidate_id", winner)
            self.assertIn("Adaptive and persistence-style baselines are diagnostics", report)
            self.assertFalse(candidates["disqualified"].astype(bool).all())
            self.assertIn("git", manifest)
            self.assertIn("input_hashes_by_surface", manifest)
            input_hashes = manifest["input_hashes_by_surface"]["fixture_surface"]
            self.assertIn("proxy_frames_sha256", input_hashes)
            self.assertIn("persona_ecology_manifest_sha256", input_hashes)
            self.assertIn("persona_ecology_panel_sha256", input_hashes)
            self.assertIn("persona_ecology_predictions_sha256", input_hashes)


def _confirmatory_targets(target_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_name": target_name,
                "target_available": True,
                "target_value": 0.1 * index,
                "as_of_date": "2026-02-15",
            }
            for index, target_name in enumerate(target_names, start=1)
        ]
    )


def _valid_confirmatory_score_rows(spec: dict[str, object]) -> pd.DataFrame:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in spec["candidate_list"]]
    target_names = [str(target_name) for target_name in spec["required_target_names"]]
    rows = [
        {
            "candidate_id": candidate_id,
            "source": source,
            "target_name": target_name,
            "rmse_scaled": 0.1,
            "n": 1,
        }
        for candidate_id in candidate_ids
        for source in ["adaptive", "llm"]
        for target_name in target_names
    ]
    rows.extend(
        {
            "candidate_id": candidate_id,
            "source": source,
            "target_name": "ALL",
            "rmse_scaled": 0.1,
            "n": len(target_names),
        }
        for candidate_id in candidate_ids
        for source in ["adaptive", "llm"]
    )
    return pd.DataFrame(rows)


def _confirmatory_manifest(spec: dict[str, object]) -> dict[str, object]:
    surface = spec["surfaces"][0]
    return {
        "run_id": spec["run_id"],
        "spec_path": "configs/test_confirmatory.json",
        "spec_sha256": "spec_sha256",
        "normalized_spec_sha256": "normalized_spec_sha256",
        "confirmatory_spent_registry_path": str(spec["confirmatory_spent_registry_path"]),
        "confirmatory_surface_keys": confirmatory_surface_keys(spec),
        "surface_ids": [surface["surface_id"]],
        "scored_asof_dates_by_surface": {
            surface["surface_id"]: surface["score_asof_dates"],
        },
        "mapping_sha256_by_surface": {surface["surface_id"]: "mapping_sha256"},
        "input_hashes_by_surface": {surface["surface_id"]: {}},
        "behavior_profiles_by_candidate": {},
        "git": {"commit": "test", "branch": "test", "dirty": False},
        "winner_candidate_id": "candidate_a",
        "winner_mean_llm_rmse_scaled": 0.1,
        "scoring_label": "confirmatory",
    }


def _confirmatory_result(
    manifest: dict[str, object],
    record: dict[str, object],
) -> dict[str, object]:
    result_manifest = {**manifest, "confirmatory_registry_record": record}
    return {
        "normalized_spec": {"schema_version": "macro_economy_tournament_v1"},
        "manifest": result_manifest,
        "candidate_table": pd.DataFrame([{"candidate_id": "candidate_a"}]),
        "score_table": pd.DataFrame([{"candidate_id": "candidate_a"}]),
        "accounting_table": pd.DataFrame([{"candidate_id": "candidate_a"}]),
        "winner": {"candidate_id": "candidate_a"},
        "report": "report\n",
    }


def _tiny_spec_dict(ecology_dir: Path | None = None, bridge_path: Path | None = None) -> dict[str, object]:
    return {
        "schema_version": "macro_economy_tournament_v1",
        "run_id": "test_macro_tournament",
        "scoring_label": "retrospective",
        "confirmatory_surface_policy": "test keeps confirmatory surfaces sealed",
        "profiles": {
            "empirical_bridge_v4": str(bridge_path or "/tmp/missing_bridge.json"),
            "empirical_bridge_v5_stabilized": str(bridge_path or "/tmp/missing_bridge.json"),
            "state_schedule": "/tmp/missing_state_policy.json",
        },
        "candidate_grid": {
            "belief_gain_global": [1.0, 1.25],
            "belief_gain_inflation": [1.0],
            "belief_gain_income": [1.0],
            "belief_gain_unemployment": [1.0],
            "behavior_mechanisms": ["empirical_bridge_v4"],
            "hybrid_state_weights": [1.0],
            "feedback_modes": ["closed_loop"],
            "feedback_gain_multipliers": [1.0],
        },
        "surfaces": [
            {
                "surface_id": "fixture_surface",
                "belief_source": "persona_ecology_replay",
                "persona_ecology_dir": str(ecology_dir or "/tmp/missing_ecology"),
                "primary_ecology_source": "llm_codex_cli_gpt-5.5",
                "ecology_period_policy": "strict",
                "data_mode": "fixture",
                "cutoff_date": "2025-12-01",
                "asof_start": "2025-12-15",
                "asof_end": "2025-12-15",
                "history_months": 18,
                "period_count": 2,
                "scoring_label": "retrospective",
            }
        ],
    }


def _candidate_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "belief_gain_global": 1.0,
        "belief_gain_inflation": 1.0,
        "belief_gain_income": 1.0,
        "belief_gain_unemployment": 1.0,
        "behavior_mechanism": "empirical_bridge_v5_stabilized",
        "hybrid_state_weight": None,
        "feedback_mode": "closed_loop",
        "feedback_gain_multiplier": 1.0,
    }
    payload.update(overrides)
    return payload


def _locked_confirmatory_spec(root: Path) -> dict[str, object]:
    conservative = _candidate_payload(
        behavior_mechanism="empirical_bridge_v4",
        belief_gain_global=3.0,
        belief_gain_inflation=1.5,
        belief_gain_income=0.5,
        belief_gain_unemployment=0.5,
        feedback_gain_multiplier=1.5,
    )
    unit_gain = _candidate_payload()
    return {
        "schema_version": "macro_economy_tournament_v1",
        "run_id": "test_confirmatory",
        "scoring_label": "confirmatory",
        "confirmatory_lock": True,
        "confirmatory_spent_registry_path": str(root / "registry.json"),
        "required_target_names": [
            "pce_mom_pct",
            "real_pce_mom_pct",
            "retail_sales_mom_pct",
            "personal_saving_rate_pct",
            "revolving_credit_mom_pct",
        ],
        "profiles": {
            "empirical_bridge_v4": str(root / "bridge_v4.json"),
            "empirical_bridge_v5_stabilized": str(root / "bridge_v5.json"),
            "state_schedule": str(root / "state_policy.json"),
        },
        "candidate_list": [
            {**conservative, "candidate_id": candidate_id_for(conservative)},
            {**unit_gain, "candidate_id": candidate_id_for(unit_gain)},
        ],
        "surfaces": [
            {
                "surface_id": "fred_confirmatory_2026_02",
                "belief_source": "persona_ecology_replay",
                "persona_ecology_dir": str(root / "ecology"),
                "primary_ecology_source": "llm_codex_cli_gpt-5.5",
                "ecology_period_policy": "hold_last",
                "data_mode": "fred",
                "cutoff_date": "2025-12-01",
                "asof_start": "2025-12-15",
                "asof_end": "2026-02-15",
                "score_asof_dates": ["2026-02-15"],
                "history_months": 18,
                "scoreable_only": True,
                "period_count": 4,
                "scoring_label": "confirmatory",
            }
        ],
    }


def _write_accepted_bridge(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "empirical_bridge_v4",
                "bridge_spec_version": "empirical_bridge_v4",
                "status": "accepted",
                "canonical_payload_sha256": "test_bridge",
                "between_coefficients": {
                    "actual_expected_inflation_1y": 0.10,
                    "actual_expected_real_income_growth": 0.20,
                    "sce_question_unemployment_higher_prob": -0.01,
                },
                "support": {
                    "actual_expected_inflation_1y": {"min": -5.0, "max": 15.0},
                    "actual_expected_real_income_growth": {"min": -12.0, "max": 12.0},
                    "sce_question_unemployment_higher_prob": {"min": 0.0, "max": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_ecology_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "ok",
        "provider": "codex_cli",
        "models": ["gpt-5.5"],
        "primary_update_source": "llm_codex_cli_gpt-5.5",
        "prior_mode": "empirical",
        "feedback_mode": "none",
        "date_mode": "relative",
        "prior_update_evidence": {"evidence_verdict": "test_fixture"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    panel_rows = []
    for period_index, period_id in enumerate(["sce_2024_10", "sce_2024_11"]):
        for respondent_id, income, liquidity, weight in [
            ("respondent_a", "low", "low", 0.55),
            ("respondent_b", "high", "high", 0.45),
        ]:
            panel_rows.append(
                {
                    "panel_row_id": f"{respondent_id}__{period_id}",
                    "period_id": period_id,
                    "period_index": period_index,
                    "respondent_id": respondent_id,
                    "survey_source": "ny_fed_sce_microdata",
                    "survey_date": "2024-10-01" if period_index == 0 else "2024-11-01",
                    "weight": weight,
                    "age_group": "35_54",
                    "income_group": income,
                    "education_group": "college_plus",
                    "gender": "female",
                    "region": "south",
                    "employment_status": "employed",
                    "homeownership": "owner",
                    "liquid_wealth_group": liquidity,
                    "prior_expected_inflation_1y": 3.0 + period_index,
                    "prior_expected_unemployment_higher_prob": 35.0 + period_index,
                    "prior_expected_real_income_growth": 1.0,
                    "actual_expected_inflation_1y": 3.1 + period_index,
                    "actual_expected_unemployment_higher_prob": 36.0 + period_index,
                    "actual_expected_real_income_growth": 1.1,
                }
            )
    pd.DataFrame(panel_rows).to_csv(root / "persona_ecology_panel.csv", index=False)
    prediction_rows = []
    for panel in panel_rows:
        for target, prior, prediction in [
            ("expected_inflation_1y", panel["prior_expected_inflation_1y"], panel["prior_expected_inflation_1y"] + 0.2),
            ("expected_unemployment_higher_prob", panel["prior_expected_unemployment_higher_prob"], panel["prior_expected_unemployment_higher_prob"] + 1.0),
            ("expected_real_income_growth", panel["prior_expected_real_income_growth"], panel["prior_expected_real_income_growth"] - 0.1),
        ]:
            prediction_rows.append(
                {
                    "schema_version": "persona_belief_ecology_v1",
                    "panel_row_id": panel["panel_row_id"],
                    "respondent_id": panel["respondent_id"],
                    "period_id": panel["period_id"],
                    "period_index": panel["period_index"],
                    "survey_source": "ny_fed_sce_microdata",
                    "survey_date": panel["survey_date"],
                    "source": "llm_codex_cli_gpt-5.5",
                    "provider": "codex_cli",
                    "model": "gpt-5.5",
                    "target_name": target,
                    "prior_prediction": prior,
                    "prediction": prediction,
                    "p10": prediction - 0.5,
                    "p50": prediction,
                    "p90": prediction + 0.5,
                    "confidence": 0.7,
                    "uncertainty": 0.5,
                    "profile_weight": 0.1,
                    "prior_weight": 0.5,
                    "environment_weight": 0.3,
                    "aggregate_feedback_weight": 0.1,
                    "cache_hit": True,
                    "call_source": "cache",
                }
            )
    pd.DataFrame(prediction_rows).to_csv(root / "persona_ecology_predictions.csv", index=False)


if __name__ == "__main__":
    unittest.main()
