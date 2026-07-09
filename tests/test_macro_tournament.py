import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pandas as pd

from macro_llm_tournament.macro_tournament import (
    CONFIRMATORY_LOCK_ERROR,
    RETROSPECTIVE_ONLY_ERROR,
    candidate_id_for,
    confirmatory_surface_keys,
    expand_candidates,
    filter_surface_targets_for_score_dates,
    is_promoted_incumbent,
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
            args = SimpleNamespace(mode="replay", max_live_calls=0)

            validate_spec(spec, args=args)

            registry = {
                "schema_version": "macro_tournament_confirmatory_registry_v1",
                "spent_surface_keys": confirmatory_surface_keys(spec),
                "records": [],
            }
            Path(spec["confirmatory_spent_registry_path"]).write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "surface already spent"):
                validate_spec(spec, args=args)

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
