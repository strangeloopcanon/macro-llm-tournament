from __future__ import annotations

import argparse
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from macro_llm_tournament.persona_ecology import (
    PERSONA_ECOLOGY_BACKSTORY_PROMPT_VERSION,
    build_ecology_cards,
    build_fixture_ecology_panel,
    persona_ecology_prompt_payload,
)
from macro_llm_tournament.persona_elicitation_campaign import (
    DEFAULT_TARGET_FIELDS,
    TARGET_SPECS,
    classify_arm1_backstory,
    draw_distributional_predictions,
    no_hints_prompt_payload,
    score_interval_calibration,
    _panel_run_complete,
    _validate_args,
)


class PersonaElicitationCampaignTests(unittest.TestCase):
    def test_ecology_backstory_prompt_uses_separate_version_and_sketch(self) -> None:
        panel = build_fixture_ecology_panel(
            respondent_count=2,
            period_count=2,
            target_fields=DEFAULT_TARGET_FIELDS,
        )
        card = build_ecology_cards(panel, target_fields=DEFAULT_TARGET_FIELDS)[1]
        payload = persona_ecology_prompt_payload(
            card,
            target_fields=DEFAULT_TARGET_FIELDS,
            prior_beliefs={target: 1.0 for target in DEFAULT_TARGET_FIELDS},
            environment=card.environment,
            date_mode="relative",
            elicitation="backstory",
        )

        self.assertEqual(payload["prompt_version"], PERSONA_ECOLOGY_BACKSTORY_PROMPT_VERSION)
        self.assertIn("persona_rule", payload)
        self.assertIn("persona_sketch", payload["required_response"])
        self.assertEqual(payload["survey_date"], "period_0")
        self.assertNotIn("targets", payload)
        self.assertNotIn("actual_expected_inflation_1y", str(payload))

    def test_distributional_draws_and_interval_calibration_use_p10_p90(self) -> None:
        respondents = pd.DataFrame(
            {
                "respondent_id": ["r1", "r2"],
                "weight": [0.5, 0.5],
                "actual_expected_inflation_1y": [2.0, 8.0],
                "actual_expected_unemployment_higher_prob": [20.0, 60.0],
                "actual_expected_real_income_growth": [1.0, -4.0],
            }
        )
        rows = []
        for respondent in ["r1", "r2"]:
            for target in DEFAULT_TARGET_FIELDS:
                spec = TARGET_SPECS[target]
                midpoint = (float(spec["lower"]) + float(spec["upper"])) / 2.0
                rows.append(
                    {
                        "respondent_id": respondent,
                        "source": "llm_codex_cli_gpt-5.5",
                        "target_name": target,
                        "prediction": midpoint,
                        "p10": float(spec["lower"]),
                        "p50": midpoint,
                        "p90": float(spec["upper"]),
                    }
                )
        predictions = pd.DataFrame(rows)

        drawn = draw_distributional_predictions(predictions, target_fields=DEFAULT_TARGET_FIELDS, seed=7)
        self.assertIn("point_prediction", drawn)
        self.assertTrue(((drawn["prediction"] >= -20.0) & (drawn["prediction"] <= 100.0)).all())
        self.assertTrue((drawn["prediction"] != drawn["point_prediction"]).any())
        interval = score_interval_calibration(respondents, predictions, target_fields=DEFAULT_TARGET_FIELDS)
        self.assertTrue((interval["coverage_p10_p90"] >= 0.5).all())

    def test_arm1_verdict_requires_spread_ks_signs_and_levels(self) -> None:
        scoreboard = pd.DataFrame(
            [
                {
                    "source": "llm_codex_cli_gpt-5.5",
                    "model": "gpt-5.5",
                    "within_variance_ratio_improvement": 3.2,
                    "ks_drop": 0.06,
                    "sign_rate_degrade": 0.01,
                    "group_mean_mae_growth": 0.10,
                },
                {
                    "source": "llm_codex_cli_gpt-5.4",
                    "model": "gpt-5.4",
                    "within_variance_ratio_improvement": 5.0,
                    "ks_drop": 0.08,
                    "sign_rate_degrade": 0.20,
                    "group_mean_mae_growth": 0.05,
                },
            ]
        )

        verdicts = classify_arm1_backstory(scoreboard)
        by_model = dict(zip(verdicts["model"], verdicts["verdict"], strict=True))
        self.assertEqual(by_model["gpt-5.5"], "backstory_recovers_spread")
        self.assertEqual(by_model["gpt-5.4"], "backstory_caricature")

    def test_campaign_live_arms_require_explicit_execute_live_and_codex_provider(self) -> None:
        args = argparse.Namespace(
            provider="codex_cli",
            mode="all",
            execute_live=False,
            sample_size=100,
        )
        with self.assertRaises(SystemExit):
            _validate_args(args, models=("gpt-5.5",), target_fields=DEFAULT_TARGET_FIELDS)

        args.execute_live = True
        args.provider = "openai_responses"
        with self.assertRaises(SystemExit):
            _validate_args(args, models=("gpt-5.5",), target_fields=DEFAULT_TARGET_FIELDS)

    def test_no_hints_prompt_contains_distribution_request_without_profile(self) -> None:
        payload = no_hints_prompt_payload("expected_unemployment_higher_prob")
        text = str(payload)
        self.assertIn("d10", payload["required_response"])
        self.assertIn("multiple_of_5_share", payload["required_response"])
        self.assertNotIn("respondent_profile", text)
        self.assertNotIn("actual_", text)

    def test_panel_run_complete_requires_ok_manifest_and_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            self.assertFalse(_panel_run_complete(output_dir))

            (output_dir / "manifest.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            self.assertFalse(_panel_run_complete(output_dir))

            for name in [
                "persona_respondents.csv",
                "persona_belief_predictions.csv",
                "persona_belief_regression_scores.csv",
                "persona_belief_variance_scores.csv",
                "persona_belief_distribution_scores.csv",
                "persona_belief_group_means.csv",
            ]:
                (output_dir / name).write_text("x\n", encoding="utf-8")
            self.assertTrue(_panel_run_complete(output_dir))


if __name__ == "__main__":
    unittest.main()
