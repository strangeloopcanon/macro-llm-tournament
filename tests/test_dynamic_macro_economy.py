from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from macro_llm_tournament.dynamic_macro_economy import (
    FORBIDDEN_PROMPT_KEYS,
    MAPPING_BY_SERIES,
    OUTPUT_FILES,
    PERIODS_PER_YEAR,
    BundleView,
    DynamicMacroError,
    ReplayThenLiveDemandClient,
    anchor_household_flows,
    apply_belief_gains,
    assert_no_prompt_target_leakage,
    canonical_behavior_profile_sha256,
    filter_bundle_targets,
    mapped_value,
    observed_signal_adaptive_payload,
    parse_args,
    run_dynamic_macro,
    score_macro,
    select_score_origins,
    validate_replay_prefix_records,
)
from macro_llm_tournament.demand_economy import (
    DemandScenario,
    _environment_for_period,
    _next_environment,
    build_fixture_demand_households,
)
from macro_llm_tournament.frozen_vintage_bundle import (
    FixtureAlfredClient,
    build_frozen_vintage_bundle,
)
from macro_llm_tournament.llm_common import LLMUnavailable


SERIES = (
    ("pce_growth_pct", "PCE", "demand", "pct_change", 0.8),
    ("real_pce_growth_pct", "PCEC96", "demand", "pct_change", 0.8),
    ("retail_sales_growth_pct", "RSAFS", "demand", "pct_change", 1.0),
    ("personal_saving_rate_change", "PSAVERT", "balance_sheet", "diff", 1.5),
    ("revolving_credit_growth_pct", "REVOLSL", "balance_sheet", "pct_change", 2.0),
    ("payroll_growth_pct", "PAYEMS", "labor", "pct_change", 0.3),
    ("unemployment_rate_level", "UNRATE", "labor", "level", 1.0),
    ("pce_price_growth_pct", "PCEPI", "prices", "pct_change", 0.3),
    (
        "real_disposable_income_growth_pct",
        "DSPIC96",
        "income_policy",
        "pct_change",
        0.8,
    ),
    ("fed_funds_rate_level", "FEDFUNDS", "income_policy", "level", 1.0),
)


class PlausibleFixtureAlfredClient(FixtureAlfredClient):
    @staticmethod
    def _value(series_id: str, observation_date) -> float:
        if series_id == "UNRATE":
            return 4.0 + 0.02 * observation_date.month
        if series_id == "FEDFUNDS":
            return 4.75 + 0.03 * observation_date.month
        return FixtureAlfredClient._value(series_id, observation_date)


def fixture_bundle() -> BundleView:
    origins = (
        {"origin_month": "2025-10-01", "as_of_date": "2025-10-15"},
        {"origin_month": "2025-11-01", "as_of_date": "2025-11-15"},
        {"origin_month": "2025-12-01", "as_of_date": "2025-12-15"},
    )
    history = []
    targets = []
    specs = []
    contamination = []
    for target_index, (name, series_id, family, transform, scale) in enumerate(SERIES):
        specs.append(
            {
                "target_name": name,
                "series_id": series_id,
                "family": family,
                "transform": transform,
                "default_scale": scale,
            }
        )
        for origin_index, origin in enumerate(origins):
            level = 100.0 + 3.0 * target_index + origin_index
            previous_level = level - 1.0
            if series_id == "UNRATE":
                level = 4.0 + 0.1 * origin_index
                previous_level = level - 0.1
            elif series_id == "FEDFUNDS":
                level = 5.0 + 0.1 * origin_index
                previous_level = level - 0.1
            elif series_id == "PSAVERT":
                level = 4.5 + 0.1 * origin_index
                previous_level = level - 0.1
            history.append(
                {
                    **origin,
                    "series_id": series_id,
                    "observation_date": [
                        "2025-09-01",
                        "2025-10-01",
                        "2025-11-01",
                    ][origin_index],
                    "value": str(previous_level),
                }
            )
            history.append(
                {
                    **origin,
                    "series_id": series_id,
                    "observation_date": origin["origin_month"],
                    "value": str(level),
                }
            )
            target_value = 0.25 + 0.03 * target_index + 0.02 * origin_index
            if transform == "level":
                target_value = 4.0 + 0.1 * target_index + 0.02 * origin_index
            target_month = origin["origin_month"]
            targets.append(
                {
                    **origin,
                    "target_name": name,
                    "series_id": series_id,
                    "family": family,
                    "transform": transform,
                    "default_scale": scale,
                    "target_value": target_value,
                    "origin_visible_denominator_value": level,
                    "target_observation_date": target_month,
                }
            )
            contamination.append(
                {
                    **origin,
                    "target_name": name,
                    "model": "gpt-5.5",
                    "model_cutoff_date": "2025-12-01",
                    "target_observation_date": target_month,
                    "first_release_as_of_date": "2026-02-15",
                    "contamination_label": (
                        "post_cutoff_holdout"
                        if target_month > "2025-12-01"
                        else "pre_cutoff_observation_post_cutoff_release"
                    ),
                }
            )
    for origin_index, origin in enumerate(origins):
        history.extend(
            [
                {
                    **origin,
                    "series_id": "CPIAUCSL",
                    "observation_date": origin["origin_month"],
                    "value": str(320.0 + origin_index),
                },
                {
                    **origin,
                    "series_id": "UMCSENT",
                    "observation_date": origin["origin_month"],
                    "value": str(65.0 + origin_index),
                },
            ]
        )
    return BundleView(
        bundle_sha256="fixture-bundle-sha256",
        origins=origins,
        history=tuple(history),
        targets=tuple(targets),
        target_specs=tuple(specs),
        target_contamination=tuple(contamination),
    )


def write_sce_panel(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "respondent_id": "sce_001",
                "period_index": 0,
                "weight": 0.55,
                "income_group": "low",
                "liquid_wealth_group": "low",
                "age_group": "prime",
                "employment_status": "employed",
                "source_survey_event_date": "2024-04-01",
                "source_estimated_public_availability_date": "2025-01-01",
                "prior_expected_inflation_1y": 3.4,
                "prior_expected_unemployment_higher_prob": 42.0,
                "prior_expected_real_income_growth": 0.4,
            },
            {
                "respondent_id": "sce_002",
                "period_index": 0,
                "weight": 0.45,
                "income_group": "high",
                "liquid_wealth_group": "high",
                "age_group": "55_plus",
                "employment_status": "employed",
                "source_survey_event_date": "2024-04-01",
                "source_estimated_public_availability_date": "2025-01-01",
                "prior_expected_inflation_1y": 2.2,
                "prior_expected_unemployment_higher_prob": 24.0,
                "prior_expected_real_income_growth": 1.6,
            },
        ]
    ).to_csv(path, index=False)


def runner_args(
    root: Path,
    panel: Path,
    *,
    mode: str = "fixture",
    feedback_mode: str = "closed_loop",
    feedback_gain: float = 1.0,
    raw_records: Path | None = None,
    replay_prefix_period_count: int = 0,
    score_origin_start: str | None = None,
    score_origin_end: str | None = None,
) -> object:
    argv = [
        "--bundle-dir",
        str(root / "bundle"),
        "--household-panel",
        str(panel),
        "--mode",
        mode,
        "--provider",
        "codex_cli",
        "--model",
        "gpt-5.5",
        "--feedback-mode",
        feedback_mode,
        "--feedback-gain",
        str(feedback_gain),
        "--bootstrap-replicates",
        "20",
        "--bootstrap-seed",
        "12345",
        "--output-dir",
        str(root / "output"),
    ]
    if raw_records is not None:
        argv.extend(["--raw-records-json", str(raw_records)])
    if replay_prefix_period_count:
        argv.extend(
            ["--replay-prefix-period-count", str(replay_prefix_period_count)]
        )
    if score_origin_start:
        argv.extend(["--score-origin-start", score_origin_start])
    if score_origin_end:
        argv.extend(["--score-origin-end", score_origin_end])
    return parse_args(argv)


def run_fixture(
    root: Path, *, feedback_mode: str = "closed_loop", feedback_gain: float = 1.0
):
    panel = root / "sce_panel.csv"
    write_sce_panel(panel)
    args = runner_args(
        root, panel, feedback_mode=feedback_mode, feedback_gain=feedback_gain
    )
    with patch(
        "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
        return_value=fixture_bundle(),
    ):
        manifest = run_dynamic_macro(args)
    return manifest, root / "output"


def prompt_payloads(output_dir: Path, candidate: str) -> list[dict]:
    cards = pd.read_csv(output_dir / "prompt_cards.csv")
    cards = cards[cards["candidate"].eq(candidate)].sort_values("period_index")
    return [json.loads(value) for value in cards["prompt_payload"]]


class DynamicMacroEconomyTests(unittest.TestCase):
    def test_policy_state_assimilation_and_rate_smoothing_are_explicit(self) -> None:
        env = {
            "periods_per_year": 12.0,
            "policy_rate": 4.25,
            "employment_rate": 0.956,
            "inflation_rate": 3.2,
        }
        scenario = DemandScenario(
            scenario_id="policy_inertia",
            label="policy inertia",
            policy_rate_smoothing=0.85,
            policy_state_mode="origin_visible",
        )
        assimilated = _environment_for_period(
            env,
            scenario,
            period_override={
                "origin_visible_state_assimilation": {
                    "policy_rate": {
                        "series_id": "FEDFUNDS",
                        "observation_date": "2026-01-01",
                        "value": 3.64,
                    }
                }
            },
        )
        self.assertEqual(assimilated["policy_rate"], 3.64)
        aggregate = {
            "output_gap_pct": 0.0,
            "aggregate_consumption": 1.0,
            "aggregate_job_loss_belief": 8.0,
            "aggregate_confidence_index": 60.0,
            "aggregate_liquid_buffer_months": 3.0,
        }
        next_env = _next_environment(
            assimilated,
            aggregate,
            scenario,
            feedback_mode="closed_loop",
        )
        unsmoothed = _next_environment(
            assimilated,
            aggregate,
            DemandScenario("unsmoothed", "unsmoothed"),
            feedback_mode="closed_loop",
        )
        self.assertGreater(next_env["policy_rate"], assimilated["policy_rate"])
        self.assertLess(next_env["policy_rate"], unsmoothed["policy_rate"])

    def test_origin_visible_policy_state_requires_a_valid_value(self) -> None:
        scenario = DemandScenario(
            "policy_assimilation",
            "policy assimilation",
            policy_state_mode="origin_visible",
        )
        with self.assertRaisesRegex(ValueError, "finite policy_rate"):
            _environment_for_period(
                {"policy_rate": 3.5},
                scenario,
                period_override={},
            )

    def test_partial_policy_assimilation_preserves_recursive_smoothing(self) -> None:
        initial = {
            "periods_per_year": 12.0,
            "policy_rate": 4.25,
            "employment_rate": 0.956,
            "inflation_rate": 3.2,
        }
        aggregate = {
            "output_gap_pct": 0.0,
            "aggregate_consumption": 1.0,
            "aggregate_job_loss_belief": 8.0,
            "aggregate_confidence_index": 60.0,
            "aggregate_liquid_buffer_months": 3.0,
        }
        override = {
            "origin_visible_state_assimilation": {
                "policy_rate": {
                    "series_id": "FEDFUNDS",
                    "observation_date": "2026-01-01",
                    "value": 3.64,
                }
            }
        }
        unsmoothed = DemandScenario(
            "unsmoothed_partial",
            "unsmoothed partial",
            policy_rate_smoothing=0.0,
            policy_state_mode="origin_visible",
            policy_state_weight=0.5,
        )
        smoothed = DemandScenario(
            "smoothed_partial",
            "smoothed partial",
            policy_rate_smoothing=0.85,
            policy_state_mode="origin_visible",
            policy_state_weight=0.5,
        )
        first_unsmoothed = _environment_for_period(
            initial, unsmoothed, period_override=override
        )
        first_smoothed = _environment_for_period(
            initial, smoothed, period_override=override
        )
        next_unsmoothed = _next_environment(
            first_unsmoothed, aggregate, unsmoothed, feedback_mode="closed_loop"
        )
        next_smoothed = _next_environment(
            first_smoothed, aggregate, smoothed, feedback_mode="closed_loop"
        )
        second_unsmoothed = _environment_for_period(
            next_unsmoothed, unsmoothed, period_override=override
        )
        second_smoothed = _environment_for_period(
            next_smoothed, smoothed, period_override=override
        )
        self.assertNotEqual(
            second_smoothed["policy_rate"], second_unsmoothed["policy_rate"]
        )
        self.assertLess(
            second_smoothed["policy_rate"], second_unsmoothed["policy_rate"]
        )

    def test_replay_live_semantic_retry_quarantines_bad_cached_payload(self) -> None:
        records = [
            {
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "variant": "llm_belief",
                "scenario_id": "recursive_monthly_path",
                "period_index": 0,
                "cache_identity": {"state_identity_sha256": "state-0"},
            }
        ]
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ReplayThenLiveDemandClient(
                "codex_cli",
                "gpt-5.5",
                root / ".cache",
                replay_records=records,
                replay_prefix_period_count=1,
                max_live_calls=3,
                semantic_retry_limit=1,
                execution_cwd=root / "provider_cwd",
            )
            bad_cache = root / ".cache" / "codex_cli" / "bad.json"
            bad_cache.parent.mkdir(parents=True, exist_ok=True)
            bad_cache.write_text('{"payload":{"beliefs":[]}}', encoding="utf-8")
            valid = {"prompt_version": "test", "beliefs_by_type": {}}
            with (
                patch.object(client._live, "belief_cache_path", return_value=bad_cache),
                patch.object(
                    client._live,
                    "belief_panel",
                    side_effect=[
                        LLMUnavailable(
                            "Duplicate demand economy household type_id: h"
                        ),
                        valid,
                    ],
                ),
            ):
                result = client._live_belief_panel_with_semantic_retry(
                    DemandScenario("recursive_monthly_path", "test"),
                    {"period_index": 1},
                    [],
                )
            self.assertEqual(result, valid)
            self.assertEqual(client.semantic_retry_count, 1)
            self.assertFalse(bad_cache.exists())
            self.assertEqual(
                len(list((root / ".cache" / "rejected_semantic").glob("*.json"))),
                1,
            )

    def test_replay_live_does_not_reclassify_provider_failure_as_semantic(self) -> None:
        records = [
            {
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "variant": "llm_belief",
                "scenario_id": "recursive_monthly_path",
                "period_index": 0,
                "cache_identity": {"state_identity_sha256": "state-0"},
            }
        ]
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = ReplayThenLiveDemandClient(
                "codex_cli",
                "gpt-5.5",
                root / ".cache",
                replay_records=records,
                replay_prefix_period_count=1,
                max_live_calls=3,
                semantic_retry_limit=1,
                execution_cwd=root / "provider_cwd",
            )
            with (
                patch.object(
                    client._live,
                    "belief_panel",
                    side_effect=LLMUnavailable("provider unavailable"),
                ),
                patch.object(
                    client._live,
                    "belief_cache_path",
                    return_value=root / "missing.json",
                ),
            ):
                with self.assertRaisesRegex(LLMUnavailable, "provider unavailable"):
                    client._live_belief_panel_with_semantic_retry(
                        DemandScenario("recursive_monthly_path", "test"),
                        {"period_index": 1},
                        [],
                    )
            self.assertEqual(client.semantic_retry_count, 0)

    def test_score_origin_range_keeps_warmup_path_but_excludes_it_from_scores(self) -> None:
        bundle = fixture_bundle()
        selected, contract = select_score_origins(
            bundle,
            start="2025-11-01",
            end="2025-12-01",
        )
        self.assertEqual(
            sorted({row["origin_month"] for row in selected.targets}),
            ["2025-11-01", "2025-12-01"],
        )
        self.assertEqual(contract["warmup_origins"], ["2025-10-01"])
        self.assertEqual(contract["scored_origin_count"], 2)

    def test_replay_prefix_requires_exact_contiguous_identity_bearing_records(self) -> None:
        records = [
            {
                "provider": "codex_cli",
                "model": "gpt-5.5",
                "variant": "llm_belief",
                "scenario_id": "recursive_monthly_path",
                "period_index": index,
                "cache_identity": {"state_identity_sha256": f"state-{index}"},
            }
            for index in range(2)
        ]
        validate_replay_prefix_records(
            records,
            provider="codex_cli",
            model="gpt-5.5",
            prefix_period_count=2,
        )
        with self.assertRaisesRegex(DynamicMacroError, "exactly periods"):
            validate_replay_prefix_records(
                records[:1],
                provider="codex_cli",
                model="gpt-5.5",
                prefix_period_count=2,
            )

    def test_household_flow_anchor_matches_origin_saving_rate_and_preserves_buffers(self) -> None:
        households = build_fixture_demand_households(12)
        households = households.assign(
            population_weight=households["population_weight"] / households["population_weight"].sum()
        )
        before_buffers = households["liquid_assets"] / (households["baseline_consumption_annual"] / 12.0)
        anchored, metadata = anchor_household_flows(
            households,
            {0: {"observed_signal_summary": {"series": {"PSAVERT": {"latest_value": 4.2}}}}},
            mode="origin_saving_rate",
        )
        weights = anchored["population_weight"]
        income = float((weights * anchored["annual_income"]).sum())
        consumption = float((weights * anchored["baseline_consumption_annual"]).sum())
        self.assertAlmostEqual(100.0 * (income - consumption) / income, 4.2)
        self.assertAlmostEqual(metadata["post_anchor_saving_rate_pct"], 4.2)
        after_buffers = anchored["liquid_assets"] / (anchored["baseline_consumption_annual"] / 12.0)
        pd.testing.assert_series_equal(before_buffers.reset_index(drop=True), after_buffers.reset_index(drop=True))

    def test_observed_adaptive_uses_observed_policy_rate_for_policy_gap(self) -> None:
        household = {
            "type_id": "h",
            "attention_weight_prices": 0.5,
            "attention_weight_jobs": 0.5,
            "attention_weight_rates": 0.5,
            "inflation_expectation_1y": 3.0,
            "income_growth_expectation_1y": 1.0,
            "baseline_job_loss_probability": 6.0,
            "job_loss_probability": 6.0,
            "unemployment_higher_probability_1y": 25.0,
            "precautionary_sensitivity": 0.5,
            "confidence_index": 55.0,
            "liquidity_group": "low",
        }
        base_period = {
            "period_id": "period_0",
            "period_index": 0,
            "output_gap_pct": 0.0,
            "policy_rate": 100.0,
            "inflation_rate": 2.0,
            "job_risk_shock_pp": 0.0,
        }

        def payload(observed_policy_rate: float) -> dict:
            period = {
                **base_period,
                "observed_signal_summary": {
                    "derived": {
                        "policy_rate_pct": observed_policy_rate,
                        "inflation_annualized_pct": 2.0,
                        "real_income_growth_annualized_pct": 1.0,
                        "unemployment_rate_change_pp": 0.0,
                        "payroll_growth_pct": 0.0,
                        "sentiment_change": 0.0,
                    }
                },
            }
            return observed_signal_adaptive_payload(
                DemandScenario("test", "test"),
                period,
                [household],
            )["beliefs"][0]

        neutral = payload(2.5)
        tight = payload(10.0)
        self.assertAlmostEqual(
            tight["perceived_job_loss_probability"]
            - neutral["perceived_job_loss_probability"],
            0.375,
        )

    def test_behavior_content_hash_ignores_machine_path_metadata(self) -> None:
        left = {
            "path": "/machine/a/manifest.json",
            "profile_json": "/machine/a/profile.json",
            "empirical_bridge_path": "/machine/a/bridge.json",
            "cells": [{"profile_id": "cell", "response": 0.4}],
        }
        right = {
            "path": "/machine/b/manifest.json",
            "profile_json": "/machine/b/profile.json",
            "empirical_bridge_path": "/machine/b/bridge.json",
            "cells": [{"profile_id": "cell", "response": 0.4}],
        }

        self.assertEqual(
            canonical_behavior_profile_sha256(left),
            canonical_behavior_profile_sha256(right),
        )

    def test_explicit_all_contamination_policy_keeps_exact_catalogue(self) -> None:
        bundle = fixture_bundle()

        selected, coverage = filter_bundle_targets(
            bundle,
            model="gpt-5.5",
            policy="all",
        )

        self.assertEqual(selected.targets, bundle.targets)
        self.assertEqual(coverage["selected_rows"], coverage["catalogue_rows"])
        self.assertEqual(coverage["excluded_rows"], 0)
        self.assertEqual(coverage["excluded_label_counts"], {})

    def test_first_release_policy_keeps_pre_cutoff_events_but_strict_policy_excludes_them(
        self,
    ) -> None:
        base = fixture_bundle()
        contamination = tuple(
            {
                **row,
                "target_observation_date": "2025-12-02",
                "contamination_label": "post_cutoff_holdout",
            }
            if row["origin_month"] == "2025-12-01"
            else row
            for row in base.target_contamination
        )
        bundle = BundleView(
            base.bundle_sha256,
            base.origins,
            base.history,
            base.targets,
            base.target_specs,
            contamination,
        )

        unavailable, unavailable_coverage = filter_bundle_targets(
            bundle,
            model="gpt-5.5",
            policy="unavailable_at_cutoff",
        )
        strict, strict_coverage = filter_bundle_targets(
            bundle,
            model="gpt-5.5",
            policy="strict_post_cutoff_event",
        )

        self.assertEqual(len(unavailable.targets), 30)
        self.assertEqual(unavailable_coverage["first_release_after_cutoff_rows"], 30)
        self.assertEqual(unavailable_coverage["complete_origin_count"], 3)
        self.assertEqual(len(strict.targets), 10)
        self.assertEqual(
            strict_coverage["excluded_label_counts"],
            {"pre_cutoff_observation_post_cutoff_release": 20},
        )
        self.assertEqual(strict_coverage["complete_origin_count"], 1)

    def test_fixture_runs_against_validated_frozen_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "bundle"
            bundle_manifest = build_frozen_vintage_bundle(
                bundle_dir,
                ["2025-10-01", "2025-11-01", "2025-12-01"],
                mode="fixture",
                client=PlausibleFixtureAlfredClient(),
            )
            panel = root / "sce_panel.csv"
            write_sce_panel(panel)
            args = runner_args(root, panel)
            args.model = "gpt-5.4"

            manifest = run_dynamic_macro(args)

            self.assertEqual(
                manifest["bundle_sha256"], bundle_manifest["bundle_sha256"]
            )
            self.assertEqual(manifest["target_count"], len(SERIES))
            self.assertEqual(manifest["origin_count"], 3)
            coverage = manifest["contamination_coverage"]
            self.assertEqual(coverage["policy"], "unavailable_at_cutoff")
            self.assertEqual(
                coverage["selected_rows"] + coverage["excluded_rows"],
                coverage["catalogue_rows"],
            )
            self.assertEqual(
                manifest["scored_target_row_count"], coverage["selected_rows"]
            )

    def test_next_state_target_mappings_ignore_present_period_state(self) -> None:
        previous = {
            "aggregate_consumption": 100.0,
            "employment_rate": 0.50,
            "inflation_rate": 40.0,
            "policy_rate": 20.0,
            "aggregate_income": 10.0,
            "next_employment_rate": 0.80,
            "next_inflation_rate": 6.0,
            "next_policy_rate": 4.0,
            "next_aggregate_income": 100.0,
        }
        current = {
            "aggregate_consumption": 110.0,
            "employment_rate": 0.40,
            "inflation_rate": 50.0,
            "policy_rate": 25.0,
            "aggregate_income": 12.0,
            "next_employment_rate": 0.88,
            "next_inflation_rate": 12.0,
            "next_policy_rate": 7.0,
            "next_aggregate_income": 110.0,
        }

        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["PAYEMS"]),
            10.0,
        )
        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["UNRATE"]),
            12.0,
        )
        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["PCEPI"]),
            100.0 * ((1.12 ** (1.0 / 12.0)) - 1.0),
        )
        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["FEDFUNDS"]),
            7.0,
        )
        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["DSPIC96"]),
            10.0,
        )
        monthly_next_inflation = 100.0 * (1.12 ** (1.0 / 12.0) - 1.0)
        expected_nominal_pce = 100.0 * (
            1.10 * (1.0 + monthly_next_inflation / 100.0) - 1.0
        )
        self.assertAlmostEqual(
            mapped_value(current, previous, MAPPING_BY_SERIES["PCE"]),
            expected_nominal_pce,
        )

    def test_twins_receive_identical_raw_history_and_observed_signals(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest, output = run_fixture(Path(temp_dir))
            llm_prompts = prompt_payloads(output, "llm")
            adaptive_prompts = prompt_payloads(output, "adaptive")
            periods = pd.read_csv(output / "periods.csv")
            anchor = manifest["initial_environment_anchor"]

            for llm_prompt, adaptive_prompt in zip(llm_prompts, adaptive_prompts):
                llm_conditions = llm_prompt["current_exogenous_conditions"][
                    "supplied_exogenous_conditions"
                ]
                adaptive_conditions = adaptive_prompt["current_exogenous_conditions"][
                    "supplied_exogenous_conditions"
                ]
                self.assertEqual(llm_conditions, adaptive_conditions)
                self.assertIn("origin_visible_macro_history", llm_conditions)
                self.assertIn("observed_signal_summary", llm_conditions)
            for prompts in (llm_prompts, adaptive_prompts):
                initial_environment = prompts[0]["current_environment"]
                for field in (
                    "employment_rate",
                    "inflation_rate",
                    "policy_rate",
                    "output_gap_pct",
                ):
                    self.assertAlmostEqual(
                        initial_environment[field], anchor[field], places=4
                    )
            for candidate, prompts in (
                ("llm", llm_prompts),
                ("adaptive", adaptive_prompts),
            ):
                candidate_periods = periods[
                    periods["candidate"].eq(candidate)
                ].sort_values("period_index")
                carry_fields = {
                    "output_gap_pct": "next_output_gap_pct",
                    "employment_rate": "next_employment_rate",
                    "inflation_rate": "next_inflation_rate",
                    "policy_rate": "next_policy_rate",
                }
                for period_index in range(1, len(prompts)):
                    for prompt_field, period_field in carry_fields.items():
                        self.assertAlmostEqual(
                            prompts[period_index]["current_environment"][prompt_field],
                            candidate_periods.iloc[period_index - 1][period_field],
                            places=4,
                        )

            spec = json.loads((output / "normalized_spec.json").read_text())
            self.assertEqual(spec["initial_environment_anchor"], anchor)
            self.assertEqual(
                spec["initial_environment_anchor_origin"],
                manifest["initial_environment_anchor_origin"],
            )

            adaptive_beliefs = pd.read_csv(output / "beliefs.csv")
            adaptive_beliefs = adaptive_beliefs[
                adaptive_beliefs["candidate"].eq("adaptive")
            ]
            self.assertTrue(
                adaptive_beliefs["reason_codes_json"]
                .str.contains("adaptive_observed_as_of_history")
                .all()
            )

    def test_fixture_feedback_changes_next_prompt_and_none_does_not(self) -> None:
        with TemporaryDirectory() as feedback_temp, TemporaryDirectory() as none_temp:
            _, feedback_output = run_fixture(
                Path(feedback_temp), feedback_mode="closed_loop", feedback_gain=1.4
            )
            _, none_output = run_fixture(
                Path(none_temp), feedback_mode="none", feedback_gain=1.4
            )

            feedback_prompts = prompt_payloads(feedback_output, "llm")
            none_prompts = prompt_payloads(none_output, "llm")
            self.assertEqual(
                feedback_prompts[0]["current_environment"]["output_gap_pct"], 0.0
            )
            self.assertEqual(
                none_prompts[1]["current_environment"]["output_gap_pct"], 0.0
            )
            self.assertNotEqual(
                feedback_prompts[1]["current_environment"]["output_gap_pct"], 0.0
            )
            self.assertNotEqual(
                feedback_prompts[1]["current_environment"],
                none_prompts[1]["current_environment"],
            )

    def test_belief_gains_apply_to_update_deltas_from_recursive_prior(self) -> None:
        panel = {
            "prompt_version": "test",
            "beliefs_by_type": {
                "h": {
                    "type_id": "h",
                    "expected_inflation_next_period": 6.0,
                    "expected_income_growth_next_period": -1.0,
                    "perceived_job_loss_probability": 14.0,
                    "expected_unemployment_higher_probability_next_period": 50.0,
                    "confidence_index": 40.0,
                    "reason_codes": [],
                }
            },
            "direct_actions_by_type": {},
        }
        first_state = {
            "type_id": "h",
            "inflation_expectation_1y": 2.0,
            "income_growth_expectation_1y": 1.0,
            "job_loss_probability": 6.0,
            "unemployment_higher_probability_1y": 30.0,
            "confidence_index": 60.0,
        }
        gains = {
            "global": 0.5,
            "inflation": 0.5,
            "income": 1.0,
            "unemployment": 1.5,
            "confidence": 1.0,
        }
        first = apply_belief_gains(panel, [first_state], gains)["beliefs_by_type"]["h"]
        self.assertAlmostEqual(first["expected_inflation_next_period"], 3.0)
        self.assertAlmostEqual(first["expected_income_growth_next_period"], 0.0)
        self.assertAlmostEqual(first["perceived_job_loss_probability"], 12.0)
        self.assertAlmostEqual(
            first["expected_unemployment_higher_probability_next_period"], 45.0
        )

        recursive_state = {
            **first_state,
            "inflation_expectation_1y": first["expected_inflation_next_period"],
            "income_growth_expectation_1y": first["expected_income_growth_next_period"],
            "job_loss_probability": first["perceived_job_loss_probability"],
            "unemployment_higher_probability_1y": first[
                "expected_unemployment_higher_probability_next_period"
            ],
            "confidence_index": first["confidence_index"],
        }
        second = apply_belief_gains(panel, [recursive_state], gains)["beliefs_by_type"][
            "h"
        ]
        self.assertAlmostEqual(second["expected_inflation_next_period"], 3.75)
        self.assertAlmostEqual(second["perceived_job_loss_probability"], 13.5)

    def test_fixture_twins_share_state_and_use_monthly_scaling(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, output = run_fixture(Path(temp_dir))
            households = pd.read_csv(output / "households.csv").set_index("type_id")
            decisions = pd.read_csv(output / "decisions.csv")
            period_zero = decisions[decisions["period_index"].eq(0)]
            for type_id, group in period_zero.groupby("type_id"):
                self.assertEqual(set(group["candidate"]), {"llm", "adaptive"})
                self.assertEqual(group["liquid_assets_before"].nunique(), 1)
                self.assertEqual(group["labor_income"].nunique(), 1)
                expected = households.loc[type_id, "annual_income"] / PERIODS_PER_YEAR
                self.assertTrue(
                    all(
                        math.isclose(value, expected) for value in group["labor_income"]
                    )
                )
            periods = pd.read_csv(output / "periods.csv")
            self.assertEqual(set(periods["periods_per_year"]), {12.0})

    def test_fixture_scores_only_locked_origins_after_recursive_warmup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            panel = root / "sce_panel.csv"
            write_sce_panel(panel)
            args = runner_args(
                root,
                panel,
                score_origin_start="2025-11-01",
                score_origin_end="2025-12-01",
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                return_value=fixture_bundle(),
            ):
                manifest = run_dynamic_macro(args)
            periods = pd.read_csv(root / "output" / "periods.csv")
            scores = pd.read_csv(root / "output" / "origin_scores.csv")
            self.assertEqual(sorted(periods["origin_month"].unique()), [
                "2025-10-01",
                "2025-11-01",
                "2025-12-01",
            ])
            self.assertEqual(sorted(scores["origin_month"].unique()), [
                "2025-11-01",
                "2025-12-01",
            ])
            self.assertEqual(manifest["score_origin_contract"]["warmup_origins"], [
                "2025-10-01"
            ])

    def test_fixture_has_no_target_leakage_and_exact_output_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest, output = run_fixture(Path(temp_dir))
            self.assertEqual(
                set(path.name for path in output.iterdir() if path.is_file()),
                set(OUTPUT_FILES),
            )
            self.assertEqual(manifest["output_contract"], list(OUTPUT_FILES))
            for payload in prompt_payloads(output, "llm") + prompt_payloads(
                output, "adaptive"
            ):
                serialized = json.dumps(payload, sort_keys=True)
                for forbidden in FORBIDDEN_PROMPT_KEYS:
                    self.assertNotIn(f'"{forbidden}"', serialized)
                self.assertIn("origin_visible_macro_context", serialized)
            spec_text = (output / "normalized_spec.json").read_text(encoding="utf-8")
            self.assertNotIn('"target_value"', spec_text)
            self.assertNotIn("latest_revision", spec_text)
            spec = json.loads(spec_text)
            provenance = manifest["household_provenance"]
            self.assertEqual(spec["household_provenance"], provenance)
            self.assertEqual(
                provenance["raw_input_file_sha256"],
                hashlib.sha256(
                    (Path(temp_dir) / "sce_panel.csv").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(len(provenance["normalized_household_state_sha256"]), 64)
            self.assertIsNone(manifest["behavior_policy_content_sha256"])

    def test_prompt_leakage_guard_rejects_nested_target_aliases(self) -> None:
        for forbidden in (
            "target_month",
            "target_realized",
            "realized",
            "forecast_error",
            "first_release",
            "latest_revision",
            "outcome",
            "origin_visible_denominator_value",
            "first_release_denominator_date",
            "first_release_denominator_value",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(DynamicMacroError, forbidden):
                    assert_no_prompt_target_leakage(
                        [{"prompt_payload": {"nested": [{forbidden: 1.0}]}}]
                    )

    def test_raw_record_replay_is_deterministic_and_identity_checked(self) -> None:
        with (
            TemporaryDirectory() as fixture_temp,
            TemporaryDirectory() as replay_one_temp,
            TemporaryDirectory() as replay_two_temp,
        ):
            _, fixture_output = run_fixture(Path(fixture_temp))
            records = fixture_output / "raw_records.json"
            comparable_files = [
                "households.csv",
                "prompt_cards.csv",
                "beliefs.csv",
                "decisions.csv",
                "periods.csv",
                "accounting.csv",
                "forecasts.csv",
                "joined_errors.csv",
                "target_scores.csv",
                "family_scores.csv",
                "origin_scores.csv",
                "raw_records.json",
                "normalized_spec.json",
            ]
            outputs = []
            for replay_temp in (replay_one_temp, replay_two_temp):
                root = Path(replay_temp)
                panel = root / "sce_panel.csv"
                write_sce_panel(panel)
                args = runner_args(root, panel, mode="replay", raw_records=records)
                with patch(
                    "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                    return_value=fixture_bundle(),
                ):
                    run_dynamic_macro(args)
                outputs.append(root / "output")
            for filename in comparable_files:
                self.assertEqual(
                    (outputs[0] / filename).read_bytes(),
                    (outputs[1] / filename).read_bytes(),
                    filename,
                )
            for output in outputs:
                self.assertTrue((output / "provider_cwd").is_dir())
                self.assertEqual(list((output / "provider_cwd").iterdir()), [])
                spec = json.loads((output / "normalized_spec.json").read_text())
                isolation = spec["provider_execution_isolation"]
                self.assertTrue(isolation["enabled"])
                self.assertEqual(isolation["relative_path"], "provider_cwd")
                self.assertNotIn("provider_cwd", spec.get("output_contract", []))

            tampered = json.loads(records.read_text(encoding="utf-8"))
            llm_record = next(
                row for row in tampered if row.get("candidate") == "llm_belief"
            )
            llm_record["cache_identity"]["model"] = "wrong-model"
            bad_records = Path(replay_one_temp) / "tampered.json"
            bad_records.write_text(json.dumps(tampered), encoding="utf-8")
            bad_root = Path(replay_one_temp) / "bad"
            bad_root.mkdir()
            bad_panel = bad_root / "sce_panel.csv"
            write_sce_panel(bad_panel)
            bad_args = runner_args(
                bad_root, bad_panel, mode="replay", raw_records=bad_records
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                return_value=fixture_bundle(),
            ):
                with self.assertRaisesRegex(
                    DynamicMacroError, "Replay identity mismatch"
                ):
                    run_dynamic_macro(bad_args)

    def test_household_tamper_changes_spec_identity_and_replay_fails(self) -> None:
        with (
            TemporaryDirectory() as original_temp,
            TemporaryDirectory() as changed_temp,
            TemporaryDirectory() as replay_temp,
        ):
            original_manifest, original_output = run_fixture(Path(original_temp))
            changed_root = Path(changed_temp)
            changed_panel = changed_root / "sce_panel.csv"
            write_sce_panel(changed_panel)
            changed = pd.read_csv(changed_panel)
            changed.loc[0, "prior_expected_inflation_1y"] += 0.75
            changed.to_csv(changed_panel, index=False)
            changed_args = runner_args(changed_root, changed_panel)
            with patch(
                "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                return_value=fixture_bundle(),
            ):
                changed_manifest = run_dynamic_macro(changed_args)

            self.assertNotEqual(
                original_manifest["household_provenance"][
                    "normalized_household_state_sha256"
                ],
                changed_manifest["household_provenance"][
                    "normalized_household_state_sha256"
                ],
            )
            self.assertNotEqual(
                original_manifest["normalized_spec_sha256"],
                changed_manifest["normalized_spec_sha256"],
            )

            replay_root = Path(replay_temp)
            replay_panel = replay_root / "sce_panel.csv"
            changed.to_csv(replay_panel, index=False)
            replay_args = runner_args(
                replay_root,
                replay_panel,
                mode="replay",
                raw_records=original_output / "raw_records.json",
            )
            with patch(
                "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                return_value=fixture_bundle(),
            ):
                with self.assertRaisesRegex(
                    DynamicMacroError,
                    "Replay identity mismatch",
                ):
                    run_dynamic_macro(replay_args)

    def test_future_available_household_input_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            panel = root / "sce_panel.csv"
            write_sce_panel(panel)
            future = pd.read_csv(panel)
            future["source_estimated_public_availability_date"] = "2026-01-01"
            future.to_csv(panel, index=False)
            args = runner_args(root, panel, mode="replay")

            with patch(
                "macro_llm_tournament.dynamic_macro_economy.load_bundle_view",
                return_value=fixture_bundle(),
            ):
                with self.assertRaisesRegex(
                    DynamicMacroError,
                    "not publicly available by the first macro origin",
                ):
                    run_dynamic_macro(args)

    def test_family_equal_macro_score_does_not_overweight_large_family(self) -> None:
        rows = []
        for candidate, errors in (
            ("llm", {"large_a": 1.0, "large_b": 1.0, "large_c": 1.0, "small_a": 3.0}),
            (
                "adaptive",
                {"large_a": 0.0, "large_b": 0.0, "large_c": 0.0, "small_a": 0.0},
            ),
        ):
            for target_name, scaled_error in errors.items():
                rows.append(
                    {
                        "candidate": candidate,
                        "origin_month": "2025-01-01",
                        "family": (
                            "large" if target_name.startswith("large") else "small"
                        ),
                        "target_name": target_name,
                        "scaled_squared_error": scaled_error**2,
                        "absolute_scaled_error": abs(scaled_error),
                        "direction_correct": True,
                    }
                )
        _, family_scores, _, macro_scores = score_macro(pd.DataFrame(rows))
        self.assertAlmostEqual(macro_scores["llm"], math.sqrt((1.0 + 9.0) / 2.0))
        self.assertAlmostEqual(macro_scores["adaptive"], 0.0)
        llm_weights = family_scores[family_scores["candidate"].eq("llm")][
            "family_equal_weight"
        ]
        self.assertEqual(list(llm_weights), [0.5, 0.5])

    def test_live_mode_rejects_output_reuse(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            panel = root / "sce_panel.csv"
            write_sce_panel(panel)
            output = root / "output"
            output.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")
            args = runner_args(root, panel, mode="live")
            args.max_live_calls = 3
            with self.assertRaisesRegex(DynamicMacroError, "refuses to reuse"):
                run_dynamic_macro(args)


if __name__ == "__main__":
    unittest.main()
