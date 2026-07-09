"""Canonical recursive monthly macro runner over a frozen vintage bundle.

The runner keeps the information boundary deliberately narrow: only
origin-visible history enters period overrides and prompts. Frozen outcomes are
joined after both matched economies have completed their continuous paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .agent_common import WORK_ROOT, cache_key, markdown_table
from .demand_economy import (
    DEMAND_ECONOMY_PROMPT_VERSION,
    DemandEconomyClient,
    DemandScenario,
    adaptive_belief_payload,
    belief_module_prompt_payload,
    build_hybrid_behavior_policy_profile,
    load_behavior_policy_profile,
    load_empirical_bridge_profile,
    load_state_behavior_policy_profile,
    normalize_belief_payload,
    normalize_demand_households,
    run_demand_economy,
)
from .frozen_vintage_bundle import load_frozen_vintage_bundle
from .llm_common import LLMUnavailable


SCHEMA_VERSION = "recursive_dynamic_macro_economy_v2"
PERIODS_PER_YEAR = 12.0
ACCOUNTING_TOLERANCE = 1e-8
DEFAULT_BOOTSTRAP_SEED = 20260709
DEFAULT_BOOTSTRAP_REPLICATES = 1000
CONTAMINATION_LABELS = (
    "post_cutoff_holdout",
    "potential_training_contamination",
    "pre_cutoff_observation_post_cutoff_release",
)
CONTAMINATION_POLICIES = (
    "unavailable_at_cutoff",
    "strict_post_cutoff_event",
    "all",
)
REQUIRED_FAMILIES = frozenset(
    {"demand", "balance_sheet", "labor", "prices", "income_policy"}
)
FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "target_name",
        "target_value",
        "target_observation_date",
        "first_release_value",
        "first_release_as_of_date",
        "latest_revision_value",
        "latest_minus_first_release",
        "revision_audit",
        "target_contamination",
    }
)
OUTPUT_FILES = (
    "normalized_spec.json",
    "manifest.json",
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
    "report.md",
)


class DynamicMacroError(ValueError):
    """Raised when a reproducibility, identity, or scoring contract fails."""


@dataclass(frozen=True)
class TargetMapping:
    series_id: str
    economy_measure: str
    economy_transform: str
    description: str


MAPPING_BY_SERIES: dict[str, TargetMapping] = {
    "PCE": TargetMapping(
        "PCE",
        "aggregate_consumption",
        "nominal_pct_change",
        "Monthly real consumption growth compounded with the model monthly price rate.",
    ),
    "PCEC96": TargetMapping(
        "PCEC96",
        "aggregate_consumption",
        "pct_change",
        "Monthly percentage change in aggregate real consumption.",
    ),
    "RSAFS": TargetMapping(
        "RSAFS",
        "aggregate_consumption",
        "pct_change",
        "Monthly percentage change in aggregate consumption demand.",
    ),
    "PSAVERT": TargetMapping(
        "PSAVERT",
        "saving_rate_pct",
        "diff",
        "Monthly change in aggregate saving as a percent of aggregate income.",
    ),
    "REVOLSL": TargetMapping(
        "REVOLSL",
        "aggregate_debt",
        "pct_change",
        "Monthly percentage change in the household debt stock.",
    ),
    "PAYEMS": TargetMapping(
        "PAYEMS",
        "next_employment_rate",
        "pct_change",
        "Monthly percentage change in post-transition model employment.",
    ),
    "UNRATE": TargetMapping(
        "UNRATE",
        "next_unemployment_rate_pct",
        "level",
        "One minus post-transition model employment, expressed in percent.",
    ),
    "PCEPI": TargetMapping(
        "PCEPI",
        "next_inflation_rate",
        "annual_rate_to_monthly_pct",
        "Post-transition annualized inflation converted to an exact monthly percentage rate.",
    ),
    "DSPIC96": TargetMapping(
        "DSPIC96",
        "next_aggregate_income",
        "pct_change",
        "Monthly percentage change in post-transition aggregate real labor income.",
    ),
    "FEDFUNDS": TargetMapping(
        "FEDFUNDS",
        "next_policy_rate",
        "level",
        "Post-transition model policy-rate level in percent.",
    ),
}


@dataclass(frozen=True)
class BundleView:
    bundle_sha256: str
    origins: tuple[dict[str, str], ...]
    history: tuple[dict[str, str], ...]
    targets: tuple[dict[str, Any], ...]
    target_specs: tuple[dict[str, Any], ...]
    target_contamination: tuple[dict[str, str], ...] = ()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched recursive monthly macro economies over a frozen bundle."
    )
    parser.add_argument("--bundle-dir", required=True)
    household_group = parser.add_mutually_exclusive_group(required=True)
    household_group.add_argument(
        "--household-panel", help="Prepared real-SCE respondent panel CSV."
    )
    household_group.add_argument(
        "--households-csv", help="Demand-economy-ready household CSV."
    )
    parser.add_argument(
        "--mode", choices=("fixture", "replay", "live"), default="fixture"
    )
    parser.add_argument("--provider", choices=("codex_cli",), default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--contamination-policy",
        choices=CONTAMINATION_POLICIES,
        default="unavailable_at_cutoff",
        help="Select by first-release availability, strict post-cutoff event date, or explicit all-target scoring.",
    )
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument(
        "--raw-records-json",
        help="Identity-bearing raw records to use instead of replay cache files.",
    )
    parser.add_argument(
        "--behavior-policy-mode",
        choices=(
            "fixed_kernel",
            "schedule",
            "state_schedule",
            "empirical_bridge",
            "empirical_bridge_state_schedule",
        ),
        default="fixed_kernel",
    )
    parser.add_argument(
        "--behavior-policy-profile",
        help="Generic profile path for the selected non-hybrid policy mode.",
    )
    parser.add_argument(
        "--behavior-policy-raw-records-json",
        help="Behavior-ecology raw records for schedule mode.",
    )
    parser.add_argument(
        "--behavior-policy-state-profile-json",
        help="State-conditioned policy profile JSON.",
    )
    parser.add_argument(
        "--empirical-bridge-json", help="Accepted empirical bridge profile JSON."
    )
    parser.add_argument("--hybrid-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--feedback-mode", choices=("closed_loop", "none"), default="closed_loop"
    )
    parser.add_argument("--feedback-gain", type=float, default=1.0)
    parser.add_argument(
        "--household-flow-anchor",
        choices=("origin_saving_rate", "none"),
        default="origin_saving_rate",
        help="Calibrate the initial aggregate consumption/saving level from origin-visible PSAVERT.",
    )
    parser.add_argument("--belief-gain-global", type=float, default=1.0)
    parser.add_argument("--belief-gain-inflation", type=float, default=1.0)
    parser.add_argument("--belief-gain-income", type=float, default=1.0)
    parser.add_argument("--belief-gain-unemployment", type=float, default=1.0)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-block-length", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_dynamic_macro(args)
    except (DynamicMacroError, LLMUnavailable, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        _canonical_json(
            {
                "output_dir": str(Path(args.output_dir)),
                "macro_scores": manifest["macro_scores"],
            }
        )
    )
    return 0


def run_dynamic_macro(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir)
    if args.mode == "live" and output_dir.exists():
        raise DynamicMacroError(
            "Live mode refuses to reuse an existing output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_bundle = load_bundle_view(Path(args.bundle_dir))
    bundle, contamination_coverage = filter_bundle_targets(
        loaded_bundle,
        model=args.model,
        policy=args.contamination_policy,
    )
    households = load_households(args)
    household_temporal_coverage = validate_household_temporal_availability(
        args,
        households,
        first_origin=bundle.origins[0],
    )
    period_overrides = build_period_overrides(bundle)
    households, household_flow_anchor = anchor_household_flows(
        households,
        period_overrides,
        mode=args.household_flow_anchor,
    )
    household_provenance = household_input_provenance(
        args,
        households,
        temporal_coverage=household_temporal_coverage,
    )
    initial_environment_anchor = build_initial_environment_anchor(period_overrides)
    behavior_profile = load_behavior_profile(args)
    behavior_profile_content_sha256 = canonical_behavior_profile_sha256(
        behavior_profile
    )
    gains = belief_gains_from_args(args)
    raw_replay_records = (
        load_raw_records(Path(args.raw_records_json)) if args.raw_records_json else None
    )

    cache_dir = (
        output_dir / ".cache"
        if args.fresh_cache
        else WORK_ROOT / "dynamic_macro_economy_cache"
    )
    provider_cwd = (
        output_dir / "provider_cwd" if args.mode in {"live", "replay"} else None
    )
    if provider_cwd is not None:
        provider_cwd.mkdir(parents=True, exist_ok=True)
        if any(provider_cwd.iterdir()):
            raise DynamicMacroError("Isolated provider_cwd must be empty at run start")
    llm_mode = (
        "raw_replay"
        if args.mode == "replay" and raw_replay_records is not None
        else args.mode
    )
    llm_base = DemandEconomyClient(
        args.provider,
        args.model,
        cache_dir,
        mode=llm_mode,
        variant="llm_belief",
        max_live_calls=args.max_live_calls,
        raw_replay_records=raw_replay_records,
        execution_cwd=provider_cwd,
    )
    llm_client = GainAdjustedDemandClient(
        llm_base,
        gains=gains,
        requested_mode=args.mode,
        replay_records=raw_replay_records,
    )
    adaptive_client = ObservedSignalAdaptiveClient()
    scenario = DemandScenario(
        scenario_id="recursive_monthly_path",
        label="Continuous frozen-origin monthly path",
        feedback_gain=float(args.feedback_gain),
        notes="One continuous trajectory; targets are withheld until scoring.",
    )

    result_frames: dict[str, list[pd.DataFrame]] = {
        "initial": [],
        "beliefs": [],
        "decisions": [],
        "periods": [],
        "accounting": [],
    }
    prompt_rows: list[dict[str, Any]] = []
    for candidate, client in (("llm", llm_client), ("adaptive", adaptive_client)):
        initial, beliefs, decisions, periods, accounting, prompts = run_demand_economy(
            households,
            [scenario],
            client,
            period_count=len(bundle.origins),
            feedback_mode=args.feedback_mode,
            behavior_policy_profile=behavior_profile,
            periods_per_year=PERIODS_PER_YEAR,
            period_overrides=period_overrides,
            initial_environment_override=initial_environment_anchor,
        )
        for frame_name, frame in (
            ("initial", initial),
            ("beliefs", beliefs),
            ("decisions", decisions),
            ("periods", periods),
            ("accounting", accounting),
        ):
            tagged = frame.copy()
            tagged.insert(0, "candidate", candidate)
            result_frames[frame_name].append(tagged)
        for row in prompts:
            prompt_rows.append({"candidate": candidate, **row})

    initial_frame = pd.concat(result_frames["initial"], ignore_index=True)
    beliefs_frame = pd.concat(result_frames["beliefs"], ignore_index=True)
    decisions_frame = pd.concat(result_frames["decisions"], ignore_index=True)
    periods_frame = pd.concat(result_frames["periods"], ignore_index=True)
    accounting_frame = pd.concat(result_frames["accounting"], ignore_index=True)
    attach_origin_identity(beliefs_frame, bundle.origins)
    attach_origin_identity(decisions_frame, bundle.origins)
    attach_origin_identity(periods_frame, bundle.origins)
    attach_origin_identity(accounting_frame, bundle.origins)

    assert_matched_initial_state(initial_frame)
    assert_complete_periods(periods_frame, bundle.origins)
    assert_accounting(accounting_frame)
    assert_no_prompt_target_leakage(prompt_rows)

    forecasts = build_forecasts(periods_frame, households, bundle)
    joined = join_and_score_forecasts(forecasts, bundle.targets)
    target_scores, family_scores, origin_scores, macro_scores = score_macro(joined)
    bootstrap = origin_block_bootstrap(
        joined,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        block_length=args.bootstrap_block_length,
    )
    raw_records = [*llm_client.raw_records, *adaptive_client.raw_records]

    spec = normalized_spec(
        args,
        bundle=bundle,
        gains=gains,
        period_overrides=period_overrides,
        initial_environment_anchor=initial_environment_anchor,
        behavior_profile_content_sha256=behavior_profile_content_sha256,
        household_provenance=household_provenance,
        household_flow_anchor=household_flow_anchor,
        contamination_coverage=contamination_coverage,
    )
    spec_sha256 = _sha256_json(spec)
    prompt_frame = prompt_rows_to_frame(prompt_rows)
    write_outputs(
        output_dir,
        spec=spec,
        households=households,
        prompts=prompt_frame,
        beliefs=beliefs_frame,
        decisions=decisions_frame,
        periods=periods_frame,
        accounting=accounting_frame,
        forecasts=forecasts,
        joined=joined,
        target_scores=target_scores,
        family_scores=family_scores,
        origin_scores=origin_scores,
        raw_records=raw_records,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "bundle_sha256": bundle.bundle_sha256,
        "normalized_spec_sha256": spec_sha256,
        "periods_per_year": PERIODS_PER_YEAR,
        "origin_count": len(bundle.origins),
        "target_count": len(bundle.target_specs),
        "scored_target_row_count": len(bundle.targets),
        "contamination_coverage": contamination_coverage,
        "provider_execution_isolation": provider_execution_isolation(args.mode),
        "initial_environment_anchor": initial_environment_anchor,
        "initial_environment_anchor_origin": bundle.origins[0],
        "household_provenance": household_provenance,
        "household_flow_anchor": household_flow_anchor,
        "behavior_policy_content_sha256": behavior_profile_content_sha256,
        "household_count": int(len(households)),
        "live_call_count": llm_client.live_call_count,
        "cache_hit_count": llm_client.cache_hit_count,
        "max_accounting_abs_residual": _max_accounting_residual(accounting_frame),
        "macro_scores": macro_scores,
        "llm_minus_adaptive": macro_scores["llm"] - macro_scores["adaptive"],
        "origin_block_bootstrap": bootstrap,
        "adaptive_role": "diagnostic_only_not_a_selection_veto",
        "output_contract": list(OUTPUT_FILES),
        "outputs": output_hashes(output_dir, exclude={"manifest.json", "report.md"}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(manifest, target_scores, family_scores)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest["outputs"] = output_hashes(output_dir, exclude={"manifest.json"})
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class GainAdjustedDemandClient:
    """Identity-checking client that applies gains to belief update deltas."""

    variant = "llm_belief"

    def __init__(
        self,
        base: DemandEconomyClient,
        *,
        gains: Mapping[str, float],
        requested_mode: str,
        replay_records: list[dict[str, Any]] | None,
    ) -> None:
        self.base = base
        self.gains = dict(gains)
        self.requested_mode = requested_mode
        self._replay_by_key = {
            (
                str(row.get("provider")),
                str(row.get("model")),
                str(row.get("variant", "llm_belief")),
                str(row.get("scenario_id")),
                int(row["period_index"]),
            ): row
            for row in (replay_records or [])
        }

    @property
    def source(self) -> str:
        return self.base.source

    @property
    def live_call_count(self) -> int:
        return self.base.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self.base.cache_hit_count

    @property
    def raw_records(self) -> list[dict[str, Any]]:
        return self.base.raw_records

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = belief_module_prompt_payload(
            scenario, period_state, household_states, variant="llm_belief"
        )
        identity = cache_identity(
            provider=self.base.provider,
            model=self.base.model,
            candidate="llm_belief",
            scenario_id=scenario.scenario_id,
            period_index=int(period_state["period_index"]),
            prompt_payload=prompt,
        )
        if self._replay_by_key:
            key = (
                self.base.provider,
                self.base.model,
                "llm_belief",
                scenario.scenario_id,
                int(period_state["period_index"]),
            )
            record = self._replay_by_key.get(key)
            if record is None or record.get("cache_identity") != identity:
                raise DynamicMacroError(
                    "Replay identity mismatch for "
                    f"provider={key[0]}, model={key[1]}, candidate={key[2]}, scenario={key[3]}, period={key[4]}"
                )
        panel = self.base.belief_panel(scenario, period_state, household_states)
        latest_record: dict[str, Any] = self.base.raw_records[-1]
        self._validate_cache_record(latest_record, identity, prompt)
        latest_record["cache_identity"] = identity
        latest_record["state_identity_sha256"] = identity["state_identity_sha256"]
        latest_record["candidate"] = "llm_belief"
        return apply_belief_gains(panel, household_states, self.gains)

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.belief_panel(scenario, period_state, household_states)

    def _validate_cache_record(
        self,
        record: dict[str, Any],
        identity: dict[str, Any],
        prompt_payload: dict[str, Any],
    ) -> None:
        if self.requested_mode not in {"replay", "live"}:
            return
        if (
            str(record.get("provider")) != self.base.provider
            or str(record.get("model")) != self.base.model
        ):
            raise DynamicMacroError(
                "Replay identity mismatch: cached provider/model does not match the requested candidate"
            )
        if self.requested_mode == "replay" and not self._replay_by_key:
            expected_name = f"demand_belief_{cache_key({'provider': self.base.provider, 'model': self.base.model, 'prompt': prompt_payload})}"
            cache_path = Path(str(record.get("cache_path", "")))
            if cache_path.stem != expected_name:
                raise DynamicMacroError(
                    "Replay identity mismatch: cache path does not match provider/model/candidate/state identity"
                )
        if (
            identity["provider"] != self.base.provider
            or identity["model"] != self.base.model
        ):
            raise DynamicMacroError("Replay identity mismatch")


class ObservedSignalAdaptiveClient:
    """Zero-call adaptive twin using the same as-of history as the LLM."""

    variant = "adaptive"
    source = "adaptive_observed_signals"

    def __init__(self) -> None:
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return 0

    @property
    def cache_hit_count(self) -> int:
        return 0

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = observed_signal_adaptive_payload(
            scenario,
            period_state,
            household_states,
        )
        normalized = normalize_belief_payload(
            household_states,
            {
                "provider": "deterministic_observed_signal_adaptive",
                "model": "adaptive_observed_signals_v1",
                "payload": payload,
            },
        )
        self.raw_records.append(
            {
                "source": self.source,
                "variant": self.variant,
                "candidate": "adaptive",
                "scenario_id": scenario.scenario_id,
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "provider": "deterministic_observed_signal_adaptive",
                "model": "adaptive_observed_signals_v1",
                "cache_hit": True,
                "cache_path": None,
                "observed_signal_summary": period_state.get("observed_signal_summary"),
                "payload": payload,
            }
        )
        return normalized

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.belief_panel(scenario, period_state, household_states)


def observed_signal_adaptive_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = period_state.get("observed_signal_summary")
    if not isinstance(summary, dict):
        raise DynamicMacroError(
            "Adaptive twin requires an observed_signal_summary for every origin"
        )
    derived = summary.get("derived")
    if not isinstance(derived, dict):
        raise DynamicMacroError(
            "Observed signal summary is missing derived as-of signals"
        )
    baseline = adaptive_belief_payload(scenario, period_state, household_states)
    states = {str(row["type_id"]): row for row in household_states}
    observed_inflation = _optional_signal(
        derived.get("inflation_annualized_pct"),
        float(period_state["inflation_rate"]),
    )
    observed_income = _optional_signal(
        derived.get("real_income_growth_annualized_pct"),
        float(period_state["output_gap_pct"]),
    )
    unemployment_change = _optional_signal(
        derived.get("unemployment_rate_change_pp"),
        0.0,
    )
    payroll_growth = _optional_signal(derived.get("payroll_growth_pct"), 0.0)
    sentiment_change = _optional_signal(derived.get("sentiment_change"), 0.0)
    output_gap = float(period_state["output_gap_pct"])
    endogenous_inflation = float(period_state["inflation_rate"])
    observed_policy_rate = _optional_signal(
        derived.get("policy_rate_pct"),
        float(period_state["policy_rate"]),
    )
    policy_gap = observed_policy_rate - 2.5
    for belief in baseline["beliefs"]:
        state = states[str(belief["type_id"])]
        prior_inflation = float(state["inflation_expectation_1y"])
        prior_income = float(state["income_growth_expectation_1y"])
        prior_job_loss = float(
            state.get("job_loss_probability", state["baseline_job_loss_probability"])
        )
        expected_inflation = (
            0.50 * prior_inflation
            + 0.30 * observed_inflation
            + 0.20 * endogenous_inflation
        )
        expected_income = (
            0.60 * prior_income + 0.25 * observed_income + 0.15 * output_gap
        )
        job_loss = prior_job_loss
        job_loss += 0.70 * max(0.0, unemployment_change)
        job_loss += 0.18 * max(0.0, -payroll_growth)
        job_loss += 0.10 * max(0.0, -output_gap) + 0.05 * max(0.0, policy_gap)
        unemployment_higher = float(
            np.clip(
                float(
                    state.get(
                        "unemployment_higher_probability_1y", prior_job_loss / 0.24
                    )
                )
                + 2.5 * unemployment_change
                + 0.35 * max(0.0, -payroll_growth),
                0.0,
                100.0,
            )
        )
        confidence = float(state["confidence_index"])
        confidence += 0.45 * sentiment_change + 0.20 * output_gap
        confidence -= 1.8 * max(0.0, unemployment_change) + 0.8 * max(
            0.0, observed_inflation - prior_inflation
        )
        precaution = 4.0 + 0.20 * job_loss - 0.025 * confidence
        belief.update(
            {
                "expected_inflation_next_period": float(
                    np.clip(expected_inflation, -5.0, 15.0)
                ),
                "expected_income_growth_next_period": float(
                    np.clip(expected_income, -12.0, 12.0)
                ),
                "perceived_job_loss_probability": float(np.clip(job_loss, 0.0, 40.0)),
                "expected_unemployment_higher_probability_next_period": unemployment_higher,
                "confidence_index": float(np.clip(confidence, 0.0, 100.0)),
                "precautionary_saving_score": float(np.clip(precaution, 0.0, 10.0)),
                "reason_codes": [
                    "adaptive_observed_as_of_history",
                    "adaptive_endogenous_feedback",
                ],
                "causal_path": [
                    "origin-visible history signals",
                    "adaptive update around recursive prior",
                    "beliefs supplied to shared demand kernel",
                ],
            }
        )
    return baseline


def cache_identity(
    *,
    provider: str,
    model: str,
    candidate: str,
    scenario_id: str,
    period_index: int,
    prompt_payload: dict[str, Any],
) -> dict[str, Any]:
    state_sha = _sha256_json(prompt_payload)
    payload = {
        "provider": provider,
        "model": model,
        "candidate": candidate,
        "scenario_id": scenario_id,
        "period_index": int(period_index),
        "state_identity_sha256": state_sha,
    }
    return {**payload, "cache_identity_sha256": _sha256_json(payload)}


def apply_belief_gains(
    panel: dict[str, Any],
    household_states: list[dict[str, Any]],
    gains: Mapping[str, float],
) -> dict[str, Any]:
    out = {
        "prompt_version": panel.get("prompt_version", DEMAND_ECONOMY_PROMPT_VERSION),
        "beliefs_by_type": {},
        "direct_actions_by_type": dict(panel.get("direct_actions_by_type", {})),
    }
    states = {str(row["type_id"]): row for row in household_states}
    global_gain = float(gains["global"])
    fields = (
        ("expected_inflation_next_period", "inflation_expectation_1y", "inflation"),
        (
            "expected_income_growth_next_period",
            "income_growth_expectation_1y",
            "income",
        ),
        ("perceived_job_loss_probability", "job_loss_probability", "unemployment"),
        (
            "expected_unemployment_higher_probability_next_period",
            "unemployment_higher_probability_1y",
            "unemployment",
        ),
        ("confidence_index", "confidence_index", "confidence"),
    )
    for type_id, raw_belief in panel["beliefs_by_type"].items():
        state = states[str(type_id)]
        belief = dict(raw_belief)
        for belief_field, state_field, gain_name in fields:
            if belief_field not in belief:
                continue
            prior = float(state.get(state_field, belief[belief_field]))
            candidate = float(belief[belief_field])
            belief[belief_field] = prior + global_gain * float(gains[gain_name]) * (
                candidate - prior
            )
        reason_codes = list(belief.get("reason_codes", []))
        reason_codes.append(
            "belief_gain_"
            f"g{global_gain:.3f}_pi{float(gains['inflation']):.3f}_inc{float(gains['income']):.3f}_"
            f"unemp{float(gains['unemployment']):.3f}"
        )
        belief["reason_codes"] = reason_codes
        out["beliefs_by_type"][str(type_id)] = belief
    return out


def belief_gains_from_args(args: argparse.Namespace) -> dict[str, float]:
    gains = {
        "global": float(args.belief_gain_global),
        "inflation": float(args.belief_gain_inflation),
        "income": float(args.belief_gain_income),
        "unemployment": float(args.belief_gain_unemployment),
        "confidence": 1.0,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in gains.values()):
        raise DynamicMacroError("Belief gains must be finite and non-negative")
    return gains


def load_bundle_view(bundle_dir: Path) -> BundleView:
    loaded = load_frozen_vintage_bundle(bundle_dir)
    manifest = dict(loaded.manifest)
    origins = tuple(_normalize_origin(row) for row in loaded.origins)
    assert_contiguous_origins(origins)
    history = tuple(_normalize_history(row) for row in loaded.history)
    raw_specs = manifest.get("target_specs") or []
    specs_by_name = {
        str(row["target_name"]): dict(row) for row in raw_specs if isinstance(row, dict)
    }
    targets = tuple(_normalize_target(row, specs_by_name) for row in loaded.targets)
    target_specs = tuple(_target_specs_from_rows(targets))
    target_contamination = tuple(
        _normalize_contamination(row) for row in loaded.target_contamination
    )
    families = {str(row["family"]) for row in target_specs}
    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        raise DynamicMacroError(
            f"Frozen bundle does not cover broad target families: {', '.join(sorted(missing_families))}"
        )
    unsupported = sorted(
        {str(row["series_id"]) for row in target_specs} - set(MAPPING_BY_SERIES)
    )
    if unsupported:
        raise DynamicMacroError(
            f"Incomplete target coverage; no economy mapping for: {', '.join(unsupported)}"
        )
    expected = {
        (row["origin_month"], spec["target_name"])
        for row in origins
        for spec in target_specs
    }
    actual = {(row["origin_month"], row["target_name"]) for row in targets}
    if len(targets) != len(actual) or actual != expected:
        raise DynamicMacroError("Incomplete target coverage across frozen origins")
    history_origins = {row["origin_month"] for row in history}
    missing_history = [
        row["origin_month"]
        for row in origins
        if row["origin_month"] not in history_origins
    ]
    if missing_history:
        raise DynamicMacroError(
            f"Missing periods in origin-visible history: {', '.join(missing_history)}"
        )
    bundle_sha = str(manifest.get("bundle_sha256", ""))
    if not bundle_sha:
        raise DynamicMacroError("Validated bundle manifest has no bundle_sha256")
    return BundleView(
        bundle_sha,
        origins,
        history,
        targets,
        target_specs,
        target_contamination,
    )


def filter_bundle_targets(
    bundle: BundleView,
    *,
    model: str,
    policy: str,
) -> tuple[BundleView, dict[str, Any]]:
    if policy not in CONTAMINATION_POLICIES:
        raise DynamicMacroError(f"Unsupported contamination policy: {policy}")
    model_rows = [row for row in bundle.target_contamination if row["model"] == model]
    expected_keys = {
        (row["origin_month"], row["target_name"]) for row in bundle.targets
    }
    actual_keys = {(row["origin_month"], row["target_name"]) for row in model_rows}
    if len(model_rows) != len(actual_keys) or actual_keys != expected_keys:
        raise DynamicMacroError(
            f"Target-contamination coverage is incomplete for requested model {model}"
        )
    contamination_by_key = {
        (row["origin_month"], row["target_name"]): row for row in model_rows
    }
    labels_by_key = {
        key: row["contamination_label"] for key, row in contamination_by_key.items()
    }
    unavailable_labels = {
        "post_cutoff_holdout",
        "pre_cutoff_observation_post_cutoff_release",
    }
    release_after_cutoff_by_key = {
        key: date.fromisoformat(row["first_release_as_of_date"])
        > date.fromisoformat(row["model_cutoff_date"])
        for key, row in contamination_by_key.items()
    }
    for key, release_after_cutoff in release_after_cutoff_by_key.items():
        label_marks_unavailable = labels_by_key[key] in unavailable_labels
        if release_after_cutoff != label_marks_unavailable:
            raise DynamicMacroError(
                "Target-contamination label conflicts with first-release availability"
            )

    def selected_by_policy(key: tuple[str, str]) -> bool:
        if policy == "all":
            return True
        if policy == "unavailable_at_cutoff":
            return release_after_cutoff_by_key[key]
        return labels_by_key[key] == "post_cutoff_holdout"

    selected = tuple(
        row
        for row in bundle.targets
        if selected_by_policy((row["origin_month"], row["target_name"]))
    )
    if not selected:
        raise DynamicMacroError(
            f"Contamination policy {policy} selects no frozen targets for model {model}"
        )
    selected_families = {row["family"] for row in selected}
    missing_families = REQUIRED_FAMILIES - selected_families
    if missing_families:
        raise DynamicMacroError(
            "Contamination-filtered targets do not cover broad families: "
            + ", ".join(sorted(missing_families))
        )
    label_counts = pd.Series(
        list(labels_by_key.values()), dtype="object"
    ).value_counts()
    selected_key_set = {(row["origin_month"], row["target_name"]) for row in selected}
    selected_label_counts = pd.Series(
        [label for key, label in labels_by_key.items() if key in selected_key_set],
        dtype="object",
    ).value_counts()
    excluded_label_counts = pd.Series(
        [label for key, label in labels_by_key.items() if key not in selected_key_set],
        dtype="object",
    ).value_counts()
    selected_keys = sorted(
        (row["origin_month"], row["target_name"]) for row in selected
    )
    selected_counts_by_origin = pd.Series(
        [row["origin_month"] for row in selected], dtype="object"
    ).value_counts()
    targets_per_complete_origin = len(bundle.target_specs)
    complete_origins = sorted(
        str(origin)
        for origin, count in selected_counts_by_origin.items()
        if int(count) == targets_per_complete_origin
    )
    partial_origins = sorted(
        str(origin)
        for origin, count in selected_counts_by_origin.items()
        if 0 < int(count) < targets_per_complete_origin
    )
    empty_origins = sorted(
        origin["origin_month"]
        for origin in bundle.origins
        if origin["origin_month"]
        not in set(selected_counts_by_origin.index.astype(str))
    )
    coverage = {
        "model": model,
        "policy": policy,
        "available_label_counts": {
            str(label): int(count) for label, count in label_counts.sort_index().items()
        },
        "selected_label_counts": {
            str(label): int(count)
            for label, count in selected_label_counts.sort_index().items()
        },
        "excluded_label_counts": {
            str(label): int(count)
            for label, count in excluded_label_counts.sort_index().items()
        },
        "catalogue_rows": len(bundle.targets),
        "first_release_after_cutoff_rows": int(
            sum(release_after_cutoff_by_key.values())
        ),
        "selected_rows": len(selected),
        "excluded_rows": len(bundle.targets) - len(selected),
        "selected_origin_count": len({row["origin_month"] for row in selected}),
        "selected_target_count": len({row["target_name"] for row in selected}),
        "selected_family_count": len(selected_families),
        "targets_per_complete_origin": targets_per_complete_origin,
        "complete_origin_count": len(complete_origins),
        "partial_origin_count": len(partial_origins),
        "empty_origin_count": len(empty_origins),
        "complete_origins": complete_origins,
        "partial_origins": partial_origins,
        "empty_origins": empty_origins,
        "selected_pairs_sha256": _sha256_json(selected_keys),
    }
    return (
        BundleView(
            bundle.bundle_sha256,
            bundle.origins,
            bundle.history,
            selected,
            bundle.target_specs,
            bundle.target_contamination,
        ),
        coverage,
    )


def _normalize_origin(row: Mapping[str, Any]) -> dict[str, str]:
    origin = str(row.get("origin_month") or row.get("origin_date") or "")
    as_of = str(row.get("as_of_date") or row.get("asof_date") or origin)
    if not origin or not as_of:
        raise DynamicMacroError("Bundle origin row is missing origin_month/as_of_date")
    return {"origin_month": origin, "as_of_date": as_of}


def _normalize_history(row: Mapping[str, Any]) -> dict[str, str]:
    origin = _normalize_origin(row)
    normalized: dict[str, Any] = {
        **origin,
        "series_id": str(row.get("series_id", "")),
        "observation_date": str(row.get("observation_date", "")),
        "value": str(row.get("value", "")),
    }
    try:
        value = float(normalized["value"])
    except ValueError as exc:
        raise DynamicMacroError(
            "Origin-visible history has a non-numeric value"
        ) from exc
    if (
        not normalized["series_id"]
        or not normalized["observation_date"]
        or not math.isfinite(value)
    ):
        raise DynamicMacroError("Origin-visible history row is incomplete")
    if date.fromisoformat(normalized["observation_date"]) > date.fromisoformat(
        origin["as_of_date"]
    ):
        raise DynamicMacroError(
            "History contains an observation beyond its origin as-of date"
        )
    return normalized


def _normalize_target(
    row: Mapping[str, Any], specs_by_name: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    origin = _normalize_origin(row)
    name = str(row.get("target_name", ""))
    spec = specs_by_name.get(name, {})
    normalized: dict[str, Any] = {
        **origin,
        "target_name": name,
        "series_id": str(row.get("series_id") or spec.get("series_id") or ""),
        "family": str(row.get("family") or spec.get("family") or ""),
        "transform": str(row.get("transform") or spec.get("transform") or ""),
        "default_scale": float(
            row.get("default_scale") or spec.get("default_scale") or np.nan
        ),
        "target_value": float(row.get("target_value", np.nan)),
        "origin_visible_denominator_value": float(
            row.get("origin_visible_denominator_value", np.nan)
        ),
        "target_observation_date": str(
            row.get("target_observation_date") or row.get("target_month") or ""
        ),
    }
    if (
        not normalized["target_name"]
        or not normalized["series_id"]
        or not normalized["family"]
        or normalized["transform"] not in {"pct_change", "diff", "level"}
        or not math.isfinite(normalized["default_scale"])
        or normalized["default_scale"] <= 0.0
        or not math.isfinite(normalized["target_value"])
    ):
        raise DynamicMacroError(
            f"Frozen target row is incomplete or invalid: {name or '<unnamed>'}"
        )
    return normalized


def _normalize_contamination(row: Mapping[str, Any]) -> dict[str, str]:
    origin = _normalize_origin(row)
    normalized = {
        **origin,
        "target_name": str(row.get("target_name", "")),
        "model": str(row.get("model", "")),
        "model_cutoff_date": str(row.get("model_cutoff_date", "")),
        "target_observation_date": str(row.get("target_observation_date", "")),
        "first_release_as_of_date": str(row.get("first_release_as_of_date", "")),
        "contamination_label": str(row.get("contamination_label", "")),
    }
    if (
        not normalized["target_name"]
        or not normalized["model"]
        or normalized["contamination_label"] not in CONTAMINATION_LABELS
    ):
        raise DynamicMacroError("Frozen target-contamination row is incomplete")
    try:
        for field in (
            "model_cutoff_date",
            "target_observation_date",
            "first_release_as_of_date",
        ):
            date.fromisoformat(normalized[field])
    except ValueError as exc:
        raise DynamicMacroError(
            "Frozen target-contamination row has an invalid availability date"
        ) from exc
    return normalized


def _target_specs_from_rows(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in targets:
        spec = {
            key: row[key]
            for key in (
                "target_name",
                "series_id",
                "family",
                "transform",
                "default_scale",
            )
        }
        previous = by_name.setdefault(str(row["target_name"]), spec)
        if previous != spec:
            raise DynamicMacroError(
                f"Target metadata changes across origins: {row['target_name']}"
            )
    return [by_name[name] for name in sorted(by_name)]


def assert_contiguous_origins(origins: tuple[dict[str, str], ...]) -> None:
    if not origins:
        raise DynamicMacroError("Frozen bundle has no origins")
    values = [date.fromisoformat(row["origin_month"]) for row in origins]
    if values != sorted(set(values)):
        raise DynamicMacroError("Frozen origins must be unique and sorted")
    for previous, current in zip(values, values[1:]):
        expected = date(
            previous.year + (previous.month == 12),
            1 if previous.month == 12 else previous.month + 1,
            1,
        )
        if current != expected:
            raise DynamicMacroError(
                f"Missing periods between frozen origins {previous} and {current}"
            )


def build_period_overrides(bundle: BundleView) -> dict[int, dict[str, Any]]:
    rows_by_origin: dict[str, list[dict[str, str]]] = {}
    for row in bundle.history:
        rows_by_origin.setdefault(row["origin_month"], []).append(row)
    overrides: dict[int, dict[str, Any]] = {}
    for period_index, origin in enumerate(bundle.origins):
        raw_by_series: dict[str, list[dict[str, Any]]] = {}
        latest: dict[str, dict[str, Any]] = {}
        origin_rows = sorted(
            rows_by_origin[origin["origin_month"]],
            key=lambda item: (item["series_id"], item["observation_date"]),
        )
        for row in origin_rows:
            raw_by_series.setdefault(row["series_id"], []).append(
                {
                    "observation_date": row["observation_date"],
                    "value": float(row["value"]),
                }
            )
            latest[row["series_id"]] = {
                "observation_date": row["observation_date"],
                "value": float(row["value"]),
            }
        if not latest:
            raise DynamicMacroError(
                f"Missing periods for origin {origin['origin_month']}"
            )
        raw_history = {key: raw_by_series[key] for key in sorted(raw_by_series)}
        overrides[period_index] = {
            "origin_month": origin["origin_month"],
            "as_of_date": origin["as_of_date"],
            "origin_visible_macro_context": {
                key: latest[key] for key in sorted(latest)
            },
            "origin_visible_macro_history": raw_history,
            "observed_signal_summary": observed_signal_summary(raw_history),
        }
    assert_no_prompt_target_leakage(
        [{"prompt_payload": value} for value in overrides.values()]
    )
    return overrides


def anchor_household_flows(
    households: pd.DataFrame,
    period_overrides: Mapping[int, dict[str, Any]],
    *,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mode not in {"origin_saving_rate", "none"}:
        raise DynamicMacroError(f"Unsupported household flow anchor: {mode}")
    if not period_overrides or min(period_overrides) != 0:
        raise DynamicMacroError("Household flow anchor requires first-origin period zero")
    out = households.copy()
    weights = out["population_weight"].astype(float)
    income = float((weights * out["annual_income"].astype(float)).sum())
    consumption = float((weights * out["baseline_consumption_annual"].astype(float)).sum())
    if income <= 0.0 or consumption <= 0.0:
        raise DynamicMacroError("Household flow anchor requires positive aggregate income and consumption")
    pre_saving_rate = 100.0 * (income - consumption) / income
    if mode == "none":
        return out, {
            "mode": mode,
            "origin_visible_saving_rate_pct": None,
            "pre_anchor_saving_rate_pct": pre_saving_rate,
            "post_anchor_saving_rate_pct": pre_saving_rate,
            "consumption_scale": 1.0,
        }

    summary = period_overrides[0].get("observed_signal_summary")
    series = summary.get("series") if isinstance(summary, dict) else None
    psavert = series.get("PSAVERT") if isinstance(series, dict) else None
    target_rate = _optional_signal(psavert.get("latest_value") if isinstance(psavert, dict) else None, np.nan)
    if not math.isfinite(target_rate) or not -10.0 <= target_rate <= 50.0:
        raise DynamicMacroError("First origin lacks a plausible origin-visible PSAVERT level")
    target_consumption = income * (1.0 - target_rate / 100.0)
    if target_consumption <= 0.0:
        raise DynamicMacroError("Origin-visible saving rate implies non-positive aggregate consumption")
    scale = target_consumption / consumption
    out["baseline_consumption_annual"] = out["baseline_consumption_annual"].astype(float) * scale
    out["liquid_assets"] = out["liquid_assets"].astype(float) * scale
    out["base_saving_rate"] = (
        1.0 - out["baseline_consumption_annual"].astype(float) / out["annual_income"].astype(float)
    ).clip(lower=-0.10, upper=0.55)
    anchored_consumption = float((weights * out["baseline_consumption_annual"].astype(float)).sum())
    post_saving_rate = 100.0 * (income - anchored_consumption) / income
    if not math.isclose(post_saving_rate, target_rate, rel_tol=0.0, abs_tol=1e-9):
        raise DynamicMacroError("Household flow anchor failed to reproduce the origin-visible saving rate")
    return out, {
        "mode": mode,
        "origin_visible_saving_rate_pct": target_rate,
        "pre_anchor_saving_rate_pct": pre_saving_rate,
        "post_anchor_saving_rate_pct": post_saving_rate,
        "consumption_scale": scale,
        "liquid_buffer_preservation": "liquid assets scaled with baseline consumption",
    }


def build_initial_environment_anchor(
    period_overrides: Mapping[int, dict[str, Any]],
) -> dict[str, float]:
    if not period_overrides or min(period_overrides) != 0:
        raise DynamicMacroError(
            "Initial environment anchor requires first-origin period zero"
        )
    summary = period_overrides[0].get("observed_signal_summary")
    derived = summary.get("derived") if isinstance(summary, dict) else None
    if not isinstance(derived, dict):
        raise DynamicMacroError(
            "First origin has no observed signal summary for anchoring"
        )

    def required(field: str) -> float:
        value = _optional_signal(derived.get(field), np.nan)
        if not math.isfinite(value):
            raise DynamicMacroError(
                f"First-origin observed signal summary is missing anchor field {field}"
            )
        return value

    unemployment_rate = required("unemployment_rate_pct")
    employment_rate = 1.0 - unemployment_rate / 100.0
    if not 0.0 <= employment_rate <= 1.0:
        raise DynamicMacroError(
            "First-origin unemployment signal cannot produce a valid employment rate"
        )
    return {
        "employment_rate": employment_rate,
        "inflation_rate": required("inflation_annualized_pct"),
        "policy_rate": required("policy_rate_pct"),
        "output_gap_pct": 0.0,
    }


def observed_signal_summary(
    history_by_series: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    series: dict[str, dict[str, Any]] = {}
    for series_id, observations in sorted(history_by_series.items()):
        if not observations:
            continue
        latest = observations[-1]
        previous = observations[-2] if len(observations) >= 2 else None
        latest_value = float(latest["value"])
        previous_value = float(previous["value"]) if previous is not None else None
        series[series_id] = {
            "latest_observation_date": str(latest["observation_date"]),
            "latest_value": latest_value,
            "previous_observation_date": (
                str(previous["observation_date"]) if previous is not None else None
            ),
            "previous_value": previous_value,
            "change": (
                latest_value - previous_value if previous_value is not None else None
            ),
            "pct_change": (
                100.0 * (latest_value / previous_value - 1.0)
                if previous_value is not None and abs(previous_value) > 1e-12
                else None
            ),
        }

    def value(series_id: str, field: str) -> float | None:
        raw = series.get(series_id, {}).get(field)
        return float(raw) if raw is not None and math.isfinite(float(raw)) else None

    inflation_monthly = value("PCEPI", "pct_change")
    if inflation_monthly is None:
        inflation_monthly = value("CPIAUCSL", "pct_change")
    income_monthly = value("DSPIC96", "pct_change")
    derived = {
        "inflation_annualized_pct": (
            _monthly_rate_to_annual_pct(inflation_monthly)
            if inflation_monthly is not None
            else None
        ),
        "unemployment_rate_pct": value("UNRATE", "latest_value"),
        "unemployment_rate_change_pp": value("UNRATE", "change"),
        "policy_rate_pct": value("FEDFUNDS", "latest_value"),
        "payroll_growth_pct": value("PAYEMS", "pct_change"),
        "real_income_growth_annualized_pct": (
            _monthly_rate_to_annual_pct(income_monthly)
            if income_monthly is not None
            else None
        ),
        "real_demand_growth_pct": value("PCEC96", "pct_change"),
        "sentiment_level": value("UMCSENT", "latest_value"),
        "sentiment_change": value("UMCSENT", "change"),
    }
    return {
        "derivation": "latest two origin-visible observations only",
        "series": series,
        "derived": derived,
    }


def load_households(args: argparse.Namespace) -> pd.DataFrame:
    if args.households_csv:
        return normalize_demand_households(pd.read_csv(Path(args.households_csv)))
    panel = pd.read_csv(Path(args.household_panel))
    required_demand = {
        "type_id",
        "annual_income",
        "baseline_consumption_annual",
        "liquid_assets",
        "debt",
    }
    if required_demand.issubset(panel.columns):
        return normalize_demand_households(panel)
    return prepare_sce_households(panel)


def household_input_provenance(
    args: argparse.Namespace,
    households: pd.DataFrame,
    *,
    temporal_coverage: dict[str, Any],
) -> dict[str, Any]:
    source_kind = "households_csv" if args.households_csv else "household_panel"
    source_path = Path(args.households_csv or args.household_panel)
    if not source_path.is_file():
        raise DynamicMacroError(f"Household input file does not exist: {source_path}")
    return {
        "source_kind": source_kind,
        "raw_input_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "normalized_household_state_sha256": canonical_household_state_sha256(
            households
        ),
        "normalized_household_count": int(len(households)),
        "temporal_coverage": temporal_coverage,
    }


def validate_household_temporal_availability(
    args: argparse.Namespace,
    households: pd.DataFrame,
    *,
    first_origin: Mapping[str, str],
) -> dict[str, Any]:
    availability_column = "source_estimated_public_availability_date"
    event_candidates = ("source_survey_event_date", "survey_event_date", "survey_date")
    first_as_of = pd.Timestamp(first_origin["as_of_date"]).normalize()
    if availability_column not in households:
        if args.mode in {"live", "replay"}:
            raise DynamicMacroError(
                f"Real household input requires {availability_column} in {args.mode} mode"
            )
        return {
            "status": "fixture_omitted",
            "first_macro_origin_month": first_origin["origin_month"],
            "first_macro_as_of_date": first_origin["as_of_date"],
            "source_event_date_min": None,
            "source_event_date_max": None,
            "source_availability_date_min": None,
            "source_availability_date_max": None,
        }

    availability = pd.to_datetime(
        households[availability_column],
        errors="coerce",
    ).dt.normalize()
    if availability.isna().any():
        raise DynamicMacroError(
            f"Household input has invalid or missing {availability_column}"
        )
    latest_availability = availability.max()
    if latest_availability > first_as_of:
        raise DynamicMacroError(
            "Household data was not publicly available by the first macro origin: "
            f"latest availability {latest_availability.date()} exceeds {first_as_of.date()}"
        )
    event_column = next(
        (column for column in event_candidates if column in households),
        None,
    )
    event_dates = None
    if event_column is not None:
        event_dates = pd.to_datetime(
            households[event_column], errors="coerce"
        ).dt.normalize()
        if event_dates.isna().any():
            raise DynamicMacroError(
                f"Household input has invalid or missing {event_column}"
            )
    return {
        "status": "available_by_first_macro_origin",
        "first_macro_origin_month": first_origin["origin_month"],
        "first_macro_as_of_date": first_origin["as_of_date"],
        "source_event_date_column": event_column,
        "source_event_date_min": (
            event_dates.min().date().isoformat() if event_dates is not None else None
        ),
        "source_event_date_max": (
            event_dates.max().date().isoformat() if event_dates is not None else None
        ),
        "source_availability_date_column": availability_column,
        "source_availability_date_min": availability.min().date().isoformat(),
        "source_availability_date_max": latest_availability.date().isoformat(),
    }


def canonical_household_state_sha256(households: pd.DataFrame) -> str:
    ordered = households.sort_values("type_id").reset_index(drop=True)
    ordered = ordered.reindex(sorted(ordered.columns), axis="columns")
    payload = json.loads(
        ordered.to_json(
            orient="records",
            double_precision=15,
            date_format="iso",
        )
    )
    return _sha256_json(payload)


def canonical_behavior_profile_sha256(
    profile: dict[str, Any] | None,
) -> str | None:
    if profile is None:
        return None
    path_keys = {
        "path",
        "profile_json",
        "policy_profile_json",
        "profile_path",
        "source_path",
        "raw_records_path",
        "raw_records_json",
        "empirical_bridge_json",
        "empirical_bridge_path",
        "state_schedule_json",
        "state_profile_path",
        "cache_path",
    }

    def strip_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): strip_paths(nested)
                for key, nested in value.items()
                if str(key) not in path_keys and not str(key).endswith("_path")
            }
        if isinstance(value, list):
            return [strip_paths(item) for item in value]
        if isinstance(value, tuple):
            return [strip_paths(item) for item in value]
        return value

    return _sha256_json(strip_paths(profile))


def prepare_sce_households(panel: pd.DataFrame) -> pd.DataFrame:
    id_column = _first_column(panel, "respondent_id", "type_id", "household_id")
    weight_column = _first_column(panel, "weight", "population_weight", "survey_weight")
    if id_column is None or weight_column is None:
        raise DynamicMacroError(
            "Prepared SCE household panel must contain respondent_id and weight columns"
        )
    base = panel.copy()
    if "period_index" in base:
        base["period_index"] = pd.to_numeric(base["period_index"], errors="raise")
        base = base[base["period_index"].eq(base["period_index"].min())]
    base = base.drop_duplicates(id_column)
    event_source_column = _first_column(
        base,
        "source_survey_event_date",
        "survey_event_date",
        "survey_date",
    )
    availability_source_column = _first_column(
        base,
        "source_estimated_public_availability_date",
        "estimated_public_availability_date",
    )
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        income_group = _group(
            row.get("income_group"), ("low", "middle", "high"), "middle"
        )
        liquid_group = _group(
            row.get("liquid_wealth_group", row.get("liquidity_group")),
            ("low", "middle", "high"),
            "middle",
        )
        employment = str(row.get("employment_status", "unknown"))
        annual_income = _finite_or(
            row.get("annual_income"),
            {"low": 38000.0, "middle": 76000.0, "high": 135000.0}[income_group],
        )
        consumption_ratio = {"low": 0.94, "middle": 0.82, "high": 0.68}[income_group]
        baseline_consumption = _finite_or(
            row.get("baseline_consumption_annual"), annual_income * consumption_ratio
        )
        liquidity_months = {
            "low": {"low": 0.6, "middle": 0.9, "high": 1.4}[income_group],
            "middle": {"low": 1.8, "middle": 2.8, "high": 4.2}[income_group],
            "high": {"low": 3.8, "middle": 5.2, "high": 8.5}[income_group],
        }[liquid_group]
        unemp_higher = _first_finite(
            row,
            (
                "prior_expected_unemployment_higher_prob",
                "actual_expected_unemployment_higher_prob",
                "unemployment_higher_probability_1y",
            ),
            35.0,
        )
        job_loss = float(np.clip(0.24 * unemp_higher, 1.0, 24.0))
        job_risk = (
            "high"
            if job_loss >= 9.0 or employment.lower() in {"unemployed", "not_employed"}
            else "low"
        )
        inflation = _first_finite(
            row,
            (
                "prior_expected_inflation_1y",
                "actual_expected_inflation_1y",
                "inflation_expectation_1y",
            ),
            3.0,
        )
        income_growth = _first_finite(
            row,
            (
                "prior_expected_real_income_growth",
                "actual_expected_real_income_growth",
                "income_growth_expectation_1y",
            ),
            1.0,
        )
        low_liquid = liquid_group == "low"
        debt_service = {"low": 0.16, "middle": 0.13, "high": 0.09}[income_group]
        base_mpc = (0.72 if low_liquid else 0.24) + {
            "low": 0.09,
            "middle": 0.0,
            "high": -0.08,
        }[income_group]
        base_mpc += 0.04 if job_risk == "high" else -0.02
        confidence = float(
            np.clip(
                63.0
                - 0.36 * unemp_higher
                + 1.2 * income_growth
                - 0.45 * max(0.0, inflation - 2.0),
                0.0,
                100.0,
            )
        )
        household = {
            "type_id": str(row[id_column]),
            "label": f"SCE respondent {row[id_column]}",
            "population_weight": float(row[weight_column]),
            "age_bucket": (
                "older"
                if str(row.get("age_group", "")).lower()
                in {"55_plus", "55+", "older", "retired"}
                else "prime"
            ),
            "income_group": income_group,
            "liquidity_group": "low" if low_liquid else "high",
            "job_loss_risk_type": job_risk,
            "employment_status": employment,
            "annual_income": annual_income,
            "baseline_consumption_annual": baseline_consumption,
            "liquid_assets": _finite_or(
                row.get("liquid_assets"),
                liquidity_months * baseline_consumption / 12.0,
            ),
            "debt": _finite_or(
                row.get("debt"),
                annual_income * debt_service * (1.3 if low_liquid else 0.8),
            ),
            "debt_service_burden": debt_service,
            "base_mpc": float(np.clip(base_mpc, 0.08, 0.92)),
            "base_saving_rate": float(
                np.clip(
                    0.12
                    + (0.08 if not low_liquid else -0.03)
                    + (0.05 if income_group == "high" else 0.0),
                    0.02,
                    0.35,
                )
            ),
            "rate_sensitivity": 0.39 if low_liquid else 0.62,
            "income_sensitivity": {"low": 0.82, "middle": 0.58, "high": 0.36}[
                income_group
            ],
            "precautionary_sensitivity": 0.34
            + (0.28 if low_liquid else 0.08)
            + (0.16 if job_risk == "high" else 0.0),
            "baseline_job_loss_probability": job_loss,
            "unemployment_higher_probability_1y": unemp_higher,
            "target_buffer_months": 1.6 if low_liquid else 5.2,
            "inflation_expectation_1y": inflation,
            "income_growth_expectation_1y": income_growth,
            "confidence_index": confidence,
            "attention_weight_prices": (
                0.68
                if income_group == "low"
                else 0.56 if income_group == "middle" else 0.46
            ),
            "attention_weight_jobs": 0.75 if job_risk == "high" else 0.48,
            "attention_weight_rates": (
                0.66 if not low_liquid or income_group == "high" else 0.40
            ),
            "income_volatility": 0.10
            + (0.08 if job_risk == "high" else 0.02)
            + (0.04 if income_group == "low" else 0.0),
            "subsistence_floor_share": (
                0.56
                if income_group == "low"
                else 0.48 if income_group == "middle" else 0.38
            ),
        }
        if event_source_column is not None:
            household["source_survey_event_date"] = row[event_source_column]
        if availability_source_column is not None:
            household["source_estimated_public_availability_date"] = row[
                availability_source_column
            ]
        rows.append(household)
    if not rows:
        raise DynamicMacroError("Prepared SCE household panel has no usable households")
    return normalize_demand_households(pd.DataFrame(rows))


def load_behavior_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    mode = args.behavior_policy_mode
    generic = (
        Path(args.behavior_policy_profile) if args.behavior_policy_profile else None
    )
    if mode == "fixed_kernel":
        return None
    if mode == "schedule":
        path = (
            Path(args.behavior_policy_raw_records_json)
            if args.behavior_policy_raw_records_json
            else generic
        )
        if path is None:
            raise DynamicMacroError(
                "schedule mode requires --behavior-policy-raw-records-json or --behavior-policy-profile"
            )
        return load_behavior_policy_profile(path)
    if mode == "state_schedule":
        path = (
            Path(args.behavior_policy_state_profile_json)
            if args.behavior_policy_state_profile_json
            else generic
        )
        if path is None:
            raise DynamicMacroError(
                "state_schedule mode requires --behavior-policy-state-profile-json or --behavior-policy-profile"
            )
        return load_state_behavior_policy_profile(path)
    if mode == "empirical_bridge":
        path = (
            Path(args.empirical_bridge_json) if args.empirical_bridge_json else generic
        )
        if path is None:
            raise DynamicMacroError(
                "empirical_bridge mode requires --empirical-bridge-json or --behavior-policy-profile"
            )
        return load_empirical_bridge_profile(path)
    state_path = (
        Path(args.behavior_policy_state_profile_json)
        if args.behavior_policy_state_profile_json
        else None
    )
    bridge_path = (
        Path(args.empirical_bridge_json) if args.empirical_bridge_json else None
    )
    if state_path is None or bridge_path is None:
        raise DynamicMacroError(
            "hybrid policy mode requires state-profile and empirical-bridge paths"
        )
    return build_hybrid_behavior_policy_profile(
        load_empirical_bridge_profile(bridge_path),
        load_state_behavior_policy_profile(state_path),
        state_weight=float(args.hybrid_state_weight),
    )


def assert_matched_initial_state(initial: pd.DataFrame) -> None:
    ignored = {"candidate", "source", "variant"}
    columns = [column for column in initial.columns if column not in ignored]
    llm = (
        initial[initial["candidate"].eq("llm")][columns]
        .sort_values("type_id")
        .reset_index(drop=True)
    )
    adaptive = (
        initial[initial["candidate"].eq("adaptive")][columns]
        .sort_values("type_id")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            llm, adaptive, check_dtype=False, check_exact=True
        )
    except AssertionError as exc:
        raise DynamicMacroError(
            "Matched twins do not share identical initial state"
        ) from exc


def assert_complete_periods(
    periods: pd.DataFrame, origins: tuple[dict[str, str], ...]
) -> None:
    expected = set(range(len(origins)))
    for candidate, frame in periods.groupby("candidate"):
        actual = set(pd.to_numeric(frame["period_index"], errors="raise").astype(int))
        if actual != expected or len(frame) != len(expected):
            raise DynamicMacroError(
                f"Missing periods for {candidate}: expected 0..{len(origins) - 1}"
            )


def assert_accounting(accounting: pd.DataFrame) -> None:
    maximum = _max_accounting_residual(accounting)
    if not math.isfinite(maximum) or maximum > ACCOUNTING_TOLERANCE:
        raise DynamicMacroError(
            f"Accounting residual {maximum:.12g} exceeds {ACCOUNTING_TOLERANCE:.1e}"
        )


def _max_accounting_residual(accounting: pd.DataFrame) -> float:
    if accounting.empty or "abs_residual" not in accounting:
        return float("inf")
    return float(pd.to_numeric(accounting["abs_residual"], errors="coerce").max())


def assert_no_prompt_target_leakage(prompt_rows: Iterable[dict[str, Any]]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = FORBIDDEN_PROMPT_KEYS.intersection(map(str, value.keys()))
            if leaked:
                raise DynamicMacroError(
                    f"Prompt target leakage detected: {', '.join(sorted(leaked))}"
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for row in prompt_rows:
        walk(row.get("prompt_payload", row))


def attach_origin_identity(
    frame: pd.DataFrame, origins: tuple[dict[str, str], ...]
) -> None:
    if frame.empty:
        return
    by_index = {index: origin for index, origin in enumerate(origins)}
    frame["origin_month"] = (
        frame["period_index"]
        .astype(int)
        .map(lambda index: by_index[index]["origin_month"])
    )
    frame["as_of_date"] = (
        frame["period_index"]
        .astype(int)
        .map(lambda index: by_index[index]["as_of_date"])
    )


def build_forecasts(
    periods: pd.DataFrame, households: pd.DataFrame, bundle: BundleView
) -> pd.DataFrame:
    initial = _initial_macro_row(households)
    rows: list[dict[str, Any]] = []
    targets_by_origin: dict[str, list[dict[str, Any]]] = {}
    for target in bundle.targets:
        targets_by_origin.setdefault(target["origin_month"], []).append(target)
    for candidate, path in periods.groupby("candidate"):
        path = path.sort_values("period_index").reset_index(drop=True)
        for index, origin in enumerate(bundle.origins):
            current = path.iloc[index]
            previous: Mapping[str, Any] = (
                initial if index == 0 else path.iloc[index - 1]
            )
            for target in targets_by_origin[origin["origin_month"]]:
                mapping = MAPPING_BY_SERIES[target["series_id"]]
                prediction = mapped_value(current, previous, mapping)
                rows.append(
                    {
                        "candidate": candidate,
                        "origin_month": origin["origin_month"],
                        "as_of_date": origin["as_of_date"],
                        "target_observation_date": target["target_observation_date"],
                        "target_name": target["target_name"],
                        "series_id": target["series_id"],
                        "family": target["family"],
                        "prediction": prediction,
                        "economy_measure": mapping.economy_measure,
                        "economy_transform": mapping.economy_transform,
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(bundle.targets) * 2
    if (
        len(frame) != expected
        or frame[["candidate", "origin_month", "target_name"]].duplicated().any()
    ):
        raise DynamicMacroError("Incomplete target coverage in economy forecasts")
    return frame.sort_values(
        ["candidate", "origin_month", "family", "target_name"]
    ).reset_index(drop=True)


def mapped_value(
    current: Mapping[str, Any], previous: Mapping[str, Any], mapping: TargetMapping
) -> float:
    current_measure = economy_measure(current, mapping.economy_measure)
    previous_measure = economy_measure(previous, mapping.economy_measure)
    if mapping.economy_transform == "pct_change":
        return _pct_change(current_measure, previous_measure)
    if mapping.economy_transform == "nominal_pct_change":
        real_growth = _pct_change(current_measure, previous_measure)
        monthly_price = _annual_rate_to_monthly_pct(
            float(current["next_inflation_rate"])
        )
        return 100.0 * (
            (1.0 + real_growth / 100.0) * (1.0 + monthly_price / 100.0) - 1.0
        )
    if mapping.economy_transform == "diff":
        return current_measure - previous_measure
    if mapping.economy_transform == "level":
        return current_measure
    if mapping.economy_transform == "annual_rate_to_monthly_pct":
        return _annual_rate_to_monthly_pct(current_measure)
    raise DynamicMacroError(f"Unknown economy transform: {mapping.economy_transform}")


def economy_measure(row: Mapping[str, Any], measure: str) -> float:
    if measure == "saving_rate_pct":
        income = float(row["aggregate_income"])
        return (
            0.0
            if abs(income) < 1e-12
            else 100.0 * float(row["aggregate_saving"]) / income
        )
    if measure == "unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["employment_rate"]))
    if measure == "next_unemployment_rate_pct":
        return 100.0 * (1.0 - float(row["next_employment_rate"]))
    return float(row[measure])


def _initial_macro_row(households: pd.DataFrame) -> dict[str, float]:
    weights = households["population_weight"].astype(float)
    aggregate_income = float(
        (weights * households["annual_income"].astype(float) / PERIODS_PER_YEAR).sum()
    )
    aggregate_saving = float(
        (
            weights
            * households["annual_income"].astype(float)
            / PERIODS_PER_YEAR
            * households["base_saving_rate"].astype(float)
        ).sum()
    )
    aggregate_consumption = float(
        (
            weights
            * households["baseline_consumption_annual"].astype(float)
            / PERIODS_PER_YEAR
        ).sum()
    )
    aggregate_debt = float((weights * households["debt"].astype(float)).sum())
    return {
        "aggregate_consumption": aggregate_consumption,
        "aggregate_income": aggregate_income,
        "aggregate_saving": aggregate_saving,
        "aggregate_debt": aggregate_debt,
        "employment_rate": 0.955,
        "inflation_rate": 2.0,
        "policy_rate": 2.5,
        "next_employment_rate": 0.955,
        "next_inflation_rate": 2.0,
        "next_policy_rate": 2.5,
        "next_aggregate_income": aggregate_income,
    }


def join_and_score_forecasts(
    forecasts: pd.DataFrame, targets: tuple[dict[str, Any], ...]
) -> pd.DataFrame:
    target_frame = pd.DataFrame(targets)
    joined = forecasts.merge(
        target_frame,
        on=[
            "origin_month",
            "as_of_date",
            "target_observation_date",
            "target_name",
            "series_id",
            "family",
        ],
        how="left",
        validate="many_to_one",
    )
    if joined["target_value"].isna().any() or len(joined) != len(forecasts):
        raise DynamicMacroError("Forecast-target join is incomplete")
    joined["error"] = joined["prediction"] - joined["target_value"]
    joined["scaled_error"] = joined["error"] / joined["default_scale"]
    joined["scaled_squared_error"] = joined["scaled_error"] ** 2
    joined["absolute_scaled_error"] = joined["scaled_error"].abs()
    transformed = ~joined["transform"].eq("level")
    joined["target_direction"] = np.sign(
        np.where(
            transformed,
            joined["target_value"],
            joined["target_value"] - joined["origin_visible_denominator_value"],
        )
    ).astype(int)
    joined["forecast_direction"] = np.sign(
        np.where(
            transformed,
            joined["prediction"],
            joined["prediction"] - joined["origin_visible_denominator_value"],
        )
    ).astype(int)
    joined["direction_correct"] = joined["target_direction"].eq(
        joined["forecast_direction"]
    )
    pivot = joined.pivot(
        index=["origin_month", "target_name"],
        columns="candidate",
        values="scaled_squared_error",
    )
    if set(pivot.columns) != {"llm", "adaptive"}:
        raise DynamicMacroError("Matched score rows are incomplete")
    delta = (pivot["llm"] - pivot["adaptive"]).rename(
        "llm_minus_adaptive_scaled_squared_error"
    )
    joined = joined.merge(
        delta, on=["origin_month", "target_name"], how="left", validate="many_to_one"
    )
    return joined.sort_values(
        ["candidate", "origin_month", "family", "target_name"]
    ).reset_index(drop=True)


def score_macro(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target_scores = joined.groupby(
        ["candidate", "family", "target_name"], as_index=False
    ).agg(
        observations=("scaled_squared_error", "size"),
        mean_squared_scaled_error=("scaled_squared_error", "mean"),
        mean_absolute_scaled_error=("absolute_scaled_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
    )
    target_scores["target_score"] = np.sqrt(target_scores["mean_squared_scaled_error"])
    target_pivot = target_scores.pivot(
        index=["family", "target_name"], columns="candidate", values="target_score"
    )
    target_delta = (target_pivot["llm"] - target_pivot["adaptive"]).rename(
        "llm_minus_adaptive_target_score"
    )
    target_scores = target_scores.merge(
        target_delta,
        on=["family", "target_name"],
        how="left",
        validate="many_to_one",
    )
    family_scores = target_scores.groupby(["candidate", "family"], as_index=False).agg(
        target_count=("target_name", "size"),
        family_mean_squared_scaled_error=("mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    family_scores["family_score"] = np.sqrt(
        family_scores["family_mean_squared_scaled_error"]
    )
    family_pivot = family_scores.pivot(
        index="family", columns="candidate", values="family_score"
    )
    family_delta = (family_pivot["llm"] - family_pivot["adaptive"]).rename(
        "llm_minus_adaptive_family_score"
    )
    family_scores = family_scores.merge(
        family_delta, on="family", how="left", validate="many_to_one"
    )
    macro_scores = {
        str(candidate): float(
            math.sqrt(group["family_mean_squared_scaled_error"].mean())
        )
        for candidate, group in family_scores.groupby("candidate")
    }
    if set(macro_scores) != {"llm", "adaptive"}:
        raise DynamicMacroError("MacroScore requires both matched candidates")
    family_scores["family_equal_weight"] = family_scores.groupby("candidate")[
        "family"
    ].transform(lambda values: 1.0 / len(values))
    family_scores["macro_score"] = family_scores["candidate"].map(macro_scores)
    family_scores["llm_minus_adaptive"] = macro_scores["llm"] - macro_scores["adaptive"]

    origin_target = joined.groupby(
        ["candidate", "origin_month", "family", "target_name"], as_index=False
    ).agg(
        mean_squared_scaled_error=("scaled_squared_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
    )
    origin_family = origin_target.groupby(
        ["candidate", "origin_month", "family"], as_index=False
    ).agg(
        family_mean_squared_scaled_error=("mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    origin_scores = origin_family.groupby(
        ["candidate", "origin_month"], as_index=False
    ).agg(
        family_count=("family", "size"),
        macro_mean_squared_scaled_error=("family_mean_squared_scaled_error", "mean"),
        direction_accuracy=("direction_accuracy", "mean"),
    )
    origin_scores["macro_score"] = np.sqrt(
        origin_scores["macro_mean_squared_scaled_error"]
    )
    origin_pivot = origin_scores.pivot(
        index="origin_month", columns="candidate", values="macro_score"
    )
    origin_delta = (origin_pivot["llm"] - origin_pivot["adaptive"]).rename(
        "llm_minus_adaptive"
    )
    origin_scores = origin_scores.merge(
        origin_delta, on="origin_month", how="left", validate="many_to_one"
    )
    return target_scores, family_scores, origin_scores, macro_scores


def origin_block_bootstrap(
    joined: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_length: int | None,
) -> dict[str, Any]:
    origins = sorted(joined["origin_month"].astype(str).unique())
    n_origins = len(origins)
    if replicates <= 0:
        raise DynamicMacroError("bootstrap replicates must be positive")
    length = int(block_length or max(1, round(math.sqrt(n_origins))))
    if length < 1 or length > n_origins:
        raise DynamicMacroError(
            "bootstrap block length must be between 1 and the origin count"
        )
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    starts = np.arange(n_origins)
    for replicate in range(replicates):
        selected: list[str] = []
        while len(selected) < n_origins:
            start = int(rng.choice(starts))
            selected.extend(
                origins[(start + offset) % n_origins] for offset in range(length)
            )
        pieces: list[pd.DataFrame] = []
        for draw_index, origin in enumerate(selected[:n_origins]):
            piece = joined[joined["origin_month"].astype(str).eq(origin)].copy()
            piece["origin_month"] = f"bootstrap_{replicate}_{draw_index}"
            pieces.append(piece)
        _, _, _, scores = score_macro(pd.concat(pieces, ignore_index=True))
        deltas.append(scores["llm"] - scores["adaptive"])
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975])
    return {
        "method": "circular_moving_origin_block_bootstrap",
        "estimand": "llm_minus_adaptive_family_equal_macro_score",
        "seed": int(seed),
        "replicates": int(replicates),
        "block_length": length,
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def normalized_spec(
    args: argparse.Namespace,
    *,
    bundle: BundleView,
    gains: dict[str, float],
    period_overrides: dict[int, dict[str, Any]],
    initial_environment_anchor: dict[str, float],
    behavior_profile_content_sha256: str | None,
    household_provenance: dict[str, Any],
    household_flow_anchor: dict[str, Any],
    contamination_coverage: dict[str, Any],
) -> dict[str, Any]:
    mappings = []
    for spec in bundle.target_specs:
        mapping = MAPPING_BY_SERIES[spec["series_id"]]
        mappings.append({**spec, **asdict(mapping)})
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_sha256": bundle.bundle_sha256,
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "provider_execution_isolation": provider_execution_isolation(args.mode),
        "periods_per_year": PERIODS_PER_YEAR,
        "feedback_mode": args.feedback_mode,
        "feedback_gain": float(args.feedback_gain),
        "belief_gains": gains,
        "initial_environment_anchor": initial_environment_anchor,
        "initial_environment_anchor_origin": bundle.origins[0],
        "household_provenance": household_provenance,
        "household_flow_anchor": household_flow_anchor,
        "behavior_policy_mode": args.behavior_policy_mode,
        "behavior_policy_content_sha256": behavior_profile_content_sha256,
        "period_overrides": {
            str(key): value for key, value in period_overrides.items()
        },
        "target_mappings": mappings,
        "contamination_coverage": contamination_coverage,
        "macro_score": {
            "target_score": "sqrt(mean squared scaled error across origins)",
            "family_score": "sqrt(mean target mean-squared-scaled-error within family)",
            "macro_score": "sqrt(mean family mean-squared-scaled-error with equal family weight)",
            "adaptive_role": "diagnostic_only_not_a_selection_veto",
            "adaptive_information": "same raw origin-visible history and derived observed signals as the LLM, plus endogenous recursive state",
        },
    }


def provider_execution_isolation(mode: str) -> dict[str, Any]:
    enabled = mode in {"live", "replay"}
    return {
        "enabled": enabled,
        "relative_path": "provider_cwd" if enabled else None,
        "workspace_policy": "empty_isolated_no_repo_access" if enabled else "not_used",
        "included_in_output_contract": False,
    }


def prompt_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "prompt_payload"},
                "prompt_payload": _canonical_json(row["prompt_payload"]),
            }
            for row in rows
        ]
    )


def write_outputs(
    output_dir: Path,
    *,
    spec: dict[str, Any],
    households: pd.DataFrame,
    prompts: pd.DataFrame,
    beliefs: pd.DataFrame,
    decisions: pd.DataFrame,
    periods: pd.DataFrame,
    accounting: pd.DataFrame,
    forecasts: pd.DataFrame,
    joined: pd.DataFrame,
    target_scores: pd.DataFrame,
    family_scores: pd.DataFrame,
    origin_scores: pd.DataFrame,
    raw_records: list[dict[str, Any]],
) -> None:
    (output_dir / "normalized_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, frame in (
        ("households.csv", households),
        ("prompt_cards.csv", prompts),
        ("beliefs.csv", beliefs),
        ("decisions.csv", decisions),
        ("periods.csv", periods),
        ("accounting.csv", accounting),
        ("forecasts.csv", forecasts),
        ("joined_errors.csv", joined),
        ("target_scores.csv", target_scores),
        ("family_scores.csv", family_scores),
        ("origin_scores.csv", origin_scores),
    ):
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "raw_records.json").write_text(
        json.dumps(raw_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_report(
    manifest: dict[str, Any], target_scores: pd.DataFrame, family_scores: pd.DataFrame
) -> str:
    bootstrap = manifest["origin_block_bootstrap"]
    contamination = manifest["contamination_coverage"]
    isolation = manifest["provider_execution_isolation"]
    household_provenance = manifest["household_provenance"]
    household_temporal = household_provenance["temporal_coverage"]
    summary = family_scores[
        ["candidate", "family", "family_score", "direction_accuracy", "macro_score"]
    ].sort_values(["candidate", "family"])
    return "\n".join(
        [
            "# Recursive Dynamic Macro Economy",
            "",
            "## Result",
            f"- LLM family-equal MacroScore: `{manifest['macro_scores']['llm']:.6f}`.",
            f"- Adaptive diagnostic MacroScore: `{manifest['macro_scores']['adaptive']:.6f}`.",
            f"- LLM minus adaptive: `{manifest['llm_minus_adaptive']:.6f}`.",
            f"- 95% origin-block bootstrap interval: `[{bootstrap['lower']:.6f}, {bootstrap['upper']:.6f}]`.",
            "- The adaptive twin is diagnostic only and does not veto or select the LLM candidate.",
            "",
            "## Family Scores",
            markdown_table(summary),
            "",
            "## Target Scores",
            markdown_table(
                target_scores[
                    [
                        "candidate",
                        "family",
                        "target_name",
                        "target_score",
                        "direction_accuracy",
                        "observations",
                    ]
                ].sort_values(["candidate", "family", "target_name"])
            ),
            "",
            "## Integrity",
            f"- Bundle SHA-256: `{manifest['bundle_sha256']}`.",
            f"- Normalized spec SHA-256: `{manifest['normalized_spec_sha256']}`.",
            f"- Normalized household-state SHA-256: `{household_provenance['normalized_household_state_sha256']}`; "
            f"raw input SHA-256: `{household_provenance['raw_input_file_sha256']}`.",
            f"- Behavior-policy canonical content SHA-256: `{manifest['behavior_policy_content_sha256']}`.",
            f"- Household public availability: `{household_temporal['status']}`; latest source availability "
            f"`{household_temporal['source_availability_date_max']}` versus first macro as-of "
            f"`{household_temporal['first_macro_as_of_date']}`.",
            f"- Shared first-origin environment anchor: `{_canonical_json(manifest['initial_environment_anchor'])}`.",
            f"- Household flow anchor: `{_canonical_json(manifest['household_flow_anchor'])}`.",
            f"- Maximum accounting residual: `{manifest['max_accounting_abs_residual']:.12g}`.",
            f"- Monthly origins: `{manifest['origin_count']}`; frozen targets: `{manifest['target_count']}`.",
            f"- Contamination policy `{contamination['policy']}` for `{contamination['model']}` selected "
            f"`{contamination['selected_rows']}` of `{contamination['catalogue_rows']}` origin-target rows.",
            f"- Original label counts available: `{_canonical_json(contamination['available_label_counts'])}`; "
            f"selected: `{_canonical_json(contamination['selected_label_counts'])}`; "
            f"excluded: `{_canonical_json(contamination['excluded_label_counts'])}`.",
            f"- Contamination-filtered origin coverage: `{contamination['complete_origin_count']}` complete, "
            f"`{contamination['partial_origin_count']}` partial, `{contamination['empty_origin_count']}` empty.",
            f"- Provider execution isolation: `{isolation['workspace_policy']}` at relative path `{isolation['relative_path']}`.",
            "- Both twins receive the same raw origin-visible history and observed-signal summary; each retains its own endogenous recursive feedback state.",
            "- Frozen outcomes, target-contamination labels, and revision audit never enter prompts.",
            "",
        ]
    )


def output_hashes(output_dir: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILES
        if name not in exclude and (output_dir / name).is_file()
    }


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("raw_records") or payload.get("records")
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise DynamicMacroError(
            "--raw-records-json must contain a JSON list of records"
        )
    return list(payload)


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == "live" and int(args.max_live_calls) <= 0:
        raise DynamicMacroError("--max-live-calls must be positive in live mode")
    if args.mode != "replay" and args.raw_records_json:
        raise DynamicMacroError("--raw-records-json is only valid in replay mode")
    if not math.isfinite(float(args.feedback_gain)) or float(args.feedback_gain) < 0.0:
        raise DynamicMacroError("--feedback-gain must be finite and non-negative")
    if (
        not math.isfinite(float(args.hybrid_state_weight))
        or not 0.0 <= float(args.hybrid_state_weight) <= 1.0
    ):
        raise DynamicMacroError("--hybrid-state-weight must be between zero and one")


def _first_column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _group(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default


def _finite_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _first_finite(row: pd.Series, columns: Iterable[str], default: float) -> float:
    for column in columns:
        if column in row.index:
            value = _finite_or(row[column], np.nan)
            if math.isfinite(value):
                return value
    return float(default)


def _pct_change(current: float, previous: float) -> float:
    if abs(previous) < 1e-12:
        raise DynamicMacroError(
            "Cannot map a percentage-change target from a zero economy baseline"
        )
    return 100.0 * (current / previous - 1.0)


def _annual_rate_to_monthly_pct(rate: float) -> float:
    if rate <= -100.0:
        raise DynamicMacroError(
            "Annual inflation rate cannot be at or below -100 percent"
        )
    return 100.0 * ((1.0 + rate / 100.0) ** (1.0 / PERIODS_PER_YEAR) - 1.0)


def _monthly_rate_to_annual_pct(rate: float) -> float:
    if rate <= -100.0:
        raise DynamicMacroError(
            "Monthly growth rate cannot be at or below -100 percent"
        )
    return 100.0 * ((1.0 + rate / 100.0) ** PERIODS_PER_YEAR - 1.0)


def _optional_signal(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
