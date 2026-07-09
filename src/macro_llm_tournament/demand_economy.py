from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, WORK_ROOT, bounded_float, cache_key, markdown_table, round_or_none
from .agent_types import build_household_type_cells
from .behavior_ecology import POLICY_PROMPT_VERSION, normalize_policy_payload
from .empirical_bridge import BRIDGE_SPEC_VERSION, DEFAULT_BRIDGE, SUPPORTED_BRIDGE_SPEC_VERSIONS, BridgeInput, transform_belief_change
from .forecast_llm import ForecastLLMClient, SUPPORTED_FORECAST_PROVIDERS
from .llm_common import LLMUnavailable


DEMAND_ECONOMY_VERSION = "hank_lite_belief_demand_economy_v3"
DEMAND_ECONOMY_PROMPT_VERSION = "hank_lite_belief_module_v5"
NAIVE_PERSONA_PROMPT_VERSION = "hank_lite_naive_persona_direct_consumption_v1"
DEMAND_BEHAVIOR_POLICY_VERSION = "demand_behavior_policy_schedule_v1"
DEMAND_STATE_BEHAVIOR_POLICY_VERSION = "demand_behavior_state_policy_schedule_v1"
DEMAND_HYBRID_BEHAVIOR_POLICY_VERSION = "demand_empirical_bridge_state_schedule_hybrid_v1"
BELIEF_MODES = ("fixture", "replay", "live", "raw_replay")
FEEDBACK_MODES = ("closed_loop", "none")
BEHAVIOR_POLICY_MODES = ("fixed_kernel", "schedule", "state_schedule", "empirical_bridge", "empirical_bridge_state_schedule")
MODEL_VARIANTS = ("representative", "adaptive", "llm_belief", "naive_persona")
LLM_VARIANTS = {"llm_belief", "naive_persona"}
NEUTRAL_POLICY_RATE = 3.0
INFLATION_TARGET = 2.0
STEADY_EMPLOYMENT_RATE = 0.955
DEFAULT_BEHAVIOR_POLICY_RAW_RECORDS = OUTPUT_ROOT / "behavior_ecology_gpt55_xhigh" / "ecology_raw_records.json"
DEFAULT_EMPIRICAL_BRIDGE_JSON = DEFAULT_BRIDGE
DEFAULT_PERIODS_PER_YEAR = 4.0
PROTECTED_PERIOD_OVERRIDE_FIELDS = frozenset({"scenario_id", "period_id", "period_index"})
FULL_LAB_LLM_METRICS = {
    "baseline_no_shock_output_gap_rms",
    "belief_feedback_amplification_ratio",
    "belief_inflation_dispersion_p1",
    "high_liquidity_impact_mpc",
    "income_mpc_gradient",
    "job_risk_impact_consumption_delta",
    "job_risk_impact_income_delta_abs",
    "liquidity_mpc_gradient",
    "low_liquidity_impact_mpc",
    "max_accounting_abs_residual",
    "rate_hike_late_inflation_delta",
    "rate_hike_mean_consumption_delta_6p",
    "rate_hike_mean_employment_delta_6p",
    "rate_hike_mean_output_gap_delta_6p",
    "steady_state_final_output_gap_abs",
    "steady_state_tail_inflation_range",
    "steady_state_tail_output_gap_range",
    "transfer_cumulative_mpc_4p",
    "transfer_impact_mpc",
}


@dataclass(frozen=True)
class DemandScenario:
    scenario_id: str
    label: str
    transfer_period: int = -1
    transfer_amount: float = 0.0
    rate_shock_start: int = -1
    rate_shock_end: int = -1
    rate_shock_pp: float = 0.0
    job_risk_shock_start: int = -1
    job_risk_shock_end: int = -1
    job_risk_shock_pp: float = 0.0
    belief_dispersion_multiplier: float = 1.0
    feedback_gain: float = 1.0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HANK-lite behavior demand economy with belief modules.")
    parser.add_argument("--provider", choices=SUPPORTED_FORECAST_PROVIDERS, default="codex_cli")
    parser.add_argument("--models", default="gpt-5.5")
    parser.add_argument("--belief-mode", "--decision-mode", dest="belief_mode", choices=BELIEF_MODES, default="fixture")
    parser.add_argument("--raw-records-json", default=None, help="Replay prior demand_raw_records.json payloads without prompt-cache lookup.")
    parser.add_argument("--belief-calibration-profile", default=None, help="Optional belief_calibration_profile.json to postprocess LLM belief payloads.")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--household-source", choices=["fixture", "csv"], default="fixture")
    parser.add_argument("--household-csv", default=None)
    parser.add_argument("--household-count", type=int, default=24)
    parser.add_argument("--period-count", type=int, default=100)
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default="closed_loop")
    parser.add_argument("--behavior-policy-mode", choices=BEHAVIOR_POLICY_MODES, default="fixed_kernel")
    parser.add_argument(
        "--behavior-policy-raw-records-json",
        default=None,
        help="Behavior-ecology raw records containing policy_transfer and policy_income_loss payloads.",
    )
    parser.add_argument(
        "--behavior-policy-state-profile-json",
        default=None,
        help="State-conditioned policy profile JSON from state_policy_schedules.",
    )
    parser.add_argument("--empirical-bridge-json", default=None, help="Accepted empirical bridge artifact.")
    parser.add_argument(
        "--hybrid-state-weight",
        type=float,
        default=1.0,
        help="Weight on state-schedule shock/transfer response in empirical_bridge_state_schedule mode.",
    )
    parser.add_argument("--variants", default="representative,adaptive,llm_belief,naive_persona")
    parser.add_argument(
        "--fixture-variants",
        default="",
        help="Comma-separated variants to force through fixture mode during a live run; useful for zero-cost baselines.",
    )
    parser.add_argument("--scenarios", default="baseline,transfer_shock,rate_hike,job_risk_shock,belief_feedback")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    fixture_variants = {part.strip() for part in args.fixture_variants.split(",") if part.strip()}
    unknown_variants = sorted(set(variants) - set(MODEL_VARIANTS))
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {', '.join(unknown_variants)}")
    unknown_fixture_variants = sorted(fixture_variants - set(MODEL_VARIANTS))
    if unknown_fixture_variants:
        raise SystemExit(f"Unknown fixture variants: {', '.join(unknown_fixture_variants)}")
    scenario_ids = [part.strip() for part in args.scenarios.split(",") if part.strip()]
    scenarios = [scenario for scenario in default_demand_scenarios() if scenario.scenario_id in set(scenario_ids)]
    missing_scenarios = sorted(set(scenario_ids) - {scenario.scenario_id for scenario in scenarios})
    if missing_scenarios:
        raise SystemExit(f"Unknown scenarios: {', '.join(missing_scenarios)}")
    if not models:
        raise SystemExit("--models must contain at least one model")
    if not variants:
        raise SystemExit("--variants must contain at least one variant")
    if not scenarios:
        raise SystemExit("--scenarios must contain at least one scenario")
    if args.period_count < 8:
        raise SystemExit("--period-count must be at least 8 for impulse-response validation")
    live_variant_count = sum(1 for variant in variants if variant in LLM_VARIANTS and variant not in fixture_variants)
    required_calls = len(models) * len(scenarios) * int(args.period_count) * live_variant_count
    if args.belief_mode == "live" and required_calls > 0:
        if args.max_live_calls <= 0:
            raise SystemExit("--max-live-calls must be positive when --belief-mode live is used")
        if not args.fresh_cache:
            raise SystemExit("--fresh-cache is required when --belief-mode live is used")
        if args.max_live_calls < required_calls:
            raise SystemExit(
                "--max-live-calls must be at least "
                f"{required_calls} for a fresh live run with {len(models)} model(s), "
                f"{live_variant_count} live variant(s), {len(scenarios)} scenario(s), and {args.period_count} periods"
            )
    if args.household_source == "csv":
        if not args.household_csv:
            raise SystemExit("--household-csv is required when --household-source csv")
        if not Path(args.household_csv).exists():
            raise SystemExit(f"--household-csv does not exist: {args.household_csv}")
    if args.belief_mode == "raw_replay":
        if not args.raw_records_json:
            raise SystemExit("--raw-records-json is required when --belief-mode raw_replay")
        if not Path(args.raw_records_json).exists():
            raise SystemExit(f"--raw-records-json does not exist: {args.raw_records_json}")
    behavior_policy_profile = _behavior_policy_profile_from_args(args)
    belief_calibration_profile = _load_belief_calibration_profile(Path(args.belief_calibration_profile)) if args.belief_calibration_profile else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"demand_economy_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fresh_demand_economy_cache" if args.fresh_cache else WORK_ROOT / "demand_economy_cache"
    raw_replay_records = _load_raw_replay_records(Path(args.raw_records_json)) if args.belief_mode == "raw_replay" else []
    households = (
        normalize_demand_households(pd.read_csv(Path(args.household_csv)))
        if args.household_source == "csv"
        else build_fixture_demand_households(args.household_count)
    )

    all_initial: list[pd.DataFrame] = []
    all_beliefs: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_periods: list[pd.DataFrame] = []
    all_accounting: list[pd.DataFrame] = []
    all_prompt_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    live_used = 0
    cache_hits = 0
    client_specs = _client_specs(variants, models, args.belief_mode, fixture_variants)
    for variant, model, client_mode in client_specs:
        client = DemandEconomyClient(
            args.provider,
            model,
            cache_dir,
            mode=client_mode,
            variant=variant,
            max_live_calls=max(0, int(args.max_live_calls) - live_used),
            raw_replay_records=raw_replay_records,
            belief_calibration_profile=belief_calibration_profile,
        )
        initial, beliefs, decisions, periods, accounting, prompt_rows = run_demand_economy(
            households,
            scenarios,
            client,
            period_count=args.period_count,
            feedback_mode=args.feedback_mode,
            behavior_policy_profile=behavior_policy_profile,
        )
        all_initial.append(initial)
        all_beliefs.append(beliefs)
        all_decisions.append(decisions)
        all_periods.append(periods)
        all_accounting.append(accounting)
        all_prompt_rows.extend(prompt_rows)
        raw_records.extend(client.raw_records)
        live_used += client.live_call_count
        cache_hits += client.cache_hit_count

    initial_frame = pd.concat(all_initial, ignore_index=True) if all_initial else pd.DataFrame()
    beliefs_frame = pd.concat(all_beliefs, ignore_index=True) if all_beliefs else pd.DataFrame()
    decisions_frame = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    periods_frame = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    accounting_frame = pd.concat(all_accounting, ignore_index=True) if all_accounting else pd.DataFrame()
    validation = score_demand_economy_validation(periods_frame, decisions_frame, beliefs_frame, accounting_frame)
    belief_targets = score_demand_belief_targets(initial_frame, beliefs_frame)
    ablations = build_ablation_table(validation, periods_frame, decisions_frame, beliefs_frame)
    evidence = classify_demand_economy_evidence(validation, ablations, mode=args.belief_mode)
    output_names = [
        "demand_households.csv",
        "demand_initial_state.csv",
        "demand_beliefs.csv",
        "demand_household_decisions.csv",
        "demand_periods.csv",
        "demand_accounting.csv",
        "demand_validation_scores.csv",
        "demand_belief_target_scores.csv",
        "demand_ablation_table.csv",
        "demand_prompt_cards.jsonl",
        "demand_raw_records.json",
        "demand_economy_report.md",
        "manifest.json",
    ]
    if behavior_policy_profile is not None:
        output_names.insert(9, "demand_behavior_policy_profile.json")

    manifest = {
        "schema_version": DEMAND_ECONOMY_VERSION,
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "naive_persona_prompt_version": NAIVE_PERSONA_PROMPT_VERSION,
        "timestamp_utc": timestamp,
        "argv": _sanitized_argv(),
        "run_command": shlex.join(_sanitized_argv()),
        "git": _git_metadata(),
        "status": "ok",
        "provider": args.provider,
        "models": models,
        "variants": variants,
        "fixture_variants": sorted(fixture_variants),
        "belief_mode": args.belief_mode,
        "fresh_cache": bool(args.fresh_cache),
        "raw_records_json": args.raw_records_json if args.belief_mode == "raw_replay" else None,
        "belief_calibration_profile": args.belief_calibration_profile,
        "belief_calibration_profile_id": belief_calibration_profile.get("profile_id") if belief_calibration_profile else None,
        "max_live_calls": int(args.max_live_calls),
        "live_call_count": int(live_used),
        "cache_hit_count": int(cache_hits),
        "household_source": args.household_source,
        "household_count": int(households.shape[0]),
        "period_count": int(args.period_count),
        "feedback_mode": args.feedback_mode,
        "behavior_policy": behavior_policy_manifest(behavior_policy_profile, mode=args.behavior_policy_mode),
        "scenario_count": len(scenarios),
        "scenarios": [scenario.__dict__ for scenario in scenarios],
        "verdict": evidence.get("evidence_verdict"),
        "passed": evidence.get("passed"),
        "full_lab_passed": evidence.get("full_lab_passed"),
        "canary_passed": evidence.get("canary_passed"),
        "evidence": evidence,
        "outputs": output_names,
    }

    households.to_csv(output_dir / "demand_households.csv", index=False)
    initial_frame.to_csv(output_dir / "demand_initial_state.csv", index=False)
    beliefs_frame.to_csv(output_dir / "demand_beliefs.csv", index=False)
    decisions_frame.to_csv(output_dir / "demand_household_decisions.csv", index=False)
    periods_frame.to_csv(output_dir / "demand_periods.csv", index=False)
    accounting_frame.to_csv(output_dir / "demand_accounting.csv", index=False)
    validation.to_csv(output_dir / "demand_validation_scores.csv", index=False)
    belief_targets.to_csv(output_dir / "demand_belief_target_scores.csv", index=False)
    ablations.to_csv(output_dir / "demand_ablation_table.csv", index=False)
    if behavior_policy_profile is not None:
        _write_json(output_dir / "demand_behavior_policy_profile.json", behavior_policy_profile)
    pd.DataFrame(all_prompt_rows).to_json(output_dir / "demand_prompt_cards.jsonl", orient="records", lines=True)
    _write_json(output_dir / "demand_raw_records.json", raw_records)
    _write_json(output_dir / "manifest.json", manifest)
    report = build_demand_economy_report(
        manifest,
        households,
        periods_frame,
        beliefs_frame,
        validation,
        belief_targets,
        ablations,
        accounting_frame,
    )
    (output_dir / "demand_economy_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote demand economy run to {output_dir}")
    print(json.dumps(_jsonable(evidence), indent=2, sort_keys=True, allow_nan=False))
    return 0


class DemandEconomyClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str = "fixture",
        variant: str = "llm_belief",
        max_live_calls: int = 0,
        raw_replay_records: list[dict[str, Any]] | None = None,
        belief_calibration_profile: dict[str, Any] | None = None,
        execution_cwd: Path | None = None,
    ):
        if mode not in BELIEF_MODES:
            raise ValueError(f"Unsupported demand-economy mode: {mode}")
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported demand-economy variant: {variant}")
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.variant = variant
        self.belief_calibration_profile = belief_calibration_profile if variant == "llm_belief" else None
        self.raw_records: list[dict[str, Any]] = []
        self._raw_replay_records = _raw_replay_record_map(raw_replay_records or [])
        llm_mode = mode if variant in LLM_VARIANTS else "fixture"
        self._llm = ForecastLLMClient(
            provider,
            model,
            cache_dir,
            mode=llm_mode,
            max_live_calls=max_live_calls,
            execution_cwd=execution_cwd,
        )

    @property
    def live_call_count(self) -> int:
        return int(self._llm.live_call_count)

    @property
    def cache_hit_count(self) -> int:
        return int(self._llm.cache_hit_count)

    @property
    def source(self) -> str:
        if self.variant in LLM_VARIANTS:
            prefix = {
                "fixture": "fixture",
                "raw_replay": "raw_replay",
                "replay": f"replay_{self.provider}",
                "live": f"live_{self.provider}",
            }[self.mode]
            suffix = "_calibrated" if self.belief_calibration_profile else ""
            return f"{self.variant}_{prefix}_{self.model}{suffix}"
        return self.variant

    def belief_cache_path(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> Path:
        if self.variant != "llm_belief":
            raise ValueError("Belief cache paths are only defined for llm_belief")
        prompt_payload = belief_module_prompt_payload(
            scenario, period_state, household_states, variant=self.variant
        )
        cache_name = (
            "demand_belief_"
            + cache_key(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "prompt": prompt_payload,
                }
            )
        )
        return self._llm.cache_path(cache_name)

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.variant == "representative":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": representative_belief_payload(scenario, period_state, household_states),
                "cache_hit": True,
                "cache_path": None,
            }
            prompt_payload = belief_module_prompt_payload(scenario, period_state, household_states, variant=self.variant)
        elif self.variant == "adaptive":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": adaptive_belief_payload(scenario, period_state, household_states),
                "cache_hit": True,
                "cache_path": None,
            }
            prompt_payload = belief_module_prompt_payload(scenario, period_state, household_states, variant=self.variant)
        elif self.variant == "naive_persona":
            prompt_payload = naive_persona_prompt_payload(scenario, period_state, household_states)
            if self.mode == "fixture":
                data = {
                    "provider": self.provider,
                    "model": self.model,
                    "payload": fixture_naive_persona_payload(scenario, period_state, household_states),
                    "cache_hit": True,
                    "cache_path": None,
                }
            else:
                prompt_text = naive_persona_prompt(scenario, period_state, household_states)
                cache_name = f"demand_naive_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt_payload})}"
                data = self._llm.json_call(prompt_text, cache_name, instructions=_demand_instructions())
        else:
            prompt_payload = belief_module_prompt_payload(scenario, period_state, household_states, variant=self.variant)
            if self.mode == "fixture":
                data = {
                    "provider": self.provider,
                    "model": self.model,
                    "payload": fixture_belief_payload(scenario, period_state, household_states),
                    "cache_hit": True,
                    "cache_path": None,
                }
            elif self.mode == "raw_replay":
                data = self._raw_replay_payload(scenario, period_state)
            else:
                prompt_text = belief_module_prompt(scenario, period_state, household_states, variant=self.variant)
                cache_name = f"demand_belief_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt_payload})}"
                data = self._llm.json_call(prompt_text, cache_name, instructions=_demand_instructions())
        normalized = normalize_period_payload(household_states, data, variant=self.variant)
        if self.belief_calibration_profile and self.variant == "llm_belief":
            normalized = apply_belief_calibration_profile(
                normalized,
                scenario,
                period_state,
                household_states,
                self.belief_calibration_profile,
            )
        self.raw_records.append(
            {
                "source": self.source,
                "variant": self.variant,
                "scenario_id": scenario.scenario_id,
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "belief_calibration_profile_id": (
                    self.belief_calibration_profile.get("profile_id") if self.belief_calibration_profile else None
                ),
                "payload": data.get("payload", data),
            }
        )
        return normalized

    def _raw_replay_payload(self, scenario: DemandScenario, period_state: dict[str, Any]) -> dict[str, Any]:
        key = (self.provider, self.model, self.variant, scenario.scenario_id, int(period_state["period_index"]))
        record = self._raw_replay_records.get(key)
        if record is None:
            raise LLMUnavailable(
                f"Raw replay record missing for provider={self.provider}, model={self.model}, "
                f"variant={self.variant}, scenario={scenario.scenario_id}, period={int(period_state['period_index'])}"
            )
        return {
            "provider": record.get("provider", self.provider),
            "model": record.get("model", self.model),
            "payload": record.get("payload", record),
            "cache_hit": True,
            "cache_path": f"raw_records:{scenario.scenario_id}:{int(period_state['period_index'])}",
        }

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.belief_panel(scenario, period_state, household_states)


def default_demand_scenarios() -> list[DemandScenario]:
    return [
        DemandScenario(
            "baseline",
            "No exogenous shock; households react only to endogenous feedback.",
            notes="Control path for stability and impulse-response scoring.",
        ),
        DemandScenario(
            "transfer_shock",
            "One-period lump-sum household transfer.",
            transfer_period=1,
            transfer_amount=1000.0,
            notes="Tests aggregate MPC and liquidity/income gradients.",
        ),
        DemandScenario(
            "rate_hike",
            "Temporary one percentage point policy-rate hike.",
            rate_shock_start=1,
            rate_shock_end=4,
            rate_shock_pp=1.0,
            notes="Tests whether consumption/output fall and inflation cools after monetary tightening.",
        ),
        DemandScenario(
            "job_risk_shock",
            "Temporary perceived job-loss-risk shock without an immediate income shock.",
            job_risk_shock_start=1,
            job_risk_shock_end=4,
            job_risk_shock_pp=8.0,
            notes="Tests precautionary saving before realized income changes.",
        ),
        DemandScenario(
            "belief_feedback",
            "No exogenous shock; stronger belief dispersion and feedback.",
            belief_dispersion_multiplier=1.8,
            feedback_gain=1.35,
            notes="Tests whether heterogeneous beliefs plus feedback amplify fluctuations.",
        ),
    ]


def _load_raw_replay_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--raw-records-json must contain a list of demand raw records")
    return [record for record in data if isinstance(record, dict)]


def _load_belief_calibration_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--belief-calibration-profile must contain a JSON object")
    if str(data.get("schema_version", "")) != "belief_dynamics_calibration_v1":
        raise SystemExit("--belief-calibration-profile has an unsupported schema_version")
    if not isinstance(data.get("demand_adjustments"), dict):
        raise SystemExit("--belief-calibration-profile is missing demand_adjustments")
    return data


def _behavior_policy_profile_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.behavior_policy_mode == "fixed_kernel":
        return None
    if args.behavior_policy_mode == "empirical_bridge":
        path = Path(args.empirical_bridge_json) if args.empirical_bridge_json else DEFAULT_EMPIRICAL_BRIDGE_JSON
        if not path.exists():
            raise SystemExit(f"Empirical bridge artifact does not exist: {path}")
        return load_empirical_bridge_profile(path)
    if args.behavior_policy_mode in {"state_schedule", "empirical_bridge_state_schedule"}:
        if not args.behavior_policy_state_profile_json:
            raise SystemExit(f"--behavior-policy-state-profile-json is required when --behavior-policy-mode {args.behavior_policy_mode}")
        path = Path(args.behavior_policy_state_profile_json)
        if not path.exists():
            raise SystemExit(f"--behavior-policy-state-profile-json does not exist: {path}")
        state_profile = load_state_behavior_policy_profile(path)
        if args.behavior_policy_mode == "state_schedule":
            return state_profile
        bridge_path = Path(args.empirical_bridge_json) if args.empirical_bridge_json else DEFAULT_EMPIRICAL_BRIDGE_JSON
        if not bridge_path.exists():
            raise SystemExit(f"Empirical bridge artifact does not exist: {bridge_path}")
        return build_hybrid_behavior_policy_profile(
            load_empirical_bridge_profile(bridge_path),
            state_profile,
            state_weight=float(args.hybrid_state_weight),
        )
    path = Path(args.behavior_policy_raw_records_json) if args.behavior_policy_raw_records_json else DEFAULT_BEHAVIOR_POLICY_RAW_RECORDS
    if not path.exists():
        raise SystemExit(f"--behavior-policy-raw-records-json does not exist: {path}")
    return load_behavior_policy_profile(path)


def load_state_behavior_policy_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--behavior-policy-state-profile-json must contain a JSON object")
    if str(data.get("schema_version")) != DEMAND_STATE_BEHAVIOR_POLICY_VERSION:
        raise SystemExit(
            "--behavior-policy-state-profile-json has unsupported schema_version: "
            f"{data.get('schema_version')!r}"
        )
    profiles = data.get("profile_rows")
    policies = data.get("state_policies")
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("--behavior-policy-state-profile-json is missing profile_rows")
    if not isinstance(policies, dict) or not policies:
        raise SystemExit("--behavior-policy-state-profile-json is missing state_policies")
    missing = sorted(str(row.get("profile_id", "")) for row in profiles if str(row.get("profile_id", "")) not in policies)
    if missing:
        raise SystemExit(f"--behavior-policy-state-profile-json policies missing profile ids: {', '.join(missing)}")
    out = dict(data)
    out["profile_json"] = str(path)
    out["profile_json_sha256"] = file_sha256(path)
    return out


def load_behavior_policy_profile(raw_records_path: Path) -> dict[str, Any]:
    raw_records = json.loads(raw_records_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise SystemExit("Behavior policy raw records must be a JSON list")
    type_cells, type_status = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=2022)
    policies: dict[str, dict[str, Any]] = {}
    record_hashes: dict[str, str] = {}
    for family in ("transfer", "income_loss"):
        record_type = f"policy_{family}"
        matches = [record for record in raw_records if isinstance(record, dict) and str(record.get("record_type")) == record_type]
        if not matches:
            raise SystemExit(f"Behavior policy raw records missing {record_type}")
        payload = matches[-1].get("payload", matches[-1])
        policies[family] = normalize_policy_payload(type_cells, {"payload": payload}, family=family)
        record_hashes[family] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    parent_manifest = _load_optional_json(raw_records_path.parent / "manifest.json")
    source_label = "policy_schedule"
    if parent_manifest:
        source_label = f"policy_{parent_manifest.get('provider', 'unknown')}_{parent_manifest.get('model', 'unknown')}"
    return {
        "schema_version": DEMAND_BEHAVIOR_POLICY_VERSION,
        "policy_prompt_version": POLICY_PROMPT_VERSION,
        "source_label": source_label,
        "raw_records_json": str(raw_records_path),
        "raw_records_sha256": file_sha256(raw_records_path),
        "policy_record_sha256": record_hashes,
        "source_manifest": {
            "path": str(raw_records_path.parent / "manifest.json") if parent_manifest else None,
            "schema_version": parent_manifest.get("schema_version") if parent_manifest else None,
            "mode": parent_manifest.get("mode") if parent_manifest else None,
            "provider": parent_manifest.get("provider") if parent_manifest else None,
            "model": parent_manifest.get("model") if parent_manifest else None,
            "claim_scope": parent_manifest.get("claim_scope") if parent_manifest else None,
        },
        "type_cell_status": type_status,
        "type_cells": type_cells.to_dict(orient="records"),
        "transfer_policies": policies["transfer"],
        "income_loss_policies": policies["income_loss"],
        "assignment_method": "nearest_scf_type_cell_by_log_income_buffer_and_debt",
        "execution_rule": (
            "The LLM-authored schedule supplies transfer allocation and job-risk spending sensitivity; "
            "deterministic code interpolates, enforces budgets, and performs accounting."
        ),
    }


def load_empirical_bridge_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Empirical bridge artifact must contain a JSON object")
    bridge_spec_version = str(data.get("bridge_spec_version") or data.get("schema_version"))
    if bridge_spec_version not in SUPPORTED_BRIDGE_SPEC_VERSIONS:
        raise ValueError(f"Unsupported empirical bridge schema: {data.get('schema_version')!r}")
    if str(data.get("status")) != "accepted":
        raise ValueError(f"Empirical bridge is fail-closed and not accepted: status={data.get('status')!r}")
    out = dict(data)
    out["bridge_spec_version"] = bridge_spec_version
    out.setdefault("schema_version", bridge_spec_version)
    out["profile_json"] = str(path)
    out["profile_json_sha256"] = file_sha256(path)
    return out


def build_hybrid_behavior_policy_profile(
    empirical_bridge_profile: dict[str, Any],
    state_schedule_profile: dict[str, Any],
    *,
    state_weight: float,
) -> dict[str, Any]:
    if str(empirical_bridge_profile.get("bridge_spec_version") or empirical_bridge_profile.get("schema_version")) not in SUPPORTED_BRIDGE_SPEC_VERSIONS:
        raise ValueError("Hybrid behavior policy requires an accepted empirical bridge profile")
    if str(state_schedule_profile.get("schema_version")) != DEMAND_STATE_BEHAVIOR_POLICY_VERSION:
        raise ValueError("Hybrid behavior policy requires a state-schedule profile")
    weight = float(np.clip(float(state_weight), 0.0, 1.0))
    return {
        "schema_version": DEMAND_HYBRID_BEHAVIOR_POLICY_VERSION,
        "state_weight": weight,
        "empirical_bridge_profile": empirical_bridge_profile,
        "state_schedule_profile": state_schedule_profile,
        "execution_rule": (
            "The empirical bridge supplies baseline consumption growth from belief changes. "
            "The state-schedule policy supplies transfer allocation and shock drag, scaled by state_weight."
        ),
    }


def behavior_policy_manifest(profile: dict[str, Any] | None, *, mode: str) -> dict[str, Any]:
    if profile is None:
        return {"mode": mode, "schema_version": None}
    if profile.get("schema_version") == DEMAND_HYBRID_BEHAVIOR_POLICY_VERSION:
        bridge = profile.get("empirical_bridge_profile", {})
        state = profile.get("state_schedule_profile", {})
        return {
            "mode": mode,
            "schema_version": profile.get("schema_version"),
            "state_weight": profile.get("state_weight"),
            "bridge_spec_version": bridge.get("bridge_spec_version"),
            "empirical_bridge_sha256": bridge.get("canonical_payload_sha256"),
            "empirical_bridge_json": bridge.get("profile_json"),
            "state_schedule_json": state.get("profile_json"),
            "state_schedule_sha256": state.get("profile_json_sha256"),
            "state_schedule_policy_count": len(state.get("state_policies", {})),
            "execution_rule": profile.get("execution_rule"),
        }
    if str(profile.get("bridge_spec_version") or profile.get("schema_version")) in SUPPORTED_BRIDGE_SPEC_VERSIONS:
        return {
            "mode": mode,
            "schema_version": profile.get("schema_version"),
            "bridge_spec_version": profile.get("bridge_spec_version"),
            "profile_json": profile.get("profile_json"),
            "profile_json_sha256": profile.get("profile_json_sha256"),
            "empirical_bridge_sha256": profile.get("canonical_payload_sha256"),
            "status": profile.get("status"),
            "fit_waves": profile.get("fit_waves"),
            "validation_waves": profile.get("validation_waves"),
            "between_coefficients": profile.get("between_coefficients"),
            "constraints": profile.get("constraints"),
            "validation_gate": profile.get("validation_gate"),
            "execution_rule": profile.get("transform_rule"),
        }
    if profile.get("schema_version") == DEMAND_STATE_BEHAVIOR_POLICY_VERSION:
        return {
            "mode": mode,
            "schema_version": profile.get("schema_version"),
            "prompt_version": profile.get("prompt_version"),
            "source_label": profile.get("source_label"),
            "provider": profile.get("provider"),
            "model": profile.get("model"),
            "profile_json": profile.get("profile_json"),
            "profile_json_sha256": profile.get("profile_json_sha256"),
            "profile_count": len(profile.get("profile_rows", [])),
            "policy_count": len(profile.get("state_policies", {})),
            "input_manifest": profile.get("input_manifest"),
            "assignment_method": profile.get("assignment_method"),
            "execution_rule": profile.get("execution_rule"),
        }
    return {
        "mode": mode,
        "schema_version": profile.get("schema_version"),
        "policy_prompt_version": profile.get("policy_prompt_version"),
        "source_label": profile.get("source_label"),
        "raw_records_json": profile.get("raw_records_json"),
        "raw_records_sha256": profile.get("raw_records_sha256"),
        "policy_record_sha256": profile.get("policy_record_sha256"),
        "source_manifest": profile.get("source_manifest"),
        "type_cell_count": len(profile.get("type_cells", [])),
        "transfer_policy_count": len(profile.get("transfer_policies", {})),
        "income_loss_policy_count": len(profile.get("income_loss_policies", {})),
        "assignment_method": profile.get("assignment_method"),
        "execution_rule": profile.get("execution_rule"),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_replay_record_map(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    mapped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for record in records:
        provider = str(record.get("provider", ""))
        model = str(record.get("model", ""))
        variant = str(record.get("variant", ""))
        scenario_id = str(record.get("scenario_id", ""))
        try:
            period_index = int(record.get("period_index"))
        except (TypeError, ValueError):
            continue
        if not provider or not model or not variant or not scenario_id:
            continue
        key = (provider, model, variant, scenario_id, period_index)
        if key in mapped:
            raise ValueError(
                "Duplicate raw replay record for "
                f"provider={provider}, model={model}, variant={variant}, scenario={scenario_id}, period={period_index}"
            )
        mapped[key] = record
    return mapped


def build_fixture_demand_households(household_count: int = 24) -> pd.DataFrame:
    income_specs = {
        "low": {"weight": 0.34, "annual_income": 36000.0, "consumption_ratio": 0.94, "debt_service": 0.16, "prior_pi": 3.4, "prior_income": 0.4},
        "middle": {"weight": 0.33, "annual_income": 72000.0, "consumption_ratio": 0.82, "debt_service": 0.13, "prior_pi": 2.8, "prior_income": 1.0},
        "high": {"weight": 0.33, "annual_income": 135000.0, "consumption_ratio": 0.68, "debt_service": 0.09, "prior_pi": 2.3, "prior_income": 1.6},
    }
    liquid_specs = {
        "low": {"weight_by_income": {"low": 0.70, "middle": 0.45, "high": 0.22}, "months": {"low": 0.5, "middle": 0.9, "high": 1.4}},
        "high": {"weight_by_income": {"low": 0.30, "middle": 0.55, "high": 0.78}, "months": {"low": 3.5, "middle": 5.0, "high": 8.5}},
    }
    risk_specs = {
        "low": {"weight_by_income": {"low": 0.45, "middle": 0.60, "high": 0.72}, "job_loss": {"low": 6.0, "middle": 4.0, "high": 2.5}},
        "high": {"weight_by_income": {"low": 0.55, "middle": 0.40, "high": 0.28}, "job_loss": {"low": 12.0, "middle": 8.0, "high": 5.0}},
    }
    age_specs = {
        "prime": {"weight": 0.62, "income_factor": 1.08, "consumption_factor": 1.02, "mpc_shift": 0.04, "rate_shift": 0.08},
        "older": {"weight": 0.38, "income_factor": 0.88, "consumption_factor": 0.92, "mpc_shift": -0.04, "rate_shift": -0.04},
    }
    rows: list[dict[str, Any]] = []
    for income_group, income in income_specs.items():
        for liquidity_group, liquidity in liquid_specs.items():
            for job_risk_type, risk in risk_specs.items():
                for age_bucket, age in age_specs.items():
                    weight = (
                        income["weight"]
                        * liquidity["weight_by_income"][income_group]
                        * risk["weight_by_income"][income_group]
                        * age["weight"]
                    )
                    annual_income = income["annual_income"] * age["income_factor"]
                    baseline_consumption = annual_income * income["consumption_ratio"] * age["consumption_factor"]
                    liquid_months = liquidity["months"][income_group]
                    liquid_assets = liquid_months * (baseline_consumption / 12.0)
                    low_liquid = liquidity_group == "low"
                    high_risk = job_risk_type == "high"
                    base_mpc = (
                        0.72
                        if low_liquid
                        else 0.24
                    )
                    base_mpc += {"low": 0.09, "middle": 0.0, "high": -0.08}[income_group]
                    base_mpc += age["mpc_shift"]
                    base_mpc += 0.04 if high_risk else -0.02
                    target_buffer = (1.5 if low_liquid else 5.0) + (1.2 if high_risk else 0.0) + (0.8 if age_bucket == "older" else 0.0)
                    attention_prices = 0.74 if income_group == "low" else 0.60 if income_group == "middle" else 0.48
                    attention_jobs = 0.78 if high_risk else 0.46
                    attention_rates = 0.68 if liquidity_group == "high" or income_group == "high" else 0.36
                    rows.append(
                        {
                            "type_id": f"{income_group}_{liquidity_group}_liquid_{job_risk_type}_risk_{age_bucket}",
                            "label": f"{income_group} income, {liquidity_group} liquid assets, {job_risk_type} job risk, {age_bucket} age",
                            "population_weight": weight,
                            "age_bucket": age_bucket,
                            "income_group": income_group,
                            "liquidity_group": liquidity_group,
                            "job_loss_risk_type": job_risk_type,
                            "employment_status": "employed_high_risk" if high_risk else "employed_low_risk",
                            "annual_income": annual_income,
                            "baseline_consumption_annual": baseline_consumption,
                            "liquid_assets": liquid_assets,
                            "debt": annual_income * income["debt_service"] * (1.4 if age_bucket == "prime" else 0.7),
                            "debt_service_burden": income["debt_service"],
                            "base_mpc": float(np.clip(base_mpc, 0.08, 0.92)),
                            "base_saving_rate": float(np.clip(0.12 + (0.10 if liquidity_group == "high" else -0.03) + (0.05 if income_group == "high" else 0.0), 0.02, 0.35)),
                            "rate_sensitivity": float(np.clip(0.28 + attention_rates * 0.42 + age["rate_shift"], 0.10, 0.90)),
                            "income_sensitivity": {"low": 0.82, "middle": 0.58, "high": 0.36}[income_group],
                            "precautionary_sensitivity": float(np.clip(0.34 + (0.30 if low_liquid else 0.06) + (0.20 if high_risk else 0.0), 0.10, 0.90)),
                            "baseline_job_loss_probability": risk["job_loss"][income_group],
                            "target_buffer_months": target_buffer,
                            "inflation_expectation_1y": income["prior_pi"] + (0.25 if high_risk else 0.0),
                            "income_growth_expectation_1y": income["prior_income"] - (0.25 if high_risk else 0.0),
                            "confidence_index": 52.0 + (8.0 if income_group == "high" else -6.0 if income_group == "low" else 0.0) - (5.0 if high_risk else 0.0),
                            "attention_weight_prices": attention_prices,
                            "attention_weight_jobs": attention_jobs,
                            "attention_weight_rates": attention_rates,
                            "income_volatility": 0.10 + (0.08 if high_risk else 0.02) + (0.04 if income_group == "low" else 0.0),
                            "subsistence_floor_share": 0.56 if income_group == "low" else 0.48 if income_group == "middle" else 0.38,
                        }
                    )
    full = normalize_demand_households(pd.DataFrame(rows))
    count = max(1, min(int(household_count or len(full)), len(full)))
    if count >= len(full):
        return full
    selected = np.linspace(0, len(full) - 1, count, dtype=int)
    return normalize_demand_households(full.iloc[selected].copy())


def normalize_demand_households(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "type_id",
        "label",
        "population_weight",
        "income_group",
        "liquidity_group",
        "annual_income",
        "baseline_consumption_annual",
        "liquid_assets",
        "debt",
        "base_mpc",
        "rate_sensitivity",
        "income_sensitivity",
        "precautionary_sensitivity",
        "baseline_job_loss_probability",
        "target_buffer_months",
        "inflation_expectation_1y",
        "income_growth_expectation_1y",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Demand household panel missing columns: {', '.join(sorted(missing))}")
    out = frame.copy()
    defaults: dict[str, Any] = {
        "age_bucket": "unknown",
        "job_loss_risk_type": "unknown",
        "employment_status": "unknown",
        "debt_service_burden": 0.10,
        "base_saving_rate": 0.12,
        "confidence_index": 50.0,
        "attention_weight_prices": 0.55,
        "attention_weight_jobs": 0.55,
        "attention_weight_rates": 0.55,
        "income_volatility": 0.12,
        "subsistence_floor_share": 0.50,
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    text_columns = {"type_id", "label", "income_group", "liquidity_group", "employment_status", "age_bucket", "job_loss_risk_type"}
    numeric_columns = sorted(set(required).union(defaults) - text_columns)
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric_columns].isna().any().any():
        raise ValueError("Demand household panel contains non-numeric required values")
    out["type_id"] = out["type_id"].astype(str)
    if out["type_id"].duplicated().any():
        raise ValueError("Demand household type_id values must be unique")
    out["population_weight"] = out["population_weight"].clip(lower=0.0)
    total_weight = float(out["population_weight"].sum())
    if total_weight <= 0:
        raise ValueError("Demand household population weights must sum to a positive value")
    out["population_weight"] = out["population_weight"] / total_weight
    out["base_mpc"] = out["base_mpc"].clip(lower=0.02, upper=0.98)
    for column in ["attention_weight_prices", "attention_weight_jobs", "attention_weight_rates"]:
        out[column] = out[column].clip(lower=0.0, upper=1.0)
    if "unemployment_higher_probability_1y" not in out:
        out["unemployment_higher_probability_1y"] = out["baseline_job_loss_probability"] / 0.24
    out["unemployment_higher_probability_1y"] = pd.to_numeric(out["unemployment_higher_probability_1y"], errors="coerce").fillna(
        out["baseline_job_loss_probability"] / 0.24
    ).clip(lower=0.0, upper=100.0)
    out["liquidity_group"] = out["liquidity_group"].astype(str).str.lower()
    out["income_group"] = out["income_group"].astype(str).str.lower()
    return out.sort_values("type_id").reset_index(drop=True)


def run_demand_economy(
    households: pd.DataFrame,
    scenarios: Iterable[DemandScenario],
    client: DemandEconomyClient,
    *,
    period_count: int = 100,
    feedback_mode: str = "closed_loop",
    behavior_policy_profile: dict[str, Any] | None = None,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    period_overrides: dict[int, dict[str, Any]] | None = None,
    initial_environment_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"Unsupported feedback mode: {feedback_mode}")
    periods_per_year = _validated_periods_per_year(periods_per_year)
    normalized_period_overrides = _normalize_period_overrides(period_overrides)
    normalized_initial_environment = _normalize_initial_environment_override(initial_environment_override)
    households = normalize_demand_households(households)
    source = client.source
    initial = _initial_household_states(households, source=source, variant=client.variant, periods_per_year=periods_per_year)
    belief_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        household_states = initial.to_dict(orient="records")
        env = {
            **_initial_environment(households, periods_per_year=periods_per_year),
            **normalized_initial_environment,
        }
        for period_index in range(int(period_count)):
            period_state = _period_state(
                env,
                scenario,
                period_index,
                periods_per_year=periods_per_year,
                period_override=normalized_period_overrides.get(period_index),
            )
            panel = client.belief_panel(scenario, period_state, household_states)
            prompt_rows.append(
                {
                    "source": source,
                    "variant": client.variant,
                    "scenario_id": scenario.scenario_id,
                    "period_id": period_state["period_id"],
                    "period_index": int(period_index),
                    "prompt_payload": _prompt_payload_for_variant(client.variant, scenario, period_state, household_states),
                }
            )
            period_beliefs = _belief_rows(panel, household_states, period_state, source=source, variant=client.variant)
            belief_rows.extend(period_beliefs)
            realized = _realize_household_period(
                households,
                household_states,
                panel,
                period_state,
                source=source,
                variant=client.variant,
                behavior_policy_profile=behavior_policy_profile,
                periods_per_year=periods_per_year,
            )
            decision_rows.extend(realized)
            aggregate = _aggregate_period(realized, scenario, period_state, source=source, variant=client.variant)
            accounting_rows.extend(_accounting_rows(realized, aggregate))
            next_household_states = _next_household_states(
                realized,
                households,
                aggregate,
                periods_per_year=periods_per_year,
            )
            next_env = _next_environment(env, aggregate, scenario, feedback_mode=feedback_mode)
            aggregate.update(
                {
                    "next_output_gap_pct": float(next_env["output_gap_pct"]),
                    "next_employment_rate": float(next_env["employment_rate"]),
                    "next_inflation_rate": float(next_env["inflation_rate"]),
                    "next_policy_rate": float(next_env["policy_rate"]),
                    "next_aggregate_income": _weighted(next_household_states, "labor_income"),
                }
            )
            period_rows.append(aggregate)
            household_states = next_household_states
            env = next_env
    return (
        initial,
        pd.DataFrame(belief_rows),
        pd.DataFrame(decision_rows),
        pd.DataFrame(period_rows),
        pd.DataFrame(accounting_rows),
        prompt_rows,
    )


def belief_module_prompt(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
    *,
    variant: str = "llm_belief",
) -> str:
    payload = belief_module_prompt_payload(scenario, period_state, household_states, variant=variant)
    return f"""
HANK-lite belief module card:
{json.dumps(payload, indent=2, sort_keys=True)}

Return exactly this JSON shape:
{{
  "prompt_version": "{DEMAND_ECONOMY_PROMPT_VERSION}",
  "beliefs": [
    {{
      "type_id": "one supplied type_id",
      "expected_inflation_next_period": 2.0,
      "expected_income_growth_next_period": 1.0,
      "perceived_job_loss_probability": 5.0,
      "expected_unemployment_higher_probability_next_period": 35.0,
      "confidence_index": 50.0,
      "precautionary_saving_score": 5.0,
      "attention_weight_prices": 0.5,
      "attention_weight_jobs": 0.5,
      "attention_weight_rates": 0.5,
      "reason_codes": ["prices", "jobs"],
      "causal_path": ["macro signal", "belief update", "saving motive"]
    }}
  ]
}}

Return one belief object for every supplied household type. Do not choose consumption or saving dollars.
Do not include markdown.
""".strip()


def belief_module_prompt_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
    *,
    variant: str = "llm_belief",
) -> dict[str, Any]:
    has_as_of_history = bool(period_state.get("origin_visible_macro_history"))
    information_label = "origin-visible as-of macro history" if has_as_of_history else "abstract current environment"
    return {
        "prompt_version": DEMAND_ECONOMY_PROMPT_VERSION,
        "variant": variant,
        "task": (
            f"Predict the beliefs of survey-seeded household cells from the supplied {information_label}. "
            "The structural code, not you, will choose consumption, saving, budgets, and aggregation."
        ),
        "belief_update_rules": [
            "Use each cell's prior_expected_inflation, prior_expected_income_growth, prior_job_loss_probability, and prior_confidence_index as survey-style anchors.",
            "Update those anchors only from the supplied current environment, origin-visible information if present, and the cell's own balance-sheet/labor-risk state.",
            "Do not collapse household cells to one representative answer; cross-cell heterogeneity is signal.",
            "In a no-shock period near steady state, beliefs should usually remain close to the cell's survey-style priors.",
            "Under normal dispersion and feedback conditions, damp one-period noise rather than extrapolating it into a persistent boom or slump.",
            "Under elevated dispersion or elevated macro-feedback conditions, preserve wider cross-cell belief differences and let fragile cells react more strongly.",
            "In an elevated macro-feedback regime, persistently weak output, lower employment, falling confidence, or rising job-risk beliefs should reinforce cautious beliefs; do not mean-revert fragile cells until the supplied current_environment improves.",
            "A current or recently absorbed liquid buffer improvement is balance-sheet information, not bad news. For low-liquidity cells, higher liquid_buffer_months after a cash-flow gain should usually reduce near-term precaution or sustain confidence for several periods unless job-risk or employment signals deteriorate.",
            "Only in normal macro_feedback_regime with no active current shock: when output_gap_pct is already clearly positive, avoid extrapolating the expansion into steadily higher income expectations or steadily lower precaution; mean-revert toward the cell's survey-style priors.",
            "After a transfer-driven policy response, do not treat policy_rate above neutral by itself as a job-risk or confidence shock when employment, output, and job-risk news are stable; low-liquidity cells can retain part of the recent buffer relief for a few relative periods.",
            "Treat precautionary_saving_score as a within-cell deviation index: 5 means normal precaution for that cell at its supplied prior state.",
            "Do not assign very low precaution merely because a cell has high liquid assets; only move below 5 when current abstract conditions improve versus that cell's own prior state.",
            "When output is only mildly above steady state and policy is responding, expected income growth should mean-revert toward the supplied prior rather than drift upward; do not convert temporary policy response after a cash transfer into a job-security scare by itself.",
            "Low-liquidity, high-job-risk, and low-income cells can rationally carry higher job-risk beliefs and precaution than high-buffer cells.",
        ],
        "contamination_control": (
            "Calendar dates and macro history are frozen at the declared as-of date. No later observations, "
            "first-release targets, revisions, or realized target paths are supplied."
            if has_as_of_history
            else "No calendar dates, named historical episodes, target realized paths, or external data are supplied."
        ),
        "economy": {
            "goods": "one nondurable consumption good",
            "capital": "none",
            "asset_market": "one liquid safe-asset buffer",
            "firms": "output follows aggregate demand with employment feedback",
            "policy": "Taylor-rule short rate with optional abstract shock",
        },
        "current_exogenous_conditions": _prompt_current_conditions(period_state, scenario),
        "current_environment": _prompt_current_environment(period_state),
        "period_id": period_state["period_id"],
        "household_cells": [_household_prompt_row(row) for row in household_states],
        "allowed_type_ids": [row["type_id"] for row in household_states],
        "required_response": {
            "beliefs": [
                {
                    "type_id": "one supplied type_id",
                    "expected_inflation_next_period": "percent, not decimal",
                    "expected_income_growth_next_period": "percent, not decimal",
                    "perceived_job_loss_probability": "percent probability, 0 to 40",
                    "confidence_index": "0 to 100",
                    "precautionary_saving_score": "0 to 10",
                    "attention_weight_prices": "0 to 1",
                    "attention_weight_jobs": "0 to 1",
                    "attention_weight_rates": "0 to 1",
                    "reason_codes": "short list from prices/jobs/income/rates/liquidity/confidence",
                    "causal_path": "compact causal chain; no historical references",
                }
            ]
        },
    }


def naive_persona_prompt(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> str:
    payload = naive_persona_prompt_payload(scenario, period_state, household_states)
    return f"""
Naive persona baseline card:
{json.dumps(payload, indent=2, sort_keys=True)}

Return exactly this JSON shape:
{{
  "prompt_version": "{NAIVE_PERSONA_PROMPT_VERSION}",
  "actions": [
    {{
      "type_id": "one supplied type_id",
      "desired_consumption_change_pct": 0.0,
      "desired_saving_change_pct": 0.0,
      "confidence": 0.5,
      "reason": "short reason"
    }}
  ]
}}

Return one action for every supplied household type. Do not include markdown.
""".strip()


def naive_persona_prompt_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "prompt_version": NAIVE_PERSONA_PROMPT_VERSION,
        "task": (
            "Naive baseline: use a plain household persona and choose consume-vs-save directly. "
            "Deterministic code will still clamp budgets, but this variant intentionally lacks the belief/structure split."
        ),
        "contamination_control": "No calendar dates, named historical episodes, target realized paths, or external data are supplied.",
        "current_exogenous_conditions": _prompt_current_conditions(period_state, scenario),
        "current_environment": _prompt_current_environment(period_state),
        "period_id": period_state["period_id"],
        "household_personas": [_household_prompt_row(row) for row in household_states],
        "allowed_type_ids": [row["type_id"] for row in household_states],
    }


def fixture_belief_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    beliefs = []
    output_gap = float(period_state["output_gap_pct"])
    policy_gap = float(period_state["policy_rate"]) - NEUTRAL_POLICY_RATE
    inflation_gap = float(period_state["inflation_rate"]) - INFLATION_TARGET
    job_shock = float(period_state["job_risk_shock_pp"])
    for row in household_states:
        high_risk = str(row.get("job_loss_risk_type", "")).lower() == "high"
        low_liquid = str(row.get("liquidity_group", "")).lower() == "low"
        prior_pi = float(row["inflation_expectation_1y"])
        prior_income = float(row["income_growth_expectation_1y"])
        attention_prices = float(row["attention_weight_prices"])
        attention_jobs = float(row["attention_weight_jobs"])
        attention_rates = float(row["attention_weight_rates"])
        expected_pi = 0.70 * prior_pi + 0.30 * float(period_state["inflation_rate"]) + 0.18 * attention_prices * inflation_gap
        expected_pi += 0.025 * max(0.0, output_gap)
        expected_income = 0.84 * prior_income + 0.060 * output_gap - 0.10 * max(0.0, policy_gap)
        job_loss = float(row["baseline_job_loss_probability"])
        job_loss += attention_jobs * (0.45 * max(0.0, -output_gap) + 0.35 * job_shock + 0.20 * max(0.0, policy_gap))
        job_loss *= float(scenario.belief_dispersion_multiplier)
        confidence = float(row["confidence_index"]) + 0.35 * output_gap - 1.4 * (job_loss - float(row["baseline_job_loss_probability"])) - 2.0 * max(0.0, inflation_gap)
        precaution = 4.0 + 0.22 * job_loss + (1.2 if low_liquid else -0.3) + (0.5 if high_risk else -0.2) - 0.035 * confidence
        beliefs.append(
            _belief_payload_row(
                row,
                expected_inflation=expected_pi,
                expected_income=expected_income,
                job_loss=job_loss,
                confidence=confidence,
                precaution=precaution,
                attention_prices=attention_prices,
                attention_jobs=attention_jobs,
                attention_rates=attention_rates,
                reason_codes=_reason_codes(inflation_gap, output_gap, policy_gap, job_shock, low_liquid=low_liquid),
            )
        )
    return {"prompt_version": DEMAND_ECONOMY_PROMPT_VERSION, "beliefs": beliefs}


def adaptive_belief_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    beliefs = []
    output_gap = float(period_state["output_gap_pct"])
    policy_gap = float(period_state["policy_rate"]) - NEUTRAL_POLICY_RATE
    inflation_gap = float(period_state["inflation_rate"]) - INFLATION_TARGET
    job_shock = float(period_state["job_risk_shock_pp"])
    for row in household_states:
        alpha = 0.78 - 0.18 * float(row["attention_weight_prices"])
        expected_pi = alpha * float(row["inflation_expectation_1y"]) + (1.0 - alpha) * float(period_state["inflation_rate"])
        expected_income = 0.88 * float(row["income_growth_expectation_1y"]) + 0.055 * output_gap
        job_loss = float(row["baseline_job_loss_probability"]) + 0.35 * max(0.0, -output_gap) + 0.55 * job_shock
        job_loss += 0.12 * max(0.0, policy_gap)
        confidence = 54.0 + 0.32 * output_gap - 1.0 * job_loss - 1.5 * max(0.0, inflation_gap)
        precaution = 3.2 + 0.18 * job_loss + 0.8 * float(row["precautionary_sensitivity"])
        beliefs.append(
            _belief_payload_row(
                row,
                expected_inflation=expected_pi,
                expected_income=expected_income,
                job_loss=job_loss,
                confidence=confidence,
                precaution=precaution,
                attention_prices=float(row["attention_weight_prices"]),
                attention_jobs=float(row["attention_weight_jobs"]),
                attention_rates=float(row["attention_weight_rates"]),
                reason_codes=_reason_codes(inflation_gap, output_gap, policy_gap, job_shock, low_liquid=str(row["liquidity_group"]) == "low"),
            )
        )
    return {"prompt_version": DEMAND_ECONOMY_PROMPT_VERSION, "beliefs": beliefs}


def representative_belief_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    frame = pd.DataFrame(household_states)
    weights = frame["population_weight"].astype(float)
    representative = {
        "inflation_expectation_1y": _weighted_frame_average(frame, "inflation_expectation_1y"),
        "income_growth_expectation_1y": _weighted_frame_average(frame, "income_growth_expectation_1y"),
        "baseline_job_loss_probability": _weighted_frame_average(frame, "baseline_job_loss_probability"),
        "confidence_index": _weighted_frame_average(frame, "confidence_index"),
        "attention_weight_prices": _weighted_frame_average(frame, "attention_weight_prices"),
        "attention_weight_jobs": _weighted_frame_average(frame, "attention_weight_jobs"),
        "attention_weight_rates": _weighted_frame_average(frame, "attention_weight_rates"),
        "precautionary_sensitivity": _weighted_frame_average(frame, "precautionary_sensitivity"),
    }
    pseudo_rows = []
    for row in household_states:
        merged = dict(row)
        merged.update(representative)
        pseudo_rows.append(merged)
    payload = adaptive_belief_payload(scenario, period_state, pseudo_rows)
    for belief in payload["beliefs"]:
        belief["reason_codes"] = ["representative_agent"]
        belief["causal_path"] = ["aggregate household", "single belief rule", "structural consumption"]
    return payload


def fixture_naive_persona_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    actions = []
    output_gap = float(period_state["output_gap_pct"])
    policy_gap = float(period_state["policy_rate"]) - NEUTRAL_POLICY_RATE
    transfer_active = float(period_state["transfer_per_household"]) > 0
    job_shock = float(period_state["job_risk_shock_pp"])
    for row in household_states:
        change = 0.25 * output_gap - 1.15 * max(0.0, policy_gap) - 0.30 * job_shock
        if transfer_active:
            change += 6.0
        change = float(np.clip(change, -10.0, 10.0))
        actions.append(
            {
                "type_id": row["type_id"],
                "desired_consumption_change_pct": change,
                "desired_saving_change_pct": -0.45 * change,
                "confidence": 0.55,
                "reason": "naive direct consume-save persona baseline",
            }
        )
    return {"prompt_version": NAIVE_PERSONA_PROMPT_VERSION, "actions": actions}


def normalize_period_payload(household_states: list[dict[str, Any]], data: dict[str, Any], *, variant: str) -> dict[str, Any]:
    if variant == "naive_persona":
        return normalize_naive_persona_payload(household_states, data)
    return normalize_belief_payload(household_states, data)


def normalize_belief_payload(household_states: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    if payload.get("prompt_version") != DEMAND_ECONOMY_PROMPT_VERSION:
        raise LLMUnavailable("Demand economy belief payload has the wrong prompt_version")
    beliefs = payload.get("beliefs")
    if not isinstance(beliefs, list):
        raise LLMUnavailable("Demand economy payload must contain beliefs")
    allowed = {str(row["type_id"]) for row in household_states}
    by_type: dict[str, dict[str, Any]] = {}
    for belief in beliefs:
        if not isinstance(belief, dict):
            raise LLMUnavailable("Each demand economy belief row must be an object")
        type_id = str(belief.get("type_id", ""))
        if type_id not in allowed:
            raise LLMUnavailable(f"Unknown demand economy household type_id: {type_id}")
        if type_id in by_type:
            raise LLMUnavailable(f"Duplicate demand economy household type_id: {type_id}")
        perceived_job_loss = bounded_float(belief, "perceived_job_loss_probability", 0.0, 40.0)
        raw_unemployment_higher = bounded_float(
            belief,
            "expected_unemployment_higher_probability_next_period",
            0.0,
            100.0,
        ) if "expected_unemployment_higher_probability_next_period" in belief else float(np.clip(perceived_job_loss / 0.24, 0.0, 100.0))
        by_type[type_id] = {
            "type_id": type_id,
            "expected_inflation_next_period": bounded_float(belief, "expected_inflation_next_period", -5.0, 15.0),
            "expected_income_growth_next_period": bounded_float(belief, "expected_income_growth_next_period", -12.0, 12.0),
            "perceived_job_loss_probability": perceived_job_loss,
            "expected_unemployment_higher_probability_next_period": raw_unemployment_higher,
            "confidence_index": bounded_float(belief, "confidence_index", 0.0, 100.0),
            "precautionary_saving_score": bounded_float(belief, "precautionary_saving_score", 0.0, 10.0),
            "attention_weight_prices": bounded_float(belief, "attention_weight_prices", 0.0, 1.0),
            "attention_weight_jobs": bounded_float(belief, "attention_weight_jobs", 0.0, 1.0),
            "attention_weight_rates": bounded_float(belief, "attention_weight_rates", 0.0, 1.0),
            "reason_codes": _string_list(belief.get("reason_codes"), limit=6),
            "causal_path": _string_list(belief.get("causal_path"), limit=6),
        }
    missing = allowed - set(by_type)
    if missing:
        raise LLMUnavailable(f"Demand economy belief payload missing household type_ids: {', '.join(sorted(missing))}")
    return {"prompt_version": DEMAND_ECONOMY_PROMPT_VERSION, "beliefs_by_type": by_type, "direct_actions_by_type": {}}


def normalize_naive_persona_payload(household_states: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    if payload.get("prompt_version") != NAIVE_PERSONA_PROMPT_VERSION:
        raise LLMUnavailable("Naive persona payload has the wrong prompt_version")
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise LLMUnavailable("Naive persona payload must contain actions")
    allowed = {str(row["type_id"]) for row in household_states}
    direct: dict[str, dict[str, Any]] = {}
    beliefs: dict[str, dict[str, Any]] = {}
    state_by_type = {str(row["type_id"]): row for row in household_states}
    for action in actions:
        if not isinstance(action, dict):
            raise LLMUnavailable("Each naive persona action must be an object")
        type_id = str(action.get("type_id", ""))
        if type_id not in allowed:
            raise LLMUnavailable(f"Unknown naive persona household type_id: {type_id}")
        if type_id in direct:
            raise LLMUnavailable(f"Duplicate naive persona household type_id: {type_id}")
        direct[type_id] = {
            "type_id": type_id,
            "desired_consumption_change_pct": bounded_float(action, "desired_consumption_change_pct", -25.0, 25.0),
            "desired_saving_change_pct": bounded_float(action, "desired_saving_change_pct", -25.0, 25.0),
            "confidence": bounded_float(action, "confidence", 0.0, 1.0),
            "reason": str(action.get("reason", ""))[:300],
        }
        state = state_by_type[type_id]
        beliefs[type_id] = {
            "type_id": type_id,
            "expected_inflation_next_period": float(state["inflation_expectation_1y"]),
            "expected_income_growth_next_period": float(state["income_growth_expectation_1y"]),
            "perceived_job_loss_probability": float(state["job_loss_probability"]),
            "expected_unemployment_higher_probability_next_period": float(state.get("unemployment_higher_probability_1y", float(state["job_loss_probability"]) / 0.24)),
            "confidence_index": 100.0 * float(direct[type_id]["confidence"]),
            "precautionary_saving_score": float(np.clip(5.0 + direct[type_id]["desired_saving_change_pct"] / 5.0, 0.0, 10.0)),
            "attention_weight_prices": 0.5,
            "attention_weight_jobs": 0.5,
            "attention_weight_rates": 0.5,
            "reason_codes": ["naive_persona"],
            "causal_path": ["persona prompt", "direct consumption", "budget clamp"],
        }
    missing = allowed - set(direct)
    if missing:
        raise LLMUnavailable(f"Naive persona payload missing household type_ids: {', '.join(sorted(missing))}")
    return {"prompt_version": NAIVE_PERSONA_PROMPT_VERSION, "beliefs_by_type": beliefs, "direct_actions_by_type": direct}


def normalize_demand_payload(household_states: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    return normalize_belief_payload(household_states, data)


def apply_belief_calibration_profile(
    panel: dict[str, Any],
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    adjustments = profile.get("demand_adjustments", {})
    if not isinstance(adjustments, dict):
        return panel
    beliefs_by_type = {type_id: dict(row) for type_id, row in panel.get("beliefs_by_type", {}).items()}
    output_gap = float(period_state.get("output_gap_pct", 0.0))
    inflation_gap = float(period_state.get("inflation_rate", INFLATION_TARGET)) - INFLATION_TARGET
    policy_gap = float(period_state.get("policy_rate", NEUTRAL_POLICY_RATE)) - NEUTRAL_POLICY_RATE
    job_shock = float(period_state.get("job_risk_shock_pp", 0.0))
    transfer = float(period_state.get("transfer_per_household", 0.0))
    dispersion = max(1.0, float(getattr(scenario, "belief_dispersion_multiplier", 1.0)))
    feedback_gain = max(0.0, float(getattr(scenario, "feedback_gain", 1.0)))
    rebound_signal = max(0.0, output_gap) + (0.35 if transfer > 0 else 0.0)
    slump_signal = max(0.0, -output_gap) + 0.35 * max(0.0, policy_gap) + 0.25 * max(0.0, job_shock)
    uncertainty_signal = abs(output_gap) + abs(inflation_gap) + 0.5 * max(0.0, job_shock) + 0.4 * (dispersion - 1.0)
    income_rebound_gain = float(adjustments.get("income_rebound_gain", 0.0) or 0.0)
    inflation_attention_gain = float(adjustments.get("inflation_attention_gain", 0.0) or 0.0)
    job_risk_regime_gain = float(adjustments.get("job_risk_regime_gain", 0.0) or 0.0)
    confidence_rebound_gain = float(adjustments.get("confidence_rebound_gain", 0.0) or 0.0)
    precaution_uncertainty_gain = float(adjustments.get("precaution_uncertainty_gain", 0.0) or 0.0)
    state_by_type = {str(row["type_id"]): row for row in household_states}
    for type_id, belief in beliefs_by_type.items():
        state = state_by_type.get(type_id, {})
        low_liquid = str(state.get("liquidity_group", "")).lower() == "low"
        high_risk = str(state.get("job_loss_risk_type", "")).lower() == "high"
        attention_prices = float(belief.get("attention_weight_prices", state.get("attention_weight_prices", 0.5)) or 0.5)
        attention_jobs = float(belief.get("attention_weight_jobs", state.get("attention_weight_jobs", 0.5)) or 0.5)
        precautionary_sensitivity = float(state.get("precautionary_sensitivity", 0.5) or 0.5)
        fragility = 1.0 + (0.35 if low_liquid else 0.0) + (0.25 if high_risk else 0.0)
        income_delta = income_rebound_gain * (0.65 * rebound_signal - 0.85 * slump_signal) * fragility
        inflation_delta = inflation_attention_gain * attention_prices * (inflation_gap + 0.10 * feedback_gain * output_gap)
        job_delta = job_risk_regime_gain * attention_jobs * (slump_signal - 0.30 * rebound_signal) * fragility
        confidence_delta = confidence_rebound_gain * (0.90 * rebound_signal - 1.10 * slump_signal - 0.30 * abs(inflation_gap))
        confidence_delta += 0.08 * transfer / 1000.0 if low_liquid else 0.02 * transfer / 1000.0
        precaution_delta = precaution_uncertainty_gain * uncertainty_signal * precautionary_sensitivity * fragility + 0.035 * job_delta
        precaution_delta -= 0.06 * rebound_signal
        belief["expected_income_growth_next_period"] = float(
            np.clip(float(belief["expected_income_growth_next_period"]) + income_delta, -12.0, 12.0)
        )
        belief["expected_inflation_next_period"] = float(
            np.clip(float(belief["expected_inflation_next_period"]) + inflation_delta, -5.0, 15.0)
        )
        belief["perceived_job_loss_probability"] = float(
            np.clip(float(belief["perceived_job_loss_probability"]) + job_delta, 0.0, 40.0)
        )
        belief["confidence_index"] = float(np.clip(float(belief["confidence_index"]) + confidence_delta, 0.0, 100.0))
        belief["precautionary_saving_score"] = float(
            np.clip(float(belief["precautionary_saving_score"]) + precaution_delta, 0.0, 10.0)
        )
        reason_codes = list(belief.get("reason_codes", []))
        if "calibrated_dynamics" not in reason_codes:
            reason_codes.append("calibrated_dynamics")
        belief["reason_codes"] = reason_codes[:6]
        causal_path = list(belief.get("causal_path", []))
        if "validation calibrated belief dynamics" not in causal_path:
            causal_path.append("validation calibrated belief dynamics")
        belief["causal_path"] = causal_path[:6]
    return {
        "prompt_version": panel.get("prompt_version", DEMAND_ECONOMY_PROMPT_VERSION),
        "beliefs_by_type": beliefs_by_type,
        "direct_actions_by_type": panel.get("direct_actions_by_type", {}),
    }


def fixture_demand_payload(
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    return fixture_belief_payload(scenario, period_state, household_states)


def score_demand_economy_validation(
    periods: pd.DataFrame,
    decisions: pd.DataFrame,
    beliefs: pd.DataFrame | None = None,
    accounting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if accounting is None:
        accounting = beliefs if beliefs is not None else pd.DataFrame()
        beliefs = pd.DataFrame()
    beliefs = beliefs if beliefs is not None else pd.DataFrame()
    accounting = accounting if accounting is not None else pd.DataFrame()
    if periods.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    max_accounting_by_source = (
        accounting.groupby("source")["abs_residual"].max().to_dict()
        if not accounting.empty and "abs_residual" in accounting
        else {}
    )
    for source, source_periods in periods.groupby("source", sort=True):
        baseline = source_periods[source_periods["scenario_id"] == "baseline"].copy()
        transfer = source_periods[source_periods["scenario_id"] == "transfer_shock"].copy()
        rate_hike = source_periods[source_periods["scenario_id"] == "rate_hike"].copy()
        job_risk = source_periods[source_periods["scenario_id"] == "job_risk_shock"].copy()
        feedback = source_periods[source_periods["scenario_id"] == "belief_feedback"].copy()
        max_accounting_residual = float(max_accounting_by_source.get(source, np.inf))
        if not baseline.empty:
            period_n = int(baseline["period_index"].nunique())
            tail_window = min(20, max(8, int(np.ceil(period_n / 2.0))))
            tail = baseline.tail(tail_window)
            steady_tail_required = period_n >= 20
            rows.extend(
                [
                    _metric_row(source, "steady_state_final_output_gap_abs", abs(float(baseline["output_gap_pct"].iloc[-1])), 0.0, 2.50, "Absolute final baseline output gap."),
                    _metric_row(
                        source,
                        "steady_state_tail_output_gap_range",
                        float(tail["output_gap_pct"].max() - tail["output_gap_pct"].min()),
                        0.0,
                        1.50,
                        "Range of baseline output gap over the final window; diagnostic only for short canaries under 20 periods.",
                        required=steady_tail_required,
                    ),
                    _metric_row(
                        source,
                        "steady_state_tail_inflation_range",
                        float(tail["inflation_rate"].max() - tail["inflation_rate"].min()),
                        0.0,
                        1.00,
                        "Range of inflation over the final window; diagnostic only for short canaries under 20 periods.",
                        required=steady_tail_required,
                    ),
                ]
            )
        if not baseline.empty and not transfer.empty:
            joined = transfer.merge(
                baseline[["period_index", "aggregate_consumption"]],
                on="period_index",
                suffixes=("_transfer", "_baseline"),
            )
            impact = joined[joined["period_index"] == 1]
            transfer_amount = float(impact["transfer_per_household"].iloc[0]) if not impact.empty else np.nan
            impact_mpc = (
                float((impact["aggregate_consumption_transfer"] - impact["aggregate_consumption_baseline"]).iloc[0] / transfer_amount)
                if transfer_amount and np.isfinite(transfer_amount)
                else np.nan
            )
            first_four = joined[(joined["period_index"] >= 1) & (joined["period_index"] <= 4)]
            cumulative_mpc = (
                float((first_four["aggregate_consumption_transfer"] - first_four["aggregate_consumption_baseline"]).sum() / transfer_amount)
                if transfer_amount and np.isfinite(transfer_amount)
                else np.nan
            )
            low_mpc, high_mpc = _group_mpcs(decisions, source=source, group_column="liquidity_group", low_label="low", high_label="high")
            low_income_mpc, high_income_mpc = _group_mpcs(decisions, source=source, group_column="income_group", low_label="low", high_label="high")
            rows.extend(
                [
                    _metric_row(source, "transfer_impact_mpc", impact_mpc, 0.20, 0.85, "Aggregate impact MPC after a one-period transfer."),
                    _metric_row(source, "transfer_cumulative_mpc_4p", cumulative_mpc, 0.20, 2.40, "Cumulative four-period transfer MPC."),
                    _metric_row(source, "low_liquidity_impact_mpc", low_mpc, 0.35, 1.00, "Impact MPC for low-liquidity households."),
                    _metric_row(source, "high_liquidity_impact_mpc", high_mpc, 0.03, 0.55, "Impact MPC for high-liquidity households."),
                    _metric_row(source, "liquidity_mpc_gradient", low_mpc - high_mpc, 0.15, 1.00, "Low-liquidity impact MPC minus high-liquidity impact MPC."),
                    _metric_row(source, "income_mpc_gradient", low_income_mpc - high_income_mpc, 0.05, 0.80, "Low-income impact MPC minus high-income impact MPC."),
                ]
            )
        if not baseline.empty and not rate_hike.empty:
            joined = rate_hike.merge(
                baseline[["period_index", "aggregate_consumption", "output_gap_pct", "employment_rate", "inflation_rate"]],
                on="period_index",
                suffixes=("_rate_hike", "_baseline"),
            )
            window = joined[(joined["period_index"] >= 1) & (joined["period_index"] <= 6)]
            mean_consumption = float((window["aggregate_consumption_rate_hike"] - window["aggregate_consumption_baseline"]).mean())
            mean_output = float((window["output_gap_pct_rate_hike"] - window["output_gap_pct_baseline"]).mean())
            mean_employment = float((window["employment_rate_rate_hike"] - window["employment_rate_baseline"]).mean())
            late = joined[(joined["period_index"] >= 4) & (joined["period_index"] <= 8)]
            late_inflation = float((late["inflation_rate_rate_hike"] - late["inflation_rate_baseline"]).mean()) if not late.empty else np.nan
            rows.extend(
                [
                    _metric_row(source, "rate_hike_mean_consumption_delta_6p", mean_consumption, -np.inf, -1e-6, "Mean six-period consumption response to a rate hike."),
                    _metric_row(source, "rate_hike_mean_output_gap_delta_6p", mean_output, -np.inf, -1e-6, "Mean six-period output-gap response to a rate hike."),
                    _metric_row(source, "rate_hike_mean_employment_delta_6p", mean_employment, -np.inf, 1e-6, "Mean six-period employment response to a rate hike."),
                    _metric_row(source, "rate_hike_late_inflation_delta", late_inflation, -np.inf, 0.05, "Later inflation response to a rate hike."),
                ]
            )
        if not baseline.empty and not job_risk.empty:
            joined = job_risk.merge(
                baseline[["period_index", "aggregate_consumption", "aggregate_income"]],
                on="period_index",
                suffixes=("_job_risk", "_baseline"),
            )
            impact = joined[joined["period_index"] == 1]
            consumption_delta = float((impact["aggregate_consumption_job_risk"] - impact["aggregate_consumption_baseline"]).iloc[0]) if not impact.empty else np.nan
            income_delta = float((impact["aggregate_income_job_risk"] - impact["aggregate_income_baseline"]).iloc[0]) if not impact.empty else np.nan
            rows.extend(
                [
                    _metric_row(source, "job_risk_impact_consumption_delta", consumption_delta, -np.inf, -1e-6, "Consumption response when perceived job risk rises before income changes."),
                    _metric_row(source, "job_risk_impact_income_delta_abs", abs(income_delta), 0.0, 1e-6, "Income should not move on impact in a pure perceived-risk shock."),
                ]
            )
        if not baseline.empty and not feedback.empty:
            baseline_rms = float(np.sqrt(np.mean(np.square(baseline["output_gap_pct"].iloc[1:]))))
            feedback_rms = float(np.sqrt(np.mean(np.square(feedback["output_gap_pct"].iloc[1:]))))
            rows.extend(
                [
                    _metric_row(source, "baseline_no_shock_output_gap_rms", baseline_rms, 0.0, np.inf, "Baseline endogenous output-gap root-mean-square movement."),
                    _metric_row(source, "belief_feedback_amplification_ratio", feedback_rms / max(baseline_rms, 1e-6), 1.05, np.inf, "Belief-feedback output-gap movement divided by baseline movement."),
                ]
            )
        if not beliefs.empty:
            source_beliefs = beliefs[(beliefs["source"] == source) & (beliefs["scenario_id"] == "baseline") & (beliefs["period_index"] == 1)]
            dispersion = _weighted_std_frame(source_beliefs, "expected_inflation_next_period", "population_weight") if not source_beliefs.empty else np.nan
            rows.append(_metric_row(source, "belief_inflation_dispersion_p1", dispersion, 0.05, np.inf, "Weighted cross-cell expected-inflation dispersion at period 1."))
        rows.append(
            _metric_row(
                source,
                "max_accounting_abs_residual",
                max_accounting_residual,
                0.0,
                ACCOUNTING_TOLERANCE,
                "Maximum absolute household-budget or goods-market accounting residual.",
            )
        )
    return pd.DataFrame(rows).sort_values(["source", "metric"]).reset_index(drop=True)


def score_demand_belief_targets(initial: pd.DataFrame, beliefs: pd.DataFrame) -> pd.DataFrame:
    if initial.empty or beliefs.empty:
        return pd.DataFrame()
    targets = {
        "expected_inflation_next_period": ("inflation_expectation_1y", 1.0),
        "expected_income_growth_next_period": ("income_growth_expectation_1y", 1.0),
        "perceived_job_loss_probability": ("job_loss_probability", 5.0),
        "confidence_index": ("confidence_index", 10.0),
    }
    initial_columns = ["source", "variant", "type_id", "population_weight", *[target for target, _scale in targets.values()]]
    missing_initial = [column for column in initial_columns if column not in initial]
    if missing_initial:
        return pd.DataFrame()
    period0 = beliefs[beliefs["period_index"] == 0].copy()
    if period0.empty:
        return pd.DataFrame()
    merged = period0.merge(
        initial[initial_columns],
        on=["source", "variant", "type_id"],
        how="inner",
        suffixes=("", "_target"),
    )
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    all_weights: list[float] = []
    for source, group in merged.groupby("source", sort=True):
        source_all_errors: list[float] = []
        source_all_weights: list[float] = []
        for belief_column, (target_column, normalizer) in targets.items():
            if belief_column not in group or target_column not in group:
                continue
            weights = group["population_weight"].astype(float)
            prediction = group[belief_column].astype(float)
            target = group[target_column].astype(float)
            error = prediction - target
            mae = _weighted_array_average(np.abs(error), weights)
            rmse = float(np.sqrt(_weighted_array_average(np.square(error), weights)))
            bias = _weighted_array_average(error, weights)
            normalized_errors = np.abs(error) / float(normalizer)
            normalized_mae = _weighted_array_average(normalized_errors, weights)
            source_all_errors.extend(normalized_errors.tolist())
            source_all_weights.extend(weights.tolist())
            all_errors.extend(normalized_errors.tolist())
            all_weights.extend(weights.tolist())
            rows.append(
                {
                    "source": source,
                    "variant": str(group["variant"].iloc[0]),
                    "period_index": 0,
                    "belief_variable": belief_column,
                    "target_variable": target_column,
                    "target_source": "survey_seed_period0",
                    "n": int(group.shape[0]),
                    "weighted_mae": mae,
                    "weighted_rmse": rmse,
                    "weighted_bias": bias,
                    "weighted_prediction_mean": _weighted_array_average(prediction, weights),
                    "weighted_target_mean": _weighted_array_average(target, weights),
                    "weighted_correlation": _weighted_corr(prediction, target, weights),
                    "normalized_mae": normalized_mae,
                    "score_direction": "lower_is_better",
                    "interpretation": "Period-0 belief prediction error against the household cell's survey-style seed.",
                }
            )
        if source_all_errors:
            source_weights = pd.Series(source_all_weights, dtype=float)
            source_errors = pd.Series(source_all_errors, dtype=float)
            rows.append(
                {
                    "source": source,
                    "variant": str(group["variant"].iloc[0]),
                    "period_index": 0,
                    "belief_variable": "ALL",
                    "target_variable": "survey_seed_period0",
                    "target_source": "survey_seed_period0",
                    "n": int(group.shape[0]),
                    "weighted_mae": _weighted_array_average(source_errors, source_weights),
                    "weighted_rmse": float(np.sqrt(_weighted_array_average(np.square(source_errors), source_weights))),
                    "weighted_bias": np.nan,
                    "weighted_prediction_mean": np.nan,
                    "weighted_target_mean": np.nan,
                    "weighted_correlation": np.nan,
                    "normalized_mae": _weighted_array_average(source_errors, source_weights),
                    "score_direction": "lower_is_better",
                    "interpretation": "Weighted normalized period-0 belief error across inflation, income, job-risk, and confidence seeds.",
                }
            )
    return pd.DataFrame(rows).sort_values(["source", "belief_variable"]).reset_index(drop=True)


def build_ablation_table(
    validation: pd.DataFrame,
    periods: pd.DataFrame,
    decisions: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if periods.empty:
        return pd.DataFrame()
    for source, group in periods.groupby("source", sort=True):
        source_validation = validation[validation["source"] == source]
        variant = str(group["variant"].iloc[0]) if "variant" in group and not group.empty else _variant_from_source(source)
        row = {
            "source": source,
            "variant": variant,
            "expected_role": _variant_expected_role(variant),
            "metric_count": int(source_validation.shape[0]),
            "passed_metric_count": int(source_validation["passed"].astype(bool).sum()) if not source_validation.empty else 0,
            "required_metric_count": int(source_validation["required"].astype(bool).sum()) if "required" in source_validation else int(source_validation.shape[0]),
            "passed_required_metric_count": (
                int((source_validation["required"].astype(bool) & source_validation["passed"].astype(bool)).sum())
                if "required" in source_validation
                else int(source_validation["passed"].astype(bool).sum())
            ),
            "all_metrics_passed": bool(source_validation["passed"].astype(bool).all()) if not source_validation.empty else False,
            "required_metrics_passed": (
                bool(source_validation.loc[source_validation["required"].astype(bool), "passed"].astype(bool).all())
                if "required" in source_validation and not source_validation.empty
                else bool(source_validation["passed"].astype(bool).all()) if not source_validation.empty else False
            ),
            "transfer_impact_mpc": _metric_value(source_validation, "transfer_impact_mpc"),
            "liquidity_mpc_gradient": _metric_value(source_validation, "liquidity_mpc_gradient"),
            "income_mpc_gradient": _metric_value(source_validation, "income_mpc_gradient"),
            "rate_hike_mean_consumption_delta_6p": _metric_value(source_validation, "rate_hike_mean_consumption_delta_6p"),
            "job_risk_impact_consumption_delta": _metric_value(source_validation, "job_risk_impact_consumption_delta"),
            "belief_feedback_amplification_ratio": _metric_value(source_validation, "belief_feedback_amplification_ratio"),
            "belief_inflation_dispersion_p1": _metric_value(source_validation, "belief_inflation_dispersion_p1"),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "source"]).reset_index(drop=True)


def classify_demand_economy_evidence(validation: pd.DataFrame, ablations: pd.DataFrame | None = None, *, mode: str) -> dict[str, Any]:
    if validation.empty:
        return {
            "evidence_verdict": "no_validation_rows",
            "passed": False,
            "passed_metric_count": 0,
            "metric_count": 0,
        }
    ablations = ablations if ablations is not None else pd.DataFrame()
    required = validation["required"].astype(bool) if "required" in validation else pd.Series(True, index=validation.index)
    required_validation = validation[required]
    passed = required_validation["passed"].astype(bool)
    required_variants = {"representative", "adaptive", "llm_belief", "naive_persona"}
    present_variants = set(ablations["variant"].astype(str)) if not ablations.empty and "variant" in ablations else set()
    all_required_present = required_variants.issubset(present_variants)
    if {"variant", "metric"}.issubset(validation.columns):
        llm_rows = validation["variant"].astype(str) == "llm_belief"
        llm_metrics = set(validation.loc[llm_rows, "metric"].astype(str))
    else:
        llm_metrics = set()
    missing_full_lab_metrics = sorted(FULL_LAB_LLM_METRICS - llm_metrics)
    full_lab_metric_surface_present = not missing_full_lab_metrics
    all_passed = bool(passed.all())
    canary_passed = bool(
        all_passed
        and mode == "live"
        and "llm_belief" in present_variants
        and (not all_required_present or not full_lab_metric_surface_present)
    )
    full_lab_passed = bool(all_passed and all_required_present and full_lab_metric_surface_present)
    if full_lab_passed:
        verdict = {
            "fixture": "fixture_hank_lite_belief_lab_ready",
            "raw_replay": "raw_replay_hank_lite_belief_lab_ready",
            "replay": "cache_replay_hank_lite_belief_lab_ready",
            "live": "live_hank_lite_belief_lab_ready",
        }.get(mode, "hank_lite_belief_lab_ready")
    elif all_passed:
        if mode == "live" and "llm_belief" in present_variants:
            verdict = "live_hank_lite_belief_canary_ready"
        elif not full_lab_metric_surface_present:
            verdict = "hank_lite_metrics_pass_but_scenario_incomplete"
        else:
            verdict = "hank_lite_metrics_pass_but_ablation_incomplete"
    else:
        verdict = "hank_lite_belief_lab_needs_work"
    return {
        "evidence_verdict": verdict,
        "passed": bool(full_lab_passed or canary_passed),
        "full_lab_passed": full_lab_passed,
        "canary_passed": canary_passed,
        "passed_required_metric_count": int(passed.sum()),
        "required_metric_count": int(required_validation.shape[0]),
        "passed_metric_count": int(validation["passed"].astype(bool).sum()),
        "metric_count": int(validation.shape[0]),
        "required_variants_present": all_required_present,
        "full_lab_metric_surface_present": full_lab_metric_surface_present,
        "missing_full_lab_metrics": missing_full_lab_metrics,
        "present_variants": sorted(present_variants),
        "failed_required_metrics": required_validation.loc[
            ~passed, ["source", "variant", "metric", "value", "target_low", "target_high"]
        ].to_dict(orient="records"),
        "failed_metrics": validation.loc[
            ~validation["passed"].astype(bool), ["source", "variant", "metric", "value", "target_low", "target_high", "required"]
        ].to_dict(orient="records"),
    }


def build_demand_economy_report(
    manifest: dict[str, Any],
    households: pd.DataFrame,
    periods: pd.DataFrame,
    beliefs: pd.DataFrame,
    validation: pd.DataFrame,
    belief_targets: pd.DataFrame,
    ablations: pd.DataFrame,
    accounting: pd.DataFrame,
) -> str:
    failed = validation[~validation["passed"].astype(bool)] if not validation.empty else pd.DataFrame()
    required_failed = (
        validation[validation["required"].astype(bool) & ~validation["passed"].astype(bool)]
        if not validation.empty and "required" in validation
        else failed
    )
    optional_failed = (
        validation[~validation["required"].astype(bool) & ~validation["passed"].astype(bool)]
        if not validation.empty and "required" in validation
        else pd.DataFrame()
    )
    period_preview = periods.sort_values(["source", "scenario_id", "period_index"]).head(64) if not periods.empty else periods
    accounting_summary = (
        accounting.groupby(["source", "scenario_id"], as_index=False)["abs_residual"].max().sort_values(["source", "scenario_id"])
        if not accounting.empty
        else accounting
    )
    lines = [
        "# HANK-Lite Belief Demand Economy",
        "",
        "## Bottom Line",
        _demand_bottom_line(manifest, required_failed, ablations, belief_targets),
        "",
        "## Setup",
        f"- Belief mode: `{manifest.get('belief_mode')}`",
        f"- Belief calibration profile: `{manifest.get('belief_calibration_profile_id') or 'none'}`",
        f"- Provider/models: `{manifest.get('provider')}` / `{', '.join(manifest.get('models', []))}`",
        f"- Variants: `{', '.join(manifest.get('variants', []))}`",
        f"- Fixture-forced variants: `{', '.join(manifest.get('fixture_variants', [])) or 'none'}`",
        f"- Live calls used: `{manifest.get('live_call_count')}` of cap `{manifest.get('max_live_calls')}`",
        f"- Household cells: `{manifest.get('household_count')}`",
        f"- Scenarios: `{manifest.get('scenario_count')}`",
        f"- Periods per scenario: `{manifest.get('period_count')}`",
        f"- Feedback mode: `{manifest.get('feedback_mode')}`",
        "",
        "## Baseline Comparison",
        _demand_baseline_comparison_table(ablations, belief_targets),
        "",
        "## Ablation Table",
        markdown_table(ablations),
        "",
        "## Survey-Seed Belief Target Scores",
        markdown_table(belief_targets),
        "",
        "## Validation Scores",
        markdown_table(validation),
        "",
        "## Required Failed Metrics",
        markdown_table(required_failed),
        "",
        "## Optional Ablation Misses",
        markdown_table(optional_failed),
        "",
        "## Household Cell Surface",
        markdown_table(households[["type_id", "population_weight", "income_group", "liquidity_group", "job_loss_risk_type", "age_bucket", "annual_income", "liquid_assets", "base_mpc"]].head(48)),
        "",
        "## Period Paths",
        markdown_table(
            period_preview[
                [
                    "source",
                    "variant",
                    "scenario_id",
                    "period_id",
                    "aggregate_consumption",
                    "output_gap_pct",
                    "employment_rate",
                    "inflation_rate",
                    "policy_rate",
                    "transfer_per_household",
                    "job_risk_shock_pp",
                ]
            ]
            if not period_preview.empty
            else period_preview
        ),
        "",
        "## Accounting Audit",
        markdown_table(accounting_summary),
        "",
        "## Interpretation",
        (
            "This runner is a HANK-lite macro lab, not a persona society. The model variants separate "
            "belief formation from structural consumption. LLM or fixture belief modules produce expectations, "
            "confidence, perceived job-loss risk, precautionary-saving scores, and reason codes. Deterministic "
            "code then enforces budgets, converts beliefs into consume-vs-save behavior, aggregates demand, "
            "updates output, employment, inflation, and policy, and audits identities every period."
        ),
        "",
        "## Manifest",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _initial_household_states(
    households: pd.DataFrame,
    *,
    source: str,
    variant: str,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    rows = []
    for _, row in households.iterrows():
        baseline_consumption = _per_period_amount(float(row["baseline_consumption_annual"]), periods_per_year)
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": source,
                "variant": variant,
                "periods_per_year": periods_per_year,
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": float(row["population_weight"]),
                "age_bucket": row["age_bucket"],
                "income_group": row["income_group"],
                "liquidity_group": row["liquidity_group"],
                "job_loss_risk_type": row["job_loss_risk_type"],
                "employment_status": row["employment_status"],
                "annual_income": float(row["annual_income"]),
                "labor_income": _per_period_amount(float(row["annual_income"]), periods_per_year),
                "baseline_consumption": baseline_consumption,
                "liquid_assets": float(row["liquid_assets"]),
                "debt": float(row["debt"]),
                "debt_service_burden": float(row["debt_service_burden"]),
                "base_mpc": float(row["base_mpc"]),
                "base_saving_rate": float(row["base_saving_rate"]),
                "rate_sensitivity": float(row["rate_sensitivity"]),
                "income_sensitivity": float(row["income_sensitivity"]),
                "precautionary_sensitivity": float(row["precautionary_sensitivity"]),
            "baseline_job_loss_probability": float(row["baseline_job_loss_probability"]),
            "job_loss_probability": float(row["baseline_job_loss_probability"]),
            "baseline_unemployment_higher_probability": float(row.get("unemployment_higher_probability_1y", float(row["baseline_job_loss_probability"]) / 0.24)),
            "unemployment_higher_probability_1y": float(row.get("unemployment_higher_probability_1y", float(row["baseline_job_loss_probability"]) / 0.24)),
            "target_buffer_months": float(row["target_buffer_months"]),
                "inflation_expectation_1y": float(row["inflation_expectation_1y"]),
                "income_growth_expectation_1y": float(row["income_growth_expectation_1y"]),
                "confidence_index": float(row["confidence_index"]),
                "attention_weight_prices": float(row["attention_weight_prices"]),
                "attention_weight_jobs": float(row["attention_weight_jobs"]),
                "attention_weight_rates": float(row["attention_weight_rates"]),
                "income_volatility": float(row["income_volatility"]),
                "subsistence_floor_share": float(row["subsistence_floor_share"]),
                "liquid_buffer_months": _buffer_months(float(row["liquid_assets"]), baseline_consumption, periods_per_year=periods_per_year),
                "transfer_buffer_relief": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("type_id").reset_index(drop=True)


def _initial_environment(
    households: pd.DataFrame,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, float]:
    baseline_consumption = float((households["population_weight"] * households["baseline_consumption_annual"] / periods_per_year).sum())
    return {
        "periods_per_year": periods_per_year,
        "baseline_aggregate_consumption": baseline_consumption,
        "aggregate_consumption": baseline_consumption,
        "output_gap_pct": 0.0,
        "employment_rate": STEADY_EMPLOYMENT_RATE,
        "inflation_rate": INFLATION_TARGET,
        "policy_rate": NEUTRAL_POLICY_RATE,
        "aggregate_job_loss_belief": float((households["population_weight"] * households["baseline_job_loss_probability"]).sum()),
        "aggregate_confidence_index": float((households["population_weight"] * households["confidence_index"]).sum()),
        "aggregate_liquid_buffer_months": 3.0,
    }


def _period_state(
    env: dict[str, float],
    scenario: DemandScenario,
    period_index: int,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    period_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rate_shock = (
        float(scenario.rate_shock_pp)
        if scenario.rate_shock_start <= period_index <= scenario.rate_shock_end and scenario.rate_shock_start >= 0
        else 0.0
    )
    job_risk_shock = (
        float(scenario.job_risk_shock_pp)
        if scenario.job_risk_shock_start <= period_index <= scenario.job_risk_shock_end and scenario.job_risk_shock_start >= 0
        else 0.0
    )
    transfer = float(scenario.transfer_amount) if period_index == scenario.transfer_period else 0.0
    state: dict[str, Any] = {
        **env,
        "scenario_id": scenario.scenario_id,
        "period_index": int(period_index),
        "period_id": f"period_{period_index}",
        "periods_per_year": periods_per_year,
        "months_per_period": _months_per_period(periods_per_year),
        "transfer_per_household": transfer,
        "policy_rate_shock_pp": rate_shock,
        "job_risk_shock_pp": job_risk_shock,
        "policy_rate": float(env["policy_rate"]) + rate_shock,
    }
    if period_override:
        state.update(period_override)
        state["supplied_exogenous_conditions"] = dict(period_override)
    else:
        state["supplied_exogenous_conditions"] = {}
    return state


def _realize_household_period(
    households: pd.DataFrame,
    household_states: list[dict[str, Any]],
    panel: dict[str, Any],
    period_state: dict[str, Any],
    *,
    source: str,
    variant: str,
    behavior_policy_profile: dict[str, Any] | None = None,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> list[dict[str, Any]]:
    household_by_type = {str(row["type_id"]): row for _, row in households.iterrows()}
    beliefs = panel["beliefs_by_type"]
    direct_actions = panel.get("direct_actions_by_type", {})
    representative_mpc = _representative_mpc(household_states) if variant == "representative" else None
    rows: list[dict[str, Any]] = []
    for state in household_states:
        type_id = str(state["type_id"])
        static = household_by_type[type_id]
        belief = beliefs[type_id]
        direct = direct_actions.get(type_id)
        labor_income = float(state["labor_income"])
        liquid_before = float(state["liquid_assets"])
        transfer = float(period_state["transfer_per_household"])
        cash_available = liquid_before + labor_income + transfer
        baseline_consumption = float(state["baseline_consumption"])
        current_buffer = _buffer_months(liquid_before, baseline_consumption, periods_per_year=periods_per_year)
        transfer_buffer_relief_before = float(state.get("transfer_buffer_relief", 0.0))
        if direct is not None:
            desired_consumption = baseline_consumption * (1.0 + float(direct["desired_consumption_change_pct"]) / 100.0)
            mpc = float(static["base_mpc"])
            desired_buffer = float(state["target_buffer_months"])
            desired_saving_rate = float(state["base_saving_rate"])
            real_rate = float(period_state["policy_rate"]) - float(belief["expected_inflation_next_period"])
            transfer_consumption_amount = transfer
            transfer_debt_repayment_amount = 0.0
            transfer_liquid_saving_amount = 0.0
            debt_repayment = 0.0
            behavior_policy_mode = "direct_action"
            behavior_policy_type_id = ""
            behavior_policy_consumption_drag = 0.0
            empirical_bridge_annual_growth_deviation_pp = 0.0
            empirical_bridge_period_growth_deviation_pp = 0.0
            empirical_bridge_consumption_delta = 0.0
            empirical_bridge_clipped_inputs_json = "{}"
            income_effect = 0.0
            expected_income_effect = 0.0
            buffer_relief_support = 0.0
            buffer_drag = 0.0
            precaution_drag = 0.0
            rate_drag = 0.0
        else:
            policy = _structural_consumption_policy(
                static,
                state,
                belief,
                period_state,
                representative_mpc=representative_mpc,
                behavior_policy_profile=behavior_policy_profile,
                periods_per_year=periods_per_year,
            )
            desired_consumption = policy["desired_consumption"]
            mpc = policy["effective_mpc"]
            desired_buffer = policy["target_buffer_months"]
            desired_saving_rate = policy["desired_saving_rate"]
            real_rate = policy["real_rate"]
            transfer_consumption_amount = policy["transfer_consumption_amount"]
            transfer_debt_repayment_amount = policy["transfer_debt_repayment_amount"]
            transfer_liquid_saving_amount = policy["transfer_liquid_saving_amount"]
            debt_repayment = policy["debt_repayment"]
            behavior_policy_mode = str(policy["behavior_policy_mode"])
            behavior_policy_type_id = str(policy["behavior_policy_type_id"])
            behavior_policy_consumption_drag = float(policy["behavior_policy_consumption_drag"])
            empirical_bridge_annual_growth_deviation_pp = float(policy.get("empirical_bridge_annual_growth_deviation_pp", 0.0))
            empirical_bridge_period_growth_deviation_pp = float(policy.get("empirical_bridge_period_growth_deviation_pp", 0.0))
            empirical_bridge_consumption_delta = float(policy.get("empirical_bridge_consumption_delta", 0.0))
            empirical_bridge_clipped_inputs_json = json.dumps(policy.get("empirical_bridge_clipped_inputs", {}), sort_keys=True)
            income_effect = float(policy.get("income_effect", 0.0))
            expected_income_effect = float(policy.get("expected_income_effect", 0.0))
            buffer_relief_support = float(policy.get("buffer_relief_support", 0.0))
            buffer_drag = float(policy.get("buffer_drag", 0.0))
            precaution_drag = float(policy.get("precaution_drag", 0.0))
            rate_drag = float(policy.get("rate_drag", 0.0))
        debt_before = float(state["debt"])
        floor_consumption = min(cash_available, float(static["subsistence_floor_share"]) * baseline_consumption)
        max_debt_repayment = max(0.0, min(debt_before, cash_available - floor_consumption))
        debt_repayment = float(np.clip(debt_repayment, 0.0, max_debt_repayment))
        consumption_ceiling = max(0.0, cash_available - debt_repayment)
        floor_consumption = min(floor_consumption, consumption_ceiling)
        consumption = float(np.clip(desired_consumption, floor_consumption, consumption_ceiling))
        saving_flow = labor_income + transfer - consumption - debt_repayment
        liquid_after = liquid_before + saving_flow
        debt_after = max(0.0, debt_before - debt_repayment)
        if transfer > 0:
            transfer_debt_repayment_amount = min(transfer_debt_repayment_amount, debt_repayment)
            transfer_consumption_amount = float(np.clip(transfer_consumption_amount, 0.0, max(0.0, transfer - transfer_debt_repayment_amount)))
            transfer_liquid_saving_amount = max(0.0, transfer - transfer_consumption_amount - transfer_debt_repayment_amount)
        else:
            transfer_consumption_amount = 0.0
            transfer_debt_repayment_amount = 0.0
            transfer_liquid_saving_amount = 0.0
        transfer_buffer_relief_after = 0.55 * transfer_buffer_relief_before + transfer_liquid_saving_amount
        budget_residual = liquid_before + labor_income + transfer - consumption - debt_repayment - liquid_after
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": source,
                "variant": variant,
                "periods_per_year": periods_per_year,
                "scenario_id": period_state["scenario_id"],
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "type_id": type_id,
                "label": state["label"],
                "age_bucket": state["age_bucket"],
                "income_group": state["income_group"],
                "liquidity_group": state["liquidity_group"],
                "job_loss_risk_type": state["job_loss_risk_type"],
                "population_weight": float(state["population_weight"]),
                "labor_income": labor_income,
                "transfer": transfer,
                "cash_available": cash_available,
                "consumption": consumption,
                "desired_consumption": float(desired_consumption),
                "saving_flow": saving_flow,
                "debt_before": debt_before,
                "debt_repayment": debt_repayment,
                "debt_after": debt_after,
                "liquid_assets_before": liquid_before,
                "liquid_assets_after": liquid_after,
                "safe_asset_absorption": saving_flow,
                "liquid_buffer_months_before": current_buffer,
                "liquid_buffer_months_after": _buffer_months(liquid_after, baseline_consumption, periods_per_year=periods_per_year),
                "base_mpc": float(static["base_mpc"]),
                "effective_mpc": mpc,
                "realized_mpc_from_transfer": transfer_consumption_amount / transfer if transfer > 0 else np.nan,
                "transfer_consumption_amount": transfer_consumption_amount,
                "transfer_debt_repayment_amount": transfer_debt_repayment_amount,
                "transfer_liquid_saving_amount": transfer_liquid_saving_amount,
                "transfer_consumption_share": transfer_consumption_amount / transfer if transfer > 0 else np.nan,
                "transfer_debt_repayment_share": transfer_debt_repayment_amount / transfer if transfer > 0 else np.nan,
                "transfer_liquid_saving_share": transfer_liquid_saving_amount / transfer if transfer > 0 else np.nan,
                "transfer_buffer_relief_before": transfer_buffer_relief_before,
                "transfer_buffer_relief_after": transfer_buffer_relief_after,
                "desired_saving_rate": desired_saving_rate,
                "target_buffer_months": desired_buffer,
                "expected_inflation_next_period": float(belief["expected_inflation_next_period"]),
                "expected_income_growth_next_period": float(belief["expected_income_growth_next_period"]),
                "job_loss_probability": float(belief["perceived_job_loss_probability"]),
                "unemployment_higher_probability": float(belief.get("expected_unemployment_higher_probability_next_period", float(belief["perceived_job_loss_probability"]) / 0.24)),
                "confidence_index": float(belief["confidence_index"]),
                "precautionary_saving_score": float(belief["precautionary_saving_score"]),
                "real_rate": real_rate,
                "direct_consumption_baseline": direct is not None,
                "behavior_policy_mode": behavior_policy_mode,
                "behavior_policy_type_id": behavior_policy_type_id,
                "behavior_policy_consumption_drag": behavior_policy_consumption_drag,
                "behavior_transfer_consumption_amount": transfer_consumption_amount,
                "behavior_income_effect": income_effect,
                "behavior_expected_income_effect": expected_income_effect,
                "behavior_empirical_bridge_consumption_delta": empirical_bridge_consumption_delta,
                "behavior_buffer_relief_support": buffer_relief_support,
                "behavior_buffer_drag": buffer_drag,
                "behavior_precaution_drag": precaution_drag,
                "behavior_rate_drag": rate_drag,
                "behavior_schedule_consumption_drag": behavior_policy_consumption_drag,
                "empirical_bridge_annual_growth_deviation_pp": empirical_bridge_annual_growth_deviation_pp,
                "empirical_bridge_period_growth_deviation_pp": empirical_bridge_period_growth_deviation_pp,
                "empirical_bridge_clipped_inputs_json": empirical_bridge_clipped_inputs_json,
                "budget_residual": budget_residual,
                "reason_codes_json": json.dumps(belief["reason_codes"], sort_keys=True),
            }
        )
    return rows


def _structural_consumption_policy(
    static: pd.Series,
    state: dict[str, Any],
    belief: dict[str, Any],
    period_state: dict[str, Any],
    *,
    representative_mpc: float | None,
    behavior_policy_profile: dict[str, Any] | None = None,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    baseline_consumption = float(state["baseline_consumption"])
    liquid_before = float(state["liquid_assets"])
    transfer = float(period_state["transfer_per_household"])
    current_buffer = _buffer_months(liquid_before, baseline_consumption, periods_per_year=periods_per_year)
    job_loss = float(belief["perceived_job_loss_probability"])
    confidence = float(belief["confidence_index"])
    expected_income = float(belief["expected_income_growth_next_period"])
    expected_inflation = float(belief["expected_inflation_next_period"])
    precaution_score = float(belief["precautionary_saving_score"])
    real_rate = float(period_state["policy_rate"]) - expected_inflation
    baseline_job_loss = float(static["baseline_job_loss_probability"])
    baseline_confidence = float(static["confidence_index"])
    baseline_expected_income = float(static["income_growth_expectation_1y"])
    baseline_expected_inflation = float(static["inflation_expectation_1y"])
    neutral_real_rate = NEUTRAL_POLICY_RATE - INFLATION_TARGET
    job_loss_gap = job_loss - baseline_job_loss
    confidence_gap = confidence - baseline_confidence
    income_expectation_gap = expected_income - baseline_expected_income
    inflation_expectation_gap = expected_inflation - baseline_expected_inflation
    real_rate_gap = real_rate - neutral_real_rate
    target_buffer = (
        float(static["target_buffer_months"])
        + 0.070 * job_loss_gap
        - 0.020 * confidence_gap
        + 0.18 * (precaution_score - 5.0)
    )
    target_buffer = float(np.clip(target_buffer, 0.5, 14.0))
    desired_saving_rate = (
        float(static["base_saving_rate"])
        + 0.0060 * job_loss_gap
        + 0.0065 * inflation_expectation_gap
        + 0.0110 * real_rate_gap
        - 0.0100 * income_expectation_gap
        - 0.0018 * confidence_gap
        + 0.010 * (precaution_score - 5.0)
    )
    desired_saving_rate = float(np.clip(desired_saving_rate, -0.10, 0.55))
    mpc = float(static["base_mpc"]) if representative_mpc is None else float(representative_mpc)
    mpc += -0.0060 * job_loss_gap
    mpc += 0.0040 * income_expectation_gap
    mpc += -0.0100 * max(0.0, real_rate_gap)
    mpc += -0.010 * (precaution_score - 5.0)
    mpc = float(np.clip(mpc, 0.03, 0.96))
    labor_income = float(state["labor_income"])
    normal_income = _per_period_amount(float(static["annual_income"]), periods_per_year)
    income_gap = labor_income / max(normal_income, 1e-9) - 1.0
    period_months = _months_per_period(periods_per_year)
    target_buffer_dollars = target_buffer * (baseline_consumption / period_months)
    baseline_target_buffer_dollars = float(static["target_buffer_months"]) * (baseline_consumption / period_months)
    baseline_buffer_gap = max(0.0, baseline_target_buffer_dollars - float(static["liquid_assets"]))
    buffer_gap = max(0.0, target_buffer_dollars - liquid_before)
    buffer_drag = 0.065 * (buffer_gap - baseline_buffer_gap)
    saving_rate_gap = desired_saving_rate - float(static["base_saving_rate"])
    precaution_drag = saving_rate_gap * labor_income * float(static["precautionary_sensitivity"])
    rate_drag = (
        max(0.0, real_rate_gap)
        * float(static["rate_sensitivity"])
        * baseline_consumption
        * 0.060
        * _rate_pass_through_factor(period_state)
    )
    income_effect = income_gap * float(static["income_sensitivity"]) * baseline_consumption
    expected_income_effect = income_expectation_gap * baseline_consumption * 0.012
    drawdown = 0.0
    buffer_relief_support = 0.030 * max(0.0, float(state.get("transfer_buffer_relief", 0.0)))
    empirical_bridge_annual_growth_deviation_pp = 0.0
    empirical_bridge_period_growth_deviation_pp = 0.0
    empirical_bridge_consumption_delta = 0.0
    empirical_bridge_clipped_inputs: dict[str, bool] = {}
    hybrid_profile = (
        behavior_policy_profile
        if behavior_policy_profile is not None
        and str(behavior_policy_profile.get("schema_version")) == DEMAND_HYBRID_BEHAVIOR_POLICY_VERSION
        else None
    )
    state_profile = hybrid_profile.get("state_schedule_profile") if hybrid_profile is not None else behavior_policy_profile
    bridge_profile = hybrid_profile.get("empirical_bridge_profile") if hybrid_profile is not None else behavior_policy_profile
    state_behavior_match = (
        _match_state_behavior_policy(static, state, state_profile, periods_per_year=periods_per_year) if state_profile is not None else None
    )
    behavior_match = (
        _match_behavior_policy(static, state, behavior_policy_profile, periods_per_year=periods_per_year)
        if behavior_policy_profile is not None and state_behavior_match is None and hybrid_profile is None
        else None
    )
    if hybrid_profile is not None and representative_mpc is None:
        fallback_transfer = _transfer_windfall_allocation(
            static,
            state,
            belief,
            target_buffer_months=target_buffer,
            desired_saving_rate=desired_saving_rate,
            real_rate=real_rate,
            representative_mpc=representative_mpc,
            periods_per_year=periods_per_year,
        )
        state_transfer = (
            _state_schedule_transfer_allocation(static, state, state_behavior_match, transfer, periods_per_year=periods_per_year)
            if state_behavior_match is not None
            else fallback_transfer
        )
        state_weight = float(np.clip(float(hybrid_profile.get("state_weight", 1.0)), 0.0, 1.0))
        transfer_allocation = _blend_transfer_allocations(fallback_transfer, state_transfer, state_weight)
        previous_input = BridgeInput(
            inflation_expectation_1y=float(state["inflation_expectation_1y"]),
            expected_real_income_growth=float(state["income_growth_expectation_1y"]),
            unemployment_higher_prob=float(state.get("unemployment_higher_probability_1y", float(state["job_loss_probability"]) / 0.24)),
            income_group=str(state["income_group"]),
            liquid_wealth_group=str(state["liquidity_group"]),
        )
        current_input = BridgeInput(
            inflation_expectation_1y=expected_inflation,
            expected_real_income_growth=expected_income,
            unemployment_higher_prob=float(belief.get("expected_unemployment_higher_probability_next_period", job_loss / 0.24)),
            income_group=str(state["income_group"]),
            liquid_wealth_group=str(state["liquidity_group"]),
        )
        bridge_result = transform_belief_change(bridge_profile, previous_input, current_input)
        empirical_bridge_annual_growth_deviation_pp = float(bridge_result.annual_growth_deviation_pp)
        empirical_bridge_period_growth_deviation_pp = empirical_bridge_annual_growth_deviation_pp / periods_per_year
        empirical_bridge_consumption_delta = baseline_consumption * empirical_bridge_period_growth_deviation_pp / 100.0
        empirical_bridge_clipped_inputs = bridge_result.clipped
        behavior_policy_mode = "empirical_bridge_state_schedule"
        behavior_policy_type_id = str(state_behavior_match["profile_id"]) if state_behavior_match is not None else ""
        behavior_policy_consumption_drag = (
            state_weight
            * _state_schedule_consumption_drag(
                static,
                state,
                belief,
                period_state,
                state_behavior_match,
                baseline_consumption,
            )
            if state_behavior_match is not None
            else 0.0
        )
        buffer_drag = 0.0
        precaution_drag = 0.0
        rate_drag = 0.0
        expected_income_effect = 0.0
    elif (
        bridge_profile is not None
        and str(bridge_profile.get("bridge_spec_version") or bridge_profile.get("schema_version"))
        in SUPPORTED_BRIDGE_SPEC_VERSIONS
        and representative_mpc is None
    ):
        transfer_allocation = _transfer_windfall_allocation(
            static,
            state,
            belief,
            target_buffer_months=target_buffer,
            desired_saving_rate=desired_saving_rate,
            real_rate=real_rate,
            representative_mpc=representative_mpc,
            periods_per_year=periods_per_year,
        )
        previous_input = BridgeInput(
            inflation_expectation_1y=float(state["inflation_expectation_1y"]),
            expected_real_income_growth=float(state["income_growth_expectation_1y"]),
            unemployment_higher_prob=float(state.get("unemployment_higher_probability_1y", float(state["job_loss_probability"]) / 0.24)),
            income_group=str(state["income_group"]),
            liquid_wealth_group=str(state["liquidity_group"]),
        )
        current_input = BridgeInput(
            inflation_expectation_1y=expected_inflation,
            expected_real_income_growth=expected_income,
            unemployment_higher_prob=float(belief.get("expected_unemployment_higher_probability_next_period", job_loss / 0.24)),
            income_group=str(state["income_group"]),
            liquid_wealth_group=str(state["liquidity_group"]),
        )
        bridge_result = transform_belief_change(bridge_profile, previous_input, current_input)
        empirical_bridge_annual_growth_deviation_pp = float(bridge_result.annual_growth_deviation_pp)
        empirical_bridge_period_growth_deviation_pp = empirical_bridge_annual_growth_deviation_pp / periods_per_year
        empirical_bridge_consumption_delta = baseline_consumption * empirical_bridge_period_growth_deviation_pp / 100.0
        empirical_bridge_clipped_inputs = bridge_result.clipped
        behavior_policy_mode = "empirical_bridge"
        behavior_policy_type_id = ""
        behavior_policy_consumption_drag = 0.0
        buffer_drag = 0.0
        precaution_drag = 0.0
        rate_drag = 0.0
        expected_income_effect = 0.0
    elif state_behavior_match is not None and representative_mpc is None:
        transfer_allocation = _state_schedule_transfer_allocation(static, state, state_behavior_match, transfer, periods_per_year=periods_per_year)
        behavior_policy_mode = "state_schedule"
        behavior_policy_type_id = str(state_behavior_match["profile_id"])
        behavior_policy_consumption_drag = _state_schedule_consumption_drag(
            static,
            state,
            belief,
            period_state,
            state_behavior_match,
            baseline_consumption,
        )
        # In state-schedule mode, the LLM-authored policy function owns the
        # belief-to-action stress response. The fixed bridge still handles
        # accounting and realized income, but its belief drags are disabled.
        buffer_drag = 0.0
        precaution_drag = 0.0
        rate_drag = 0.0
        expected_income_effect = 0.0
    elif behavior_match is not None and representative_mpc is None:
        transfer_allocation = _schedule_transfer_allocation(static, state, behavior_match, transfer, periods_per_year=periods_per_year)
        behavior_policy_mode = "schedule"
        behavior_policy_type_id = str(behavior_match["type_id"])
        behavior_policy_consumption_drag = _schedule_consumption_drag(static, state, belief, period_state, behavior_match, baseline_consumption)
    else:
        transfer_allocation = _transfer_windfall_allocation(
            static,
            state,
            belief,
            target_buffer_months=target_buffer,
            desired_saving_rate=desired_saving_rate,
            real_rate=real_rate,
            representative_mpc=representative_mpc,
            periods_per_year=periods_per_year,
        )
        behavior_policy_mode = "fixed_kernel"
        behavior_policy_type_id = ""
        behavior_policy_consumption_drag = 0.0
    transfer_consumption_amount = float(transfer_allocation["consumption_share"] * transfer)
    transfer_debt_repayment_amount = float(min(float(state["debt"]), transfer_allocation["debt_repayment_share"] * transfer))
    transfer_liquid_saving_amount = max(0.0, transfer - transfer_consumption_amount - transfer_debt_repayment_amount)
    desired_consumption = (
        baseline_consumption
        + transfer_consumption_amount
        + income_effect
        + expected_income_effect
        + empirical_bridge_consumption_delta
        + buffer_relief_support
        + drawdown
        - buffer_drag
        - precaution_drag
        - rate_drag
        - behavior_policy_consumption_drag
    )
    return {
        "desired_consumption": float(desired_consumption),
        "effective_mpc": transfer_allocation["consumption_share"] if transfer > 0 else mpc,
        "target_buffer_months": target_buffer,
        "desired_saving_rate": desired_saving_rate,
        "real_rate": real_rate,
        "transfer_consumption_amount": transfer_consumption_amount,
        "transfer_debt_repayment_amount": transfer_debt_repayment_amount,
        "transfer_liquid_saving_amount": transfer_liquid_saving_amount,
        "debt_repayment": transfer_debt_repayment_amount,
        "behavior_policy_mode": behavior_policy_mode,
        "behavior_policy_type_id": behavior_policy_type_id,
        "behavior_policy_consumption_drag": behavior_policy_consumption_drag,
        "income_effect": income_effect,
        "expected_income_effect": expected_income_effect,
        "empirical_bridge_consumption_delta": empirical_bridge_consumption_delta,
        "buffer_relief_support": buffer_relief_support,
        "buffer_drag": buffer_drag,
        "precaution_drag": precaution_drag,
        "rate_drag": rate_drag,
        "empirical_bridge_annual_growth_deviation_pp": empirical_bridge_annual_growth_deviation_pp,
        "empirical_bridge_period_growth_deviation_pp": empirical_bridge_period_growth_deviation_pp,
        "empirical_bridge_clipped_inputs": empirical_bridge_clipped_inputs,
    }


def _match_behavior_policy(
    static: pd.Series,
    state: dict[str, Any],
    profile: dict[str, Any] | None,
    *,
    periods_per_year: float,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    if profile.get("schema_version") == DEMAND_STATE_BEHAVIOR_POLICY_VERSION:
        return None
    type_cells = profile.get("type_cells")
    if not isinstance(type_cells, list) or not type_cells:
        return None
    transfer_policies = profile.get("transfer_policies", {})
    income_loss_policies = profile.get("income_loss_policies", {})
    if not isinstance(transfer_policies, dict) or not isinstance(income_loss_policies, dict):
        return None
    household_features = _policy_matching_features(static, state, periods_per_year=periods_per_year)
    best_cell: dict[str, Any] | None = None
    best_distance = float("inf")
    for cell in type_cells:
        if not isinstance(cell, dict):
            continue
        type_id = str(cell.get("type_id", ""))
        if type_id not in transfer_policies or type_id not in income_loss_policies:
            continue
        cell_features = _policy_matching_features_from_cell(cell, periods_per_year=periods_per_year)
        distance = (
            1.35 * (household_features["log_income"] - cell_features["log_income"]) ** 2
            + 1.00 * (household_features["log_buffer"] - cell_features["log_buffer"]) ** 2
            + 0.55 * (household_features["debt_to_income"] - cell_features["debt_to_income"]) ** 2
            + 0.35 * (household_features["liquidity_low"] - cell_features["liquidity_low"]) ** 2
        )
        if distance < best_distance:
            best_distance = distance
            best_cell = cell
    if best_cell is None:
        return None
    type_id = str(best_cell["type_id"])
    return {
        "type_id": type_id,
        "distance": float(best_distance),
        "type_cell": best_cell,
        "transfer_policy": transfer_policies[type_id],
        "income_loss_policy": income_loss_policies[type_id],
    }


def _match_state_behavior_policy(
    static: pd.Series,
    state: dict[str, Any],
    profile: dict[str, Any] | None,
    *,
    periods_per_year: float,
) -> dict[str, Any] | None:
    if profile is None or profile.get("schema_version") != DEMAND_STATE_BEHAVIOR_POLICY_VERSION:
        return None
    profiles = profile.get("profile_rows")
    policies = profile.get("state_policies")
    if not isinstance(profiles, list) or not isinstance(policies, dict):
        return None
    household_features = _state_policy_matching_features(static, state, periods_per_year=periods_per_year)
    best_profile: dict[str, Any] | None = None
    best_distance = float("inf")
    for row in profiles:
        if not isinstance(row, dict):
            continue
        profile_id = str(row.get("profile_id", ""))
        if profile_id not in policies:
            continue
        row_features = _state_policy_matching_features_from_profile(row)
        distance = (
            1.25 * (household_features["log_income"] - row_features["log_income"]) ** 2
            + 1.10 * (household_features["log_buffer"] - row_features["log_buffer"]) ** 2
            + 0.65 * (household_features["debt_to_income"] - row_features["debt_to_income"]) ** 2
            + 0.50 * (household_features["job_loss_risk"] - row_features["job_loss_risk"]) ** 2
            + 0.35 * (household_features["older"] - row_features["older"]) ** 2
        )
        if distance < best_distance:
            best_distance = distance
            best_profile = row
    if best_profile is None:
        return None
    profile_id = str(best_profile["profile_id"])
    return {
        "profile_id": profile_id,
        "distance": float(best_distance),
        "profile_row": best_profile,
        "state_policy": policies[profile_id],
    }


def _policy_matching_features(static: pd.Series, state: dict[str, Any], *, periods_per_year: float) -> dict[str, float]:
    annual_income = max(float(static["annual_income"]), 1.0)
    baseline_consumption = max(
        float(state.get("baseline_consumption", _per_period_amount(float(static["baseline_consumption_annual"]), periods_per_year))),
        1.0,
    )
    liquid_assets = max(float(state.get("liquid_assets", static["liquid_assets"])), 0.0)
    debt = max(float(state.get("debt", static["debt"])), 0.0)
    buffer_months = _buffer_months(liquid_assets, baseline_consumption, periods_per_year=periods_per_year)
    return {
        "log_income": float(np.log1p(annual_income)),
        "log_buffer": float(np.log1p(buffer_months)),
        "debt_to_income": float(np.clip(debt / annual_income, 0.0, 6.0)),
        "liquidity_low": 1.0 if str(state.get("liquidity_group", static.get("liquidity_group", ""))).lower() == "low" else 0.0,
    }


def _state_policy_matching_features(static: pd.Series, state: dict[str, Any], *, periods_per_year: float) -> dict[str, float]:
    annual_income = max(float(static["annual_income"]), 1.0)
    baseline_consumption = max(
        float(state.get("baseline_consumption", _per_period_amount(float(static["baseline_consumption_annual"]), periods_per_year))),
        1.0,
    )
    liquid_assets = max(float(state.get("liquid_assets", static["liquid_assets"])), 0.0)
    debt = max(float(state.get("debt", static["debt"])), 0.0)
    buffer_months = _buffer_months(liquid_assets, baseline_consumption, periods_per_year=periods_per_year)
    return {
        "log_income": float(np.log1p(annual_income)),
        "log_buffer": float(np.log1p(buffer_months)),
        "debt_to_income": float(np.clip(debt / annual_income, 0.0, 6.0)),
        "job_loss_risk": 1.0 if str(state.get("job_loss_risk_type", static.get("job_loss_risk_type", ""))).lower() == "high" else 0.0,
        "older": 1.0 if str(state.get("age_bucket", static.get("age_bucket", ""))).lower() == "older" else 0.0,
    }


def _state_policy_matching_features_from_profile(row: dict[str, Any]) -> dict[str, float]:
    annual_income = max(float(row.get("annual_income", 1.0)), 1.0)
    buffer_months = max(float(row.get("liquid_buffer_months", 0.0)), 0.0)
    debt_to_income = float(np.clip(float(row.get("debt_to_income", 0.0)), 0.0, 6.0))
    return {
        "log_income": float(np.log1p(annual_income)),
        "log_buffer": float(np.log1p(buffer_months)),
        "debt_to_income": debt_to_income,
        "job_loss_risk": 1.0 if str(row.get("job_loss_risk_type", "")).lower() == "high" else 0.0,
        "older": 1.0 if str(row.get("age_bucket", "")).lower() == "older" else 0.0,
    }


def _policy_matching_features_from_cell(cell: dict[str, Any], *, periods_per_year: float) -> dict[str, float]:
    annual_income = max(float(cell.get("annual_income", 1.0)), 1.0)
    consumption = max(_per_period_amount(float(cell.get("consumption_proxy_annual", annual_income * 0.75)), periods_per_year), 1.0)
    liquid_assets = max(float(cell.get("liquid_assets", 0.0)), 0.0)
    debt = max(float(cell.get("debt", 0.0)), 0.0)
    buffer_months = float(cell.get("liquid_buffer_months", _buffer_months(liquid_assets, consumption, periods_per_year=periods_per_year)))
    return {
        "log_income": float(np.log1p(annual_income)),
        "log_buffer": float(np.log1p(max(0.0, buffer_months))),
        "debt_to_income": float(np.clip(debt / annual_income, 0.0, 6.0)),
        "liquidity_low": 1.0 if buffer_months < 1.5 else 0.0,
    }


def _state_schedule_transfer_allocation(
    static: pd.Series,
    state: dict[str, Any],
    match: dict[str, Any],
    transfer: float,
    *,
    periods_per_year: float,
) -> dict[str, float]:
    if transfer <= 0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    policy = match["state_policy"]
    schedule = policy["transfer_schedule"]
    period_income = max(float(state.get("labor_income", _per_period_amount(float(static["annual_income"]), periods_per_year))), 1.0)
    ratio = float(transfer) / period_income
    spending_share = _interp_schedule(schedule, "ratio", "total_spending_share", ratio, log_x=True)
    debt_share = _interp_schedule(schedule, "ratio", "debt_repayment_share", ratio, log_x=True)
    liquid_share = _interp_schedule(schedule, "ratio", "liquid_saving_share", ratio, log_x=True)
    total = spending_share + debt_share + liquid_share
    if total > 1.0 and total > 0:
        spending_share, debt_share, liquid_share = spending_share / total, debt_share / total, liquid_share / total
    if total <= 0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    return {
        "consumption_share": float(np.clip(spending_share, 0.0, 1.0)),
        "debt_repayment_share": float(np.clip(debt_share, 0.0, 1.0)),
        "liquid_saving_share": float(np.clip(max(0.0, 1.0 - spending_share - debt_share), 0.0, 1.0)),
    }


def _schedule_transfer_allocation(
    static: pd.Series,
    state: dict[str, Any],
    match: dict[str, Any],
    transfer: float,
    *,
    periods_per_year: float,
) -> dict[str, float]:
    if transfer <= 0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    policy = match["transfer_policy"]
    schedule = policy["schedule"]
    monthly_income = max(float(state.get("labor_income", _per_period_amount(float(static["annual_income"]), periods_per_year))) / _months_per_period(periods_per_year) * 0.78, 1.0)
    ratio = float(transfer) / monthly_income
    ratios = np.array([float(point["ratio"]) for point in schedule], dtype=float)
    log_ratio = float(np.log(np.clip(ratio, float(ratios.min()), float(ratios.max()))))

    def interp(field: str) -> float:
        values = np.array([float(point[field]) for point in schedule], dtype=float)
        return float(np.interp(log_ratio, np.log(ratios), values))

    consumption_share = interp("total_spending_share")
    debt_share = interp("debt_repayment_share")
    liquid_share = interp("liquid_saving_share")
    total = consumption_share + debt_share + liquid_share
    if total > 1.0 and total > 0:
        consumption_share, debt_share, liquid_share = consumption_share / total, debt_share / total, liquid_share / total
    if total <= 0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    return {
        "consumption_share": float(np.clip(consumption_share, 0.0, 1.0)),
        "debt_repayment_share": float(np.clip(debt_share, 0.0, 1.0)),
        "liquid_saving_share": float(np.clip(max(0.0, 1.0 - consumption_share - debt_share), 0.0, 1.0)),
    }


def _blend_transfer_allocations(left: dict[str, float], right: dict[str, float], right_weight: float) -> dict[str, float]:
    weight = float(np.clip(float(right_weight), 0.0, 1.0))
    out = {
        key: (1.0 - weight) * float(left.get(key, 0.0)) + weight * float(right.get(key, 0.0))
        for key in ("consumption_share", "debt_repayment_share", "liquid_saving_share")
    }
    total = sum(max(0.0, value) for value in out.values())
    if total <= 0.0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    return {
        "consumption_share": float(np.clip(max(0.0, out["consumption_share"]) / total, 0.0, 1.0)),
        "debt_repayment_share": float(np.clip(max(0.0, out["debt_repayment_share"]) / total, 0.0, 1.0)),
        "liquid_saving_share": float(np.clip(max(0.0, out["liquid_saving_share"]) / total, 0.0, 1.0)),
    }


def _state_schedule_consumption_drag(
    static: pd.Series,
    state: dict[str, Any],
    belief: dict[str, Any],
    period_state: dict[str, Any],
    match: dict[str, Any],
    baseline_consumption: float,
) -> float:
    policy = match["state_policy"]
    baseline_job_loss = float(static["baseline_job_loss_probability"])
    baseline_confidence = float(static["confidence_index"])
    baseline_inflation = float(static["inflation_expectation_1y"])
    neutral_real_rate = NEUTRAL_POLICY_RATE - INFLATION_TARGET
    real_rate = float(period_state["policy_rate"]) - float(belief["expected_inflation_next_period"])
    job_loss_gap = max(0.0, float(belief["perceived_job_loss_probability"]) - baseline_job_loss)
    inflation_gap = max(0.0, float(belief["expected_inflation_next_period"]) - baseline_inflation)
    confidence_drop = max(0.0, baseline_confidence - float(belief["confidence_index"]))
    real_rate_gap = max(0.0, real_rate - neutral_real_rate)
    job_cut = _interp_schedule(policy["job_risk_schedule"], "job_loss_probability_gap_pp", "consumption_cut_share", job_loss_gap)
    inflation_cut = _interp_schedule(policy["inflation_schedule"], "inflation_expectation_gap_pp", "consumption_cut_share", inflation_gap)
    confidence_cut = _interp_schedule(policy["confidence_schedule"], "confidence_drop_points", "consumption_cut_share", confidence_drop)
    rate_cut = _interp_schedule(policy["real_rate_schedule"], "real_rate_gap_pp", "consumption_cut_share", real_rate_gap)
    max_cut = float(np.clip(float(policy.get("max_consumption_cut_share", 0.25)), 0.0, 0.35))
    # Stress channels overlap in real household decisions; using the largest
    # implied cut avoids the over-additive bridge that previous Phase 4 runs exposed.
    cut_share = min(max_cut, max(job_cut, inflation_cut, confidence_cut, rate_cut))
    return float(np.clip(cut_share, 0.0, 0.35) * baseline_consumption)


def _schedule_consumption_drag(
    static: pd.Series,
    state: dict[str, Any],
    belief: dict[str, Any],
    period_state: dict[str, Any],
    match: dict[str, Any],
    baseline_consumption: float,
) -> float:
    policy = match["income_loss_policy"]
    baseline_job_loss = float(static["baseline_job_loss_probability"])
    job_loss_gap = max(0.0, float(belief["perceived_job_loss_probability"]) - baseline_job_loss)
    job_shock = max(0.0, float(period_state.get("job_risk_shock_pp", 0.0)))
    precaution_gap = max(0.0, float(belief["precautionary_saving_score"]) - 5.0) / 5.0
    confidence_gap = max(0.0, float(static["confidence_index"]) - float(belief["confidence_index"])) / 50.0
    onset = float(policy["onset_drop_share"])
    receipt_drift = float(policy["receipt_monthly_drift_share"])
    exhaustion = float(policy["exhaustion_drop_share"])
    if job_shock > 0:
        active_drop = onset + 0.25 * receipt_drift
    else:
        active_drop = 0.55 * onset + 0.20 * exhaustion
    risk_intensity = min(1.0, (job_loss_gap + 0.65 * job_shock) / 18.0)
    drag_share = active_drop * risk_intensity + 0.030 * precaution_gap + 0.020 * confidence_gap
    return float(np.clip(drag_share, 0.0, 0.35) * baseline_consumption)


def _interp_schedule(
    schedule: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    x_value: float,
    *,
    log_x: bool = False,
) -> float:
    points = sorted(
        (
            (float(point[x_field]), float(point[y_field]))
            for point in schedule
            if isinstance(point, dict) and x_field in point and y_field in point
        ),
        key=lambda pair: pair[0],
    )
    if not points:
        return 0.0
    xs = np.array([max(pair[0], 1e-9) for pair in points], dtype=float)
    ys = np.array([pair[1] for pair in points], dtype=float)
    x = float(np.clip(max(float(x_value), 1e-9), float(xs.min()), float(xs.max())))
    if log_x:
        return float(np.interp(np.log(x), np.log(xs), ys))
    return float(np.interp(x, xs, ys))


def _transfer_windfall_allocation(
    static: pd.Series,
    state: dict[str, Any],
    belief: dict[str, Any],
    *,
    target_buffer_months: float,
    desired_saving_rate: float,
    real_rate: float,
    representative_mpc: float | None,
    periods_per_year: float,
) -> dict[str, float]:
    if representative_mpc is not None:
        consumption_share = float(np.clip(representative_mpc, 0.08, 0.55))
    else:
        base_mpc = float(static["base_mpc"])
        low_liquid = str(state["liquidity_group"]).lower() == "low"
        low_income = str(state["income_group"]).lower() == "low"
        high_income = str(state["income_group"]).lower() == "high"
        high_risk = str(state["job_loss_risk_type"]).lower() == "high"
        baseline_confidence = float(static["confidence_index"])
        confidence_gap = float(belief["confidence_index"]) - baseline_confidence
        precaution_gap = float(belief["precautionary_saving_score"]) - 5.0
        neutral_real_rate = NEUTRAL_POLICY_RATE - INFLATION_TARGET
        real_rate_gap = float(real_rate) - neutral_real_rate
        current_buffer = _buffer_months(float(state["liquid_assets"]), float(state["baseline_consumption"]), periods_per_year=periods_per_year)
        buffer_gap_months = max(0.0, float(target_buffer_months) - current_buffer)
        if low_liquid:
            consumption_share = 0.56 + 0.35 * (base_mpc - 0.72)
            consumption_share += 0.025 if low_income else -0.025 if high_income else 0.0
            consumption_share += 0.015 if high_risk else 0.0
            consumption_share += 0.0015 * confidence_gap
            consumption_share -= 0.012 * max(0.0, precaution_gap)
            consumption_share -= 0.010 * max(0.0, real_rate_gap)
            consumption_share -= 0.008 * min(buffer_gap_months, 4.0)
            consumption_share = float(np.clip(consumption_share, 0.42, 0.68))
        else:
            consumption_share = 0.108 + 0.32 * (base_mpc - 0.18)
            consumption_share += 0.020 if low_income else -0.015 if high_income else 0.0
            consumption_share += 0.010 if high_risk else 0.0
            consumption_share += 0.0010 * confidence_gap
            consumption_share -= 0.010 * max(0.0, precaution_gap)
            consumption_share -= 0.010 * max(0.0, real_rate_gap)
            consumption_share -= 0.006 * min(buffer_gap_months, 4.0)
            consumption_share = float(np.clip(consumption_share, 0.05, 0.15))
    non_consumed_share = max(0.0, 1.0 - consumption_share)
    debt_service = float(static["debt_service_burden"])
    debt_share = 0.32 + 0.50 * (debt_service - 0.12)
    income_group = str(state["income_group"]).lower()
    liquidity_group = str(state["liquidity_group"]).lower()
    job_risk = str(state["job_loss_risk_type"]).lower()
    debt_share += 0.030 if income_group == "low" else -0.020 if income_group == "high" else 0.0
    debt_share += 0.025 if job_risk == "high" else 0.0
    debt_share -= 0.015 if liquidity_group == "high" else 0.0
    debt_share -= 0.035 * max(0.0, float(desired_saving_rate) - float(static["base_saving_rate"]))
    debt_share = float(np.clip(debt_share, 0.22, 0.42))
    debt_share = min(debt_share, non_consumed_share)
    liquid_saving_share = max(0.0, 1.0 - consumption_share - debt_share)
    total = consumption_share + debt_share + liquid_saving_share
    if total <= 0:
        return {"consumption_share": 0.0, "debt_repayment_share": 0.0, "liquid_saving_share": 0.0}
    return {
        "consumption_share": float(consumption_share / total),
        "debt_repayment_share": float(debt_share / total),
        "liquid_saving_share": float(liquid_saving_share / total),
    }


def _rate_pass_through_factor(period_state: dict[str, Any]) -> float:
    if float(period_state.get("policy_rate_shock_pp", 0.0)) <= 0.0:
        return 1.0
    period_index = int(period_state.get("period_index", 0))
    if period_index <= 1:
        return 0.45
    if period_index == 2:
        return 0.75
    return 1.0


def _aggregate_period(
    realized: list[dict[str, Any]],
    scenario: DemandScenario,
    period_state: dict[str, Any],
    *,
    source: str,
    variant: str,
) -> dict[str, Any]:
    aggregate_consumption = _weighted(realized, "consumption")
    aggregate_income = _weighted(realized, "labor_income")
    aggregate_transfer = _weighted(realized, "transfer")
    aggregate_saving = _weighted(realized, "saving_flow")
    aggregate_debt_repayment = _weighted(realized, "debt_repayment")
    aggregate_liquid_assets = _weighted(realized, "liquid_assets_after")
    aggregate_debt = _weighted(realized, "debt_after")
    aggregate_job_loss = _weighted(realized, "job_loss_probability")
    aggregate_confidence = _weighted(realized, "confidence_index")
    aggregate_buffer = _weighted(realized, "liquid_buffer_months_after")
    aggregate_transfer_consumption = _weighted(realized, "transfer_consumption_amount")
    aggregate_transfer_debt_repayment = _weighted(realized, "transfer_debt_repayment_amount")
    aggregate_transfer_liquid_saving = _weighted(realized, "transfer_liquid_saving_amount")
    aggregate_income_effect = _weighted(realized, "behavior_income_effect")
    aggregate_expected_income_effect = _weighted(realized, "behavior_expected_income_effect")
    aggregate_bridge_delta = _weighted(realized, "behavior_empirical_bridge_consumption_delta")
    aggregate_buffer_relief = _weighted(realized, "behavior_buffer_relief_support")
    aggregate_buffer_drag = _weighted(realized, "behavior_buffer_drag")
    aggregate_precaution_drag = _weighted(realized, "behavior_precaution_drag")
    aggregate_rate_drag = _weighted(realized, "behavior_rate_drag")
    aggregate_schedule_drag = _weighted(realized, "behavior_schedule_consumption_drag")
    baseline = float(period_state["baseline_aggregate_consumption"])
    output = aggregate_consumption
    output_gap = 100.0 * (output / max(baseline, 1e-9) - 1.0)
    return {
        "schema_version": DEMAND_ECONOMY_VERSION,
        "source": source,
        "variant": variant,
        "periods_per_year": float(period_state["periods_per_year"]),
        "scenario_id": scenario.scenario_id,
        "scenario_label": scenario.label,
        "period_id": period_state["period_id"],
        "period_index": int(period_state["period_index"]),
        "aggregate_consumption": aggregate_consumption,
        "aggregate_income": aggregate_income,
        "aggregate_transfer": aggregate_transfer,
        "aggregate_saving": aggregate_saving,
        "aggregate_debt_repayment": aggregate_debt_repayment,
        "safe_asset_absorption": aggregate_saving,
        "aggregate_liquid_assets": aggregate_liquid_assets,
        "aggregate_debt": aggregate_debt,
        "aggregate_job_loss_belief": aggregate_job_loss,
        "aggregate_confidence_index": aggregate_confidence,
        "aggregate_liquid_buffer_months": aggregate_buffer,
        "aggregate_transfer_consumption": aggregate_transfer_consumption,
        "aggregate_transfer_debt_repayment": aggregate_transfer_debt_repayment,
        "aggregate_transfer_liquid_saving": aggregate_transfer_liquid_saving,
        "aggregate_behavior_income_effect": aggregate_income_effect,
        "aggregate_behavior_expected_income_effect": aggregate_expected_income_effect,
        "aggregate_behavior_empirical_bridge_consumption_delta": aggregate_bridge_delta,
        "aggregate_behavior_buffer_relief_support": aggregate_buffer_relief,
        "aggregate_behavior_buffer_drag": aggregate_buffer_drag,
        "aggregate_behavior_precaution_drag": aggregate_precaution_drag,
        "aggregate_behavior_rate_drag": aggregate_rate_drag,
        "aggregate_behavior_schedule_consumption_drag": aggregate_schedule_drag,
        "aggregate_transfer_consumption_share": aggregate_transfer_consumption / aggregate_transfer if aggregate_transfer > 0 else np.nan,
        "aggregate_transfer_debt_repayment_share": aggregate_transfer_debt_repayment / aggregate_transfer if aggregate_transfer > 0 else np.nan,
        "aggregate_transfer_liquid_saving_share": aggregate_transfer_liquid_saving / aggregate_transfer if aggregate_transfer > 0 else np.nan,
        "output": output,
        "output_gap_pct": output_gap,
        "employment_rate": float(period_state["employment_rate"]),
        "inflation_rate": float(period_state["inflation_rate"]),
        "policy_rate": float(period_state["policy_rate"]),
        "policy_rate_shock_pp": float(period_state["policy_rate_shock_pp"]),
        "job_risk_shock_pp": float(period_state["job_risk_shock_pp"]),
        "transfer_per_household": float(period_state["transfer_per_household"]),
        "supplied_exogenous_conditions_json": json.dumps(period_state.get("supplied_exogenous_conditions", {}), sort_keys=True),
        "goods_market_residual": output - aggregate_consumption,
    }


def _next_household_states(
    realized: list[dict[str, Any]],
    households: pd.DataFrame,
    aggregate: dict[str, Any],
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> list[dict[str, Any]]:
    static_by_type = {str(row["type_id"]): row for _, row in households.iterrows()}
    employment_factor = float(aggregate["employment_rate"]) / STEADY_EMPLOYMENT_RATE
    inflation_gap = float(aggregate["inflation_rate"]) - INFLATION_TARGET
    output_gap = float(aggregate["output_gap_pct"])
    rows: list[dict[str, Any]] = []
    job_risk_persistence = _quarterly_persistence_for_frequency(0.78, periods_per_year)
    job_risk_adjustment = _quarterly_flow_for_frequency(0.08, 0.78, periods_per_year)
    inflation_belief_persistence = _quarterly_persistence_for_frequency(0.72, periods_per_year)
    inflation_feedback = _quarterly_flow_for_frequency(0.10, 0.72, periods_per_year)
    income_belief_persistence = _quarterly_persistence_for_frequency(0.68, periods_per_year)
    income_feedback = _quarterly_flow_for_frequency(0.012, 0.68, periods_per_year)
    confidence_persistence = _quarterly_persistence_for_frequency(0.70, periods_per_year)
    confidence_feedback = _quarterly_flow_for_frequency(0.040, 0.70, periods_per_year)
    for row in realized:
        static = static_by_type[str(row["type_id"])]
        annual_income = float(static["annual_income"])
        labor_income = _per_period_amount(annual_income, periods_per_year) * np.clip(employment_factor * (1.0 + 0.0015 * output_gap), 0.70, 1.25)
        baseline_consumption = _per_period_amount(float(static["baseline_consumption_annual"]), periods_per_year)
        job_loss = float(row["job_loss_probability"])
        job_loss = float(
            np.clip(
                job_risk_persistence * job_loss
                + (1.0 - job_risk_persistence) * float(static["baseline_job_loss_probability"])
                + job_risk_adjustment * max(0.0, -output_gap),
                0.5,
                35.0,
            )
        )
        unemployment_higher = float(
            np.clip(
                job_risk_persistence * float(row.get("unemployment_higher_probability", job_loss / 0.24))
                + (1.0 - job_risk_persistence)
                * float(static.get("unemployment_higher_probability_1y", float(static["baseline_job_loss_probability"]) / 0.24))
                + job_risk_adjustment * max(0.0, -output_gap) / 0.24,
                0.0,
                100.0,
            )
        )
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": row["source"],
                "variant": row["variant"],
                "periods_per_year": periods_per_year,
                "type_id": row["type_id"],
                "label": row["label"],
                "population_weight": float(row["population_weight"]),
                "age_bucket": row["age_bucket"],
                "income_group": row["income_group"],
                "liquidity_group": row["liquidity_group"],
                "job_loss_risk_type": row["job_loss_risk_type"],
                "employment_status": str(static.get("employment_status", "unknown")),
                "annual_income": annual_income,
                "labor_income": float(labor_income),
                "baseline_consumption": baseline_consumption,
                "liquid_assets": float(row["liquid_assets_after"]),
                "debt": float(row.get("debt_after", static["debt"])),
                "debt_service_burden": float(static["debt_service_burden"]),
                "base_mpc": float(static["base_mpc"]),
                "base_saving_rate": float(static["base_saving_rate"]),
                "rate_sensitivity": float(static["rate_sensitivity"]),
                "income_sensitivity": float(static["income_sensitivity"]),
                "precautionary_sensitivity": float(static["precautionary_sensitivity"]),
                "baseline_job_loss_probability": float(static["baseline_job_loss_probability"]),
                "job_loss_probability": job_loss,
                "baseline_unemployment_higher_probability": float(static.get("unemployment_higher_probability_1y", float(static["baseline_job_loss_probability"]) / 0.24)),
                "unemployment_higher_probability_1y": unemployment_higher,
                "target_buffer_months": float(static["target_buffer_months"]),
                "inflation_expectation_1y": float(
                    np.clip(
                        inflation_belief_persistence * float(row["expected_inflation_next_period"])
                        + (1.0 - inflation_belief_persistence) * float(static["inflation_expectation_1y"])
                        + inflation_feedback * inflation_gap,
                        -2.0,
                        12.0,
                    )
                ),
                "income_growth_expectation_1y": float(
                    np.clip(
                        income_belief_persistence * float(row["expected_income_growth_next_period"])
                        + (1.0 - income_belief_persistence) * float(static["income_growth_expectation_1y"])
                        + income_feedback * output_gap,
                        -8.0,
                        8.0,
                    )
                ),
                "confidence_index": float(
                    np.clip(
                        confidence_persistence * float(row["confidence_index"])
                        + (1.0 - confidence_persistence) * float(static["confidence_index"])
                        + confidence_feedback * output_gap,
                        0.0,
                        100.0,
                    )
                ),
                "attention_weight_prices": float(static["attention_weight_prices"]),
                "attention_weight_jobs": float(static["attention_weight_jobs"]),
                "attention_weight_rates": float(static["attention_weight_rates"]),
                "income_volatility": float(static["income_volatility"]),
                "subsistence_floor_share": float(static["subsistence_floor_share"]),
                "liquid_buffer_months": _buffer_months(float(row["liquid_assets_after"]), baseline_consumption, periods_per_year=periods_per_year),
                "transfer_buffer_relief": float(row.get("transfer_buffer_relief_after", 0.0)),
            }
        )
    return rows


def _next_environment(
    env: dict[str, float],
    aggregate: dict[str, Any],
    scenario: DemandScenario,
    *,
    feedback_mode: str,
) -> dict[str, float]:
    if feedback_mode == "none":
        return {
            **env,
            "aggregate_consumption": float(aggregate["aggregate_consumption"]),
            "aggregate_job_loss_belief": float(aggregate["aggregate_job_loss_belief"]),
            "aggregate_confidence_index": float(aggregate["aggregate_confidence_index"]),
            "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
        }
    periods_per_year = _validated_periods_per_year(env.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR))
    output_gap = float(aggregate["output_gap_pct"])
    gain = float(scenario.feedback_gain)
    employment_persistence = _quarterly_persistence_for_frequency(0.82, periods_per_year)
    inflation_persistence = _quarterly_persistence_for_frequency(0.64, periods_per_year)
    inflation_output_feedback = _quarterly_flow_for_frequency(0.024, 0.64, periods_per_year)
    next_employment = float(
        np.clip(
            employment_persistence * float(env["employment_rate"])
            + (1.0 - employment_persistence) * (STEADY_EMPLOYMENT_RATE + 0.0010 * gain * output_gap),
            0.82,
            0.99,
        )
    )
    next_inflation = float(
        np.clip(
            inflation_persistence * float(env["inflation_rate"])
            + (1.0 - inflation_persistence) * INFLATION_TARGET
            + inflation_output_feedback * gain * output_gap,
            -2.0,
            12.0,
        )
    )
    next_policy = float(
        np.clip(
            NEUTRAL_POLICY_RATE + 1.35 * (next_inflation - INFLATION_TARGET) + 0.18 * output_gap,
            0.0,
            12.0,
        )
    )
    return {
        **env,
        "aggregate_consumption": float(aggregate["aggregate_consumption"]),
        "output_gap_pct": output_gap,
        "employment_rate": next_employment,
        "inflation_rate": next_inflation,
        "policy_rate": next_policy,
        "aggregate_job_loss_belief": float(aggregate["aggregate_job_loss_belief"]),
        "aggregate_confidence_index": float(aggregate["aggregate_confidence_index"]),
        "aggregate_liquid_buffer_months": float(aggregate["aggregate_liquid_buffer_months"]),
    }


def _accounting_rows(realized: list[dict[str, Any]], aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "source": row["source"],
            "variant": row["variant"],
            "scenario_id": row["scenario_id"],
            "period_id": row["period_id"],
            "period_index": int(row["period_index"]),
            "unit": row["type_id"],
            "identity": "household_cash_budget",
            "residual": float(row["budget_residual"]),
            "abs_residual": abs(float(row["budget_residual"])),
            "passed": abs(float(row["budget_residual"])) <= ACCOUNTING_TOLERANCE,
        }
        for row in realized
    ]
    rows.extend(
        {
            "source": row["source"],
            "variant": row["variant"],
            "scenario_id": row["scenario_id"],
            "period_id": row["period_id"],
            "period_index": int(row["period_index"]),
            "unit": row["type_id"],
            "identity": "household_debt_stock",
            "residual": float(row["debt_before"] - row["debt_repayment"] - row["debt_after"]),
            "abs_residual": abs(float(row["debt_before"] - row["debt_repayment"] - row["debt_after"])),
            "passed": abs(float(row["debt_before"] - row["debt_repayment"] - row["debt_after"])) <= ACCOUNTING_TOLERANCE,
        }
        for row in realized
    )
    residual = float(aggregate["goods_market_residual"])
    rows.append(
        {
            "source": aggregate["source"],
            "variant": aggregate["variant"],
            "scenario_id": aggregate["scenario_id"],
            "period_id": aggregate["period_id"],
            "period_index": int(aggregate["period_index"]),
            "unit": "aggregate",
            "identity": "one_good_output_equals_consumption",
            "residual": residual,
            "abs_residual": abs(residual),
            "passed": abs(residual) <= ACCOUNTING_TOLERANCE,
        }
    )
    return rows


def _belief_rows(
    panel: dict[str, Any],
    household_states: list[dict[str, Any]],
    period_state: dict[str, Any],
    *,
    source: str,
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states = {str(row["type_id"]): row for row in household_states}
    for type_id, belief in panel["beliefs_by_type"].items():
        state = states[type_id]
        rows.append(
            {
                "schema_version": DEMAND_ECONOMY_VERSION,
                "source": source,
                "variant": variant,
                "scenario_id": period_state["scenario_id"],
                "period_id": period_state["period_id"],
                "period_index": int(period_state["period_index"]),
                "type_id": type_id,
                "population_weight": float(state["population_weight"]),
                "income_group": state["income_group"],
                "liquidity_group": state["liquidity_group"],
                "job_loss_risk_type": state["job_loss_risk_type"],
                "age_bucket": state["age_bucket"],
                "expected_inflation_next_period": float(belief["expected_inflation_next_period"]),
                "expected_income_growth_next_period": float(belief["expected_income_growth_next_period"]),
                "perceived_job_loss_probability": float(belief["perceived_job_loss_probability"]),
                "expected_unemployment_higher_probability_next_period": float(
                    belief.get("expected_unemployment_higher_probability_next_period", float(belief["perceived_job_loss_probability"]) / 0.24)
                ),
                "confidence_index": float(belief["confidence_index"]),
                "precautionary_saving_score": float(belief["precautionary_saving_score"]),
                "attention_weight_prices": float(belief["attention_weight_prices"]),
                "attention_weight_jobs": float(belief["attention_weight_jobs"]),
                "attention_weight_rates": float(belief["attention_weight_rates"]),
                "reason_codes_json": json.dumps(belief["reason_codes"], sort_keys=True),
                "causal_path_json": json.dumps(belief["causal_path"], sort_keys=True),
            }
        )
    return rows


def _client_specs(
    variants: list[str],
    models: list[str],
    belief_mode: str,
    fixture_variants: set[str] | None = None,
) -> list[tuple[str, str, str]]:
    fixture_variants = fixture_variants or set()
    specs: list[tuple[str, str, str]] = []
    for variant in variants:
        if variant in fixture_variants:
            if variant in LLM_VARIANTS:
                specs.extend((variant, model, "fixture") for model in models)
            else:
                specs.append((variant, "structural", "fixture"))
            continue
        if belief_mode != "live":
            mode = belief_mode if variant in LLM_VARIANTS else "fixture"
            if variant in LLM_VARIANTS:
                specs.extend((variant, model, mode) for model in models)
            else:
                specs.append((variant, "structural", "fixture"))
        elif variant in LLM_VARIANTS and variant not in fixture_variants:
            specs.extend((variant, model, "live") for model in models)
        elif variant in LLM_VARIANTS:
            specs.extend((variant, model, "fixture") for model in models)
        else:
            specs.append((variant, "structural", "fixture"))
    return specs


def _prompt_payload_for_variant(
    variant: str,
    scenario: DemandScenario,
    period_state: dict[str, Any],
    household_states: list[dict[str, Any]],
) -> dict[str, Any]:
    if variant == "naive_persona":
        return naive_persona_prompt_payload(scenario, period_state, household_states)
    return belief_module_prompt_payload(scenario, period_state, household_states, variant=variant)


def _prompt_current_conditions(period_state: dict[str, Any], scenario: DemandScenario) -> dict[str, Any]:
    transfer = float(period_state["transfer_per_household"])
    rate_shock = float(period_state["policy_rate_shock_pp"])
    job_risk_shock = float(period_state["job_risk_shock_pp"])
    active = []
    if transfer > 0:
        active.append("lump_sum_transfer_now")
    if abs(rate_shock) > 0:
        active.append("policy_rate_shock_now")
    if abs(job_risk_shock) > 0:
        active.append("job_risk_news_now")
    if float(scenario.belief_dispersion_multiplier) > 1.0:
        active.append("elevated_belief_dispersion_regime_now")
    if float(scenario.feedback_gain) > 1.0:
        active.append("elevated_macro_feedback_regime_now")
    return {
        "active_current_shocks": active or ["none"],
        "periods_per_year": round_or_none(period_state.get("periods_per_year")),
        "months_per_period": round_or_none(period_state.get("months_per_period")),
        "transfer_per_household": round_or_none(transfer),
        "policy_rate_shock_pp": round_or_none(rate_shock),
        "job_risk_news_shock_pp": round_or_none(job_risk_shock),
        "belief_dispersion_regime": "elevated" if float(scenario.belief_dispersion_multiplier) > 1.0 else "normal",
        "macro_feedback_regime": "elevated" if float(scenario.feedback_gain) > 1.0 else "normal",
        "supplied_exogenous_conditions": _jsonable(period_state.get("supplied_exogenous_conditions", {})),
        "future_shock_path_disclosed": False,
    }


def _prompt_current_environment(period_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "period_index": round_or_none(period_state["period_index"]),
        "periods_per_year": round_or_none(period_state.get("periods_per_year")),
        "months_per_period": round_or_none(period_state.get("months_per_period")),
        "output_gap_pct": round_or_none(period_state["output_gap_pct"]),
        "employment_rate": round_or_none(period_state["employment_rate"]),
        "inflation_rate": round_or_none(period_state["inflation_rate"]),
        "policy_rate": round_or_none(period_state["policy_rate"]),
        "transfer_per_household": round_or_none(period_state["transfer_per_household"]),
        "job_risk_news_pp": round_or_none(period_state["job_risk_shock_pp"]),
        "aggregate_job_loss_belief": round_or_none(period_state.get("aggregate_job_loss_belief")),
        "aggregate_confidence_index": round_or_none(period_state.get("aggregate_confidence_index")),
        "aggregate_liquid_buffer_months": round_or_none(period_state.get("aggregate_liquid_buffer_months")),
        "supplied_exogenous_conditions": _jsonable(period_state.get("supplied_exogenous_conditions", {})),
    }


def _household_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt_row = {
        "type_id": row["type_id"],
        "label": row["label"],
        "age_bucket": row.get("age_bucket"),
        "income_group": row["income_group"],
        "liquidity_group": row["liquidity_group"],
        "job_loss_risk_type": row.get("job_loss_risk_type"),
        "population_weight": round_or_none(row["population_weight"]),
        "periods_per_year": round_or_none(row.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR)),
        "period_labor_income": round_or_none(row["labor_income"]),
        "period_baseline_consumption": round_or_none(row["baseline_consumption"]),
        "liquid_assets": round_or_none(row["liquid_assets"]),
        "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
        "debt_service_burden": round_or_none(row.get("debt_service_burden")),
        "base_mpc": round_or_none(row["base_mpc"]),
        "prior_expected_inflation": round_or_none(row["inflation_expectation_1y"]),
        "prior_expected_income_growth": round_or_none(row["income_growth_expectation_1y"]),
        "prior_job_loss_probability": round_or_none(row["job_loss_probability"]),
        "prior_confidence_index": round_or_none(row["confidence_index"]),
        "attention_weight_prices": round_or_none(row["attention_weight_prices"]),
        "attention_weight_jobs": round_or_none(row["attention_weight_jobs"]),
        "attention_weight_rates": round_or_none(row["attention_weight_rates"]),
    }
    if np.isclose(float(row.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR)), DEFAULT_PERIODS_PER_YEAR):
        prompt_row["quarterly_labor_income"] = round_or_none(row["labor_income"])
        prompt_row["quarterly_baseline_consumption"] = round_or_none(row["baseline_consumption"])
    return prompt_row


def _belief_payload_row(
    row: dict[str, Any],
    *,
    expected_inflation: float,
    expected_income: float,
    job_loss: float,
    confidence: float,
    precaution: float,
    attention_prices: float,
    attention_jobs: float,
    attention_rates: float,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "type_id": row["type_id"],
        "expected_inflation_next_period": float(np.clip(expected_inflation, -5.0, 15.0)),
        "expected_income_growth_next_period": float(np.clip(expected_income, -12.0, 12.0)),
        "perceived_job_loss_probability": float(np.clip(job_loss, 0.0, 40.0)),
        "expected_unemployment_higher_probability_next_period": float(np.clip(job_loss / 0.24, 0.0, 100.0)),
        "confidence_index": float(np.clip(confidence, 0.0, 100.0)),
        "precautionary_saving_score": float(np.clip(precaution, 0.0, 10.0)),
        "attention_weight_prices": float(np.clip(attention_prices, 0.0, 1.0)),
        "attention_weight_jobs": float(np.clip(attention_jobs, 0.0, 1.0)),
        "attention_weight_rates": float(np.clip(attention_rates, 0.0, 1.0)),
        "reason_codes": reason_codes,
        "causal_path": ["abstract signal", "belief update", "structural consumption rule"],
    }


def _reason_codes(
    inflation_gap: float,
    output_gap: float,
    policy_gap: float,
    job_shock: float,
    *,
    low_liquid: bool,
) -> list[str]:
    codes: list[str] = []
    if abs(inflation_gap) > 0.15:
        codes.append("prices")
    if output_gap < -0.25:
        codes.append("jobs")
    if policy_gap > 0.15:
        codes.append("rates")
    if job_shock > 0:
        codes.append("job_security")
    if low_liquid:
        codes.append("liquidity")
    return codes or ["steady_state"]


def _representative_mpc(household_states: list[dict[str, Any]]) -> float:
    return float(sum(float(row["base_mpc"]) * float(row["population_weight"]) for row in household_states))


def _group_mpcs(
    decisions: pd.DataFrame,
    *,
    source: str,
    group_column: str,
    low_label: str,
    high_label: str,
) -> tuple[float, float]:
    subset = decisions[
        (decisions["source"] == source)
        & (decisions["period_index"] == 1)
        & (decisions["scenario_id"].isin(["baseline", "transfer_shock"]))
    ].copy()
    if subset.empty:
        return np.nan, np.nan
    baseline = subset[subset["scenario_id"] == "baseline"][["type_id", "consumption"]].rename(columns={"consumption": "baseline_consumption"})
    transfer = subset[subset["scenario_id"] == "transfer_shock"].merge(baseline, on="type_id", how="inner")
    if transfer.empty:
        return np.nan, np.nan
    transfer["type_mpc"] = (transfer["consumption"] - transfer["baseline_consumption"]) / transfer["transfer"].replace(0.0, np.nan)
    grouped = (
        transfer.groupby(group_column)
        .apply(lambda group: float((group["type_mpc"] * group["population_weight"]).sum() / group["population_weight"].sum()))
        .to_dict()
    )
    return float(grouped.get(low_label, np.nan)), float(grouped.get(high_label, np.nan))


def _liquidity_mpcs(decisions: pd.DataFrame, *, source: str) -> tuple[float, float]:
    return _group_mpcs(decisions, source=source, group_column="liquidity_group", low_label="low", high_label="high")


def _metric_row(
    source: str,
    metric: str,
    value: float,
    target_low: float,
    target_high: float,
    interpretation: str,
    *,
    required: bool | None = None,
) -> dict[str, Any]:
    if not np.isfinite(value):
        passed = False
    else:
        passed = bool(float(value) >= float(target_low) and float(value) <= float(target_high))
    variant = _variant_from_source(source)
    return {
        "source": source,
        "variant": variant,
        "metric": metric,
        "value": float(value) if np.isfinite(value) else np.nan,
        "target_low": float(target_low) if np.isfinite(target_low) else target_low,
        "target_high": float(target_high) if np.isfinite(target_high) else target_high,
        "passed": passed,
        "required": _required_metric_for_verdict(variant, metric) if required is None else bool(required),
        "interpretation": interpretation,
    }


def _metric_value(validation: pd.DataFrame, metric: str) -> float | None:
    row = validation[validation["metric"] == metric]
    if row.empty:
        return None
    value = float(row["value"].iloc[0])
    return value if np.isfinite(value) else None


def _weighted(rows: list[dict[str, Any]], column: str) -> float:
    return float(sum(float(row[column]) * float(row["population_weight"]) for row in rows))


def _weighted_frame_average(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    weights = frame["population_weight"].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return np.nan
    return float((frame[column].astype(float) * weights).sum() / total)


def _weighted_array_average(values: Any, weights: Any) -> float:
    values_series = pd.Series(values, dtype=float)
    weights_series = pd.Series(weights, dtype=float)
    total = float(weights_series.sum())
    if total <= 0 or values_series.empty:
        return np.nan
    return float((values_series * weights_series).sum() / total)


def _weighted_corr(x: Any, y: Any, weights: Any) -> float:
    x_series = pd.Series(x, dtype=float)
    y_series = pd.Series(y, dtype=float)
    weights_series = pd.Series(weights, dtype=float)
    total = float(weights_series.sum())
    if total <= 0 or x_series.shape[0] < 2:
        return np.nan
    mean_x = _weighted_array_average(x_series, weights_series)
    mean_y = _weighted_array_average(y_series, weights_series)
    cov = _weighted_array_average((x_series - mean_x) * (y_series - mean_y), weights_series)
    var_x = _weighted_array_average((x_series - mean_x) ** 2, weights_series)
    var_y = _weighted_array_average((y_series - mean_y) ** 2, weights_series)
    denom = float(np.sqrt(max(0.0, var_x) * max(0.0, var_y)))
    if denom <= 0:
        return np.nan
    return float(cov / denom)


def _weighted_std_frame(frame: pd.DataFrame, column: str, weight_column: str) -> float:
    if frame.empty:
        return np.nan
    weights = frame[weight_column].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return np.nan
    values = frame[column].astype(float)
    mean = float((values * weights).sum() / total)
    variance = float(((values - mean) ** 2 * weights).sum() / total)
    return float(np.sqrt(max(0.0, variance)))


def _validated_periods_per_year(periods_per_year: float) -> float:
    value = float(periods_per_year)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("periods_per_year must be a positive finite float")
    return value


def _months_per_period(periods_per_year: float) -> float:
    return 12.0 / _validated_periods_per_year(periods_per_year)


def _per_period_amount(annual_amount: float, periods_per_year: float) -> float:
    return float(annual_amount) / _validated_periods_per_year(periods_per_year)


def _quarterly_persistence_for_frequency(quarterly_persistence: float, periods_per_year: float) -> float:
    persistence = float(quarterly_persistence)
    if not 0.0 <= persistence < 1.0:
        raise ValueError("quarterly_persistence must be in [0, 1)")
    frequency = _validated_periods_per_year(periods_per_year)
    if np.isclose(frequency, DEFAULT_PERIODS_PER_YEAR):
        return persistence
    return float(persistence ** (DEFAULT_PERIODS_PER_YEAR / frequency))


def _quarterly_flow_for_frequency(
    quarterly_flow: float,
    quarterly_persistence: float,
    periods_per_year: float,
) -> float:
    persistence = float(quarterly_persistence)
    if not 0.0 <= persistence < 1.0:
        raise ValueError("quarterly_persistence must be in [0, 1)")
    frequency_persistence = _quarterly_persistence_for_frequency(persistence, periods_per_year)
    return float(quarterly_flow) * (1.0 - frequency_persistence) / (1.0 - persistence)


def _normalize_period_overrides(period_overrides: dict[int, dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if period_overrides is None:
        return {}
    if not isinstance(period_overrides, dict):
        raise ValueError("period_overrides must be a dict keyed by period index")
    out: dict[int, dict[str, Any]] = {}
    for raw_index, raw_override in period_overrides.items():
        try:
            period_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"period_overrides key must be an integer period index, got {raw_index!r}") from exc
        if period_index < 0:
            raise ValueError(f"period_overrides period index must be non-negative, got {period_index}")
        if not isinstance(raw_override, dict):
            raise ValueError(f"period_overrides[{period_index}] must be a dict of exogenous fields")
        forbidden = sorted(PROTECTED_PERIOD_OVERRIDE_FIELDS.intersection(raw_override))
        if forbidden:
            raise ValueError(
                "period_overrides cannot override protected period identity fields: "
                + ", ".join(forbidden)
            )
        override = dict(raw_override)
        _assert_finite_override_values(override, path=f"period_overrides[{period_index}]")
        out[period_index] = override
    return out


def _normalize_initial_environment_override(value: dict[str, Any] | None) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("initial_environment_override must be a dict")
    allowed = {
        "output_gap_pct",
        "employment_rate",
        "inflation_rate",
        "policy_rate",
        "aggregate_job_loss_belief",
        "aggregate_confidence_index",
        "aggregate_liquid_buffer_months",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("initial_environment_override contains unsupported fields: " + ", ".join(unknown))
    _assert_finite_override_values(value, path="initial_environment_override")
    out = {key: float(raw) for key, raw in value.items()}
    if "employment_rate" in out and not 0.0 <= out["employment_rate"] <= 1.0:
        raise ValueError("initial_environment_override employment_rate must be between zero and one")
    return out


def _assert_finite_override_values(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_finite_override_values(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_finite_override_values(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise ValueError(f"{path} contains a nonfinite numeric value")
        return


def _buffer_months(liquid_assets: float, quarterly_consumption: float, *, periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
    monthly_consumption = max(float(quarterly_consumption) / _months_per_period(periods_per_year), 1e-9)
    return float(max(0.0, float(liquid_assets)) / monthly_consumption)


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:80] for item in value[:limit]]


def _variant_from_source(source: str) -> str:
    for variant in MODEL_VARIANTS:
        if source.startswith(variant):
            return variant
    return source


def _variant_expected_role(variant: str) -> str:
    return {
        "representative": "Smooth representative-agent baseline with weak heterogeneity.",
        "adaptive": "Hand-coded heterogeneous belief baseline; proves closure without LLMs.",
        "llm_belief": "Main architecture: belief module plus structural consumption/accounting.",
        "naive_persona": "Bad baseline: direct consume/save prompt with budget clamp.",
    }.get(variant, "")


def _required_metric_for_verdict(variant: str, metric: str) -> bool:
    optional_for_representative = {
        "belief_feedback_amplification_ratio",
        "belief_inflation_dispersion_p1",
        "high_liquidity_impact_mpc",
        "low_liquidity_impact_mpc",
        "liquidity_mpc_gradient",
        "income_mpc_gradient",
    }
    optional_for_adaptive = {"belief_feedback_amplification_ratio"}
    required_for_naive = {
        "steady_state_final_output_gap_abs",
        "steady_state_tail_output_gap_range",
        "steady_state_tail_inflation_range",
        "max_accounting_abs_residual",
    }
    if variant == "llm_belief":
        return True
    if variant == "adaptive":
        return metric not in optional_for_adaptive
    if variant == "representative":
        return metric not in optional_for_representative
    if variant == "naive_persona":
        return metric in required_for_naive
    return True


def _demand_instructions() -> str:
    return """
Return only valid JSON. Use only the abstract economy state supplied in the prompt.
Do not browse, inspect files, run commands, cite historical episodes, or infer calendar dates.
For the belief-module variant, do not choose consumption or saving dollars.
""".strip()


def _demand_baseline_comparison_frame(ablations: pd.DataFrame, belief_targets: pd.DataFrame) -> pd.DataFrame:
    if ablations.empty:
        return pd.DataFrame()
    comparison = ablations.copy()
    if not belief_targets.empty and "belief_variable" in belief_targets:
        belief_mae = belief_targets[belief_targets["belief_variable"] == "ALL"][["source", "normalized_mae"]].rename(
            columns={"normalized_mae": "survey_seed_normalized_mae"}
        )
        comparison = comparison.merge(belief_mae, on="source", how="left")
    else:
        comparison["survey_seed_normalized_mae"] = np.nan

    if "belief_feedback_amplification_ratio" not in comparison:
        comparison["belief_feedback_amplification_ratio"] = np.nan
    llm_mae = comparison.loc[comparison["variant"].astype(str) == "llm_belief", "survey_seed_normalized_mae"].dropna()
    best_llm_mae = float(llm_mae.min()) if not llm_mae.empty else np.nan
    comparison["mae_delta_vs_best_belief_module"] = comparison["survey_seed_normalized_mae"] - best_llm_mae
    comparison.loc[comparison["variant"].astype(str) == "naive_persona", "mae_delta_vs_best_belief_module"] = np.nan
    comparison["belief_target_note"] = comparison["variant"].map(
        {
            "llm_belief": "belief-module actor",
            "adaptive": "coded belief baseline",
            "representative": "pooled representative baseline",
            "naive_persona": "mechanical fixture seed echo",
        }
    )
    columns = [
        "source",
        "variant",
        "belief_target_note",
        "metric_count",
        "passed_metric_count",
        "required_metric_count",
        "passed_required_metric_count",
        "all_metrics_passed",
        "required_metrics_passed",
        "survey_seed_normalized_mae",
        "mae_delta_vs_best_belief_module",
        "transfer_impact_mpc",
        "liquidity_mpc_gradient",
        "income_mpc_gradient",
        "belief_feedback_amplification_ratio",
        "rate_hike_mean_consumption_delta_6p",
        "job_risk_impact_consumption_delta",
    ]
    existing = [column for column in columns if column in comparison]
    return comparison[existing].sort_values(["variant", "source"]).reset_index(drop=True)


def _demand_baseline_comparison_table(ablations: pd.DataFrame, belief_targets: pd.DataFrame) -> str:
    return markdown_table(_demand_baseline_comparison_frame(ablations, belief_targets))


def _variant_summary_row(ablations: pd.DataFrame, belief_targets: pd.DataFrame, variant: str) -> pd.Series | None:
    comparison = _demand_baseline_comparison_frame(ablations, belief_targets)
    subset = comparison[comparison["variant"].astype(str) == variant].copy()
    if subset.empty:
        return None
    sort_columns = ["all_metrics_passed", "required_metrics_passed", "survey_seed_normalized_mae"]
    ascending = [False, False, True]
    return subset.sort_values(sort_columns, ascending=ascending, na_position="last").iloc[0]


def _fmt_report_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _demand_bottom_line(
    manifest: dict[str, Any],
    failed: pd.DataFrame,
    ablations: pd.DataFrame,
    belief_targets: pd.DataFrame,
) -> str:
    evidence = manifest.get("evidence", {})
    verdict = evidence.get("evidence_verdict", "unknown")
    if failed.empty and evidence.get("required_variants_present") and evidence.get("full_lab_metric_surface_present"):
        llm = _variant_summary_row(ablations, belief_targets, "llm_belief")
        adaptive = _variant_summary_row(ablations, belief_targets, "adaptive")
        representative = _variant_summary_row(ablations, belief_targets, "representative")
        if manifest.get("belief_mode") == "live" and llm is not None:
            llm_passed = f"{int(llm.get('passed_metric_count', 0))}/{int(llm.get('metric_count', 0))}"
            return (
                f"`{verdict}`. The live belief-module economy clears the full dynamic gate: the LLM variant passes "
                f"{llm_passed} validation metrics, all required metrics pass across the ablation surface, and "
                "accounting identities hold each period. On survey-seed belief targets, the live LLM has normalized "
                f"MAE {_fmt_report_float(llm.get('survey_seed_normalized_mae'))}, versus adaptive "
                f"{_fmt_report_float(adaptive.get('survey_seed_normalized_mae') if adaptive is not None else np.nan)} "
                "and representative "
                f"{_fmt_report_float(representative.get('survey_seed_normalized_mae') if representative is not None else np.nan)}; "
                "the naive persona row is a mechanical fixture seed echo, so its zero seed error is not evidence of "
                "belief formation. The live LLM is also the only variant in this run to pass feedback amplification "
                f"({_fmt_report_float(llm.get('belief_feedback_amplification_ratio'), 2)}x) while preserving transfer "
                "MPCs, liquidity and income MPC gradients, monetary-shock contraction, job-risk precaution, and "
                "budget closure."
            )
        if manifest.get("belief_mode") == "fixture":
            return (
                f"`{verdict}`. The zero-call fixture clears accounting, steady-state, transfer, monetary, "
                "job-risk, feedback, and ablation-coverage checks. It is a clean lab sanity check before live "
                "belief-module sweeps."
            )
        return (
            f"`{verdict}`. The HANK-lite demand economy clears accounting, steady-state, transfer, monetary, "
            "job-risk, feedback, and ablation-coverage checks."
        )
    if failed.empty and not evidence.get("full_lab_metric_surface_present", True):
        missing = evidence.get("missing_full_lab_metrics", [])
        missing_text = ", ".join(str(metric) for metric in missing[:4])
        if len(missing) > 4:
            missing_text += f", plus {len(missing) - 4} more"
        return (
            f"`{verdict}`. The metrics included in this run pass, accounting identities hold, and the ablation "
            f"surface is present, but this is not the full lab gate because missing metrics remain: {missing_text}."
        )
    if failed.empty:
        return f"`{verdict}`. Dynamic metrics pass, but the ablation surface is incomplete."
    return (
        f"`{verdict}`. The harness ran, but some dynamic validation checks failed. "
        "Treat the failed-metric table as the next calibration target before live model spend."
    )


def _sanitized_argv() -> list[str]:
    raw = sys.argv
    if raw and Path(raw[0]).name == "demand_economy.py":
        return ["python3", "-m", "macro_llm_tournament.demand_economy", *raw[1:]]
    return list(raw)


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - git can be unavailable in packaged use
        return {"error": str(exc)[:200]}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
