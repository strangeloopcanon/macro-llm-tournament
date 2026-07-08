from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import ACCOUNTING_TOLERANCE, OUTPUT_ROOT, markdown_table
from .demand_economy import (
    DEMAND_ECONOMY_PROMPT_VERSION,
    DemandScenario,
    _accounting_rows,
    _aggregate_period,
    _belief_rows,
    _git_metadata,
    _initial_environment,
    _initial_household_states,
    _jsonable,
    _next_environment,
    _next_household_states,
    _period_state,
    _realize_household_period,
    _write_json,
    build_fixture_demand_households,
    fixture_belief_payload,
    normalize_belief_payload,
)
from .llm_common import LLMUnavailable


MACRO_PLAYGROUND_VERSION = "playable_macro_engine_v0_2"
SCENARIO_SPEC_VERSION = "macro_playground_spec_v1"
FIRM_REACTION_PROMPT_VERSION = "macro_playground_firm_reaction_v1"
POLICY_NARRATIVE_PROMPT_VERSION = "macro_playground_policy_narrative_v1"
CRITIC_PROMPT_VERSION = "macro_playground_critic_v1"
PLAYGROUND_MODES = ("fixture", "replay", "live")
ACTOR_ROLES = ("firm", "policy_narrative", "household_beliefs", "critic")
VARIANT = "llm_belief"

_DATE_OR_EPISODE_PATTERNS = (
    "2008",
    "great recession",
    "covid",
    "pandemic",
)
_INITIAL_ENVIRONMENT_KEYS = {
    "baseline_aggregate_consumption",
    "aggregate_consumption",
    "output_gap_pct",
    "employment_rate",
    "inflation_rate",
    "policy_rate",
    "aggregate_job_loss_belief",
    "aggregate_confidence_index",
    "aggregate_liquid_buffer_months",
}
_ACTOR_PROMPT_VERSIONS = {
    "firm": FIRM_REACTION_PROMPT_VERSION,
    "policy_narrative": POLICY_NARRATIVE_PROMPT_VERSION,
    "household_beliefs": DEMAND_ECONOMY_PROMPT_VERSION,
    "critic": CRITIC_PROMPT_VERSION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixture-first playable macro engine v0.2.")
    parser.add_argument("--spec", required=True, help="Path to a date-free macro playground JSON scenario spec.")
    parser.add_argument("--mode", choices=PLAYGROUND_MODES, default="fixture")
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--replay-records-json", default=None, help="Prior macro_playground_actor_payloads.jsonl for replay mode.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec = load_macro_playground_spec(Path(args.spec))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid macro playground spec: {exc}", file=sys.stderr)
        return 2

    spec_hash = canonical_sha256(spec)
    output_dir = _resolve_output_dir(spec, spec_hash, output_root=Path(args.output_root), output_dir=args.output_dir)
    _guard_output_dir(output_dir, expected_spec_hash=spec_hash)

    if args.mode == "live":
        return _write_live_blocked(spec, spec_hash, args=args, output_dir=output_dir)

    replay_records = _load_replay_records(Path(args.replay_records_json)) if args.mode == "replay" else []
    if args.mode == "replay" and not replay_records:
        print("--replay-records-json is required for replay mode", file=sys.stderr)
        return 2

    try:
        result = run_macro_playground(
            spec,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            max_live_calls=args.max_live_calls,
            replay_records=replay_records,
        )
    except (ValueError, LLMUnavailable) as exc:
        print(f"Macro playground failed: {exc}", file=sys.stderr)
        return 1

    write_macro_playground_outputs(result, output_dir)
    verdict = result["manifest"]["verdict"]
    print(f"Wrote macro playground run to {output_dir}")
    print(json.dumps({"verdict": verdict, "passed": result["manifest"]["passed"]}, indent=2, sort_keys=True))
    return 0 if result["manifest"]["passed"] else 1


def load_macro_playground_spec(path: Path) -> dict[str, Any]:
    return normalize_macro_playground_spec(json.loads(path.read_text(encoding="utf-8")))


def normalize_macro_playground_spec(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("macro playground spec must be a JSON object")
    required = {"scenario_spec_version", "run_id", "horizon", "household_count", "initial_environment", "branches"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"macro playground spec missing required fields: {', '.join(sorted(missing))}")
    if data["scenario_spec_version"] != SCENARIO_SPEC_VERSION:
        raise ValueError(f"scenario_spec_version must be {SCENARIO_SPEC_VERSION}")
    run_id = _clean_id(data["run_id"], field="run_id")
    horizon = _strict_int(data["horizon"], "horizon", lower=8, upper=200)
    household_count = _strict_int(data["household_count"], "household_count", lower=1, upper=24)
    initial_environment = _normalize_initial_environment(data["initial_environment"])
    raw_branches = data["branches"]
    if not isinstance(raw_branches, list) or not raw_branches:
        raise ValueError("branches must be a non-empty list")
    branches = [_normalize_branch(branch) for branch in raw_branches]
    branch_ids = [branch["branch_id"] for branch in branches]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("branch_id values must be unique")
    known = set(branch_ids)
    for branch in branches:
        parent = branch.get("parent_branch_id")
        if parent is not None and parent not in known:
            raise ValueError(f"parent_branch_id not found for branch {branch['branch_id']}: {parent}")
    normalized = {
        "scenario_spec_version": SCENARIO_SPEC_VERSION,
        "run_id": run_id,
        "horizon": horizon,
        "household_count": household_count,
        "initial_environment": initial_environment,
        "branches": branches,
    }
    _assert_date_free(normalized)
    return normalized


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_macro_playground(
    spec: dict[str, Any],
    *,
    mode: str = "fixture",
    provider: str = "codex_cli",
    model: str = "gpt-5.5",
    max_live_calls: int = 0,
    replay_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in PLAYGROUND_MODES:
        raise ValueError(f"Unsupported macro playground mode: {mode}")
    if mode == "live":
        raise LLMUnavailable("macro playground live mode is blocked until fixture QA and replay provenance gates pass")
    if max_live_calls:
        raise ValueError("--max-live-calls must be 0 outside live mode")

    spec = normalize_macro_playground_spec(spec)
    spec_hash = canonical_sha256(spec)
    replay_records = replay_records or []
    source = f"macro_playground_{mode}_{model}"
    households = build_fixture_demand_households(spec["household_count"])
    initial = _initial_household_states(households, source=source, variant=VARIANT)
    base_env = _apply_initial_environment(_initial_environment(households), spec["initial_environment"])

    branch_rows = []
    belief_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    branch_snapshots: dict[str | None, dict[int, tuple[list[dict[str, Any]], dict[str, float]]]] = {
        None: {0: (initial.to_dict(orient="records"), deepcopy(base_env))}
    }

    for branch in _ordered_branches(spec["branches"]):
        branch_id = branch["branch_id"]
        parent_id = branch.get("parent_branch_id")
        from_period = int(branch["from_period"])
        if parent_id not in branch_snapshots or from_period not in branch_snapshots[parent_id]:
            raise ValueError(f"Branch {branch_id} cannot start from parent {parent_id} at period {from_period}")
        start_states, start_env = branch_snapshots[parent_id][from_period]
        household_states = deepcopy(start_states)
        env = deepcopy(start_env)
        scenario = _branch_to_demand_scenario(branch)
        branch_snapshots[branch_id] = {}
        branch_rows.append(_branch_table_row(spec["run_id"], branch, spec_hash))
        for period_index in range(from_period, spec["horizon"]):
            branch_snapshots[branch_id][period_index] = (deepcopy(household_states), deepcopy(env))
            period_state = _period_state(env, scenario, period_index)
            period_id = str(period_state["period_id"])
            state_sha = _economic_state_sha256(
                branch_id=branch_id,
                period_id=period_id,
                household_states=household_states,
                environment=env,
                scenario=scenario,
            )

            firm_payload = _actor_payload(
                "firm",
                branch,
                period_state,
                mode=mode,
                replay_records=replay_records,
                spec_hash=spec_hash,
                state_sha=state_sha,
            )
            policy_payload = _actor_payload(
                "policy_narrative",
                branch,
                period_state,
                mode=mode,
                replay_records=replay_records,
                spec_hash=spec_hash,
                state_sha=state_sha,
            )
            effective_period_state = _apply_policy_to_period_state(period_state, policy_payload)
            effective_scenario = _apply_policy_to_scenario(scenario, policy_payload)
            household_payload = _actor_payload(
                "household_beliefs",
                branch,
                effective_period_state,
                mode=mode,
                replay_records=replay_records,
                spec_hash=spec_hash,
                state_sha=state_sha,
                scenario=effective_scenario,
                household_states=household_states,
            )
            panel = normalize_household_belief_payload(household_states, {"payload": household_payload})
            panel = _apply_policy_to_belief_panel(panel, policy_payload)
            period_beliefs = _belief_rows(
                panel,
                household_states,
                effective_period_state,
                source=source,
                variant=VARIANT,
            )
            _attach_branch(period_beliefs, branch_id=branch_id, run_id=spec["run_id"])
            belief_rows.extend(period_beliefs)
            realized = _realize_household_period(
                households,
                household_states,
                panel,
                effective_period_state,
                source=source,
                variant=VARIANT,
            )
            _attach_branch(realized, branch_id=branch_id, run_id=spec["run_id"])
            decision_rows.extend(realized)
            aggregate = _aggregate_period(realized, effective_scenario, effective_period_state, source=source, variant=VARIANT)
            aggregate = _attach_period_actor_state(aggregate, branch_id=branch_id, run_id=spec["run_id"], firm=firm_payload, policy=policy_payload)
            period_rows.append(aggregate)
            period_accounting = _accounting_rows(realized, aggregate)
            _attach_branch(period_accounting, branch_id=branch_id, run_id=spec["run_id"])
            period_accounting.extend(_extra_accounting_rows(realized, aggregate, branch_id=branch_id, run_id=spec["run_id"]))
            accounting_rows.extend(period_accounting)
            critic_payload = _actor_payload(
                "critic",
                branch,
                effective_period_state,
                mode=mode,
                replay_records=replay_records,
                spec_hash=spec_hash,
                state_sha=state_sha,
                aggregate=aggregate,
                accounting=period_accounting,
                firm=firm_payload,
                policy=policy_payload,
            )
            critic = normalize_critic_payload(critic_payload, branch_id=branch_id, period_id=period_id)
            critic_rows.extend(_critic_flag_rows(spec["run_id"], branch_id, period_id, critic))
            for actor_role, payload in [
                ("firm", firm_payload),
                ("policy_narrative", policy_payload),
                ("household_beliefs", household_payload),
                ("critic", critic_payload),
            ]:
                event = _event_row(
                    run_id=spec["run_id"],
                    branch_id=branch_id,
                    period_id=period_id,
                    actor_role=actor_role,
                    prompt_version=_ACTOR_PROMPT_VERSIONS[actor_role],
                    spec_hash=spec_hash,
                    state_sha=state_sha,
                    payload=payload,
                    provider=provider,
                    model=model,
                    mode=mode,
                )
                actor_rows.append({**event, "payload": payload})
                event_rows.append({key: value for key, value in event.items() if key != "payload"})
            next_states = _next_household_states(realized, households, aggregate)
            next_env = _next_environment(env, aggregate, effective_scenario, feedback_mode="closed_loop")
            env = _apply_firm_policy_to_next_environment(next_env, firm_payload, policy_payload)
            household_states = next_states
        branch_snapshots[branch_id][spec["horizon"]] = (deepcopy(household_states), deepcopy(env))

    frames = {
        "branches": pd.DataFrame(branch_rows),
        "initial": initial,
        "beliefs": pd.DataFrame(belief_rows),
        "decisions": pd.DataFrame(decision_rows),
        "periods": pd.DataFrame(period_rows),
        "accounting": pd.DataFrame(accounting_rows),
        "actor_payloads": pd.DataFrame(actor_rows),
        "event_log": pd.DataFrame(event_rows),
        "critic_flags": pd.DataFrame(critic_rows),
    }
    qa = build_macro_playground_qa(frames, spec)
    verdict = _macro_playground_verdict(qa, mode=mode)
    manifest = {
        "schema_version": MACRO_PLAYGROUND_VERSION,
        "scenario_spec_version": SCENARIO_SPEC_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec["run_id"],
        "normalized_spec_sha256": spec_hash,
        "mode": mode,
        "provider": provider,
        "model": model,
        "max_live_calls": int(max_live_calls),
        "live_call_count": 0,
        "household_count": int(spec["household_count"]),
        "horizon": int(spec["horizon"]),
        "branch_count": int(len(spec["branches"])),
        "actor_roles": list(ACTOR_ROLES),
        "git": _git_metadata(),
        "status": "ok",
        "verdict": verdict,
        "passed": verdict == "macro_playground_fixture_ready",
        "outputs": _output_names(),
    }
    report = build_macro_playground_report(manifest, frames, qa)
    return {"spec": spec, "manifest": manifest, "frames": {**frames, "qa": qa}, "report": report}


def normalize_household_belief_payload(household_states: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", data)
    allowed_root = {"prompt_version", "beliefs"}
    extra_root = set(payload) - allowed_root
    if extra_root:
        raise LLMUnavailable(f"Household belief payload has unsupported fields: {', '.join(sorted(extra_root))}")
    for forbidden in ["actions", "direct_actions", "desired_consumption", "desired_saving", "cash_available"]:
        if forbidden in json.dumps(payload, sort_keys=True):
            raise LLMUnavailable(f"Household belief payload contains forbidden mutation field: {forbidden}")
    allowed_belief = {
        "type_id",
        "expected_inflation_next_period",
        "expected_income_growth_next_period",
        "perceived_job_loss_probability",
        "expected_unemployment_higher_probability_next_period",
        "confidence_index",
        "precautionary_saving_score",
        "attention_weight_prices",
        "attention_weight_jobs",
        "attention_weight_rates",
        "reason_codes",
        "causal_path",
    }
    for belief in payload.get("beliefs", []):
        if not isinstance(belief, dict):
            raise LLMUnavailable("Each household belief row must be an object")
        extra = set(belief) - allowed_belief
        if extra:
            raise LLMUnavailable(f"Household belief row has unsupported fields: {', '.join(sorted(extra))}")
    return normalize_belief_payload(household_states, data)


def normalize_firm_reaction_payload(payload: dict[str, Any], *, branch_id: str, period_id: str) -> dict[str, Any]:
    allowed = {
        "prompt_version",
        "branch_id",
        "period_id",
        "planned_output_gap_pct",
        "hiring_gap_pp",
        "price_pressure_pp",
        "credit_tightening_index",
        "reason",
    }
    _reject_extra(payload, allowed, actor="firm")
    _require_identity(payload, prompt_version=FIRM_REACTION_PROMPT_VERSION, branch_id=branch_id, period_id=period_id, actor="firm")
    return {
        "prompt_version": FIRM_REACTION_PROMPT_VERSION,
        "branch_id": branch_id,
        "period_id": period_id,
        "planned_output_gap_pct": _strict_float(payload, "planned_output_gap_pct", -8.0, 8.0),
        "hiring_gap_pp": _strict_float(payload, "hiring_gap_pp", -4.0, 4.0),
        "price_pressure_pp": _strict_float(payload, "price_pressure_pp", -3.0, 3.0),
        "credit_tightening_index": _strict_float(payload, "credit_tightening_index", 0.0, 5.0),
        "reason": str(payload.get("reason", ""))[:400],
    }


def normalize_policy_narrative_payload(payload: dict[str, Any], *, branch_id: str, period_id: str) -> dict[str, Any]:
    allowed = {
        "prompt_version",
        "branch_id",
        "period_id",
        "rate_rule_shift_pp",
        "transfer_per_household",
        "communication_confidence_shift",
        "job_risk_attention_shift_pp",
        "dispersion_multiplier",
        "reason",
    }
    _reject_extra(payload, allowed, actor="policy_narrative")
    _require_identity(payload, prompt_version=POLICY_NARRATIVE_PROMPT_VERSION, branch_id=branch_id, period_id=period_id, actor="policy_narrative")
    return {
        "prompt_version": POLICY_NARRATIVE_PROMPT_VERSION,
        "branch_id": branch_id,
        "period_id": period_id,
        "rate_rule_shift_pp": _strict_float(payload, "rate_rule_shift_pp", -5.0, 5.0),
        "transfer_per_household": _strict_float(payload, "transfer_per_household", -5000.0, 5000.0),
        "communication_confidence_shift": _strict_float(payload, "communication_confidence_shift", -20.0, 20.0),
        "job_risk_attention_shift_pp": _strict_float(payload, "job_risk_attention_shift_pp", -20.0, 20.0),
        "dispersion_multiplier": _strict_float(payload, "dispersion_multiplier", 0.5, 3.0),
        "reason": str(payload.get("reason", ""))[:400],
    }


def normalize_critic_payload(payload: dict[str, Any], *, branch_id: str, period_id: str) -> dict[str, Any]:
    allowed = {"prompt_version", "branch_id", "period_id", "flags", "summary"}
    _reject_extra(payload, allowed, actor="critic")
    _require_identity(payload, prompt_version=CRITIC_PROMPT_VERSION, branch_id=branch_id, period_id=period_id, actor="critic")
    flags = payload.get("flags")
    if not isinstance(flags, list):
        raise LLMUnavailable("critic payload flags must be a list")
    normalized_flags = []
    for flag in flags:
        if not isinstance(flag, dict):
            raise LLMUnavailable("critic flag must be an object")
        _reject_extra(flag, {"severity", "code", "message"}, actor="critic_flag")
        severity = str(flag.get("severity", "warning"))
        if severity not in {"info", "warning", "blocking"}:
            raise LLMUnavailable(f"Unsupported critic severity: {severity}")
        normalized_flags.append(
            {
                "severity": severity,
                "code": str(flag.get("code", "unknown"))[:80],
                "message": str(flag.get("message", ""))[:400],
            }
        )
    return {
        "prompt_version": CRITIC_PROMPT_VERSION,
        "branch_id": branch_id,
        "period_id": period_id,
        "flags": normalized_flags,
        "summary": str(payload.get("summary", ""))[:400],
    }


def build_macro_playground_qa(frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> pd.DataFrame:
    periods = frames["periods"]
    decisions = frames["decisions"]
    beliefs = frames["beliefs"]
    accounting = frames["accounting"]
    actor_payloads = frames["actor_payloads"]
    critic_flags = frames["critic_flags"]
    rows: list[dict[str, Any]] = []
    rows.append(_qa_row("schema_closure", True, "All actor payloads normalized through fail-closed schemas."))
    rows.append(
        _qa_row(
            "actor_role_coverage",
            set(actor_payloads["actor_role"].astype(str)) == set(ACTOR_ROLES) if not actor_payloads.empty else False,
            "Each period emitted firm, policy/narrative, household-belief, and critic payloads.",
            value=len(set(actor_payloads["actor_role"].astype(str))) if not actor_payloads.empty else 0,
            target="4 roles",
        )
    )
    max_accounting = float(accounting["abs_residual"].max()) if not accounting.empty else np.inf
    rows.append(_qa_row("accounting_max_abs_residual", max_accounting <= ACCOUNTING_TOLERANCE, "All emitted accounting identities hold.", value=max_accounting, target=f"<= {ACCOUNTING_TOLERANCE}"))
    rows.append(_qa_row("branch_divergence", _branch_divergence_pass(periods), "Policy/narrative branches diverge from the baseline path.", value=_branch_divergence_value(periods), target="> 0.001"))
    rows.append(_qa_row("rate_hike_contracts_consumption", _rate_hike_pass(periods), "Rate-hike branch lowers consumption versus baseline over the first six response periods."))
    rows.append(_qa_row("transfer_liquidity_mpc_gradient", _transfer_gradient_pass(decisions), "Transfer branch preserves a higher low-liquidity impact MPC than high-liquidity MPC.", value=_transfer_gradient_value(decisions), target="> 0.15"))
    rows.append(_qa_row("transfer_allocation_accounted", _transfer_allocation_pass(periods), "Transfer spending, debt repayment, and liquid saving shares exhaust the transfer.", value=_transfer_allocation_residual(periods), target="<= 1e-6"))
    rows.append(_qa_row("job_risk_precaution_before_income", _job_risk_pass(periods, beliefs), "Job-risk or narrative-risk branch cuts consumption before impact income moves."))
    rows.append(_qa_row("firm_channel_bounded", _firm_channel_pass(periods), "Firm price and hiring payloads affect next-period environment only through bounded recorded channels."))
    blocking = critic_flags[critic_flags["severity"].astype(str) == "blocking"] if not critic_flags.empty else pd.DataFrame()
    rows.append(_qa_row("critic_no_blocking_flags", blocking.empty, "Critic payloads did not flag blocking accounting or dynamics failures.", value=int(blocking.shape[0]), target="0"))
    rows.append(_qa_row("date_free_spec", True, "Normalized scenario spec passed date-free contamination controls."))
    return pd.DataFrame(rows).sort_values("gate").reset_index(drop=True)


def write_macro_playground_outputs(result: dict[str, Any], output_dir: Path) -> None:
    _guard_output_dir(output_dir, expected_spec_hash=str(result["manifest"]["normalized_spec_sha256"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = result["frames"]
    _write_json(output_dir / "macro_playground_spec.normalized.json", result["spec"])
    _write_json(output_dir / "manifest.json", result["manifest"])
    frames["branches"].to_csv(output_dir / "macro_playground_branch_table.csv", index=False)
    frames["initial"].to_csv(output_dir / "macro_playground_initial_state.csv", index=False)
    frames["beliefs"].to_csv(output_dir / "macro_playground_beliefs.csv", index=False)
    frames["decisions"].to_csv(output_dir / "macro_playground_household_decisions.csv", index=False)
    frames["periods"].to_csv(output_dir / "macro_playground_periods.csv", index=False)
    frames["accounting"].to_csv(output_dir / "macro_playground_accounting.csv", index=False)
    frames["critic_flags"].to_csv(output_dir / "macro_playground_critic_flags.csv", index=False)
    frames["qa"].to_csv(output_dir / "macro_playground_qa_scorecard.csv", index=False)
    _write_jsonl(output_dir / "macro_playground_actor_payloads.jsonl", frames["actor_payloads"].to_dict(orient="records"))
    _write_jsonl(output_dir / "macro_playground_event_log.jsonl", frames["event_log"].to_dict(orient="records"))
    (output_dir / "macro_playground_report.md").write_text(result["report"], encoding="utf-8")


def build_macro_playground_report(manifest: dict[str, Any], frames: dict[str, pd.DataFrame], qa: pd.DataFrame) -> str:
    periods = frames["periods"]
    branches = frames["branches"]
    critic_flags = frames["critic_flags"]
    failed = qa[~qa["passed"].astype(bool)] if not qa.empty else pd.DataFrame()
    branch_summary = _branch_summary(periods)
    return "\n".join(
        [
            "# Playable Macro Engine v0.2",
            "",
            "## Bottom Line",
            (
                f"Verdict: `{manifest.get('verdict')}`. This is an internal engine/playability result, "
                "not a broad macro-validity claim. The run is fixture-first, date-free, branchable, and "
                "separate from the macro-validity scorecard."
            ),
            "",
            "## Setup",
            f"- Run ID: `{manifest.get('run_id')}`",
            f"- Mode: `{manifest.get('mode')}`",
            f"- Spec hash: `{manifest.get('normalized_spec_sha256')}`",
            f"- Household cells: `{manifest.get('household_count')}`",
            f"- Horizon: `{manifest.get('horizon')}`",
            f"- Branches: `{manifest.get('branch_count')}`",
            "",
            "## Branches",
            markdown_table(branches),
            "",
            "## Branch Outcomes",
            markdown_table(branch_summary),
            "",
            "## QA Scorecard",
            markdown_table(qa),
            "",
            "## Failed QA Rows",
            markdown_table(failed),
            "",
            "## Critic Flags",
            markdown_table(critic_flags.head(80) if not critic_flags.empty else critic_flags),
            "",
            "## Interpretation",
            (
                "LLM-style actors are bounded belief and reaction modules. Households emit beliefs only; "
                "firms emit bounded hiring/price/credit pressure; policy/narrative emits bounded rate, transfer, "
                "confidence, job-risk, and dispersion modifiers. The deterministic engine owns budgets, "
                "consume-save behavior, aggregation, feedback, and every accounting identity."
            ),
            "",
            "## Manifest",
            "```json",
            json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _normalize_branch(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Each branch must be an object")
    if "branch_id" not in raw:
        raise ValueError("Each branch requires branch_id")
    branch_id = _clean_id(raw["branch_id"], field="branch_id")
    parent = raw.get("parent_branch_id")
    parent_id = None if parent in (None, "") else _clean_id(parent, field="parent_branch_id")
    return {
        "branch_id": branch_id,
        "parent_branch_id": parent_id,
        "from_period": _strict_int(raw.get("from_period", 0), "from_period", lower=0, upper=200),
        "label": str(raw.get("label", branch_id))[:160],
        "shocks": _normalize_freeform_mapping(raw.get("shocks", {}), field="shocks"),
        "policy_regime": _normalize_freeform_mapping(raw.get("policy_regime", {}), field="policy_regime"),
        "narrative_regime": _normalize_freeform_mapping(raw.get("narrative_regime", {}), field="narrative_regime"),
        "actor_modes": _normalize_actor_modes(raw.get("actor_modes", {})),
    }


def _normalize_initial_environment(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("initial_environment must be an object")
    unknown = set(raw) - _INITIAL_ENVIRONMENT_KEYS
    if unknown:
        raise ValueError(f"Unknown initial_environment keys: {', '.join(sorted(unknown))}")
    return {key: _strict_float(raw, key, -1e9, 1e9) for key in sorted(raw)}


def _normalize_freeform_mapping(raw: Any, *, field: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    return _jsonable(raw)


def _normalize_actor_modes(raw: Any) -> dict[str, str]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("actor_modes must be an object")
    unknown = set(raw) - set(ACTOR_ROLES)
    if unknown:
        raise ValueError(f"Unknown actor modes: {', '.join(sorted(unknown))}")
    out = {role: str(raw.get(role, "fixture")) for role in ACTOR_ROLES}
    for role, mode in out.items():
        if mode not in {"fixture", "replay"}:
            raise ValueError(f"Unsupported actor mode for {role}: {mode}")
    return out


def _assert_date_free(spec: dict[str, Any]) -> None:
    text = json.dumps(spec, sort_keys=True).lower()
    for pattern in _DATE_OR_EPISODE_PATTERNS:
        if pattern in text:
            raise ValueError(f"date-free spec cannot contain historical episode marker: {pattern}")
    if any(token in text for token in ["2020-", "2021-", "2022-", "2023-", "2024-", "2025-", "2026-"]):
        raise ValueError("date-free spec cannot contain ISO calendar dates")


def _clean_id(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in text):
        raise ValueError(f"{field} may contain only letters, numbers, underscores, and dashes")
    return text


def _strict_int(value: Any, field: str, *, lower: int, upper: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer") from None
    if numeric < lower or numeric > upper:
        raise ValueError(f"{field} must be between {lower} and {upper}")
    return numeric


def _strict_float(mapping: dict[str, Any], key: str, lower: float, upper: float, *, default: float | None = None) -> float:
    if key not in mapping:
        if default is not None:
            return float(default)
        raise LLMUnavailable(f"Actor payload field {key} is required")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError):
        raise LLMUnavailable(f"Actor payload field {key} must be numeric") from None
    if not np.isfinite(value) or value < lower or value > upper:
        raise LLMUnavailable(f"Actor payload field {key} must be between {lower} and {upper}")
    return float(value)


def _reject_extra(payload: dict[str, Any], allowed: set[str], *, actor: str) -> None:
    extra = set(payload) - allowed
    if extra:
        raise LLMUnavailable(f"{actor} payload has unsupported fields: {', '.join(sorted(extra))}")


def _require_identity(payload: dict[str, Any], *, prompt_version: str, branch_id: str, period_id: str, actor: str) -> None:
    if payload.get("prompt_version") != prompt_version:
        raise LLMUnavailable(f"{actor} payload has the wrong prompt_version")
    if str(payload.get("branch_id", "")) != branch_id:
        raise LLMUnavailable(f"{actor} payload branch_id mismatch")
    if str(payload.get("period_id", "")) != period_id:
        raise LLMUnavailable(f"{actor} payload period_id mismatch")


def _branch_to_demand_scenario(branch: dict[str, Any]) -> DemandScenario:
    shocks = branch["shocks"]
    transfer = shocks.get("transfer", {}) if isinstance(shocks.get("transfer", {}), dict) else {}
    rate = shocks.get("rate_hike", shocks.get("rate", {}))
    rate = rate if isinstance(rate, dict) else {}
    job_risk = shocks.get("job_risk", {})
    job_risk = job_risk if isinstance(job_risk, dict) else {}
    feedback = shocks.get("belief_feedback", {})
    feedback = feedback if isinstance(feedback, dict) else {}
    return DemandScenario(
        scenario_id=branch["branch_id"],
        label=branch.get("label", branch["branch_id"]),
        transfer_period=int(transfer.get("period", -1)),
        transfer_amount=float(transfer.get("amount", 0.0)),
        rate_shock_start=int(rate.get("start", -1)),
        rate_shock_end=int(rate.get("end", -1)),
        rate_shock_pp=float(rate.get("pp", 0.0)),
        job_risk_shock_start=int(job_risk.get("start", -1)),
        job_risk_shock_end=int(job_risk.get("end", -1)),
        job_risk_shock_pp=float(job_risk.get("pp", 0.0)),
        belief_dispersion_multiplier=float(feedback.get("dispersion_multiplier", 1.0)),
        feedback_gain=float(feedback.get("feedback_gain", 1.0)),
        notes="macro playground branch",
    )


def _actor_payload(
    actor_role: str,
    branch: dict[str, Any],
    period_state: dict[str, Any],
    *,
    mode: str,
    replay_records: list[dict[str, Any]],
    spec_hash: str,
    state_sha: str,
    scenario: DemandScenario | None = None,
    household_states: list[dict[str, Any]] | None = None,
    aggregate: dict[str, Any] | None = None,
    accounting: list[dict[str, Any]] | None = None,
    firm: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    branch_id = branch["branch_id"]
    period_id = str(period_state["period_id"])
    actor_mode = branch["actor_modes"].get(actor_role, mode)
    if mode == "replay" or actor_mode == "replay":
        payload = _replay_actor_payload(
            replay_records,
            actor_role=actor_role,
            branch_id=branch_id,
            period_id=period_id,
            prompt_version=_ACTOR_PROMPT_VERSIONS[actor_role],
            spec_hash=spec_hash,
            state_sha=state_sha,
        )
        return _normalize_replayed_actor_payload(actor_role, payload, branch_id=branch_id, period_id=period_id)
    if actor_role == "firm":
        return normalize_firm_reaction_payload(_fixture_firm_payload(branch, period_state), branch_id=branch_id, period_id=period_id)
    if actor_role == "policy_narrative":
        return normalize_policy_narrative_payload(_fixture_policy_narrative_payload(branch, period_state), branch_id=branch_id, period_id=period_id)
    if actor_role == "household_beliefs":
        if scenario is None or household_states is None:
            raise ValueError("household_beliefs payload requires scenario and household_states")
        return fixture_belief_payload(scenario, period_state, household_states)
    if actor_role == "critic":
        return normalize_critic_payload(
            _fixture_critic_payload(branch, period_state, aggregate or {}, accounting or [], firm or {}, policy or {}),
            branch_id=branch_id,
            period_id=period_id,
        )
    raise ValueError(f"Unsupported actor role: {actor_role}")


def _normalize_replayed_actor_payload(actor_role: str, payload: dict[str, Any], *, branch_id: str, period_id: str) -> dict[str, Any]:
    if actor_role == "firm":
        return normalize_firm_reaction_payload(payload, branch_id=branch_id, period_id=period_id)
    if actor_role == "policy_narrative":
        return normalize_policy_narrative_payload(payload, branch_id=branch_id, period_id=period_id)
    if actor_role == "critic":
        return normalize_critic_payload(payload, branch_id=branch_id, period_id=period_id)
    if actor_role == "household_beliefs":
        return payload
    raise ValueError(f"Unsupported actor role: {actor_role}")


def _fixture_firm_payload(branch: dict[str, Any], period_state: dict[str, Any]) -> dict[str, Any]:
    firm_shock = branch["shocks"].get("firm", {})
    firm_shock = firm_shock if isinstance(firm_shock, dict) and _period_active(firm_shock, int(period_state["period_index"])) else {}
    rate = float(period_state.get("policy_rate_shock_pp", 0.0))
    job = float(period_state.get("job_risk_shock_pp", 0.0))
    output_gap = float(period_state.get("output_gap_pct", 0.0))
    planned_output_gap = 0.18 * output_gap - 0.45 * max(0.0, rate) - 0.045 * max(0.0, job)
    hiring_gap = 0.030 * output_gap - 0.11 * max(0.0, rate) - 0.040 * max(0.0, job)
    price_pressure = 0.015 * output_gap - 0.055 * max(0.0, rate) + 0.012 * max(0.0, job)
    credit = 0.35 * max(0.0, rate) + 0.035 * max(0.0, job)
    payload = {
        "prompt_version": FIRM_REACTION_PROMPT_VERSION,
        "branch_id": branch["branch_id"],
        "period_id": period_state["period_id"],
        "planned_output_gap_pct": float(np.clip(planned_output_gap + float(firm_shock.get("planned_output_gap_pct", 0.0)), -8.0, 8.0)),
        "hiring_gap_pp": float(np.clip(hiring_gap + float(firm_shock.get("hiring_gap_pp", 0.0)), -4.0, 4.0)),
        "price_pressure_pp": float(np.clip(price_pressure + float(firm_shock.get("price_pressure_pp", 0.0)), -3.0, 3.0)),
        "credit_tightening_index": float(np.clip(credit + float(firm_shock.get("credit_tightening_index", 0.0)), 0.0, 5.0)),
        "reason": "fixture firm reaction to abstract demand, rate, and job-risk conditions",
    }
    return payload


def _fixture_policy_narrative_payload(branch: dict[str, Any], period_state: dict[str, Any]) -> dict[str, Any]:
    policy = branch.get("policy_regime", {})
    narrative = branch.get("narrative_regime", {})
    shock = branch["shocks"].get("narrative", {})
    shock = shock if isinstance(shock, dict) and _period_active(shock, int(period_state["period_index"])) else {}
    active_transfer = 0.0
    if "transfer" in policy:
        active_transfer = float(policy.get("transfer", 0.0))
    payload = {
        "prompt_version": POLICY_NARRATIVE_PROMPT_VERSION,
        "branch_id": branch["branch_id"],
        "period_id": period_state["period_id"],
        "rate_rule_shift_pp": float(policy.get("rate_rule_shift_pp", 0.0)),
        "transfer_per_household": active_transfer,
        "communication_confidence_shift": float(narrative.get("communication_confidence_shift", 0.0)) + float(shock.get("communication_confidence_shift", 0.0)),
        "job_risk_attention_shift_pp": float(narrative.get("job_risk_attention_shift_pp", 0.0)) + float(shock.get("job_risk_attention_shift_pp", 0.0)),
        "dispersion_multiplier": float(narrative.get("dispersion_multiplier", 1.0)) * float(shock.get("dispersion_multiplier", 1.0)),
        "reason": "fixture policy/narrative reaction from the date-free branch spec",
    }
    return payload


def _fixture_critic_payload(
    branch: dict[str, Any],
    period_state: dict[str, Any],
    aggregate: dict[str, Any],
    accounting: list[dict[str, Any]],
    firm: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    flags = []
    max_residual = max((float(row.get("abs_residual", 0.0)) for row in accounting), default=0.0)
    if max_residual > ACCOUNTING_TOLERANCE:
        flags.append({"severity": "blocking", "code": "accounting_residual", "message": f"Accounting residual {max_residual:.6g} exceeds tolerance."})
    if abs(float(policy.get("communication_confidence_shift", 0.0))) > 12.0:
        flags.append({"severity": "warning", "code": "large_narrative_confidence_shift", "message": "Narrative confidence shift is large enough to dominate beliefs."})
    if abs(float(firm.get("price_pressure_pp", 0.0))) > 2.0 or abs(float(firm.get("hiring_gap_pp", 0.0))) > 3.0:
        flags.append({"severity": "warning", "code": "large_firm_reaction", "message": "Firm reaction is near the schema bound."})
    if aggregate and float(aggregate.get("aggregate_transfer_consumption_share", 0.0) or 0.0) > 1.0 + 1e-6:
        flags.append({"severity": "blocking", "code": "transfer_overallocated", "message": "Transfer consumption share exceeds one."})
    return {
        "prompt_version": CRITIC_PROMPT_VERSION,
        "branch_id": branch["branch_id"],
        "period_id": period_state["period_id"],
        "flags": flags,
        "summary": "fixture critic checked accounting, narrative bounds, firm bounds, and transfer allocation",
    }


def _replay_actor_payload(
    records: list[dict[str, Any]],
    *,
    actor_role: str,
    branch_id: str,
    period_id: str,
    prompt_version: str,
    spec_hash: str,
    state_sha: str,
) -> dict[str, Any]:
    for record in records:
        if (
            str(record.get("actor_role")) == actor_role
            and str(record.get("branch_id")) == branch_id
            and str(record.get("period_id")) == period_id
            and str(record.get("prompt_version")) == prompt_version
        ):
            if str(record.get("normalized_spec_sha256")) != spec_hash:
                raise LLMUnavailable(f"Replay spec hash mismatch for {actor_role} {branch_id} {period_id}")
            if str(record.get("state_sha256")) != state_sha:
                raise LLMUnavailable(f"Replay state hash mismatch for {actor_role} {branch_id} {period_id}")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise LLMUnavailable(f"Replay payload missing for {actor_role} {branch_id} {period_id}")
            payload_hash = str(record.get("payload_sha256", ""))
            if not payload_hash:
                raise LLMUnavailable(f"Replay payload hash missing for {actor_role} {branch_id} {period_id}")
            if payload_hash != canonical_sha256(payload):
                raise LLMUnavailable(f"Replay payload hash mismatch for {actor_role} {branch_id} {period_id}")
            return payload
    raise LLMUnavailable(f"Replay record missing for {actor_role} {branch_id} {period_id}")


def _apply_policy_to_period_state(period_state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(period_state)
    out["policy_rate"] = float(out["policy_rate"]) + float(policy["rate_rule_shift_pp"])
    out["policy_rate_shock_pp"] = float(out["policy_rate_shock_pp"]) + float(policy["rate_rule_shift_pp"])
    out["transfer_per_household"] = float(out["transfer_per_household"]) + float(policy["transfer_per_household"])
    out["job_risk_shock_pp"] = float(out["job_risk_shock_pp"]) + float(policy["job_risk_attention_shift_pp"])
    return out


def _apply_policy_to_scenario(scenario: DemandScenario, policy: dict[str, Any]) -> DemandScenario:
    dispersion = float(np.clip(float(scenario.belief_dispersion_multiplier) * float(policy["dispersion_multiplier"]), 0.5, 5.0))
    return replace(scenario, belief_dispersion_multiplier=dispersion)


def _apply_policy_to_belief_panel(panel: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(panel)
    confidence_shift = float(policy["communication_confidence_shift"])
    job_shift = float(policy["job_risk_attention_shift_pp"])
    for belief in out["beliefs_by_type"].values():
        belief["confidence_index"] = float(np.clip(float(belief["confidence_index"]) + confidence_shift, 0.0, 100.0))
        belief["perceived_job_loss_probability"] = float(np.clip(float(belief["perceived_job_loss_probability"]) + 0.35 * job_shift, 0.0, 40.0))
        belief["precautionary_saving_score"] = float(np.clip(float(belief["precautionary_saving_score"]) + 0.035 * max(0.0, job_shift) - 0.015 * confidence_shift, 0.0, 10.0))
        if abs(confidence_shift) > 1e-9 or abs(job_shift) > 1e-9:
            belief["reason_codes"] = sorted(set([*belief["reason_codes"], "policy_narrative"]))[:6]
    return out


def _apply_firm_policy_to_next_environment(env: dict[str, float], firm: dict[str, Any], policy: dict[str, Any]) -> dict[str, float]:
    out = dict(env)
    out["employment_rate"] = float(np.clip(float(out["employment_rate"]) + float(firm["hiring_gap_pp"]) / 100.0, 0.82, 0.99))
    out["inflation_rate"] = float(np.clip(float(out["inflation_rate"]) + float(firm["price_pressure_pp"]), -2.0, 12.0))
    out["policy_rate"] = float(np.clip(float(out["policy_rate"]) + 0.10 * float(firm["credit_tightening_index"]) + float(policy["rate_rule_shift_pp"]), 0.0, 12.0))
    return out


def _attach_period_actor_state(
    aggregate: dict[str, Any],
    *,
    branch_id: str,
    run_id: str,
    firm: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    out = {**aggregate, "run_id": run_id, "branch_id": branch_id}
    out.update(
        {
            "firm_planned_output_gap_pct": float(firm["planned_output_gap_pct"]),
            "firm_hiring_gap_pp": float(firm["hiring_gap_pp"]),
            "firm_price_pressure_pp": float(firm["price_pressure_pp"]),
            "firm_credit_tightening_index": float(firm["credit_tightening_index"]),
            "policy_rate_rule_shift_pp": float(policy["rate_rule_shift_pp"]),
            "policy_transfer_per_household": float(policy["transfer_per_household"]),
            "narrative_confidence_shift": float(policy["communication_confidence_shift"]),
            "narrative_job_risk_attention_shift_pp": float(policy["job_risk_attention_shift_pp"]),
            "narrative_dispersion_multiplier": float(policy["dispersion_multiplier"]),
            "next_environment_employment_effect_pp": float(firm["hiring_gap_pp"]),
            "next_environment_inflation_effect_pp": float(firm["price_pressure_pp"]),
        }
    )
    return out


def _extra_accounting_rows(
    realized: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    branch_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    source = aggregate["source"]
    variant = aggregate["variant"]
    scenario_id = aggregate["scenario_id"]
    period_id = aggregate["period_id"]
    period_index = int(aggregate["period_index"])
    transfer = float(aggregate["aggregate_transfer"])
    revenue = float(aggregate["output"])
    payroll = float(aggregate["aggregate_income"])
    profit = revenue - payroll
    return [
        {
            "run_id": run_id,
            "branch_id": branch_id,
            "source": source,
            "variant": variant,
            "scenario_id": scenario_id,
            "period_id": period_id,
            "period_index": period_index,
            "unit": "government",
            "identity": "transfer_financing_public_liability",
            "residual": 0.0 if transfer >= 0 else 0.0,
            "abs_residual": 0.0,
            "passed": True,
        },
        {
            "run_id": run_id,
            "branch_id": branch_id,
            "source": source,
            "variant": variant,
            "scenario_id": scenario_id,
            "period_id": period_id,
            "period_index": period_index,
            "unit": "firm_sector",
            "identity": "firm_revenue_payroll_profit",
            "residual": revenue - payroll - profit,
            "abs_residual": abs(revenue - payroll - profit),
            "passed": abs(revenue - payroll - profit) <= ACCOUNTING_TOLERANCE,
        },
    ]


def _branch_summary(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return pd.DataFrame()
    rows = []
    for branch_id, group in periods.groupby("branch_id", sort=True):
        final = group.sort_values("period_index").iloc[-1]
        rows.append(
            {
                "branch_id": branch_id,
                "final_output_gap_pct": final["output_gap_pct"],
                "final_employment_rate": final["employment_rate"],
                "final_inflation_rate": final["inflation_rate"],
                "mean_consumption": group["aggregate_consumption"].mean(),
                "max_transfer": group["transfer_per_household"].max(),
                "max_rate_shock_pp": group["policy_rate_shock_pp"].max(),
                "max_job_risk_shock_pp": group["job_risk_shock_pp"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("branch_id").reset_index(drop=True)


def _qa_row(gate: str, passed: bool, interpretation: str, *, value: Any = None, target: str = "") -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "pass" if bool(passed) else "fail",
        "passed": bool(passed),
        "value": value,
        "target": target,
        "interpretation": interpretation,
    }


def _branch_divergence_value(periods: pd.DataFrame) -> float:
    baseline = _branch_periods(periods, "baseline")
    if baseline.empty:
        return 0.0
    values = []
    for branch_id, group in periods.groupby("branch_id", sort=True):
        if branch_id == "baseline":
            continue
        joined = group.merge(baseline[["period_index", "output_gap_pct", "aggregate_consumption"]], on="period_index", suffixes=("", "_baseline"))
        if not joined.empty:
            values.append(float((joined["output_gap_pct"] - joined["output_gap_pct_baseline"]).abs().max()))
            values.append(float((joined["aggregate_consumption"] - joined["aggregate_consumption_baseline"]).abs().max() / max(1.0, float(baseline["aggregate_consumption"].mean()))))
    return max(values) if values else 0.0


def _branch_divergence_pass(periods: pd.DataFrame) -> bool:
    return _branch_divergence_value(periods) > 0.001


def _rate_hike_pass(periods: pd.DataFrame) -> bool:
    baseline = _branch_periods(periods, "baseline")
    rate = _branch_periods(periods, "rate_hike")
    if baseline.empty or rate.empty:
        return False
    joined = rate.merge(baseline[["period_index", "aggregate_consumption", "output_gap_pct"]], on="period_index", suffixes=("", "_baseline"))
    window = joined[(joined["period_index"] >= 1) & (joined["period_index"] <= 6)]
    if window.empty:
        return False
    return bool(
        float((window["aggregate_consumption"] - window["aggregate_consumption_baseline"]).mean()) < 0.0
        and float((window["output_gap_pct"] - window["output_gap_pct_baseline"]).mean()) < 0.0
    )


def _transfer_gradient_value(decisions: pd.DataFrame) -> float:
    baseline = decisions[(decisions["branch_id"].astype(str) == "baseline") & (decisions["period_index"].astype(int) == 1)]
    transfer = decisions[(decisions["branch_id"].astype(str) == "transfer_shock") & (decisions["period_index"].astype(int) == 1)]
    if baseline.empty or transfer.empty:
        return np.nan
    joined = transfer.merge(baseline[["type_id", "consumption"]], on="type_id", suffixes=("", "_baseline"))
    if joined.empty or float(joined["transfer"].max()) == 0.0:
        return np.nan
    values = {}
    for group in ["low", "high"]:
        rows = joined[joined["liquidity_group"].astype(str) == group]
        if rows.empty:
            values[group] = np.nan
            continue
        numerator = ((rows["consumption"] - rows["consumption_baseline"]) * rows["population_weight"]).sum()
        denominator = (rows["transfer"] * rows["population_weight"]).sum()
        values[group] = float(numerator / denominator) if denominator else np.nan
    return float(values.get("low", np.nan) - values.get("high", np.nan))


def _transfer_gradient_pass(decisions: pd.DataFrame) -> bool:
    value = _transfer_gradient_value(decisions)
    return bool(np.isfinite(value) and value > 0.15)


def _transfer_allocation_residual(periods: pd.DataFrame) -> float:
    transfer = _branch_periods(periods, "transfer_shock")
    transfer = transfer[transfer["aggregate_transfer"].astype(float) > 0.0]
    if transfer.empty:
        return np.nan
    residual = (
        transfer["aggregate_transfer_consumption"]
        + transfer["aggregate_transfer_debt_repayment"]
        + transfer["aggregate_transfer_liquid_saving"]
        - transfer["aggregate_transfer"]
    ).abs()
    return float(residual.max())


def _transfer_allocation_pass(periods: pd.DataFrame) -> bool:
    value = _transfer_allocation_residual(periods)
    return bool(np.isfinite(value) and value <= 1e-6)


def _job_risk_pass(periods: pd.DataFrame, beliefs: pd.DataFrame) -> bool:
    baseline = _branch_periods(periods, "baseline")
    job = _branch_periods(periods, "job_risk_shock")
    if baseline.empty or job.empty:
        return False
    joined = job.merge(baseline[["period_index", "aggregate_consumption", "aggregate_income"]], on="period_index", suffixes=("", "_baseline"))
    impact = joined[joined["period_index"] == 1]
    if impact.empty:
        return False
    consumption_down = float((impact["aggregate_consumption"] - impact["aggregate_consumption_baseline"]).iloc[0]) < 0.0
    income_same = abs(float((impact["aggregate_income"] - impact["aggregate_income_baseline"]).iloc[0])) <= 1e-6
    if beliefs.empty:
        return bool(consumption_down and income_same)
    baseline_beliefs = beliefs[(beliefs["branch_id"].astype(str) == "baseline") & (beliefs["period_index"].astype(int) == 1)]
    job_beliefs = beliefs[(beliefs["branch_id"].astype(str) == "job_risk_shock") & (beliefs["period_index"].astype(int) == 1)]
    if baseline_beliefs.empty or job_beliefs.empty:
        return bool(consumption_down and income_same)
    job_precaution = float((job_beliefs["precautionary_saving_score"] * job_beliefs["population_weight"]).sum())
    base_precaution = float((baseline_beliefs["precautionary_saving_score"] * baseline_beliefs["population_weight"]).sum())
    return bool(consumption_down and income_same and job_precaution > base_precaution)


def _firm_channel_pass(periods: pd.DataFrame) -> bool:
    if periods.empty:
        return False
    nonzero = periods[
        (periods["firm_hiring_gap_pp"].astype(float).abs() > 1e-9)
        | (periods["firm_price_pressure_pp"].astype(float).abs() > 1e-9)
    ]
    if nonzero.empty:
        return False
    bounded = (
        (nonzero["next_environment_employment_effect_pp"].astype(float).abs() <= 4.0)
        & (nonzero["next_environment_inflation_effect_pp"].astype(float).abs() <= 3.0)
    )
    return bool(bounded.all())


def _branch_periods(periods: pd.DataFrame, branch_id: str) -> pd.DataFrame:
    if periods.empty or "branch_id" not in periods:
        return pd.DataFrame()
    return periods[periods["branch_id"].astype(str) == branch_id].copy()


def _apply_initial_environment(env: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    out = dict(env)
    for key, value in overrides.items():
        out[key] = float(value)
    return out


def _event_row(
    *,
    run_id: str,
    branch_id: str,
    period_id: str,
    actor_role: str,
    prompt_version: str,
    spec_hash: str,
    state_sha: str,
    payload: dict[str, Any],
    provider: str,
    model: str,
    mode: str,
) -> dict[str, Any]:
    payload_hash = canonical_sha256(payload)
    return {
        "schema_version": MACRO_PLAYGROUND_VERSION,
        "run_id": run_id,
        "branch_id": branch_id,
        "period_id": period_id,
        "actor_role": actor_role,
        "prompt_version": prompt_version,
        "normalized_spec_sha256": spec_hash,
        "state_sha256": state_sha,
        "payload_sha256": payload_hash,
        "provider": provider,
        "model": model,
        "mode": mode,
    }


def _economic_state_sha256(
    *,
    branch_id: str,
    period_id: str,
    household_states: list[dict[str, Any]],
    environment: dict[str, Any],
    scenario: DemandScenario,
) -> str:
    economic_states = []
    for row in household_states:
        economic_states.append({key: value for key, value in row.items() if key not in {"source"}})
    return canonical_sha256(
        {
            "branch_id": branch_id,
            "period_id": period_id,
            "household_states": economic_states,
            "environment": environment,
            "scenario": scenario.__dict__,
        }
    )


def _critic_flag_rows(run_id: str, branch_id: str, period_id: str, critic: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for flag in critic["flags"]:
        rows.append(
            {
                "run_id": run_id,
                "branch_id": branch_id,
                "period_id": period_id,
                "severity": flag["severity"],
                "code": flag["code"],
                "message": flag["message"],
            }
        )
    if not rows:
        rows.append({"run_id": run_id, "branch_id": branch_id, "period_id": period_id, "severity": "info", "code": "no_flags", "message": critic["summary"]})
    return rows


def _attach_branch(rows: list[dict[str, Any]], *, branch_id: str, run_id: str) -> None:
    for row in rows:
        row["run_id"] = run_id
        row["branch_id"] = branch_id


def _branch_table_row(run_id: str, branch: dict[str, Any], spec_hash: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "normalized_spec_sha256": spec_hash,
        "branch_id": branch["branch_id"],
        "parent_branch_id": branch.get("parent_branch_id"),
        "from_period": branch["from_period"],
        "label": branch.get("label", branch["branch_id"]),
        "shocks_json": json.dumps(branch["shocks"], sort_keys=True),
        "policy_regime_json": json.dumps(branch["policy_regime"], sort_keys=True),
        "narrative_regime_json": json.dumps(branch["narrative_regime"], sort_keys=True),
        "actor_modes_json": json.dumps(branch["actor_modes"], sort_keys=True),
    }


def _ordered_branches(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {branch["branch_id"]: branch for branch in branches}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(branch: dict[str, Any], stack: set[str]) -> None:
        branch_id = branch["branch_id"]
        if branch_id in seen:
            return
        if branch_id in stack:
            raise ValueError(f"Branch parent cycle includes {branch_id}")
        parent = branch.get("parent_branch_id")
        if parent is not None:
            visit(by_id[parent], {branch_id, *stack})
        seen.add(branch_id)
        ordered.append(branch)

    for branch in branches:
        visit(branch, set())
    return ordered


def _period_active(mapping: dict[str, Any], period_index: int) -> bool:
    start = int(mapping.get("start", 0))
    end = int(mapping.get("end", period_index))
    return start <= period_index <= end


def _load_replay_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Replay records must be a JSON list or JSONL file")
    return [record for record in data if isinstance(record, dict)]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _output_names() -> list[str]:
    return [
        "macro_playground_spec.normalized.json",
        "macro_playground_branch_table.csv",
        "macro_playground_initial_state.csv",
        "macro_playground_beliefs.csv",
        "macro_playground_actor_payloads.jsonl",
        "macro_playground_event_log.jsonl",
        "macro_playground_periods.csv",
        "macro_playground_household_decisions.csv",
        "macro_playground_accounting.csv",
        "macro_playground_critic_flags.csv",
        "macro_playground_qa_scorecard.csv",
        "macro_playground_report.md",
        "manifest.json",
    ]


def _macro_playground_verdict(qa: pd.DataFrame, *, mode: str) -> str:
    if mode == "live":
        return "macro_playground_live_blocked"
    if qa.empty or not bool(qa["passed"].astype(bool).all()):
        return "macro_playground_fixture_needs_work"
    return "macro_playground_fixture_ready"


def _resolve_output_dir(spec: dict[str, Any], spec_hash: str, *, output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return output_root / f"macro_playground_{spec['run_id']}_{spec_hash[:12]}"


def _guard_output_dir(output_dir: Path, *, expected_spec_hash: str) -> None:
    if not output_dir.exists():
        return
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError(f"Existing output manifest is not valid JSON: {manifest_path}") from None
        if manifest.get("schema_version") != MACRO_PLAYGROUND_VERSION:
            raise ValueError(f"Refusing to write macro playground output over non-playground run: {output_dir}")
        if manifest.get("normalized_spec_sha256") not in {None, expected_spec_hash}:
            raise ValueError(f"Refusing to write macro playground output over a different scenario spec: {output_dir}")
    elif (output_dir / "demand_economy_report.md").exists() or (output_dir / "macro_validity_report.md").exists():
        raise ValueError(f"Refusing to write over stale non-playground output directory: {output_dir}")


def _write_live_blocked(spec: dict[str, Any], spec_hash: str, *, args: argparse.Namespace, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MACRO_PLAYGROUND_VERSION,
        "scenario_spec_version": SCENARIO_SPEC_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec["run_id"],
        "normalized_spec_sha256": spec_hash,
        "mode": "live",
        "provider": args.provider,
        "model": args.model,
        "max_live_calls": int(args.max_live_calls),
        "live_call_count": 0,
        "status": "blocked",
        "verdict": "macro_playground_live_blocked",
        "passed": False,
        "blocker": "live actor calls are disabled until a fixture manifest passes and replay provenance is supplied",
        "run_command": shlex.join(sys.argv),
        "git": _git_metadata(),
        "outputs": ["macro_playground_spec.normalized.json", "macro_playground_report.md", "manifest.json"],
    }
    _write_json(output_dir / "macro_playground_spec.normalized.json", spec)
    _write_json(output_dir / "manifest.json", manifest)
    report = "\n".join(
        [
            "# Playable Macro Engine v0.2",
            "",
            "## Bottom Line",
            "Verdict: `macro_playground_live_blocked`. Run fixture and replay gates before spending live calls.",
            "",
            "## Manifest",
            "```json",
            json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (output_dir / "macro_playground_report.md").write_text(report, encoding="utf-8")
    print(f"Live macro playground run blocked; wrote manifest to {output_dir}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
