"""Run one recursive firm-income feedback period after a household ecology run."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .ecology import (
    PROJECT_ROOT,
    _artifact_sha256,
    _file_sha256,
    _git_state,
    _history_materialization_binding,
    _initial_credit,
    _jsonable,
    _source_sha256,
    _state_from_row,
    _write_json,
)
from .ecology_engine import run_monthly_ecology
from .ecology_feedback import (
    AggregateFirmPeriodOne,
    build_simulated_environment_payload,
    compute_aggregate_firm_feedback,
)
from .ecology_history import ECOLOGY_HISTORY_SCHEMA_VERSION
from .ecology_households import (
    HOUSEHOLD_PROMPT_VERSION,
    HouseholdElicitor,
    LiveCallBudget,
    canonical_sha256,
    household_card,
    normalize_household_payload,
)
from .ecology_models import EmployerState, HouseholdState
from .ecology_transition import transition_household_states


SCHEMA_VERSION = "household_ecology_two_period_feedback_v3"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-1-run", type=Path, required=True)
    parser.add_argument("--households", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--mode", choices=("live", "replay"), required=True)
    parser.add_argument("--provider", choices=("codex_cli",), default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _population_masses(states: list[Any]) -> dict[str, float]:
    total = sum(row.population_weight for row in states)
    if total <= 0.0:
        raise ValueError("household population weights must sum to a positive value")
    return {
        row.household_id: len(states) * row.population_weight / total for row in states
    }


def _settled_period_one(
    states: list[HouseholdState],
    decisions: pd.DataFrame,
    employer: dict[str, Any],
) -> AggregateFirmPeriodOne:
    masses = _population_masses(states)
    settled_dollars = sum(
        masses[str(row.household_id)] * float(row.consumption_usd)
        for row in decisions.itertuples(index=False)
    )
    price = float(employer["current_price_per_unit_usd"])
    settled_units = settled_dollars / max(price, 1e-12)
    if not math.isclose(
        settled_units,
        float(employer["units_sold"]),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("period-1 settled household demand disagrees with producer sales")
    return AggregateFirmPeriodOne(
        demand_units=settled_units,
        sales_units=float(employer["units_sold"]),
        opening_inventory_units=float(employer["inventory_start_units"]),
        closing_inventory_units=float(employer["inventory_end_units"]),
        base_output_units=float(employer["output_units"]),
        productivity_index=1.0,
        producer_employment_index=1.0,
        producer_wage_index=1.0,
    )


def _validated_parent_artifact(
    run_dir: Path,
    manifest: dict[str, Any],
    name: str,
) -> Path:
    path = run_dir / name
    expected = manifest.get("artifacts", {}).get(name)
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"period-1 manifest does not bind {name}")
    if not path.is_file() or _artifact_sha256(path) != expected:
        raise ValueError(f"period-1 artifact hash mismatch: {name}")
    return path


def _period_one_history_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    binding = manifest.get("history_materialization")
    required = {
        "schema_version",
        "manifest_sha256",
        "history_sha256",
        "through_event_month",
        "publication_lag_months",
        "input_provenance",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("period-1 manifest requires complete history materialization provenance")
    if binding.get("schema_version") != ECOLOGY_HISTORY_SCHEMA_VERSION:
        raise ValueError("period-1 history materialization schema is unsupported")
    for field in ("manifest_sha256", "history_sha256"):
        if not isinstance(binding.get(field), str) or len(binding[field]) != 64:
            raise ValueError("period-1 history materialization hashes are malformed")
    if (
        not isinstance(binding.get("through_event_month"), str)
        or not binding["through_event_month"].strip()
        or not isinstance(binding.get("publication_lag_months"), int)
        or binding["publication_lag_months"] < 0
        or not isinstance(binding.get("input_provenance"), dict)
    ):
        raise ValueError("period-1 history materialization provenance is malformed")
    return binding


def _load_period_one(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("accounting_passed"):
        raise ValueError("period-1 run must pass accounting")
    if manifest.get("household_prompt_version") != HOUSEHOLD_PROMPT_VERSION:
        raise ValueError("period-1 run prompt version does not match current feedback runner")
    _period_one_history_binding(manifest)
    economy_path = _validated_parent_artifact(run_dir, manifest, "median_economy.json")
    decisions_path = _validated_parent_artifact(
        run_dir, manifest, "household_decisions.csv"
    )
    origin_path = _validated_parent_artifact(
        run_dir, manifest, "normalized_origin_information.json"
    )
    economy = json.loads(economy_path.read_text(encoding="utf-8"))
    decisions = pd.read_csv(decisions_path)
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    decisions = decisions.loc[decisions["scenario"].astype(str).eq("median")].copy()
    if len(decisions) != int(manifest["household_count"]):
        raise ValueError("period-1 decisions must contain one median row per household")
    if decisions["household_id"].astype(str).duplicated().any():
        raise ValueError("period-1 household decisions contain duplicate identities")
    return manifest, economy, decisions, origin


def _validate_source_inputs(
    manifest: dict[str, Any],
    households: Path,
    history: Path,
) -> tuple[str, str]:
    household_input_sha256 = _file_sha256(households)
    history_input_sha256 = _file_sha256(history)
    if household_input_sha256 != manifest.get("household_input_sha256"):
        raise ValueError("household input does not match the period-1 run")
    if history_input_sha256 != manifest.get("history_input_sha256"):
        raise ValueError("history input does not match the period-1 run")
    if _history_materialization_binding(history, required=True) != _period_one_history_binding(
        manifest
    ):
        raise ValueError("history materialization does not match the period-1 run")
    return household_input_sha256, history_input_sha256


def _period_one_replay_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    replay_equivalence_sha256 = manifest.get("replay_equivalence_sha256")
    if not isinstance(replay_equivalence_sha256, str) or not replay_equivalence_sha256:
        raise ValueError("period-1 manifest requires replay equivalence provenance")
    artifacts = manifest.get("artifacts", {})
    consumed_artifacts = {
        name: artifacts.get(name)
        for name in (
            "median_economy.json",
            "household_decisions.csv",
            "normalized_origin_information.json",
        )
    }
    if not all(isinstance(value, str) and value for value in consumed_artifacts.values()):
        raise ValueError("period-1 manifest does not bind every consumed artifact")
    return {
        "replay_equivalence_sha256": replay_equivalence_sha256,
        "consumed_artifacts": consumed_artifacts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    period_1_manifest, period_1_economy, decisions, origin = _load_period_one(
        args.period_1_run.resolve()
    )
    household_input_sha256, history_input_sha256 = _validate_source_inputs(
        period_1_manifest,
        args.households,
        args.history,
    )
    period_1_replay_binding = _period_one_replay_binding(period_1_manifest)
    source = pd.read_csv(args.households)
    ids = set(decisions["household_id"].astype(str))
    source = source.loc[source["type_id"].astype(str).isin(ids)].copy()
    if len(source) != len(ids):
        raise ValueError("household source does not cover every period-1 household")
    period_1_states = [_state_from_row(row) for _, row in source.sort_values("type_id").iterrows()]
    masses = _population_masses(period_1_states)
    employer_1 = period_1_economy["employer"]
    period_one = _settled_period_one(period_1_states, decisions, employer_1)
    feedback = compute_aggregate_firm_feedback(period_one)
    period_2_states, transitions = transition_household_states(
        period_1_states, decisions, feedback
    )
    history = pd.read_csv(args.history)
    histories = {
        str(key): group.to_dict("records") for key, group in history.groupby("household_id")
    }
    environment = build_simulated_environment_payload(feedback)["simulated_environment"]
    cards: list[dict[str, Any]] = []
    for state in period_2_states:
        card_state = source.loc[source["type_id"].astype(str).eq(state.household_id)].iloc[0].to_dict()
        card_state.update(dataclasses.asdict(state))
        card_state["state_provenance"] = "simulated_feedback_state"
        card_state["employed"] = bool((state.employment_share or 0.0) > 0.0)
        own_history = [
            row for row in histories.get(state.household_id, [])
            if str(row.get("public_availability_date", "")) <= str(origin["as_of_date"])
            and str(row.get("event_date", "")) <= str(origin["as_of_date"])
        ]
        card = household_card(
            card_state,
            origin=origin,
            own_history=own_history,
            simulated_environment=environment,
        )
        lowered = json.dumps(card, sort_keys=True).lower()
        if "actual_" in lowered or "target_value" in lowered:
            raise ValueError("period-2 card contains a forbidden realization field")
        cards.append(card)

    journal = args.cache_dir / "live_attempts" / HOUSEHOLD_PROMPT_VERSION / "feedback_period_2"
    budget = LiveCallBudget(args.max_live_calls, journal)

    def elicit(card: dict[str, Any]) -> dict[str, Any]:
        return HouseholdElicitor(
            args.provider,
            args.model,
            args.cache_dir,
            args.mode,
            args.max_live_calls,
            PROJECT_ROOT,
            budget,
        ).elicit(card)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(elicit, cards))
    responses = {
        state.household_id: normalize_household_payload(record["payload"], state.household_id)
        for state, record in zip(period_2_states, records, strict=True)
    }
    employer_2 = EmployerState(
        employer_id=str(employer_1["employer_id"]),
        productivity_per_hour=float(period_one.base_output_units) / max(
            float(employer_1["employment_count"]) * 160.0, 1.0
        ),
        monthly_capacity_units=max(
            float(employer_1["capacity_units"]), feedback.planned_output_units
        ),
        inventory_units=float(employer_1["inventory_end_units"]),
        price_per_unit_usd=float(employer_1["next_price_per_unit_usd"]),
        target_headcount=max(
            1.0,
            float(employer_1["employment_count"]) * feedback.producer_employment_index,
        ),
        wage_offer_usd=float(employer_1["average_hourly_wage_usd"])
        * feedback.producer_wage_index,
        target_inventory_units=feedback.target_inventory_units,
        variable_nonlabor_cost_per_unit_usd=0.15,
        inventory_carry_cost_per_unit_usd=0.005,
    )
    credit_2 = _initial_credit(period_2_states, origin)
    result_2 = run_monthly_ecology(
        period_2_states,
        responses,
        employer_2,
        credit_2,
        scenario="median",
        institution_mode="household_demand",
        planned_output_units=feedback.planned_output_units,
        producer_employment_count=employer_2.target_headcount,
    )
    replay_sha = canonical_sha256(
        {
            "period_1_replay_binding": period_1_replay_binding,
            "household_input_sha256": household_input_sha256,
            "history_input_sha256": history_input_sha256,
            "cards": cards,
            "payloads": [record["payload"] for record in records],
            "feedback": _jsonable(feedback),
            "period_2_result": _jsonable(result_2),
        }
    )
    request_sha = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "period_1_replay_binding": period_1_replay_binding,
            "household_input_sha256": household_input_sha256,
            "history_input_sha256": history_input_sha256,
            "source_sha256": _source_sha256(),
            "provider": args.provider,
            "model": args.model,
            "card_sha256s": [canonical_sha256(card) for card in cards],
        }
    )
    reference = args.cache_dir / "feedback_live_references" / f"{request_sha}.json"
    reference_payload = {
        "request_sha256": request_sha,
        "replay_equivalence_sha256": replay_sha,
    }
    reference_payload["reference_sha256"] = canonical_sha256(reference_payload)
    if args.mode == "live":
        reference.parent.mkdir(parents=True, exist_ok=True)
        if reference.exists() and json.loads(reference.read_text(encoding="utf-8")) != reference_payload:
            raise ValueError("feedback live reference already exists with a different result")
        _write_json(reference, reference_payload)
    else:
        if not reference.exists():
            raise ValueError("feedback replay requires an immutable live reference")
        expected = json.loads(reference.read_text(encoding="utf-8"))
        if expected != reference_payload:
            raise ValueError("feedback replay equivalence mismatch")

    _write_json(staging / "period_2_household_cards.json", cards)
    _write_json(staging / "period_2_household_responses.json", records)
    _write_json(staging / "period_2_economy.json", result_2)
    pd.DataFrame(transitions).to_csv(staging / "household_state_transitions.csv", index=False)
    pd.DataFrame(
        [{"period": 1, **dataclasses.asdict(period_one)}, {"period": 2, **dataclasses.asdict(feedback)}]
    ).to_csv(staging / "firm_state_transitions.csv", index=False)
    pd.DataFrame(
        [_jsonable(row) | {"period": 2} for row in result_2.accounting_residuals]
    ).to_csv(staging / "accounting_audit.csv", index=False)
    period_1_consumption = float(employer_1["revenue_usd"])
    producer_employment_index_2 = float(result_2.employer.employment_count) / max(
        float(employer_1["employment_count"]), 1e-12
    )
    producer_wage_index_2 = float(result_2.employer.average_hourly_wage_usd) / max(
        float(employer_1["average_hourly_wage_usd"]), 1e-12
    )
    paths = pd.DataFrame(
        [
            {
                "period": 1,
                "period_role": "observed_simulation_state",
                "consumption_usd": period_1_consumption,
                "output_units": float(employer_1["output_units"]),
                "inventory_units": float(employer_1["inventory_end_units"]),
                "producer_employment_index": 1.0,
                "producer_wage_index": 1.0,
                "family_wage_income_usd": sum(
                    masses[row.household_id] * row.monthly_family_wage_income_usd
                    for row in period_1_states
                ),
            },
            {
                "period": 2,
                "period_role": "recursive_simulated_state",
                "consumption_usd": float(result_2.employer.revenue_usd),
                "output_units": float(result_2.employer.output_units),
                "inventory_units": float(result_2.employer.inventory_end_units),
                "producer_employment_index": producer_employment_index_2,
                "producer_wage_index": producer_wage_index_2,
                "family_wage_income_usd": sum(
                    masses[row.household_id] * row.monthly_family_wage_income_usd
                    for row in period_2_states
                ),
            },
        ]
    )
    paths["consumption_growth_from_period_1_pct"] = 100.0 * (
        paths["consumption_usd"] / max(period_1_consumption, 1.0) - 1.0
    )
    paths.to_csv(staging / "dynamic_macro_paths.csv", index=False)
    period_2_target_month = (
        pd.Timestamp(period_1_manifest["target_month"]) + pd.DateOffset(months=1)
    ).strftime("%Y-%m-%d")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "unscored_two_period_mechanism_experiment",
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "period_1_run": str(args.period_1_run.resolve()),
        "period_1_manifest_sha256": _file_sha256(args.period_1_run / "manifest.json"),
        "period_1_replay_equivalence_sha256": period_1_replay_binding[
            "replay_equivalence_sha256"
        ],
        "period_1_consumed_artifact_sha256": period_1_replay_binding[
            "consumed_artifacts"
        ],
        "household_input_sha256": household_input_sha256,
        "history_input_sha256": history_input_sha256,
        "origin_month": period_1_manifest["origin_month"],
        "period_2_target_month": period_2_target_month,
        "household_count": len(period_2_states),
        "accepted_household_response_count": len(records),
        "live_call_count": budget.used,
        "cache_hit_count": sum(bool(row.get("cache_hit")) for row in records),
        "failed_provider_attempt_count": budget.failed,
        "accounting_passed": all(row.passed for row in result_2.accounting_residuals),
        "max_abs_accounting_residual": result_2.max_abs_residual(),
        "feedback_parameters": {
            "inventory_target_share": 0.08,
            "inventory_gap_closure": 0.35,
            "employment_gap_closure": 0.25,
            "wage_employment_elasticity": 0.10,
            "max_monthly_wage_change": 0.02,
        },
        "public_information_reused_from_period_1": True,
        "simulated_environment_separately_labelled": True,
        "respondent_employment_status_changed": False,
        "aggregate_producer_employment_realized_in_settlement": True,
        "target_values_loaded_into_prompts": False,
        "period_2_output_planned_from_period_1_demand": True,
        "period_2_public_information_policy": (
            "reuse origin-safe public information; add simulated producer state only; "
            "no future observations"
        ),
        "period_2_recurring_spending_policy": (
            "carry period-1 settled committed and discretionary spending; exclude "
            "one-off purchases; origin-visible aggregate growth remains context only"
        ),
        "request_sha256": request_sha,
        "replay_equivalence_sha256": replay_sha,
        "replay_verified": args.mode == "replay",
        "live_reference_sha256": reference_payload["reference_sha256"],
        "source_sha256": _source_sha256(),
        **_git_state(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }
    report = [
        "# Two-Period LLM Household Economy",
        "",
        "This unscored mechanism run closes one feedback loop: period-one household demand changes producer output, inventories, aggregate employment, wages, and family wage income; the same LLM households then decide again from their carried balances and updated income.",
        "",
        f"Period-two household consumption changes **{paths.iloc[1]['consumption_growth_from_period_1_pct']:+.2f}%** from period one. Settled producer employment moves to **{producer_employment_index_2:.3f}** and the average-wage index to **{producer_wage_index_2:.3f}** (period one = 1).",
        "",
        f"Accounting: **{'PASS' if manifest['accounting_passed'] else 'FAIL'}**; maximum residual `{manifest['max_abs_accounting_residual']:.3g}`.",
        "",
        "The firm is deliberately mechanical, not another LLM. It plans output from prior demand and realizes aggregate employment and wages in settlement. Respondent job statuses remain fixed because the data do not identify which family earner changes hours. Period-two sales still emerge from fresh household choices.",
        "",
        "This run is a mechanism trace, not a causal estimate of the feedback effect: it has no matched no-feedback household-call arm.",
    ]
    (staging / "feedback_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for path in sorted(staging.iterdir()):
        manifest["artifacts"][path.name] = _artifact_sha256(path)
    _write_json(staging / "manifest.json", manifest)
    staging.rename(output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
