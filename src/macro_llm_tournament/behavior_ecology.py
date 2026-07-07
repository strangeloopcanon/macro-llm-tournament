"""Individual-household behavior ecology.

Tests whether the compression failure of cell-level behavior elicitation
(models emit nearly the same response across very different shocks) survives
when the simulation is run the natural way: each simulated household gets a
concrete balance sheet, faces one shock in the first person, decides in
dollars, and the population aggregates emerge from weighted summation.

Two arms:
- ``ecology``: one call per household x scenario; first-person "what do you
  actually do" prompt; dollar answers converted to shares by code.
- ``policy``: one call per scenario family; the model states a response
  *schedule* (spending share as a function of windfall size relative to
  monthly income, or the UI onset/receipt/exhaustion path) per type cell;
  code evaluates the schedule at each scenario's actual ratio.

The pre-registered primary metrics target differentiation, not level:
lottery size gradient and UI exhaustion cliff. Both scenario families were
already spent as behavior holdouts, so this run is declared diagnostic
selection surface, not confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, bounded_float, cache_key
from .agent_llm import AgentLLMClient
from .agent_types import build_household_type_cells
from .behavior_gate import (
    BEHAVIOR_SCENARIOS,
    BehaviorScenario,
    _behavior_action_row,
    _liquidity_group,
    aggregate_behavior_actions,
    behavior_targets_frame,
    run_behavior_controls,
    score_behavior_targets,
)
from .llm_common import LLMUnavailable

BEHAVIOR_ECOLOGY_VERSION = "behavior_ecology_v1"
ECOLOGY_PROMPT_VERSION = "behavior_ecology_household_v1"
POLICY_PROMPT_VERSION = "behavior_ecology_policy_v1"
POLICY_RATIO_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)

NATURAL_BEHAVIOR_PREAMBLE = """
Return only valid JSON. You are answering as one specific American household, privately and
honestly, about what you would actually do. Real families often do not follow financial
advice, often do not do what surveys say they should, and often spend windfalls faster or
cut spending later than experts recommend. Do not give the prudent answer or the socially
expected answer; give the realistic one for this family. Use only the information supplied.
Do not browse, inspect files, run commands, or cite research studies.
""".strip()

POLICY_PREAMBLE = """
Return only valid JSON. You are a careful empirical economist describing how measured US
household behavior varies with the size and type of an income shock, based on your general
knowledge of observed (not advised) behavior. Use only the information supplied. Do not
browse, inspect files, run commands, or cite specific study estimates.
""".strip()

# Pre-registered before scoring: bands derive from the public target bands
# already in the catalog (small minus large lottery spending shares; UI
# exhaustion minus receipt drop shares). Point-elicitation references are the
# descriptive-prompt arms from the four-arm probe.
PREREGISTERED_METRICS = {
    "lottery_size_gradient": {
        "definition": "aggregate_total_spending_share(small_lottery) - aggregate_total_spending_share(large_lottery)",
        "target_low": 0.30,
        "target_high": 0.68,
        "point_elicitation_reference_gpt55_descriptive": -0.013,
        "point_elicitation_reference_opus48_descriptive": 0.078,
    },
    "ui_exhaustion_cliff": {
        "definition": "aggregate_nondurable_spending_drop_share(ui_exhaustion) - aggregate_nondurable_spending_drop_share(ui_receipt)",
        "target_low": 0.10,
        "target_high": 0.13,
        "point_elicitation_reference_gpt55_descriptive": 0.017,
        "point_elicitation_reference_opus48_descriptive": 0.012,
    },
}
CLAIM_SCOPE = (
    "Diagnostic selection-surface run. The lottery and UI families were already spent as holdouts in "
    "earlier rounds, so nothing here is confirmatory; the question is whether individual-household "
    "grounding or policy-schedule elicitation restores cross-scenario differentiation that cell-level "
    "point elicitation compresses."
)


def sample_households(type_cells: pd.DataFrame, *, households_per_cell: int, seed: int) -> pd.DataFrame:
    """Draw concrete households around each SCF type cell with seeded jitter."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, cell in type_cells.iterrows():
        for draw in range(households_per_cell):
            income_jitter = float(rng.lognormal(mean=0.0, sigma=0.25))
            liquid_jitter = float(rng.lognormal(mean=0.0, sigma=0.50))
            monthly_take_home = round(float(cell["annual_income"]) / 12.0 * income_jitter * 0.78, -1)
            checking_and_savings = round(float(cell["liquid_assets"]) * liquid_jitter, -1)
            monthly_spending = round(float(cell["consumption_proxy_annual"]) / 12.0 * income_jitter, -1)
            total_debt = round(float(cell["debt"]) * float(rng.uniform(0.7, 1.3)), -2)
            revolving_debt = round(min(total_debt, float(cell["credit_limit_proxy"]) * float(rng.uniform(0.2, 0.9))), -2)
            family_size = int(rng.integers(1, 5))
            buffer_months = checking_and_savings / max(monthly_spending, 1.0)
            rows.append(
                {
                    "household_id": f"{cell['type_id']}_{draw:02d}",
                    "type_id": str(cell["type_id"]),
                    "cell_label": str(cell["label"]),
                    "population_weight": float(cell["population_weight"]) / households_per_cell,
                    "liquidity_group": _liquidity_group(cell),
                    "monthly_take_home_usd": monthly_take_home,
                    "monthly_spending_usd": monthly_spending,
                    "checking_and_savings_usd": checking_and_savings,
                    "revolving_debt_usd": revolving_debt,
                    "other_debt_usd": max(0.0, total_debt - revolving_debt),
                    "family_size": family_size,
                    "buffer_months": round(buffer_months, 2),
                }
            )
    return pd.DataFrame(rows)


def _household_prompt_block(household: pd.Series) -> dict[str, Any]:
    return {
        "monthly_take_home_income_usd": household["monthly_take_home_usd"],
        "typical_monthly_spending_usd": household["monthly_spending_usd"],
        "checking_and_savings_balance_usd": household["checking_and_savings_usd"],
        "credit_card_and_revolving_debt_usd": household["revolving_debt_usd"],
        "other_debt_usd_mortgage_auto_student": household["other_debt_usd"],
        "months_of_spending_covered_by_liquid_balance": household["buffer_months"],
        "people_in_household": int(household["family_size"]),
    }


def ecology_household_prompt(scenario: BehaviorScenario, household: pd.Series) -> str:
    if scenario.scenario_type == "income_loss":
        payload = {
            "prompt_version": ECOLOGY_PROMPT_VERSION,
            "you_are": "Answer in the first person as this exact household. Be honest about what you would actually do, not what you should do.",
            "your_household": _household_prompt_block(household),
            "situation": scenario.prompt_context,
            "income_loss_share_of_monthly_income": scenario.income_loss_pct,
            "question": (
                "Given this situation, report your monthly spending on everyday nondurables and services "
                "(groceries, fuel, bills, eating out) in the reference month described by the situation, and in the "
                "month being asked about. Answer in dollars for this household."
            ),
            "required_response": {
                "monthly_nondurable_spending_reference_usd": "dollars in the reference month described by the situation",
                "monthly_nondurable_spending_now_usd": "dollars in the month the situation asks about",
                "one_line_reason": "one short sentence in the first person",
            },
        }
    else:
        amount = float(scenario.transfer_amount)
        payload = {
            "prompt_version": ECOLOGY_PROMPT_VERSION,
            "you_are": "Answer in the first person as this exact household. Be honest about what you would actually do, not what you should do.",
            "your_household": _household_prompt_block(household),
            "situation": scenario.prompt_context,
            "amount_received_usd": amount,
            "horizon_months": scenario.horizon_months,
            "question": (
                f"You receive ${amount:,.0f} as described. Over the next {scenario.horizon_months} months, what do you "
                "actually do with it? Split the full amount in dollars."
            ),
            "required_response": {
                "spend_on_nondurables_and_services_usd": "dollars actually spent on everyday goods, services, bills, food",
                "spend_on_durables_usd": "dollars actually spent on durable goods (appliances, vehicles, electronics, furniture)",
                "repay_debt_usd": "dollars actually used to pay down debt",
                "keep_as_liquid_savings_usd": "dollars still sitting in checking/savings at the end of the horizon",
                "one_line_reason": "one short sentence in the first person",
                "sum_rule": f"the four dollar amounts must sum to {amount:,.0f}",
            },
        }
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_ecology_payload(scenario: BehaviorScenario, household: pd.Series) -> dict[str, Any]:
    """Deterministic liquidity-flavored fixture so tests run without a network."""
    buffer_months = float(household["buffer_months"])
    tightness = float(np.clip(1.0 - buffer_months / 6.0, 0.05, 0.95))
    if scenario.scenario_type == "income_loss":
        reference = float(household["monthly_spending_usd"]) * 0.8
        drop_share = float(np.clip(scenario.income_loss_pct * (0.25 + 0.5 * tightness), 0.0, 0.9))
        return {
            "monthly_nondurable_spending_reference_usd": round(reference, 0),
            "monthly_nondurable_spending_now_usd": round(reference * (1.0 - drop_share), 0),
            "one_line_reason": "fixture income-loss household",
        }
    amount = float(scenario.transfer_amount)
    size_ratio = amount / max(float(household["monthly_take_home_usd"]), 1.0)
    spend_share = float(np.clip(0.85 * tightness / (1.0 + 0.35 * np.log1p(size_ratio)), 0.02, 0.95))
    debt_share = float(np.clip(0.4 * tightness * (1.0 - spend_share), 0.0, 0.9))
    save_share = max(0.0, 1.0 - spend_share - debt_share)
    return {
        "spend_on_nondurables_and_services_usd": round(amount * spend_share * 0.8, 0),
        "spend_on_durables_usd": round(amount * spend_share * 0.2, 0),
        "repay_debt_usd": round(amount * debt_share, 0),
        "keep_as_liquid_savings_usd": round(amount * save_share, 0),
        "one_line_reason": "fixture transfer household",
    }


def normalize_ecology_payload(scenario: BehaviorScenario, household: pd.Series, data: dict[str, Any]) -> dict[str, Any]:
    """Convert a household dollar answer into the behavior-gate share schema."""
    payload = data.get("payload", data)
    if scenario.scenario_type == "income_loss":
        reference = bounded_float(payload, "monthly_nondurable_spending_reference_usd", 1.0, 1e7)
        now = bounded_float(payload, "monthly_nondurable_spending_now_usd", 0.0, 1e7)
        drop_share = float(np.clip((reference - now) / reference, -1.0, 1.0))
        return {
            "total_spending_share": drop_share,
            "nondurable_spending_share": drop_share,
            "durable_spending_share": 0.0,
            "debt_repayment_share": 0.0,
            "liquid_saving_share": 0.0,
            "confidence": 0.5,
            "reason": str(payload.get("one_line_reason", ""))[:300],
        }
    amount = float(scenario.transfer_amount)
    nondurable = bounded_float(payload, "spend_on_nondurables_and_services_usd", 0.0, amount * 2)
    durable = bounded_float(payload, "spend_on_durables_usd", 0.0, amount * 2)
    debt = bounded_float(payload, "repay_debt_usd", 0.0, amount * 2)
    liquid = bounded_float(payload, "keep_as_liquid_savings_usd", 0.0, amount * 2)
    total = nondurable + durable + debt + liquid
    if total <= 0:
        raise LLMUnavailable(f"Ecology payload for {scenario.scenario_id}/{household['household_id']} allocates nothing")
    scale = amount / total
    nondurable, durable, debt, liquid = (v * scale / amount for v in (nondurable, durable, debt, liquid))
    return {
        "total_spending_share": nondurable + durable,
        "nondurable_spending_share": nondurable,
        "durable_spending_share": durable,
        "debt_repayment_share": debt,
        "liquid_saving_share": liquid,
        "confidence": 0.5,
        "reason": str(payload.get("one_line_reason", ""))[:300],
    }


def policy_prompt(scenarios: Iterable[BehaviorScenario], type_cells: pd.DataFrame, *, family: str) -> str:
    cells = [
        {
            "type_id": str(row["type_id"]),
            "label": str(row["label"]),
            "monthly_take_home_income_usd": round(float(row["annual_income"]) / 12.0 * 0.78, -1),
            "checking_and_savings_balance_usd": round(float(row["liquid_assets"]), -1),
            "total_debt_usd": round(float(row["debt"]), -2),
            "months_of_spending_covered_by_liquid_balance": round(
                float(row["liquid_assets"]) / max(float(row["consumption_proxy_annual"]) / 12.0, 1.0), 2
            ),
        }
        for _, row in type_cells.iterrows()
    ]
    if family == "income_loss":
        payload = {
            "prompt_version": POLICY_PROMPT_VERSION,
            "task": (
                "For each household type, describe the measured path of everyday nondurable spending through an "
                "unemployment spell with unemployment insurance: the drop at job loss while benefits arrive, the "
                "extra month-to-month drift while benefits continue, and the further drop in the month benefits "
                "are exhausted. Report observed behavior, not advice."
            ),
            "household_types": cells,
            "required_response": {
                "policies": [
                    {
                        "type_id": "one supplied type_id",
                        "onset_drop_share": "0 to 1 drop vs pre-unemployment monthly nondurable spending",
                        "receipt_monthly_drift_share": "0 to 1 additional month-over-month drop while benefits continue",
                        "exhaustion_drop_share": "0 to 1 drop in the exhaustion month vs the prior benefit month",
                        "reason": "short reason",
                    }
                ]
            },
        }
    else:
        payload = {
            "prompt_version": POLICY_PROMPT_VERSION,
            "task": (
                "For each household type, describe how the household actually uses a one-time windfall as a function "
                "of its size. The grid point r is windfall divided by monthly take-home income. Report the measured "
                "shares over a 12-month horizon: spent (nondurables+durables), used to repay debt, and kept liquid. "
                "Report observed behavior, not advice."
            ),
            "windfall_to_monthly_income_ratio_grid": list(POLICY_RATIO_GRID),
            "household_types": cells,
            "required_response": {
                "policies": [
                    {
                        "type_id": "one supplied type_id",
                        "schedule": [
                            {
                                "ratio": "one grid ratio",
                                "total_spending_share": "0 to 1",
                                "nondurable_share_of_spending": "0 to 1",
                                "debt_repayment_share": "0 to 1",
                                "liquid_saving_share": "0 to 1",
                            }
                        ],
                        "reason": "short reason",
                    }
                ]
            },
        }
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_policy_payload(type_cells: pd.DataFrame, *, family: str) -> dict[str, Any]:
    policies = []
    for _, cell in type_cells.iterrows():
        buffer_months = float(cell["liquid_assets"]) / max(float(cell["consumption_proxy_annual"]) / 12.0, 1.0)
        tightness = float(np.clip(1.0 - buffer_months / 6.0, 0.05, 0.95))
        if family == "income_loss":
            policies.append(
                {
                    "type_id": str(cell["type_id"]),
                    "onset_drop_share": round(0.05 + 0.10 * tightness, 3),
                    "receipt_monthly_drift_share": round(0.005 * tightness, 4),
                    "exhaustion_drop_share": round(0.06 + 0.12 * tightness, 3),
                    "reason": "fixture income-loss policy",
                }
            )
        else:
            schedule = []
            for ratio in POLICY_RATIO_GRID:
                spend = float(np.clip(0.9 * tightness / (1.0 + 0.4 * np.log1p(ratio)), 0.02, 0.95))
                debt = float(np.clip(0.35 * tightness * (1.0 - spend), 0.0, 0.9))
                schedule.append(
                    {
                        "ratio": ratio,
                        "total_spending_share": round(spend, 3),
                        "nondurable_share_of_spending": 0.8,
                        "debt_repayment_share": round(debt, 3),
                        "liquid_saving_share": round(max(0.0, 1.0 - spend - debt), 3),
                    }
                )
            policies.append({"type_id": str(cell["type_id"]), "schedule": schedule, "reason": "fixture windfall policy"})
    return {"policies": policies}


def normalize_policy_payload(type_cells: pd.DataFrame, data: dict[str, Any], *, family: str) -> dict[str, dict[str, Any]]:
    payload = data.get("payload", data)
    policies = payload.get("policies")
    if not isinstance(policies, list):
        raise LLMUnavailable(f"Policy payload for family {family} is missing policies list")
    expected = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, Any]] = {}
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        type_id = str(policy.get("type_id", ""))
        if type_id not in expected:
            continue
        if family == "income_loss":
            by_type[type_id] = {
                "onset_drop_share": bounded_float(policy, "onset_drop_share", 0.0, 1.0),
                "receipt_monthly_drift_share": bounded_float(policy, "receipt_monthly_drift_share", 0.0, 1.0),
                "exhaustion_drop_share": bounded_float(policy, "exhaustion_drop_share", 0.0, 1.0),
            }
        else:
            schedule = policy.get("schedule")
            if not isinstance(schedule, list) or not schedule:
                continue
            points = []
            for point in schedule:
                if not isinstance(point, dict):
                    continue
                points.append(
                    {
                        "ratio": bounded_float(point, "ratio", 1e-4, 1e4),
                        "total_spending_share": bounded_float(point, "total_spending_share", 0.0, 1.0),
                        "nondurable_share_of_spending": bounded_float(point, "nondurable_share_of_spending", 0.0, 1.0),
                        "debt_repayment_share": bounded_float(point, "debt_repayment_share", 0.0, 1.0),
                        "liquid_saving_share": bounded_float(point, "liquid_saving_share", 0.0, 1.0),
                    }
                )
            if points:
                by_type[type_id] = {"schedule": sorted(points, key=lambda p: p["ratio"])}
    missing = sorted(expected - set(by_type))
    if missing:
        raise LLMUnavailable(f"Policy payload for family {family} is missing type ids: {', '.join(missing)}")
    return by_type


def evaluate_policy_at_scenario(
    scenario: BehaviorScenario,
    type_cell: pd.Series,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if scenario.scenario_type == "income_loss":
        if scenario.scenario_id == "ui_receipt_monthly_path_style":
            drop = float(policy["receipt_monthly_drift_share"])
        elif scenario.scenario_id == "ui_exhaustion_income_loss_style":
            drop = float(policy["exhaustion_drop_share"])
        else:
            drop = float(policy["onset_drop_share"])
        return {
            "total_spending_share": drop,
            "nondurable_spending_share": drop,
            "durable_spending_share": 0.0,
            "debt_repayment_share": 0.0,
            "liquid_saving_share": 0.0,
            "confidence": 0.5,
            "reason": "policy schedule evaluation",
        }
    monthly_income = max(float(type_cell["annual_income"]) / 12.0 * 0.78, 1.0)
    ratio = float(scenario.transfer_amount) / monthly_income
    schedule = policy["schedule"]
    ratios = np.array([p["ratio"] for p in schedule])
    log_r = np.log(np.clip(ratio, ratios.min(), ratios.max()))

    def interp(field: str) -> float:
        values = np.array([p[field] for p in schedule])
        return float(np.interp(log_r, np.log(ratios), values))

    spend = interp("total_spending_share")
    nondurable_of_spend = interp("nondurable_share_of_spending")
    debt = interp("debt_repayment_share")
    liquid = interp("liquid_saving_share")
    total = spend + debt + liquid
    if total > 1.0 and total > 0:
        spend, debt, liquid = spend / total, debt / total, liquid / total
    return {
        "total_spending_share": spend,
        "nondurable_spending_share": spend * nondurable_of_spend,
        "durable_spending_share": spend * (1.0 - nondurable_of_spend),
        "debt_repayment_share": debt,
        "liquid_saving_share": liquid,
        "confidence": 0.5,
        "reason": "policy schedule evaluation",
    }


class EcologyLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        mode: str,
        max_live_calls: int,
        raw_policy_records_path: Path | None = None,
    ):
        self.provider = provider
        self.model = model
        self.mode = mode
        self.raw_policy_records_path = raw_policy_records_path
        self._raw_policy_records = _load_policy_record_payloads(raw_policy_records_path) if raw_policy_records_path else {}
        self._household_client = AgentLLMClient(
            provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls, system_preamble=NATURAL_BEHAVIOR_PREAMBLE
        )
        self._policy_client = AgentLLMClient(
            provider, model, cache_dir, mode=mode, max_live_calls=max_live_calls, system_preamble=POLICY_PREAMBLE
        )
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._household_client.live_call_count + self._policy_client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._household_client.cache_hit_count + self._policy_client.cache_hit_count

    def household_decision(self, scenario: BehaviorScenario, household: pd.Series) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {"provider": self.provider, "model": self.model, "payload": fixture_ecology_payload(scenario, household)}
        else:
            prompt = ecology_household_prompt(scenario, household)
            name = f"ecology_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"
            data = self._household_client._codex_call(prompt, name)
        normalized = normalize_ecology_payload(scenario, household, data)
        self.raw_records.append(
            {
                "record_type": "household_decision",
                "scenario_id": scenario.scenario_id,
                "household_id": str(household["household_id"]),
                "cache_hit": bool(data.get("cache_hit", False)),
                "payload": data.get("payload", data),
            }
        )
        return normalized

    def policy(self, scenarios: Iterable[BehaviorScenario], type_cells: pd.DataFrame, *, family: str) -> dict[str, dict[str, Any]]:
        if family in self._raw_policy_records:
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": self._raw_policy_records[family],
                "cache_hit": True,
                "cache_path": str(self.raw_policy_records_path),
            }
        elif self.mode == "fixture":
            data = {"provider": self.provider, "model": self.model, "payload": fixture_policy_payload(type_cells, family=family)}
        else:
            prompt = policy_prompt(scenarios, type_cells, family=family)
            name = f"ecology_policy_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"
            data = self._policy_client._codex_call(prompt, name)
        normalized = normalize_policy_payload(type_cells, data, family=family)
        self.raw_records.append(
            {
                "record_type": f"policy_{family}",
                "cache_hit": bool(data.get("cache_hit", False)),
                "payload": data.get("payload", data),
            }
        )
        return normalized


def _load_policy_record_payloads(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--policy-raw-records-json must contain a JSON list")
    out: dict[str, Any] = {}
    for family in ("transfer", "income_loss"):
        record_type = f"policy_{family}"
        matches = [record for record in data if isinstance(record, dict) and str(record.get("record_type")) == record_type]
        if matches:
            out[family] = matches[-1].get("payload", matches[-1])
    return out


def run_ecology_arm(
    scenarios: Iterable[BehaviorScenario],
    households: pd.DataFrame,
    type_cells: pd.DataFrame,
    *,
    client: EcologyLLMClient,
) -> pd.DataFrame:
    cell_by_id = {str(row["type_id"]): row for _, row in type_cells.iterrows()}
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for _, household in households.iterrows():
            response = client.household_decision(scenario, household)
            row = _behavior_action_row(scenario, cell_by_id[str(household["type_id"])], response, source=f"ecology_{client.provider}_{client.model}")
            row["population_weight"] = float(household["population_weight"])
            row["household_id"] = str(household["household_id"])
            rows.append(row)
    return pd.DataFrame(rows)


def run_policy_arm(
    scenarios: Iterable[BehaviorScenario],
    type_cells: pd.DataFrame,
    *,
    client: EcologyLLMClient,
) -> pd.DataFrame:
    scenario_list = list(scenarios)
    transfer = [s for s in scenario_list if s.scenario_type != "income_loss"]
    income_loss = [s for s in scenario_list if s.scenario_type == "income_loss"]
    rows: list[dict[str, Any]] = []
    for family, family_scenarios in (("transfer", transfer), ("income_loss", income_loss)):
        if not family_scenarios:
            continue
        policies = client.policy(family_scenarios, type_cells, family=family)
        for scenario in family_scenarios:
            for _, cell in type_cells.iterrows():
                action = evaluate_policy_at_scenario(scenario, cell, policies[str(cell["type_id"])])
                rows.append(_behavior_action_row(scenario, cell, action, source=f"policy_{client.provider}_{client.model}"))
    return pd.DataFrame(rows)


def differentiation_metrics(aggregates: pd.DataFrame, source: str) -> dict[str, Any]:
    def value(scenario_id: str, column: str) -> float:
        rows = aggregates[(aggregates["source"] == source) & (aggregates["scenario_id"] == scenario_id)]
        return float(rows.iloc[0][column]) if not rows.empty else float("nan")

    lottery_gradient = value("small_lottery_windfall_style", "aggregate_total_spending_share") - value(
        "large_lottery_windfall_style", "aggregate_total_spending_share"
    )
    ui_cliff = value("ui_exhaustion_income_loss_style", "aggregate_nondurable_spending_drop_share") - value(
        "ui_receipt_monthly_path_style", "aggregate_nondurable_spending_drop_share"
    )
    out: dict[str, Any] = {"source": source}
    for metric_name, metric_value in (("lottery_size_gradient", lottery_gradient), ("ui_exhaustion_cliff", ui_cliff)):
        spec = PREREGISTERED_METRICS[metric_name]
        low, high = float(spec["target_low"]), float(spec["target_high"])
        reference = float(spec["point_elicitation_reference_gpt55_descriptive"])
        inside = bool(np.isfinite(metric_value) and low <= metric_value <= high)
        distance = 0.0 if inside else float(min(abs(metric_value - low), abs(metric_value - high))) if np.isfinite(metric_value) else float("nan")
        reference_distance = float(min(abs(reference - low), abs(reference - high)))
        if inside:
            verdict = "inside_target_band"
        elif np.isfinite(distance) and distance < reference_distance:
            verdict = "closer_than_point_elicitation"
        else:
            verdict = "no_better_than_point_elicitation"
        out[metric_name] = float(metric_value) if np.isfinite(metric_value) else None
        out[f"{metric_name}_verdict"] = verdict
        out[f"{metric_name}_distance_to_band"] = distance
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the individual-household behavior ecology.")
    parser.add_argument("--provider", choices=["codex_cli", "cursor_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--households-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--arms", default="ecology,policy")
    parser.add_argument("--scenario-ids", default=None, help="Comma-separated scenario filter (for parallel shards)")
    parser.add_argument("--scf-wave", type=int, default=2022)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--policy-raw-records-json",
        default=None,
        help="Existing behavior_ecology raw records to reuse for policy schedules without new calls.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if args.mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --mode live is used")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"behavior_ecology_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else WORK_ROOT / "behavior_ecology_cache"
    policy_raw_records_path = Path(args.policy_raw_records_json) if args.policy_raw_records_json else None
    if policy_raw_records_path is not None and not policy_raw_records_path.exists():
        raise SystemExit(f"--policy-raw-records-json does not exist: {policy_raw_records_path}")

    scenario_filter = {s.strip() for s in args.scenario_ids.split(",")} if args.scenario_ids else None
    scenarios = [s for s in BEHAVIOR_SCENARIOS if scenario_filter is None or s.scenario_id in scenario_filter]

    manifest: dict[str, Any] = {
        "schema_version": BEHAVIOR_ECOLOGY_VERSION,
        "timestamp_utc": timestamp,
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "arms": arms,
        "households_per_cell": int(args.households_per_cell),
        "seed": int(args.seed),
        "scenario_ids": sorted(s.scenario_id for s in scenarios),
        "max_live_calls": int(args.max_live_calls),
        "ecology_prompt_version": ECOLOGY_PROMPT_VERSION,
        "policy_prompt_version": POLICY_PROMPT_VERSION,
        "policy_raw_records_json": str(policy_raw_records_path) if policy_raw_records_path else None,
        "policy_raw_records_sha256": hashlib.sha256(policy_raw_records_path.read_bytes()).hexdigest() if policy_raw_records_path else None,
        "preregistered_metrics": PREREGISTERED_METRICS,
        "claim_scope": CLAIM_SCOPE,
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        type_cells, _ = build_household_type_cells(work_dir=WORK_ROOT / "scf", wave=args.scf_wave)
        households = sample_households(type_cells, households_per_cell=args.households_per_cell, seed=args.seed)
        client = EcologyLLMClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.mode,
            max_live_calls=args.max_live_calls,
            raw_policy_records_path=policy_raw_records_path,
        )

        frames: list[pd.DataFrame] = []
        if "ecology" in arms:
            frames.append(run_ecology_arm(scenarios, households, type_cells, client=client))
        if "policy" in arms:
            frames.append(run_policy_arm(scenarios, type_cells, client=client))
        controls = run_behavior_controls(scenarios, type_cells)
        actions = pd.concat([*frames, controls], ignore_index=True)
        aggregates = aggregate_behavior_actions(actions)
        targets = behavior_targets_frame(target_scope="aggregate")
        targets = targets[targets["scenario_id"].isin({s.scenario_id for s in scenarios})]
        scores = score_behavior_targets(aggregates, targets)
        candidate_sources = sorted(
            source for source in aggregates["source"].unique() if source.startswith(("ecology_", "policy_"))
        )
        metrics = pd.DataFrame(
            [differentiation_metrics(aggregates, source) for source in [*candidate_sources, "liquidity_rule"]]
        )
        households.to_csv(output_dir / "ecology_households.csv", index=False)
        actions.to_csv(output_dir / "ecology_actions.csv", index=False)
        aggregates.to_csv(output_dir / "ecology_aggregates.csv", index=False)
        scores.to_csv(output_dir / "ecology_target_scores.csv", index=False)
        metrics.to_csv(output_dir / "ecology_differentiation_metrics.csv", index=False)
        (output_dir / "ecology_raw_records.json").write_text(
            json.dumps(client.raw_records, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest.update(
            {
                "status": "ok",
                "household_count": int(households.shape[0]),
                "action_count": int(actions.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "differentiation_metrics": metrics.to_dict(orient="records"),
                "outputs": [
                    "ecology_households.csv",
                    "ecology_actions.csv",
                    "ecology_aggregates.csv",
                    "ecology_target_scores.csv",
                    "ecology_differentiation_metrics.csv",
                    "ecology_raw_records.json",
                ],
            }
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(output_dir)
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
