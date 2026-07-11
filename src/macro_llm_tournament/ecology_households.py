"""Isolated household elicitation for the rolling microeconomy."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ecology_provider import CodexJSONClient
from .ecology_models import HouseholdResponse, QuantileTriplet


HOUSEHOLD_PROMPT_VERSION = "household_ecology_monthly_v1"


class LiveCallBudget:
    def __init__(self, maximum: int, journal_dir: Path | None = None) -> None:
        self.maximum = max(0, int(maximum))
        self.used = 0
        self._lock = threading.Lock()
        self.journal_dir = journal_dir
        self._attempts: dict[str, Path] = {}

    def reserve(self, cache_name: str) -> None:
        with self._lock:
            if self.used >= self.maximum:
                raise ValueError(f"ecology live-call cap reached ({self.maximum}); cache miss for {cache_name}")
            self.used += 1
            if self.journal_dir is not None:
                self.journal_dir.mkdir(parents=True, exist_ok=True)
                path = self.journal_dir / f"attempt_{self.used:04d}.json"
                payload = {
                    "attempt_number": self.used,
                    "cache_name": cache_name,
                    "status": "started",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "finished_at_utc": None,
                    "response_sha256": None,
                    "error_sha256": None,
                }
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                self._attempts[cache_name] = path

    def complete(self, cache_name: str, *, response: Any | None = None, error: str | None = None) -> None:
        with self._lock:
            path = self._attempts.get(cache_name)
            if path is None:
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status"] = "accepted" if error is None else "failed"
            payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            payload["response_sha256"] = canonical_sha256(response) if response is not None else None
            payload["error_sha256"] = hashlib.sha256(error.encode()).hexdigest() if error is not None else None
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def household_card(
    state: Mapping[str, Any],
    *,
    origin: Mapping[str, Any],
    own_history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one leakage-safe card; no other household state is admitted."""

    household_id = str(state.get("household_id") or state.get("type_id") or "")
    if not household_id:
        raise ValueError("household state lacks household_id/type_id")
    public = {
        "origin_month": origin["origin_month"],
        "as_of_date": origin["as_of_date"],
        "macro_context": origin.get("origin_visible_macro_context", {}),
        "macro_history": origin.get("origin_visible_macro_history", {}),
        "public_events": origin.get("public_events", []),
    }
    if origin.get("prior_simulated_state"):
        public["prior_simulated_state"] = origin["prior_simulated_state"]
    private = {
        "household_id": household_id,
        "profile": {
            key: state.get(key)
            for key in (
                "age_bucket",
                "income_group",
                "liquidity_group",
                "employment_status",
            )
        },
        "current_state": {
            "annual_income": state.get("annual_income", 0.0),
            "monthly_consumption": state.get(
                "monthly_consumption",
                float(state.get("baseline_consumption_annual", 0.0)) / 12.0,
            ),
            "liquid_assets": state.get("deposit_balance_usd", state.get("liquid_assets", 0.0)),
            "revolving_debt": state.get(
                "revolving_debt_usd", state.get("revolving_debt", state.get("debt", 0.0))
            ),
            "credit_limit": state.get("credit_limit"),
            "hours_worked": state.get("hours_worked", 160.0),
            "employed": state.get(
                "employed",
                str(state.get("employment_status", "unknown")).lower() not in {"unemployed", "not_employed"},
            ),
        },
        "previous_beliefs": {
            "inflation_expectation_1y": state.get("inflation_expectation_1y"),
            "income_growth_expectation_1y": state.get("income_growth_expectation_1y"),
            "job_loss_probability": state.get(
                "job_loss_probability", state.get("baseline_job_loss_probability")
            ),
        },
        "previous_intentions": state.get("previous_intentions", {}),
        "previous_outcomes": state.get("previous_outcomes", {}),
        "survey_history": sorted(
            [dict(row) for row in own_history], key=lambda row: str(row.get("event_date", ""))
        ),
    }
    return _clean({
        "prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "household": private,
        "public_information": public,
    })


def household_prompt(card: Mapping[str, Any]) -> str:
    return f"""
You are one household making a one-month-ahead forecast and plan. Use only the
household's own state/history and the dated public information below. Do not
invent a biography. Report intended choices, not socially desirable choices.
You do not directly execute transactions; deterministic accounting code will
enforce budgets and credit limits.

INPUT
{json.dumps(card, indent=2, sort_keys=True)}

Return exactly one JSON object with this shape:
{{
  "prompt_version": "{HOUSEHOLD_PROMPT_VERSION}",
  "household_id": "{card['household']['household_id']}",
  "expected_inflation_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "expected_income_growth_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "job_loss_probability_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "planned_consumption_change_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "planned_work_hours": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "planned_job_search_hours": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "target_buffer_months": 0.0,
  "buffer_contribution_intent_usd": 0.0,
  "debt_payment_intent_usd": 0.0,
  "borrowing_intent_usd": 0.0,
  "reason_codes": ["short_machine_readable_reason"]
}}
""".strip()


def fixture_response(card: Mapping[str, Any]) -> dict[str, Any]:
    state = card["household"]["current_state"]
    profile = card["household"]["profile"]
    liquid = float(state.get("liquid_assets") or 0.0)
    monthly = max(1.0, float(state.get("monthly_consumption") or 1.0))
    debt = max(0.0, float(state.get("revolving_debt") or 0.0))
    low_liquidity = liquid < 2.0 * monthly or profile.get("liquidity_group") == "low"
    employed = bool(state.get("employed"))
    inflation = 3.1
    income = 0.4 if employed else -1.8
    job_risk = 12.0 if employed else 65.0
    consumption = -0.8 if low_liquidity else 0.2
    return {
        "prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "household_id": card["household"]["household_id"],
        "expected_inflation_pct": {"p10": inflation - 1.2, "p50": inflation, "p90": inflation + 1.8},
        "expected_income_growth_pct": {"p10": income - 2.0, "p50": income, "p90": income + 2.0},
        "job_loss_probability_pct": {"p10": max(0.0, job_risk - 8.0), "p50": job_risk, "p90": min(100.0, job_risk + 16.0)},
        "planned_consumption_change_pct": {"p10": consumption - 2.0, "p50": consumption, "p90": consumption + 2.0},
        "planned_work_hours": {"p10": 120.0 if employed else 0.0, "p50": 160.0 if employed else 0.0, "p90": 180.0 if employed else 40.0},
        "planned_job_search_hours": {"p10": 1.0, "p50": 4.0 if employed else 30.0, "p90": 12.0 if employed else 80.0},
        "target_buffer_months": 3.0 if low_liquidity else 5.0,
        "buffer_contribution_intent_usd": monthly * (0.02 if low_liquidity else 0.08),
        "debt_payment_intent_usd": min(debt, max(0.0, monthly * (0.04 if low_liquidity else 0.1))),
        "borrowing_intent_usd": max(0.0, monthly - liquid) * (0.15 if low_liquidity else 0.0),
        "reason_codes": ["fixture_liquidity_and_employment_state"],
    }


_PAYLOAD_FIELDS = {
    "prompt_version",
    "household_id",
    "expected_inflation_pct",
    "expected_income_growth_pct",
    "job_loss_probability_pct",
    "planned_consumption_change_pct",
    "planned_work_hours",
    "planned_job_search_hours",
    "target_buffer_months",
    "buffer_contribution_intent_usd",
    "debt_payment_intent_usd",
    "borrowing_intent_usd",
    "reason_codes",
}


def normalize_household_payload(payload: Mapping[str, Any], household_id: str) -> HouseholdResponse:
    if set(payload) != _PAYLOAD_FIELDS:
        missing = sorted(_PAYLOAD_FIELDS - set(payload))
        extra = sorted(set(payload) - _PAYLOAD_FIELDS)
        raise ValueError(f"household response schema mismatch; missing={missing}, extra={extra}")
    if payload["prompt_version"] != HOUSEHOLD_PROMPT_VERSION:
        raise ValueError("household response prompt version mismatch")
    if payload["household_id"] != household_id:
        raise ValueError("household response identity mismatch")
    reasons = payload["reason_codes"]
    if not isinstance(reasons, list) or not reasons or not all(
        isinstance(reason, str) and 0 < len(reason) <= 80 for reason in reasons
    ):
        raise ValueError("reason_codes must be a non-empty list of short strings")

    def triplet(name: str) -> QuantileTriplet:
        value = payload[name]
        if not isinstance(value, Mapping) or set(value) != {"p10", "p50", "p90"}:
            raise ValueError(f"{name} must contain only p10/p50/p90")
        return QuantileTriplet(float(value["p10"]), float(value["p50"]), float(value["p90"]))

    response = HouseholdResponse(
        expected_inflation_pct=triplet("expected_inflation_pct"),
        expected_income_growth_pct=triplet("expected_income_growth_pct"),
        job_loss_probability_pct=triplet("job_loss_probability_pct"),
        planned_consumption_change_pct=triplet("planned_consumption_change_pct"),
        planned_work_hours=triplet("planned_work_hours"),
        planned_job_search_hours=triplet("planned_job_search_hours"),
        target_buffer_months=float(payload["target_buffer_months"]),
        buffer_contribution_intent_usd=float(payload["buffer_contribution_intent_usd"]),
        debt_payment_intent_usd=float(payload["debt_payment_intent_usd"]),
        borrowing_intent_usd=float(payload["borrowing_intent_usd"]),
    )
    response.validate()
    return response


@dataclass
class HouseholdElicitor:
    provider: str
    model: str
    cache_dir: Path
    mode: str
    max_live_calls: int
    execution_cwd: Path
    call_budget: LiveCallBudget | None = None

    def __post_init__(self) -> None:
        if self.provider != "codex_cli":
            raise ValueError("household ecology currently supports codex_cli only")
        if self.mode == "fixture":
            self.client = None
            return
        self.client = CodexJSONClient(
            model=self.model,
            cache_dir=self.cache_dir,
            mode=self.mode,
            max_live_calls=1 if self.call_budget is not None else self.max_live_calls,
            execution_cwd=self.execution_cwd,
        )

    def elicit(self, card: Mapping[str, Any]) -> dict[str, Any]:
        if self.mode == "fixture":
            return {"payload": fixture_response(card), "cache_hit": True}
        assert self.client is not None
        prompt, instructions, cache_name = household_request_identity(
            self.provider, self.model, card
        )
        cache_miss = not self.client.cache_path(cache_name).exists()
        if cache_miss and self.mode == "live" and self.call_budget is not None:
            self.call_budget.reserve(cache_name)
        try:
            result = self.client.json_call(prompt, cache_name, instructions=instructions)
        except Exception as exc:
            if cache_miss and self.call_budget is not None:
                self.call_budget.complete(cache_name, error=f"{type(exc).__name__}: {exc}")
            raise
        if cache_miss and self.call_budget is not None:
            self.call_budget.complete(cache_name, response=result.get("payload"))
        return result

    @property
    def live_call_count(self) -> int:
        return self.client.live_call_count if self.client is not None else 0

    @property
    def cache_hit_count(self) -> int:
        return self.client.cache_hit_count if self.client is not None else 0


def household_request_identity(
    provider: str, model: str, card: Mapping[str, Any]
) -> tuple[str, str, str]:
    prompt = household_prompt(card)
    instructions = (
        "Return only valid JSON. Do not use tools, browse, inspect files, or infer future outcomes. "
        "Use only the supplied household card and public information."
    )
    cache_name = "household_ecology_" + canonical_sha256(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "instructions": instructions,
        }
    )
    return prompt, instructions, cache_name
