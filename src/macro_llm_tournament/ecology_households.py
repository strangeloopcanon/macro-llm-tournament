"""Isolated household elicitation for the rolling microeconomy."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ecology_provider import (
    CODEX_INSTRUCTION_CONTEXT_VERSION,
    CODEX_TOOL_ISOLATION_VERSION,
    CodexJSONClient,
)
from .ecology_models import HouseholdPolicyBranch, HouseholdResponse, QuantileTriplet


HOUSEHOLD_PROMPT_VERSION = "household_ecology_monthly_v18"


class LiveCallBudget:
    def __init__(self, maximum: int, journal_dir: Path | None = None) -> None:
        self.maximum = max(0, int(maximum))
        self.used = 0
        self.accepted = 0
        self.failed = 0
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
                path = self.journal_dir / (
                    f"attempt_{self.used:04d}_{cache_name[:12]}_{uuid.uuid4().hex}.json"
                )
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
            if error is None:
                self.accepted += 1
            else:
                self.failed += 1
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


def _own_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only the household's dated, previously reported beliefs."""

    rename = {
        "actual_expected_inflation_1y": "reported_expected_inflation_1y",
        "actual_expected_real_income_growth": "reported_expected_real_income_growth",
        "actual_expected_unemployment_higher_prob": "reported_expected_unemployment_higher_prob",
        "sce_personal_job_loss_probability_1y": "reported_personal_job_loss_probability_1y",
    }
    allowed = {
        "event_date",
        "public_availability_date",
        "observation_status",
        "responded",
        "employment_status",
        *rename,
    }
    return {
        rename.get(key, key): value
        for key, value in row.items()
        if key in allowed
    }


def household_card(
    state: Mapping[str, Any],
    *,
    origin: Mapping[str, Any],
    own_history: Iterable[Mapping[str, Any]],
    simulated_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one leakage-safe card; no other household state is admitted."""

    household_id = str(state.get("household_id") or state.get("type_id") or "")
    if not household_id:
        raise ValueError("household state lacks household_id/type_id")
    compact = origin.get("compact_macro_information")
    public = (
        dict(compact)
        if isinstance(compact, Mapping)
        else {
            "origin_month": origin["origin_month"],
            "as_of_date": origin["as_of_date"],
            "macro_context": origin.get("origin_visible_macro_context", {}),
            "macro_history": origin.get("origin_visible_macro_history", {}),
            "public_events": origin.get("public_events", []),
        }
    )
    has_monthly_state = "baseline_monthly_consumption_usd" in state
    state_provenance = str(
        state.get(
            "state_provenance",
            "fixed_survey_scf_anchor"
            if has_monthly_state
            else "survey_seeded_initial_state",
        )
    )
    if state_provenance not in {
        "survey_seeded_initial_state",
        "fixed_survey_scf_anchor",
        "simulated_feedback_state",
    }:
        raise ValueError(f"unsupported household state provenance: {state_provenance}")
    monthly_consumption = (
        state.get("baseline_monthly_consumption_usd")
        if has_monthly_state
        else state.get(
            "monthly_consumption",
            float(state.get("baseline_consumption_annual", 0.0)) / 12.0,
        )
    )
    hours_worked = (
        state.get("baseline_monthly_hours")
        if has_monthly_state
        else state.get("hours_worked", 160.0)
    )
    employed = (
        float(hours_worked or 0.0) > 0.0
        if has_monthly_state
        else state.get(
            "employed",
            str(state.get("employment_status", "unknown")).lower()
            not in {"unemployed", "not_employed"},
        )
    )
    annualized_wage_income = (
        float(state.get("hourly_wage_usd") or 0.0) * float(hours_worked or 0.0) * 12.0
        if has_monthly_state
        else state.get("annual_income", 0.0)
    )
    recent_nominal_drift = float(origin.get("routine_nominal_spending_drift_pct", 0.0))
    committed_baseline = state.get("baseline_committed_consumption_usd")
    discretionary_baseline = state.get("baseline_discretionary_consumption_usd")
    if committed_baseline is None or discretionary_baseline is None:
        committed_baseline = 0.65 * float(monthly_consumption or 0.0)
        discretionary_baseline = max(
            0.0,
            float(monthly_consumption or 0.0) - float(committed_baseline),
        )
    status_quo_multiplier = max(0.0, 1.0 + recent_nominal_drift / 100.0)
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
            "provenance": state_provenance,
            "annualized_wage_income": annualized_wage_income,
            "monthly_consumption": monthly_consumption,
            "liquid_assets": state.get("deposit_balance_usd", state.get("liquid_assets", 0.0)),
            "revolving_debt": state.get(
                "revolving_debt_usd", state.get("revolving_debt", state.get("debt", 0.0))
            ),
            "credit_limit": state.get(
                "revolving_credit_limit_usd", state.get("credit_limit")
            ),
            "hourly_wage": state.get("hourly_wage_usd"),
            "hours_worked": hours_worked,
            "employed": employed,
            "employment_share": state.get(
                "employment_share",
                1.0 if employed else 0.0,
            ),
            "monthly_nonwage_income": state.get("monthly_nonwage_income_usd", 0.0),
            "monthly_household_earned_income": state.get(
                "monthly_household_earned_income_usd", 0.0
            ),
            "monthly_family_wage_income": state.get(
                "monthly_family_wage_income_usd", 0.0
            ),
            "monthly_family_business_income": state.get(
                "monthly_family_business_income_usd", 0.0
            ),
            "monthly_transfer_income": state.get("monthly_transfer_income_usd", 0.0),
            "monthly_taxes_nondeposit_saving_and_omitted_outflows": state.get(
                "monthly_omitted_fixed_outflow_usd", 0.0
            ),
            "monthly_baseline_total_saving_target": state.get(
                "monthly_baseline_total_saving_target_usd", 0.0
            ),
            "monthly_baseline_liquid_saving_target": state.get(
                "monthly_baseline_liquid_saving_target_usd", 0.0
            ),
            "monthly_baseline_cash_deficit": state.get(
                "monthly_baseline_cash_deficit_usd", 0.0
            ),
            "baseline_committed_consumption": state.get(
                "baseline_committed_consumption_usd"
            ),
            "baseline_discretionary_consumption": state.get(
                "baseline_discretionary_consumption_usd"
            ),
            "recent_observed_nominal_spending_drift_pct": recent_nominal_drift,
            "status_quo_next_month_committed_consumption": (
                float(committed_baseline) * status_quo_multiplier
            ),
            "status_quo_next_month_discretionary_consumption": (
                float(discretionary_baseline) * status_quo_multiplier
            ),
            "minimum_debt_payment": state.get("minimum_debt_payment_usd", 0.0),
            "spending_baseline_semantics": (
                "origin_safe_estimate_of_recent_typical_monthly_spending_in_"
                "the_household_state_dollar_scale"
            ),
        },
        "previous_beliefs": {
            "inflation_expectation_1y": state.get("inflation_expectation_1y"),
            "income_growth_expectation_1y": state.get("income_growth_expectation_1y"),
            "personal_job_loss_probability_1y": state.get(
                "personal_job_loss_probability_1y"
            ),
            "unemployment_higher_probability_1y": state.get(
                "unemployment_higher_probability_1y"
            ),
        },
        "survey_history": sorted(
            [_own_history_row(row) for row in own_history],
            key=lambda row: str(row.get("event_date", "")),
        ),
    }
    card = {
        "prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "household": private,
        "public_information": public,
    }
    if simulated_environment is not None:
        card["simulated_environment"] = dict(simulated_environment)
    return _clean(card)


def household_prompt(card: Mapping[str, Any]) -> str:
    return f"""
You are one real household making its next-month choices. Use only this
household's own state/history and the dated public information below. Do not
invent a biography and do not answer as a financial adviser. State what this
household will actually do, including ordinary inertia, habits, and imperfect
adjustment. Deterministic code will execute the policy and enforce its budget,
credit limit, and minimum debt payment.

If a separately labelled simulated_environment is present, it describes the
economy's own prior-period outcome and current simulated firm conditions. It is
not new public news and it is not a realized historical target.

The inflation and income-growth fields are expected percentage changes over
the next 12 months. They are beliefs, not an instruction to cut next month's
cash income. Unless the card contains a specific next-month income event, treat
the supplied monthly income components as continuing next month. The personal
job-loss prior, when present, is the respondent's
reported chance of losing the current job over 12 months. Translate that personal
prior into a next-month probability rather than copying it. The separate
unemployment-higher probability is an aggregate U.S. outlook and must never be
treated as personal job-loss risk. If this household is currently not employed,
return zero for job-loss probability.

This rolling household-demand forecast holds the observed employment share,
hours, and wage income fixed for the next month. The job-loss and labor-hour
answers are recorded beliefs and conditional plans for later counterfactual
work; they do not change employment or income in this run. The executor applies
the employed policy to the currently employed share and the not-employed policy
to the currently not-employed share. Do not make either policy react to a job
loss that this rolling executor does not realize. Work and job-search hours are
totals for next month, conditional on the corresponding labor state. Do not
rebuild a prudent budget from zero. The
baseline committed and discretionary amounts are an origin-safe estimate of
this household's recent typical monthly expenditures, expressed in the same
dollar scale as its income and balance sheet. Treat that estimate as the
household's current normal spending level; it is not a newly observed bank
statement. The status_quo_next_month amounts apply the latest origin-visible
nominal spending drift to that normal level. They are a neutral, dated starting
point, not a future realization or target. Begin there and make household-specific
changes only when this household's state, prior behavior, or dated information
supports them. For each employment branch, return the total
nominal dollars this household will actually spend next month, not an adjustment,
residual, prudent budget, or recommendation. Include ordinary recurring purchases,
expected price changes, known bill changes, and actual quantity or habit changes.
If the household buys the same basket while prices rise, next month's nominal
expenditure should normally be higher. Do not copy aggregate PCE growth or
manufacture an aggregate forecast. Report unconstrained intended spending; do
not reduce it merely to create a prudent savings plan. Code applies balances and
constraints. Deposits are the residual after spending, debt payments, borrowing,
and income, so do not choose a separate deposit contribution.
extra_debt_payment_usd excludes the stated minimum payment.
current_state.annualized_wage_income is current hourly wage times current
monthly hours times 12; it is not a separate survey-income measure.
current_state.monthly_household_earned_income is SCF family earned income. It is
held fixed in this household-demand forecast because the data do not identify
this respondent's wage share. Respondent employment must not be interpreted as
the employment status of every earner in the household.
current_state.monthly_taxes_nondeposit_saving_and_omitted_outflows is a fixed
budget outflow calibrated from the declared total-saving rate and the household's
liquid-buffer gap. It includes taxes, recurring obligations, and saving routed
outside liquid deposits. The separate liquid-saving target closes any buffer gap
gradually over twelve months. Do not count the fixed outflow as money available
for spending or debt repayment. A positive baseline cash deficit means the matched
income and expenditure sources imply that this household normally draws down cash
or borrows; do not silently turn it into a saver.

For every p10/p50/p90 block, return finite numbers satisfying p10 <= p50 <=
p90. Inflation must be within [-25, 40], income growth within [-50, 50], job
loss probability within [0, 100], work hours within [0, 320], and job-search hours within [0, 200]. Planned work hours
are hours conditional on being employed next month; if currently unemployed,
report hours conditional on finding work. The rolling household-demand executor
does not use these labor fields to alter the observed employment state.
Consumption, debt-payment, and borrowing fields must be nonnegative.

INPUT
{json.dumps(card, indent=2, sort_keys=True)}

Return exactly one JSON object with this shape:
{{
  "prompt_version": "{HOUSEHOLD_PROMPT_VERSION}",
  "household_id": "{card['household']['household_id']}",
  "expected_inflation_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "expected_income_growth_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "job_loss_probability_pct": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "planned_work_hours": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "planned_job_search_hours": {{"p10": 0.0, "p50": 0.0, "p90": 0.0}},
  "employed_policy": {{
    "next_month_committed_consumption_nominal_usd": 0.0,
    "next_month_discretionary_consumption_nominal_usd": 0.0,
    "extra_debt_payment_usd": 0.0,
    "borrowing_intent_usd": 0.0
  }},
  "not_employed_policy": {{
    "next_month_committed_consumption_nominal_usd": 0.0,
    "next_month_discretionary_consumption_nominal_usd": 0.0,
    "extra_debt_payment_usd": 0.0,
    "borrowing_intent_usd": 0.0
  }},
  "reason_codes": ["short_machine_readable_reason"]
}}
""".strip()


def fixture_response(card: Mapping[str, Any]) -> dict[str, Any]:
    state = card["household"]["current_state"]
    profile = card["household"]["profile"]
    liquid = float(state.get("liquid_assets") or 0.0)
    monthly = max(1.0, float(state.get("monthly_consumption") or 1.0))
    debt = max(0.0, float(state.get("revolving_debt") or 0.0))
    employed = bool(state.get("employed"))
    inflation = 3.1
    income = 0.4 if employed else -1.8
    job_risk = 12.0 if employed else 0.0
    committed = float(
        state.get("status_quo_next_month_committed_consumption")
        or state.get("baseline_committed_consumption")
        or monthly * 0.65
    )
    discretionary = float(
        state.get("status_quo_next_month_discretionary_consumption")
        or max(0.0, monthly - committed)
    )
    return {
        "prompt_version": HOUSEHOLD_PROMPT_VERSION,
        "household_id": card["household"]["household_id"],
        "expected_inflation_pct": {"p10": inflation - 1.2, "p50": inflation, "p90": inflation + 1.8},
        "expected_income_growth_pct": {"p10": income - 2.0, "p50": income, "p90": income + 2.0},
        "job_loss_probability_pct": {"p10": max(0.0, job_risk - 8.0), "p50": job_risk, "p90": min(100.0, job_risk + 16.0)},
        "planned_work_hours": {"p10": 120.0 if employed else 0.0, "p50": 160.0 if employed else 0.0, "p90": 180.0 if employed else 40.0},
        "planned_job_search_hours": {"p10": 1.0, "p50": 4.0 if employed else 30.0, "p90": 12.0 if employed else 80.0},
        "employed_policy": {
            "next_month_committed_consumption_nominal_usd": committed,
            "next_month_discretionary_consumption_nominal_usd": discretionary,
            "extra_debt_payment_usd": min(debt, monthly * 0.03),
            "borrowing_intent_usd": 0.0,
        },
        "not_employed_policy": {
            "next_month_committed_consumption_nominal_usd": committed * 0.92,
            "next_month_discretionary_consumption_nominal_usd": discretionary * 0.45,
            "extra_debt_payment_usd": 0.0,
            "borrowing_intent_usd": max(0.0, monthly - liquid) * 0.15,
        },
        "reason_codes": ["fixture_liquidity_and_employment_state"],
    }


_PAYLOAD_FIELDS = {
    "prompt_version",
    "household_id",
    "expected_inflation_pct",
    "expected_income_growth_pct",
    "job_loss_probability_pct",
    "planned_work_hours",
    "planned_job_search_hours",
    "employed_policy",
    "not_employed_policy",
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
        isinstance(reason, str) and 0 < len(reason) <= 240 for reason in reasons
    ):
        raise ValueError("reason_codes must be a non-empty list of strings up to 240 characters")

    def triplet(name: str) -> QuantileTriplet:
        value = payload[name]
        if not isinstance(value, Mapping) or set(value) != {"p10", "p50", "p90"}:
            raise ValueError(f"{name} must contain only p10/p50/p90")
        return QuantileTriplet(float(value["p10"]), float(value["p50"]), float(value["p90"]))

    def policy(name: str) -> HouseholdPolicyBranch:
        value = payload[name]
        fields = {
            "next_month_committed_consumption_nominal_usd",
            "next_month_discretionary_consumption_nominal_usd",
            "extra_debt_payment_usd",
            "borrowing_intent_usd",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"{name} policy schema mismatch")
        return HouseholdPolicyBranch(
            next_month_committed_consumption_nominal_usd=float(
                value["next_month_committed_consumption_nominal_usd"]
            ),
            next_month_discretionary_consumption_nominal_usd=float(
                value["next_month_discretionary_consumption_nominal_usd"]
            ),
            extra_debt_payment_usd=float(value["extra_debt_payment_usd"]),
            borrowing_intent_usd=float(value["borrowing_intent_usd"]),
        )

    response = HouseholdResponse(
        expected_inflation_pct=triplet("expected_inflation_pct"),
        expected_income_growth_pct=triplet("expected_income_growth_pct"),
        job_loss_probability_pct=triplet("job_loss_probability_pct"),
        planned_consumption_change_pct=None,
        planned_work_hours=triplet("planned_work_hours"),
        planned_job_search_hours=triplet("planned_job_search_hours"),
        target_buffer_months=0.0,
        buffer_contribution_intent_usd=0.0,
        debt_payment_intent_usd=0.0,
        borrowing_intent_usd=0.0,
        employed_policy=policy("employed_policy"),
        not_employed_policy=policy("not_employed_policy"),
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
            max_live_calls=2 if self.call_budget is not None else self.max_live_calls,
            execution_cwd=self.execution_cwd,
        )

    def elicit(self, card: Mapping[str, Any]) -> dict[str, Any]:
        if self.mode == "fixture":
            return {"payload": fixture_response(card), "cache_hit": True}
        assert self.client is not None
        prompt, instructions, cache_name = household_request_identity(
            self.provider, self.model, card
        )
        cache_path = self.client.cache_path(cache_name)
        for attempt in range(2):
            cache_miss = not cache_path.exists()
            reserved = cache_miss and self.mode == "live" and self.call_budget is not None
            if reserved:
                self.call_budget.reserve(cache_name)
            try:
                result = self.client.json_call(prompt, cache_name, instructions=instructions)
                normalize_household_payload(result["payload"], str(card["household"]["household_id"]))
            except Exception as exc:
                if self.mode == "live":
                    cache_path.unlink(missing_ok=True)
                if reserved and self.call_budget is not None:
                    self.call_budget.complete(cache_name, error=f"{type(exc).__name__}: {exc}")
                if attempt == 0 and self.mode == "live":
                    continue
                raise
            if reserved and self.call_budget is not None:
                self.call_budget.complete(cache_name, response=result.get("payload"))
            return result
        raise AssertionError("unreachable household elicitation retry state")

    @property
    def live_call_count(self) -> int:
        return self.client.live_call_count if self.client is not None else 0

    @property
    def cache_hit_count(self) -> int:
        return self.client.cache_hit_count if self.client is not None else 0

    @property
    def tool_isolation_version(self) -> str | None:
        return CODEX_TOOL_ISOLATION_VERSION if self.client is not None else None


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
            "tool_isolation_version": CODEX_TOOL_ISOLATION_VERSION,
            "instruction_context_version": CODEX_INSTRUCTION_CONTEXT_VERSION,
        }
    )
    return prompt, instructions, cache_name
