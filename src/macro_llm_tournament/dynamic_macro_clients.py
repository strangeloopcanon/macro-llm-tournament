"""Demand clients, replay contracts, and belief-payload identity checks."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .agent_common import cache_key
from .demand_economy import (
    DEMAND_ECONOMY_PROMPT_VERSION,
    DemandEconomyClient,
    DemandScenario,
    adaptive_belief_payload,
    belief_module_prompt_payload,
    normalize_belief_payload,
)
from .dynamic_macro_common import (
    DynamicMacroError,
    _optional_signal,
    _sha256_json,
)
from .llm_common import LLMUnavailable


class ReplayThenLiveDemandClient:
    """Replay a locked recursive prefix, then call the live provider for new states."""

    variant = "llm_belief"

    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        *,
        replay_records: list[dict[str, Any]],
        replay_prefix_period_count: int,
        max_live_calls: int,
        semantic_retry_limit: int,
        execution_cwd: Path | None,
    ) -> None:
        validate_replay_prefix_records(
            replay_records,
            provider=provider,
            model=model,
            prefix_period_count=replay_prefix_period_count,
        )
        self.provider = provider
        self.model = model
        self.replay_prefix_period_count = int(replay_prefix_period_count)
        self.semantic_retry_limit = int(semantic_retry_limit)
        self.cache_dir = cache_dir
        self._replay = DemandEconomyClient(
            provider,
            model,
            cache_dir,
            mode="raw_replay",
            variant=self.variant,
            raw_replay_records=replay_records,
        )
        self._live = DemandEconomyClient(
            provider,
            model,
            cache_dir,
            mode="live",
            variant=self.variant,
            max_live_calls=max_live_calls,
            execution_cwd=execution_cwd,
        )
        self._raw_records: list[dict[str, Any]] = []
        self._replayed_record_count = 0
        self._semantic_retry_count = 0
        self._rejected_semantic_payloads: list[dict[str, Any]] = []

    @property
    def source(self) -> str:
        return f"llm_belief_replay_live_{self.provider}_{self.model}"

    @property
    def live_call_count(self) -> int:
        return self._live.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._replay.cache_hit_count + self._live.cache_hit_count

    @property
    def replayed_record_count(self) -> int:
        return self._replayed_record_count

    @property
    def semantic_retry_count(self) -> int:
        return self._semantic_retry_count

    @property
    def rejected_semantic_payloads(self) -> list[dict[str, Any]]:
        return list(self._rejected_semantic_payloads)

    @property
    def raw_records(self) -> list[dict[str, Any]]:
        return self._raw_records

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        client = (
            self._replay
            if int(period_state["period_index"]) < self.replay_prefix_period_count
            else self._live
        )
        if client is self._replay:
            panel = client.belief_panel(scenario, period_state, household_states)
        else:
            panel = self._live_belief_panel_with_semantic_retry(
                scenario, period_state, household_states
            )
        if client is self._replay:
            self._replayed_record_count += 1
        self._raw_records.append(dict(client.raw_records[-1]))
        return panel

    def _live_belief_panel_with_semantic_retry(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for attempt in range(self.semantic_retry_limit + 1):
            try:
                return self._live.belief_panel(
                    scenario, period_state, household_states
                )
            except LLMUnavailable as exc:
                if attempt >= self.semantic_retry_limit:
                    raise
                cache_path = self._live.belief_cache_path(
                    scenario, period_state, household_states
                )
                if not cache_path.is_file():
                    raise
                rejected_dir = self.cache_dir / "rejected_semantic"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                payload_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                rejected_path = rejected_dir / (
                    f"{cache_path.stem}.attempt_{attempt + 1}.{payload_sha[:12]}.json"
                )
                cache_path.replace(rejected_path)
                self._semantic_retry_count += 1
                self._rejected_semantic_payloads.append(
                    {
                        "period_index": int(period_state["period_index"]),
                        "payload_sha256": payload_sha,
                        "relative_path": str(
                            rejected_path.relative_to(self.cache_dir.parent)
                        ),
                        "reason": str(exc)[:500],
                    }
                )
        raise AssertionError("unreachable semantic retry loop")

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.belief_panel(scenario, period_state, household_states)


def validate_replay_prefix_records(
    records: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    prefix_period_count: int,
) -> None:
    if prefix_period_count <= 0:
        raise DynamicMacroError("replay_live requires a positive replay prefix")
    matching = [
        row
        for row in records
        if str(row.get("provider")) == provider
        and str(row.get("model")) == model
        and str(row.get("variant")) == "llm_belief"
        and str(row.get("scenario_id")) == "recursive_monthly_path"
    ]
    try:
        periods = [int(row["period_index"]) for row in matching]
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicMacroError("Replay prefix records have malformed period indexes") from exc
    expected = list(range(prefix_period_count))
    if sorted(periods) != expected or len(periods) != len(set(periods)):
        raise DynamicMacroError(
            f"Replay prefix must contain exactly periods {expected}; found {sorted(periods)}"
        )
    for row in matching:
        identity = row.get("cache_identity")
        if not isinstance(identity, dict) or not identity.get("state_identity_sha256"):
            raise DynamicMacroError("Replay prefix record lacks its state identity")


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
        replay_prefix_period_count: int = 0,
    ) -> None:
        self.base = base
        self.gains = dict(gains)
        self.requested_mode = requested_mode
        self.replay_prefix_period_count = int(replay_prefix_period_count)
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
    def replayed_record_count(self) -> int:
        return int(getattr(self.base, "replayed_record_count", 0))

    @property
    def semantic_retry_count(self) -> int:
        return int(getattr(self.base, "semantic_retry_count", 0))

    @property
    def rejected_semantic_payloads(self) -> list[dict[str, Any]]:
        return list(getattr(self.base, "rejected_semantic_payloads", []))

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
            replay_required = (
                self.requested_mode == "replay"
                or int(period_state["period_index"]) < self.replay_prefix_period_count
            )
            if replay_required and (
                record is None or record.get("cache_identity") != identity
            ):
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
        if self.requested_mode not in {"replay", "replay_live", "live"}:
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
