from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_behavior import (
    bank_credit_multiplier,
    forecast_for_prompt,
    forecast_uncertainty,
    firm_hiring_index,
    firm_price_pressure_index,
    response_by_variable,
    standardized_signal,
    updated_belief,
)
from .agent_common import (
    AGENT_LLM_CACHE_ROOT,
    AGENT_LLM_PROMPT_VERSION,
    PROJECT_ROOT,
    bounded_float,
    cache_key,
    extract_json,
    round_or_none,
)
from .forecast_cards import ForecastCard
from .llm_common import LLMUnavailable


class AgentLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path = AGENT_LLM_CACHE_ROOT,
        *,
        mode: str = "fixture",
        max_live_calls: int = 0,
        system_preamble: str | None = None,
    ):
        if mode not in {"fixture", "replay", "live"}:
            raise ValueError(f"Unsupported agent LLM mode: {mode}")
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.system_preamble = system_preamble
        self.max_live_calls = max(0, int(max_live_calls or 0))
        self.live_call_count = 0
        self.cache_hit_count = 0
        self.raw_records: list[dict[str, Any]] = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "live" and self.max_live_calls <= 0:
            raise ValueError("--max-agent-live-calls must be positive when --agent-mode live is used")

    def _provider_dir(self) -> Path:
        path = self.cache_dir / self.provider
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, cache_name: str) -> Path:
        return self._provider_dir() / f"{cache_name}.json"

    def agent_cache_name(self, prompt: str) -> str:
        return f"typed_agent_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"

    def agent_panel(
        self,
        card: ForecastCard,
        forecast: pd.Series,
        type_cells: pd.DataFrame,
        prior_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_agent_payload(card, forecast, type_cells, prior_states),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = agent_prompt(card, forecast, type_cells, prior_states)
            data = self._codex_call(prompt, self.agent_cache_name(prompt))
        normalized = normalize_agent_payload(card, forecast, type_cells, data)
        self.raw_records.append(
            {
                "card_id": card.card_id,
                "origin": card.origin,
                "variable": card.variable,
                "source": str(forecast["source"]),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized

    def _record_live_call(self, cache_name: str) -> None:
        if self.live_call_count >= self.max_live_calls:
            raise LLMUnavailable(f"Agent live-call cap reached ({self.max_live_calls}); cache miss for {cache_name}")
        self.live_call_count += 1

    def _read_cache(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.cache_hit_count += 1
        data["cache_hit"] = True
        data["cache_path"] = str(path)
        return data

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        data = {
            "provider": self.provider,
            "model": self.model,
            "payload": payload,
            "cache_hit": False,
            "cache_path": str(path),
            "response_created_utc": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    def _codex_binary(self) -> str:
        binary = os.getenv("CODEX_CLI_BIN") or shutil.which("codex")
        if not binary:
            raise LLMUnavailable("codex CLI binary not found; set CODEX_CLI_BIN or install codex")
        return binary

    def _cursor_binary(self) -> str:
        binary = os.getenv("CURSOR_CLI_BIN") or shutil.which("cursor-agent")
        if not binary:
            raise LLMUnavailable("cursor-agent CLI binary not found; set CURSOR_CLI_BIN or install cursor-agent")
        return binary

    def _cursor_neutral_workspace(self) -> Path:
        # cursor-agent in ask mode can read its workspace, so point it at an
        # empty directory: the model must answer from the prompt alone and can
        # never inspect repo data, targets, or caches.
        path = self.cache_dir / "cursor_neutral_workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _codex_call(self, prompt: str, cache_name: str) -> dict[str, Any]:
        cache_path = self.cache_path(cache_name)
        if cache_path.exists():
            return self._read_cache(cache_path)
        if self.mode == "replay":
            raise LLMUnavailable(f"Agent replay mode cache miss for {cache_name}")
        if self.mode != "live":
            raise LLMUnavailable(f"Agent mode {self.mode} cannot make live calls")
        self._record_live_call(cache_name)
        if self.provider == "cursor_cli":
            return self._run_cursor_cli(prompt, cache_path)
        return self._run_codex_cli(prompt, cache_path)

    def _wrapped_prompt(self, prompt: str) -> str:
        if self.system_preamble is not None:
            return f"{self.system_preamble.strip()}\n\n{prompt.strip()}"
        return _agent_system_prompt(prompt)

    def _codex_config_args(self) -> list[str]:
        reasoning_effort = os.getenv("CODEX_CLI_REASONING_EFFORT") or os.getenv("CODEX_REASONING_EFFORT")
        if not reasoning_effort:
            return []
        return ["-c", f'model_reasoning_effort="{reasoning_effort.strip()}"']

    def _run_codex_cli(self, prompt: str, cache_path: Path) -> dict[str, Any]:
        last_message_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.last_message.txt")
        command = [
            self._codex_binary(),
            "exec",
            *self._codex_config_args(),
            "--model",
            self.model,
            "--cd",
            str(PROJECT_ROOT),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            str(last_message_path),
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                input=self._wrapped_prompt(prompt),
                text=True,
                capture_output=True,
                cwd=str(PROJECT_ROOT),
                timeout=float(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "240")),
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise LLMUnavailable(f"Codex CLI failed with exit code {result.returncode}: {stderr[:700]}")
            text = last_message_path.read_text(encoding="utf-8") if last_message_path.exists() else result.stdout
            return self._write_cache(cache_path, extract_json(text or result.stdout or ""))
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(f"Codex CLI timed out after {exc.timeout} seconds") from exc
        finally:
            if last_message_path.exists():
                last_message_path.unlink()

    def _run_cursor_cli(self, prompt: str, cache_path: Path) -> dict[str, Any]:
        command = cursor_cli_command(self._cursor_binary(), self.model, self._cursor_neutral_workspace())
        try:
            result = subprocess.run(
                command,
                input=self._wrapped_prompt(prompt),
                text=True,
                capture_output=True,
                cwd=str(self._cursor_neutral_workspace()),
                timeout=float(os.getenv("CURSOR_CLI_TIMEOUT_SECONDS", "600")),
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise LLMUnavailable(f"Cursor CLI failed with exit code {result.returncode}: {stderr[:700]}")
            return self._write_cache(cache_path, extract_json(result.stdout or ""))
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(f"Cursor CLI timed out after {exc.timeout} seconds") from exc


def cursor_cli_command(binary: str, model: str, workspace: Path) -> list[str]:
    return [
        binary,
        "-p",
        "--output-format",
        "text",
        "--model",
        model,
        "--mode",
        "ask",
        "--trust",
        "--workspace",
        str(workspace),
    ]


def agent_prompt(
    card: ForecastCard,
    forecast: pd.Series,
    type_cells: pd.DataFrame,
    prior_states: list[dict[str, Any]],
) -> str:
    type_rows = [
        {
            "type_id": row["type_id"],
            "label": row["label"],
            "population_weight": round_or_none(row["population_weight"]),
            "annual_income": round_or_none(row["annual_income"]),
            "liquid_assets": round_or_none(row["liquid_assets"]),
            "illiquid_assets": round_or_none(row["illiquid_assets"]),
            "debt": round_or_none(row["debt"]),
            "consumption_proxy_annual": round_or_none(row["consumption_proxy_annual"]),
            "liquid_buffer_months": round_or_none(row["liquid_buffer_months"]),
            "credit_limit_proxy": round_or_none(row["credit_limit_proxy"]),
        }
        for _, row in type_cells.iterrows()
    ]
    state_rows = [
        {
            "type_id": state["type_id"],
            "expected_inflation_1y": round_or_none(state["expected_inflation_1y"]),
            "expected_real_income_growth": round_or_none(state["expected_real_income_growth"]),
            "expected_unemployment_rate": round_or_none(state["expected_unemployment_rate"]),
            "expected_short_rate": round_or_none(state["expected_short_rate"]),
            "confidence": round_or_none(state["confidence"]),
            "uncertainty": round_or_none(state["uncertainty"]),
            "desired_liquid_buffer_months": round_or_none(state["desired_liquid_buffer_months"]),
            "credit_access": state["credit_access"],
        }
        for state in prior_states
    ]
    payload = {
        "prompt_version": AGENT_LLM_PROMPT_VERSION,
        "task": (
            "For each representative household type, update beliefs and choose desired behavior. "
            "These are desired actions before accounting reconciliation; deterministic code will enforce budgets."
        ),
        "as_of_rule": "Use only the supplied as-of forecast card, macro belief forecast, type cells, and prior states.",
        "forecast_card": card.prompt_payload,
        "belief_forecast": forecast_for_prompt(forecast),
        "household_type_cells": type_rows,
        "prior_agent_states": state_rows,
        "allowed_household_type_ids": [row["type_id"] for row in type_rows],
        "required_response": {
            "household_actions": [
                {
                    "type_id": "one supplied type_id",
                    "consumption_change_pct": "desired percent change versus baseline quarterly consumption, -20 to 20",
                    "liquid_buffer_change_pct": "desired percent change in liquid buffer, -25 to 25",
                    "borrowing_desire_index": "positive means borrow more, negative means repay debt, -5 to 5",
                    "portfolio_rebalance_to_liquid_pct": "positive moves illiquid wealth to liquid, -15 to 15",
                    "job_search_intensity_index": "-3 to 6",
                    "expected_inflation_1y": "updated percent expectation",
                    "expected_real_income_growth": "updated percent expectation",
                    "expected_unemployment_rate": "updated percent expectation",
                    "expected_short_rate": "updated percent expectation",
                    "confidence": "0 to 1",
                    "uncertainty": "0 to 1.5",
                }
            ],
            "firm": {
                "hiring_index": "-5 to 5",
                "price_pressure_index": "-5 to 5",
                "confidence": "0 to 1",
            },
            "bank": {
                "credit_supply_multiplier": "0.35 to 1.20",
                "credit_tightening_index": "0 to 1",
                "confidence": "0 to 1",
            },
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_agent_payload(
    card: ForecastCard,
    forecast: pd.Series,
    type_cells: pd.DataFrame,
    prior_states: list[dict[str, Any]],
) -> dict[str, Any]:
    state_by_type = {str(state["type_id"]): state for state in prior_states}
    household_actions: list[dict[str, Any]] = []
    signal = standardized_signal(card, float(forecast["point_forecast"]))
    uncertainty = forecast_uncertainty(forecast, signal)
    for _, type_cell in type_cells.iterrows():
        prior_state = state_by_type[str(type_cell["type_id"])]
        response = response_by_variable(
            card.variable,
            signal,
            liquidity=float(type_cell["liquidity_sensitivity"]),
            rate=float(type_cell["rate_sensitivity"]),
            unemployment=float(type_cell["unemployment_sensitivity"]),
            portfolio=float(type_cell["portfolio_sensitivity"]),
            uncertainty=uncertainty,
        )
        household_actions.append(
            {
                "type_id": str(type_cell["type_id"]),
                "consumption_change_pct": response["consumption_change_pct"],
                "liquid_buffer_change_pct": response["desired_liquid_buffer_change_pct"],
                "borrowing_desire_index": response["borrowing_desire_index"],
                "portfolio_rebalance_to_liquid_pct": response["portfolio_rebalance_to_liquid_pct"],
                "job_search_intensity_index": response["job_search_intensity_index"],
                "expected_inflation_1y": updated_belief(prior_state, card.variable, "CPI", "expected_inflation_1y", forecast),
                "expected_real_income_growth": updated_belief(prior_state, card.variable, "RGDP", "expected_real_income_growth", forecast),
                "expected_unemployment_rate": updated_belief(prior_state, card.variable, "UNEMP", "expected_unemployment_rate", forecast),
                "expected_short_rate": updated_belief(prior_state, card.variable, "TBILL", "expected_short_rate", forecast),
                "confidence": float(np.clip(1.0 - uncertainty / 8.0, 0.05, 0.95)),
                "uncertainty": float(np.clip(uncertainty / 4.0, 0.05, 1.50)),
            }
        )
    bank_multiplier = bank_credit_multiplier(card, [{"belief_signal_vs_history": signal, "population_weight": 1.0}])
    return {
        "prompt_version": AGENT_LLM_PROMPT_VERSION,
        "household_actions": household_actions,
        "firm": {
            "hiring_index": firm_hiring_index(card, signal, 0.0),
            "price_pressure_index": firm_price_pressure_index(card, signal, 0.0),
            "confidence": 0.5,
        },
        "bank": {
            "credit_supply_multiplier": bank_multiplier,
            "credit_tightening_index": float(np.clip(1.0 - bank_multiplier, 0.0, 1.0)),
            "confidence": 0.5,
        },
        "reason": "deterministic fixture agent payload",
    }


def normalize_agent_payload(
    card: ForecastCard,
    forecast: pd.Series,
    type_cells: pd.DataFrame,
    data: dict[str, Any],
) -> dict[str, Any]:
    payload = data.get("payload", data)
    actions = payload.get("household_actions")
    if not isinstance(actions, list):
        raise LLMUnavailable(f"Agent payload for card {card.card_id} is missing household_actions list")
    expected_ids = set(type_cells["type_id"].astype(str))
    by_type: dict[str, dict[str, float]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        type_id = str(action.get("type_id", ""))
        if type_id not in expected_ids:
            continue
        by_type[type_id] = {
            "consumption_change_pct": bounded_float(action, "consumption_change_pct", -20.0, 20.0),
            "liquid_buffer_change_pct": bounded_float(action, "liquid_buffer_change_pct", -25.0, 25.0),
            "borrowing_desire_index": bounded_float(action, "borrowing_desire_index", -5.0, 5.0),
            "portfolio_rebalance_to_liquid_pct": bounded_float(action, "portfolio_rebalance_to_liquid_pct", -15.0, 15.0),
            "job_search_intensity_index": bounded_float(action, "job_search_intensity_index", -3.0, 6.0),
            "expected_inflation_1y": bounded_float(action, "expected_inflation_1y", -5.0, 20.0),
            "expected_real_income_growth": bounded_float(action, "expected_real_income_growth", -20.0, 20.0),
            "expected_unemployment_rate": bounded_float(action, "expected_unemployment_rate", 0.0, 35.0),
            "expected_short_rate": bounded_float(action, "expected_short_rate", -5.0, 20.0),
            "confidence": bounded_float(action, "confidence", 0.0, 1.0),
            "uncertainty": bounded_float(action, "uncertainty", 0.0, 1.5),
        }
    missing = sorted(expected_ids - set(by_type))
    if missing:
        raise LLMUnavailable(f"Agent payload for card {card.card_id} is missing type ids: {', '.join(missing)}")
    firm = payload.get("firm")
    bank = payload.get("bank")
    if not isinstance(firm, dict):
        raise LLMUnavailable(f"Agent payload for card {card.card_id} is missing firm response")
    if not isinstance(bank, dict):
        raise LLMUnavailable(f"Agent payload for card {card.card_id} is missing bank response")
    return {
        "household_by_type": by_type,
        "firm": {
            "hiring_index": bounded_float(firm, "hiring_index", -5.0, 5.0),
            "price_pressure_index": bounded_float(firm, "price_pressure_index", -5.0, 5.0),
            "confidence": bounded_float(firm, "confidence", 0.0, 1.0),
        },
        "bank": {
            "credit_supply_multiplier": bounded_float(bank, "credit_supply_multiplier", 0.35, 1.20),
            "credit_tightening_index": bounded_float(bank, "credit_tightening_index", 0.0, 1.0),
            "confidence": bounded_float(bank, "confidence", 0.0, 1.0),
        },
        "agent_source": f"agent_{data.get('provider', 'unknown')}_{data.get('model', 'unknown')}",
    }


def _agent_system_prompt(prompt: str) -> str:
    return f"""
Return only valid JSON. You are a structured typed-agent simulator inside a macroeconomic
forecast experiment. Use only the supplied as-of information. Do not browse, inspect files,
run commands, cite realized outcomes, or reveal hidden targets. Your outputs are desired
behavior and beliefs; deterministic code will enforce accounting constraints after your reply.

{prompt.strip()}
""".strip()
