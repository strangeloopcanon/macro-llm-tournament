from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .env import load_secret_env
from .forecast_cards import ForecastCard
from .llm_common import LLMUnavailable

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "llm_cache"
FORECAST_LLM_PROMPT_VERSION = "spf_direct_forecast_v1"
RECALL_PROMPT_VERSION = "spf_forecast_recall_probe_v1"
SUPPORTED_FORECAST_PROVIDERS = ("codex_cli", "openai_responses", "gemini_cli")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
    if not match:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    return json.loads(match.group(1))


def _cache_key(kind: str, provider: str, model: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"kind": kind, "provider": provider, "model": model, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _json_instructions() -> str:
    return """
Return only valid JSON. You are participating in a macroeconomic forecast tournament.
Use only the information inside the prompt. Do not browse, inspect files, run commands,
or cite realized outcomes. Produce numeric forecasts in the requested units.
""".strip()


def _system_prompt(prompt: str, *, instructions: str | None = None) -> str:
    return f"""
{instructions or _json_instructions()}
{prompt.strip()}
""".strip()


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


class ForecastLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path = WORK_ROOT,
        *,
        mode: str = "fixture",
        max_live_calls: int = 0,
    ):
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.max_live_calls = max(0, int(max_live_calls or 0))
        self.live_call_count = 0
        self.cache_hit_count = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "live" and self.max_live_calls <= 0:
            raise ValueError("--max-live-calls must be positive when --llm-mode live is used")

    def _provider_dir(self) -> Path:
        path = self.cache_dir / self.provider
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, cache_name: str) -> Path:
        return self._provider_dir() / f"{cache_name}.json"

    def forecast_cache_name(self, card: ForecastCard) -> str:
        key = _cache_key(
            "forecast_card",
            self.provider,
            self.model,
            {"prompt_version": FORECAST_LLM_PROMPT_VERSION, "card_id": card.card_id, "prompt": card.prompt_payload},
        )
        return f"spf_forecast_{key}"

    def recall_cache_name(self) -> str:
        key = _cache_key(
            "forecast_recall_probe",
            self.provider,
            self.model,
            {"prompt_version": RECALL_PROMPT_VERSION},
        )
        return f"spf_recall_probe_{key}"

    def _record_live_call(self, cache_name: str) -> None:
        if self.live_call_count >= self.max_live_calls:
            raise LLMUnavailable(f"Forecast live-call cap reached ({self.max_live_calls}); cache miss for {cache_name}")
        self.live_call_count += 1

    def _read_cache(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text())
        self.cache_hit_count += 1
        data["cache_hit"] = True
        data["cache_path"] = str(path)
        data["response_read_utc"] = _utc_now()
        return data

    def _write_cache(self, path: Path, payload: dict[str, Any], **metadata: Any) -> dict[str, Any]:
        data = {
            "provider": self.provider,
            "model": self.model,
            "payload": payload,
            "cache_hit": False,
            "cache_path": str(path),
            "response_created_utc": _utc_now(),
        }
        data.update({key: _jsonable(value) for key, value in metadata.items()})
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    def _codex_binary(self) -> str:
        binary = os.getenv("CODEX_CLI_BIN") or shutil.which("codex")
        if not binary:
            raise LLMUnavailable("codex CLI binary not found; set CODEX_CLI_BIN or install codex")
        return binary

    def _gemini_binary(self) -> str:
        binary = os.getenv("GEMINI_CLI_BIN") or shutil.which("gemini")
        if not binary:
            raise LLMUnavailable("gemini CLI binary not found; set GEMINI_CLI_BIN or install gemini")
        return binary

    def _gemini_system_settings_path(self) -> Path:
        configured = os.getenv("GEMINI_CLI_SYSTEM_SETTINGS_PATH")
        if configured:
            return Path(configured)
        path = self.cache_dir / "gemini_cli" / "system_settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "security": {
                            "auth": {
                                "selectedType": "gemini-api-key",
                                "enforcedType": "gemini-api-key",
                            }
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return path

    def _gemini_env(self) -> dict[str, str]:
        load_secret_env()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY is required for provider gemini_cli")
        env = dict(os.environ)
        env["GEMINI_API_KEY"] = api_key
        env["GEMINI_DEFAULT_AUTH_TYPE"] = "gemini-api-key"
        env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(self._gemini_system_settings_path())
        env.setdefault("TERM", "xterm-256color")
        return env

    def _codex_call(self, prompt: str, cache_name: str, *, instructions: str | None = None) -> dict[str, Any]:
        cache_path = self.cache_path(cache_name)
        if cache_path.exists():
            return self._read_cache(cache_path)
        if self.mode == "replay":
            raise LLMUnavailable(f"Replay mode cache miss for {cache_name}")
        if self.mode != "live":
            raise LLMUnavailable(f"LLM mode {self.mode} cannot make live forecast calls")
        self._record_live_call(cache_name)
        last_message_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.last_message.txt")
        command = [
            self._codex_binary(),
            "exec",
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
                input=_system_prompt(prompt, instructions=instructions),
                text=True,
                capture_output=True,
                cwd=str(PROJECT_ROOT),
                timeout=float(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "240")),
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise LLMUnavailable(f"Codex CLI failed with exit code {result.returncode}: {stderr[:700]}")
            text = last_message_path.read_text() if last_message_path.exists() else result.stdout
            payload = _extract_json(text or result.stdout or "")
            data = self._write_cache(cache_path, payload)
            data["provider_binary"] = command[0]
            return data
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(f"Codex CLI timed out after {exc.timeout} seconds") from exc
        finally:
            if last_message_path.exists():
                last_message_path.unlink()

    def _openai_client(self) -> Any:
        if OpenAI is None:
            raise LLMUnavailable("openai package not installed; add `openai` or use provider codex_cli")
        load_secret_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMUnavailable("OPENAI_API_KEY is required for provider openai_responses")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
        if os.getenv("OPENAI_ORG_ID"):
            kwargs["organization"] = os.getenv("OPENAI_ORG_ID")
        if os.getenv("OPENAI_PROJECT_ID"):
            kwargs["project"] = os.getenv("OPENAI_PROJECT_ID")
        return OpenAI(**kwargs)

    def _openai_responses_call(self, prompt: str, cache_name: str, *, instructions: str | None = None) -> dict[str, Any]:
        cache_path = self.cache_path(cache_name)
        if cache_path.exists():
            return self._read_cache(cache_path)
        if self.mode == "replay":
            raise LLMUnavailable(f"Replay mode cache miss for {cache_name}")
        if self.mode != "live":
            raise LLMUnavailable(f"LLM mode {self.mode} cannot make live forecast calls")
        self._record_live_call(cache_name)
        max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096"))
        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low")
        try:
            response = self._openai_client().responses.create(
                model=self.model,
                instructions=instructions or _json_instructions(),
                input=prompt.strip(),
                max_output_tokens=max_output_tokens,
                reasoning={"effort": reasoning_effort},
                store=False,
                timeout=float(os.getenv("OPENAI_API_TIMEOUT_SECONDS", "240")),
            )
            text = _response_output_text(response)
            payload = _extract_json(text)
            return self._write_cache(
                cache_path,
                payload,
                provider_response_id=getattr(response, "id", None),
                provider_response_status=getattr(response, "status", None),
                provider_usage=getattr(response, "usage", None),
                provider_model=getattr(response, "model", None),
            )
        except Exception as exc:
            if isinstance(exc, (LLMUnavailable, ValueError, json.JSONDecodeError)):
                raise
            raise LLMUnavailable(f"OpenAI Responses API call failed: {type(exc).__name__}: {str(exc)[:700]}") from exc

    def _gemini_cli_call(self, prompt: str, cache_name: str, *, instructions: str | None = None) -> dict[str, Any]:
        cache_path = self.cache_path(cache_name)
        if cache_path.exists():
            return self._read_cache(cache_path)
        if self.mode == "replay":
            raise LLMUnavailable(f"Replay mode cache miss for {cache_name}")
        if self.mode != "live":
            raise LLMUnavailable(f"LLM mode {self.mode} cannot make live forecast calls")
        self._record_live_call(cache_name)
        command = [
            self._gemini_binary(),
            "--model",
            self.model,
            "--prompt",
            "",
            "--output-format",
            "json",
            "--skip-trust",
        ]
        try:
            result = subprocess.run(
                command,
                input=_system_prompt(prompt, instructions=instructions),
                text=True,
                capture_output=True,
                cwd=str(PROJECT_ROOT),
                timeout=float(os.getenv("GEMINI_CLI_TIMEOUT_SECONDS", "300")),
                check=False,
                env=self._gemini_env(),
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise LLMUnavailable(f"Gemini CLI failed with exit code {result.returncode}: {stderr[:700]}")
            wrapper = _extract_json(result.stdout or "")
            if not isinstance(wrapper, dict):
                raise ValueError("Gemini CLI returned a non-object JSON wrapper")
            text = str(wrapper.get("response") or "")
            payload = _extract_json(text)
            return self._write_cache(
                cache_path,
                payload,
                provider_binary=command[0],
                provider_session_id=wrapper.get("session_id"),
                provider_stats=wrapper.get("stats"),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(f"Gemini CLI timed out after {exc.timeout} seconds") from exc
        except Exception as exc:
            if isinstance(exc, (LLMUnavailable, ValueError, json.JSONDecodeError)):
                raise
            raise LLMUnavailable(f"Gemini CLI call failed: {type(exc).__name__}: {str(exc)[:700]}") from exc

    def _call(self, prompt: str, cache_name: str, *, instructions: str | None = None) -> dict[str, Any]:
        if self.provider == "codex_cli":
            return self._codex_call(prompt, cache_name, instructions=instructions)
        if self.provider == "openai_responses":
            return self._openai_responses_call(prompt, cache_name, instructions=instructions)
        if self.provider == "gemini_cli":
            return self._gemini_cli_call(prompt, cache_name, instructions=instructions)
        raise LLMUnavailable(f"Unsupported forecast provider: {self.provider}")

    def json_call(self, prompt: str, cache_name: str, *, instructions: str) -> dict[str, Any]:
        return self._call(prompt, cache_name, instructions=instructions)

    def forecast_card(self, card: ForecastCard) -> dict[str, Any]:
        if self.mode == "fixture":
            return fixture_forecast(card, self.provider, self.model)
        prompt = forecast_prompt(card)
        return self._call(prompt, self.forecast_cache_name(card))

    def recall_probe(self) -> dict[str, Any]:
        if self.mode == "fixture":
            return {
                "provider": self.provider,
                "model": self.model,
                "payload": {
                    "prompt_version": RECALL_PROMPT_VERSION,
                    "knows_spf_dataset": False,
                    "knows_forecast_literature": False,
                    "contamination_risk": "fixture",
                    "reason": "fixture mode",
                },
                "cache_hit": True,
            }
        prompt = """
Return JSON with:
{
  "prompt_version": "spf_forecast_recall_probe_v1",
  "knows_spf_dataset": true/false,
  "knows_forecast_literature": true/false,
  "contamination_risk": "low"|"medium"|"high",
  "known_patterns": ["..."],
  "reason": "short explanation"
}

Question: Without using tools, what do you know about the Philadelphia Fed Survey of
Professional Forecasters and behavioral-expectations forecast tests such as underreaction,
overreaction, constant-gain learning, and diagnostic expectations?
"""
        instructions = """
Return only valid JSON. This is a general contamination probe.
Do not browse, inspect files, run commands, or cite hidden outcome data.
""".strip()
        return self._call(prompt, self.recall_cache_name(), instructions=instructions)


def forecast_prompt(card: ForecastCard) -> str:
    payload = dict(card.prompt_payload)
    return f"""
Forecast card:
{json.dumps(payload, indent=2, sort_keys=True)}

Return exactly this JSON shape:
{{
  "prompt_version": "{FORECAST_LLM_PROMPT_VERSION}",
  "point_forecast": 0.0,
  "p10": 0.0,
  "p50": 0.0,
  "p90": 0.0,
  "confidence": 0.0,
  "forecaster_draws": [
    {{"forecaster_id": "sim_01", "forecast": 0.0}}
  ],
  "reason": "short reason based only on the supplied history"
}}

Use 8 forecaster_draws. Do not include markdown.
""".strip()


def fixture_forecast(card: ForecastCard, provider: str, model: str) -> dict[str, Any]:
    recent = card.rolling_signal_mean_4
    trend = 0.25 * card.recent_signal_change_4
    point = float(recent + trend)
    width = max(0.1, 1.35 * float(card.recent_signal_volatility_8 or 0.0))
    draws = [point + offset * width for offset in np.linspace(-0.75, 0.75, 8)]
    return {
        "provider": provider,
        "model": model,
        "payload": {
            "prompt_version": FORECAST_LLM_PROMPT_VERSION,
            "point_forecast": point,
            "p10": point - width,
            "p50": point,
            "p90": point + width,
            "confidence": 0.5,
            "forecaster_draws": [
                {"forecaster_id": f"fixture_{idx + 1:02d}", "forecast": float(value)}
                for idx, value in enumerate(draws)
            ],
            "reason": "deterministic fixture forecast from recent realized trend",
        },
        "cache_hit": True,
        "cache_path": None,
    }


def normalize_forecast_payload(card: ForecastCard, data: dict[str, Any], *, source: str) -> dict[str, Any]:
    payload = data.get("payload", data)
    point = _required_float(payload, "point_forecast")
    p10 = _required_float(payload, "p10")
    p50 = _required_float(payload, "p50")
    p90 = _required_float(payload, "p90")
    confidence = _required_float(payload, "confidence")
    if not (p10 <= p50 <= p90):
        raise LLMUnavailable(f"Invalid forecast interval for card {card.card_id}: expected p10 <= p50 <= p90")
    if not 0.0 <= confidence <= 1.0:
        raise LLMUnavailable(f"Invalid confidence for card {card.card_id}: expected 0 <= confidence <= 1")
    draws = payload.get("forecaster_draws") or []
    if not isinstance(draws, list) or len(draws) != 8:
        raise LLMUnavailable(f"Invalid forecaster_draws for card {card.card_id}: expected exactly 8 draws")
    draw_values = [_finite_float(row.get("forecast"), default=np.nan) for row in draws if isinstance(row, dict)]
    draw_values = [value for value in draw_values if np.isfinite(value)]
    if len(draw_values) != 8:
        raise LLMUnavailable(f"Invalid forecaster_draws for card {card.card_id}: all 8 draws must be numeric")
    return {
        "card_id": card.card_id,
        "source": source,
        "provider": data.get("provider") or source,
        "model": data.get("model"),
        "variable": card.variable,
        "origin": card.origin,
        "origin_index": card.origin_index,
        "horizon": card.horizon,
        "point_forecast": point,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "confidence": confidence,
        "panel_mean": float(np.mean(draw_values)) if draw_values else np.nan,
        "panel_std": float(np.std(draw_values)) if len(draw_values) > 1 else np.nan,
        "params_json": json.dumps({"reason": str(payload.get("reason", ""))[:300]}, sort_keys=True),
        "cache_hit": bool(data.get("cache_hit", False)),
        "cache_path": data.get("cache_path"),
    }


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, *, default: float) -> float:
    if _is_finite(value):
        return float(value)
    return float(default)


def _required_float(payload: dict[str, Any], field: str) -> float:
    if not _is_finite(payload.get(field)):
        raise LLMUnavailable(f"Invalid forecast payload: missing numeric `{field}`")
    return float(payload[field])


def run_llm_forecasts(client: ForecastLLMClient, cards: Iterable[ForecastCard]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    source = f"llm_{client.provider}_{client.model}".replace("/", "_")
    for card in cards:
        data = client.forecast_card(card)
        rows.append(normalize_forecast_payload(card, data, source=source))
        raw_records.append(
            {
                "card_id": card.card_id,
                "provider": client.provider,
                "model": client.model,
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
    return pd.DataFrame(rows), raw_records
