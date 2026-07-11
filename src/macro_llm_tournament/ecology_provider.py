"""Minimal Codex CLI JSON transport with durable replay caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


class ProviderUnavailable(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("provider response must be one JSON object")
    return payload


class CodexJSONClient:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        mode: str,
        max_live_calls: int,
        execution_cwd: Path,
    ) -> None:
        if mode not in {"replay", "live"}:
            raise ValueError("CodexJSONClient mode must be replay or live")
        self.provider = "codex_cli"
        self.model = model
        self.cache_dir = cache_dir
        self.mode = mode
        self.max_live_calls = max(0, int(max_live_calls))
        self.execution_cwd = execution_cwd.resolve()
        self.live_call_count = 0
        self.cache_hit_count = 0

    def cache_path(self, cache_name: str) -> Path:
        path = self.cache_dir / self.provider / f"{cache_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def json_call(self, prompt: str, cache_name: str, *, instructions: str) -> dict[str, Any]:
        cache_path = self.cache_path(cache_name)
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("provider") != self.provider or data.get("model") != self.model:
                raise ProviderUnavailable("cached provider/model identity mismatch")
            self.cache_hit_count += 1
            return data | {"cache_hit": True, "cache_path": str(cache_path)}
        if self.mode == "replay":
            raise ProviderUnavailable(f"replay cache miss for {cache_name}")
        if self.live_call_count >= self.max_live_calls:
            raise ProviderUnavailable(f"live-call cap reached for {cache_name}")
        binary = os.getenv("CODEX_CLI_BIN") or shutil.which("codex")
        if not binary:
            raise ProviderUnavailable("codex CLI not found")
        self.live_call_count += 1
        message_path = cache_path.with_suffix(f".{os.getpid()}.last_message.txt")
        command = [
            binary,
            "exec",
            "--model",
            self.model,
            *self._reasoning_args(),
            "--cd",
            str(self.execution_cwd),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            str(message_path),
            "-",
        ]
        system_prompt = (
            instructions.strip()
            + "\n\n"
            + prompt.strip()
            + "\n\nReturn only the requested JSON object."
        )
        try:
            result = subprocess.run(
                command,
                input=system_prompt,
                text=True,
                capture_output=True,
                cwd=self.execution_cwd,
                timeout=float(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "600")),
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise ProviderUnavailable(
                    f"codex CLI exited {result.returncode}: {detail[:700]}"
                )
            text = message_path.read_text(encoding="utf-8") if message_path.exists() else result.stdout
            payload = _extract_json(text)
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": payload,
                "cache_hit": False,
                "cache_path": str(cache_path),
                "response_created_utc": _utc_now(),
            }
            cache_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return data
        except subprocess.TimeoutExpired as exc:
            raise ProviderUnavailable(f"codex CLI timed out after {exc.timeout} seconds") from exc
        finally:
            message_path.unlink(missing_ok=True)

    @staticmethod
    def _reasoning_args() -> list[str]:
        effort = os.getenv("CODEX_CLI_REASONING_EFFORT") or os.getenv("CODEX_REASONING_EFFORT")
        return ["-c", f'model_reasoning_effort="{effort.strip()}"'] if effort else []


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
