"""Minimal Codex CLI JSON transport with durable replay caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


CODEX_TOOL_ISOLATION_VERSION = "codex_cli_no_shell_web_textfiles_v2"
CODEX_INSTRUCTION_CONTEXT_VERSION = "instruction_free_codex_home_v1"
_DISABLED_CODEX_FEATURES = (
    "shell_tool",
    "apps",
    "enable_mcp_apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "in_app_browser",
    "multi_agent",
    "multi_agent_v2",
    "tool_search",
    "plugin_sharing",
    "workspace_dependencies",
    "memories",
    "image_generation",
    "imagegenext",
)


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
            if (
                data.get("provider") != self.provider
                or data.get("model") != self.model
                or data.get("request_sha256") != cache_name
                or data.get("tool_isolation_version") != CODEX_TOOL_ISOLATION_VERSION
                or data.get("instruction_context_version")
                != CODEX_INSTRUCTION_CONTEXT_VERSION
            ):
                raise ProviderUnavailable("cached request identity or execution context mismatch")
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
        system_prompt = (
            instructions.strip()
            + "\n\n"
            + prompt.strip()
            + "\n\nReturn only the requested JSON object."
        )
        with tempfile.TemporaryDirectory(prefix="household-ecology-codex-") as isolated:
            isolated_cwd = Path(isolated)
            isolated_codex_home = isolated_cwd / ".codex-home"
            isolated_codex_home.mkdir()
            configured_codex_home = Path(
                os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
            ).expanduser()
            auth_source = configured_codex_home / "auth.json"
            if not auth_source.is_file():
                raise ProviderUnavailable(
                    "Codex authentication file is unavailable for isolated execution"
                )
            (isolated_codex_home / "auth.json").symlink_to(auth_source.resolve())
            message_path = isolated_cwd / "last_message.txt"
            command = [
                binary,
                "exec",
                "--model",
                self.model,
                *self._reasoning_args(),
                "--cd",
                str(isolated_cwd),
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--ignore-user-config",
                "-c",
                'web_search="disabled"',
                *(item for feature in _DISABLED_CODEX_FEATURES for item in ("--disable", feature)),
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--output-last-message",
                str(message_path),
                "-",
            ]
            try:
                environment = os.environ.copy()
                environment["CODEX_HOME"] = str(isolated_codex_home)
                result = subprocess.run(
                    command,
                    input=system_prompt,
                    text=True,
                    capture_output=True,
                    cwd=isolated_cwd,
                    env=environment,
                    timeout=float(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "600")),
                    check=False,
                )
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "").strip()
                    raise ProviderUnavailable(
                        f"codex CLI exited {result.returncode}: {detail[:700]}"
                    )
                text = (
                    message_path.read_text(encoding="utf-8")
                    if message_path.exists()
                    else result.stdout
                )
                payload = _extract_json(text)
                data = {
                    "provider": self.provider,
                    "model": self.model,
                    "request_sha256": cache_name,
                    "payload": payload,
                    "cache_hit": False,
                    "cache_path": str(cache_path),
                    "response_created_utc": _utc_now(),
                    "tool_isolation_version": CODEX_TOOL_ISOLATION_VERSION,
                    "instruction_context_version": CODEX_INSTRUCTION_CONTEXT_VERSION,
                }
                cache_path.write_text(
                    json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                return data
            except subprocess.TimeoutExpired as exc:
                raise ProviderUnavailable(
                    f"codex CLI timed out after {exc.timeout} seconds"
                ) from exc

    @staticmethod
    def _reasoning_args() -> list[str]:
        effort = os.getenv("CODEX_CLI_REASONING_EFFORT") or os.getenv("CODEX_REASONING_EFFORT")
        return ["-c", f'model_reasoning_effort="{effort.strip()}"'] if effort else []


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
