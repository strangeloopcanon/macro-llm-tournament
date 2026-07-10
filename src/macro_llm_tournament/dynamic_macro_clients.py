"""Demand clients, replay contracts, and belief-payload identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
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


LIVE_ATTEMPT_SCHEMA_VERSION = "dynamic_macro_live_attempt_v1"
_LIVE_ATTEMPT_STATUSES = frozenset({"started", "accepted", "failed"})
_ATTEMPT_FILE_RE = re.compile(r"attempt_(\d{4,})\.json")


def canonical_live_attempts(journal_dir: Path) -> list[dict[str, Any]]:
    """Return the public ledger derived from durable pre-call journals."""
    if not journal_dir.exists():
        return []
    if not journal_dir.is_dir():
        raise DynamicMacroError("Live-attempt journal path must be a directory")

    journals: list[tuple[int, Path]] = []
    for path in sorted(journal_dir.iterdir()):
        match = _ATTEMPT_FILE_RE.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise DynamicMacroError(f"Malformed live-attempt journal entry: {path}")
        journals.append((int(match.group(1)), path))
    if [number for number, _ in journals] != list(range(1, len(journals) + 1)):
        raise DynamicMacroError("Live-attempt journal sequence is not contiguous")

    rows: list[dict[str, Any]] = []
    for attempt_number, path in journals:
        try:
            journal = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DynamicMacroError(f"Malformed live-attempt journal: {path}") from exc
        row = _canonical_live_attempt_row(journal, attempt_number=attempt_number)
        row["journal_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    validate_live_attempt_ledger(rows)
    return rows


def validate_live_attempt_ledger(
    rows: Any, *, allow_started: bool = False
) -> list[dict[str, Any]]:
    """Validate the serialized ledger without consulting private cache files."""
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise DynamicMacroError("Live-attempt ledger must be a JSON list of objects")
    required = {
        "schema_version",
        "attempt_number",
        "attempt_id",
        "provider",
        "model",
        "period_index",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "cache_file",
        "cache_file_sha256",
        "error_sha256",
        "journal_sha256",
    }
    for expected_number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise DynamicMacroError("Live-attempt ledger row has an invalid schema")
        if (
            row["schema_version"] != LIVE_ATTEMPT_SCHEMA_VERSION
            or row["attempt_number"] != expected_number
            or row["attempt_id"] != f"live_attempt_{expected_number:04d}"
            or not isinstance(row["provider"], str)
            or not row["provider"]
            or not isinstance(row["model"], str)
            or not row["model"]
            or not isinstance(row["period_index"], int)
            or row["period_index"] < 0
            or row["status"] not in _LIVE_ATTEMPT_STATUSES
            or not isinstance(row["started_at_utc"], str)
            or not row["started_at_utc"]
            or not isinstance(row["cache_file"], str)
            or not row["cache_file"]
            or not _is_sha256(row["journal_sha256"])
        ):
            raise DynamicMacroError("Live-attempt ledger row is invalid")
        if row["status"] == "started":
            if not allow_started or row["finished_at_utc"] is not None:
                raise DynamicMacroError("Live-attempt ledger contains an unfinished call")
        elif not isinstance(row["finished_at_utc"], str) or not row["finished_at_utc"]:
            raise DynamicMacroError("Live-attempt ledger row lacks a completion timestamp")
        if row["status"] == "accepted" and not _is_sha256(row["cache_file_sha256"]):
            raise DynamicMacroError("Accepted live-attempt ledger row lacks its cache hash")
        if row["status"] == "failed" and not _is_sha256(row["error_sha256"]):
            raise DynamicMacroError("Failed live-attempt ledger row lacks its error hash")
        if row["cache_file_sha256"] is not None and not _is_sha256(row["cache_file_sha256"]):
            raise DynamicMacroError("Live-attempt ledger cache hash is invalid")
        if row["error_sha256"] is not None and not _is_sha256(row["error_sha256"]):
            raise DynamicMacroError("Live-attempt ledger error hash is invalid")
    return list(rows)


def _canonical_live_attempt_row(
    journal: Any, *, attempt_number: int
) -> dict[str, Any]:
    if not isinstance(journal, dict):
        raise DynamicMacroError("Live-attempt journal must be a JSON object")
    required = {
        "schema_version",
        "attempt_number",
        "attempt_id",
        "provider",
        "model",
        "period_index",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "cache_file",
        "cache_file_sha256",
        "error_sha256",
    }
    if set(journal) != required:
        raise DynamicMacroError("Live-attempt journal has an invalid schema")
    return {key: journal[key] for key in sorted(required)}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class LiveAttemptJournal:
    """Durably record every cache-miss call before it reaches the provider."""

    def __init__(self, journal_dir: Path, *, provider: str, model: str) -> None:
        self.journal_dir = journal_dir
        self.provider = provider
        self.model = model

    def start(self, period_state: Mapping[str, Any], cache_path: Path) -> Path:
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.journal_dir.iterdir())
        numbers: list[int] = []
        for path in existing:
            match = _ATTEMPT_FILE_RE.fullmatch(path.name)
            if match is None or not path.is_file() or path.is_symlink():
                raise DynamicMacroError(f"Malformed live-attempt journal entry: {path}")
            numbers.append(int(match.group(1)))
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise DynamicMacroError("Live-attempt journal sequence is not contiguous")
        attempt_number = len(numbers) + 1
        path = self.journal_dir / f"attempt_{attempt_number:04d}.json"
        _write_durable_json(
            path,
            {
                "schema_version": LIVE_ATTEMPT_SCHEMA_VERSION,
                "attempt_number": attempt_number,
                "attempt_id": f"live_attempt_{attempt_number:04d}",
                "provider": self.provider,
                "model": self.model,
                "period_index": int(period_state["period_index"]),
                "status": "started",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "finished_at_utc": None,
                "cache_file": _cache_file_label(cache_path),
                "cache_file_sha256": None,
                "error_sha256": None,
            },
        )
        return path

    def finish_for_cache(
        self, path: Path, cache_path: Path, *, status: str, error: str | None = None
    ) -> None:
        if status not in {"accepted", "failed"}:
            raise DynamicMacroError("Live-attempt journal has an invalid completion status")
        try:
            journal = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DynamicMacroError(f"Malformed live-attempt journal: {path}") from exc
        _canonical_live_attempt_row(journal, attempt_number=int(journal.get("attempt_number", 0)))
        if journal["status"] != "started":
            raise DynamicMacroError("Live-attempt journal was already completed")
        journal.update(
            {
                "status": status,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "cache_file_sha256": (
                    hashlib.sha256(cache_path.read_bytes()).hexdigest()
                    if cache_path.is_file()
                    else None
                ),
                "error_sha256": (
                    hashlib.sha256(error.encode("utf-8")).hexdigest()
                    if error is not None
                    else None
                ),
            }
        )
        _write_durable_json(path, journal)


def _cache_file_label(path: Path) -> str:
    return path.name


def _write_durable_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


class JournaledLiveDemandClient:
    """Add pre-call audit journals to the live DemandEconomyClient path."""

    def __init__(self, base: DemandEconomyClient, journal: LiveAttemptJournal) -> None:
        self._base = base
        self._journal = journal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def belief_cache_path(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> Path:
        return self._base.belief_cache_path(scenario, period_state, household_states)

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cache_path = self.belief_cache_path(scenario, period_state, household_states)
        attempt_path = (
            None if cache_path.is_file() else self._journal.start(period_state, cache_path)
        )
        try:
            panel = self._base.belief_panel(scenario, period_state, household_states)
        except Exception as exc:
            if attempt_path is not None:
                self._journal.finish_for_cache(
                    attempt_path, cache_path, status="failed", error=str(exc)
                )
            raise
        if attempt_path is not None:
            self._journal.finish_for_cache(attempt_path, cache_path, status="accepted")
        return panel

    def decision_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.belief_panel(scenario, period_state, household_states)


def seed_live_cache(
    seed_dir: Path,
    cache_dir: Path,
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    source_dir = seed_dir / provider
    if not source_dir.is_dir():
        raise DynamicMacroError(
            f"Seed cache lacks provider directory: {source_dir}"
        )
    destination = cache_dir / provider
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in sorted(source_dir.glob("*.json")):
        if source.is_symlink() or not source.is_file():
            raise DynamicMacroError("Seed cache entries must be regular JSON files")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DynamicMacroError(f"Invalid seed cache JSON: {source}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("provider") != provider
            or payload.get("model") != model
            or not isinstance(payload.get("payload"), dict)
        ):
            raise DynamicMacroError(f"Seed cache identity mismatch: {source}")
        target = destination / source.name
        if target.exists():
            raise DynamicMacroError(f"Seed cache destination collision: {target}")
        shutil.copy2(source, target)
        rows.append(
            {
                "file": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    if not rows:
        raise DynamicMacroError("Seed cache contains no JSON records")
    return {
        "source_dir_name": seed_dir.name,
        "provider": provider,
        "model": model,
        "record_count": len(rows),
        "records": rows,
    }


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
        journal_dir: Path | None = None,
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
        if max_live_calls > 0:
            self._live: Any = JournaledLiveDemandClient(
                DemandEconomyClient(
                    provider,
                    model,
                    cache_dir,
                    mode="live",
                    variant=self.variant,
                    max_live_calls=max_live_calls,
                    execution_cwd=execution_cwd,
                ),
                LiveAttemptJournal(
                    journal_dir or cache_dir / "live_attempts",
                    provider=provider,
                    model=model,
                ),
            )
        else:
            self._live = _DisabledLiveDemandClient(provider, model, cache_dir)
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
            cache_path = self._live.belief_cache_path(
                scenario, period_state, household_states
            )
            try:
                panel = self._live.belief_panel(
                    scenario, period_state, household_states
                )
                return panel
            except LLMUnavailable as exc:
                if attempt >= self.semantic_retry_limit:
                    raise
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


class _DisabledLiveDemandClient:
    """Fail closed if a full-prefix zero-call replay unexpectedly needs a provider."""

    variant = "llm_belief"

    def __init__(self, provider: str, model: str, cache_dir: Path) -> None:
        self.provider = provider
        self.model = model
        self.cache_dir = cache_dir
        self.live_call_count = 0
        self.cache_hit_count = 0
        self.raw_records: list[dict[str, Any]] = []

    def belief_cache_path(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> Path:
        del scenario, period_state, household_states
        return self.cache_dir / "disabled_live_call.json"

    def belief_panel(
        self,
        scenario: DemandScenario,
        period_state: dict[str, Any],
        household_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del scenario, period_state, household_states
        raise LLMUnavailable("Live calls are disabled for this full-prefix replay")


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
