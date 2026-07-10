"""Focused batch-contract coverage for the recursive dynamic macro client."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from macro_llm_tournament.demand_economy import (
    DemandEconomyClient,
    build_fixture_demand_households,
    default_demand_scenarios,
    run_demand_economy,
)
from macro_llm_tournament.dynamic_macro_clients import (
    GainAdjustedDemandClient,
    LiveAttemptJournal,
    canonical_live_attempts,
    validate_live_attempt_ledger,
)
from macro_llm_tournament.dynamic_macro_common import DynamicMacroError
from macro_llm_tournament.dynamic_macro_economy import economy_measure
from macro_llm_tournament.llm_common import LLMUnavailable


def _households(count: int) -> pd.DataFrame:
    base = build_fixture_demand_households()
    rows = []
    for index in range(count):
        row = base.iloc[index % len(base)].copy()
        row["type_id"] = f"household_{index:03d}"
        row["label"] = f"Household {index:03d}"
        row["population_weight"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _run(households: pd.DataFrame, *, max_households_per_call: int):
    with TemporaryDirectory() as temp_dir:
        client = DemandEconomyClient(
            "codex_cli",
            "gpt-5.5",
            Path(temp_dir),
            mode="fixture",
            max_households_per_call=max_households_per_call,
        )
        result = run_demand_economy(
            households,
            [default_demand_scenarios()[0]],
            client,
            period_count=1,
        )
        return result, client


def test_81_households_use_one_deterministic_batch_without_sample_weights() -> None:
    (_initial, beliefs, _decisions, _periods, _accounting, prompts), client = _run(
        _households(81), max_households_per_call=100
    )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt["batch_index"] == 0
    assert prompt["batch_count"] == 1
    assert prompt["household_type_ids"] == sorted(prompt["household_type_ids"])
    serialized = json.dumps(prompt["prompt_payload"], sort_keys=True)
    assert "population_weight" not in serialized
    assert set(beliefs["type_id"]) == set(prompt["household_type_ids"])
    assert len(client.last_call_records) == 1


def test_200_households_split_into_two_exact_batches_and_merge_coverage() -> None:
    (_initial, beliefs, _decisions, _periods, _accounting, prompts), client = _run(
        _households(200), max_households_per_call=100
    )

    assert [row["batch_index"] for row in prompts] == [0, 1]
    assert [len(row["household_type_ids"]) for row in prompts] == [100, 100]
    assert {row["batch_count"] for row in prompts} == {2}
    assert all(len(row["household_type_ids_sha256"]) == 64 for row in prompts)
    assert all(len(row["prompt_payload_sha256"]) == 64 for row in prompts)
    assert len(set(beliefs["type_id"])) == 200
    assert len(client.last_call_records) == 2
    assert {row["batch_index"] for row in client.last_call_records} == {0, 1}


def test_incomplete_multi_batch_replay_fails_before_accepting_a_period() -> None:
    households = _households(200)
    _result, fixture_client = _run(households, max_households_per_call=100)
    with TemporaryDirectory() as temp_dir:
        replay_client = DemandEconomyClient(
            "codex_cli",
            "gpt-5.5",
            Path(temp_dir),
            mode="raw_replay",
            raw_replay_records=[fixture_client.raw_records[0]],
            max_households_per_call=100,
        )
        with pytest.raises(LLMUnavailable, match="batch=1"):
            run_demand_economy(
                households,
                [default_demand_scenarios()[0]],
                replay_client,
                period_count=1,
            )
    assert replay_client.raw_records == []


def test_shuffled_inputs_produce_identical_batch_layout_and_prompt_hashes() -> None:
    first, _ = _run(_households(200), max_households_per_call=100)
    second, _ = _run(
        _households(200).sample(frac=1.0, random_state=7).reset_index(drop=True),
        max_households_per_call=100,
    )
    first_prompts = first[-1]
    second_prompts = second[-1]
    assert [row["household_type_ids"] for row in first_prompts] == [
        row["household_type_ids"] for row in second_prompts
    ]
    assert [row["prompt_payload_sha256"] for row in first_prompts] == [
        row["prompt_payload_sha256"] for row in second_prompts
    ]


def test_gain_wrapper_preserves_actual_batch_prompt_cards() -> None:
    with TemporaryDirectory() as temp_dir:
        base = DemandEconomyClient(
            "codex_cli",
            "gpt-5.5",
            Path(temp_dir),
            mode="fixture",
            max_households_per_call=100,
        )
        client = GainAdjustedDemandClient(
            base,
            gains={
                "global": 1.0,
                "inflation": 1.0,
                "income": 1.0,
                "unemployment": 1.0,
                "confidence": 1.0,
            },
            requested_mode="fixture",
            replay_records=None,
        )
        *_frames, prompts = run_demand_economy(
            _households(200),
            [default_demand_scenarios()[0]],
            client,
            period_count=1,
            max_households_per_call=100,
        )
    assert [row["batch_index"] for row in prompts] == [0, 1]


def test_one_batch_legacy_raw_replay_remains_compatible_and_final_state_is_exposed() -> None:
    households = _households(4)
    (_result, fixture_client) = _run(households, max_households_per_call=100)
    legacy = dict(fixture_client.raw_records[0])
    for key in (
        "batch_index",
        "batch_count",
        "household_type_ids",
        "household_type_ids_sha256",
        "prompt_payload_sha256",
        "provider_called",
        "response_sha256",
    ):
        legacy.pop(key, None)
    with TemporaryDirectory() as temp_dir:
        replay_client = DemandEconomyClient(
            "codex_cli",
            "gpt-5.5",
            Path(temp_dir),
            mode="raw_replay",
            raw_replay_records=[legacy],
            max_households_per_call=100,
        )
        _initial, beliefs, _decisions, _periods, _accounting, prompts = run_demand_economy(
            households,
            [default_demand_scenarios()[0]],
            replay_client,
            period_count=1,
        )
    assert len(prompts) == 1
    assert len(beliefs) == 4
    final_states = replay_client.final_household_states["baseline"]
    assert len(final_states) == 4
    assert {"static_income_group", "static_liquidity_group", "static_job_loss_risk_type"}.issubset(final_states[0])
    assert all(
        row["income_group"] == row["static_income_group"] for row in final_states
    )
    assert all(row["employment_status"] == "employed" for row in final_states)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_raw_replay_rejects_inexact_batch_household_coverage(mutation: str) -> None:
    households = _households(4)
    _, fixture_client = _run(households, max_households_per_call=100)
    record = json.loads(json.dumps(fixture_client.raw_records[0]))
    beliefs = record["payload"]["beliefs"]
    if mutation == "missing":
        beliefs.pop()
    elif mutation == "duplicate":
        beliefs[-1] = dict(beliefs[0])
    else:
        extra = dict(beliefs[0])
        extra["type_id"] = "not_supplied"
        beliefs.append(extra)
    with TemporaryDirectory() as temp_dir:
        replay_client = DemandEconomyClient(
            "codex_cli",
            "gpt-5.5",
            Path(temp_dir),
            mode="raw_replay",
            raw_replay_records=[record],
            max_households_per_call=100,
        )
        with pytest.raises(LLMUnavailable):
            run_demand_economy(
                households,
                [default_demand_scenarios()[0]],
                replay_client,
                period_count=1,
            )


def test_weighted_dollar_aggregation_and_ratio_of_totals() -> None:
    (result, _client) = _run(_households(4), max_households_per_call=100)
    _initial, _beliefs, decisions, periods, _accounting, _prompts = result
    weights = decisions["population_weight"].astype(float)
    expected_income = float((weights * decisions["labor_income"]).sum())
    expected_saving = float((weights * decisions["saving_flow"]).sum())
    expected_assets = float((weights * decisions["liquid_assets_after"]).sum())
    expected_consumption = float((weights * decisions["consumption"]).sum())
    row = periods.iloc[0]
    assert row["aggregate_income"] == pytest.approx(expected_income)
    assert row["aggregate_saving"] == pytest.approx(expected_saving)
    assert row["aggregate_assets_to_consumption_ratio"] == pytest.approx(
        expected_assets / expected_consumption
    )
    assert economy_measure(row, "saving_rate_pct") == pytest.approx(
        100.0 * expected_saving / expected_income
    )


def test_v2_ledger_records_exact_batch_identity_and_hashes() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        cache_path = root / "cache.json"
        cache_path.write_text('{"payload": {}}', encoding="utf-8")
        journal = LiveAttemptJournal(root / "live_attempts", provider="codex_cli", model="gpt-5.5")
        path = journal.start(
            {"period_index": 2},
            cache_path,
            batch_index=1,
            batch_count=2,
            household_type_ids=["household_100", "household_101"],
            prompt_payload_sha256="a" * 64,
        )
        journal.finish_for_cache(path, cache_path, status="accepted", response={"ok": True})
        rows = canonical_live_attempts(root / "live_attempts")
    assert rows[0]["schema_version"] == "dynamic_macro_live_attempt_v2"
    assert rows[0]["batch_index"] == 1
    assert rows[0]["household_type_ids"] == ["household_100", "household_101"]
    assert rows[0]["provider_called"] is True
    assert len(rows[0]["response_sha256"]) == 64


def test_legacy_v1_ledger_rows_remain_readable() -> None:
    legacy = {
        "schema_version": "dynamic_macro_live_attempt_v1",
        "attempt_number": 1,
        "attempt_id": "live_attempt_0001",
        "provider": "codex_cli",
        "model": "gpt-5.5",
        "period_index": 0,
        "status": "failed",
        "started_at_utc": "2026-07-09T00:00:00+00:00",
        "finished_at_utc": "2026-07-09T00:00:01+00:00",
        "cache_file": "legacy.json",
        "cache_file_sha256": None,
        "error_sha256": "b" * 64,
        "journal_sha256": "c" * 64,
    }
    assert validate_live_attempt_ledger([legacy]) == [legacy]
    with pytest.raises(DynamicMacroError):
        validate_live_attempt_ledger([{**legacy, "unexpected": True}])
