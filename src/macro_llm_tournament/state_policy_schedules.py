from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agent_common import OUTPUT_ROOT, WORK_ROOT, bounded_float, cache_key, markdown_table
from .agent_llm import AgentLLMClient
from .demand_economy import build_fixture_demand_households, normalize_demand_households
from .llm_common import LLMUnavailable
from .phase4_matched_twins import load_persona_ecology_bundle


STATE_POLICY_VERSION = "demand_behavior_state_policy_schedule_v1"
STATE_POLICY_PROMPT_VERSION = "demand_state_policy_schedule_v1"
TRANSFER_RATIO_GRID = (0.05, 0.10, 0.30, 1.00, 3.00, 10.00)
JOB_LOSS_GAP_GRID = (0.0, 3.0, 6.0, 12.0, 20.0)
INFLATION_GAP_GRID = (0.0, 2.0, 5.0, 8.0)
CONFIDENCE_DROP_GRID = (0.0, 5.0, 15.0, 30.0)
REAL_RATE_GAP_GRID = (0.0, 1.0, 3.0, 5.0)

STATE_POLICY_PREAMBLE = """
Return only valid JSON. You are writing bounded household behavior policy functions for a
macroeconomic simulation. The households should behave like ordinary US households, not like
financial-advice examples. Do not emit one-off actions. Emit reusable schedules that deterministic
code can evaluate as household states and beliefs change. Use only the information supplied. Do not
browse, inspect files, run commands, or cite studies.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build state-conditioned household behavior policy schedules.")
    parser.add_argument("--provider", choices=["codex_cli", "cursor_cli"], default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=["fixture", "replay", "live"], default="fixture")
    parser.add_argument("--max-live-calls", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--household-source", choices=["fixture", "csv", "persona_ecology_replay"], default="fixture")
    parser.add_argument("--household-csv", default=None)
    parser.add_argument("--household-count", type=int, default=24)
    parser.add_argument("--persona-ecology-dir", default=None)
    parser.add_argument("--primary-ecology-source", default="")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "live" and args.max_live_calls <= 0:
        raise SystemExit("--max-live-calls must be positive when --mode live is used")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"state_policy_schedules_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else WORK_ROOT / "state_policy_schedule_cache"
    manifest: dict[str, Any] = {
        "schema_version": STATE_POLICY_VERSION,
        "prompt_version": STATE_POLICY_PROMPT_VERSION,
        "timestamp_utc": timestamp,
        "argv": _sanitized_argv(),
        "run_command": shlex.join(_sanitized_argv()),
        "provider": args.provider,
        "model": args.model,
        "mode": args.mode,
        "max_live_calls": int(args.max_live_calls),
        "household_source": args.household_source,
        "persona_ecology_dir": args.persona_ecology_dir,
        "primary_ecology_source": args.primary_ecology_source,
        "status": "running",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        households, input_manifest = load_policy_households(args)
        profile_rows = build_state_policy_profiles(households)
        client = StatePolicyClient(
            args.provider,
            args.model,
            cache_dir,
            mode=args.mode,
            max_live_calls=args.max_live_calls,
        )
        policies = client.state_policies(profile_rows)
        profile = build_state_policy_profile(
            profile_rows,
            policies,
            provider=args.provider,
            model=args.model,
            mode=args.mode,
            input_manifest=input_manifest,
            raw_records=client.raw_records,
        )
        households.to_csv(output_dir / "state_policy_households.csv", index=False)
        profile_rows.to_csv(output_dir / "state_policy_profiles.csv", index=False)
        (output_dir / "state_policy_raw_records.json").write_text(
            json.dumps(client.raw_records, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "state_behavior_policy_profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
        )
        report = build_state_policy_report(manifest, profile_rows, profile)
        (output_dir / "state_policy_schedules_report.md").write_text(report, encoding="utf-8")
        manifest.update(
            {
                "status": "ok",
                "household_count": int(households.shape[0]),
                "profile_count": int(profile_rows.shape[0]),
                "live_call_count": int(client.live_call_count),
                "cache_hit_count": int(client.cache_hit_count),
                "profile_schema_version": profile["schema_version"],
                "policy_profile_json": "state_behavior_policy_profile.json",
                "outputs": [
                    "state_policy_households.csv",
                    "state_policy_profiles.csv",
                    "state_policy_raw_records.json",
                    "state_behavior_policy_profile.json",
                    "state_policy_schedules_report.md",
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


def load_policy_households(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.household_source == "csv":
        if not args.household_csv:
            raise SystemExit("--household-csv is required when --household-source csv")
        path = Path(args.household_csv)
        if not path.exists():
            raise SystemExit(f"--household-csv does not exist: {path}")
        return normalize_demand_households(pd.read_csv(path)), {"source": "csv", "path": str(path)}
    if args.household_source == "persona_ecology_replay":
        if not args.persona_ecology_dir:
            raise SystemExit("--persona-ecology-dir is required when --household-source persona_ecology_replay")
        bundle = load_persona_ecology_bundle(args)
        return bundle["households"], {
            "source": "persona_ecology_replay",
            "root": str(bundle["root"]),
            "ecology_source": bundle["source"],
            "manifest_sha256": bundle["manifest_sha256"],
            "panel_sha256": bundle["panel_sha256"],
            "predictions_sha256": bundle["predictions_sha256"],
        }
    return build_fixture_demand_households(args.household_count), {"source": "fixture", "household_count": int(args.household_count)}


def build_state_policy_profiles(households: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["income_group", "liquidity_group", "job_loss_risk_type", "age_bucket"]
    rows: list[dict[str, Any]] = []
    for keys, group in households.groupby(group_cols, dropna=False):
        income_group, liquidity_group, job_loss_risk_type, age_bucket = (str(value) for value in keys)
        weight = group["population_weight"].astype(float).clip(lower=0.0)
        total_weight = float(weight.sum())
        if total_weight <= 0:
            weight = pd.Series(np.full(group.shape[0], 1.0 / max(group.shape[0], 1)), index=group.index)
            total_weight = 1.0

        def wav(column: str) -> float:
            values = pd.to_numeric(group[column], errors="coerce").astype(float)
            return float((values * weight).sum() / total_weight)

        baseline_consumption = wav("baseline_consumption_annual") / 4.0
        liquid_assets = wav("liquid_assets")
        annual_income = wav("annual_income")
        debt = wav("debt")
        profile_id = "_".join(
            _slug(part)
            for part in (
                str(income_group),
                str(liquidity_group),
                str(job_loss_risk_type),
                str(age_bucket),
            )
        )
        rows.append(
            {
                "profile_id": profile_id,
                "label": f"{income_group} income, {liquidity_group} liquid, {job_loss_risk_type} job risk, {age_bucket}",
                "income_group": income_group,
                "liquidity_group": liquidity_group,
                "job_loss_risk_type": job_loss_risk_type,
                "age_bucket": age_bucket,
                "n_households": int(group.shape[0]),
                "population_weight": total_weight,
                "annual_income": annual_income,
                "quarterly_labor_income": annual_income / 4.0,
                "baseline_consumption": baseline_consumption,
                "liquid_assets": liquid_assets,
                "liquid_buffer_months": liquid_assets / max(baseline_consumption / 3.0, 1.0),
                "debt": debt,
                "debt_to_income": debt / max(annual_income, 1.0),
                "base_mpc": wav("base_mpc"),
                "base_saving_rate": wav("base_saving_rate"),
                "baseline_job_loss_probability": wav("baseline_job_loss_probability"),
                "inflation_expectation_1y": wav("inflation_expectation_1y"),
                "income_growth_expectation_1y": wav("income_growth_expectation_1y"),
                "confidence_index": wav("confidence_index"),
                "precautionary_sensitivity": wav("precautionary_sensitivity"),
                "rate_sensitivity": wav("rate_sensitivity"),
            }
        )
    return pd.DataFrame(rows).sort_values("profile_id").reset_index(drop=True)


def state_policy_prompt(profile_rows: pd.DataFrame) -> str:
    profiles = [
        {
            "profile_id": str(row["profile_id"]),
            "label": str(row["label"]),
            "population_weight": round(float(row["population_weight"]), 5),
            "n_households": int(row["n_households"]),
            "annual_income_usd": round(float(row["annual_income"]), -2),
            "quarterly_labor_income_usd": round(float(row["quarterly_labor_income"]), -2),
            "baseline_quarterly_consumption_usd": round(float(row["baseline_consumption"]), -2),
            "liquid_assets_usd": round(float(row["liquid_assets"]), -2),
            "liquid_buffer_months": round(float(row["liquid_buffer_months"]), 2),
            "debt_to_income": round(float(row["debt_to_income"]), 3),
            "base_mpc": round(float(row["base_mpc"]), 3),
            "base_saving_rate": round(float(row["base_saving_rate"]), 3),
            "baseline_job_loss_probability_pct": round(float(row["baseline_job_loss_probability"]), 2),
            "baseline_inflation_expectation_pct": round(float(row["inflation_expectation_1y"]), 2),
            "baseline_real_income_growth_expectation_pct": round(float(row["income_growth_expectation_1y"]), 2),
            "baseline_confidence_index": round(float(row["confidence_index"]), 1),
        }
        for _, row in profile_rows.iterrows()
    ]
    payload = {
        "prompt_version": STATE_POLICY_PROMPT_VERSION,
        "task": (
            "For each household-state profile, write a bounded behavioral policy function. "
            "The simulator will evaluate these schedules using that household's current balance sheet and updated beliefs. "
            "Use realistic measured household behavior, including liquidity constraints, debt pressure, child/household needs, "
            "and underreaction. Do not emit advice or one-time actions."
        ),
        "profiles": profiles,
        "transfer_to_quarterly_income_ratio_grid": list(TRANSFER_RATIO_GRID),
        "belief_gap_grids": {
            "job_loss_probability_gap_pp": list(JOB_LOSS_GAP_GRID),
            "inflation_expectation_gap_pp": list(INFLATION_GAP_GRID),
            "confidence_drop_points": list(CONFIDENCE_DROP_GRID),
            "real_rate_gap_pp": list(REAL_RATE_GAP_GRID),
        },
        "required_response": {
            "policies": [
                {
                    "profile_id": "one supplied profile_id",
                    "transfer_schedule": [
                        {
                            "ratio": "one transfer_to_quarterly_income_ratio grid value",
                            "total_spending_share": "0 to 1 share of transfer spent within the quarter",
                            "debt_repayment_share": "0 to 1 share of transfer used for debt repayment",
                            "liquid_saving_share": "0 to 1 share of transfer kept liquid",
                        }
                    ],
                    "job_risk_schedule": [
                        {
                            "job_loss_probability_gap_pp": "one grid value",
                            "consumption_cut_share": "0 to 0.35 share of baseline quarterly consumption cut",
                        }
                    ],
                    "inflation_schedule": [
                        {
                            "inflation_expectation_gap_pp": "one grid value",
                            "consumption_cut_share": "0 to 0.25 share of baseline quarterly consumption cut",
                        }
                    ],
                    "confidence_schedule": [
                        {
                            "confidence_drop_points": "one grid value",
                            "consumption_cut_share": "0 to 0.25 share of baseline quarterly consumption cut",
                        }
                    ],
                    "real_rate_schedule": [
                        {
                            "real_rate_gap_pp": "one grid value",
                            "consumption_cut_share": "0 to 0.20 share of baseline quarterly consumption cut",
                        }
                    ],
                    "max_consumption_cut_share": "0 to 0.35 cap after overlapping stress channels",
                    "reason": "short reason",
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def fixture_state_policy_payload(profile_rows: pd.DataFrame) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    for _, row in profile_rows.iterrows():
        low_liquid = str(row["liquidity_group"]).lower() == "low"
        low_income = str(row["income_group"]).lower() == "low"
        high_risk = str(row["job_loss_risk_type"]).lower() == "high"
        base_mpc = float(row["base_mpc"])
        debt_to_income = float(row["debt_to_income"])
        buffer_months = float(row["liquid_buffer_months"])
        stress = float(np.clip((1.5 - buffer_months) / 1.5, 0.0, 1.0))
        transfer_schedule = []
        for ratio in TRANSFER_RATIO_GRID:
            spend = base_mpc / (1.0 + 0.22 * np.log1p(ratio))
            spend += 0.05 if low_income else -0.03
            spend += 0.04 if high_risk else 0.0
            spend = float(np.clip(spend, 0.05, 0.92))
            debt = float(np.clip((0.08 + 0.25 * stress + 0.16 * min(debt_to_income, 1.0)) * (1.0 - spend), 0.0, 0.55))
            liquid = max(0.0, 1.0 - spend - debt)
            transfer_schedule.append(
                {
                    "ratio": float(ratio),
                    "total_spending_share": round(spend, 4),
                    "debt_repayment_share": round(debt, 4),
                    "liquid_saving_share": round(liquid, 4),
                }
            )
        risk_scale = 0.004 + (0.004 if low_liquid else 0.0015) + (0.002 if high_risk else 0.0)
        policies.append(
            {
                "profile_id": str(row["profile_id"]),
                "transfer_schedule": transfer_schedule,
                "job_risk_schedule": [
                    {
                        "job_loss_probability_gap_pp": float(gap),
                        "consumption_cut_share": round(float(np.clip(gap * risk_scale, 0.0, 0.28)), 4),
                    }
                    for gap in JOB_LOSS_GAP_GRID
                ],
                "inflation_schedule": [
                    {
                        "inflation_expectation_gap_pp": float(gap),
                        "consumption_cut_share": round(float(np.clip(gap * (0.006 if low_income else 0.003), 0.0, 0.12)), 4),
                    }
                    for gap in INFLATION_GAP_GRID
                ],
                "confidence_schedule": [
                    {
                        "confidence_drop_points": float(drop),
                        "consumption_cut_share": round(float(np.clip(drop * (0.0025 if low_liquid else 0.0015), 0.0, 0.11)), 4),
                    }
                    for drop in CONFIDENCE_DROP_GRID
                ],
                "real_rate_schedule": [
                    {
                        "real_rate_gap_pp": float(gap),
                        "consumption_cut_share": round(float(np.clip(gap * (0.010 if not low_liquid else 0.004), 0.0, 0.10)), 4),
                    }
                    for gap in REAL_RATE_GAP_GRID
                ],
                "max_consumption_cut_share": 0.30 if low_liquid else 0.22,
                "reason": "fixture state policy",
            }
        )
    return {"policies": policies}


def normalize_state_policy_payload(profile_rows: pd.DataFrame, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = data.get("payload", data)
    policies = payload.get("policies")
    if not isinstance(policies, list):
        raise LLMUnavailable("State policy payload is missing policies list")
    expected = set(profile_rows["profile_id"].astype(str))
    by_profile: dict[str, dict[str, Any]] = {}
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        profile_id = str(policy.get("profile_id", ""))
        if profile_id not in expected:
            continue
        by_profile[profile_id] = {
            "transfer_schedule": _normalize_schedule(
                policy,
                "transfer_schedule",
                "ratio",
                TRANSFER_RATIO_GRID,
                {
                    "total_spending_share": (0.0, 1.0),
                    "debt_repayment_share": (0.0, 1.0),
                    "liquid_saving_share": (0.0, 1.0),
                },
            ),
            "job_risk_schedule": _normalize_schedule(
                policy,
                "job_risk_schedule",
                "job_loss_probability_gap_pp",
                JOB_LOSS_GAP_GRID,
                {"consumption_cut_share": (0.0, 0.35)},
            ),
            "inflation_schedule": _normalize_schedule(
                policy,
                "inflation_schedule",
                "inflation_expectation_gap_pp",
                INFLATION_GAP_GRID,
                {"consumption_cut_share": (0.0, 0.25)},
            ),
            "confidence_schedule": _normalize_schedule(
                policy,
                "confidence_schedule",
                "confidence_drop_points",
                CONFIDENCE_DROP_GRID,
                {"consumption_cut_share": (0.0, 0.25)},
            ),
            "real_rate_schedule": _normalize_schedule(
                policy,
                "real_rate_schedule",
                "real_rate_gap_pp",
                REAL_RATE_GAP_GRID,
                {"consumption_cut_share": (0.0, 0.20)},
            ),
            "max_consumption_cut_share": bounded_float(policy, "max_consumption_cut_share", 0.0, 0.35),
            "reason": str(policy.get("reason", ""))[:300],
        }
        _renormalize_transfer_schedule(by_profile[profile_id]["transfer_schedule"])
    missing = sorted(expected - set(by_profile))
    if missing:
        raise LLMUnavailable(f"State policy payload missing profile ids: {', '.join(missing)}")
    return by_profile


def _normalize_schedule(
    policy: dict[str, Any],
    name: str,
    x_field: str,
    grid: Iterable[float],
    fields: dict[str, tuple[float, float]],
) -> list[dict[str, float]]:
    raw = policy.get(name)
    if not isinstance(raw, list):
        raise LLMUnavailable(f"State policy is missing {name}")
    points: dict[float, dict[str, float]] = {}
    for point in raw:
        if not isinstance(point, dict):
            continue
        x = bounded_float(point, x_field, min(grid), max(grid))
        row = {x_field: float(x)}
        for field, bounds in fields.items():
            row[field] = bounded_float(point, field, bounds[0], bounds[1])
        points[float(x)] = row
    if not points:
        raise LLMUnavailable(f"State policy schedule {name} has no valid points")
    return [points[x] for x in sorted(points)]


def _renormalize_transfer_schedule(schedule: list[dict[str, float]]) -> None:
    for point in schedule:
        spend = float(point["total_spending_share"])
        debt = float(point["debt_repayment_share"])
        liquid = float(point["liquid_saving_share"])
        total = spend + debt + liquid
        if total > 1.0 and total > 0.0:
            point["total_spending_share"] = spend / total
            point["debt_repayment_share"] = debt / total
            point["liquid_saving_share"] = liquid / total
        else:
            point["liquid_saving_share"] = max(0.0, 1.0 - spend - debt)


class StatePolicyClient:
    def __init__(self, provider: str, model: str, cache_dir: Path, *, mode: str, max_live_calls: int):
        self.provider = provider
        self.model = model
        self.mode = mode
        self._client = AgentLLMClient(
            provider,
            model,
            cache_dir,
            mode=mode,
            max_live_calls=max_live_calls,
            system_preamble=STATE_POLICY_PREAMBLE,
        )
        self.raw_records: list[dict[str, Any]] = []

    @property
    def live_call_count(self) -> int:
        return self._client.live_call_count

    @property
    def cache_hit_count(self) -> int:
        return self._client.cache_hit_count

    def state_policies(self, profile_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if self.mode == "fixture":
            data = {
                "provider": self.provider,
                "model": self.model,
                "payload": fixture_state_policy_payload(profile_rows),
                "cache_hit": True,
                "cache_path": None,
            }
        else:
            prompt = state_policy_prompt(profile_rows)
            cache_name = f"state_policy_{cache_key({'provider': self.provider, 'model': self.model, 'prompt': prompt})}"
            data = self._client._codex_call(prompt, cache_name)
        normalized = normalize_state_policy_payload(profile_rows, data)
        self.raw_records.append(
            {
                "record_type": "state_policy_schedule",
                "prompt_version": STATE_POLICY_PROMPT_VERSION,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "cache_hit": bool(data.get("cache_hit", False)),
                "cache_path": data.get("cache_path"),
                "payload": data.get("payload", data),
            }
        )
        return normalized


def build_state_policy_profile(
    profile_rows: pd.DataFrame,
    policies: dict[str, dict[str, Any]],
    *,
    provider: str,
    model: str,
    mode: str,
    input_manifest: dict[str, Any],
    raw_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": STATE_POLICY_VERSION,
        "prompt_version": STATE_POLICY_PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "mode": mode,
        "source_label": f"state_policy_{provider}_{model}",
        "input_manifest": input_manifest,
        "profile_count": int(profile_rows.shape[0]),
        "profile_rows": profile_rows.to_dict(orient="records"),
        "state_policies": policies,
        "raw_record_sha256": {
            record.get("record_type", f"record_{index}"): _sha256_json(record.get("payload", record))
            for index, record in enumerate(raw_records)
        },
        "assignment_method": "nearest_state_profile_by_income_buffer_debt_job_risk_age",
        "execution_rule": (
            "The LLM-authored state schedule supplies transfer allocation and belief-stress consumption cuts; "
            "deterministic code interpolates schedules, caps overlapping channels, enforces budgets, and performs accounting."
        ),
    }


def build_state_policy_report(manifest: dict[str, Any], profile_rows: pd.DataFrame, profile: dict[str, Any]) -> str:
    preview = profile_rows[
        [
            "profile_id",
            "n_households",
            "population_weight",
            "annual_income",
            "liquid_buffer_months",
            "debt_to_income",
            "baseline_job_loss_probability",
        ]
    ].head(12)
    return "\n".join(
        [
            "# State-Conditioned Policy Schedules",
            "",
            f"Status: `{profile.get('schema_version')}`.",
            "",
            "These are reusable household behavior policies. They are not period-by-period consumption answers.",
            "The demand economy evaluates them against current household state and updated beliefs, then enforces accounting.",
            "",
            "## Profile Preview",
            "",
            markdown_table(preview),
            "",
            "## Manifest",
            "",
            "```json",
            json.dumps(
                {
                    "provider": manifest.get("provider"),
                    "model": manifest.get("model"),
                    "mode": manifest.get("mode"),
                    "household_source": manifest.get("household_source"),
                    "profile_count": profile.get("profile_count"),
                    "execution_rule": profile.get("execution_rule"),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "unknown"


def _sha256_json(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sanitized_argv() -> list[str]:
    import sys

    return [arg if "key" not in arg.lower() and "token" not in arg.lower() else "<redacted>" for arg in sys.argv]


if __name__ == "__main__":
    raise SystemExit(main())
