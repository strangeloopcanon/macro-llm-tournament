"""Deterministically attach SCF 2022 financial states to persistent SCE households.

The SCE cohort supplies identity, beliefs, and sampling weights.  The SCF only
supplies a matched baseline financial state; it is never used to redraw or
rename households.  SCF public data contain five implicates per ``yy1`` case,
so records are collapsed before any matching takes place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .ecology_models import (
    DEFAULT_CREDIT_MINIMUM_PAYMENT_FLOOR_USD,
    DEFAULT_CREDIT_MINIMUM_PAYMENT_RATE_PCT,
)

SCHEMA_VERSION = "ecology_scf_financial_states_v5"
# Preserve the v4 donor draw so this schema change alters payment semantics only.
DONOR_SELECTION_VERSION = "ecology_scf_financial_states_v4"
DEFAULT_SEED = 20260714
INCOME_CALIPER_MIN_RATIO = 0.5
INCOME_CALIPER_MAX_RATIO = 2.0
MAX_BASELINE_SPENDING_SHARE = 0.05
MIN_BASELINE_SPENDING_EFFECTIVE_HOUSEHOLDS = 100.0
SCF_MEMBER_NAME = "rscfp2022.dta"
SCF_REQUIRED_COLUMNS = (
    "y1",
    "yy1",
    "wgt",
    "age",
    "edcl",
    "lf",
    "own",
    "income",
    "wageinc",
    "bussefarminc",
    "ssretinc",
    "transfothinc",
    "foodhome",
    "foodaway",
    "fooddelv",
    "rent",
    "mortpay",
    "liq",
    "ccbal",
    "noccbal",
    "revpay",
    "saved",
)


class FinancialStateError(ValueError):
    """Raised when the SCE or SCF input cannot form a reproducible state."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _number(value: object, *, floor: float | None = None) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        return 0.0
    result = float(number)
    return max(floor, result) if floor is not None else result


def _mode(values: pd.Series) -> object:
    """Return a stable mode, including an explicit tie break."""

    clean = values.dropna()
    if clean.empty:
        return np.nan
    counts = clean.value_counts(dropna=True)
    winners = sorted(counts[counts.eq(counts.max())].index.tolist(), key=str)
    return winners[0]


def _income_group(value: object) -> str:
    income = _number(value)
    if income < 45_000:
        return "low"
    if income < 140_000:
        return "middle"
    return "high"


def _age_group(value: object) -> str:
    age = _number(value)
    if age < 35:
        return "young"
    if age < 55:
        return "prime"
    return "older"


def _target_age_group(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "unknown", "none"}:
        return None
    if any(token in text for token in ("young", "18_34", "18-34", "under_35", "under 35")):
        return "young"
    if any(token in text for token in ("prime", "35_54", "35-54", "35 to 54")):
        return "prime"
    if any(token in text for token in ("older", "55_plus", "55+", "55-", "55 to")):
        return "older"
    return None


def _education_group(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "unknown", "none"}:
        return None
    if any(token in text for token in ("college", "bachelor", "graduate", "postgrad")):
        return "college_plus"
    if "some" in text or "associate" in text:
        return "some_college"
    if any(token in text for token in ("high", "hs", "less", "secondary")):
        return "hs_or_less"
    return None


def _scf_education_group(value: object) -> str:
    edcl = _number(value)
    if edcl >= 4:
        return "college_plus"
    if edcl >= 3:
        return "some_college"
    return "hs_or_less"


def _employment_group(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "unknown", "none"}:
        return None
    if any(token in text for token in ("unemployed", "not_employed", "retired", "not employed")):
        return "not_employed"
    if "employ" in text or "work" in text:
        return "employed"
    return None


def _homeownership(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "unknown", "none"}:
        return None
    if text in {"1", "true", "yes", "owner", "own", "homeowner"}:
        return "owner"
    if text in {"0", "false", "no", "renter", "rent"}:
        return "renter"
    return None


def _first_present(row: pd.Series, columns: Iterable[str]) -> object:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return None


def _household_id_column(frame: pd.DataFrame) -> str:
    for column in ("household_id", "type_id"):
        if column in frame.columns:
            values = frame[column].astype(str).str.strip()
            if values.ne("").all() and values.ne("nan").all() and values.is_unique:
                return column
    raise FinancialStateError("households must have unique non-blank household_id or type_id")


def _target_match_fields(row: pd.Series) -> dict[str, str]:
    income = _first_present(row, ("income_group",))
    income_text = str(income).strip().lower()
    fields: dict[str, str] = {}
    if income_text in {"low", "middle", "high"}:
        fields["income_group"] = income_text
    age = _target_age_group(_first_present(row, ("age_bucket", "age_group", "age")))
    if age:
        fields["age_group"] = age
    education = _education_group(_first_present(row, ("education", "education_group", "education_bucket")))
    if education is None and "cohort_stratum" in row.index:
        education = _education_group(str(row["cohort_stratum"]).split("|")[-1])
    if education:
        fields["education_group"] = education
    employment = _employment_group(_first_present(row, ("employment_status", "employment")))
    if employment:
        fields["employment_group"] = employment
    ownership = _homeownership(_first_present(row, ("homeownership", "homeowner", "owns_home", "own")))
    if ownership:
        fields["homeownership_group"] = ownership
    return fields


def _target_annual_income(row: pd.Series) -> float:
    value = _first_present(row, ("annual_income", "annual_income_usd"))
    income = _number(value)
    if income <= 0.0:
        raise FinancialStateError(
            "each household must have a positive annual-income anchor"
        )
    return income


def collapse_scf_implicates(scf: pd.DataFrame) -> pd.DataFrame:
    """Select one coherent canonical implicate per SCF household.

    Averaging the five implicates would create financial vectors that no sampled
    household reported. The lowest ``y1`` implicate is a deterministic, documented
    choice; survey weights remain attached to that same vector.
    """

    missing = sorted(set(SCF_REQUIRED_COLUMNS) - set(scf.columns))
    if missing:
        raise FinancialStateError(f"SCF input is missing required columns: {', '.join(missing)}")
    source = scf.loc[:, list(SCF_REQUIRED_COLUMNS)].copy()
    source["y1"] = pd.to_numeric(source["y1"], errors="coerce")
    source["yy1"] = pd.to_numeric(source["yy1"], errors="coerce")
    if source["yy1"].isna().any() or source["yy1"].duplicated().all():
        raise FinancialStateError("SCF yy1 must identify donor records")
    for column in SCF_REQUIRED_COLUMNS[2:]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    grouped = source.groupby("yy1", sort=True, dropna=False)
    counts = grouped.size()
    donors = (
        source.sort_values(["yy1", "y1"], kind="stable")
        .groupby("yy1", sort=True, as_index=False)
        .first()
    )
    donors["scf_implicate_count"] = donors["yy1"].map(counts).astype(int)
    if (donors["scf_implicate_count"] != 5).any():
        bad = donors.loc[donors["scf_implicate_count"].ne(5), "yy1"].head(5).tolist()
        raise FinancialStateError(f"each SCF yy1 must have five implicates; invalid yy1 values: {bad}")
    if donors["wgt"].isna().any() or (donors["wgt"] <= 0).any():
        raise FinancialStateError("collapsed SCF donor weights must be positive")
    donors["income_group"] = donors["income"].map(_income_group)
    donors["age_group"] = donors["age"].map(_age_group)
    donors["education_group"] = donors["edcl"].map(_scf_education_group)
    donors["employment_group"] = np.where(donors["lf"].eq(1), "employed", "not_employed")
    donors["homeownership_group"] = np.where(donors["own"].eq(1), "owner", "renter")
    return donors.sort_values("yy1", kind="stable").reset_index(drop=True)


def read_scf_2022_donors(scf_zip: Path | str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read ``rscfp2022.dta`` directly from its official ZIP and collapse it."""

    archive_path = Path(scf_zip)
    if not archive_path.is_file():
        raise FinancialStateError(f"SCF ZIP does not exist: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name.lower() == SCF_MEMBER_NAME]
        if len(members) != 1:
            raise FinancialStateError(f"SCF ZIP must contain exactly one {SCF_MEMBER_NAME}")
        member = members[0]
        with archive.open(member) as handle:
            scf = pd.read_stata(handle, columns=list(SCF_REQUIRED_COLUMNS))
        member_sha256 = hashlib.sha256(archive.read(member)).hexdigest()
    return collapse_scf_implicates(scf), {
        "scf_zip_path": str(archive_path),
        "scf_zip_sha256": _sha256(archive_path),
        "scf_member": member,
        "scf_member_sha256": member_sha256,
    }


def _select_donor(
    donors: pd.DataFrame,
    match_fields: dict[str, str],
    *,
    household_id: str,
    target_annual_income: float,
    seed: int,
) -> tuple[pd.Series, tuple[str, ...], int]:
    active = dict(match_fields)
    relaxed: list[str] = []
    # Employment determines whether a household receives labor income in the
    # ecology, so it is the last field we permit the donor match to relax.
    relax_order = (
        "homeownership_group",
        "education_group",
        "age_group",
        "income_group",
        "employment_group",
    )
    while True:
        candidates = donors
        for column, value in active.items():
            candidates = candidates.loc[candidates[column].eq(value)]
        donor_income = pd.to_numeric(candidates["income"], errors="coerce")
        income_ratio = donor_income / target_annual_income
        candidates = candidates.loc[
            donor_income.gt(0.0)
            & income_ratio.ge(INCOME_CALIPER_MIN_RATIO)
            & income_ratio.le(INCOME_CALIPER_MAX_RATIO)
        ]
        if not candidates.empty:
            break
        next_field = next((field for field in relax_order if field in active), None)
        if next_field is None:
            raise FinancialStateError(
                f"no SCF donor satisfies the income caliper for {household_id}"
            )
        active.pop(next_field)
        relaxed.append(next_field)
    candidates = candidates.sort_values("yy1", kind="stable").reset_index(drop=True)
    profile = "|".join(f"{field}={value}" for field, value in sorted(active.items()))
    token = f"{DONOR_SELECTION_VERSION}|{int(seed)}|{household_id}|{profile}"
    draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) / 2**64
    weights = candidates["wgt"].to_numpy(dtype=float)
    index = int(np.searchsorted(np.cumsum(weights) / weights.sum(), draw, side="right"))
    index = min(index, len(candidates) - 1)
    return candidates.iloc[index], tuple(relaxed), len(candidates)


def _financial_fields(
    donor: pd.Series, *, target_annual_income: float
) -> dict[str, float]:
    donor_annual_income = max(0.0, _number(donor["income"]))
    if donor_annual_income <= 0.0:
        raise FinancialStateError("selected SCF donor income must be positive")
    financial_state_scale = target_annual_income / donor_annual_income
    annual_income = target_annual_income
    monthly_income = annual_income / 12.0
    raw_wage = max(
        0.0, _number(donor["wageinc"]) * financial_state_scale / 12.0
    )
    raw_business = max(
        0.0, _number(donor["bussefarminc"]) * financial_state_scale / 12.0
    )
    raw_transfers = max(
        0.0,
        (_number(donor["ssretinc"]) + _number(donor["transfothinc"]))
        * financial_state_scale
        / 12.0,
    )
    positive_components = raw_wage + raw_business + raw_transfers
    component_scale = (
        min(1.0, monthly_income / positive_components)
        if positive_components > 0.0
        else 1.0
    )
    monthly_wage = raw_wage * component_scale
    monthly_business_income = raw_business * component_scale
    transfers = raw_transfers * component_scale
    monthly_earned_income = monthly_wage + monthly_business_income
    monthly_nonwage = max(0.0, monthly_income - monthly_earned_income - transfers)
    monthly_food = max(
        0.0,
        sum(
            _number(donor[column])
            for column in ("foodhome", "foodaway", "fooddelv")
        )
        * financial_state_scale
        / 12.0,
    )
    # SCF rent is reported monthly, while the food variables are annualized.
    committed = (
        max(0.0, _number(donor["rent"])) * financial_state_scale
        + max(0.0, _number(donor["mortpay"])) * financial_state_scale
        + monthly_food
    )
    consumption_propensity_proxy = 0.80 if _number(donor["saved"]) > 0.0 else 0.97
    total_consumption_proxy = max(committed, monthly_income * consumption_propensity_proxy)
    discretionary = max(0.0, total_consumption_proxy - committed)
    debt = max(0.0, _number(donor["ccbal"])) * financial_state_scale
    deposits = max(0.0, _number(donor["liq"])) * financial_state_scale
    # Public SCF summary data contain balances but not a credit-limit total.
    # This conservative capacity proxy is donor-conditioned and never uses SCE
    # outcomes: 25% of income, at least $1,000 for a card borrower, and never
    # below the observed balance.
    has_revolving_access = debt > 0.0 or _number(donor["noccbal"]) > 0.0
    credit_limit = (
        max(debt, 1_000.0, annual_income * 0.25)
        if has_revolving_access
        else 0.0
    )
    utilization = 0.0 if credit_limit == 0.0 else debt / credit_limit
    observed_revolving_payment = (
        max(0.0, _number(donor["revpay"])) * financial_state_scale
    )
    # SCF REVPAY is an observed composite of card, BNPL, and revolving-line
    # payments. It is not the contractual minimum on the card balance stored
    # here, so the executable state uses the credit engine's actual contract.
    minimum_payment = (
        min(
            debt,
            max(
                DEFAULT_CREDIT_MINIMUM_PAYMENT_FLOOR_USD,
                debt * DEFAULT_CREDIT_MINIMUM_PAYMENT_RATE_PCT / 100.0,
            ),
        )
        if debt > 0.0
        else 0.0
    )
    return {
        "annual_income_usd": annual_income,
        "scf_donor_annual_income_usd": donor_annual_income,
        "scf_income_to_anchor_ratio": donor_annual_income / annual_income,
        "financial_state_scale": financial_state_scale,
        "monthly_wage_income_usd": monthly_wage,
        "monthly_business_income_usd": monthly_business_income,
        "monthly_earned_income_usd": monthly_earned_income,
        "monthly_nonwage_income_usd": monthly_nonwage,
        "monthly_transfers_benefits_usd": transfers,
        "income_component_reconciliation_scale": component_scale,
        "income_components_reconcile_to_annual_income": True,
        "baseline_committed_consumption_monthly_usd": committed,
        "baseline_discretionary_consumption_monthly_usd": discretionary,
        "liquid_deposits_usd": deposits,
        "revolving_debt_usd": debt,
        "revolving_credit_limit_usd": credit_limit,
        "revolving_credit_utilization": utilization,
        "scf_observed_revolving_payment_monthly_usd": observed_revolving_payment,
        "recurring_minimum_debt_payment_usd": minimum_payment,
    }


def financial_state_population_audit(
    states: pd.DataFrame, *, enforce: bool = True
) -> dict[str, float | int]:
    required = {
        "population_weight",
        "baseline_committed_consumption_monthly_usd",
        "baseline_discretionary_consumption_monthly_usd",
    }
    missing = sorted(required - set(states.columns))
    if missing:
        raise FinancialStateError(
            f"financial-state audit is missing columns: {', '.join(missing)}"
        )
    weights = pd.to_numeric(states["population_weight"], errors="coerce")
    baseline = (
        pd.to_numeric(
            states["baseline_committed_consumption_monthly_usd"],
            errors="coerce",
        )
        + pd.to_numeric(
            states["baseline_discretionary_consumption_monthly_usd"],
            errors="coerce",
        )
    )
    contributions = weights * baseline
    total = float(contributions.sum())
    if (
        weights.isna().any()
        or baseline.isna().any()
        or (weights <= 0.0).any()
        or (baseline < 0.0).any()
        or total <= 0.0
    ):
        raise FinancialStateError(
            "financial-state population audit requires positive finite weights and spending"
        )
    shares = contributions / total
    audit: dict[str, float | int] = {
        "household_count": int(len(states)),
        "max_baseline_spending_contribution_share": float(shares.max()),
        "top_five_baseline_spending_contribution_share": float(
            shares.nlargest(min(5, len(shares))).sum()
        ),
        "baseline_spending_effective_households": float(
            1.0 / float(np.square(shares).sum())
        ),
    }
    if enforce and len(states) >= 100:
        if audit["max_baseline_spending_contribution_share"] > MAX_BASELINE_SPENDING_SHARE:
            raise FinancialStateError(
                "one household contributes more than 5% of baseline spending"
            )
        if (
            audit["baseline_spending_effective_households"]
            < MIN_BASELINE_SPENDING_EFFECTIVE_HOUSEHOLDS
        ):
            raise FinancialStateError(
                "baseline spending effective household count is below 100"
            )
    return audit


def generate_financial_states(
    households: pd.DataFrame,
    scf: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    scf_provenance: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return persistent households with deterministic SCF-conditioned state fields."""

    if households.empty:
        raise FinancialStateError("households input must not be empty")
    id_column = _household_id_column(households)
    donors = (
        scf.copy().sort_values("yy1", kind="stable").reset_index(drop=True)
        if "scf_implicate_count" in scf.columns
        else collapse_scf_implicates(scf)
    )
    prepared = households.copy()
    prepared["household_id"] = prepared[id_column].astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for _, row in prepared.sort_values("household_id", kind="stable").iterrows():
        household_id = str(row["household_id"])
        match_fields = _target_match_fields(row)
        target_annual_income = _target_annual_income(row)
        donor, relaxed, candidate_count = _select_donor(
            donors,
            match_fields,
            household_id=household_id,
            target_annual_income=target_annual_income,
            seed=seed,
        )
        record = row.to_dict()
        record.update(
            _financial_fields(donor, target_annual_income=target_annual_income)
        )
        record.update(
            {
                "household_id": household_id,
                "financial_state_schema_version": SCHEMA_VERSION,
                "financial_state_seed": int(seed),
                "scf_donor_id_sha256": hashlib.sha256(
                    f"scf2022|{int(donor['yy1'])}".encode("utf-8")
                ).hexdigest(),
                "scf_donor_weight": float(donor["wgt"]),
                "scf_donor_implicate_count": int(donor["scf_implicate_count"]),
                "scf_donor_income_group": str(donor["income_group"]),
                "scf_donor_age_group": str(donor["age_group"]),
                "scf_donor_employment_group": str(donor["employment_group"]),
                "scf_match_fields": ",".join(sorted(match_fields)),
                "scf_match_relaxed_fields": ",".join(relaxed),
                "scf_match_rule": "exact" if not relaxed else "relaxed_" + "_then_".join(relaxed),
                "scf_income_caliper_candidate_count": int(candidate_count),
                "financial_state_source": "SCF_2022_public_summary_canonical_first_implicate",
                "consumption_baseline_measurement": "SCF_food_housing_plus_saved_status_calibrated_proxy",
            }
        )
        if scf_provenance:
            record["scf_source_member_sha256"] = scf_provenance.get("scf_member_sha256", "")
        rows.append(record)
    result = (
        pd.DataFrame(rows)
        .sort_values("household_id", kind="stable")
        .reset_index(drop=True)
    )
    financial_state_population_audit(result)
    return result


def write_financial_states(
    households_csv: Path | str,
    scf_zip: Path | str,
    output_csv: Path | str,
    manifest_path: Path | str | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Read inputs, write the state CSV plus an explicit reproducibility manifest."""

    households_path = Path(households_csv)
    output_path = Path(output_csv)
    if not households_path.is_file():
        raise FinancialStateError(f"households CSV does not exist: {households_path}")
    donors, provenance = read_scf_2022_donors(scf_zip)
    states = generate_financial_states(
        pd.read_csv(households_path), donors, seed=seed, scf_provenance=provenance
    )
    population_audit = financial_state_population_audit(states)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(output_path, index=False, lineterminator="\n")
    manifest_file = Path(manifest_path) if manifest_path else output_path.with_suffix(".manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "households_input": {"path": str(households_path), "sha256": _sha256(households_path)},
        "scf_input": provenance,
        "output_csv": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "row_count": int(len(states)),
            "columns": states.columns.tolist(),
        },
        "matching": {
            "base_fields": ["income_group", "age_group", "education_group", "employment_group", "homeownership_group"],
            "relaxation_order": [
                "homeownership_group",
                "education_group",
                "age_group",
                "income_group",
                "employment_group",
            ],
            "implicate_rule": "lowest y1 implicate per yy1; no cross-implicate averaging",
            "selection": "stable household-id hash mapped through sorted donor survey-weight CDF",
            "selection_version": DONOR_SELECTION_VERSION,
            "income_caliper_ratio": [
                INCOME_CALIPER_MIN_RATIO,
                INCOME_CALIPER_MAX_RATIO,
            ],
            "financial_scaling": (
                "all SCF dollar flows and stocks scaled by "
                "SCE income anchor / SCF donor income"
            ),
            "target_or_actual_data_used": False,
            "revolving_payment_semantics": {
                "observed_field": "scf_observed_revolving_payment_monthly_usd",
                "observed_measure": "SCF_REVPAY_composite_card_BNPL_and_revolving_line_payment",
                "executable_field": "recurring_minimum_debt_payment_usd",
                "minimum_payment_rate_pct": DEFAULT_CREDIT_MINIMUM_PAYMENT_RATE_PCT,
                "minimum_payment_floor_usd": DEFAULT_CREDIT_MINIMUM_PAYMENT_FLOOR_USD,
            },
        },
        "population_contribution_audit": population_audit,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--households", type=Path, required=True, help="Persistent SCE household CSV")
    parser.add_argument(
        "--scf-zip",
        type=Path,
        default=Path("work/scf/2022/scfp2022s.zip"),
        help="Official SCF 2022 summary ZIP containing rscfp2022.dta",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = write_financial_states(
        args.households, args.scf_zip, args.output_csv, args.manifest, seed=args.seed
    )
    print(json.dumps({"output_csv": manifest["output_csv"], "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI entry point.
    main()
