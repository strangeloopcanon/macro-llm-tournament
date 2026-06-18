from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import xlrd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "work" / "spf"
PHILLY_FED_BASE_URL = "https://www.philadelphiafed.org"
MISSING_SENTINEL = 42.0


@dataclass(frozen=True)
class SPFVariable:
    code: str
    name: str
    units: str
    page_url: str
    error_data_url: str


SPF_VARIABLES: dict[str, SPFVariable] = {
    "CPI": SPFVariable(
        code="CPI",
        name="CPI inflation rate",
        units="annualized percentage points",
        page_url="https://www.philadelphiafed.org/surveys-and-data/cpi-spf",
        error_data_url=(
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/data-files/CPI/"
            "Data_SPF_Error_Statistics_CPI_1_AIC.xls"
            "?sc_lang=en&hash=E1779D894F61DDC4C9D6C1BECC5ED860"
        ),
    ),
    "RGDP": SPFVariable(
        code="RGDP",
        name="real GDP growth",
        units="annualized percentage points",
        page_url="https://www.philadelphiafed.org/surveys-and-data/rgdp",
        error_data_url=(
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/data-files/RGDP/"
            "Data_SPF_Error_Statistics_RGDP_3_AIC.xls"
            "?sc_lang=en&hash=0EBA1EB4FD421168FE19C1BF1361A88D"
        ),
    ),
    "UNEMP": SPFVariable(
        code="UNEMP",
        name="civilian unemployment rate",
        units="percentage points",
        page_url="https://www.philadelphiafed.org/surveys-and-data/unemp",
        error_data_url=(
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/data-files/UNEMP/"
            "Data_SPF_Error_Statistics_UNEMP_1_AIC.xls"
            "?sc_lang=en&hash=0FECEA90D4FAF338D812F52CB46D506A"
        ),
    ),
    "TBILL": SPFVariable(
        code="TBILL",
        name="3-month Treasury bill rate",
        units="percentage points",
        page_url="https://www.philadelphiafed.org/surveys-and-data/tbill",
        error_data_url=(
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/data-files/TBILL/"
            "Data_SPF_Error_Statistics_TBILL_1_AIC.xls"
            "?sc_lang=en&hash=A7D7F3BB48C2D832144673E8DEFC5F04"
        ),
    ),
    "TBOND": SPFVariable(
        code="TBOND",
        name="10-year Treasury bond rate",
        units="percentage points",
        page_url="https://www.philadelphiafed.org/surveys-and-data/tbond",
        error_data_url=(
            "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
            "survey-of-professional-forecasters/data-files/TBOND/"
            "Data_SPF_Error_Statistics_TBOND_1_AIC.xls"
            "?sc_lang=en&hash=219CD142F3E29EDF29EE501B8353502D"
        ),
    ),
}


def parse_variable_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        raw = [part.strip().upper() for part in value.split(",")]
    else:
        raw = [str(part).strip().upper() for part in value]
    variables = [part for part in raw if part]
    unknown = [part for part in variables if part not in SPF_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown SPF variable(s): {', '.join(unknown)}")
    return variables


def quarter_index(origin: str) -> int:
    text = origin.strip().replace("Q", "")
    year, quarter = text.split(":")
    return int(year) * 4 + int(quarter)


def origin_from_index(index: int) -> str:
    year = index // 4
    quarter = index - year * 4
    if quarter == 0:
        year -= 1
        quarter = 4
    return f"{year}:Q{quarter}"


def normalize_origin(raw: Any) -> str:
    text = str(raw).strip().replace(" ", "")
    if ":" not in text:
        raise ValueError(f"Unexpected SPF origin value: {raw!r}")
    year, quarter = text.split(":", 1)
    return f"{int(year)}:Q{int(quarter)}"


def is_missing_value(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(numeric):
        return True
    return abs(numeric - MISSING_SENTINEL) < 1e-9


def clean_numeric(value: Any) -> float:
    if is_missing_value(value):
        return float("nan")
    return float(value)


def download_spf_error_file(variable: str, work_dir: Path = WORK_ROOT, *, refresh: bool = False) -> Path:
    spec = SPF_VARIABLES[variable.upper()]
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"Data_SPF_Error_Statistics_{spec.code}.xls"
    if path.exists() and not refresh:
        return path
    response = requests.get(spec.error_data_url, timeout=45)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def read_xls_table(path: Path) -> pd.DataFrame:
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    if sheet.nrows < 2:
        raise ValueError(f"SPF workbook {path} has no data rows")
    columns = [str(value).strip() for value in sheet.row_values(0)]
    rows = [sheet.row_values(row_idx) for row_idx in range(1, sheet.nrows)]
    return pd.DataFrame(rows, columns=columns)


def load_spf_error_data(
    variables: Iterable[str],
    *,
    work_dir: Path = WORK_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for variable in parse_variable_list(variables):
        spec = SPF_VARIABLES[variable]
        path = download_spf_error_file(variable, work_dir, refresh=refresh)
        raw = read_xls_table(path)
        for _, row in raw.iterrows():
            origin = normalize_origin(row["Date"])
            origin_year = int(origin.split(":")[0])
            origin_quarter = int(origin.split("Q")[1])
            origin_idx = quarter_index(origin)
            for horizon in range(1, 6):
                records.append(
                    {
                        "variable": variable,
                        "variable_name": spec.name,
                        "units": spec.units,
                        "origin": origin,
                        "origin_year": origin_year,
                        "origin_quarter": origin_quarter,
                        "origin_index": origin_idx,
                        "horizon": horizon,
                        "spf_forecast": clean_numeric(row.get(f"SPFfor_Step{horizon}")),
                        "official_iar_forecast": clean_numeric(row.get(f"IARfor_Step{horizon}")),
                        "official_no_change_forecast": clean_numeric(row.get(f"NCfor_Step{horizon}")),
                        "official_dar_forecast": clean_numeric(row.get(f"DARfor_Step{horizon}")),
                        "official_darm_forecast": clean_numeric(row.get(f"DARMfor_Step{horizon}")),
                        "realized": clean_numeric(row.get(f"Realiz{horizon}")),
                        "source_url": spec.error_data_url,
                        "variable_page_url": spec.page_url,
                    }
                )
    frame = pd.DataFrame(records)
    frame = frame.sort_values(["variable", "horizon", "origin_index"]).reset_index(drop=True)
    return frame
