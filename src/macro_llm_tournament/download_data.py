from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import requests

from .forecast_cards import build_forecast_cards, cards_to_frame
from .forecast_data import PROJECT_ROOT, SPF_VARIABLES, download_spf_error_file, load_spf_error_data
from .fred_vintage import build_vintage_context_for_cards
from .survey_beliefs import load_survey_belief_targets


WORK_ROOT = PROJECT_ROOT / "work"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series_id}"

FRED_SERIES: tuple[str, ...] = (
    "GDPC1",
    "PCECC96",
    "GPDIC1",
    "INDPRO",
    "PAYEMS",
    "UNRATE",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "PCEPILFE",
    "FEDFUNDS",
    "TB3MS",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "M2SL",
    "MICH",
    "UMCSENT",
    "PSAVERT",
    "DSPIC96",
    "RSXFS",
    "HOUST",
)

SCF_INDEX_URL = "https://www.federalreserve.gov/econres/scfindex.htm"
SCF_PREVIOUS_SURVEYS_URL = "https://www.federalreserve.gov/econres/scf-previous-surveys.htm"
SCF_MODERN_WAVES = frozenset(range(1989, 2023, 3))
SCF_CURATED_FILE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^scfp\d{4}(?:excel|s)?\.zip$",  # summary extract public data
        r"^scf\d{4}_tables_.*\.xlsx$",  # 1989-current historical table files
        r"^tables1_2_.*\.xlsx$",
        r"^bulletin\.macro\.txt$",
        r"^creating.*scf.*tables.*\.docx$",
        r"^codebk\d{2,4}\.txt$",
        r"^\d{2,4}map\.(?:xls|xlsx|txt)$",
        r"^\d{4}_scf_changes\.txt$",
        r"^standard_error_documentation\.pdf$",
        r"^scfoutline\.\d{4}\.pdf$",
        r"^\d{4}_showcards\.pdf$",
        r"^scf\d{2,4}\.(?:txt|docx)$",
        r"^networth.*flowchart\.pdf$",
        r"^scf2022\.zip$",  # keep the already-used 2022 full public CPORT file
    )
)


@dataclass(frozen=True)
class DownloadResult:
    group: str
    name: str
    url: str
    path: str
    status: str
    bytes: int = 0
    sha256: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public datasets used by the forecast tournament.")
    parser.add_argument("--work-dir", default=str(WORK_ROOT))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-scf", action="store_true")
    parser.add_argument("--skip-alfred-current", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []
    results.extend(download_spf_bundle(work_dir, refresh=args.refresh))
    results.extend(download_survey_beliefs(work_dir, refresh=args.refresh))
    results.extend(download_fred_graph_bundle(work_dir, refresh=args.refresh, alfred_current=not args.skip_alfred_current))
    results.extend(download_default_card_vintage_context(work_dir, refresh=args.refresh))
    if not args.skip_scf:
        results.extend(download_scf_bundle(work_dir, refresh=args.refresh))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": relative_path(work_dir),
        "summary": summarize(results),
        "datasets": [result.__dict__ for result in results],
    }
    (work_dir / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (work_dir / "DATA_README.md").write_text(render_readme(manifest), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print(work_dir / "DATA_MANIFEST.json")
    return 0 if not any(result.status == "error" for result in results) else 1


def download_spf_bundle(work_dir: Path, *, refresh: bool) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    spf_dir = work_dir / "spf"
    for variable in sorted(SPF_VARIABLES):
        try:
            path = download_spf_error_file(variable, spf_dir, refresh=refresh)
            results.append(result_from_path("spf_error_statistics", variable, SPF_VARIABLES[variable].error_data_url, path))
        except Exception as exc:
            results.append(error_result("spf_error_statistics", variable, SPF_VARIABLES[variable].error_data_url, spf_dir, exc))
    results.extend(download_spf_detail_files(work_dir / "spf_detail", refresh=refresh))
    return results


def download_spf_detail_files(work_dir: Path, *, refresh: bool) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for variable, spec in sorted(SPF_VARIABLES.items()):
        try:
            response = requests.get(spec.page_url, timeout=45)
            response.raise_for_status()
            links = extract_spf_download_links(spec.page_url, response.text)
        except Exception as exc:
            results.append(error_result("spf_detail_page", variable, spec.page_url, work_dir / variable, exc))
            continue
        for url in links:
            name = Path(urlparse(url).path).name
            path = work_dir / variable / name
            results.append(download_file("spf_detail", f"{variable}/{name}", url, path, refresh=refresh))
    return results


def extract_spf_download_links(page_url: str, html_text: str) -> list[str]:
    raw_links: set[str] = set()
    patterns = (
        r'https?://[^"\']+?\.(?:xlsx|xls)(?:\?[^"\']*)?',
        r'/[^"\']+?\.(?:xlsx|xls)(?:\?[^"\']*)?',
    )
    for pattern in patterns:
        for raw in re.findall(pattern, html_text, flags=re.IGNORECASE):
            url = html.unescape(urljoin(page_url, raw))
            if "survey-of-professional-forecasters/data-files" not in url.lower():
                continue
            if "assets/images" in url.lower():
                continue
            raw_links.add(url)
    return sorted(raw_links)


def download_survey_beliefs(work_dir: Path, *, refresh: bool) -> list[DownloadResult]:
    survey_dir = work_dir / "survey_beliefs"
    before = set(survey_dir.glob("*")) if survey_dir.exists() else set()
    try:
        frame, status = load_survey_belief_targets(work_dir=survey_dir, refresh=refresh, mode="require")
    except Exception as exc:
        return [error_result("survey_beliefs", "michigan_sce", "Michigan FRED + NY Fed SCE", survey_dir, exc)]
    after = set(survey_dir.glob("*"))
    summary_path = survey_dir / "survey_belief_targets.csv"
    results = [
        result_from_path(
            "survey_beliefs",
            path.name,
            source_url_for_survey_file(path.name),
            path,
            status="cached" if path in before and not refresh else "downloaded",
        )
        for path in sorted(after | before)
        if path.is_file() and path != summary_path
    ]
    frame.to_csv(summary_path, index=False)
    results.append(result_from_path("survey_beliefs", summary_path.name, json.dumps(status, sort_keys=True), summary_path))
    return results


def source_url_for_survey_file(name: str) -> str:
    if "michigan" in name.lower() or name == "mich.csv":
        return "https://fred.stlouisfed.org/series/MICH"
    if "sce" in name.lower() or name.endswith(".xlsx"):
        return "https://www.newyorkfed.org/microeconomics/sce"
    return "survey belief derived file"


def download_fred_graph_bundle(work_dir: Path, *, refresh: bool, alfred_current: bool) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for series_id in FRED_SERIES:
        url = FRED_GRAPH_URL.format(series_id=series_id)
        path = work_dir / "fred_current" / f"{series_id}.csv"
        results.append(download_file("fred_current", series_id, url, path, refresh=refresh))
        if alfred_current:
            alfred_url = ALFRED_GRAPH_URL.format(series_id=series_id)
            alfred_path = work_dir / "alfred_current_vintage" / f"{series_id}.csv"
            results.append(download_file("alfred_current_vintage", series_id, alfred_url, alfred_path, refresh=refresh))
    return results


def download_default_card_vintage_context(work_dir: Path, *, refresh: bool) -> list[DownloadResult]:
    vintage_dir = work_dir / "fred_card_vintage"
    try:
        variables = ["CPI", "RGDP", "UNEMP", "TBILL"]
        spf_data = load_spf_error_data(variables, work_dir=work_dir / "spf", refresh=False)
        cards = build_forecast_cards(
            spf_data,
            variables=variables,
            horizons=[1],
            holdout_start_year=2015,
            holdout_end_year=2024,
            history_quarters=24,
            card_count=24,
        )
        contexts, vintage_rows, status = build_vintage_context_for_cards(
            cards,
            work_dir=work_dir / "fred_vintage",
            refresh=refresh,
            mode="best_effort",
        )
        vintage_dir.mkdir(parents=True, exist_ok=True)
        status_path = vintage_dir / "fred_vintage_status.json"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        cards_path = vintage_dir / "forecast_cards_for_vintage_context.csv"
        cards_to_frame(cards).to_csv(cards_path, index=False)
        context_path = vintage_dir / "fred_vintage_context.csv"
        vintage_rows.to_csv(context_path, index=False)
        context_json_path = vintage_dir / "fred_vintage_context_by_card.json"
        context_json_path.write_text(json.dumps(contexts, indent=2, sort_keys=True), encoding="utf-8")
        results = [
            result_from_path("fred_card_vintage", status_path.name, "FRED/ALFRED series observations API status", status_path),
            result_from_path("fred_card_vintage", cards_path.name, "Derived SPF cards for default vintage context", cards_path),
            result_from_path("fred_card_vintage", context_path.name, "FRED/ALFRED series observations API rows", context_path),
            result_from_path("fred_card_vintage", context_json_path.name, "FRED/ALFRED context by forecast card", context_json_path),
        ]
        if status.get("status") == "missing_api_key":
            return [
                DownloadResult(
                    group="fred_card_vintage",
                    name="default_24_card_vintage_context",
                    url="https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
                    path=relative_path(vintage_dir),
                    status="skipped",
                    error="FRED_API_KEY not found in .env or process environment.",
                ),
                *results,
            ]
        return results
    except Exception as exc:
        return [error_result("fred_card_vintage", "default_24_card_vintage_context", "FRED/ALFRED API", vintage_dir, exc)]


def download_scf_bundle(work_dir: Path, *, refresh: bool) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    try:
        page_urls = discover_scf_wave_pages()
    except Exception as exc:
        return [error_result("scf", "wave_page_discovery", SCF_PREVIOUS_SURVEYS_URL, work_dir / "scf", exc)]

    seen_urls: set[str] = set()
    for page_url in page_urls:
        wave = scf_wave_from_page_url(page_url)
        try:
            response = requests.get(page_url, timeout=45)
            response.raise_for_status()
            links = extract_scf_download_links(page_url, response.text)
        except Exception as exc:
            results.append(error_result("scf", f"{wave}/page", page_url, work_dir / "scf" / wave, exc))
            continue
        for url in links:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            name = safe_url_filename(url)
            path = work_dir / "scf" / wave / name
            results.append(download_file("scf", f"{wave}/{name}", url, path, refresh=refresh))
    return results


def discover_scf_wave_pages() -> list[str]:
    response = requests.get(SCF_PREVIOUS_SURVEYS_URL, timeout=45)
    response.raise_for_status()
    pages = set()
    for url in extract_links(SCF_PREVIOUS_SURVEYS_URL, response.text):
        match = re.search(r"/scf_(\d{4})\.htm$", urlparse(url).path, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group(1))
        if year in SCF_MODERN_WAVES:
            pages.add(url)
    return [SCF_INDEX_URL, *sorted(pages, key=scf_page_sort_key)]


def extract_scf_download_links(page_url: str, html_text: str) -> list[str]:
    links = []
    for url in extract_links(page_url, html_text):
        parsed = urlparse(url)
        if "/econres/files/" not in parsed.path.lower():
            continue
        if not re.search(r"\.(?:zip|xlsx|xls|txt|docx|pdf)$", parsed.path, flags=re.IGNORECASE):
            continue
        name = safe_url_filename(url)
        if any(pattern.match(name) for pattern in SCF_CURATED_FILE_PATTERNS):
            links.append(url)
    return sorted(set(links))


def extract_links(page_url: str, html_text: str) -> list[str]:
    links = []
    for raw in re.findall(r'href=["\']([^"\']+)["\']', html_text, flags=re.IGNORECASE):
        links.append(html.unescape(urljoin(page_url, raw)))
    return links


def scf_page_sort_key(url: str) -> tuple[int, str]:
    wave = scf_wave_from_page_url(url)
    return (int(wave) if wave.isdigit() else 9999, url)


def scf_wave_from_page_url(url: str) -> str:
    if url.rstrip("/").endswith("scfindex.htm"):
        return "2022"
    match = re.search(r"/scf_(\d{4})\.htm$", urlparse(url).path, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def safe_url_filename(url: str) -> str:
    return unquote(Path(urlparse(url).path).name).replace("/", "_")


def download_file(group: str, name: str, url: str, path: Path, *, refresh: bool) -> DownloadResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return result_from_path(group, name, url, path, status="cached")
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
        return result_from_path(group, name, url, path, status="downloaded")
    except Exception as exc:
        return error_result(group, name, url, path, exc)


def result_from_path(
    group: str,
    name: str,
    url: str,
    path: Path,
    *,
    status: str = "downloaded",
) -> DownloadResult:
    data = path.read_bytes()
    return DownloadResult(
        group=group,
        name=name,
        url=url,
        path=relative_path(path),
        status=status,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def error_result(group: str, name: str, url: str, path: Path, exc: Exception) -> DownloadResult:
    return DownloadResult(group=group, name=name, url=url, path=relative_path(path), status="error", error=str(exc))


def summarize(results: Iterable[DownloadResult]) -> dict[str, object]:
    rows = list(results)
    by_group: dict[str, dict[str, int]] = {}
    for row in rows:
        by_group.setdefault(row.group, {"files": 0, "bytes": 0, "errors": 0, "skipped": 0})
        by_group[row.group]["files"] += int(row.status not in {"error", "skipped"})
        by_group[row.group]["bytes"] += row.bytes
        by_group[row.group]["errors"] += int(row.status == "error")
        by_group[row.group]["skipped"] += int(row.status == "skipped")
    return {
        "files": sum(1 for row in rows if row.status not in {"error", "skipped"}),
        "bytes": sum(row.bytes for row in rows),
        "errors": sum(1 for row in rows if row.status == "error"),
        "skipped": sum(1 for row in rows if row.status == "skipped"),
        "groups": by_group,
    }


def render_readme(manifest: dict[str, object]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Local Data Manifest",
        "",
        "This directory is ignored by git. It contains downloaded public data for local experiments.",
        "",
        f"- Created UTC: `{manifest['created_utc']}`",
        f"- Files: `{summary['files']}`",
        f"- Bytes: `{summary['bytes']}`",
        f"- Errors: `{summary['errors']}`",
        "",
        "See `DATA_MANIFEST.json` for source URLs, relative local paths, byte counts, and SHA-256 checksums.",
        "",
    ]
    return "\n".join(lines)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
