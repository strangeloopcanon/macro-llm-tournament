"""Content-addressed provenance for the executable Python package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_CONTRACT_SCHEMA_VERSION = "macro_llm_source_contract_v1"
SOURCE_ROOT = "src/macro_llm_tournament"


class SourceProvenanceError(ValueError):
    pass


def build_source_contract(project_root: Path) -> dict[str, Any]:
    source_root = project_root.resolve() / SOURCE_ROOT
    if not source_root.is_dir():
        raise SourceProvenanceError(f"Source root is missing: {source_root}")
    files = {
        str(path.relative_to(project_root.resolve())): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_root.glob("*.py"))
        if path.is_file()
    }
    if not files:
        raise SourceProvenanceError("Source contract cannot be empty")
    return {
        "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
        "source_root": SOURCE_ROOT,
        "file_count": len(files),
        "files": files,
        "tree_sha256": _canonical_sha(files),
    }


def validate_source_contract(
    expected: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    if expected.get("schema_version") != SOURCE_CONTRACT_SCHEMA_VERSION:
        raise SourceProvenanceError("Unsupported source contract schema")
    actual = build_source_contract(project_root)
    if actual != expected:
        raise SourceProvenanceError("Executable source tree does not match the locked contract")
    return actual


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
