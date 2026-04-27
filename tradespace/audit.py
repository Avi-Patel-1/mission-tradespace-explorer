from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .reports import read_csv


REQUIRED_STUDY_FILES = [
    "resolved_config.json",
    "results.csv",
    "samples.csv",
    "summary.json",
    "sensitivity.csv",
    "report.html",
]


def _load_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        warnings.append(f"{path.name}: invalid JSON at line {exc.lineno}, column {exc.colno}")
        return {}


def _sqlite_count(path: Path, warnings: list[str]) -> int | None:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(path)
        count = int(conn.execute("select count(*) from runs where status = 'complete'").fetchone()[0])
        conn.close()
        return count
    except sqlite3.Error as exc:
        warnings.append(f"{path.name}: could not inspect SQLite database: {exc}")
        return None


def audit_study(study_dir: str | Path) -> dict[str, Any]:
    study = Path(study_dir)
    warnings: list[str] = []
    errors: list[str] = []

    if not study.exists():
        return {"ok": False, "errors": [f"{study} does not exist"], "warnings": [], "files": {}}
    if not study.is_dir():
        return {"ok": False, "errors": [f"{study} is not a directory"], "warnings": [], "files": {}}

    files = {name: (study / name).exists() for name in REQUIRED_STUDY_FILES}
    for name, exists in files.items():
        if not exists:
            errors.append(f"{name}: required study artifact is missing")

    results = read_csv(study / "results.csv")
    samples = read_csv(study / "samples.csv")
    sensitivity = read_csv(study / "sensitivity.csv")
    summary = _load_json(study / "summary.json", warnings)
    manifest = _load_json(study / "manifest.json", warnings)
    resolved = _load_json(study / "resolved_config.json", warnings)

    if results and samples and len(results) != len(samples):
        errors.append(f"row count mismatch: results.csv has {len(results)} rows, samples.csv has {len(samples)}")
    if summary:
        expected_runs = int(summary.get("runs", -1))
        if expected_runs != -1 and expected_runs != len(results):
            errors.append(f"summary.json runs={expected_runs}, but results.csv has {len(results)} rows")
        success_rate = summary.get("success_rate")
        if success_rate is not None and not (0.0 <= float(success_rate) <= 1.0):
            errors.append("summary.json success_rate must be between 0 and 1")
    if manifest:
        manifest_runs = int(manifest.get("runs", -1))
        if manifest_runs != -1 and manifest_runs != len(results):
            errors.append(f"manifest.json runs={manifest_runs}, but results.csv has {len(results)} rows")
        if summary and manifest.get("config_hash") and summary.get("config_hash") and manifest["config_hash"] != summary["config_hash"]:
            errors.append("config hash mismatch between manifest.json and summary.json")
    if resolved and summary and resolved.get("seed") is None:
        warnings.append("resolved_config.json does not include seed")

    sqlite_runs = _sqlite_count(study / "study.sqlite", warnings)
    if sqlite_runs is not None and results and sqlite_runs < len(results):
        warnings.append(f"study.sqlite has {sqlite_runs} completed rows for {len(results)} result rows")

    if not sensitivity:
        warnings.append("sensitivity.csv has no rows")
    if results:
        failures = sorted({str(row.get("failure_reason", "none")) for row in results})
    else:
        failures = []

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "files": files,
        "rows": {
            "results": len(results),
            "samples": len(samples),
            "sensitivity": len(sensitivity),
            "sqlite_completed": sqlite_runs,
        },
        "summary": {
            "runs": summary.get("runs"),
            "success_rate": summary.get("success_rate"),
            "config_hash": summary.get("config_hash"),
        },
        "failure_reasons": failures,
    }
