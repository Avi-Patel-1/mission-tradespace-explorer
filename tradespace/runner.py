from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import config_hash, load_config, resolve_config, strip_internal, validate_or_raise
from .mission_models import simulate_model
from .reports import summarize, write_csv, write_json, write_report
from .samplers import generate_samples
from .sensitivity import sensitivity_table
from .storage import SQLiteStudy


def _resolved(config: dict[str, Any]) -> dict[str, Any]:
    return resolve_config(config, config.get("__config_dir"))


def _write_manifest(out: Path, config: dict[str, Any], run_records: list[dict[str, Any]], *, status: str) -> None:
    manifest = {
        "study_name": config.get("study", {}).get("name", "study"),
        "status": status,
        "config_hash": config_hash(config),
        "seed": config.get("seed"),
        "sampler": config.get("sampler", {}),
        "runs": len(run_records),
        "run_records": run_records,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    write_json(out / "manifest.json", manifest)


def run_monte_carlo(
    config: dict[str, Any],
    out_dir: str | Path,
    runs: int | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    resolved = _resolved(config)
    if runs is not None:
        resolved = {**resolved, "runs": int(runs)}
    validate_or_raise(resolved)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", strip_internal(resolved))

    n = int(resolved.get("runs", 1000))
    base_seed = int(resolved.get("seed", 2026))
    sampler = resolved.get("sampler", {})
    method = str(sampler.get("method", "monte_carlo"))
    chunk_size = max(1, int(sampler.get("chunk_size", n)))
    outputs = resolved.get("outputs", {})
    write_manifest = bool(outputs.get("write_manifest", True))
    write_html = bool(outputs.get("write_html", True))
    root = resolved.get("__config_dir")
    plan = generate_samples(
        resolved.get("nominal", {}),
        resolved.get("uncertainties", {}),
        n,
        base_seed,
        method=method,
        root=root,
    )

    hash_value = config_hash(resolved)
    write_sqlite = bool(outputs.get("write_sqlite", True))
    db = SQLiteStudy(out / "study.sqlite") if write_sqlite else None
    completed = db.completed_runs(config_hash=hash_value) if (db is not None and resume) else {}
    if db is not None:
        db.set_metadata("config_hash", hash_value)
        db.set_metadata("seed", base_seed)
        db.set_metadata("sampler", sampler)

    samples: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    try:
        for index, seed_i, params, sample in plan:
            if index in completed:
                stored_sample, stored_result, stored_seed = completed[index]
                samples.append({"run": index, **stored_sample})
                results.append({"run": index, **stored_result})
                run_records.append({"run": index, "seed": stored_seed, "status": "loaded"})
                continue
            result = simulate_model(params, resolved)
            samples.append({"run": index, **sample})
            results.append({"run": index, **result})
            run_records.append({"run": index, "seed": seed_i, "status": "complete"})
            if db is not None:
                db.record_run(index, seed_i, sample, result, config_hash=hash_value)
            if write_manifest and len(run_records) % chunk_size == 0:
                _write_manifest(out, resolved, run_records, status="running")
    finally:
        if db is not None:
            db.close()

    sensitivity = sensitivity_table(samples, results)
    summary = summarize(results, sensitivity)
    summary["config_hash"] = hash_value
    summary["sampler"] = method
    summary["resumed_runs"] = sum(1 for record in run_records if record["status"] == "loaded")
    write_csv(out / "samples.csv", samples)
    write_csv(out / "results.csv", results)
    write_csv(out / "sensitivity.csv", sensitivity)
    write_json(out / "summary.json", summary)
    write_report(out, results, summary, sensitivity, samples=samples, write_html=write_html)
    if write_manifest:
        _write_manifest(out, resolved, run_records, status="complete")
    return summary


def run_resume(study_dir: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    study = Path(study_dir)
    if config_path is not None:
        config = load_config(config_path)
    else:
        resolved_path = study / "resolved_config.json"
        if not resolved_path.exists():
            raise FileNotFoundError(f"{resolved_path} is required when --config is not provided")
        config = json.loads(resolved_path.read_text())
        config = resolve_config(config, study)
    return run_monte_carlo(config, study, resume=True)
