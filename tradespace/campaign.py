from __future__ import annotations

import copy
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import config_hash, load_config, resolve_config, strip_internal, validate_or_raise
from .mission_models import simulate_model
from .optimization import optimize_config
from .reliability import analyze_reliability, run_stress_test, write_margin_report
from .reports import read_csv, summarize, write_csv, write_json
from .runner import run_monte_carlo
from .samplers import generate_samples
from .sensitivity import sensitivity_table
from .sweep import run_sweep


CAMPAIGN_STATUS = "campaign_status.json"
SCENARIO_STATUS = "scenario_status.json"
SCENARIO_CONFIG = "scenario_config.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "scenario"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _path(raw: str | Path, root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (root / path)


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(manifest.get("scenarios"), list) or not manifest["scenarios"]:
        raise ValueError("campaign.scenarios: expected a nonempty list")
    return manifest_path, manifest


def _campaign_name(manifest: dict[str, Any], manifest_path: Path) -> str:
    campaign = manifest.get("campaign", {})
    if isinstance(campaign, dict) and campaign.get("name"):
        return str(campaign["name"])
    if manifest.get("name"):
        return str(manifest["name"])
    return manifest_path.stem


def _campaign_out_dir(manifest: dict[str, Any], manifest_path: Path, out_dir: str | Path | None) -> Path:
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()
    raw = manifest.get("out_dir") or manifest.get("output_dir")
    if raw:
        return _path(raw, manifest_path.parent).resolve()
    return (manifest_path.parent / "outputs" / _slug(_campaign_name(manifest, manifest_path))).resolve()


def _scenario_out_dir(campaign_dir: Path, scenario: dict[str, Any]) -> Path:
    raw = scenario.get("out") or scenario.get("out_dir")
    if raw:
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else campaign_dir / path
    return campaign_dir / _slug(str(scenario.get("name", "scenario")))


def _scenario_config(manifest: dict[str, Any], scenario: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    source = scenario.get("config") or scenario.get("config_path") or manifest.get("base_config")
    if source:
        config = load_config(_path(source, manifest_dir))
    else:
        raw = scenario.get("config_data") or manifest.get("base_config_data")
        if not isinstance(raw, dict):
            raise ValueError(f"{scenario.get('name', 'scenario')}: missing config, base_config, or config_data")
        config = resolve_config(raw, manifest_dir)

    overrides = {}
    if isinstance(manifest.get("overrides"), dict):
        overrides = _merge(overrides, manifest["overrides"])
    if isinstance(scenario.get("overrides"), dict):
        overrides = _merge(overrides, scenario["overrides"])
    if overrides:
        config = resolve_config(_merge(config, overrides), config.get("__config_dir", manifest_dir))

    if "seed" in scenario:
        config["seed"] = int(scenario["seed"])
    if "runs" in scenario:
        config["runs"] = int(scenario["runs"])
    if "sampler" in scenario:
        config.setdefault("sampler", {})
        config["sampler"]["method"] = str(scenario["sampler"])

    config.setdefault("study", {})
    config["study"]["name"] = str(scenario.get("name", config["study"].get("name", "scenario")))
    return resolve_config(config, config.get("__config_dir", manifest_dir))


def expand_campaign(manifest_path: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    path, manifest = _load_manifest(manifest_path)
    campaign_dir = _campaign_out_dir(manifest, path, out_dir)
    rows = []
    seen: set[str] = set()
    for index, scenario in enumerate(manifest["scenarios"]):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{index}]: expected object")
        name = str(scenario.get("name") or f"scenario_{index + 1}")
        slug = _slug(name)
        if slug in seen:
            raise ValueError(f"scenarios[{index}].name: duplicate scenario slug '{slug}'")
        seen.add(slug)
        config = _scenario_config(manifest, scenario, path.parent)
        rows.append(
            {
                "index": index,
                "name": name,
                "slug": slug,
                "command": str(scenario.get("command", "run")),
                "out_dir": str(_scenario_out_dir(campaign_dir, scenario)),
                "config_hash": config_hash(config),
                "runs": int(config.get("runs", 0)),
                "sampler": config.get("sampler", {}).get("method", "monte_carlo"),
            }
        )
    return {
        "campaign": _campaign_name(manifest, path),
        "manifest": str(path),
        "out_dir": str(campaign_dir),
        "scenarios": rows,
    }


def _status_path(out_dir: Path) -> Path:
    return out_dir / SCENARIO_STATUS


def _read_status(out_dir: Path) -> dict[str, Any] | None:
    path = _status_path(out_dir)
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return status if isinstance(status, dict) else None


def _write_status(out_dir: Path, payload: dict[str, Any]) -> None:
    write_json(_status_path(out_dir), payload)


def _scenario_row(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("summary", {}) if isinstance(status.get("summary"), dict) else {}
    return {
        "scenario": status.get("name", ""),
        "command": status.get("command", ""),
        "status": status.get("status", ""),
        "config_hash": status.get("config_hash", ""),
        "out_dir": status.get("out_dir", ""),
        "runs": summary.get("runs", summary.get("cases", summary.get("evaluations", ""))),
        "success_rate": summary.get("success_rate", ""),
        "pareto_rows": summary.get("pareto_rows", ""),
        "best_score": summary.get("best_score", ""),
        "error": status.get("error", ""),
    }


def _run_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(payload["manifest_path"])
    campaign_dir = Path(payload["campaign_dir"])
    scenario = payload["scenario"]
    out_dir = Path(payload["out_dir"])
    name = str(scenario.get("name"))
    command = str(scenario.get("command", "run")).replace("_", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    log_path = out_dir / "scenario.log"
    try:
        config = _scenario_config(payload["manifest"], scenario, manifest_path.parent)
        validate_or_raise(config)
        write_json(out_dir / SCENARIO_CONFIG, strip_internal(config))
        hash_value = config_hash(config)
        status = {
            "name": name,
            "command": command,
            "status": "running",
            "started_utc": started,
            "config_hash": hash_value,
            "out_dir": str(out_dir),
            "campaign_dir": str(campaign_dir),
        }
        _write_status(out_dir, status)

        if command in {"run", "monte-carlo", "robustness"}:
            summary = run_monte_carlo(config, out_dir, runs=scenario.get("runs"))
        elif command == "sweep":
            summary = run_sweep(config, out_dir)
        elif command == "optimize":
            method = str(scenario.get("method", "genetic"))
            iterations = int(scenario.get("iterations", 20))
            population = int(scenario.get("population", 16))
            summary = optimize_config(out_dir / SCENARIO_CONFIG, out_dir, method=method, iterations=iterations, population=population)
        elif command == "stress-test":
            max_cases = int(scenario.get("max_cases", 64))
            summary = run_stress_test(out_dir / SCENARIO_CONFIG, out_dir, max_cases=max_cases)
        elif command == "reliability":
            run_monte_carlo(config, out_dir, runs=scenario.get("runs"))
            summary = analyze_reliability(out_dir)
        elif command == "margin-report":
            run_monte_carlo(config, out_dir, runs=scenario.get("runs"))
            summary = write_margin_report(out_dir)
        else:
            raise ValueError(f"{name}: unsupported campaign command '{command}'")

        finished = _now()
        status.update({"status": "complete", "finished_utc": finished, "summary": summary})
        _write_status(out_dir, status)
        log_path.write_text(json.dumps({"status": "complete", "started_utc": started, "finished_utc": finished}, indent=2) + "\n")
        return status
    except Exception as exc:
        finished = _now()
        status = {
            "name": name,
            "command": command,
            "status": "failed",
            "started_utc": started,
            "finished_utc": finished,
            "out_dir": str(out_dir),
            "error": str(exc),
        }
        _write_status(out_dir, status)
        log_path.write_text(traceback.format_exc())
        return status


def _write_campaign_status(campaign_dir: Path, payload: dict[str, Any]) -> None:
    write_json(campaign_dir / CAMPAIGN_STATUS, payload)


def _write_campaign_outputs(campaign_dir: Path, campaign: str, manifest_path: Path, statuses: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_scenario_row(status) for status in statuses]
    complete = sum(1 for status in statuses if status.get("status") == "complete")
    failed = sum(1 for status in statuses if status.get("status") == "failed")
    skipped = sum(1 for status in statuses if status.get("skipped"))
    summary = {
        "campaign": campaign,
        "manifest": str(manifest_path),
        "out_dir": str(campaign_dir),
        "scenarios": len(statuses),
        "complete": complete,
        "failed": failed,
        "skipped": skipped,
        "updated_utc": _now(),
    }
    write_csv(campaign_dir / "campaign_summary.csv", rows)
    _write_campaign_status(campaign_dir, {**summary, "scenario_status": statuses})
    return summary


def run_campaign(
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    resume: bool = True,
    retry_failed: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    path, manifest = _load_manifest(manifest_path)
    campaign = _campaign_name(manifest, path)
    campaign_dir = _campaign_out_dir(manifest, path, out_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_json(campaign_dir / "campaign_manifest.json", manifest)

    planned = expand_campaign(path, campaign_dir)
    statuses: list[dict[str, Any]] = []
    tasks = []
    for row, scenario in zip(planned["scenarios"], manifest["scenarios"], strict=True):
        scenario = copy.deepcopy(scenario)
        scenario["name"] = row["name"]
        scenario_out = Path(row["out_dir"])
        existing = _read_status(scenario_out)
        if existing and existing.get("status") == "complete" and resume:
            existing = {**existing, "skipped": True}
            statuses.append(existing)
            continue
        if existing and existing.get("status") == "failed" and resume and not retry_failed:
            existing = {**existing, "skipped": True}
            statuses.append(existing)
            continue
        tasks.append(
            {
                "manifest_path": str(path),
                "manifest": manifest,
                "campaign_dir": str(campaign_dir),
                "scenario": scenario,
                "out_dir": row["out_dir"],
            }
        )

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futures = [pool.submit(_run_scenario, task) for task in tasks]
            for future in as_completed(futures):
                statuses.append(future.result())
                _write_campaign_outputs(campaign_dir, campaign, path, statuses)
    else:
        for task in tasks:
            statuses.append(_run_scenario(task))
            _write_campaign_outputs(campaign_dir, campaign, path, statuses)

    statuses = sorted(statuses, key=lambda item: str(item.get("name", "")))
    return _write_campaign_outputs(campaign_dir, campaign, path, statuses)


def replay_run(campaign_dir: str | Path, scenario_name: str, run_index: int = 0) -> dict[str, Any]:
    root = Path(campaign_dir).expanduser().resolve()
    scenario_dir = root / _slug(scenario_name)
    status = _read_status(scenario_dir)
    if status is None:
        raise FileNotFoundError(f"{scenario_dir / SCENARIO_STATUS} not found")
    config_path = scenario_dir / SCENARIO_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    config = load_config(config_path)
    n = int(config.get("runs", max(run_index + 1, 1)))
    if run_index < 0 or run_index >= n:
        raise ValueError(f"run_index must satisfy 0 <= run_index < {n}")
    sampler = config.get("sampler", {})
    plan = generate_samples(
        config.get("nominal", {}),
        config.get("uncertainties", {}),
        n,
        int(config.get("seed", 2026)),
        method=str(sampler.get("method", "monte_carlo")),
        root=config.get("__config_dir"),
    )
    selected = next((row for row in plan if row[0] == run_index), None)
    if selected is None:
        raise ValueError(f"run {run_index} not generated")
    index, seed, params, sample = selected
    result = simulate_model(params, config)
    replay = {
        "campaign_dir": str(root),
        "scenario": status.get("name", scenario_name),
        "run": index,
        "seed": seed,
        "sample": sample,
        "result": result,
        "config_hash": config_hash(config),
        "replayed_utc": _now(),
    }
    out = root / "replays"
    write_json(out / f"{_slug(scenario_name)}_run_{run_index}.json", replay)
    return replay


def _load_campaign_rows(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(path).expanduser().resolve()
    summary_path = root / "campaign_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")
    return root, read_csv(summary_path)


def compare_campaigns(left_dir: str | Path, right_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    left_root, left_rows = _load_campaign_rows(left_dir)
    right_root, right_rows = _load_campaign_rows(right_dir)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    right_by_name = {str(row.get("scenario")): row for row in right_rows}
    comparison = []
    for left in left_rows:
        name = str(left.get("scenario"))
        right = right_by_name.get(name)
        if right is None:
            comparison.append({"scenario": name, "status": "missing_right"})
            continue
        row = {
            "scenario": name,
            "left_status": left.get("status"),
            "right_status": right.get("status"),
            "left_runs": left.get("runs", ""),
            "right_runs": right.get("runs", ""),
        }
        for metric in ("success_rate", "pareto_rows", "best_score"):
            try:
                row[f"{metric}_delta"] = float(right.get(metric, 0.0)) - float(left.get(metric, 0.0))
            except (TypeError, ValueError):
                row[f"{metric}_delta"] = ""
        comparison.append(row)
    left_names = {str(row.get("scenario")) for row in left_rows}
    for right in right_rows:
        name = str(right.get("scenario"))
        if name not in left_names:
            comparison.append({"scenario": name, "status": "missing_left"})

    write_csv(out / "campaign_comparison.csv", comparison)
    summary = {
        "left": str(left_root),
        "right": str(right_root),
        "out_dir": str(out),
        "matched": sum(1 for row in comparison if "left_status" in row and "right_status" in row),
        "left_only": sum(1 for row in comparison if row.get("status") == "missing_right"),
        "right_only": sum(1 for row in comparison if row.get("status") == "missing_left"),
        "rows": len(comparison),
    }
    write_json(out / "campaign_comparison.json", summary)
    return summary


def summarize_campaign_study(campaign_dir: str | Path) -> dict[str, Any]:
    root, rows = _load_campaign_rows(campaign_dir)
    study_rows = []
    for row in rows:
        result_path = Path(str(row.get("out_dir", ""))) / "results.csv"
        if not result_path.exists():
            continue
        results = read_csv(result_path)
        samples_path = Path(str(row.get("out_dir", ""))) / "samples.csv"
        samples = read_csv(samples_path) if samples_path.exists() else []
        study_rows.append({"scenario": row.get("scenario"), **summarize(results, sensitivity_table(samples, results))})
    write_json(root / "campaign_rollup.json", study_rows)
    return {"campaign": str(root), "studies": len(study_rows)}
