from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import config_hash, resolve_config, strip_internal, validate_or_raise
from .mission_models import simulate_model
from .pareto import DEFAULT_OBJECTIVES, pareto_front, score_designs
from .reports import summarize, write_csv, write_json, write_report
from .samplers import bounded_random_cases, generate_samples, grid_cases, run_seed
from .sensitivity import sensitivity_table


def _build_cases(sweep: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    mode = str(sweep.get("mode", "grid")).lower()
    if mode == "grid":
        return grid_cases(sweep.get("values", {}))
    if mode in {"latin", "random", "sobol"}:
        return bounded_random_cases(sweep.get("bounds", {}), int(sweep.get("samples", 40)), seed, method=mode)
    if mode == "evolutionary":
        # Lightweight random-plus-elite pass using only NumPy.
        bounds = sweep.get("bounds", {})
        population = bounded_random_cases(bounds, int(sweep.get("population", 24)), seed, method="latin")
        rng = np.random.default_rng(seed + 31)
        names = list(bounds.keys())
        for _ in range(int(sweep.get("generations", 2))):
            mutants = []
            for parent in population[: max(2, len(population) // 3)]:
                child = dict(parent)
                for name in names:
                    low, high = float(bounds[name][0]), float(bounds[name][1])
                    child[name] = float(np.clip(child[name] + rng.normal(0.0, 0.08 * (high - low)), low, high))
                mutants.append(child)
            population.extend(mutants)
        return population
    if mode == "nested":
        return grid_cases(sweep.get("values", {}))
    return grid_cases(sweep.get("values", {}))


def _aggregate_designs(rows: list[dict[str, Any]], design_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["case"]), []).append(row)
    aggregates = []
    for case, case_rows in grouped.items():
        miss = np.array([float(row["miss_distance_m"]) for row in case_rows], dtype=float)
        qbar = np.array([float(row["max_qbar_pa"]) for row in case_rows], dtype=float)
        impulse = np.array([float(row["impulse_n_s"]) for row in case_rows], dtype=float)
        margin = np.array([float(row["robustness_margin"]) for row in case_rows], dtype=float)
        success = np.array([1.0 if row.get("success") else 0.0 for row in case_rows], dtype=float)
        design_values = {key: case_rows[0].get(key) for key in design_keys}
        aggregates.append(
            {
                "case": case,
                **design_values,
                "runs": len(case_rows),
                "success_rate": float(success.mean()),
                "success": bool(success.mean() >= 0.5),
                "miss_distance_m": float(np.percentile(miss, 50)),
                "miss_p95_m": float(np.percentile(miss, 95)),
                "max_qbar_pa": float(np.percentile(qbar, 95)),
                "impulse_n_s": float(np.mean(impulse)),
                "robustness_margin": float(np.percentile(margin, 5)),
                "failure_reason": "aggregate",
            }
        )
    return aggregates


def run_sweep(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    resolved = resolve_config(config, config.get("__config_dir"))
    validate_or_raise(resolved)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", strip_internal(resolved))

    seed = int(resolved.get("seed", 2026))
    sweep = resolved.get("sweep", {})
    outputs = resolved.get("outputs", {})
    write_manifest = bool(outputs.get("write_manifest", True))
    write_html = bool(outputs.get("write_html", True))
    cases = _build_cases(sweep, seed)
    design_keys = sorted({key for case in cases for key in case})
    reps = int(sweep.get("uncertainty_runs", sweep.get("replicates", 1)))
    root = resolved.get("__config_dir")
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases):
        nominal = {**resolved.get("nominal", {}), **case}
        if reps > 1 or sweep.get("include_uncertainty", False):
            plan = generate_samples(
                nominal,
                resolved.get("uncertainties", {}),
                max(reps, 1),
                run_seed(seed, case_index),
                method=str(sweep.get("uncertainty_sampler", resolved.get("sampler", {}).get("method", "monte_carlo"))),
                root=root,
            )
            for rep, seed_i, params, sample in plan:
                result = simulate_model(params, resolved)
                rows.append({"case": case_index, "replicate": rep, **case, **result})
                sample_rows.append({"case": case_index, "replicate": rep, **case, **sample})
        else:
            params = {**nominal, "run_seed": run_seed(seed, case_index)}
            result = simulate_model(params, resolved)
            rows.append({"case": case_index, "replicate": 0, **case, **result})
            sample_rows.append({"case": case_index, "replicate": 0, **case, "run_seed": params["run_seed"]})

    design_rows = _aggregate_designs(rows, design_keys)
    constraints = sweep.get("constraints", resolved.get("constraints", {}))
    objectives = sweep.get("objectives", DEFAULT_OBJECTIVES)
    ranked = score_designs(design_rows, objectives=objectives, constraints=constraints)
    pareto_rows = pareto_front([row for row in ranked if row.get("feasible")], objectives)
    sensitivity = sensitivity_table(sample_rows or rows, rows)
    summary = summarize(rows, sensitivity)
    summary.update(
        {
            "cases": len(cases),
            "design_rows": len(design_rows),
            "pareto_rows": len(pareto_rows),
            "best_case": ranked[0] if ranked else {},
            "config_hash": config_hash(resolved),
        }
    )

    write_csv(out / "sweep_results.csv", rows)
    write_csv(out / "results.csv", rows)
    write_csv(out / "samples.csv", sample_rows)
    write_csv(out / "design_comparison.csv", ranked)
    write_csv(out / "pareto.csv", pareto_rows)
    write_csv(out / "sensitivity.csv", sensitivity)
    write_json(out / "sweep_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out, rows, summary, sensitivity, samples=sample_rows, pareto_rows=pareto_rows, write_html=write_html)
    if write_manifest:
        write_json(
            out / "manifest.json",
            {
                "study_name": resolved.get("study", {}).get("name", "sweep"),
                "status": "complete",
                "config_hash": summary["config_hash"],
                "seed": seed,
                "sweep": sweep,
                "cases": len(cases),
                "rows": len(rows),
                "case_records": [{"case": i, "status": "complete", **case} for i, case in enumerate(cases)],
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
    return summary
