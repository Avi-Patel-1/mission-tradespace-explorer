from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_config, validate_or_raise
from .mission_models import simulate_model
from .pareto import DEFAULT_OBJECTIVES, pareto_front, score_designs
from .reports import svg_scatter, write_csv, write_json
from .samplers import bounded_random_cases, run_seed


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _variable_spec(config: dict[str, Any]) -> dict[str, Any]:
    opt = config.get("optimize", {})
    if isinstance(opt, dict) and isinstance(opt.get("variables"), dict):
        return opt["variables"]
    sweep = config.get("sweep", {})
    if isinstance(sweep, dict) and isinstance(sweep.get("bounds"), dict):
        return {name: {"bounds": bounds} for name, bounds in sweep["bounds"].items()}
    if isinstance(sweep, dict) and isinstance(sweep.get("values"), dict):
        return {name: {"values": values} for name, values in sweep["values"].items()}
    return {
        "guidance_gain": {"bounds": [1.5, 3.0]},
        "thrust_n": {"bounds": [1900.0, 2500.0]},
    }


def _sample_candidate(spec: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    candidate = {}
    for name, rule in spec.items():
        if isinstance(rule, dict) and "values" in rule:
            values = list(rule["values"])
            candidate[name] = values[int(rng.integers(0, len(values)))]
        else:
            bounds = rule.get("bounds", rule) if isinstance(rule, dict) else rule
            low, high = float(bounds[0]), float(bounds[1])
            candidate[name] = float(rng.uniform(low, high))
    return candidate


def _repair(candidate: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(candidate)
    for name, rule in spec.items():
        if isinstance(rule, dict) and "values" in rule:
            values = list(rule["values"])
            repaired[name] = min(values, key=lambda value: abs(_as_float(value) - _as_float(repaired.get(name))))
        else:
            bounds = rule.get("bounds", rule) if isinstance(rule, dict) else rule
            repaired[name] = float(np.clip(_as_float(repaired.get(name)), float(bounds[0]), float(bounds[1])))
    return repaired


def _mutate(candidate: dict[str, Any], spec: dict[str, Any], rng: np.random.Generator, scale: float) -> dict[str, Any]:
    child = dict(candidate)
    for name, rule in spec.items():
        if rng.random() > 0.6:
            continue
        if isinstance(rule, dict) and "values" in rule:
            values = list(rule["values"])
            child[name] = values[int(rng.integers(0, len(values)))]
        else:
            bounds = rule.get("bounds", rule) if isinstance(rule, dict) else rule
            width = float(bounds[1]) - float(bounds[0])
            child[name] = _as_float(child.get(name)) + rng.normal(0.0, scale * width)
    return _repair(child, spec)


def _crossover(a: dict[str, Any], b: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in a}


def _score(row: dict[str, Any], objectives: list[dict[str, str]], constraints: dict[str, Any]) -> float:
    penalty = 0.0
    if constraints.get("require_success", False) and not row.get("success"):
        penalty += 1e6
    for metric, rule in constraints.items():
        if metric == "require_success" or not isinstance(rule, dict):
            continue
        value = _as_float(row.get(metric))
        if "max" in rule:
            penalty += max(0.0, value - float(rule["max"])) * 1e3
        if "min" in rule:
            penalty += max(0.0, float(rule["min"]) - value) * 1e3
    total = penalty
    for objective in objectives:
        value = _as_float(row.get(objective["metric"]))
        weight = float(objective.get("weight", 1.0))
        total += weight * (value if objective.get("sense", "min") == "min" else -value)
    return float(total)


def _evaluate(config: dict[str, Any], design: dict[str, Any], index: int) -> dict[str, Any]:
    params = {**config.get("nominal", {}), **design, "run_seed": run_seed(int(config.get("seed", 2026)), index)}
    result = simulate_model(params, config)
    return {"candidate": index, **design, **result}


def _initial_population(spec: dict[str, Any], population: int, seed: int) -> list[dict[str, Any]]:
    continuous = {name: rule.get("bounds", rule) for name, rule in spec.items() if not (isinstance(rule, dict) and "values" in rule)}
    rng = np.random.default_rng(seed)
    cases = bounded_random_cases(continuous, population, seed, method="latinized_sobol") if continuous else [{} for _ in range(population)]
    population_rows = []
    for case in cases:
        row = dict(case)
        for name, rule in spec.items():
            if isinstance(rule, dict) and "values" in rule:
                values = list(rule["values"])
                row[name] = values[int(rng.integers(0, len(values)))]
        population_rows.append(_repair(row, spec))
    return population_rows


def optimize_config(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    method: str = "genetic",
    iterations: int = 40,
    population: int = 24,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_or_raise(config)
    config = resolve_config(config, config.get("__config_dir"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = config.get("optimize", {}) if isinstance(config.get("optimize"), dict) else {}
    spec = _variable_spec(config)
    objectives = opt.get("objectives", config.get("sweep", {}).get("objectives", DEFAULT_OBJECTIVES))
    constraints = opt.get("constraints", config.get("sweep", {}).get("constraints", {"require_success": False}))
    seed = int(config.get("seed", 2026))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    candidates = _initial_population(spec, population, seed)
    eval_index = 0
    for generation in range(max(1, iterations)):
        evaluated = []
        for candidate in candidates:
            row = _evaluate(config, candidate, eval_index)
            row["generation"] = generation
            row["optimizer_score"] = _score(row, objectives, constraints)
            rows.append(row)
            evaluated.append(row)
            eval_index += 1
        evaluated.sort(key=lambda row: float(row["optimizer_score"]))
        elites = [{name: row[name] for name in spec} for row in evaluated[: max(2, population // 4)]]
        if method == "random":
            candidates = [_sample_candidate(spec, rng) for _ in range(population)]
        elif method in {"anneal", "simulated_annealing"}:
            base = elites[0]
            temp = max(0.02, 1.0 - generation / max(iterations, 1))
            candidates = [_mutate(base, spec, rng, 0.25 * temp) for _ in range(population)]
        elif method in {"coordinate", "pattern", "pattern_search"}:
            base = elites[0]
            candidates = [base]
            for name, rule in spec.items():
                if isinstance(rule, dict) and "values" in rule:
                    for value in rule["values"]:
                        c = dict(base)
                        c[name] = value
                        candidates.append(_repair(c, spec))
                else:
                    bounds = rule.get("bounds", rule) if isinstance(rule, dict) else rule
                    step = (float(bounds[1]) - float(bounds[0])) * max(0.02, 0.25 / (generation + 1))
                    for direction in [-1.0, 1.0]:
                        c = dict(base)
                        c[name] = _as_float(c[name]) + direction * step
                        candidates.append(_repair(c, spec))
            while len(candidates) < population:
                candidates.append(_mutate(base, spec, rng, 0.1))
            candidates = candidates[:population]
        else:
            children = elites.copy()
            while len(children) < population:
                parent_a = elites[int(rng.integers(0, len(elites)))]
                parent_b = elites[int(rng.integers(0, len(elites)))]
                children.append(_mutate(_crossover(parent_a, parent_b, rng), spec, rng, 0.15))
            candidates = children

    ranked = score_designs(rows, objectives=objectives, constraints=constraints)
    front = pareto_front([row for row in ranked if row.get("feasible")], objectives)
    write_csv(out / "optimization_results.csv", rows)
    write_csv(out / "design_ranking.csv", ranked)
    write_csv(out / "pareto.csv", front)
    if rows:
        x = np.array([_as_float(row.get("impulse_n_s")) for row in rows], dtype=float)
        y = np.array([_as_float(row.get("miss_distance_m")) for row in rows], dtype=float)
        (out / "optimization_trade.svg").write_text(svg_scatter(x, y, "Optimization Trade", "impulse_n_s", "miss_distance_m"))
    summary = {
        "method": method,
        "iterations": iterations,
        "population": population,
        "evaluations": len(rows),
        "best_design": ranked[0] if ranked else {},
        "pareto_rows": len(front),
    }
    write_json(out / "optimization_summary.json", summary)
    return summary
