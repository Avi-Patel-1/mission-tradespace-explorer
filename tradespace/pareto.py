from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .reports import read_csv, svg_scatter, write_csv


DEFAULT_OBJECTIVES = [
    {"metric": "miss_distance_m", "sense": "min"},
    {"metric": "impulse_n_s", "sense": "min"},
    {"metric": "robustness_margin", "sense": "max"},
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return value in {True, "True", "true", 1, "1", "yes"}


def is_feasible(row: dict[str, Any], constraints: dict[str, Any] | None = None) -> bool:
    if constraints is None:
        constraints = {}
    if constraints.get("require_success", True) and "success" in row and not _is_true(row["success"]):
        return False
    for metric, rule in constraints.items():
        if metric == "require_success":
            continue
        if not isinstance(rule, dict):
            continue
        value = row.get(metric)
        if "max" in rule and _as_float(value, float("inf")) > float(rule["max"]):
            return False
        if "min" in rule and _as_float(value, -float("inf")) < float(rule["min"]):
            return False
        if "equals" in rule and value != rule["equals"]:
            return False
    return True


def _objective_value(row: dict[str, Any], objective: dict[str, str]) -> float:
    value = _as_float(row.get(objective["metric"]), 0.0)
    return value if objective.get("sense", "min") == "min" else -value


def dominates(a: dict[str, Any], b: dict[str, Any], objectives: list[dict[str, str]]) -> bool:
    a_values = [_objective_value(a, objective) for objective in objectives]
    b_values = [_objective_value(b, objective) for objective in objectives]
    return all(av <= bv for av, bv in zip(a_values, b_values)) and any(av < bv for av, bv in zip(a_values, b_values))


def pareto_front(rows: list[dict[str, Any]], objectives: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    objectives = objectives or DEFAULT_OBJECTIVES
    front = []
    for i, row in enumerate(rows):
        if any(dominates(other, row, objectives) for j, other in enumerate(rows) if i != j):
            continue
        front.append(dict(row, pareto=True))
    return front


def dominance_rank(rows: list[dict[str, Any]], objectives: list[dict[str, str]] | None = None) -> list[int]:
    objectives = objectives or DEFAULT_OBJECTIVES
    ranks = []
    for i, row in enumerate(rows):
        ranks.append(sum(1 for j, other in enumerate(rows) if i != j and dominates(other, row, objectives)))
    return ranks


def score_designs(
    rows: list[dict[str, Any]],
    *,
    objectives: list[dict[str, str]] | None = None,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    objectives = objectives or DEFAULT_OBJECTIVES
    ranked = [dict(row) for row in rows]
    ranks = dominance_rank(ranked, objectives)
    for row, rank in zip(ranked, ranks):
        row["feasible"] = is_feasible(row, constraints)
        row["dominance_rank"] = rank
        row["robustness_score"] = _as_float(row.get("robustness_margin"), 0.0) - 0.02 * _as_float(row.get("miss_distance_m"), 0.0)

    for objective in objectives:
        metric = objective["metric"]
        values = np.array([_as_float(row.get(metric), 0.0) for row in ranked], dtype=float)
        if len(values) == 0 or float(np.ptp(values)) <= 1e-12:
            percentiles = np.full(len(values), 0.5)
        else:
            order = np.argsort(values)
            if objective.get("sense", "min") == "max":
                order = order[::-1]
            percentiles = np.empty(len(values), dtype=float)
            percentiles[order] = np.linspace(1.0, 0.0, len(values))
        for row, percentile in zip(ranked, percentiles):
            row[f"{metric}_percentile_score"] = float(percentile)

    for row in ranked:
        objective_scores = [float(row.get(f"{objective['metric']}_percentile_score", 0.0)) for objective in objectives]
        row["percentile_score"] = float(np.mean(objective_scores)) if objective_scores else 0.0
    ranked.sort(key=lambda row: (not bool(row["feasible"]), int(row["dominance_rank"]), -float(row["percentile_score"])))
    return ranked


def write_pareto(
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    objectives: list[dict[str, str]] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(input_csv))
    ranked = score_designs(rows, objectives=objectives, constraints=constraints)
    feasible = [row for row in ranked if row.get("feasible")]
    front = pareto_front(feasible, objectives or DEFAULT_OBJECTIVES)
    write_csv(out / "design_ranking.csv", ranked)
    write_csv(out / "pareto.csv", front)
    if rows:
        x = np.array([_as_float(row.get("impulse_n_s"), 0.0) for row in rows], dtype=float)
        y = np.array([_as_float(row.get("miss_distance_m"), 0.0) for row in rows], dtype=float)
        (out / "pareto_plot.svg").write_text(svg_scatter(x, y, "Design Trade: Impulse vs Miss", "impulse_n_s", "miss_distance_m"))
    summary = {
        "input_rows": len(rows),
        "feasible_rows": len(feasible),
        "pareto_rows": len(front),
        "best_design": ranked[0] if ranked else {},
    }
    (out / "pareto_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
