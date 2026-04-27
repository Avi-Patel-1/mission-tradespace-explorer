from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

import numpy as np


EXCLUDED_SAMPLE_COLUMNS = {"run", "case", "replicate", "run_seed", "seed"}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not math.isfinite(value) else value


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _numeric_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in samples:
        for key, value in row.items():
            if key in EXCLUDED_SAMPLE_COLUMNS:
                continue
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                names.add(key)
    return sorted(names)


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, math.nan)))
        except (TypeError, ValueError):
            out.append(math.nan)
    return np.array(out, dtype=float)


def _partial_rank(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    if len(controls) == 0 or len(x) < 8:
        return _corr(_rank(x), _rank(y))
    ranked_x = _rank(x)
    ranked_y = _rank(y)
    control_matrix = np.column_stack([_rank(c) for c in controls[: min(len(controls), 5)]])
    control_matrix = np.column_stack([np.ones(len(x)), control_matrix])
    try:
        beta_x, *_ = np.linalg.lstsq(control_matrix, ranked_x, rcond=None)
        beta_y, *_ = np.linalg.lstsq(control_matrix, ranked_y, rcond=None)
    except np.linalg.LinAlgError:
        return _corr(ranked_x, ranked_y)
    return _corr(ranked_x - control_matrix @ beta_x, ranked_y - control_matrix @ beta_y)


def sensitivity_table(samples: list[dict[str, Any]], results: list[dict[str, Any]], metric: str = "miss_distance_m") -> list[dict[str, float | str]]:
    if not samples or not results:
        return []

    names = _numeric_names(samples)
    y = _series(results, metric)
    success = np.array([1.0 if r.get("success") in {True, "True", "true", 1, "1"} else 0.0 for r in results], dtype=float)
    reason_counts = Counter(str(r.get("failure_reason", "none")) for r in results)
    reasons = [reason for reason, _ in reason_counts.most_common(5) if reason != "none"]
    x_by_name = {name: _series(samples, name) for name in names}

    rows: list[dict[str, float | str]] = []
    for name in names:
        x = x_by_name[name]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            continue
        x_m = x[mask]
        y_m = y[mask]
        pearson = _corr(x_m, y_m)
        rank_corr = _corr(_rank(x_m), _rank(y_m))
        controls = [
            values[mask]
            for other, values in x_by_name.items()
            if other != name and np.isfinite(values[mask]).all() and float(np.std(values[mask])) > 1e-12
        ]
        partial = _partial_rank(x_m, y_m, controls)
        success_corr = _corr(x_m, success[mask])
        low_cut, high_cut = np.percentile(x_m, [25, 75])
        low_metric = float(np.median(y_m[x_m <= low_cut])) if np.any(x_m <= low_cut) else float(np.median(y_m))
        high_metric = float(np.median(y_m[x_m >= high_cut])) if np.any(x_m >= high_cut) else float(np.median(y_m))
        strongest_reason = "none"
        strongest_reason_corr = 0.0
        for reason in reasons:
            indicator = np.array([1.0 if str(r.get("failure_reason", "none")) == reason else 0.0 for r in results], dtype=float)
            reason_corr = _corr(x_m, indicator[mask])
            if abs(reason_corr) > abs(strongest_reason_corr):
                strongest_reason = reason
                strongest_reason_corr = reason_corr
        rows.append(
            {
                "parameter": name,
                "metric": metric,
                "pearson": pearson,
                "rank_correlation": rank_corr,
                "partial_rank_correlation": partial,
                "success_correlation": success_corr,
                "failure_reason": strongest_reason,
                "failure_reason_correlation": strongest_reason_corr,
                "tornado_low": low_metric,
                "tornado_high": high_metric,
                "tornado_delta": high_metric - low_metric,
                "variance_contribution_proxy": pearson * pearson,
                "score": abs(rank_corr) + 0.5 * abs(pearson) + 0.25 * abs(success_corr),
            }
        )

    total_variance_proxy = sum(float(row["variance_contribution_proxy"]) for row in rows) or 1.0
    for row in rows:
        row["variance_contribution_proxy"] = float(row["variance_contribution_proxy"]) / total_variance_proxy
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def finite_difference_sensitivity(
    nominal: dict[str, Any],
    perturbations: dict[str, float],
    simulate: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    metric: str = "miss_distance_m",
) -> list[dict[str, float | str]]:
    base = simulate(dict(nominal))
    base_metric = float(base[metric])
    rows: list[dict[str, float | str]] = []
    for name, step in perturbations.items():
        if step == 0.0:
            continue
        perturbed = dict(nominal)
        perturbed[name] = float(perturbed.get(name, 0.0)) + float(step)
        value = float(simulate(perturbed)[metric])
        rows.append(
            {
                "parameter": name,
                "metric": metric,
                "finite_difference_step": float(step),
                "finite_difference_slope": (value - base_metric) / float(step),
            }
        )
    rows.sort(key=lambda row: abs(float(row["finite_difference_slope"])), reverse=True)
    return rows
