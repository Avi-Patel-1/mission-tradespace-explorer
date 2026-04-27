from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .reports import read_csv, write_csv, write_json


def _as_bool(value: Any) -> bool:
    return value in {True, "True", "true", 1, "1", "yes"}


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, math.nan)))
        except (TypeError, ValueError):
            values.append(math.nan)
    return np.array(values, dtype=float)


def _numeric_names(samples: list[dict[str, Any]]) -> list[str]:
    excluded = {"run", "run_seed", "case", "replicate"}
    return sorted({key for row in samples for key, value in row.items() if key not in excluded and isinstance(value, (int, float))})


def _r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return 0.0
    ss_res = float(np.sum((y[mask] - pred[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _mutual_information_proxy(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < bins or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / max(float(np.sum(hist)), 1e-12)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask_nonzero = pxy > 0
    return float(np.sum(pxy[mask_nonzero] * np.log(pxy[mask_nonzero] / np.maximum(expected[mask_nonzero], 1e-12))))


def _binned_variance_proxy(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < bins or float(np.var(y)) <= 1e-12:
        return 0.0
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    means = []
    weights = []
    for low, high in zip(edges[:-1], edges[1:]):
        bucket = (x >= low) & (x <= high)
        if np.any(bucket):
            means.append(float(np.mean(y[bucket])))
            weights.append(float(np.mean(bucket)))
    if not means:
        return 0.0
    return float(np.average((np.array(means) - np.mean(y)) ** 2, weights=np.array(weights)) / max(float(np.var(y)), 1e-12))


def _linear_prediction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    pred = np.full(len(y), np.nan, dtype=float)
    if int(mask.sum()) < 3 or float(np.std(x[mask])) <= 1e-12:
        return pred
    design = np.column_stack([np.ones(int(mask.sum())), x[mask]])
    beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
    pred[mask] = design @ beta
    return pred


def deep_sensitivity(samples: list[dict[str, Any]], results: list[dict[str, Any]], metric: str = "miss_distance_m", seed: int = 0) -> list[dict[str, Any]]:
    names = _numeric_names(samples)
    y = _series(results, metric)
    success = np.array([1.0 if _as_bool(row.get("success")) else 0.0 for row in results], dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for name in names:
        x = _series(samples, name)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 5:
            continue
        pred = _linear_prediction(x, y)
        base_r2 = _r2_score(y, pred)
        x_perm = x.copy()
        rng.shuffle(x_perm)
        perm_r2 = _r2_score(y, _linear_prediction(x_perm, y))
        rank_corr = float(np.corrcoef(_rank(x[mask]), _rank(y[mask]))[0, 1]) if float(np.std(x[mask])) > 1e-12 and float(np.std(y[mask])) > 1e-12 else 0.0
        success_mask = success.astype(bool)
        fail_mask = ~success_mask
        success_median = float(np.median(x[success_mask & np.isfinite(x)])) if np.any(success_mask & np.isfinite(x)) else math.nan
        failure_median = float(np.median(x[fail_mask & np.isfinite(x)])) if np.any(fail_mask & np.isfinite(x)) else math.nan
        elasticity = 0.0
        if abs(float(np.mean(x[mask]))) > 1e-12 and abs(float(np.mean(y[mask]))) > 1e-12:
            cov = float(np.cov(x[mask], y[mask])[0, 1])
            slope = cov / max(float(np.var(x[mask])), 1e-12)
            elasticity = slope * float(np.mean(x[mask])) / float(np.mean(y[mask]))
        stability = []
        checkpoints = np.unique(np.clip(np.geomspace(8, int(mask.sum()), min(8, int(mask.sum()))).astype(int), 2, int(mask.sum())))
        valid_x = x[mask]
        valid_y = y[mask]
        for count in checkpoints:
            if count > 3 and float(np.std(valid_x[:count])) > 1e-12 and float(np.std(valid_y[:count])) > 1e-12:
                stability.append(float(np.corrcoef(_rank(valid_x[:count]), _rank(valid_y[:count]))[0, 1]))
        rows.append(
            {
                "parameter": name,
                "metric": metric,
                "rank_correlation": rank_corr,
                "permutation_importance": max(0.0, base_r2 - perm_r2),
                "mutual_information_proxy": _mutual_information_proxy(x, y),
                "sobol_first_order_proxy": _binned_variance_proxy(x, y),
                "success_median": success_median,
                "failure_median": failure_median,
                "success_failure_median_delta": failure_median - success_median if np.isfinite(success_median) and np.isfinite(failure_median) else math.nan,
                "elasticity": elasticity,
                "stability_std": float(np.std(stability)) if stability else math.nan,
                "score": abs(rank_corr) + _mutual_information_proxy(x, y) + max(0.0, base_r2 - perm_r2),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def write_deep_sensitivity(study_dir: str | Path, *, metric: str = "miss_distance_m") -> dict[str, Any]:
    study = Path(study_dir)
    rows = deep_sensitivity(read_csv(study / "samples.csv"), read_csv(study / "results.csv"), metric=metric)
    write_csv(study / "sensitivity_deep.csv", rows)
    payload = {"metric": metric, "rows": len(rows), "top_driver": rows[0] if rows else {}}
    write_json(study / "sensitivity_deep_summary.json", payload)
    return payload
