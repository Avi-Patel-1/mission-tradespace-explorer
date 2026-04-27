from __future__ import annotations

import math
from typing import Any

import numpy as np

from .distributions import _normal_ppf
from .reports import read_csv, write_csv, write_json


def _as_bool(value: Any) -> bool:
    return value in {True, "True", "true", 1, "1", "yes"}


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(math.nan)
    return np.array(values, dtype=float)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = abs(_normal_ppf(0.5 + confidence / 2.0))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: np.ndarray, statistic: str = "mean", confidence: float = 0.95, resamples: int = 400, seed: int = 0) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": math.nan, "low": math.nan, "high": math.nan}
    rng = np.random.default_rng(seed)
    if statistic == "median":
        fn = np.median
    elif statistic.startswith("q"):
        quantile = float(statistic[1:]) / 100.0
        fn = lambda x: np.quantile(x, quantile)
    else:
        fn = np.mean
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        draws[i] = float(fn(rng.choice(values, size=len(values), replace=True)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(fn(values)),
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
    }


def jackknife_estimate(values: np.ndarray, statistic: str = "mean") -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return {"estimate": float(values[0]) if n else math.nan, "bias": math.nan, "std_error": math.nan}
    fn = np.median if statistic == "median" else np.mean
    estimate = float(fn(values))
    leave_one = np.array([float(fn(np.delete(values, i))) for i in range(n)], dtype=float)
    mean_leave_one = float(np.mean(leave_one))
    bias = (n - 1) * (mean_leave_one - estimate)
    std_error = math.sqrt((n - 1) * float(np.mean((leave_one - mean_leave_one) ** 2)))
    return {"estimate": estimate, "bias": float(bias), "std_error": float(std_error)}


def empirical_cdf_rows(values: np.ndarray, *, points: int = 101) -> list[dict[str, float]]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    xs = np.quantile(values, np.linspace(0.0, 1.0, points))
    sorted_values = np.sort(values)
    rows = []
    for x in xs:
        rows.append({"value": float(x), "cdf": float(np.searchsorted(sorted_values, x, side="right") / len(sorted_values))})
    return rows


def quantile_convergence(values: np.ndarray, quantiles: list[float] | None = None, checkpoints: int = 12) -> list[dict[str, float]]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    quantiles = quantiles or [0.05, 0.5, 0.95]
    counts = np.unique(np.clip(np.geomspace(5, len(values), min(checkpoints, len(values))).astype(int), 1, len(values)))
    rows = []
    for count in counts:
        subset = values[:count]
        row = {"runs": int(count)}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = float(np.quantile(subset, q))
        rows.append(row)
    return rows


def stopping_rule_status(values: np.ndarray, *, metric: str, confidence: float = 0.95, tolerance: float = 0.05, statistic: str = "mean") -> dict[str, Any]:
    ci = bootstrap_ci(values, statistic=statistic, confidence=confidence)
    estimate = abs(float(ci["estimate"]))
    width = float(ci["high"]) - float(ci["low"])
    relative_width = width / max(estimate, 1e-9)
    return {
        "metric": metric,
        "statistic": statistic,
        "confidence": confidence,
        "tolerance": tolerance,
        "estimate": ci["estimate"],
        "low": ci["low"],
        "high": ci["high"],
        "relative_width": relative_width,
        "stop": bool(relative_width <= tolerance),
    }


def rare_event_probability(results: list[dict[str, Any]], *, event: str = "failed", confidence: float = 0.95) -> dict[str, Any]:
    if event == "failed":
        hits = sum(1 for row in results if not _as_bool(row.get("success")))
    elif event.startswith("failure_reason="):
        reason = event.split("=", 1)[1]
        hits = sum(1 for row in results if str(row.get("failure_reason", "")) == reason)
    else:
        hits = sum(1 for row in results if _as_bool(row.get(event)))
    low, high = wilson_interval(hits, len(results), confidence)
    p = hits / max(len(results), 1)
    return {"event": event, "runs": len(results), "hits": hits, "probability": p, "low": low, "high": high}


def distribution_diagnostics(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    names = sorted({key for row in samples for key, value in row.items() if isinstance(value, (int, float)) and key not in {"run", "run_seed", "case", "replicate"}})
    for name in names:
        values = _series(samples, name)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values))
        centered = values - mean
        skew = float(np.mean(centered**3) / max(std**3, 1e-12))
        kurtosis = float(np.mean(centered**4) / max(std**4, 1e-12) - 3.0)
        rows.append(
            {
                "parameter": name,
                "n": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": mean,
                "std": std,
                "skew": skew,
                "excess_kurtosis": kurtosis,
                "unique_fraction": float(len(set(values.tolist())) / len(values)),
            }
        )
    return rows


def write_uq_artifacts(study_dir: str, *, metric: str = "miss_distance_m") -> dict[str, Any]:
    from pathlib import Path

    study = Path(study_dir)
    results = read_csv(study / "results.csv")
    samples = read_csv(study / "samples.csv")
    values = _series(results, metric)
    success_count = sum(1 for row in results if _as_bool(row.get("success")))
    success_ci = wilson_interval(success_count, len(results))
    payload = {
        "metric": metric,
        "bootstrap_mean": bootstrap_ci(values, "mean"),
        "bootstrap_median": bootstrap_ci(values, "median"),
        "jackknife_mean": jackknife_estimate(values, "mean"),
        "stopping_rule": stopping_rule_status(values, metric=metric),
        "success_probability": {
            "estimate": success_count / max(len(results), 1),
            "low": success_ci[0],
            "high": success_ci[1],
        },
        "rare_event_failed": rare_event_probability(results, event="failed"),
    }
    write_json(study / "uq_summary.json", payload)
    write_csv(study / f"{metric}_ecdf.csv", empirical_cdf_rows(values))
    write_csv(study / f"{metric}_quantile_convergence.csv", quantile_convergence(values))
    write_csv(study / "distribution_diagnostics.csv", distribution_diagnostics(samples))
    return payload
