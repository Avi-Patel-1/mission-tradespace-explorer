from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _clip(value: float, spec: dict[str, Any]) -> float:
    if "low" in spec:
        value = max(float(spec["low"]), value)
    if "high" in spec:
        value = min(float(spec["high"]), value)
    return value


def _as_probability(unit: float | None, rng: np.random.Generator) -> float:
    if unit is None:
        return float(rng.random())
    return min(max(float(unit), np.finfo(float).eps), 1.0 - np.finfo(float).eps)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    # Peter J. Acklam's rational approximation, sufficient for sampling diagnostics.
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _triangular_ppf(p: float, low: float, mode: float, high: float) -> float:
    if high <= low:
        return low
    cutoff = (mode - low) / (high - low)
    if p < cutoff:
        return low + math.sqrt(p * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - p) * (high - low) * (high - mode))


@lru_cache(maxsize=64)
def _read_table(path: str, column: str, weight_column: str | None) -> tuple[tuple[Any, ...], tuple[float, ...] | None]:
    values: list[Any] = []
    weights: list[float] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row[column]
            try:
                value: Any = float(raw)
            except (TypeError, ValueError):
                value = raw
            values.append(value)
            if weight_column:
                weights.append(float(row[weight_column]))
    if not values:
        raise ValueError(f"table distribution has no rows: {path}")
    return tuple(values), tuple(weights) if weight_column else None


def _normalize_weights(weights: Any, count: int) -> np.ndarray | None:
    if weights is None:
        return None
    probs = np.array(weights, dtype=float)
    if len(probs) != count:
        raise ValueError("probability count does not match value count")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return probs / total


def _resolve_table_path(spec: dict[str, Any], root: str | Path | None) -> Path:
    path = Path(str(spec["path"])).expanduser()
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return path.resolve()


def sample_distribution(
    spec: dict[str, Any],
    rng: np.random.Generator,
    *,
    run_index: int = 0,
    root: str | Path | None = None,
    unit: float | None = None,
) -> Any:
    kind = str(spec.get("type", "constant"))
    p = _as_probability(unit, rng)

    if kind == "constant":
        return spec.get("value")

    if kind == "uniform":
        low, high = float(spec["low"]), float(spec["high"])
        return float(low + p * (high - low))

    if kind == "normal":
        value = float(spec["mean"]) + float(spec["std"]) * _normal_ppf(p)
        return _clip(value, spec)

    if kind == "triangular":
        return float(_triangular_ppf(p, float(spec["low"]), float(spec["mode"]), float(spec["high"])))

    if kind == "lognormal":
        value = math.exp(float(spec["mean"]) + float(spec["sigma"]) * _normal_ppf(p))
        if "scale" in spec:
            value *= float(spec["scale"])
        return _clip(float(value), spec)

    if kind == "truncated_normal":
        mean, std = float(spec["mean"]), float(spec["std"])
        low, high = float(spec["low"]), float(spec["high"])
        lo_cdf = _normal_cdf((low - mean) / std)
        hi_cdf = _normal_cdf((high - mean) / std)
        bounded_p = lo_cdf + p * max(hi_cdf - lo_cdf, 1e-12)
        return float(min(max(mean + std * _normal_ppf(bounded_p), low), high))

    if kind in {"beta", "bounded_beta"}:
        value = float(rng.beta(float(spec["alpha"]), float(spec["beta"])))
        low, high = float(spec["low"]), float(spec["high"])
        return float(low + value * (high - low))

    if kind in {"choice", "discrete"}:
        values = list(spec["values"])
        probs = _normalize_weights(spec.get("probabilities"), len(values))
        if probs is None:
            return values[min(int(p * len(values)), len(values) - 1)]
        cdf = np.cumsum(probs)
        return values[int(np.searchsorted(cdf, p, side="right"))]

    if kind == "bernoulli":
        hit = p < float(spec["p"])
        if "true_value" in spec or "false_value" in spec:
            return spec.get("true_value", True) if hit else spec.get("false_value", False)
        return bool(hit)

    if kind == "correlated_normal":
        mean = np.array(spec["mean"], dtype=float)
        cov = np.array(spec["cov"], dtype=float)
        names = list(spec["names"])
        values = rng.multivariate_normal(mean, cov)
        lows = spec.get("low")
        highs = spec.get("high")
        if lows is not None:
            values = np.maximum(values, np.array(lows, dtype=float))
        if highs is not None:
            values = np.minimum(values, np.array(highs, dtype=float))
        return {name: float(value) for name, value in zip(names, values)}

    if kind == "mixture":
        components = list(spec["components"])
        weights = [float(c.get("weight", 1.0)) if isinstance(c, dict) else 1.0 for c in components]
        probs = _normalize_weights(weights, len(components))
        index = int(rng.choice(len(components), p=probs))
        component = components[index]
        component_spec = component.get("distribution", component) if isinstance(component, dict) else component
        return sample_distribution(component_spec, rng, run_index=run_index, root=root)

    if kind in {"empirical", "table"}:
        path = _resolve_table_path(spec, root)
        values, weights = _read_table(str(path), str(spec["column"]), spec.get("weight_column"))
        probs = _normalize_weights(weights, len(values)) if weights is not None else None
        index = int(rng.choice(len(values), p=probs))
        return values[index]

    if kind in {"sweep", "sequence"}:
        values = list(spec["values"])
        if not values:
            raise ValueError("sweep distribution requires at least one value")
        return values[run_index % len(values)]

    raise ValueError(f"unsupported distribution type: {kind}")


def sample_parameters(
    nominal: dict[str, Any],
    uncertainties: dict[str, Any],
    rng: np.random.Generator,
    *,
    run_index: int = 0,
    root: str | Path | None = None,
    units: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(nominal)
    sample_log: dict[str, Any] = {}
    for name, spec in uncertainties.items():
        value = sample_distribution(spec, rng, run_index=run_index, root=root, unit=(units or {}).get(name))
        if isinstance(value, dict):
            params.update(value)
            sample_log.update(value)
        else:
            params[name] = value
            sample_log[name] = value
    return params, sample_log
