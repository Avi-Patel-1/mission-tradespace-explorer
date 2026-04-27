from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .reports import read_csv, write_csv, write_json


EXCLUDED_COLUMNS = {"run", "run_seed", "case", "replicate", "success", "failed"}


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_features(samples: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in samples:
        for key, value in row.items():
            if key not in EXCLUDED_COLUMNS and isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    return np.array([[_as_float(row.get(name)) for name in features] for row in rows], dtype=float)


def _target(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    return np.array([_as_float(row.get(metric)) for row in rows], dtype=float)


def _clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return x[mask], y[mask]


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale <= 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def _poly_terms(dim: int, degree: int) -> list[tuple[int, ...]]:
    terms: list[tuple[int, ...]] = [()]
    terms.extend((i,) for i in range(dim))
    if degree >= 2:
        for i in range(dim):
            terms.append((i, i))
        for i in range(dim):
            for j in range(i + 1, dim):
                terms.append((i, j))
    if degree >= 3:
        for i in range(dim):
            terms.append((i, i, i))
    return terms


def _poly_design(x: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    design = np.ones((len(x), len(terms)), dtype=float)
    for col, term in enumerate(terms):
        if not term:
            continue
        for idx in term:
            design[:, col] *= x[:, idx]
    return design


def _ridge_solve(design: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(design.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) == 0:
        return {"rmse": math.nan, "mae": math.nan, "r2": math.nan}
    residual = y[mask] - pred[mask]
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    ss_res = float(np.sum(residual**2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot),
    }


def _kfold_indices(n: int, folds: int, seed: int) -> list[np.ndarray]:
    folds = max(2, min(folds, n))
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    return [chunk for chunk in np.array_split(order, folds) if len(chunk)]


def _fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int, ridge: float) -> dict[str, Any]:
    z, mean, scale = _standardize(x)
    terms = _poly_terms(x.shape[1], degree)
    design = _poly_design(z, terms)
    coef = _ridge_solve(design, y, ridge)
    pred = design @ coef
    return {
        "model_type": "polynomial",
        "degree": degree,
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "terms": [list(term) for term in terms],
        "coefficients": coef.tolist(),
        "train_metrics": _metrics(y, pred),
    }


def _predict_polynomial(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - np.array(model["x_mean"], dtype=float)) / np.array(model["x_scale"], dtype=float)
    terms = [tuple(term) for term in model["terms"]]
    return _poly_design(z, terms) @ np.array(model["coefficients"], dtype=float)


def _fit_rbf(x: np.ndarray, y: np.ndarray, centers: int, ridge: float, seed: int) -> dict[str, Any]:
    z, mean, scale = _standardize(x)
    rng = np.random.default_rng(seed)
    count = min(max(2, centers), len(z))
    center_idx = rng.choice(len(z), size=count, replace=False)
    c = z[center_idx]
    distances = np.linalg.norm(z[:, None, :] - c[None, :, :], axis=2)
    length_scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    phi = np.exp(-(distances**2) / max(2.0 * length_scale * length_scale, 1e-12))
    design = np.column_stack([np.ones(len(phi)), phi])
    weights = _ridge_solve(design, y, ridge)
    pred = design @ weights
    return {
        "model_type": "rbf",
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "centers": c.tolist(),
        "length_scale": length_scale,
        "weights": weights.tolist(),
        "train_metrics": _metrics(y, pred),
    }


def _predict_rbf(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - np.array(model["x_mean"], dtype=float)) / np.array(model["x_scale"], dtype=float)
    centers = np.array(model["centers"], dtype=float)
    distances = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    phi = np.exp(-(distances**2) / max(2.0 * float(model["length_scale"]) ** 2, 1e-12))
    design = np.column_stack([np.ones(len(phi)), phi])
    return design @ np.array(model["weights"], dtype=float)


def predict_array(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model["model_type"] == "polynomial":
        return _predict_polynomial(model, x)
    if model["model_type"] == "rbf":
        return _predict_rbf(model, x)
    raise ValueError(f"unsupported surrogate type: {model['model_type']}")


def _cross_validate(x: np.ndarray, y: np.ndarray, model_type: str, degree: int, centers: int, ridge: float, seed: int) -> dict[str, float]:
    preds = np.full(len(y), math.nan, dtype=float)
    for fold, test_idx in enumerate(_kfold_indices(len(y), min(5, len(y)), seed)):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if model_type == "rbf":
            model = _fit_rbf(x[train_idx], y[train_idx], min(centers, len(train_idx)), ridge, seed + fold)
        else:
            model = _fit_polynomial(x[train_idx], y[train_idx], degree, ridge)
        preds[test_idx] = predict_array(model, x[test_idx])
    return _metrics(y, preds)


def fit_surrogate(
    study_dir: str | Path,
    out_path: str | Path,
    *,
    metric: str = "miss_distance_m",
    model_type: str = "polynomial",
    degree: int = 2,
    centers: int = 24,
    ridge: float = 1e-6,
    seed: int = 2026,
) -> dict[str, Any]:
    study = Path(study_dir)
    samples = read_csv(study / "samples.csv")
    results = read_csv(study / "results.csv")
    features = _numeric_features(samples)
    if not features:
        raise ValueError("no numeric sample features available for surrogate fitting")
    x, y = _clean_xy(_matrix(samples, features), _target(results, metric))
    if len(y) < max(5, len(features) + 2):
        raise ValueError("not enough finite rows for surrogate fitting")
    if model_type == "rbf":
        body = _fit_rbf(x, y, centers, ridge, seed)
    elif model_type == "polynomial":
        body = _fit_polynomial(x, y, degree, ridge)
    else:
        raise ValueError("model_type must be polynomial or rbf")
    model = {
        "version": 1,
        "metric": metric,
        "features": features,
        "rows": int(len(y)),
        "study_dir": str(study),
        "cross_validation": _cross_validate(x, y, model_type, degree, centers, ridge, seed),
        **body,
    }
    write_json(Path(out_path), model)
    return model


def predict_rows(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = _matrix(rows, list(model["features"]))
    pred = predict_array(model, x)
    out = []
    for row, value in zip(rows, pred):
        out.append({**row, f"predicted_{model['metric']}": float(value)})
    return out


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def predict_surrogate(
    model_path: str | Path,
    *,
    params: dict[str, Any] | None = None,
    input_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> dict[str, Any]:
    model = load_model(model_path)
    if input_csv is not None:
        rows = read_csv(Path(input_csv))
        predictions = predict_rows(model, rows)
        if out_csv is not None:
            write_csv(Path(out_csv), predictions)
        values = [row[f"predicted_{model['metric']}"] for row in predictions]
        return {"rows": len(predictions), "metric": model["metric"], "min": float(np.min(values)), "max": float(np.max(values))}
    if params is None:
        raise ValueError("provide params or input_csv")
    prediction = predict_rows(model, [params])[0][f"predicted_{model['metric']}"]
    return {"metric": model["metric"], "prediction": float(prediction), "features": model["features"]}
