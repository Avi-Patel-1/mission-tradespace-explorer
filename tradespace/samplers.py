from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .distributions import sample_parameters


SamplePlan = list[tuple[int, int, dict[str, Any], dict[str, Any]]]


SOBOL_POLYNOMIALS: dict[int, tuple[int, int, list[int]]] = {
    1: (1, 0, [1]),
    2: (2, 1, [1, 3]),
    3: (3, 1, [1, 3, 1]),
    4: (3, 2, [1, 1, 3]),
    5: (4, 1, [1, 3, 5, 13]),
    6: (4, 4, [1, 1, 5, 5]),
    7: (5, 2, [1, 3, 3, 9, 7]),
    8: (5, 4, [1, 1, 7, 11, 19]),
    9: (5, 7, [1, 3, 5, 1, 27]),
    10: (5, 11, [1, 1, 5, 13, 3]),
}


def run_seed(base_seed: int, index: int) -> int:
    return int((int(base_seed) + 7919 * int(index) + 104729) % (2**32 - 1))


def _primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denom = 1.0
    i = index
    while i:
        i, remainder = divmod(i, base)
        denom *= base
        result += remainder / denom
    return result


def halton(n: int, dim: int, *, skip: int = 1) -> np.ndarray:
    bases = _primes(dim)
    data = np.zeros((n, dim), dtype=float)
    for row in range(n):
        for col, base in enumerate(bases):
            data[row, col] = _van_der_corput(row + skip, base)
    return data


def sobol(n: int, dim: int, *, skip: int = 1, scramble_seed: int | None = None) -> np.ndarray:
    """Generate a deterministic Sobol digital net for moderate dimensions.

    The implementation uses a compact primitive-polynomial table and optional
    digital-shift scrambling. It is intentionally dependency-light and covers
    the small-to-medium dimensional studies this package targets.
    """

    if dim < 1:
        return np.zeros((n, 0), dtype=float)
    if dim > max(SOBOL_POLYNOMIALS) + 1:
        raise ValueError(f"sobol sampler supports up to {max(SOBOL_POLYNOMIALS) + 1} dimensions")
    bits = 32
    direction = np.zeros((dim, bits), dtype=np.uint32)
    direction[0] = np.array([1 << (bits - i - 1) for i in range(bits)], dtype=np.uint32)
    for d in range(1, dim):
        degree, coeff, initials = SOBOL_POLYNOMIALS[d]
        for i in range(degree):
            direction[d, i] = np.uint32(initials[i] << (bits - i - 1))
        for i in range(degree, bits):
            value = direction[d, i - degree] ^ (direction[d, i - degree] >> degree)
            for k in range(1, degree):
                if (coeff >> (degree - 1 - k)) & 1:
                    value ^= direction[d, i - k]
            direction[d, i] = value

    shift = np.zeros(dim, dtype=np.uint32)
    if scramble_seed is not None:
        shift = np.random.default_rng(scramble_seed).integers(0, 2**32 - 1, size=dim, dtype=np.uint32)

    data = np.zeros((n, dim), dtype=float)
    state = np.zeros(dim, dtype=np.uint32)
    previous_index = 0
    for row in range(n + skip):
        if row == 0:
            state[:] = 0
        else:
            changed_bit = (previous_index & -previous_index).bit_length() - 1
            state ^= direction[:, changed_bit]
        previous_index = row + 1
        if row >= skip:
            data[row - skip] = (state ^ shift).astype(np.float64) / 2**32
    return np.clip(data, np.finfo(float).eps, 1.0 - np.finfo(float).eps)


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((n, dim), dtype=float)
    for col in range(dim):
        points = (np.arange(n, dtype=float) + rng.random(n)) / max(n, 1)
        rng.shuffle(points)
        data[:, col] = points
    return data


def latinized_sobol(n: int, dim: int, seed: int) -> np.ndarray:
    base = sobol(n, dim, skip=max(1, seed % 4093), scramble_seed=seed)
    rng = np.random.default_rng(seed)
    out = np.zeros_like(base)
    for col in range(dim):
        order = np.argsort(base[:, col], kind="mergesort")
        strata = (np.arange(n, dtype=float) + rng.random(n)) / max(n, 1)
        out[order, col] = strata
    return out


def stratified(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    strata = max(1, int(math.sqrt(n)))
    data = np.zeros((n, dim), dtype=float)
    base = np.linspace(0.0, 1.0, strata + 1)
    for row in range(n):
        bucket = row % strata
        for col in range(dim):
            low, high = base[bucket], base[bucket + 1]
            data[row, col] = low + rng.random() * (high - low)
    rng.shuffle(data, axis=0)
    return data


def orthogonal_array(n: int, dim: int, seed: int, *, levels: int | None = None) -> np.ndarray:
    if dim <= 0:
        return np.zeros((n, 0), dtype=float)
    levels = levels or max(2, int(math.ceil(math.sqrt(n))))
    base_rows = list(itertools.product(range(levels), repeat=min(dim, 2)))
    rng = np.random.default_rng(seed)
    data = np.zeros((n, dim), dtype=float)
    for row in range(n):
        pair = base_rows[row % len(base_rows)]
        for col in range(dim):
            level = pair[col % len(pair)] if col < 2 else (pair[0] + (col + 1) * pair[-1]) % levels
            data[row, col] = (level + rng.random()) / levels
    rng.shuffle(data, axis=0)
    return data


def morris_trajectories(dim: int, trajectories: int, seed: int, *, levels: int = 4) -> np.ndarray:
    if dim <= 0:
        return np.zeros((0, 0), dtype=float)
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1.0))
    points: list[np.ndarray] = []
    grid = np.linspace(0.0, 1.0 - delta, levels // 2 + 1)
    for _ in range(trajectories):
        x = rng.choice(grid, size=dim).astype(float)
        points.append(x.copy())
        for axis in rng.permutation(dim):
            direction = 1.0 if x[axis] + delta <= 1.0 else -1.0
            x = x.copy()
            x[axis] = np.clip(x[axis] + direction * delta, 0.0, 1.0)
            points.append(x.copy())
    return np.array(points, dtype=float)


def factorial_design(levels_by_dim: list[int], *, fraction: int = 1, seed: int = 0) -> np.ndarray:
    grids = [np.linspace(0.0, 1.0, max(2, int(level))) for level in levels_by_dim]
    full = np.array(list(itertools.product(*grids)), dtype=float)
    if fraction <= 1 or len(full) <= 1:
        return full
    rng = np.random.default_rng(seed)
    keep = np.arange(len(full))
    rng.shuffle(keep)
    keep = np.sort(keep[: max(1, len(full) // fraction)])
    return full[keep]


def unit_matrix(method: str, n: int, dim: int, seed: int) -> np.ndarray | None:
    if dim <= 0:
        return None
    normalized = method.lower().replace("-", "_")
    if normalized in {"monte_carlo", "crude", "crude_monte_carlo", "random"}:
        return None
    if normalized in {"latin", "latin_hypercube", "lhs"}:
        return latin_hypercube(n, dim, seed)
    if normalized == "stratified":
        return stratified(n, dim, seed)
    if normalized == "halton":
        return halton(n, dim, skip=max(1, seed % 997))
    if normalized in {"sobol", "sobol_like", "low_discrepancy"}:
        return sobol(n, dim, skip=max(1, seed % 997), scramble_seed=seed)
    if normalized in {"latinized_sobol", "latin_sobol"}:
        return latinized_sobol(n, dim, seed)
    if normalized in {"orthogonal", "orthogonal_array", "oa"}:
        return orthogonal_array(n, dim, seed)
    return None


def generate_samples(
    nominal: dict[str, Any],
    uncertainties: dict[str, Any],
    n: int,
    seed: int,
    *,
    method: str = "monte_carlo",
    root: str | Path | None = None,
) -> SamplePlan:
    names = list(uncertainties.keys())
    units = unit_matrix(method, n, len(names), seed)
    plan: SamplePlan = []
    for i in range(n):
        rng = np.random.default_rng(run_seed(seed, i))
        unit_row = {name: float(units[i, j]) for j, name in enumerate(names)} if units is not None else None
        params, sample = sample_parameters(
            nominal,
            uncertainties,
            rng,
            run_index=i,
            root=root,
            units=unit_row,
        )
        seed_i = run_seed(seed, i)
        params["run_seed"] = seed_i
        sample["run_seed"] = seed_i
        plan.append((i, seed_i, params, sample))
    return plan


def grid_cases(values: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    names = list(values.keys())
    value_lists = [list(values[name]) for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def bounded_random_cases(bounds: dict[str, list[float]], n: int, seed: int, *, method: str = "random") -> list[dict[str, float]]:
    names = list(bounds.keys())
    units = unit_matrix(method, n, len(names), seed)
    rng = np.random.default_rng(seed)
    if units is None:
        units = rng.random((n, len(names)))
    cases: list[dict[str, float]] = []
    for i in range(n):
        case: dict[str, float] = {}
        for j, name in enumerate(names):
            low, high = float(bounds[name][0]), float(bounds[name][1])
            case[name] = float(low + units[i, j] * (high - low))
        cases.append(case)
    return cases
