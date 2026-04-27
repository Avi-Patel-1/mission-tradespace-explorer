# Architecture

The package is organized around deterministic study execution and flat, reviewable artifacts.

## Core Flow

1. `config.py` loads, inherits, resolves, validates, and hashes JSON configs.
2. `samplers.py` creates deterministic unit-space designs and sampled parameter plans.
3. `distributions.py` maps unit draws or random generators into physical parameter values.
4. `mission_models.py` dispatches to registered screening models.
5. `runner.py` and `sweep.py` execute individual Monte Carlo and design studies.
6. `sensitivity.py`, `deep_sensitivity.py`, `uq.py`, and `reliability.py` analyze completed studies.
7. `optimization.py` and `surrogate.py` handle direct design search and response-model fitting.
8. `campaign.py` expands scenario manifests into resumable study directories.
9. `reports.py` and `storage.py` write static artifacts and SQLite records.
10. `cli.py` exposes the workflows through `python3 -m tradespace`.

## Design Principles

- Deterministic seeds and reproducible artifacts.
- Dependency-light numerical methods using NumPy and the Python standard library.
- Flat CSV/JSON outputs that are easy to inspect, diff, archive, and import.
- Screening-model scope with explicit failure reasons and robustness margins.
