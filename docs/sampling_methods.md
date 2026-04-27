# Sampling Methods

The sampler block selects how uncertainty dimensions are covered.

- `monte_carlo`: independent random draws.
- `latin_hypercube`: one stratified draw per dimension and interval.
- `stratified`: repeatable bins across each uncertainty dimension.
- `sobol`, `sobol_like`, or `halton`: low-discrepancy approximation using prime-base van der Corput sequences.
- `sobol`: compact Sobol digital net for moderate dimensions with deterministic digital shift.
- `halton`: prime-base low-discrepancy sequence.
- `latinized_sobol`: Sobol coverage remapped into Latin-hypercube strata.
- `orthogonal_array`: balanced pairwise coverage for screening studies.
- Morris trajectories and factorial designs are available as library functions for screening and design of experiments.
- `grid`: Cartesian product over explicit values for sweep studies.
- `nested`: grid over design variables with random uncertainty runs inside each design.
- `random`: uniform random design exploration across bounds.
- `evolutionary`: random population with a small mutation pass for broad design search.

Every run records its seed in `samples.csv`, `manifest.json`, and optional SQLite storage.
