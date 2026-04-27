# Distributions

Distribution specs live under `uncertainty.distributions`. Keys are parameter names; values define sample behavior.

Supported types:

- `constant`: fixed value.
- `uniform`: `low`, `high`.
- `normal`: `mean`, `std`, optional `low`, `high`.
- `triangular`: `low`, `mode`, `high`.
- `lognormal`: `mean`, `sigma`, optional bounds.
- `truncated_normal`: `mean`, `std`, `low`, `high`.
- `beta` or `bounded_beta`: `alpha`, `beta`, `low`, `high`.
- `choice` or `discrete`: `values`, optional `probabilities`.
- `bernoulli`: event probability `p`.
- `correlated_normal`: `names`, `mean`, `cov`.
- `mixture`: weighted component distributions.
- `table` or `empirical`: CSV `path`, `column`, optional `weight_column`.
- `sweep` or `sequence`: deterministic values cycled by run index.

All random draws use NumPy generators created from deterministic per-run seeds.
