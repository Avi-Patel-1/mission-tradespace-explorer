# Sensitivity Methods

The default sensitivity table ranks numeric sampled inputs against `miss_distance_m`.

Columns include:

- `pearson`: linear correlation with the metric.
- `rank_correlation`: correlation of ordinal ranks.
- `partial_rank_correlation`: rank correlation after removing up to five other sampled dimensions.
- `success_correlation`: correlation with the success indicator.
- `variance_contribution_proxy`: normalized squared-correlation proxy.
- `failure_reason` and `failure_reason_correlation`: strongest listed failure mode association.
- `tornado_low`, `tornado_high`, `tornado_delta`: median metric shift between low and high parameter quartiles.

Finite-difference helpers are available for one-at-a-time local checks when a nominal case and perturbation sizes are supplied.
