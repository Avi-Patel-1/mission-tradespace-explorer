# Uncertainty Quantification

The `uq-report` command writes uncertainty estimates for completed studies.

```bash
python3 -m tradespace uq-report --study outputs/demo_expanded --metric miss_distance_m
```

Outputs:

- `uq_summary.json`: bootstrap mean/median intervals, jackknife estimate, confidence-based stopping rule status, success probability confidence bounds, and rare-event probability.
- `miss_distance_m_ecdf.csv`: empirical CDF table.
- `miss_distance_m_quantile_convergence.csv`: quantile convergence across sample count.
- `distribution_diagnostics.csv`: sample distribution diagnostics such as mean, standard deviation, skewness, kurtosis, and unique fraction.

These artifacts support convergence checks, uncertainty communication, and repeatable campaign acceptance criteria.
