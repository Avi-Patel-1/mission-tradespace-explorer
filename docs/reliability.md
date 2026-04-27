# Reliability

Reliability workflows operate on completed study directories.

```bash
python3 -m tradespace reliability --study outputs/demo_expanded
python3 -m tradespace stress-test --config examples/configs/mission_mc.json --out outputs/stress_demo
python3 -m tradespace margin-report --study outputs/demo_expanded
```

Outputs include:

- `reliability_summary.json`: success probability, Wilson confidence bounds, reliability index, rare-event probability, FORM-like and SORM-like approximations.
- `constraint_violations.csv`: violation probabilities and confidence bounds.
- `failure_mode_ranking.csv`: failure reason frequencies.
- `failure_mode_clusters.csv`: simple clustering of failed samples.
- `margin_decomposition.csv`: uncertainty-source contribution to robustness margin shifts.
- `robustness_margin_histogram.svg`: margin distribution.
- `stress_results.csv`: corner stress-case metrics.

The FORM-like method fits a linearized limit-state surface in standardized sampled coordinates. The SORM-like value applies a curvature proxy based on margin skewness. These are screening diagnostics, not substitutes for high-fidelity certification analysis.
