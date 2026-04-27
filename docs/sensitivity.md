# Sensitivity

The standard `sensitivity.csv` is correlation-focused. The expanded sensitivity workflow adds broader diagnostics:

```bash
python3 -m tradespace sensitivity-deep --study outputs/demo_expanded
```

Outputs:

- `sensitivity_deep.csv`
- `sensitivity_deep_summary.json`

Columns include:

- `permutation_importance`: drop in single-parameter linear fit score after shuffling.
- `mutual_information_proxy`: binned nonlinear dependence proxy.
- `sobol_first_order_proxy`: binned conditional-mean variance proxy.
- `success_failure_median_delta`: parameter median shift between successful and failed runs.
- `elasticity`: local relative metric sensitivity.
- `stability_std`: rank-correlation stability across growing sample counts.

Use this report to separate stable drivers from noisy correlations and to identify interactions or nonlinear effects worth a deeper study.
