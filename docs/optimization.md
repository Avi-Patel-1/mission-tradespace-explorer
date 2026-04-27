# Optimization

The optimizer evaluates mission models directly from a config.

```bash
python3 -m tradespace optimize --config examples/configs/guidance_gain_optimization.json --out outputs/optimization_demo --method genetic --iterations 20 --population 16
```

Supported methods:

- `genetic`: elite selection, crossover, mutation, and constraint repair.
- `random`: random design exploration across bounds.
- `anneal` or `simulated_annealing`: mutation scale decreases with generation.
- `coordinate`, `pattern`, or `pattern_search`: local coordinate perturbations around the best design.

Variables can be continuous bounds or discrete values:

```json
"optimize": {
  "variables": {
    "guidance_gain": { "bounds": [1.4, 3.1] },
    "thrust_n": { "bounds": [1900.0, 2500.0] },
    "mode": { "values": [1, 2, 3] }
  }
}
```

Outputs:

- `optimization_results.csv`: every evaluated design.
- `design_ranking.csv`: Pareto-aware feasibility and percentile ranking.
- `pareto.csv`: non-dominated feasible designs.
- `optimization_summary.json`: best design, evaluation count, and method metadata.
- `optimization_trade.svg`: static objective scatter plot.
