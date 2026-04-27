# Pareto and Design Selection

Trade-space studies write design-level tables in addition to run-level results.

## Selection Products

- `design_comparison.csv`: every design with feasibility, dominance rank, robustness score, and percentile score.
- `pareto.csv`: non-dominated feasible designs.
- `pareto_plot.svg`: static objective scatter plot.

## Objectives

Objectives are configured as metric/sense pairs. The default set minimizes miss distance and impulse while maximizing robustness margin.

## Constraints

Constraint dictionaries support `min`, `max`, and `equals`. `require_success` controls whether unsuccessful rows are filtered before ranking.
