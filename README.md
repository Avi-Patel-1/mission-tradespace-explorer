# Monte Carlo Trade-Space Explorer

A dependency-light mission-analysis toolkit for Monte Carlo performance studies, robustness checks, and design trade exploration. It is intended for early guidance, navigation, vehicle, and mission concept studies where teams need deterministic runs, traceable assumptions, and reviewable static outputs.

## Capabilities

- JSON configuration with defaults, inheritance, named mission blocks, and path-based validation errors
- Monte Carlo, Latin-hypercube, stratified, low-discrepancy, grid, random, nested, and simple evolutionary study modes
- Sobol, Halton, Latinized Sobol, orthogonal-array, Morris, and factorial design utilities
- Distribution library covering constant, uniform, clipped normal, triangular, lognormal, truncated normal, bounded beta, discrete choice, Bernoulli, correlated normal, mixtures, table samples, and deterministic sequences
- Mission model registry with point-mass intercept, ballistic intercept, rocket ascent, UAV endurance, communication link budget, and rover energy screening models
- Mission metrics for miss distance, crossrange/downrange error, terminal state, time of flight, qbar, load proxy, impulse, control effort, fuel proxy, robustness margin, and constraint violations
- Sensitivity analysis with Pearson, rank, partial-rank, success-probability, variance-proxy, failure-reason, and tornado-table outputs
- Deep sensitivity reports with permutation importance, mutual-information proxy, binned variance proxy, conditional deltas, elasticity, and stability metrics
- Reliability reports with success probability confidence bounds, reliability index, FORM/SORM-like approximations, margin decomposition, failure ranking, and failure clustering
- Polynomial and radial-basis surrogate models with cross-validation and JSON export
- Genetic, random, annealing, coordinate, and pattern-search optimization workflows
- Manifest-driven scenario campaigns with continuation, retry, replay, and comparison artifacts
- Trade-space analysis with constraint filtering, Pareto fronts, top design ranking, dominance counts, robustness scoring, and percentile scoring
- Optional SQLite storage using the Python standard library, with deterministic replay and export to CSV/JSON
- Static reports with CSV, JSON, HTML, and SVG plots

## Quick Start

```bash
python3 -m unittest discover -s tests
python3 -m tradespace validate --config examples/configs/mission_mc.json
python3 -m tradespace inspect-config --config examples/configs/mission_mc.json
python3 -m tradespace run --config examples/configs/mission_mc.json --out outputs/demo_expanded --runs 1000
python3 -m tradespace sweep --config examples/configs/guidance_gain_sweep.json --out outputs/sweep_expanded
python3 -m tradespace pareto --input outputs/sweep_expanded/sweep_results.csv --out outputs/pareto_expanded
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --out outputs/campaign_resume_example
python3 -m tradespace report --study outputs/demo_expanded
python3 -m tradespace audit --study outputs/demo_expanded
```

## Configuration Shape

Configs are JSON files with named blocks:

```json
{
  "seed": 2026,
  "runs": 1000,
  "sampler": { "method": "latin_hypercube" },
  "mission": { "target_x_m": 3000.0, "success_miss_m": 230.0 },
  "vehicle": { "mass_kg": 42.0, "thrust_n": 1560.0 },
  "environment": { "wind_x_mps": 0.0 },
  "guidance": { "guidance_gain": 2.25 },
  "sensors": { "sensor_noise_m": 0.0 },
  "uncertainty": {
    "distributions": {
      "wind_x_mps": { "type": "normal", "mean": 0.0, "std": 6.0, "low": -18.0, "high": 18.0 }
    }
  }
}
```

Legacy `nominal` and `uncertainties` blocks are still accepted. Named blocks are flattened into the mission parameter set, then uncertainty samples override those nominal values per run.

## Commands

- `validate`: check config structure and distribution definitions
- `inspect-config`: print the resolved config after defaults and inheritance
- `init-config`: write a starter Monte Carlo or sweep config
- `list-models`: list mission models registered in the simulation library
- `run`: execute a Monte Carlo campaign
- `sweep`: evaluate a trade-space study
- `robustness`: run a robustness campaign with the same execution engine as `run`
- `pareto`: rank a CSV of design results and write Pareto outputs
- `report`: rebuild the static report for a study directory
- `audit`: check that study outputs are present and internally consistent
- `reliability`: write reliability and robustness diagnostics for an existing study
- `stress-test`: generate and run uncertainty corner stress cases from a config
- `margin-report`: write margin decomposition artifacts
- `uq-report`: write bootstrap, jackknife, ECDF, and quantile-convergence artifacts
- `sensitivity-deep`: write expanded sensitivity diagnostics
- `fit-surrogate`: train a surrogate from a completed study
- `surrogate-predict`: predict from a surrogate JSON model
- `optimize`: run direct mission-model design optimization
- `rank-designs`: rank design CSV rows
- `robust-pareto`: compute a robust Pareto ranking from aggregate design rows
- `campaign`: plan or run a manifest of named scenarios with continuation and retry behavior
- `replay-run`: reproduce one deterministic Monte Carlo run from a campaign scenario
- `compare-campaigns`: compare two campaign summary directories
- `config-diff`, `freeze-config`, `generate-template`: config management utilities
- single-study continuation from the SQLite run table
- `export`: export a study SQLite database to CSV and JSON

Useful CLI overrides:

```bash
python3 -m tradespace init-config --kind sweep --name thrust_gain_check --out examples/configs/new_sweep.json
python3 -m tradespace run --config examples/configs/mission_mc.json --out outputs/check --runs 100 --seed 12 --sampler stratified --no-sqlite
python3 -m tradespace sweep --config examples/configs/guidance_gain_sweep.json --out outputs/check_sweep --uncertainty-runs 3
python3 -m tradespace pareto --input outputs/check_sweep/sweep_results.csv --out outputs/check_pareto --objective miss_distance_m:min --objective impulse_n_s:min --constraint "miss_distance_m<=260"
python3 -m tradespace reliability --study outputs/demo_expanded
python3 -m tradespace uq-report --study outputs/demo_expanded
python3 -m tradespace sensitivity-deep --study outputs/demo_expanded
python3 -m tradespace fit-surrogate --study outputs/demo_expanded --out outputs/demo_expanded/miss_surrogate.json
python3 -m tradespace optimize --config examples/configs/guidance_gain_optimization.json --out outputs/optimization_demo
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --out outputs/campaign_resume_example --jobs 2
python3 -m tradespace replay-run --campaign outputs/campaign_resume_example --scenario baseline_latin --run-index 0
```

## Output Artifacts

- `samples.csv`: sampled inputs and run seeds
- `results.csv`: mission metrics for each run
- `sensitivity.csv`: influence metrics and tornado-chart data
- `summary.json`: aggregate statistics, failure reasons, and top sensitivities
- `manifest.json`: reproducibility metadata and per-run status
- `study.sqlite`: optional run-level storage with inputs, outputs, status, seed, and config hash
- `report.html`: static report with embedded SVG charts
- `pareto.csv` and `design_comparison.csv`: trade-space selection products when applicable
- `campaign_summary.csv` and `campaign_status.json`: scenario campaign tracking artifacts

## Examples

The `examples/configs` directory includes baseline mission Monte Carlo, guidance gain sweep, thrust/mass trade, high-wind robustness, sensor-bias sensitivity, qbar-constrained design, Pareto design exploration, failure-mode stress test, empirical wind-table, UAV endurance, rocket ascent, communication link budget, rover energy, and reliability-report studies.

The `examples/campaigns` directory includes a resumable campaign manifest that combines Monte Carlo, sweep, and optimization-style scenario execution.

## Methodology Summary

The model is a screening tool, not a high-fidelity flight dynamics solver. It uses a point-mass propagation loop with configurable timestep, thrust, drag, density, wind, guidance steering, and sensor perturbations. The analysis layer emphasizes repeatability, transparent assumptions, and comparable outputs across many mission concepts.
