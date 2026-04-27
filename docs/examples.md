# Examples

Example configs are stored in `examples/configs`.
Campaign manifests are stored in `examples/campaigns`.

- `mission_mc.json`: baseline Monte Carlo mission study.
- `guidance_gain_sweep.json`: gain and thrust grid with uncertainty runs per design.
- `thrust_mass_trade.json`: thrust, mass, and burn-time grid.
- `high_wind_robustness.json`: wind-layer robustness case.
- `sensor_bias_sensitivity.json`: correlated sensor bias and noise study.
- `qbar_constrained_design.json`: design search with tighter qbar and load limits.
- `pareto_design_exploration.json`: random design-space exploration for Pareto selection.
- `failure_mode_stress_test.json`: stressed inputs intended to exercise failure reason logic.
- `empirical_wind_table.json`: weighted wind cases sampled from `examples/data/wind_cases.csv`.
- `reliability_report_example.json`: reliability-focused point-mass study with tighter success and load/qbar margins.
- `uav_endurance_trade.json`: UAV battery, power, speed, and wind uncertainty study.
- `rocket_ascent_screening.json`: ascent screening study with mass, thrust, propellant, Isp, and drag-loss uncertainty.
- `comm_link_budget_trade.json`: communication link-budget trade with range and gain uncertainty.
- `rover_energy_mission.json`: rover battery and terrain-energy mission study.
- `guidance_gain_optimization.json`: optimizer example for guidance gain, thrust, and burn time.

Run any example with:

```bash
python3 -m tradespace validate --config examples/configs/mission_mc.json
python3 -m tradespace run --config examples/configs/mission_mc.json --out outputs/example_run --runs 200
```

Sweep examples use:

```bash
python3 -m tradespace sweep --config examples/configs/guidance_gain_sweep.json --out outputs/example_sweep
```

Reliability and UQ examples:

```bash
python3 -m tradespace run --config examples/configs/reliability_report_example.json --out outputs/reliability_example
python3 -m tradespace reliability --study outputs/reliability_example
python3 -m tradespace uq-report --study outputs/reliability_example
python3 -m tradespace sensitivity-deep --study outputs/reliability_example
```

Optimization and surrogate examples:

```bash
python3 -m tradespace optimize --config examples/configs/guidance_gain_optimization.json --out outputs/optimization_example
python3 -m tradespace fit-surrogate --study outputs/demo_expanded --out outputs/demo_expanded/miss_surrogate.json
python3 -m tradespace rank-designs --input outputs/optimization_example/optimization_results.csv --out outputs/optimization_ranked
```

Campaign example:

```bash
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --plan-only
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --out outputs/campaign_resume_example
python3 -m tradespace replay-run --campaign outputs/campaign_resume_example --scenario baseline_latin --run-index 0
```
