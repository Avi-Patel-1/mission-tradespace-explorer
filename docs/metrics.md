# Metrics

Each run writes a flat row to `results.csv`.

Core metrics:

- `miss_distance_m`: closest approach to the target.
- `crossrange_error_m`, `downrange_error_m`: target-relative error at closest approach.
- `terminal_altitude_m`, `terminal_speed_mps`, `terminal_flight_path_angle_deg`: terminal state.
- `time_of_flight_s`: propagated time.
- `max_qbar_pa`: peak dynamic pressure.
- `max_load_g`: non-gravitational acceleration proxy.
- `impulse_n_s`: integrated thrust command.
- `fuel_used_kg`: mass depletion proxy.
- `integrated_control_effort`: steering activity proxy.
- `robustness_margin`: minimum normalized margin across miss, qbar, and load checks.
- `success`, `failed`, `failure_reason`: outcome classification.
- `qbar_violation_pa`, `load_violation_g`, `miss_violation_m`, `ground_violation_m`: constraint violations.
