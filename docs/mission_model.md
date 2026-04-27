# Mission Model

The propagation loop uses a 3D point-mass state with position, velocity, mass, thrust, drag, and gravity. A 2D study is represented by zero crossrange terms.

## Features

- Target-intercept guidance toward a sensor-corrupted line of sight.
- Optional loft angle before the vehicle approaches the target.
- Wind layers interpolated by altitude.
- Constant or exponential density model, plus density layers.
- Drag from dynamic pressure, reference area, and drag coefficient.
- Thrust cutoff by burn time and a simple mass depletion proxy.
- Actuator lag through first-order command filtering.
- Qbar, load, ground, range, and time termination checks.

## Failure Reasons

Runs report `none`, `qbar_limit`, `load_limit`, `ground_intercept`, `range_limit`, `time_limit`, or `miss_distance`.
