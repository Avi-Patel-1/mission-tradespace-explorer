from __future__ import annotations

import math
from typing import Any

import numpy as np

G0 = 9.80665


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        out = np.zeros_like(vector, dtype=float)
        out[0] = 1.0
        return out
    return vector / norm


def _interpolate_layers(layers: Any, altitude_m: float, key: str, default: float) -> float:
    if not isinstance(layers, list) or not layers:
        return default
    valid = sorted(
        (float(layer.get("altitude_m", 0.0)), float(layer.get(key, default)))
        for layer in layers
        if isinstance(layer, dict)
    )
    if not valid:
        return default
    if altitude_m <= valid[0][0]:
        return valid[0][1]
    if altitude_m >= valid[-1][0]:
        return valid[-1][1]
    for (a0, v0), (a1, v1) in zip(valid[:-1], valid[1:]):
        if a0 <= altitude_m <= a1:
            frac = (altitude_m - a0) / max(a1 - a0, 1e-9)
            return v0 + frac * (v1 - v0)
    return default


def _density(params: dict[str, Any], altitude_m: float) -> float:
    base = max(float(params.get("density_kg_m3", 1.1)), 0.0)
    layers = params.get("density_layers")
    if isinstance(layers, list) and layers:
        return max(_interpolate_layers(layers, altitude_m, "density_kg_m3", base), 0.0)
    if str(params.get("density_model", "exponential")) == "constant":
        return base
    scale = max(float(params.get("density_scale_height_m", 8500.0)), 1.0)
    return max(base * math.exp(-max(altitude_m, 0.0) / scale), 0.0)


def _wind(params: dict[str, Any], altitude_m: float) -> np.ndarray:
    layers = params.get("wind_layers")
    return np.array(
        [
            _interpolate_layers(layers, altitude_m, "wind_x_mps", float(params.get("wind_x_mps", 0.0))),
            _interpolate_layers(layers, altitude_m, "wind_y_mps", float(params.get("wind_y_mps", 0.0))),
            _interpolate_layers(layers, altitude_m, "wind_z_mps", float(params.get("wind_z_mps", 0.0))),
        ],
        dtype=float,
    )


def _initial_velocity(params: dict[str, Any]) -> np.ndarray:
    speed = float(params.get("initial_speed_mps", 145.0))
    gamma = math.radians(float(params.get("initial_flight_path_deg", 8.0)))
    heading = math.radians(float(params.get("initial_heading_deg", 0.0)))
    horizontal = speed * math.cos(gamma)
    return np.array(
        [
            horizontal * math.cos(heading),
            horizontal * math.sin(heading),
            speed * math.sin(gamma),
        ],
        dtype=float,
    )


def _guidance_gain(params: dict[str, Any], speed: float, qbar: float) -> float:
    base = float(params.get("guidance_gain", 2.25))
    speed_ref = max(float(params.get("guidance_speed_ref_mps", 150.0)), 1.0)
    qbar_ref = max(float(params.get("guidance_qbar_ref_pa", 45000.0)), 1.0)
    gain = base
    gain += float(params.get("guidance_gain_speed_slope", 0.0)) * ((speed - speed_ref) / speed_ref)
    gain += float(params.get("guidance_gain_qbar_slope", 0.0)) * (qbar / qbar_ref)
    return float(np.clip(gain, 0.1, 8.0))


def simulate_mission(params: dict[str, Any]) -> dict[str, float | bool | str]:
    dt = max(float(params.get("dt_s", 0.05)), 1e-3)
    max_time = max(float(params.get("max_time_s", 70.0)), dt)
    target = np.array(
        [
            float(params.get("target_x_m", 3000.0)),
            float(params.get("target_y_m", 0.0)),
            float(params.get("target_z_m", 0.0)),
        ],
        dtype=float,
    )
    pos = np.array(
        [
            float(params.get("initial_x_m", 0.0)),
            float(params.get("initial_y_m", 0.0)),
            float(params.get("initial_z_m", 110.0)),
        ],
        dtype=float,
    )
    vel = _initial_velocity(params)
    mass_initial = max(float(params.get("mass_kg", 42.0)), 0.1)
    mass = mass_initial
    thrust = max(float(params.get("thrust_n", 1560.0)), 0.0)
    burn_time = max(float(params.get("burn_time_s", 8.0)), 0.0)
    propellant = max(float(params.get("propellant_mass_kg", 0.0)), 0.0)
    mass_flow = float(params.get("mass_flow_kgps", propellant / burn_time if burn_time > 0.0 else 0.0))
    area = max(float(params.get("reference_area_m2", 0.032)), 1e-6)
    cd = max(float(params.get("drag_cd", 0.22)), 0.0)
    success_miss = max(float(params.get("success_miss_m", 230.0)), 0.0)
    ground_altitude = float(params.get("ground_altitude_m", 0.0))
    qbar_limit = float(params.get("max_qbar_pa", float("inf")))
    load_limit = float(params.get("max_load_g", float("inf")))
    max_range = float(params.get("max_range_m", target[0] + 2500.0))
    loft_angle = math.radians(float(params.get("loft_angle_deg", 0.0)))
    actuator_tau = max(float(params.get("actuator_time_constant_s", 0.12)), 0.0)
    rng = np.random.default_rng(int(params.get("run_seed", 0)))

    sensor_bias = np.array(
        [
            float(params.get("sensor_bias_x_m", 0.0)),
            float(params.get("sensor_bias_y_m", 0.0)),
            float(params.get("sensor_bias_z_m", 0.0)),
        ],
        dtype=float,
    )
    sensor_noise = max(float(params.get("sensor_noise_m", 0.0)), 0.0)

    command_dir = _unit(vel)
    min_miss = float(np.linalg.norm(target - pos))
    closest_pos = pos.copy()
    max_qbar = 0.0
    max_load_g = 0.0
    impulse = 0.0
    control_effort = 0.0
    hard_failure = False
    failure_reason = "none"
    t = 0.0

    while t < max_time:
        altitude = float(pos[2])
        wind = _wind(params, altitude)
        rel = vel - wind
        rel_speed = float(np.linalg.norm(rel))
        rho = _density(params, altitude)
        qbar = 0.5 * rho * rel_speed * rel_speed
        max_qbar = max(max_qbar, qbar)

        sensed_target = target + sensor_bias + rng.normal(0.0, sensor_noise, 3)
        los = sensed_target - pos
        if loft_angle != 0.0 and pos[0] < target[0] * 0.65:
            los = los + np.array([0.0, 0.0, math.tan(loft_angle) * max(target[0] - pos[0], 0.0) * 0.16])
        desired = _unit(los)
        vel_dir = _unit(vel)
        gain = _guidance_gain(params, float(np.linalg.norm(vel)), qbar)
        raw_command = _unit(desired + gain * 0.18 * (desired - vel_dir))
        if actuator_tau > 0.0:
            alpha = min(dt / actuator_tau, 1.0)
            command_dir = _unit((1.0 - alpha) * command_dir + alpha * raw_command)
        else:
            command_dir = raw_command

        active_thrust = thrust if t <= burn_time else 0.0
        drag = -qbar * cd * area * _unit(rel)
        gravity = np.array([0.0, 0.0, -G0])
        non_grav_accel = (active_thrust * command_dir + drag) / max(mass, 0.1)
        accel = non_grav_accel + gravity
        load_g = float(np.linalg.norm(non_grav_accel) / G0)
        max_load_g = max(max_load_g, load_g)

        pos = pos + vel * dt
        vel = vel + accel * dt
        if active_thrust > 0.0 and mass_flow > 0.0:
            mass = max(mass_initial - propellant, mass - mass_flow * dt)
        t += dt
        impulse += active_thrust * dt
        control_effort += float(np.linalg.norm(raw_command - vel_dir) ** 2) * dt

        miss = float(np.linalg.norm(target - pos))
        if miss < min_miss:
            min_miss = miss
            closest_pos = pos.copy()

        if qbar > qbar_limit:
            hard_failure = True
            failure_reason = "qbar_limit"
            break
        if load_g > load_limit:
            hard_failure = True
            failure_reason = "load_limit"
            break
        if pos[2] <= ground_altitude and min_miss > success_miss:
            hard_failure = True
            failure_reason = "ground_intercept"
            break
        if float(np.linalg.norm(pos[:2])) > max_range:
            hard_failure = True
            failure_reason = "range_limit"
            break
        if min_miss <= max(success_miss * 0.1, 1.0):
            break
        if pos[0] > target[0] + max(1000.0, success_miss * 3.0) and np.dot(target - pos, vel) < 0.0:
            break

    terminal_speed = float(np.linalg.norm(vel))
    horizontal_speed = max(float(np.linalg.norm(vel[:2])), 1e-9)
    terminal_fpa = math.degrees(math.atan2(float(vel[2]), horizontal_speed))
    qbar_violation = max(0.0, max_qbar - qbar_limit) if math.isfinite(qbar_limit) else 0.0
    load_violation = max(0.0, max_load_g - load_limit) if math.isfinite(load_limit) else 0.0
    miss_violation = max(0.0, min_miss - success_miss)
    ground_violation = max(0.0, ground_altitude - float(pos[2]))
    success = (min_miss <= success_miss) and not hard_failure
    if not success and failure_reason == "none":
        failure_reason = "time_limit" if t >= max_time else "miss_distance"
    robustness_margin = min(
        success_miss - min_miss,
        (qbar_limit - max_qbar) / 1000.0 if math.isfinite(qbar_limit) else success_miss,
        (load_limit - max_load_g) * 10.0 if math.isfinite(load_limit) else success_miss,
    )

    return {
        "miss_distance_m": min_miss,
        "crossrange_error_m": float(closest_pos[1] - target[1]),
        "downrange_error_m": float(closest_pos[0] - target[0]),
        "terminal_altitude_m": float(pos[2]),
        "terminal_speed_mps": terminal_speed,
        "terminal_flight_path_angle_deg": float(terminal_fpa),
        "time_of_flight_s": float(t),
        "max_qbar_pa": float(max_qbar),
        "max_load_g": float(max_load_g),
        "impulse_n_s": float(impulse),
        "fuel_used_kg": float(max(0.0, mass_initial - mass)),
        "integrated_control_effort": float(control_effort),
        "robustness_margin": float(robustness_margin),
        "qbar_violation_pa": float(qbar_violation),
        "load_violation_g": float(load_violation),
        "miss_violation_m": float(miss_violation),
        "ground_violation_m": float(ground_violation),
        "final_x_m": float(pos[0]),
        "final_y_m": float(pos[1]),
        "final_z_m": float(pos[2]),
        "success": bool(success),
        "failed": bool(not success),
        "failure_reason": failure_reason,
    }
