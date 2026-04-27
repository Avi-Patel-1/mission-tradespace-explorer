from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .mission import G0, simulate_mission


MissionFunction = Callable[[dict[str, Any]], dict[str, Any]]


def _success(result: dict[str, Any], threshold_key: str, value_key: str, default_threshold: float) -> None:
    threshold = float(result.pop(threshold_key, default_threshold))
    value = float(result.get(value_key, 0.0))
    result["success"] = bool(value <= threshold)
    result["failed"] = not result["success"]
    result["failure_reason"] = "none" if result["success"] else value_key


def ballistic_intercept(params: dict[str, Any]) -> dict[str, Any]:
    speed = float(params.get("initial_speed_mps", 300.0))
    angle = math.radians(float(params.get("launch_angle_deg", params.get("initial_flight_path_deg", 35.0))))
    target_x = float(params.get("target_x_m", 5000.0))
    target_z = float(params.get("target_z_m", 0.0))
    z0 = float(params.get("initial_z_m", 0.0))
    wind_x = float(params.get("wind_x_mps", 0.0))
    cd_loss = max(float(params.get("drag_cd", 0.0)), 0.0) * 0.08
    vx = speed * math.cos(angle) * (1.0 - cd_loss) + wind_x
    vz = speed * math.sin(angle) * (1.0 - 0.5 * cd_loss)
    t = target_x / max(vx, 1e-6)
    z = z0 + vz * t - 0.5 * G0 * t * t
    miss = abs(z - target_z)
    range_m = vx * max((vz + math.sqrt(max(vz * vz + 2 * G0 * z0, 0.0))) / G0, 0.0)
    result = {
        "miss_distance_m": float(miss),
        "downrange_error_m": float(range_m - target_x),
        "crossrange_error_m": 0.0,
        "terminal_altitude_m": float(z),
        "terminal_speed_mps": float(math.hypot(vx, vz - G0 * t)),
        "time_of_flight_s": float(t),
        "max_qbar_pa": 0.5 * float(params.get("density_kg_m3", 1.1)) * speed * speed,
        "max_load_g": 1.0,
        "impulse_n_s": 0.0,
        "integrated_control_effort": 0.0,
        "robustness_margin": float(params.get("success_miss_m", 100.0)) - miss,
        "success_miss_m": float(params.get("success_miss_m", 100.0)),
    }
    _success(result, "success_miss_m", "miss_distance_m", 100.0)
    return result


def rocket_ascent(params: dict[str, Any]) -> dict[str, Any]:
    mass0 = max(float(params.get("mass_kg", 1200.0)), 1.0)
    thrust = max(float(params.get("thrust_n", 22000.0)), 0.0)
    burn = max(float(params.get("burn_time_s", 35.0)), 0.0)
    propellant = max(float(params.get("propellant_mass_kg", 420.0)), 0.0)
    isp = max(float(params.get("isp_s", 230.0)), 1.0)
    drag_loss = float(params.get("drag_loss_mps", 90.0))
    gravity_loss = G0 * burn * 0.55
    mass1 = max(mass0 - propellant, 1.0)
    ideal_dv = isp * G0 * math.log(mass0 / mass1)
    thrust_margin = thrust / max(mass0 * G0, 1e-9) - 1.0
    burnout_speed = max(ideal_dv - gravity_loss - drag_loss, 0.0)
    altitude = 0.5 * max(burnout_speed, 0.0) * burn * 0.62
    target_altitude = float(params.get("target_altitude_m", 80000.0))
    margin = altitude - target_altitude
    success = margin >= 0.0 and thrust_margin > 0.15
    return {
        "miss_distance_m": abs(margin),
        "terminal_altitude_m": float(altitude),
        "terminal_speed_mps": float(burnout_speed),
        "time_of_flight_s": float(burn),
        "max_qbar_pa": float(params.get("max_qbar_estimate_pa", 45000.0) * (1.0 + max(thrust_margin, 0.0))),
        "max_load_g": float(max(thrust / max(mass1 * G0, 1.0), 0.0)),
        "impulse_n_s": float(thrust * burn),
        "fuel_used_kg": float(propellant),
        "integrated_control_effort": 0.0,
        "robustness_margin": float(margin),
        "success": bool(success),
        "failed": not bool(success),
        "failure_reason": "none" if success else ("thrust_to_weight" if thrust_margin <= 0.15 else "target_altitude"),
    }


def uav_endurance(params: dict[str, Any]) -> dict[str, Any]:
    battery_wh = float(params.get("battery_wh", 850.0))
    cruise_power_w = max(float(params.get("cruise_power_w", 240.0)), 1.0)
    payload_power_w = max(float(params.get("payload_power_w", 35.0)), 0.0)
    speed = max(float(params.get("cruise_speed_mps", 18.0)), 1.0)
    wind = float(params.get("wind_x_mps", 0.0))
    reserve_fraction = float(params.get("reserve_fraction", 0.18))
    target_range = float(params.get("target_range_m", 42000.0))
    usable_wh = battery_wh * max(0.0, 1.0 - reserve_fraction)
    endurance_h = usable_wh / (cruise_power_w + payload_power_w)
    ground_speed = max(speed - wind, 1.0)
    range_m = endurance_h * 3600.0 * ground_speed
    margin = range_m - target_range
    return {
        "miss_distance_m": max(0.0, -margin),
        "range_m": float(range_m),
        "endurance_h": float(endurance_h),
        "terminal_speed_mps": float(ground_speed),
        "time_of_flight_s": float(endurance_h * 3600.0),
        "max_qbar_pa": 0.5 * float(params.get("density_kg_m3", 1.1)) * ground_speed * ground_speed,
        "max_load_g": 1.0,
        "impulse_n_s": 0.0,
        "fuel_used_kg": 0.0,
        "integrated_control_effort": 0.0,
        "robustness_margin": float(margin),
        "success": bool(margin >= 0.0),
        "failed": bool(margin < 0.0),
        "failure_reason": "none" if margin >= 0.0 else "range_shortfall",
    }


def comm_link_budget(params: dict[str, Any]) -> dict[str, Any]:
    tx_power_dbw = float(params.get("tx_power_dbw", 10.0))
    tx_gain_dbi = float(params.get("tx_gain_dbi", 8.0))
    rx_gain_dbi = float(params.get("rx_gain_dbi", 12.0))
    frequency_hz = max(float(params.get("frequency_hz", 2.2e9)), 1.0)
    range_m = max(float(params.get("range_m", 120000.0)), 1.0)
    bandwidth_hz = max(float(params.get("bandwidth_hz", 1.0e6)), 1.0)
    noise_figure_db = float(params.get("noise_figure_db", 3.0))
    required_snr_db = float(params.get("required_snr_db", 8.0))
    c = 299792458.0
    fspl_db = 20.0 * math.log10(4.0 * math.pi * range_m * frequency_hz / c)
    noise_dbw = -228.6 + 10.0 * math.log10(290.0) + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    received_dbw = tx_power_dbw + tx_gain_dbi + rx_gain_dbi - fspl_db - float(params.get("losses_db", 2.0))
    snr_db = received_dbw - noise_dbw
    margin = snr_db - required_snr_db
    return {
        "miss_distance_m": max(0.0, -margin),
        "link_margin_db": float(margin),
        "snr_db": float(snr_db),
        "range_m": float(range_m),
        "terminal_speed_mps": 0.0,
        "time_of_flight_s": 0.0,
        "max_qbar_pa": 0.0,
        "max_load_g": 0.0,
        "impulse_n_s": 0.0,
        "integrated_control_effort": 0.0,
        "robustness_margin": float(margin),
        "success": bool(margin >= 0.0),
        "failed": bool(margin < 0.0),
        "failure_reason": "none" if margin >= 0.0 else "link_margin",
    }


def rover_energy(params: dict[str, Any]) -> dict[str, Any]:
    battery_wh = float(params.get("battery_wh", 1200.0))
    drive_power_w = max(float(params.get("drive_power_w", 180.0)), 1.0)
    payload_power_w = max(float(params.get("payload_power_w", 45.0)), 0.0)
    speed_mps = max(float(params.get("speed_mps", 0.45)), 1e-3)
    distance_m = float(params.get("target_distance_m", 6000.0))
    slope_deg = math.radians(float(params.get("slope_deg", 4.0)))
    terrain_factor = max(float(params.get("terrain_factor", 1.15)), 0.1)
    adjusted_power = (drive_power_w * terrain_factor * (1.0 + max(math.sin(slope_deg), 0.0))) + payload_power_w
    duration_h = distance_m / speed_mps / 3600.0
    energy_used = adjusted_power * duration_h
    margin = battery_wh - energy_used
    return {
        "miss_distance_m": max(0.0, -margin),
        "energy_margin_wh": float(margin),
        "energy_used_wh": float(energy_used),
        "range_m": float(distance_m if margin >= 0 else distance_m * battery_wh / max(energy_used, 1e-9)),
        "terminal_speed_mps": float(speed_mps),
        "time_of_flight_s": float(duration_h * 3600.0),
        "max_qbar_pa": 0.0,
        "max_load_g": 0.0,
        "impulse_n_s": 0.0,
        "integrated_control_effort": 0.0,
        "robustness_margin": float(margin),
        "success": bool(margin >= 0.0),
        "failed": bool(margin < 0.0),
        "failure_reason": "none" if margin >= 0.0 else "energy_shortfall",
    }


MISSION_REGISTRY: dict[str, MissionFunction] = {
    "point_mass": simulate_mission,
    "target_intercept": simulate_mission,
    "ballistic_intercept": ballistic_intercept,
    "rocket_ascent": rocket_ascent,
    "uav_endurance": uav_endurance,
    "comm_link_budget": comm_link_budget,
    "rover_energy": rover_energy,
}


def available_models() -> list[str]:
    return sorted(MISSION_REGISTRY)


def model_name_from_config(config: dict[str, Any]) -> str:
    mission = config.get("mission", {}) if isinstance(config.get("mission"), dict) else {}
    study = config.get("study", {}) if isinstance(config.get("study"), dict) else {}
    return str(mission.get("model", study.get("model", config.get("model", "point_mass"))))


def simulate_model(params: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    model_name = model_name_from_config(config or {})
    if model_name not in MISSION_REGISTRY:
        raise ValueError(f"unsupported mission model: {model_name}")
    result = MISSION_REGISTRY[model_name](params)
    result["mission_model"] = model_name
    return result
