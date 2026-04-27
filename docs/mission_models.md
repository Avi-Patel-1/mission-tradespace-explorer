# Mission Models

Mission models are registered in `tradespace.mission_models`. Select a model with `mission.model` in the config. The default model is `point_mass`.

Available models:

- `point_mass` and `target_intercept`: guided 3D point-mass intercept model.
- `ballistic_intercept`: closed-form ballistic screen with drag-loss proxy.
- `rocket_ascent`: rocket ascent screen using ideal delta-v, gravity loss, drag loss, thrust-to-weight margin, and target altitude.
- `uav_endurance`: range/endurance screen with battery capacity, power draw, reserve, speed, and wind.
- `comm_link_budget`: communication link-budget screen with free-space path loss, noise figure, bandwidth, and required SNR.
- `rover_energy`: energy-limited rover traverse model with terrain, slope, payload power, speed, and battery margin.

Every model returns `success`, `failed`, `failure_reason`, `robustness_margin`, and common report columns where meaningful. This keeps Monte Carlo, reliability, Pareto, and reporting tools compatible across mission types.
