# Methodology

The toolkit supports fast screening studies for guided mission concepts. It combines deterministic sampling, a point-mass flight model, constraint checks, sensitivity metrics, and static reporting.

## Study Flow

1. Load JSON config, apply defaults, and merge any inherited config.
2. Flatten mission, vehicle, environment, guidance, and sensor blocks into nominal parameters.
3. Draw uncertainty samples with a deterministic seed plan.
4. Propagate each mission case and record metrics plus failure reason.
5. Compute summary statistics, sensitivities, failure tables, and report plots.

## Model Scope

The mission model is intentionally compact. It is useful for comparing design options, ranking uncertainty drivers, and finding promising regions of a trade space. It should be calibrated before use as a final performance prediction.
