# Surrogate Models

Surrogate models are fitted from completed study directories.

```bash
python3 -m tradespace fit-surrogate --study outputs/demo_expanded --out outputs/demo_expanded/miss_surrogate.json --metric miss_distance_m --model-type polynomial
python3 -m tradespace surrogate-predict --model outputs/demo_expanded/miss_surrogate.json --params '{"guidance_gain":2.2,"thrust_n":2200,"mass_kg":42,"drag_cd":0.22,"wind_x_mps":0,"wind_z_mps":0,"sensor_noise_m":8,"sensor_bias_x_m":0,"sensor_bias_z_m":0}'
```

Supported model types:

- `polynomial`: ridge-regularized polynomial response surface with linear, square, and pairwise interaction terms.
- `rbf`: radial basis function surrogate with deterministic center selection and Gaussian basis functions.

The JSON model records feature names, normalization, coefficients or centers, training metrics, and cross-validation metrics. CSV batch prediction is supported with `--input-csv` and `--out-csv`.
