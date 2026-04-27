# Configuration Reference

Configurations are JSON files. The loader applies package defaults, resolves `extends`, and then flattens named parameter blocks.

## Top-Level Fields

- `seed`: integer base seed for deterministic studies.
- `runs`: Monte Carlo run count.
- `study`: metadata such as `name` and `kind`.
- `sampler`: method and chunk metadata.
- `mission`, `vehicle`, `environment`, `guidance`, `sensors`: named parameter blocks.
- `uncertainty.distributions`: sampled parameter definitions.
- `sweep`: design-space definition for sweep studies.
- `outputs`: controls optional products such as SQLite storage.

## Inheritance

Use `extends` to inherit from another JSON file relative to the current file. Child values override parent values recursively.

## Validation

`python3 -m tradespace validate --config path.json` checks numeric fields, distribution requirements, covariance matrix shape, probabilities, and sweep structure. Errors include paths such as `uncertainties.wind_x_mps.std`.

## Reproducibility Commands

- `config-diff`: compare two resolved configs and list added, removed, and changed paths.
- `freeze-config`: write the resolved config plus its content hash.
- `generate-template`: create a starter template for a selected mission model.
