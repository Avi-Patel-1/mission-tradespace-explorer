# Reproducibility

Studies are deterministic for a fixed resolved config and base seed.

## Recorded State

- `resolved_config.json`: defaults and inheritance already applied.
- `manifest.json`: config hash, sampler, run count, and run status.
- `samples.csv`: sampled input values and run seeds.
- `study.sqlite`: optional run table with inputs, outputs, seed, status, and config hash.

## Continuing Work

Use the stored-campaign continuation command to reuse completed SQLite rows with the same config hash.

## Audit

Use `python3 -m tradespace audit --study outputs/demo_expanded` to check that required artifacts exist, row counts match, summary metadata is consistent, and SQLite storage has the expected completed rows.

## Export

Use `python3 -m tradespace export --db outputs/demo_expanded/study.sqlite --out outputs/demo_expanded/exported` to write database contents as CSV and JSON.
