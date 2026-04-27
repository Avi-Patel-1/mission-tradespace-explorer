# Campaigns

Campaign manifests run named scenario families from one JSON file. Each scenario remains a normal study directory, so existing `audit`, `report`, `reliability`, `fit-surrogate`, `export`, and replay workflows still work.

## Manifest Shape

```json
{
  "campaign": { "name": "resume_campaign_example" },
  "out_dir": "../../outputs/campaign_resume_example",
  "base_config": "../configs/mission_mc.json",
  "scenarios": [
    {
      "name": "baseline_latin",
      "command": "run",
      "runs": 32,
      "sampler": "latin_hypercube"
    },
    {
      "name": "guidance_gain_grid",
      "command": "sweep",
      "config": "../configs/guidance_gain_sweep.json"
    }
  ]
}
```

Scenario `overrides` are deep-merged onto the base config before validation. Scenario-level `runs`, `seed`, and `sampler` fields are convenience overrides for the resolved config.

## Supported Scenario Commands

- `run`, `monte-carlo`, and `robustness`: execute `run_monte_carlo`.
- `sweep`: execute `run_sweep`.
- `optimize`: execute `optimize_config` with optional `method`, `iterations`, and `population`.
- `stress-test`: execute stress corner cases with optional `max_cases`.
- `reliability`: run a Monte Carlo study, then write reliability artifacts.
- `margin-report`: run a Monte Carlo study, then write margin artifacts.

## CLI

```bash
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --plan-only
python3 -m tradespace campaign --manifest examples/campaigns/resume_campaign.json --out outputs/campaign_resume_example --jobs 2
python3 -m tradespace replay-run --campaign outputs/campaign_resume_example --scenario baseline_latin --run-index 0
python3 -m tradespace compare-campaigns --left outputs/campaign_resume_example --right outputs/campaign_candidate --out outputs/campaign_compare
```

With default continuation behavior, complete scenarios are skipped on repeat runs. Failed scenarios are preserved unless `--retry-failed` is passed.

## Artifacts

- `campaign_manifest.json`: manifest copy used for the run.
- `campaign_status.json`: top-level campaign status with scenario records.
- `campaign_summary.csv`: scenario status, output paths, run counts, success rates, Pareto counts, and errors.
- `<scenario>/scenario_config.json`: resolved config used by that scenario.
- `<scenario>/scenario_status.json`: per-scenario status, config hash, timing, and summary.
- `<scenario>/scenario.log`: success marker or traceback.
- `replays/<scenario>_run_<index>.json`: deterministic replay for one Monte Carlo run.
- `campaign_comparison.csv`: baseline-vs-candidate scenario comparison.
