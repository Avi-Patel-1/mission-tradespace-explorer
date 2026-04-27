import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tradespace.audit import audit_study
from tradespace.campaign import compare_campaigns, expand_campaign, replay_run, run_campaign
from tradespace.config import apply_overrides, load_config, validate_config
from tradespace.config_tools import diff_configs, freeze_config, generate_template
from tradespace.deep_sensitivity import write_deep_sensitivity
from tradespace.distributions import sample_distribution
from tradespace.mission import simulate_mission
from tradespace.mission_models import available_models, simulate_model
from tradespace.optimization import optimize_config
from tradespace.pareto import pareto_front, score_designs, write_pareto
from tradespace.reliability import analyze_reliability, run_stress_test
from tradespace.reports import read_csv, rewrite_report_from_study, write_csv
from tradespace.runner import run_monte_carlo
from tradespace.samplers import factorial_design, generate_samples, grid_cases, halton, latin_hypercube, latinized_sobol, morris_trajectories, orthogonal_array, run_seed, sobol
from tradespace.storage import SQLiteStudy, export_sqlite
from tradespace.surrogate import fit_surrogate, predict_surrogate
from tradespace.sweep import run_sweep
from tradespace.uq import bootstrap_ci, jackknife_estimate, quantile_convergence, rare_event_probability, write_uq_artifacts


ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_scalar_distributions(self):
        rng = np.random.default_rng(1)
        specs = [
            {"type": "constant", "value": 4.0},
            {"type": "uniform", "low": 1.0, "high": 2.0},
            {"type": "normal", "mean": 0.0, "std": 1.0, "low": -2.0, "high": 2.0},
            {"type": "triangular", "low": 0.0, "mode": 0.5, "high": 1.0},
            {"type": "lognormal", "mean": 0.0, "sigma": 0.25, "low": 0.1, "high": 3.0},
            {"type": "truncated_normal", "mean": 5.0, "std": 1.0, "low": 3.0, "high": 7.0},
            {"type": "beta", "alpha": 2.0, "beta": 3.0, "low": -1.0, "high": 1.0},
            {"type": "choice", "values": [1, 2, 3], "probabilities": [0.2, 0.3, 0.5]},
            {"type": "bernoulli", "p": 0.5},
            {"type": "sweep", "values": [10, 20]},
        ]
        for spec in specs:
            value = sample_distribution(spec, rng, run_index=1)
            self.assertIsNotNone(value)

    def test_correlated_mixture_and_table_distributions(self):
        rng = np.random.default_rng(4)
        correlated = {
            "type": "correlated_normal",
            "names": ["a", "b"],
            "mean": [0.0, 0.0],
            "cov": [[1.0, 0.7], [0.7, 1.0]],
        }
        pairs = [sample_distribution(correlated, rng) for _ in range(600)]
        draws = np.array([[pair["a"], pair["b"]] for pair in pairs])
        self.assertGreater(np.cov(draws.T)[0, 1], 0.2)

        mixture = {
            "type": "mixture",
            "components": [
                {"weight": 0.1, "distribution": {"type": "constant", "value": -1}},
                {"weight": 0.9, "distribution": {"type": "constant", "value": 1}},
            ],
        }
        self.assertIn(sample_distribution(mixture, rng), {-1, 1})

        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "values.csv"
            table.write_text("value,weight\n2,1\n5,3\n")
            value = sample_distribution({"type": "table", "path": "values.csv", "column": "value", "weight_column": "weight"}, rng, root=tmp)
            self.assertIn(value, {2.0, 5.0})


class SamplerAndConfigTests(unittest.TestCase):
    def test_config_validation_and_inheritance(self):
        config = load_config(ROOT / "examples/configs/guidance_gain_sweep.json")
        self.assertEqual(validate_config(config), [])
        self.assertIn("nominal", config)
        self.assertIn("wind_x_mps", config["uncertainties"])

    def test_config_validation_catches_bad_sweep_and_sampler(self):
        errors = validate_config({"sampler": {"method": "unknown"}, "sweep": {"mode": "latin", "bounds": {"x": [2.0, 1.0]}}})
        self.assertTrue(any("sampler.method" in error for error in errors))
        self.assertTrue(any("sweep.bounds.x" in error for error in errors))

        bad_distributions = validate_config(
            {
                "uncertainties": {
                    "wind_x_mps": {"type": "uniform", "low": -1.0},
                    "bias": {"type": "correlated_normal", "names": ["x", "y"], "mean": [0.0, 0.0], "cov": [[1.0, 2.0], [2.0, 1.0]]},
                    "mode": {"type": "choice", "values": [1, 2], "probabilities": [0.0, 0.0]},
                    "table_value": {"type": "table", "path": "missing.csv", "column": "value"},
                }
            }
        )
        self.assertTrue(any("uniform.high" in error or "wind_x_mps.high" in error for error in bad_distributions))
        self.assertTrue(any("positive semidefinite" in error for error in bad_distributions))
        self.assertTrue(any("positive probability" in error for error in bad_distributions))
        self.assertTrue(any("does not exist" in error for error in bad_distributions))

    def test_apply_overrides_and_legacy_nominal_resolution(self):
        config = apply_overrides({"nominal": {"target_x_m": 1200.0}}, seed=99, sampler="stratified", write_sqlite=False)
        self.assertEqual(config["seed"], 99)
        self.assertEqual(config["sampler"]["method"], "stratified")
        self.assertFalse(config["outputs"]["write_sqlite"])
        self.assertIn("mass_kg", config["nominal"])

    def test_all_example_configs_validate(self):
        for path in sorted((ROOT / "examples/configs").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertEqual(validate_config(load_config(path)), [])

    def test_sampler_determinism_and_coverage(self):
        a = generate_samples({"x": 1.0}, {"x": {"type": "uniform", "low": 0.0, "high": 1.0}}, 8, 11, method="latin_hypercube")
        b = generate_samples({"x": 1.0}, {"x": {"type": "uniform", "low": 0.0, "high": 1.0}}, 8, 11, method="latin_hypercube")
        self.assertEqual([row[3]["x"] for row in a], [row[3]["x"] for row in b])
        self.assertEqual(run_seed(5, 2), run_seed(5, 2))
        self.assertEqual(latin_hypercube(5, 2, 1).shape, (5, 2))
        self.assertEqual(halton(5, 3).shape, (5, 3))
        self.assertEqual(sobol(8, 3, scramble_seed=1).shape, (8, 3))
        self.assertEqual(latinized_sobol(8, 2, 1).shape, (8, 2))
        self.assertEqual(orthogonal_array(9, 3, 1).shape, (9, 3))
        self.assertEqual(morris_trajectories(3, 2, 1).shape, (8, 3))
        self.assertEqual(factorial_design([2, 3]).shape, (6, 2))
        self.assertEqual(len(grid_cases({"a": [1, 2], "b": [3, 4]})), 4)


class MissionAndAnalysisTests(unittest.TestCase):
    def test_mission_returns_expanded_metrics(self):
        result = simulate_mission({"target_x_m": 1000.0, "thrust_n": 1200.0, "burn_time_s": 4.0, "success_miss_m": 500.0})
        for key in ["miss_distance_m", "crossrange_error_m", "terminal_speed_mps", "max_load_g", "robustness_margin", "failure_reason"]:
            self.assertIn(key, result)
        self.assertGreater(result["time_of_flight_s"], 0)

    def test_mission_model_registry_screening_models(self):
        self.assertIn("uav_endurance", available_models())
        for model in ["ballistic_intercept", "rocket_ascent", "uav_endurance", "comm_link_budget", "rover_energy"]:
            with self.subTest(model=model):
                result = simulate_model({"run_seed": 1, "target_x_m": 1000.0, "success_miss_m": 500.0}, {"mission": {"model": model}})
                self.assertIn("success", result)
                self.assertIn("robustness_margin", result)
                self.assertEqual(result["mission_model"], model)

    def test_failure_reason_logic(self):
        result = simulate_mission({"target_x_m": 3000.0, "max_qbar_pa": 100.0, "thrust_n": 1800.0})
        self.assertEqual(result["failure_reason"], "qbar_limit")
        self.assertFalse(result["success"])

    def test_pareto_and_constraint_filtering(self):
        rows = [
            {"case": 0, "success": True, "miss_distance_m": 10.0, "impulse_n_s": 100.0, "robustness_margin": 5.0},
            {"case": 1, "success": True, "miss_distance_m": 20.0, "impulse_n_s": 80.0, "robustness_margin": 6.0},
            {"case": 2, "success": False, "miss_distance_m": 5.0, "impulse_n_s": 200.0, "robustness_margin": -5.0},
        ]
        ranked = score_designs(rows)
        self.assertEqual(len(ranked), 3)
        self.assertGreaterEqual(len(pareto_front([row for row in ranked if row["feasible"]])), 1)


class RunnerStorageAndCliTests(unittest.TestCase):
    def test_monte_carlo_outputs_storage_and_report(self):
        config = load_config(ROOT / "examples/configs/mission_mc.json")
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_monte_carlo(config, tmp, runs=24)
            out = Path(tmp)
            self.assertEqual(summary["runs"], 24)
            for name in ["report.html", "sensitivity.csv", "failure_reasons.csv", "study.sqlite", "manifest.json"]:
                self.assertTrue((out / name).exists(), name)
            exported = export_sqlite(out / "study.sqlite", out / "exported")
            self.assertEqual(exported["runs"], 24)
            audit = audit_study(out)
            self.assertTrue(audit["ok"], audit)
            self.assertEqual(audit["rows"]["results"], 24)
            reliability = analyze_reliability(out)
            self.assertIn("reliability_index", reliability)
            uq = write_uq_artifacts(out)
            self.assertIn("bootstrap_mean", uq)
            deep = write_deep_sensitivity(out)
            self.assertIn("top_driver", deep)
            regenerated = rewrite_report_from_study(out)
            self.assertEqual(regenerated["runs"], 24)
            self.assertEqual(regenerated["config_hash"], summary["config_hash"])

            surrogate_path = out / "surrogate.json"
            model = fit_surrogate(out, surrogate_path, metric="miss_distance_m", model_type="polynomial")
            self.assertIn("cross_validation", model)
            prediction = predict_surrogate(surrogate_path, params={name: 1.0 for name in model["features"]})
            self.assertIn("prediction", prediction)

    def test_sweep_outputs_and_pareto_command_backend(self):
        config = {
            "seed": 9,
            "runs": 12,
            "mission": {"target_x_m": 2000.0, "success_miss_m": 400.0},
            "vehicle": {"thrust_n": 1450.0, "burn_time_s": 7.0},
            "sweep": {
                "mode": "grid",
                "values": {"guidance_gain": [1.6, 2.2], "thrust_n": [1400.0, 1600.0]},
                "constraints": {"require_success": False, "miss_distance_m": {"max": 600.0}},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_sweep(config, tmp)
            out = Path(tmp)
            self.assertEqual(summary["cases"], 4)
            self.assertTrue((out / "sweep_results.csv").exists())
            self.assertTrue((out / "manifest.json").exists())
            pareto_summary = write_pareto(out / "sweep_results.csv", out / "pareto")
            self.assertEqual(pareto_summary["input_rows"], 4)

    def test_output_flags_disable_optional_artifacts(self):
        config = load_config(ROOT / "examples/configs/mission_mc.json")
        config["outputs"]["write_sqlite"] = False
        config["outputs"]["write_html"] = False
        config["outputs"]["write_manifest"] = False
        with tempfile.TemporaryDirectory() as tmp:
            run_monte_carlo(config, tmp, runs=5)
            out = Path(tmp)
            self.assertTrue((out / "results.csv").exists())
            self.assertFalse((out / "study.sqlite").exists())
            self.assertFalse((out / "report.html").exists())
            self.assertFalse((out / "manifest.json").exists())

    def test_sqlite_storage_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "study.sqlite"
            with SQLiteStudy(db) as study:
                study.set_metadata("name", "roundtrip")
                study.record_run(0, 123, {"x": 1.0}, {"miss_distance_m": 2.0}, config_hash="abc")
                completed = study.completed_runs(config_hash="abc")
            self.assertIn(0, completed)
            export = export_sqlite(db, Path(tmp) / "export")
            self.assertEqual(export["runs"], 1)

    def test_uq_helpers_and_stress_test(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertLess(bootstrap_ci(values)["low"], bootstrap_ci(values)["high"])
        self.assertGreater(jackknife_estimate(values)["std_error"], 0.0)
        self.assertGreaterEqual(len(quantile_convergence(values)), 1)
        self.assertEqual(rare_event_probability([{"success": True}, {"success": False}])["hits"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_stress_test(ROOT / "examples/configs/mission_mc.json", tmp, max_cases=4)
            self.assertEqual(summary["cases"], 4)
            self.assertTrue((Path(tmp) / "stress_results.csv").exists())

    def test_optimization_and_config_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            opt = optimize_config(ROOT / "examples/configs/guidance_gain_optimization.json", out / "opt", iterations=2, population=5)
            self.assertGreaterEqual(opt["evaluations"], 10)
            self.assertTrue((out / "opt" / "optimization_results.csv").exists())
            frozen = freeze_config(ROOT / "examples/configs/mission_mc.json", out / "frozen.json")
            self.assertIn("config_hash", frozen)
            template = generate_template(out / "template.json", kind="sweep", model="uav_endurance")
            self.assertEqual(template["model"], "uav_endurance")
            diff = diff_configs(ROOT / "examples/configs/mission_mc.json", out / "template.json")
            self.assertTrue(diff["changed"] or diff["added"] or diff["removed"])

    def test_campaign_run_resume_replay_and_compare(self):
        manifest = ROOT / "examples/campaigns/resume_campaign.json"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign_a"
            plan = expand_campaign(manifest, out)
            self.assertEqual(len(plan["scenarios"]), 4)

            summary = run_campaign(manifest, out, jobs=1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["complete"], 4)
            self.assertTrue((out / "campaign_summary.csv").exists())

            resumed = run_campaign(manifest, out, jobs=1)
            self.assertEqual(resumed["skipped"], 4)
            replay = replay_run(out, "baseline_latin", 0)
            self.assertEqual(replay["run"], 0)
            self.assertIn("result", replay)

            out_b = Path(tmp) / "campaign_b"
            second = run_campaign(manifest, out_b, jobs=1)
            self.assertEqual(second["failed"], 0)
            comparison = compare_campaigns(out, out_b, Path(tmp) / "comparison")
            self.assertEqual(comparison["matched"], 4)
            self.assertTrue((Path(tmp) / "comparison" / "campaign_comparison.csv").exists())

    def test_cli_smoke(self):
        result = subprocess.run(
            [sys.executable, "-m", "tradespace", "validate", "--config", "examples/configs/mission_mc.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"valid": true', result.stdout)
        inspect = subprocess.run(
            [sys.executable, "-m", "tradespace", "inspect-config", "--config", "examples/configs/mission_mc.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"nominal"', inspect.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            starter = Path(tmp) / "starter.json"
            init = subprocess.run(
                [sys.executable, "-m", "tradespace", "init-config", "--kind", "sweep", "--name", "starter", "--out", str(starter)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"written"', init.stdout)
            self.assertTrue(starter.exists())
            self.assertEqual(validate_config(load_config(starter)), [])
            run_dir = Path(tmp) / "cli_run"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tradespace",
                    "run",
                    "--config",
                    "examples/configs/mission_mc.json",
                    "--out",
                    str(run_dir),
                    "--runs",
                    "5",
                    "--seed",
                    "12",
                    "--sampler",
                    "stratified",
                    "--no-sqlite",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"runs": 5', run.stdout)
            self.assertFalse((run_dir / "study.sqlite").exists())
            audit = subprocess.run(
                [sys.executable, "-m", "tradespace", "audit", "--study", str(run_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"ok": true', audit.stdout)

    def test_csv_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv(path, [{"run": 0, "success": True, "miss_distance_m": 1.2}])
            rows = read_csv(path)
            self.assertEqual(rows[0]["success"], True)
            self.assertAlmostEqual(rows[0]["miss_distance_m"], 1.2)


if __name__ == "__main__":
    unittest.main()
