from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments.plan_validity_target_credit import (
    TARGETS, fit_targets, fitting_readout, held_readout, labels, prediction_error, predict,
)
from freq_hrl.experiments.pointmaze_timing_baselines import _CAUSAL_DYNAMIC_NAMES
from scripts import submit_plan_validity_target_credit_scheduleurm as submitter
from scripts import run_plan_validity_target_credit as worker
from scripts import analyze_plan_validity_target_credit as analyzer


def bank():
    rng = np.random.default_rng(604)
    output = []
    for seed in range(8):
        for step in (50, 100, 150):
            x = rng.normal(size=len(_CAUSAL_DYNAMIC_NAMES))
            effect = x[0] + x[1] * x[2]
            futures = [{"local_advantage": effect + noise,
                "adaptive_window_advantage": -effect + noise,
                "full_advantage": 2 * effect + noise} for noise in rng.normal(0, 0.1, 8)]
            output.append({"seed": seed, "opportunity_step": step,
                "feature_names": list(_CAUSAL_DYNAMIC_NAMES), "causal_features": x.tolist(),
                "conditional_replicas": futures})
    return output


class TargetCreditTest(unittest.TestCase):
    def test_future_blocks_and_capacity_are_matched(self):
        rows = bank()
        rules, oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        self.assertEqual(set(rules), set(TARGETS))
        models = [r.model["model"] for r in rules.values()]
        for model in models:
            self.assertAlmostEqual(model["effective_df_slopes"], 8)
            self.assertEqual(model["coefficient_count"], 40)
            self.assertEqual(model["alpha"], models[0]["alpha"])
        for target, rule in rules.items():
            self.assertAlmostEqual(rule.threshold, np.quantile(oof[target], 0.75))
            scores = np.array([rule.score(tuple(r["feature_names"]), np.array(r["causal_features"]), 0, 1) for r in rows])
            np.testing.assert_allclose(scores, predict(rule, rows))

    def test_independent_future_B_cannot_change_coefficients_or_thresholds(self):
        rows = bank()
        first, first_oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        changed = deepcopy(rows)
        for row in changed:
            for future in row["conditional_replicas"][4:]:
                for name in TARGETS.values():
                    future[name] += 1000
        second, second_oof = fit_targets(changed, effective_df=8, threshold_quantile=0.75)
        self.assertEqual({k: v.to_dict() for k, v in first.items()}, {k: v.to_dict() for k, v in second.items()})
        for target in TARGETS:
            np.testing.assert_array_equal(first_oof[target], second_oof[target])

    def test_train_error_is_not_noise_corrected_using_shared_fit_noise(self):
        rows = bank()
        rules, oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        result = fitting_readout(rows, rules, oof)
        for target in TARGETS:
            self.assertIsNone(result[target]["own_training_A"]["noise_corrected_mse"])
            self.assertIsNotNone(result[target]["own_independent_future_B"]["noise_corrected_mse"])
            self.assertEqual(result[target]["own_prefix_oof_B"]["paths"], 8)
        held = held_readout(rows, rules)
        self.assertEqual(set(held["G"]), {"A", "B", "all"})

    def test_negative_noise_correction_is_preserved(self):
        rows = bank()[:1]
        for i, future in enumerate(rows[0]["conditional_replicas"]):
            future["full_advantage"] = (-1) ** i
        result = prediction_error(np.zeros(1), rows, "G", "B", independent=True)
        self.assertLess(result["noise_corrected_mse"], 0)
        rows[0]["conditional_replicas"].pop()
        with self.assertRaises(ValueError):
            labels(rows, "L", "A")

    def test_reservation_retains_hold_and_never_reuses_old_eval_paths(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_local_episode_diagnostic_2026-11-10.json").read_text())
        plan = draft_plan()
        result = submitter.prepare(source, plan, "fixture", Path("protocols/fixture.json"))
        reserved = {s for c in seed_cells(plan) for seeds in c["seed_roles"].values() for s in seeds}
        for cell in result["cells"]:
            self.assertEqual(cell["training_prefix_seeds"], cell["expected_configuration"]["branch_fit_seeds"])
            self.assertFalse(set(cell["held_prefix_seeds"]) & reserved)
            self.assertFalse(set(cell["held_prefix_seeds"]) & set(cell["previous_evaluation_seeds"]))
        for phase, jobs in result["jobs"].items():
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertIn("--phase " + phase, job["cmd"])
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["protect_from_auto_eviction"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        self.assertFalse(result["confirmation_released"])
        self.assertEqual(result["candidate_recipe_decision"], "PILOT_RECIPE_HOLD")

    def test_missing_training_paths_are_not_replaced_with_evaluation_data(self):
        with self.assertRaisesRegex(ValueError, "three fitting paths"):
            fit_targets(bank()[:6], effective_df=8, threshold_quantile=0.75)

    def test_rules_are_frozen_before_independent_readout_and_new_prefixes(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_local_episode_diagnostic_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        cell = manifest["cells"][0]
        for row in cell["sources"].values():
            row["validation_ise"] = 1
        rules, _ = fit_targets(bank(), effective_df=8, threshold_quantile=0.75)
        originals = {m: rules["L"] for m in cell["sources"]}
        records = {"termination_history": {"termination_readiness": {"eligible_for_fresh_confirmation": False}}}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "run"
            (directory / "private").mkdir(parents=True)
            gate_path = directory.parent / "precheck/public/result.json"
            gate_path.parent.mkdir(parents=True)
            gate = {"status": "complete", "stage": "precheck_passed", "run_name": "fixture",
                    "optimizer_seed": cell["optimizer_seed"]}
            gate_path.write_text(json.dumps(gate))
            calls = []

            def collect(*args, **kwargs):
                calls.append(kwargs["seeds"])
                if len(calls) == 2:
                    self.assertTrue((directory / "private/frozen_rules.json").exists())
                return bank()

            with patch.object(worker, "load_sources", return_value=(records,
                {m: object() for m in originals}, originals, SimpleNamespace(**cell["expected_configuration"]))), \
                 patch.object(worker, "run_episode", return_value=(None, {"tracking_squared_error_integral": 1})), \
                 patch.object(worker, "collect_bank", side_effect=collect):
                result, raw = worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertEqual(calls, [cell["training_prefix_seeds"], cell["held_prefix_seeds"]])
                self.assertEqual(result["stage"], "repair_readout_complete")
                self.assertEqual(len(result["episodes"]), 8)
                self.assertFalse(result["termination_development_eligible"])
                gate_path.write_text(json.dumps({**gate, "status": "failed"}))
                calls.clear()
                with self.assertRaisesRegex(ValueError, "successful matching native precheck"):
                    worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertFalse(calls)

    def test_analysis_does_not_omit_missing_roots_or_clear_hold(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_local_episode_diagnostic_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        with tempfile.TemporaryDirectory() as folder:
            result = analyzer.summarize(manifest, root=Path(folder))
        self.assertEqual(len(result["missing"]), 4)
        self.assertEqual(result["task_means"], {"pointmaze": None, "reacher_tracking": None})
        self.assertFalse(result["confirmation_released"])

    def test_private_and_public_costs_both_include_work_after_last_progress_update(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def measured_cell(cell, manifest, phase, directory, publish):
            count_work("environment_steps", 7)
            publish({"stage": "rules_frozen_before_holdout"})
            count_work("environment_steps", 11)
            return {"stage": "repair_readout_complete"}, None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out",
                    "training_prefix_seeds": [2, 3, 4], "held_prefix_seeds": [5, 6]}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=measured_cell):
                self.assertEqual(worker.main(["--manifest", str(manifest), "--cell-index", "0", "--phase", "run"]), 0)
            private = json.loads((root / "out/run/private/result.json").read_text())
            public = json.loads((root / "out/run/public/result.json").read_text())
            self.assertEqual(private["measured_cost"], public["measured_cost"])
            self.assertEqual(private["measured_cost"]["counts"]["environment_steps"], 18)


if __name__ == "__main__":
    unittest.main()
