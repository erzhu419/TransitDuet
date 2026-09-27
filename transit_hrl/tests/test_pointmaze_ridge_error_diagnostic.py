from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_ridge_error_diagnostic as diagnostic
from scripts import pointmaze_ridge_error_diagnostic_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class RidgeErrorDiagnosticTest(unittest.TestCase):
    def test_signed_terms_and_exact_sum(self):
        result = diagnostic.decompose(np.array([1.]), np.array([[.5]]),
                                      np.array([2.]), np.array([.25]), np.array([3.]), np.array([.5]))
        self.assertEqual(result["training_label_realization_mse_estimate"][0], -.0625)
        self.assertEqual(result["signal_mismatch_mse_estimate"][0], 3.4375)
        self.assertEqual(result["interaction_estimate"][0], .125)
        self.assertEqual(result["total_mse_estimate"][0], 3.5)

    def test_corrections_are_unbiased_under_independent_fresh_means(self):
        p, h = np.array([2.]), np.array([[.5, -.25]])
        train, vt, query, vq = np.array([1., 2.]), np.array([.25, 1.]), np.array([3.]), np.array([.5])
        samples = [diagnostic.decompose(p, h, train + np.array(s[:2]) * np.sqrt(vt), vt,
                                        query + s[2] * np.sqrt(vq), vq)
                   for s in product((-1, 1), repeat=3)]
        signal = h @ train
        expected = {"training_label_realization_mse_estimate": (p - signal) ** 2,
                    "signal_mismatch_mse_estimate": (signal - query) ** 2,
                    "interaction_estimate": 2 * (p - signal) * (signal - query),
                    "total_mse_estimate": (p - query) ** 2}
        for key, value in expected.items():
            np.testing.assert_allclose(np.mean([r[key] for r in samples], axis=0), value)
        for r in samples:
            np.testing.assert_allclose(sum(r[k] for k in expected if k != "total_mse_estimate"),
                                       r["total_mse_estimate"])

    def preflight_inputs(self):
        options = cell_options(208001, preflight=True)
        args = SimpleNamespace(optimizer_seed=208001,
                               branch_fit_seeds=list(options["branch_fit"]),
                               trigger_eval_seeds=list(options["trigger_eval"]),
                               **spec.input_results(208001, preflight=True),
                               **spec.sampling_options(preflight=True))
        _, pairs = diagnostic.load_cached_pairs(args)
        ridge = json.loads(args.prediction_result.read_text())["cells"][0]
        fresh = json.loads(args.fresh_result.read_text())["cells"][0]
        return args, pairs, ridge, fresh

    def test_saved_preflight_replays_without_parameter_changes(self):
        args, pairs, ridge, fresh = self.preflight_inputs()
        snapshot = deepcopy((pairs, ridge, fresh))
        result = diagnostic.run_cell(args)
        self.assertEqual(result["influence_linear_systems"], 2)
        for key in ("additional_primitive_steps", "controller_training_iterations",
                    "critic_parameter_updates", "evaluation_paths_used"):
            self.assertEqual(result[key], 0)
        np.testing.assert_allclose(result["metrics"]["total_mse_estimate"],
                                   fresh["metrics"]["conditional_mean_mse_estimate"]["ridge_averaged"])
        diagnostic.diagnose(pairs, ridge, fresh, fresh_replicates=4)
        self.assertEqual((pairs, ridge, fresh), snapshot)

    def test_rejects_path_leakage_changed_prediction_and_budget(self):
        _, pairs, ridge, fresh = self.preflight_inputs()
        fold = next(f for f in ridge["folds"] if f["label"] == "averaged")
        with self.assertRaisesRegex(ValueError, "path isolation"):
            diagnostic.frozen_operator(pairs, pairs, fold)
        bad = deepcopy(fresh)
        bad["rows"][0]["predictions"]["ridge_averaged"] += .1
        with self.assertRaisesRegex(ValueError, "frozen prediction"):
            diagnostic.diagnose(pairs, ridge, bad, fresh_replicates=4)
        with self.assertRaisesRegex(ValueError, "budget"):
            diagnostic.diagnose(pairs, ridge, fresh, fresh_replicates=64)
        bad = deepcopy(fresh)
        bad["rows"].append(bad["rows"][0])
        with self.assertRaisesRegex(ValueError, "exact frozen opportunities"):
            diagnostic.diagnose(pairs, ridge, bad, fresh_replicates=4)

    def test_tasks_stage_only_four_existing_caches_on_dynamic_cpu_pool(self):
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_ridge_error", root, preflight=preflight, protocol_spec=spec)
                self.assertEqual(task["cpu"], 1)
                self.assertEqual(task["ram_mb"], 1536)
                self.assertIsNone(task["require_node"])
                sources = spec.input_results(root, preflight=preflight)
                self.assertEqual(len(sources), 4)
                for path in sources.values():
                    self.assertIn(str(path.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
