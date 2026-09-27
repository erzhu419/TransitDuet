from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_contextual_pair_diagnostic as diagnostic
from freq_hrl.experiments.pointmaze_fresh_future_diagnostic import fixed_comparisons
from scripts import pointmaze_contextual_pair_diagnostic_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class ContextualPairDiagnosticTest(unittest.TestCase):
    def test_shared_context_modulates_contrast_and_preserves_antisymmetry(self):
        names = ["physical_0", "force_current_0", "remaining_fraction"]
        fold = {"feature_names": names, "feature_mean": [0., 0., 0.], "feature_scale": [1., 1., 1.]}
        pairs = [{"seed": 10 + i, "check_step": 20,
                  "now_endpoint": {"features": [0., c, 1.], "feature_names": names},
                  "wait_endpoint": {"features": [1., c, 1.], "feature_names": names}}
                 for i, c in enumerate((2., -2., 0.))]
        result = diagnostic.designs(pairs, fold, root=1)
        np.testing.assert_equal(result["linear"], [[1., 0., 0.]] * 3)
        np.testing.assert_allclose(result["contextual"][:, -1], np.tanh([2., -2., 0.]))
        for r in pairs:
            r["now_endpoint"], r["wait_endpoint"] = r["wait_endpoint"], r["now_endpoint"]
        swapped = diagnostic.designs(pairs, fold, root=1)
        for method in result:
            np.testing.assert_allclose(swapped[method], -result[method])
        for method in diagnostic.METHODS:
            self.assertTrue(np.all(np.linalg.norm(result[method][:, 3:], axis=1) <= 1.))
        np.testing.assert_allclose(np.linalg.norm(result["contextual"], axis=1),
                                   np.linalg.norm(result["random_context"], axis=1))

    def test_dual_solution_matches_weighted_primal_ridge(self):
        rng = np.random.default_rng(23)
        train, query, target = rng.normal(size=(4, 7)), rng.normal(size=(2, 7)), rng.normal(size=4)
        weights = np.array([.1, .2, .3, .4])
        pred, fit = diagnostic.fit_design(train, query, target, weights)
        gram = train.T @ (weights[:, None] * train)
        system = gram + np.eye(7)
        coefficients = np.linalg.solve(system, train.T @ (weights * target))
        np.testing.assert_allclose(pred, query @ coefficients, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(fit["effective_degrees_of_freedom"], np.trace(np.linalg.solve(system, gram)))
        self.assertAlmostEqual(fit["training_mse"], np.sum(weights * (train @ coefficients - target) ** 2))

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

    def test_real_schema_capacity_and_training_only_normalization(self):
        _, pairs, ridge, _ = self.preflight_inputs()
        fold = next(f for f in ridge["folds"] if f["label"] == "averaged")
        snapshot = deepcopy(fold)
        state, context = diagnostic.feature_groups(fold["feature_names"])
        self.assertEqual((len(state), len(context)), (13, 25))
        original = diagnostic.designs(pairs, fold, root=208001)
        self.assertEqual(original["contextual"].shape, (2, 365))
        changed = deepcopy(pairs)
        for endpoint in ("now_endpoint", "wait_endpoint"):
            changed[-1][endpoint]["features"][context[0]] += 1e6
        altered = diagnostic.designs(changed, fold, root=208001)
        self.assertEqual(fold, snapshot)
        for method in original:
            np.testing.assert_equal(original[method][0], altered[method][0])

    def test_scoring_labels_do_not_change_fits_and_inputs_stay_frozen(self):
        args, pairs, ridge, fresh = self.preflight_inputs()
        snapshot = deepcopy((pairs, ridge, fresh))
        result = diagnostic.run_cell(args)
        changed = deepcopy(fresh)
        for row in changed["rows"]:
            row["replicate_tail_advantages"] = [v + 1. for v in row["replicate_tail_advantages"]]
        changed["metrics"] = fixed_comparisons(changed["rows"])
        altered = diagnostic.qualify(pairs, ridge, changed, root=208001, fresh_replicates=4)
        self.assertEqual([r["predictions"] for r in result["rows"]], [r["predictions"] for r in altered["rows"]])
        self.assertEqual(result["folds"], altered["folds"])
        self.assertNotEqual(result["metrics"], altered["metrics"])
        self.assertEqual((pairs, ridge, fresh), snapshot)
        self.assertEqual(result["ridge_fits"], 4)
        for key in ("additional_primitive_steps", "controller_training_iterations", "critic_optimizer_steps", "evaluation_paths_used"):
            self.assertEqual(result[key], 0)

    def test_rejects_fold_leakage_and_scoring_budget_change(self):
        _, pairs, ridge, fresh = self.preflight_inputs()
        bad = deepcopy(ridge)
        fold = next(f for f in bad["folds"] if f["label"] == "averaged")
        fold["training_paths"].append(fold["held_out_path"])
        with self.assertRaisesRegex(ValueError, "training paths"):
            diagnostic.qualify(pairs, bad, fresh, root=208001, fresh_replicates=4)
        with self.assertRaisesRegex(ValueError, "label budget"):
            diagnostic.qualify(pairs, ridge, fresh, root=208001, fresh_replicates=64)

    def test_gate_requires_all_controls_and_retains_negative_estimates(self):
        rows = [{"repeat_mean": 1., "mean_variance": .2,
                 "predictions": {"zero": 0., "linear": 2., "contextual": 1., "random_context": 2.}}]
        self.assertTrue(diagnostic.score(rows)["development_gate_passed"])
        self.assertEqual(diagnostic.score(rows)["conditional_mean_mse_estimate"]["contextual"], -.2)
        for control in ("linear", "random_context"):
            changed = deepcopy(rows)
            changed[0]["predictions"][control] = 1.
            self.assertFalse(diagnostic.score(changed)["development_gate_passed"])

    def test_scheduler_uses_existing_caches_and_single_dynamic_cpu(self):
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_contextual_pair", root, preflight=preflight, protocol_spec=spec)
                self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 1536, None))
                for source in spec.input_results(root, preflight=preflight).values():
                    self.assertIn(str(source.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
