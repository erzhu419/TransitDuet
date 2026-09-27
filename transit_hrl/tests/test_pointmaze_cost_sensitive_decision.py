from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_cost_sensitive_decision as decision
from scripts import pointmaze_cost_sensitive_decision_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class CostSensitiveDecisionTest(unittest.TestCase):
    def preflight(self):
        options = cell_options(208001, preflight=True)
        args = SimpleNamespace(optimizer_seed=208001, branch_fit_seeds=list(options["branch_fit"]),
                               trigger_eval_seeds=list(options["trigger_eval"]),
                               **spec.input_results(208001, preflight=True),
                               **spec.sampling_options(preflight=True))
        _, pairs = decision.load_cached_pairs(args)
        fresh = json.loads(args.fresh_result.read_text())["cells"][0]
        return args, pairs, fresh

    def test_logistic_gradient_matches_finite_difference(self):
        x = np.array([[1., 2.], [1., -1.], [1., .5]])
        signs, weights = np.array([1., -1., 0.]), np.array([.2, .8, 0.])
        theta = np.array([.3, -.7])
        _, gradient = decision.logistic_objective(theta, x, signs, weights)
        eps = 1e-6
        numeric = [(decision.logistic_objective(theta + eps * v, x, signs, weights)[0]
                    - decision.logistic_objective(theta - eps * v, x, signs, weights)[0]) / (2 * eps)
                   for v in np.eye(2)]
        np.testing.assert_allclose(gradient, numeric, rtol=1e-8, atol=1e-8)

    def test_cost_weighting_changes_rare_harm_decision_and_is_unit_invariant(self):
        x, y = np.ones((3, 1)), np.array([1., 1., -10.])
        weighted = decision.fit_model(x, y, "cost_sensitive")
        uniform = decision.fit_model(x, y, "uniform")
        self.assertLess(weighted["weights"][0], 0.)
        self.assertGreater(uniform["weights"][0], 0.)
        np.testing.assert_allclose(weighted["weights"], decision.fit_model(x, y * 1000, "cost_sensitive")["weights"])
        np.testing.assert_allclose(weighted["weights"], -np.array(decision.fit_model(x, -y, "cost_sensitive")["weights"]))

    def test_zero_labels_and_ridge_normal_equations(self):
        x = np.array([[1., 2.], [1., -1.], [1., .5]])
        for method in ("cost_sensitive", "uniform", "mse"):
            fit = decision.fit_model(x, np.zeros(3), method)
            np.testing.assert_equal(fit["weights"], [0., 0.])
            self.assertFalse(any(x @ fit["weights"] > 0))
        y = np.array([1., 2., -4.])
        fit = decision.fit_model(x, y, "mse")
        expected = np.linalg.solve(x.T @ x / 3 + np.eye(2), x.T @ (y / np.mean(abs(y))) / 3)
        np.testing.assert_allclose(fit["weights"], expected, rtol=1e-12, atol=1e-12)

    def test_predictions_use_only_online_features_and_train_normalization(self):
        _, pairs, _ = self.preflight()
        fold = decision.fit_fold(pairs[:1])
        matrix, names = decision.advantage_matrix(pairs[:1])
        self.assertEqual(len(names), 41)
        np.testing.assert_equal(fold["feature_mean"], matrix[0])
        np.testing.assert_equal(fold["feature_scale"], np.ones(41))
        changed = deepcopy(pairs[1:])
        for r in changed:
            r["window_ise_advantage"] += 1000
            r["replicate_tail_advantages"] = [1000.] * 4
            r["now_endpoint"]["features"] = [1000.]
            r["wait_endpoint"]["features"] = [-1000.]
        before, after = decision.predict(fold, pairs[1:]), decision.predict(fold, changed)
        for method in decision.LEARNED:
            np.testing.assert_equal(before[method], after[method])

    def test_fresh_labels_and_held_path_labels_do_not_enter_fits(self):
        args, pairs, fresh = self.preflight()
        def run(p, f):
            return decision.qualify(p, f, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                                    eval_seeds=args.trigger_eval_seeds, fresh_replicates=4)
        snapshot = deepcopy((pairs, fresh))
        original = run(pairs, fresh)
        altered = deepcopy(fresh)
        for r in altered["rows"]:
            r["replicate_tail_advantages"] = [a + 10 for a in r["replicate_tail_advantages"]]
        result = run(pairs, altered)
        self.assertEqual(original["folds"], result["folds"])
        self.assertEqual([r["scores"] for r in original["rows"]], [r["scores"] for r in result["rows"]])
        self.assertNotEqual(original["rows"][0]["scoring_total_mean"], result["rows"][0]["scoring_total_mean"])
        changed = deepcopy(pairs)
        changed[0]["window_ise_advantage"] += 100
        changed[0]["replicate_tail_advantages"] = [100.] * 4
        altered = run(changed, fresh)
        held = pairs[0]["seed"]
        self.assertEqual(next(f for f in original["folds"] if f["held_out_path"] == held),
                         next(f for f in altered["folds"] if f["held_out_path"] == held))
        self.assertEqual((pairs, fresh), snapshot)
        self.assertEqual(original["decision_fits"], 8)
        for key in ("additional_primitive_steps", "controller_training_iterations", "controller_policy_updates", "evaluation_paths_used"):
            self.assertEqual(original[key], 0)

    def test_paired_benefit_sign_mc_se_and_all_control_gate(self):
        rows = []
        for now, total in ((True, 2.), (False, -1.)):
            rows.append({"scoring_total_mean": total, "scoring_mean_variance": .25,
                         "now_choices": {"cost_sensitive": now, "short_window": not now,
                                         "uniform": not now, "mse": not now,
                                         "always_wait": False, "always_now": True}})
        result = decision.summarize(rows)
        self.assertTrue(result["development_gate_passed"])
        self.assertEqual(result["candidate_vs_control"]["short_window"]["mean_ise_benefit"], 1.5)
        self.assertEqual(result["candidate_vs_control"]["always_wait"]["conditional_mc_standard_error"], .25)
        for r in rows:
            r["now_choices"]["uniform"] = r["now_choices"]["cost_sensitive"]
        self.assertFalse(decision.summarize(rows)["development_gate_passed"])

    def test_rejects_missing_states_wrong_budget_and_seed_overlap(self):
        args, pairs, fresh = self.preflight()
        options = dict(root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                       eval_seeds=args.trigger_eval_seeds, fresh_replicates=4)
        missing = deepcopy(fresh)
        missing["rows"].pop()
        with self.assertRaisesRegex(ValueError, "whole-path"):
            decision.qualify(pairs, missing, **options)
        with self.assertRaisesRegex(ValueError, "label budget"):
            decision.qualify(pairs, fresh, **{**options, "fresh_replicates": 64})
        with self.assertRaisesRegex(ValueError, "whole-path"):
            decision.qualify(pairs, fresh, **{**options, "eval_seeds": args.branch_fit_seeds})

    def test_runner_and_scheduler_keep_three_caches_and_single_dynamic_cpu(self):
        args, _, _ = self.preflight()
        self.assertEqual(decision.run_cell(args)["decision_fits"], 8)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_cost_sensitive", root, preflight=preflight, protocol_spec=spec)
                self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 1536, None))
                self.assertEqual(len(spec.input_results(root, preflight=preflight)), 3)
                for p in spec.input_results(root, preflight=preflight).values():
                    self.assertIn(str(p.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
