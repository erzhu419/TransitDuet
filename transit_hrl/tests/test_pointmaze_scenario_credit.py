import copy
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_scenario_credit as experiment
from scripts import pointmaze_scenario_credit_stage80_spec as spec
from scripts.submit_pointmaze_scenario_credit_stage80_scheduleurm import task_specification, qualification_task
from test_pointmaze_feasible_credit import FeasibleCreditTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_upper_paths import predictor


class ScenarioCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def collect(self, model, args, roster, *, collect=True, variant="base", weights=None):
        experiment.native.init_worker(model.config, args)
        weights = experiment.native.joint.inference_weights(model) if weights is None else weights
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        with patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), patch.object(
                experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
            pairs = [experiment.worker_native((weights, s["scenario_seed"], n, variant, 50, predictor(),
                0. if variant == "zero" else .02, envelope, collect)) for s in roster for n in s["noise_seeds"]]
        return [pairs[i:i+2] for i in range(0, len(pairs), 2)]

    def test_leave_other_out_excludes_own_return_and_removes_scenario_offset_without_bias(self):
        r = np.arange(24, dtype=float).reshape(2, 3, 4)
        expected = r - np.stack([(r[:, [j for j in range(3) if j != i]].mean(1)) for i in range(3)], axis=1)
        np.testing.assert_array_equal(experiment.leave_other_out(r), expected)
        np.testing.assert_array_equal(experiment.leave_other_out(r + np.array([100., -40.])[:, None, None]), expected)
        estimates = []
        for a, b in product((-1., 1.), repeat=2):
            signals = experiment.leave_other_out(np.array([[[a+100.], [b+100.]]]))
            estimates.append(float((signals * np.array([[[a], [b]]])).mean()))
        self.assertEqual(np.mean(estimates), 1.)
        with self.assertRaises(ValueError):experiment.leave_other_out(np.ones((2, 1, 4)))

    def test_group_noise_counts_scenarios_not_dependent_rollouts(self):
        g = np.array([[1., 3.], [3., 1.], [5., 8.], [7., 6.]])
        observed = experiment.group_noise(g, 2, 2)
        expected = experiment.native.independent.gradient_noise(np.array([[2., 2.], [6., 7.]]))
        self.assertEqual(observed, expected)
        self.assertEqual(observed["episodes"], 2)

    def test_scenario_pair_has_identical_exogenous_history_independent_actions_and_exact_replay(self):
        model = FeasibleCreditTest().source()
        args = spec.arguments(310011, preflight=True)
        roster = spec.seed_roles(310011, preflight=True)["credit_A"][:1]
        pairs = self.collect(model, args, roster)[0]
        experiment.check_scenario_pair(pairs, roster[0])
        self.assertFalse(np.array_equal(pairs[0][0].lower.action, pairs[1][0].lower.action))
        replay = self.collect(model, args, roster)[0]
        for (first, row), (again, replay_row) in zip(pairs, replay):
            np.testing.assert_array_equal(first.lower.action, again.lower.action)
            self.assertEqual(row["episode_return"], replay_row["episode_return"])
        bad = copy.deepcopy(pairs)
        bad[1][0].lower.state[100, 7] += 1.
        with self.assertRaises(AssertionError):experiment.check_scenario_pair(bad, roster[0])
        bad = copy.deepcopy(pairs)
        bad[1][1]["lower_seed"] = bad[0][1]["lower_seed"]
        with self.assertRaises(ValueError):experiment.check_scenario_pair(bad, roster[0])

    def test_same_data_both_baselines_freeze_source_and_count_four_backward_signals(self):
        model = FeasibleCreditTest().source()
        before = copy.deepcopy(model.state_dict())
        args, roles = spec.arguments(310011, preflight=True), spec.seed_roles(310011, preflight=True)
        batches = {name: self.collect(model, args, roles["credit_" + name]) for name in ("A", "B")}
        cost = dict.fromkeys(spec.budget(preflight=True), 0)
        candidates, credit = experiment.credit_directions(model, batches, period=50, horizon=args.horizon, cost=cost)
        experiment.native.curves.support.assert_frozen(model, before)
        self.assertEqual(set(candidates), set(spec.VARIANTS) - {"base", "zero"})
        self.assertEqual(cost["objective_checks"], 8)
        self.assertEqual(cost["mc_calls"], 16)
        self.assertEqual(cost["actor_score_forward_batches"], 16)
        self.assertEqual(cost["actor_score_backward_batches"], 64)
        self.assertEqual(cost["fisher_jvp_batches"], 8)
        self.assertEqual(cost["exact_kl_forward_batches"], 16)
        self.assertEqual(cost["actor_parameter_perturbations"], 8)
        for a in ("upper", "lower"):
            self.assertEqual(set(credit["actors"][a]["methods"]), set(spec.METHODS))
            for m in spec.METHODS:
                for row in credit["actors"][a]["methods"][m]["scenario_group_noise"].values():
                    self.assertEqual(row["episodes"], 2)
        effects = experiment.credit_effects(50, credit)
        self.assertEqual(len(effects), 4)
        self.assertTrue(np.isfinite(list(effects.values())).all())

    def test_preregistered_budget_disjoint_roles_completion_only_dynamic_scheduler(self):
        budget = spec.budget(preflight=False)
        self.assertEqual((budget["native_episodes"] * 8, budget["native_steps"] * 8), (6144, 7372800))
        self.assertEqual(budget["scenario_pair_checks"] * 8, 512)
        self.assertEqual(budget["actor_score_forward_batches"] * 8, 3072)
        self.assertEqual(budget["actor_score_backward_batches"] * 8, 12288)
        self.assertEqual(len(spec.ENDPOINTS), 46)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                scenarios = [s["scenario_seed"] for name in ("A", "B") for s in roles["credit_" + name]]
                noises = [n for name in ("A", "B") for s in roles["credit_" + name] for n in s["noise_seeds"]]
                all_roles = scenarios + noises + roles["native_evaluation"]
                self.assertEqual(len(set(all_roles)), len(all_roles))
                task = task_specification("unit_stage80", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage80", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])

    def test_all_reward_and_credit_endpoints_share_one_root_bootstrap_family(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda cell, **kw: cell), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.native.np, "quantile", wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(quantile.call_args.args[1], [.05 / 92, 1 - .05 / 92])
        self.assertEqual(set(summary["endpoints"]), set(spec.ENDPOINTS))
        self.assertEqual(summary["cost"]["native_episodes"], 6144)
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
