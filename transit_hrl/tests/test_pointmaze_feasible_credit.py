import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_feasible_credit as experiment
from freq_hrl.experiments import pointmaze_learned_plan as learned
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_feasible_credit_stage79_spec as spec
from scripts.submit_pointmaze_feasible_credit_stage79_scheduleurm import task_specification, qualification_task
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_upper_paths import predictor


class FeasibleCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self):
        torch.manual_seed(79)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False, lower_value_state_dim=392))
        return learned.make_model(model)

    def collect(self, model, args, seeds, *, period=50, weights=None, alpha=.02, variant="base", collect=True):
        experiment.init_worker(model.config, args)
        weights = experiment.joint.inference_weights(model) if weights is None else weights
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        with patch.object(experiment.joint, "_make_task", side_effect=lambda **kw: DenseTask()), patch.object(
                experiment.joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
            return [experiment.worker_native((weights, s, variant, period, predictor(), alpha, envelope, collect)) for s in seeds]

    def test_true_episode_upper_and_lower_mc_match_native_undiscounted_reward(self):
        model = self.source()
        args = spec.arguments(310011, preflight=True)
        for p in spec.PERIODS:
            batch, row = self.collect(model, args, [79090001], period=p)[0]
            returns = experiment.task_returns(batch, p, args.horizon, row["episode_return"])
            np.testing.assert_allclose(returns["upper"], returns["lower"][::p], atol=.002)
            self.assertEqual(experiment._WORKER[0].config.gamma, 1.)
            self.assertNotEqual(model.config.gamma, 1.)
            bad = copy.deepcopy(batch)
            bad.lower.done[p-1] = 1.
            with self.assertRaises(ValueError):experiment.task_returns(bad, p, args.horizon, row["episode_return"])
            bad = copy.deepcopy(batch)
            bad.upper.reward *= .99
            with self.assertRaises(AssertionError):experiment.task_returns(bad, p, args.horizon, row["episode_return"])

    def test_fresh_crossfit_scores_leave_source_and_optimizers_frozen_and_count_passes(self):
        model = self.source()
        before = copy.deepcopy(model.state_dict())
        args = spec.arguments(310011, preflight=True)
        roles = spec.seed_roles(310011, preflight=True)
        batches = {name: self.collect(model, args, roles["credit_" + name]) for name in ("A", "B")}
        cost = dict.fromkeys(spec.budget(preflight=True), 0)
        candidates, credit = experiment.credit_directions(model, batches, period=50, horizon=args.horizon, cost=cost)
        experiment.curves.support.assert_frozen(model, before)
        self.assertEqual(set(candidates), {a + "_" + s for a in ("upper", "lower") for s in ("plus", "minus")})
        self.assertEqual(cost["objective_checks"], 4)
        self.assertEqual(cost["mc_calls"], 8)
        self.assertEqual(cost["actor_score_forward_batches"], 8)
        self.assertEqual(cost["actor_score_backward_batches"], 24)
        self.assertEqual(cost["fisher_jvp_batches"], 3)
        self.assertEqual(cost["exact_kl_forward_batches"], 6)
        for name in ("A", "B"):
            rate = np.mean([r["episode_return"] for _, r in batches[name]]) / args.horizon
            self.assertEqual(credit["baseline_reward_rates"][name], rate)
        for name in ("upper", "lower"):
            self.assertEqual(credit["actors"][name]["geometry"]["radius_check"], "passed")
            untouched = "lower_actor" if name == "upper" else "upper_actor"
            torch.testing.assert_close(candidates[name + "_plus"][untouched], before[untouched], atol=0, rtol=0)
            torch.testing.assert_close(candidates[name + "_minus"][untouched], before[untouched], atol=0, rtol=0)

    def test_upper_update_pairs_underlying_gaussian_noise_not_proposals(self):
        model, seeds = self.source(), [79095001]
        args = spec.arguments(310011, preflight=True)
        weights = experiment.joint.inference_weights(model)
        changed = copy.deepcopy(weights)
        changed["upper_actor"]["net.2.bias"] += .03
        changed["upper_actor"]["log_std"] -= .02
        evaluation = {}
        for variant in spec.VARIANTS:
            w = changed if variant == "upper_plus" else weights
            pairs = self.collect(model, args, seeds, weights=w, variant=variant,
                alpha=0. if variant == "zero" else .02, collect=False)
            self.assertIsNone(pairs[0][0])
            evaluation[variant] = [r for _, r in pairs]
        experiment.paired_effects(50, evaluation, seeds)
        evaluation["upper_plus"][0]["upper_standard_noise"][0][0] += .001
        with self.assertRaises(AssertionError):experiment.paired_effects(50, evaluation, seeds)

    def test_preregistered_budget_seeds_and_unpinned_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"] * 8, b["native_steps"] * 8), (3584, 4300800))
        self.assertEqual(len(spec.ENDPOINTS), 18)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertEqual(len(set(sum(roles.values(), []))), sum(map(len, roles.values())))
                task = task_specification("unit_stage79", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage79", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])

    def test_all_roots_and_all18_contrasts_share_one_corrected_family(self):
        cells = [{"root": root, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for root in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda cell, **kw: cell), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), patch.object(experiment.np, "quantile", wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(quantile.call_args.args[1], [.05 / 36, 1 - .05 / 36])
        self.assertEqual(set(summary["endpoints"]), set(spec.ENDPOINTS))
        self.assertTrue(all(e["ci"] == [2., 2.] for e in summary["endpoints"].values()))
        self.assertEqual(summary["native_trial_prerequisite"], "hold_Stage67_credit_gate_unchanged")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
