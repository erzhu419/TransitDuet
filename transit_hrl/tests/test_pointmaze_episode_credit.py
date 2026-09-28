import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_episode_credit as experiment
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch, SMDPPPOConfig
from scripts import pointmaze_episode_credit_stage46_spec as spec
from scripts.submit_pointmaze_episode_credit_stage46_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class EpisodeCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model_batch(self):
        torch.manual_seed(46)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1, lower_state_dim=1, upper_action_dim=1, lower_action_dim=1,
            hidden_dim=0, lower_cost_critic=False, entropy_coef=0., epochs=1, minibatch_size=4))
        with torch.no_grad():
            model.lower_actor.net[0].weight.zero_()
            model.lower_actor.net[0].bias.zero_()
        state, action = np.ones((4, 1), dtype=np.float32), np.array([[-1.], [-1.], [1.], [1.]], dtype=np.float32)
        with torch.no_grad():
            logp, _ = model.lower_actor.log_prob_entropy(torch.from_numpy(state), torch.from_numpy(action))
        batch = LevelTrajectoryBatch(state=state, action=action, old_logp=logp.numpy(), old_value=np.zeros(4, dtype=np.float32),
            reward=np.array([-1., -1., 1., 1.], dtype=np.float32), done=np.ones(4, dtype=np.float32),
            duration=np.ones(4, dtype=np.int64))
        return model, batch

    def core_update(self, model, batch, advantage=None):
        np.random.seed(46)
        return model._update_level(level="lower", batch=batch, actor=model.lower_actor, value_net=model.lower_value,
            actor_optimizer=model.lower_actor_optimizer, value_optimizer=model.lower_value_optimizer,
            actor_advantage=advantage)

    def test_explicit_gae_is_identical_and_zero_actor_credit_keeps_original_critic_update(self):
        model, batch = self.model_batch()
        same, zero = copy.deepcopy(model), copy.deepcopy(model)
        advantage, _ = model._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        original_actor = copy.deepcopy(model.lower_actor.state_dict())
        metrics = self.core_update(model, batch)
        self.assertEqual(metrics, self.core_update(same, batch, advantage))
        self.assertEqual(model.config, same.config)
        torch.testing.assert_close({k: v for k, v in model.state_dict().items() if k != "config"},
                                   {k: v for k, v in same.state_dict().items() if k != "config"}, atol=0, rtol=0)
        self.core_update(zero, batch, np.zeros(batch.size, dtype=np.float32))
        torch.testing.assert_close(zero.lower_actor.state_dict(), original_actor, atol=0, rtol=0)
        torch.testing.assert_close(zero.lower_value.state_dict(), model.lower_value.state_dict(), atol=0, rtol=0)
        torch.testing.assert_close(zero.lower_value_optimizer.state_dict(), model.lower_value_optimizer.state_dict(), atol=0, rtol=0)

    def test_actor_override_changes_gradient_direction_and_rejects_wrong_shape_before_steps(self):
        model, batch = self.model_batch()
        opposite = copy.deepcopy(model)
        advantage, _ = model._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        self.core_update(model, batch)
        self.core_update(opposite, batch, -advantage)
        self.assertGreater(float(model.lower_actor.net[0].bias[0]), 0.)
        self.assertLess(float(opposite.lower_actor.net[0].bias[0]), 0.)
        for invalid in (np.ones(batch.size - 1), np.full(batch.size, np.nan)):
            fresh, _ = self.model_batch()
            with self.assertRaisesRegex(ValueError, "actor_advantage"):
                self.core_update(fresh, batch, invalid)
            self.assertFalse(fresh.lower_actor_optimizer.state)
            self.assertFalse(fresh.lower_value_optimizer.state)

    def test_loo_complete_returns_and_exact_delayed_task_score(self):
        np.testing.assert_array_equal(experiment.score_targets([[1, 2, 3], [4, 5, 6]]), [[-9, -6, -3], [9, 6, 3]])
        paths = list(itertools.product((0, 1), repeat=2))
        estimates = []
        for own, other in itertools.product(paths, repeat=2):
            # Reward after renewal depends on both earlier and later actions.
            rewards = [[0., float(a[0] * a[1])] for a in (own, other)]
            centered = experiment.score_targets(rewards)
            estimates.append(float(np.dot(np.asarray(own) - .5, centered[0])))
        self.assertAlmostEqual(float(np.mean(estimates)), .25)
        with self.assertRaisesRegex(ValueError, "two complete episodes"):
            experiment.score_targets([[1., 2.]])

    def test_native_credit_inputs_and_critic_target_identity(self):
        model, batch = self.model_batch()
        mc = copy.deepcopy(model)
        rewards = batch.reward.reshape(2, 2).astype(np.float64)
        np.random.seed(46)
        left = experiment.update(model, batch, rewards, "gae")
        np.random.seed(46)
        right = experiment.update(mc, batch, rewards, "episode_mc")
        for key in ("critic_target_mean", "critic_target_std", "task_return_mean", "gae_mc_normalized_mse", "gae_mc_normalized_dot"):
            self.assertEqual(left[key], right[key])
        torch.testing.assert_close(model.lower_value.state_dict(), mc.lower_value.state_dict(), atol=0, rtol=0)
        with self.assertRaises(AssertionError):
            experiment.update(model, batch, rewards + 1., "episode_mc")
        with self.assertRaisesRegex(ValueError, "two complete native episodes"):
            experiment.update(model, batch, batch.reward[None, :], "episode_mc")

    def test_existing_checkpoint_native_style_pipeline(self):
        original = spec.previous.source.source
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=8, lower_cost_critic=False, epochs=1, minibatch_size=128))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, source_checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, source_checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(source_checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_file.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec.previous.source, "ROOT", directory), patch.object(original, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                for method in spec.METHODS:
                    calibration.train(310001, method, preflight=True, output=spec.source_result(310001, method, preflight=True),
                                      specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 15600)
                self.assertEqual(result["native_trace_audits"], 52)
                self.assertEqual(result["seed_roles"]["training"], [11090001, 11090002, 11090003, 11090004])
                self.assertTrue(all(p["first_critic_update"] == "passed" for p in result["first_batch_pairs"].values()))
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["method_cost"]["primitive_steps"], 15600)
                self.assertEqual(summary["optimizer_steps"], {"actor_optimizer_steps": 40, "value_optimizer_steps": 40})
                self.assertNotIn("primary_endpoints", summary)

    def test_seed_roles_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for values in roles.values() for s in values}
                self.assertEqual(len(new), sum(len(v) for v in roles.values()))
                for prior in (spec.previous, spec.previous.source, spec.previous.source.source):
                    old = {s for values in prior.seed_roles(root, preflight=preflight).values() for s in values}
                    self.assertFalse(new.intersection(old))
                self.assertFalse(new.intersection(seen))
                seen.update(new)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 7680000)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 6400)
        for preflight in (True, False):
            task = task_specification("unit_stage46", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)

    def synthetic_results(self):
        results = []
        opt, budget = spec.options(preflight=False), spec.budget(preflight=False)
        for index, root in enumerate(spec.roots(preflight=False)):
            roles, training = spec.seed_roles(root, preflight=False), {}
            for method in spec.METHODS:
                training[method] = {}
                for treatment in spec.TREATMENTS:
                    history = training[method][treatment] = []
                    for iteration in range(1, opt["learning_iterations"] + 1):
                        begin = (iteration - 1) * opt["rollouts_per_iteration"]
                        seeds = roles["training"][begin:begin + opt["rollouts_per_iteration"]]
                        history.append({"iteration": iteration, "primitive_steps": 9600,
                            "actor_credit": treatment, "critic_credit": "original_task_option_gae", "native_episodes": 8,
                            "actor_optimizer_steps": 2, "value_optimizer_steps": 2,
                            **{k: .1 for k in ("actor_advantage_mean", "actor_advantage_std", "critic_target_mean", "critic_target_std",
                                              "task_return_mean", "gae_mc_normalized_mse", "gae_mc_normalized_dot")},
                            "inference_counts": {"upper_inference_calls": 8, "lower_inference_calls": 9600, "gate_inference_calls": 8},
                            "rollout_sampling": [{"seed": seed, "policy_seed": spec.policy_seed(root, seed),
                                **{k: v for k, v in spec.rollout_arguments(root, seed, phase="train", mode="training").items() if k != "sample"}} for seed in seeds]})
            evaluation = {}
            for policy in spec.POLICIES:
                value = 100. if policy == "frozen" else 101. + index if policy.endswith(":episode_mc") else 95. - index
                evaluation[policy] = {str(i): {mode: [{**{k: value for k in spec.METRICS}, "episode_length": 1200,
                    "upper_inference_calls": 1, "lower_inference_calls": 1200, "gate_inference_calls": 1,
                    "seed": seed, "policy_seed": spec.policy_seed(root, seed), "deployment_mode": mode,
                    **{k: v for k, v in spec.rollout_arguments(root, seed, phase="eval", mode=mode).items() if k != "sample"}}
                    for seed in roles["evaluation"]] for mode in spec.MODES}
                    for i in ((0,) if policy == "frozen" else spec.snapshots(preflight=False))}
            results.append({"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "root": root, "preflight": False, "options": opt, "seed_roles": roles, "budget": budget,
                "training": training, "evaluation_rows": evaluation, "fixed_upper_gate_networks": "passed",
                "first_batch_pairs": {m: {"status": "passed", "transitions": 9600, "native_task_rewards": "passed", "first_critic_update": "passed"} for m in spec.METHODS},
                "expected_steps_per_update": {m: 2 for m in spec.METHODS}, "native_trace_audits": budget["native_trace_audits"],
                "inference_counts": {phase: {"primitive_steps": budget[key], "lower_inference_calls": budget[key],
                    "upper_inference_calls": budget[key] // 1200, "gate_inference_calls": budget[key] // 1200}
                    for phase, key in (("train", "training_primitive_steps"), ("eval", "evaluation_primitive_steps"))}})
        return results

    def test_independent_statistics_and_credit_accounting(self):
        results = self.synthetic_results()
        summary = experiment.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 8})
        for method in spec.METHODS:
            self.assertAlmostEqual(summary["primary_endpoints"][method + ":mc_gae_final"]["mean"], 13.)
            self.assertEqual(summary["primary_endpoints"][method + ":mc_frozen_final"]["effect"], "positive")
            self.assertEqual(summary["primary_endpoints"][method + ":gae_frozen_final"]["effect"], "negative")
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            experiment.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"][spec.METHODS[0]]["episode_mc"][0]["critic_target_mean"] += .1
        with self.assertRaisesRegex(ValueError, "paired initial targets differ"):
            experiment.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"][spec.METHODS[0]]["episode_mc"][1]["actor_optimizer_steps"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer or target accounting"):
            experiment.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
