import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.experiments import pointmaze_update_direction as direction
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch, SMDPPPOConfig
from scripts import pointmaze_update_direction_stage43_spec as spec
from scripts.submit_pointmaze_update_direction_stage43_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class UpdateDirectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, state_dim=1):
        torch.manual_seed(43)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=state_dim, lower_state_dim=state_dim, upper_action_dim=1, lower_action_dim=1,
            hidden_dim=0, lower_cost_critic=False, init_log_std=0., epochs=1, minibatch_size=128))

    def batch(self, model):
        state = np.ones((4, 1), dtype=np.float32)
        action = np.array([[-1.], [-1.], [1.], [1.]], dtype=np.float32)
        with torch.no_grad():
            logp, _ = model.lower_actor.log_prob_entropy(torch.from_numpy(state), torch.from_numpy(action))
        return LevelTrajectoryBatch(state=state, action=action, old_logp=logp.numpy(),
            old_value=np.zeros(4, dtype=np.float32), reward=np.array([1., -1., .5, -.5], dtype=np.float32),
            done=np.ones(4, dtype=np.float32), duration=np.ones(4, dtype=np.int64))

    def test_clipped_surrogate_matches_ppo_formula_without_updates(self):
        before = self.model()
        after = self.model()
        after.load_state_dict(copy.deepcopy(before.state_dict()))
        with torch.no_grad():
            after.lower_actor.net[0].bias += .5
        batch = self.batch(before)
        original = joint.inference_weights(before)
        terms = direction.surrogate_terms(before, after.lower_actor, batch)
        with torch.no_grad():
            logp, entropy = after.lower_actor.log_prob_entropy(torch.from_numpy(batch.state), torch.from_numpy(batch.action))
        ratio = np.exp(np.clip(logp.numpy() - batch.old_logp, -20., 20.))
        advantage = batch.reward / (batch.reward.std() + 1e-8)
        clipped = np.clip(ratio, 1. - before.config.clip_ratio, 1. + before.config.clip_ratio)
        self.assertAlmostEqual(terms["clipped_surrogate"], np.minimum(ratio * advantage, clipped * advantage).mean(), places=7)
        self.assertAlmostEqual(terms["actor_objective"], terms["clipped_surrogate"] + before.config.entropy_coef * entropy.mean().item(), places=7)
        for name, values in original.items():
            torch.testing.assert_close(getattr(before, name).state_dict(), values, rtol=0, atol=0)
        same = direction.surrogate_terms(before, before.lower_actor, batch)
        self.assertEqual(same["clipped_surrogate"] - direction.surrogate_terms(before, before.lower_actor, batch)["clipped_surrogate"], 0.)

    def test_full_episode_score_has_independent_baseline_and_analytic_direction(self):
        rewards = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_array_equal(direction.score_targets(rewards), [[-4., -2.], [4., 2.]])
        before = self.model()
        with torch.no_grad():
            before.lower_actor.net[0].weight.zero_()
            before.lower_actor.net[0].bias.zero_()
        batch = self.batch(before)
        original = joint.inference_weights(before)
        for shift in (.1, -.1, 0.):
            after = self.model()
            after.load_state_dict(copy.deepcopy(before.state_dict()))
            with torch.no_grad():
                after.lower_actor.net[0].bias += shift
            measured = direction.task_direction(before, after, batch, rewards)
            self.assertAlmostEqual(measured["heldout_task_direction"], 6. * shift, places=6)
        for name, values in original.items():
            torch.testing.assert_close(getattr(before, name).state_dict(), values, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in before.lower_actor.parameters()))
        self.assertFalse(before.lower_actor_optimizer.state)
        with self.assertRaisesRegex(ValueError, "two complete episodes"):
            direction.score_targets([[1., 2.]])
        with self.assertRaisesRegex(ValueError, "transitions differ"):
            direction.task_direction(before, before, batch, np.ones((2, 3)))

    def test_reconstruction_matches_source_and_detects_wrong_batch(self):
        before, after = self.model(), self.model()
        batch = self.batch(before)
        advantage, target = before._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        credit = {"task_reward_sum": 4., "option_count": 4}
        row = {"seed": 1, "policy_seed": 2, "lower_training_credit": credit}
        saved = {"primitive_steps": 4, "reward_sum": 0., "task_reward_sum": 4., "done_count": 4,
                 "option_count": 4, "old_value_mean": 0., "gae_target_mean": float(target.mean()),
                 "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std()),
                 "rollout_sampling": [{"seed": 1, "policy_seed": 2}],
                 "policy_drift": calibration.policy_drift(after.lower_actor, before.lower_actor, batch.state)}
        result = {"options": {"warmup_iterations": 0}, "training": [saved], "first_learning_credit": {"seed": 1, **credit}}
        self.assertEqual(direction.check_first_batch(before, after, batch, [row], result)["status"], "passed")
        changed = copy.deepcopy(batch)
        changed.old_value += .1
        with self.assertRaises(AssertionError):
            direction.check_first_batch(before, after, changed, [row], result)

    def test_existing_checkpoint_pipeline_fresh_native_style_paths_and_accounting(self):
        torch.manual_seed(43)
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
            with patch.object(spec, "ROOT", directory), patch.object(spec.source, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(direction, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                for method in spec.METHODS:
                    output = spec.source_result(310001, method, preflight=True)
                    calibration.train(310001, method, preflight=True, output=output,
                                      specification=spec.source, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = direction.diagnose(310001, preflight=True, output=directory / "diagnostic/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 7200)
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertEqual(result["native_trace_audits"], 24)
                self.assertEqual(result["zero_displacement_task_direction"], 0.)
                self.assertEqual(set(result["methods"]), set(spec.METHODS))
                self.assertEqual(result["seed_roles"]["evaluation"], [10793001, 10793002])
                for rows in result["evaluation_rows"]["frozen"].values():
                    self.assertTrue(all(not r["upper_sample"] and not r["gate_sample"] for r in rows))
                summary = direction.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["method_cost"]["primitive_steps"], 7200)
                self.assertNotIn("primary_endpoints", summary)

    def test_frozen_streams_endpoints_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                prior = {s for values in spec.source.seed_roles(root, preflight=preflight).values() for s in values}
                self.assertFalse(set(roles["evaluation"]).intersection(prior))
                self.assertFalse(set(roles["evaluation"]).intersection(seen))
                seen.update(roles["evaluation"])
                self.assertTrue(set(roles["reconstruction"]).issubset(prior))
        self.assertEqual(spec.CI_FAMILY_SIZE, 16)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 1843200)
        for preflight in (True, False):
            task = task_specification("unit_stage43", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertEqual(task["ram_mb"], 4096 if preflight else 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)
            self.assertIn("Training complete: result.json written", task["cmd"])

    def test_paired_root_bootstrap_all_endpoints_and_no_missing_roots(self):
        results = []
        for index, root in enumerate(spec.roots(preflight=False)):
            methods = {m: {"training_surrogate_gain": .01 * (index + 1), "heldout_task_direction": -.1 * (index + 1)} for m in spec.METHODS}
            evaluation = {p: {mode: [{**{k: 100. + index * (p != "frozen") for k in spec.METRICS},
                "seed": seed, "policy_seed": spec.policy_seed(root, seed), "deployment_mode": mode,
                **{k: v for k, v in spec.rollout_arguments(root, seed, mode=mode, reference=p == "frozen").items() if k != "sample"}}
                for seed in spec.seed_roles(root, preflight=False)["evaluation"]] for mode in spec.MODES} for p in spec.POLICIES}
            results.append({"root": root, "status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
                "contract": spec.contract(), "preflight": False, "seed_roles": spec.seed_roles(root, preflight=False),
                "budget": spec.budget(preflight=False), "optimizer_steps": 0, "methods": methods,
                "evaluation_rows": evaluation, "inference_counts": {"evaluation": {k: 0 for k in
                    ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}}, "native_trace_audits": 0})
        summary = direction.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 16})
        for m in spec.METHODS:
            self.assertEqual(summary["primary_endpoints"][m + ":training_surrogate_gain"]["effect"], "positive")
            self.assertEqual(summary["primary_endpoints"][m + ":heldout_task_direction"]["effect"], "negative")
            self.assertAlmostEqual(summary["primary_endpoints"][m + ":deterministic_return"]["mean"], 3.5)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            direction.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["evaluation_rows"]["frozen"]["lower_sampled"][0]["lower_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sampling changed"):
            direction.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
