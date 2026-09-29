import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_state_baseline as experiment
from freq_hrl.experiments import pointmaze_episode_credit as episodes
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch, SMDPPPOConfig
from scripts import pointmaze_state_baseline_stage47_spec as spec
from scripts.submit_pointmaze_state_baseline_stage47_scheduleurm import task_specification
from test_pointmaze_episode_credit import EpisodeCreditTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class StateBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        torch.manual_seed(47)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=2, lower_state_dim=2, lower_value_state_dim=4,
            upper_action_dim=1, lower_action_dim=1, hidden_dim=0, lower_cost_critic=False,
            epochs=2, minibatch_size=4, lower_learning_rate=.01))
        rng = np.random.default_rng(47)
        state, action = rng.normal(size=(12, 2)).astype(np.float32), rng.normal(size=(12, 1)).astype(np.float32)
        value_state = np.column_stack((state, np.tile(np.arange(4) / 4, 3), np.tile(1 - np.arange(4) / 4, 3))).astype(np.float32)
        with torch.no_grad():
            logp, _ = model.lower_actor.log_prob_entropy(torch.from_numpy(state), torch.from_numpy(action))
            value = model.lower_value(torch.from_numpy(value_state))
        batch = LevelTrajectoryBatch(state=state, action=action, value_state=value_state,
            old_logp=logp.numpy(), old_value=value.numpy(), reward=rng.normal(size=12).astype(np.float32),
            done=np.tile([0., 0., 0., 1.], 3).astype(np.float32), duration=np.ones(12, dtype=np.int64))
        return model, batch, batch.reward.reshape(3, 4).astype(np.float64)

    def test_entire_query_episode_excluded_and_future_query_states_do_not_change_past_prediction(self):
        model, batch, rewards = self.data()
        states = batch.value_state.reshape(3, 4, 4)
        _, baseline, _ = experiment.state_credit(states, rewards, model.config, root=310001, iteration=1)
        changed = rewards.copy()
        changed[0] += np.arange(4) * 100.
        _, other, _ = experiment.state_credit(states, changed, model.config, root=310001, iteration=1)
        np.testing.assert_array_equal(baseline[0], other[0])
        future = states.copy()
        future[0, 2:] += 50.
        _, other, _ = experiment.state_credit(future, rewards, model.config, root=310001, iteration=1)
        np.testing.assert_array_equal(baseline[0, :2], other[0, :2])
        _, time, _ = experiment.state_credit(states, np.tile(rewards[1], (3, 1)), model.config, root=310001, iteration=1)
        np.testing.assert_allclose(time, np.tile(np.cumsum(rewards[1, ::-1])[::-1], (3, 1)), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "three complete episodes"):
            experiment.state_credit(states[:2], rewards[:2], model.config, root=310001, iteration=1)

    def test_fit_learns_state_residual_without_mutating_global_rng_or_policy(self):
        model, batch, rewards = self.data()
        states = batch.value_state.reshape(3, 4, 4)
        before = copy.deepcopy(model.state_dict())
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        advantage, predictions, detail = experiment.state_credit(states, rewards, model.config, root=310001, iteration=1)
        time = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1] - episodes.score_targets(rewards)
        self.assertGreater(float(np.linalg.norm(predictions - time)), 0.)
        self.assertEqual(detail["baseline_optimizer_steps"], 12)
        experiment.gradient_dispersion(model, batch, rewards, advantage)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
        for left, right in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(left, right)
        self.assertEqual(model.config, before["config"] if isinstance(before["config"], SMDPPPOConfig) else SMDPPPOConfig(**before["config"]))
        torch.testing.assert_close({k: v for k, v in model.state_dict().items() if k != "config"},
                                   {k: v for k, v in before.items() if k != "config"}, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in model.lower_actor.parameters()))
        self.assertFalse(model.lower_actor_optimizer.state)
        self.assertFalse(model.lower_value_optimizer.state)

    def test_episode_gradient_dispersion_matches_direct_parameter_score(self):
        model, batch, rewards = self.data()
        advantage = np.arange(12, dtype=np.float32) ** 2
        observed = experiment.gradient_dispersion(model, batch, rewards, advantage)
        for treatment, values in (("episode_mc", episodes.score_targets(rewards).reshape(-1)), ("state_mc", advantage)):
            weights, gradients = model._normalize(values).reshape(3, 4), []
            for i in range(3):
                rows = slice(i * 4, (i + 1) * 4)
                logp, _ = model.lower_actor.log_prob_entropy(torch.from_numpy(batch.state[rows]), torch.from_numpy(batch.action[rows]))
                gradient = torch.autograd.grad((logp * torch.from_numpy(weights[i])).mean(), list(model.lower_actor.parameters()))
                gradients.append(torch.cat([g.reshape(-1) for g in gradient]).double())
            gradients = torch.stack(gradients)
            expected = float((gradients - gradients.mean(dim=0)).square().sum() / 2)
            self.assertAlmostEqual(observed[treatment]["trace_dispersion"], expected, places=12)
        self.assertEqual(observed["gradient_backward_calls"], 6)

    def test_all_three_credit_arms_preserve_first_critic_and_adam(self):
        model, batch, rewards = self.data()
        controls = []
        for treatment in spec.TREATMENTS:
            copied = copy.deepcopy(model)
            np.random.seed(spec.shuffle_seed(310001, 1))
            result = experiment.update(copied, batch, rewards, treatment, root=310001, iteration=1)
            self.assertEqual(result["actor_credit"], treatment)
            controls.append((copied.lower_value.state_dict(), copied.lower_value_optimizer.state_dict()))
        for other in controls[1:]:
            torch.testing.assert_close(other, controls[0], atol=0, rtol=0)

    def test_existing_checkpoint_three_arm_pipeline_and_auxiliary_accounting(self):
        original = spec.previous.previous.source.source
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=8, lower_cost_critic=False, epochs=1, minibatch_size=128))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_file.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec.previous.previous.source, "ROOT", directory), patch.object(original, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), patch.object(episodes, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                calibration.train(310001, "task_clock", preflight=True, output=spec.source_result(310001, "task_clock", preflight=True),
                                  specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 15600)
                self.assertEqual(result["native_trace_audits"], 52)
                self.assertEqual(set(result["first_batch_pair_details"]["task_clock"]), {"episode_mc", "state_mc"})
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["optimizer_steps"], {"actor_optimizer_steps": 60, "value_optimizer_steps": 60})
                self.assertEqual(summary["auxiliary_cost"], {"baseline_optimizer_steps": 64, "gradient_backward_calls": 16})
                self.assertNotIn("primary_endpoints", summary)

    def test_fresh_seeds_fixed_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for values in roles.values() for s in values}
                self.assertEqual(len(new), sum(len(v) for v in roles.values()))
                for prior in (spec.previous, spec.previous.previous, spec.previous.previous.source.source):
                    old = {s for values in prior.seed_roles(root, preflight=preflight).values() for s in values}
                    self.assertFalse(new.intersection(old))
                self.assertFalse(new.intersection(seen))
                seen.update(new)
        self.assertEqual(spec.CI_FAMILY_SIZE, 7)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 5836800)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 4864)
        for preflight in (True, False):
            task = task_specification("unit_stage47", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)

    def synthetic_results(self):
        results = EpisodeCreditTest().synthetic_results()
        for cell in results:
            root, method = cell["root"], "task_clock"
            cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract(), options=spec.options(preflight=False),
                        budget=spec.budget(preflight=False), seed_roles=spec.seed_roles(root, preflight=False))
            cell["training"] = {method: cell["training"][method]}
            cell["training"][method]["state_mc"] = copy.deepcopy(cell["training"][method]["episode_mc"])
            settings = {"state_dim": 392, "hidden_dim": 0, "epochs": 1, "minibatch_size": 1024,
                        "learning_rate": .003, "value_coef": .5, "max_grad_norm": .5}
            fit_steps = math.ceil(7 * 1200 / 1024)
            for treatment, history in cell["training"][method].items():
                for row in history:
                    iteration = row["iteration"]
                    seeds = cell["seed_roles"]["training"][(iteration - 1) * 8:iteration * 8]
                    row["actor_credit"] = treatment
                    row["rollout_sampling"] = [{"seed": seed, "policy_seed": spec.policy_seed(root, seed),
                        **{k: v for k, v in spec.rollout_arguments(root, seed, phase="train", mode="training").items() if k != "sample"}} for seed in seeds]
                    if treatment == "state_mc":
                        row.update(settings=settings, folds=[{"held_out": i, "fit_indices": [j for j in range(8) if j != i],
                            "seed": spec.baseline_seed(root, iteration, i), "optimizer_steps": fit_steps,
                            "held_out_mse": .1, "time_loo_mse": .2, "fit_normalized_mse": .05} for i in range(8)],
                            baseline_optimizer_steps=8 * fit_steps, gradient_dispersion={"episode_mc": {"trace_dispersion": .2},
                            "state_mc": {"trace_dispersion": .1}, "gradient_backward_calls": 16})
            cell["evaluation_rows"] = {p: v for p, v in cell["evaluation_rows"].items() if p in spec.POLICIES}
            cell["evaluation_rows"][method + ":state_mc"] = copy.deepcopy(cell["evaluation_rows"][method + ":episode_mc"])
            for policy, stages in cell["evaluation_rows"].items():
                for stage in stages.values():
                    for mode, rows in stage.items():
                        for row, seed in zip(rows, cell["seed_roles"]["evaluation"]):
                            row.update(seed=seed, policy_seed=spec.policy_seed(root, seed),
                                       **spec.rollout_arguments(root, seed, phase="eval", mode=mode))
                            if policy.endswith(":state_mc"):
                                row["episode_return"] += 5.
            cell["first_batch_pairs"] = {method: cell["first_batch_pairs"][method]}
            cell["first_batch_pair_details"] = {method: {t: cell["first_batch_pairs"][method] for t in spec.TREATMENTS[1:]}}
            cell["expected_steps_per_update"] = {method: 2}
            cell["native_trace_audits"] = cell["budget"]["native_trace_audits"]
            for phase, key in (("train", "training_primitive_steps"), ("eval", "evaluation_primitive_steps")):
                steps = cell["budget"][key]
                cell["inference_counts"][phase] = {"primitive_steps": steps, "lower_inference_calls": steps,
                                                   "upper_inference_calls": steps // 1200, "gate_inference_calls": steps // 1200}
        return results

    def test_statistics_and_actual_fold_accounting(self):
        results = self.synthetic_results()
        summary = experiment.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 7})
        self.assertEqual(summary["primary_endpoints"]["state_mc_final"]["mean"], 5.)
        self.assertEqual(summary["primary_endpoints"]["state_frozen_final"]["effect"], "positive")
        self.assertAlmostEqual(summary["primary_endpoints"]["first_gradient_dispersion_reduction"]["mean"], .1)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            experiment.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["state_mc"][0]["folds"][0]["fit_indices"].append(0)
        with self.assertRaisesRegex(ValueError, "fit provenance changed"):
            experiment.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["state_mc"][0]["baseline_optimizer_steps"] += 1
        with self.assertRaisesRegex(ValueError, "auxiliary cost accounting changed"):
            experiment.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
