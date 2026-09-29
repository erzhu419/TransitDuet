import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from freq_hrl.experiments import pointmaze_fisher_direction as experiment
from freq_hrl.experiments import pointmaze_episode_credit as episodes
from freq_hrl.experiments import pointmaze_episode_kl as bounded
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_fisher_direction_stage50_spec as spec
from scripts.submit_pointmaze_fisher_direction_stage50_scheduleurm import task_specification
from test_pointmaze_state_baseline import StateBaselineTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class FisherDirectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fisher_matches_gaussian_analytic_hessian_and_dense_solve(self):
        model, batch, rewards = StateBaselineTest().data()
        weights = model._normalize(episodes.score_targets(rewards).reshape(-1))
        gradient, fisher = experiment.score_geometry(model.lower_actor, batch, weights, model.config.entropy_coef)
        actor = copy.deepcopy(model.lower_actor).double()
        state = torch.from_numpy(batch.state).double()
        old = actor.distribution(state)
        std = old.stddev.detach().numpy()[0, 0]
        # The fixture is a scalar linear Gaussian: log_std, weights, bias.
        design = np.column_stack((batch.state.astype(np.float64), np.ones(batch.size)))
        exact = np.zeros((len(gradient), len(gradient)))
        exact[0, 0] = 2.
        exact[1:, 1:] = design.T @ design / batch.size / std ** 2
        eye = np.eye(len(gradient))
        observed = np.column_stack([fisher(e) for e in eye])
        np.testing.assert_allclose(observed, exact, atol=1e-12, rtol=0)
        delta, detail = experiment.direction(model.lower_actor, batch, weights, model.config.entropy_coef, natural=True, horizon=4)
        solution = np.linalg.solve(exact + spec.FISHER_DAMPING * eye, gradient)
        coefficient = np.sqrt(2 * spec.KL_BUDGET / (4 * solution @ exact @ solution))
        np.testing.assert_allclose(delta, coefficient * solution, atol=1e-10, rtol=0)
        self.assertEqual(detail["cg_info"], 0)
        self.assertEqual(detail["fisher_vector_products"], detail["cg_iterations"] + 2)
        self.assertLess(detail["cg_relative_residual"], 1e-10)
        initial = parameters_to_vector(actor.parameters()).detach().clone()
        with torch.no_grad():
            vector_to_parameters(initial + torch.as_tensor(delta) * 1e-4, actor.parameters())
            new = actor.distribution(state)
            old = torch.distributions.Normal(old.mean.detach(), old.stddev.detach())
            actual = torch.distributions.kl_divergence(old, new).sum(-1).mean().item() * 4
        self.assertAlmostEqual(actual / 1e-8, spec.KL_BUDGET, places=4)

    def test_score_gradient_is_full_batch_entropy_included_and_euclidean_same_credit(self):
        model, batch, rewards = StateBaselineTest().data()
        weights = model._normalize(episodes.score_targets(rewards).reshape(-1))
        actor = copy.deepcopy(model.lower_actor).double()
        state, action = torch.from_numpy(batch.state).double(), torch.from_numpy(batch.action).double()
        logp, entropy = actor.log_prob_entropy(state, action)
        objective = (logp * torch.as_tensor(weights)).mean() + model.config.entropy_coef * entropy.mean()
        direct = torch.cat([p.reshape(-1) for p in torch.autograd.grad(objective, tuple(actor.parameters()))]).numpy()
        observed, _ = experiment.score_geometry(model.lower_actor, batch, weights, model.config.entropy_coef)
        np.testing.assert_array_equal(direct, observed)
        delta, detail = experiment.direction(model.lower_actor, batch, weights, model.config.entropy_coef, natural=False, horizon=4)
        np.testing.assert_allclose(delta, observed * detail["initial_coefficient"], atol=0, rtol=0)
        self.assertEqual(detail["cg_iterations"], 0)
        self.assertEqual(detail["fisher_vector_products"], 1)
        self.assertGreater(detail["score_direction_dot"], 0)

    def test_direct_updates_keep_actor_adam_and_exact_original_critic(self):
        model, batch, rewards = StateBaselineTest().data()
        np.random.seed(spec.shuffle_seed(310001, 1))
        expected = copy.deepcopy(model)
        original = episodes.update(expected, batch, rewards, "gae", specification=spec)
        for treatment in ("euclidean_mc", "fisher_mc"):
            observed = copy.deepcopy(model)
            adam = copy.deepcopy(observed.lower_actor_optimizer.state_dict())
            record = experiment.update(observed, batch, rewards, treatment, root=310001, iteration=1)
            self.assertTrue(record["accepted"])
            self.assertLessEqual(record["deployed_terms"]["max_episode_kl"], spec.KL_BUDGET)
            self.assertEqual(experiment.optimizer_steps(record, original["value_optimizer_steps"]),
                             {"actor_optimizer_steps": 0, "value_optimizer_steps": original["value_optimizer_steps"]})
            torch.testing.assert_close(observed.lower_actor_optimizer.state_dict(), adam, atol=0, rtol=0)
            for name in ("lower_value", "lower_value_optimizer"):
                torch.testing.assert_close(getattr(observed, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)
            self.assertFalse(torch.equal(parameters_to_vector(observed.lower_actor.parameters()), parameters_to_vector(model.lower_actor.parameters())))

    def test_rejection_restores_actor_and_does_not_repeat_critic_or_solver(self):
        model, batch, rewards = StateBaselineTest().data()
        initial = copy.deepcopy(model.lower_actor.state_dict())
        with patch.object(experiment, "direction", wraps=experiment.direction) as solver, \
                patch.object(bounded, "policy_terms", wraps=bounded.policy_terms) as terms:
            def rejected(*args):
                row = terms._mock_wraps(*args)
                if row["max_episode_kl"] > 0:
                    row["episode_kl"] = [1.] * 3
                    row["max_episode_kl"] = row["mean_episode_kl"] = 1.
                return row
            terms.side_effect = rejected
            record = experiment.update(model, batch, rewards, "fisher_mc", root=310001, iteration=1)
        self.assertFalse(record["accepted"])
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(record["parameter_proposals"], 13)
        self.assertEqual(record["retained_parameter_updates"], 0)
        self.assertEqual(record["value_optimizer_steps"], record["retained_value_steps"])
        torch.testing.assert_close(model.lower_actor.state_dict(), initial, atol=0, rtol=0)
        experiment.optimizer_steps(record, record["value_optimizer_steps"])
        changed = copy.deepcopy(record)
        changed["geometry"]["fisher_vector_products"] -= 1
        with self.assertRaisesRegex(ValueError, "computation accounting"):
            experiment.optimizer_steps(changed, record["value_optimizer_steps"])

    def test_adam_controls_use_registered_kl_bound_without_reset(self):
        model, batch, rewards = StateBaselineTest().data()
        for treatment in ("gae", "episode_mc"):
            observed, expected = copy.deepcopy(model), copy.deepcopy(model)
            a = experiment.update(observed, batch, rewards, treatment, root=310001, iteration=1)
            b = bounded.update(expected, batch, rewards, treatment, root=310001, iteration=1, specification=spec)
            self.assertEqual(a, b)
            for name in bounded.LOWER_STATE:
                torch.testing.assert_close(getattr(observed, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)

    def test_shared_native_style_pipeline_and_aggregate_costs(self):
        calibration_spec = spec.previous.previous.previous.previous.source.source
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=8, lower_cost_critic=False, lower_learning_rate=3e-4, epochs=1, minibatch_size=128))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_file.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec.previous.previous.previous.previous.source, "ROOT", directory), \
                    patch.object(calibration_spec, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), patch.object(episodes, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                calibration.train(310001, "task_clock", preflight=True, output=spec.source_result(310001, "task_clock", preflight=True),
                                  specification=calibration_spec, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                summary = experiment.aggregate([result], preflight=True)
            self.assertEqual(result["budget"]["total_primitive_steps"], 20400)
            self.assertEqual(summary["status"], "preflight_passed")
            self.assertEqual(summary["direction_computation"]["score_backward_calls"], 4)
            self.assertEqual(summary["direction_computation"]["retained_value_steps"], 80)
            self.assertNotIn("primary_endpoints", summary)
            incomplete = copy.deepcopy(result)
            incomplete["first_batch_pair_details"]["task_clock"].pop("fisher_mc")
            with self.assertRaisesRegex(ValueError, "first batch pair incomplete"):
                experiment.aggregate([incomplete], preflight=True)

    def test_budget_fresh_seeds_endpoints_and_dynamic_placement(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                values = {s for role in spec.seed_roles(root, preflight=preflight).values() for s in role}
                self.assertFalse(values & seen)
                self.assertFalse(values & {s for role in spec.previous.seed_roles(root, preflight=preflight).values() for s in role})
                seen |= values
            task = task_specification("unit_stage50", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertNotIn("result_dir", task)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 7680000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        means = {s: {"lower_sampled": {p: {"episode_return": v} for p, v in zip(spec.POLICIES, (0, 1, 2, 3, 4))}}
                 for s in ("0", "1", "16")}
        self.assertEqual(spec.contrasts(means, preflight=False), dict(zip(spec.ENDPOINTS, (2, 2, 1, 4, 1, 3, 2, 1))))


if __name__ == "__main__":
    unittest.main()
