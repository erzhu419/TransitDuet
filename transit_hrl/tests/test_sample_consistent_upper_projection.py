import unittest
from unittest import mock

import numpy as np
import torch

from freq_hrl.rl import FrequencySeparatedActorCriticPPO, HierarchicalRolloutBuilder, SMDPPPOConfig
from freq_hrl.rl import smdp_actor_critic as core
from freq_hrl.experiments.mujoco import control_validation as control
from freq_hrl.experiments.mujoco.control_validation import environment_dimensions


class SampleConsistentUpperProjectionTest(unittest.TestCase):
    def test_identity_projection_does_not_penalize_exploration(self):
        for objective in ("raw_mean", "raw_sample", "action_sample"):
            mean = torch.tensor([[0.2, -0.3]], requires_grad=True)
            offset = torch.tensor([[0.7, -0.5]])
            target = (mean.detach() + offset)
            if objective == "action_sample":
                target = target.tanh()
            loss = core._projection_consistency_error(
                mean, target, objective=objective, sample_offset=offset,
            ).sum()
            loss.backward()
            if objective == "raw_mean":
                self.assertGreater(mean.grad.abs().max().item(), 0.1)
            else:
                self.assertEqual(loss.item(), 0.0)
                self.assertEqual(mean.grad.abs().max().item(), 0.0)

    def test_action_loss_gradient_matches_finite_difference(self):
        mean = torch.tensor([[0.3, -1.1]], dtype=torch.float64, requires_grad=True)
        offset = torch.tensor([[0.6, 0.2]], dtype=torch.float64)
        target = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        def loss(value):
            return core._projection_consistency_error(
                value, target, objective="action_sample", sample_offset=offset,
            ).sum()
        loss(mean).backward()
        for index in range(2):
            delta = torch.zeros_like(mean)
            delta[0, index] = 1e-6
            numeric = (loss(mean.detach() + delta) - loss(mean.detach() - delta)) / 2e-6
            self.assertAlmostEqual(mean.grad[0, index].item(), numeric.item(), places=8)

    def test_ppo_epochs_keep_upper_offset_fixed_and_lower_raw(self):
        for mode in ("scalarized", "reward_guarded_projection"):
            with self.subTest(mode=mode):
                model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                    upper_state_dim=3, lower_state_dim=2,
                    upper_action_dim=1, lower_action_dim=1, hidden_dim=8,
                    upper_learning_rate=1e-2, lower_learning_rate=1e-2,
                    upper_projection_consistency_coef=5.0, lower_projection_consistency_coef=5.0,
                    upper_projection_target_aggregation="decision_time",
                    upper_projection_consistency_objective="action_sample",
                    projection_consistency_update_mode=mode,
                    entropy_coef=0.0, epochs=4, minibatch_size=8,
                ))
                builder = HierarchicalRolloutBuilder(gamma=0.99, upper_projection_target_aggregation="decision_time")
                for i in range(2):
                    builder.begin_upper(state=np.full(3, i, np.float32), action=np.array([1.0 + i], np.float32), logp=0., value=0.)
                    builder.add_lower(state=np.full(2, i, np.float32), action=np.array([-1.], np.float32), logp=0., value=0., reward=0., done=True,
                                      upper_projection_target=np.array([-0.5], np.float32), lower_projection_target=np.array([0.5], np.float32))
                batch = builder.build()
                states = torch.as_tensor(batch.upper.state)
                before = model.upper_actor.distribution(states).mean.detach().clone()
                expected = torch.sort(torch.as_tensor(batch.upper.action) - before, dim=0).values
                calls = []
                actual_error = core._projection_consistency_error
                def capture(mean, target, *, objective, sample_offset=None):
                    calls.append((objective, None if sample_offset is None else sample_offset.clone()))
                    return actual_error(mean, target, objective=objective, sample_offset=sample_offset)
                with mock.patch.object(core, "_projection_consistency_error", side_effect=capture):
                    model.update(batch)
                upper_calls = [offset for objective, offset in calls if objective == "action_sample"]
                self.assertGreaterEqual(len(upper_calls), 4)
                for offset in upper_calls:
                    torch.testing.assert_close(torch.sort(offset, dim=0).values, expected, rtol=0, atol=0)
                    self.assertFalse(offset.requires_grad)
                self.assertTrue(any(objective == "raw_mean" for objective, _ in calls))
                self.assertTrue(all(offset is None for objective, offset in calls if objective == "raw_mean"))
                after = model.upper_actor.distribution(states).mean.detach()
                self.assertGreater((after - before).abs().max().item(), 1e-5)

    def test_sample_objectives_reject_noncausal_aggregation(self):
        for objective in ("raw_sample", "action_sample"):
            with self.assertRaisesRegex(ValueError, "decision-time"):
                FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                    upper_state_dim=3, lower_state_dim=2, upper_action_dim=1, lower_action_dim=1,
                    upper_projection_consistency_objective=objective,
                ))

    def test_bounded_target_changes_neither_execution_nor_lower_target(self):
        observation_dim, action_dim = environment_dimensions("HalfCheetah-v5", episode_horizon=32)
        model = control._hierarchical_model(
            state_dim=control.mujoco_policy_state_dim(observation_dim, action_dim, terminal_reserve_projection=True),
            action_dim=action_dim, hidden_dim=8, learning_rate=3e-4, leakage_constraint=False,
            upper_projection_target_aggregation="decision_time",
        )
        values = dict(logp=0.0, value=0.0, cost_value=0.0)
        upper = np.linspace(-1.3, 1.1, action_dim, dtype=np.float32)
        lower = np.linspace(0.9, -0.7, action_dim, dtype=np.float32)
        options = dict(seed=2063, env_id="HalfCheetah-v5", disturbance_mode="mixed", steps=32,
                       upper_period=4, frequency_routing=True, leakage_constraint=False, sample=False,
                       episode_horizon=32, terminal_reserve_projection=True, collect_trajectory=True,
                       upper_hf_rms_budget=0.075, lower_lf_rms_budget=0.0475,
                       upper_action_scale=0.8)
        with mock.patch.object(model, "act_upper", return_value={**values, "action": upper}), mock.patch.object(
            model, "act_lower", return_value={**values, "action": lower, "mean_action": lower},
        ):
            raw, raw_metrics = control.rollout_hierarchical(model, **options)
            model.config.upper_projection_consistency_objective = "action_sample"
            bounded, bounded_metrics = control.rollout_hierarchical(model, **options)
        np.testing.assert_allclose(bounded.upper.projection_target, np.tanh(raw.upper.projection_target), atol=1e-7)
        np.testing.assert_array_equal(bounded.lower.projection_target, raw.lower.projection_target)
        self.assertEqual(raw_metrics["episode_return"], bounded_metrics["episode_return"])


if __name__ == "__main__":
    unittest.main()
