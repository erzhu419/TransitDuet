import unittest

import numpy as np
import torch

from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    SMDPPPOConfig,
    deployment_frequency_stats,
    deterministic_actor_action,
)


class DeploymentFrequencyStatsTest(unittest.TestCase):
    def test_upper_hold_duration_is_expanded_before_high_pass(self):
        constant = deployment_frequency_stats(
            torch.tensor([[1.0], [1.0]]),
            torch.tensor([2, 2]),
            torch.tensor([False, True]),
            window=2,
            band="high",
            rms_budget=0.1,
        )
        alternating = deployment_frequency_stats(
            torch.tensor([[1.0], [-1.0]]),
            torch.tensor([2, 2]),
            torch.tensor([False, True]),
            window=2,
            band="high",
            rms_budget=0.1,
        )
        self.assertEqual(constant.primitive_steps, 4)
        self.assertAlmostEqual(float(constant.power), 0.0, places=7)
        self.assertAlmostEqual(float(alternating.power), 0.25, places=7)

    def test_episode_boundary_resets_causal_filter(self):
        reset = deployment_frequency_stats(
            torch.tensor([[1.0], [-1.0]]),
            torch.ones(2, dtype=torch.long),
            torch.tensor([True, True]),
            window=2,
            band="high",
            rms_budget=0.1,
        )
        continuous = deployment_frequency_stats(
            torch.tensor([[1.0], [-1.0]]),
            torch.ones(2, dtype=torch.long),
            torch.tensor([False, True]),
            window=2,
            band="high",
            rms_budget=0.1,
        )
        self.assertEqual(reset.segment_count, 2)
        self.assertAlmostEqual(float(reset.power), 0.0, places=7)
        self.assertAlmostEqual(float(continuous.power), 0.5, places=7)

    def test_lower_low_pass_constraint_is_differentiable(self):
        actions = torch.full((5, 2), 0.4, requires_grad=True)
        stats = deployment_frequency_stats(
            actions,
            torch.ones(5, dtype=torch.long),
            torch.tensor([False, False, False, False, True]),
            window=3,
            band="low",
            rms_budget=0.1,
        )
        self.assertAlmostEqual(
            float(stats.power.detach().cpu().item()), 0.16, places=6
        )
        self.assertAlmostEqual(
            float(stats.signed_excess.detach().cpu().item()), 0.15, places=6
        )
        stats.signed_excess.backward()
        self.assertTrue(torch.all(torch.isfinite(actions.grad)))
        self.assertGreater(float(torch.linalg.vector_norm(actions.grad)), 0.0)


class DeploymentFrequencyPPOTest(unittest.TestCase):
    @staticmethod
    def _batch(model: FrequencySeparatedActorCriticPPO):
        rng = np.random.default_rng(42)
        builder = HierarchicalRolloutBuilder(gamma=0.99)
        for macro in range(2):
            upper_state = rng.normal(size=3).astype(np.float32)
            upper = model.act_upper(upper_state, sample=False)
            builder.begin_upper(
                state=upper_state,
                action=upper["action"],
                logp=upper["logp"],
                value=upper["value"],
            )
            for lower_step in range(4):
                lower_state = rng.normal(size=2).astype(np.float32)
                lower = model.act_lower(lower_state, sample=False)
                builder.add_lower(
                    state=lower_state,
                    action=lower["action"],
                    logp=lower["logp"],
                    value=lower["value"],
                    reward=float(lower_step - 1),
                    done=macro == 1 and lower_step == 3,
                )
        return builder.build()

    def test_actor_mean_constraint_is_separate_from_exploration_std(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            deployment_action_transform="tanh",
            lower_deployment_frequency_rms_budget=0.01,
            lower_deployment_frequency_window=3,
            lower_deployment_frequency_dual_lr=0.5,
            lower_deployment_frequency_lambda_init=1.0,
            lower_deployment_frequency_step_scale=10.0,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        with torch.no_grad():
            model.lower_actor.net[-1].bias.fill_(0.5)
        state = torch.zeros((4, 2), dtype=torch.float32)
        action = deterministic_actor_action(
            model.lower_actor, state, transform="tanh", scale=1.0
        )
        stats = deployment_frequency_stats(
            action,
            torch.ones(4, dtype=torch.long),
            torch.tensor([False, False, False, True]),
            window=3,
            band="low",
            rms_budget=0.01,
        )
        gradient = torch.autograd.grad(
            stats.signed_excess,
            tuple(model.lower_actor.parameters()),
            allow_unused=True,
        )
        self.assertIsNone(gradient[0])
        self.assertTrue(any(item is not None for item in gradient[1:]))

        metrics = model.update(self._batch(model))
        self.assertEqual(metrics["lower_deployment_frequency_enabled"], 1.0)
        self.assertEqual(
            metrics["lower_deployment_frequency_primitive_steps"], 8.0
        )
        self.assertEqual(
            metrics["lower_deployment_frequency_segment_count"], 1.0
        )
        self.assertGreater(
            metrics["lower_deployment_frequency_violation_before"], 0.0
        )
        self.assertGreater(
            metrics["lower_deployment_frequency_lambda_after"], 1.0
        )
        self.assertEqual(
            metrics["upper_deployment_frequency_enabled"], 0.0
        )

    def test_deployment_dual_state_round_trips(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            upper_deployment_frequency_rms_budget=0.2,
            upper_deployment_frequency_lambda_init=1.5,
            lower_deployment_frequency_rms_budget=0.1,
            lower_deployment_frequency_lambda_init=2.5,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        payload = model.state_dict()
        restored = FrequencySeparatedActorCriticPPO(config)
        restored.load_state_dict(payload)
        self.assertEqual(restored.upper_deployment_frequency_lambda, 1.5)
        self.assertEqual(restored.lower_deployment_frequency_lambda, 2.5)

    def test_active_constraint_requires_positive_budget(self):
        with self.assertRaisesRegex(ValueError, "positive RMS budget"):
            FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                upper_state_dim=3,
                lower_state_dim=2,
                upper_action_dim=1,
                lower_action_dim=1,
                lower_deployment_frequency_dual_lr=0.1,
            ))


if __name__ == "__main__":
    unittest.main()
