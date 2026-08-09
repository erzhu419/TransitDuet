import copy
import unittest

import numpy as np
import torch

from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    LevelTrajectoryBatch,
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
        self.assertAlmostEqual(
            float(stats.normalized_signed_excess.detach().cpu().item()),
            15.0,
            places=5,
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

    def test_relative_target_uses_frozen_actor_on_the_same_states(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            deployment_action_transform="tanh",
            lower_deployment_frequency_rms_budget=0.001,
            lower_deployment_frequency_reference_reduction_fraction=0.1,
            lower_deployment_frequency_lambda_init=1.0,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        with torch.no_grad():
            model.lower_actor.net[-1].bias.fill_(0.4)
        model.capture_actor_anchor()
        with torch.no_grad():
            model.lower_actor.net[-1].bias.fill_(0.5)
        metrics = model.update(self._batch(model))
        reference = metrics["lower_deployment_frequency_reference_power"]
        target = metrics["lower_deployment_frequency_target_power"]
        self.assertGreater(reference, 1e-6)
        self.assertAlmostEqual(target, 0.9 * reference, places=6)
        self.assertEqual(
            metrics[
                "lower_deployment_frequency_reference_reduction_fraction"
            ],
            0.1,
        )

    def test_iterative_projection_outperforms_one_step_under_one_reward_budget(self):
        common = {
            "upper_state_dim": 3,
            "lower_state_dim": 2,
            "upper_action_dim": 1,
            "lower_action_dim": 1,
            "hidden_dim": 8,
            "epochs": 1,
            "minibatch_size": 8,
            "deployment_action_transform": "tanh",
            "lower_deployment_frequency_rms_budget": 0.01,
            "lower_deployment_frequency_window": 3,
            "lower_deployment_frequency_lambda_init": 1.0,
            "lower_deployment_frequency_step_scale": 10.0,
            "lower_deployment_frequency_reward_tolerance": 1e-8,
        }
        torch.manual_seed(7)
        source = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**common))
        with torch.no_grad():
            source.lower_actor.net[-1].bias.fill_(0.5)
        batch = self._batch(source).lower
        initial_state = copy.deepcopy(source.state_dict())

        def project(steps: int) -> dict[str, float]:
            model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                **common,
                lower_deployment_frequency_max_projection_steps=steps,
            ))
            model.load_state_dict(copy.deepcopy(initial_state))
            return model._update_deployment_frequency_constraint(
                level="lower",
                batch=batch,
                actor=model.lower_actor,
            )

        one_step = project(1)
        four_steps = project(4)
        prefix = "lower_deployment_frequency_"
        self.assertEqual(one_step[prefix + "projection_steps_accepted"], 1.0)
        self.assertEqual(four_steps[prefix + "projection_steps_attempted"], 4.0)
        self.assertEqual(four_steps[prefix + "projection_steps_accepted"], 4.0)
        self.assertLess(
            four_steps[prefix + "power_after"],
            one_step[prefix + "power_after"],
        )
        self.assertLess(
            four_steps[prefix + "normalized_signed_excess_after"],
            one_step[prefix + "normalized_signed_excess_after"],
        )
        self.assertLessEqual(
            four_steps[prefix + "guard_reward_loss_delta"],
            common["lower_deployment_frequency_reward_tolerance"] + 1e-7,
        )
        self.assertEqual(
            four_steps[prefix + "projection_step_budget_exhausted"], 1.0
        )

    def test_groupwise_projection_constrains_the_worst_rollout(self):
        common = {
            "upper_state_dim": 3,
            "lower_state_dim": 2,
            "upper_action_dim": 1,
            "lower_action_dim": 1,
            "hidden_dim": 0,
            "deployment_action_transform": "tanh",
            "lower_deployment_frequency_rms_budget": 0.5,
            "lower_deployment_frequency_window": 3,
            "lower_deployment_frequency_lambda_init": 1.0,
            "lower_deployment_frequency_step_scale": 1.0,
        }
        pooled = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**common))
        robust = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            **common,
            deployment_frequency_groupwise_robust=True,
        ))
        with torch.no_grad():
            pooled.lower_actor.net[-1].weight.zero_()
            pooled.lower_actor.net[-1].weight[0, 0] = 3.0
            pooled.lower_actor.net[-1].bias.zero_()
        robust.load_state_dict(copy.deepcopy(pooled.state_dict()))
        state = np.zeros((10, 2), dtype=np.float32)
        state[0, 0] = 1.0
        batch = LevelTrajectoryBatch(
            state=state,
            action=np.zeros((10, 1), dtype=np.float32),
            reward=np.zeros(10, dtype=np.float32),
            duration=np.ones(10, dtype=np.int64),
            done=np.asarray(
                [True] + [False] * 8 + [True], dtype=np.float32
            ),
            old_logp=np.zeros(10, dtype=np.float32),
            old_value=np.zeros(10, dtype=np.float32),
            deployment_frequency_group=np.asarray(
                [0] + [1] * 9, dtype=np.int64
            ),
        )
        pooled_metrics = pooled._update_deployment_frequency_constraint(
            level="lower", batch=batch, actor=pooled.lower_actor
        )
        robust_metrics = robust._update_deployment_frequency_constraint(
            level="lower", batch=batch, actor=robust.lower_actor
        )
        prefix = "lower_deployment_frequency_"
        self.assertEqual(
            pooled_metrics[prefix + "projection_target_reached_before"], 1.0
        )
        self.assertEqual(
            pooled_metrics[prefix + "projection_steps_attempted"], 0.0
        )
        self.assertEqual(robust_metrics[prefix + "group_count"], 2.0)
        self.assertEqual(
            robust_metrics[prefix + "projection_target_reached_before"], 0.0
        )
        self.assertGreater(
            robust_metrics[prefix + "projection_steps_attempted"], 0.0
        )
        self.assertEqual(
            robust_metrics[prefix + "group_reward_budget_violation_count"],
            0.0,
        )

    def test_anchor_state_replay_expands_only_frequency_groups(self):
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
            lower_deployment_frequency_lambda_init=1.0,
            deployment_frequency_groupwise_robust=True,
            deployment_frequency_anchor_state_replay=True,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        current = self._batch(model).lower
        reference = copy.deepcopy(current)
        metrics = model._update_deployment_frequency_constraint(
            level="lower",
            batch=current,
            actor=model.lower_actor,
            reference_batch=reference,
        )
        prefix = "lower_deployment_frequency_"
        self.assertEqual(metrics[prefix + "group_count"], 2.0)
        self.assertEqual(
            metrics[prefix + "reward_guard_group_count"], 1.0
        )
        self.assertEqual(
            metrics[prefix + "anchor_state_replay_enabled"], 1.0
        )
        self.assertEqual(
            metrics[prefix + "anchor_state_replay_transitions"],
            float(reference.size),
        )

    def test_ppo_trust_region_blocks_worst_group_frequency_drift(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            deployment_action_transform="tanh",
            lower_deployment_frequency_rms_budget=0.1,
            lower_deployment_frequency_lambda_init=1.0,
            lower_deployment_frequency_reward_tolerance=1e-8,
            deployment_frequency_groupwise_robust=True,
            deployment_frequency_ppo_trust_region=True,
            deployment_frequency_ppo_trust_region_backtracks=8,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        with torch.no_grad():
            for parameter in model.lower_actor.parameters():
                parameter.zero_()
        batch = self._batch(model).lower
        captured = model._capture_deployment_frequency_ppo_guard(
            level="lower",
            batch=batch,
            actor=model.lower_actor,
            actor_optimizer=model.lower_actor_optimizer,
            reference_batch=None,
        )
        self.assertIsNotNone(captured)
        with torch.no_grad():
            model.lower_actor.net[-1].bias.fill_(1.0)
        metrics = model._apply_deployment_frequency_ppo_guard(
            level="lower",
            batch=batch,
            actor=model.lower_actor,
            actor_optimizer=model.lower_actor_optimizer,
            reference_batch=None,
            captured=captured,
        )
        prefix = "lower_deployment_frequency_ppo_guard_"
        self.assertGreater(
            metrics[prefix + "frequency_excess_full_step"],
            metrics[prefix + "frequency_excess_before"],
        )
        self.assertLess(metrics[prefix + "step_fraction"], 1.0)
        self.assertLessEqual(
            metrics[prefix + "frequency_excess_after"], 1e-7
        )
        self.assertEqual(
            metrics[prefix + "group_reward_budget_violation_count"], 0.0
        )
        self.assertEqual(metrics[prefix + "optimizer_restored"], 1.0)

    def test_replay_and_trust_region_require_groupwise_constraints(self):
        with self.assertRaisesRegex(ValueError, "groupwise robust"):
            FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                upper_state_dim=3,
                lower_state_dim=2,
                upper_action_dim=1,
                lower_action_dim=1,
                deployment_frequency_anchor_state_replay=True,
            ))

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
