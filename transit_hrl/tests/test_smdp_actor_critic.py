import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    PromotionRolloutBuilder,
    SMDPPPOConfig,
    TemporalDecisionScheduler,
    concat_hierarchical_batches,
)
from freq_hrl.experiments.trading.ppo_actor_critic import (
    frequency_separated_feature_vectors,
    smdp_parameter_count,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    count_parameters,
)
from freq_hrl.rl.smdp_actor_critic import (
    _reward_guarded_adam_step,
    _project_constraint_gradients,
    _reward_guarded_constraint_step,
)


class TemporalDecisionSchedulerTest(unittest.TestCase):
    def test_schedule_and_promotion_respect_minimum_duration(self):
        scheduler = TemporalDecisionScheduler(upper_period=5, min_upper_duration=2)
        self.assertEqual(scheduler.decision_reason(0), "initial")
        scheduler.mark_decision(0)
        self.assertIsNone(scheduler.decision_reason(1, promotion=True))
        self.assertEqual(scheduler.decision_reason(2, promotion=True), "promotion")
        scheduler.mark_decision(2)
        self.assertIsNone(scheduler.decision_reason(6))
        self.assertEqual(scheduler.decision_reason(7), "scheduled")


class HierarchicalRolloutBuilderTest(unittest.TestCase):
    @staticmethod
    def _add_step(builder, reward, done=False):
        builder.add_lower(
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            action=np.asarray([0.1], dtype=np.float32),
            logp=-0.2,
            value=0.3,
            reward=reward,
            cost=0.25,
            done=done,
        )

    def test_upper_action_is_recorded_once_per_macro_interval(self):
        builder = HierarchicalRolloutBuilder(gamma=0.9)
        builder.begin_upper(
            state=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            action=np.asarray([0.4], dtype=np.float32),
            logp=-0.5,
            value=0.2,
        )
        self._add_step(builder, 1.0)
        self._add_step(builder, 2.0)
        builder.begin_upper(
            state=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
            action=np.asarray([0.8], dtype=np.float32),
            logp=-0.7,
            value=0.4,
        )
        self._add_step(builder, 3.0, done=True)
        batch = builder.build()

        self.assertEqual(batch.upper.size, 2)
        self.assertEqual(batch.lower.size, 3)
        np.testing.assert_array_equal(batch.upper.duration, np.asarray([2, 1]))
        self.assertAlmostEqual(float(batch.upper.reward[0]), 1.0 + 0.9 * 2.0, places=6)
        np.testing.assert_allclose(batch.upper.old_logp, np.asarray([-0.5, -0.7]))
        np.testing.assert_array_equal(batch.upper.done, np.asarray([0.0, 1.0]))

    def test_finish_marks_timeout_as_terminal_boundary(self):
        builder = HierarchicalRolloutBuilder(gamma=0.95)
        builder.begin_upper(
            state=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            logp=0.0,
            value=0.0,
        )
        self._add_step(builder, 1.0)
        builder.finish(terminal=True)
        batch = builder.build()
        self.assertEqual(float(batch.upper.done[-1]), 1.0)
        self.assertEqual(float(batch.lower.done[-1]), 1.0)

    def test_sparse_promotion_gate_owns_rewards_until_next_decision(self):
        builder = PromotionRolloutBuilder(gamma=0.9)
        builder.begin(
            state=np.asarray([1.0, 0.0], dtype=np.float32),
            action=1.0,
            logp=-0.2,
            value=0.1,
        )
        builder.add_reward(1.0, counterfactual_advantage=0.5)
        builder.add_reward(2.0, counterfactual_advantage=-0.25)
        builder.begin(
            state=np.asarray([0.0, 1.0], dtype=np.float32),
            action=0.0,
            logp=-0.3,
            value=0.2,
        )
        builder.add_reward(
            3.0, counterfactual_advantage=0.75, done=True
        )
        batch = builder.build()
        self.assertIsNotNone(batch)
        np.testing.assert_array_equal(batch.duration, [2, 1])
        np.testing.assert_allclose(batch.reward, [2.8, 3.0])
        np.testing.assert_allclose(
            batch.counterfactual_advantage, [0.275, 0.75]
        )
        np.testing.assert_array_equal(batch.action.reshape(-1), [1.0, 0.0])

    def test_promotion_gate_can_close_at_scheduled_upper_boundary(self):
        builder = PromotionRolloutBuilder(gamma=0.9)
        builder.begin(
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            action=0.0,
            logp=-0.2,
            value=0.4,
        )
        builder.add_reward(1.0)
        builder.add_reward(2.0)
        builder.close(done=False)
        builder.begin(
            state=np.asarray([3.0, 4.0], dtype=np.float32),
            action=1.0,
            logp=-0.3,
            value=0.5,
        )
        builder.add_reward(3.0)
        builder.finish(terminal=True)

        batch = builder.build()
        self.assertIsNotNone(batch)
        np.testing.assert_array_equal(batch.duration, [2, 1])
        np.testing.assert_array_equal(batch.done, [0.0, 1.0])
        self.assertAlmostEqual(float(batch.reward[0]), 1.0 + 0.9 * 2.0)
        self.assertAlmostEqual(float(batch.reward[1]), 3.0)


class FrequencySeparatedActorCriticTest(unittest.TestCase):
    @staticmethod
    def _batch(seed=0):
        rng = np.random.default_rng(seed)
        builder = HierarchicalRolloutBuilder(gamma=0.99)
        for macro in range(2):
            builder.begin_upper(
                state=rng.normal(size=3).astype(np.float32),
                action=rng.normal(size=1).astype(np.float32),
                logp=-0.5,
                value=0.1,
            )
            for lower_step in range(2):
                builder.add_lower(
                    state=rng.normal(size=2).astype(np.float32),
                    action=rng.normal(size=1).astype(np.float32),
                    logp=-0.4,
                    value=0.2,
                    reward=0.1 * (macro + 1),
                    cost=0.05 * lower_step,
                    done=macro == 1 and lower_step == 1,
                )
        return builder.build()

    def test_independent_smdp_updates_run(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            epochs=1,
            minibatch_size=8,
            lower_dual_lr=0.1,
            lower_cost_target=0.01,
        ))
        metrics = model.update(self._batch())
        self.assertEqual(metrics["upper_transitions"], 2.0)
        self.assertEqual(metrics["lower_transitions"], 4.0)
        self.assertEqual(metrics["upper_mean_duration"], 2.0)
        self.assertIn("upper_policy_loss", metrics)
        self.assertIn("lower_policy_loss", metrics)
        self.assertGreater(metrics["constraint_lambda"], 0.0)

    def test_lower_cost_critic_can_use_a_distinct_causal_state(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            lower_cost_state_dim=3,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            lower_dual_lr=0.1,
        ))
        actor_state = np.asarray([0.25, -0.5], dtype=np.float32)
        first = model.act_lower(
            actor_state,
            sample=False,
            cost_state=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )
        second = model.act_lower(
            actor_state,
            sample=False,
            cost_state=np.asarray([-3.0, -2.0, -1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(first["action"], second["action"])
        self.assertEqual(first["value"], second["value"])

        rng = np.random.default_rng(17)
        builder = HierarchicalRolloutBuilder(gamma=0.99)
        builder.begin_upper(
            state=rng.normal(size=3).astype(np.float32),
            action=rng.normal(size=1).astype(np.float32),
            logp=-0.5,
            value=0.1,
        )
        for step in range(3):
            builder.add_lower(
                state=rng.normal(size=2).astype(np.float32),
                cost_state=rng.normal(size=3).astype(np.float32),
                action=rng.normal(size=1).astype(np.float32),
                logp=-0.4,
                value=0.2,
                reward=0.1,
                cost=0.05,
                done=step == 2,
            )
        batch = builder.build()
        self.assertEqual(batch.lower.cost_state.shape, (3, 3))
        metrics = model.update(batch)
        self.assertEqual(metrics["lower_transitions"], 3.0)

    def test_distinct_cost_state_is_required_when_dimensions_differ(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            lower_cost_state_dim=3,
            upper_action_dim=1,
            lower_action_dim=1,
        ))
        with self.assertRaisesRegex(ValueError, "lower cost state"):
            model.act_lower(np.zeros(2, dtype=np.float32), sample=False)
        with self.assertRaisesRegex(ValueError, "explicit cost_state"):
            model.update(self._batch())

    def test_smdp_truncation_uses_duration_aware_bootstrap(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            gamma=0.9,
            gae_lambda=0.95,
        ))
        advantage, returns = model._gae(
            signal=np.asarray([1.0], dtype=np.float32),
            done=np.asarray([1.0], dtype=np.float32),
            duration=np.asarray([2], dtype=np.int64),
            values=np.asarray([2.0], dtype=np.float32),
            next_value=np.asarray([5.0], dtype=np.float32),
            terminal=np.asarray([0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(advantage[0]), 3.05, places=6)
        self.assertAlmostEqual(float(returns[0]), 5.05, places=6)

    def test_zero_cost_batch_cannot_inject_constraint_actor_gradient(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            lower_lambda_init=0.5,
            lower_zero_init_cost_value=True,
            lower_skip_inactive_cost_value_update=True,
        )
        torch.manual_seed(31)
        constrained = FrequencySeparatedActorCriticPPO(config)
        unconstrained = FrequencySeparatedActorCriticPPO(config)
        unconstrained.load_state_dict(copy.deepcopy(constrained.state_dict()))
        unconstrained.constraint_lambda = 0.0
        batch = self._batch(37)
        batch = replace(
            batch,
            lower=replace(
                batch.lower,
                cost=np.zeros(batch.lower.size, dtype=np.float32),
            ),
        )

        np.random.seed(41)
        constrained_metrics = constrained.update(copy.deepcopy(batch))
        np.random.seed(41)
        unconstrained.update(copy.deepcopy(batch))
        constrained_actor = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in constrained.lower_actor.parameters()
        ])
        unconstrained_actor = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in unconstrained.lower_actor.parameters()
        ])
        torch.testing.assert_close(constrained_actor, unconstrained_actor)
        self.assertEqual(constrained_metrics["lower_cost_actor_active"], 0.0)
        self.assertEqual(
            constrained_metrics["lower_cost_value_optimizer_steps"], 0.0
        )
        with torch.no_grad():
            cost_prediction = constrained.lower_cost_value(
                torch.randn(5, config.lower_state_dim)
            )
        torch.testing.assert_close(
            cost_prediction, torch.zeros_like(cost_prediction)
        )

    def test_negligible_cost_stays_below_constraint_activation_threshold(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            lower_lambda_init=0.5,
            lower_cost_activation_threshold=1e-5,
            lower_zero_init_cost_value=True,
            lower_skip_inactive_cost_value_update=True,
        )
        torch.manual_seed(43)
        constrained = FrequencySeparatedActorCriticPPO(config)
        unconstrained = FrequencySeparatedActorCriticPPO(config)
        unconstrained.load_state_dict(copy.deepcopy(constrained.state_dict()))
        unconstrained.constraint_lambda = 0.0
        batch = self._batch(47)
        batch = replace(
            batch,
            lower=replace(
                batch.lower,
                cost=np.full(batch.lower.size, 1e-6, dtype=np.float32),
            ),
        )

        np.random.seed(53)
        metrics = constrained.update(copy.deepcopy(batch))
        np.random.seed(53)
        unconstrained.update(copy.deepcopy(batch))
        constrained_actor = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in constrained.lower_actor.parameters()
        ])
        unconstrained_actor = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in unconstrained.lower_actor.parameters()
        ])
        torch.testing.assert_close(constrained_actor, unconstrained_actor)
        self.assertEqual(metrics["lower_cost_actor_active"], 0.0)
        self.assertEqual(metrics["lower_cost_value_optimizer_steps"], 0.0)

    def test_constraint_gradient_projection_removes_reward_conflict(self):
        reward = [torch.tensor([-2.0, 0.0])]
        constraint = [torch.tensor([2.0, -2.0])]
        projected, diagnostics = _project_constraint_gradients(
            reward, constraint
        )
        self.assertEqual(diagnostics["gradient_conflict"], 1.0)
        self.assertLess(diagnostics["gradient_cosine"], 0.0)
        self.assertIsNotNone(projected[0])
        self.assertAlmostEqual(
            float(torch.dot(reward[0], projected[0]).item()),
            0.0,
            places=6,
        )
        torch.testing.assert_close(
            projected[0], torch.tensor([0.0, -2.0])
        )

    def test_reward_guard_accepts_only_non_regressing_cost_correction(self):
        parameter = torch.nn.Parameter(torch.zeros(2))

        def reward_loss():
            return (parameter[0] - 1.0).square()

        def constraint_loss():
            return (
                (parameter[0] + 1.0).square()
                + (parameter[1] - 1.0).square()
            )

        reward_before = float(reward_loss().item())
        constraint_before = float(constraint_loss().item())
        diagnostics = _reward_guarded_constraint_step(
            parameters=[parameter],
            reward_loss_fn=reward_loss,
            constraint_loss_fn=constraint_loss,
            step_size=0.1,
            max_grad_norm=10.0,
            max_backtracks=4,
            reward_tolerance=0.0,
        )
        self.assertEqual(diagnostics["gradient_conflict"], 1.0)
        self.assertEqual(diagnostics["accepted"], 1.0)
        self.assertLessEqual(float(reward_loss().item()), reward_before)
        self.assertLess(float(constraint_loss().item()), constraint_before)
        self.assertAlmostEqual(float(parameter[0].item()), 0.0, places=6)
        self.assertGreater(float(parameter[1].item()), 0.0)

    def test_reward_guarded_adam_candidate_beats_reward_only_candidate(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([parameter], lr=0.1)

        def reward_loss():
            return (parameter[0] - 1.0).square()

        def constraint_loss():
            return (
                (parameter[0] + 1.0).square()
                + (parameter[1] - 1.0).square()
            )

        diagnostics = _reward_guarded_adam_step(
            parameters=[parameter],
            optimizer=optimizer,
            reward_actor_loss_fn=reward_loss,
            reward_guard_loss_fn=reward_loss,
            constraint_loss_fn=constraint_loss,
            constraint_scale=1.0,
            max_grad_norm=10.0,
            max_backtracks=4,
            reward_tolerance=0.0,
        )
        self.assertEqual(diagnostics["gradient_conflict"], 1.0)
        self.assertEqual(diagnostics["accepted"], 1.0)
        self.assertLessEqual(diagnostics["reward_loss_delta"], 0.0)
        self.assertLess(diagnostics["constraint_loss_delta"], 0.0)
        self.assertGreater(float(parameter[0].item()), 0.0)
        self.assertGreater(float(parameter[1].item()), 0.0)

    def test_reward_guarded_adam_fallback_matches_reward_only_adam(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        reference = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        reference_optimizer = torch.optim.Adam([reference], lr=0.1)

        def reward_loss():
            return (parameter[0] - 1.0).square()

        def constraint_loss():
            return (parameter[0] + 1.0).square()

        diagnostics = _reward_guarded_adam_step(
            parameters=[parameter],
            optimizer=optimizer,
            reward_actor_loss_fn=reward_loss,
            reward_guard_loss_fn=reward_loss,
            constraint_loss_fn=constraint_loss,
            constraint_scale=1.0,
            max_grad_norm=10.0,
            max_backtracks=4,
            reward_tolerance=0.0,
        )
        reference_optimizer.zero_grad()
        (reference[0] - 1.0).square().backward()
        reference_optimizer.step()
        self.assertEqual(diagnostics["gradient_conflict"], 1.0)
        self.assertEqual(diagnostics["accepted"], 0.0)
        torch.testing.assert_close(parameter, reference)
        self.assertEqual(
            optimizer.state_dict()["state"].keys(),
            reference_optimizer.state_dict()["state"].keys(),
        )
        for key, value in optimizer.state_dict()["state"][0].items():
            torch.testing.assert_close(
                value,
                reference_optimizer.state_dict()["state"][0][key],
            )

    def test_reward_guarded_constraint_mode_runs_and_reports_diagnostics(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=8,
            epochs=1,
            minibatch_size=8,
            lower_lambda_init=1.0,
            lower_constraint_update_mode="reward_guarded_adam_projection",
            lower_constraint_max_backtracks=3,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        metrics = model.update(self._batch(59))
        self.assertGreater(metrics["lower_constraint_guard_attempted"], 0.0)
        self.assertIn("lower_constraint_guard_accepted", metrics)
        self.assertIn("lower_constraint_gradient_conflict", metrics)
        self.assertGreaterEqual(
            metrics["lower_constraint_guard_backtracks"], 0.0
        )

    def test_unknown_constraint_update_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "update_mode"):
            FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
                upper_state_dim=3,
                lower_state_dim=2,
                upper_action_dim=1,
                lower_action_dim=1,
                lower_constraint_update_mode="unknown",
            ))

    def test_concatenation_preserves_explicit_cost_bootstrap(self):
        batches = []
        for seed in (43, 47):
            batch = self._batch(seed)
            lower = batch.lower
            batches.append(replace(
                batch,
                lower=replace(
                    lower,
                    next_value=np.zeros(lower.size, dtype=np.float32),
                    terminal=np.asarray(
                        [0.0] * (lower.size - 1) + [1.0], dtype=np.float32
                    ),
                    next_cost_value=np.full(
                        lower.size, 0.25, dtype=np.float32
                    ),
                ),
            ))
        combined = concat_hierarchical_batches(batches)
        self.assertIsNotNone(combined.lower.next_cost_value)
        np.testing.assert_allclose(combined.lower.next_cost_value, 0.25)

    def test_hf_tactical_policy_has_an_independent_ppo_stream(self):
        rng = np.random.default_rng(17)
        builder = HierarchicalRolloutBuilder(gamma=0.99)
        builder.begin_upper(
            state=rng.normal(size=3).astype(np.float32),
            action=rng.normal(size=1).astype(np.float32),
            logp=-0.5,
            value=0.1,
        )
        for step in range(3):
            builder.add_lower(
                state=rng.normal(size=2).astype(np.float32),
                action=rng.normal(size=1).astype(np.float32),
                logp=-0.4,
                value=0.2,
                reward=0.1,
                cost=0.01,
                hf_state=rng.normal(size=4).astype(np.float32),
                hf_action=rng.normal(size=2).astype(np.float32),
                hf_logp=-0.3,
                hf_value=0.05,
                hf_reward=(-0.02 if step == 0 else 0.03),
                done=step == 2,
            )
        batch = builder.build()
        self.assertIsNotNone(batch.hf)
        self.assertEqual(batch.hf.size, batch.lower.size)

        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hf_state_dim=4,
            hf_action_dim=2,
            epochs=1,
            minibatch_size=8,
        ))
        action = model.act_hf(np.zeros(4, dtype=np.float32), sample=False)
        self.assertEqual(np.asarray(action["action"]).shape, (2,))
        metrics = model.update(batch)
        self.assertEqual(metrics["hf_transitions"], 3.0)
        self.assertGreater(metrics["hf_actor_optimizer_steps"], 0.0)
        self.assertGreater(metrics["hf_value_optimizer_steps"], 0.0)
        self.assertIn("hf_actor", model.state_dict())
        self.assertEqual(count_parameters(model), smdp_parameter_count(model.config))

    def test_hf_trajectory_presence_cannot_change_mid_episode(self):
        builder = HierarchicalRolloutBuilder(gamma=0.99)
        builder.begin_upper(
            state=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            logp=0.0,
            value=0.0,
        )
        builder.add_lower(
            state=np.zeros(2, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            logp=0.0,
            value=0.0,
            reward=0.0,
            hf_state=np.zeros(2, dtype=np.float32),
            hf_action=np.zeros(1, dtype=np.float32),
            hf_logp=0.0,
            hf_value=0.0,
            hf_reward=0.0,
            done=False,
        )
        with self.assertRaisesRegex(ValueError, "consistent"):
            builder.add_lower(
                state=np.zeros(2, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )

    def test_learned_promotion_gate_has_independent_ppo_stream(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            promotion_state_dim=4,
            promotion_init_logit=-2.0,
            promotion_entropy_coef=0.0,
            promotion_rate_budget=0.05,
            promotion_rate_coef=1.0,
            promotion_counterfactual_coef=1.0,
            promotion_advantage_learning_rate=1e-2,
            promotion_advantage_coef=1.0,
            promotion_advantage_huber_delta=0.1,
            epochs=1,
            minibatch_size=8,
        ))
        initial = model.act_promotion(
            np.zeros(4, dtype=np.float32), sample=False
        )
        self.assertEqual(initial["action"], 0.0)
        self.assertLess(initial["probability"], 0.5)

        calibrated = model.act_promotion(
            np.zeros(4, dtype=np.float32),
            sample=False,
            deterministic_threshold=0.1,
        )
        self.assertEqual(calibrated["action"], 1.0)
        self.assertAlmostEqual(
            calibrated["probability"], initial["probability"]
        )
        with self.assertRaisesRegex(ValueError, "in \\(0, 1\\)"):
            model.act_promotion(
                np.zeros(4, dtype=np.float32),
                sample=False,
                deterministic_threshold=1.0,
            )
        self.assertIsNotNone(model.promotion_advantage)
        with torch.no_grad():
            for parameter in model.promotion_advantage.parameters():
                parameter.zero_()
            model.promotion_advantage.net[-1].bias.fill_(0.2)
        advantage_action = model.act_promotion(
            np.zeros(4, dtype=np.float32),
            sample=False,
            deterministic_mode="counterfactual_advantage",
            advantage_threshold=0.1,
        )
        self.assertEqual(advantage_action["action"], 1.0)
        self.assertAlmostEqual(
            advantage_action["predicted_counterfactual_advantage"], 0.2
        )
        with torch.no_grad():
            model.promotion_advantage.net[-1].bias.fill_(-0.2)
        advantage_action = model.act_promotion(
            np.zeros(4, dtype=np.float32),
            sample=False,
            deterministic_mode="counterfactual_advantage",
            advantage_threshold=0.1,
        )
        self.assertEqual(advantage_action["action"], 0.0)
        predictions = model.predict_promotion_advantage(
            np.zeros((3, 4), dtype=np.float32)
        )
        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(np.allclose(predictions, -0.2))

        gate = PromotionRolloutBuilder(gamma=0.99)
        gate.begin(
            state=np.ones(4, dtype=np.float32),
            action=1.0,
            logp=-2.0,
            value=0.0,
        )
        gate.add_reward(1.0, counterfactual_advantage=0.5)
        gate.begin(
            state=np.zeros(4, dtype=np.float32),
            action=0.0,
            logp=-0.2,
            value=0.0,
        )
        gate.add_reward(
            0.2, counterfactual_advantage=-0.25, done=True
        )
        batch = self._batch()
        batch.promotion = gate.build()
        metrics = model.update(batch)
        self.assertEqual(metrics["promotion_transitions"], 2.0)
        self.assertGreater(metrics["promotion_actor_optimizer_steps"], 0.0)
        self.assertGreater(metrics["promotion_value_optimizer_steps"], 0.0)
        self.assertGreater(
            metrics["promotion_advantage_optimizer_steps"], 0.0
        )
        self.assertGreater(metrics["promotion_advantage_loss"], 0.0)
        self.assertGreater(metrics["promotion_rate_loss"], 0.0)
        self.assertGreater(metrics["promotion_probability_mean"], 0.05)
        self.assertNotEqual(
            metrics["promotion_counterfactual_surrogate"], 0.0
        )
        self.assertIn("promotion_advantage", model.state_dict())

    def test_concatenation_preserves_level_specific_counts(self):
        batch = concat_hierarchical_batches([self._batch(1), self._batch(2)])
        self.assertEqual(batch.upper.size, 4)
        self.assertEqual(batch.lower.size, 8)
        self.assertEqual(int(np.sum(batch.upper.duration)), batch.lower.size)

    def test_critic_reward_scale_does_not_suppress_actor_step(self):
        config = SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            epochs=1,
            minibatch_size=8,
        )
        torch.manual_seed(19)
        reference = FrequencySeparatedActorCriticPPO(config)
        small = FrequencySeparatedActorCriticPPO(config)
        large = FrequencySeparatedActorCriticPPO(config)
        state = copy.deepcopy(reference.state_dict())
        small.load_state_dict(copy.deepcopy(state))
        large.load_state_dict(copy.deepcopy(state))

        small_batch = self._batch(9)
        large_batch = copy.deepcopy(small_batch)
        for level in (small_batch.upper, small_batch.lower, large_batch.upper, large_batch.lower):
            level.old_value[:] = 0.0
        large_batch.upper.reward *= 1000.0
        large_batch.lower.reward *= 1000.0

        np.random.seed(23)
        small.update(small_batch)
        np.random.seed(23)
        large.update(large_batch)
        small_actor = torch.cat([p.detach().reshape(-1) for p in small.upper_actor.parameters()])
        large_actor = torch.cat([p.detach().reshape(-1) for p in large.upper_actor.parameters()])
        torch.testing.assert_close(small_actor, large_actor, rtol=1e-5, atol=1e-6)


class TradingRoutingContractTest(unittest.TestCase):
    @staticmethod
    def _features(high):
        z = np.asarray([0.1, -0.2], dtype=np.float64)
        return {
            "x_low": z,
            "x_low_forecast": z.reshape(1, -1),
            "x_low_uncertainty": np.abs(z),
            "x_mid": z * 0.5,
            "x_high": np.asarray(high, dtype=np.float64),
            "x_high_delta": np.asarray(high, dtype=np.float64) * 0.1,
            "x_high_energy": np.asarray([0.3, 0.4], dtype=np.float64),
            "x_high_persistence": np.asarray([0.6, 0.7], dtype=np.float64),
            "shock_age": np.asarray([2.0, 2.0], dtype=np.float64),
            "promotion": {"promote": False, "promotion_strength": 0.0, "shock_age": 0.0},
        }

    def test_raw_high_frequency_cannot_change_upper_vector(self):
        position = np.asarray([0.2, -0.1], dtype=np.float64)
        target = np.asarray([0.4, 0.1], dtype=np.float64)
        upper_a, lower_a = frequency_separated_feature_vectors(
            self._features([0.5, -0.5]), position, target
        )
        upper_b, lower_b = frequency_separated_feature_vectors(
            self._features([50.0, -50.0]), position, target
        )
        np.testing.assert_array_equal(upper_a, upper_b)
        self.assertFalse(np.array_equal(lower_a, lower_b))


if __name__ == "__main__":
    unittest.main()
