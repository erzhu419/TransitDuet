import copy
import unittest

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
from freq_hrl.experiments.trading.ppo_actor_critic import frequency_separated_feature_vectors


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
        builder.add_reward(1.0)
        builder.add_reward(2.0)
        builder.begin(
            state=np.asarray([0.0, 1.0], dtype=np.float32),
            action=0.0,
            logp=-0.3,
            value=0.2,
        )
        builder.add_reward(3.0, done=True)
        batch = builder.build()
        self.assertIsNotNone(batch)
        np.testing.assert_array_equal(batch.duration, [2, 1])
        np.testing.assert_allclose(batch.reward, [2.8, 3.0])
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

    def test_learned_promotion_gate_has_independent_ppo_stream(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            promotion_state_dim=4,
            promotion_init_logit=-2.0,
            epochs=1,
            minibatch_size=8,
        ))
        initial = model.act_promotion(
            np.zeros(4, dtype=np.float32), sample=False
        )
        self.assertEqual(initial["action"], 0.0)
        self.assertLess(initial["probability"], 0.5)

        gate = PromotionRolloutBuilder(gamma=0.99)
        gate.begin(
            state=np.ones(4, dtype=np.float32),
            action=1.0,
            logp=-2.0,
            value=0.0,
        )
        gate.add_reward(1.0)
        gate.begin(
            state=np.zeros(4, dtype=np.float32),
            action=0.0,
            logp=-0.2,
            value=0.0,
        )
        gate.add_reward(0.2, done=True)
        batch = self._batch()
        batch.promotion = gate.build()
        metrics = model.update(batch)
        self.assertEqual(metrics["promotion_transitions"], 2.0)
        self.assertGreater(metrics["promotion_actor_optimizer_steps"], 0.0)
        self.assertGreater(metrics["promotion_value_optimizer_steps"], 0.0)

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
