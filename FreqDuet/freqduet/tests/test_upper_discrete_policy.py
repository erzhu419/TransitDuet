import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from upper.resac_upper import RESACUpperTrainer


class UpperDiscretePolicyTest(unittest.TestCase):
    def setUp(self):
        self.candidates = np.asarray([
            [-30.0, -30.0],
            [-15.0, 0.0],
            [0.0, 0.0],
            [10.0, 20.0],
        ], dtype=np.float32)

    def _trainer(self, discrete_critic="continuous_action"):
        return RESACUpperTrainer(
            state_dim=3,
            action_dim=2,
            action_low=[-60.0, -60.0],
            action_high=[20.0, 20.0],
            action_candidates=self.candidates,
            discrete_critic=discrete_critic,
            ensemble_size=3,
            hidden_dim=16,
        )

    def test_actions_are_members_of_curve_library(self):
        trainer = self._trainer()
        state = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
        for deterministic in (False, True):
            for _ in range(20):
                action = trainer.policy_net.get_action(
                    state, deterministic=deterministic)
                self.assertTrue(np.any(np.all(
                    np.isclose(self.candidates, action), axis=1)))

    def test_exact_discrete_sac_update_and_checkpoint(self):
        trainer = self._trainer()
        rng = np.random.RandomState(7)
        for i in range(40):
            state = rng.normal(size=3).astype(np.float32)
            action = self.candidates[i % len(self.candidates)]
            next_state = rng.normal(size=3).astype(np.float32)
            trainer.replay_buffer.push(
                state, action, -float(i % 5), next_state, i % 11 == 0)

        metrics = trainer.update(batch_size=16)
        self.assertTrue(np.isfinite(metrics['upper_q_loss']))
        self.assertTrue(np.isfinite(metrics['upper_policy_loss']))
        self.assertGreater(metrics['upper_duration_steps_mean'], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'upper.pt'
            trainer.save(path)
            restored = self._trainer()
            restored.load(path)
            for left, right in zip(
                    trainer.policy_net.parameters(),
                    restored.policy_net.parameters()):
                self.assertTrue(torch.equal(left, right))

            incompatible = RESACUpperTrainer(
                state_dim=3,
                action_dim=2,
                action_low=[-60.0, -60.0],
                action_high=[20.0, 20.0],
                action_candidates=self.candidates + 1.0,
                ensemble_size=3,
                hidden_dim=16,
            )
            with self.assertRaisesRegex(ValueError, "action library"):
                incompatible.load(path)

    def test_log_prob_rejects_actions_outside_library(self):
        trainer = self._trainer()
        with self.assertRaisesRegex(ValueError, "outside its library"):
            trainer.policy_net.log_prob(
                np.zeros(3, dtype=np.float32),
                np.asarray([1.0, 1.0], dtype=np.float32),
            )

    def test_indexed_critic_uses_exact_action_heads(self):
        trainer = self._trainer(discrete_critic="indexed")
        state = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ], dtype=torch.float32)
        actions = torch.from_numpy(np.asarray([
            self.candidates[1],
            self.candidates[3],
        ], dtype=np.float32))

        all_values = trainer.q_net.all_values(state)
        selected = trainer.q_net(state, actions)
        self.assertEqual(tuple(all_values.shape), (3, 2, 4))
        self.assertTrue(torch.equal(selected[:, 0], all_values[:, 0, 1]))
        self.assertTrue(torch.equal(selected[:, 1], all_values[:, 1, 3]))

        rng = np.random.RandomState(11)
        for i in range(40):
            trainer.replay_buffer.push(
                rng.normal(size=3).astype(np.float32),
                self.candidates[i % len(self.candidates)],
                -float(i % 7),
                rng.normal(size=3).astype(np.float32),
                i % 13 == 0,
            )
        metrics = trainer.update(batch_size=16)
        self.assertTrue(np.isfinite(metrics['upper_q_loss']))
        self.assertTrue(np.isfinite(metrics['upper_policy_loss']))

    def test_checkpoint_rejects_different_discrete_critic(self):
        trainer = self._trainer(discrete_critic="indexed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'upper.pt'
            trainer.save(path)
            with self.assertRaisesRegex(ValueError, "discrete critic"):
                self._trainer(discrete_critic="continuous_action").load(path)

    def test_mean_l1_regularization_is_parameter_count_invariant(self):
        trainer = self._trainer(discrete_critic="indexed")
        summed = trainer.q_net.compute_l1_norm("sum")
        averaged = trainer.q_net.compute_l1_norm("mean")
        self.assertTrue(torch.all(summed > averaged))
        self.assertTrue(torch.all(averaged >= 0.0))

    def test_duration_aware_replay_accepts_fractional_steps(self):
        trainer = self._trainer(discrete_critic="indexed")
        rng = np.random.RandomState(29)
        for i in range(40):
            trainer.replay_buffer.push(
                rng.normal(size=3).astype(np.float32),
                self.candidates[i % len(self.candidates)],
                -float(i % 3),
                rng.normal(size=3).astype(np.float32),
                False,
                duration_steps=0.5 + (i % 4) * 0.5,
            )
        metrics = trainer.update(batch_size=16)
        self.assertGreater(metrics["upper_duration_steps_mean"], 0.0)


if __name__ == '__main__':
    unittest.main()
