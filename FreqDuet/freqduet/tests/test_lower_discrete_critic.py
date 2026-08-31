import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lower.resac_lagrangian import RESACLagrangianTrainer


class _ReplayBatch:
    def __init__(self, bins, state_dim=3, seed=17):
        self.bins = np.asarray(bins, dtype=np.float32)
        self.state_dim = int(state_dim)
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_size):
        states = self.rng.normal(
            size=(batch_size, self.state_dim)).astype(np.float32)
        next_states = self.rng.normal(
            size=(batch_size, self.state_dim)).astype(np.float32)
        actions = np.asarray([
            self.bins[i % len(self.bins)] for i in range(batch_size)
        ], dtype=np.float32).reshape(-1, 1)
        rewards = self.rng.normal(size=(batch_size, 1)).astype(np.float32)
        costs = self.rng.uniform(
            0.0, 0.2, size=(batch_size, 1)).astype(np.float32)
        done = np.zeros((batch_size, 1), dtype=np.float32)
        trip_ids = np.arange(batch_size, dtype=np.int64)
        return (
            states, actions, rewards, costs, next_states, done, trip_ids)


class LowerDiscreteCriticTest(unittest.TestCase):
    def setUp(self):
        self.bins = [0.0, 5.0, 15.0]

    def _trainer(self, discrete_critic="continuous_action", bins=None):
        return RESACLagrangianTrainer(
            state_dim=3,
            action_dim=1,
            action_range=15.0,
            action_bins=self.bins if bins is None else bins,
            discrete_critic=discrete_critic,
            ensemble_size=3,
            hidden_dim=8,
            weight_reg_mode="mean",
            policy_sample_seed=23,
        )

    def test_indexed_critic_gathers_exact_action_heads(self):
        trainer = self._trainer(discrete_critic="indexed")
        state = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ], dtype=torch.float32)
        action = torch.tensor([[5.0], [15.0]], dtype=torch.float32)

        all_values = trainer.q_net.all_values(state)
        selected = trainer.q_net(state, action)

        self.assertEqual(tuple(all_values.shape), (3, 2, 3))
        self.assertTrue(torch.equal(selected[:, 0], all_values[:, 0, 1]))
        self.assertTrue(torch.equal(selected[:, 1], all_values[:, 1, 2]))

    def test_indexed_critic_rejects_action_outside_library(self):
        trainer = self._trainer(discrete_critic="indexed")
        with self.assertRaisesRegex(ValueError, "outside its library"):
            trainer.q_net(
                torch.zeros((1, 3), dtype=torch.float32),
                torch.tensor([[7.0]], dtype=torch.float32),
            )

    def test_zero_hold_advantage_is_anchored_at_zero_action(self):
        trainer = self._trainer(discrete_critic="zero_hold_advantage")
        critic = trainer.q_net
        with torch.no_grad():
            for weight, bias in zip(critic.weights, critic.biases):
                weight.zero_()
                bias.zero_()
            critic.biases[-1][:, 0, :] = torch.tensor(
                [2.0, 3.0, -2.0])

        values = critic.all_values(torch.zeros((2, 3), dtype=torch.float32))
        expected = torch.tensor([2.0, 5.0, 0.0], dtype=torch.float32)
        self.assertEqual(critic.biases[-1].shape[-1], len(self.bins))
        self.assertTrue(torch.equal(values[0, 0], expected))
        self.assertTrue(torch.equal(values[2, 1], expected))

    def test_indexed_modes_complete_one_sac_update(self):
        replay = _ReplayBatch(self.bins)
        for mode in ("indexed", "zero_hold_advantage"):
            with self.subTest(mode=mode):
                trainer = self._trainer(discrete_critic=mode)
                metrics = trainer.update(replay, batch_size=12)
                self.assertTrue(np.isfinite(metrics["q_loss"]))
                self.assertTrue(np.isfinite(metrics["policy_loss"]))
                self.assertGreater(metrics["q_action_span_mean"], 0.0)
                self.assertGreater(
                    metrics["q_zero_hold_advantage_abs_mean"], 0.0)

    def test_indexed_mode_requires_categorical_actions(self):
        with self.assertRaisesRegex(ValueError, "requires action_bins"):
            RESACLagrangianTrainer(
                state_dim=3,
                action_dim=1,
                action_bins=None,
                discrete_critic="indexed",
            )

    def test_zero_hold_advantage_requires_zero_second_action(self):
        with self.assertRaisesRegex(ValueError, "zero-second action"):
            self._trainer(
                discrete_critic="zero_hold_advantage",
                bins=[5.0, 10.0, 15.0],
            )

    def test_checkpoint_rejects_different_critic_semantics(self):
        trainer = self._trainer(discrete_critic="indexed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lower.pt"
            trainer.save(path)
            with self.assertRaisesRegex(ValueError, "discrete critic"):
                self._trainer(
                    discrete_critic="zero_hold_advantage").load(path)

    def test_checkpoint_rejects_different_action_library(self):
        trainer = self._trainer(discrete_critic="indexed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lower.pt"
            trainer.save(path)
            with self.assertRaisesRegex(ValueError, "action library"):
                self._trainer(
                    discrete_critic="indexed",
                    bins=[0.0, 10.0, 15.0],
                ).load(path)

    def test_training_checkpoint_locks_discrete_critic(self):
        trainer = self._trainer(discrete_critic="zero_hold_advantage")
        state = trainer.training_state_dict()
        self.assertEqual(state["format"], "freqduet-lower-training-v8")
        self.assertEqual(state["discrete_critic"], "zero_hold_advantage")
        with self.assertRaisesRegex(ValueError, "discrete critic"):
            self._trainer(discrete_critic="indexed").load_training_state_dict(
                state)


if __name__ == "__main__":
    unittest.main()
