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

    def _trainer(self):
        return RESACUpperTrainer(
            state_dim=3,
            action_dim=2,
            action_low=[-60.0, -60.0],
            action_high=[20.0, 20.0],
            action_candidates=self.candidates,
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


if __name__ == '__main__':
    unittest.main()
