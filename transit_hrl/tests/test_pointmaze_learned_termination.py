import unittest

import numpy as np

from freq_hrl.experiments.pointmaze_learned_termination import (
    FEATURE_DIM,
    TerminationPPO,
)


class TerminationPPOTest(unittest.TestCase):
    def test_on_policy_update_changes_actor_after_normalization(self):
        policy = TerminationPPO(seed=71, hidden_dim=16)
        warm = []
        for offset in range(4):
            state = np.full(FEATURE_DIM, float(offset), dtype=np.float32)
            warm.append({**policy.act(state, sample=True), "reward": 0.0})
        policy.set_normalizer([warm])
        trajectories = []
        for episode in range(4):
            records = []
            for offset in range(5):
                state = np.full(
                    FEATURE_DIM, float(episode + offset), dtype=np.float32
                )
                record = policy.act(state, sample=True)
                records.append({**record, "reward": -0.1 * (offset + 1)})
            trajectories.append(records)
        before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
        result = policy.update(trajectories)
        self.assertEqual(result["decision_count"], 20)
        self.assertGreater(policy.optimizer_steps, 0)
        self.assertTrue(any(
            not np.array_equal(old.numpy(), new.detach().numpy())
            for old, new in zip(before, policy.actor.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
