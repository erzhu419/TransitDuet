import unittest

import numpy as np

from freq_hrl.experiments.pointmaze_learned_termination import (
    FEATURE_DIM,
    TerminationPPO,
    credited_records,
)


class TerminationPPOTest(unittest.TestCase):
    def test_full_return_credit_does_not_decay_with_decision_count(self):
        episode = [
            {"reward": -1.0, "value": 0.0},
            {"reward": -2.0, "value": 0.0},
        ]
        mc = credited_records([episode], gae_lambda=1.0)
        old = credited_records([episode], gae_lambda=0.95)
        more_checks = credited_records([[
            {"reward": -1.0, "value": 0.0},
            {"reward": -1.0, "value": 0.0},
            {"reward": -1.0, "value": 0.0},
        ]], gae_lambda=1.0)
        self.assertEqual([r["return_target"] for r in mc], [-2.0, -3.0])
        self.assertEqual(more_checks[-1]["return_target"], mc[-1]["return_target"])
        self.assertAlmostEqual(old[1]["return_target"], -2.9)

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
