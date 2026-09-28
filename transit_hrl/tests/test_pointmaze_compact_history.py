import unittest

import numpy as np

from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    _compact_history, pointmaze_plan_value_checkpoint_rank, summarize_numeric_rows,
)
from freq_hrl.rl import FrequencySeparatedActorCriticPPO, HierarchicalRolloutBuilder, SMDPPPOConfig
from freq_hrl.rl.training import train_frequency_separated_ppo


class CompactPointMazeHistoryTest(unittest.TestCase):
    def test_real_trainer_keys_survive_compaction(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1, lower_state_dim=1, upper_action_dim=1, lower_action_dim=1,
            hidden_dim=4, epochs=1, minibatch_size=4))

        def rollout(_model, seed, sample):
            builder = HierarchicalRolloutBuilder(gamma=.99)
            zero = np.zeros(1, dtype=np.float32)
            builder.begin_upper(state=zero, action=zero, logp=0., value=0.)
            builder.add_lower(state=zero, action=zero, logp=0., value=0., reward=0., done=True)
            return (builder.build() if sample else None), {
                "episode_return": float(seed), "tracking_squared_error_integral": 100. - seed,
                "tracking_success_rate": .5, "tracking_rmse": 2., "episode_length": 1}

        payload, _, _ = train_frequency_separated_ppo(
            model=model, train_seeds=[1], selection_seeds=[10, 20], eval_seeds=[30],
            iterations=3, rollout_fn=rollout, objective_fn=lambda r: r["episode_return"],
            summary_fn=summarize_numeric_rows, training_seed_fn=lambda seed, i: seed + i,
            checkpoint_evaluation_interval=2, checkpoint_minimum_iteration=0,
            checkpoint_rank_fn=pointmaze_plan_value_checkpoint_rank,
            checkpoint_rank_names=("negative_mean_tracking_squared_error_integral", "mean_dense_episode_return"),
            checkpoint_rank_contract="lexicographic_negative_mean_tracking_squared_error_integral_then_mean_dense_return_v1")
        compact = _compact_history(payload["history"])
        self.assertEqual([r["iteration"] for r in compact], [-1, 0, 1, 2])
        self.assertEqual([r["checkpoint_evaluation_performed"] for r in compact], [True, False, True, True])
        self.assertEqual(compact[0]["score"], 15.)
        self.assertIsNone(compact[1]["score"])
        self.assertEqual(compact[2]["checkpoint_selection_rank"], {
            "negative_mean_tracking_squared_error_integral": -85., "mean_dense_episode_return": 15.})
        self.assertEqual(compact[2]["episode_return_mean"], 15.)
        self.assertEqual(compact[2]["tracking_squared_error_integral_mean"], 85.)
        self.assertEqual([r["training_rollout_seeds"] for r in compact[1:]], [[1], [2], [3]])
        self.assertEqual([r["sampled_episode_length_mean"] for r in compact[1:]], [1., 1., 1.])
        self.assertFalse(compact[0]["checkpoint_selection_eligible"])
        self.assertTrue(compact[2]["checkpoint_selection_eligible"])
        self.assertEqual([r["iteration"] for r in compact if r["checkpoint_selected"]],
                         [payload["selected_checkpoint_iteration"]])
        for original, row in zip(payload["history"], compact):
            for key in ("upper_policy_loss", "upper_value_loss", "lower_policy_loss", "lower_value_loss"):
                self.assertEqual(row[key], original[key])
        self.assertLess(len(compact[-1]), len(payload["history"][-1]))


if __name__ == "__main__":
    unittest.main()
