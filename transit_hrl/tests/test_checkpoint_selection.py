import unittest

import numpy as np

from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    RobustValidationCheckpointSelector,
    SMDPPPOConfig,
)
from freq_hrl.rl.training import train_frequency_separated_ppo, train_joint_ppo


class RobustValidationCheckpointSelectorTest(unittest.TestCase):
    def test_trailing_window_rejects_an_isolated_validation_spike(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
            smoothing_window=3,
            min_delta=0.05,
        )

        first = selector.consider(
            score=0.6, state={"step": 0}, iteration=0
        )
        second = selector.consider(
            score=-0.6, state={"step": 1}, iteration=1
        )
        third = selector.consider(
            score=0.2, state={"step": 2}, iteration=2
        )

        self.assertFalse(first["checkpoint_selection_eligible"])
        self.assertFalse(first["checkpoint_selected"])
        self.assertFalse(second["checkpoint_selected"])
        self.assertTrue(third["checkpoint_selected"])
        self.assertEqual(selector.selected_iteration, 2)
        self.assertEqual(selector.best_state, {"step": 2})

        metadata = selector.metadata(total_iterations=5)
        self.assertEqual(
            metadata["checkpoint_selection_protocol"],
            "trailing_mean_material_improvement_v1",
        )
        self.assertEqual(metadata["checkpoint_smoothing_window"], 3)
        self.assertEqual(metadata["checkpoint_plateau_tail_iterations"], 2)

    def test_window_one_preserves_strict_best_score_selection(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
        )
        selector.consider(score=0.1, state={"step": 0}, iteration=0)
        selector.consider(score=0.05, state={"step": 1}, iteration=1)

        self.assertAlmostEqual(selector.best_score, 0.1)
        self.assertEqual(selector.selected_iteration, 0)
        self.assertEqual(selector.best_state, {"step": 0})
        self.assertEqual(
            selector.metadata(total_iterations=2)[
                "checkpoint_selection_protocol"
            ],
            "disjoint_validation_paths",
        )

    def test_invalid_selection_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                smoothing_window=0,
            )
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                min_delta=-1e-3,
            )

    def test_smdp_trainer_accepts_a_registered_custom_score_contract(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))
        score_calls = []

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
            }

        def robust_score(rows):
            score_calls.append(len(rows))
            return min(float(row["reward_mean"]) for row in rows)

        payload, heldout, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10, 20],
            eval_seeds=[30],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_score_fn=robust_score,
            checkpoint_score_contract="minimum_validation_reward_v1",
        )

        self.assertEqual(score_calls, [2, 2])
        self.assertEqual(payload["initial_validation_score"], 10.0)
        self.assertEqual(
            payload["checkpoint_score_contract"],
            "minimum_validation_reward_v1",
        )
        self.assertEqual(heldout[0]["reward_mean"], 30.0)

    def test_joint_trainer_uses_the_same_custom_score_contract(self):
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=1,
            action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))

        def rollout_fn(_model, seed, train):
            batch = JointTrajectoryBatch(
                state=np.zeros((1, 1), dtype=np.float32),
                action=np.zeros((1, 1), dtype=np.float32),
                reward=np.zeros(1, dtype=np.float32),
                done=np.ones(1, dtype=np.float32),
                old_logp=np.zeros(1, dtype=np.float32),
                old_value=np.zeros(1, dtype=np.float32),
            )
            return (batch if train else None), {"reward_mean": float(seed)}

        payload, _, _ = train_joint_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10, 20],
            eval_seeds=[],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_score_fn=lambda rows: min(
                float(row["reward_mean"]) for row in rows
            ),
            checkpoint_score_contract="minimum_validation_reward_v1",
        )
        self.assertEqual(payload["initial_validation_score"], 10.0)
        self.assertEqual(
            payload["checkpoint_score_contract"],
            "minimum_validation_reward_v1",
        )


if __name__ == "__main__":
    unittest.main()
