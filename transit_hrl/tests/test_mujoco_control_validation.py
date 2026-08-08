import unittest

import numpy as np

from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    action_from_unit_box,
    deterministic_actuation_disturbance,
)
from freq_hrl.experiments.mujoco.control_validation import (
    capacity_matched_flat_hidden_dim,
    environment_dimensions,
    rollout_hierarchical,
    train_mujoco_method,
    _hierarchical_model,
)


def mujoco_available() -> bool:
    try:
        import gymnasium as gym

        env = gym.make("HalfCheetah-v5", render_mode=None, max_episode_steps=8)
        env.reset(seed=1)
        env.close()
        return True
    except Exception:
        return False


class MujocoFrequencyAdapterTest(unittest.TestCase):
    def test_causal_bands_are_invariant_to_future_noise(self):
        prefix = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, -0.5]])

        def encode(future):
            decomposer = CausalBandDecomposer(slow_alpha=0.1, fast_alpha=0.5)
            rows = [decomposer.reset(prefix[0])]
            for row in np.concatenate([prefix[1:], future], axis=0):
                rows.append(decomposer.update(row))
            return rows

        left = encode(np.asarray([[2.0, 2.0], [3.0, 3.0]]))
        right = encode(np.asarray([[-20.0, 10.0], [50.0, -40.0]]))
        for index in range(len(prefix)):
            for band in ("slow", "mid", "high", "delta"):
                np.testing.assert_allclose(left[index][band], right[index][band])

    def test_disturbance_modes_are_deterministic_and_frequency_separated(self):
        low = np.asarray([
            deterministic_actuation_disturbance(
                mode="low_frequency", step=step, action_dim=1, seed=7, horizon=256
            )[0]
            for step in range(256)
        ])
        high = np.asarray([
            deterministic_actuation_disturbance(
                mode="high_frequency", step=step, action_dim=1, seed=7, horizon=256
            )[0]
            for step in range(256)
        ])
        repeated = np.asarray([
            deterministic_actuation_disturbance(
                mode="high_frequency", step=step, action_dim=1, seed=7, horizon=256
            )[0]
            for step in range(256)
        ])
        np.testing.assert_allclose(high, repeated)
        low_roughness = float(np.mean(np.square(np.diff(low))))
        high_roughness = float(np.mean(np.square(np.diff(high))))
        self.assertGreater(high_roughness, 10.0 * low_roughness)

    def test_action_mapping_respects_box_bounds(self):
        mapped = action_from_unit_box(
            np.asarray([-2.0, 0.0, 2.0]),
            np.asarray([-2.0, -1.0, 0.0]),
            np.asarray([2.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(mapped, [-2.0, 1.0, 4.0])

    def test_flat_capacity_match_is_close(self):
        hidden, actual, ratio = capacity_matched_flat_hidden_dim(
            target_parameter_count=40_000,
            state_dim=40,
            action_dim=6,
        )
        self.assertGreater(hidden, 0)
        self.assertGreater(actual, 0)
        self.assertLess(abs(ratio - 1.0), 0.03)


@unittest.skipUnless(mujoco_available(), "MuJoCo runtime is unavailable")
class MujocoControlIntegrationTest(unittest.TestCase):
    def test_hierarchical_rollout_uses_asynchronous_transitions(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", steps=24
        )
        model = _hierarchical_model(
            state_dim=2 * observation_dim + action_dim,
            action_dim=action_dim,
            hidden_dim=8,
            learning_rate=3e-4,
            leakage_constraint=True,
        )
        batch, row = rollout_hierarchical(
            model,
            seed=11,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=24,
            upper_period=6,
            frequency_routing=True,
            leakage_constraint=True,
            sample=True,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.lower.size, 24)
        self.assertEqual(batch.upper.size, 4)
        self.assertEqual(row["protocol_valid"], 1.0)
        self.assertTrue(np.isfinite(row["episode_return"]))

    def test_shared_training_core_smoke(self):
        payload, rows, _ = train_mujoco_method(
            method="freq_hrl_no_leakage",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[41],
            selection_seeds=[43],
            eval_seeds=[47],
            steps=24,
            iterations=1,
            optimizer_seed=53,
            upper_period=6,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=4,
        )
        self.assertEqual(payload["domain"], "mujoco")
        self.assertEqual(payload["protocol_version"], "freq_hrl_mujoco_shared_core_v1")
        self.assertTrue(payload["frequency_routing_enabled"])
        self.assertEqual(payload["checkpoint_evaluation_interval"], 4)
        self.assertEqual(payload["checkpoint_validation_observation_count"], 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["protocol_valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
