import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    action_from_unit_box,
    deterministic_actuation_disturbance,
)
from freq_hrl.experiments.mujoco.control_validation import (
    _with_explicit_bootstrap,
    capacity_matched_flat_hidden_dim,
    environment_dimensions,
    rollout_hierarchical,
    train_mujoco_method,
    write_cell,
    _hierarchical_model,
)
from freq_hrl.rl import JointTrajectoryBatch


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
    def test_flat_batch_bootstrap_does_not_require_cost_fields(self):
        batch = JointTrajectoryBatch(
            state=np.zeros((2, 3), dtype=np.float32),
            action=np.zeros((2, 1), dtype=np.float32),
            reward=np.ones(2, dtype=np.float32),
            done=np.asarray([0.0, 1.0], dtype=np.float32),
            old_logp=np.zeros(2, dtype=np.float32),
            old_value=np.asarray([0.25, 0.5], dtype=np.float32),
        )
        bootstrapped = _with_explicit_bootstrap(
            batch,
            boundary_next_values=[0.75],
            boundary_terminals=[0.0],
        )
        np.testing.assert_allclose(bootstrapped.next_value, [0.5, 0.75])
        np.testing.assert_allclose(bootstrapped.terminal, [0.0, 0.0])

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

    def test_written_checkpoint_has_independent_file_hash(self):
        model = torch.nn.Linear(2, 1)
        payload = {
            "history": [{"iteration": 0}],
            "method": "unit",
            "environment": "unit",
            "disturbance_mode": "standard",
            "frozen_parameter_sha256": "a" * 64,
            "frozen_checkpoint_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_cell(output, payload, [{"metric": 1.0}], model)
            summary = json.loads(
                (output / "cell_summary.json").read_text(encoding="utf-8")
            )
            actual = hashlib.sha256(
                (output / "checkpoint.pt").read_bytes()
            ).hexdigest()
            self.assertEqual(summary["checkpoint_file_sha256"], actual)
            self.assertEqual(
                summary["checkpoint_integrity_contract"],
                "independent_parameter_and_serialized_file_sha256_v1",
            )


@unittest.skipUnless(mujoco_available(), "MuJoCo runtime is unavailable")
class MujocoControlIntegrationTest(unittest.TestCase):
    def test_hierarchical_rollout_uses_asynchronous_transitions(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", episode_horizon=24
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
        self.assertIsNotNone(batch.upper.next_value)
        self.assertIsNotNone(batch.lower.next_value)
        self.assertIsNotNone(batch.lower.next_cost_value)
        self.assertTrue(np.any(np.abs(batch.lower.next_cost_value[:-1]) > 0.0))
        self.assertEqual(float(batch.upper.terminal[-1]), 0.0)
        self.assertEqual(float(batch.lower.terminal[-1]), 0.0)
        self.assertEqual(row["bootstrap_boundary_count"], 1)

    def test_training_budget_continues_across_hopper_terminations(self):
        observation_dim, action_dim = environment_dimensions(
            "Hopper-v5", episode_horizon=1000
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
            seed=1,
            env_id="Hopper-v5",
            disturbance_mode="mixed",
            steps=128,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=True,
            sample=True,
            episode_horizon=1000,
        )
        self.assertEqual(batch.lower.size, 128)
        self.assertGreater(row["natural_episode_count"], 0)
        self.assertEqual(row["transition_budget_exact"], 1.0)
        self.assertGreater(row["mdp_terminal_count"], 0)
        self.assertEqual(
            batch.lower.next_value.shape,
            batch.lower.old_value.shape,
        )

    def test_shared_training_core_smoke(self):
        payload, rows, _ = train_mujoco_method(
            method="freq_hrl_no_leakage",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[41],
            selection_seeds=[43],
            eval_seeds=[47],
            steps=24,
            episode_horizon=32,
            iterations=1,
            optimizer_seed=53,
            upper_period=6,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=4,
            evaluation_disturbance_modes=["standard", "ood_chirp"],
        )
        self.assertEqual(payload["domain"], "mujoco")
        self.assertEqual(payload["protocol_version"], "freq_hrl_mujoco_shared_core_v4")
        self.assertTrue(payload["frequency_routing_enabled"])
        self.assertEqual(payload["checkpoint_evaluation_interval"], 4)
        self.assertEqual(payload["checkpoint_validation_observation_count"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["disturbance_mode"] for row in rows},
            {"standard", "ood_chirp"},
        )
        self.assertTrue(all(row["protocol_valid"] == 1.0 for row in rows))
        self.assertEqual(
            payload["history"][-1]["sampled_transition_budget_exact_mean"],
            1.0,
        )
        self.assertEqual(payload["evaluation_episode_horizon"], 32)
        self.assertEqual(
            payload["bootstrap_contract"],
            "explicit_reward_and_cost_next_value_with_separate_trace_boundary_"
            "and_mdp_terminal",
        )
        self.assertIsNotNone(payload["history"][-1]["lower_cost_actor_active"])
        self.assertEqual(len(payload["frozen_checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
