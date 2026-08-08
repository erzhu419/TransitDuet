import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    CausalResponsibilityTransfer,
    action_from_unit_box,
    deterministic_actuation_disturbance,
)
from freq_hrl.experiments.mujoco.control_validation import (
    SAFE_SELECTOR_BASELINE_BRANCH,
    _with_explicit_bootstrap,
    capacity_matched_flat_hidden_dim,
    environment_dimensions,
    mujoco_policy_state_dim,
    rollout_hierarchical,
    select_safe_mujoco_branch,
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
    @staticmethod
    def _selector_rows(
        reward: float,
        drift: float,
        raw_drift: float | None = None,
        upper_hf_power: float = 0.0025,
    ):
        return [
            {
                "disturbance_mode": mode,
                "seed": seed,
                "episode_return": reward,
                "LowerLFDriftAbs": drift,
                "RawLowerLFDriftAbs": (
                    drift if raw_drift is None else raw_drift
                ),
                "UpperHFPowerAbs": upper_hf_power,
            }
            for seed in (1, 2)
            for mode in ("standard", "mixed")
        ]

    def test_safe_selector_requires_reward_floor_and_drift_reduction(self):
        result = select_safe_mujoco_branch({
            SAFE_SELECTOR_BASELINE_BRANCH: self._selector_rows(100.0, 1.0),
            "responsibility_guarded_adam_projection": self._selector_rows(
                99.5, 0.7, raw_drift=0.7
            ),
            "behavior_guarded_adam_projection": self._selector_rows(
                80.0, 0.2, raw_drift=0.2
            ),
            "behavior_guarded_upper_smooth": self._selector_rows(
                80.0, 0.2, raw_drift=0.2
            ),
            "behavior_scalarized_upper_smooth": self._selector_rows(
                80.0, 0.2, raw_drift=0.2
            ),
        }, bootstrap_seed=7, bootstrap_draws=200)
        self.assertEqual(
            result["selected_branch"],
            "responsibility_guarded_adam_projection",
        )
        self.assertTrue(result["branch_diagnostics"][
            "responsibility_guarded_adam_projection"
        ]["feasible"])
        self.assertFalse(
            result["branch_diagnostics"][
                "behavior_guarded_adam_projection"
            ]["feasible"]
        )

    def test_safe_selector_falls_back_when_no_candidate_is_pareto_safe(self):
        result = select_safe_mujoco_branch({
            SAFE_SELECTOR_BASELINE_BRANCH: self._selector_rows(100.0, 1.0),
            "responsibility_guarded_adam_projection": self._selector_rows(
                95.0, 0.5
            ),
            "behavior_guarded_adam_projection": self._selector_rows(
                101.0, 0.95
            ),
            "behavior_guarded_upper_smooth": self._selector_rows(
                95.0, 0.5
            ),
            "behavior_scalarized_upper_smooth": self._selector_rows(
                101.0, 0.95
            ),
        }, bootstrap_seed=11, bootstrap_draws=200)
        self.assertEqual(result["selected_branch"], SAFE_SELECTOR_BASELINE_BRANCH)
        self.assertEqual(
            result["selection_status"], "fallback_to_no_leakage"
        )

    def test_safe_selector_rejects_responsibility_only_improvement(self):
        rows = {
            SAFE_SELECTOR_BASELINE_BRANCH: self._selector_rows(
                100.0, 1.0, raw_drift=1.0
            ),
            "responsibility_guarded_adam_projection": self._selector_rows(
                100.0, 0.6, raw_drift=1.1
            ),
            "behavior_guarded_adam_projection": self._selector_rows(
                100.0, 0.6, raw_drift=0.6
            ),
            "behavior_guarded_upper_smooth": self._selector_rows(
                100.0, 0.6, raw_drift=0.6, upper_hf_power=0.04
            ),
            "behavior_scalarized_upper_smooth": self._selector_rows(
                80.0, 0.2, raw_drift=0.2
            ),
        }
        result = select_safe_mujoco_branch(
            rows, bootstrap_seed=13, bootstrap_draws=200
        )
        self.assertEqual(
            result["selected_branch"], "behavior_guarded_adam_projection"
        )
        self.assertFalse(result["branch_diagnostics"][
            "responsibility_guarded_adam_projection"
        ]["minimum_raw_drift_reduction_supported"])
        self.assertFalse(result["branch_diagnostics"][
            "behavior_guarded_upper_smooth"
        ]["upper_hf_budget_supported"])

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

    def test_responsibility_transfer_is_causal_and_exactly_reconstructs(self):
        def trace(future):
            transfer = CausalResponsibilityTransfer(
                mode="causal_lf_transfer", alpha=0.25
            )
            transfer.reset(2)
            rows = []
            sequence = [
                np.asarray([0.4, -0.2]),
                np.asarray([0.6, -0.4]),
                *future,
            ]
            for index, raw_lower in enumerate(sequence):
                if index % 2 == 0:
                    assignment = transfer.begin_macro(
                        np.asarray([0.2, 0.85])
                    )
                split = transfer.split_lower(raw_lower)
                np.testing.assert_allclose(
                    np.asarray(assignment["upper_responsibility"])
                    + np.asarray(split["lower_responsibility"]),
                    np.asarray(assignment["upper_policy"]) + raw_lower,
                    atol=1e-7,
                )
                rows.append((
                    np.asarray(assignment["upper_responsibility"]).copy(),
                    np.asarray(split["lower_responsibility"]).copy(),
                    np.asarray(split["raw_lower_lf_after"]).copy(),
                ))
            return rows

        left = trace([np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])])
        right = trace([np.asarray([-1.0, -1.0]), np.asarray([-1.0, -1.0])])
        for left_row, right_row in zip(left[:2], right[:2]):
            for left_value, right_value in zip(left_row, right_row):
                np.testing.assert_allclose(left_value, right_value)

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
    def test_canonical_policy_state_is_pathwise_decomposition_invariant(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", episode_horizon=64
        )
        model = _hierarchical_model(
            state_dim=mujoco_policy_state_dim(observation_dim, action_dim),
            action_dim=action_dim,
            hidden_dim=8,
            learning_rate=3e-4,
            leakage_constraint=False,
        )
        common = dict(
            seed=127,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=64,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=False,
            sample=True,
            episode_horizon=64,
        )
        torch.manual_seed(131)
        np.random.seed(131)
        additive_batch, additive = rollout_hierarchical(
            model, responsibility_mode="additive", **common
        )
        torch.manual_seed(131)
        np.random.seed(131)
        transfer_batch, transfer = rollout_hierarchical(
            model, responsibility_mode="causal_lf_transfer", **common
        )

        np.testing.assert_array_equal(
            additive_batch.upper.state, transfer_batch.upper.state
        )
        np.testing.assert_array_equal(
            additive_batch.lower.state, transfer_batch.lower.state
        )
        np.testing.assert_array_equal(
            additive_batch.upper.action, transfer_batch.upper.action
        )
        np.testing.assert_array_equal(
            additive_batch.lower.action, transfer_batch.lower.action
        )
        np.testing.assert_array_equal(
            additive_batch.lower.reward, transfer_batch.lower.reward
        )
        self.assertEqual(
            additive["episode_return"], transfer["episode_return"]
        )
        self.assertEqual(
            additive["RawLowerActionRMS"], transfer["RawLowerActionRMS"]
        )
        self.assertEqual(
            additive["RawLowerLFDriftAbs"],
            transfer["RawLowerLFDriftAbs"],
        )
        self.assertLessEqual(
            transfer["ResponsibilityReconstructionRMS"], 1e-7
        )
        self.assertFalse(np.array_equal(
            additive_batch.lower.cost_state,
            transfer_batch.lower.cost_state,
        ))

    def test_hierarchical_rollout_uses_asynchronous_transitions(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", episode_horizon=24
        )
        model = _hierarchical_model(
            state_dim=mujoco_policy_state_dim(observation_dim, action_dim),
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
        self.assertEqual(
            batch.lower.next_cost_value.shape,
            batch.lower.old_value.shape,
        )
        np.testing.assert_allclose(batch.lower.next_cost_value, 0.0)
        self.assertEqual(float(batch.upper.terminal[-1]), 0.0)
        self.assertEqual(float(batch.lower.terminal[-1]), 0.0)
        self.assertEqual(row["bootstrap_boundary_count"], 1)

    def test_behavior_constraint_and_upper_penalty_do_not_relabel_return(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", episode_horizon=64
        )
        model = _hierarchical_model(
            state_dim=mujoco_policy_state_dim(observation_dim, action_dim),
            action_dim=action_dim,
            hidden_dim=8,
            learning_rate=3e-4,
            leakage_constraint=True,
        )
        common = dict(
            seed=211,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=64,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=True,
            lower_lf_rms_budget=1e-3,
            responsibility_mode="causal_lf_transfer",
            sample=True,
            episode_horizon=64,
        )
        torch.manual_seed(223)
        np.random.seed(223)
        responsibility_batch, responsibility_row = rollout_hierarchical(
            model,
            leakage_constraint_scope="responsibility",
            upper_transition_rms_budget=1e-4,
            upper_transition_penalty_coef=0.0,
            **common,
        )
        torch.manual_seed(223)
        np.random.seed(223)
        behavior_batch, behavior_row = rollout_hierarchical(
            model,
            leakage_constraint_scope="joint_behavior",
            upper_transition_rms_budget=1e-4,
            upper_transition_penalty_coef=1.0,
            **common,
        )
        np.testing.assert_array_equal(
            responsibility_batch.lower.reward,
            behavior_batch.lower.reward,
        )
        self.assertEqual(
            responsibility_row["episode_return"],
            behavior_row["episode_return"],
        )
        self.assertTrue(np.all(
            behavior_batch.lower.cost + 1e-12
            >= responsibility_batch.lower.cost
        ))
        self.assertGreater(
            behavior_row["UpperContinuityPenaltyTotal"], 0.0
        )
        self.assertLess(
            float(np.sum(behavior_batch.upper.reward)),
            float(np.sum(responsibility_batch.upper.reward)),
        )

    def test_training_budget_continues_across_hopper_terminations(self):
        observation_dim, action_dim = environment_dimensions(
            "Hopper-v5", episode_horizon=1000
        )
        model = _hierarchical_model(
            state_dim=mujoco_policy_state_dim(observation_dim, action_dim),
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
            responsibility_mode="causal_lf_transfer",
        )
        self.assertEqual(payload["domain"], "mujoco")
        self.assertEqual(
            payload["protocol_version"],
            "freq_hrl_mujoco_shared_core_v14_behavior_safe_training",
        )
        self.assertTrue(payload["frequency_routing_enabled"])
        self.assertEqual(payload["training_disturbance_modes"], ["standard"])
        self.assertEqual(payload["upper_action_scale"], 1.0)
        self.assertEqual(payload["lower_action_scale"], 1.0)
        self.assertEqual(payload["responsibility_mode"], "causal_lf_transfer")
        self.assertEqual(
            payload["policy_filter_state_contract"],
            "canonical_raw_lf_and_previous_raw_lower_actor_state_v1",
        )
        self.assertEqual(
            payload["lower_cost_state_contract"],
            "causal_responsibility_anchor_raw_lower_lf_and_"
            "responsibility_lf_cost_critic_only_v2",
        )
        self.assertEqual(payload["role_capacity_status"], "symmetric")
        self.assertEqual(payload["upper_to_lower_action_capacity_ratio"], 1.0)
        self.assertEqual(
            payload["lower_constraint_update_mode"],
            "reward_guarded_adam_projection",
        )
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
        for metric in (
            "UpperActionRMS",
            "LowerActionRMS",
            "UpperActionEnergyShare",
            "AdditiveActionClipRate",
            "RawLowerLFDriftAbs",
            "RawLowerLFRmsOnlineMean",
            "LowerConstraintCostMean",
            "UpperTransitionDeltaRMSMean",
            "UpperContinuityPenaltyTotal",
            "ResponsibilityTransferRMS",
            "ResponsibilityReconstructionRMS",
        ):
            self.assertIn(metric, rows[0])
            self.assertTrue(np.isfinite(rows[0][metric]))
        self.assertLessEqual(rows[0]["ResponsibilityReconstructionRMS"], 1e-7)
        self.assertEqual(
            payload["bootstrap_contract"],
            "explicit_reward_and_cost_next_value_with_separate_trace_boundary_"
            "and_mdp_terminal",
        )
        self.assertIsNotNone(payload["history"][-1]["lower_cost_actor_active"])
        self.assertEqual(len(payload["frozen_checkpoint_sha256"]), 64)

    def test_safe_selector_keeps_branch_selection_out_of_heldout_paths(self):
        payload, rows, _ = train_mujoco_method(
            method="freq_hrl_safe_selector",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[101],
            selection_seeds=[103],
            safety_selection_seeds=[107],
            eval_seeds=[109],
            steps=16,
            episode_horizon=16,
            iterations=1,
            optimizer_seed=113,
            upper_period=4,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            training_disturbance_modes=["standard"],
            evaluation_disturbance_modes=["standard"],
        )
        self.assertEqual(payload["safe_selector_training_compute_multiplier"], 5)
        self.assertEqual(payload["safe_selector_selection_seeds"], [107])
        self.assertEqual(payload["branch_training_eval_seeds"], [])
        self.assertEqual(payload["eval_seeds"], [109])
        self.assertEqual(
            payload["heldout_test_access_status"],
            "loaded_once_after_safe_branch_selection",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 109)
        self.assertEqual(rows[0]["evaluation_role"], "heldout_test")
        self.assertIn(
            payload["safe_selector"]["selected_branch"],
            payload["safe_selector_branch_training"],
        )

    def test_standard_condition_keeps_frequency_and_generic_inputs_equivalent(self):
        common = dict(
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[61],
            selection_seeds=[67],
            eval_seeds=[71],
            steps=16,
            episode_horizon=16,
            iterations=1,
            optimizer_seed=73,
            upper_period=4,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            training_disturbance_modes=["standard"],
            evaluation_disturbance_modes=["standard"],
        )
        frequency, frequency_rows, _ = train_mujoco_method(
            method="freq_hrl_no_leakage", **common
        )
        generic, generic_rows, _ = train_mujoco_method(
            method="generic_hrl", **common
        )
        self.assertEqual(
            frequency["frozen_parameter_sha256"],
            generic["frozen_parameter_sha256"],
        )
        self.assertEqual(
            frequency_rows[0]["episode_return"],
            generic_rows[0]["episode_return"],
        )


if __name__ == "__main__":
    unittest.main()
