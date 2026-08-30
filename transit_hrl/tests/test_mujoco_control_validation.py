import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    CausalLowerActionRouter,
    CausalResponsibilityTransfer,
    action_from_unit_box,
    deterministic_actuation_disturbance,
)
from freq_hrl.experiments.mujoco.control_validation import (
    SAFE_SELECTOR_BASELINE_BRANCH,
    MUJOCO_CONTROL_PROTOCOL_VERSION,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17,
    _model_parameter_sha256,
    _leakage_constraint_cost,
    _with_explicit_bootstrap,
    behavior_robust_checkpoint_diagnostics,
    capacity_matched_flat_hidden_dim,
    crossed_checkpoint_selection_paths,
    crossed_deployment_frequency_guard_paths,
    crossed_deployment_frequency_reference_paths,
    deployment_frequency_constraint_contract,
    environment_dimensions,
    latent_behavior_feasibility_rank,
    paired_relative_frequency_feasibility_diagnostics,
    paired_relative_frequency_feasibility_rank,
    paired_closed_loop_guard_snapshot,
    lower_action_router_training_strength,
    load_paired_mujoco_checkpoint,
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
    def test_deployment_constraint_contract_separates_guard_ablations(self):
        contracts = {
            (replay, trust, closed): deployment_frequency_constraint_contract(
                requested=True,
                groupwise=True,
                anchor_state_replay=replay,
                ppo_trust_region=trust,
                closed_loop_trust_region=closed,
            )
            for replay in (False, True)
            for trust in (False, True)
            for closed in (False, True)
        }
        self.assertEqual(len(set(contracts.values())), 8)
        self.assertIn("state_replay", contracts[(True, False, False)])
        self.assertIn("trust_region", contracts[(False, True, False)])
        self.assertIn("state_replay", contracts[(True, True, True)])
        self.assertIn("joint_actor", contracts[(False, False, True)])
        cvar = deployment_frequency_constraint_contract(
            requested=True,
            groupwise=True,
            anchor_state_replay=True,
            ppo_trust_region=True,
            closed_loop_trust_region=True,
            projection_objective="violation_cvar",
            projection_cvar_alpha=0.5,
            closed_loop_risk_mode="mode_cvar",
            closed_loop_cvar_alpha=0.5,
        )
        self.assertIn("projection_cvar_alpha_0.5", cvar)
        self.assertIn("mode_cvar_constraints_alpha_0.5", cvar)
        self.assertTrue(cvar.endswith("v10"))

    def test_v1417_mechanism_cannot_use_the_v1416_protocol_label(self):
        with self.assertRaisesRegex(ValueError, "v14.17 mechanisms"):
            train_mujoco_method(
                method="freq_hrl",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                train_seeds=[11],
                selection_seeds=[13],
                eval_seeds=[17],
                steps=8,
                iterations=1,
                optimizer_seed=19,
                constraint_dual_normalization="ema_abs",
                control_protocol_version=(
                    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16
                ),
            )
        self.assertTrue(MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17.endswith(
            "native_pd_cvar"
        ))

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
            "behavior_guarded_upper_hf": self._selector_rows(
                80.0, 0.2, raw_drift=0.2
            ),
            "behavior_scalarized_upper_hf": self._selector_rows(
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

    def test_router_homotopy_has_registered_linear_and_cosine_paths(self):
        linear = [
            lower_action_router_training_strength(
                iteration=iteration,
                total_iterations=4,
                target_strength=0.1,
                schedule="delayed_linear",
                warmup_fraction=0.25,
                ramp_fraction=0.5,
            )
            for iteration in range(4)
        ]
        self.assertTrue(np.allclose(linear, [0.0, 0.05, 0.1, 0.1]))
        cosine = lower_action_router_training_strength(
            iteration=2,
            total_iterations=8,
            target_strength=0.1,
            schedule="delayed_cosine",
            warmup_fraction=0.25,
            ramp_fraction=0.5,
        )
        expected = 0.1 * (0.5 - 0.5 * np.cos(np.pi * 0.25))
        self.assertAlmostEqual(cosine, expected)
        self.assertEqual(
            mujoco_policy_state_dim(
                17, 6, observe_router_strength=True
            ),
            mujoco_policy_state_dim(17, 6) + 1,
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
            "behavior_guarded_upper_hf": self._selector_rows(
                95.0, 0.5
            ),
            "behavior_scalarized_upper_hf": self._selector_rows(
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
            "behavior_guarded_upper_hf": self._selector_rows(
                100.0, 0.6, raw_drift=0.6, upper_hf_power=0.04
            ),
            "behavior_scalarized_upper_hf": self._selector_rows(
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
            "behavior_guarded_upper_hf"
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

    def test_lower_action_router_is_causal_observable_and_high_passes(self):
        prefix = [
            np.asarray([0.6, -0.4]),
            np.asarray([0.6, -0.4]),
            np.asarray([0.6, -0.4]),
        ]

        def trace(future):
            router = CausalLowerActionRouter(
                mode="causal_ema_high_pass", alpha=0.5
            )
            router.reset(2)
            rows = []
            for latent in [*prefix, *future]:
                before = router.context
                routed = router.route(latent, action_limit=1.0)
                rows.append((before, routed["effective"], router.context))
            return rows

        left = trace([np.asarray([1.0, 1.0])])
        right = trace([np.asarray([-1.0, -1.0])])
        for left_row, right_row in zip(left[:len(prefix)], right[:len(prefix)]):
            for left_value, right_value in zip(left_row, right_row):
                np.testing.assert_allclose(left_value, right_value)
        effective_norms = [
            float(np.linalg.norm(row[1])) for row in left[:len(prefix)]
        ]
        self.assertGreater(effective_norms[0], effective_norms[-1])
        np.testing.assert_allclose(left[0][0], np.zeros(2))

    def test_direct_lower_action_router_preserves_legacy_action(self):
        router = CausalLowerActionRouter(mode="direct", alpha=0.25)
        router.reset(2)
        latent = np.asarray([0.25, -0.75])
        routed = router.route(latent, action_limit=1.0)
        np.testing.assert_allclose(routed["effective"], latent)
        np.testing.assert_allclose(router.context, latent)
        self.assertEqual(float(routed["clip_rate"]), 0.0)

    def test_partial_lower_action_router_has_registered_dc_gain(self):
        router = CausalLowerActionRouter(
            mode="causal_ema_high_pass",
            alpha=0.5,
            strength=0.1,
        )
        router.reset(1)
        effective = None
        for _ in range(64):
            effective = router.route(
                np.asarray([0.6]), action_limit=1.0
            )["effective"]
        self.assertIsNotNone(effective)
        np.testing.assert_allclose(effective, [0.54], atol=1e-8)

    def test_conservative_router_transfers_removed_action_exactly(self):
        router = CausalLowerActionRouter(
            mode="causal_ema_conservative_transfer",
            alpha=1.0,
            strength=1.0,
        )
        router.reset(2)
        router.route(np.asarray([1.0, -1.0]), action_limit=1.0)
        routed = router.route(
            np.asarray([-1.0, 1.0]), action_limit=0.5
        )
        np.testing.assert_allclose(
            np.asarray(routed["upper_transfer"])
            + np.asarray(routed["effective"]),
            routed["latent"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            routed["transfer_reconstruction_error"],
            np.zeros(2),
            atol=1e-12,
        )
        self.assertEqual(float(routed["clip_rate"]), 1.0)

    def test_joint_band_projection_is_causal_and_reconstructs_total(self):
        prefix = [
            (np.asarray([0.4, -0.2]), np.asarray([0.1, 0.3])),
            (np.asarray([0.4, -0.2]), np.asarray([0.1, 0.3])),
        ]

        def trace(future_lower):
            router = CausalLowerActionRouter(
                mode="causal_joint_band_projection",
                alpha=0.5,
                strength=1.0,
            )
            router.reset(2)
            rows = []
            for upper, lower in [
                *prefix,
                (np.asarray([0.4, -0.2]), future_lower),
            ]:
                routed = router.route(
                    lower,
                    upper_action=upper,
                    action_limit=2.0,
                )
                projected_upper = (
                    upper + np.asarray(routed["upper_transfer"])
                )
                projected_lower = np.asarray(routed["effective"])
                np.testing.assert_allclose(
                    projected_upper + projected_lower,
                    upper + lower,
                    atol=1e-7,
                )
                rows.append((
                    np.asarray(routed["baseline_before"]).copy(),
                    projected_upper.copy(),
                    projected_lower.copy(),
                ))
            return rows

        left = trace(np.asarray([1.0, 1.0]))
        right = trace(np.asarray([-1.0, -1.0]))
        for left_row, right_row in zip(left[:2], right[:2]):
            for left_value, right_value in zip(left_row, right_row):
                np.testing.assert_allclose(left_value, right_value)
        np.testing.assert_allclose(
            left[0][1], np.asarray([0.5, 0.1]), atol=1e-7
        )
        np.testing.assert_allclose(
            left[1][1], np.asarray([0.5, 0.1]), atol=1e-7
        )

    def test_joint_band_projection_requires_upper_action(self):
        router = CausalLowerActionRouter(
            mode="causal_joint_band_projection",
            alpha=0.5,
            strength=0.5,
        )
        router.reset(1)
        with self.assertRaisesRegex(ValueError, "requires the current upper"):
            router.route(np.asarray([0.2]))

    def test_physical_constraint_cost_avoids_small_budget_blowup(self):
        budget = {
            "budget_excess_squared": 100.0,
            "power_excess": 0.01,
        }
        self.assertEqual(
            _leakage_constraint_cost(
                budget, mode="ratio_excess_squared"
            ),
            100.0,
        )
        self.assertEqual(
            _leakage_constraint_cost(budget, mode="power_excess"),
            0.01,
        )

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

    def test_crossed_checkpoint_paths_cover_every_root_condition_pair(self):
        paths, assignments = crossed_checkpoint_selection_paths(
            [101, 103],
            ["standard", "mixed", "ood_chirp"],
            env_id="HalfCheetah-v5",
        )
        repeated, repeated_assignments = crossed_checkpoint_selection_paths(
            [101, 103],
            ["standard", "mixed", "ood_chirp"],
            env_id="HalfCheetah-v5",
        )
        self.assertEqual(paths, repeated)
        self.assertEqual(assignments, repeated_assignments)
        self.assertEqual(len(paths), 6)
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(
            {mode: list(assignments.values()).count(mode) for mode in set(assignments.values())},
            {"standard": 2, "mixed": 2, "ood_chirp": 2},
        )

    def test_closed_loop_guard_paths_have_an_independent_namespace(self):
        selection_paths, _ = crossed_checkpoint_selection_paths(
            [101, 103],
            ["standard", "mixed"],
            env_id="HalfCheetah-v5",
        )
        guard_paths, assignments = crossed_deployment_frequency_guard_paths(
            [101, 103],
            ["standard", "mixed"],
            env_id="HalfCheetah-v5",
        )
        repeated, repeated_assignments = (
            crossed_deployment_frequency_guard_paths(
                [101, 103],
                ["standard", "mixed"],
                env_id="HalfCheetah-v5",
            )
        )
        self.assertEqual(guard_paths, repeated)
        self.assertEqual(assignments, repeated_assignments)
        self.assertEqual(len(guard_paths), 4)
        self.assertFalse(set(guard_paths) & set(selection_paths))
        self.assertEqual(set(assignments.values()), {"standard", "mixed"})

    def test_frequency_reference_paths_have_an_independent_namespace(self):
        selection_paths, _ = crossed_checkpoint_selection_paths(
            [101, 103],
            ["standard", "mixed"],
            env_id="HalfCheetah-v5",
        )
        guard_paths, _ = crossed_deployment_frequency_guard_paths(
            [101, 103],
            ["standard", "mixed"],
            env_id="HalfCheetah-v5",
        )
        reference_paths, assignments = (
            crossed_deployment_frequency_reference_paths(
                [101, 103],
                ["standard", "mixed"],
                env_id="HalfCheetah-v5",
            )
        )
        repeated, repeated_assignments = (
            crossed_deployment_frequency_reference_paths(
                [101, 103],
                ["standard", "mixed"],
                env_id="HalfCheetah-v5",
            )
        )
        self.assertEqual(reference_paths, repeated)
        self.assertEqual(assignments, repeated_assignments)
        self.assertEqual(len(reference_paths), 4)
        self.assertFalse(set(reference_paths) & set(selection_paths))
        self.assertFalse(set(reference_paths) & set(guard_paths))
        self.assertEqual(set(assignments.values()), {"standard", "mixed"})

    def test_behavior_robust_checkpoint_score_penalizes_worst_endpoint(self):
        rows = [
            {
                "disturbance_mode": mode,
                "reward_mean": reward,
                "LowerLFDriftAbs": lower ** 2,
                "RawLowerLFDriftAbs": raw_lower ** 2,
                "UpperHFPowerAbs": upper ** 2,
                "LatentLowerLFDriftAbs": (lower * 1.5) ** 2,
                "LatentUpperHFPowerAbs": (upper * 1.5) ** 2,
            }
            for mode, reward, lower, raw_lower, upper in (
                ("standard", 2.0, 0.04, 0.04, 0.08),
                ("mixed", 1.0, 0.10, 0.12, 0.20),
            )
        ]
        diagnostics = behavior_robust_checkpoint_diagnostics(
            rows,
            expected_modes=["standard", "mixed"],
            lower_lf_rms_budget=0.05,
            upper_hf_rms_budget=0.10,
            constraint_penalty=10.0,
        )
        self.assertEqual(diagnostics["worst_condition_reward_mean"], 1.0)
        self.assertGreater(
            diagnostics["normalized_constraint_penalty"], 0.0
        )
        self.assertLess(diagnostics["score"], 1.0)
        latent_diagnostics = behavior_robust_checkpoint_diagnostics(
            rows,
            expected_modes=["standard", "mixed"],
            lower_lf_rms_budget=0.05,
            upper_hf_rms_budget=0.10,
            constraint_penalty=10.0,
            include_latent=True,
        )
        self.assertGreater(
            latent_diagnostics["normalized_constraint_penalty"],
            diagnostics["normalized_constraint_penalty"],
        )
        self.assertIn(
            "latent_lower_violation",
            latent_diagnostics["worst_normalized_violations"],
        )
        self.assertIn(
            "latent_upper_hf_violation",
            latent_diagnostics["worst_normalized_violations"],
        )

    def test_latent_feasibility_rank_prioritizes_the_worst_endpoint(self):
        def rows(latent_lower: float, latent_upper: float, reward: float):
            return [
                {
                    "disturbance_mode": mode,
                    "reward_mean": reward,
                    "LowerLFDriftAbs": 0.04 ** 2,
                    "RawLowerLFDriftAbs": 0.04 ** 2,
                    "LatentLowerLFDriftAbs": latent_lower ** 2,
                    "UpperHFPowerAbs": 0.08 ** 2,
                    "LatentUpperHFPowerAbs": latent_upper ** 2,
                }
                for mode in ("standard", "mixed")
            ]

        imbalanced = latent_behavior_feasibility_rank(
            rows(0.05, 0.20, 10.0),
            expected_modes=("standard", "mixed"),
            lower_lf_rms_budget=0.05,
            upper_hf_rms_budget=0.10,
        )
        balanced = latent_behavior_feasibility_rank(
            rows(0.07, 0.15, 1.0),
            expected_modes=("standard", "mixed"),
            lower_lf_rms_budget=0.05,
            upper_hf_rms_budget=0.10,
        )

        self.assertGreater(balanced, imbalanced)
        self.assertLess(balanced[2], imbalanced[2])

    def test_paired_relative_rank_enforces_reward_and_all_endpoints(self):
        def rows(reward: float, frequency: float):
            return [
                {
                    "disturbance_mode": mode,
                    "seed": seed,
                    "reward_mean": reward,
                    "LowerLFDriftAbs": frequency,
                    "RawLowerLFDriftAbs": frequency,
                    "LatentLowerLFDriftAbs": frequency,
                    "UpperHFPowerAbs": frequency,
                    "LatentUpperHFPowerAbs": frequency,
                }
                for mode in ("standard", "mixed")
                for seed in (11, 13)
            ]

        baseline = rows(100.0, 1.0)
        feasible = paired_relative_frequency_feasibility_rank(
            rows(99.0, 0.90),
            baseline_rows=baseline,
            expected_modes=("standard", "mixed"),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        high_reward_leaky = paired_relative_frequency_feasibility_rank(
            rows(110.0, 1.20),
            baseline_rows=baseline,
            expected_modes=("standard", "mixed"),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        self.assertEqual(feasible[0], 0.0)
        self.assertGreater(feasible, high_reward_leaky)
        diagnostics = paired_relative_frequency_feasibility_diagnostics(
            rows(110.0, 1.20),
            baseline_rows=baseline,
            expected_modes=("standard", "mixed"),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        self.assertEqual(diagnostics["constraint_count"], 12)
        self.assertEqual(tuple(diagnostics["rank"]), high_reward_leaky)
        self.assertIn(
            diagnostics["worst_constraint"]["endpoint"],
            {
                "LowerLFDriftAbs",
                "RawLowerLFDriftAbs",
                "LatentLowerLFDriftAbs",
                "UpperHFPowerAbs",
                "LatentUpperHFPowerAbs",
            },
        )
        with self.assertRaisesRegex(ValueError, "identical unique paths"):
            paired_relative_frequency_feasibility_rank(
                rows(99.0, 0.90)[:-1],
                baseline_rows=baseline,
                expected_modes=("standard", "mixed"),
                lower_reduction_fraction=0.05,
                upper_reduction_fraction=0.05,
                lower_power_floor=1e-6,
                upper_power_floor=1e-6,
            )

        snapshot = paired_closed_loop_guard_snapshot(
            rows(99.0, 0.90),
            baseline_rows=baseline,
            expected_modes=("standard", "mixed"),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        self.assertEqual(snapshot["path_count"], 4)
        self.assertEqual(snapshot["constraint_count"], 12)
        self.assertEqual(snapshot["reward_violation_count"], 0)
        self.assertEqual(snapshot["frequency_violation_count"], 0)
        self.assertEqual(snapshot["frequency_violation_merit"], 0.0)
        self.assertEqual(snapshot["worst_frequency_violation"], 0.0)
        self.assertTrue(snapshot["contract"].endswith(
            "frequency_endpoints_with_restoration_merit_v2"
        ))

    def test_pathwise_relative_rank_catches_failure_hidden_by_mode_mean(self):
        baseline = [
            {
                "disturbance_mode": "standard",
                "seed": seed,
                "reward_mean": 100.0,
                "LowerLFDriftAbs": 1.0,
                "RawLowerLFDriftAbs": 1.0,
                "LatentLowerLFDriftAbs": 1.0,
                "UpperHFPowerAbs": 1.0,
                "LatentUpperHFPowerAbs": 1.0,
            }
            for seed in (11, 13)
        ]
        candidate = []
        for seed, frequency in ((11, 0.1), (13, 1.7)):
            candidate.append({
                "disturbance_mode": "standard",
                "seed": seed,
                "reward_mean": 100.0,
                "LowerLFDriftAbs": frequency,
                "RawLowerLFDriftAbs": frequency,
                "LatentLowerLFDriftAbs": frequency,
                "UpperHFPowerAbs": frequency,
                "LatentUpperHFPowerAbs": frequency,
            })
        common = dict(
            baseline_rows=baseline,
            expected_modes=("standard",),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        mode_mean = paired_relative_frequency_feasibility_diagnostics(
            candidate, **common
        )
        pathwise = paired_relative_frequency_feasibility_diagnostics(
            candidate, pathwise_robust=True, **common
        )
        self.assertEqual(mode_mean["rank"][0], 0.0)
        self.assertLess(pathwise["rank"][0], 0.0)
        self.assertEqual(mode_mean["constraint_count"], 6)
        self.assertEqual(pathwise["constraint_count"], 12)
        self.assertEqual(pathwise["comparison_group_count"], 2)
        self.assertEqual(pathwise["aggregation"], "pathwise")
        self.assertEqual(pathwise["worst_constraint"]["seed"], 13)

    def test_mode_cvar_bounds_tail_risk_without_all_path_feasibility(self):
        baseline = [
            {
                "disturbance_mode": "standard",
                "seed": seed,
                "reward_mean": 100.0,
                "LowerLFDriftAbs": 1.0,
                "RawLowerLFDriftAbs": 1.0,
                "LatentLowerLFDriftAbs": 1.0,
                "UpperHFPowerAbs": 1.0,
                "LatentUpperHFPowerAbs": 1.0,
            }
            for seed in (11, 13, 17, 19)
        ]
        candidate = []
        for seed, frequency in zip(
            (11, 13, 17, 19), (0.5, 0.5, 0.5, 1.1)
        ):
            row = dict(baseline[0])
            row.update({
                "seed": seed,
                "LowerLFDriftAbs": frequency,
                "RawLowerLFDriftAbs": frequency,
                "LatentLowerLFDriftAbs": frequency,
                "UpperHFPowerAbs": frequency,
                "LatentUpperHFPowerAbs": frequency,
            })
            candidate.append(row)
        common = dict(
            baseline_rows=baseline,
            expected_modes=("standard",),
            lower_reduction_fraction=0.05,
            upper_reduction_fraction=0.05,
            lower_power_floor=1e-6,
            upper_power_floor=1e-6,
        )
        pathwise = paired_relative_frequency_feasibility_diagnostics(
            candidate, pathwise_robust=True, **common
        )
        cvar_half = paired_relative_frequency_feasibility_diagnostics(
            candidate, risk_mode="mode_cvar", cvar_alpha=0.5, **common
        )
        cvar_top_quartile = paired_relative_frequency_feasibility_diagnostics(
            candidate, risk_mode="mode_cvar", cvar_alpha=0.75, **common
        )

        self.assertLess(pathwise["rank"][0], 0.0)
        self.assertEqual(cvar_half["rank"][0], 0.0)
        self.assertLess(cvar_top_quartile["rank"][0], 0.0)
        self.assertEqual(cvar_half["constraint_count"], 6)
        self.assertEqual(cvar_half["comparison_group_count"], 1)
        self.assertEqual(cvar_half["aggregation"], "disturbance_mode_cvar")
        self.assertEqual(cvar_half["risk_mode"], "mode_cvar")
        self.assertEqual(cvar_half["cvar_alpha"], 0.5)

        snapshot = paired_closed_loop_guard_snapshot(
            candidate, risk_mode="mode_cvar", cvar_alpha=0.5, **common
        )
        self.assertEqual(snapshot["risk_mode"], "mode_cvar")
        self.assertIn("mode_cvar", snapshot["contract"])

    def test_written_checkpoint_has_independent_file_hash(self):
        model = torch.nn.Linear(2, 1)
        payload = {
            "history": [{"iteration": 0}],
            "method": "unit",
            "environment": "unit",
            "disturbance_mode": "standard",
            "optimizer_seed": 7,
            "code_revision": "b" * 40,
            "source_manifest_sha256": "c" * 64,
            "frozen_parameter_sha256": "a" * 64,
            "frozen_checkpoint_sha256": "a" * 64,
            "lower_action_router_mode": "direct",
            "lower_action_router_observe_strength": False,
            "responsibility_mode": "additive",
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

    def test_paired_checkpoint_loader_verifies_provenance_and_parameters(self):
        torch.manual_seed(401)
        baseline = _hierarchical_model(
            state_dim=5,
            action_dim=2,
            hidden_dim=8,
            learning_rate=3e-4,
            leakage_constraint=False,
        )
        parameter_sha256 = _model_parameter_sha256(baseline)
        code_revision = "d" * 40
        source_manifest = "e" * 64
        payload = {
            "history": [{"iteration": 0}],
            "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
            "method": "freq_hrl_no_leakage",
            "environment": "HalfCheetah-v5",
            "disturbance_mode": "standard",
            "optimizer_seed": 409,
            "code_revision": code_revision,
            "source_manifest_sha256": source_manifest,
            "lower_action_router_mode": "direct",
            "lower_action_router_strength": 0.0,
            "lower_action_router_observe_strength": True,
            "responsibility_mode": "causal_lf_transfer",
            "selected_checkpoint_iteration": 3,
            "frozen_parameter_sha256": parameter_sha256,
            "frozen_checkpoint_sha256": parameter_sha256,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_cell(output, payload, [{"metric": 1.0}], baseline)
            torch.manual_seed(419)
            candidate = _hierarchical_model(
                state_dim=5,
                action_dim=2,
                hidden_dim=8,
                learning_rate=3e-4,
                leakage_constraint=False,
                lower_actor_anchor_coef=0.5,
                actor_anchor_zero_state_indices=(4,),
            )
            self.assertNotEqual(
                _model_parameter_sha256(candidate), parameter_sha256
            )
            metadata = load_paired_mujoco_checkpoint(
                candidate,
                checkpoint_path=output / "checkpoint.pt",
                summary_path=output / "cell_summary.json",
                env_id="HalfCheetah-v5",
                optimizer_seed=409,
                expected_code_revision=code_revision,
                expected_source_manifest_sha256=source_manifest,
                reset_upper_deployment_frequency_lambda=1.25,
                reset_lower_deployment_frequency_lambda=2.5,
            )
            self.assertEqual(
                _model_parameter_sha256(candidate), parameter_sha256
            )
            self.assertEqual(
                metadata["checkpoint_parameter_sha256"], parameter_sha256
            )
            self.assertEqual(
                metadata["loaded_upper_deployment_frequency_lambda"], 0.0
            )
            self.assertEqual(
                metadata["reset_upper_deployment_frequency_lambda"], 1.25
            )
            self.assertEqual(candidate.upper_deployment_frequency_lambda, 1.25)
            self.assertEqual(candidate.lower_deployment_frequency_lambda, 2.5)
            with self.assertRaisesRegex(ValueError, "summary contract"):
                load_paired_mujoco_checkpoint(
                    candidate,
                    checkpoint_path=output / "checkpoint.pt",
                    summary_path=output / "cell_summary.json",
                    env_id="Hopper-v5",
                    optimizer_seed=409,
                    expected_code_revision=code_revision,
                    expected_source_manifest_sha256=source_manifest,
                )


@unittest.skipUnless(mujoco_available(), "MuJoCo runtime is unavailable")
class MujocoControlIntegrationTest(unittest.TestCase):
    def test_explicit_v1416_protocol_keeps_control_arm_comparable(self):
        payload, _, _ = train_mujoco_method(
            method="freq_hrl_no_leakage",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[811],
            selection_seeds=[821],
            eval_seeds=[823],
            steps=8,
            episode_horizon=8,
            iterations=1,
            optimizer_seed=827,
            upper_period=4,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            training_disturbance_modes=["standard"],
            evaluation_disturbance_modes=["standard"],
            control_protocol_version=(
                MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16
            ),
        )
        self.assertEqual(
            payload["protocol_version"],
            MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
        )
        self.assertEqual(
            payload["protocol_version_selection"],
            MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
        )
        self.assertFalse(payload["deployment_frequency_constraint_enabled"])

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

    def test_conservative_router_is_closed_loop_action_invariant(self):
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
            seed=137,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=64,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=False,
            responsibility_mode="additive",
            lower_action_router_mode=(
                "causal_ema_conservative_transfer"
            ),
            lower_action_router_alpha=0.04,
            lower_action_router_observe_strength=False,
            sample=False,
            episode_horizon=64,
        )
        _, control = rollout_hierarchical(
            model, lower_action_router_strength=0.0, **common
        )
        _, transferred = rollout_hierarchical(
            model, lower_action_router_strength=0.15, **common
        )

        for metric in (
            "episode_return",
            "reward_mean",
            "action_energy",
            "action_smoothness",
            "UpperPolicyActionRMS",
            "LatentLowerActionRMS",
            "LatentLowerLFDriftAbs",
        ):
            self.assertEqual(control[metric], transferred[metric], metric)
        for trace in (
            "RewardTraceSHA256",
            "ExecutedActionTraceSHA256",
            "LatentPolicyTraceSHA256",
        ):
            self.assertEqual(control[trace], transferred[trace], trace)
        self.assertGreater(transferred["LowerRouterRemovedRMS"], 0.0)
        self.assertEqual(
            transferred["LowerRouterUpperTransferRMS"],
            transferred["LowerRouterRemovedRMS"],
        )
        self.assertEqual(
            transferred["LowerRouterFunctionPreserving"], 1.0
        )
        self.assertLessEqual(
            transferred["LowerRouterActionReconstructionRMS"], 1e-7
        )
        self.assertLessEqual(
            transferred["ResponsibilityReconstructionRMS"], 1e-7
        )
        self.assertLess(
            transferred["RawLowerLFDriftAbs"],
            control["RawLowerLFDriftAbs"],
        )

    def test_joint_projection_preserves_path_and_reduces_both_leakages(self):
        observation_dim, action_dim = environment_dimensions(
            "HalfCheetah-v5", episode_horizon=128
        )
        torch.manual_seed(701)
        np.random.seed(701)
        model = _hierarchical_model(
            state_dim=mujoco_policy_state_dim(observation_dim, action_dim),
            action_dim=action_dim,
            hidden_dim=16,
            learning_rate=3e-4,
            leakage_constraint=False,
        )
        common = dict(
            seed=709,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=128,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=False,
            responsibility_mode="additive",
            lower_action_router_mode="causal_joint_band_projection",
            lower_action_router_alpha=0.04,
            lower_action_router_observe_strength=False,
            upper_constraint_mode="disabled",
            sample=False,
            episode_horizon=128,
        )
        _, control = rollout_hierarchical(
            model, lower_action_router_strength=0.0, **common
        )
        _, projected = rollout_hierarchical(
            model, lower_action_router_strength=0.5, **common
        )

        for trace in (
            "RewardTraceSHA256",
            "ExecutedActionTraceSHA256",
            "LatentPolicyTraceSHA256",
        ):
            self.assertEqual(control[trace], projected[trace], trace)
        self.assertLess(
            projected["UpperHFPowerAbs"], control["UpperHFPowerAbs"]
        )
        self.assertLess(
            projected["LowerLFDriftAbs"], control["LowerLFDriftAbs"]
        )
        self.assertEqual(
            projected["LatentUpperHFPowerAbs"],
            control["LatentUpperHFPowerAbs"],
        )
        self.assertEqual(
            projected["LatentLowerLFDriftAbs"],
            control["LatentLowerLFDriftAbs"],
        )
        self.assertEqual(projected["LowerRouterFunctionPreserving"], 1.0)
        self.assertLessEqual(
            projected["ResponsibilityReconstructionRMS"], 1e-7
        )

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
            upper_hf_rms_budget=1e-4,
            upper_hf_penalty_coef=0.0,
            **common,
        )
        torch.manual_seed(223)
        np.random.seed(223)
        behavior_batch, behavior_row = rollout_hierarchical(
            model,
            leakage_constraint_scope="joint_behavior",
            upper_hf_rms_budget=1e-4,
            upper_hf_penalty_coef=1.0,
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
            behavior_row["UpperHFPenaltyTotal"], 0.0
        )
        self.assertLess(
            float(np.sum(behavior_batch.upper.reward)),
            float(np.sum(responsibility_batch.upper.reward)),
        )

    def test_latent_joint_scope_cannot_hide_actor_leakage_with_projection(self):
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
            seed=227,
            env_id="HalfCheetah-v5",
            disturbance_mode="mixed",
            steps=64,
            upper_period=8,
            frequency_routing=True,
            leakage_constraint=True,
            lower_lf_rms_budget=1e-3,
            upper_hf_rms_budget=1e-4,
            upper_hf_penalty_coef=1.0,
            responsibility_mode="additive",
            lower_action_router_mode="causal_joint_band_projection",
            lower_action_router_strength=0.5,
            sample=True,
            episode_horizon=64,
        )
        torch.manual_seed(229)
        np.random.seed(229)
        behavior_batch, behavior_row = rollout_hierarchical(
            model, leakage_constraint_scope="joint_behavior", **common
        )
        torch.manual_seed(229)
        np.random.seed(229)
        latent_batch, latent_row = rollout_hierarchical(
            model, leakage_constraint_scope="joint_behavior_latent", **common
        )
        self.assertTrue(np.all(
            latent_batch.lower.cost + 1e-12 >= behavior_batch.lower.cost
        ))
        self.assertGreaterEqual(
            latent_row["UpperHFPenaltyTotal"],
            behavior_row["UpperHFPenaltyTotal"],
        )
        self.assertEqual(
            latent_row["LatentUpperHFPowerAbs"],
            behavior_row["LatentUpperHFPowerAbs"],
        )
        self.assertEqual(
            latent_row["LatentLowerLFDriftAbs"],
            behavior_row["LatentLowerLFDriftAbs"],
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
            MUJOCO_CONTROL_PROTOCOL_VERSION,
        )
        self.assertTrue(payload["frequency_routing_enabled"])
        self.assertEqual(payload["training_disturbance_modes"], ["standard"])
        self.assertEqual(payload["upper_action_scale"], 1.0)
        self.assertEqual(payload["lower_action_scale"], 1.0)
        self.assertEqual(payload["responsibility_mode"], "causal_lf_transfer")
        self.assertEqual(
            payload["policy_filter_state_contract"],
            "canonical_raw_lf_and_observed_lower_router_state_v2",
        )
        self.assertEqual(
            payload["lower_cost_state_contract"],
            "causal_responsibility_anchor_32_step_raw_and_responsibility_"
            "rolling_lf_cost_critic_only_v3",
        )
        self.assertEqual(payload["role_capacity_status"], "symmetric")
        self.assertEqual(payload["upper_to_lower_action_capacity_ratio"], 1.0)
        self.assertEqual(
            payload["lower_constraint_update_mode"],
            "reward_guarded_adam_projection",
        )
        self.assertFalse(payload["deployment_frequency_constraint_enabled"])
        self.assertEqual(
            payload["deployment_frequency_action_source"], "disabled"
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
        self.assertEqual(payload["heldout_evaluation_pass_count"], 1)
        for metric in (
            "UpperActionRMS",
            "LowerActionRMS",
            "LatentLowerActionRMS",
            "LowerRouterRemovedRMS",
            "LowerRouterClipRate",
            "UpperActionEnergyShare",
            "AdditiveActionClipRate",
            "LatentUpperHFPowerAbs",
            "RawLowerLFDriftAbs",
            "LatentLowerLFDriftAbs",
            "RawLowerLFRmsOnlineMean",
            "LowerLFPowerOnlineMean",
            "RawLowerLFPowerOnlineMean",
            "LatentLowerLFPowerOnlineMean",
            "LowerConstraintCostMean",
            "UpperTransitionDeltaRMSMean",
            "UpperHFPowerOnlineMean",
            "UpperHFPenaltyTotal",
            "ResponsibilityTransferRMS",
            "ResponsibilityReconstructionRMS",
        ):
            self.assertIn(metric, rows[0])
            self.assertTrue(np.isfinite(rows[0][metric]))
        self.assertAlmostEqual(
            rows[0]["LowerLFPowerOnlineMean"],
            rows[0]["LowerLFDriftAbs"],
            places=10,
        )
        self.assertAlmostEqual(
            rows[0]["RawLowerLFPowerOnlineMean"],
            rows[0]["RawLowerLFDriftAbs"],
            places=10,
        )
        self.assertAlmostEqual(
            rows[0]["LatentLowerLFPowerOnlineMean"],
            rows[0]["LatentLowerLFDriftAbs"],
            places=10,
        )
        self.assertAlmostEqual(
            rows[0]["UpperHFPowerOnlineMean"],
            rows[0]["UpperHFPowerAbs"],
            places=10,
        )
        self.assertLessEqual(rows[0]["ResponsibilityReconstructionRMS"], 1e-7)
        self.assertEqual(
            payload["bootstrap_contract"],
            "explicit_reward_and_cost_next_value_with_separate_trace_boundary_"
            "and_mdp_terminal",
        )
        self.assertIsNotNone(payload["history"][-1]["lower_cost_actor_active"])
        self.assertEqual(len(payload["frozen_checkpoint_sha256"]), 64)

    def test_deployment_frequency_constraint_is_audited_separately(self):
        payload, _, model = train_mujoco_method(
            method="freq_hrl",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[141],
            selection_seeds=[143],
            eval_seeds=[147],
            steps=24,
            episode_horizon=32,
            iterations=1,
            optimizer_seed=153,
            upper_period=6,
            hidden_dim=8,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            upper_constraint_mode="primal_dual",
            upper_dual_lr=0.0,
            lower_dual_lr=0.0,
            upper_deployment_frequency_dual_lr=10.0,
            upper_deployment_frequency_lambda_init=1.0,
            upper_deployment_frequency_step_scale=10.0,
            upper_deployment_frequency_rms_budget=0.001,
        )
        self.assertTrue(payload["deployment_frequency_constraint_enabled"])
        self.assertTrue(
            payload["upper_deployment_frequency_constraint_enabled"]
        )
        self.assertFalse(
            payload["lower_deployment_frequency_constraint_enabled"]
        )
        self.assertEqual(
            payload["deployment_frequency_action_source"],
            "deterministic_squashed_actor_mean",
        )
        self.assertEqual(
            payload["upper_deployment_frequency_rms_budget"], 0.001
        )
        self.assertGreaterEqual(
            model.upper_deployment_frequency_lambda, 0.0
        )

    def test_crossed_behavior_selection_and_upper_primal_dual_are_wired(self):
        payload, rows, _ = train_mujoco_method(
            method="freq_hrl",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[151, 157],
            selection_seeds=[163],
            eval_seeds=[167],
            steps=16,
            episode_horizon=16,
            iterations=1,
            optimizer_seed=173,
            upper_period=4,
            hidden_dim=8,
            lower_lf_rms_budget=1e-3,
            upper_hf_rms_budget=1e-4,
            upper_constraint_mode="primal_dual",
            upper_dual_lr=0.1,
            lower_dual_lr=0.2,
            leakage_cost_mode="power_excess",
            lower_action_router_mode="causal_ema_high_pass",
            lower_action_router_strength=0.1,
            checkpoint_selection_mode="crossed_conditions",
            checkpoint_score_mode="latent_behavior_feasibility_first",
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            training_disturbance_modes=["standard", "mixed"],
            evaluation_disturbance_modes=["standard"],
        )
        self.assertEqual(payload["checkpoint_selection_path_count"], 2)
        self.assertEqual(
            set(payload["selection_seed_condition_assignment"].values()),
            {"standard", "mixed"},
        )
        self.assertEqual(payload["upper_constraint_mode"], "primal_dual")
        self.assertEqual(payload["leakage_constraint_cost_mode"], "power_excess")
        self.assertEqual(payload["lower_dual_lr"], 0.2)
        self.assertEqual(
            payload["lower_action_router_mode"], "causal_ema_high_pass"
        )
        self.assertEqual(payload["lower_action_router_strength"], 0.1)
        self.assertEqual(rows[0]["LowerActionRouterStrength"], 0.1)
        self.assertEqual(
            payload["checkpoint_score_mode"],
            "latent_behavior_feasibility_first",
        )
        self.assertEqual(
            payload["checkpoint_selection_protocol"],
            "state_aligned_lexicographic_validation_v1",
        )
        self.assertIn(
            "negative_worst_endpoint_violation",
            payload["checkpoint_selected_rank"],
        )
        self.assertGreater(
            payload["history"][-1]["upper_constraint_mean"], 0.0
        )
        self.assertGreater(
            payload["history"][-1]["upper_constraint_lambda"], 0.0
        )
        self.assertGreater(rows[0]["UpperConstraintCostMean"], 0.0)
        self.assertLessEqual(rows[0]["UpperConstraintCostMax"], 1.0)
        self.assertLessEqual(rows[0]["LowerConstraintCostMax"], 1.0)
        self.assertEqual(rows[0]["UpperHFPenaltyTotal"], 0.0)

    def test_router_curriculum_is_training_only_and_observed(self):
        payload, rows, _ = train_mujoco_method(
            method="freq_hrl_no_leakage",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[211],
            selection_seeds=[223],
            eval_seeds=[227],
            steps=8,
            episode_horizon=8,
            iterations=4,
            optimizer_seed=229,
            upper_period=4,
            hidden_dim=8,
            lower_action_router_mode="causal_ema_high_pass",
            lower_action_router_alpha=0.04,
            lower_action_router_strength=0.1,
            lower_action_router_training_schedule="delayed_linear",
            lower_action_router_warmup_fraction=0.25,
            lower_action_router_ramp_fraction=0.5,
            lower_action_router_observe_strength=True,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=4,
        )
        expected = [0.0, 0.05, 0.1, 0.1]
        self.assertTrue(np.allclose(
            payload["lower_action_router_training_strengths_by_iteration"],
            expected,
        ))
        self.assertTrue(np.allclose(
            [
                row["sampled_lower_action_router_strength"]
                for row in payload["history"][1:]
            ],
            expected,
        ))
        self.assertEqual(
            payload["history"][0]["LowerActionRouterStrength_mean"],
            0.1,
        )
        self.assertEqual(
            payload["history"][-1]["LowerActionRouterStrength_mean"],
            0.1,
        )
        self.assertEqual(rows[0]["LowerActionRouterStrength"], 0.1)
        self.assertTrue(payload["lower_action_router_observe_strength"])

    def test_paired_router_continuation_starts_from_exact_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_dir = Path(directory) / "baseline"
            baseline_payload, baseline_rows, baseline_model = (
                train_mujoco_method(
                    method="freq_hrl_no_leakage",
                    env_id="HalfCheetah-v5",
                    disturbance_mode="standard",
                    train_seeds=[271],
                    selection_seeds=[277],
                    eval_seeds=[281],
                    steps=8,
                    episode_horizon=8,
                    iterations=1,
                    optimizer_seed=283,
                    upper_period=4,
                    hidden_dim=8,
                    lower_action_router_mode="direct",
                    lower_action_router_strength=0.0,
                    lower_action_router_observe_strength=True,
                    responsibility_mode="causal_lf_transfer",
                    checkpoint_smoothing_window=1,
                    checkpoint_min_delta=0.0,
                    checkpoint_evaluation_interval=1,
                    evaluation_disturbance_modes=["standard"],
                )
            )
            write_cell(
                baseline_dir,
                baseline_payload,
                baseline_rows,
                baseline_model,
            )
            candidate_payload, _, _ = train_mujoco_method(
                method="freq_hrl_no_leakage",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                train_seeds=[293],
                selection_seeds=[307],
                eval_seeds=[311],
                steps=8,
                episode_horizon=8,
                iterations=1,
                optimizer_seed=283,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_mode="causal_ema_high_pass",
                lower_action_router_alpha=0.04,
                lower_action_router_strength=0.1,
                lower_action_router_observe_strength=True,
                responsibility_mode="causal_lf_transfer",
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                upper_actor_anchor_coef=0.05,
                lower_actor_anchor_coef=0.25,
                checkpoint_smoothing_window=1,
                checkpoint_min_delta=0.0,
                checkpoint_evaluation_interval=1,
                evaluation_disturbance_modes=["standard"],
            )
            continuation = candidate_payload[
                "paired_checkpoint_continuation"
            ]
            self.assertTrue(continuation["enabled"])
            self.assertEqual(
                continuation["checkpoint_parameter_sha256"],
                baseline_payload["frozen_parameter_sha256"],
            )
            self.assertEqual(
                candidate_payload["actor_anchor_zero_state_indices"],
                [candidate_payload["config"]["lower_state_dim"] - 1],
            )
            self.assertEqual(
                candidate_payload["actor_anchor_contract"],
                "frozen_matched_direct_policy_at_zero_router_context_"
                "analytic_gaussian_kl_v1",
            )
            self.assertGreaterEqual(
                candidate_payload["history"][-1]["lower_actor_anchor_kl"],
                0.0,
            )

            direct_payload, _, _ = train_mujoco_method(
                method="freq_hrl_no_leakage",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                train_seeds=[313],
                selection_seeds=[317],
                eval_seeds=[331],
                steps=8,
                episode_horizon=8,
                iterations=1,
                optimizer_seed=283,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_mode="direct",
                lower_action_router_strength=0.0,
                lower_action_router_observe_strength=True,
                responsibility_mode="causal_lf_transfer",
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                checkpoint_smoothing_window=1,
                checkpoint_min_delta=0.0,
                checkpoint_evaluation_interval=1,
                evaluation_disturbance_modes=["standard"],
            )
            self.assertTrue(
                direct_payload["paired_checkpoint_continuation"]["enabled"]
            )
            self.assertEqual(
                direct_payload["lower_action_router_mode"], "direct"
            )

    def test_closed_loop_guard_uses_disjoint_actual_rollout_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_dir = Path(directory) / "closed_loop_baseline"
            common = dict(
                method="freq_hrl",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                steps=8,
                episode_horizon=8,
                iterations=1,
                optimizer_seed=601,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_observe_strength=True,
                checkpoint_smoothing_window=1,
                checkpoint_min_delta=0.0,
                checkpoint_evaluation_interval=1,
                training_disturbance_modes=["standard"],
                evaluation_disturbance_modes=["standard"],
            )
            baseline_payload, baseline_rows, baseline_model = (
                train_mujoco_method(
                    train_seeds=[607],
                    selection_seeds=[613],
                    eval_seeds=[617],
                    **common,
                )
            )
            write_cell(
                baseline_dir,
                baseline_payload,
                baseline_rows,
                baseline_model,
            )
            candidate_payload, _, _ = train_mujoco_method(
                train_seeds=[619],
                selection_seeds=[631],
                deployment_frequency_closed_loop_guard_seeds=[641],
                eval_seeds=[643],
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                upper_deployment_frequency_lambda_init=1.0,
                lower_deployment_frequency_lambda_init=1.0,
                upper_deployment_frequency_rms_budget=1.0,
                lower_deployment_frequency_rms_budget=1.0,
                deployment_frequency_groupwise_robust=True,
                deployment_frequency_closed_loop_trust_region=True,
                deployment_frequency_closed_loop_trust_region_backtracks=2,
                deployment_frequency_closed_loop_restoration_filter=True,
                deployment_frequency_closed_loop_restoration_min_reduction=(
                    1e-4
                ),
                deployment_frequency_closed_loop_restoration_funnel_multiplier=(
                    3.0
                ),
                **common,
            )

            self.assertTrue(candidate_payload[
                "deployment_frequency_closed_loop_trust_region"
            ])
            self.assertEqual(candidate_payload[
                "deployment_frequency_closed_loop_guard_seed_roots"
            ], [641])
            self.assertEqual(candidate_payload[
                "deployment_frequency_closed_loop_guard_path_count"
            ], 1)
            self.assertEqual(candidate_payload[
                "deployment_frequency_closed_loop_guard_baseline"
            ]["heldout_rows_used"], 0)
            self.assertGreaterEqual(candidate_payload[
                "deployment_frequency_closed_loop_guard_evaluation_count"
            ], 3)
            self.assertTrue(candidate_payload[
                "deployment_frequency_closed_loop_restoration_filter_enabled"
            ])
            self.assertGreaterEqual(candidate_payload[
                "deployment_frequency_closed_loop_restoration_funnel_limit"
            ], 0.0)
            self.assertTrue(candidate_payload["history"][1][
                "deployment_frequency_closed_loop_guard_trial_trace"
            ])
            guard_paths = set(map(
                int,
                candidate_payload[
                    "deployment_frequency_closed_loop_guard_condition_assignment"
                ],
            ))
            self.assertFalse(guard_paths & {619, 631, 641, 643})

    def test_v1416_crossed_pathwise_restoration_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_dir = Path(directory) / "v1416_baseline"
            common = dict(
                method="freq_hrl",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                steps=8,
                episode_horizon=8,
                iterations=1,
                optimizer_seed=659,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_observe_strength=True,
                checkpoint_smoothing_window=1,
                checkpoint_min_delta=0.0,
                checkpoint_evaluation_interval=1,
                training_disturbance_modes=["standard"],
                evaluation_disturbance_modes=["standard"],
                control_protocol_version=(
                    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16
                ),
            )
            baseline_payload, baseline_rows, baseline_model = (
                train_mujoco_method(
                    train_seeds=[607],
                    selection_seeds=[613],
                    eval_seeds=[617],
                    **common,
                )
            )
            write_cell(
                baseline_dir,
                baseline_payload,
                baseline_rows,
                baseline_model,
            )
            training_replay_payload, _, _ = train_mujoco_method(
                train_seeds=[661],
                selection_seeds=[673],
                deployment_frequency_closed_loop_guard_seeds=[677],
                eval_seeds=[683],
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                upper_deployment_frequency_lambda_init=1.0,
                lower_deployment_frequency_lambda_init=1.0,
                upper_deployment_frequency_rms_budget=0.001,
                lower_deployment_frequency_rms_budget=0.001,
                upper_deployment_frequency_reference_reduction_fraction=0.05,
                lower_deployment_frequency_reference_reduction_fraction=0.05,
                deployment_frequency_groupwise_robust=True,
                deployment_frequency_anchor_state_replay=True,
                deployment_frequency_closed_loop_trust_region=True,
                deployment_frequency_closed_loop_trust_region_backtracks=2,
                deployment_frequency_closed_loop_restoration_filter=True,
                **common,
            )
            self.assertEqual(
                training_replay_payload[
                    "deployment_frequency_anchor_state_replay_seed_roots"
                ],
                [],
            )
            self.assertEqual(
                training_replay_payload[
                    "deployment_frequency_anchor_state_replay_seed_source"
                ],
                "iteration_zero_training_paths",
            )
            self.assertEqual(
                training_replay_payload[
                    "deployment_frequency_anchor_state_replay_path_count"
                ],
                1,
            )
            candidate_payload, _, _ = train_mujoco_method(
                train_seeds=[619],
                selection_seeds=[631],
                deployment_frequency_anchor_state_replay_seeds=[653],
                deployment_frequency_closed_loop_guard_seeds=[641],
                eval_seeds=[643],
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                upper_deployment_frequency_lambda_init=1.0,
                lower_deployment_frequency_lambda_init=1.0,
                upper_deployment_frequency_rms_budget=0.001,
                lower_deployment_frequency_rms_budget=0.001,
                upper_deployment_frequency_reference_reduction_fraction=0.05,
                lower_deployment_frequency_reference_reduction_fraction=0.05,
                deployment_frequency_groupwise_robust=True,
                deployment_frequency_anchor_state_replay=True,
                deployment_frequency_projection_objective="violation_l2",
                deployment_frequency_pathwise_robust=True,
                deployment_frequency_restoration_freeze_reward_actor=True,
                deployment_frequency_closed_loop_trust_region=True,
                deployment_frequency_closed_loop_trust_region_backtracks=2,
                deployment_frequency_closed_loop_restoration_filter=True,
                **common,
            )
            self.assertEqual(
                candidate_payload["protocol_version"],
                MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
            )
            self.assertEqual(
                candidate_payload[
                    "deployment_frequency_anchor_state_replay_seed_roots"
                ],
                [653],
            )
            self.assertEqual(
                candidate_payload[
                    "deployment_frequency_anchor_state_replay_seed_source"
                ],
                "explicit",
            )
            self.assertEqual(
                candidate_payload[
                    "deployment_frequency_anchor_state_replay_path_count"
                ],
                1,
            )
            self.assertTrue(
                candidate_payload["deployment_frequency_pathwise_robust"]
            )
            self.assertTrue(candidate_payload[
                "deployment_frequency_restoration_freeze_reward_actor"
            ])
            self.assertEqual(
                candidate_payload[
                    "deployment_frequency_closed_loop_guard_constraint_count"
                ],
                6,
            )
            self.assertEqual(
                candidate_payload["history"][1][
                    "deployment_frequency_reward_actor_frozen"
                ],
                1.0,
            )
            self.assertEqual(
                candidate_payload["history"][1][
                    "upper_actor_optimizer_steps"
                ],
                0.0,
            )
            self.assertEqual(
                candidate_payload["history"][1][
                    "lower_actor_optimizer_steps"
                ],
                0.0,
            )

    def test_conservative_router_continuation_uses_same_hidden_state_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_dir = Path(directory) / "conservative_baseline"
            common = dict(
                method="freq_hrl_no_leakage",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                train_seeds=[337],
                selection_seeds=[347],
                eval_seeds=[349],
                steps=8,
                episode_horizon=8,
                iterations=1,
                optimizer_seed=353,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_mode=(
                    "causal_ema_conservative_transfer"
                ),
                lower_action_router_alpha=0.04,
                lower_action_router_observe_strength=False,
                responsibility_mode="additive",
                checkpoint_smoothing_window=1,
                checkpoint_min_delta=0.0,
                checkpoint_evaluation_interval=1,
                evaluation_disturbance_modes=["standard"],
            )
            baseline_payload, baseline_rows, baseline_model = (
                train_mujoco_method(
                    lower_action_router_strength=0.0,
                    **common,
                )
            )
            write_cell(
                baseline_dir,
                baseline_payload,
                baseline_rows,
                baseline_model,
            )

            candidate_payload, candidate_rows, _ = train_mujoco_method(
                lower_action_router_strength=0.15,
                initial_checkpoint_path=baseline_dir / "checkpoint.pt",
                initial_checkpoint_summary_path=(
                    baseline_dir / "cell_summary.json"
                ),
                initial_checkpoint_router_mode=(
                    "causal_ema_conservative_transfer"
                ),
                checkpoint_minimum_iteration=0,
                **common,
            )

            continuation = candidate_payload[
                "paired_checkpoint_continuation"
            ]
            self.assertTrue(continuation["enabled"])
            self.assertEqual(
                continuation["checkpoint_router_mode"],
                "causal_ema_conservative_transfer",
            )
            self.assertFalse(
                continuation["checkpoint_router_observe_strength"]
            )
            self.assertEqual(
                continuation["checkpoint_responsibility_mode"], "additive"
            )
            self.assertEqual(
                candidate_payload["actor_anchor_zero_state_indices"], []
            )
            self.assertEqual(
                candidate_payload["selected_checkpoint_iteration"], 0
            )
            self.assertEqual(
                candidate_payload["actor_anchor_contract"],
                "frozen_matched_conservative_policy_same_state_analytic_"
                "gaussian_kl_v2",
            )
            self.assertTrue(all(
                row["LowerRouterFunctionPreserving"] == 1.0
                and row["LowerRouterActionReconstructionRMS"] <= 1e-7
                for row in candidate_rows
            ))

    def test_router_curriculum_rejects_hidden_strength(self):
        with self.assertRaisesRegex(
            ValueError,
            "must expose its strength",
        ):
            train_mujoco_method(
                method="freq_hrl_no_leakage",
                env_id="HalfCheetah-v5",
                disturbance_mode="standard",
                train_seeds=[233],
                selection_seeds=[239],
                eval_seeds=[241],
                steps=8,
                episode_horizon=8,
                iterations=2,
                optimizer_seed=251,
                upper_period=4,
                hidden_dim=8,
                lower_action_router_mode="causal_ema_high_pass",
                lower_action_router_strength=0.1,
                lower_action_router_training_schedule="delayed_linear",
                lower_action_router_warmup_fraction=0.0,
                lower_action_router_ramp_fraction=0.5,
                lower_action_router_observe_strength=False,
            )

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
