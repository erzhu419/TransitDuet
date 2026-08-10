import argparse
import copy
import csv
import tempfile
import unittest
from pathlib import Path

from scripts import mujoco_v14_13_anchor_replay_trust_screen_spec as v1413
from scripts import mujoco_v14_14_closed_loop_actor_guard_screen_spec as v1414
from scripts import mujoco_v14_15_closed_loop_restoration_filter_screen_spec as spec
from freq_hrl.experiments.mujoco.control_validation import (
    MUJOCO_CONTROL_PROTOCOL_VERSION,
)
from freq_hrl.experiments.reproducibility import (
    git_source_manifest_sha256,
)
from scripts.analyze_mujoco_v14_15_closed_loop_restoration_filter_preflight import (
    _projection_diagnostics,
    _anchor_replay_diagnostics,
    _read_rows,
    _trust_region_diagnostics,
)
from scripts.submit_mujoco_v14_15_closed_loop_restoration_filter_screen_scheduleurm import (
    _closed_loop_guard_contract_valid,
    build_scheduler_spec,
    build_training_command,
)


class MujocoV1415ClosedLoopRestorationFilterScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="mujoco_v14_15_unit",
            python_executable="python3",
            priority="normal",
            nodes=[f"node{index:03d}" for index in range(1, 7)],
        )

    def test_frozen_core_identity_is_source_bound(self):
        root = Path(__file__).resolve().parents[1]
        manifest = git_source_manifest_sha256(
            root,
            Path("freq_hrl"),
            revision=spec.FROZEN_ALGORITHM_REVISION,
        )
        self.assertEqual(manifest, spec.FROZEN_SOURCE_MANIFEST_SHA256)
        self.assertIn("v14_15", spec.DEVELOPMENT_PROTOCOL_VERSION)
        self.assertIn("v14_15", spec.FROZEN_CORE_PROTOCOL_VERSION)
        self.assertEqual(
            spec.FROZEN_CORE_PROTOCOL_VERSION,
            MUJOCO_CONTROL_PROTOCOL_VERSION,
        )

    def test_seed_namespace_is_fresh_and_role_disjoint(self):
        current_roles = (
            spec.OPTIMIZER_SEEDS,
            spec.PRETRAIN_SEEDS,
            spec.PRETRAIN_SELECTION_SEEDS,
            spec.CONTINUATION_TRAIN_SEEDS,
            spec.CONTINUATION_SELECTION_SEEDS,
            spec.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
            spec.DEVELOPMENT_EVALUATION_SEEDS,
        )
        current = [seed for role in current_roles for seed in role]
        self.assertEqual(len(current), len(set(current)))
        previous_roles = (
                v1413.OPTIMIZER_SEEDS,
                v1413.PRETRAIN_SEEDS,
                v1413.PRETRAIN_SELECTION_SEEDS,
                v1413.CONTINUATION_TRAIN_SEEDS,
                v1413.CONTINUATION_SELECTION_SEEDS,
                v1413.DEVELOPMENT_EVALUATION_SEEDS,
                v1414.OPTIMIZER_SEEDS,
                v1414.PRETRAIN_SEEDS,
                v1414.PRETRAIN_SELECTION_SEEDS,
                v1414.CONTINUATION_TRAIN_SEEDS,
                v1414.CONTINUATION_SELECTION_SEEDS,
                v1414.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
                v1414.DEVELOPMENT_EVALUATION_SEEDS,
        )
        previous = {seed for role in previous_roles for seed in role}
        self.assertFalse(previous.intersection(current))

    def test_restoration_and_strict_control_commands_are_explicit(self):
        args = self._args()
        joint_arm = spec.AUTHORIZING_ARMS[0]
        joint_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=joint_arm,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/joint"),
        )
        self.assertIn("--deployment-frequency-groupwise-robust", joint_command)
        self.assertIn("--deployment-frequency-anchor-state-replay", joint_command)
        self.assertIn("--deployment-frequency-ppo-trust-region", joint_command)
        self.assertIn(
            "--deployment-frequency-closed-loop-trust-region", joint_command
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-guard-seeds", joint_command
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-restoration-filter",
            joint_command,
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-restoration-min-reduction 0.0001",
            joint_command,
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-restoration-funnel-multiplier 2.0",
            joint_command,
        )
        self.assertIn(
            "--upper-deployment-frequency-reward-tolerance 0.001",
            joint_command,
        )
        self.assertIn(spec.FROZEN_ALGORITHM_REVISION, joint_command)
        strict_control = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.STRICT_CLOSED_LOOP_CONTROL_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/strict_control"),
        )
        self.assertIn(
            "--deployment-frequency-anchor-state-replay", strict_control
        )
        self.assertIn(
            "--deployment-frequency-ppo-trust-region", strict_control
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-trust-region", strict_control
        )
        self.assertNotIn(
            "--deployment-frequency-closed-loop-restoration-filter",
            strict_control,
        )

    def test_scheduler_is_dynamic_across_six_linux_nodes(self):
        payload = build_scheduler_spec(
            self._args(),
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.GROUPWISE_ARMS[0],
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(payload["require_node"])
        self.assertEqual(
            payload["allowed_nodes"],
            [f"node{index:03d}" for index in range(1, 7)],
        )
        self.assertTrue(payload["reroute_on_node_down"])
        self.assertEqual(payload["cpu"], 1)

    def test_projection_audit_rejects_missing_replay_or_reward_groups(self):
        arm = spec.ARMS[spec.AUTHORIZING_ARMS[0]]
        row = {"iteration": 0}
        for level in ("upper", "lower"):
            prefix = f"{level}_deployment_frequency"
            row.update({
                f"{prefix}_enabled": 1.0,
                f"{prefix}_projection_target_reached_before": 0.0,
                f"{prefix}_projection_steps_attempted": 1.0,
                f"{prefix}_projection_steps_accepted": 1.0,
                f"{prefix}_normalized_signed_excess_before": 1.0,
                f"{prefix}_normalized_signed_excess_after": 0.5,
                f"{prefix}_guard_reward_loss_delta": 0.0,
                f"{prefix}_group_reward_budget_violation_count": 0.0,
                f"{prefix}_groups_target_reached_after": 1.0,
                f"{prefix}_group_count": 8.0,
                f"{prefix}_reward_guard_group_count": 4.0,
                f"{prefix}_anchor_state_replay_enabled": 1.0,
                f"{prefix}_groupwise_robust": 1.0,
                f"{prefix}_projection_steps_requested": float(
                    arm[f"{prefix}_max_projection_steps"]
                ),
                f"{prefix}_projection_reward_tolerance": float(
                    arm[f"{prefix}_reward_tolerance"]
                ),
            })
        valid = _projection_diagnostics([row], arm)
        self.assertTrue(valid["pass"])
        self.assertEqual(valid["group_reward_budget_violation_count"], 0)

        missing_replay = copy.deepcopy(row)
        missing_replay[
            "upper_deployment_frequency_anchor_state_replay_enabled"
        ] = 0.0
        self.assertFalse(
            _projection_diagnostics([missing_replay], arm)["pass"]
        )

        pooled_reward = copy.deepcopy(row)
        pooled_reward[
            "upper_deployment_frequency_reward_guard_group_count"
        ] = 1.0
        self.assertFalse(
            _projection_diagnostics([pooled_reward], arm)["pass"]
        )

        unguarded = copy.deepcopy(row)
        unguarded[
            "lower_deployment_frequency_group_reward_budget_violation_count"
        ] = 1.0
        self.assertFalse(_projection_diagnostics([unguarded], arm)["pass"])

    def test_trust_and_replay_audits_require_nonzero_safe_updates(self):
        arm = spec.ARMS[spec.AUTHORIZING_ARMS[0]]
        row = {"iteration": 0}
        for level in ("upper", "lower"):
            prefix = f"{level}_deployment_frequency_ppo_guard"
            row.update({
                f"{prefix}_enabled": 1.0,
                f"{prefix}_step_fraction": 0.125,
                f"{prefix}_frequency_excess_before": 0.05,
                f"{prefix}_frequency_excess_after": 0.04,
                f"{prefix}_frequency_group_count": 8.0,
                f"{prefix}_reward_group_count": 4.0,
                f"{prefix}_group_reward_budget_violation_count": 0.0,
            })
        self.assertTrue(_trust_region_diagnostics([row], arm)["pass"])
        frozen = copy.deepcopy(row)
        frozen[
            "lower_deployment_frequency_ppo_guard_step_fraction"
        ] = 0.0
        self.assertFalse(_trust_region_diagnostics([frozen], arm)["pass"])

        summary = {
            "deployment_frequency_anchor_state_replay_enabled": True,
            "deployment_frequency_anchor_state_replay_path_count": 4,
            "deployment_frequency_anchor_state_replay_contract": (
                "deterministic_frozen_anchor_deployment_trajectory_v1"
            ),
            "deployment_frequency_anchor_state_replay_upper_transitions": 8,
            "deployment_frequency_anchor_state_replay_lower_transitions": 128,
        }
        self.assertTrue(_anchor_replay_diagnostics(summary, arm)["pass"])

    def test_closed_loop_merge_contract_audits_restoration_trace(self):
        prefix = "deployment_frequency_closed_loop_guard_"
        contract = spec.EXPECTED_CLOSED_LOOP_GUARD_CONTRACT
        initial_rank = [-0.2, -1.0, 0.02]
        feasible_rank = [0.0, 0.0, 0.02]
        initial = {
            f"{prefix}enabled": 1.0,
            f"{prefix}contract": contract,
            f"{prefix}path_count": float(
                spec.EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT
            ),
            f"{prefix}constraint_count": float(
                spec.EXPECTED_CLOSED_LOOP_GUARD_CONSTRAINT_COUNT
            ),
            f"{prefix}rank_before": initial_rank,
            f"{prefix}rank_full_step": initial_rank,
            f"{prefix}rank_after": initial_rank,
            f"{prefix}full_step_reward_violation_count": 0.0,
            f"{prefix}reward_violation_count": 0.0,
            f"{prefix}full_step_frequency_violation_count": 20.0,
            f"{prefix}frequency_violation_count": 20.0,
            f"{prefix}step_fraction": 0.0,
            f"{prefix}accepted": 0.0,
            f"{prefix}evaluation_count": 1.0,
            f"{prefix}final_actor_rms": 0.0,
            f"{prefix}restoration_filter_enabled": 1.0,
            f"{prefix}restoration_phase_before": "restoration",
            f"{prefix}restoration_phase_after": "restoration",
            f"{prefix}restoration_funnel_limit": 0.6,
            f"{prefix}restoration_merit_before": 1.0,
            f"{prefix}restoration_merit_full_step": 1.0,
            f"{prefix}restoration_merit_after": 1.0,
            f"{prefix}worst_frequency_violation_before": 0.2,
            f"{prefix}worst_frequency_violation_full_step": 0.2,
            f"{prefix}worst_frequency_violation_after": 0.2,
            f"{prefix}trial_trace": [],
            "iteration": -1,
        }
        history = [initial]
        previous_rank = initial_rank
        previous_count = 20
        previous_merit = 1.0
        previous_worst = 0.2
        for iteration in range(spec.CONTINUATION_ITERATIONS):
            count = 0
            merit = 0.0
            worst = 0.0
            rank = feasible_rank
            history.append({
                f"{prefix}enabled": 1.0,
                f"{prefix}contract": contract,
                f"{prefix}path_count": float(
                    spec.EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT
                ),
                f"{prefix}constraint_count": float(
                    spec.EXPECTED_CLOSED_LOOP_GUARD_CONSTRAINT_COUNT
                ),
                f"{prefix}rank_before": previous_rank,
                f"{prefix}rank_full_step": rank,
                f"{prefix}rank_after": rank,
                f"{prefix}full_step_reward_violation_count": 0.0,
                f"{prefix}reward_violation_count": 0.0,
                f"{prefix}full_step_frequency_violation_count": float(count),
                f"{prefix}frequency_violation_count": float(count),
                f"{prefix}step_fraction": 1.0,
                f"{prefix}accepted": 1.0,
                f"{prefix}evaluation_count": 1.0,
                f"{prefix}final_actor_rms": 0.01,
                f"{prefix}restoration_filter_enabled": 1.0,
                f"{prefix}restoration_phase_before": (
                    "restoration" if previous_count > 0 else "maintenance"
                ),
                f"{prefix}restoration_phase_after": "maintenance",
                f"{prefix}restoration_funnel_limit": 0.6,
                f"{prefix}restoration_merit_before": previous_merit,
                f"{prefix}restoration_merit_full_step": merit,
                f"{prefix}restoration_merit_after": merit,
                f"{prefix}worst_frequency_violation_before": previous_worst,
                f"{prefix}worst_frequency_violation_full_step": worst,
                f"{prefix}worst_frequency_violation_after": worst,
                f"{prefix}trial_trace": [{
                    "fraction": 1.0,
                    "accepted": True,
                    "rejection_reasons": [],
                    "reward_violation_count": 0,
                    "frequency_violation_count": count,
                    "rank": rank,
                    "frequency_violation_merit": merit,
                    "worst_frequency_violation": worst,
                }],
                "iteration": iteration,
            })
            previous_rank = rank
            previous_count = count
            previous_merit = merit
            previous_worst = worst
        assignments = {
            str(index): mode
            for index, mode in enumerate(
                mode
                for mode in spec.TRAINING_DISTURBANCE_MODES
                for _ in spec.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS
            )
        }
        summary = {
            f"{prefix}enabled": True,
            f"{prefix}contract": contract,
            f"{prefix}path_count": spec.EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT,
            f"{prefix}constraint_count": (
                spec.EXPECTED_CLOSED_LOOP_GUARD_CONSTRAINT_COUNT
            ),
            f"{prefix}evaluation_count": (
                2 + spec.CONTINUATION_ITERATIONS
            ),
            f"{prefix}effective_update_count": spec.CONTINUATION_ITERATIONS,
            f"{prefix}initial_rank": initial_rank,
            f"{prefix}training_final_rank": feasible_rank,
            f"{prefix}selected_rank": feasible_rank,
            f"{prefix}initial_frequency_violation_count": 20,
            f"{prefix}selected_frequency_violation_count": 0,
            f"{prefix}selected_reward_violation_count": 0,
            f"{prefix}initial_frequency_violation_merit": 1.0,
            f"{prefix}initial_worst_frequency_violation": 0.2,
            f"{prefix}training_final_frequency_violation_merit": 0.0,
            f"{prefix}training_final_worst_frequency_violation": 0.0,
            f"{prefix}selected_frequency_violation_merit": 0.0,
            f"{prefix}selected_worst_frequency_violation": 0.0,
            "deployment_frequency_closed_loop_restoration_filter_enabled": True,
            "deployment_frequency_closed_loop_restoration_min_reduction": 1e-4,
            "deployment_frequency_closed_loop_restoration_funnel_multiplier": 3.0,
            "deployment_frequency_closed_loop_restoration_funnel_limit": 0.6,
            "deployment_frequency_closed_loop_guard_seed_roots": list(
                spec.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS
            ),
            "deployment_frequency_closed_loop_guard_condition_assignment": (
                assignments
            ),
            "deployment_frequency_closed_loop_guard_baseline": {
                "enabled": True,
                "row_count": spec.EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT,
                "checkpoint_selection_rows_used": 0,
                "heldout_rows_used": 0,
                "parameter_sha256": "a" * 64,
            },
        }
        self.assertTrue(_closed_loop_guard_contract_valid(
            summary,
            history,
            expected=True,
            restoration_expected=True,
            restoration_min_reduction=1e-4,
            restoration_funnel_multiplier=3.0,
        ))
        regressed = copy.deepcopy(history)
        regressed[-1][f"{prefix}trial_trace"][0][
            "frequency_violation_count"
        ] = 1
        self.assertFalse(_closed_loop_guard_contract_valid(
            summary,
            regressed,
            expected=True,
            restoration_expected=True,
            restoration_min_reduction=1e-4,
            restoration_funnel_multiplier=3.0,
        ))
        missing_trace = copy.deepcopy(history)
        missing_trace[1][f"{prefix}trial_trace"] = []
        self.assertFalse(_closed_loop_guard_contract_valid(
            summary,
            missing_trace,
            expected=True,
            restoration_expected=True,
            restoration_min_reduction=1e-4,
            restoration_funnel_multiplier=3.0,
        ))

    def test_constraint_contracts_distinguish_restoration_from_strict(self):
        disabled = spec.deployment_constraint_contract(
            spec.MATCHED_COMPARATOR_ARM
        )
        strict = spec.deployment_constraint_contract(
            spec.STRICT_CLOSED_LOOP_CONTROL_ARM
        )
        restoration = spec.deployment_constraint_contract(
            spec.AUTHORIZING_ARMS[0]
        )
        self.assertEqual(len({disabled, strict, restoration}), 3)
        self.assertEqual(disabled, "disabled")
        self.assertIn("closed_loop", strict)
        self.assertNotIn("two_phase", strict)
        self.assertIn("anchor_state_replay", restoration)
        self.assertIn("ppo_trust_region", restoration)
        self.assertIn("two_phase", restoration)

    def test_evaluation_registry_is_owned_by_v14_15(self):
        fields = ("disturbance_mode", "seed", "episode_return")
        rows = [
            {
                "disturbance_mode": mode,
                "seed": seed,
                "episode_return": 0.0,
            }
            for mode in spec.EVALUATION_DISTURBANCE_MODES
            for seed in spec.DEVELOPMENT_EVALUATION_SEEDS
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(_read_rows(path)), len(rows))

            rows[-1] = dict(rows[0])
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "v14.15 evaluation"):
                _read_rows(path)


if __name__ == "__main__":
    unittest.main()
