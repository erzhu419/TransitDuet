import argparse
import copy
import csv
import tempfile
import unittest
from pathlib import Path

from scripts import mujoco_v14_12_groupwise_robust_screen_spec as v1412
from scripts import mujoco_v14_13_anchor_replay_trust_screen_spec as spec
from freq_hrl.experiments.mujoco.control_validation import (
    MUJOCO_CONTROL_PROTOCOL_VERSION,
)
from freq_hrl.experiments.reproducibility import (
    git_source_manifest_sha256,
)
from scripts.analyze_mujoco_v14_13_anchor_replay_trust_preflight import (
    _projection_diagnostics,
    _anchor_replay_diagnostics,
    _read_rows,
    _trust_region_diagnostics,
)
from scripts.submit_mujoco_v14_13_anchor_replay_trust_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
)


class MujocoV1413AnchorReplayTrustScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="mujoco_v14_13_unit",
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
        self.assertIn("v14_13", spec.DEVELOPMENT_PROTOCOL_VERSION)
        self.assertIn("v14_13", spec.FROZEN_CORE_PROTOCOL_VERSION)
        self.assertNotEqual(
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
            spec.DEVELOPMENT_EVALUATION_SEEDS,
        )
        current = [seed for role in current_roles for seed in role]
        self.assertEqual(len(current), len(set(current)))
        previous = {
            seed
            for role in (
                v1412.OPTIMIZER_SEEDS,
                v1412.PRETRAIN_SEEDS,
                v1412.PRETRAIN_SELECTION_SEEDS,
                v1412.CONTINUATION_TRAIN_SEEDS,
                v1412.CONTINUATION_SELECTION_SEEDS,
                v1412.DEVELOPMENT_EVALUATION_SEEDS,
            )
            for seed in role
        }
        self.assertFalse(previous.intersection(current))

    def test_joint_and_ablation_commands_are_explicit(self):
        args = self._args()
        joint_arm = "group_replay1_trust1_eps1e2_k8"
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
            "--upper-deployment-frequency-reward-tolerance 0.01",
            joint_command,
        )
        self.assertIn(spec.FROZEN_ALGORITHM_REVISION, joint_command)
        replay_only = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm="group_replay1_trust0_eps1e2_k8",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/replay_only"),
        )
        self.assertIn(
            "--deployment-frequency-anchor-state-replay", replay_only
        )
        self.assertNotIn(
            "--deployment-frequency-ppo-trust-region", replay_only
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

    def test_constraint_contracts_distinguish_all_mechanism_ablations(self):
        groupwise = spec.deployment_constraint_contract(
            "group_replay0_trust0_eps1e8_k8"
        )
        replay = spec.deployment_constraint_contract(
            "group_replay1_trust0_eps1e2_k8"
        )
        trust = spec.deployment_constraint_contract(
            "group_replay0_trust1_eps1e2_k8"
        )
        joint = spec.deployment_constraint_contract(
            "group_replay1_trust1_eps1e2_k8"
        )
        self.assertEqual(len({groupwise, replay, trust, joint}), 4)
        self.assertNotIn("anchor_state_replay", groupwise)
        self.assertIn("anchor_state_replay", replay)
        self.assertNotIn("ppo_trust_region", replay)
        self.assertIn("ppo_trust_region", trust)
        self.assertNotIn("anchor_state_replay", trust)
        self.assertIn("anchor_state_replay", joint)
        self.assertIn("ppo_trust_region", joint)

    def test_evaluation_registry_is_owned_by_v14_13(self):
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
            with self.assertRaisesRegex(ValueError, "v14.13 evaluation"):
                _read_rows(path)


if __name__ == "__main__":
    unittest.main()
