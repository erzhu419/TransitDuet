import argparse
import copy
import csv
import tempfile
import unittest
from pathlib import Path

from scripts import mujoco_v14_11_iterative_projection_screen_spec as v1411
from scripts import mujoco_v14_12_groupwise_robust_screen_spec as spec
from scripts.analyze_mujoco_v14_12_groupwise_robust_preflight import (
    _projection_diagnostics,
    _read_rows,
)
from scripts.submit_hyperparameter_pilot_scheduleurm import source_identity
from scripts.submit_mujoco_v14_12_groupwise_robust_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
)


class MujocoV1412GroupwiseRobustScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="mujoco_v14_12_unit",
            python_executable="python3",
            priority="normal",
            nodes=[f"node{index:03d}" for index in range(1, 7)],
        )

    def test_frozen_core_identity_is_source_bound(self):
        revision, manifest = source_identity(
            spec.FROZEN_ALGORITHM_REVISION
        )
        self.assertEqual(revision, spec.FROZEN_ALGORITHM_REVISION)
        self.assertEqual(manifest, spec.FROZEN_SOURCE_MANIFEST_SHA256)
        self.assertIn("v14_12", spec.DEVELOPMENT_PROTOCOL_VERSION)
        self.assertIn("v14_12", spec.FROZEN_CORE_PROTOCOL_VERSION)

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
                v1411.OPTIMIZER_SEEDS,
                v1411.PRETRAIN_SEEDS,
                v1411.PRETRAIN_SELECTION_SEEDS,
                v1411.CONTINUATION_TRAIN_SEEDS,
                v1411.CONTINUATION_SELECTION_SEEDS,
                v1411.DEVELOPMENT_EVALUATION_SEEDS,
            )
            for seed in role
        }
        self.assertFalse(previous.intersection(current))

    def test_groupwise_command_and_actor_anchor_are_explicit(self):
        args = self._args()
        group_arm = "group_s050_asym_u003_l008_s310_r05_k8_a001"
        group_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=group_arm,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/group"),
        )
        self.assertIn("--deployment-frequency-groupwise-robust", group_command)
        self.assertIn("--upper-actor-anchor-coef 0.01", group_command)
        self.assertIn("--lower-actor-anchor-coef 0.01", group_command)
        self.assertIn(spec.FROZEN_ALGORITHM_REVISION, group_command)
        pooled_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.POOLED_COMPARATOR_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/pooled"),
        )
        self.assertNotIn(
            "--deployment-frequency-groupwise-robust", pooled_command
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

    def test_projection_audit_rejects_pooled_or_unguarded_rows(self):
        arm = spec.ARMS[spec.GROUPWISE_ARMS[0]]
        row = {"iteration": 0}
        for level in ("upper", "lower"):
            prefix = f"{level}_deployment_frequency"
            row.update({
                f"{prefix}_enabled": 1.0,
                f"{prefix}_projection_target_reached_before": 0.0,
                f"{prefix}_projection_steps_attempted": 2.0,
                f"{prefix}_projection_steps_accepted": 2.0,
                f"{prefix}_normalized_signed_excess_before": 1.0,
                f"{prefix}_normalized_signed_excess_after": 0.5,
                f"{prefix}_guard_reward_loss_delta": 0.0,
                f"{prefix}_group_reward_budget_violation_count": 0.0,
                f"{prefix}_groups_target_reached_after": 1.0,
                f"{prefix}_group_count": 4.0,
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

        pooled = copy.deepcopy(row)
        pooled["upper_deployment_frequency_group_count"] = 1.0
        self.assertFalse(_projection_diagnostics([pooled], arm)["pass"])

        unguarded = copy.deepcopy(row)
        unguarded[
            "lower_deployment_frequency_group_reward_budget_violation_count"
        ] = 1.0
        self.assertFalse(_projection_diagnostics([unguarded], arm)["pass"])

    def test_evaluation_registry_is_owned_by_v14_12(self):
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
            with self.assertRaisesRegex(ValueError, "v14.12 evaluation"):
                _read_rows(path)


if __name__ == "__main__":
    unittest.main()
