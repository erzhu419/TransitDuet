import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import mujoco_v12_confirmatory_spec as v12
from scripts import mujoco_v13_behavioral_confirmatory_spec as v13
from scripts import mujoco_v14_behavior_screen_spec as v14
from scripts import mujoco_v14_1_upper_pd_screen_spec as v14_1
from scripts import mujoco_v14_2_physical_router_screen_spec as v14_2
from scripts import mujoco_v14_3_partial_router_screen_spec as v14_3
from scripts import mujoco_v14_4_router_homotopy_screen_spec as v14_4
from scripts import mujoco_v14_5_paired_anchor_screen_spec as v14_5
from scripts import mujoco_v14_6_conservative_transfer_screen_spec as v14_6
from scripts import mujoco_v14_7_joint_learned_projection_screen_spec as spec
from scripts.analyze_mujoco_v14_7_joint_learned_projection_screen import analyze
from scripts.submit_mujoco_v14_7_joint_learned_projection_screen_scheduleurm import (
    ANCHOR_ARM,
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
    task_signature,
)


class MujocoV147JointLearnedProjectionScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="unit_v14_7_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )

    def test_registry_commands_and_dynamic_scheduler_contract(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 384)
        self.assertEqual(len(set(cells)), 384)
        self.assertEqual(sum(cell[0] == "anchor" for cell in cells), 48)
        self.assertEqual(
            sum(cell[0] == "continuation" for cell in cells), 336
        )

        args = self._args()
        anchor_command = build_training_command(
            args,
            phase="anchor",
            environment="HalfCheetah-v5",
            arm=ANCHOR_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/anchor"),
        )
        self.assertNotIn("--initial-checkpoint-path", anchor_command)
        self.assertIn(
            "--lower-action-router-mode causal_joint_band_projection",
            anchor_command,
        )
        self.assertIn("--method freq_hrl", anchor_command)
        self.assertIn("--upper-dual-lr 0.0", anchor_command)
        self.assertIn("--lower-dual-lr 0.0", anchor_command)

        calibration_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.CALIBRATION_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/calibration"),
        )
        self.assertIn("--lower-action-router-strength 0.5", calibration_command)
        self.assertIn("--checkpoint-score-mode mean_reward", calibration_command)

        learned = "joint_s050_pd_u003_l003"
        learned_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=learned,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/learned"),
        )
        self.assertIn("--initial-checkpoint-path", learned_command)
        self.assertIn(
            "--initial-checkpoint-router-mode causal_joint_band_projection",
            learned_command,
        )
        self.assertIn("--upper-dual-lr 0.03", learned_command)
        self.assertIn("--lower-dual-lr 0.03", learned_command)
        self.assertIn(
            "--checkpoint-score-mode behavior_robust", learned_command
        )
        self.assertIn("--checkpoint-constraint-penalty 10.0", learned_command)

        scheduler = build_scheduler_spec(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=learned,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 1)
        self.assertEqual(scheduler["allowed_nodes"], args.nodes)
        self.assertTrue(scheduler["reroute_on_node_down"])
        self.assertEqual(len(scheduler["wait_for_files"]), 2)
        self.assertEqual(
            scheduler["signature"],
            task_signature(
                args.run_name,
                phase="continuation",
                environment="HalfCheetah-v5",
                arm=learned,
                optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            ),
        )

    def test_all_v147_seed_roles_are_fresh_and_disjoint(self):
        old = set()
        for module in (
            v12, v13, v14, v14_1, v14_2, v14_3, v14_4, v14_5, v14_6
        ):
            for name in dir(module):
                value = getattr(module, name)
                if "SEED" in name and isinstance(value, (tuple, list)):
                    old.update(int(item) for item in value)
        roles = (
            spec.OPTIMIZER_SEEDS,
            spec.PRETRAIN_SEEDS,
            spec.PRETRAIN_SELECTION_SEEDS,
            spec.CONTINUATION_TRAIN_SEEDS,
            spec.CONTINUATION_SELECTION_SEEDS,
            spec.DEVELOPMENT_EVALUATION_SEEDS,
        )
        flattened = [int(seed) for role in roles for seed in role]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertFalse(set(flattened) & old)

    @staticmethod
    def _write_checkpoint(path: Path, value: float) -> None:
        state = {
            "model_state_dict": {
                "upper_actor": {"weight": torch.full((2, 2), value)},
                "lower_actor": {"weight": torch.full((2, 2), value)},
            }
        }
        torch.save(state, path)

    @staticmethod
    def _write_rows(
        path: Path,
        *,
        arm: str,
        reward: float,
        lower_lf: float,
        raw_lower_lf: float,
        trace_suffix: str,
    ) -> None:
        fields = [
            "disturbance_mode", "seed", "episode_return",
            "LowerLFDriftAbs", "RawLowerLFDriftAbs",
            "LatentLowerLFDriftAbs", "RawLowerActionRMS",
            "LowerRouterClipRate", "LowerActionRouterStrength",
            "LowerRouterUpperTransferRMS",
            "LowerRouterFunctionPreserving",
            "LowerRouterActionReconstructionRMS",
            "UpperHFPowerAbs", "ResponsibilityReconstructionRMS",
            "RewardTraceSHA256", "ExecutedActionTraceSHA256",
            "LatentPolicyTraceSHA256",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for mode in spec.EVALUATION_DISTURBANCE_MODES:
                for evaluation_seed in spec.DEVELOPMENT_EVALUATION_SEEDS:
                    writer.writerow({
                        "disturbance_mode": mode,
                        "seed": evaluation_seed,
                        "episode_return": reward,
                        "LowerLFDriftAbs": lower_lf,
                        "RawLowerLFDriftAbs": raw_lower_lf,
                        "LatentLowerLFDriftAbs": lower_lf,
                        "RawLowerActionRMS": 1.0,
                        "LowerRouterClipRate": 0.0,
                        "LowerActionRouterStrength": spec.ARMS[arm][
                            "lower_action_router_strength"
                        ],
                        "LowerRouterUpperTransferRMS": (
                            0.0 if arm == spec.COMPARATOR_ARM else 0.1
                        ),
                        "LowerRouterFunctionPreserving": 1.0,
                        "LowerRouterActionReconstructionRMS": 0.0,
                        "UpperHFPowerAbs": 0.0025,
                        "ResponsibilityReconstructionRMS": 0.0,
                        "RewardTraceSHA256": trace_suffix * 64,
                        "ExecutedActionTraceSHA256": trace_suffix * 64,
                        "LatentPolicyTraceSHA256": trace_suffix * 64,
                    })

    def test_analyzer_separates_calibration_from_learned_behavior(self):
        comparator = spec.COMPARATOR_ARM
        calibration = spec.CALIBRATION_ARM
        learned = "joint_s050_pd_u003_l003"
        reduced_arms = {
            name: spec.ARMS[name]
            for name in (comparator, calibration, learned)
        }
        patchers = (
            patch.object(spec, "ENVIRONMENTS", ("HalfCheetah-v5",)),
            patch.object(
                spec,
                "EVALUATION_DISTURBANCE_MODES",
                ("standard", "ood_chirp"),
            ),
            patch.object(spec, "OPTIMIZER_SEEDS", spec.OPTIMIZER_SEEDS[:2]),
            patch.object(
                spec,
                "DEVELOPMENT_EVALUATION_SEEDS",
                spec.DEVELOPMENT_EVALUATION_SEEDS[:2],
            ),
            patch.object(spec, "ARMS", reduced_arms),
            patch.object(spec, "CANDIDATE_ARMS", (calibration, learned)),
            patch.object(spec, "LEARNED_ARMS", (learned,)),
            patch.object(spec, "MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS", 1),
            patch.object(spec, "BOOTSTRAP_DRAWS", 200),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            (run / "merged").mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_7_development_outcome_access",
                "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
                "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
                "frozen_source_manifest_sha256": (
                    spec.FROZEN_SOURCE_MANIFEST_SHA256
                ),
            }), encoding="utf-8")
            expected_cells = (
                len(spec.ENVIRONMENTS)
                * (len(spec.ARMS) + 1)
                * len(spec.OPTIMIZER_SEEDS)
            )
            (run / "merged" / "cell_manifest.json").write_text(json.dumps({
                "status": "development_screen_complete_unanalyzed",
                "cell_count": expected_cells,
            }), encoding="utf-8")
            (run / "merged" / "run_scoped_result_sync.json").write_text(
                json.dumps({"status": "run_scoped_result_sync_complete"}),
                encoding="utf-8",
            )

            for seed in spec.OPTIMIZER_SEEDS:
                anchor = (
                    run / "anchors" / "HalfCheetah-v5" / f"replicate_{seed}"
                )
                anchor.mkdir(parents=True)
                (anchor / "cell_summary.json").write_text("{}", encoding="utf-8")
                (anchor / "training_history.json").write_text("[]", encoding="utf-8")
                (anchor / "evaluation_rows.csv").write_text("", encoding="utf-8")
                self._write_checkpoint(anchor / "checkpoint.pt", 0.0)

            arm_values = {
                comparator: (100.0, 1.0, 1.0, "a", 0.0, "a"),
                calibration: (100.0, 0.8, 0.7, "a", 0.0, "a"),
                learned: (105.0, 0.75, 0.65, "b", 0.01, "b"),
            }
            for arm, values in arm_values.items():
                for seed in spec.OPTIMIZER_SEEDS:
                    cell = (
                        run / "cells" / "HalfCheetah-v5" / arm
                        / f"replicate_{seed}"
                    )
                    cell.mkdir(parents=True)
                    (cell / "cell_summary.json").write_text(json.dumps({
                        "selected_checkpoint_iteration": 11,
                        "upper_actor_anchor_parameter_rms": 0.01,
                        "lower_actor_anchor_parameter_rms": 0.02,
                        "frozen_parameter_sha256": values[5] * 64,
                    }), encoding="utf-8")
                    (cell / "training_history.json").write_text(json.dumps([{
                        "upper_actor_anchor_kl": 0.001,
                        "lower_actor_anchor_kl": 0.002,
                    }]), encoding="utf-8")
                    self._write_checkpoint(cell / "checkpoint.pt", values[4])
                    self._write_rows(
                        cell / "evaluation_rows.csv",
                        arm=arm,
                        reward=values[0],
                        lower_lf=values[1],
                        raw_lower_lf=values[2],
                        trace_suffix=values[3],
                    )

            decision = analyze(run)
            self.assertEqual(decision["status"], "learned_candidate_selected")
            self.assertEqual(decision["selected_arm"], learned)
            self.assertTrue(decision["calibration_validation_pass"])
            self.assertFalse(decision["arm_status"][calibration][
                "development_selection_pass"
            ])
            self.assertTrue(decision["arm_status"][learned][
                "learned_parameter_gate_pass"
            ])
            self.assertTrue(decision["arm_status"][learned][
                "learned_behavior_gate_pass"
            ])

            for seed in spec.OPTIMIZER_SEEDS:
                learned_rows = (
                    run / "cells" / "HalfCheetah-v5" / learned
                    / f"replicate_{seed}" / "evaluation_rows.csv"
                )
                self._write_rows(
                    learned_rows,
                    arm=learned,
                    reward=105.0,
                    lower_lf=0.75,
                    raw_lower_lf=0.65,
                    trace_suffix="a",
                )
            unchanged_behavior = analyze(run)
            self.assertEqual(
                unchanged_behavior["status"],
                "no_learned_behavior_safe_candidate",
            )
            self.assertFalse(unchanged_behavior["arm_status"][learned][
                "learned_behavior_gate_pass"
            ])


if __name__ == "__main__":
    unittest.main()
