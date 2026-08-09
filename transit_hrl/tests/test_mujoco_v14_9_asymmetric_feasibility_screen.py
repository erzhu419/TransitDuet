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
from scripts import mujoco_v14_7_joint_learned_projection_screen_spec as v14_7
from scripts import mujoco_v14_8_latent_matched_screen_spec as v14_8
from scripts import mujoco_v14_9_asymmetric_feasibility_screen_spec as spec
from scripts.analyze_mujoco_v14_9_asymmetric_feasibility_screen import analyze
from scripts.submit_mujoco_v14_9_asymmetric_feasibility_screen_scheduleurm import (
    ANCHOR_ARM,
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
    task_signature,
)


class MujocoV149AsymmetricFeasibilityScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="unit_v14_9_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )

    def test_registry_commands_and_dynamic_scheduler_contract(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 576)
        self.assertEqual(len(set(cells)), 576)
        self.assertEqual(sum(cell[0] == "anchor" for cell in cells), 48)
        self.assertEqual(
            sum(cell[0] == "continuation" for cell in cells), 528
        )

        args = self._args()
        anchor = build_training_command(
            args,
            phase="anchor",
            environment="HalfCheetah-v5",
            arm=ANCHOR_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/anchor"),
        )
        self.assertNotIn("--initial-checkpoint-path", anchor)
        self.assertIn(
            "--leakage-constraint-scope joint_behavior_latent", anchor
        )
        self.assertIn("--checkpoint-score-mode mean_reward", anchor)

        calibration = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.CALIBRATION_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/calibration"),
        )
        self.assertIn("--lower-action-router-strength 0.5", calibration)
        self.assertIn("--checkpoint-score-mode mean_reward", calibration)
        self.assertIn("--upper-dual-lr 0.0", calibration)

        matched = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.MATCHED_COMPARATOR_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/matched"),
        )
        self.assertIn(
            "--checkpoint-score-mode latent_behavior_feasibility_first",
            matched,
        )
        self.assertIn("--checkpoint-constraint-penalty 10.0", matched)
        self.assertIn("--upper-dual-lr 0.0", matched)

        learned_arm = "feasible_s050_u020_l300"
        learned = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=learned_arm,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/learned"),
        )
        self.assertIn("--upper-dual-lr 0.2", learned)
        self.assertIn("--lower-dual-lr 3.0", learned)
        self.assertIn("--initial-checkpoint-path", learned)

        scheduler = build_scheduler_spec(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=learned_arm,
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
                arm=learned_arm,
                optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            ),
        )

    def test_all_v149_seed_roles_are_fresh_and_disjoint(self):
        old = set()
        for module in (
            v12, v13, v14, v14_1, v14_2, v14_3, v14_4, v14_5, v14_6,
            v14_7, v14_8,
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
        torch.save({
            "model_state_dict": {
                "upper_actor": {"weight": torch.full((2, 2), value)},
                "lower_actor": {"weight": torch.full((2, 2), value)},
            }
        }, path)

    @staticmethod
    def _write_rows(
        path: Path,
        *,
        arm: str,
        reward: float,
        lower_lf: float,
        raw_lower_lf: float,
        latent_lower_lf: float,
        upper_hf: float,
        latent_upper_hf: float,
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
            "UpperHFPowerAbs", "LatentUpperHFPowerAbs",
            "ResponsibilityReconstructionRMS", "RewardTraceSHA256",
            "ExecutedActionTraceSHA256", "LatentPolicyTraceSHA256",
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
                        "LatentLowerLFDriftAbs": latent_lower_lf,
                        "RawLowerActionRMS": 1.0,
                        "LowerRouterClipRate": 0.0,
                        "LowerActionRouterStrength": spec.ARMS[arm][
                            "lower_action_router_strength"
                        ],
                        "LowerRouterUpperTransferRMS": (
                            0.0 if arm == spec.BASE_CONTROL_ARM else 0.1
                        ),
                        "LowerRouterFunctionPreserving": 1.0,
                        "LowerRouterActionReconstructionRMS": 0.0,
                        "UpperHFPowerAbs": upper_hf,
                        "LatentUpperHFPowerAbs": latent_upper_hf,
                        "ResponsibilityReconstructionRMS": 0.0,
                        "RewardTraceSHA256": trace_suffix * 64,
                        "ExecutedActionTraceSHA256": trace_suffix * 64,
                        "LatentPolicyTraceSHA256": trace_suffix * 64,
                    })

    def test_analyzer_uses_separate_calibration_and_matched_controls(self):
        learned = "feasible_s050_u020_l300"
        arm_names = (
            spec.BASE_CONTROL_ARM,
            spec.CALIBRATION_ARM,
            spec.MATCHED_COMPARATOR_ARM,
            learned,
        )
        reduced_arms = {name: spec.ARMS[name] for name in arm_names}
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
            patch.object(spec, "LEARNED_ARMS", (learned,)),
            patch.object(spec, "CANDIDATE_ARMS", (learned,)),
            patch.object(spec, "EVALUATED_ARMS", (spec.CALIBRATION_ARM, learned)),
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
                "status": "frozen_before_v14_9_development_outcome_access",
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

            values = {
                spec.BASE_CONTROL_ARM: (
                    100.0, 1.0, 1.0, 1.0, 0.0049, 0.0049, "a", 0.0, "a",
                ),
                spec.CALIBRATION_ARM: (
                    100.0, 0.75, 0.70, 1.0, 0.0030, 0.0049, "a", 0.0, "c",
                ),
                spec.MATCHED_COMPARATOR_ARM: (
                    100.0, 0.80, 0.70, 1.0, 0.0030, 0.0049, "m", 0.02, "m",
                ),
                learned: (
                    105.0, 0.60, 0.50, 0.70, 0.0025, 0.0030, "b", 0.03, "b",
                ),
            }
            for arm, arm_values in values.items():
                for seed in spec.OPTIMIZER_SEEDS:
                    cell = (
                        run / "cells" / "HalfCheetah-v5" / arm
                        / f"replicate_{seed}"
                    )
                    cell.mkdir(parents=True)
                    (cell / "cell_summary.json").write_text(json.dumps({
                        "selected_checkpoint_iteration": 11,
                        "frozen_parameter_sha256": arm_values[8] * 64,
                    }), encoding="utf-8")
                    (cell / "training_history.json").write_text(json.dumps([{
                        "upper_constraint_lambda": 0.1,
                        "lower_constraint_lambda": 0.1,
                    }]), encoding="utf-8")
                    self._write_checkpoint(cell / "checkpoint.pt", arm_values[7])
                    self._write_rows(
                        cell / "evaluation_rows.csv",
                        arm=arm,
                        reward=arm_values[0],
                        lower_lf=arm_values[1],
                        raw_lower_lf=arm_values[2],
                        latent_lower_lf=arm_values[3],
                        upper_hf=arm_values[4],
                        latent_upper_hf=arm_values[5],
                        trace_suffix=arm_values[6],
                    )

            decision = analyze(run)
            self.assertEqual(decision["status"], "learned_candidate_selected")
            self.assertEqual(decision["selected_arm"], learned)
            self.assertTrue(decision["calibration_validation_pass"])
            self.assertTrue(decision["arm_status"][spec.CALIBRATION_ARM][
                "calibration_actor_identity_gate_pass"
            ])
            self.assertTrue(decision["arm_status"][learned][
                "learned_actor_difference_gate_pass"
            ])
            self.assertEqual(
                decision["arm_status"][learned]["comparator_arm"],
                spec.MATCHED_COMPARATOR_ARM,
            )

            for seed in spec.OPTIMIZER_SEEDS:
                path = (
                    run / "cells" / "HalfCheetah-v5" / learned
                    / f"replicate_{seed}" / "evaluation_rows.csv"
                )
                self._write_rows(
                    path,
                    arm=learned,
                    reward=105.0,
                    lower_lf=0.60,
                    raw_lower_lf=0.50,
                    latent_lower_lf=1.10,
                    upper_hf=0.0025,
                    latent_upper_hf=0.0030,
                    trace_suffix="b",
                )
            latent_failure = analyze(run)
            self.assertEqual(
                latent_failure["status"],
                "no_latent_behavior_safe_candidate",
            )
            self.assertFalse(all(
                row["latent_lower_drift_pass"]
                for row in latent_failure["environment_condition_rows"]
                if row["arm"] == learned
            ))


if __name__ == "__main__":
    unittest.main()
