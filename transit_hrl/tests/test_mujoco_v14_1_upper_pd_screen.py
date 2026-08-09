import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import mujoco_v12_confirmatory_spec as v12
from scripts import mujoco_v13_behavioral_confirmatory_spec as v13
from scripts import mujoco_v14_behavior_screen_spec as v14
from scripts import mujoco_v14_1_upper_pd_screen_spec as spec
from scripts.analyze_mujoco_v14_1_upper_pd_screen import analyze
from scripts.submit_mujoco_v14_1_upper_pd_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV141UpperPDScreenTest(unittest.TestCase):
    def test_frozen_registry_has_144_dynamic_cells(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 144)
        self.assertEqual(len(set(cells)), 144)
        args = argparse.Namespace(
            run_name="unit_v14_1_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )
        scheduler_spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            arm="crossed_joint_pd2",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(scheduler_spec["allowed_nodes"], args.nodes)
        command = build_training_command(
            args,
            environment="HalfCheetah-v5",
            arm="crossed_joint_pd2",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
        )
        self.assertIn("--upper-constraint-mode primal_dual", command)
        self.assertIn("--upper-dual-lr 2.0", command)
        self.assertIn("--checkpoint-selection-mode crossed_conditions", command)
        self.assertIn("--checkpoint-score-mode behavior_robust", command)
        self.assertIn(
            "--code-revision " + spec.FROZEN_ALGORITHM_REVISION,
            command,
        )

    def test_development_seeds_do_not_reuse_v12_to_v14_roles(self):
        old = set()
        for module in (v12, v13, v14):
            for name in dir(module):
                value = getattr(module, name)
                if "SEED" in name and isinstance(value, (tuple, list)):
                    old.update(int(item) for item in value)
        current = set(
            spec.OPTIMIZER_SEEDS
            + spec.TRAIN_SEEDS
            + spec.CHECKPOINT_SELECTION_SEEDS
            + spec.DEVELOPMENT_EVALUATION_SEEDS
        )
        self.assertFalse(current & old)

    def test_analyzer_requires_every_environment_condition_gate(self):
        arm_values = {
            "additive_baseline": (100.0, 1.0, 1.0, 0.0025),
            "crossed_joint_beta0": (100.0, 0.6, 0.6, 0.0400),
            "crossed_joint_beta0p5": (99.5, 0.6, 0.6, 0.0064),
            "crossed_joint_pd0p5": (99.0, 0.7, 0.7, 0.0049),
            "crossed_joint_pd2": (99.5, 0.5, 0.5, 0.0036),
            "crossed_joint_pd8": (95.0, 0.2, 0.2, 0.0016),
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "results" / "unit_v14_1"
            merged = run / "merged"
            merged.mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_1_development_outcome_access",
                "development_protocol_version": (
                    spec.DEVELOPMENT_PROTOCOL_VERSION
                ),
                "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
                "frozen_source_manifest_sha256": (
                    spec.FROZEN_SOURCE_MANIFEST_SHA256
                ),
            }), encoding="utf-8")
            (merged / "cell_manifest.json").write_text(json.dumps({
                "status": "development_screen_complete_unanalyzed",
                "cell_count": 144,
            }), encoding="utf-8")
            for environment in spec.ENVIRONMENTS:
                for arm in spec.ARMS:
                    values = arm_values[arm]
                    for optimizer_seed in spec.OPTIMIZER_SEEDS:
                        cell = (
                            run / "cells" / environment / arm
                            / f"replicate_{optimizer_seed}"
                        )
                        cell.mkdir(parents=True)
                        (cell / "cell_summary.json").write_text(
                            "{}\n", encoding="utf-8"
                        )
                        with (cell / "evaluation_rows.csv").open(
                            "w", newline="", encoding="utf-8"
                        ) as handle:
                            fields = [
                                "disturbance_mode",
                                "seed",
                                "episode_return",
                                "LowerLFDriftAbs",
                                "RawLowerLFDriftAbs",
                                "UpperHFPowerAbs",
                                "ResponsibilityReconstructionRMS",
                            ]
                            writer = csv.DictWriter(
                                handle, fieldnames=fields, lineterminator="\n"
                            )
                            writer.writeheader()
                            for mode in spec.EVALUATION_DISTURBANCE_MODES:
                                for evaluation_seed in (
                                    spec.DEVELOPMENT_EVALUATION_SEEDS
                                ):
                                    writer.writerow({
                                        "disturbance_mode": mode,
                                        "seed": evaluation_seed,
                                        "episode_return": values[0],
                                        "LowerLFDriftAbs": values[1],
                                        "RawLowerLFDriftAbs": values[2],
                                        "UpperHFPowerAbs": values[3],
                                        "ResponsibilityReconstructionRMS": 0.0,
                                    })
            decision = analyze(run)
        self.assertEqual(decision["status"], "development_candidate_selected")
        self.assertEqual(decision["selected_arm"], "crossed_joint_pd2")
        self.assertEqual(decision["gate_granularity"], "environment_by_disturbance_mode")
        self.assertEqual(
            decision["arm_status"]["crossed_joint_pd2"]["passed_gate_count"],
            15,
        )
        self.assertTrue(all(
            not row["upper_hf_budget_pass"]
            for row in decision["environment_condition_rows"]
            if row["arm"] == "crossed_joint_beta0"
        ))
        self.assertTrue(all(
            not row["reward_noninferiority_pass"]
            for row in decision["environment_condition_rows"]
            if row["arm"] == "crossed_joint_pd8"
        ))


if __name__ == "__main__":
    unittest.main()
