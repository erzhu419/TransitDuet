import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import mujoco_v14_behavior_screen_spec as spec
from scripts.analyze_mujoco_v14_behavior_screen import analyze
from scripts.submit_mujoco_v14_behavior_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV14BehaviorScreenTest(unittest.TestCase):
    def test_frozen_registry_has_120_unpinned_cells(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 120)
        self.assertEqual(len(set(cells)), 120)
        args = argparse.Namespace(
            run_name="unit_v14_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )
        scheduler_spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            arm="joint_beta2",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(scheduler_spec["allowed_nodes"], args.nodes)
        command = build_training_command(
            args,
            environment="HalfCheetah-v5",
            arm="joint_beta2",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
        )
        self.assertIn("--leakage-constraint-scope joint_behavior", command)
        self.assertIn("--upper-hf-penalty-coef 2.0", command)
        self.assertIn(
            "--code-revision " + spec.FROZEN_ALGORITHM_REVISION,
            command,
        )

    def test_development_analyzer_selects_jointly_safe_beta(self):
        arm_values = {
            "additive_baseline": (100.0, 1.0, 1.0, 0.0025),
            "joint_beta0": (100.0, 0.6, 0.6, 0.0400),
            "joint_beta0p5": (99.5, 0.6, 0.6, 0.0064),
            "joint_beta2": (99.0, 0.5, 0.5, 0.0049),
            "joint_beta8": (95.0, 0.2, 0.2, 0.0016),
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "results" / "unit_v14"
            merged = run / "merged"
            merged.mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_development_outcome_access",
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
                "cell_count": 120,
            }), encoding="utf-8")
            for environment in spec.ENVIRONMENTS:
                for arm in spec.ARMS:
                    values = arm_values[arm]
                    for seed in spec.OPTIMIZER_SEEDS:
                        cell = (
                            run / "cells" / environment / arm
                            / f"replicate_{seed}"
                        )
                        cell.mkdir(parents=True)
                        (cell / "cell_summary.json").write_text(
                            "{}\n", encoding="utf-8"
                        )
                        with (cell / "evaluation_rows.csv").open(
                            "w", newline="", encoding="utf-8"
                        ) as handle:
                            writer = csv.DictWriter(handle, fieldnames=[
                                "episode_return",
                                "LowerLFDriftAbs",
                                "RawLowerLFDriftAbs",
                                "UpperHFPowerAbs",
                                "ResponsibilityReconstructionRMS",
                            ], lineterminator="\n")
                            writer.writeheader()
                            for _ in range(40):
                                writer.writerow({
                                    "episode_return": values[0],
                                    "LowerLFDriftAbs": values[1],
                                    "RawLowerLFDriftAbs": values[2],
                                    "UpperHFPowerAbs": values[3],
                                    "ResponsibilityReconstructionRMS": 0.0,
                                })
            decision = analyze(run)
        self.assertEqual(decision["status"], "development_candidate_selected")
        self.assertEqual(decision["selected_arm"], "joint_beta2")
        beta0 = [
            row for row in decision["environment_rows"]
            if row["arm"] == "joint_beta0"
        ]
        self.assertTrue(all(not row["upper_hf_budget_pass"] for row in beta0))
        beta8 = [
            row for row in decision["environment_rows"]
            if row["arm"] == "joint_beta8"
        ]
        self.assertTrue(all(
            not row["reward_noninferiority_pass"] for row in beta8
        ))


if __name__ == "__main__":
    unittest.main()
