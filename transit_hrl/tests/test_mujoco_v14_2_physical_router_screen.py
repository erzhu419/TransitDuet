import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import mujoco_v12_confirmatory_spec as v12
from scripts import mujoco_v13_behavioral_confirmatory_spec as v13
from scripts import mujoco_v14_behavior_screen_spec as v14
from scripts import mujoco_v14_1_upper_pd_screen_spec as v14_1
from scripts import mujoco_v14_2_physical_router_screen_spec as spec
from scripts.analyze_mujoco_v14_2_physical_router_screen import analyze
from scripts.submit_mujoco_v14_2_physical_router_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV142PhysicalRouterScreenTest(unittest.TestCase):
    def test_frozen_registry_has_432_dynamic_cells(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 432)
        self.assertEqual(len(set(cells)), 432)
        args = argparse.Namespace(
            run_name="unit_v14_2_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )
        scheduler_spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            arm="crossed_router_a010_pd_u2_l8",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(scheduler_spec["allowed_nodes"], args.nodes)
        command = build_training_command(
            args,
            environment="HalfCheetah-v5",
            arm="crossed_router_a010_pd_u2_l8",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
        )
        self.assertIn("--upper-constraint-mode primal_dual", command)
        self.assertIn("--upper-dual-lr 2.0", command)
        self.assertIn("--lower-dual-lr 8.0", command)
        self.assertIn("--leakage-cost-mode power_excess", command)
        self.assertIn("--lower-action-router-mode causal_ema_high_pass", command)
        self.assertIn("--lower-action-router-alpha 0.1", command)
        self.assertIn("--checkpoint-selection-mode crossed_conditions", command)
        self.assertIn("--checkpoint-score-mode behavior_robust", command)
        self.assertIn(
            "--code-revision " + spec.FROZEN_ALGORITHM_REVISION,
            command,
        )

    def test_development_seeds_do_not_reuse_v12_to_v141_roles(self):
        old = set()
        for module in (v12, v13, v14, v14_1):
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
        test_arm_names = (
            "additive_reward_baseline",
            "crossed_router_a010_reward",
            "crossed_direct_pd_u2_l8",
            "crossed_router_a004_pd_u2_l8",
            "crossed_router_a010_pd_u2_l8",
            "crossed_router_a010_pd_u8_l32",
        )
        patchers = (
            patch.object(spec, "ENVIRONMENTS", spec.ENVIRONMENTS[:2]),
            patch.object(
                spec,
                "EVALUATION_DISTURBANCE_MODES",
                spec.EVALUATION_DISTURBANCE_MODES[:2],
            ),
            patch.object(
                spec,
                "ARMS",
                {name: spec.ARMS[name] for name in test_arm_names},
            ),
            patch.object(spec, "CANDIDATE_ARMS", test_arm_names[1:]),
            patch.object(spec, "OPTIMIZER_SEEDS", spec.OPTIMIZER_SEEDS[:4]),
            patch.object(
                spec,
                "DEVELOPMENT_EVALUATION_SEEDS",
                spec.DEVELOPMENT_EVALUATION_SEEDS[:2],
            ),
            patch.object(spec, "BOOTSTRAP_DRAWS", 200),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        arm_values = {
            "additive_reward_baseline": (100.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0025),
            "crossed_direct_reward": (100.0, 0.6, 1.1, 1.1, 0.8, 0.0, 0.0025),
            "crossed_router_a004_reward": (80.0, 0.5, 0.5, 0.8, 0.5, 0.0, 0.0025),
            "crossed_router_a010_reward": (99.5, 0.5, 0.5, 0.8, 0.5, 0.2, 0.0025),
            "crossed_direct_pd_u2_l8": (99.5, 0.5, 0.5, 0.8, 0.5, 0.0, 0.0025),
            "crossed_router_a004_pd_u2_l8": (99.5, 0.5, 0.5, 0.8, 0.5, 0.0, 0.0025),
            "crossed_router_a010_pd_u0p5_l2": (99.5, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0025),
            "crossed_router_a010_pd_u2_l8": (99.5, 0.5, 0.5, 0.8, 0.5, 0.0, 0.0025),
            "crossed_router_a010_pd_u8_l32": (95.0, 0.2, 0.2, 0.5, 0.5, 0.0, 0.0025),
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "results" / "unit_v14_2"
            merged = run / "merged"
            merged.mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_2_development_outcome_access",
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
                "cell_count": (
                    len(spec.ENVIRONMENTS)
                    * len(spec.ARMS)
                    * len(spec.OPTIMIZER_SEEDS)
                ),
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
                        selected_iteration = (
                            -1 if arm == "crossed_direct_pd_u2_l8" else 8
                        )
                        (cell / "cell_summary.json").write_text(json.dumps({
                            "selected_checkpoint_iteration": selected_iteration,
                        }) + "\n", encoding="utf-8")
                        saturated = arm == "crossed_router_a004_pd_u2_l8"
                        (cell / "training_history.json").write_text(json.dumps([
                            {
                                "upper_constraint_lambda": (
                                    20.0 if saturated else 1.0
                                ),
                                "constraint_lambda": (
                                    20.0 if saturated else 1.0
                                ),
                            }
                        ]) + "\n", encoding="utf-8")
                        with (cell / "evaluation_rows.csv").open(
                            "w", newline="", encoding="utf-8"
                        ) as handle:
                            fields = [
                                "disturbance_mode",
                                "seed",
                                "episode_return",
                                "LowerLFDriftAbs",
                                "RawLowerLFDriftAbs",
                                "LatentLowerLFDriftAbs",
                                "RawLowerActionRMS",
                                "LowerRouterClipRate",
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
                                    at_floor = (
                                        environment == "Hopper-v5"
                                        and mode == "standard"
                                    )
                                    responsibility = (
                                        spec.DRIFT_MATERIALITY_FLOOR / 10.0
                                        if at_floor and arm == spec.BASELINE_ARM
                                        else (
                                            spec.DRIFT_MATERIALITY_FLOOR / 2.0
                                            if at_floor else values[1]
                                        )
                                    )
                                    raw_lower = (
                                        spec.DRIFT_MATERIALITY_FLOOR / 10.0
                                        if at_floor and arm == spec.BASELINE_ARM
                                        else (
                                            spec.DRIFT_MATERIALITY_FLOOR / 2.0
                                            if at_floor else values[2]
                                        )
                                    )
                                    writer.writerow({
                                        "disturbance_mode": mode,
                                        "seed": evaluation_seed,
                                        "episode_return": values[0],
                                        "LowerLFDriftAbs": responsibility,
                                        "RawLowerLFDriftAbs": raw_lower,
                                        "LatentLowerLFDriftAbs": values[3],
                                        "RawLowerActionRMS": values[4],
                                        "LowerRouterClipRate": values[5],
                                        "UpperHFPowerAbs": values[6],
                                        "ResponsibilityReconstructionRMS": 0.0,
                                    })
            decision = analyze(run)
        self.assertEqual(decision["status"], "development_candidate_selected")
        self.assertEqual(
            decision["selected_arm"], "crossed_router_a010_pd_u2_l8"
        )
        self.assertEqual(decision["gate_granularity"], "environment_by_disturbance_mode")
        self.assertEqual(
            decision["arm_status"][
                "crossed_router_a010_pd_u2_l8"
            ]["passed_gate_count"],
            4,
        )
        self.assertTrue(all(
            not row["router_clip_pass"]
            for row in decision["environment_condition_rows"]
            if row["arm"] == "crossed_router_a010_reward"
        ))
        self.assertTrue(all(
            not row["reward_noninferiority_pass"]
            for row in decision["environment_condition_rows"]
            if row["arm"] == "crossed_router_a010_pd_u8_l32"
        ))
        at_floor_rows = [
            row for row in decision["environment_condition_rows"]
            if row["arm"] == "crossed_router_a010_pd_u2_l8"
            and row["environment"] == "Hopper-v5"
            and row["disturbance_mode"] == "standard"
        ]
        self.assertEqual(len(at_floor_rows), 1)
        self.assertEqual(
            at_floor_rows[0]["raw_lower_drift_gate_type"],
            "absolute_floor_noninferiority",
        )
        self.assertTrue(at_floor_rows[0]["raw_lower_drift_pass"])
        self.assertFalse(
            decision["arm_status"]["crossed_direct_pd_u2_l8"][
                "trained_checkpoint_gate_pass"
            ]
        )
        self.assertFalse(
            decision["arm_status"]["crossed_router_a004_pd_u2_l8"][
                "dual_saturation_gate_pass"
            ]
        )


if __name__ == "__main__":
    unittest.main()
