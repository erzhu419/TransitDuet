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
from scripts import mujoco_v14_2_physical_router_screen_spec as v14_2
from scripts import mujoco_v14_3_partial_router_screen_spec as v14_3
from scripts import mujoco_v14_4_router_homotopy_screen_spec as spec
from scripts.analyze_mujoco_v14_4_router_homotopy_screen import (
    _validate_router_strength,
    analyze,
)
from scripts.submit_mujoco_v14_4_router_homotopy_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV144RouterHomotopyScreenTest(unittest.TestCase):
    def test_frozen_registry_has_384_dynamic_cells(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 384)
        self.assertEqual(len(set(cells)), 384)
        args = argparse.Namespace(
            run_name="unit_v14_4_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )
        scheduler_spec = build_scheduler_spec(
            args,
            environment="HalfCheetah-v5",
            arm="router_a004_s010_linear_w0125_r0375",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(scheduler_spec["allowed_nodes"], args.nodes)
        command = build_training_command(
            args,
            environment="HalfCheetah-v5",
            arm="router_a004_s010_linear_w0125_r0375",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
        )
        self.assertIn("--upper-constraint-mode disabled", command)
        self.assertIn("--upper-dual-lr 0.0", command)
        self.assertIn("--lower-dual-lr 0.0", command)
        self.assertIn("--leakage-cost-mode power_excess", command)
        self.assertIn("--lower-action-router-mode causal_ema_high_pass", command)
        self.assertIn("--lower-action-router-alpha 0.04", command)
        self.assertIn("--lower-action-router-strength 0.1", command)
        self.assertIn(
            "--lower-action-router-training-schedule delayed_linear",
            command,
        )
        self.assertIn(
            "--lower-action-router-warmup-fraction 0.125",
            command,
        )
        self.assertIn(
            "--lower-action-router-ramp-fraction 0.375",
            command,
        )
        self.assertIn("--lower-action-router-observe-strength", command)
        self.assertIn("--checkpoint-selection-mode crossed_conditions", command)
        self.assertIn("--checkpoint-score-mode mean_reward", command)
        self.assertIn(
            "--code-revision " + spec.FROZEN_ALGORITHM_REVISION,
            command,
        )
        strengths = spec.expected_router_training_strengths(
            "router_a004_s010_linear_w0125_r0375"
        )
        self.assertEqual(len(strengths), spec.ITERATIONS)
        self.assertEqual(strengths[0], 0.0)
        self.assertAlmostEqual(strengths[-1], 0.1)
        self.assertTrue(all(
            value == 0.1
            for value in spec.expected_router_training_strengths(
                "router_a004_s010_constant"
            )
        ))

    def test_development_seeds_do_not_reuse_v12_to_v143_roles(self):
        old = set()
        for module in (v12, v13, v14, v14_1, v14_2, v14_3):
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
            "causal_transfer_direct_baseline",
            "router_a004_s010_constant",
            "router_a004_s010_linear_w0125_r0375",
            "router_a004_s015_cosine_w025_r050",
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
            # return, responsibility, raw, latent, action RMS, clip, upper power
            "causal_transfer_direct_baseline": (
                100.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0025
            ),
            "router_a004_s010_constant": (
                99.5, 0.98, 0.88, 0.94, 0.94, 0.0, 0.0025
            ),
            "router_a004_s010_linear_w0125_r0375": (
                99.5, 0.95, 0.80, 0.90, 0.90, 0.0, 0.0025
            ),
            "router_a004_s015_cosine_w025_r050": (
                95.0, 0.90, 0.70, 0.85, 0.85, 0.0, 0.0025
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "results" / "unit_v14_4"
            merged = run / "merged"
            merged.mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_4_development_outcome_access",
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
                    strength = float(
                        spec.ARMS[arm]["lower_action_router_strength"]
                    )
                    for optimizer_seed in spec.OPTIMIZER_SEEDS:
                        cell = (
                            run / "cells" / environment / arm
                            / f"replicate_{optimizer_seed}"
                        )
                        cell.mkdir(parents=True)
                        (cell / "cell_summary.json").write_text(json.dumps({
                            "selected_checkpoint_iteration": 8,
                        }) + "\n", encoding="utf-8")
                        (cell / "training_history.json").write_text(json.dumps([
                            {
                                "upper_constraint_lambda": 0.0,
                                "constraint_lambda": 0.0,
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
                                "LowerActionRouterStrength",
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
                                        "LowerActionRouterStrength": strength,
                                        "UpperHFPowerAbs": values[6],
                                        "ResponsibilityReconstructionRMS": 0.0,
                                    })
            decision = analyze(run)
        self.assertEqual(decision["status"], "development_candidate_selected")
        self.assertEqual(
            decision["selected_arm"],
            "router_a004_s010_linear_w0125_r0375",
        )
        self.assertEqual(
            decision["gate_granularity"],
            "environment_by_disturbance_mode",
        )
        self.assertEqual(
            decision["arm_status"][
                "router_a004_s010_linear_w0125_r0375"
            ][
                "passed_gate_count"
            ],
            4,
        )
        self.assertTrue(all(
            row["lower_action_router_strength"] == 0.10
            for row in decision["environment_condition_rows"]
            if row["arm"] == "router_a004_s010_linear_w0125_r0375"
        ))
        self.assertTrue(all(
            row["responsibility_drift_gate_type"]
            in ("relative_noninferiority", "absolute_floor_noninferiority")
            for row in decision["environment_condition_rows"]
            if row["arm"] == "router_a004_s010_linear_w0125_r0375"
        ))
        self.assertTrue(all(
            not row["reward_noninferiority_pass"]
            for row in decision["environment_condition_rows"]
            if row["arm"] == "router_a004_s015_cosine_w025_r050"
        ))
        at_floor_rows = [
            row for row in decision["environment_condition_rows"]
            if row["arm"] == "router_a004_s010_linear_w0125_r0375"
            and row["environment"] == "Hopper-v5"
            and row["disturbance_mode"] == "standard"
        ]
        self.assertEqual(len(at_floor_rows), 1)
        self.assertEqual(
            at_floor_rows[0]["raw_lower_drift_gate_type"],
            "absolute_floor_noninferiority",
        )
        self.assertTrue(at_floor_rows[0]["raw_lower_drift_pass"])

    def test_analyzer_rejects_router_strength_mismatch(self):
        rows = [
            {"LowerActionRouterStrength": "0.10"},
            {"LowerActionRouterStrength": "0.15"},
        ]
        with self.assertRaisesRegex(ValueError, "strength mismatch"):
            _validate_router_strength(
                rows,
                expected=0.10,
                cell_label="HalfCheetah-v5/router/1",
            )


if __name__ == "__main__":
    unittest.main()
