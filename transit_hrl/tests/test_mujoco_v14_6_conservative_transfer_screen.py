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
from scripts import mujoco_v14_4_router_homotopy_screen_spec as v14_4
from scripts import mujoco_v14_5_paired_anchor_screen_spec as v14_5
from scripts import mujoco_v14_6_conservative_transfer_screen_spec as spec
from scripts.analyze_mujoco_v14_6_conservative_transfer_screen import analyze
from scripts.submit_mujoco_v14_6_conservative_transfer_screen_scheduleurm import (
    ANCHOR_ARM,
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV146ConservativeTransferScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="unit_v14_6_screen",
            python_executable="/compute/python",
            priority="normal",
            nodes=[f"node00{index}" for index in range(1, 7)],
        )

    def test_frozen_registry_has_48_anchors_and_384_continuations(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 432)
        self.assertEqual(len(set(cells)), 432)
        self.assertEqual(sum(cell[0] == "anchor" for cell in cells), 48)
        self.assertEqual(
            sum(cell[0] == "continuation" for cell in cells), 384
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
            f"--iterations {spec.PRETRAIN_ITERATIONS}", anchor_command
        )
        self.assertIn(
            "--lower-action-router-mode causal_ema_conservative_transfer",
            anchor_command,
        )
        self.assertNotIn(
            "--lower-action-router-observe-strength", anchor_command
        )
        self.assertIn(
            "--checkpoint-minimum-iteration "
            f"{spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION}",
            anchor_command,
        )
        candidate = "conservative_s0100"
        continuation_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=candidate,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/candidate"),
        )
        self.assertIn("--initial-checkpoint-path", continuation_command)
        self.assertIn(
            "--initial-checkpoint-router-mode "
            "causal_ema_conservative_transfer",
            continuation_command,
        )
        self.assertIn("--upper-actor-anchor-coef 0.0", continuation_command)
        self.assertIn("--lower-actor-anchor-coef 0.0", continuation_command)
        self.assertIn(
            "--checkpoint-minimum-iteration "
            f"{spec.CONTINUATION_CHECKPOINT_MINIMUM_ITERATION}",
            continuation_command,
        )
        self.assertIn(
            f"--iterations {spec.CONTINUATION_ITERATIONS}",
            continuation_command,
        )
        scheduler = build_scheduler_spec(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=candidate,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 1)
        self.assertEqual(scheduler["allowed_nodes"], args.nodes)
        self.assertEqual(len(scheduler["wait_for_files"]), 2)
        self.assertTrue(all(
            "anchors/HalfCheetah-v5" in path
            for path in scheduler["wait_for_files"]
        ))

    def test_all_v146_seed_roles_are_fresh_and_disjoint(self):
        old = set()
        for module in (
            v12, v13, v14, v14_1, v14_2, v14_3, v14_4, v14_5
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

    def test_analyzer_requires_exact_function_preserving_continuation(self):
        comparator = spec.COMPARATOR_ARM
        candidate = "conservative_s0100"
        patchers = (
            patch.object(spec, "ENVIRONMENTS", ("HalfCheetah-v5",)),
            patch.object(spec, "EVALUATION_DISTURBANCE_MODES", (
                "standard", "ood_chirp"
            )),
            patch.object(spec, "OPTIMIZER_SEEDS", spec.OPTIMIZER_SEEDS[:2]),
            patch.object(
                spec,
                "DEVELOPMENT_EVALUATION_SEEDS",
                spec.DEVELOPMENT_EVALUATION_SEEDS[:2],
            ),
            patch.object(spec, "ARMS", {
                comparator: spec.ARMS[comparator],
                candidate: spec.ARMS[candidate],
            }),
            patch.object(spec, "CANDIDATE_ARMS", (candidate,)),
            patch.object(spec, "BOOTSTRAP_DRAWS", 200),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            (run / "merged").mkdir(parents=True)
            (run / "preregistration.json").write_text(json.dumps({
                "status": "frozen_before_v14_6_development_outcome_access",
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

            for seed in spec.OPTIMIZER_SEEDS:
                anchor = (
                    run / "anchors" / "HalfCheetah-v5"
                    / f"replicate_{seed}"
                )
                anchor.mkdir(parents=True)
                for name in (
                    "cell_summary.json",
                    "training_history.json",
                    "evaluation_rows.csv",
                ):
                    (anchor / name).write_text("{}", encoding="utf-8")
            for arm in spec.ARMS:
                values = (
                    (100.0, 1.0, 1.0, 1.0)
                    if arm == comparator else (100.0, 0.8, 0.7, 1.0)
                )
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
                        "frozen_parameter_sha256": "f" * 64,
                    }), encoding="utf-8")
                    (cell / "training_history.json").write_text(json.dumps([
                        {
                            "upper_actor_anchor_kl": 0.001,
                            "lower_actor_anchor_kl": 0.002,
                        }
                    ]), encoding="utf-8")
                    with (cell / "evaluation_rows.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        fields = [
                            "disturbance_mode", "seed", "episode_return",
                            "LowerLFDriftAbs", "RawLowerLFDriftAbs",
                            "LatentLowerLFDriftAbs", "RawLowerActionRMS",
                            "LowerRouterClipRate", "LowerActionRouterStrength",
                            "LowerRouterUpperTransferRMS",
                            "LowerRouterFunctionPreserving",
                            "LowerRouterActionReconstructionRMS",
                            "UpperHFPowerAbs", "ResponsibilityReconstructionRMS",
                            "RewardTraceSHA256",
                            "ExecutedActionTraceSHA256",
                            "LatentPolicyTraceSHA256",
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
                                    "LatentLowerLFDriftAbs": values[3],
                                    "RawLowerActionRMS": 1.0,
                                    "LowerRouterClipRate": 0.0,
                                    "LowerRouterUpperTransferRMS": (
                                        0.0 if arm == comparator else 0.1
                                    ),
                                    "LowerRouterFunctionPreserving": 1.0,
                                    "LowerRouterActionReconstructionRMS": 0.0,
                                    "LowerActionRouterStrength": (
                                        spec.ARMS[arm][
                                            "lower_action_router_strength"
                                        ]
                                    ),
                                    "UpperHFPowerAbs": 0.0025,
                                    "ResponsibilityReconstructionRMS": 0.0,
                                    "RewardTraceSHA256": "a" * 64,
                                    "ExecutedActionTraceSHA256": "b" * 64,
                                    "LatentPolicyTraceSHA256": "c" * 64,
                                })
            decision = analyze(run)
            self.assertEqual(
                decision["status"], "development_candidate_selected"
            )
            self.assertEqual(decision["selected_arm"], candidate)
            self.assertEqual(
                decision["arm_status"][candidate]["passed_gate_count"], 2
            )
            self.assertTrue(decision["arm_status"][candidate][
                "exact_selected_parameter_hash_gate_pass"
            ])
            self.assertTrue(all(
                row["exact_trace_pass"] and row["exact_return_pass"]
                for row in decision["environment_condition_rows"]
            ))

            corrupted_path = (
                run / "cells" / "HalfCheetah-v5" / candidate
                / f"replicate_{spec.OPTIMIZER_SEEDS[0]}"
                / "evaluation_rows.csv"
            )
            with corrupted_path.open(
                newline="", encoding="utf-8"
            ) as handle:
                corrupted_rows = list(csv.DictReader(handle))
            corrupted_rows[0]["RewardTraceSHA256"] = "d" * 64
            with corrupted_path.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(corrupted_rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(corrupted_rows)
            corrupted = analyze(run)
            self.assertEqual(
                corrupted["status"], "no_behavior_safe_candidate"
            )
            self.assertFalse(any(
                row["exact_trace_pass"]
                for row in corrupted["environment_condition_rows"]
                if row["disturbance_mode"] == "standard"
            ))


if __name__ == "__main__":
    unittest.main()
