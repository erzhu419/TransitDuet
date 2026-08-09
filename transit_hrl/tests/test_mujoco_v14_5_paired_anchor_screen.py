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
from scripts import mujoco_v14_5_paired_anchor_screen_spec as spec
from scripts.analyze_mujoco_v14_5_paired_anchor_screen import analyze
from scripts.submit_mujoco_v14_5_paired_anchor_screen_scheduleurm import (
    ANCHOR_ARM,
    build_scheduler_spec,
    build_training_command,
    experiment_cells,
)


class MujocoV145PairedAnchorScreenTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            run_name="unit_v14_5_screen",
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
        candidate = "router_s010_ua005_la010"
        continuation_command = build_training_command(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=candidate,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit/candidate"),
        )
        self.assertIn("--initial-checkpoint-path", continuation_command)
        self.assertIn("--upper-actor-anchor-coef 0.05", continuation_command)
        self.assertIn("--lower-actor-anchor-coef 0.1", continuation_command)
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

    def test_all_v145_seed_roles_are_fresh_and_disjoint(self):
        old = set()
        for module in (v12, v13, v14, v14_1, v14_2, v14_3, v14_4):
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

    def test_analyzer_uses_compute_matched_direct_continuation(self):
        comparator = spec.COMPARATOR_ARM
        candidate = "router_s010_ua005_la010"
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
                "status": "frozen_before_v14_5_development_outcome_access",
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
                    if arm == comparator else (100.5, 0.8, 0.7, 0.9)
                )
                for seed in spec.OPTIMIZER_SEEDS:
                    cell = (
                        run / "cells" / "HalfCheetah-v5" / arm
                        / f"replicate_{seed}"
                    )
                    cell.mkdir(parents=True)
                    (cell / "cell_summary.json").write_text(json.dumps({
                        "selected_checkpoint_iteration": 4,
                        "upper_actor_anchor_parameter_rms": 0.01,
                        "lower_actor_anchor_parameter_rms": 0.02,
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
                            "UpperHFPowerAbs", "ResponsibilityReconstructionRMS",
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
                                    "LowerActionRouterStrength": (
                                        spec.ARMS[arm][
                                            "lower_action_router_strength"
                                        ]
                                    ),
                                    "UpperHFPowerAbs": 0.0025,
                                    "ResponsibilityReconstructionRMS": 0.0,
                                })
            decision = analyze(run)
            self.assertEqual(
                decision["status"], "development_candidate_selected"
            )
            self.assertEqual(decision["selected_arm"], candidate)
            self.assertEqual(
                decision["arm_status"][candidate]["passed_gate_count"], 2
            )


if __name__ == "__main__":
    unittest.main()
