import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from freq_hrl.rl.smdp_actor_critic import (
    HierarchicalTrajectoryBatch,
    LevelTrajectoryBatch,
)
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as v14_25
from scripts import mujoco_v14_26_robust_paired_fd_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    _paired_finite_difference_estimate,
)
from scripts.submit_mujoco_v14_26_robust_paired_fd_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _level_batch(cost: float) -> LevelTrajectoryBatch:
    return LevelTrajectoryBatch(
        state=np.zeros((1, 1), dtype=np.float32),
        action=np.zeros((1, 2), dtype=np.float32),
        reward=np.zeros(1, dtype=np.float32),
        duration=np.ones(1, dtype=np.int64),
        done=np.ones(1, dtype=np.float32),
        old_logp=np.zeros(1, dtype=np.float32),
        old_value=np.zeros(1, dtype=np.float32),
        cost=np.asarray([cost], dtype=np.float32),
    )


def _hierarchical_batch(upper_cost: float, lower_cost: float):
    return HierarchicalTrajectoryBatch(
        upper=_level_batch(upper_cost),
        lower=_level_batch(lower_cost),
    )


class MujocoV1426RobustPairedFdPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_26_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_paired_estimator_recovers_balanced_linear_direction(self):
        gradient = np.asarray([2.0, 1.0], dtype=np.float64)
        amplitude = 0.25
        directions = (
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, -1.0]),
        )
        results = []
        jobs = []
        modes = ("stationary_low", "stationary_high", "burst", "shift")
        for mode_index, mode in enumerate(modes):
            for direction_index, direction in enumerate(directions):
                seed = 10 * mode_index + direction_index
                upper_bias = amplitude * direction
                lower_bias = amplitude * direction
                variants = (
                    ("control", None, None),
                    ("upper_plus", upper_bias, None),
                    ("upper_minus", -upper_bias, None),
                    ("lower_plus", None, lower_bias),
                    ("lower_minus", None, -lower_bias),
                )
                for variant, upper, lower in variants:
                    upper_cost = 10.0 + (
                        0.0 if upper is None else float(np.dot(gradient, upper))
                    )
                    lower_cost = 10.0 + (
                        0.0 if lower is None else float(np.dot(gradient, lower))
                    )
                    jobs.append({
                        "path": {"seed": seed, "disturbance_mode": mode},
                        "intervention": {
                            "variant": variant,
                            "upper_bias": upper,
                            "lower_bias": lower,
                        },
                    })
                    results.append({
                        "intervention_variant": variant,
                        "batch": _hierarchical_batch(upper_cost, lower_cost),
                    })
        expected = gradient / np.sqrt(np.mean(np.square(gradient)))
        for level in ("upper", "lower"):
            estimated, per_mode, metrics = _paired_finite_difference_estimate(
                results,
                jobs,
                level=level,
                gamma=0.99,
                max_return_decisions=None,
            )
            np.testing.assert_allclose(estimated, expected, atol=1e-6)
            for mode in modes:
                np.testing.assert_allclose(per_mode[mode], gradient, atol=1e-6)
            self.assertEqual(metrics["path_count"], 8)
            self.assertEqual(metrics["parameter_count"], 2)

    def test_frozen_roots_are_fresh_and_holdout_is_powered(self):
        roles = (
            spec.CRITIC_TRAIN_ROOTS,
            spec.CRITIC_HOLDOUT_ROOTS,
            spec.DESIGN_ROOTS,
            spec.VALIDATION_ROOTS,
        )
        flattened = [root for role in roles for root in role]
        self.assertEqual(len(flattened), 48)
        self.assertEqual(len(set(flattened)), 48)
        previous = set(
            v14_25.CRITIC_TRAIN_ROOTS + v14_25.CRITIC_HOLDOUT_ROOTS
            + v14_25.DESIGN_ROOTS + v14_25.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))
        self.assertEqual(spec.EXPECTED_PAIRED_HOLDOUT_DIRECTION_COUNT, 32)
        self.assertEqual(spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT, 160)

    def test_launcher_freezes_paired_direction_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn(
            "--actor-direction-source paired_finite_difference", command
        )
        self.assertIn("--minimum-paired-holdout-cosine 0.0", command)
        self.assertIn("--actor-update-scope output_bias", command)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_26_robust_paired_fd_preflight_scheduleurm.py"
        )
        process = subprocess.run(
            [sys.executable, str(launcher), "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--run-name", process.stdout)


if __name__ == "__main__":
    unittest.main()
