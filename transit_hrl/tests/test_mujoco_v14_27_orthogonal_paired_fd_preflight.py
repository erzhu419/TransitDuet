import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import mujoco_v14_26_robust_paired_fd_preflight_spec as v14_26
from scripts import mujoco_v14_27_orthogonal_paired_fd_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    _paired_finite_difference_estimate,
    balanced_hadamard_direction,
    paired_output_bias_interventions,
)
from scripts.submit_mujoco_v14_27_orthogonal_paired_fd_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)
from tests.test_mujoco_v14_26_robust_paired_fd_preflight import (
    _hierarchical_batch,
)


class _ToyActor(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, output_dim),
        )


class _ToyModel:
    def __init__(self):
        self.upper_actor = _ToyActor(3)
        self.lower_actor = _ToyActor(2)


class MujocoV1427OrthogonalPairedFdPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_27_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_hadamard_rows_are_balanced_and_orthogonal(self):
        matrix = np.stack([
            balanced_hadamard_direction(8, index=index, order=8)
            for index in range(8)
        ])
        np.testing.assert_array_equal(matrix @ matrix.T, 8.0 * np.eye(8))
        interventions = paired_output_bias_interventions(
            _ToyModel(),
            seed=7,
            bias_rms=0.25,
            direction_scheme="balanced_hadamard",
            direction_index=3,
            hadamard_order=8,
        )
        for item in interventions:
            for level in ("upper", "lower"):
                bias = item[f"{level}_bias"]
                if bias is not None:
                    self.assertAlmostEqual(
                        float(np.sqrt(np.mean(np.square(bias)))), 0.25
                    )

    def test_orthogonal_solver_recovers_linear_cost_gradient(self):
        gradient = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)
        amplitude = 0.25
        results = []
        jobs = []
        modes = ("stationary_low", "stationary_high", "burst", "shift")
        for mode_index, mode in enumerate(modes):
            for root_index in range(16):
                direction = balanced_hadamard_direction(
                    3, index=root_index % 8, order=8
                )
                bias = amplitude * direction
                root_offset = 0.01 * (root_index + mode_index)
                for variant, upper in (
                    ("control", None),
                    ("upper_plus", bias),
                    ("upper_minus", -bias),
                ):
                    cost = 10.0 + root_offset + (
                        0.0 if upper is None else float(np.dot(gradient, upper))
                    )
                    jobs.append({
                        "path": {
                            "seed": 100 * mode_index + root_index,
                            "disturbance_mode": mode,
                        },
                        "intervention": {
                            "variant": variant,
                            "upper_bias": upper,
                            "lower_bias": None,
                        },
                    })
                    results.append({
                        "intervention_variant": variant,
                        "batch": _hierarchical_batch(cost, 10.0),
                    })
        estimated, per_mode, metrics = _paired_finite_difference_estimate(
            results,
            jobs,
            level="upper",
            gamma=0.99,
            max_return_decisions=None,
            estimator="orthogonal_least_squares",
        )
        expected = gradient / np.sqrt(np.mean(np.square(gradient)))
        np.testing.assert_allclose(estimated, expected, atol=2e-6)
        for mode in modes:
            np.testing.assert_allclose(per_mode[mode], gradient, atol=2e-6)
            self.assertEqual(metrics["per_mode_design_rank"][mode], 3)
        self.assertEqual(metrics["global_design_rank"], 3)
        self.assertEqual(metrics["path_count"], 64)

    def test_frozen_roots_are_fresh_and_hadamard_rows_repeat_twice(self):
        roles = (
            spec.CRITIC_TRAIN_ROOTS,
            spec.CRITIC_HOLDOUT_ROOTS,
            spec.DESIGN_ROOTS,
            spec.VALIDATION_ROOTS,
        )
        flattened = [root for role in roles for root in role]
        self.assertEqual(len(flattened), 64)
        self.assertEqual(len(set(flattened)), 64)
        previous = set(
            v14_26.CRITIC_TRAIN_ROOTS + v14_26.CRITIC_HOLDOUT_ROOTS
            + v14_26.DESIGN_ROOTS + v14_26.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))
        self.assertEqual(
            len(spec.CRITIC_TRAIN_ROOTS)
            // spec.CRITIC_INTERVENTION_HADAMARD_ORDER,
            2,
        )
        self.assertEqual(spec.EXPECTED_CRITIC_TRAIN_PATH_COUNT, 320)
        self.assertEqual(spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT, 320)

    def test_launcher_freezes_orthogonal_design_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn(
            "--critic-intervention-direction-scheme balanced_hadamard", command
        )
        self.assertIn("--critic-intervention-hadamard-order 8", command)
        self.assertIn(
            "--paired-direction-estimator orthogonal_least_squares", command
        )
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_27_orthogonal_paired_fd_preflight_scheduleurm.py"
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
