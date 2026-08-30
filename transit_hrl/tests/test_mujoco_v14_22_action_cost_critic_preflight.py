import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    actor_mean_parameter_vector,
    apply_actor_mean_parameter_delta,
)
from scripts.submit_mujoco_v14_22_action_cost_critic_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


class _ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(3, 4)
        self.head = torch.nn.Linear(4, 2)
        self.log_std = torch.nn.Parameter(torch.tensor([-1.0, -2.0]))


class _ToyModel:
    def __init__(self):
        self.upper_actor = _ToyActor()
        self.lower_actor = _ToyActor()


class MujocoV1422ActionCostCriticPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_22_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_actor_mean_delta_alignment_excludes_log_standard_deviation(self):
        torch.manual_seed(7)
        model = _ToyModel()
        before = actor_mean_parameter_vector(model)
        upper_log_std = model.upper_actor.log_std.detach().clone()
        lower_log_std = model.lower_actor.log_std.detach().clone()
        delta = np.linspace(-1e-3, 1e-3, before.size)
        apply_actor_mean_parameter_delta(model, delta)
        np.testing.assert_allclose(
            actor_mean_parameter_vector(model), before + delta,
            rtol=2e-5, atol=2e-8,
        )
        torch.testing.assert_close(model.upper_actor.log_std, upper_log_std)
        torch.testing.assert_close(model.lower_actor.log_std, lower_log_std)

    def test_frozen_root_roles_are_fresh_disjoint_and_crossed(self):
        roles = (
            spec.CRITIC_TRAIN_ROOTS,
            spec.CRITIC_HOLDOUT_ROOTS,
            spec.DESIGN_ROOTS,
            spec.VALIDATION_ROOTS,
        )
        flattened = [root for role in roles for root in role]
        self.assertEqual(len(flattened), 48)
        self.assertEqual(len(set(flattened)), 48)
        old = set(
            v14_20.DESIGN_ROOTS + v14_20.VALIDATION_ROOTS
            + v14_21.DESIGN_ROOTS + v14_21.VALIDATION_ROOTS
        )
        self.assertFalse(old & set(flattened))
        self.assertEqual(spec.EXPECTED_CRITIC_TRAIN_PATH_COUNT, 48)
        self.assertEqual(spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT, 16)
        self.assertEqual(spec.EXPECTED_DESIGN_PATH_COUNT, 64)
        self.assertEqual(spec.EXPECTED_VALIDATION_PATH_COUNT, 64)

    def test_launcher_declares_action_critic_and_dynamic_scheduler_contract(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--workers 24", command)
        self.assertIn("--critic-minimum-action-permutation-mse-increase 0.0", command)
        self.assertIn("--risk-mode mode_mean", command)
        self.assertTrue(command.endswith("&& echo DONE"))
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], spec.CPU_PER_TASK)
        self.assertEqual(scheduler["ram_mb"], spec.RAM_MB_PER_TASK)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_22_action_cost_critic_preflight_scheduleurm.py"
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
