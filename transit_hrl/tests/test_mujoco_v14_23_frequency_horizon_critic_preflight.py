import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as spec
from scripts.submit_mujoco_v14_23_frequency_horizon_critic_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


class MujocoV1423FrequencyHorizonCriticPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_23_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_frozen_frequency_horizons_and_fresh_root_roles(self):
        self.assertEqual(spec.UPPER_COST_RETURN_HORIZON_DECISIONS, 8)
        self.assertEqual(spec.LOWER_COST_RETURN_HORIZON_DECISIONS, 32)
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
            v14_22.CRITIC_TRAIN_ROOTS + v14_22.CRITIC_HOLDOUT_ROOTS
            + v14_22.DESIGN_ROOTS + v14_22.VALIDATION_ROOTS
        )
        self.assertFalse(previous & set(flattened))

    def test_launcher_freezes_horizons_and_dynamic_resources(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--upper-cost-return-horizon-decisions 8", command)
        self.assertIn("--lower-cost-return-horizon-decisions 32", command)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        self.assertTrue(command.endswith("&& echo DONE"))
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 24)
        self.assertEqual(scheduler["ram_mb"], 16384)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_launcher_is_directly_executable(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "scripts"
            / "submit_mujoco_v14_23_frequency_horizon_critic_preflight_scheduleurm.py"
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
