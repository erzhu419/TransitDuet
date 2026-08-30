import unittest
from argparse import Namespace

from scripts import mujoco_v14_21_distributional_actor_preflight_spec as spec
from scripts.submit_mujoco_v14_21_distributional_actor_preflight_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


class MujocoV1421DistributionalActorPreflightTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_21_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=[
                "node001", "node002", "node003", "node004", "node005",
                "node006",
            ],
        )

    def test_frozen_roots_are_fresh_disjoint_and_crossed(self):
        self.assertEqual(len(spec.DESIGN_ROOTS), 16)
        self.assertEqual(len(spec.VALIDATION_ROOTS), 16)
        self.assertFalse(set(spec.DESIGN_ROOTS) & set(spec.VALIDATION_ROOTS))
        self.assertEqual(spec.EXPECTED_PATH_COUNT, 64)

    def test_launcher_declares_parallel_distributional_contract(self):
        self.assertEqual(len(selected_cells()), 3)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--workers 16", command)
        self.assertIn("--risk-mode mode_mean", command)
        self.assertIn(f"--probe-version {spec.PROBE_VERSION}", command)
        self.assertTrue(command.endswith("&& echo DONE"))
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], spec.CPU_PER_TASK)
        self.assertEqual(scheduler["ram_mb"], spec.RAM_MB_PER_TASK)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))


if __name__ == "__main__":
    unittest.main()
