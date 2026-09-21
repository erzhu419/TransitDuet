import unittest

from scripts import pointmaze_exogenous_stage5_spec as spec
from scripts import pointmaze_routing_stage4_v2_spec as stage4
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_exogenous_stage5_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeExogenousStageFiveSchedulerTest(unittest.TestCase):
    def test_matrix_and_seed_roles_are_fresh(self):
        self.assertEqual(len(spec.cells(preflight=False)), 16)
        self.assertEqual(len(spec.cells(preflight=True)), 2)
        old_seeds = {
            seed
            for root in stage4.OPTIMIZER_SEEDS
            for values in stage4.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for root in spec.OPTIMIZER_SEEDS:
            roles = spec.seed_roles(root)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 36)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(observed.intersection(values))
            self.assertFalse(old_seeds.intersection(values))
            observed.update(values)

    def test_tasks_are_dynamic_single_core_and_use_frozen_runner(self):
        cell = spec.cells(preflight=True)[0]
        task = task_specification("unit_stage5_preflight", cell, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 2560)
        command = training_command(
            "unit_stage5_preflight", cell, preflight=True
        )
        self.assertIn("run_pointmaze_exogenous_stage5.py", command)
        self.assertIn("--iterations 2", command)
        self.assertIn("--horizon 64", command)
        self.assertIn("--force-period-seconds 0.04 0.04", command)

    def test_protocol_freezes_external_stream_and_gate(self):
        self.assertEqual(
            spec.PROTOCOL, "pointmaze_exogenous_control_stage5_v1"
        )
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)
        self.assertEqual(spec.TARGET_SPEED, 1.0)
        self.assertEqual(spec.FORCE_RMS, 0.12)
        self.assertEqual(spec.FORCE_PERIOD_SECONDS, (0.04, 0.04))
        self.assertEqual(spec.HISTORY_SECONDS, 0.32)
        self.assertEqual(spec.UPPER_PERIOD_SECONDS, 0.25)


if __name__ == "__main__":
    unittest.main()

