import unittest

from scripts import pointmaze_goal_stage2_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_goal_stage2_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeGoalStageTwoSchedulerTest(unittest.TestCase):
    def test_formal_matrix_and_seed_roles_are_independent(self):
        self.assertEqual(len(spec.cells(preflight=False)), 16)
        self.assertEqual(len(spec.cells(preflight=True)), 2)
        observed = set()
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            roles = spec.seed_roles(optimizer_seed)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 16)
            self.assertEqual(len(set(values)), 16)
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_task_is_dynamic_single_core_and_explicitly_completes(self):
        cell = spec.cells(preflight=True)[0]
        task = task_specification("unit_preflight", cell, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 2048)
        self.assertTrue(task["allow_no_ckpt"])
        command = training_command("unit_preflight", cell, preflight=True)
        self.assertIn("--iterations 2", command)
        self.assertIn("--horizon 64", command)
        self.assertIn("complete: result.json written", command)

    def test_frozen_revision_is_full_sha(self):
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)


if __name__ == "__main__":
    unittest.main()
