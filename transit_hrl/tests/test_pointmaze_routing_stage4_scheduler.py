import unittest

from scripts import pointmaze_routing_stage4_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_routing_stage4_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeRoutingStageFourSchedulerTest(unittest.TestCase):
    def test_formal_matrix_and_seed_roles_are_fresh_and_paired(self):
        self.assertEqual(len(spec.cells(preflight=False)), 120)
        self.assertEqual(len(spec.cells(preflight=True)), 15)
        observed = set()
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            roles = spec.seed_roles(optimizer_seed)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 36)
            self.assertEqual(len(set(values)), 36)
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_tasks_are_dynamic_single_core_and_only_sync_json(self):
        cell = spec.cells(preflight=True)[0]
        task = task_specification("unit_preflight", cell, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 2560)
        self.assertTrue(task["allow_no_ckpt"])
        command = training_command("unit_preflight", cell, preflight=True)
        self.assertIn("--iterations 2", command)
        self.assertIn("--horizon 64", command)
        self.assertIn("complete: result.json written", command)

    def test_protocol_freezes_the_routing_controls(self):
        self.assertEqual(spec.PROTOCOL, "pointmaze_frequency_routing_stage4_v1")
        self.assertEqual(
            spec.METHODS,
            (
                "hrl_history",
                "hrl_causal_filter",
                "hrl_multiscale_all",
                "hrl_multiscale_routed",
                "hrl_multiscale_swapped",
            ),
        )
        self.assertEqual(
            spec.SCENARIOS,
            ("clean", "fast_observation_noise", "slow_drift_fast_action"),
        )
        self.assertEqual(spec.CHECKPOINT_EVALUATION_INTERVAL, 96)
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)
        self.assertEqual(spec.RUNTIME_EXPECTATIONS["mujoco"], "3.2.7")


if __name__ == "__main__":
    unittest.main()
