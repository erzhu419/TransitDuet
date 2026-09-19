import unittest

from scripts import pointmaze_multiscale_stage3_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_multiscale_stage3_scheduleurm import (
    _inventory_by_signature,
    task_specification,
    training_command,
)


class PointMazeMultiscaleStageThreeSchedulerTest(unittest.TestCase):
    def test_formal_factorial_and_seed_roles_are_fresh_and_paired(self):
        self.assertEqual(len(spec.cells(preflight=False)), 160)
        self.assertEqual(len(spec.cells(preflight=True)), 20)
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
        self.assertIn("--history-seconds 0.32", command)
        self.assertIn("complete: result.json written", command)

    def test_protocol_freezes_the_registered_factorial(self):
        self.assertEqual(spec.PROTOCOL, "pointmaze_multiscale_goal_stage3_v2")
        self.assertEqual(
            spec.METHODS,
            (
                "flat_history",
                "flat_multiscale",
                "hrl_history",
                "hrl_multiscale",
                "flat_causal_filter",
            ),
        )
        self.assertEqual(
            spec.SCENARIOS,
            (
                "clean",
                "fast_observation_noise",
                "slow_drift_fast_action",
                "persistent_action_shift",
            ),
        )
        self.assertEqual(spec.CHECKPOINT_EVALUATION_INTERVAL, 96)
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)
        self.assertEqual(spec.RUNTIME_EXPECTATIONS["mujoco"], "3.2.7")

    def test_result_sync_prefers_successful_retry_over_failed_attempt(self):
        signature = "Freq-HRL/test/cell"
        selected = _inventory_by_signature([
            {
                "id": "t-retry",
                "signature": signature,
                "status": "done",
                "node": "node003",
            },
            {
                "id": "t-original",
                "signature": signature,
                "status": "failed",
                "node": "node006",
            },
        ])
        self.assertEqual(selected[signature]["id"], "t-retry")


if __name__ == "__main__":
    unittest.main()
