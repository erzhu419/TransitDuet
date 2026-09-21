import unittest

from scripts import pointmaze_exogenous_routing_stage6_spec as spec
from scripts import pointmaze_exogenous_stage5_v2_spec as stage5
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_exogenous_stage5_scheduleurm import (
    _experiment_protocol,
    _preflight_optimizer_seeds,
    task_specification,
    training_command,
)


class PointMazeExogenousRoutingStageSixSchedulerTest(unittest.TestCase):
    def test_matrix_and_all_seed_roles_are_fresh_and_paired(self):
        self.assertEqual(len(spec.METHODS), 8)
        self.assertEqual(len(spec.cells(preflight=False)), 64)
        self.assertEqual(len(spec.cells(preflight=True)), 8)
        self.assertEqual(_preflight_optimizer_seeds(spec), (184001,))
        old = {
            seed
            for root in (*stage5.PREFLIGHT_OPTIMIZER_SEEDS, *stage5.OPTIMIZER_SEEDS)
            for values in stage5.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for root in (*spec.PREFLIGHT_OPTIMIZER_SEEDS, *spec.OPTIMIZER_SEEDS):
            roles = spec.seed_roles(root)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 40)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(old.intersection(values))
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_tasks_are_dynamic_and_use_the_stage_six_runner(self):
        for method in spec.METHODS:
            cell = (method, spec.PREFLIGHT_OPTIMIZER_SEEDS[0])
            command = training_command(
                "unit_stage6", cell, preflight=True, protocol_spec=spec
            )
            self.assertIn(spec.RUNNER_SCRIPT, command)
            self.assertIn(f"--methods {method}", command)
            self.assertIn("--iterations 2", command)
            self.assertIn("--horizon 64", command)
            task = task_specification(
                "unit_stage6", cell, preflight=True, protocol_spec=spec
            )
            self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["cpu"], 1)
            self.assertEqual(task["ram_mb"], 2560)
            self.assertIn("stage6", task["description"])

    def test_protocol_freezes_strict_routing_claim_before_results(self):
        self.assertEqual(
            spec.PROTOCOL,
            "pointmaze_exogenous_frequency_routing_stage6_v1",
        )
        self.assertEqual(_experiment_protocol(spec), spec.EXPERIMENT_PROTOCOL)
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)
        self.assertEqual(
            spec.CLAIM_GATE["routed_vs_all_band_success"], "positive_ci"
        )
        self.assertEqual(
            spec.CLAIM_GATE["routed_vs_swapped_success"], "positive_ci"
        )
        self.assertEqual(
            spec.CLAIM_GATE["hierarchy_x_multiscale_success_interaction"],
            "positive_ci",
        )


if __name__ == "__main__":
    unittest.main()
