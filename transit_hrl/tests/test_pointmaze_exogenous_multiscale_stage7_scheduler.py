import unittest

from scripts import pointmaze_exogenous_multiscale_stage7_spec as spec
from scripts import pointmaze_exogenous_routing_stage6_spec as stage6
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_exogenous_stage5_scheduleurm import (
    _experiment_protocol,
    _preflight_optimizer_seeds,
    task_specification,
    training_command,
)


class PointMazeExogenousMultiscaleStageSevenSchedulerTest(unittest.TestCase):
    def test_fixed_matrix_and_seed_roles_are_fresh(self):
        self.assertEqual(len(spec.METHODS), 4)
        self.assertEqual(len(spec.OPTIMIZER_SEEDS), 16)
        self.assertEqual(len(spec.cells(preflight=False)), 64)
        self.assertEqual(len(spec.cells(preflight=True)), 4)
        self.assertEqual(_preflight_optimizer_seeds(spec), (194001,))
        old = {
            seed
            for root in (*stage6.PREFLIGHT_OPTIMIZER_SEEDS, *stage6.OPTIMIZER_SEEDS)
            for values in stage6.seed_roles(root).values()
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

    def test_tasks_use_frozen_runner_and_dynamic_cpu_pool(self):
        for method in spec.METHODS:
            cell = (method, spec.PREFLIGHT_OPTIMIZER_SEEDS[0])
            command = training_command(
                "unit_stage7", cell, preflight=True, protocol_spec=spec
            )
            self.assertIn(spec.RUNNER_SCRIPT, command)
            self.assertIn(f"--methods {method}", command)
            task = task_specification(
                "unit_stage7", cell, preflight=True, protocol_spec=spec
            )
            self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["cpu"], 1)
            self.assertEqual(task["ram_mb"], 2560)
            self.assertIn("stage7", task["description"])

    def test_confirmation_gate_is_frozen_without_sequential_extension(self):
        self.assertEqual(_experiment_protocol(spec), spec.EXPERIMENT_PROTOCOL)
        self.assertEqual(spec.EVIDENCE_STAGE, "confirmation")
        self.assertEqual(
            spec.CLAIM_GATE["hrl_multiscale_vs_hrl_history_success"],
            "positive_ci",
        )
        self.assertEqual(
            spec.CLAIM_GATE["hrl_multiscale_vs_flat_multiscale_success"],
            "positive_ci",
        )
        self.assertEqual(
            spec.CLAIM_GATE["hierarchy_x_multiscale_success_interaction"],
            "positive_ci",
        )
        self.assertEqual(spec.CLAIM_GATE["sequential_root_extension"], "forbidden")


if __name__ == "__main__":
    unittest.main()
