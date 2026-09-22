import unittest

from scripts import pointmaze_exogenous_multiscale_stage7_spec as stage7
from scripts import pointmaze_plan_value_stage8_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_plan_value_stage8_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazePlanValueStageEightSchedulerTest(unittest.TestCase):
    def test_matrix_and_seed_roles_are_frozen_and_fresh(self):
        self.assertEqual(len(spec.METHODS), 2)
        self.assertEqual(len(spec.OPTIMIZER_SEEDS), 8)
        self.assertEqual(len(spec.cells(preflight=False)), 16)
        self.assertEqual(len(spec.cells(preflight=True)), 2)
        self.assertEqual(spec.PREFLIGHT_OPTIMIZER_SEEDS, (204901,))
        old = {
            seed
            for root in (
                *stage7.PREFLIGHT_OPTIMIZER_SEEDS,
                *stage7.OPTIMIZER_SEEDS,
            )
            for values in stage7.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for root in (*spec.PREFLIGHT_OPTIMIZER_SEEDS, *spec.OPTIMIZER_SEEDS):
            roles = spec.seed_roles(root)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 32)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(old.intersection(values))
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_preflight_is_small_but_contains_hidden_regime_events(self):
        options = spec.cell_options(
            spec.PREFLIGHT_OPTIMIZER_SEEDS[0], preflight=True
        )
        self.assertEqual(options["iterations"], 2)
        self.assertEqual(options["horizon"], 240)
        self.assertEqual(len(options["train"]), 1)
        self.assertEqual(len(options["selection"]), 1)
        self.assertEqual(len(options["evaluation"]), 1)
        self.assertGreaterEqual(
            options["horizon"] * 0.01,
            2.0 * options["regime_dwell_seconds"][0],
        )

    def test_tasks_use_dynamic_linux_pool_and_compact_results(self):
        for method in spec.METHODS:
            cell = (method, spec.PREFLIGHT_OPTIMIZER_SEEDS[0])
            command = training_command(
                "unit_stage8", cell, preflight=True
            )
            self.assertIn(spec.RUNNER_SCRIPT, command)
            self.assertIn(f"--methods {method}", command)
            self.assertIn("--upper-period-seconds 0.5", command)
            self.assertIn("--history-seconds 0.64", command)
            task = task_specification(
                "unit_stage8", cell, preflight=True
            )
            self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["cpu"], 1)
            self.assertEqual(task["ram_mb"], 1536)
            self.assertEqual(task["stage_excludes"], [
                "results",
                "data",
                "freq_transitduet",
                "scheduler_results",
                "**/__pycache__",
            ])
            self.assertTrue(task["allow_no_ckpt"])

    def test_gate_authorizes_only_stage9_and_forbids_root_extension(self):
        self.assertEqual(
            spec.EVIDENCE_STAGE, "task_qualification_development"
        )
        self.assertEqual(
            spec.CLAIM_GATE["evidence_role"],
            "task_qualification_not_algorithm_confirmation",
        )
        self.assertEqual(
            spec.CLAIM_GATE["primary_endpoint"],
            "tracking_squared_error_integral",
        )
        self.assertEqual(
            spec.CLAIM_GATE["sequential_root_extension"], "forbidden"
        )
        self.assertTrue(
            spec.CLAIM_GATE[
                "causal_observation_precedes_quarter_second_cost"
            ]
        )


if __name__ == "__main__":
    unittest.main()
