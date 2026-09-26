import unittest
from collections import Counter

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    balanced_jitter_schedule,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    plan_renewal_opportunities,
)
from scripts import pointmaze_budgeted_trigger_stage9_spec as spec
from scripts import pointmaze_compact_plan_validity_stage8c_spec as stage8c
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_budgeted_trigger_stage9_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeBudgetedTriggerStageNineTest(unittest.TestCase):
    def test_frozen_roles_are_fresh_and_disjoint(self):
        old = {
            seed
            for root in (*stage8c.PREFLIGHT_OPTIMIZER_SEEDS, *stage8c.OPTIMIZER_SEEDS)
            for values in stage8c.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for root in (
            *spec.PREFLIGHT_OPTIMIZER_SEEDS,
            *spec.OPTIMIZER_SEEDS,
            *spec.CONFIRMATION_OPTIMIZER_SEEDS,
        ):
            values = [
                seed for role in spec.seed_roles(root).values() for seed in role
            ]
            self.assertEqual(len(values), 40)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(old.intersection(values))
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_all_registered_branch_fit_paths_have_balanced_opportunities(self):
        for preflight, confirmation in ((True, False), (False, False), (False, True)):
            for _, root in spec.cells(
                preflight=preflight, confirmation=confirmation,
            ):
                options = spec.cell_options(root, preflight=preflight)
                count = options["max_events_per_class"]
                for seed in options["branch_fit"]:
                    driver = PointMazeRegimeDriver(
                        seed=seed,
                        horizon=options["horizon"],
                        dt_seconds=0.01,
                        regime_dwell_seconds=options["regime_dwell_seconds"],
                        target_speed_modes=options["target_speed_modes"],
                        force_pulse_amplitude=options["force_pulse_amplitude"],
                        force_pulse_duration_seconds=options["force_pulse_duration_seconds"],
                        force_pulse_gap_seconds=options["force_pulse_gap_seconds"],
                        distractor_amplitude=options["distractor_amplitude"],
                        distractor_dwell_seconds=options["distractor_dwell_seconds"],
                    )
                    schedule = balanced_jitter_schedule(
                        seed=seed,
                        horizon=options["horizon"],
                        period_steps=50,
                        max_offset_steps=options["max_offset_steps"],
                    )
                    rows = plan_renewal_opportunities(
                        horizon=options["horizon"],
                        period_steps=50,
                        branch_window_steps=50,
                        regime_change_steps=driver.regime_change_steps,
                        force_pulse_steps=driver.pulse_start_steps,
                        distractor_change_steps=driver.distractor_change_steps,
                        seed=seed,
                        max_events_per_class=count,
                        blocked_steps=schedule,
                    )
                    counts = Counter(row.category for row in rows)
                    self.assertEqual(set(counts), set(BRANCH_CATEGORIES))
                    self.assertEqual(set(counts.values()), {count})
                    self.assertTrue(set(row.step for row in rows).isdisjoint(schedule))

    def test_task_uses_dynamic_single_core_and_exact_budget_options(self):
        cell = spec.cells(preflight=True)[0]
        command = training_command("unit_stage9", cell[1], preflight=True)
        self.assertIn("--max-offset-steps 25", command)
        self.assertIn("--check-stride-steps 5", command)
        self.assertIn("--trigger-eval-seeds", command)
        task = task_specification("unit_stage9", cell[1], preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertTrue(task["allow_no_ckpt"])

    def test_confirmation_uses_fresh_roots_and_same_frozen_options(self):
        cells = spec.cells(preflight=False, confirmation=True)
        self.assertEqual(len(cells), 8)
        self.assertFalse(set(spec.OPTIMIZER_SEEDS).intersection(root for _, root in cells))
        self.assertEqual(
            spec.CONFIRMATION_CLAIM_GATE["authorization_scope"],
            "stage9_pointmaze_confirmation_result_only",
        )
        for _, root in cells:
            options = spec.cell_options(root, preflight=False)
            self.assertEqual(len(options["trigger_eval"]), 16)
            task = task_specification(
                "unit_stage9_confirmation", root,
                preflight=False, confirmation=True,
            )
            self.assertEqual(task["project"], spec.CONFIRMATION_EXPERIMENT_PROTOCOL)
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])


if __name__ == "__main__":
    unittest.main()
