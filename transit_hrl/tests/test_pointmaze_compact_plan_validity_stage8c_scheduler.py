import unittest

from collections import Counter

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    plan_renewal_opportunities,
)
from scripts import pointmaze_compact_plan_validity_stage8c_spec as spec
from scripts import pointmaze_plan_validity_stage8b_spec as stage8b
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_compact_plan_validity_stage8c_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeCompactPlanValidityStageEightCSchedulerTest(unittest.TestCase):
    def test_matrix_and_seed_roles_are_frozen_and_fresh(self):
        self.assertEqual(len(spec.OPTIMIZER_SEEDS), 8)
        self.assertEqual(len(spec.cells(preflight=False)), 8)
        self.assertEqual(spec.PREFLIGHT_OPTIMIZER_SEEDS, (207001,))
        old = {
            seed
            for root in (
                *stage8b.PREFLIGHT_OPTIMIZER_SEEDS,
                *stage8b.OPTIMIZER_SEEDS,
            )
            for values in stage8b.seed_roles(root).values()
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

    def test_preflight_exercises_grouped_cross_validation(self):
        options = spec.cell_options(207001, preflight=True)
        self.assertEqual(options["iterations"], 2)
        self.assertEqual(options["horizon"], 240)
        self.assertEqual(options["max_events_per_class"], 1)
        self.assertEqual(len(options["train"]), 1)
        self.assertEqual(len(options["selection"]), 1)
        self.assertEqual(len(options["branch_fit"]), 2)
        self.assertEqual(len(options["branch_eval"]), 2)

    def test_every_frozen_branch_path_has_balanced_opportunities(self):
        for root in spec.OPTIMIZER_SEEDS:
            options = spec.cell_options(root, preflight=False)
            for role in ("branch_fit", "branch_eval"):
                for seed in options[role]:
                    driver = PointMazeRegimeDriver(
                        seed=seed,
                        horizon=options["horizon"],
                        dt_seconds=0.01,
                        regime_dwell_seconds=options[
                            "regime_dwell_seconds"
                        ],
                        target_speed_modes=options["target_speed_modes"],
                        force_pulse_amplitude=options[
                            "force_pulse_amplitude"
                        ],
                        force_pulse_duration_seconds=options[
                            "force_pulse_duration_seconds"
                        ],
                        force_pulse_gap_seconds=options[
                            "force_pulse_gap_seconds"
                        ],
                        distractor_amplitude=options[
                            "distractor_amplitude"
                        ],
                        distractor_dwell_seconds=options[
                            "distractor_dwell_seconds"
                        ],
                    )
                    counts = Counter(
                        item.category
                        for item in plan_renewal_opportunities(
                            horizon=options["horizon"],
                            period_steps=50,
                            branch_window_steps=50,
                            regime_change_steps=driver.regime_change_steps,
                            force_pulse_steps=driver.pulse_start_steps,
                            distractor_change_steps=(
                                driver.distractor_change_steps
                            ),
                            seed=seed,
                            max_events_per_class=options[
                                "max_events_per_class"
                            ],
                        )
                    )
                    self.assertEqual(set(counts), set(BRANCH_CATEGORIES))
                    self.assertEqual(set(counts.values()), {4})

    def test_task_is_dynamic_single_core_and_explicit_about_alpha_grid(self):
        cell = spec.cells(preflight=True)[0]
        command = training_command("unit_stage8c", cell, preflight=True)
        self.assertIn(spec.RUNNER_SCRIPT, command)
        self.assertIn("--ridge-alpha-grid 0.01 0.1 1.0", command)
        self.assertIn("--branch-fit-seeds", command)
        self.assertIn("--branch-eval-seeds", command)
        task = task_specification("unit_stage8c", cell, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertTrue(task["allow_no_ckpt"])
        self.assertEqual(task["stage_excludes"], [
            "results",
            "data",
            "freq_transitduet",
            "scheduler_results",
            "**/__pycache__",
        ])

    def test_gate_is_conjunctive_and_forbids_root_extension(self):
        self.assertEqual(
            spec.CLAIM_GATE["primary_endpoint"],
            (
                "causal_validity_interactions_selected_utility_minus_"
                "current_compact_quadratic"
            ),
        )
        self.assertEqual(
            spec.CLAIM_GATE["authorization_scope"],
            "stage9_budgeted_trigger_development_only",
        )
        self.assertEqual(
            spec.CLAIM_GATE["sequential_root_extension"], "forbidden"
        )
        self.assertEqual(
            spec.CLAIM_GATE["candidate_utility_beats_current_only"],
            "positive_root_ci",
        )


if __name__ == "__main__":
    unittest.main()
