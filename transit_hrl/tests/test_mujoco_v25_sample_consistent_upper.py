import shlex
import unittest

from freq_hrl.experiments.mujoco.control_validation import build_parser
from scripts import mujoco_v25_sample_consistent_upper_spec as spec
from scripts.submit_mujoco_v25_sample_consistent_upper_scheduleurm import cells, task_spec, training_command
from scripts.analyze_mujoco_v25_sample_consistent_upper import development_gates


class SampleConsistentUpperScreenTest(unittest.TestCase):
    def test_commands_parse_and_cells_are_unpinned_and_unique(self):
        for preflight, count in ((True, 12), (False, 48)):
            jobs = cells(preflight)
            self.assertEqual(len(set(jobs)), count)
            signatures = set()
            for cell in jobs:
                tokens = shlex.split(training_command("test", cell, preflight=preflight))
                args = build_parser().parse_args(tokens[tokens.index("--") + 1:] + ["--output-dir", "/tmp/v25-test"])
                self.assertEqual(args.upper_projection_consistency_objective, cell[1] if cell[1] != spec.ZERO else "raw_mean")
                self.assertEqual(args.upper_projection_target_aggregation, "decision_time")
                job = task_spec("test", cell, preflight=preflight)
                self.assertIsNone(job["require_node"])
                self.assertEqual(job["cpu"], 1)
                self.assertEqual(len(job["allowed_nodes"]), 6)
                signatures.add(job["signature"])
            self.assertEqual(len(signatures), count)

    def test_development_and_preflight_seed_roles_are_disjoint(self):
        roots = [*spec.OPTIMIZER_SEEDS, *spec.TRAIN_SEEDS, *spec.SELECTION_SEEDS, *spec.EVAL_SEEDS]
        roots.extend(seed for role in spec.PREFLIGHT_SEEDS.values() for seed in role)
        self.assertEqual(len(roots), len(set(roots)))

    def test_gate_never_selects_better_diagnostic_or_pools_harmed_environment(self):
        values = {cell: dict(reward=100., component=0.2, total=0.2) for cell in cells()}
        for env, arm, seed in values:
            if arm in (spec.CANDIDATE, spec.DIAGNOSTIC):
                values[(env, arm, seed)] = dict(reward=110., component=0.18, total=0.18)
        _, gates = development_gates(values)
        self.assertTrue(all(gates.values()))
        for seed in spec.OPTIMIZER_SEEDS:
            values[("Walker2d-v5", spec.CANDIDATE, seed)]["reward"] = 80.
        _, gates = development_gates(values)
        self.assertFalse(gates["reward_regression"])
        self.assertFalse(gates["reward_wins"])
        for env, _, seed in values:
            values[(env, spec.CANDIDATE, seed)]["reward"] = 100.
        _, gates = development_gates(values)
        self.assertFalse(gates["reward_wins"])


if __name__ == "__main__":
    unittest.main()
