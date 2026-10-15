from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_system_learner import SystemStateEncoder
from lower.resac_lagrangian import CategoricalPolicy
from run_freqduet_v60_deployment_factorial import (CASES, ITEMS, METRICS, factorial_summary,
                                                  item_name, make_policy, sampler_seed)


class DeploymentFactorialTest(unittest.TestCase):
    def test_fixed_matrix_has_every_paired_cell_once(self):
        self.assertEqual(len(ITEMS), 32)
        self.assertEqual(len({item_name(i) for i in range(32)}), 32)
        for case in range(4):
            for actor in ("initial", "updated"):
                reps = [rep for c, (a, mode, rep) in ITEMS if (c, a, mode) == (case, actor, "stochastic")]
                self.assertEqual(reps, [0, 1, 2])

    def test_paired_sampling_resets_generator_even_when_weights_are_loaded(self):
        old = CategoricalPolicy(2, (0, 5, 10, 15, 20, 30, 45), hidden_dim=8)
        encoder = SystemStateEncoder(2, 3, 10, 5, 100)
        seed = sampler_seed(12011, 0)
        initial = make_policy(old, encoder, seed)
        initial.get_action(np.zeros(encoder.dimension, dtype=np.float32))
        payload = dict(policy=initial.state_dict(), encoder=encoder.__dict__, steps=64)
        loaded = make_policy(old, encoder, seed, payload)
        fresh = make_policy(old, encoder, seed)
        self.assertTrue(torch.equal(loaded.sampling_state(), fresh.sampling_state()))
        states = np.ones(encoder.dimension, dtype=np.float32)
        for _ in range(20):
            np.testing.assert_array_equal(loaded.get_action(states), fresh.get_action(states))
        self.assertNotEqual(sampler_seed(12011, 0), sampler_seed(12011, 1))

    def test_initial_actor_keeps_original_probabilities_bit_exact(self):
        old = CategoricalPolicy(2, (0, 5, 10, 15, 20, 30, 45), hidden_dim=8)
        encoder = SystemStateEncoder(2, 3, 10, 5, 100)
        policy = make_policy(old, encoder, 10)
        features = torch.randn(10, encoder.dimension)
        with torch.no_grad():
            before = old.dist_info(features[:, :2].contiguous())[0]
            after = policy.dist_info(features)[0]
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def rows(self):
        rows = []
        for case, (actor, mode, rep) in ITEMS:
            seed, scenario = CASES[case]
            value = case * 100 + (2 if actor == "updated" else 0)
            value += (-5 if mode == "stochastic" else 0)
            value += (rep - 1 if mode == "stochastic" else 0)
            row = dict(train_seed=seed, scenario_seed=scenario, actor=actor, mode=mode, rep=rep,
                       cv_budget_squared=1000000, **{m: float(value) for m in METRICS})
            rows.append(row)
        return rows

    def test_sampling_benefit_is_not_mistaken_for_learning(self):
        result = factorial_summary(self.rows())
        effects = result["mean_effects"]
        self.assertEqual(effects["learning_deterministic"]["journey_min"], 2)
        self.assertEqual(effects["learning_stochastic"]["journey_min"], 2)
        self.assertEqual(effects["sampling_initial"]["journey_min"], -5)
        self.assertEqual(effects["combined_updated_stochastic"]["journey_min"], -3)
        self.assertEqual(effects["interaction"]["journey_min"], 0)
        self.assertEqual(len(result["cases"]), 4)

    def test_missing_or_duplicate_replicate_stops_aggregation(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            factorial_summary(rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            factorial_summary(rows + [rows[0]])

    def test_constraint_averages_squared_cv_inside_each_context(self):
        rows = self.rows()
        for row in rows:
            row["cv"] = (row["rep"] + 1) / 10
            row["cv_squared"] = row["cv"] ** 2
        result = factorial_summary(rows)
        values = result["cases"][0]["means"]["initial_stochastic"]
        self.assertAlmostEqual(values["cv_squared"], (.01 + .04 + .09) / 3)
        self.assertNotAlmostEqual(values["cv_squared"], values["cv"] ** 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
