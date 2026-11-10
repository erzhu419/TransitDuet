from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.plan_validity_target_credit import (
    TARGETS, fit_targets, fitting_readout, held_readout, labels, prediction_error, predict,
)
from freq_hrl.experiments.pointmaze_timing_baselines import _CAUSAL_DYNAMIC_NAMES
from scripts import submit_plan_validity_target_credit_scheduleurm as submitter


def bank():
    rng = np.random.default_rng(604)
    output = []
    for seed in range(8):
        for step in (50, 100, 150):
            x = rng.normal(size=len(_CAUSAL_DYNAMIC_NAMES))
            effect = x[0] + x[1] * x[2]
            futures = [{"local_advantage": effect + noise,
                "adaptive_window_advantage": -effect + noise,
                "full_advantage": 2 * effect + noise} for noise in rng.normal(0, 0.1, 8)]
            output.append({"seed": seed, "opportunity_step": step,
                "feature_names": list(_CAUSAL_DYNAMIC_NAMES), "causal_features": x.tolist(),
                "conditional_replicas": futures})
    return output


class TargetCreditTest(unittest.TestCase):
    def test_future_blocks_and_capacity_are_matched(self):
        rows = bank()
        rules, oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        self.assertEqual(set(rules), set(TARGETS))
        models = [r.model["model"] for r in rules.values()]
        for model in models:
            self.assertAlmostEqual(model["effective_df_slopes"], 8)
            self.assertEqual(model["coefficient_count"], 40)
            self.assertEqual(model["alpha"], models[0]["alpha"])
        for target, rule in rules.items():
            self.assertAlmostEqual(rule.threshold, np.quantile(oof[target], 0.75))
            scores = np.array([rule.score(tuple(r["feature_names"]), np.array(r["causal_features"]), 0, 1) for r in rows])
            np.testing.assert_allclose(scores, predict(rule, rows))

    def test_independent_future_B_cannot_change_coefficients_or_thresholds(self):
        rows = bank()
        first, first_oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        changed = deepcopy(rows)
        for row in changed:
            for future in row["conditional_replicas"][4:]:
                for name in TARGETS.values():
                    future[name] += 1000
        second, second_oof = fit_targets(changed, effective_df=8, threshold_quantile=0.75)
        self.assertEqual({k: v.to_dict() for k, v in first.items()}, {k: v.to_dict() for k, v in second.items()})
        for target in TARGETS:
            np.testing.assert_array_equal(first_oof[target], second_oof[target])

    def test_train_error_is_not_noise_corrected_using_shared_fit_noise(self):
        rows = bank()
        rules, oof = fit_targets(rows, effective_df=8, threshold_quantile=0.75)
        result = fitting_readout(rows, rules, oof)
        for target in TARGETS:
            self.assertIsNone(result[target]["own_training_A"]["noise_corrected_mse"])
            self.assertIsNotNone(result[target]["own_independent_future_B"]["noise_corrected_mse"])
            self.assertEqual(result[target]["own_prefix_oof_B"]["paths"], 8)
        held = held_readout(rows, rules)
        self.assertEqual(set(held["G"]), {"A", "B", "all"})

    def test_negative_noise_correction_is_preserved(self):
        rows = bank()[:1]
        for i, future in enumerate(rows[0]["conditional_replicas"]):
            future["full_advantage"] = (-1) ** i
        result = prediction_error(np.zeros(1), rows, "G", "B", independent=True)
        self.assertLess(result["noise_corrected_mse"], 0)
        rows[0]["conditional_replicas"].pop()
        with self.assertRaises(ValueError):
            labels(rows, "L", "A")

    def test_reservation_retains_hold_and_never_reuses_old_eval_paths(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_local_episode_diagnostic_2026-11-10.json").read_text())
        plan = draft_plan()
        result = submitter.prepare(source, plan, "fixture", Path("protocols/fixture.json"))
        reserved = {s for c in seed_cells(plan) for seeds in c["seed_roles"].values() for s in seeds}
        for cell in result["cells"]:
            self.assertEqual(cell["training_prefix_seeds"], cell["expected_configuration"]["branch_fit_seeds"])
            self.assertFalse(set(cell["held_prefix_seeds"]) & reserved)
            self.assertFalse(set(cell["held_prefix_seeds"]) & set(cell["previous_evaluation_seeds"]))
        for phase, jobs in result["jobs"].items():
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertIn("--phase " + phase, job["cmd"])
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["protect_from_auto_eviction"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        self.assertFalse(result["confirmation_released"])
        self.assertEqual(result["candidate_recipe_decision"], "PILOT_RECIPE_HOLD")

    def test_missing_training_paths_are_not_replaced_with_evaluation_data(self):
        with self.assertRaisesRegex(ValueError, "three fitting paths"):
            fit_targets(bank()[:6], effective_df=8, threshold_quantile=0.75)


if __name__ == "__main__":
    unittest.main()
