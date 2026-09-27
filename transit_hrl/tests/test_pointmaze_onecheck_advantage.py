from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments.pointmaze_onecheck_advantage import (
    advantage_matrix, fit_advantage, predict_advantage, select_checks,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _causal_plan_features
from scripts import pointmaze_onecheck_advantage_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


def causal_row(seed):
    names, values, _ = _causal_plan_features(
        observation=SimpleNamespace(physical=np.zeros(4), target_error=np.ones(2),
                                    achieved_goal=np.zeros(2)),
        feature_builder=SimpleNamespace(history=np.zeros((64, 6))),
        subgoal=np.ones(2), plan_age_steps=50,
        time_scale=PhysicalTimeScaleContract(dt_seconds=0.01, upper_period_seconds=0.5,
                                            history_seconds=0.64, fast_period_seconds=0.04),
    )
    return {"seed": seed, "feature_names": names, "causal_features": values,
            "offset_fraction": 0.4, "remaining_fraction": 0.8,
            "renew_ise_advantage": 0.01}


class OneCheckAdvantageTest(unittest.TestCase):
    def test_checks_are_before_factual_call_and_never_at_deadline(self):
        schedule = [0, *[50 * i + i % 6 * 5 for i in range(1, 24)]]
        kwargs = dict(root=209011, seed=2180201, schedule=schedule,
                      period=50, stride=5, deadline=25, count=12)
        checks = select_checks(**kwargs)
        self.assertEqual(checks, select_checks(**kwargs))
        self.assertEqual(len({s // 50 for s in checks}), 12)
        self.assertTrue(all(s % 5 == 0 and s % 50 < 25 and 1 <= s // 50 < 23
                            and s <= schedule[s // 50] for s in checks))

    def test_clock_features_reach_deployed_score(self):
        row = causal_row(1)
        matrix, names = advantage_matrix([row])
        self.assertEqual(matrix.shape, (1, 41))
        weights = np.zeros(42)
        weights[-2:] = [2.0, 3.0]
        fitted = {"feature_names": names, "model": {
            "feature_mean": np.zeros(41), "feature_scale": np.ones(41),
            "weights": weights,
        }}
        self.assertAlmostEqual(predict_advantage(fitted, row["feature_names"],
                                                row["causal_features"], 0.4, 0.8), 3.2)
        self.assertAlmostEqual(predict_advantage(fitted, row["feature_names"],
                                                row["causal_features"], 0.2, 0.5), 1.9)

    def test_fit_does_not_calibrate_threshold_or_use_evaluation_seeds(self):
        rows = [causal_row(seed) for seed in (1, 1, 2, 2)]
        fitted = fit_advantage(rows, alpha_grid=[1.0], fit_seeds=[1, 2], eval_seeds=[3])
        self.assertEqual(fitted["threshold"], 0.0)
        self.assertEqual(fitted["model"]["group_count"], 2)
        with self.assertRaisesRegex(ValueError, "seed roles"):
            fit_advantage(rows, alpha_grid=[1.0], fit_seeds=[1, 2], eval_seeds=[2])

    def test_submission_is_training_not_diagnostic_and_no_extra_roots(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_onecheck", 209011, preflight=False,
                                  protocol_spec=spec)
        self.assertIn("--pairs-per-seed 12", task["cmd"])
        self.assertNotIn("--pairs-per-class", task["cmd"])
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertEqual(task["cpu"], 1)
        self.assertIsNone(task["require_node"])


if __name__ == "__main__":
    unittest.main()
