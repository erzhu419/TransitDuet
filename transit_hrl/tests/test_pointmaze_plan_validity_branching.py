import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    evaluate_plan_renewal_pair,
    fit_plan_validity_predictors,
    plan_renewal_opportunities,
)
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    build_pointmaze_plan_value_model,
    pointmaze_plan_value_dimensions,
)


class PointMazePlanValidityBranchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gymnasium  # noqa: F401
            import gymnasium_robotics  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("gymnasium-robotics is not installed")
        cls.time_scale = PhysicalTimeScaleContract(
            dt_seconds=0.01,
            upper_period_seconds=0.50,
            history_seconds=0.64,
            fast_period_seconds=0.04,
        )
        cls.task_options = {
            "regime_dwell_seconds": (0.80, 1.60),
            "target_speed_modes": (-1.25, -0.55, 0.55, 1.25),
            "force_pulse_amplitude": 0.18,
            "force_pulse_duration_seconds": (0.04, 0.10),
            "force_pulse_gap_seconds": (0.45, 1.10),
            "distractor_amplitude": 0.50,
            "distractor_dwell_seconds": (0.35, 0.85),
        }

    def test_opportunities_cover_registered_classes_off_fixed_boundaries(self):
        driver = PointMazeRegimeDriver(
            seed=71,
            horizon=240,
            dt_seconds=0.01,
            **self.task_options,
        )
        opportunities = plan_renewal_opportunities(
            horizon=240,
            period_steps=50,
            branch_window_steps=50,
            regime_change_steps=driver.regime_change_steps,
            force_pulse_steps=driver.pulse_start_steps,
            distractor_change_steps=driver.distractor_change_steps,
            seed=71,
            max_events_per_class=1,
        )
        self.assertEqual(
            {item.category for item in opportunities}, set(BRANCH_CATEGORIES)
        )
        self.assertEqual(
            len({item.step for item in opportunities}), len(opportunities)
        )
        self.assertTrue(all(item.step % 50 for item in opportunities))
        self.assertTrue(all(1 <= item.step <= 190 for item in opportunities))

    def test_keep_and_renew_replay_share_an_exact_prefix(self):
        horizon = 240
        driver = PointMazeRegimeDriver(
            seed=71,
            horizon=horizon,
            dt_seconds=0.01,
            **self.task_options,
        )
        opportunity = next(
            item
            for item in plan_renewal_opportunities(
                horizon=horizon,
                period_steps=50,
                branch_window_steps=20,
                regime_change_steps=driver.regime_change_steps,
                force_pulse_steps=driver.pulse_start_steps,
                distractor_change_steps=driver.distractor_change_steps,
                seed=71,
                max_events_per_class=1,
            )
            if item.category == "regime_lag_100ms"
        )
        dimensions = pointmaze_plan_value_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=horizon,
            time_scale=self.time_scale,
            task_options=self.task_options,
        )
        model, _ = build_pointmaze_plan_value_model(
            method="hrl_regime_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=11,
        )
        row = evaluate_plan_renewal_pair(
            model,
            seed=71,
            opportunity=opportunity,
            branch_window_steps=20,
            env_id=DEFAULT_ENV_ID,
            horizon=horizon,
            time_scale=self.time_scale,
            maximum_subgoal_delta=0.75,
            task_options=self.task_options,
            optimizer_seed=11,
            split="test",
        )
        self.assertEqual(row["prefix_max_abs_difference"], 0.0)
        self.assertEqual(row["feature_max_abs_difference"], 0.0)
        self.assertEqual(len(row["causal_features"]), 37)
        self.assertFalse(row["candidate_feature_has_future_access"])
        self.assertFalse(row["candidate_feature_has_regime_label"])
        self.assertTrue(np.isfinite(row["renew_ise_advantage"]))

    def test_predictors_fit_only_registered_feature_views(self):
        names = ["plan_age_fraction", "signal", "nuisance", "context"]
        masks = {
            "age_only": [0],
            "plan_state": [0, 1],
            "change_magnitude": [0, 2],
            "causal_history": [0, 1, 2, 3],
        }

        def rows(offset: int):
            result = []
            for seed in range(offset, offset + 4):
                for index, category in enumerate(BRANCH_CATEGORIES):
                    features = [
                        index / 6.0,
                        float(index - 2),
                        float(index % 2),
                        float(seed % 3),
                    ]
                    result.append({
                        "seed": seed,
                        "category": category,
                        "feature_names": names,
                        "feature_masks": masks,
                        "causal_features": features,
                        "oracle_regime_context": [
                            float(i == seed % 4) for i in range(4)
                        ],
                        "renew_ise_advantage": (
                            0.4 * features[1]
                            + 0.2 * features[3]
                            - 0.1 * features[2]
                        ),
                    })
            return result

        fit_rows = rows(10)
        eval_rows = rows(20)
        result = fit_plan_validity_predictors(
            fit_rows,
            eval_rows,
            ridge_alpha=1.0,
            selection_rate=0.25,
        )
        self.assertEqual(set(result["predictors"]), {
            "age_only",
            "plan_state",
            "change_magnitude",
            "causal_history",
            "causal_history_plus_regime",
        })
        for row in eval_rows:
            for name in result["predictors"]:
                self.assertTrue(np.isfinite(row[f"prediction_{name}"]))


if __name__ == "__main__":
    unittest.main()
