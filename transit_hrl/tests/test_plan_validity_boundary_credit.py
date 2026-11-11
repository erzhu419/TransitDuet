import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments import plan_validity_boundary_credit as credit
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, DenseTask, options


class BoundaryCreditTest(unittest.TestCase):
    def test_variance_respects_window_clusters_and_keeps_negative_estimates(self):
        result = credit.variance_components([[1, 3], [5, 7]])
        self.assertEqual(result["within_window"], 2)
        self.assertEqual(result["between_window_corrected"], 7)
        self.assertEqual(result["total_corrected"], 9)
        self.assertEqual(result["nested_mean_mc_se"], 2)
        same = credit.variance_components([[1, 3], [1, 3]])
        self.assertEqual(same["between_window_corrected"], -1)
        self.assertEqual(same["nested_mean_mc_se"], 0)

    def test_boundary_context_excludes_the_boundary_action_itself(self):
        row = {"opportunity_step": 105, "feature_names": ["x"], "causal_features": [1.]}
        data = {"feature_rows": [row], "decision_steps": [0, 50, 105, 155]}
        before = credit.boundary_row(data, step=105, horizon=500, period=50)
        self.assertFalse(before["call_spent"])
        data["decision_steps"][2] = 100
        self.assertTrue(credit.boundary_row(data, step=105, horizon=500, period=50)["call_spent"])

    def test_nested_suffix_redraw_preserves_both_realized_windows_and_replay_budget(self):
        args = options()
        args.horizon = 500
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        rule = clock_rule()
        controller = Controller()
        ledger = ComputeLedger()
        forks = []
        conditional_future = PointMazeRegimeDriver.conditional_future

        def record_fork(driver, *, step, seed):
            forks.append((step, seed))
            return conditional_future(driver, step=step, seed=seed)

        with patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})), \
             patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))), \
             patch.object(PointMazeRegimeDriver, "conditional_future", new=record_fork):
            with ledger.recording("nested"):
                obs = observe(controller, rule, spec=spec, args=args, seed=71, probes_per_path=1)
                result = credit.collect_boundary_pair(controller, rule, obs, 55, spec=spec, args=args, windows=2, suffixes=2)
            self.assertEqual(result["boundary_step"], 105)
            self.assertEqual(ledger.summary()["counts"]["environment_steps"], 11 * args.horizon)
            self.assertEqual(ledger.summary()["counts"].get("ridge_fit_calls", 0), 0)
            self.assertEqual(forks, [fork for w in result["windows"] for suffix in w["suffix_draws"]
                for _ in range(2) for fork in ((55, w["window_seed"]), (105, suffix["suffix_seed"]))])
            for window in result["windows"]:
                for index, arm in enumerate(("now", "wait_one_check")):
                    original = shared.run_episode(controller, task_spec=spec, seed=71, args=args,
                        time_scale=shared.time_contract(args), rule=rule, intervention_step=55, arm=arm,
                        continuation_seed=window["window_seed"], feature_steps=(105,))[1]
                    self.assertEqual(original["window_ise"], window["window_costs"][index])
                    self.assertEqual(credit.boundary_row(original, step=105, horizon=500, period=50),
                        window["boundary_context"][arm])
                for row in window["suffix_draws"]:
                    self.assertAlmostEqual(row["W"] + row["tail"], row["G"], places=10)
            self.assertEqual(result["variance_components"]["W"]["within_window"], 0)

    def test_boundary_seed_without_a_sampled_window_is_rejected_before_native_access(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        with patch.object(shared.PlanValidityTaskSpec, "open") as native:
            with self.assertRaisesRegex(ValueError, "sampled intervention window"):
                shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                    time_scale=shared.time_contract(args), rule=clock_rule(), intervention_step=55,
                    arm="now", boundary_future_seed=1)
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
