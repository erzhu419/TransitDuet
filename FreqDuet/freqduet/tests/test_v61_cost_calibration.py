from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v61_cost_calibration import (ITEMS, calibration_metrics, dual_timing, remaining_cost)
import run_freqduet_v60_deployment_factorial as deployment


def optimizer_state(steps, lr):
    return dict(state={0: {"step": torch.tensor(float(steps))}}, param_groups=[{"lr": lr}])


class CostCalibrationTest(unittest.TestCase):
    def test_only_fixed_updated_stochastic_rollouts_are_selected(self):
        self.assertEqual(len(ITEMS), 12)
        for index in ITEMS:
            self.assertEqual(deployment.ITEMS[index][1][:2], ("updated", "stochastic"))

    def test_complete_returns_retain_negative_cost_and_last_terminal(self):
        increments, returns = remaining_cost([.01, .20, .10], .05)
        np.testing.assert_allclose(increments, [.19, -.10, -.05])
        np.testing.assert_allclose(returns, [.04, -.15, -.05])
        self.assertAlmostEqual(float(increments.sum()), .04)

    def test_same_clock_zero_cost_still_has_full_continuation(self):
        increments, returns = remaining_cost([.01, .01, .03], .05)
        self.assertEqual(increments[0], 0)
        self.assertEqual(returns[0], returns[1])

    def test_zero_benchmark_can_beat_biased_critic(self):
        metrics = calibration_metrics([1., 1.], [.02, -.02])
        self.assertAlmostEqual(metrics["bias"], 1)
        self.assertAlmostEqual(metrics["zero_mae"], .02)
        self.assertGreater(metrics["mae"], metrics["zero_mae"])
        self.assertEqual(metrics["actual_negative_fraction"], .5)

    def test_short_bellman_error_does_not_imply_calibrated_full_return(self):
        one_step = calibration_metrics([.4, .4], [.4, .4])
        full_day = calibration_metrics([.4, .4], [.02, .01])
        self.assertEqual(one_step["mae"], 0)
        self.assertGreater(full_day["mae"], .3)

    def test_first_positive_dual_after_final_actor_block_has_no_feedback(self):
        marker = dict(updates_per_day=2, updates=4, episodes=[
            dict(cv_squared=.05, regularity_budget_cv_squared=.05, lambda_before=1., lambda_after=1.),
            dict(cv_squared=.10, regularity_budget_cv_squared=.05, lambda_before=1., lambda_after=1.01)])
        updates = [dict(lambda_value=1.) for _ in range(4)]
        state = dict(policy_optimizer=optimizer_state(4, .0003),
                     lambda_optimizer=optimizer_state(2, .0001), log_lambda=torch.tensor(np.log(1.01)))
        timing = dual_timing(marker, updates, state)
        self.assertEqual(timing["actor_updates_using_first_positive_dual"], 0)
        self.assertEqual(timing["same_day_updates_using_post_violation_lambda"], 0)
        self.assertEqual(timing["dual_updates"], [2])
        updates[-1]["lambda_value"] = 1.01
        self.assertEqual(dual_timing(marker, updates, state)["actor_updates_using_first_positive_dual"], 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
