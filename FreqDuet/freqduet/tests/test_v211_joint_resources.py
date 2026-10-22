import unittest

import numpy as np

import run_freqduet_v211_joint_resources as v


class JointResourceTests(unittest.TestCase):
    def test_original_shared_authority(self):
        np.testing.assert_array_equal(v.upper_candidate([50.], [-60.], [60.], 30.), [60.])
        np.testing.assert_array_equal(v.upper_candidate([-50.], [-60.], [60.], -30.), [-60.])
        with self.assertRaisesRegex(AssertionError, 'shared'):
            v.upper_candidate([0., 0.], [-60., -60.], [60., 60.], 30.)

    def test_lower_moves_only_original_feasible_bins(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 1], bool)
        self.assertEqual(v.lower_candidate(1, 1, mask), 3)
        self.assertEqual(v.lower_candidate(0, -1, mask), 0)
        self.assertEqual(v.lower_candidate(6, 1, mask), 6)
        with self.assertRaisesRegex(AssertionError, 'infeasible'):
            v.lower_candidate(2, 0, mask)

    def test_physical_time_cannot_create_slack(self):
        self.assertEqual(v.time_account(100., 600., 40.)['other_elapsed_s'], 460.)
        with self.assertRaisesRegex(AssertionError, 'physical vehicle-time'):
            v.time_account(100., 120., 45.)

    def test_incomplete_service_keeps_executed_and_pending_holding_separate(self):
        result = v.holding_account([{'holding_s': 30.}], [{'holding_s': 5.}], 50., False)
        self.assertEqual(result['executed_holding_s'], 35.)
        self.assertEqual(result['unexecuted_command_s'], 15.)
        with self.assertRaises(AssertionError):
            v.holding_account([{'holding_s': 30.}], [], 50., True)

    def test_futures_disjoint_and_common_to_candidates(self):
        self.assertFalse(set(v.SELECT_FUTURES) & set(v.EVAL_FUTURES))
        seeds = {v.future_seed(c, f) for c in range(4) for f in (*v.SELECT_FUTURES, *v.EVAL_FUTURES)}
        self.assertEqual(len(seeds), 128)
        self.assertNotIn(2119000000, seeds)

    def test_selection_cannot_buy_passenger_gain_with_other_costs(self):
        y = np.zeros((15, len(v.METRICS)))
        y[1, 0], y[1, v.METRICS.index('cv_squared')] = -10., .01
        y[2, 0], y[3, 0] = -3., -3.
        self.assertEqual(v.select_candidate(y, list(range(15))), 2)
        self.assertEqual(v.select_candidate(y, [0, 1]), 0)

    def test_selection_ignores_held_evaluation_winner(self):
        training = np.zeros((15, len(v.METRICS)))
        evaluation = training.copy()
        training[2, 0], evaluation[3, 0] = -2., -100.
        selected = v.select_candidate(training, list(range(15)))
        self.assertEqual(selected, 2)
        self.assertEqual(evaluation[selected, 0], 0.)

    def test_every_constraint_is_in_joint_gate(self):
        stats = {k: dict(ci95=[-2., -1.]) for k in v.METRICS}
        self.assertTrue(v.joint_gate(stats))
        stats['inclusive_launch_delay_s']['ci95'][1] = .1
        self.assertFalse(v.joint_gate(stats))

    def test_missing_lower_opportunity_is_not_a_zero_effect_or_a_pass(self):
        self.assertFalse(v.lower_execution('bilateral025', [])['complete'])
        one = [dict(base_bin=1, selected_bin=0)]
        self.assertFalse(v.lower_execution('bilateral025', one)['complete'])
        self.assertEqual(v.lower_execution('bilateral025', one)['changed_commands'], 1)
        self.assertTrue(v.lower_execution('bilateral025', one*2)['complete'])
        self.assertTrue(v.lower_execution('native', [])['complete'])


if __name__ == '__main__':
    unittest.main()
