from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v86_feedback_timing import ITEMS, cleared_tail_at, headway_prefix, window_values


class FeedbackTimingTest(unittest.TestCase):
    def test_all_existing_prefixes_futures_and_regimes_are_kept(self):
        self.assertEqual(len(ITEMS), 24 * 8 * 3)

    def test_cleared_tail_keeps_constant_waiting_and_zero_new_onboard(self):
        start = np.array([10., 20., 0., 0., 30.])
        final = start + np.array([100., 0., 0., 0., 0.])
        np.testing.assert_array_equal(cleared_tail_at(150., 100., start, 200., final), [60., 20., 0., 0., 30.])

    def test_prefix_cv_excludes_later_arrivals(self):
        events = [dict(arrival_time_s=10., headway_s=None), dict(arrival_time_s=20., headway_s=10.),
                  dict(arrival_time_s=40., headway_s=20.)]
        self.assertEqual(headway_prefix(events, 20.)["cv_squared"], 0.)
        self.assertAlmostEqual(headway_prefix(events, 40.)["cv_squared"], 1. / 9.)

    def test_window_exposure_adds_but_cv_is_endpoint_statistic(self):
        def sample(t, wait, onboard, cv):
            return dict(time_s=t, waiting_person_s=wait, onboard_person_s=onboard,
                dispatch_backlog_trip_s=0., holding_vehicle_s=0., holding_person_s=0.,
                readiness_delay_sum_s=0., headways=dict(cv_squared=cv))
        row = dict(target_time_s=100., samples=[sample(100., 10., 20., .1), sample(1000., 110., 70., .2)])
        result = window_values(row, 0, 900)
        self.assertEqual(result["passenger_person_s"], 150.)
        self.assertEqual(result["cumulative_cv_squared"], .2)


if __name__ == "__main__":
    unittest.main()
