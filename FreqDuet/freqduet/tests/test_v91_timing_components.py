from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v91_timing_components as audit


class TimingComponentsTest(unittest.TestCase):
    def events(self, times, launch=0.):
        return [dict(trip_id=i, station_id=2, direction=True, time_s=t,
                     release=launch, realized_holding=0., other_elapsed=t - launch) for i, t in enumerate(times)]

    def test_holding_is_executed_counter_difference_not_command_or_previous_trip(self):
        r = audit.timing_event(dict(time_s=100., holding_counter_at_launch=75.), 85., 160., 2, 3, True)
        self.assertEqual(r["realized_holding"], 10.)
        self.assertEqual(r["other_elapsed"], 50.)
        self.assertEqual(sum(r[k] for k in audit.COMPONENTS), r["time_s"])

    def test_negative_covariances_are_retained(self):
        c = np.asarray([[100., 10., 0.], [200., -10., 0.], [300., 0., 0.]])
        m = audit.moments(c.sum(1), c)
        self.assertLess(m["covariance_s2"]["realized_holding"], 0.)
        self.assertAlmostEqual(sum(m["covariance_s2"].values()), m["variance_s2"])

    def test_common_clock_advance_does_not_change_headways(self):
        b = audit.timing_series(self.events([100., 200., 400.]))
        c = audit.timing_series(self.events([90., 190., 390.]))
        delta = audit.component_effect(b, c)
        self.assertEqual(delta["cv_squared"], 0.)
        self.assertEqual(delta["mean_headway_s"], 0.)
        self.assertEqual(delta["last_boundary_mean_s"], -5.)
        self.assertEqual(delta["first_boundary_mean_s"], 5.)

    def test_variance_and_service_span_components_reconstruct_effect(self):
        b = audit.timing_series(self.events([100., 200., 400.]))
        c = audit.timing_series(self.events([100., 210., 380.]))
        delta = audit.component_effect(b, c)
        self.assertAlmostEqual(delta["mean_headway_s"], -10.)
        self.assertAlmostEqual(delta["last_boundary_mean_s"], -10.)
        self.assertAlmostEqual(sum(delta[k + "_variance_at_base_mean"] for k in audit.COMPONENTS) +
                               delta["mean_normalization"], delta["cv_squared"])

    def test_sample_count_changes_have_explicit_boundary_normalization(self):
        b = audit.timing_series(self.events([100., 200., 400.]))
        c = audit.timing_series(self.events([100., 200., 300., 400.]))
        delta = audit.component_effect(b, c)
        self.assertAlmostEqual(delta["count_normalization_mean_s"], -50.)
        self.assertAlmostEqual(delta["mean_headway_s"], -50.)

    def test_actual_predecessor_changes_are_not_position_matched_trips(self):
        r = audit.order_changes({"stop": [1, 2, 3]}, {"stop": [1, 3, 2]})
        self.assertEqual(r["matched"], 3)
        self.assertEqual(r["changed_predecessor"], 2)


if __name__ == "__main__":
    unittest.main()
