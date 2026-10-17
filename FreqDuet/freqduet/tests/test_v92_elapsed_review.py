from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v92_elapsed_review as audit


class ElapsedReviewTest(unittest.TestCase):
    def row(self, trip, station, time, held, other):
        return dict(trip_id=trip, station_id=station, direction=True, time_s=time,
                    release=time - held - other, realized_holding=held, other_elapsed=other)

    def test_stop_time_uses_actual_holding_and_is_accumulated_after_departure(self):
        a = [self.row(1, 2, 100., 0., 20.), self.row(1, 3, 150., 10., 60.)]
        d = [self.row(1, 2, 120., 10., 30.), self.row(1, 3, 180., 20., 80.)]
        split = audit.split_stop_time(dict(arrivals=a, departures=d))
        self.assertEqual(split[((1, 2, True), "arrival")][2], 0.)
        self.assertEqual(split[((1, 2, True), "departure")][2], 10.)
        self.assertEqual(split[((1, 3, True), "arrival")][2], 10.)
        self.assertEqual(split[((1, 3, True), "departure")][2], 30.)
        self.assertEqual(split[((1, 3, True), "arrival")][3], 50.)

    def test_impossible_executed_holding_is_not_clipped_into_a_dwell_label(self):
        a = self.row(1, 2, 100., 0., 20.)
        d = self.row(1, 2, 110., 20., 10.)
        with self.assertRaises(AssertionError):
            audit.split_stop_time(dict(arrivals=[a], departures=[d]))

    def test_cross_covariance_is_kept_instead_of_treating_variances_as_additive(self):
        c = np.array([[100., 10., 5., 0.], [200., -10., 0., 5.], [300., 0., 5., 0.]])
        m = audit.covariance_moments(c.sum(1), c)
        self.assertLess(m["covariance"][0, 1], 0.)
        self.assertNotAlmostEqual(np.trace(m["covariance"]), m["variance"])
        self.assertAlmostEqual(m["covariance"].sum(), m["variance"])

    def test_changed_predecessor_contributions_include_large_rare_edges(self):
        def record(h, predecessor):
            return dict(headway=h, predecessor=predecessor)
        b = dict(samples=2, mean=100., cv_squared=1., edges={
            1: record(0., 0), 2: record(200., 1)})
        c = dict(samples=2, mean=100., cv_squared=.25, edges={
            1: record(50., 0), 2: record(150., 3)})
        result = audit.edge_contributions(b, c)
        self.assertAlmostEqual(result["changed_predecessor_cv_squared"], -.375)
        self.assertAlmostEqual(result["same_predecessor_cv_squared"], -.375)

    def test_covariance_and_boundary_effects_reconstruct_their_parent_differences(self):
        c = np.asarray([[100., 10., 5., 0.], [200., 0., 5., 0.], [300., 0., 10., 0.]])
        b = audit.covariance_moments(c.sum(1), c)
        b.update(first=np.zeros(4), last=c.sum(0), edges={i: dict(predecessor=i-1, headway=h)
            for i, h in enumerate(c.sum(1))})
        d = c.copy();d[:, 1] += [0., 5., -5.]
        x = audit.covariance_moments(d.sum(1), d)
        x.update(first=np.zeros(4), last=d.sum(0), edges={i: dict(predecessor=i-1, headway=h)
            for i, h in enumerate(d.sum(1))})
        e = audit.effects(b, x)
        self.assertAlmostEqual(sum(e[k + "_allocated"] for k in audit.COMPONENTS) + e["mean_normalization"],
                               x["cv_squared"] - b["cv_squared"])
        self.assertAlmostEqual(sum(v for k, v in e.items() if k.endswith("_mean_s")), x["mean"] - b["mean"])


if __name__ == "__main__":
    unittest.main()
