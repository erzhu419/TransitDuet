import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v126_dispatch_readiness import launch_rows, effect, partition, timing


class DispatchTests(unittest.TestCase):
    def rows(self, times, lags):
        return launch_rows([dict(trip_id=i, time_s=t, readiness_s=w)
            for i, (t, w) in enumerate(zip(times, lags))], {i: True for i in range(len(times))})

    def test_eligibility_and_lag_differences_recover_actual_gaps(self):
        rows, points = self.rows([100., 120., 155.], [0., 30., 5.])
        self.assertEqual(list(rows[0]['coordinates']), [-10., 30.])
        self.assertEqual(list(rows[1]['coordinates']), [60., -25.])
        self.assertEqual(points[1, 0, True]['coordinates'][0], 90.)
        self.assertEqual(timing.moment(rows)['mean'], 27.5)

    def test_mean_readiness_reduction_is_not_a_variance_improvement(self):
        b, _ = self.rows([0., 100., 200., 300.], [0., 50., 50., 50.])
        c, _ = self.rows([0., 60., 190., 300.], [0., 10., 40., 50.])
        delta = effect(timing.moment(b), timing.moment(c))
        self.assertGreater(delta['cv_squared'], 0.)
        parts = partition(b, c, lambda r: int(r['key'][0] > 1))
        self.assertAlmostEqual(sum(r['pooled_cv2_delta'] for r in parts), delta['cv_squared'])


if __name__ == '__main__':
    unittest.main()
