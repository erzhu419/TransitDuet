from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v169_residual_transfer as audit


class ResidualTransferTests(unittest.TestCase):
    def test_signed_components_and_cancellation(self):
        v = audit.residual_values(np.array([[-10., 7., -3.]]))[0]
        np.testing.assert_array_equal(v, [-10., 10., 7., 7., -3., 3., -6., 6., 7.])

    def test_episode_weights_not_query_counts(self):
        v = audit.episode_mean(np.array([[0.], [0.], [12.]]), np.array([1, 1, 2]))
        self.assertEqual(v[0], 6.)
        self.assertIsNone(audit.episode_mean(np.empty((0, 1)), np.array([], dtype=int)))

    def test_narrower_interval_need_not_contract_upper_bound(self):
        v = audit.bound_values(np.array([100.]), np.array([[[50., 120.], [0., 110.]]]), np.array([64.]))[0]
        m = dict(zip(audit.BOUND_METRICS, v))
        self.assertEqual(m['width_delta_s'], -40.)
        self.assertEqual(m['upper_delta_s'], 10.)
        self.assertEqual(m['upper_contracted'], 0.)

    def test_upper_miss_partition_and_reference_coverage(self):
        bounds = np.array([[[90., 110.], [0., 120.]]]*4)
        v = audit.bound_values(np.array([100., 115., 125., 110.]), bounds, np.array([64.]*4))
        m = {n: v[:, i] for i, n in enumerate(audit.BOUND_METRICS)}
        np.testing.assert_array_equal(m['above']-m['reference_above'],
            m['candidate_only_upper']-m['reference_only_upper'])
        np.testing.assert_array_equal(sum(m[n] for n in ('candidate_only_upper', 'reference_only_upper',
            'both_upper', 'neither_upper')), np.ones(4))
        np.testing.assert_array_equal(m['new_upper_reference_covered'], [0., 1., 0., 0.])

    def fixture(self):
        return [np.array([2951001]*3), np.array([1, 1, 2]), np.array([2]*3),
            np.array([True, True, False]), np.array([True, False, False]),
            np.array([[-10., 0., 0.], [10., 0., 0.], [np.nan]*3]), np.array([100., 100., np.nan]),
            np.array([[[90., 110.], [80., 120.]], [[np.nan]*2]*2, [[np.nan]*2]*2]),
            np.array([64., 18., np.nan])]

    def test_native_and_finite_masks_and_empty_strata_preserved(self):
        data = self.fixture()
        before = deepcopy(data)
        rows = audit.cells(1, 'old', 'calibration64_in_sample', data)
        matched = next(r for r in rows if r['scope'] == 'hour2' and r['mask'] == 'matched')
        self.assertEqual(matched['native_fraction'], .5)
        self.assertEqual(matched['finite_fraction'], .25)
        self.assertEqual(matched['running_bias_s'], -10.)
        empty = next(r for r in rows if r['scope'] == 'hour13' and r['mask'] == 'candidate_only_upper')
        self.assertIsNone(empty['joint_bias_s'])
        self.assertEqual(empty['selected_queries'], 0)
        for a, b in zip(data, before):
            np.testing.assert_array_equal(a, b)

    def test_missing_native_residual_and_reference_are_not_silently_dropped(self):
        for index in (5, 7):
            data = self.fixture()
            data[index][0] = np.nan
            with self.assertRaises(AssertionError):
                audit.cells(1, 'old', 'calibration64_in_sample', data)

    def test_sparse_marginals_are_json_null_not_nan(self):
        rows = audit.cells(1, 'old', 'calibration64_in_sample', self.fixture())
        result = audit.describe(rows, [1], dict(calibration_days=[2951001]))
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
