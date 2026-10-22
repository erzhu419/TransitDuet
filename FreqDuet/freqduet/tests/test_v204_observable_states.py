import unittest
from unittest.mock import patch

import numpy as np

import preregister_freqduet_v204_observable_states as v


class ObservableStateInventoryTests(unittest.TestCase):
    def test_all_joint_slots_including_cross_state_and_outcome(self):
        names = list(v.saved.VARS)
        cols, missing = v.requirements(names)
        vector = np.arange(1., len(names)+1)
        expected = vector[[names.index(n) for n in v.FIELDS]]
        np.testing.assert_allclose(v.saved.moment(vector)[cols],
            np.r_[1., expected, np.outer(expected, expected)[np.triu_indices(len(expected))]])
        self.assertEqual(missing, [])
        self.assertEqual(len(set(cols)), 78)

    def test_missing_state_is_a_gap_not_a_zero_filled_moment(self):
        cols, missing = v.requirements([n for n in v.saved.VARS if n != 'matched_n'])
        self.assertIsNone(cols)
        self.assertEqual(missing, ['matched_n'])

    def test_support_at_18_and_19_days_and_no_mask_replacement(self):
        mass = np.zeros((64, 9))
        mass[:19, 0] = mass[32:51, 0] = .1
        mass[:18, 1] = mass[32:51, 1] = .2
        counts = np.stack([(part > 1e-12).sum(axis=0) for part in (mass[:32], mass[32:])])
        keep = np.all(counts >= 19, axis=0)
        v.check_support(mass, counts, keep)
        self.assertTrue(keep[0])
        self.assertFalse(keep[1])
        wrong = keep.copy()
        wrong[1] = True
        with self.assertRaisesRegex(AssertionError, 'support differs'):
            v.check_support(mass, counts, wrong)

    def test_raw_moments_get_parent_factor_once_shared_by_all_bins(self):
        m = np.zeros((64, 1, 9, v.saved.SIZE))
        m[..., v.compact.COLS] = np.arange(1., 9.)
        factors = np.array([[2.], [3.]])
        out = v.balanced_compact(m, factors)
        np.testing.assert_allclose(out[:32], np.broadcast_to(2.*np.arange(1., 9.), out[:32].shape))
        np.testing.assert_allclose(out[32:], np.broadcast_to(3.*np.arange(1., 9.), out[32:].shape))
        np.testing.assert_allclose(m[..., v.compact.COLS], np.broadcast_to(np.arange(1., 9.), out.shape))

    def test_protocol_is_observable_only_and_seals_no_fit_boundary(self):
        with patch.dict('os.environ', FREQDUET_DIAGNOSTIC_COMMIT='fixture'):
            p = v.registration()
        self.assertEqual(len(v.STATES), 9)
        self.assertTrue(set(v.STATES) <= set(v.saved.prior.CONTROLS))
        self.assertEqual(p['keys'], [[0, 7, 13], [0, 12, 13]])
        self.assertFalse(p['state_operator_fitted'])
        self.assertFalse(p['state_associations_estimated'])
        self.assertFalse(p['confirmation_started'])
        self.assertEqual(p['required']['packed_slots'], 78)
        self.assertEqual(p['frozen']['window_s'], 3600.)
        self.assertEqual(p['frozen']['tails'], [.1, .1])


if __name__ == '__main__':
    unittest.main()
