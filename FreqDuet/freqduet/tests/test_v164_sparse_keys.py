from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v164_sparse_keys as audit


class SparseKeyTests(unittest.TestCase):
    def test_hour13_support_weights_keep_episode_and_day_balance(self):
        weights = audit.support_weights(np.array([10, 10, 10, 10, 10, 11]),
            np.array([0, 0, 0, 0, 1, 2]), 1, [10, 11])
        np.testing.assert_allclose(weights, [1/16]*4+[1/4, 1/2])
        result = audit.partition(weights, np.array([True]*4+[False, False]),
            np.array([True]*4+[False, True]))
        self.assertAlmostEqual(result['finite_mass'], .25)
        self.assertAlmostEqual(result['native_abstention_mass'], .25)
        self.assertAlmostEqual(result['key_abstention_mass'], .5)

    def test_missing_day_is_not_silently_renormalized(self):
        with self.assertRaisesRegex(AssertionError, 'grid incomplete'):
            audit.support_weights(np.array([10]), np.array([0]), 1, [10, 11])

    def test_actor_specific_day_support_is_not_added_when_describing_patterns(self):
        rows = [dict(actor=a, mode='old', values=dict(stage='service'), day_count=d,
            finite=False, queries=100, mass=.01) for a, d in ((1, 17), (2, 18))]
        grouped = audit.group_keys(rows, ['stage'], 'stage')[0]
        self.assertEqual(grouped['keys'], 2)
        self.assertEqual(grouped['actors'], 2)
        self.assertEqual((grouped['minimum_low_key_days'], grouped['maximum_low_key_days']), (17, 18))
        self.assertEqual(grouped['key_abstentions'], 200)
        self.assertAlmostEqual(grouped['key_abstention_mass'], .02)

    def test_finite_and_sparse_keys_remain_separate(self):
        rows = [dict(actor=1, mode='old', values=dict(stage='service'), day_count=d,
            finite=good, queries=10, mass=.1) for d, good in ((18, False), (19, True))]
        grouped = audit.group_keys(rows, ['stage'], 'stage')[0]
        self.assertEqual(grouped['low_support_keys'], 1)
        self.assertEqual(grouped['finite_queries'], 10)
        self.assertEqual(grouped['key_abstentions'], 10)
        self.assertEqual(grouped['maximum_low_key_days'], 18)


if __name__ == '__main__':
    unittest.main()
