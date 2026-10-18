from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_freqduet_v104_system_credit import teacher, model, conflicts


class SystemCreditReviewTests(unittest.TestCase):
    def test_held_futures_do_not_change_teacher_law(self):
        y = np.tile(np.arange(7, dtype=float)[None, :, None], (4, 1, 3))
        p = np.full(7, 1 / 7)
        b, _ = teacher(y, p, (0, 1), np.ones(3))
        y[2:] *= -100
        other, _ = teacher(y, p, (0, 1), np.ones(3))
        np.testing.assert_array_equal(b, other)
        self.assertTrue(np.all((b - p) @ y[:2].mean(0) < 0))
        self.assertTrue(np.all((b - p) @ y[2:].mean(0) > 0))

    def test_common_levels_leave_law_and_measured_changes_unchanged(self):
        y = np.tile(np.arange(7, dtype=float)[None, :, None], (4, 1, 3))
        shift = np.arange(12, dtype=float).reshape(4, 1, 3) * 10000
        p = np.full(7, 1 / 7)
        b, _ = teacher(y, p, (0, 1), np.ones(3))
        other, _ = teacher(y + shift, p, (0, 1), np.ones(3))
        np.testing.assert_array_equal(b, other)
        np.testing.assert_allclose((b - p) @ y[2], (b - p) @ (y + shift)[2], atol=1e-9)

    def test_future_rows_are_not_independent_contexts(self):
        values = dict(zip(model.GOALS, (-1., 0., 1.)))
        rows = [dict(seed=1, day=2, target=3, effects=dict(teacher=values)) for _ in range(4)]
        result = conflicts(rows)['teacher']
        self.assertEqual(result['contexts'], 1)
        self.assertEqual(result['joint_nonharm'], 0)
        self.assertEqual(result['harm_counts']['readiness_delay_s'], 1)


if __name__ == '__main__':
    unittest.main()
