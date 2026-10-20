import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v159_residual_archive as check


class ReadbackTests(unittest.TestCase):
    def test_episode_weighting_matches_frozen_describe(self):
        rows = [dict(episode=(1, 2), e_running=1.), dict(episode=(1, 2), e_running=3.),
            dict(episode=(2, 2), e_running=8.)]
        result = check.describe(rows, [True]*3)
        self.assertEqual(result['metrics']['e_running'], 5.)
        self.assertEqual(result, check.audit.stats.describe(rows))

    def test_metrics_match_v158(self):
        errors = dict(running=3., service=-2., residence=1.)
        self.assertEqual(check.metrics((1, 2), errors, 2.), check.frozen.error_record((1, 2), errors, 2.))
        with self.assertRaises(AssertionError):
            check.metrics((1, 2), errors, 3.)

    def test_missing_label_cannot_acquire_zero_residual(self):
        record = dict(label=dict(full_eta_s=None), forecasts=None, components=None)
        self.assertEqual(check.component_record(record), {})

    def test_unfinished_support_fails(self):
        from collections import Counter
        row = dict(query_time_s=10., forecasts={'v151': {}}, support=dict(service=[dict(
            query_time_s=10., support=dict(local_n=1, local_survivor_n=1, prior_n=0,
                prior_days=0, prior_survivor_n=0, prior_survivor_days=0,
                latest_current_s=10., historical_fallback=False, conditional_mass=1.))]))
        with self.assertRaisesRegex(AssertionError, 'unfinished'):
            check.support_record(row, Counter())


if __name__ == '__main__':
    unittest.main()
