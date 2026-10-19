import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v143_interval_support import support_context, start_cohort
from freqduet_stop_phase_eta import ServiceResiduals


class SupportDiagnosisTests(unittest.TestCase):
    def history(self, values):
        return ServiceResiduals([(0, 1, t, d) for t, d in values])

    def test_cold_prefix_window_and_survivor_depletion_are_distinct(self):
        for samples, elapsed, expected in (([], 0, 'no_completed_history'),
                ([(1., 10.)]*8, 0, 'prefix_less_than_nine'),
                ([(1., 10.)]*9+[(4999., 10.)], 0, 'window_less_than_nine'),
                ([(4999., 10.)]*9, 11, 'elapsed_survivor_depletion'),
                ([(4999., 10.)]*9, 0, 'finite')):
            context, _ = support_context(self.history(samples), 0, 1, 5000., elapsed)
            self.assertEqual(context['reason'], expected)

    def test_simultaneous_completion_excluded_equal_duration_retained(self):
        context, values = support_context(self.history([(99., 10.), (100., 999.)]), 0, 1, 100., 10.)
        self.assertEqual(context['past_n'], 1)
        self.assertEqual(values, [(99., 10.)])

    def test_all_past_fallback_is_not_triggered_by_survivor_depletion(self):
        context, values = support_context(self.history([(1., 999.), (4999., 1.)]), 0, 1, 5000., 10.)
        self.assertFalse(context['historical_fallback'])
        self.assertEqual(values, [])
        context, values = support_context(self.history([(1., 999.)]), 0, 1, 5000., 10.)
        self.assertTrue(context['historical_fallback'])
        self.assertEqual(values, [(1., 999.)])

    def test_pending_diagnostics_do_not_enter_prefix(self):
        events = [(90., 99.), (95., 100.), (96., 110.), (100., 120.)]
        result = start_cohort(events, 100., 3.)
        self.assertEqual((result['started_n'], result['completed_started_n'], result['pending_n']), (3, 1, 2))
        self.assertEqual(result['pending_already_older_than_query_n'], 2)
        context, _ = support_context(self.history([(b, b-a) for a, b in events]), 0, 1, 100., 3.)
        self.assertEqual(context['past_n'], 1)


if __name__ == '__main__':
    unittest.main()
