from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v162_capture as audit


class CaptureReadbackTests(unittest.TestCase):
    def packet(self):
        return dict(query_time_s=100., observable=dict(position=None),
            native_stage=dict(kind='moving-launch', reason='not_applicable'),
            support=dict(service=[dict(query_time_s=100., predicted_arrival_time_s=200., support=dict(
                latest_current_s=99., local_n=3, local_survivor_n=2, prior_n=12, prior_survivor_n=10,
                prior_days=8, prior_survivor_days=7, conditional_mass=.5))],
                completed_components=[], current_residence=None, v151_detail={}))

    def test_strict_completed_support_and_absent_support(self):
        q = self.packet()
        counts = audit.Counter()
        audit.support_record(q, counts)
        self.assertEqual(counts['service_calls'], 1)
        for t in (100., 101.):
            bad = deepcopy(q)
            bad['support']['service'][0]['support']['latest_current_s'] = t
            with self.assertRaisesRegex(AssertionError, 'strictly completed'):
                audit.support_record(bad, audit.Counter())
        q['support']['service'][0]['support'] = None
        audit.support_record(q, counts)
        self.assertEqual(counts['explicit_service_fallback'], 1)

    def test_native_rank_abstention_is_not_a_missing_field(self):
        q = self.packet()
        q['native_stage'] = dict(kind='service', reason='prefix_less_than_nine', survivor_n=8,
            pool_n=8, past_n=8, latest_survivor_lag_s=1., support_class=None,
            historical_fallback=False, start_time_s=90.)
        audit.support_record(q, audit.Counter())
        q['native_stage']['support_class'] = 'finite_window'
        with self.assertRaisesRegex(AssertionError, 'support class'):
            audit.support_record(q, audit.Counter())

    def test_missing_support_channel_is_rejected(self):
        q = self.packet()
        del q['support']['current_residence']
        with self.assertRaisesRegex(AssertionError, 'channels missing'):
            audit.support_record(q, audit.Counter())

    def test_crossed_marginals_preserve_all_paths(self):
        rows = [dict(actor=a, day=d, counts=dict(queries=5)) for a in (1, 2) for d in (10, 11)]
        for field in ('actor', 'day'):
            groups = audit.marginals(rows, field)
            self.assertEqual([r['paths'] for r in groups], [2, 2])
            self.assertEqual(sum(r['counts']['queries'] for r in groups), 20)

    def test_key_support_counts_distinct_days_not_queries(self):
        key = '[1,"old","active"]'
        rows = [dict(day=day, key_counts={key: 100}) for day in (10, 10, 11)]
        result = audit.key_support(rows, 3)['old']
        self.assertEqual(result['distinct_day_distribution'], {2: 1})
        self.assertEqual(result['queries_in_keys_at_minimum'], 0)
        self.assertEqual(result['supported_resolved_queries'], 300)


if __name__ == '__main__':
    unittest.main()
