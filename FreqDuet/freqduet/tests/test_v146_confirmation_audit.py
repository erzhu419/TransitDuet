from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v146_confirmation import check_support, margins


def entry(coverage=.9, score_delta=-2.):
    base = dict(requests=10, finite_n=6, finite_resolved_n=6, finite_unresolved_n=0,
        unresolved_n=0, finite_resolved_episodes=2, label_statuses={'resolved': 10},
        support_reasons={'finite': 6, 'elapsed_survivor_depletion': 4}, episode_coverage=.6)
    return dict(base=base, candidate=dict(base, episode_coverage=coverage), paired=dict(
        base_score=20. if score_delta is not None else None,
        candidate_score=20.+score_delta if score_delta is not None else None, score_delta=score_delta))


def cell(seed, day, coverage=.9, delta=-2.):
    return dict(seed=seed, day=day, stats=dict(service_clearance=entry(coverage, delta)))


class ConfirmationAuditTests(unittest.TestCase):
    def test_coverage_changes_do_not_change_abstention(self):
        check_support(cell(1, 10)['stats'])

    def test_equal_finite_totals_do_not_hide_support_reason_changes(self):
        stats = cell(1, 10)['stats']
        stats['service_clearance']['candidate']['support_reasons'] = {'finite': 6, 'prefix_less_than_nine': 4}
        with self.assertRaisesRegex(AssertionError, 'support or labels'):
            check_support(stats)

    def test_ready_and_active_service_stay_identical(self):
        for scope in ('ready_all', 'service_active'):
            e = entry()
            e['candidate'] = deepcopy(e['base'])
            check_support({scope: e})
            e['candidate']['episode_coverage'] = .99
            with self.assertRaisesRegex(AssertionError, 'negative-control'):
                check_support({scope: e})

    def test_margins_give_cells_equal_weight_on_both_axes(self):
        cells = [cell(1, 10, .9, -2), cell(1, 20, 1., -4),
            cell(2, 10, .7, 2), cell(2, 20, .8, 0)]
        actors, a = margins(cells, 'service_clearance', 'seed')
        days, d = margins(cells, 'service_clearance', 'day')
        self.assertAlmostEqual(actors[0]['candidate_coverage'], .95)
        self.assertAlmostEqual(days[0]['candidate_coverage'], .8)
        self.assertEqual(a['score_delta']['improved_n'], 1)
        self.assertEqual(a['score_delta']['worse_n'], 1)
        self.assertEqual(d['score_delta']['unchanged_n'], 1)
        self.assertAlmostEqual(a['candidate_coverage']['mean'], d['candidate_coverage']['mean'])

    def test_missing_cells_are_not_replaced_by_subset_means(self):
        rows, summary = margins([cell(1, 10), cell(1, 20, None, None), cell(2, 10)],
            'service_clearance', 'seed')
        self.assertEqual(rows[0]['contributing_cells'], 1)
        self.assertIsNone(rows[0]['candidate_coverage'])
        self.assertIsNone(rows[0]['score_delta'])
        self.assertEqual(summary['complete_coverage_groups'], 1)


if __name__ == '__main__':
    unittest.main()
