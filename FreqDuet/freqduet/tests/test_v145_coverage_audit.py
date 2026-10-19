from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v145_coverage import margins, hour_contrast, add, pool_summary, labels


def cell(seed, day, a=.7, b=.86):
    return dict(seed=seed, day=day, strata=dict(service={
        scope: dict(episode_coverage=value, finite_fraction=.8 if value is not None else None)
        for scope, value in (('hour13', a), ('rest', b))}))


class CoverageAuditTests(unittest.TestCase):
    def test_axis_margins_do_not_fill_missing_cells(self):
        rows, summary = margins([cell(1, 10), cell(1, 20, None), cell(2, 10), cell(2, 20)],
            'service', 'hour13', 'seed')
        self.assertEqual(rows[0]['contributing_cells'], 1)
        self.assertAlmostEqual(rows[0]['episode_coverage'], .7)
        self.assertEqual(summary['complete_groups'], 1)
        self.assertEqual(summary['below_target_ids'], [2])

    def test_paired_hour_contrast_keeps_crossed_grid(self):
        stats = hour_contrast([cell(s, d) for s in (1, 2) for d in (10, 20)], 'service')
        self.assertAlmostEqual(stats['mean'], -.16)
        self.assertAlmostEqual(stats['ci95'][0], -.16)

    def test_missing_hour_cell_has_no_replacement_bootstrap(self):
        stats = hour_contrast([cell(1, 10), cell(1, 20, None)], 'service')
        self.assertEqual(stats['paired_cells'], 1)
        self.assertIsNone(stats['ci95'])
        self.assertIsNone(stats['mean'])

    def test_unsupported_and_unresolved_do_not_become_coverage_failures(self):
        bucket = Counter()
        row = dict(pool_n=10, survivor_n=8, rank=dict(finite=False), label_status='resolved')
        add(bucket, row)
        add(bucket, dict(row, rank=dict(finite=True), label_status='missing_departure_tail'))
        stats = pool_summary(bucket)
        self.assertEqual((stats['requests'], stats['finite_n'], stats['finite_resolved_n']), (2, 1, 0))
        self.assertIsNone(stats['query_coverage'])

    def test_miss_direction_distance_and_support_are_separate(self):
        row = dict(pool_n=10, survivor_n=10, rank=dict(finite=True, n=10, lower_s=4., upper_s=15.),
            label_status='resolved', covered=False, below=True, above=False, truth_s=1.,
            hour=13, support_reason='finite', episode=[1, 2, False])
        bucket = Counter()
        add(bucket, row)
        stats = pool_summary(bucket)
        self.assertEqual(stats['mean_lower_shortfall_s'], 3.)
        self.assertEqual(stats['pool_depleted_n'], 0)
        self.assertIn('hour13/rank_support/9-18', labels(row))
        self.assertIn('hour13/below', labels(row))


if __name__ == '__main__':
    unittest.main()
