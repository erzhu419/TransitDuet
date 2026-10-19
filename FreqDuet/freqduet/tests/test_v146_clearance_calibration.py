from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from freqduet_clearance_calibration import (
    fit, apply, interval_score, cell_stats, weighted_upper_tail,
)
from run_freqduet_v146_clearance_calibration import DAYS, SAMPLERS, ITEMS, registration


def row(truth=2., finite=True, kind='service', now=100., episode=1):
    return dict(kind=kind, time_s=now, hour=13, episode=[episode, 3, False],
        rank=dict(finite=finite, n=10 if finite else 8, lower_s=4., upper_s=14. if finite else None),
        label_status='resolved', truth_s=truth, support_reason='finite' if finite else 'prefix_less_than_nine',
        covered=False if finite else None, below=True if finite else None, above=False if finite else None,
        v140_s=12., v141_s=12.)


def cells(days=range(9), replicate=False):
    return [dict(seed=1, day=d, mode='old', cutoff=100., rows=
        [row(episode=1)]*(10 if replicate and d == 0 else 1)+[row(truth=10., episode=2)]) for d in days]


class ClearanceCalibrationTests(unittest.TestCase):
    def test_registered_cohorts_and_samplers_are_disjoint(self):
        self.assertEqual([len(v) for v in ITEMS.values()], [768, 768])
        self.assertFalse(set(DAYS['calibration']) & set(DAYS['confirmation']))
        self.assertFalse(set(SAMPLERS['calibration'].values()) & set(SAMPLERS['confirmation'].values()))
        self.assertEqual(registration()['window_s'], 3600.)
        self.assertEqual((registration()['lower_tail'], registration()['upper_tail']), (.1, .1))

    def test_fixed_weighted_tail_has_no_parameter_search(self):
        self.assertEqual(weighted_upper_tail([0., 10.], [.95, .05]), 0.)
        self.assertEqual(weighted_upper_tail([0., 10.], [.5, .5]), 10.)

    def test_only_finite_clearance_service_bounds_change(self):
        model = dict(status='sealed', lower_extension_s=3., upper_extension_s=2.)
        for r in (row(finite=False), row(kind='ready'), row(now=99.)):
            out = apply(r, model, 100.)
            self.assertEqual(out['rank'], r['rank'])
            self.assertFalse(out['calibration_applied'])
        r = row()
        before = deepcopy(r)
        out = apply(r, model, 100.)
        self.assertEqual(r, before)
        self.assertEqual((out['rank']['lower_s'], out['rank']['upper_s']), (1., 16.))
        self.assertTrue(out['covered'])
        self.assertEqual((out['v140_s'], out['v141_s']), (12., 12.))

    def test_lower_clips_at_physical_zero(self):
        self.assertEqual(apply(row(), dict(status='sealed', lower_extension_s=10., upper_extension_s=0.),
            100.)['rank']['lower_s'], 0.)

    def test_episode_query_replication_does_not_change_fit(self):
        a, b = [fit(cells(replicate=v), (1,), ('old',), tuple(range(9))) for v in (False, True)]
        self.assertEqual(a['status'], 'sealed')
        self.assertEqual(a['lower_extension_s'], 2.)
        self.assertEqual(a['upper_extension_s'], 0.)
        self.assertEqual(a['lower_extension_s'], b['lower_extension_s'])

    def test_fewer_than_nine_days_cannot_seal(self):
        self.assertEqual(fit(cells(range(8)), (1,), ('old',), tuple(range(8)))['status'], 'calibration_inconclusive')

    def test_unresolved_finite_labels_block_sealing(self):
        data = cells()
        data[0]['rows'][0]['label_status'] = 'missing_same_vehicle_ready'
        self.assertEqual(fit(data, (1,), ('old',), tuple(range(9)))['status'], 'calibration_inconclusive')

    def test_unsealed_calibration_cannot_be_applied(self):
        with self.assertRaises(AssertionError):
            apply(row(), dict(status='calibration_inconclusive'), 100.)

    def test_proper_interval_score_penalizes_unlimited_width(self):
        r = row()
        r['rank'].update(lower_s=0., upper_s=1000.)
        self.assertEqual(interval_score(r), 1000.)
        r = row()
        self.assertEqual(interval_score(r), 30.)
        new = apply(r, dict(status='sealed', lower_extension_s=2., upper_extension_s=0.), 100.)
        self.assertEqual(interval_score(new), 12.)

    def test_cell_pairing_preserves_abstention_and_empty_scopes(self):
        stats = cell_stats([row(), row(finite=False)],
            dict(status='sealed', lower_extension_s=2., upper_extension_s=0.), 100.)
        a = stats['service_clearance']
        self.assertEqual(a['base']['finite_n'], a['candidate']['finite_n'])
        self.assertEqual(a['candidate']['requests'], 2)
        self.assertEqual(a['paired']['score_delta'], -18.)
        self.assertIsNone(stats['service_active']['paired']['score_delta'])
        self.assertIsNone(stats['ready_all']['candidate']['episode_coverage'])


if __name__ == '__main__':
    unittest.main()
