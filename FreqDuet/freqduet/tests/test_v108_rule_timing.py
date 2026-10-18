from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v108_rule_timing as audit


class RuleTimingTest(unittest.TestCase):
    def events(self, times, offset=0.):
        return [dict(trip_id=i, station_id=2, direction=True, time_s=t + offset,
            release=offset, realized_holding=0., other_elapsed=t) for i, t in enumerate(times)]

    def test_source_order_cohort_and_frozen_grid(self):
        self.assertEqual(audit.SEEDS, (audit.prior.SEEDS[0], next(s for s in audit.prior.SEEDS if s >= 900000)))
        self.assertEqual(len(audit.ITEMS), 288)
        self.assertEqual(len(set(audit.ITEMS)), 288)
        self.assertTrue(set(audit.ITEMS) <= set(audit.prior.ITEMS))
        self.assertEqual({x[1] for x in audit.ITEMS}, set(audit.prior.DAYS))
        self.assertTrue(all(audit.prior.prior.RULES[m][1] == .25 for m in audit.prior.PRIMARY))

    def test_moment_effect_reconstructs_cv_with_negative_covariance(self):
        b = np.array([[100., 10., 0.], [200., -10., 0.], [300., 0., 0.]])
        c = b.copy(); c[:, 1] *= .5
        base, candidate = [audit.timing.moments(v.sum(1), v) for v in (b, c)]
        effect = audit.moment_effect(base, candidate)
        self.assertLess(base['covariance_s2']['realized_holding'], 0.)
        self.assertAlmostEqual(sum(effect[k + '_variance_at_base_mean'] for k in audit.timing.COMPONENTS)
            + effect['mean_normalization'], effect['cv_squared'])

    def test_clock_shift_does_not_manufacture_regularity_change(self):
        base = audit.timing.timing_series(self.events([100., 200., 400.]))
        candidate = audit.timing.timing_series(self.events([100., 200., 400.], 50.))
        self.assertEqual(audit.moment_effect(base, candidate)['cv_squared'], 0.)
        self.assertEqual(audit.moment_effect(base, candidate)['raw_variance_s2'], 0.)

    def test_scopes_use_actual_stop_and_four_hour_support(self):
        events = self.events([100., 200., 14500., 14600.])
        rule = SimpleNamespace(summary={'arrival': audit.timing.timing_series(events)}, arrivals=events)
        scopes = audit.scoped_moments(rule, 'arrival')
        self.assertEqual(set(scopes), {'full', 'band/0', 'band/1', 'stop/2/1'})
        self.assertEqual(scopes['full']['samples'], 3)
        self.assertEqual(sum(scopes[k]['samples'] for k in ('band/0', 'band/1')), 3)

    def test_six_covariance_terms_and_boundaries_reconstruct_full_effect(self):
        def pack(events):
            return dict(coordinates={'arrival': audit.coordinates(events)},
                        series={'arrival': audit.timing.timing_series(events)})
        base = pack(self.events([100., 200., 400.]))
        candidate = pack(self.events([100., 210., 380.]))
        effect = audit.full_effect(base, candidate, 'arrival')
        self.assertAlmostEqual(sum(effect[k] for k in ('release_diagonal', 'holding_diagonal',
            'other_diagonal', 'release_holding_cross', 'release_other_cross', 'holding_other_cross'))
            + effect['mean_normalization'], effect['cv_squared'])
        self.assertAlmostEqual(sum(effect[k] for k in ('last_boundary_mean_s', 'first_boundary_mean_s',
            'count_normalization_mean_s')), effect['mean_headway_s'])

    def test_cache_mismatch_is_rejected_before_passive_accounting(self):
        item = audit.ITEMS[0]
        cached = dict(zip(('seed', 'day', 'method', 'rep'), item))
        cached.update(results={k: 0 for k in (*audit.prior.METRICS, 'action_counts')}, contract={}, coverage={})
        audit.check_cached(cached, cached, item)
        changed = dict(cached, results=dict(cached['results'], cv_squared=1.))
        with self.assertRaises(AssertionError):
            audit.check_cached(changed, cached, item)

    def test_sampler_averaging_precedes_shared_deterministic_comparison(self):
        events = self.events([100., 200., 300.])
        series = audit.timing.timing_series(events)
        rule = SimpleNamespace(summary={'arrival': series}, arrivals=events)
        scopes = audit.scoped_moments(rule, 'arrival')
        values = dict(native=[10., 30., 50.], preserve=[60., 90., 120.],
                      zero=[5.], forward025=[9.], bilateral025=[7.])
        rows = [dict(seed=1, day=2, method=m, rep=rep,
            results=dict(passenger_person_s=value, cv_squared=0.),
            series={k: series for k in audit.KINDS},
            coordinates={k: audit.coordinates(events) for k in audit.KINDS},
            scopes={k: scopes for k in audit.KINDS})
            for m, vs in values.items() for rep, value in enumerate(vs)]
        with patch.object(audit, 'SEEDS', (1,)), patch.object(audit.prior, 'DAYS', (2,)), \
                patch.object(audit.prior, 'METRICS', ('passenger_person_s', 'cv_squared')):
            service, full, _, unmatched = audit.paired_rows(rows)
        self.assertEqual(len(service), 1)
        self.assertEqual(service[0]['effects']['forward025_vs_native']['passenger_person_s'], -21.)
        self.assertEqual(service[0]['effects']['bilateral025_vs_preserve']['passenger_person_s'], -83.)
        self.assertEqual(service[0]['effects']['bilateral025_vs_forward025']['passenger_person_s'], -2.)
        self.assertEqual(full[0]['effects']['forward025_vs_native/arrival']['matched'], 3.)
        self.assertEqual(unmatched, [])


if __name__ == '__main__':
    unittest.main()
