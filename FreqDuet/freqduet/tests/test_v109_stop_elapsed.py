from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v109_stop_elapsed as audit


class StopElapsedTest(unittest.TestCase):
    def records(self):
        arrivals, departures = [], []
        for trip, launch in enumerate((100., 220., 420.)):
            for station, offset in ((1, 0.), (2, 100.)):
                held, stop = (0., 0.) if station == 1 else (5. * trip, 10. + trip)
                time = launch + offset + held + stop
                common = dict(trip_id=trip, station_id=station, direction=True, release=launch)
                arrivals.append(dict(common, time_s=time, realized_holding=held, other_elapsed=time-launch-held))
                departures.append(dict(common, time_s=time+held+10.+trip,
                    realized_holding=2*held, other_elapsed=time+10.+trip-launch-held))
        return dict(arrivals=sorted(arrivals, key=lambda r: r['time_s']),
                    departures=sorted(departures, key=lambda r: r['time_s']))

    def test_scope_identity_preserves_parent_and_support(self):
        records = self.records()
        split = audit.elapsed.split_stop_time(records)
        blocks = audit.scoped_series(records, split, 'arrival')
        self.assertEqual(set(blocks), {'full', 'band/0', 'stop/1/1', 'stop/2/1'})
        for scope, events in (('full', records['arrivals']),
                              ('stop/2/1', [r for r in records['arrivals'] if r['station_id'] == 2])):
            audit.assert_parent(blocks[scope], audit.prior.timing.timing_series(events))
        self.assertEqual(sum(blocks[k]['samples'] for k in ('stop/1/1', 'stop/2/1')), blocks['full']['samples'])

    def test_parent_allocation_mismatch_is_rejected(self):
        records = self.records()
        measured = audit.scoped_series(records, audit.elapsed.split_stop_time(records), 'arrival')['full']
        parent = audit.prior.timing.timing_series(records['arrivals'])
        parent['covariance_s2']['other_elapsed'] += 1.
        with self.assertRaises(AssertionError):
            audit.assert_parent(measured, parent)

    def test_clock_band_keeps_edge_crossing_the_boundary(self):
        records = dict(arrivals=[dict(trip_id=i, station_id=1, direction=True, time_s=t,
            release=t, realized_holding=0., other_elapsed=0.) for i, t in enumerate((100., 200., 14500.))])
        records['departures'] = list(records['arrivals'])
        scopes = audit.scoped_series(records, audit.elapsed.split_stop_time(records), 'arrival')
        self.assertEqual(scopes['band/1']['samples'], 1)
        self.assertEqual(scopes['band/1']['mean'], 14300.)

    def test_split_cross_terms_reconstruct_other_elapsed_contribution(self):
        b = np.array([[100., 10., 5., 0.], [200., -10., 0., 5.], [300., 0., 5., 0.]])
        c = b.copy(); c[:, 1] *= .5
        base, candidate = [audit.elapsed.covariance_moments(v.sum(1), v) for v in (b, c)]
        effect = audit.moment_effect(base, candidate)
        self.assertAlmostEqual(sum(effect[k + '_allocated'] for k in audit.elapsed.COMPONENTS)
            + effect['mean_normalization'], effect['cv_squared'])
        self.assertAlmostEqual(sum(v for k, v in effect.items() if k.endswith(('_cross', '_diagonal')))
            + effect['mean_normalization'], effect['cv_squared'])
        self.assertLess(base['covariance'][0, 1], 0.)

    def test_unchanged_paths_and_sampler_averaging_have_zero_effect(self):
        records = self.records()
        scopes = {k: audit.scoped_series(records, audit.elapsed.split_stop_time(records), k)
                  for k in audit.prior.KINDS}
        paths = [dict(method=m, rep=rep, scopes=scopes) for m in audit.prior.prior.METHODS
                 for rep in (range(3) if m in ('native', 'preserve') else (0,))]
        full, local = audit.compare_paths(paths)
        self.assertTrue(all(v['cv_squared'] == 0. for v in full.values()))
        self.assertTrue(all(v['cv_squared'] == 0. for blocks in local.values() for v in blocks.values()))
        self.assertEqual(audit.average_effects([dict(x=1.), dict(x=2.), dict(x=6.)]), dict(x=3.))

    def test_review_keeps_all_consumed_actor_days(self):
        self.assertEqual(len(audit.CELLS), 32)
        self.assertEqual({s for s, _ in audit.CELLS}, set(audit.prior.SEEDS))
        self.assertEqual({d for _, d in audit.CELLS}, set(audit.prior.prior.DAYS))


if __name__ == '__main__':
    unittest.main()
