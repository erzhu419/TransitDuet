import unittest

import numpy as np

from audit_freqduet_v245_holding_propagation import (
    edge_accounting, four_way_parts, paired_holding, restore_label, series, unique)
from audit_freqduet_v245_elapsed_components import split


def fixture(times, order=(0, 1, 2, 3), direct=None):
    events, coordinates = [], {}
    for i, (time, trip) in enumerate(zip(times, order)):
        held = 0. if direct is None else direct[i]
        r = dict(trip_id=trip, station_id=7, direction=False, time_s=time)
        events.append(r)
        coordinates[trip, 7, False] = np.array([float(trip*10), held, 0., time-trip*10-held])
    return series(events, coordinates)


class HoldingPropagationTest(unittest.TestCase):
    def test_covariance_allocation_and_mean_normalization(self):
        b = fixture([0., 10., 21., 30.])
        c = fixture([0., 9., 23., 33.], direct=[0., -1., -2., -3.])
        parts = four_way_parts(b, c)
        self.assertAlmostEqual(sum(parts.values()), c['cv2']-b['cv2'])
        self.assertNotEqual(parts['normalization'], 0.)

    def test_changed_neighbours_remain_explicit(self):
        b = fixture([0., 10., 21., 30.])
        c = fixture([0., 11., 22., 33.], order=(0, 2, 1, 3))
        r = edge_accounting(b, c)
        self.assertEqual(r['matched_edges'], 0)
        self.assertEqual(r['added_edges'], 3)
        self.assertEqual(r['removed_edges'], 3)
        self.assertAlmostEqual(r['added_cv2']+r['removed_cv2']+r['normalization'], c['cv2']-b['cv2'])

    def test_identical_paths_have_no_loss(self):
        b = fixture([0., 10., 21., 30.])
        r = edge_accounting(b, b)
        self.assertEqual(r['matched_cv2'], 0.)
        self.assertEqual(r['added_edges'], 0)
        self.assertTrue(all(x == 0 for x in four_way_parts(b, b).values()))

    def test_local_compensation_and_order_crossing_not_confused(self):
        self.assertEqual(restore_label(5, 10, 30)[0], 'cut_removed_local_compensation')
        self.assertEqual(restore_label(5, 30, 10)[0], 'cut_helped_local_gap')
        self.assertEqual(restore_label(15, 30, 10), ('would_change_neighbour_order', None))
        self.assertEqual(restore_label(5, None, 20), ('boundary_unresolved', None))

    def test_native_rebound_not_mislabelled_direct_cut(self):
        ident = dict(bus_id=1)
        bq = {(1, 7, False): dict(issued_s=10, native_bin=2, identity=ident)}
        cq = {(1, 7, False): dict(issued_s=5, native_bin=3, identity=ident)}
        def path(q, actual, direct, native):
            return dict(queries=q, stats=dict(actual_holding_s=actual,
                direct=dict(direct_delta_s=direct),native_requested_s=native))
        r = paired_holding(path(bq, 10, 0, 10), path(cq, 5, -10, 15))
        self.assertEqual(r['actual_delta_s'], -5)
        self.assertEqual(r['direct_delta_s'], -10)
        self.assertEqual(r['native_response_s'], 5)

    def test_duplicate_occurrences_are_not_silently_overwritten(self):
        with self.assertRaises(AssertionError):
            unique([dict(id=1),dict(id=1)], lambda r:r['id'])

    def test_observed_stop_excludes_holding_and_current_stop(self):
        arrivals, departures = [], []
        for trip, shift in enumerate((0, 10, 23)):
            for station, time, holding, dep, dep_hold in ((1, shift+1, 0, shift+6, 2),
                                                        (2, shift+9, 2, shift+13, 3)):
                common = dict(trip_id=trip, station_id=station, direction=False, release=shift)
                arrivals.append(dict(common,time_s=time,realized_holding=holding,other_elapsed=time-shift-holding))
                departures.append(dict(common,time_s=dep,realized_holding=dep_hold,other_elapsed=dep-shift-dep_hold))
        r = split(dict(arrivals=arrivals,departures=departures))
        self.assertAlmostEqual(r['cv2'], np.var([10.,13.,10.,13.])/np.mean([10.,13.,10.,13.])**2)
        self.assertEqual(r['edges'][2,False,0,1]['x'][2], 0.)

    def test_negative_nonholding_stop_is_not_clipped_to_zero(self):
        common = dict(trip_id=0,station_id=1,direction=False,release=0)
        trace = dict(arrivals=[dict(common,time_s=1,realized_holding=0,other_elapsed=1)],
            departures=[dict(common,time_s=2,realized_holding=3,other_elapsed=-1)])
        with self.assertRaises(AssertionError):
            split(trace)


if __name__ == '__main__':
    unittest.main()
