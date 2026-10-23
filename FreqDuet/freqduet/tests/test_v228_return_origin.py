import copy
import unittest

from analyze_freqduet_v228_return_origin import compare, trip_difference, trip_times


def fixture():
    def event(stop, t, h):
        return dict(trip_id=0, station_id=stop, direction=True, time_s=t,
                    release=10., realized_holding=h, other_elapsed=t-10.-h)
    trace = dict(arrivals=[event(1, 20., 0.), event(2, 50., 5.)],
                 departures=[event(1, 30., 5.), event(2, 60., 10.)])
    chain = dict(launches={0: dict(bus=7, direction=True, actual_s=10.)}, parents={0: None},
                 returns={0: dict(bus_id=7, launch_s=10., return_s=70., holding_s=10., other_elapsed_s=50.)})
    return dict(trace=trace, ledger=dict(executed_holding_s=10.)), chain


class ReturnOriginTest(unittest.TestCase):
    def test_exact_partition(self):
        d, s = fixture()
        t = trip_times(d, s)[0]['totals']
        self.assertEqual([t[k] for k in ('holding_s', 'nonholding_stop_s', 'interstop_nonholding_s',
            'launch_boundary_nonholding_s', 'return_boundary_nonholding_s')], [10., 10., 20., 10., 10.])

    def test_missing_departure_is_not_zero(self):
        d, s = fixture()
        d['trace']['departures'].pop()
        with self.assertRaisesRegex(AssertionError, 'unmatched'):
            trip_times(d, s)

    def test_repeated_stop_not_silently_overwritten(self):
        d, s = fixture()
        d['trace']['arrivals'].append(d['trace']['arrivals'][0])
        with self.assertRaisesRegex(AssertionError, 'repeated'):
            trip_times(d, s)

    def test_negative_stop_interval_rejected(self):
        d, s = fixture()
        d['trace']['departures'][0]['realized_holding'] = 11.
        d['trace']['departures'][0]['other_elapsed'] = 9.
        with self.assertRaisesRegex(AssertionError, 'negative'):
            trip_times(d, s)

    def test_unobserved_holding_remains_explicit(self):
        d, s = fixture()
        d['trace']['arrivals'][1]['realized_holding'] = 6.
        d['trace']['arrivals'][1]['other_elapsed'] -= 1.
        t = trip_times(d, s)[0]['totals']
        self.assertEqual(t['holding_outside_stop_s'], 1.)
        self.assertEqual(t['interstop_nonholding_s'], 19.)

    def test_inherited_launch_is_not_new_duration(self):
        d, s = fixture()
        b = trip_times(d, s)[0]
        c = copy.deepcopy(b)
        c['launch_s'] += 3.; c['return_s'] += 3.
        for p in c['stops']:
            p['arrival_s'] += 3.; p['departure_s'] += 3.
        r = trip_difference(c, b)
        self.assertEqual(r['new_duration_delta_s'], 0.)
        self.assertEqual(r['first_positive_clock']['kind'], 'launch')
        self.assertIsNone(r['first_increase'])

    def test_local_stop_delay_propagates(self):
        d, s = fixture()
        b = trip_times(d, s)[0]
        c = copy.deepcopy(b)
        c['totals']['nonholding_stop_s'] += 2.
        c['stops'][0]['stop_nonholding_s'] += 2.
        c['stops'][0]['departure_s'] += 2.
        c['stops'][1]['arrival_s'] += 2.; c['stops'][1]['departure_s'] += 2.
        c['return_s'] += 2.
        r = trip_difference(c, b)
        self.assertEqual(r['new_duration_delta_s'], 2.)
        self.assertEqual(r['first_positive_clock'], dict(kind='departure', station_id=1, offset_s=2.))

    def test_same_path_has_no_delay_origin(self):
        d, s = fixture()
        t = trip_times(d, s)
        row, detail = compare(t, t, s, s)
        self.assertEqual(row['buses_with_positive_return'], 0)
        self.assertEqual(row['delayed_same_chain_launches']['count'], 0)
        self.assertEqual(len(detail['physical_chain_prefixes']), 1)


if __name__ == '__main__':
    unittest.main()
