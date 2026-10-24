import copy
import unittest

from analyze_freqduet_v250_compensation_supply import (
    PARTS, component_summary, observable_queries, prepare_account, supply_labels)


def fixture():
    identity = dict(bus_id=2, trip_id=22, station_id=7, direction=False)
    parts = dict(zip(PARTS, (100., 20., -20.)))
    p = dict(identity=identity, cycle_key=[2, 22, 3], prepare_s=1000.,
        prepare_seq=20, prepare_order=1200, forward_s=100., backward_s=140.,
        own=dict(launch_s=100., launch_order=1002, holding_s=60.),
        front=dict(launch_s=0., departure_s=900., departure_order=1100, holding_s=40.),
        front_identity=dict(bus_id=1, trip_id=21, direction=False), parts=parts)
    join = dict(prepare_seq=20, prepare_order=1200, query_seq=21, query_order=1210,
        query_occurrence=0, prepare_s=1000., query_s=1010., prepare_holding_s=60., query_holding_s=65.)
    timing = dict(prepares=[dict(mapped=p, change=dict(status='first_supported_prepare', parts=None))],
                  queries=[dict(prepare_index=0, join=join)])
    q = dict(identity=identity, legacy_query_event_seq=21, prepare_order=1200,
        query_order=1210, query_occurrence=0, query_s=1010., native_bin=2,
        feasible_bins=list(range(7)))
    view = dict(cycle_key=[2, 22, 3], query_s=1010., native_event_seq=21,
        executed_holding_s=65., own_association=dict(obligation=dict(trip_id=30, kind='due')),
        target_profile=dict(uncovered_due=[30]))
    extra = dict(q, snapshot_read_s=1010., terminal_masked=False, onboard=3.)
    return timing, dict(queries=[q]), dict(query_extensions=[extra]), dict(queries=[dict(q, view=view)])


class CompensationSupplyTests(unittest.TestCase):
    def test_distinct_clocks_and_holdings(self):
        summary, rows, _ = observable_queries(*fixture())
        self.assertEqual(summary['prepare_to_query_H_increases'], 1)
        self.assertEqual((rows[0]['prepare_H_s'], rows[0]['H_s']), (60., 65.))

    def test_wrong_query_join_fails(self):
        args = fixture()
        args[0]['queries'][0]['join']['query_seq'] = 1210
        with self.assertRaises(AssertionError):
            observable_queries(*args)

    def test_future_association_does_not_hide_other_due(self):
        args = fixture()
        args[3]['queries'][0]['view']['own_association']['obligation'] = dict(trip_id=31, kind='future')
        summary, rows, first = observable_queries(*args)
        self.assertEqual(rows[0]['group'], 'other_due')
        self.assertEqual(summary['own_future_with_other_due'], 1)
        self.assertFalse(first)

    def test_compensation_against_native_not_zero(self):
        _, rows, _ = observable_queries(*fixture())
        self.assertGreater(rows[0]['zero_loss_s2'], 0)
        self.assertFalse(rows[0]['nonharm_cut'])
        self.assertEqual(len(rows[0]['frontier']['candidates']), 7)
        self.assertIsNone(rows[0]['frontier']['selected_bin'])

    def test_mask_and_missing_rear_not_nonharm(self):
        for mode in ('mask', 'rear'):
            args = fixture()
            if mode == 'mask':
                args[2]['query_extensions'][0]['terminal_masked'] = True
            else:
                args[0]['prepares'][0]['mapped']['backward_s'] = None
            _, rows, _ = observable_queries(*args)
            self.assertIsNone(rows[0]['nonharm_cut'])
            self.assertIsNone(rows[0]['zero_loss_s2'])

    def test_zero_action_and_missing_zero_candidate_distinct(self):
        args = fixture()
        args[1]['queries'][0]['native_bin'] = 0
        summary, _, _ = observable_queries(*args)
        self.assertEqual(summary['all_queries']['positive'], 0)
        args = fixture()
        args[1]['queries'][0]['feasible_bins'] = [2, 3]
        _, rows, _ = observable_queries(*args)
        self.assertIsNone(rows[0]['zero_loss_s2'])
        self.assertIsNone(rows[0]['nonharm_cut'])

    def test_components_and_covariance_close(self):
        result = component_summary([dict(gap_s=100., **dict(zip(PARTS, (100., 20., -20.)))),
                                    dict(gap_s=130., **dict(zip(PARTS, (120., 40., -30.))))])
        self.assertEqual(sum(result['mean_s'].values()), 115.)
        self.assertAlmostEqual(sum(result['covariance_with_gap_s2'].values()), result['variance_s2'])

    def test_prepare_ledger_includes_zero_query_cycle(self):
        timing = fixture()[0]
        p = copy.deepcopy(timing['prepares'][0]['mapped'])
        p.update(prepare_s=1100., prepare_seq=22, prepare_order=1300)
        p['own']['holding_s'] = 70.
        p['front'].update(departure_s=1000., departure_order=1250, holding_s=50.)
        timing['prepares'].append(dict(mapped=p,
            change=dict(status='same_physical_cycle_and_front', parts=dict.fromkeys(PARTS, 0.))))
        cycle = dict(cycle_key=[2, 22, 3], executed_holding_s=80., queries=0)
        summary, _ = prepare_account(timing, [cycle])
        self.assertEqual(summary['H_ledger'], dict(cycles=1, first_prepare_prefix_s=60.,
            cross_prepare_increment_s=10., return_tail_s=10., total_s=80.))
        p['own']['holding_s'] = 0.
        with self.assertRaises(AssertionError):
            prepare_account(timing, [cycle])

    def test_future_labels_are_separate_and_missing_is_explicit(self):
        args = fixture()
        before = copy.deepcopy(args)
        _, rows, _ = observable_queries(*args)
        cycle = dict(trip_id=22, bus_id=2, next_trip=None, holding_s=80.,
                     next_public_delay_s=None, return_s=1200.)
        model = dict(cycles={22: cycle})
        data = dict(trace=dict(actual_launches=[]), launch_plans={})
        resource = dict(metadata=dict(native_dt_s=10.))
        labels = supply_labels(rows, model, data, resource)
        self.assertEqual(labels[0]['status'], 'no_next_launch')
        self.assertIsNone(labels[0]['postreturn_wait_s'])
        cycle.update(next_trip=30, next_public_delay_s=20.)
        data['trace']['actual_launches'] = [dict(trip=30, actual_s=1210., eligible_s=1190.)]
        data['launch_plans']['30'] = 1190.
        self.assertEqual(supply_labels(rows, model, data, resource)[0]['status'], 'match')
        cycle['next_trip'] = 31
        data['trace']['actual_launches'][0]['trip'] = 31
        data['launch_plans']['31'] = 1190.
        self.assertEqual(supply_labels(rows, model, data, resource)[0]['status'], 'mismatch')
        self.assertEqual(args, before)
        self.assertEqual(observable_queries(*args)[1], rows)


if __name__ == '__main__':
    unittest.main()
