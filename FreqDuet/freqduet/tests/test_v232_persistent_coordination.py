import copy
import unittest

from preregister_freqduet_v232_persistent_coordination import CycleState, registration, resource_profile


def snapshot():
    meta = dict(native_dt_s=1., native_inventory_order=[8, 2, 5], native_timetable_order=[10, 11, 12])
    row = dict(snapshot_read_s=10., snapshot_event_seq=3, cycle_ref=[2, 1, 1],
        executed_holding_before_s=5., message_refs={'true': 'initial', 'false': 'initial'},
        pending_plan_version_refs={'10': 'initial', '11': 'initial', '12': 'initial'},
        ready=[dict(bus_id=8, direction=True, position_sample_s=10., return_observed=False,
                    last_observed_return_s=None)], active=[dict(bus_id=b, trip_id=t, direction=False,
                    remaining_distance_m=d, position_sample_s=10.) for b, t, d in ((2, 1, 200.), (5, 2, 800.))],
        plans=[dict(trip_id=t, direction=True, planned_s=p, public_s=p, target_s=10.)
               for t, p in ((10, 10.), (11, 20.), (12, 30.))])
    return row, meta


class PersistentCoordinationTest(unittest.TestCase):
    def test_current_ready_is_spent_once_and_future_is_not_due_shortage(self):
        row, meta = snapshot()
        p = resource_profile(row, meta)['directions']['true']
        self.assertEqual(p['ready_coverage'], [dict(trip_id=10, bus_id=8, kind='due')])
        self.assertEqual(p['uncovered_due'], [])
        self.assertEqual(p['future_without_current_ready'], [11, 12])
        self.assertEqual([r['obligation']['trip_id'] for r in p['position_associations']], [11, 12])

    def test_incoming_positions_never_back_ready_advance(self):
        row, meta = snapshot()
        row['ready'] = []
        p = resource_profile(row, meta)['directions']['true']
        self.assertEqual(p['ready_coverage'], [])
        self.assertEqual(p['uncovered_due'], [10])
        self.assertEqual(p['position_associations'][0]['obligation']['trip_id'], 10)

    def test_distance_ties_follow_original_inventory_order(self):
        row, meta = snapshot()
        for b in row['active']:
            b['remaining_distance_m'] = 200.
        meta['native_inventory_order'] = [8, 5, 2]
        p = resource_profile(row, meta)['directions']['true']
        self.assertEqual([b['bus_id'] for b in p['position_associations']], [5, 2])

    def test_current_tick_return_covers_next_pass(self):
        row, meta = snapshot()
        row['ready'][0].update(return_observed=True, last_observed_return_s=10.)
        p = resource_profile(row, meta)
        self.assertEqual(p['next_dispatch_s'], 11.)
        self.assertEqual(p['directions']['true']['ready_coverage'][0]['bus_id'], 8)

    def test_unmatched_obligations_keep_their_actual_plan_version(self):
        row, meta = snapshot()
        row['active'] = row['active'][:1]
        row['pending_plan_version_refs']['12'] = 2
        p = resource_profile(row, meta)['directions']['true']
        self.assertEqual(len(p['position_associations']), 1)
        self.assertEqual(p['unbacked_obligations'][-1]['trip_id'], 12)
        self.assertEqual(p['unbacked_obligations'][-1]['last_write_version'], 2)

    def test_query_rejects_future_version_or_late_position_samples(self):
        for problem in ('future_version', 'late_position'):
            state = CycleState()
            state.launch(2, 1, False, 1, 0.)
            row, meta = snapshot()
            if problem == 'future_version':
                row['message_refs']['true'] = 4
            else:
                row['active'][0]['position_sample_s'] = 11.
            with self.assertRaises(AssertionError):
                state.query(row, meta)

    def test_replans_and_cleared_pressure_do_not_reset_physical_holding(self):
        state = CycleState()
        state.launch(2, 1, False, 1, 0.)
        row, meta = snapshot()
        first = state.query(row, meta)
        changed = copy.deepcopy(row)
        changed.update(snapshot_event_seq=5, executed_holding_before_s=10.,
                       message_refs={'true': 4, 'false': 'initial'})
        changed['plans'] = changed['plans'][:1]
        changed['pending_plan_version_refs'] = {'10': 4}
        second = state.query(changed, meta)
        self.assertEqual(second['executed_holding_s'], 10.)
        self.assertEqual(second['delta_executed_since_last_sample_s'], 5.)
        self.assertIsNone(second['own_association']['obligation'])
        self.assertEqual(first['own_association']['obligation']['trip_id'], 11)
        with self.assertRaisesRegex(AssertionError, 'holding prefix reset'):
            state.query(dict(changed, snapshot_event_seq=6, executed_holding_before_s=0.), meta)

    def test_only_actual_next_launch_opens_new_cycle_and_observed_parent(self):
        state = CycleState()
        state.launch(2, 1, False, 1, 0.)
        row, meta = snapshot()
        before = state.query(row, meta)
        state.returned_to_terminal(2, 1, 4, 20., 15.)
        self.assertNotIn(2, state.active)
        self.assertEqual(state.returned[2]['executed_holding_s'], 15.)
        opened = state.launch(2, 9, True, 5, 21.)
        self.assertEqual(opened['observed_previous_cycle']['cycle_key'], [2, 1, 1])
        self.assertEqual(state.active[2]['executed_holding_s'], 0.)
        self.assertEqual(before['cycle_key'], [2, 1, 1])
        self.assertNotIn('next_trip', before)

    def test_protocol_opens_no_execution_or_fit_budget(self):
        p = registration('test')
        self.assertEqual(set(p['budget'].values()), {0})
        self.assertFalse(p['runtime_connected'] or p['selector_defined'] or p['replay_authorized'])
        self.assertEqual(p['frozen']['fleet'], 12)
        self.assertEqual(p['frozen']['holding_bins_s'], [0, 5, 10, 15, 20, 30, 45])


if __name__ == '__main__':
    unittest.main()
