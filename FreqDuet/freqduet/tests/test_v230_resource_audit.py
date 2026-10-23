import copy
import unittest

from audit_freqduet_v230_resource_capture import audit_prefix


def fixture():
    plans = [dict(trip_id=t, direction=d, planned_s=s) for t, d, s in ((0, True, 0.), (1, False, 11.))]
    initial = dict(bus_id=4, direction=True, return_observed=False, last_observed_return_s=None)
    kinds = ('fresh_upper', 'plan_update', 'launch', 'prepare', 'lower_query', 'return',
             'plan_update', 'launch', 'prepare', 'lower_query', 'return')
    times = (0., 0., 0., 2., 3., 10., 11., 11., 13., 14., 20.)
    events = [dict(kind=k, event_seq=i, time_s=t, trip_id=int(i >= 6),
        bus_id=4, station_id=2, direction=i < 6) for i, (k, t) in enumerate(zip(kinds, times))]
    new = dict(plans[1], planned_s=10.)
    callbacks = [dict(trip_id=0, time_s=0., direction=True,
                     changes=[dict(trip_id=1, before=plans[1], after=new)]),
                 dict(trip_id=1, time_s=11., direction=False, changes=[])]
    launches = [dict(trip=t, bus=4, direction=d, actual_s=s)
                for t, d, s in ((0, True, 0.), (1, False, 11.))]
    returns = [dict(trip_id=t, bus_id=4, launch_s=l, return_s=r, holding_s=5.,
        other_elapsed_s=r-l-5., inbound_direction=d, target_direction=not d)
        for t, d, l, r in ((0, True, 0., 10.), (1, False, 11., 20.))]
    legacy = dict(events=events, fresh_upper=[dict(plans=plans, ready=[initial])],
        physical_contract=dict(dt_s=1.), trace=dict(upper_callbacks=callbacks,
        actual_launches=launches, vehicle_returns=returns,
        arrivals=[dict(trip_id=t, station_id=2, direction=d, time_s=s, realized_holding=2.)
                  for t, d, s in ((0, True, 1.), (1, False, 12.))]))
    queries = [dict(legacy_query_event_seq=i, query_s=times[i],
        identity={k: events[i][k] for k in ('bus_id', 'trip_id', 'direction', 'station_id')}) for i in (4, 9)]
    versions = [dict(version_id='initial')]+[dict(version_id=i, initial=False, native_event_seq=i,
        time_s=times[i], direction=c['direction'], callback_trip_id=c['trip_id'],
        fresh_upper_event_seq=0 if i == 1 else None, changes=c['changes'])
        for i, c in zip((1, 6), callbacks)]
    rows = [dict(legacy_query_event_seq=i, snapshot_event_seq=i, snapshot_read_s=times[i],
        plans=[new] if i == 4 else [], pending_plan_version_refs={'1': 1} if i == 4 else {},
        message_refs={'true': 1, 'false': 'initial' if i == 4 else 6},
        cycle_ref=[4, int(i == 9), 2 if i == 4 else 7], executed_holding_before_s=2.,
        ready=[], active=[dict(bus_id=4, trip_id=int(i == 9), direction=i == 4, position_sample_s=times[i])])
        for i in (4, 9)]
    cycles = []
    for launch_seq, return_seq, ret in zip((2, 7), (5, 10), returns):
        base = dict(bus_id=4, trip_id=ret['trip_id'], direction=ret['inbound_direction'],
                    launch_native_event_seq=launch_seq)
        cycles.extend([dict(base, kind='launch', native_event_seq=launch_seq,
                            time_s=times[launch_seq], executed_holding_s=0.),
                       dict(base, kind='return', native_event_seq=return_seq,
                            time_s=times[return_seq], executed_holding_s=5., account=ret)])
    data = dict(metadata=dict(initial_published_plans=plans, native_inventory_order=[4],
        native_timetable_order=[0, 1], native_dt_s=1.), upper_versions=versions, query_extensions=rows,
        cycle_events=cycles, supply_outcomes=[dict(bus_id=4, inbound_trip_id=None,
            inbound_return_event_seq=None, next_trip_id=0, next_launch_event_seq=2, status='initial_stock'),
            dict(bus_id=4, inbound_trip_id=0, inbound_return_event_seq=5,
                 next_trip_id=1, next_launch_event_seq=7, status='first_observed_same_bus_launch')],
        terminal_cycles=[dict(bus_id=4, trip_id=1, status='returned_no_later_launch', native_return_event_seq=10)],
        open_cycles=[])
    return data, dict(queries=queries), legacy


class ResourceAuditTest(unittest.TestCase):
    def test_observed_prefix_and_first_same_bus_cycle(self):
        result = audit_prefix(*fixture())
        self.assertEqual(result['actual_holding_prefixes'], 2)
        self.assertEqual(result['next_same_bus_launches'], 1)
        self.assertEqual(result['cached_callbacks'], 1)

    def test_unchanged_trip_keeps_last_write_not_latest_callback(self):
        data, sidecar, legacy = fixture()
        data['query_extensions'][0]['pending_plan_version_refs']['1'] = 'initial'
        with self.assertRaisesRegex(AssertionError, 'actual last write'):
            audit_prefix(data, sidecar, legacy)

    def test_actual_prefix_cannot_be_replaced_by_final_or_commanded_holding(self):
        data, sidecar, legacy = fixture()
        data['query_extensions'][0]['executed_holding_before_s'] = 5.
        with self.assertRaisesRegex(AssertionError, 'actual arrival prefix'):
            audit_prefix(data, sidecar, legacy)

    def test_same_bus_link_cannot_skip_first_later_launch(self):
        data, sidecar, legacy = fixture()
        data['supply_outcomes'][1]['next_launch_event_seq'] = 10
        with self.assertRaises(AssertionError):
            audit_prefix(data, sidecar, legacy)

    def test_query_cannot_use_unwritten_plan_or_future_message(self):
        for field in ('plans', 'message_refs'):
            data, sidecar, legacy = fixture()
            row = data['query_extensions'][0]
            row[field] = copy.deepcopy(row[field])
            if field == 'plans':
                row[field][0]['planned_s'] = 9.
            else:
                row[field]['false'] = 6
            with self.assertRaises(AssertionError):
                audit_prefix(data, sidecar, legacy)


if __name__ == '__main__':
    unittest.main()
