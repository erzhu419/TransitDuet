import copy
import unittest

from analyze_freqduet_v233_cycle_state import evaluate, label_sample, walk_saved


def fixture():
    meta = dict(native_dt_s=1., native_inventory_order=[8, 2, 5])
    q = dict(snapshot_read_s=10., snapshot_event_seq=3, cycle_ref=[2, 1, 1],
        executed_holding_before_s=5., message_refs={'true': 'initial', 'false': 'initial'},
        pending_plan_version_refs={str(t): 'initial' for t in (10, 11, 12)},
        ready=[dict(bus_id=8, direction=True, position_sample_s=10., return_observed=False,
                    last_observed_return_s=None)],
        active=[dict(bus_id=b, trip_id=t, direction=False, remaining_distance_m=d, position_sample_s=10.)
                for b, t, d in ((2, 1, 200.), (5, 2, 800.))],
        plans=[dict(trip_id=t, direction=True, planned_s=p) for t, p in ((10, 10.), (11, 20.), (12, 30.))])
    second = copy.deepcopy(q)
    second.update(snapshot_event_seq=5, executed_holding_before_s=10.)
    second['message_refs']['true'] = 4
    second['pending_plan_version_refs']['11'] = 4
    second['active'][0]['remaining_distance_m'] = 900.
    second['active'][1]['remaining_distance_m'] = 100.
    kinds = ('launch', 'launch', 'lower_query', 'plan_update', 'lower_query', 'return', 'return', 'launch', 'return')
    times = (0., 1., 10., 10., 10., 20., 21., 22., 30.)
    events = [dict(event_seq=i, kind=k, time_s=t) for i, (k, t) in enumerate(zip(kinds, times), 1)]
    cycles = [dict(native_event_seq=i, kind=k, time_s=t, bus_id=b, trip_id=trip,
                   direction=d, executed_holding_s=h)
              for i, k, t, b, trip, d, h in ((1, 'launch', 0., 2, 1, False, 0.),
                  (2, 'launch', 1., 5, 2, False, 0.), (6, 'return', 20., 5, 2, False, 0.),
                  (7, 'return', 21., 2, 1, False, 15.), (8, 'launch', 22., 2, 9, True, 0.),
                  (9, 'return', 30., 2, 9, True, 0.))]
    versions = [dict(initial=False, native_event_seq=4, changes=[dict(trip_id=11)])]
    return events, cycles, [q, second], versions, meta


class CycleStateAuditTest(unittest.TestCase):
    def test_plan_write_and_rank_change_preserve_the_same_holding_account(self):
        model = walk_saved(*fixture())
        self.assertEqual(model['ledger']['executed_holding_s'], 15.)
        self.assertEqual(model['ledger']['query_increment_s'], 10.)
        self.assertEqual(model['ledger']['after_last_query_s'], 5.)
        self.assertEqual(model['intervals']['write_intervals_carrying_positive_H'], 1)
        self.assertEqual(model['intervals']['association_changed'], 1)
        self.assertEqual(model['intervals']['position_rank_changed'], 1)
        self.assertEqual(model['samples'][0]['association']['trip_id'], 11)
        self.assertEqual(model['samples'][1]['association']['trip_id'], 12)

    def test_first_actual_next_launch_cannot_be_replaced_by_a_later_same_bus_slot(self):
        sample = walk_saved(*fixture())['samples'][0]
        returns = {(2, 1): dict(native_event_seq=7), (5, 2): dict(native_event_seq=6)}
        launch = {11: dict(bus_id=2, native_event_seq=12)}
        result = label_sample(sample, 9, launch, returns)
        self.assertEqual(result['status'], 'mismatch')
        self.assertEqual(result['mismatch_reason'], 'same_bus_but_not_first_next_launch')
        self.assertFalse(result['distance_rank_correct'])
        self.assertEqual(label_sample(sample, None, launch, returns)['status'], 'missing_next_launch')

    def test_equal_clock_returns_keep_native_order_not_input_list_order(self):
        sample = walk_saved(*fixture())['samples'][0]
        returns = {(2, 1): dict(native_event_seq=7, time_s=21.),
                   (5, 2): dict(native_event_seq=6, time_s=21.)}
        result = label_sample(sample, 11, {11: dict(bus_id=2, native_event_seq=8)}, returns)
        self.assertEqual(result['future_return_rank'], 2)

    def test_future_evaluation_never_changes_causal_views_or_zero_fills_final_cycles(self):
        events, cycles, queries, versions, meta = fixture()
        model = walk_saved(events, cycles, queries, versions, meta)
        before = copy.deepcopy(model)
        summary, labels = evaluate(model, [dict(bus_id=2, inbound_trip_id=1, next_trip_id=9)], cycles)
        self.assertEqual(model, before)
        self.assertEqual(summary['all_queries']['mismatch'], 2)
        self.assertEqual(summary['cycles_without_query'], 2)
        self.assertNotIn('actual_next_trip', model['samples'][0])
        self.assertEqual(len(labels), 2)


if __name__ == '__main__':
    unittest.main()
