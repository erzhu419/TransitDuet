import copy
import unittest

from analyze_freqduet_v252_early_supply import evaluate, extract_prefixes, label_tally


def query(seq, time, held, plan=11, kind='future', version='initial', rank=1):
    p = None if plan is None else dict(trip_id=plan, kind=kind, planned_s=20., last_write_version=version)
    own = dict(bus_id=2, inbound_trip_id=1, position_sample_s=time, distance_rank=rank,
               remaining_distance_m=200., posted_lead_s=20.-time, obligation=p)
    return dict(view=dict(cycle_key=[2, 1, 1], query_s=time, native_event_seq=seq,
        executed_holding_s=held, own_association=own,
        target_profile=dict(uncovered_due=[plan] if kind == 'due' else [], position_associations=[own])))


class EarlySupplyAuditTests(unittest.TestCase):
    def test_future_to_due_keeps_same_start_through_write_and_rank_change(self):
        rows = [query(3, 10., 0.), query(5, 15., 5., version=4, rank=2),
                query(7, 21., 15., kind='due', version=4, rank=2)]
        p = extract_prefixes(rows, [dict(initial=False, native_event_seq=4, changes=[11])])
        self.assertEqual(len(p['episodes']), 1)
        e = p['episodes'][0]
        self.assertEqual((e['own_version_changes'], e['rank_changes'], e['crossed_write_intervals']), (1, 1, 1))
        self.assertEqual(e['first_due']['query_s']-e['start']['query_s'], 11.)
        self.assertEqual(e['internal_H_s'], 15.)

    def test_switch_boundary_H_is_not_assigned_to_new_episode(self):
        rows = [query(3, 10., 0.), query(5, 15., 5.), query(7, 20., 15., plan=12),
                query(9, 25., 20., plan=12)]
        p = extract_prefixes(rows, [])
        a, b = p['episodes']
        self.assertEqual(a['last']['query_s'], 15.)
        self.assertEqual((a['internal_H_s'], b['internal_H_s']), (5., 5.))
        boundary = [r for r in p['intervals'] if r['kind'] == 'association_boundary']
        self.assertEqual(boundary[0]['H_delta_s'], 10.)
        self.assertEqual(a['end_reason'], 'reassociated')

    def test_regained_plan_is_not_joined_to_old_early_start(self):
        p = extract_prefixes([query(3, 10., 0.), query(5, 15., 5., plan=None),
            query(7, 20., 15.), query(9, 25., 20., kind='due')], [])
        a, b = p['episodes']
        self.assertIsNone(a['first_due'])
        self.assertEqual(b['first_due']['query_s']-b['start']['query_s'], 5.)
        c = p['cycles'][0]
        self.assertNotEqual(c['first_link_episode'], c['first_due_episode'])
        self.assertEqual(c['first_link']['query_s'], 10.)

    def test_never_due_and_unassociated_rows_are_retained(self):
        p = extract_prefixes([query(3, 10., 0., plan=None), query(5, 15., 5.),
                              query(7, 20., 15., plan=None)], [])
        self.assertEqual(len(p['samples']), 3)
        self.assertIsNone(p['cycles'][0]['first_due'])
        self.assertEqual(p['episodes'][0]['end_reason'], 'lost')

    def test_future_outcomes_and_future_writes_do_not_change_prefix(self):
        rows = [query(3, 10., 0.), query(5, 15., 5.)]
        before = copy.deepcopy(rows)
        a = extract_prefixes(rows, [])
        for row in rows:
            row['view'].update(actual_next_trip=999, actual_return_s=9999.)
        b = extract_prefixes(rows, [dict(initial=False, native_event_seq=100, changes=[999])])
        self.assertEqual(a, b)
        self.assertNotIn('actual_next_trip', before[0]['view'])

    def test_prefix_extraction_does_not_mutate_inputs(self):
        rows = [query(3, 10., 0.), query(5, 15., 5., plan=12)]
        before = copy.deepcopy(rows)
        extract_prefixes(rows, [])
        self.assertEqual(rows, before)

    def test_unknown_and_mismatch_labels_keep_their_denominators(self):
        def row(status, reason=None, rank=None):
            return dict(status=status, mismatch_reason=reason, distance_rank_correct=rank)
        t = label_tally([row('match', rank=True), row('mismatch', 'predicted_slot_served_by_other_bus'),
                         row('missing_next_launch'), row('labelled_without_association')])
        self.assertEqual(t['samples'], 4)
        self.assertEqual(t['match_fraction'], .5)
        self.assertEqual(t['distance_rank_labelled'], 1)
        self.assertEqual(t['status']['missing_next_launch'], 1)

    def test_actual_H_closes_with_return_tail_and_zero_query_cycle(self):
        queries = [query(3, 10., 0.), query(5, 21., 5., kind='due')]
        capture = dict(terminal=dict(open_cycles=[]), cycles=[
            dict(cycle_key=[2, 1, 1], status='returned', executed_holding_s=10.),
            dict(cycle_key=[2, 11, 7], status='returned', executed_holding_s=0.)],
            observed_links=[dict(previous=dict(cycle_key=[2, 1, 1]), current_cycle=[2, 11, 7])])
        resource = dict(supply_outcomes=[dict(bus_id=2, inbound_trip_id=1, next_trip_id=11)],
            cycle_events=[dict(kind='launch', trip_id=1, bus_id=2, native_event_seq=1),
                dict(kind='return', trip_id=1, bus_id=2, native_event_seq=6),
                dict(kind='launch', trip_id=11, bus_id=2, native_event_seq=7),
                dict(kind='return', trip_id=11, bus_id=2, native_event_seq=8)],
            query_extensions=[dict(snapshot_event_seq=seq, cycle_ref=[2, 1, 1],
                shadow_frontier=dict(native_bin=1, status='supported',
                    candidates=[dict(holding_s=0, local_gap_cost_delta_s2=10.)])) for seq in (3, 5)])
        prefix = extract_prefixes(queries, [])
        before = copy.deepcopy(prefix)
        result, _ = evaluate(prefix, capture, resource)
        self.assertEqual(result['cycles']['no_query'], 1)
        self.assertEqual(result['cycles']['returned_without_next_launch'], 1)
        self.assertEqual(result['H_ledger']['total_s'], 10.)
        self.assertEqual(result['H_ledger']['query_return_tail_s'], 5.)
        self.assertEqual(result['future_to_due_same_episode']['actual_prefix_H_s']['sum'], 5.)
        self.assertEqual(result['labels']['first_cycle']['status']['match'], 1)
        self.assertEqual(result['query_doses']['future']['certified_executed_H_s'], 5.)
        self.assertEqual(prefix, before)


if __name__ == '__main__':
    unittest.main()
