import unittest

import preregister_freqduet_v215_positive_event as v


def event():
    return dict(time_s=7500., event_seq=100, snapshot_read_s=7500., snapshot_event_seq=100,
        state_captured_s=7499., bus_id=9, trip_id=31, direction=False, station_id=3,
        terminal_masked=False, forward_valid=True, follower_valid=True, base_bin=3, feasible_bins=list(range(7)),
        ready=[], active=[dict(bus_id=b, trip_id=t, direction=False, remaining_distance_m=d)
                          for b, t, d in ((8, 29, 500.), (9, 31, 1000.), (10, 33, 2000.))],
        plans=[dict(trip_id=i, direction=True, launched=False, planned_s=7400.) for i in (40, 42)])


def upper():
    q = event()
    q.update(time_s=7501., event_seq=110, snapshot_read_s=7501., snapshot_event_seq=110,
             direction=True, fresh_upper=True)
    return q


class PositiveEventTests(unittest.TestCase):
    def test_positive_support_precedes_distance_and_does_not_lock_nearest_vehicle(self):
        a = v.lower_anchor(event())
        self.assertEqual((a['bus_id'], a['base_bin'], a['smaller_bin'], a['inbound_rank']), (9, 3, 2, 2))
        for key, value in (('base_bin', 0), ('forward_valid', False), ('follower_valid', False),
                           ('terminal_masked', True), ('time_s', 8100.)):
            q = event()
            q[key] = value
            self.assertIsNone(v.lower_anchor(q))

    def test_observed_queue_and_vehicle_availability_both_required(self):
        for change in ('ready', 'wrong_direction', 'not_due', 'launched', 'insufficient_queue'):
            q = event()
            if change == 'ready': q['ready'] = [dict(direction=True)]
            if change == 'wrong_direction':
                for p in q['plans']: p['direction'] = False
            if change == 'not_due':
                for p in q['plans']: p['planned_s'] = 7600.
            if change == 'launched':
                for p in q['plans']: p['launched'] = True
            if change == 'insufficient_queue': q['plans'] = q['plans'][:1]
            self.assertIsNone(v.lower_anchor(q), change)

    def test_missing_or_stale_query_snapshot_abstains_without_backfill(self):
        for key in ('ready', 'active', 'plans', 'state_captured_s', 'event_seq'):
            q = event()
            del q[key]
            self.assertIsNone(v.lower_anchor(q))
        for key, value in (('snapshot_read_s', 7499.), ('snapshot_event_seq', 99), ('state_captured_s', 7501.)):
            q = event()
            q[key] = value
            self.assertIsNone(v.lower_anchor(q))

    def test_future_return_and_assignment_never_select_event(self):
        q = event()
        a = v.lower_anchor(q)
        q['future_return_s'], q['future_assigned_trip'] = 999999., 999
        for b in q['active']: b['future_eta_s'] = -b['remaining_distance_m']
        self.assertEqual(v.lower_anchor(q), a)

    def test_same_vehicle_next_trip_is_not_same_inbound_resource(self):
        q = event()
        q['active'][1]['trip_id'] = 99
        self.assertIsNone(v.lower_anchor(q))

    def test_upper_is_native_future_event_not_reuse_or_past_override(self):
        a = v.lower_anchor(event())
        self.assertTrue(v.upper_followup(a, upper()))
        for key, value in (('fresh_upper', False), ('direction', False), ('time_s', 8400.)):
            q = upper()
            q[key] = value
            self.assertFalse(v.upper_followup(a, q))
        q = upper()
        q.update(time_s=7500., snapshot_read_s=7500., event_seq=99, snapshot_event_seq=99)
        self.assertFalse(v.upper_followup(a, q))
        q.update(event_seq=101, snapshot_event_seq=101)
        self.assertTrue(v.upper_followup(a, q))

    def test_upper_missing_returned_resource_or_original_queue_cannot_be_invented(self):
        a = v.lower_anchor(event())
        for change in ('returned', 'reused', 'queue_served', 'ready'):
            q = upper()
            if change == 'returned': q['active'].pop(1)
            if change == 'reused': q['active'][1]['trip_id'] = 99
            if change == 'queue_served':
                for p in q['plans']: p['trip_id'] += 100
            if change == 'ready': q['ready'] = [dict(direction=True)]
            self.assertFalse(v.upper_followup(a, q), change)

    def test_new_days_and_future_banks_are_separate_from_v214(self):
        p = v.registration('test')
        select = {v.future_seed(c, f) for c in range(4) for f in p['selection_futures']}
        evaluate = {v.future_seed(c, f) for c in range(4) for f in p['evaluation_futures']}
        self.assertFalse(select & evaluate)
        self.assertFalse((select | evaluate) & {2159000000+c for c in range(4)})
        self.assertFalse({c[1] for c in v.CONTEXTS} & {2061001, 2061004})
        self.assertEqual(len(select)*len(v.CANDIDATES), p['budget']['selection_max'])
        self.assertEqual((p['new_rollouts'], p['training_updates'], p['main_policy_changed']), (0, 0, False))


if __name__ == '__main__':
    unittest.main()
