import unittest

from analyze_freqduet_v237_persistent_supply import (
    fixed_neighbour, next_clock, resource_class, supply_tally, version_intervals)


def view(seq, holding, delta, due=False):
    return dict(view=dict(cycle_key=[1, 10, 0], native_event_seq=seq,
        executed_holding_s=holding, delta_executed_since_last_sample_s=delta,
        target_profile=dict(uncovered_due=[20] if due else []), own_association=dict(obligation=None)))


class PersistentSupplyTest(unittest.TestCase):
    def test_pressure_is_current_own_due_not_just_any_future_unbacked_plan(self):
        row = dict(own_association=dict(obligation=dict(trip_id=20, kind='future')),
                   target_profile=dict(uncovered_due=[]))
        self.assertEqual(resource_class(row), 'future_unbacked')
        row['target_profile']['uncovered_due'] = [19]
        self.assertEqual(resource_class(row), 'other_due')
        row['own_association']['obligation'] = dict(trip_id=19, kind='due')
        self.assertEqual(resource_class(row), 'own_due')

    def test_fixed_neighbour_formula_and_missing_not_zero(self):
        for h, f, b in ((5., 100., 120.), (5., 120., 90.)):
            r = fixed_neighbour(h, f, b)
            self.assertEqual(r['zero_minus_native_s2'], (f-h)**2+(b+h)**2-f*f-b*b)
        self.assertEqual(fixed_neighbour(5., 100., 120.)['kind'], 'benefit')
        self.assertEqual(fixed_neighbour(5., 120., 90.)['kind'], 'nonbenefit')
        self.assertIsNone(fixed_neighbour(5., None, 100.)['zero_minus_native_s2'])
        self.assertIsNone(fixed_neighbour(5., 3., 100.)['zero_minus_native_s2'])

    def test_return_bound_and_waiting_are_distinct_from_plan_delay(self):
        cycle = dict(return_s=100.)
        r = next_clock(cycle, dict(actual_s=101., eligible_s=90.), 95., 1.)
        self.assertEqual(r['kind'], 'return_bound_after_posted_plan')
        self.assertEqual(r['postreturn_wait_s'], 0.)
        r = next_clock(cycle, dict(actual_s=110., eligible_s=110.), 110., 1.)
        self.assertEqual(r['kind'], 'postreturn_wait_not_after_posted_plan')
        self.assertEqual(r['postreturn_wait_s'], 9.)
        self.assertIsNone(next_clock(cycle, None, None, 1.)['next_posted_delay_s'])

    def test_actual_writes_are_not_confused_with_no_write_callbacks(self):
        qs = [view(2, 5., 5.), view(5, 10., 5.), view(8, 15., 5.)]
        for q, version in zip(qs, ('initial', 3, 3)):
            q['view']['own_association']['obligation'] = dict(trip_id=20, last_write_version=version)
        vs = [dict(initial=False, native_event_seq=3, changes=[dict(trip_id=20)]),
              dict(initial=False, native_event_seq=6, changes=[])]
        cycles = [dict(after_last_query_s=5., executed_holding_s=20.)]
        summary, rows = version_intervals(qs, vs, cycles)
        self.assertEqual([r['kind'] for r in rows], ['actual_write', 'callback_without_write'])
        self.assertEqual(summary['closed_holding_s'], 20.)
        self.assertEqual(summary['groups']['actual_write']['carrying_positive_H'], 1)
        self.assertEqual(summary['groups']['actual_write']['own_plan_revised_with_positive_H'], 1)
        self.assertEqual(summary['groups']['callback_without_write']['own_plan_revised'], 0)

    def test_interval_holding_delta_not_reassigned_to_current_pressure(self):
        qs = [view(2, 5., 5.), view(5, 10., 5., due=True), view(8, 15., 5.)]
        cycles = [dict(after_last_query_s=5., executed_holding_s=20.)]
        summary, rows = version_intervals(qs, [], cycles)
        self.assertEqual(summary['first_query_prefix_s'], 5.)
        self.assertEqual([r['interval_delta_H_s'] for r in rows], [5., 5.])
        self.assertEqual(summary['groups']['no_callback']['pressure_cleared'], 1)
        self.assertNotIn('holding_caused_by_current_pressure_s', rows[0])

    def test_prefix_reset_cannot_pass_cross_version_analysis(self):
        with self.assertRaises(AssertionError):
            version_intervals([view(2, 10., 10.), view(5, 0., -10.)], [],
                              [dict(after_last_query_s=0., executed_holding_s=0.)])

    def test_missing_next_launch_aggregates_remain_null_not_zero(self):
        result = supply_tally([dict(holding_s=5., next_trip=None)], {})
        self.assertEqual(result['next_launch_labels'], 0)
        self.assertIsNone(result['next_public_delay_s'])
        self.assertIsNone(result['postreturn_wait_after_native_tick_s'])
        known = supply_tally([dict(trip_id=1, holding_s=5., next_trip=2, next_public_delay_s=0.)],
                             {1: dict(postreturn_wait_s=0.)})
        self.assertEqual(known['next_launch_labels'], 1)
        self.assertEqual(known['next_public_delay_s'], 0.)


if __name__ == '__main__':
    unittest.main()
