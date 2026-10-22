import copy
import unittest

import freqduet_v218_cycle_policy as v
from run_freqduet_v218_cycle_smoke import next_launch
from upper.timetable_planner import TimetableCurvePlanner


def query():
    return dict(time_s=7500., event_seq=10, ready=[],
        active=[dict(bus_id=9, trip_id=31, direction=False, remaining_distance_m=1000.)],
        plans=[dict(trip_id=42, direction=True, launched=False, planned_s=7400.)],
        bus_id=9, trip_id=31, station_id=3, direction=False, base_bin=2, feasible_bins=list(range(7)),
        forward_valid=True, follower_valid=True, terminal_masked=False, context=dict(target_headway_s=360.))


class CyclePolicyTests(unittest.TestCase):
    def cycle(self):
        return dict(event_seq=5, time_s=7300., task_ids=[42])

    def test_due_shortage_backloads_native_curve_without_creating_stock(self):
        q = query()
        before = copy.deepcopy(q)
        c = v.cycle_context(q)
        self.assertEqual(c['shift_s'], -30.)
        self.assertEqual(c['ready'], [0, 0])
        self.assertEqual(q, before)

    def test_covering_next_cycle_allows_frontloading_otherwise_native(self):
        q = query()
        q['ready'] = [dict(direction=True)]
        self.assertEqual(v.cycle_context(q)['shift_s'], 30.)
        q['plans'].append(dict(trip_id=44, direction=True, launched=False, planned_s=8000.))
        self.assertEqual(v.cycle_context(q)['shift_s'], 0.)
        q['plans'] = []
        self.assertEqual(v.cycle_context(q)['shift_s'], 0.)

    def test_shared_antisymmetry_is_time_not_opposite_direction_controls(self):
        p = TimetableCurvePlanner(coefficient_parameterization='antisymmetric_linear_v5',
                                  basis_per_direction=2, shared_directions=True)
        for d in (False, True):
            self.assertEqual(p.delta_at([30.], d, 0.), -30.)
            self.assertEqual(p.delta_at([30.], d, p.horizon_s), 30.)

    def test_reduce_excess_release_holding_error_one_native_bin(self):
        selected, reason, link = v.lower_request(query(), self.cycle(), 370.)
        self.assertEqual((selected, reason, link['due_trip_ids']), (1, 'reduce_current_resource_request', [42]))

    def test_keep_beneficial_release_compensation(self):
        selected, reason, _ = v.lower_request(query(), self.cycle(), 345.)
        self.assertEqual((selected, reason), (2, 'preserve_release_compensation'))

    def test_current_queue_and_same_inbound_trip_are_required(self):
        for change in ('ready', 'not_due', 'new_trip', 'task_gone'):
            q, cycle = query(), self.cycle()
            if change == 'ready': q['ready'] = [dict(direction=True)]
            if change == 'not_due': q['plans'][0]['planned_s'] = 7600.
            if change == 'new_trip': q['active'][0]['trip_id'] = 33
            if change == 'task_gone': cycle['task_ids'] = [44]
            self.assertEqual(v.lower_request(q, cycle, 370.)[0], 2, change)

    def test_no_zero_to_zero_intervention_missing_support_or_past_cycle(self):
        for change in (dict(base_bin=0), dict(follower_valid=False), dict(terminal_masked=True),
                       dict(feasible_bins=[2]), dict(event_seq=4)):
            q = dict(query(), **change)
            self.assertEqual(v.lower_request(q, self.cycle(), 370.)[0], q['base_bin'])
        self.assertEqual(v.lower_request(query(), self.cycle(), None)[1], 'missing_completed_predecessor')
        self.assertEqual(v.lower_request(query(), None, 370.)[1], 'no_prior_cycle')

    def test_future_labels_cannot_change_live_request_or_cycle(self):
        q = query()
        q['future_return_s'], q['future_assigned_trip'] = -100., 999
        self.assertEqual(v.cycle_context(q), v.cycle_context(query()))
        self.assertEqual(v.lower_request(q, self.cycle(), 370.), v.lower_request(query(), self.cycle(), 370.))

    def test_supply_control_is_immediate_launch_not_a_later_matching_assignment(self):
        data = dict(events=[dict(kind='return', bus_id=9, trip_id=31, event_seq=10),
            dict(kind='launch', bus_id=9, trip_id=40, event_seq=11),
            dict(kind='launch', bus_id=9, trip_id=42, event_seq=12)])
        self.assertEqual(next_launch(data, 9, 31), 40)
        self.assertIsNone(next_launch(data, 9, 33))

    def test_budget_is_execution_only_with_no_future_search(self):
        p = v.registration('test')
        self.assertEqual(p['budget']['smoke_paths_max'], len(v.CONTEXTS)*len(v.ARMS))
        self.assertEqual(p['budget']['independent_future_paths'], 0)
        self.assertFalse(p['future_banks_opened'])
        self.assertFalse(p['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
