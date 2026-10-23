import unittest

from upper.timetable_planner import TimetableCurvePlanner
import freqduet_v219_projection as preview
import freqduet_v220_supply_projection as supply
import freqduet_v221_release_gate as gate
from test_v219_projection import query


class ReleaseGateTest(unittest.TestCase):
    def setUp(self):
        self.task = dict(trip_id=42, reserved_bus_id=11, native_s=7560., proposed_s=7558.6)
        self.old = dict(trip=42, bus=11, actual_s=7560.)
        self.new = dict(trip=42, bus=11, actual_s=7559.)

    def check_gate(self, old=None, new=None, old_plan=7560., new_plan=7558.6):
        return gate.task_execution(self.task, self.old if old is None else old,
            self.new if new is None else new, old_plan, new_plan, 1.)

    def test_one_second_same_reserved_bus_passes(self):
        self.assertTrue(self.check_gate()['passed'])

    def test_plan_change_without_actual_launch_advance_fails(self):
        self.assertFalse(self.check_gate(new=dict(self.new, actual_s=7560.))['passed'])

    def test_another_bus_does_not_count_as_authority(self):
        self.assertFalse(self.check_gate(new=dict(self.new, bus=12))['passed'])
        self.assertFalse(self.check_gate(old=dict(self.old, bus=12))['passed'])

    def test_missing_launch_or_overwritten_plan_fails(self):
        self.assertFalse(gate.task_execution(self.task, self.old, None, 7560., 7558.6, 1.)['passed'])
        self.assertFalse(self.check_gate(new_plan=7558.)['passed'])
        self.assertFalse(self.check_gate(old_plan=None)['passed'])

    def test_subtick_actual_advance_does_not_pass(self):
        self.assertFalse(self.check_gate(new=dict(self.new, actual_s=7559.5))['passed'])

    def test_budget_has_only_sixteen_fixed_slots(self):
        spec = gate.registration('test-source')
        self.assertEqual(spec['max_started_paths'], 16)
        self.assertEqual(sum(spec['stages'].values()), 16)
        self.assertEqual(len(spec['contexts']), 4)
        self.assertEqual(spec['model_updates'], 0)
        self.assertFalse(spec['independent_futures_opened'])

    def test_materialization_updates_both_adjacent_targets_and_cached_metadata(self):
        q = query((1, 2))
        q['direction'] = True
        p = TimetableCurvePlanner(basis_per_direction=2, shared_directions=True, plan_all_directions=False,
            coefficient_parameterization='antisymmetric_linear_v5', terminal_schedule_mode='exact_headway_curve',
            headway_budget_mode='rolling_zero_sum_delta_v6', delta_min_s=-60, delta_max_s=60)
        trips = preview.query_timetable(q, [])
        native, desired = preview.preview(p, trips, q, [0.]), preview.preview(p, trips, q, [10.5])
        solution, _ = supply.project(p, q, trips, native, desired)
        current = next(t for t in trips if t.launch_turn == 40)
        summary = p.apply(trips, current, [0.], origin_launch_s=7200., write_scheduled_launch=True, plan_id=40)
        result = gate.overwrite_curve(p, q, trips, native, solution, summary)
        actual = {t.launch_turn: dict(planned_s=t._freqduet_scheduled_launch, target_s=t.target_headway)
                  for t in trips if t.direction}
        for trip in (40, 42, 44, 46):
            self.assertEqual(actual[trip], solution[trip])
        advanced = next(t for t in trips if t.launch_turn == 42)
        following = next(t for t in trips if t.launch_turn == 44)
        cached = p.cached_plan_summary(advanced, plan_id=40)
        self.assertEqual(cached['scheduled_mean'], solution[42]['planned_s'])
        self.assertEqual(cached['target_headway'], solution[42]['target_s'])
        self.assertEqual(following._freqduet_predecessor_scheduled_launch, solution[42]['planned_s'])
        self.assertGreater(following.target_headway, 360.)
        self.assertEqual(result['target_headway'], summary['target_headway'])
        self.assertLess(abs(result['projected_headway_delta_sum_s']), 1e-6)


if __name__ == '__main__':
    unittest.main()
