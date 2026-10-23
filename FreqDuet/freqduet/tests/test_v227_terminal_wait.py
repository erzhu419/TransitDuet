from pathlib import Path
import tempfile
import unittest

from analyze_freqduet_v227_terminal_wait import audit_path, frozen_floor, plan_at, plans, ready_at, terminal_periods


def fixture():
    initial = [dict(trip_id=i, direction=(i == 2), planned_s=float(p)) for i,p in enumerate((0,11,10))]
    launches = [dict(trip=i, bus=b, direction=(i == 2), actual_s=float(t), public_baseline_s=float(p),
                     eligible_s=float(e)) for i,b,t,p,e in ((0,0,0,0,0),(1,1,11,9,11),(2,0,12,10,10))]
    callbacks = [dict(trip_id=0,time_s=0.,changes=[]),dict(trip_id=2,time_s=10.,changes=[]),
        dict(trip_id=1,time_s=11.,changes=[dict(trip_id=2,before=dict(planned_s=10.),after=dict(planned_s=12.))])]
    events = [dict(kind='plan_update',trip_id=0,time_s=0.),dict(kind='launch',trip_id=0,bus_id=0,time_s=0.),
        dict(kind='plan_update',trip_id=2,time_s=10.),dict(kind='return',trip_id=0,bus_id=0,time_s=10.),
        dict(kind='plan_update',trip_id=1,time_s=11.),dict(kind='launch',trip_id=1,bus_id=1,time_s=11.),
        dict(kind='launch',trip_id=2,bus_id=0,time_s=12.),dict(kind='return',trip_id=1,bus_id=1,time_s=20.),
        dict(kind='return',trip_id=2,bus_id=0,time_s=22.)]
    returns = [dict(trip_id=i,bus_id=b,return_s=float(t),target_direction=(i != 2))
               for i,b,t in ((0,0,10),(1,1,20),(2,0,22))]
    return dict(physical_contract=dict(dt_s=1.,physical_vehicle_count=2),
        fresh_upper=[dict(time_s=0.,trip_id=0,active=[],plans=initial,
            ready=[dict(bus_id=0,direction=False),dict(bus_id=1,direction=False)])],
        trace=dict(actual_launches=launches,vehicle_returns=returns,upper_callbacks=callbacks),
        events=events,launch_plans={'0':0.,'1':11.,'2':12.},
        results=dict(readiness_delay_total_s=2.,public_launch_delay_total_s=4.))


class TerminalWaitTest(unittest.TestCase):
    def test_replan_wait_is_not_labeled_as_no_vehicle(self):
        result = audit_path(fixture())
        row = result['rows'][2]
        self.assertEqual(row['post_eligible_plan_block_s'], 1.)
        self.assertEqual(row['post_eligible_no_ready_s'], 1.)
        self.assertEqual(row['post_eligible_open_ready_residual_s'], 0.)
        self.assertEqual(row['readiness_s'], 2.)

    def test_return_after_dispatch_needs_next_tick(self):
        data = fixture(); history = plans(data); terminal = terminal_periods(data,history)
        self.assertEqual(ready_at(terminal,10.,history['rank'][2],True), [])
        self.assertEqual([r['bus_id'] for r in ready_at(terminal,11.,history['rank'][2],True)], [0])
        row = audit_path(data)['rows'][2]
        self.assertEqual(row['return_after_dispatch_tick_s'], 1.)
        self.assertEqual(row['idle_plan_block_s'], 1.)
        self.assertEqual(row['idle_plan_open_s'], 0.)
        self.assertEqual(row['terminal_idle_s'], 2.)

    def test_later_trip_same_clock_cannot_change_earlier_dispatch_pass(self):
        h = dict(rank={2:2},keys={2:[(0.,-1),(10.,3)]},histories={2:[((0.,-1),20.),((10.,3),9.)]})
        self.assertEqual(plan_at(h,2,10.),20.)
        self.assertEqual(plan_at(h,2,11.),9.)

    def test_missing_plan_change_or_final_plan_fails(self):
        data = fixture()
        data['trace']['upper_callbacks'][-1]['changes'][0]['before']['planned_s'] = 9.
        with self.assertRaisesRegex(AssertionError,'missing intermediate plan'):
            audit_path(data)
        data = fixture();data['launch_plans']['2'] = 11.
        with self.assertRaisesRegex(AssertionError,'launch-time plan mismatch'):
            audit_path(data)

    def test_fifo_tie_uses_original_vehicle_order(self):
        data = fixture()
        for row in data['trace']['actual_launches']:
            row['bus'] = 1-row['bus']
        for row in data['trace']['vehicle_returns']:
            row['bus_id'] = 1-row['bus_id']
        for event in data['events']:
            if 'bus_id' in event:
                event['bus_id'] = 1-event['bus_id']
        with self.assertRaisesRegex(AssertionError,'native FIFO'):
            audit_path(data)

    def test_overlapping_plan_and_supply_block_is_retained(self):
        data = fixture()
        event = data['events'].pop(3)
        event['time_s'] = 11.
        data['events'].insert(5,event)
        data['trace']['vehicle_returns'][0]['return_s'] = 11.
        row = audit_path(data)['rows'][2]
        self.assertEqual(row['post_eligible_plan_block_s'],1.)
        self.assertEqual(row['post_eligible_no_ready_s'],1.)
        self.assertEqual(row['post_eligible_joint_plan_supply_block_s'],1.)

    def test_initial_stock_does_not_invent_a_return(self):
        result = audit_path(fixture())
        self.assertEqual(result['initial_stock'],2)
        self.assertEqual(result['incoming_chains'],1)
        self.assertNotIn('terminal_idle_s',result['rows'][0])
        self.assertIsNone(result['rows'][0]['return_s'])

    def test_final_fractional_plan_keeps_rounding_separate(self):
        data = fixture()
        data['trace']['upper_callbacks'][-1]['changes'][0]['after']['planned_s'] = 11.2
        data['launch_plans']['2'] = 11.2
        row = audit_path(data)['rows'][2]
        self.assertAlmostEqual(row['posted_plan_delta_s'],1.2)
        self.assertAlmostEqual(row['plan_rounding_s'],.8)
        self.assertEqual(row['final_plan_to_actual_s'],0.)

    def test_actual_config_parent_location_and_closest_floor_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'configs').mkdir()
            (root/'config_v2.yaml').write_text('upper:\n  timetable_planner:\n    terminal_headway_floor: {enable: true, ratio: 0.5}\n')
            (root/'configs/base.yaml').write_text('_extends: config_v2.yaml\nupper:\n  timetable_planner:\n    terminal_headway_floor: {enable: false, ratio: 0.0}\n')
            (root/'configs/main.yaml').write_text('_extends: base.yaml\n')
            result = frozen_floor(root/'configs/main.yaml')
            self.assertFalse(result['resolved']['enable'])
            self.assertEqual(len(result['chain']),3)


if __name__ == '__main__':
    unittest.main()
