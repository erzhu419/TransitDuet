import ast
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v162_calibration_capture as capture


class CalibrationCaptureTests(unittest.TestCase):
    def snapshot(self):
        bus = NS(trip_id=1, bus_id=2, last_station=NS(station_id=3), direction=True,
            pre_action_follower_departure_gap=20., travel_distance=1000., current_speed=5.)
        follower = NS(trip_id=2, bus_id=4, direction=True, travel_distance=900., current_speed=4.,
            state=NS(name='TRAVEL'))
        ready = {(1, 3, True): dict(follower_source='same_time_avl_route_speed_eta',
            predicted_follower_bus_id=4, predicted_follower_trip_id=2)}
        return capture.snapshot(bus, [bus, follower], ready, [dict(time_s=9.)], [], {1: {}}, 10.)

    def row(self):
        return dict(trip_id=1, station_id=3, direction=True, prediction_time_s=10., policy_time_s=11.,
            evidence=dict(ready_time_s=10., follower_gap_prediction_s=20.,
                follower_source='same_time_avl_route_speed_eta'),
            context=dict(follower_evidence_valid=True, follower_departure_gap_s=20.,
                follower_evidence_source='same_time_avl_route_speed_eta'))

    def test_live_snapshot_preserves_identity_and_no_load(self):
        raw = self.snapshot()
        self.assertEqual((raw['follower']['trip_id'], raw['follower']['bus_id']), (2, 4))
        self.assertEqual(raw['events_seen']['last_arrival_s'], 9.)
        self.assertIsNone(raw['query_load'])
        self.assertFalse(raw['query_load_available'])
        self.assertNotIn('holding_time', raw['follower'])

    def test_ready_and_policy_clocks_checked_separately(self):
        raw, row = self.snapshot(), self.row()
        raw['query_key'] = list(raw['query_key'])
        capture.validate_snapshot(raw, row, 0)
        for field, value in (('query_time_s', 11.), ('ready_callback_order', 1),
                ('follower_gap_prediction_s', 21.), ('query_load', 20)):
            broken = deepcopy(raw)
            broken[field] = value
            with self.assertRaises(AssertionError):
                capture.validate_snapshot(broken, row, 0)

    def test_query_snapshot_rejects_future_event_clock(self):
        raw = self.snapshot()
        raw['query_key'] = list(raw['query_key'])
        raw['events_seen']['last_departure_s'] = 11.
        with self.assertRaisesRegex(AssertionError, 'future event'):
            capture.validate_snapshot(raw, self.row(), 0)

    def stage(self, count, ready_time=20., command=None, status='historical_current_service'):
        current = (2, 1, True)
        path = dict(arrivals={current: dict(time_s=5.)},
            ready={current: dict(ready_time_s=ready_time)}, trace={} if command is None else {current: command})
        p = dict(now=10., direction=True, route=dict(ids=[1, 3]), origin=0, target=1)
        row = dict(evidence=dict(predicted_follower_trip_id=2, predicted_follower_bus_id=4))
        samples = [(float(i), 10.) for i in range(1, count+1)]+[(10., 100.)]
        H = namedtuple('History', 'groups times')
        history = H({(True, 1): samples}, {(True, 1): [t for t, _ in samples]})
        indexed = {2: NS(motion=NS(dt=[], at=[5.]))}
        return capture.stage_support(path, row, p, status, (None, history, history, indexed))

    def test_native_support_never_counts_same_clock_completion(self):
        result = self.stage(8)
        self.assertEqual(result['survivor_n'], 8)
        self.assertIsNone(result['support_class'])
        self.assertEqual(result['reason'], 'prefix_less_than_nine')
        result = self.stage(9)
        self.assertEqual(result['support_class'], 'finite_window')
        self.assertEqual(result['latest_survivor_lag_s'], 1.)

    def test_known_command_requires_strictly_earlier_policy_clock(self):
        result = self.stage(9, ready_time=8., command=dict(policy_time_s=10.))
        self.assertEqual(result['kind'], 'ready-residence')
        result = self.stage(9, ready_time=8., command=dict(policy_time_s=9.))
        self.assertEqual(result['kind'], 'known-command')
        self.assertEqual(result['support_class'], 'not_applicable')

    def test_unfinished_service_time_not_used_as_feature(self):
        a, b = self.stage(9, ready_time=20.), self.stage(9, ready_time=999.)
        self.assertEqual(a, b)
        self.assertNotIn('ready_time_s', a)

    def test_horizon_uses_frozen_point_and_exact_boundaries(self):
        self.assertEqual([capture.horizon_band(x) for x in (0., 900., 901., 1800., 1801.)],
            ['[0,900]', '[0,900]', '(900,1800]', '(900,1800]', '(1800,infinity)'])
        self.assertIsNone(capture.horizon_band(None))
        self.assertIsNone(capture.horizon_band(-1.))

    def test_case_builder_never_enumerates_confirmation(self):
        plan = dict(reservation=dict(actors=[1, 2], calibration_days=[101, 102], confirmation_days=[201, 202]))
        self.assertEqual(capture.calibration_cases(plan), ((1, 101), (1, 102), (2, 101), (2, 102)))

    def test_query_output_precedes_outcome_attachment_and_has_no_target_label_lookup(self):
        tree = ast.parse(Path(capture.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ('snapshot', 'query_packet', 'save_queries'):
            called = {n.func.attr for n in ast.walk(functions[name]) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)}
            self.assertTrue(called.isdisjoint({'label_record', 'target_label', 'actual_parts'}))
        calls = {n.func.id: n.lineno for n in ast.walk(functions['path_case']) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)}
        self.assertLess(calls['save_queries'], calls['attach_labels'])


if __name__ == '__main__':
    unittest.main()
