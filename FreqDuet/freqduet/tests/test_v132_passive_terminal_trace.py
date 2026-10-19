from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v132_passive_terminal_trace as run


class PassiveTraceTests(unittest.TestCase):
    def test_query_calls_controller_once_and_detaches_evidence(self):
        @dataclass
        class Context:
            forward_headway_s: float = 300.
        observer = run.PassiveTrace(12011, {})
        stop = (16, 5, True)
        observer.decisions = [dict(trip_id=16, station_id=5, direction=True)]
        observer.evidence[stop] = dict(ready_time_s=3240.)
        observer.online[stop] = dict(gap_s=540.)
        observer.runner = SimpleNamespace(_pilot_agent=0, _pending_lower_action_context={0: Context()})
        law = object()
        with patch.object(run.original.PlanAudit, '__call__', return_value=law) as control:
            self.assertIs(observer(None, None, None, None), law)
            control.assert_called_once_with(None, None, None, None)
        observer.online[stop]['gap_s'] = 10.
        observer.evidence[stop]['ready_time_s'] = 9999.
        self.assertEqual(observer.queries[stop]['online']['gap_s'], 540.)
        self.assertEqual(observer.queries[stop]['evidence']['ready_time_s'], 3240.)

    def test_first_change_uses_policy_clock_not_trip_or_arrival_order(self):
        def row(trip, time, action):
            return dict(trip_id=trip, station_id=5, direction=True, policy_time_s=time,
                chosen_s=action, feedback_error_s=0.,
                evidence=dict(follower_source='journey', predicted_follower_bus_id=7))
        baseline = [row(1, 3300., 0.), row(16, 3240., 0.)]
        treated = [row(1, 3300., 5.), row(16, 3240., 30.)]
        result = run.compare_paths(baseline, treated)
        self.assertEqual(result['first_changed_queries'][0]['candidate']['trip_id'], 16)
        self.assertEqual(result['total_chosen_delta_s'], 35.)

    def test_missing_decisions_are_not_silently_intersected_away(self):
        row = dict(trip_id=1, station_id=2, direction=False, policy_time_s=10., chosen_s=5.,
            feedback_error_s=20., evidence=dict(follower_source='journey', predicted_follower_bus_id=7))
        result = run.compare_paths([row], [])
        self.assertEqual(result['baseline_only'], 1)
        self.assertEqual(result['total_chosen_delta_s'], -5.)

    def test_brief_excludes_unobserved_legacy_avl_cache(self):
        row = dict(feedback_error_s=142.5, ready_avl=dict(follower_snapshot=None))
        result = run.brief(row)
        self.assertNotIn('ready_avl', result)
        self.assertEqual(result['unclipped_request_s'], 35.625)


if __name__ == '__main__':
    unittest.main()
