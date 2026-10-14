from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v38_mechanism import cumulative_at
from run_freqduet_v38_observed_replay import FEATURE_NAMES, observable_snapshot


class ObservableSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.bus = NS(bus_id=7, direction=True, capacity=60, passengers=[1] * 12,
                      effective_route=[NS(distance=1000, speed_limit=10)] * 3,
                      travel_distance=1000, last_service_dwell_s=10,
                      last_left_behind_count=3, future_terminal_time=2000)
        self.trip = NS(launched=False, direction=False, launch_time=1200)
        self.env = NS(current_time=1000, stations=list(range(5)),
                      timetables=[self.trip], latent_waiting=999,
                      _ready_vehicle_count=lambda direction: 1 if direction else 2)
        context = NS(target_headway_s=360, forward_headway_s=280,
                     follower_departure_gap_s=420, follower_evidence_valid=True)
        self.runner = NS(env=self.env, _pending_lower_action_context={7: context})

    def snapshot(self):
        return observable_snapshot(self.runner, self.bus, [7, 2], 5, 0, 5, 10)

    def test_observable_geometry_and_resource_margin(self):
        features = self.snapshot()
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["return_eta_proxy_s"], 215)
        self.assertEqual(features["return_margin_proxy_s"], -15)
        self.assertEqual(features["ready_origin"], 1)
        self.assertEqual(features["ready_return"], 2)
        self.assertEqual(features["load_ratio"], .2)

    def test_future_outcomes_and_latent_queue_cannot_change_features(self):
        before = self.snapshot()
        self.bus.future_terminal_time = 9999
        self.env.latent_waiting = 0
        self.env.future_journey = -100
        self.trip.future_actual_launch = 1500
        self.bus.effective_route[0].speed_limit = 2
        self.assertEqual(before, self.snapshot())

    def test_current_backlog_and_published_plan_are_observed(self):
        self.trip._freqduet_scheduled_launch = 900
        features = self.snapshot()
        self.assertEqual(features["overdue_return_count"], 1)
        self.assertEqual(features["overdue_return_age_s"], 100)
        self.assertEqual(features["published_return_deadline_s"], -100)
        self.trip.launched = True
        features = self.snapshot()
        self.assertEqual(features["has_return_departure"], 0)
        self.assertEqual(features["return_margin_proxy_s"], 0)


class ExposureCutsTest(unittest.TestCase):
    def test_common_clock_cuts_use_only_completed_ticks(self):
        timeline = np.array([[1, 10, 20, 1], [2, 30, 60, 3], [3, 60, 120, 6]])
        np.testing.assert_array_equal(cumulative_at(timeline, 0), [0, 0, 0])
        np.testing.assert_array_equal(cumulative_at(timeline, 2), [30, 60, 3])
        np.testing.assert_array_equal(cumulative_at(timeline, np.inf), [60, 120, 6])


if __name__ == "__main__":
    unittest.main()
