import unittest
from types import SimpleNamespace

from env.evaluation import (
    EpisodeProtocol,
    HeadwayEventRecorder,
    composite_service_cost,
    compute_wait_metrics,
)


class EpisodeProtocolTest(unittest.TestCase):
    def test_service_window_and_fixed_horizon(self):
        protocol = EpisodeProtocol.from_config(
            {
                "service_start_hour": 4,
                "service_end_hour": 23,
                "demand_end_time_s": 72000,
                "evaluation_end_time_s": 90000,
            },
            {},
            timetable_last_launch_s=70000,
        )
        self.assertEqual(protocol.demand_end_time_s, 72000)
        self.assertTrue(protocol.demand_active(71999))
        self.assertFalse(protocol.demand_active(72000))
        self.assertEqual(protocol.evaluation_end_time_s, 90000)

    def test_default_demand_window_ends_at_fixed_last_dispatch(self):
        protocol = EpisodeProtocol.from_config(
            {"service_start_hour": 4, "service_end_hour": 23},
            {},
            timetable_last_launch_s=70380,
        )
        self.assertEqual(protocol.demand_end_time_s, 70380)

    def test_early_finish_waits_until_demand_window_closes(self):
        protocol = EpisodeProtocol.from_config(
            {"service_start_hour": 6, "service_end_hour": 6},
            {},
            timetable_last_launch_s=0,
        )
        done, _ = protocol.should_terminate(100, True, False)
        self.assertFalse(done)
        done, reason = protocol.should_terminate(3600, True, False)
        self.assertTrue(done)
        self.assertEqual(reason, "service_cleared")


class MeasurementTest(unittest.TestCase):
    def test_unboarded_passengers_contribute_restricted_wait(self):
        boarded = SimpleNamespace(appear_time=0, boarding_time=120)
        waiting = SimpleNamespace(appear_time=60, boarding_time=None)
        station = SimpleNamespace(total_passenger=[boarded, waiting])
        metrics = compute_wait_metrics([station], censor_time_s=300)
        self.assertEqual(metrics["passengers_generated"], 2)
        self.assertEqual(metrics["passengers_unserved"], 1)
        self.assertAlmostEqual(metrics["avg_wait_observed_min"], 2.0)
        self.assertAlmostEqual(metrics["avg_wait_censored_min"], 3.0)

    def test_headway_uses_every_stop_arrival_event(self):
        recorder = HeadwayEventRecorder()
        recorder.record(3, True, 100, 0)
        recorder.record(3, True, 400, 2)
        recorder.record(3, True, 760, 4)
        recorder.record(4, True, 120, 0)
        recorder.record(4, True, 420, 2)
        summary = recorder.summary()
        self.assertEqual(summary["headway_event_count"], 5)
        self.assertEqual(summary["headway_sample_count"], 3)
        self.assertAlmostEqual(summary["headway_mean_s"], 320.0)

    def test_service_cost_penalises_suppressed_service(self):
        valid, _ = composite_service_cost(5.0, 12, 0.2, 12)
        invalid, components = composite_service_cost(
            4.0,
            10,
            0.1,
            12,
            passenger_unserved_rate=0.2,
            trip_completion_rate=0.8,
        )
        self.assertGreater(invalid, valid)
        self.assertAlmostEqual(components["incomplete_service"], 0.2)


if __name__ == "__main__":
    unittest.main()
