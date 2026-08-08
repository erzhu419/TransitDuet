import unittest

from env.station import Station


class FixedScenarioTape:
    def poisson(self, *_args, **_kwargs):
        return 2

    def uniform(self, _low, _high, _namespace, *_key):
        index = int(_key[-1])
        return (0.5, 10.5)[index]


class PassengerArrivalCausalityTest(unittest.TestCase):
    def test_sampled_window_does_not_expose_future_passengers(self):
        origin = Station(
            station_type=1,
            station_id=1,
            station_name="A",
            direction=True,
            od={"06:00:00": {"B": 3600.0}},
        )
        destination = Station(
            station_type=1,
            station_id=2,
            station_name="B",
            direction=True,
            od=None,
        )

        scheduled = origin.schedule_passenger_window(
            0.0,
            [origin, destination],
            passenger_update_interval=20,
            scenario_tape=FixedScenarioTape(),
        )

        self.assertEqual(scheduled, 2)
        self.assertEqual(len(origin.waiting_passengers), 0)
        self.assertEqual(len(origin.total_passenger), 0)
        self.assertEqual(origin.release_passengers(0.0), 0)
        self.assertEqual(origin.release_passengers(1.0), 1)
        self.assertAlmostEqual(origin.total_passenger[0].appear_time, 0.5)
        self.assertEqual(origin.release_passengers(10.0), 0)
        self.assertEqual(origin.release_passengers(11.0), 1)
        self.assertAlmostEqual(origin.total_passenger[1].appear_time, 10.5)

    def test_release_reports_od_only_when_arrival_is_visible(self):
        origin = Station(
            station_type=1,
            station_id=1,
            station_name="A",
            direction=True,
            od={"06:00:00": {"B": 3600.0}},
        )
        destination = Station(1, 2, "B", True, None)
        origin.schedule_passenger_window(
            0.0,
            [origin, destination],
            passenger_update_interval=20,
            scenario_tape=FixedScenarioTape(),
        )

        count, od = origin.release_passengers(1.0, return_details=True)

        self.assertEqual(count, 1)
        self.assertEqual(od, {(1, 2, True): 1})


if __name__ == "__main__":
    unittest.main()
