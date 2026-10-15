from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_fleet_pool_forecast import (
    PastFleetTelemetry, forecast_pool, forecast_second_identity, freeze_pool, remaining_trip_s,
)


def observation():
    route = dict(station_ids=[0, 1], lengths_m=[10.0], travel_s=[10.0],
                 dwell_s=[0.0, 0.0], hold_s=[0.0, 0.0], cycle_s=20.0)
    vehicle = dict(bus_id=7, trip_id=1, direction=True, on_route=True,
                   station_index=0, progress_m=0.0, phase="travel", remaining_stop_s=0.0,
                   remaining_service_s=0.0, ready_since_s=-1.0)
    return dict(time_s=0.0, dt_s=1.0, routes={"True": route, "False": dict(route)},
                vehicles=[vehicle], commitments=[], last_dispatch={"True": -9999, "False": -9999},
                decision=dict(bus_id=8, trip_id=99, direction=False, station_id=1, actor_action_s=0))


def commitment(trip, direction, scheduled, gap=0):
    return dict(trip_id=trip, direction=direction, scheduled_s=scheduled,
                min_gap_s=gap, first_eligible_s=None, upper_queried=True)


class FleetRolloutTest(unittest.TestCase):
    def test_return_is_available_only_next_tick_and_vehicle_is_reused(self):
        obs = observation()
        obs["commitments"] = [commitment(2, False, 10), commitment(3, True, 30)]
        forecast = forecast_pool(obs, 40)
        self.assertEqual([(row["trip_id"], row["bus_id"], row["time_s"]) for row in forecast["launches"]],
                         [(2, 7, 11), (3, 7, 32)])
        self.assertEqual(forecast["backlog_trip_seconds"], 3)

    def test_empty_terminal_does_not_create_a_vehicle(self):
        obs = observation()
        obs["vehicles"][0].update(on_route=False, direction=True)
        obs["commitments"] = [commitment(2, False, 5)]
        forecast = forecast_pool(obs, 25)
        self.assertEqual(forecast["launches"], [])
        self.assertEqual(forecast["unfulfilled_within_window"], 1)
        self.assertEqual(forecast["backlog_trip_seconds"], 20)

    def test_fifo_pool_and_headway_are_both_preserved(self):
        obs = observation()
        vehicle = obs["vehicles"][0]
        vehicle.update(on_route=False, direction=True, ready_since_s=4)
        obs["vehicles"].append(dict(vehicle, bus_id=8, ready_since_s=2))
        obs["commitments"] = [commitment(2, True, 5), commitment(3, True, 5, gap=10)]
        launches = forecast_pool(obs, 20)["launches"]
        self.assertEqual([(row["bus_id"], row["time_s"]) for row in launches], [(8, 5), (7, 15)])

    def test_holding_effect_propagates_through_next_service(self):
        obs = observation()
        obs["commitments"] = [commitment(2, False, 10)]
        baseline = forecast_pool(obs, 60)
        candidate = forecast_pool(obs, 60, return_offsets={1: 15})
        self.assertEqual(candidate["launches"][0]["time_s"] - baseline["launches"][0]["time_s"], 15)
        self.assertEqual(candidate["backlog_trip_seconds"] - baseline["backlog_trip_seconds"], 15)

    def test_no_terminal_dwell_is_charged(self):
        obs = observation()
        obs["routes"]["True"]["dwell_s"][-1] = 999
        self.assertEqual(remaining_trip_s(obs, obs["vehicles"][0]), 10)

    def test_second_decision_can_belong_to_a_not_yet_launched_trip(self):
        obs = observation()
        obs["decision"].update(direction=False, station_id=1, bus_id=8)
        obs["commitments"] = [commitment(2, False, 15)]
        second = forecast_second_identity(obs, forecast_pool(obs, 50))
        self.assertEqual((second["trip_id"], second["bus_id"]), (2, 7))

    def test_predictor_does_not_read_future_labels(self):
        obs = observation()
        before = forecast_pool(obs, 60)
        obs.update(future_returns=object(), seed=9999, future_journey=object())
        obs["vehicles"][0]["actual_return_s"] = 999999
        self.assertEqual(before, forecast_pool(obs, 60))


class ObservablePoolTest(unittest.TestCase):
    def test_snapshot_ignores_latent_queue_and_stochastic_route_speeds(self):
        stations = [NS(station_name=name, station_id=idx) for name, idx in zip("ABCBA", (0, 1, 2, 1, 0))]
        routes = [NS(start_stop=a, end_stop=b, distance=50, speed_limit=2) for a, b in zip("ABCB", "BCBA")]
        bus = NS(bus_id=7, trip_id=3, direction=True, last_station=stations[1], in_station=True,
                 last_action_time=None, last_board_time=5, launch_time=0, last_service_dwell_s=4,
                 last_action_s=0, back_to_terminal_time=None, on_route=True,
                 travel_distance=50, passengers=[1], capacity=50)
        env = NS(current_time=10, time_step=1, stations=stations, routes=routes, bus_all=[bus],
                 timetables=[], _last_dispatch_time={True: 0, False: 0})
        telemetry = PastFleetTelemetry()
        for direction in (True, False):
            telemetry.record_movement(direction, 0, 10, 1)
        runner = NS(env=env)
        identity = dict(bus_id=7, trip_id=3, station_id=1, direction=True, time_s=10)
        before = freeze_pool(runner, telemetry, identity, 15)
        env.latent_queue = 999
        bus.future_terminal_time = 99999
        for route in routes:
            route.speed_limit = 999
        self.assertEqual(before, freeze_pool(runner, telemetry, identity, 15))
        self.assertEqual(before["vehicles"][0]["remaining_stop_s"], 17)


if __name__ == "__main__":
    unittest.main()
