from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_closed_loop_forecast import FlowHistory, HoldingHistory, fluid_rollout, response_action


def observation():
    route = dict(station_ids=[0, 1, 2], lengths_m=[100, 100], travel_s=[10, 10],
                 dwell_s=[0, 4, 0], hold_s=[0, 0, 0], cycle_s=26)
    flows = [dict(rate_per_s=0.0, queue=0.0, alight_fraction=0.0) for _ in range(3)]
    vehicle = dict(bus_id=7, trip_id=1, direction=True, on_route=True, station_index=0,
                   progress_m=0, phase="travel", remaining_stop_s=0, remaining_service_s=0,
                   load=5.0, capacity=50.0, ready_since_s=-1, predecessor_trip=None,
                   previous_hold_s=0, target_headway_s=360)
    return dict(time_s=0.0, dt_s=1.0, routes={"True": route, "False": dict(route)},
                flows={"True": flows, "False": [dict(row) for row in flows]},
                vehicles=[vehicle], commitments=[], decision=None,
                last_dispatch={"True": -9999, "False": -9999},
                last_arrivals={"True:1": dict(time_s=-400, trip_id=9)},
                last_departures={"True:1": dict(time_s=-360, trip_id=9)},
                holding_model=dict(coef=[1.0] + [0.0] * 11, samples=100, latest_time_s=-1))


class ClosedLoopTest(unittest.TestCase):
    def test_holding_feedback_changes_travel_and_onboard_exposure(self):
        observed = observation()
        historical = fluid_rollout(observed, 100, "historical")
        response = fluid_rollout(observed, 100, "response")
        self.assertEqual(response["holding_vehicle_seconds"], 45)
        self.assertEqual(response["returns"][0]["time_s"] - historical["returns"][0]["time_s"], 45)
        self.assertEqual(response["onboard_person_seconds"] - historical["onboard_person_seconds"], 225)

    def test_capacity_and_passenger_mass_are_conserved(self):
        observed = observation()
        observed["vehicles"][0]["load"] = 49
        observed["flows"]["True"][1].update(queue=1000, rate_per_s=.1)
        result = fluid_rollout(observed, 100, "historical")
        self.assertEqual(result["max_load"], 50)
        self.assertAlmostEqual(result["final_queue"] + result["final_onboard"] + result["alighted_total"],
                               1000 + 49 + .1 * 100)
        self.assertEqual(result["vehicles_created"], 0)

    def test_unserved_fluid_queue_integrates_arrivals(self):
        observed = observation()
        observed["vehicles"][0]["on_route"] = False
        observed["flows"]["True"][1].update(queue=2, rate_per_s=.1)
        result = fluid_rollout(observed, 10, "response")
        self.assertAlmostEqual(result["waiting_person_seconds"], 25)
        self.assertEqual(result["onboard_person_seconds"], 0)

    def test_future_labels_cannot_change_rollout(self):
        observed = observation()
        before = fluid_rollout(observed, 100, "response")
        observed.update(future_od=object(), scenario_seed=9999, latent_queue=object())
        self.assertEqual(before, fluid_rollout(observed, 100, "response"))

    def test_holding_fit_excludes_same_time_and_future_decisions(self):
        history = HoldingHistory()
        features = dict(forward_gap_s=360, target_headway_s=360, follower_gap_s=0,
                        follower_valid=False, load_ratio=.5, previous_hold_s=0,
                        station_phase=.5, direction=True, time_s=99)
        history.record(99, features, 15)
        history.record(100, features, 45)
        history.record(101, features, 45)
        model = history.freeze(100)
        self.assertEqual(model["samples"], 1)
        self.assertEqual(model["latest_time_s"], 99)

    def test_left_behind_balance_recovers_arrivals_beyond_boardings(self):
        history = FlowHistory()
        history.arrival(True, 1, 0, 0, 0, 0, 20)
        history.arrival(True, 1, 100, 50, 2, 10, 30)
        observed = observation()
        observed["time_s"] = 110
        flows = history.freeze(observed)
        self.assertAlmostEqual(flows["True"][1]["rate_per_s"], .6)
        self.assertAlmostEqual(flows["True"][1]["queue"], 36)

    def test_holding_response_uses_predicted_headway(self):
        model = dict(coef=[0, 1] + [0] * 10)
        values = dict(forward_gap_s=360, target_headway_s=360, follower_gap_s=0,
                      follower_valid=False, load_ratio=.5, previous_hold_s=0,
                      station_phase=.5, direction=True, time_s=99)
        self.assertEqual(response_action(model, values, 15), 0)
        self.assertEqual(response_action(model, dict(values, forward_gap_s=720), 15), 45)


if __name__ == "__main__":
    unittest.main()
