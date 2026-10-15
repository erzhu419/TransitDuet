from pathlib import Path
from io import StringIO
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_boundary_apc import APCTrace, arrival_intervals, audit_flows, boundary_events
from freqduet_travel_distribution import completed_legs, fit_travel
from test_v41_closed_loop import observation


class BoundaryAPCTest(unittest.TestCase):
    def test_terminal_clocks_complete_first_and_last_legs(self):
        routes = observation()["routes"]
        events = pd.DataFrame([dict(stage=stage, trip_id=1, bus_id=7, station_id=1, direction=True, time_s=time)
                               for stage, time in [("arrival", 20), ("departure", 30)]])
        launch = pd.DataFrame([dict(trip_id=1, bus_id=7, direction=True, time_s=5)])
        terminal = pd.DataFrame([dict(trip_id=1, bus_id=7, direction=True, time_s=50)])
        legs = completed_legs(boundary_events(events, launch, terminal, routes), routes)
        self.assertEqual(legs.sort_values("station_id").duration_s.tolist(), [15, 20])
        observed = observation()
        observed["time_s"] = 1800
        self.assertEqual(fit_travel(legs, observed)["counts"]["True:0"], 1)
        restored = pd.read_csv(StringIO(legs.to_csv(index=False)), dtype={"direction": str})
        self.assertEqual(fit_travel(restored, observed)["counts"], fit_travel(legs, observed)["counts"])

    def test_reverse_terminal_uses_direction_not_reused_bus(self):
        routes = {"True": dict(station_ids=[0, 1, 2]), "False": dict(station_ids=[2, 1, 0])}
        events = pd.DataFrame(columns=["stage", "direction", "trip_id", "bus_id", "station_id", "time_s"])
        launch = pd.DataFrame([dict(trip_id=3, bus_id=7, direction=False, time_s=5)])
        terminal = pd.DataFrame([dict(trip_id=3, bus_id=7, direction=False, time_s=50)])
        result = boundary_events(events, launch, terminal, routes)
        self.assertEqual(result.station_id.tolist(), [2, 0])

    def test_apc_trace_preserves_left_behind_mass_balance(self):
        history = APCTrace()
        history.arrival(True, 1, 10, 5, 0, 0, 20)
        history.arrival(True, 1, 110, 50, 1, 10, 30)
        gaps = arrival_intervals(pd.DataFrame(history.rows))
        self.assertEqual(gaps.arrivals.tolist(), [60])
        observed = observation()
        observed["time_s"] = 120
        self.assertEqual(history.freeze(observed)["True"][1]["rate_per_s"], .6)

    def test_negative_arrival_mass_is_not_hidden_by_clipping(self):
        apc = pd.DataFrame([dict(direction=True, station_id=1, time_s=t, boarded=b, left_behind=q)
                            for t, b, q in [(10, 0, 20), (110, 0, 10)]])
        self.assertEqual(arrival_intervals(apc).arrivals.iloc[0], -10)

    def test_future_apc_changes_labels_but_not_recent_prefix_rate(self):
        apc = pd.DataFrame([dict(direction=True, station_id=1, time_s=t, boarded=b,
                                left_behind=0, alighted=0, preload=0)
                            for t, b in [(10, 0), (110, 10), (210, 20)]])
        observed = observation()
        observed["time_s"] = 200
        observed["flows"]["True"][1].update(rate_per_s=.1, alight_fraction=0)
        first = audit_flows(apc, arrival_intervals(apc), observed)
        apc.loc[apc.time_s == 210, "boarded"] = 30
        second = audit_flows(apc, arrival_intervals(apc), observed)
        a = next(row for row in first if row["mode"] == "recent_4")
        b = next(row for row in second if row["mode"] == "recent_4")
        self.assertEqual(a["error"] + a["actual"], b["error"] + b["actual"])
        self.assertEqual(a["error"] - b["error"], 10)


if __name__ == "__main__":
    unittest.main()
