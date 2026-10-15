from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_turnover_holding import augment_labels, freeze_turnover, turnover_features, TurnoverResponse


def data():
    labels = pd.DataFrame([dict(time_s=t, bus_id=0, trip_id=trip, action_s=hold, features=[0.] * 12)
                          for t, trip, hold in [(20, 1, 10), (100, 2, 5)]])
    launches = pd.DataFrame([dict(time_s=t, bus_id=0, trip_id=trip, scheduled_s=scheduled)
                            for t, trip, scheduled in [(0, 1, 0), (60, 2, 55)]])
    terminals = pd.DataFrame([dict(time_s=50, bus_id=0, trip_id=1)])
    return labels, launches, terminals


class TurnoverHoldingTest(unittest.TestCase):
    def test_future_completed_trip_cannot_change_earlier_context(self):
        labels, launches, terminals = data()
        expected = augment_labels(labels, launches, terminals).augmented_features.tolist()
        future_labels = pd.DataFrame([dict(time_s=250, bus_id=0, trip_id=3, action_s=45, features=[0.] * 12)])
        future_launches = pd.DataFrame([dict(time_s=200, bus_id=0, trip_id=3, scheduled_s=200)])
        future_terminals = pd.DataFrame([dict(time_s=300, bus_id=0, trip_id=3)])
        actual = augment_labels(pd.concat([labels, future_labels]), pd.concat([launches, future_launches]),
            pd.concat([terminals, future_terminals])).augmented_features.iloc[:2].tolist()
        np.testing.assert_allclose(actual, expected)

    def test_freeze_excludes_simultaneous_holding_label(self):
        labels, launches, terminals = data()
        history = freeze_turnover(labels, launches, terminals, dict(time_s=100,
            vehicles=[dict(bus_id=0, trip_id=2, on_route=True)]))
        self.assertEqual(history["0"]["current"]["count"], 0)
        self.assertEqual(history["0"]["previous"]["mean_hold_s"], 10)

    def test_cold_start_context_and_observed_launch_delay(self):
        context = turnover_features(None, dict(launch_s=10, scheduled_s=5), 20)
        np.testing.assert_array_equal(context[:5], np.zeros(5))
        self.assertAlmostEqual(context[5], 5 / 3600)

    def test_predicted_return_updates_next_trip_context(self):
        history = {"0": dict(previous=None, current=dict(trip_id=1, launch_s=0, scheduled_s=0,
            count=1, total=10., last=10.))}
        response = TurnoverResponse(dict(coef=None, prior=[0, 1, 0, 0, 0, 0, 0]), history,
            {"True": dict(station_ids=[0, 1, 2]), "False": dict(station_ids=[2, 1, 0])}, 0)
        response.observe(dict(stage="arrival", bus_id=0, trip_id=1, direction=True, station_id=2,
            time_s=100, launch_time_s=0, scheduled_launch_s=0))
        response.observe(dict(stage="ready", bus_id=0, trip_id=2, direction=False, station_id=1,
            time_s=220, launch_time_s=200, scheduled_launch_s=190))
        state = response.history["0"]
        context = turnover_features(state["previous"], state["current"], 220)
        self.assertEqual(context[0], 1)
        self.assertAlmostEqual(context[4], 100 / 3600)
        response.observe(dict(stage="action", bus_id=0, trip_id=2, direction=False, station_id=1,
            time_s=220, launch_time_s=200, scheduled_launch_s=190, action_s=5, evidence_valid=True))
        response.observe(dict(stage="arrival", bus_id=0, trip_id=2, direction=False, station_id=0,
            time_s=300, launch_time_s=200, scheduled_launch_s=190))
        self.assertEqual(response.history["0"]["previous"]["mean_hold_s"], 5)
        self.assertEqual(history["0"]["current"]["total"], 10)


if __name__ == "__main__":
    unittest.main()
