from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_turnover_holding import augment_labels, freeze_turnover, turnover_features, TurnoverResponse, fit_turnover_tilt, tilted_probabilities


def data():
    labels = pd.DataFrame([dict(time_s=t, bus_id=0, trip_id=trip, action_s=hold, residual_s=hold, features=[0.] * 12)
                          for t, trip, hold in [(20, 1, 10), (100, 2, 5)]])
    launches = pd.DataFrame([dict(time_s=t, bus_id=0, trip_id=trip, scheduled_s=scheduled)
                            for t, trip, scheduled in [(0, 1, 0), (60, 2, 55)]])
    terminals = pd.DataFrame([dict(time_s=50, bus_id=0, trip_id=1)])
    return labels, launches, terminals


class TurnoverHoldingTest(unittest.TestCase):
    def test_zero_correction_preserves_base_distribution_and_seven_actions(self):
        prior = np.arange(1., 8.) / 28
        model = dict(coef=None, prior=prior.tolist())
        correction = dict(coef=[0.] * 7, scale=[1.] * 7)
        np.testing.assert_allclose(tilted_probabilities(model, [], np.ones(7), correction), prior, atol=1e-12)

    def test_tilt_fit_excludes_future_and_has_zero_cold_start(self):
        rows = pd.DataFrame([dict(time_s=t, action_s=y, context=np.ones(7), base_probabilities=np.ones(7) / 7)
            for t, y in [(1, 45), (2, 45), (10, 0), (11, 0)]])
        expected = fit_turnover_tilt(rows.iloc[:2], 10)
        actual = fit_turnover_tilt(rows, 10)
        self.assertEqual(expected, actual)
        self.assertEqual(actual["latest_time_s"], 2)
        self.assertGreater(sum(actual["coef"]), 0)
        self.assertEqual(fit_turnover_tilt(rows, 1)["coef"], [0.] * 7)

    def test_future_completed_trip_cannot_change_earlier_context(self):
        labels, launches, terminals = data()
        expected = augment_labels(labels, launches, terminals).augmented_features.tolist()
        future_labels = pd.DataFrame([dict(time_s=250, bus_id=0, trip_id=3, action_s=45, residual_s=20, features=[0.] * 12)])
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
        self.assertEqual(history["0"]["previous"]["mean_residual_s"], 10)

    def test_large_correctly_predicted_action_is_not_residual_memory(self):
        labels, launches, terminals = data()
        labels.loc[labels.trip_id == 1, ["action_s", "residual_s"]] = [45, 0]
        history = freeze_turnover(labels, launches, terminals, dict(time_s=100,
            vehicles=[dict(bus_id=0, trip_id=2, on_route=True)]))
        self.assertEqual(history["0"]["previous"]["mean_residual_s"], 0)

    def test_cold_start_has_no_turnover_correction(self):
        context = turnover_features(None, dict(launch_s=10, scheduled_s=5), 20)
        np.testing.assert_array_equal(context, np.zeros(7))

    def test_turnover_context_expires_after_first_eligible_action(self):
        previous = dict(scored_count=1, mean_residual_s=10, last_residual_s=10, duration_s=100, end_s=100)
        current = dict(launch_s=200, scheduled_s=190, decisions=0)
        self.assertTrue(np.any(turnover_features(previous, current, 220)))
        current["decisions"] = 1
        np.testing.assert_array_equal(turnover_features(previous, current, 230), np.zeros(7))

    def test_predicted_return_updates_next_trip_context(self):
        history = {"0": dict(previous=None, current=dict(trip_id=1, launch_s=0, scheduled_s=0,
            decisions=1, count=1, total=10., last=10.))}
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
        state["current"]["base_mean_s"] = 2.
        response.observe(dict(stage="action", bus_id=0, trip_id=2, direction=False, station_id=1,
            time_s=220, launch_time_s=200, scheduled_launch_s=190, action_s=5, evidence_valid=True))
        np.testing.assert_array_equal(turnover_features(state["previous"], state["current"], 230), np.zeros(7))
        response.observe(dict(stage="arrival", bus_id=0, trip_id=2, direction=False, station_id=0,
            time_s=300, launch_time_s=200, scheduled_launch_s=190))
        self.assertEqual(response.history["0"]["previous"]["mean_residual_s"], 3)
        self.assertEqual(history["0"]["current"]["total"], 10)


if __name__ == "__main__":
    unittest.main()
