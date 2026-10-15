from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_closed_loop_forecast import fluid_rollout
from freqduet_travel_distribution import completed_legs, fit_travel, TravelPath, score_draws, travel_ensemble
from test_v41_closed_loop import observation


class TravelDistributionTest(unittest.TestCase):
    def test_only_finished_legs_before_issue_are_used(self):
        observed = observation()
        observed["time_s"] = 1800
        legs = pd.DataFrame([dict(direction="True", station_id=0, start_s=10, end_s=end,
                                  duration_s=end - 10) for end in (20, 1800, 1801)])
        model = fit_travel(legs, observed)
        self.assertEqual(model["samples"], 1)
        self.assertEqual(model["latest_completion_s"], 20)
        self.assertEqual(model["means"]["True:0"], 10)

    def test_pairing_uses_route_trip_and_adjacent_stop(self):
        events = pd.DataFrame([dict(stage=stage, trip_id=trip, station_id=stop, direction=direction, time_s=time)
            for stage, trip, stop, direction, time in [
                ("departure", 1, 0, True, 5), ("arrival", 1, 1, True, 20),
                ("departure", 2, 1, True, 30), ("arrival", 2, 2, True, 50),
                ("departure", 3, 0, False, 100)]])
        legs = completed_legs(events, observation()["routes"])
        self.assertEqual(legs.duration_s.tolist(), [15, 20])

    def test_shared_block_preserves_cross_vehicle_and_temporal_pattern(self):
        model = dict(now_s=0, dt_s=1, bin_s=300, horizon_s=900,
            means={"True:0": 20, "False:0": 40}, block_starts=[0],
            profiles={"True:0": [10, 20, 30], "False:0": [20, 40, 60]})
        path = TravelPath(model, "shared_block", np.random.default_rng(1))
        vehicle = dict(direction=True, travel_station_id=0)
        self.assertEqual([path(vehicle, time, 1) for time in (0, 299, 300, 600)], [10, 10, 20, 30])
        self.assertEqual(path(dict(vehicle, direction=False), 300, 1), 40)

    def test_independent_draws_do_not_force_shared_road_shock(self):
        model = dict(now_s=0, dt_s=1, bin_s=300, horizon_s=300,
            means={"True:0": 20}, block_starts=[0, 1, 2], profiles={"True:0": [10, 20, 30]})
        vehicle = dict(direction=True, travel_station_id=0)
        path = TravelPath(model, "independent", np.random.default_rng(3))
        self.assertGreater(len({path(vehicle, 0, 1) for _ in range(30)}), 1)

    def test_profile_mean_equals_completed_mean(self):
        observed = observation()
        observed["time_s"] = 3600
        legs = pd.DataFrame([dict(direction="True", station_id=0, start_s=t, end_s=t+d, duration_s=d)
                             for t, d in [(20, 10), (400, 30), (800, 20)]])
        model = fit_travel(legs, observed)
        self.assertAlmostEqual(np.mean(model["profiles"]["True:0"]), model["means"]["True:0"])
        self.assertEqual(model["counts"]["False:0"], 0)

    def test_circular_block_has_same_marginal_mean_at_every_lead(self):
        model = dict(now_s=0, dt_s=1, bin_s=300, horizon_s=900,
            means={"True:0": 20}, block_starts=[0, 1, 2], profiles={"True:0": [10, 20, 30]})
        vehicle = dict(direction=True, travel_station_id=0)
        path = TravelPath(model, "shared_block", np.random.default_rng(1))
        for time in (0, 300, 600):
            values = []
            for start in model["block_starts"]:
                path.start = start
                values.append(path(vehicle, time, 1))
            self.assertEqual(np.mean(values), 20)

    def test_missing_events_remain_censored_in_clock_scores(self):
        from run_freqduet_v46_travel_blocks import state_scores
        predicted = pd.DataFrame([dict(trip_id=1, station_id=1, stage="arrival", draw=0, time_s=20)])
        actual = pd.DataFrame([dict(trip_id=1, station_id=1, stage="arrival", time_s=10)])
        score = state_scores(predicted, actual, 0, "shared_block")[0]
        self.assertEqual(score["paths"], 1)
        self.assertAlmostEqual(score["error"], (20 + 63 * 1800) / 64 - 10)

    def test_premature_events_are_not_dropped_from_clock_scores(self):
        from run_freqduet_v46_travel_blocks import state_scores
        predicted = pd.DataFrame([dict(trip_id=1, station_id=1, stage="arrival", draw=i, time_s=20)
                                  for i in range(64)])
        actual = pd.DataFrame([dict(trip_id=1, station_id=1, stage="arrival", time_s=2000)])
        score = state_scores(predicted, actual, 0, "shared_block")[0]
        self.assertEqual(score["lead"], "later_than_horizon")
        self.assertEqual(score["error"], -1780)

    def test_partial_leg_uses_remaining_distance_and_tick_floor(self):
        observed = observation()
        observed["vehicles"][0]["progress_m"] = 50
        fractions, events = [], []

        def travel(vehicle, time, fraction):
            fractions.append(fraction)
            return 20 * fraction

        fluid_rollout(observed, 100, "historical", event_sink=events.append, travel_predictor=travel)
        self.assertEqual(fractions, [.5, 1.0])
        self.assertEqual(events[0]["time_s"], 10)

    def test_travel_ensemble_preserves_rng_and_capacity(self):
        observed = observation()
        observed["time_s"] = 1800
        observed["vehicles"][0]["load"] = 49
        observed["flows"]["True"][1]["queue"] = 1000
        model = fit_travel(pd.DataFrame(columns=["end_s", "start_s", "direction", "station_id", "duration_s"]), observed)
        holding = dict(coef=None, prior=[1, 0, 0, 0, 0, 0, 0])
        np.random.seed(42)
        before = np.random.get_state()
        rows = []
        travel_ensemble(observed, 100, holding, model, "shared_block", draws=2,
                        event_sink=lambda draw, row: rows.append(row))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertLessEqual(max(row["load"] for row in rows), 50)

    def test_crps_retains_both_bias_and_uncertainty(self):
        self.assertEqual(score_draws([1, 1], 2)["crps"], 1)
        self.assertEqual(score_draws([0, 2], 1)["crps"], .5)


if __name__ == "__main__":
    unittest.main()
