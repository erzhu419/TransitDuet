from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_residual_dependence import (issued_origin, identified_legs, identify_holding,
    describe_moments, moments, variance_parts, pair_audit, window_audit, incoming_travel_holding,
    holding_records, travel_records)


def records():
    return pd.DataFrame(dict(time_s=[3700, 3800, 4600, 4700], end_s=[3750, 3850, 4650, 4750],
        origin=[3600] * 4, bus_id=[1, 2, 1, 2], trip_id=[10, 20, 10, 20],
        direction=[True] * 4, station_id=[1, 1, 2, 2],
        residual=[1., 2., 3., 4.], centered=[-1., 1., -1., 1.], variance=[4.] * 4))


class ResidualDependenceTest(unittest.TestCase):
    def test_origin_is_strictly_before_departure_not_before_completion(self):
        clocks = [3600, 4500, 5400]
        self.assertIsNone(issued_origin(clocks, 3600))
        self.assertEqual(issued_origin(clocks, 4500), 3600)
        self.assertEqual(issued_origin(clocks, 4501), 4500)
        self.assertIsNone(issued_origin(clocks, 49501))

    def test_identity_does_not_join_two_physical_buses(self):
        events = pd.DataFrame([dict(direction=True, trip_id=1, station_id=station,
            bus_id=bus, stage=stage, time_s=time) for bus, stage, station, time in
            [(1, "departure", 0, 10), (2, "arrival", 1, 20)]])
        legs, missing = identified_legs(events, {"True": dict(station_ids=[0, 1])})
        self.assertTrue(legs.empty)
        self.assertEqual(missing, 1)

    def test_teacher_identity_is_exact_and_ambiguous_clock_is_rejected(self):
        teacher = [dict(time_s=10., action_s=5., features=[1, 0, 0, 0, 0, .1, 0, 0, .5, .25, 1, 0])]
        holds = pd.DataFrame([dict(time_s=10., action_s=5., station_id=1, bus_id=1, trip_id=2)])
        events = holds.assign(stage="action", direction=True, load=5)
        routes = {"True": dict(station_ids=[0, 1, 2])}
        result = identify_holding(teacher, holds, events, routes)
        self.assertEqual(result.bus_id.tolist(), [1])
        duplicated = pd.concat([holds, holds.assign(bus_id=2, trip_id=3)])
        with self.assertRaises(pd.errors.MergeError):
            identify_holding(teacher, duplicated, duplicated.assign(stage="action", direction=True, load=5), routes)

    def test_covariance_partition_includes_negative_terms_and_closes(self):
        frame = records()
        frame.loc[3, "direction"] = False
        frame.loc[1, "residual"] = -2
        result = variance_parts(frame, "residual")
        self.assertEqual(result["squared_sum"], 36)
        self.assertEqual(sum(result[k] for k in ["diagonal", "same_bus_cross", "same_stop_other_bus_cross",
            "same_direction_other_cross", "opposite_direction_other_cross"]), 36)

    def test_pearson_matches_numpy(self):
        x, y = [1, 2, 4], [3, 7, 5]
        self.assertAlmostEqual(describe_moments(moments(x, y))["correlation"], np.corrcoef(x, y)[0, 1])

    def test_exact_zero_correlation_control(self):
        self.assertEqual(describe_moments(moments([-1, -1, 1, 1], [-1, 1, -1, 1]))["correlation"], 0)
        self.assertEqual(describe_moments(moments([-1, -1, 1, 1], [-1, -1, 1, 1]))["correlation"], 1)

    def test_shared_model_covariance_only_same_origin(self):
        frame = records()
        frame.loc[2:, "origin"] = 4500
        result = pair_audit(frame, np.tile([-1., 1.], (4, 1)), np.full(4, 2))
        serial = result.loc[(result.relation == "same_bus_same_trip") & (result.lag == "1")]
        self.assertTrue((serial.same_origin_pairs == 0).all())
        adjacent = result.loc[(result.relation == "same_directed_stop_different_bus") & (result["transform"] == "residual")]
        self.assertTrue((adjacent.same_origin_model_covariance == 1).all())

    def test_windows_do_not_duplicate_endpoint_labels(self):
        frame = records()
        frame.loc[1, "time_s"] = 4500
        windows, _ = window_audit(frame)
        first = windows.loc[(windows.horizon_s == 900) & (windows["transform"] == "residual")]
        self.assertEqual(first.events.sum(), len(frame))
        self.assertEqual(first.events.iloc[0], 2)

    def test_cross_component_requires_leg_finished_before_action(self):
        travel = records().assign(arrival_station_id=[1, 1, 2, 2], station_id=[0, 0, 1, 1])
        holding = records()
        with self.assertRaisesRegex(ValueError, "preceded"):
            incoming_travel_holding(travel, holding)

    def test_holding_and_travel_reject_simultaneous_fit_labels(self):
        labels = records().assign(features=[[1] * 12] * 4, action_s=0)
        captures = {"clock_3600": dict(pool=dict(time_s=3600),
            distribution_model=dict(latest_time_s=3600))}
        with self.assertRaisesRegex(ValueError, "nonprefix holding"):
            holding_records(labels, captures)
        legs = records().assign(start_s=lambda f: f.time_s, duration_s=50, arrival_station_id=2)
        with self.assertRaisesRegex(ValueError, "nonprefix travel"):
            travel_records(legs, {3600: dict(now_s=3600, latest_completion_s=3600)})

    def test_cluster_summary_averages_actors_before_resampling(self):
        from run_freqduet_v49_residual_dependence import cluster_summary
        frame = pd.DataFrame(dict(kind=["x"] * 3, eval_seed=[1, 1, 2], value=[0., 2., 5.]))
        result = cluster_summary(frame, ["kind"], ["value"])
        self.assertEqual(result.demand_clusters.iloc[0], 2)
        self.assertEqual(result["mean"].iloc[0], 3.)

    def test_permutation_control_exposes_negative_correlation_from_centering(self):
        from run_freqduet_v49_centering_null import permutation_audit
        frame = records().assign(bus_id=[1, 2, 3, 4], trip_id=[10, 20, 30, 40], hour=1,
            time_s=[3700, 3800, 3900, 4000], station_id=1,
            residual=[-1.5, -.5, .5, 1.5], centered=[-1.5, -.5, .5, 1.5])
        travel = frame.assign(station_id=0, arrival_station_id=1)
        result = permutation_audit(travel, frame)
        selected = result.loc[(result.component == "holding") &
            (result.relation == "same_directed_stop_different_bus") & (result["transform"] == "centered")]
        self.assertEqual(selected.observed.iloc[0], 1)
        self.assertLess(selected.permutation_mean.iloc[0], -.1)
        self.assertEqual(selected.permutations.iloc[0], 200)


if __name__ == "__main__":
    unittest.main()
