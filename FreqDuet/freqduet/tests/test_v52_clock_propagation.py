import unittest

import numpy as np
import pandas as pd

from run_freqduet_v52_clock_propagation import COMPONENTS, arrival_scores, decompose, items, previous


class ClockPropagationTests(unittest.TestCase):
    def test_clock_plan_covers_hour_crossings_and_common_clocks(self):
        plan = items()
        self.assertEqual(len(plan), 72)
        clocks = {row[-1] for row in plan}
        self.assertTrue(set(previous.COMMON).issubset(clocks))
        self.assertTrue(clocks.issubset(previous.CLOCKS))
        self.assertTrue(any(t // 3600 != (t + 1800) // 3600 for t in clocks))

    def setUp(self):
        self.observed = dict(time_s=100, vehicles=[dict(trip_id=1, on_route=True)],
            routes={"True": dict(station_ids=[0, 1, 2])})
        self.launches = pd.DataFrame([dict(trip_id=1, time_s=0), dict(trip_id=2, time_s=150)])
        self.actual = pd.DataFrame([dict(trip_id=1, station_id=station, stage=stage, time_s=time, direction=True)
            for station, stage, time in [(1, "arrival", 120), (1, "ready", 130),
                (1, "action", 131), (1, "departure", 141), (2, "arrival", 180)]])
        self.predicted = self.actual.copy()
        self.predicted.time_s = [125, 138, 139, 151, 195]

    def test_initial_chain_telescopes(self):
        row = decompose(self.predicted, self.actual, self.observed, self.launches)[-1]
        self.assertEqual(row["initial_boundary"], 5)
        self.assertEqual(row["service"], 3)
        self.assertEqual(row["holding"], 2)
        self.assertEqual(row["travel"], 5)
        self.assertEqual(sum(row[c] for c in COMPONENTS), row["error"])

    def test_future_launch_anchor(self):
        actual = self.actual.copy()
        actual.trip_id = 2
        actual.time_s += 100
        actual = pd.concat([pd.DataFrame([dict(trip_id=2, station_id=0, stage="departure", time_s=150, direction=True)]), actual])
        predicted = actual.copy()
        predicted.time_s += 7
        row = decompose(predicted, actual, self.observed, self.launches)[-1]
        self.assertEqual(row["launch"], 7)
        self.assertEqual(row["travel"], 0)
        self.assertEqual(row["initial_boundary"], 0)

    def test_missing_path_is_capped_not_removed(self):
        traces = [self.predicted.to_dict("records") for _ in range(63)] + [self.predicted.iloc[:1].to_dict("records")]
        row = arrival_scores(traces, self.actual, self.observed, self.launches).iloc[-1]
        self.assertEqual(row.available_paths, 63)
        self.assertEqual(row.complete_paths, 63)
        self.assertAlmostEqual(row.error, (63 * 15 + 1900 - 180) / 64)

    def test_missing_intermediate_stage_not_attributed(self):
        predicted = self.predicted.loc[self.predicted.stage != "ready"]
        row = decompose(predicted, self.actual, self.observed, self.launches)[-1]
        self.assertTrue(row["available"])
        self.assertFalse(row["complete_chain"])
        self.assertTrue(np.isnan(row["service"]))


if __name__ == "__main__":
    unittest.main()
