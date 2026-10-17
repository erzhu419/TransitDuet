from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v83_holding_propagation import adjacent_pair, expected_effect, command_rows


class HoldingPropagationTest(unittest.TestCase):
    def test_adjacent_pairs_follow_actual_departure_order(self):
        events = [[2, 5, 1, 400.], [1, 5, 1, 100.], [3, 5, 1, 700.], [7, 5, 0, 500.]]
        result = adjacent_pair(events, 2, 5, True, 300.)
        self.assertEqual(result["loss"], 0.)
        self.assertEqual(result["follower_trip_id"], 3)

    def test_missing_future_event_is_not_imputed(self):
        self.assertIsNone(adjacent_pair([[1, 5, 1, 100.], [2, 5, 1, 400.]], 2, 5, True, 300.))

    def test_same_tick_preserves_simulator_event_order(self):
        events = [[8, 5, 1, 100.], [7, 5, 1, 400.], [2, 5, 1, 400.], [9, 5, 1, 700.]]
        result = adjacent_pair(events, 7, 5, True, 300.)
        self.assertEqual(result["follower_trip_id"], 2)
        self.assertEqual(result["follower_gap_s"], 0.)

    def test_probability_weighted_response_cancels_common_prefix(self):
        base = np.full(7, 1. / 7.)
        law = np.arange(1., 8.) / 28.
        values = np.arange(7.) * 10.
        self.assertAlmostEqual(expected_effect(law, base, values),
                               expected_effect(law, base, values + 1e9))
        self.assertEqual(expected_effect(base, base, values), 0.)

    def test_command_identity_includes_time_trip_stop_and_direction(self):
        rows = command_rows([dict(time_s=42., trip_id=4, station_id=7, direction=True, action=3)])
        self.assertEqual(rows, [[42., 4, 7, 1, 3]])


if __name__ == "__main__":
    unittest.main()
