from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v90_conditional_regularity as audit


class ConditionalRegularityTest(unittest.TestCase):
    def test_exactly_reuses_v89_primary_cells_not_a_new_candidate(self):
        self.assertEqual(audit.ITEMS, audit.source.ITEMS)
        self.assertEqual(audit.MODES, ("late", "no_gain_early"))
        self.assertEqual(2 * len(audit.ITEMS), 192)

    def test_arrival_timing_matches_trip_identity_not_list_position(self):
        a = dict(trip_id=1, station_id=2, direction=True, arrival_time_s=100., headway_s=None)
        b = dict(trip_id=2, station_id=2, direction=True, arrival_time_s=200., headway_s=100.)
        base = dict(events=[a, b], departures={(2, 2, True): 230.})
        other = dict(events=[dict(b, arrival_time_s=220., headway_s=120.), a], departures={(2, 2, True): 255.})
        result = audit.paired_timing(base, other)
        group = result["groups"][str((2, True))]
        self.assertEqual(result["unmatched_base"], 0)
        self.assertEqual(group["arrival_delta_s"]["mean"], 10.)
        self.assertEqual(group["headway_delta_s"]["mean"], 20.)
        self.assertEqual(group["headway_delta_s"]["count"], 1)
        self.assertEqual(group["departure_delta_s"]["mean"], 25.)

    def test_missing_follower_stays_unlabelled(self):
        record = dict(trip_id=1, station_id=2, direction=True)
        capture = SimpleNamespace(followers={(1, 2, True): None}, ready={}, departures={})
        self.assertIsNone(audit.follower_label(capture, record, {}))

    def test_completed_follower_preserves_ready_departure_and_prediction(self):
        record = dict(trip_id=1, station_id=2, direction=True,
                      context=SimpleNamespace(follower_departure_gap_s=160., follower_evidence_valid=True))
        capture = SimpleNamespace(followers={(1, 2, True): 2}, ready={(1, 2, True): 100., (2, 2, True): 240.},
                                  departures={(1, 2, True): 120., (2, 2, True): 255.})
        label = audit.follower_label(capture, record, {(2, 2): 15.})
        self.assertEqual(label["actual_follower_gap_s"], 140.)
        self.assertEqual(label["raw_gap_prediction_error_s"], 20.)
        self.assertEqual(label["follower_departure_time_s"], 255.)
        self.assertEqual(label["follower_action_s"], 15.)

    def test_cv_parts_separate_variance_and_mean_normalization(self):
        def arrival(mean, cv):
            return {k: dict(mean_headway_s=mean, cv_squared=cv, within_group_cv_squared=cv * .75,
                            between_group_cv_squared=cv * .25) for k in ("stop_direction", "four_hour_band")}
        result = audit.cv_parts(arrival(100., .04), arrival(110., .05))
        self.assertAlmostEqual(result["variance_at_base_mean"] + result["mean_normalization"], .01)
        self.assertAlmostEqual(result["stop_direction_within_group_cv_squared"] +
                               result["stop_direction_between_group_cv_squared"], .01)

    def test_noise_does_not_clip_unresolved_between_day_variance(self):
        values = np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])[..., None]
        result = audit.repeat_variance(values)
        self.assertLess(result["between_day_variance_untruncated"][0], 0.)
        self.assertEqual(result["mixed_repeat_sign"][0], 4)


if __name__ == "__main__":
    unittest.main()
