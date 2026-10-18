from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v112_decision_trace as run


class DecisionTraceTests(unittest.TestCase):
    def context(self, **changes):
        return SimpleNamespace(**(dict(evidence_valid=True, follower_evidence_valid=True,
            forward_headway_s=360., follower_departure_gap_s=360., target_headway_s=360.) | changes))

    def test_original_formula_and_quantization_are_unchanged(self):
        mask = np.ones(7, bool)
        for residual in (-100., -60., -10., 0., 10., 60.):
            c = self.context(forward_headway_s=320., follower_departure_gap_s=400.)
            row = run.decision_math(c, (360., -residual-40., residual), mask, True)
            self.assertEqual(row['chosen_s'], run.command(run.prior.desired_holding(c, residual), mask))
            self.assertAlmostEqual(row['actual_feedforward_s']+row['clipping_delta_s']+
                row['quantization_delta_s'], row['chosen_s']-row['base_shadow_s'])
            json.dumps(row, allow_nan=False)

    def test_persistent_level_is_not_an_innovation(self):
        row = run.decision_math(self.context(), (360., 60., -60.), np.ones(7, bool), True, -60.)
        self.assertEqual(row['chosen_s'], 45.)
        self.assertEqual(row['innovation_s'], 0.)
        self.assertEqual(row['innovation_shadow_s'], 0.)
        self.assertEqual(row['base_shadow_s'], 0.)

    def test_missing_forward_and_missing_follower_stay_distinct(self):
        missing = run.decision_math(self.context(evidence_valid=False), None, np.ones(7, bool), True)
        self.assertIsNone(missing['residual_s'])
        self.assertEqual(missing['chosen_s'], 0.)
        fallback = run.decision_math(self.context(follower_evidence_valid=False,
            follower_departure_gap_s=None, forward_headway_s=320.), (360., -60., 20.), np.ones(7, bool), True)
        self.assertFalse(fallback['valid_bilateral'])
        self.assertEqual(fallback['feedback_error_s'], 40.)
        self.assertEqual(fallback['chosen_s'], 0.)

    def test_feasible_mask_and_shorter_tie_are_preserved(self):
        self.assertEqual(run.command(12.5, np.ones(7, bool)), 10.)
        self.assertEqual(run.command(100., np.asarray([True]+[False]*6)), 0.)

    def test_ready_prediction_does_not_include_future_holding(self):
        row = dict(trip_id=1, follower_trip_id=3, station_id=2, direction=True,
            valid_bilateral=True, ready_time_s=100., follower_gap_prediction_s=100.)
        key = (3, 2, True)
        label = run.completed_follower(row, {key: 200.},
            {key: dict(time_s=246., realized_holding=55.)}, {key: dict(realized_holding=10.)}, {(1, 2, True): 3})
        self.assertEqual(label['prediction_error_s'], 0.)
        self.assertEqual(label['follower_executed_holding_s'], 45.)
        self.assertEqual(label['follower_after_ready_elapsed_s'], 46.)
        self.assertEqual(label['follower_departure_gap_from_own_ready_s'], 146.)
        self.assertIsNone(run.completed_follower(row, {}, {}, {}, {}))

    def test_grid_only_reuses_consumed_v110_paths(self):
        self.assertEqual(len(run.ITEMS), 16)
        self.assertEqual(len(set(run.ITEMS)), 16)
        self.assertTrue(all((*item, 0) in run.prior.ITEMS for item in run.ITEMS))


if __name__ == '__main__':
    unittest.main()
