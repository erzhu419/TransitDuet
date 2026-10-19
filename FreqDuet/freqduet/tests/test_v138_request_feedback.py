import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v138_request_feedback import clip_slope, split_request, source_detail, clock_bands, request


def state(own, front, follower, prediction, holding):
    f = sum(own)-sum(front)
    t = sum(follower)-sum(own)
    return dict(regime='bilateral', u=(prediction-f)/8., prediction=prediction,
        forward=f, true_gap=t, h=holding, front_id=(0, 0), own_bus=1, follower_id=(2, 2),
        follower_is_next=True, source=('matched', 'journey', 'predicted', True),
        own_coordinates=np.asarray(own, float), front_coordinates=np.asarray(front, float),
        follower_coordinates=np.asarray(follower, float))


class RequestFeedbackTests(unittest.TestCase):
    def test_observed_timing_prediction_and_quantization_reconcile(self):
        b = state([100, 10, 100], [0, 5, 100], [200, 20, 110], 121., 0.)
        c = state([100, 12, 101], [0, 8, 100], [200, 23, 112], 140., 5.)
        name, parts, slope = split_request(b, c)
        self.assertEqual(name, 'stable_matched')
        self.assertEqual(slope, 1.)
        self.assertAlmostEqual(sum(parts.values()), 5.)
        self.assertNotEqual(parts['cumulative_holding_s'], 0.)
        self.assertNotEqual(parts['forecast_error_s'], 0.)

    def test_saturation_does_not_turn_unexecuted_requests_into_holding(self):
        self.assertEqual(clip_slope(60., 100.), 0.)
        self.assertAlmostEqual(clip_slope(40., 60.), .25)
        self.assertEqual(clip_slope(-10., -1.), 0.)
        self.assertEqual(clip_slope(10., 10.), 1.)

    def test_follower_identity_change_is_not_called_prediction_error(self):
        b = state([100, 0, 100], [0, 0, 100], [200, 0, 120], 140., 5.)
        c = dict(b, u=10., h=10., follower_id=(4, 4))
        name, parts, _ = split_request(b, c)
        self.assertEqual(name, 'follower_identity_change')
        self.assertEqual(parts['follower_identity_change_s'], 5.)
        self.assertEqual(parts['forecast_error_s'], 0.)

    def test_source_switch_retains_its_own_request_bucket(self):
        b = state([100, 0, 100], [0, 0, 100], [200, 0, 120], 140., 5.)
        c = dict(b, u=10., h=10., prediction=180., source=('matched', 'route', 'fallback', False))
        self.assertEqual(split_request(b, c)[0], 'source_change')
        self.assertEqual(source_detail(b, c)['forecast_error_s'], 5.)

    def test_request_regimes_use_original_gain(self):
        row = dict(context=dict(evidence_valid=True, follower_evidence_valid=False,
            target_headway_s=360., forward_headway_s=320.))
        self.assertEqual(request(row), ('forward', 10., None))
        row['context']['evidence_valid'] = False
        self.assertEqual(request(row), ('none', 0., None))

    def test_clock_relocation_is_not_a_change_in_action_amount(self):
        rows = [dict(base_time_s=899., candidate_time_s=901., delta_s=0., base_holding_s=10.,
            person_delta_s=0., base_person_s=30.)]
        bands = clock_bands(rows)
        self.assertEqual(bands[0]['issued_holding_delta_s'], -10.)
        self.assertEqual(bands[1]['issued_holding_delta_s'], 10.)
        self.assertEqual(bands[1]['anchored_holding_s'], 0.)
        self.assertEqual(sum(r['issued_person_delta_s'] for r in bands.values()), 0.)


if __name__ == '__main__':
    unittest.main()
