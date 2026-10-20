from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v148_fallback_clock as run


def row():
    return dict(prediction_time_s=100., online=dict(component_used=False),
        evidence=dict(eta_prediction_s=30., service_dwell_proxy_s=10.))


def saved():
    return dict(v140_status='unchanged', v141_status='unchanged_v140', v141_s=43.)


class RawClockTests(unittest.TestCase):
    def test_raw_clock_proxy_and_existing_adjustment_sum_to_saved_error(self):
        parts = run.raw_parts(row(), saved(), 120., 128.)
        self.assertEqual(parts, dict(raw_arrival_clock=10., raw_target_service_proxy=2., raw_gap_adjustment=3.))
        self.assertEqual(sum(parts.values()), 43.-28.)

    def test_already_arrived_target_is_kept_as_a_signed_arrival_label(self):
        parts = run.raw_parts(row(), saved(), 90., 110.)
        self.assertEqual(parts['raw_arrival_clock'], 40.)
        self.assertEqual(parts['raw_target_service_proxy'], -10.)
        self.assertEqual(sum(parts.values()), 43.-10.)

    def test_structured_forecasts_are_not_counted_again_as_raw_error(self):
        for version in ('moving', 'stop', 'ready'):
            r, s = row(), saved()
            if version == 'moving':
                r['online']['component_used'] = True
            elif version == 'stop':
                s['v140_status'] = 'passive_stop_component'
            else:
                s['v141_status'] = 'passive_unissued_ready'
            self.assertFalse(run.opaque(r, s))
            self.assertEqual(sum(run.raw_parts(r, s, 120., 128.).values()), 0.)

    def test_mae_split_keeps_the_original_full_delta_error_sign(self):
        r = run.record([1], dict(raw_arrival_clock=10., raw_target_service_proxy=-3., raw_gap_adjustment=1.), -2.)
        self.assertEqual(r['unallocated_bias_s'], 8.)
        self.assertEqual(r['unallocated_mae_allocation_s'], -8.)
        self.assertEqual(sum(r[p+'_mae_allocation_s'] for p in run.PARTS), -8.)


if __name__ == '__main__':
    unittest.main()
