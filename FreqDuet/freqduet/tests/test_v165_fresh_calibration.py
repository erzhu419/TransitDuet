from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v165_fresh_calibration as fresh


def inputs():
    cal, confirm = list(range(101, 133)), list(range(201, 233))
    parent = dict(status='v160_metadata_and_preregistration_complete', execution_source='14d930ee25',
        new_days_collected=0, interval_fit=False, registration=dict(status='preregistered_not_collected',
            collector_submitted=False, calibration_days=cal, confirmation_days=confirm, actors=list(range(12)),
            modes=['old', 'handoff'], checkpoint_ep=199,
            sampler_seeds={str(d): 1600000000+i for i, d in enumerate(cal+confirm)},
            service_point_source='1b783ea8f3', running_shadow_source='5be4cea534',
            v146_extensions_s=[3., 0.], window_s=3600., tails=[.1, .1],
            freshness=dict(overlap_days=[], overlap_samplers=[])))
    reference = fresh.capture.protocol.registration(parent)
    reference['execution_source'] = '09553bc920'
    model = dict(status='sealed', execution_source='35ae15b185', confirmation_started=False,
        confirmation_outcomes_read=False, confirmation_release_eligible=False,
        support_admission=dict(decision='insufficient_support'), calibration_days=cal,
        frozen_protocol=deepcopy(reference))
    diagnosis = dict(status='v164_readonly_sparse_key_audit_complete', execution_source='4366c6c4b1',
        v163_counts_and_weighted_support_reproduced=True)
    evidence = dict(new_day_conflicts=[], new_sampler_conflicts=[], confirmation_outcome_conflicts=[], errors=[])
    return reference, model, diagnosis, evidence


class FreshCalibrationTests(unittest.TestCase):
    def test_method_and_parents_unchanged(self):
        values = inputs()
        before = deepcopy(values)
        result = fresh.registration(*values)
        self.assertEqual(values, before)
        for field in fresh.METHOD_FIELDS:
            self.assertEqual(result[field], values[0][field])
        self.assertFalse(result['execution']['collector_submitted'])
        self.assertFalse(result['execution']['interval_fit'])

    def test_fixed_new_64_days_exclude_old_32_and_preserve_confirmation(self):
        reference, *rest = inputs()
        result = fresh.registration(reference, *rest)['reservation']
        self.assertEqual(len(result['calibration_days']), 64)
        self.assertEqual(result['calibration_paths'], 1536)
        self.assertEqual(result['confirmation_paths'], 768)
        self.assertFalse(set(result['calibration_days']) & set(result['excluded_development_days']))
        self.assertEqual(result['confirmation_days'], reference['reservation']['confirmation_days'])
        for day in result['confirmation_days']:
            self.assertEqual(result['sampler_seeds'][str(day)], reference['reservation']['sampler_seeds'][str(day)])
        self.assertEqual(len(set(result['sampler_seeds'].values())), 96)

    def test_method_drift_and_confirmation_exposure_stop_registration(self):
        values = inputs()
        values[1]['frozen_protocol']['support']['minimum_calibration_days'] = 18
        with self.assertRaisesRegex(AssertionError, 'frozen method'):
            fresh.registration(*values)
        values = inputs()
        values[1]['confirmation_started'] = True
        with self.assertRaisesRegex(AssertionError, 'blocked V163'):
            fresh.registration(*values)

    def test_conflicts_and_unreadable_current_metadata_block_seal(self):
        for field in ('new_day_conflicts', 'new_sampler_conflicts', 'confirmation_outcome_conflicts', 'errors'):
            values = inputs()
            values[3][field] = ['conflict']
            with self.assertRaisesRegex(AssertionError, 'not established'):
                fresh.registration(*values)

    def test_reserved_confirmation_is_not_consumed_outcome(self):
        self.assertEqual(fresh.conflicts(dict(status='preregistered_not_collected', confirmation_days=[201]),
            [201])['confirmation_outcome_conflicts'], [])
        self.assertEqual(fresh.conflicts(dict(status='complete', days=[201]),
            [201])['confirmation_outcome_conflicts'], [201])

    def test_day_and_sampler_reuse_detected_independently(self):
        day = fresh.CALIBRATION[0]
        result = fresh.conflicts(dict(calibration_days=[day], sampler_seed=fresh.SAMPLERS[day]), [])
        self.assertEqual(result['new_day_conflicts'], [day])
        self.assertEqual(result['new_sampler_conflicts'], [fresh.SAMPLERS[day]])
        self.assertEqual(fresh.filename_day(f'12011_{day}_old.json'), day)
        self.assertEqual(fresh.filename_day(f'12011_{day}_old.jsonl.gz'), day)
        self.assertEqual(fresh.filename_day(f'12011_{day}.json'), day)
        self.assertEqual(fresh.filename_day(f'12011_{day}.npz'), day)
        self.assertIsNone(fresh.filename_day('status.json'))


if __name__ == '__main__':
    unittest.main()
