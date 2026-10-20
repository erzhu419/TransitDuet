from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v155_archive_running_prior as audit


class ArchiveRunningPriorTests(unittest.TestCase):
    def metadata(self):
        return dict(status='v145_independent_interval_validation_complete', registration=dict(
            registration_source='6eaa4a389b', archived_actors=list(audit.frozen.frozen.ACTORS),
            demand_days=[2451001], modes=['old', 'handoff'], checkpoint_ep=199, window_s=3600.))

    def test_archive_prior_overlap_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit.archive_cases(self.metadata(), dict(days=[2451001]), dict(calibration_days=[2751001]))

    def test_disjoint_archive_uses_all_registered_actors(self):
        cases = audit.archive_cases(self.metadata(), dict(days=[2751001]), dict(calibration_days=[2751001]))
        self.assertEqual(len(cases), 12)

    def test_absent_low_support_is_not_zero_or_pass(self):
        results = {m: {g: dict(metrics={f: dict(ci95=[-2., -1.]) for f in
            ('delta_full_mae_s', 'delta_running_mae_s')}) for g in ('all', 'hour2', 'clearance', 'hour13')}
            for m in ('old', 'handoff')}
        result = audit.report(results)
        self.assertFalse(result['fixed_joint_gate'])
        self.assertEqual(len(result['insufficient_crossed_support']), 2)
        self.assertEqual(result['unmet_interval_requirements'], [])

    def test_clearance_harm_is_separate_from_absent_support(self):
        results = {m: {g: dict(metrics={f: dict(ci95=[-2., -1.]) for f in
            ('delta_full_mae_s', 'delta_running_mae_s')}) for g in
            ('all', 'hour2', 'clearance', 'hour13', 'low_support')} for m in ('old', 'handoff')}
        results['old']['clearance']['metrics']['delta_full_mae_s']['ci95'] = [-1., 1.]
        result = audit.report(results)
        self.assertFalse(result['fixed_joint_gate'])
        self.assertEqual(result['insufficient_crossed_support'], [])
        self.assertEqual(result['unmet_interval_requirements'], ['old/clearance/delta_full_mae_s'])


if __name__ == '__main__':
    unittest.main()
