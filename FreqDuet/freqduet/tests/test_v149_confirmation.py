from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v149_confirmation as audit


class ConfirmationTests(unittest.TestCase):
    def test_service_change_is_not_allocated_to_current_service_or_running(self):
        parts = audit.components.zeros()
        parts.update(current_service=2., running=3., downstream_service=12.)
        result = audit.service_parts(parts, -5., True)
        self.assertEqual((result['current_service'], result['running'], result['downstream_service']), (2., 3., 7.))
        self.assertEqual(sum(result.values()), sum(parts.values())-5.)
        self.assertEqual(parts['downstream_service'], 12.)

    def test_opaque_fallback_cannot_hide_a_service_change(self):
        parts = audit.components.zeros()
        parts['unallocated'] = 10.
        self.assertEqual(audit.service_parts(parts, 0., False), parts)
        with self.assertRaises(AssertionError):
            audit.service_parts(parts, -3., False)

    def test_no_observed_switch_has_zero_burden_not_zero_conditional_error(self):
        result = audit.burden([], 20)
        self.assertEqual(result['metrics'], {a+'_mae_s': 0. for a in audit.ARMS})
        self.assertEqual(audit.burden([], 0)['metrics'], {})


if __name__ == '__main__':
    unittest.main()
