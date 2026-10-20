from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v156_clearance_support as audit


class ClearanceSupportTests(unittest.TestCase):
    def test_fixed_day_deletion_does_not_refit_or_mutate(self):
        cells = [dict(station=2, days=[dict(day=3, mean=20.), dict(day=4, mean=40.)])]
        self.assertEqual(audit.retain_days(cells, [3])[0]['days'], [dict(day=3, mean=20.)])
        self.assertEqual(audit.retain_days(cells, [])[0]['days'], [])
        self.assertEqual(len(cells[0]['days']), 2)

    def test_service_clock_feedback_is_not_labeled_running_or_holding(self):
        reference = audit.parts.zeros()
        reference.update(running=100., downstream_service=20., holding=10.)
        packet = audit.shadow_packet(reference, 130., 144., 110.)
        self.assertEqual(packet['downstream_service'], 24.)
        self.assertEqual(packet['holding'], 10.)
        self.assertEqual(sum(packet.values()), 144.)
        self.assertEqual(reference['running'], 100.)

    def test_reference_point_mismatch_is_not_absorbed_as_service(self):
        reference = audit.parts.zeros()
        reference['running'] = 100.
        with self.assertRaises(AssertionError):
            audit.shadow_packet(reference, 120., 130., 110.)

    def test_endpoint_difference_remains_paired(self):
        record = audit.full_record([1], 30., 25., 20.)
        self.assertEqual(record['delta_full_mae_s'], -5.)
        self.assertEqual(record['reference_bias_s'], 10.)


if __name__ == '__main__':
    unittest.main()
