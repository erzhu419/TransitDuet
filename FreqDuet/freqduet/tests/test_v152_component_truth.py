import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v152_component_truth as audit


def packet(service, running, holding, ticks=0., joint=0.):
    result = audit.parts.zeros()
    result.update(downstream_service=service, running=running, holding=holding,
        execution_wait=ticks, current_ready_joint=joint)
    return result


class ComponentTruthTests(unittest.TestCase):
    def test_improved_service_can_remove_error_cancellation(self):
        r = audit.component_errors(dict(v149=packet(30., 80., 0.),
            joint=packet(20., 80., 0.)), packet(20., 100., 0.))
        self.assertEqual(r['delta_service_mae_s'], -10.)
        self.assertEqual(r['delta_full_mae_s'], 10.)
        self.assertEqual(r['delta_cancellation_s'], -10.)

    def test_worsened_service_is_not_called_lost_cancellation(self):
        r = audit.component_errors(dict(v149=packet(20., 100., 0.),
            joint=packet(10., 100., 0.)), packet(20., 100., 0.))
        self.assertEqual(r['delta_service_mae_s'], 10.)
        self.assertEqual(r['delta_full_mae_s'], 10.)
        self.assertEqual(r['delta_cancellation_s'], 0.)

    def test_joint_residence_has_comparable_total_not_fabricated_holding(self):
        r = audit.component_errors(dict(v149=packet(20., 100., 15., 5.),
            joint=packet(20., 100., 0., joint=20.)), packet(20., 100., 17., 3.))
        self.assertEqual(r['delta_residence_mae_s'], 0.)
        self.assertEqual(r['joint_full_mae_s'], 0.)

    def test_unallocated_packet_cannot_be_reported_as_components(self):
        p = packet(0., 0., 0.)
        p['unallocated'] = 4.
        with self.assertRaises(AssertionError):
            audit.component_errors(dict(v149=p, joint=p), packet(0., 0., 0.))


if __name__ == '__main__':
    unittest.main()
