from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_freqduet_v156_tradeoffs as summary


def fixture():
    arms = {}
    for name, running, cancel in (('full', -2., -.5), ('single', -3., -.25), ('none', 0., 0.)):
        metrics = dict(reference_mae_s=20., delta_running_mae_s=running,
            delta_service_mae_s=0., delta_residence_mae_s=0.,
            delta_cancellation_s=cancel, delta_full_mae_s=running-2*cancel)
        arms[name] = {g: dict(queries=7, metrics=deepcopy(metrics)) for g in summary.SCOPES}
    return dict(seed=1, day=2, modes=dict(old=arms))


class TradeoffTests(unittest.TestCase):
    def test_direct_contrast_and_cancellation_sign(self):
        result = summary.paired(fixture())['groups']
        self.assertEqual(result['old/clearance']['metrics']['delta_full_mae_s'], -1.5)
        self.assertEqual(result['old/full/clearance_budget']['metrics']['cancellation_loss_s'], 1.)

    def test_different_support_is_not_a_paired_comparison(self):
        cell = fixture()
        cell['modes']['old']['single']['clearance']['queries'] = 6
        with self.assertRaises(AssertionError):
            summary.paired(cell)

    def test_different_reference_is_not_absorbed(self):
        cell = fixture()
        cell['modes']['old']['single']['all']['metrics']['reference_mae_s'] = 21.
        with self.assertRaises(AssertionError):
            summary.paired(cell)


if __name__ == '__main__':
    unittest.main()
