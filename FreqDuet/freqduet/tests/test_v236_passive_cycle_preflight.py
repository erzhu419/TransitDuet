import json
from pathlib import Path
import tempfile
import unittest

from preregister_freqduet_v234_passive_cycle_preflight import PAIR_CHECKS
from run_freqduet_v236_passive_cycle_preflight import claim_slot, write


class PassivePairBudgetTest(unittest.TestCase):
    def test_only_two_fixed_slots_and_no_duplicate_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(AssertionError):
                claim_slot(root, 'joint', PAIR_CHECKS)
            claim_slot(root, 'v107', PAIR_CHECKS)
            with self.assertRaises(FileExistsError):
                claim_slot(root, 'v107', PAIR_CHECKS)
            self.assertEqual(len(list((root/'started').glob('*.json'))), 1)

    def test_upper_cannot_start_without_v107_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                claim_slot(root, 'upper', PAIR_CHECKS)
            self.assertFalse((root/'started/c0_upper.json').exists())

    def test_every_v107_gate_is_required_before_upper_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passing = dict(slot='c0_v107', passed=True, **{k: True for k in PAIR_CHECKS})
            for key in PAIR_CHECKS:
                write(root/'checks/c0_v107.json', dict(passing, **{key: False}))
                with self.assertRaises(AssertionError):
                    claim_slot(root, 'upper', PAIR_CHECKS)
            self.assertFalse((root/'started/c0_upper.json').exists())
            write(root/'checks/c0_v107.json', passing)
            claim_slot(root, 'upper', PAIR_CHECKS)

    def test_failed_v107_blocks_upper_and_retains_consumed_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim_slot(root, 'v107', PAIR_CHECKS)
            write(root/'checks/c0_v107.json', dict(slot='c0_v107', passed=False, failure='recording failure'))
            with self.assertRaises(AssertionError):
                claim_slot(root, 'upper', PAIR_CHECKS)
            self.assertEqual([p.stem for p in (root/'started').glob('*.json')], ['c0_v107'])

    def test_frozen_observer_acceptance_survives_json_round_trip(self):
        import random
        import numpy as np
        import torch
        from freqduet_v235_passive_cycle_capture import validate_capture
        from test_v235_passive_cycle_capture import Fixture
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        f = Fixture()
        try:
            f.complete()
            originals = json.loads(json.dumps(f.originals()))
            capture = json.loads(json.dumps(f.audit.cycle_capture.data(True)))
            result = validate_capture(originals, originals, capture)
            self.assertTrue(all(result[k] is True for k in PAIR_CHECKS))
        finally:
            f.audit.detach()
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])


if __name__ == '__main__':
    unittest.main()
