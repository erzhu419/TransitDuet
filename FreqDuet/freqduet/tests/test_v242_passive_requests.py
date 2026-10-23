import json
from pathlib import Path
import tempfile
import unittest

from run_freqduet_v242_passive_requests import accepted, claim_slot, GATES, write


class PassiveSlotTest(unittest.TestCase):
    def test_only_reserved_passive_slots_and_no_duplicate_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ('joint', 'smoke', 'lower_supply'):
                with self.assertRaises(AssertionError):
                    claim_slot(root, arm, 'test')
            claim_slot(root, 'v107', 'test')
            with self.assertRaises(FileExistsError):
                claim_slot(root, 'v107', 'test')
            record = json.loads((root/'started/passive_c0_v107.json').read_text())
            self.assertFalse(record['retry_authorized'])

    def test_upper_requires_every_equality_and_recording_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                claim_slot(root, 'upper_only', 'test')
            passing = dict(slot='passive_c0_v107', source='test', passed=True,
                           **{k: True for k in GATES})
            for key in GATES:
                write(root/'checks/passive_c0_v107.json', dict(passing, **{key: False}))
                with self.assertRaises(AssertionError):
                    claim_slot(root, 'upper_only', 'test')
            self.assertFalse((root/'started/passive_c0_upper_only.json').exists())
            write(root/'checks/passive_c0_v107.json', passing)
            claim_slot(root, 'upper_only', 'test')
            record = json.loads((root/'started/passive_c0_upper_only.json').read_text())
            self.assertEqual(record['engine_arm'], 'upper')

    def test_failed_start_remains_consumed_and_blocks_upper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim_slot(root, 'v107', 'test')
            write(root/'checks/passive_c0_v107.json', dict(slot='passive_c0_v107', source='test', passed=False))
            with self.assertRaises(AssertionError):
                claim_slot(root, 'upper_only', 'test')
            self.assertEqual(len(list((root/'started').glob('*.json'))), 1)

    def test_missing_or_partial_gate_is_not_acceptance(self):
        self.assertFalse(accepted(dict(passed=True)))
        self.assertFalse(accepted(dict(passed=False, **{k: True for k in GATES})))
        self.assertTrue(accepted(dict(passed=True, **{k: True for k in GATES})))


if __name__ == '__main__':
    unittest.main()
