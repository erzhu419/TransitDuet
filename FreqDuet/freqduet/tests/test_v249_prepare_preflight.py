import json
from pathlib import Path
import tempfile
import unittest

from run_freqduet_v249_prepare_preflight import COLLECTION, GATES, accepted, claim_slot


class PassiveSlotsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/COLLECTION

    def tearDown(self):
        self.temp.cleanup()

    def test_first_start_is_charged_and_cannot_restart(self):
        claim_slot(self.root, 'v107', 'source')
        self.assertEqual(len(list((self.root/'started').glob('*.json'))), 1)
        with self.assertRaises(FileExistsError):
            claim_slot(self.root, 'v107', 'source')

    def test_upper_needs_a_complete_passing_first_result(self):
        with self.assertRaises(FileNotFoundError):
            claim_slot(self.root, 'upper', 'source')
        p = self.root/'checks/c0_v107.json'
        p.parent.mkdir(parents=True)
        for missing in GATES:
            row = dict(source='source', arm='v107', passed=True, **dict.fromkeys(GATES, True))
            row[missing] = False
            p.write_text(json.dumps(row))
            with self.subTest(gate=missing), self.assertRaises(AssertionError):
                claim_slot(self.root, 'upper', 'source')

    def test_only_two_fixed_slots_and_same_source_are_allowed(self):
        with self.assertRaises(AssertionError):
            claim_slot(self.root, 'other_context', 'source')
        p = self.root/'checks/c0_v107.json'
        p.parent.mkdir(parents=True)
        row = dict(source='source', arm='v107', passed=True, **dict.fromkeys(GATES, True))
        p.write_text(json.dumps(row))
        with self.assertRaises(AssertionError):
            claim_slot(self.root, 'upper', 'different')
        claim_slot(self.root, 'v107', 'source')
        claim_slot(self.root, 'upper', 'source')
        self.assertEqual(len(list((self.root/'started').glob('*.json'))), 2)

    def test_a_failed_result_cannot_pass_by_partial_flags(self):
        row = dict(passed=False, **dict.fromkeys(GATES, True))
        self.assertFalse(accepted(row))
        self.assertFalse(accepted(dict(passed=True)))
        row['passed'] = True
        self.assertTrue(accepted(row))


if __name__ == '__main__':
    unittest.main()
