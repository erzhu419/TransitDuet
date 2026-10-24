import json
from pathlib import Path
import tempfile
import unittest

from run_freqduet_v256_passive_pulse import COLLECTION, GATES, accepted, claim_slot, path_output


class PassivePulseSlotsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/COLLECTION

    def tearDown(self):
        self.temp.cleanup()

    def first_result(self, **changes):
        row = dict(source='source', mode='observer_only', passed=True, **dict.fromkeys(GATES, True))
        row.update(changes)
        path = self.root/'checks/passive_c0_v107.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))

    def test_started_attempt_cannot_retry(self):
        claim_slot(self.root, 'observer_only', 'source')
        with self.assertRaises(FileExistsError):
            claim_slot(self.root, 'observer_only', 'source')

    def test_path_preserves_frozen_runtime_proof_parent(self):
        for mode in ('observer_only','forced_native_once'):
            output = path_output(self.root, mode)
            self.assertEqual(output.parent, self.root.parent)
            self.assertEqual(output.parent/'runtime_optimization_20261014',
                             self.root.parent/'runtime_optimization_20261014')
        self.assertNotEqual(path_output(self.root, 'observer_only'),
                            path_output(self.root, 'forced_native_once'))

    def test_noop_needs_every_first_path_gate(self):
        with self.assertRaises(FileNotFoundError):
            claim_slot(self.root, 'forced_native_once', 'source')
        for gate in ('passed', *GATES):
            self.first_result(**{gate: False})
            with self.subTest(gate=gate), self.assertRaises(AssertionError):
                claim_slot(self.root, 'forced_native_once', 'source')

    def test_exactly_two_fixed_slots_same_source(self):
        self.first_result()
        with self.assertRaises(AssertionError):
            claim_slot(self.root, 'forced_native_once', 'different')
        with self.assertRaises(AssertionError):
            claim_slot(self.root, 'smoke', 'source')
        claim_slot(self.root, 'observer_only', 'source')
        claim_slot(self.root, 'forced_native_once', 'source')
        self.assertEqual(len(list((self.root/'started').glob('*.json'))), 2)

    def test_incomplete_receipts_never_admit_next_path(self):
        self.assertFalse(accepted(dict(passed=True)))
        self.assertFalse(accepted(dict(passed=False, **dict.fromkeys(GATES, True))))
        self.first_result(mode='forced_native_once')
        with self.assertRaises(AssertionError):
            claim_slot(self.root, 'forced_native_once', 'source')


if __name__ == '__main__':
    unittest.main()
