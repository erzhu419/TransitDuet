import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_freqduet_v244_six_arm_smoke import claim_slot, write, save, read, aggregate, ARMS, METRICS


class SmokeSlotsTest(unittest.TestCase):
    def test_only_six_fixed_slots_and_no_second_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ('formal', 'joint', 'c1_v107'):
                with self.assertRaises(AssertionError):
                    claim_slot(root, arm, 'test')
            claim_slot(root, 'v107', 'test')
            with self.assertRaises(FileExistsError):
                claim_slot(root, 'v107', 'test')
            self.assertEqual(len(list((root/'started').glob('*.json'))), 1)

    def test_five_remaining_arms_require_passing_reference_and_fork(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ARMS[1:]:
                with self.assertRaises(FileNotFoundError):
                    claim_slot(root, arm, 'test')
            path = root/'checks/smoke_c0_v107.json'
            for passed, ready in ((False, False), (False, True), (True, False)):
                write(path, dict(source='test', passed=passed, reference_ready=ready))
                with self.assertRaises(AssertionError):
                    claim_slot(root, 'joint_supply', 'test')
            write(path, dict(source='test', passed=True, reference_ready=True))
            for arm in ARMS[1:]:
                claim_slot(root, arm, 'test')
            self.assertEqual(len(list((root/'started').glob('*.json'))), 5)

    def test_failure_does_not_refund_reference_slot_or_enable_others(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim_slot(root, 'v107', 'test')
            write(root/'checks/smoke_c0_v107.json', dict(source='test', passed=False, reference_ready=False))
            with self.assertRaises(FileExistsError):
                claim_slot(root, 'v107', 'test')
            with self.assertRaises(AssertionError):
                claim_slot(root, 'upper_only', 'test')
            started = json.loads((root/'started/smoke_c0_v107.json').read_text())
            self.assertFalse(started['retry_authorized'])

    def test_native_future_types_survive_server_record_roundtrip(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'record.pkl.gz'
            payload = dict(phase={('trip', 1): np.array([3., 7.])}, rng=torch.tensor([1, 2], dtype=torch.uint8))
            save(path, payload)
            restored = read(path)
            self.assertTrue(np.array_equal(restored['phase']['trip', 1], payload['phase']['trip', 1]))
            self.assertTrue(torch.equal(restored['rng'], payload['rng']))

    def test_aggregation_retains_tick_for_frozen_execution_gate(self):
        module = 'run_freqduet_v244_six_arm_smoke'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ARMS:
                write(root/'checks'/f'smoke_c0_{arm}.json', dict(
                    passed=True, source='test', metrics={k: 0 for k in METRICS}))
                write(root/'started'/f'smoke_c0_{arm}.json', dict(started=True))
            stored = dict.fromkeys(('arm', 'anchor', 'future', 'physical_contract',
                'ledger', 'completion', 'uniforms', 'execution', 'tasks', 'dispatches',
                'launch_plans', 'requests', 'main_policy_changed'))
            stored['tick_s'] = 1.
            legacy = SimpleNamespace(pilot=SimpleNamespace(
                joint=SimpleNamespace(assert_same_state=None), event=SimpleNamespace(alignment=None)),
                gate=SimpleNamespace(task_execution=None))
            resource = SimpleNamespace(frozen=lambda: SimpleNamespace(legacy=legacy))
            with patch.dict('sys.modules', {'run_freqduet_v230_resource_capture': resource}), \
                 patch(module+'.read', return_value=stored), \
                 patch(module+'.adapter.check_smoke', return_value=dict(passed=True)) as gate:
                result = aggregate(root, 'test')
            self.assertEqual(gate.call_args.args[-1], 1.)
            self.assertEqual(result['status'], 'v244_smoke_engineering_passed_formal_closed')
            self.assertEqual(result['reserved_starts_consumed'], 8)
            self.assertFalse(result['formal_authorized'])


if __name__ == '__main__':
    unittest.main()
