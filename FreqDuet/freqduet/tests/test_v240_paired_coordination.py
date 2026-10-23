import copy
import json
from pathlib import Path
import tempfile
import unittest

from preregister_freqduet_v238_persistent_requests import shadow_request
from test_v238_persistent_requests import snapshot
from preregister_freqduet_v240_paired_coordination import (
    ARMS, CONSTRAINTS, SMOKE_SEED, blind_request, future_seed, joint_gate,
    namespace_audit, registration, reserved_slots)


class PairedProtocolTest(unittest.TestCase):
    def test_blind_matches_frozen_v238_when_own_exposure_exists(self):
        fields = ('shadow_bin', 'shadow_s', 'target_s', 'compensation_floor_s',
                  'local_gap_delta_vs_native_s2', 'local_gain_vs_zero_s2')
        for holding in (0, 5, 27, 45, 90, 300):
            for native in range(7):
                state, row, meta = snapshot(holding, native=native, difference=360)
                expected = shadow_request(state.query(row, meta), row)
                actual = blind_request(holding, row['shadow_frontier'])
                self.assertEqual(tuple(expected[k] for k in fields), tuple(actual[k] for k in fields))

    def test_blind_does_not_need_or_fabricate_a_supply_obligation(self):
        state, row, meta = snapshot()
        row['plans'], row['pending_plan_version_refs'] = [], {}
        original = copy.deepcopy(row)
        supply = shadow_request(state.query(row, meta), row)
        blind = blind_request(row['executed_holding_before_s'], row['shadow_frontier'])
        self.assertEqual((supply['shadow_s'], blind['shadow_s']), (20, 10))
        self.assertEqual(row, original)

    def test_same_native_mask_ties_zero_and_unsupported_fallbacks(self):
        for bins, holding, expected in (((0, 3, 4, 6), 45, 15), (range(7), 27, 15),
                                        ((2, 4), 45, 20)):
            _, row, _ = snapshot(holding, bins=bins)
            self.assertEqual(blind_request(holding, row['shadow_frontier'])['shadow_s'], expected)
        for status in ('terminal_masked', 'missing_forward', 'missing_follower'):
            _, row, _ = snapshot()
            row['shadow_frontier']['status'] = status
            self.assertEqual(blind_request(45, row['shadow_frontier'])['shadow_s'], 20)
        _, row, _ = snapshot(native=0)
        self.assertEqual(blind_request(1000, row['shadow_frontier'])['shadow_s'], 0)

    def test_h0_aliases_native_without_extra_physical_arms(self):
        for native in range(7):
            _, row, _ = snapshot(native=native, difference=360)
            self.assertEqual(blind_request(0, row['shadow_frontier'])['shadow_bin'], native)
        self.assertEqual(len(ARMS), 6)
        self.assertEqual(registration('test')['H0_aliases'],
                         dict(native_upper='v107', v220_upper='upper_only'))

    def test_reserved_starts_and_future_halves_are_fixed_and_disjoint(self):
        slots = reserved_slots()
        self.assertEqual(len(slots), 104)
        self.assertEqual(len({s['slot'] for s in slots}), 104)
        for phase, count in (('passive', 2), ('smoke', 6), ('formal', 96)):
            self.assertEqual(sum(s['phase'] == phase for s in slots), count)
        seeds = {future_seed(c, f) for c in range(4) for f in range(4)}
        self.assertEqual(len(seeds), 16)
        self.assertNotIn(SMOKE_SEED, seeds)
        left = {s['future_seed'] for s in slots if s.get('evaluation_half') == 'A'}
        right = {s['future_seed'] for s in slots if s.get('evaluation_half') == 'B'}
        self.assertEqual((len(left), len(right), len(left & right)), (8, 8, 0))
        with self.assertRaises(AssertionError):
            future_seed(0, 4)

    def test_gate_requires_strict_passenger_benefit_and_all_nonharm_bounds(self):
        stats = {m: dict(ci95=[-1, 0]) for m in CONSTRAINTS}
        stats['passenger_person_s'] = dict(ci95=[-2, -1])
        self.assertTrue(joint_gate(stats))
        stats['passenger_person_s']['ci95'][1] = 0
        self.assertFalse(joint_gate(stats))
        stats['passenger_person_s']['ci95'][1] = -1
        for metric in CONSTRAINTS:
            bad = copy.deepcopy(stats)
            bad[metric]['ci95'][1] = 0.0001
            self.assertFalse(joint_gate(bad))

    def test_namespace_audit_checks_registered_metadata_not_raw_archives(self):
        snapshot_meta = dict(namespace='FreqDuet/v240', conflicts=[])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root/'protocol_v6_v239_example'
            stage.mkdir()
            path = stage/'protocol.json'
            path.write_text(json.dumps(dict(future_namespace='2390000000+context')))
            report = namespace_audit(root, snapshot_meta)
            self.assertEqual((report['registered_metadata_files'], report['conflicts']), (1, []))
            path.write_text(json.dumps(dict(future_namespace='2400000000+context')))
            with self.assertRaises(AssertionError):
                namespace_audit(root, snapshot_meta)

    def test_seal_authorizes_no_replay_or_training(self):
        p = registration('test')
        self.assertEqual(set(p['current_budget'].values()), {0})
        self.assertFalse(p['runtime_connected'] or p['replay_authorized'] or
                         p['training_authorized'] or p['main_policy_changed'] or
                         p['reserved_budget']['launch_authorized'])
        self.assertEqual(p['reserved_budget']['total_starts_max'], len(reserved_slots()))


if __name__ == '__main__':
    unittest.main()
