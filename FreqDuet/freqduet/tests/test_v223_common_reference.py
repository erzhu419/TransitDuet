import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from freqduet_v223_common_reference import ACTIONS, command, launch_phase, registration


def context(forward=300., backward=420., **changes):
    return SimpleNamespace(**(dict(evidence_valid=True, follower_evidence_valid=True,
        forward_headway_s=forward, follower_departure_gap_s=backward,
        target_headway_s=360.) | changes))


def launch(**changes):
    return dict(trip_id=7, bus_id=3, direction=True, launch_event_seq=2,
        native_tick_s=100., coordinated_tick_s=99., actual_launch_s=99.,
        reserved_bus_id=3) | changes


class CommonReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile only the unchanged oracle function; no rollout imports execute.
        source = Path(__file__).resolve().parents[1]/'scripts/run_freqduet_v106_service_tradeoff.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'rule_action')
        step_source = source.with_name('freqduet_system_policy_step.py')
        bins = next(n for n in ast.parse(step_source.read_text()).body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ACTIONS'
                                                        for t in n.targets))
        step_namespace = dict(np=np)
        exec(compile(ast.Module(body=[bins], type_ignores=[]), str(step_source), 'exec'), step_namespace)
        namespace = dict(np=np, pilot=SimpleNamespace(step=SimpleNamespace(ACTIONS=step_namespace['ACTIONS'])))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        cls.native_rule = staticmethod(namespace['rule_action'])

    def test_zero_phase_exact_native_with_feasibility_clipping_and_ties(self):
        for mask in ((True,)*7, (True, True, False, True, False, True, False), (True,)+(False,)*6):
            bins = np.flatnonzero(mask).tolist()
            for f in (0., 100., 300., 360., 900.):
                for b in (0., 320., 420., 900.):
                    c = context(f, b)
                    self.assertEqual(command(c, bins, (0., 0., 0.))['bin'],
                                     self.native_rule(c, 'bilateral', .25, np.array(mask)))
        self.assertEqual(command(context(300., 320.), range(7), (0., 0., 0.))['bin'], 0)

    def test_native_quantization_near_every_boundary(self):
        for left, right in zip(ACTIONS[:-1], ACTIONS[1:]):
            for epsilon in (-1e-7, 0., 1e-7):
                raw = (left+right)/2+epsilon
                c = context(300., 300.+8*raw)
                self.assertEqual(command(c, range(7), (0., 0., 0.))['bin'],
                                 self.native_rule(c, 'bilateral', .25, np.ones(7, bool)))

    def test_common_phase_translation_has_no_effect(self):
        a = command(context(), range(7), (-1., -1., -1.))
        self.assertEqual(a['raw_s'], 15.)
        self.assertEqual(a['phase_correction_s'], 0.)

    def test_relative_phase_is_not_compensated_again(self):
        for phases in ((0., -1., 0.), (-1., 0., -1.), (-1., -1., 0.)):
            p, own, follower = phases
            c = context(300.+own-p, 420.+follower-own)
            self.assertEqual(command(c, range(7), phases)['raw_s'], 15.)

    def test_persistent_phase_is_not_spent_again_at_each_stop(self):
        for _ in range(20):
            self.assertEqual(command(context(299., 421.), range(7), (0., -1., 0.))['raw_s'], 15.)

    def test_real_nonholding_gap_change_is_not_removed(self):
        r = command(context(199., 421.), range(7), (0., -1., 0.))
        self.assertEqual(r['raw_s'], 27.5)

    def test_executed_holding_enters_measured_gap_once(self):
        r = command(context(304., 416.), range(7), (0., -1., 0.))
        self.assertEqual(r['raw_s'], 13.75)

    def test_sub_bin_offset_does_not_force_one_bin_reduction(self):
        r = command(context(), range(7), (0., -1., 0.))
        self.assertEqual(r['phase_correction_s'], -.25)
        self.assertEqual(r['action_s'], 15.)
        self.assertEqual(command(context(300., 320.8), range(7), (0., -1., 0.))['bin'], 0)

    def test_unsupported_phase_keeps_native_not_partial_offset(self):
        for phases in (None, (None, -1., 0.), (0., -1., None)):
            r = command(context(), range(7), phases)
            self.assertEqual(r['mode'], 'native_missing_phase')
            self.assertEqual(r['raw_s'], 15.)

    def test_forward_fallback_is_unchanged_not_double_counted(self):
        c = context(follower_evidence_valid=False, follower_departure_gap_s=None)
        r = command(c, range(7), (0., -1., 0.))
        self.assertEqual(r['mode'], 'native_forward_fallback')
        self.assertEqual(r['bin'], self.native_rule(c, 'bilateral', .25, np.ones(7, bool)))
        c.evidence_valid = False
        self.assertEqual(command(c, range(7), (0., -1., 0.))['bin'], 0)

    def test_launch_phase_requires_causal_same_identity_and_delivered_clock(self):
        identity = (7, 3, True)
        self.assertEqual(launch_phase(launch(), identity, 3), -1.)
        for row, who, seq in ((None, identity, 3), (launch(), (7, 4, True), 3),
            (launch(), (8, 3, True), 3), (launch(), (7, 3, False), 3),
            (launch(), identity, 2), (launch(actual_launch_s=100.), identity, 3),
            (launch(reserved_bus_id=4), identity, 3),
            (launch(coordinated_tick_s=101., actual_launch_s=101.), identity, 3)):
            self.assertIsNone(launch_phase(row, who, seq))

    def test_unmodified_late_launch_has_known_zero_not_unknown(self):
        self.assertEqual(launch_phase(launch(coordinated_tick_s=100., actual_launch_s=120.),
                                     (7, 3, True), 3), 0.)

    def test_registration_opens_no_runtime_or_data_budget(self):
        r = json.loads(json.dumps(registration('test_source')))
        self.assertEqual(sum(r['budget'].values()), 0)
        self.assertFalse(r['runtime_connected'])
        self.assertFalse(r['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
