import json
from types import SimpleNamespace
import unittest

from freqduet_v223_common_reference import command
from preregister_freqduet_v225_action_smoke import (
    FIRST_PATH_CHECKS, NO_HARM, descriptive_joint_passed, first_path_passed, registration)


class ActionSmokeContractTest(unittest.TestCase):
    def test_seal_has_no_execution_or_archive_budget(self):
        spec = registration('test')
        self.assertEqual(sum(spec['current_budget'].values()), 0)
        self.assertFalse(spec['reserved_smoke']['launch_authorized'])
        self.assertFalse(spec['replay_implemented'])
        self.assertFalse(spec['independent_futures_opened'])
        self.assertFalse(spec['main_policy_changed'])

    def test_only_four_fixed_joint_paths_and_existing_controls(self):
        reserve = registration('test')['reserved_smoke']
        self.assertEqual(reserve['max_started_paths'], 4)
        self.assertEqual(reserve['extra_reference_paths'], 0)
        self.assertEqual(reserve['max_workers'], 2)
        self.assertEqual(reserve['first_slot'], 'c0_joint')
        self.assertEqual(reserve['subsequent_slots'], ['c1_joint', 'c2_joint', 'c3_joint'])
        self.assertEqual([(s['actor'], s['demand_day']) for s in reserve['slots']],
            [(12037, 2151001), (12037, 2151002), (933083, 2151001), (933083, 2151002)])
        for ci, slot in enumerate(reserve['slots']):
            self.assertEqual(slot['slot'], f'c{ci}_joint')
            self.assertEqual(slot['sampler_seed'], 187000000+slot['actor']+100*slot['demand_day'])
            self.assertEqual(set(slot['references']), {'v107', 'upper'})
            self.assertTrue(slot['references']['upper'].endswith(f'/c{ci}_upper.json.gz'))

    def test_engineering_gate_requires_every_check_and_executed_change(self):
        checks = dict.fromkeys(FIRST_PATH_CHECKS, True)
        self.assertTrue(first_path_passed(checks, 1))
        self.assertFalse(first_path_passed(checks, 0))
        for key in FIRST_PATH_CHECKS:
            self.assertFalse(first_path_passed(dict(checks, **{key: False}), 1))
            self.assertFalse(first_path_passed(dict(checks, **{key: None}), 1))
        # An unfavorable endpoint must not select which other contexts get run.
        self.assertTrue(first_path_passed(dict(checks, passenger_delta=100.), 1))

    def test_joint_comparison_requires_both_controls_and_all_constraints(self):
        effects = {arm: dict.fromkeys(NO_HARM, 0.) for arm in ('v107', 'upper')}
        for row in effects.values():
            row['passenger_person_s'] = -1.
        self.assertTrue(descriptive_joint_passed(effects))
        for arm in effects:
            for metric in NO_HARM:
                changed = {a: dict(v) for a, v in effects.items()}
                changed[arm][metric] = 1e-12
                self.assertFalse(descriptive_joint_passed(changed))
            changed = {a: dict(v) for a, v in effects.items()}
            changed[arm]['passenger_person_s'] = 0.
            self.assertFalse(descriptive_joint_passed(changed))

    def test_lower_only_zero_phase_or_fallback_is_exactly_native(self):
        bins = tuple(range(7))
        for forward_valid, follower_valid in ((True, True), (True, False), (False, True)):
            for f, b in ((100., 105.), (80., 180.), (180., 80.), (100., 260.)):
                ctx = SimpleNamespace(evidence_valid=forward_valid,
                    follower_evidence_valid=follower_valid, target_headway_s=180.,
                    forward_headway_s=f, follower_departure_gap_s=b)
                native = command(ctx, bins)
                for phases in ((0., 0., 0.), (None, 0., 0.)):
                    result = command(ctx, bins, phases)
                    for key in ('raw_s', 'clipped_s', 'bin', 'action_s'):
                        self.assertEqual(result[key], native[key])

    def test_live_law_allows_both_change_directions_and_abstains(self):
        ctx = SimpleNamespace(evidence_valid=True, follower_evidence_valid=True,
            target_headway_s=180., forward_headway_s=100., follower_departure_gap_s=200.)
        # Native raw12.5 ties toward10s. A delivered predecessor advance raises it.
        self.assertEqual(command(ctx, range(7))['action_s'], 10.)
        self.assertEqual(command(ctx, range(7), (-1., 0., 0.))['action_s'], 15.)
        ctx.follower_departure_gap_s = 201.
        # Native raw12.625 goes to15s; own advance moves the same boundary downward.
        self.assertEqual(command(ctx, range(7))['action_s'], 15.)
        self.assertEqual(command(ctx, range(7), (0., -1., 0.))['action_s'], 10.)
        self.assertEqual(command(ctx, range(7), (0., None, 0.))['action_s'], 15.)

    def test_protocol_roundtrip_and_frozen_sources(self):
        spec = registration('test')
        self.assertEqual(json.loads(json.dumps(spec)), spec)
        self.assertEqual(spec['frozen']['law_source'], '7939ddaab6')
        self.assertEqual(spec['frozen']['passive_collector_source'], '17043d5c03')
        self.assertEqual(spec['frozen']['gain'], .25)
        self.assertEqual(spec['frozen']['fleet'], 12)
        self.assertEqual(spec['frozen']['action_bins_s'], [0., 5., 10., 15., 20., 30., 45.])


if __name__ == '__main__':
    unittest.main()
