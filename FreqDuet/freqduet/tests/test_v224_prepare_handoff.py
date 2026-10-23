import copy
import json
import unittest

from preregister_freqduet_v224_prepare_handoff import registration, require_legacy_equal


class PrepareHandoffContractTest(unittest.TestCase):
    def test_current_stage_has_no_data_or_execution_budget(self):
        r = registration('test')
        self.assertEqual(sum(r['current_budget'].values()), 0)
        self.assertFalse(r['reserved_preflight']['launch_authorized'])
        self.assertFalse(r['collector_implemented'])
        self.assertFalse(r['main_policy_changed'])

    def test_reserved_slots_are_exact_archived_arms_and_seeds(self):
        r = registration('test')['reserved_preflight']
        self.assertEqual(r['max_started_paths'], 8)
        self.assertEqual(r['extra_reference_paths'], 0)
        self.assertEqual([(s['context'], s['arm']) for s in r['slots']],
                         [(c, arm) for c in range(4) for arm in ('v107', 'upper')])
        self.assertEqual(len({s['slot'] for s in r['slots']}), 8)
        for s in r['slots']:
            self.assertEqual(s['sampler_seed'], 187000000+s['actor']+100*s['demand_day'])
            self.assertTrue(s['archived_reference'].endswith('/'+s['slot']+'.json.gz'))

    def test_first_pair_blocks_the_other_six(self):
        r = registration('test')['reserved_preflight']
        self.assertEqual(r['first_pair'], ['c0_v107', 'c0_upper'])
        self.assertEqual(r['subsequent_slots'], [s['slot'] for s in r['slots'][2:]])
        self.assertEqual(len(r['subsequent_slots']), 6)

    def test_role_identity_clock_and_phase_fields_are_explicit(self):
        fields = registration('test')['prepare_fields']
        self.assertTrue({'trip_id', 'bus_id', 'direction', 'station_id'} <= set(fields['current']))
        self.assertTrue({'trip_id', 'bus_id', 'direction', 'departure_s', 'departure_order'} <= set(fields['front']))
        self.assertTrue({'trip_id', 'bus_id', 'direction', 'avl_sample_s', 'avl_sample_order',
                         'avl_sample_origin'} <= set(fields['follower']))
        self.assertTrue({'phase_s', 'status', 'launch_order', 'admission_order', 'native_tick_s',
                         'coordinated_tick_s', 'actual_launch_s', 'plan_at_launch_s'} <= set(fields['role_phase']))
        self.assertIn('prepare_occurrence', fields['clock'])
        self.assertIn('legacy_prepare_event_seq', fields['clock'])
        self.assertIn('prepare_order', fields['query_join'])

    def test_equal_legacy_payloads_pass_without_modification(self):
        a = dict(trace=dict(actions=[0, 1], uniforms=[[1, .2]], launches=[100.]),
                 results=dict(journey_min=16., cv_squared=.05, readiness_delay_s=100.))
        b = copy.deepcopy(a)
        require_legacy_equal(a, b)
        self.assertEqual(a, b)

    def test_action_order_random_or_plan_timing_changes_fail(self):
        a = dict(trace=dict(actions=[0, 1], uniforms=[[1, .2]], launches=[100.],
                            plans=[99.5], event_order=[0, 1]))
        for key, changed in dict(actions=[1, 0], uniforms=[[1, .3]], launches=[101.],
                                 plans=[99.6], event_order=[1, 0]).items():
            b = copy.deepcopy(a)
            b['trace'][key] = changed
            with self.assertRaisesRegex(AssertionError, 'legacy mismatch: trace'):
                require_legacy_equal(a, b)

    def test_endpoint_change_or_new_legacy_field_fails(self):
        a = dict(results=dict(journey_min=16., cv_squared=.05))
        for key in a['results']:
            b = copy.deepcopy(a)
            b['results'][key] += 1e-12
            with self.assertRaises(AssertionError):
                require_legacy_equal(a, b)
        with self.assertRaisesRegex(AssertionError, 'key set changed'):
            require_legacy_equal(a, dict(a, prepare_handoff=[]))

    def test_protocol_roundtrips_with_explicit_no_collection(self):
        r = json.loads(json.dumps(registration('test')))
        self.assertEqual(r['stage'], 'v224_prepare_identity_phase_handoff_preregistration')
        self.assertFalse(r['reserved_preflight']['launch_authorized'])
        self.assertEqual(r['frozen']['shadow_law_source'], '7939ddaab6')


if __name__ == '__main__':
    unittest.main()
