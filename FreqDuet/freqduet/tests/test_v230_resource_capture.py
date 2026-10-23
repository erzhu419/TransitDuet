import copy
import unittest

from preregister_freqduet_v230_resource_capture import (
    FIRST_PAIR_CHECKS, check_cycle_progress, check_query_link, first_pair_passed, registration)


def example():
    identity = dict(bus_id=0, trip_id=30, direction=True, station_id=7)
    links = dict(legacy_query_event_seq=12, query_order=100, prepare_order=95, query_occurrence=0)
    query = dict(links, query_s=101., identity=identity)
    prepare = dict(clock=dict(handoff_order=95, prepare_s=100.), current=identity)
    launch = dict(kind='launch', event_seq=3, time_s=10., bus_id=0, trip_id=30)
    versions = {'initial': dict(initial=True), 9: dict(initial=False, direction=False,
        native_event_seq=9, time_s=100., version_id=9)}
    row = dict(links, snapshot_read_s=101., snapshot_event_seq=12, onboard=5,
        executed_holding_before_s=20., terminal_masked=False,
        ready=[dict(bus_id=i, position_sample_s=101.) for i in range(1, 12)],
        active=[dict(bus_id=0, trip_id=30, position_sample_s=101.)],
        plans=[dict(trip_id=40)], cycle_ref=[0, 30, 3],
        message_refs={'false': 9, 'true': 'initial'}, pending_plan_version_refs={'40': 9})
    return row, query, prepare, launch, versions


class ResourceCaptureContractTest(unittest.TestCase):
    def test_reserved_paths_reuse_both_archived_controls(self):
        spec = registration('test')
        self.assertEqual(set(spec['current_budget'].values()), {0})
        self.assertFalse(spec['collection_authorized'])
        budget = spec['reserved_passive']
        self.assertFalse(budget['launch_authorized'])
        self.assertEqual(budget['max_started_paths'], len(budget['slots']))
        self.assertEqual(budget['max_started_paths'], 8)
        self.assertEqual(budget['extra_reference_paths'], 0)
        self.assertEqual(len({s['slot'] for s in budget['slots']}), 8)
        for slot in budget['slots']:
            self.assertIn(slot['arm'], ('v107', 'upper'))
            self.assertEqual(slot['sampler_seed'], 187000000+slot['actor']+100*slot['demand_day'])
            self.assertTrue(slot['handoff_reference'].endswith('/'+slot['slot']+'.json.gz'))

    def test_first_pair_needs_both_complete_engineering_results(self):
        rows = [dict(slot=s, **dict.fromkeys(FIRST_PAIR_CHECKS, True)) for s in ('c0_v107', 'c0_upper')]
        self.assertTrue(first_pair_passed(rows))
        self.assertFalse(first_pair_passed(rows[:1]))
        for key in FIRST_PAIR_CHECKS:
            changed = copy.deepcopy(rows)
            changed[1][key] = False
            self.assertFalse(first_pair_passed(changed))

    def test_query_clock_keeps_older_prepare_evidence(self):
        check_query_link(*example())

    def test_query_snapshot_cannot_be_mislabeled_as_prepare(self):
        args = list(example())
        args[0]['snapshot_read_s'] = 100.
        with self.assertRaisesRegex(AssertionError, 'query clock'):
            check_query_link(*args)

    def test_native_and_sidecar_ordinals_cannot_be_swapped(self):
        args = list(example())
        args[0]['snapshot_event_seq'] = args[0]['query_order']
        with self.assertRaisesRegex(AssertionError, 'snapshot ordinal'):
            check_query_link(*args)

    def test_same_clock_future_callback_does_not_enter_query(self):
        args = list(example())
        args[4][13] = dict(initial=False, direction=False, native_event_seq=13, time_s=101., version_id=13)
        args[0]['message_refs']['false'] = 13
        with self.assertRaisesRegex(AssertionError, 'latest completed'):
            check_query_link(*args)

    def test_future_plan_write_cannot_enter_pending_plan_provenance(self):
        args = list(example())
        args[4][13] = dict(initial=False, direction=False, native_event_seq=13, time_s=101., version_id=13)
        args[0]['pending_plan_version_refs']['40'] = 13
        with self.assertRaisesRegex(AssertionError, 'future plan'):
            check_query_link(*args)

    def test_missing_payload_is_not_zero_support(self):
        args = list(example())
        args[0].pop('onboard')
        with self.assertRaisesRegex(AssertionError, 'missing extension'):
            check_query_link(*args)

    def test_other_trip_on_reused_bus_cannot_supply_cycle_identity(self):
        args = list(example())
        args[3]['trip_id'] = 29
        with self.assertRaises(AssertionError):
            check_query_link(*args)

    def test_prefix_is_not_reset_by_upper_replan(self):
        first = example()[0]
        later = copy.deepcopy(first)
        later.update(legacy_query_event_seq=20, message_refs={'false': 19, 'true': 'initial'})
        check_cycle_progress(first, later)
        later['executed_holding_before_s'] = 0.
        with self.assertRaisesRegex(AssertionError, 'reset'):
            check_cycle_progress(first, later)


if __name__ == '__main__':
    unittest.main()
