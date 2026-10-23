import copy
import unittest

from preregister_freqduet_v234_passive_cycle_preflight import (
    JOIN_FIELDS, PAIR_CHECKS, check_originals, check_queries, pair_passed, registration)


def query_fixture():
    links = dict(zip(JOIN_FIELDS, (7, 11, 10, 1)))
    identity = dict(bus_id=2, trip_id=1, direction=False, station_id=4)
    resource = dict(links, cycle_ref=[2, 1, 1], snapshot_read_s=10., snapshot_event_seq=7)
    old = dict(links, identity=identity, query_s=10.)
    view = dict(cycle_key=[2, 1, 1], query_s=10., native_event_seq=7, executed_holding_s=5.,
        delta_executed_since_last_sample_s=5., previous_query_s=None,
        message_refs={'true': 6, 'false': 'initial'}, pending_plan_version_refs={'11': 6},
        target_profile=dict(uncovered_due=[], future_without_current_ready=[11],
            unbacked_obligations=[dict(trip_id=11, last_write_version=6)],
            position_associations=[dict(bus_id=2, obligation=dict(trip_id=11, last_write_version=6))]),
        own_association=dict(bus_id=2, obligation=dict(trip_id=11, last_write_version=6)))
    return dict(links, identity=identity, view=copy.deepcopy(view)), resource, old, view


class PassiveCyclePreflightTest(unittest.TestCase):
    def test_only_two_reserved_starts_no_current_authorization(self):
        p = registration('test')
        self.assertEqual(set(p['current_budget'].values()), {0})
        b = p['reserved_preflight']
        self.assertEqual(b['max_started_paths'], 2)
        self.assertEqual(b['order'], ['c0_v107', 'c0_upper'])
        self.assertEqual([s['slot'] for s in b['slots']], b['order'])
        self.assertEqual(b['max_workers'], 1)
        self.assertEqual(b['extra_reference_paths'], 0)
        self.assertEqual({s['demand_day'] for s in b['slots']}, {2151001})
        self.assertFalse(b['launch_authorized'] or p['collection_authorized'] or p['runtime_connected'])
        self.assertFalse(p['collector_implemented'] or p['main_policy_changed'])

    def test_no_partial_or_wrong_pair_acceptance(self):
        pair = [dict(slot=s, **{k: True for k in PAIR_CHECKS}) for s in ('c0_v107', 'c0_upper')]
        self.assertTrue(pair_passed(pair))
        self.assertFalse(pair_passed(pair[:1]))
        self.assertFalse(pair_passed([pair[0], pair[0]]))
        for key in PAIR_CHECKS:
            bad = copy.deepcopy(pair)
            bad[1].pop(key)
            self.assertFalse(pair_passed(bad), key)

    def test_complete_old_payloads_not_just_endpoints(self):
        ref = dict(legacy=dict(trace=[dict(action=3, time_s=10.)], results=dict(journey=18.)),
                   v224=dict(queries=[dict(query_order=11)], observer_checks=20),
                   v230=dict(query_extensions=[dict(snapshot_event_seq=7)], counts=dict(queries=1)))
        check_originals(ref, copy.deepcopy(ref))
        for name in ref:
            bad = copy.deepcopy(ref)
            bad[name]['extra'] = 0
            with self.assertRaises(AssertionError):
                check_originals(ref, bad)
        bad = copy.deepcopy(ref)
        bad['legacy']['trace'][0]['time_s'] += .001
        with self.assertRaises(AssertionError):
            check_originals(ref, bad)
        bad = copy.deepcopy(ref)
        bad['v224']['observer_checks'] += 1
        with self.assertRaises(AssertionError):
            check_originals(ref, bad)

    def test_all_query_occurrences_and_clocks_required(self):
        row, resource, old, view = query_fixture()
        check_queries([row], [resource], [old], [view])
        with self.assertRaises(AssertionError):
            check_queries([], [resource], [old], [view])
        for key, value in (('query_occurrence', 2), ('prepare_order', 9)):
            with self.assertRaises(AssertionError):
                check_queries([dict(row, **{key: value})], [resource], [old], [view])
        with self.assertRaises(AssertionError):
            check_queries([row], [dict(resource, snapshot_read_s=11.)], [old], [view])

    def test_frozen_view_catches_holding_reset_and_lost_plan_version(self):
        row, resource, old, view = query_fixture()
        for key, value in (('executed_holding_s', 0.), ('pending_plan_version_refs', {})):
            bad = copy.deepcopy(row)
            bad['view'][key] = value
            with self.assertRaises(AssertionError):
                check_queries([bad], [resource], [old], [view])

    def test_no_future_query_backfill(self):
        row, resource, old, view = query_fixture()
        row['actual_next_trip'] = 11
        with self.assertRaises(AssertionError):
            check_queries([row], [resource], [old], [view])
        row.pop('actual_next_trip')
        row['view']['actual_return_s'] = 20.
        with self.assertRaises(AssertionError):
            check_queries([row], [resource], [old], [view])

    def test_null_association_and_zero_holding_remain_rows(self):
        row, resource, old, view = query_fixture()
        view.update(executed_holding_s=0., delta_executed_since_last_sample_s=0.,
                    own_association=dict(bus_id=2, obligation=None))
        row['view'] = copy.deepcopy(view)
        check_queries([row], [resource], [old], [view])
        row['view']['own_association']['obligation'] = 0
        with self.assertRaises(AssertionError):
            check_queries([row], [resource], [old], [view])


if __name__ == '__main__':
    unittest.main()
