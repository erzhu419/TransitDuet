import copy
import unittest

from preregister_freqduet_v247_prepare_timing import (
    OLD_PAYLOADS, join_query, map_prepare, prepare_sample, registration)
from preregister_freqduet_v246_compensation_supply import persistent_change


def fixture():
    identity = dict(bus_id=2, trip_id=22, station_id=7, direction=False)
    native = dict(identity, kind='prepare', time_s=1000., event_seq=20)
    launch = dict(bus_id=2, time_s=100., holding_counter_at_launch=300.)
    front = dict(bus_id=1, trip_id=21, station_id=7, direction=False,
        status='supported', departure_s=900., departure_order=1100, launch_order=1001)
    phases = dict(current=dict(identity, actual_launch_s=100., launch_order=1002),
                  front=dict(front, actual_launch_s=0.))
    old = dict(clock=dict(prepare_s=1000., record_read_s=1000., handoff_order=1200,
        legacy_prepare_event_seq=20, prepare_occurrence=0), current=identity,
        native=dict(forward_valid=True, follower_valid=True, forward_gap_s=100., follower_gap_s=140.),
        front=front, phases=phases)
    completed = dict(timing=dict(trip_id=21, station_id=7, direction=False, time_s=900.,
        release=0., realized_holding=40., other_elapsed=860.), launch=dict(bus_id=1, time_s=0.))
    sample = prepare_sample(native, launch, [2,22,3], 360., 1000., completed)
    return old, sample


class PrepareTimingTests(unittest.TestCase):
    def test_known_components_close(self):
        old, sample = fixture()
        row = map_prepare(old, sample)
        self.assertEqual(row['parts'], dict(release_s=100., executed_holding_gap_s=20.,
                                           nonholding_elapsed_gap_s=-20.))
        self.assertEqual(row['own']['holding_s'], 60.)

    def test_originals_not_mutated(self):
        old, sample = fixture()
        before = copy.deepcopy((old, sample))
        map_prepare(old, sample)
        self.assertEqual((old, sample), before)

    def test_same_front_persists_across_stations(self):
        old, sample = fixture()
        first = map_prepare(old, sample)
        old['current']['station_id'] = old['front']['station_id'] = 8
        old['clock'].update(prepare_s=1100., record_read_s=1100.,
                            legacy_prepare_event_seq=22, handoff_order=1300)
        old['front'].update(departure_s=1000., departure_order=1250)
        sample['native'].update(station_id=8, time_s=1100., event_seq=22)
        sample.update(read_s=1100., holding_s=70., counter_s=370.)
        sample['completed_front']['timing'].update(station_id=8, time_s=1000.,
            realized_holding=50., other_elapsed=950.)
        second = map_prepare(old, sample)
        self.assertEqual(first['front_identity'], second['front_identity'])
        delta = persistent_change(first, second)
        self.assertEqual(delta['status'], 'same_physical_cycle_and_front')
        self.assertEqual(sum(delta['parts'].values()), 0.)

    def test_zero_holding_is_supported(self):
        old, sample = fixture()
        sample.update(holding_s=0., counter_s=sample['launch_counter_s'])
        self.assertEqual(map_prepare(old, sample)['own']['holding_s'], 0.)

    def test_missing_front_is_not_zero(self):
        old, sample = fixture()
        sample['completed_front'] = None
        row = map_prepare(old, sample)
        self.assertIsNone(row['parts'])
        self.assertIsNone(row['front']['holding_s'])
        self.assertEqual(row['front_status'], 'completed_timing_unavailable')

    def test_no_native_front_stays_unknown(self):
        old, sample = fixture()
        old['native']['forward_valid'] = False
        row = map_prepare(old, sample)
        self.assertIsNone(row['forward_s'])
        self.assertIsNone(row['front_identity'])

    def test_completed_prefix_is_copied_at_prepare(self):
        _, sample = fixture()
        front = sample['completed_front']
        frozen = prepare_sample(sample['native'], dict(bus_id=2, time_s=100.,
            holding_counter_at_launch=300.), [2,22,3], 360., 1000., front)
        front['timing']['realized_holding'] = 999.
        self.assertEqual(frozen['completed_front']['timing']['realized_holding'], 40.)

    def test_wrong_occurrence_rejected(self):
        old, sample = fixture()
        old['clock']['legacy_prepare_event_seq'] += 1
        with self.assertRaises(AssertionError):
            map_prepare(old, sample)

    def test_wrong_front_identity_rejected(self):
        old, sample = fixture()
        sample['completed_front']['launch']['bus_id'] = 9
        with self.assertRaises(AssertionError):
            map_prepare(old, sample)

    def test_future_front_rejected(self):
        old, sample = fixture()
        sample['completed_front']['timing']['time_s'] = 1010.
        with self.assertRaises(AssertionError):
            map_prepare(old, sample)

    def test_same_time_front_requires_prior_old_order(self):
        old, sample = fixture()
        old['front'].update(departure_s=1000., departure_order=1199)
        old['native']['forward_gap_s'] = 0.
        sample['completed_front']['timing'].update(time_s=1000., other_elapsed=960.)
        self.assertEqual(sum(map_prepare(old, sample)['parts'].values()), 0.)
        old['front']['departure_order'] = 1200
        with self.assertRaises(AssertionError):
            map_prepare(old, sample)

    def test_broken_front_component_closure_rejected(self):
        old, sample = fixture()
        sample['completed_front']['timing']['other_elapsed'] += 5.
        with self.assertRaises(AssertionError):
            map_prepare(old, sample)

    def test_late_sample_and_reset_rejected(self):
        for read, counter in ((1001., 360.), (1000., 299.)):
            with self.assertRaises(AssertionError):
                prepare_sample(dict(kind='prepare', time_s=1000., event_seq=20,
                    bus_id=2, trip_id=22), dict(bus_id=2, time_s=100.,
                    holding_counter_at_launch=300.), [2,22,3], counter, read)

    def test_query_holding_remains_distinct(self):
        old, sample = fixture()
        row = map_prepare(old, sample)
        query = dict(identity=old['current'], prepare_order=1200, query_order=1210,
                     legacy_query_event_seq=21, query_s=1010., query_occurrence=0)
        legacy = dict(old['current'], prepare_event_seq=20, event_seq=21, time_s=1010.,
                      state_captured_s=1000., executed_holding_before_s=65.)
        view = dict(cycle_key=[2,22,3], native_event_seq=21, query_s=1010., executed_holding_s=65.)
        joined = join_query(row, query, legacy, view)
        self.assertEqual((joined['prepare_holding_s'], joined['query_holding_s']), (60.,65.))
        query.update(query_occurrence=1, query_order=1211, legacy_query_event_seq=22)
        legacy['event_seq'] = view['native_event_seq'] = 22
        self.assertEqual(join_query(row, query, legacy, view)['query_occurrence'], 1)
        view['cycle_key'] = [2,23,4]
        with self.assertRaises(AssertionError):
            join_query(row, query, legacy, view)

    def test_budget_and_five_original_payloads_frozen(self):
        spec = registration('test')
        self.assertEqual(set(OLD_PAYLOADS), {'legacy','v224','v230','v235','v241'})
        self.assertEqual(spec['reserved_preflight']['max_started_paths'], 2)
        self.assertTrue(all(v == 0 for v in spec['current_budget'].values()))
        for key in ('selector_defined', 'runtime_connected', 'collector_implemented',
                    'collection_authorized', 'replay_authorized', 'main_policy_changed'):
            self.assertFalse(spec[key])
        self.assertFalse(spec['reserved_preflight']['launch_authorized'])


if __name__ == '__main__':
    unittest.main()
