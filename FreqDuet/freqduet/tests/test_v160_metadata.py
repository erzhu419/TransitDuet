from collections import namedtuple
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import complete_freqduet_v160_metadata as metadata


class MetadataTests(unittest.TestCase):
    def test_residence_strict_prefix_and_equal_elapsed_survivors(self):
        Duration = namedtuple('Duration', 'groups times')
        duration = Duration({(True, 2): [(5., 8.), (9., 2.), (10., 100.)]}, {(True, 2): [5., 9., 10.]})
        result = metadata.residence_support(duration, (True, 2), 10., 2.)
        self.assertEqual((result['selected_n'], result['survivor_n']), (2, 2))
        self.assertEqual((result['latest_survivor_s'], result['mean_remaining_s']), (9., 3.))

    def test_residence_no_window_samples_uses_frozen_historical_fallback(self):
        Duration = namedtuple('Duration', 'groups times')
        duration = Duration({(True, 2): [(5., 8.)]}, {(True, 2): [5.]})
        result = metadata.residence_support(duration, (True, 2), 4000., 2.)
        self.assertTrue(result['historical_fallback'])
        self.assertEqual(result['mean_remaining_s'], 6.)

    def test_shadow_arrival_uses_saved_running_entry_and_preserves_partial_clock(self):
        call = dict(direction=True, station=2, predicted_arrival_time_s=108.)
        running = {(True, 2): dict(predicted_entry_time_s=100., fixed_prior=dict(mean=20.))}
        self.assertEqual(metadata.shadow_clock(call, running)[0], 120.)
        self.assertEqual(metadata.shadow_clock(call, {})[0], 108.)

    def test_shadow_phase_support_reconciles_saved_service_delta(self):
        class Law:
            cutoff = 110.

            def estimate(self, direction, station, now, arrival, elapsed):
                return dict(mean=2. if arrival < self.cutoff else 1., latest_current_s=9.)

        record = dict(query_time_s=10., support=dict(
            running=[dict(direction=True, station=2, predicted_entry_time_s=100., fixed_prior=dict(mean=20.))],
            service=[dict(direction=True, station=2, predicted_arrival_time_s=108., elapsed_service_s=0.,
                support=dict(mean=2., latest_current_s=9.))],
            completed_components=[dict(station=2, components=dict(service=dict(mean=3.)))]),
            components=dict(v151=dict(prediction_s=dict(service=4.)), eight_day=dict(prediction_s=dict(service=3.))))
        calls, delta = metadata.shadow_services(record, Law())
        self.assertEqual(delta, -1.)
        self.assertTrue(calls[0]['phase_changed'])
        self.assertEqual(calls[0]['historical_support']['mean'], 1.)

    def test_validity_and_clocks_do_not_backfill_late_load(self):
        row = dict(policy_time_s=11., query_load=40, evidence=dict(ready_time_s=10.,
            follower_source='same_time_avl_route_speed_eta', follower_gap_prediction_s=20.),
            context=dict(follower_evidence_source='same_time_avl_route_speed_eta',
                follower_departure_gap_s=20., follower_evidence_valid=True))
        record = dict(query_time_s=10., query_ready_order=2, label=dict(available_time_s=30.))
        value = metadata.follower_clock(row, record)
        self.assertTrue(value['follower_evidence_valid'])
        self.assertEqual((value['ready_snapshot_time_s'], value['policy_snapshot_time_s']), (10., 11.))
        self.assertIsNone(value['query_load'])
        self.assertNotIn('policy_load', value)
        row['context']['follower_evidence_source'] = 'changed_after_ready'
        with self.assertRaisesRegex(AssertionError, 'ready-cached'):
            metadata.follower_clock(row, record)

    def test_sampler_parser_ignores_day_keys_and_actor_ids(self):
        value = dict(actors=[1], sampler_seeds={'2851001': 1600000000},
            registration=dict(samplers=dict(calibration={'100': 99})))
        self.assertEqual(metadata.sampler_ids(value), {99, 1600000000})

    def test_generator_range_end_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prior.py'
            path.write_text('DAYS = range(100, 103)\n')
            self.assertEqual(metadata.literal_conflicts(path, {102, 103}), {102})

    def test_reserved_cohorts_and_samplers_are_disjoint(self):
        self.assertEqual((len(metadata.CALIBRATION), len(metadata.CONFIRMATION)), (32, 32))
        self.assertFalse(set(metadata.CALIBRATION) & set(metadata.CONFIRMATION))
        self.assertEqual(len(set(metadata.SAMPLERS.values())), 64)


if __name__ == '__main__':
    unittest.main()
