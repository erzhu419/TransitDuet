import copy
import unittest

import run_freqduet_v213_passive_coordination as v


def query(trip, time, command, *, station=2, evidence=True):
    return dict(trip_id=trip, station_id=station, direction=True, time_s=float(time), ordinal=trip,
        command_s=float(command), onboard=3, context=dict(evidence_valid=evidence,
            forward_headway_s=100., target_headway_s=360.))


def path(commands):
    queries = [query(i, 10*i, c) for i, c in enumerate(commands)]
    return dict(result=dict(lower=[dict(identity=[i, 2, True, i], time_s=10*i) for i in (0, 1)],
        ledger=dict(commanded_holding_s=sum(commands)), results=dict.fromkeys(v.prior.METRICS, 0.)),
        trace=dict(lower_queries=queries, departures=[dict(trip_id=i, station_id=2, direction=True,
            time_s=10*i+c) for i, c in enumerate(commands)], actual_launches=[], upper_callbacks=[]))


class PassiveCoordinationTests(unittest.TestCase):
    def test_mechanical_gain_distinguishes_correction_and_overshoot(self):
        self.assertGreater(v.mechanical_gain(100., 500., 360., 20.), 0.)
        self.assertLess(v.mechanical_gain(500., 100., 360., 20.), 0.)
        self.assertIsNone(v.mechanical_gain(100., 5., 360., 10.))

    def test_low_support_not_encoded_as_zero_gain(self):
        rows = [query(1, 10., 0., evidence=False), query(2, 20., 0.)]
        labels = v.completed_labels(rows, [dict(trip_id=1, station_id=2, direction=True, time_s=10.)])
        self.assertIsNone(labels[1, 2, True]['ready_stage_gain'])
        self.assertIsNone(labels[2, 2, True]['follower_ready_gap_s'])

    def test_completed_follower_label_clock_and_identity(self):
        rows = [query(1, 10., 5.), query(2, 110., 0.)]
        labels = v.completed_labels(rows, [dict(trip_id=1, station_id=2, direction=True, time_s=15.)])
        label = labels[1, 2, True]
        self.assertEqual(label['follower_ready_gap_s'], 100.)
        self.assertEqual(label['follower_label_available_s'], 110.)
        self.assertEqual(label['follower_trip_id'], 2)

    def test_duplicate_keys_are_not_silently_collapsed(self):
        with self.assertRaisesRegex(AssertionError, 'repeated lower'):
            v.unique_decisions([query(1, 10., 0.), query(1, 20., 0.)])

    def test_plan_updates_record_before_after_without_mutation(self):
        before = {1: dict(planned_s=100.)}
        after = {1: dict(planned_s=110.)}
        self.assertEqual(v.plan_changes(before, after)[0]['before']['planned_s'], 100.)
        self.assertEqual(before, {1: dict(planned_s=100.)})
        with self.assertRaises(AssertionError):
            v.plan_changes(before, {})

    def test_timeline_partition_closes_without_calling_net_reduction_direct(self):
        b, c = path([5., 10., 0., 0.]), path([0., 10., 5., 0.])
        result = v.paired_timeline(b, c)
        self.assertEqual(result['seconds']['registered_command_delta_s'], -5.)
        self.assertEqual(result['seconds']['post_second_command_delta_s'], 5.)
        self.assertEqual(result['seconds']['all_command_delta_s'], 0.)
        self.assertEqual(result['counts']['post_second_increases'], 1)
        self.assertEqual(result['counts']['post_second_increase_locally_counterproductive'], 1)

    def test_missing_events_kept_as_unmatched_not_matched_zero(self):
        b, c = path([0., 0., 0.]), path([0., 0., 5.])
        c['trace']['lower_queries'][2]['trip_id'] = 8
        result = v.paired_timeline(b, c)
        self.assertEqual(result['counts']['unmatched_candidate'], 1)
        self.assertEqual(result['counts']['unmatched_base'], 1)
        self.assertEqual(result['seconds']['unmatched_command_delta_s'], 5.)

    def test_actual_launch_change_survives_equal_mean(self):
        rows = [dict(trip=i, bus=i, actual_s=100.*i, public_baseline_s=100.*i) for i in (1, 2)]
        changed = copy.deepcopy(rows)
        changed[0]['actual_s'] += 5.
        changed[1]['actual_s'] -= 5.
        result = v.compare_launches(rows, changed)
        self.assertEqual(result['changed_time'], 2)
        self.assertEqual(result['absolute_time_delta_s'], 10.)


if __name__ == '__main__':
    unittest.main()
