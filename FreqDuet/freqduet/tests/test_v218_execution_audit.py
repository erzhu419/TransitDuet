import unittest

import analyze_freqduet_v218_execution as audit


class ExecutionAuditTest(unittest.TestCase):
    def test_cumulative_holding_is_not_counted_twice(self):
        rows = [dict(trip_id=2, station_id=i, direction=True,
                     time_s=i*10, realized_holding=h) for i, h in ((1, 5), (2, 15), (3, 15))]
        self.assertEqual(list(audit.holding_by_stop(rows).values()), [5, 10, 0])

    def test_plan_only_is_not_actual_execution(self):
        self.assertEqual(audit.change_pattern(dict(next_plan_s=5, next_launch_s=0)), 'plan_only')

    def test_changed_assignment_is_not_a_matched_chain(self):
        a = dict(bus=1, inbound=2, outbound=3, direction=True)
        self.assertIsNone(audit.chain_delta(a, dict(a, outbound=5)))
        self.assertIsNone(audit.chain_delta(a, None))

    def test_chain_clock_conservation(self):
        a = dict(bus=1, inbound=2, outbound=3, direction=True, inbound_launch_s=0,
                 inbound_holding_s=10, other_elapsed_s=90, return_s=100,
                 next_launch_s=110, return_to_launch_s=10, next_plan_s=105)
        b = dict(a, inbound_holding_s=5, return_s=95, next_launch_s=108, return_to_launch_s=13)
        d = audit.chain_delta(a, b)
        self.assertEqual(d['next_launch_s'], 2)
        self.assertEqual(d['return_s'], 5)
        with self.assertRaises(AssertionError):
            audit.chain_delta(a, dict(b, return_s=96))

    def test_first_subsequent_launch_not_later_matching_assignment(self):
        data = dict(trace=dict(actual_launches=[], vehicle_returns=[], departures=[]),
                    ledger=dict(executed_holding_s=0), events=[
            dict(kind='return', bus_id=1, trip_id=2, event_seq=0),
            dict(kind='launch', bus_id=1, trip_id=3, event_seq=1),
            dict(kind='return', bus_id=1, trip_id=3, event_seq=2),
            dict(kind='launch', bus_id=1, trip_id=5, event_seq=3)])
        self.assertEqual(audit.index(data)['next_trip'][1, 2], 3)


if __name__ == '__main__':
    unittest.main()
