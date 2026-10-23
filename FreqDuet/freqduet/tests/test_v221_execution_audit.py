import copy
import unittest

from analyze_freqduet_v221_execution import holding_account, differences, METRICS


def example():
    reference = dict(ledger=dict(executed_holding_s=90.), trace=dict(vehicle_returns=[
        dict(trip_id=1, holding_s=50.), dict(trip_id=2, holding_s=40.)]))
    data = dict(ledger=dict(executed_holding_s=85.), tasks=[dict(trip_id=1)],
        responses=[dict(trip_id=1, station_id=3, direction=True, base_bin=5, selected_bin=4, executed_before_s=5.)],
        trace=dict(vehicle_returns=[dict(trip_id=1, holding_s=40.), dict(trip_id=2, holding_s=45.)],
            departures=[dict(trip_id=1, station_id=3, direction=True, realized_holding=25.)]))
    return data, reference


class ExecutionAuditTest(unittest.TestCase):
    def test_direct_decrement_is_not_the_net_system_change(self):
        data, reference = example()
        r = holding_account(data, reference)
        self.assertEqual(r['direct_decrement_vs_live_bilateral_s'], 10.)
        self.assertEqual(r['task_trip_holding_delta_s'], -10.)
        self.assertEqual(r['other_trip_holding_delta_s'], 5.)
        self.assertEqual(r['total_holding_delta_s'], -5.)
        self.assertEqual(r['accounting_residual_vs_direct_decrement_s'], 5.)

    def test_missing_service_or_departure_is_not_zero(self):
        data, reference = example()
        missing = copy.deepcopy(data)
        missing['trace']['vehicle_returns'].pop()
        with self.assertRaises(AssertionError):
            holding_account(missing, reference)
        data['trace']['departures'] = []
        with self.assertRaises(AssertionError):
            holding_account(data, reference)

    def test_issued_and_executed_action_must_match(self):
        data, reference = example()
        data['trace']['departures'][0]['realized_holding'] += 1.
        with self.assertRaises(AssertionError):
            holding_account(data, reference)

    def test_journey_minutes_are_converted_to_seconds(self):
        zero = dict.fromkeys(METRICS, 0.)
        a = dict(zero, journey_min=-.01)
        self.assertAlmostEqual(differences(a, zero)['journey_s'], -.6)


if __name__ == '__main__':
    unittest.main()
