import copy
import unittest

from analyze_freqduet_v222_holding_response import (
    compare, decision_terms, fixed_neighbour_delta, index_decisions, loss_delta, rule_parts,
)


def context(forward=300., follower=420., valid=True):
    return dict(forward_headway_s=forward, follower_departure_gap_s=follower,
        target_headway_s=360., evidence_valid=valid, follower_evidence_valid=valid)


def decision(ctx, native, actual, order):
    return dict(context=ctx, rule=rule_parts(ctx), native_s=native, actual_s=actual,
        override_s=actual-native, order=order, time_s=100.+order)


def repeated_data():
    events, actions, uniforms, responses = [], [], [], []
    for occurrence in range(2):
        stamp = dict(trip_id=1, station_id=2, direction=True, bus_id=3)
        events.extend([dict(stamp, kind='prepare', event_seq=2*occurrence, time_s=9.+occurrence),
                       dict(stamp, kind='lower_query', event_seq=2*occurrence+1, time_s=10.+occurrence)])
        actions.append(dict(trip_id=1, station_id=2, direction=True, time_s=10.+occurrence,
                            action=0, context=context(valid=False)))
        uniforms.append(([1, 2, 1, occurrence], dict(time_s=10.+occurrence, action_s=0.)))
        responses.append(dict(trip_id=1, station_id=2, direction=True, time_s=10.+occurrence,
                              base_bin=0, selected_bin=0, event_seq=2*occurrence+1))
    return dict(events=events, tasks=[dict(trip_id=1)], responses=responses,
                trace=dict(lower_actions=actions, action_uniforms=uniforms))


class HoldingResponseTest(unittest.TestCase):
    def test_repeated_logical_calls_are_not_overwritten(self):
        rows = index_decisions(repeated_data())
        self.assertEqual(list(rows), [(1, 2, True, 0), (1, 2, True, 1)])
        self.assertEqual(rows[(1, 2, True, 0)]['prepare_s'], 9.)

    def test_missing_native_request_is_not_imputed(self):
        data = repeated_data()
        data['responses'].pop()
        with self.assertRaises(AssertionError):
            index_decisions(data)

    def test_frozen_rule_mismatch_is_detected(self):
        data = repeated_data()
        data['responses'][0]['base_bin'] = 1
        with self.assertRaises(AssertionError):
            index_decisions(data)

    def test_proxy_improvement_can_harm_actual_forward_gap(self):
        self.assertLess(loss_delta(400., 360., 15., 10.), 0.)
        self.assertGreater(loss_delta(300., 360., 15., 10.), 0.)

    def test_state_response_and_direct_decrement_cancel(self):
        old = decision(context(), 15., 15., 0)
        now = decision(context(280., 440.), 20., 15., 0)
        r = decision_terms(now, old)
        self.assertEqual(r['native_s'], 5.)
        self.assertEqual(r['override_s'], -5.)
        self.assertEqual(r['net_s'], 0.)
        self.assertEqual(r['forward_s'], 2.5)
        self.assertEqual(r['follower_s'], 2.5)

    def test_quantization_and_clipping_are_separate(self):
        old = decision(context(), 15., 15., 0)
        now = decision(context(300., 436.), 15., 15., 0)
        self.assertEqual(decision_terms(now, old)['quantization_s'], -2.)
        old = decision(context(0., 400.), 45., 45., 0)
        now = decision(context(0., 500.), 45., 45., 0)
        self.assertEqual(decision_terms(now, old)['clipping_s'], -12.5)
        self.assertEqual(rule_parts(context(300., 440.))['bin'], 3)

    def test_later_native_request_is_not_a_direct_override(self):
        b = {(1, 2, True, 0): decision(context(320., 400.), 10., 10., 0),
             (1, 3, True, 0): decision(context(300., 380.), 10., 10., 1)}
        c = {(1, 2, True, 0): decision(context(320., 400.), 10., 5., 0),
             (1, 3, True, 0): decision(context(280., 400.), 15., 15., 1)}
        data = dict(tasks=[dict(trip_id=1)], trace=dict(vehicle_returns=[dict(trip_id=1, holding_s=20.)]))
        r = compare(data, copy.deepcopy(data), c, b)
        self.assertEqual(r['phases']['first']['override_s'], -5.)
        self.assertEqual(r['phases']['later_native']['native_s'], 5.)
        self.assertEqual(r['totals']['net_s'], 0.)
        self.assertEqual(r['untraced_holding_delta_s'], 0.)

    def test_fixed_neighbours_do_not_allow_crossing(self):
        self.assertIsNone(fixed_neighbour_delta(300., 2., 360., 5.))
        self.assertGreater(fixed_neighbour_delta(300., 420., 360., 5.), 0.)


if __name__ == '__main__':
    unittest.main()
