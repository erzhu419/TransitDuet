import copy
from types import SimpleNamespace
import unittest

import numpy as np

from analyze_freqduet_v222_holding_response import rule_parts
from analyze_freqduet_v226_coordination import index, response, response_terms, supply, supply_compare
from audit_freqduet_v213_coordination import cv_parts, moments
from freqduet_v223_common_reference import command


def decision(forward, backward, actual=None, mask=0.):
    ctx = dict(evidence_valid=True, follower_evidence_valid=True,
        forward_headway_s=forward, follower_departure_gap_s=backward, target_headway_s=180.)
    parts = rule_parts(ctx)
    native = (0., 5., 10., 15., 20., 30., 45.)[parts['bin']]+mask
    actual = native if actual is None else actual
    return dict(rule=parts, native_s=native, actual_s=actual, override_s=actual-native, mask_s=mask)


def chain(shift=0., held=0., other=0., idle=0.):
    returned = 100.+shift+held+other
    departure = returned+20.+idle
    data = dict(trace=dict(actual_launches=[
        dict(trip=1, bus=3, actual_s=10.+shift, public_baseline_s=10., eligible_s=10.),
        dict(trip=2, bus=3, actual_s=departure, public_baseline_s=120., eligible_s=120.)],
        vehicle_returns=[dict(trip_id=1, bus_id=3, launch_s=10.+shift, return_s=returned,
                             holding_s=held, other_elapsed_s=90.+other)]),
        events=[dict(kind='launch', trip_id=1, bus_id=3), dict(kind='return', trip_id=1, bus_id=3),
                dict(kind='launch', trip_id=2, bus_id=3)],
        launch_plans={'1': 10., '2': 120.}, results=dict(
            public_launch_delay_total_s=max(0., shift)+max(0., departure-120.),
            readiness_delay_total_s=shift+departure-120.))
    return data


def repeated():
    ctx = dict(evidence_valid=True, follower_evidence_valid=True, forward_headway_s=100.,
               follower_departure_gap_s=200., target_headway_s=180.)
    who = dict(trip_id=7, bus_id=3, station_id=2, direction=True)
    chosen = command(SimpleNamespace(**ctx), range(7), (0., 0., 0.))
    actions, queries, prepares, uniforms = [], [], [], []
    for i in range(2):
        actions.append(dict(who, context=ctx, time_s=500.+i, action=2))
        prepares.append(dict(current=who, context=ctx, clock=dict(handoff_order=2*i),
            phases={r: dict(status='known_zero', phase_s=0.) for r in ('front', 'current', 'follower')},
            front=dict(status='supported', trip_id=5, bus_id=2, direction=True),
            follower=dict(status='supported', trip_id=9, bus_id=4, direction=True)))
        queries.append(dict(identity=who, prepare_order=2*i, query_occurrence=i, query_s=500.+i,
                            feasible_bins=list(range(7)), shadow=chosen, native_bin=2))
        uniforms.append([[7, 2, 1, i], dict(time_s=500.+i, action_s=10.)])
    return dict(arm='upper', ledger=dict(executed_holding_s=20.),
        trace=dict(lower_actions=actions, action_uniforms=uniforms, vehicle_returns=[dict(holding_s=20.)])), \
        dict(prepares=prepares, queries=queries)


class CoordinationTest(unittest.TestCase):
    def test_occurrences_are_retained_and_reconstruct_holding(self):
        data, sidecar = repeated()
        rows = index(data, sidecar)
        self.assertEqual(list(rows), [(7, 2, True, 0), (7, 2, True, 1)])
        sidecar['queries'][1]['query_occurrence'] = 0
        with self.assertRaisesRegex(AssertionError, 'occurrence join failed'):
            index(data, sidecar)

    def test_unmatched_holding_is_separate_not_a_native_state_effect(self):
        data, sidecar = repeated()
        current = index(data, sidecar)
        base_data = copy.deepcopy(data)
        base_data['ledger']['executed_holding_s'] = 10.
        baseline = {next(iter(current)): next(iter(current.values()))}
        result, _ = response(data, base_data, current, baseline)
        self.assertEqual(result['current_only'], 1)
        self.assertEqual(result['current_only_holding_s'], 10.)
        self.assertEqual(result['totals']['native_s'], 0.)

    def test_direct_override_and_state_response_close(self):
        reference = decision(100., 201.)
        current = decision(100., 200., actual=5.)
        parts = response_terms(current, reference)
        self.assertEqual(parts['native_s'], -5.)
        self.assertEqual(parts['override_s'], -5.)
        self.assertEqual(parts['net_s'], -10.)
        self.assertAlmostEqual(parts['follower_s'], -.125)

    def test_feasibility_is_not_mislabeled_as_quantization(self):
        reference = decision(100., 200.)
        current = decision(100., 200., mask=-10.)
        parts = response_terms(current, reference)
        self.assertEqual(parts['feasibility_s'], -10.)
        self.assertEqual(parts['quantization_s'], 0.)
        self.assertEqual(parts['native_s'], -10.)

    def test_return_to_next_launch_decomposition(self):
        reference, current = supply(chain()), supply(chain(shift=1., held=5., other=2., idle=-3.))
        result, rows = supply_compare(current, reference)
        row = next(r for r in rows if r['trip_id'] == 2)
        self.assertEqual(row['actual_launch_delta_s'], 5.)
        self.assertEqual(row['parent_holding_delta_s'], 5.)
        self.assertEqual(row['terminal_idle_delta_s'], -3.)
        self.assertEqual(result['counts']['same_inbound_trip'], 1)
        self.assertEqual(result['counts']['initial_stock_both'], 1)

    def test_changed_inbound_source_is_unmatched_not_zero(self):
        reference = supply(chain())
        current = copy.deepcopy(reference)
        current['parents'][2] = None
        result, rows = supply_compare(current, reference)
        self.assertEqual(result['counts']['unmatched_supply'], 1)
        row = next(r for r in rows if r['trip_id'] == 2)
        self.assertNotIn('parent_holding_delta_s', row)

    def test_ledger_inconsistency_or_nonphysical_return_fails(self):
        data = chain()
        data['results']['readiness_delay_total_s'] += 1.
        with self.assertRaises(AssertionError):
            supply(data)
        data = chain()
        data['trace']['vehicle_returns'][0]['return_s'] = 200.
        with self.assertRaises(AssertionError):
            supply(data)

    def test_cv_parts_reconstruct_variance_and_normalization(self):
        def events(offset):
            rows = []
            for i, launch in enumerate((0., 100., 220., 300.)):
                hold = float(i*offset)
                rows.append(dict(trip_id=i, station_id=2, direction=True,
                    time_s=launch+30.+hold, release=launch, realized_holding=hold, other_elapsed=30.))
            return rows
        before, after = moments(events(0.)), moments(events(5.))
        parts = cv_parts(before, after)
        self.assertAlmostEqual(sum(parts.values()), after['cv2']-before['cv2'])
        self.assertNotEqual(parts['normalization'], 0.)
        np.testing.assert_allclose(sum(after['covariance'].flat), after['variance'])


if __name__ == '__main__':
    unittest.main()
