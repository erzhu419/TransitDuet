from collections import Counter
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from freqduet_v230_resource_capture import ResourceCapture, frontier, ready_coverage


class ResourceExtensionTest(unittest.TestCase):
    def query(self):
        return dict(context=dict(evidence_valid=True, follower_evidence_valid=True,
            forward_headway_s=200., follower_departure_gap_s=240.), base_bin=1,
            onboard=3, terminal_masked=False, feasible_bins=[0, 1, 3])

    def test_shadow_contains_only_feasible_actions_and_native(self):
        row = frontier(self.query())
        self.assertEqual([r['bin'] for r in row['candidates']], [0, 1, 3])
        self.assertEqual(row['native_bin'], 1)
        self.assertEqual(row['candidates'][1]['delta_holding_s'], 0)
        self.assertEqual(row['candidates'][0]['immediate_onboard_delta_person_s'], -15)
        self.assertNotIn('selected_bin', row)

    def test_unsupported_quadratic_is_null_not_zero(self):
        q = self.query()
        q['context']['follower_evidence_valid'] = False
        q['context']['follower_departure_gap_s'] = None
        self.assertEqual(frontier(q)['status'], 'missing_follower')
        self.assertTrue(all(c['local_gap_cost_delta_s2'] is None for c in frontier(q)['candidates']))

    def test_native_fifo_ties_use_inventory_order_not_bus_id(self):
        row = dict(snapshot_read_s=10., active=[], ready=[dict(bus_id=b, direction=True,
            return_observed=False, last_observed_return_s=None) for b in (8, 2)],
            plans=[dict(trip_id=t, direction=True, planned_s=10.) for t in (30, 29, 31)])
        metadata = dict(native_dt_s=1., native_inventory_order=[8, 2])
        result = ready_coverage(row, metadata)
        self.assertEqual(result['covered_by_ready_bus'], {'30': 8, '29': 2})
        self.assertEqual(result['uncovered']['true'], [31])

    def test_current_tick_return_covers_next_pass_not_already_completed_pass(self):
        row = dict(snapshot_read_s=10., active=[], ready=[dict(bus_id=0, direction=False,
            return_observed=True, last_observed_return_s=10.)],
            plans=[dict(trip_id=1, direction=False, planned_s=10.)])
        result = ready_coverage(row, dict(native_dt_s=1., native_inventory_order=[0]))
        self.assertEqual(result['next_dispatch_s'], 11.)
        self.assertEqual(result['covered_by_ready_bus'], {'1': 0})

    def test_future_plan_does_not_spend_ready_stock_before_a_due_plan(self):
        row = dict(snapshot_read_s=10., active=[], ready=[dict(bus_id=0, direction=False,
            return_observed=False, last_observed_return_s=None)], plans=[
            dict(trip_id=1, direction=False, planned_s=20.),
            dict(trip_id=2, direction=False, planned_s=10.)])
        self.assertEqual(ready_coverage(row, dict(native_dt_s=1., native_inventory_order=[0]))[
            'covered_by_ready_bus'], {'2': 0})

    def test_callback_records_actual_write_and_explicit_cache_reuse(self):
        old = dict(trip_id=0, planned_s=10.)
        new = dict(trip_id=0, planned_s=9.)
        audit = SimpleNamespace(callbacks=[dict(trip_id=0, time_s=0., direction=True,
            changes=[dict(trip_id=0, before=old, after=new)])])
        c = ResourceCapture(audit)
        c.plans, c.plan_versions = {0: old}, {0: 'initial'}
        c.event(dict(kind='fresh_upper', trip_id=0, direction=True, time_s=0., event_seq=0))
        c.event(dict(kind='plan_update', trip_id=0, direction=True, time_s=0., event_seq=1))
        self.assertEqual(c.plans[0], new)
        self.assertEqual(c.versions[1]['fresh_upper_event_seq'], 0)
        audit.callbacks.append(dict(trip_id=1, time_s=10., direction=True, changes=[]))
        c.event(dict(kind='plan_update', trip_id=1, direction=True, time_s=10., event_seq=5))
        self.assertIsNone(c.versions[5]['fresh_upper_event_seq'])
        self.assertEqual(c.latest['true'], 5)
        self.assertEqual(c.plan_versions[0], 1)

    def test_observer_has_own_counters_and_rejects_rng_draws(self):
        audit = SimpleNamespace(handoff=SimpleNamespace(calls=Counter(), checked=Counter()))
        c = ResourceCapture(audit)
        with c.read('copy'):
            copy.deepcopy([1, 2])
        self.assertEqual(audit.handoff.checked, Counter())
        self.assertEqual(c.checks, Counter(copy=1))
        state = np.random.get_state()
        try:
            with self.assertRaises(AssertionError):
                with c.read('bad'):
                    np.random.random()
        finally:
            np.random.set_state(state)


if __name__ == '__main__':
    unittest.main()
