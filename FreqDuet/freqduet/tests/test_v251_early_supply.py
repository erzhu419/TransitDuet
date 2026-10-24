import copy
import unittest

from preregister_freqduet_v229_supply_coordination import local_gap_cost_change
from preregister_freqduet_v232_persistent_coordination import CycleState, resource_profile
from preregister_freqduet_v251_early_supply import link_prefix, registration


def fixture():
    meta = dict(native_dt_s=1., native_inventory_order=[8, 2, 5])
    row = dict(snapshot_read_s=10., snapshot_event_seq=3, cycle_ref=[2, 1, 1],
        executed_holding_before_s=5., message_refs={'true': 'initial', 'false': 'initial'},
        pending_plan_version_refs={'10': 'initial', '11': 'initial', '12': 'initial'},
        ready=[dict(bus_id=8, direction=True, position_sample_s=10.,
                    return_observed=False, last_observed_return_s=None)],
        active=[dict(bus_id=b, trip_id=t, direction=False, remaining_distance_m=d,
                     position_sample_s=10.) for b, t, d in ((2, 1, 200.), (5, 2, 800.))],
        plans=[dict(trip_id=t, direction=True, planned_s=p, public_s=p, target_s=10.)
               for t, p in ((10, 10.), (11, 20.), (12, 30.))])
    cycle = CycleState()
    cycle.launch(2, 1, False, 1, 0.)
    return cycle.query(row, meta)


def later(view, seq=5, now=12., held=15., plan=11, kind='future', version='initial'):
    v = copy.deepcopy(view)
    v.update(native_event_seq=seq, query_s=now, executed_holding_s=held)
    v['own_association']['position_sample_s'] = now
    p = v['own_association']['obligation']
    if plan is None:
        v['own_association']['obligation'] = None
    else:
        p.update(trip_id=plan, kind=kind, last_write_version=version)
    v['target_profile']['uncovered_due'] = [plan] if kind == 'due' and plan is not None else []
    return v


class EarlySupplyTests(unittest.TestCase):
    def test_future_watch_starts_before_due_without_defining_shortage(self):
        a = link_prefix(None, fixture())
        self.assertTrue(a['own_future'])
        self.assertFalse(a['target_due_present'] or a['own_due'])
        b = link_prefix(a, later(fixture(), now=21., kind='due'))
        self.assertEqual(b['first_link'], a['first_link'])
        self.assertEqual(b['episode']['start'], a['episode']['start'])
        self.assertEqual(b['H_since_episode_start_s'], 10.)
        self.assertEqual(b['first_own_due']['query_s'], 21.)

    def test_replan_preserves_episode_and_physical_H(self):
        a = link_prefix(None, fixture())
        v = later(fixture(), version=4)
        v['own_association']['obligation']['planned_s'] = 25.
        b = link_prefix(a, v)
        self.assertEqual(b['transition'], 'revised')
        self.assertEqual(b['episode'], a['episode'])
        self.assertEqual(b['H_s'], 15.)

    def test_lost_association_is_not_a_historical_reservation(self):
        v = fixture()
        a = link_prefix(None, v)
        b = link_prefix(a, later(v, plan=None))
        self.assertEqual(b['transition'], 'lost')
        self.assertIsNone(b['episode'])
        self.assertIsNone(b['current']['obligation'])
        self.assertIsNone(b['H_since_episode_start_s'])
        self.assertEqual(b['first_link'], a['first_link'])
        regained = link_prefix(b, later(v, seq=7, now=14., held=20.))
        self.assertEqual(regained['episodes_started'], 2)
        self.assertEqual(regained['episode']['start']['H_s'], 20.)
        self.assertEqual(regained['H_since_cycle_first_link_s'], 15.)

    def test_reassociation_closes_episode_without_resetting_H(self):
        a = link_prefix(None, fixture())
        b = link_prefix(a, later(fixture(), plan=12))
        self.assertEqual(b['transition'], 'reassociated')
        self.assertEqual(b['closed_episode'], a['episode'])
        self.assertEqual(b['H_s'], 15.)
        self.assertEqual(b['H_since_episode_start_s'], 0.)

    def test_own_future_can_coexist_with_other_due(self):
        v = fixture()
        v['target_profile']['uncovered_due'] = [10]
        a = link_prefix(None, v)
        self.assertTrue(a['own_future'] and a['target_due_present'])
        self.assertFalse(a['own_due'])

    def test_query_clock_cycle_and_H_resets_rejected(self):
        a = link_prefix(None, fixture())
        for changed in (later(fixture(), held=0.), later(fixture(), seq=3),
                        dict(later(fixture()), cycle_key=[2, 9, 4])):
            with self.assertRaises(AssertionError):
                link_prefix(a, changed)

    def test_missing_association_and_zero_H_are_not_missing_state(self):
        v = later(fixture(), plan=None, held=0.)
        a = link_prefix(None, v)
        self.assertIsNone(a['first_link'])
        self.assertEqual(a['H_s'], 0.)
        self.assertEqual(a['episodes_started'], 0)

    def test_labels_cannot_change_prefix_and_inputs_stay_unchanged(self):
        v = fixture()
        before = copy.deepcopy(v)
        a = link_prefix(None, v)
        self.assertEqual(v, before)
        v.update(actual_return_s=999., actual_next_trip=100, future_cost=-123.)
        self.assertEqual(link_prefix(None, v), a)
        b = link_prefix(a, later(v))
        self.assertEqual(a, link_prefix(None, before))
        self.assertFalse(any('bin' in k or 'credit' in k or 'slack' in k for k in b))

    def test_H_does_not_reallocate_ready_stock_or_incoming_positions(self):
        # Resource allocation is the frozen V232 function;H is not an input it reads.
        row = dict(snapshot_read_s=10., ready=[],
            active=[dict(bus_id=2, trip_id=1, direction=False,
                         remaining_distance_m=200., position_sample_s=10.)],
            plans=[dict(trip_id=11, direction=True, planned_s=20.)],
            pending_plan_version_refs={'11': 'initial'}, executed_holding_before_s=0.)
        meta = dict(native_dt_s=1., native_inventory_order=[2])
        first = resource_profile(row, meta)
        self.assertEqual(first['directions']['true']['position_associations'][0]['obligation']['trip_id'], 11)
        row['executed_holding_before_s'] = 450.
        self.assertEqual(resource_profile(row, meta), first)

    def test_local_tradeoff_is_against_native_not_zero(self):
        f, b, h = 100., 140., 10.
        self.assertGreater(local_gap_cost_change(f, b, 0., h), 0.)
        self.assertGreater(local_gap_cost_change(f, b, 5., h), 0.)
        self.assertLess(local_gap_cost_change(f, b, 5., 0.), 0.)
        self.assertLess(local_gap_cost_change(f, b, 20., h), 0.)

    def test_seal_has_zero_budgets_and_no_runtime_authority(self):
        p = registration('test')
        self.assertEqual(set(p['budget'].values()), {0})
        self.assertEqual(p['v238_source'], 'b4b1d00203')
        self.assertFalse(any(p[k] for k in ('runtime_connected', 'selector_defined',
            'archive_analysis_authorized', 'collection_authorized', 'replay_authorized',
            'training_authorized', 'main_policy_changed', 'v238_changed')))


if __name__ == '__main__':
    unittest.main()
