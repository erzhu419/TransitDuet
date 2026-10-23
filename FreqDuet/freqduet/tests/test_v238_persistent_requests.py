import copy
import unittest

from preregister_freqduet_v232_persistent_coordination import CycleState
from preregister_freqduet_v238_persistent_requests import ACTIONS, registration, shadow_request


def snapshot(holding=45, difference=160, native=4, bins=range(7)):
    meta = dict(native_dt_s=1, native_inventory_order=[2], native_timetable_order=[11])
    row = dict(snapshot_read_s=10, snapshot_event_seq=3, cycle_ref=[2, 1, 1],
        executed_holding_before_s=holding, message_refs={'false': 'initial', 'true': 'initial'},
        pending_plan_version_refs={'11': 'initial'}, ready=[],
        active=[dict(bus_id=2, trip_id=1, direction=False, remaining_distance_m=200,
                     position_sample_s=10)],
        plans=[dict(trip_id=11, direction=True, planned_s=20, public_s=20, target_s=10)],
        shadow_frontier=dict(status='supported', native_bin=native, candidates=[
            dict(bin=b, holding_s=ACTIONS[b],
                 local_gap_cost_delta_s2=2*(ACTIONS[b]-ACTIONS[native])*
                     (ACTIONS[b]+ACTIONS[native]-difference)) for b in bins]))
    state = CycleState()
    state.launch(2, 1, False, 1, 0)
    return state, row, meta


def request(holding=45, **kwargs):
    state, row, meta = snapshot(holding, **kwargs)
    return shadow_request(state.query(row, meta), row)


class PersistentRequestsTest(unittest.TestCase):
    def test_no_holding_is_exact_native_and_more_h_never_raises_request(self):
        requests = [request(h) for h in (0, 5, 45, 90, 180, 10000)]
        self.assertEqual(requests[0]['shadow_s'], 20)
        self.assertEqual(requests[2]['shadow_s'], 10)
        self.assertEqual(requests[-1]['shadow_s'], 5)
        self.assertEqual([r['shadow_s'] for r in requests],
                         sorted((r['shadow_s'] for r in requests), reverse=True))

    def test_positive_compensation_is_not_a_local_native_nonharm_gate(self):
        r = request()
        self.assertGreater(r['local_gain_vs_zero_s2'], 0)
        self.assertGreater(r['local_gap_delta_vs_native_s2'], 0)
        self.assertEqual(r['shadow_s'], 10)

    def test_projection_uses_native_mask_and_ties_keep_more_compensation(self):
        # 20/(1+27/45)=12.5,so 10/15 tie retains 15.
        self.assertEqual(request(27)['shadow_s'], 15)
        self.assertEqual(request(45, bins=(0, 3, 4, 6))['shadow_s'], 15)
        self.assertEqual(request(1000, bins=(0, 4))['shadow_s'], 20)
        self.assertEqual(request(45, bins=(2, 4))['reason'], 'no_native_feasible_zero_reference')

    def test_no_plan_or_current_ready_backing_keeps_native(self):
        for no_plan in (False, True):
            state, row, meta = snapshot()
            if no_plan:
                row['plans'], row['pending_plan_version_refs'] = [], {}
            else:
                meta['native_inventory_order'].append(8)
                row['ready'] = [dict(bus_id=8, direction=True, position_sample_s=10,
                                    return_observed=False, last_observed_return_s=None)]
            r = shadow_request(state.query(row, meta), row)
            self.assertEqual(r['shadow_s'], 20)
            self.assertEqual(r['reason'], 'no_own_unbacked_obligation')

    def test_future_and_due_both_expose_cycle_without_a_lead_time_budget(self):
        for plan_time, kind in ((100000, 'future'), (10, 'due')):
            state, row, meta = snapshot()
            row['plans'][0]['planned_s'] = plan_time
            r = shadow_request(state.query(row, meta), row)
            self.assertEqual((r['exposure'], r['shadow_s']), (kind, 10))

    def test_missing_or_terminal_evidence_and_zero_keep_native(self):
        for status in ('missing_forward', 'missing_follower', 'terminal_masked'):
            state, row, meta = snapshot()
            row['shadow_frontier']['status'] = status
            row['shadow_frontier']['candidates'][0]['local_gap_cost_delta_s2'] = None
            r = shadow_request(state.query(row, meta), row)
            self.assertEqual((r['reason'], r['shadow_s']), (status, 20))
        self.assertEqual(request(native=0)['shadow_s'], 0)
        self.assertEqual(request(difference=10)['reason'], 'native_lacks_positive_bilateral_gain_vs_zero')

    def test_plan_and_association_changes_preserve_actual_h_without_shadow_credit(self):
        state, row, meta = snapshot()
        before = shadow_request(state.query(row, meta), row)
        changed = copy.deepcopy(row)
        changed.update(snapshot_event_seq=5, message_refs={'false': 4, 'true': 'initial'},
                       pending_plan_version_refs={'12': 4})
        changed['plans'][0]['trip_id'] = 12
        after_view = state.query(changed, meta)
        after = shadow_request(after_view, changed)
        self.assertEqual((after['executed_holding_s'], after['shadow_s']), (45, before['shadow_s']))
        self.assertEqual(after['obligation_trip'], 12)
        self.assertEqual(after_view['delta_executed_since_last_sample_s'], 0)
        changed.update(snapshot_event_seq=6, executed_holding_before_s=50)
        self.assertEqual(state.query(changed, meta)['delta_executed_since_last_sample_s'], 5)
        state.returned_to_terminal(2, 1, 7, 20, 50)
        state.launch(2, 2, True, 8, 21)
        self.assertEqual(state.active[2]['executed_holding_s'], 0)

    def test_relief_and_new_position_rank_do_not_recover_a_fresh_budget(self):
        state, row, meta = snapshot()
        state.query(row, meta)
        relieved = copy.deepcopy(row)
        relieved.update(snapshot_event_seq=4, plans=[], pending_plan_version_refs={})
        self.assertEqual(shadow_request(state.query(relieved, meta), relieved)['shadow_s'], 20)
        restored = copy.deepcopy(row)
        restored['snapshot_event_seq'] = 5
        restored['active'][0]['remaining_distance_m'] = 100
        view = state.query(restored, meta)
        self.assertEqual((view['executed_holding_s'], shadow_request(view, restored)['shadow_s']), (45, 10))

    def test_purity_and_future_labels_are_not_inputs(self):
        state, row, meta = snapshot()
        view = state.query(row, meta)
        frozen = copy.deepcopy((view, row))
        before = shadow_request(view, row)
        row.update(actual_future_neighbour=99, return_s=-1, actual_next_launch=0,
                   future_supply_bus=100, posthoc_holding_benefit=False)
        self.assertEqual(before, shadow_request(view, row))
        for key in tuple(row.keys()-frozen[1].keys()):
            row.pop(key)
        self.assertEqual((view, row), frozen)
        self.assertEqual(state.active[2]['executed_holding_s'], 45)

    def test_wrong_query_or_holding_is_rejected_not_silently_reused(self):
        state, row, meta = snapshot()
        view = state.query(row, meta)
        for key, wrong in (('cycle_ref', [2, 99, 1]), ('snapshot_read_s', 11),
                           ('snapshot_event_seq', 4), ('executed_holding_before_s', 0)):
            bad = dict(row, **{key: wrong})
            with self.assertRaises(AssertionError):
                shadow_request(view, bad)

    def test_native_seven_bin_permissions_and_all_budgets_remain_frozen(self):
        p = registration('synthetic')
        self.assertEqual(p['frozen']['feasible_action_bins_s'], list(ACTIONS))
        self.assertEqual(p['frozen']['fleet'], 12)
        self.assertEqual(set(p['budget'].values()), {0})
        self.assertFalse(p['runtime_connected'] or p['archived_shadow_evaluated'] or
                         p['replay_authorized'] or p['training_authorized'] or p['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
