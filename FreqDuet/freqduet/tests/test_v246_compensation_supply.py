import copy
import unittest

from preregister_freqduet_v246_compensation_supply import (
    ACTIONS, CONSTRAINTS, CycleState, completed_forward_parts, evaluation_label,
    evidence_frontier, joint_gate, persistent_change, registration)


def inputs():
    p = dict(cycle_key=[2,9,1], front_identity=[1,7,False], prepare_s=150.,prepare_seq=10,
        forward_s=30.,backward_s=270.,
        own=dict(launch_s=100.,launch_seq=1,holding_s=10.),
        front=dict(launch_s=80.,departure_s=120.,departure_seq=8,holding_s=5.))
    q = dict(query_s=151.,query_seq=11,native_bin=5,onboard=10,terminal_masked=False)
    v = dict(cycle_key=[2,9,1],query_s=151.,native_event_seq=11,executed_holding_s=12.,
        own_association=dict(obligation=dict(trip_id=20,kind='due')),
        target_profile=dict(uncovered_due=[20]))
    return p,q,v


def intervals(passenger=(-2.,-1.),cv=(-.2,0.),delay=(-3.,-1.)):
    r = {k:dict(ci95=[-.1,0.]) for k in CONSTRAINTS}
    r.update(passenger_person_s=dict(ci95=list(passenger)),cv_squared=dict(ci95=list(cv)),
             public_launch_delay_total_s=dict(ci95=list(delay)))
    return r


class CompensationSupplyTest(unittest.TestCase):
    def test_forward_coordinates_add_holding_once(self):
        p,_,_ = inputs()
        self.assertEqual(completed_forward_parts(p),dict(release_s=20.,executed_holding_gap_s=5.,
                                                        nonholding_elapsed_gap_s=5.))

    def test_query_holding_does_not_backfill_prepare(self):
        p,q,v = inputs()
        r = evidence_frontier(p,q,v,range(7))
        self.assertEqual(r['prepared_parts']['executed_holding_gap_s'],5.)
        self.assertEqual((r['prepare_holding_s'],r['query_holding_s']),(10.,12.))

    def test_all_seven_bins_remain_and_positive_local_regret_is_explicit(self):
        p,q,v = inputs()
        r = evidence_frontier(p,q,v,range(7))
        self.assertEqual([c['holding_s'] for c in r['candidates']],list(ACTIONS))
        self.assertGreater(r['candidates'][3]['local_loss_vs_native_s2'],0.)
        self.assertEqual(r['candidates'][3]['compensation_evidence'],'local_compensation_risk')
        self.assertIsNone(r['selected_bin'])
        self.assertFalse(r['executed'] or r['selector_defined'])

    def test_strict_local_nonharm_does_not_fabricate_reduction_support(self):
        p,q,v = inputs()
        rows = evidence_frontier(p,q,v,range(7))['candidates']
        self.assertTrue(all(c['local_loss_vs_native_s2']>0 for c in rows if c['bin']<q['native_bin']))

    def test_native_mask_is_preserved_without_new_permission(self):
        p,q,v = inputs()
        self.assertEqual([c['bin'] for c in evidence_frontier(p,q,v,[0,3,5])['candidates']],[0,3,5])

    def test_future_unbacked_is_not_current_shortage(self):
        p,q,v = inputs()
        v['own_association']['obligation']['kind']='future'
        v['target_profile']['uncovered_due']=[]
        r=evidence_frontier(p,q,v,range(7))
        self.assertTrue(all(c['supply_exposure']=='future_only_not_due_shortage' for c in r['candidates']))

    def test_missing_prepare_holding_and_rear_are_unknown_not_zero(self):
        p,q,v = inputs()
        p['own'].pop('holding_s');p['backward_s']=None
        r=evidence_frontier(p,q,v,range(7))
        self.assertIsNone(r['prepared_parts'])
        self.assertIsNone(r['prepare_holding_s'])
        self.assertTrue(all(c['local_loss_vs_native_s2'] is None for c in r['candidates']))
        self.assertEqual(persistent_change(None,p)['status'],'forward_components_unavailable')

    def test_future_front_departure_is_not_a_completed_prefix(self):
        p,_,_ = inputs();p['front']['departure_seq']=12
        with self.assertRaises(AssertionError):completed_forward_parts(p)

    def test_persistence_separates_new_disturbance_from_executed_holding(self):
        p,_,_ = inputs();c=copy.deepcopy(p)
        c.update(prepare_s=160.,prepare_seq=20,forward_s=35.)
        c['own']['holding_s']=15.;c['front'].update(departure_s=125.,departure_seq=18)
        r=persistent_change(p,c)
        self.assertEqual(r['parts'],dict(release_s=0.,executed_holding_gap_s=5.,nonholding_elapsed_gap_s=0.))
        c['front_identity']=[3,8,False]
        self.assertIsNone(persistent_change(p,c)['parts'])

    def test_only_observed_next_launch_resets_physical_cycle(self):
        state=CycleState();state.launch(2,9,False,1,100.)
        self.assertEqual(state.active[2]['executed_holding_s'],0.)
        state.returned_to_terminal(2,9,20,200.,30.)
        r=state.launch(2,10,True,21,201.)
        self.assertEqual(r['observed_previous_cycle']['executed_holding_s'],30.)
        self.assertEqual(state.active[2]['executed_holding_s'],0.)

    def test_real_regularity_harm_not_covariance_sign_defines_relative_benefit(self):
        e=dict(executed_holding_reduction=True)
        self.assertEqual(evaluation_label(intervals(cv=(.01,.02)),{},e),
                         'retained_holding_has_relative_regularity_benefit')
        self.assertEqual(evaluation_label(intervals(cv=(-.01,.02)),{},e),'unresolved_or_tradeoff')
        self.assertEqual(evaluation_label(intervals(cv=(.01,.02)),{},{}),'no_verified_holding_reduction')

    def test_unchanged_supply_cannot_pass_resource_claim(self):
        e=dict(executed_holding_reduction=True,earlier_same_bus_return=True,
               earlier_first_same_bus_launch=False,serves_query_recorded_due_obligation=True)
        self.assertEqual(evaluation_label(intervals(),intervals(),e),
                         'system_gain_without_verified_supply_chain_gain')
        e['earlier_first_same_bus_launch']=True
        self.assertEqual(evaluation_label(intervals(),intervals(delay=(-1.,0.)),e),
                         'system_gain_without_incremental_dispatch_benefit')
        self.assertEqual(evaluation_label(intervals(),intervals(),e),
                         'independently_supported_supply_redundancy_relative_to_action_pair')

    def test_missing_cost_is_not_zero_nonharm(self):
        r=intervals();r.pop('public_launch_delay_total_s')
        self.assertIsNone(joint_gate(r))

    def test_seal_authorizes_no_runtime_or_data_and_freezes_v238(self):
        p=registration('test')
        self.assertEqual(set(p['budget'].values()),{0})
        self.assertEqual(p['frozen']['v238_source'],'b4b1d00203')
        self.assertEqual(p['frozen']['holding_bins_s'],list(ACTIONS))
        self.assertFalse(p['selector_defined'] or p['runtime_connected'] or p['replay_authorized'])


if __name__ == '__main__':unittest.main()
