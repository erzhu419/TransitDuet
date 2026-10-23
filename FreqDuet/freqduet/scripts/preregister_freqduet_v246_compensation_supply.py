#!/usr/bin/env python3
"""Seal observable compensation/supply identification; no online selector."""

import argparse
import json
from pathlib import Path
import unittest

from preregister_freqduet_v229_supply_coordination import assigned_fields, local_gap_cost_change
from preregister_freqduet_v232_persistent_coordination import CycleState


ACTIONS = (0, 5, 10, 15, 20, 30, 45)
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
               'trips_unlaunched', 'trips_incomplete')


def forward_parts_available(prepared):
    return prepared['forward_s'] is not None and all(
        prepared[role].get(field) is not None for role, fields in (
            ('own', ('launch_s','launch_order','holding_s')),
            ('front', ('launch_s','departure_s','departure_order','holding_s'))) for field in fields)


def completed_forward_parts(prepared):
    """Both clocks are already observed at PREPARE; never synthesize rear timing."""
    own, front = prepared['own'], prepared['front']
    now = prepared['prepare_s']
    assert own['launch_s'] <= now and front['launch_s'] <= front['departure_s'] <= now
    assert own['launch_order'] < prepared['prepare_order']
    assert front['departure_order'] < prepared['prepare_order']
    assert own['holding_s'] >= 0 and front['holding_s'] >= 0
    release = own['launch_s']-front['launch_s']
    holding = own['holding_s']-front['holding_s']
    elapsed = (now-own['launch_s']-own['holding_s']
               -(front['departure_s']-front['launch_s']-front['holding_s']))
    assert abs(release+holding+elapsed-prepared['forward_s']) <= 1e-8
    return dict(release_s=release, executed_holding_gap_s=holding, nonholding_elapsed_gap_s=elapsed)


def persistent_change(previous, current):
    if not forward_parts_available(current) or (previous is not None and not forward_parts_available(previous)):
        return dict(status='forward_components_unavailable', parts=None)
    if previous is None:
        return dict(status='first_supported_prepare', parts=None)
    same = previous['cycle_key'] == current['cycle_key'] and previous['front_identity'] == current['front_identity']
    if not same:
        return dict(status='cycle_or_front_changed', parts=None)
    assert previous['prepare_seq'] < current['prepare_seq']
    assert previous['prepare_order'] < current['prepare_order']
    a, b = completed_forward_parts(previous), completed_forward_parts(current)
    delta = {k: b[k]-a[k] for k in a}
    assert abs(sum(delta.values())-(current['forward_s']-previous['forward_s'])) <= 1e-8
    return dict(status='same_physical_cycle_and_front', parts=delta)


def evidence_frontier(prepared, query, view, feasible_bins):
    """Two orthogonal evidence axes; every legal action is retained, none executed."""
    assert prepared['cycle_key'] == view['cycle_key']
    assert prepared['prepare_s'] <= query['query_s'] == view['query_s']
    assert prepared['prepare_seq'] < query['query_seq'] == view['native_event_seq']
    assert prepared['prepare_order'] < query['query_order']
    prepare_holding = prepared['own'].get('holding_s')
    if prepare_holding is not None:
        assert view['executed_holding_s'] >= prepare_holding
    assert query['native_bin'] in feasible_bins and len(set(feasible_bins)) == len(feasible_bins)
    assert all(b in range(len(ACTIONS)) for b in feasible_bins)
    parts = completed_forward_parts(prepared) if forward_parts_available(prepared) else None
    native = ACTIONS[query['native_bin']]
    association = view['own_association']
    obligation = association['obligation']
    if obligation is None:
        exposure = 'other_due' if view['target_profile']['uncovered_due'] else 'none'
    elif obligation['kind'] == 'due':
        assert obligation['trip_id'] in view['target_profile']['uncovered_due']
        exposure = 'own_due_exposure'
    else:
        assert obligation['kind'] == 'future'
        exposure = 'future_only_not_due_shortage'
    candidates = []
    for b in feasible_bins:
        action = ACTIONS[b]
        loss = None if (query['terminal_masked'] or prepared['forward_s'] is None or
                        prepared['backward_s'] is None) else local_gap_cost_change(
            prepared['forward_s'], prepared['backward_s'], action, native)
        risk = ('backward_unresolved' if loss is None else 'local_compensation_risk' if loss > 0
                else 'local_proxy_nonharm')
        candidates.append(dict(bin=b, holding_s=action, delta_s=action-native,
            immediate_onboard_delta_person_s=query['onboard']*(action-native),
            local_loss_vs_native_s2=loss, compensation_evidence=risk, supply_exposure=exposure,
            interpretation='Candidate evidence,not identified necessity or supply redundancy.'))
    return dict(cycle_key=list(view['cycle_key']), prepared_parts=parts,
        prepare_s=prepared['prepare_s'], query_s=query['query_s'],
        prepare_holding_s=prepare_holding, query_holding_s=view['executed_holding_s'],
        native_bin=query['native_bin'], candidates=candidates,
        selected_bin=None, executed=False, selector_defined=False)


def joint_gate(intervals):
    keys = ('passenger_person_s', *CONSTRAINTS)
    if any(k not in intervals or intervals[k].get('ci95') is None for k in keys):
        return None
    return intervals['passenger_person_s']['ci95'][1] < 0 and all(
        intervals[k]['ci95'][1] <= 0 for k in CONSTRAINTS)


def evaluation_label(native_intervals, blind_intervals, supply_execution):
    """Future-only evaluation label; this function is never a policy input."""
    if supply_execution.get('executed_holding_reduction') is not True:
        return 'no_verified_holding_reduction'
    cv = native_intervals.get('cv_squared', {}).get('ci95')
    if cv is not None and cv[0] > 0:
        return 'retained_holding_has_relative_regularity_benefit'
    if joint_gate(native_intervals) is not True:
        return 'unresolved_or_tradeoff'
    if joint_gate(blind_intervals) is not True:
        return 'general_moderation_or_unresolved_resource_increment'
    required = ('executed_holding_reduction', 'earlier_same_bus_return',
                'earlier_first_same_bus_launch', 'serves_query_recorded_due_obligation')
    if not all(supply_execution.get(k) is True for k in required):
        return 'system_gain_without_verified_supply_chain_gain'
    delay = blind_intervals['public_launch_delay_total_s']['ci95']
    if delay[1] >= 0:
        return 'system_gain_without_incremental_dispatch_benefit'
    return 'independently_supported_supply_redundancy_relative_to_action_pair'


def registration(source):
    return dict(stage='v246_compensation_supply_identification_preregistration', source=source,
        authorization='Design seal,synthetic reference checks and source-interface inspection only.',
        parent_sources=dict(accounting='39cd832b46', elapsed='8050fd604e'),
        supersedes_design_source='d5ebe1c77d',
        amendment='Separate original native event_seq from original V224 sidecar order. '
            'Preserve the earlier seal;no trajectory,criterion or action budget changed.',
        budget=dict(trajectory_scans=0, shadow_archive_queries=0, passive_paths=0,
            intervention_paths=0, new_futures=0, model_fits=0, training_updates=0),
        frozen=dict(fleet=12, holding_bins_s=list(ACTIONS), v238_source='b4b1d00203',
            base='V107 bilateral025', upper='Native B200 or exact frozen V220,original +/-60 '
                'bounds,EMA.35,900s cadence,anchors,rolling-zero-sum and direction-local caches.',
            other='Main,actors,critics,duals,all predictors,calibrations,windows,quantiles,'
                'native masks,service clearing and physical dispatch authority unchanged.'),
        hypothesis='Disturbance-compensation risk and current due-service exposure are '
            'different state axes. A persistent coordinator must identify a useful action '
            'difference on both axes,not relabel cumulative holding as removable slack.',
        causal_state=dict(prepare='Keep original F/B,target,own/front/follower identities and '
            'native PREPARE/event provenance. Record own actual launch and executed H AT '
            'PREPARE plus the already completed matched-front departure/launch/H;never '
            'substitute query H into prepare. Launch/departure/prepare/query ORDER comparisons '
            'stay within original V224 sidecar order;native prepare/query SEQ comparisons '
            'stay within original EventClock. Join the existing prepare occurrence;never '
            'compare numeric ordinals across domains. Rear ETA stays a native prediction,not a '
            'completed future identity or timing coordinate.',
            query='Current posted plans/last-write versions,ready FIFO inventory,all incoming '
                'positions,query load/mask/native bin and separate query H,joined by exact '
                'bus/trip/observed-launch cycle and native prepare occurrence.',
            unavailable='Missing prepare H or matched-front H means unavailable decomposition,'
                'not zero. Incomplete follower travel/service/holding remains unknown. '
                'No label backfill,extra native estimator call or hidden road/demand state.'),
        compensation='At prepare,F=delta actual launch+delta executed H+delta completed/current '
            'nonholding elapsed. These terms are coordinates,not causal necessity labels. '
            'For every feasible a retain R(a)-R(native),not just R(a)-R(0). Never subtract '
            'H again or separately charge release and nonholding terms as holding budgets. '
            'Across stations,compare components only for the same own physical cycle and '
            'matched front identity;identity changes stay explicit and do not erase H.',
        supply='Reuse V232 one-use ready allocation and position-only incoming association. '
            'Separate own already-due uncovered obligation,other uncovered due,unbacked '
            'future and no exposure. A future unbacked plan is not a current shortage. '
            'Own-due association is not a reservation,known next assignment,ETA or slack. '
            'Upper advances remain backed only by currently eligible ready vehicles.',
        persistence='Reuse the unchanged V232 physical cycle ledger. Only actual execution '
            'increases H;only an observed actual new launch opens H=0. Refresh current '
            'obligations at every native query,through plan revisions and relief. Never '
            'turn observed H,desired savings,posted lead or local compensation into '
            'spendable credit;never suppress a later native compensation request from '
            'an earlier reduction. No added terminal waiting or out-of-cadence replanning.',
        frontier='Retain ALL native-feasible seven-bin actions,including zero when legal,'
            'native and larger actions. Tag compensation risk and supply exposure separately. '
            'Positive local regret is a tradeoff requiring evidence,not a forbidden action '
            'or automatic redundancy. Missing rear evidence remains unresolved. H/45 '
            'attenuation,forward-only substitution,forced reductions and fitted weighted '
            'scores are not this design. No online action ranking is sealed here.',
        guard_feasibility='A strict R(a)<=R(native) cut guard generally admits no lower bin '
            'when quarter-gain native lies below the two-gap optimum. Test that algebra '
            'synthetically;do not call a no-op guard a repaired controller or raise gains '
            'to manufacture support. R(a)<R(0) alone also does not preserve native benefit.',
        future_intervention='Before any new path,SEPARATELY freeze ONE executable rule mapping '
            'this observable frontier/history to one feasible command,its exact passive '
            'schema and fixed started-path budget. Native fallback is mandatory while '
            'that mapping is absent. No certified action is inferred from V244 outcomes. '
            'The later rule must refresh causal state at every native query to clearing,'
            'not replay a fixed action tape or choose an ex-post successful anchor.',
        identification='Relative necessity requires independently paired full-horizon '
            'actual-CV2 harm CI lower>0 when the specified holding is removed. Supply '
            'redundancy requires passenger benefit and CV2/readiness/original-public '
            'total-delay/completion nonharm PLUS reduced executed H,earlier same-bus '
            'return and FIRST next actual launch serving a query-recorded due obligation.'
            'Require incremental dispatch benefit versus a supply-blind control before '
            'claiming resource coordination. All are evaluation labels,never runtime inputs.',
        controls='V107,upper-only,new lower-only,joint,and supply-blind lower/joint using '
            'the SAME later compensation/decision function with ONLY resource linkage '
            'removed. Keep frozen V238 as a historical comparator,not as a modified rule.'
            'Required strong external controls:exact-fixed+bilateral and exact-fixed+zero;'
            'B200 remains the learned-controller reference. Any memoryless ablation '
            'must remove actual history,not assert H=0 aliases a different candidate.',
        causal_contrasts='Joint vs V107,upper-only,lower-only and blind-joint;factorial '
            'interaction (joint-upper)-(lower-V107);resource interaction '
            '(joint-blind_joint)-(lower-blind_lower). Report realized holding doses;'
            'do not match/filter/reweight on post-action H to manufacture a resource effect.',
        future_statistics='New calibration/development demand days and independent confirmation '
            'days/futures must be separately reserved and verified unconsumed BEFORE use. '
            'V244 and its context are development only. Keep actual frozen V68/V69 '
            'coupling,current-window draws and branch-only events. Pair all arms within '
            'unit;average future repeats then crossed actor/day bootstrap(5000 draws,'
            'seed246),report each actor/day and two predetermined independent future '
            'halves. No query-level replication,posthoc dose matching,context replacement '
            'or relaxed zero-nonharm margins. This seal reserves/starts NO day or future.',
        acceptance='Joint passenger95%CI upper<0 and actual CV2,readiness,original-public '
            'total launch delay,incomplete/unlaunched CI uppers<=0 against EACH primary '
            'comparator;preserve missing/censored outcomes. Resource-specific claim ALSO '
            'requires public-delay CI upper<0 vs blind and actually executed same-chain '
            'supply changes. Geometry,covariance,synthetic tests or unchanged dispatches '
            'cannot pass that claim. Passing internal controls is not dominance over '
            'strong fixed;report its separate paired tradeoff.',
        stop='A single source/support feasibility audit comes next under separate approval. '
            'If completed-clock inputs or a nontrivial observable action distinction are '
            'absent,stop BEFORE replay;do not continue H-scale/threshold/guard variants. '
            'A tradeoff-only or blind-equivalent confirmation stops promotion. No automatic '
            'future-bank opening,extra samples or policy switch.',
        limitations='Forward decomposition does not reconstruct unfinished rear dynamics or '
            'identify causal mediation. Existing V245 covariance signs do not prove removal '
            'is safe. This is an identification/persistent-state protocol,NOT a fully '
            'specified executable coordinator;online selection still needs its own seal.',
        runtime_connected=False, selector_defined=False, replay_authorized=False,
        training_authorized=False, formal_authorized=False, main_policy_changed=False)


def source_audit(scripts):
    query = assigned_fields(scripts/'run_freqduet_v216_passive_query.py', '__call__', 'q')
    current = assigned_fields(scripts/'freqduet_v224_prepare_capture.py', 'prepare', 'identity')
    packet = assigned_fields(scripts/'freqduet_v224_prepare_capture.py', 'prepare', 'packet')
    assert {'executed_holding_before_s','context','onboard','feasible_bins'} <= set(query)
    assert {'clock','native','front','follower','phases','current'} <= set(packet)
    return dict(source_only=True, archive_completeness_verified=False,
        existing_query_holding_available=True, prepare_current_fields=current,
        prepare_holding_retained='holding_s' in current,
        required_extension=['own executed H sampled at native prepare,not query',
            'matched completed-front launch/H/departure timing with original occurrence',
            'explicit supporting event/identity/clock joins,with native and V224 domains separate'],
        unresolved_rear='No completed future rear timing permitted.',
        next_gate='Seal a minimal passive extension contract and no-op equivalence budget '
            'separately;no capture or replay is authorized by this field audit.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','parent-summary','elapsed-summary','source'):
        p.add_argument('--'+name,required=True,type=str if name=='source' else Path)
    a = p.parse_args()
    for path, source, status in ((a.parent_summary,'39cd832b46','v245_readonly_holding_propagation_complete'),
        (a.elapsed_summary,'8050fd604e','v245_observed_stop_split_complete')):
        parent = json.loads(path.read_text())
        assert (parent['source'],parent['status']) == (source,status)
        assert parent['new_rollouts'] == parent['training_updates'] == 0
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v246_compensation_supply.py')
    assert tests.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful()
    protocol = registration(a.source)
    path = a.out_root/'protocol.json'
    assert not path.exists(), 'sealed protocol is not overwritten'
    audit = source_audit(Path(__file__).resolve().parent)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    summary = dict(status='v246_identification_and_persistent_state_protocol_sealed',
        source=a.source,tests_passed=result.testsRun,budget=protocol['budget'],
        supersedes_design_source=protocol['supersedes_design_source'],clock_domains_separated=True,
        source_interfaces=audit,protocol_path=str(path),runtime_connected=False,
        selector_defined=False,model_fits=0,replay_authorized=False,formal_authorized=False,
        main_policy_changed=False,v238_changed=False,next_step=audit['next_gate'])
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('V246_PROTOCOL_SEALED_NO_RUNTIME '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
