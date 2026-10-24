#!/usr/bin/env python3
"""Seal action-difference and blinded-control estimands; no replay or selector."""

import argparse
import json
import math
from pathlib import Path
import unittest

from preregister_freqduet_v246_compensation_supply import ACTIONS, CONSTRAINTS


def holding_differences(reference, alternative):
    """Actual executed doses, not requested bins or sums of repeated H prefixes."""
    fields = ('anchor_executed_s', 'anchor_cycle_H_s', 'all_cycles_H_s')
    assert all(reference[k] >= 0 and alternative[k] >= 0 for k in fields)
    for row in (reference, alternative):
        assert row['anchor_executed_s'] <= row['anchor_cycle_H_s'] <= row['all_cycles_H_s']
    direct, cycle, total = (alternative[k]-reference[k] for k in fields)
    later = cycle-direct
    other = total-cycle
    assert math.isclose(direct+later+other, total, rel_tol=0., abs_tol=1e-6)
    return dict(direct_executed_delta_s=direct, later_same_cycle_delta_s=later,
        other_cycles_delta_s=other, all_cycles_delta_s=total,
        holding_rebound_observed=direct < 0 and later > 0)


def supply_differences(reference, alternative, native_dt_s, query_obligation):
    """Evaluation only;both next rows must be FIRST actual same-bus launches."""
    assert reference['cycle_key'] == alternative['cycle_key']
    assert reference['launch_s'] == alternative['launch_s']
    if reference['return_s'] is None or alternative['return_s'] is None:
        return dict(status='incomplete_anchor_cycle', return_delta_s=None,
                    first_next_delta_s=None, public_delay_delta_s=None)
    for row in (reference, alternative):
        assert math.isclose(row['return_s'], row['launch_s']+row['H_s']+row['other_elapsed_s'],
                            rel_tol=0., abs_tol=1e-6)
    returned = alternative['return_s']-reference['return_s']
    out = dict(return_delta_s=returned, cycle_H_delta_s=alternative['H_s']-reference['H_s'],
        nonholding_delta_s=alternative['other_elapsed_s']-reference['other_elapsed_s'],
        first_next_delta_s=None, return_to_next_wait_delta_s=None, public_delay_delta_s=None,
        reference_first_next_trip=reference['next_trip'], alternative_first_next_trip=alternative['next_trip'],
        serves_query_obligation=None, query_due_obligation_match=None)
    if reference['next_trip'] is None or alternative['next_trip'] is None:
        return dict(out, status='missing_first_next_launch')
    if reference['next_trip'] != alternative['next_trip']:
        return dict(out, status='changed_first_next_obligation')
    assert reference['next_public_s'] == alternative['next_public_s']
    waits = [r['next_launch_s']-r['return_s']-native_dt_s for r in (reference, alternative)]
    assert all(w >= 0 for w in waits)
    nxt = alternative['next_launch_s']-reference['next_launch_s']
    assert math.isclose(returned+waits[1]-waits[0], nxt, rel_tol=0., abs_tol=1e-6)
    serves = query_obligation is not None and reference['next_trip'] == query_obligation['trip_id']
    return dict(out, status='same_first_next_obligation', first_next_delta_s=nxt,
        return_to_next_wait_delta_s=waits[1]-waits[0],
        public_delay_delta_s=max(0., alternative['next_launch_s']-alternative['next_public_s'])-
            max(0., reference['next_launch_s']-reference['next_public_s']),
        serves_query_obligation=serves,
        query_due_obligation_match=serves and query_obligation['kind'] == 'due')


def check_future_split(selection_roots, evaluation_roots):
    assert len(selection_roots) == len(set(selection_roots)) == 8
    assert len(evaluation_roots) == len(set(evaluation_roots)) == 8
    assert set(selection_roots).isdisjoint(evaluation_roots), 'future banks overlap'
    return dict(selection=8, evaluation=8, evaluation_halves=[4, 4])


def registration(source):
    return dict(stage='v253_compensation_cost_supply_action_difference_preregistration', source=source,
        parent_source='c491406e47', lineage_source='7e9faca963', v238_source='b4b1d00203',
        authorization='Protocol and synthetic accounting checks only;no archive or runtime access.',
        current_budget=dict(trajectory_scans=0, passive_paths=0, intervention_paths=0,
            future_starts=0, model_fits=0, training_updates=0, shadow_actions_executed=0),
        frozen=dict(fleet=12, holding_bins_s=list(ACTIONS), base='V107 bilateral025',
            other='V238,main,actors,critics,duals,predictors,calibrations,windows,quantiles,'
                'masks and native upper law/cadence/bounds/EMA/anchors remain unchanged.'),
        purpose='Measure executable action DIFFERENCES and their full-horizon consequences,'
            'not fit return levels or relabel stable supply matching as removable H. '
            'V252 is consumed development evidence,not independent confirmation.',
        atomic_anchor='For a later separately approved finite context manifest,use the FIRST '
            'native nonterminal lower query at t>=7200s with positive already-sampled native '
            'holding,supported frozen PREPARE F/B and at least two native-feasible bins. '
            'Anchor selection itself is supply-blind:do not require future/due association,'
            'large H or a later matching supply identity. Keep the original query occurrence,'
            'bus/trip/observed-launch key,plans,positions,H and both clock domains. '
            'Absent anchor is explicit and stops that fixed slot;no later winning replacement.',
        atomic_grid='At that SAME pre-command physical state,externally enumerate ALL and '
            'ONLY native-feasible seven-bin commands versus unchanged native,including '
            'smaller,native and larger bins. This is a finite one-query counterfactual '
            'experiment,NOT a state-to-bin rule or uniform reduction. Native-bin override '
            'is an engineering alias,not a second treatment. Masks never expand;no H '
            'credit,negative holding,new vehicles,terminal wait or larger upper gain.',
        continuation='After the one registered pulse,both branches resume their own native '
            'bilateral025 lower feedback and unchanged upper feedback at native cadence '
            'through original clearing. Frozen actor/law does NOT mean frozen actions. '
            'No saved later-action tape,permanent suppression or fixed-plan substitution. '
            'Any sustained/joint coordinator needs a separately frozen rule and budget.',
        local_cost='Retain cached PREPARE identities,ages,H and completed-front decomposition. '
            'Record R(a)-R(h)=2(a-h)[a+h-(B-F)] independently of supply evidence. '
            'Never deduct cumulative H from F/B again. Positive local regret is a '
            'tradeoff to measure,not a cut licence or a strict local-native-nonharm guard.',
        holding_account='Use ACTUALLY executed anchor doses. Delta total H = delta anchor '
            'dose + delta later H on the same inbound physical cycle + delta H on all '
            'other cycles. Later H=cycle-H difference minus direct executed difference,'
            'including missing/added later queries via physical ledgers,not time-aligned '
            'query matches. Positive later H after a reduction is rebound,not automatically '
            'harm:report its passenger and actual-CV2 consequences. Never sum repeated H '
            'prefixes or attribute all later holding to the most recent plan.',
        supply_account='For the common inbound cycle,delta return = delta executed cycle H '
            '+ delta nonholding elapsed (observed launch is fixed). If FIRST next actual '
            'same-bus trip agrees across branches,delta next launch = delta return + '
            'delta postreturn wait;retain the mandatory native dispatch tick separately. '
            'Compare original-public delay for the SAME next trip. Earlier return without '
            'earlier launch is not delivered supply benefit. Changed next trip,missing '
            'next launch or incomplete cycle remain explicit,not zero. Never search for '
            'a later launch that matches the predicted plan. Future labels are evaluation '
            'only,never anchor/state filters. System outcomes include ALL units,not a '
            'post-action matched-supply subset;chain closure is accounting,not mediation.',
        outcomes=['passenger_person_s', *CONSTRAINTS],
        metric_contract='Full-day waiting+onboard exposure including unfinished demand;'
            'frozen actual both-direction arrival CV2,readiness mean/total,original-public '
            'SUM max(actual launch-public baseline,0) INCLUDING control waiting,and '
            'unlaunched/incomplete counts. Short-window/local proxy improvements never '
            'replace full-day endpoints. Keep passenger journey denominator and served '
            'population unchanged;do not interpret CV normalization alone as benefit.',
        blind_controls=dict(
            atomic_sham='At the SAME anchor and externally fixed bin,resource-visible and '
                'resource-hidden executions MUST be identical. This is one physical arm '
                'with two metadata views,not an incremental coordination experiment. '
                'Do not schedule duplicate alias rollouts or manufacture different bins.',
            causal_resource_control='Before a later coordinator trial,freeze resource-aware '
                'and supply-blind intervention-allocation rules TOGETHER. Same compensation '
                'function,actual H,masks,authority,calibration/training budget and fallbacks;'
                'blind removes ONLY plans,positions,associations and their resource history. '
                'Do not substitute H=0,force a fabricated obligation,or match realized doses. '
                'The exact rule and its action map are intentionally NOT defined here.',
            admission='No resource-specific efficacy trial may start until both rules are '
                'sealed before future access and a same-state passive comparison shows '
                'genuine request distinctions. Identical outputs mean an alias,not failure '
                'of the blind implementation;stop rather than retune on V244/V249/V252.',
            contrasts='Later lower-aware vs lower-blind and joint-aware vs joint-blind '
                'identify resource increments;retain V107,upper-only,lower-only,joint '
                'and both blind arms,V238 historical and strong fixed+bilateral/fixed+zero. '
                'Use the V246 factorial/resource interactions for any hierarchy claim.'),
        selection_and_evaluation=dict(
            plan='Eight selection futures S0--S7 and eight DISJOINT evaluation futures '
                'E0--E7 per fixed context;E0--E3/E4--E7 are fixed halves. These are '
                'symbolic roles only:ZERO seeds,contexts or rollout slots are reserved now. '
                'Freeze actual roots,contexts,source/authority and started-path limits '
                'after metadata-only unused checks and separate approval BEFORE access.',
            coupling='Use existing V68 fork_future and V69 event-keyed categorical uniforms '
                'within each context/root. Retain already sampled current-window demand/'
                'road and day latents. Fork only after exact common pre-command state and '
                'native action capture;do not resample that action. Common logical events '
                'share streams;branch-only events and identity shifts remain real. '
                'Different future innovations do not create independent demand days.',
            discovery='After ALL S outcomes,lock ONE finite-context action contrast:among '
                'commands whose selection MEAN passenger difference<0 and every listed '
                'constraint difference<=0,choose least passenger time;treat ties by smaller '
                'absolute dose change then original bin order. If none,lock native/no effect. '
                'This is an offline experiment choice,not an online selector or fitted '
                'state-to-bin map. Never use next-supply identity as a selection filter.',
            confirmation='Seal chosen contrast IDs and all S results before any E outcome. '
                'Evaluate ALL legal atomic commands on the fixed E bank regardless of '
                'which wins S;reuse native aliases. Primary estimate is the S-locked '
                'contrast,all other commands are descriptive,not new primary winners. '
                'No E re-selection,reverse cross-fitting,optional stopping,extra futures '
                'or context replacement. Resource-aware/blind rules,if later admitted,'
                'must share the SAME budget and keep their choices frozen through E.'),
        statistics='Average repeats within context,then crossed actor/day paired bootstrap '
            '5000 draws,seed253;also conditional-future intervals on fixed contexts. '
            'Share resamples across metrics/arms/contrasts. Report each actor/day and both '
            'E halves,selection-to-evaluation change and paired noise vs action signal. '
            'No query-level pseudo-replication or best-E-bin confirmatory claim.',
        gate='Primary locked contrast:passenger95%CI upper<0 and actual CV2/readiness/'
            'original-public delay/completion CI uppers<=0 under both interval types;'
            'each fixed E half also has passenger improvement and constraint nonharm. '
            'V246 system/resource gates are NOT relaxed. Local harm labels require actual '
            'paired CV2 evidence,not R. Earlier matched return/first next launch is a '
            'mechanism requirement,not sufficient policy benefit. Resource attribution '
            'ALSO needs actual incremental delay benefit against non-alias blind control '
            'and V246 query-recorded DUE obligation criterion. An early future-watch '
            'chain is reported secondarily,never substituted for that criterion.',
        next_gate='Separately seal context/future manifest,finite atomic-pulse budget and '
            'exact passive/no-op preflight interface before any simulator start. This '
            'protocol authorizes NONE. A later online/sustained coordinator and genuine '
            'blind rule need their own design seal;atomic response evidence cannot deploy one.',
        limitations='Single-pulse conditional action effects are not a sustained controller '
            'or frequency-separation result. Blind metadata alone cannot alter fixed actions. '
            'V252 matching on one consumed actor/day does not establish new-day reliability. '
            'Finite-context offline oracle gains need independent-future replication and '
            'cannot become a deployable state-action table. No current result is claimed.',
        runtime_connected=False, selector_defined=False, offline_choices_made=False,
        collection_authorized=False, replay_authorized=False, training_authorized=False,
        future_manifest_reserved=False, main_policy_changed=False, v238_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--parent-summary', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    a = parser.parse_args()
    parent = json.loads(a.parent_summary.read_text())
    assert parent['status'] == 'v252_early_supply_audit_complete' and parent['source'] == 'c491406e47'
    assert parent['analyzed_paths'] == 2 and not parent['selector_defined']
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v253_action_supply.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    spec = registration(a.source)
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v253_action_difference_protocol_sealed', source=a.source,
        parent_source=spec['parent_source'], tests=tested.testsRun, budget=spec['current_budget'],
        runtime_connected=False, selector_defined=False, offline_choices_made=False,
        collection_authorized=False, replay_authorized=False, training_authorized=False,
        future_manifest_reserved=False, main_policy_changed=False, v238_changed=False,
        blind_alias_scope='Same fixed command at same anchor is an alias,not resource efficacy.',
        next_gate=spec['next_gate'])
    (a.out_root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V253_SEALED '+json.dumps(summary, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
