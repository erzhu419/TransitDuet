#!/usr/bin/env python3
"""Seal early observable supply linkage and compensation estimands, not actions."""

import argparse
import copy
import json
from pathlib import Path
import unittest


def link_prefix(previous, view):
    """Unconnected reference over a V232 causal view; no action or ETA output."""
    key, now, seq = view['cycle_key'], view['query_s'], view['native_event_seq']
    held, own = view['executed_holding_s'], view['own_association']
    assert own['bus_id'] == key[0] and own['inbound_trip_id'] == key[1]
    assert own['position_sample_s'] == now and held >= 0
    if previous is not None:
        assert previous['cycle_key'] == key, 'new physical cycle requires its observed launch'
        assert previous['query_seq'] < seq and previous['query_s'] <= now
        assert previous['H_s'] <= held, 'plan/association changes cannot reset executed H'
    old = None if previous is None else previous['episode']
    obligation = own['obligation']
    current = dict(query_seq=seq, query_s=now, H_s=held,
        obligation=None if obligation is None else copy.deepcopy(obligation),
        distance_rank=own['distance_rank'], remaining_distance_m=own['remaining_distance_m'],
        posted_lead_s=own['posted_lead_s'])
    if obligation is not None:
        assert obligation['kind'] in ('future', 'due')
        assert obligation['last_write_version'] == 'initial' or obligation['last_write_version'] < seq
    old_id = None if old is None else old['start']['obligation']['trip_id']
    new_id = None if obligation is None else obligation['trip_id']
    changed = old_id != new_id
    episode = (None if obligation is None else dict(start=copy.deepcopy(current))
               if changed or old is None else copy.deepcopy(old))
    revised = old is not None and new_id == old_id and obligation['last_write_version'] != previous['current']['obligation']['last_write_version']
    transition = ('lost' if changed and new_id is None else 'opened' if changed and old_id is None
                  else 'reassociated' if changed else 'revised' if revised else 'retained')
    state = dict(cycle_key=list(key), query_seq=seq, query_s=now, H_s=held,
        current=current, episode=episode, transition=transition,
        closed_episode=copy.deepcopy(old) if changed and old is not None else None,
        first_link=None if previous is None else copy.deepcopy(previous['first_link']),
        first_own_due=None if previous is None else copy.deepcopy(previous['first_own_due']),
        episodes_started=(0 if previous is None else previous['episodes_started'])+
            int(episode is not None and (changed or old is None)),
        target_due_present=bool(view['target_profile']['uncovered_due']),
        own_due=obligation is not None and obligation['kind'] == 'due',
        own_future=obligation is not None and obligation['kind'] == 'future')
    if obligation is not None and state['first_link'] is None:
        state['first_link'] = copy.deepcopy(current)
    if state['own_due'] and state['first_own_due'] is None:
        state['first_own_due'] = copy.deepcopy(current)
    state['H_since_episode_start_s'] = None if episode is None else held-episode['start']['H_s']
    # Historical first-link markers are never the current obligation or a budget.
    state['H_since_cycle_first_link_s'] = None if state['first_link'] is None else held-state['first_link']['H_s']
    return state


def registration(source):
    return dict(stage='v251_early_causal_supply_linkage_preregistration', source=source,
        parent_source='5a72961382', v238_source='b4b1d00203',
        authorization='Design seal and synthetic prefix tests only;no trajectory reads.',
        budget=dict(trajectory_scans=0, passive_paths=0, intervention_paths=0,
            archive_shadow_requests=0, new_futures=0, model_fits=0, training_updates=0),
        frozen=dict(fleet=12, holding_bins_s=[0,5,10,15,20,30,45],
            base='V107 bilateral025;native B200 or frozen V220 upper only.',
            other='V238,main,actors,critics,duals,predictors,calibrations,windows,quantiles,'
                'masks,upper cadence/bounds/EMA/anchors/zero-sum constraints unchanged.'),
        motivation='V250 first-own-due H mostly predates due exposure. V232/V238 already '
            'associate future obligations:earlier triggering alone is NOT a new controller '
            'or a demonstrated repair. Test whether earlier observable lineage is stable '
            'and leaves prospective action opportunity without relabelling compensation as slack.',
        information=dict(
            resource='ONLY the existing V232 current published pending plans/versions,'
                'current ready FIFO inventory,compatible incoming directions/positions,'
                'observed launch cycle and executed H at native query. Clocks and original '
                'inventory ties are provenance. No ETA,new predictor,load-based risk score,'
                'hidden road/demand state or future identity. H changes bookkeeping only,'
                'not ready allocation or positional association.',
            compensation='Keep the existing baseline PREPARE F/B and completed-front '
                'launch/H/nonholding decomposition as a SEPARATE diagnostic,not extra '
                'resource-link inputs. No new estimator call or future rear label. '
                'PREPARE H never takes query H;native and V224 order domains stay separate.',
            availability='V230 query payload/versions and V235 cycle views already contain '
                'resource fields;V248 contains PREPARE H/front timing and exact joins. '
                'Source interfaces inspected,not archives. No earlier-than-first-query '
                'resource snapshot is claimed;unobserved prefixes remain unavailable.'),
        linkage='Reuse unchanged V232 one-use ready allocation and incoming distance order. '
            'Unbacked future is a WATCH,not due shortage. Keep own future and target other-due '
            'as simultaneous axes. No forecasted arrivals fund advances. At EVERY query '
            'refresh from current plans/positions;never protect an old assignment.',
        persistence='At first observed own unbacked association,record native query seq/time,'
            'physical cycle,plan ID/version/kind,position rank/distance,posted lead and H. '
            'Keep a bounded first-link history marker. A consecutive episode is the same '
            'physical cycle AND same currently associated plan ID,through plan revisions '
            'and future-to-due transitions. Loss/reassociation closes it immediately;'
            'regaining the same plan starts a NEW episode. Preserve physical H across ALL '
            'such changes;reset only with an actually observed new launch. Never promote '
            'historical association to current obligation,ready bus,reservation or debt.',
        hypothesis_1='Earlier information:report first-link vs first-own-due clocks/H for '
            'ALL cycles. Separately report future-to-due lead and H change WITHIN each '
            'consecutive same-plan episode;never join disjoint episodes or different plans '
            'to inflate lead. Keep never-due,lost,unassociated,no-query and incomplete '
            'cycles explicit. No horizon,H cutoff,distance threshold or successful-tail filter.',
        hypothesis_2='Reliability:describe plan-write/rank/association changes and pressure '
            'relief. Only AFTER prefix extraction evaluate every early association against '
            'FIRST next same-bus launch. Keep mismatches,no-next and switched plans;do not '
            'backdate the eventual correct plan or pick the earliest successful episode. '
            'Separate episode/query exposure from one-sample-per-physical-cycle reporting.',
        compensation_tradeoff='For every native-feasible a,retain a-h and '
            'R(a)-R(h)=2(a-h)[a+h-(B-F)]. Keep native,smaller AND larger legal bins,'
            'terminal masks and unresolved rear evidence. F=launch gap+executed-H gap+'
            'nonholding gap is accounting;do not subtract H again or call one component '
            'removable slack. Positive local regret is an explicit tradeoff,not automatic '
            'rejection or permission. R(a)<R(0) does not preserve native benefit;strict '
            'R(a)<=R(h) cut guards are not a repair. No H/scale attenuation,uniform cut,'
            'gain escalation,weighted score,action rank,threshold or online bin mapping.',
        causal_effects='Timing and covariance cannot identify action benefit. A later '
            'SEPARATELY sealed executable candidate must be compared at matched states '
            'with native subsequent feedback and independent selection/evaluation futures. '
            'Measure actual holding,subsequent holding,nonholding time,same-bus return,'
            'FIRST next launch and original-public total delay. Never hold later actions '
            'fixed or dose-match/filter on post-action H. Local proxy vs full-day actual '
            'CV2 remains separate. Historical H cannot be recovered by a later command.',
        future_controls='Inherit V246:V107,upper-only,new lower-only,joint,resource-blind '
            'lower/joint using the SAME future action formula with only resource linkage '
            'removed;frozen V238 as historical comparator;strong exact-fixed+bilateral '
            'and exact-fixed+zero externally. No new upper law or controller exists here.',
        future_gate='V246 gates unchanged:paired passenger95%CI upper<0;actual CV2,'
            'readiness,original-public total delay,incomplete/unlaunched CI uppers<=0 '
            'against EACH primary control. Resource-specific claim also needs incremental '
            'public-delay benefit vs blind and actual same-chain supply changes under '
            'the V246 due-obligation criterion. Early future watch is not a substitute '
            'for that criterion. Report both independent future halves and actor/day '
            'crossed paired statistics,not query pseudo-replication. No margins relaxed.',
        next_gate='Seek separate approval for ONE read-only audit of the two accepted '
            'V249 paths using this exact prefix protocol;no new path. If early fields '
            'are absent or links churn before a useful prospective decision,report the '
            'limitation and stop before selector/replay. Earlier linkage alone admits '
            'neither execution nor a safe H cut. Any later action rule and started-path '
            'budget require separate preregistration before passive integration/smoke.',
        limitations='V250 is one consumed actor/day. Position order is not return order '
            'or known assignment. Future lack of current stock does not identify future '
            'shortage without time information;no travel-time model is added. Interval H '
            'cannot be attributed to one obligation. Synthetic prefix correctness is not '
            'recording equivalence,causal benefit or a frequency-separation result.',
        runtime_connected=False, selector_defined=False, archive_analysis_authorized=False,
        collection_authorized=False, replay_authorized=False, training_authorized=False,
        main_policy_changed=False, v238_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--parent-summary', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent_summary.read_text())
    assert parent['status'] == 'v250_read_only_complete' and parent['source'] == '5a72961382'
    assert parent['analyzed_paths'] == 2 and parent['new_replays'] == parent['training_updates'] == 0
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v251_early_supply.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    spec = registration(args.source)
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v251_early_supply_protocol_sealed', source=args.source,
        parent_source=spec['parent_source'], tests=tested.testsRun, budget=spec['budget'],
        runtime_connected=False, selector_defined=False, archive_analysis_authorized=False,
        collection_authorized=False, replay_authorized=False, training_authorized=False,
        main_policy_changed=False, v238_changed=False, next_gate=spec['next_gate'])
    (args.out_root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V251_SEALED '+json.dumps(summary, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
