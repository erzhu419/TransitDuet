#!/usr/bin/env python3
"""Seal passive resource capture, clocks and path limits; no rollout entrypoint."""

import argparse
import json
from pathlib import Path
import unittest

from preregister_freqduet_v224_prepare_handoff import CONTEXTS


PAYLOAD_FIELDS = ('snapshot_read_s', 'snapshot_event_seq', 'ready', 'active', 'plans',
                  'onboard', 'executed_holding_before_s', 'terminal_masked')
LINK_FIELDS = ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence')
FIRST_PAIR_CHECKS = ('legacy_exact', 'v224_sidecar_exact', 'query_payload_complete',
    'query_prepare_clocks_valid', 'upper_versions_strict_prefix', 'physical_cycles_valid',
    'ready_and_plan_orders_preserved', 'ready_dispatch_boundary_valid',
    'observer_rng_and_native_calls_unchanged',
    'shadow_never_executed')


def first_pair_passed(results):
    return (len(results) == 2 and {r['slot'] for r in results} == {'c0_v107', 'c0_upper'}
            and all(r.get(k) is True for r in results for k in FIRST_PAIR_CHECKS))


def check_query_link(row, query, prepare, launch, versions):
    """Validate new payload against original query/prepare and observed events."""
    assert set((*PAYLOAD_FIELDS, *LINK_FIELDS, 'cycle_ref', 'message_refs',
                'pending_plan_version_refs')) <= row.keys(), 'missing extension field'
    assert all(row[k] == query[k] for k in LINK_FIELDS), 'query occurrence join changed'
    clock = prepare['clock']
    assert clock['handoff_order'] == row['prepare_order'] < row['query_order']
    assert clock['prepare_s'] <= row['snapshot_read_s'] == query['query_s'], 'wrong query clock'
    assert row['snapshot_event_seq'] == row['legacy_query_event_seq'], 'wrong native snapshot ordinal'
    identity = query['identity']
    assert all(prepare['current'][k] == identity[k] for k in ('bus_id', 'trip_id', 'direction', 'station_id'))
    assert launch['kind'] == 'launch' and launch['event_seq'] < row['legacy_query_event_seq']
    assert launch['time_s'] <= query['query_s']
    assert row['cycle_ref'] == [identity['bus_id'], identity['trip_id'], launch['event_seq']]
    assert (launch['bus_id'], launch['trip_id']) == (identity['bus_id'], identity['trip_id'])
    assert row['executed_holding_before_s'] >= 0 and row['onboard'] >= 0
    buses = row['ready'] + row['active']
    assert len(buses) == len({b['bus_id'] for b in buses}) == 12, 'inventory is not the original fleet'
    assert all(b['position_sample_s'] == query['query_s'] for b in buses)
    assert any((b['bus_id'], b['trip_id']) == (identity['bus_id'], identity['trip_id'])
               for b in row['active']), 'query vehicle absent from current inventory'
    assert set(row['message_refs']) == {'false', 'true'}
    assert set(row['pending_plan_version_refs']) == {str(p['trip_id']) for p in row['plans']}
    for direction, version in row['message_refs'].items():
        completed = [v for v in versions.values() if not v['initial']
                     and v['direction'] == (direction == 'true')
                     and v['native_event_seq'] < row['legacy_query_event_seq']]
        expected = max(completed, key=lambda v: v['native_event_seq'])['version_id'] if completed else 'initial'
        assert version == expected, 'not the latest completed direction callback'
    for version in (*row['message_refs'].values(), *row['pending_plan_version_refs'].values()):
        event = versions[version]
        if not event['initial']:
            assert event['native_event_seq'] < row['legacy_query_event_seq'], 'future plan version'
            assert event['time_s'] <= query['query_s']


def check_cycle_progress(previous, current):
    if previous['cycle_ref'] == current['cycle_ref']:
        assert previous['legacy_query_event_seq'] < current['legacy_query_event_seq']
        assert previous['executed_holding_before_s'] <= current['executed_holding_before_s'], \
            'executed holding reset within a vehicle cycle'


def registration(source):
    reference = 'protocol_v6_v224_prepare_collection'
    slots = [dict(slot=f'c{ci}_{arm}', context=ci, actor=actor, demand_day=day, arm=arm,
        sampler_seed=187000000+actor+100*day,
        legacy_reference=f'{reference}/records/c{ci}_{arm}.json.gz',
        handoff_reference=f'{reference}/sidecars/c{ci}_{arm}.json.gz')
        for ci, (actor, day) in enumerate(CONTEXTS) for arm in ('v107', 'upper')]
    return dict(stage='v230_passive_resource_capture_preregistration', source=source,
        parent_source='aa48aa173e', current_authorization='Seal and synthetic tests only.',
        current_budget=dict(archive_scans=0, passive_paths=0, intervention_paths=0,
                            training_updates=0, model_fits=0, new_futures=0),
        reserved_passive=dict(max_started_paths=8, slots=slots,
            first_pair=['c0_v107', 'c0_upper'], subsequent_slots=[s['slot'] for s in slots[2:]],
            max_workers=4, extra_reference_paths=0, launch_authorized=False,
            failure_rule='Started attempts,including incomplete/failed paths,consume their '
                'fixed slot. No automatic retry,extra references,context replacement or budget '
                'extension. Reuse complete saved outputs for analysis-only recovery. c0 pair '
                'must pass ALL engineering checks before the other six;outcomes cannot pick paths.'),
        frozen=dict(controller_source='2b0115af7b', passive_source='17043d5c03',
            controls='Only archived V107 and V220 upper arms. No V225 common-reference '
                'execution or V221 old forced-cut lower/joint branch.',
            runtime='FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet',
            authority='12 vehicles,seven bins,native masks and V220 ready-backed projection; '
                'no gain,EMA,bounds,anchors,900s cadence,cache,dispatch or timetable change.',
            unchanged='Actors,critics,duals,all predictor/calibration/filter versions and main.'),
        files=dict(legacy=f'{reference}/records reused as references;new legacy copies stay separate',
            original_sidecar='Preserve the complete V224 sidecar key set and contents,including '
                'all old prepare/query/phase/event rows,observer counts and native-call counts.',
            new_extension='Separate resource_sidecars/cN_ARM.json.gz only. Do not append fields '
                'to original legacy or V224 rows. No extra marks in their event counters.'),
        schema=dict(metadata=['native_dt_s', 'native_inventory_order', 'native_timetable_order',
                'initial_published_plans', 'initial_version_id'],
            query_extension=[*LINK_FIELDS, *PAYLOAD_FIELDS, 'cycle_ref', 'message_refs',
                             'pending_plan_version_refs', 'shadow_frontier'],
            ready='Copy each original q.ready row in order:bus_id,direction,position_sample_s, '
                'return_observed,last_observed_return_s. Null initial-stock return is not0.',
            active='Copy q.active exactly,including bus/trip,direction,state,station IDs, '
                'progress,remaining distance and position_sample_s;no ETA or future assignment.',
            plans='Copy q.plans exactly in native iteration order:trip_id,direction,launched, '
                'planned_s,public_s,target_s. Full native order comes from initial metadata.',
            upper_version=['version_id', 'native_event_seq', 'time_s', 'direction',
                'fresh_upper_event_seq', 'callback_trip_id', 'changes'],
            cycle_event=['kind', 'native_event_seq', 'time_s', 'bus_id', 'trip_id', 'direction',
                'launch_native_event_seq', 'executed_holding_s'],
            shadow_frontier=['status', 'native_bin', 'candidates'],
            shadow_candidate=['bin', 'holding_s', 'delta_holding_s',
                'immediate_onboard_delta_person_s', 'local_gap_cost_delta_s2']),
        query_clocks='Copy the ALREADY CAPTURED original live q inside the existing native '
            'query call before issue/draw. snapshot_read_s=q.time_s and snapshot_event_seq= '
            'q.event_seq. Join old V224 query row by legacy native query event and occurrence; '
            'join its original prepare order/identity,cached context and state_captured_s. '
            'Resource/load/prefix are QUERY samples;F/B and identities keep their original '
            'PREPARE clocks. Do not overwrite either or call obs.snapshot/estimators again. '
            'Existing aliases query_s/native_bin/prepare.context remain references,not duplicated '
            'missing-field claims. Native and V224 ordinal domains remain distinct.',
        versions='Emit initial published plans once with version_id=initial. After EACH '
            'completed original plan_update callback,copy its existing callback changes and '
            'native event;version_id is that native event_seq. fresh_upper_event_seq is the '
            'actual fresh query inside THIS callback,else null for reuse. No last-fresh '
            'snapshot masquerading as post-callback plans. Track latest completed callback '
            'per direction and latest actual changed-plan version per pending trip;unchanged '
            'trips retain their prior version. Use actual before/after rows,not requested '
            'projection output. A query may reference only strict-prior completed versions. '
            'Query posted plans must reconstruct from initial plus actual changed-plan prefix. '
            'This captures reuse without introducing any extra upper decision or plan write.',
        cycles='Open bus/trip/observed-launch-event cycle from original launch rows;direction '
            'comes from that actual dispatch. Each query references this strict-prior launch. '
            'Its executed_holding_before_s is the existing observed prefix,not commanded '
            'holding or a return label. It cannot decrease within a cycle or reset at a replan. '
            'Append completed return from its original event/account;then link FIRST later '
            'same-bus launch as a separate outcome row. Never backfill this link into query '
            'features. New trip on reused bus opens a new cycle;initial stock is not a '
            'fictional inbound return. Retain all open/missing/end-of-day cycle statuses.',
        ready_timing='Record native dt and stable original inventory/timetable order once. '
            'Do not sort copied q lists. Any later ready-coverage shadow must use native '
            'pending dispatch order,FIFO by observed return with original inventory-order '
            'ties,one use per compatible ready bus. Return at R is usable only at R+dt. '
            'Explicitly anchor coverage to the NEXT native dispatch pass;query inventory '
            'is not retrospective inventory for a pass already completed. Incoming distance '
            'does not fund an advance or certify the next actual supply assignment.',
        shadow='Enumerate exact native feasible bins using unchanged q/V224 metadata. Record '
            'delta holding,current query onboard exposure,and V229 quadratic difference '
            'only if cached F/B evidence is supported. Terminal/missing-evidence rows remain '
            'with explicit status and null unavailable R,not omitted or filled0. Preserve '
            'the native bin. No new selected bin,compensation label,ETA,score fitting or '
            'executed shadow. Existing V223 shadow in the old sidecar remains unchanged.',
        equality='Reuse the established require_legacy_equal oracle against each V224 '
            'legacy record AND old V224 sidecar. Recursive field/value/list-order equality '
            'with no tolerance,ignored fields,new legacy keys or wall-time comparison. '
            'This includes all actions,inputs,probabilities,uniforms,plans,events,physical '
            'contract,holding ledgers,arrivals/departures/launches/returns and all endpoints. '
            'New observer checks use their OWN counters/RNG snapshots;do not increment '
            'old observer counters or call a native computation for logging.',
        first_pair_gate=list(FIRST_PAIR_CHECKS),
        recording_gate='Exactly one new row per existing lower-query occurrence,including '
            'masked/unsupported/zero-native rows. Every original field present,12 unique '
            'vehicles with current query stamps,complete plan order/version joins and '
            'strict-prefix cycle links. Preserve unused/repeated prepares. Close completed '
            'cycle holding/elapsed to the original return and issued-versus-executed ledger. '
            'Check upper-version transitions,readiness boundary and no upper-cycle prefix '
            'reset;RNG and native-call deltas inside each NEW observer body must be0. '
            'An absent field is recording failure,not scientific low support.',
        report='Per fixed path:legacy/old-sidecar equality,query count,field/clock/version/'
            'cycle coverage,unsupported reasons,masked/zero actions,feasible-bin counts '
            'and pressure/positive-native overlap. Keep all contexts regardless of counts. '
            'Do not call alternative feasible bins actual action authority or compare '
            'passenger outcomes as efficacy:all passive endpoints must equal their controls.',
        decision='Current seal authorizes NO collection. Next implement the independent '
            'extension and ask for the fixed passive pair/six-stage execution authorization. '
            'A pass only admits read-only shadow-support analysis. Missing support stops '
            'progression;no gain/threshold tuning or new selector. New action tests,fits, '
            'training or independent futures each require separately sealed protocols.',
        execution='scheduler high,manual dispatch,node001--006. Current tests and parent '
            'small-summary check only;no runtime/model imports or trajectory reads. Raw '
            'paths/sidecars stay remote;only tiny receipt/count summaries return locally.',
        collector_implemented=False, collection_authorized=False, selector_defined=False,
        main_policy_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--v229-root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v230_resource_capture.py')
    assert tests.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise SystemExit(1)
    parent = json.loads((args.v229_root/'summary/status.json').read_text())
    assert parent['status'] == 'v229_supply_coordination_design_sealed'
    assert parent['source'] == 'aa48aa173e' and set(parent['budget'].values()) == {0}
    protocol = registration(args.source)
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root/'protocol.json'
    assert not path.exists(), 'preserve the original sealed V230 protocol'
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    summary = dict(status='v230_passive_resource_capture_protocol_sealed', source=args.source,
        parent_source=parent['source'], tests=result.testsRun, current_budget=protocol['current_budget'],
        reserved_passive_paths=8, first_pair=['c0_v107', 'c0_upper'], subsequent_paths=6,
        extra_reference_paths=0, started_paths=0, collector_implemented=False,
        collection_authorized=False, selector_defined=False, main_policy_changed=False,
        protocol_path=str(path), next_step=protocol['decision'])
    (args.out_root/'summary').mkdir(exist_ok=True)
    (args.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V230_CAPTURE_PROTOCOL_SEALED_NO_COLLECTION '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
