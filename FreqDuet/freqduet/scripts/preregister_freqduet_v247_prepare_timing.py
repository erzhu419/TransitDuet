#!/usr/bin/env python3
"""Seal PREPARE timing fields and two future passive slots; no live observer."""

import argparse
import ast
import copy
import json
from pathlib import Path
import unittest

from preregister_freqduet_v246_compensation_supply import completed_forward_parts


IDENTITY = ('bus_id', 'trip_id', 'station_id', 'direction')
OLD_PAYLOADS = dict(legacy='records', v224='sidecars', v230='resource_sidecars',
                    v235='cycle_sidecars', v241='request_sidecars')


def prepare_sample(native, launch, cycle_key, counter_s, read_s, completed_front=None):
    """Pure reference for a future read at the existing native prepare stamp."""
    assert native['kind'] == 'prepare' and read_s == native['time_s']
    assert list(cycle_key[:2]) == [native['bus_id'], native['trip_id']]
    assert cycle_key[2] < native['event_seq']  # Native EventClock domain only.
    assert launch['bus_id'] == native['bus_id'] and launch['time_s'] <= read_s
    held = counter_s-launch['holding_counter_at_launch']
    assert 0 <= held <= read_s-launch['time_s']
    return dict(native=copy.deepcopy(native), read_s=read_s, cycle_key=list(cycle_key),
        launch_s=launch['time_s'], counter_s=counter_s,
        launch_counter_s=launch['holding_counter_at_launch'], holding_s=held,
        completed_front=copy.deepcopy(completed_front))


def map_prepare(old, sample):
    """Join original V224 occurrence; no query H or later front lookup is accepted."""
    clock, own, native = old['clock'], old['current'], sample['native']
    assert all(own[k] == native[k] for k in IDENTITY)
    assert clock['legacy_prepare_event_seq'] == native['event_seq']
    assert clock['prepare_s'] == clock['record_read_s'] == sample['read_s'] == native['time_s']
    phase = old['phases']['current']
    assert all(phase[k] == own[k] for k in ('bus_id', 'trip_id', 'direction'))
    assert phase['actual_launch_s'] == sample['launch_s']
    assert phase['launch_order'] < clock['handoff_order']  # V224 domain only.
    row = dict(identity=copy.deepcopy(own), cycle_key=list(sample['cycle_key']),
        prepare_s=native['time_s'], prepare_seq=native['event_seq'],
        prepare_order=clock['handoff_order'], prepare_occurrence=clock['prepare_occurrence'],
        forward_s=old['native']['forward_gap_s'] if old['native']['forward_valid'] else None,
        backward_s=old['native']['follower_gap_s'] if old['native']['follower_valid'] else None,
        own=dict(launch_s=sample['launch_s'], launch_order=phase['launch_order'],
                 holding_s=sample['holding_s']),
        front_identity=None, front=dict(launch_s=None, launch_order=None,
            departure_s=None, departure_order=None, holding_s=None),
        front_status='no_native_front', parts=None)
    if row['forward_s'] is None:
        return row
    front, observed = old['front'], sample['completed_front']
    assert front['status'] == 'supported'
    assert front['station_id'] == own['station_id'] and front['direction'] == own['direction']
    row['front_identity'] = {k: front[k] for k in ('bus_id', 'trip_id', 'direction')}
    row['front_status'] = 'completed_timing_unavailable'
    if observed is None:
        return row
    timing, launch = observed['timing'], observed['launch']
    assert all(timing[k] == front[k] for k in ('trip_id', 'station_id', 'direction'))
    assert launch['bus_id'] == front['bus_id']
    assert timing['time_s'] == front['departure_s'] <= row['prepare_s']
    assert front['launch_order'] < front['departure_order'] < row['prepare_order']
    assert timing['release'] == launch['time_s'] == old['phases']['front']['actual_launch_s']
    assert all(old['phases']['front'][k] == front[k] for k in ('bus_id', 'trip_id', 'direction'))
    assert timing['realized_holding'] >= 0 and timing['other_elapsed'] >= 0
    assert abs(timing['release']+timing['realized_holding']+timing['other_elapsed']
               -timing['time_s']) <= 1e-8
    row['front'] = dict(launch_s=timing['release'], launch_order=front['launch_order'],
        departure_s=timing['time_s'], departure_order=front['departure_order'],
        holding_s=timing['realized_holding'])
    row['parts'] = completed_forward_parts(row)
    row['front_status'] = 'completed_at_prepare'
    return row


def join_query(prepared, old_query, legacy_query, cycle_view):
    assert prepared['identity'] == old_query['identity']
    assert all(legacy_query[k] == prepared['identity'][k] for k in IDENTITY)
    assert prepared['prepare_seq'] == legacy_query['prepare_event_seq']
    assert prepared['prepare_order'] == old_query['prepare_order']
    assert prepared['prepare_s'] == legacy_query['state_captured_s']
    assert prepared['cycle_key'] == cycle_view['cycle_key']
    assert prepared['prepare_seq'] < legacy_query['event_seq'] == old_query['legacy_query_event_seq']
    assert cycle_view['native_event_seq'] == legacy_query['event_seq']
    assert prepared['prepare_order'] < old_query['query_order']
    assert prepared['prepare_s'] <= old_query['query_s'] == legacy_query['time_s'] == cycle_view['query_s']
    assert prepared['own']['holding_s'] <= legacy_query['executed_holding_before_s'] == cycle_view['executed_holding_s']
    return dict(prepare_seq=prepared['prepare_seq'], prepare_order=prepared['prepare_order'],
        query_seq=legacy_query['event_seq'], query_order=old_query['query_order'],
        query_occurrence=old_query['query_occurrence'], prepare_s=prepared['prepare_s'],
        query_s=old_query['query_s'], prepare_holding_s=prepared['own']['holding_s'],
        query_holding_s=cycle_view['executed_holding_s'])


def registration(source):
    reference = 'protocol_v6_v242_passive_requests_collection'
    slots = [dict(slot='c0_'+arm, actor=12037, demand_day=2151001,
        sampler_seed=187000000+12037+100*2151001, arm=arm,
        references={k: f'{reference}/{folder}/c0_{arm}.json.gz'
                    for k, folder in OLD_PAYLOADS.items()}) for arm in ('v107', 'upper')]
    return dict(stage='v247_prepare_timing_mapping_and_passive_budget', source=source,
        parent_source='1828d9cd02', reference_source='c6710744f9',
        authorization='Field mapping,source-only inspection and synthetic tests;no capture or replay.',
        current_budget=dict(archive_scans=0, passive_paths=0, intervention_paths=0,
            new_futures=0, model_fits=0, training_updates=0, shadow_actions=0),
        reserved_preflight=dict(max_started_paths=2, max_workers=1, slots=slots,
            order=['c0_v107','c0_upper'], extra_reference_paths=0, launch_authorized=False,
            failures_consume_slots=True, automatic_retry=False,
            progression='V107 must pass EVERY exact-equality and new-record gate before upper. '
                'A simulator start consumes its slot even if it fails or is incomplete. '
                'No replacement context,extra reference run,retry or expansion without approval.'),
        field_mapping=dict(
            own='At the EXISTING native EventClock prepare stamp,after native _prepare_for_action '
                'returns,read audit.buses[trip] episode_hold_vehicle_seconds minus '
                'audit.launches[trip].holding_counter_at_launch. Retain actual launch_s,bus/trip,'
                'station/direction,native prepare_seq,read_s,and current V235 physical cycle_key. '
                'Read the cycle key only;do not call CycleState.query or update its ledger.',
            front='At that SAME native prepare stamp,copy the already captured TimingAudit '
                'departure row for native forward_predecessor_trip_id/station/direction AND '
                'its stored actual launch row. This is a completed prefix,not a later bus read. '
                'After original V224 prepare callback returns,join its packet by '
                'legacy_prepare_event_seq. Match front bus/trip/station/direction,departure_s,'
                'original departure_order and launch_order,actual release/H/nonholding elapsed. '
                'Persistent front identity is bus/trip/direction;station belongs to the '
                'observation join,not the cross-station identity. '
                'Use an independent completed-event index when implementing;never rescan the '
                'full history per PREPARE or add marks to the old sidecar.',
            clocks='Native actual launch seq,prepare seq and query seq stay in EventClock. '
                'V224 launch/departure/prepare/query order stays in V224. Join by existing '
                'occurrence and identity,not ordinal equality across domains. Same-time '
                'departures require original departure_order<prepare_order. Copy before '
                'further simulation;query H is separately read and never fills PREPARE H.',
            absent='Preserve every native prepare,including unused,masked,zero-H and unsupported '
                'ones. Missing front timing is null with reason,not zero or a future lookup. '
                'Every old forward-valid PREPARE must obtain complete matched timing before '
                'preflight can pass. No-front rows remain no-front without inventing support.',
            query='One independent join per old query occurrence to its exact native prepare '
                'seq and V224 prepare_order;retain separate prepare/query clocks and H. '
                'Repeated queries may share a prepare but must keep distinct query occurrence.'),
        equality='ENTIRE five reference payloads legacy,V224,V230,V235,V241 must pass frozen '
            'require_legacy_equal recursively,with exact values,key sets,list order and no '
            'tolerance,rounding,sorting,ignored fields or digest substitution. This includes '
            'inputs,probabilities,uniforms,actions,all event and execution clocks,plans,'
            'callbacks,launches,returns,ledgers,endpoints,all old sidecar checks/counters '
            'and old shadow requests. Preserve legacy data types. Wall runtime is not a '
            'simulator endpoint. Compare full payloads remotely;pull only a small receipt.',
        recording_gate='Exactly one new sample and mapped row per existing native PREPARE '
            'and old V224 prepare occurrence,in order;one join per old query. Independently '
            'check own H against original launch baseline and actual counter sampled AT '
            'PREPARE,front timing against the completed prefix and old departure event,'
            'and F=release_gap+executed_H_gap+nonholding_elapsed_gap. Check each clock domain '
            'separately. Report unsupported/unused/zero-H rows and open/zero-query cycles;'
            'no missing query or forward-valid component is accepted. Plan versions never '
            'reset H;front changes remain explicit. Later queries/returns are checks only,'
            'never backfill sources for earlier samples.',
        passive_boundary='New independent SIXTH payload only. A future recorder may observe '
            'existing stamp/departure/prepare/query callbacks but returns their unchanged '
            'values exactly once. No extra native computation,snapshot,estimator,policy,'
            'planner,step or random draw. No old mark/stamp or observer read-context call '
            'added. New observer owns its counters and RNG/native-call invariance checks. '
            'V238 and V241 old shadows remain unchanged;no new online selection or shadow '
            'action request is computed by this extension.',
        frozen=dict(fleet=12, holding_bins_s=[0,5,10,15,20,30,45], v238_source='b4b1d00203',
            unchanged='V107/native B200 or frozen V220 upper,original masks/authority,'
                'actors,critics,duals,main,V151,V146,all predictors,windows and quantiles.'),
        stop='Any original-payload difference,missing occurrence,unsupported valid-front '
            'timing,clock/identity/closure failure or observer side effect stops progression. '
            'An unexercised state is reported,not an excuse to add paths. Passive equality '
            'does not establish action benefit or authorize smoke,training or formal tests.',
        next_step='Implement only the independent recorder against this mapping;synthetic '
            'checks first. Obtain separate approval before either reserved real path.',
        execution='scheduler high,explicit manual dispatch,node001--006;source/synthetic '
            'checks only now. No raw archive,CSV,cache or checkpoint download.',
        collector_implemented=False, runtime_connected=False, selector_defined=False,
        collection_authorized=False, replay_authorized=False, main_policy_changed=False)


def source_audit(scripts):
    files = {k: (scripts/name).read_text() for k, name in dict(
        query='run_freqduet_v216_passive_query.py', timing='run_freqduet_v91_timing_components.py',
        handoff='freqduet_v224_prepare_capture.py', reference='run_freqduet_v242_passive_requests.py').items()}
    nodes = {k: ast.parse(v) for k, v in files.items()}
    native = next(n for n in ast.walk(nodes['query']) if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    text = ast.get_source_segment(files['query'], native)
    assert text.index('self.native_prepare(') < text.index("self.clock.stamp('prepare'")
    for key, tokens in dict(timing=('holding_counter_at_launch', 'self.departures.append(capture(',
            'realized_holding=held', 'other_elapsed='),
        handoff=('legacy_prepare_event_seq', 'prepare_occurrence', 'departure_order', 'self.by_prepare_seq'),
        reference=("files['requests']", "root/'request_sidecars'", "'v242_passive_pair_passed'")).items():
        assert all(token in files[key] for token in tokens), ('source mapping changed', key)
    return dict(source_only=True, field_sources_present=True, native_prepare_after_body=True,
        own_prepare_holding_new=True, front_timing_from_completed_departure=True,
        old_payloads=list(OLD_PAYLOADS), archive_completeness_verified=False,
        runtime_equivalence_verified=False, trajectories_opened=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'parent-summary', 'reference-summary'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    parent = json.loads(a.parent_summary.read_text())
    assert (parent['source'], parent['status']) == (
        '1828d9cd02', 'v246_identification_and_persistent_state_protocol_sealed')
    reference = json.loads(a.reference_summary.read_text())
    assert (reference['source'], reference['status']) == ('c6710744f9', 'v242_passive_pair_passed')
    assert reference['checked_paths'] == reference['started_paths'] == 2
    assert not parent['selector_defined'] and not parent['runtime_connected']
    audit = source_audit(Path(__file__).resolve().parent)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v247_prepare_timing.py')
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    spec = registration(a.source)
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    status = dict(status='v247_prepare_mapping_and_minimal_budget_sealed', source=a.source,
        tests_passed=result.testsRun, source_interfaces=audit, current_budget=spec['current_budget'],
        reserved_preflight=spec['reserved_preflight'], runtime_connected=False,
        collector_implemented=False, selector_defined=False, collection_authorized=False,
        main_policy_changed=False, v238_changed=False, next_step=spec['next_step'])
    (a.out_root/'summary').mkdir()
    (a.out_root/'summary/status.json').write_text(json.dumps(status, indent=2)+'\n')
    print('V247_PASSIVE_MAPPING_SEALED '+json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
