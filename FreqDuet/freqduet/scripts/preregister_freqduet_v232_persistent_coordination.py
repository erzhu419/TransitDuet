#!/usr/bin/env python3
"""Seal persistent-cycle state semantics; synthetic reference model only."""

import argparse
import ast
import copy
import json
import math
from pathlib import Path
import unittest


def resource_profile(row, metadata):
    """Current ready coverage plus position-only association, never a dispatch."""
    dt, now = metadata['native_dt_s'], row['snapshot_read_s']
    tick = (math.floor(now/dt)+1)*dt
    rank = {b: i for i, b in enumerate(metadata['native_inventory_order'])}
    result = dict(next_dispatch_s=tick, directions={},
                  association_semantics='Position-only hypothesis;not assignment,ETA,reservation or slack.')
    for direction in (False, True):
        ready = [b for b in row['ready'] if b['direction'] == direction and
                 (not b['return_observed'] or b['last_observed_return_s']+dt <= tick)]
        ready.sort(key=lambda b: (b['last_observed_return_s'] if b['return_observed'] else -math.inf,
                                  rank[b['bus_id']]))
        plans = [p for p in row['plans'] if p['direction'] == direction]
        due = [p for p in plans if math.ceil(p['planned_s']/dt)*dt <= tick]
        future = [p for p in plans if math.ceil(p['planned_s']/dt)*dt > tick]
        covered, unbacked, consumed = [], [], 0
        for kind, obligations in (('due', due), ('future', future)):
            for p in obligations:
                if consumed < len(ready):
                    covered.append(dict(trip_id=p['trip_id'], bus_id=ready[consumed]['bus_id'], kind=kind))
                    consumed += 1
                else:
                    unbacked.append(dict(p, kind=kind,
                        last_write_version=row['pending_plan_version_refs'][str(p['trip_id'])]))
        incoming = [b for b in row['active'] if b['direction'] != direction]
        incoming.sort(key=lambda b: (b['remaining_distance_m'], rank[b['bus_id']]))
        associations = []
        for i, b in enumerate(incoming):
            p = unbacked[i] if i < len(unbacked) else None
            associations.append(dict(bus_id=b['bus_id'], inbound_trip_id=b['trip_id'],
                distance_rank=i+1, position_sample_s=b['position_sample_s'],
                remaining_distance_m=b['remaining_distance_m'],
                obligation=None if p is None else dict(p),
                posted_lead_s=None if p is None else p['planned_s']-tick))
        result['directions'][str(direction).lower()] = dict(ready_coverage=covered,
            uncovered_due=[p['trip_id'] for p in unbacked if p['kind'] == 'due'],
            future_without_current_ready=[p['trip_id'] for p in unbacked if p['kind'] == 'future'],
            unbacked_obligations=unbacked, position_associations=associations)
    return result


class CycleState:
    """Unconnected reference transitions, exercised on synthetic inputs only."""
    def __init__(self):
        self.active, self.returned = {}, {}
        self.last_seq, self.last_time = -1, -math.inf

    def advance(self, seq, now):
        assert seq > self.last_seq and now >= self.last_time
        self.last_seq, self.last_time = seq, now

    def launch(self, bus, trip, direction, seq, now):
        self.advance(seq, now)
        assert bus not in self.active
        parent = self.returned.pop(bus, None)
        self.active[bus] = dict(cycle_key=[bus, trip, seq], direction=direction,
            launch_s=now, executed_holding_s=0., last_query_s=None, last_profile=None)
        return dict(cycle_key=[bus, trip, seq], observed_previous_cycle=copy.deepcopy(parent))

    def query(self, row, metadata):
        seq, now = row['snapshot_event_seq'], row['snapshot_read_s']
        self.advance(seq, now)
        bus, trip, launch_seq = row['cycle_ref']
        state = self.active[bus]
        assert state['cycle_key'] == [bus, trip, launch_seq] and launch_seq < seq
        own = next(b for b in row['active'] if b['bus_id'] == bus)
        assert (own['trip_id'], own['direction']) == (trip, state['direction'])
        assert all(b['position_sample_s'] == now for b in row['ready']+row['active'])
        assert all(v == 'initial' or v < seq for v in
                   (*row['message_refs'].values(), *row['pending_plan_version_refs'].values())), \
            'plan version is not a strictly completed prefix'
        executed = row['executed_holding_before_s']
        assert executed >= state['executed_holding_s'], 'holding prefix reset inside cycle'
        profile = resource_profile(row, metadata)
        target = str(not state['direction']).lower()
        association = next(b for b in profile['directions'][target]['position_associations'] if b['bus_id'] == bus)
        view = dict(cycle_key=list(state['cycle_key']), query_s=now, native_event_seq=seq,
            executed_holding_s=executed, delta_executed_since_last_sample_s=executed-state['executed_holding_s'],
            previous_query_s=state['last_query_s'], message_refs=dict(row['message_refs']),
            pending_plan_version_refs=dict(row['pending_plan_version_refs']),
            target_profile=copy.deepcopy(profile['directions'][target]), own_association=copy.deepcopy(association))
        state.update(executed_holding_s=executed, last_query_s=now, last_profile=copy.deepcopy(view))
        return view

    def returned_to_terminal(self, bus, trip, seq, now, executed_holding_s):
        self.advance(seq, now)
        state = self.active.pop(bus)
        assert state['cycle_key'][1] == trip and executed_holding_s >= state['executed_holding_s']
        row = dict(cycle_key=state['cycle_key'], return_s=now, return_event_seq=seq,
                   executed_holding_s=executed_holding_s)
        self.returned[bus] = copy.deepcopy(row)
        return row


def registration(source):
    return dict(stage='v232_persistent_physical_cycle_coordination_preregistration', source=source,
        parent_source='2dc70e4bf7', authorization='Seal protocol,synthetic transitions,source-interface audit only.',
        budget=dict(archive_scans=0, passive_paths=0, intervention_paths=0, shadow_requests=0,
                    training_updates=0, model_fits=0, new_futures=0),
        frozen=dict(base='V107 bilateral025', controller_source='2b0115af7b', capture_source='eec7a20c71',
            fleet=12, holding_bins_s=[0, 5, 10, 15, 20, 30, 45],
            upper='V220 ready-backed native projection;one shared scalar,original bounds/EMA.35/'
                '900s cadence/anchors/rolling-zero-sum constraints,direction-local callbacks and caches. '
                'No early launch backed by an incoming bus,no additional terminal control or larger gain.',
            unchanged='Actors,critics,duals,all predictor/calibration/filter versions and main.'),
        hypothesis='Persistent coordination must represent the specific physical vehicle cycle '
            'and posted obligations,not apply a terminal-wide holding reduction. Position-only '
            'association is a falsifiable causal-input hypothesis,not a known future supply identity.',
        inputs=['native query clock and event order', 'bus/trip/observed-launch cycle identity',
            'current published pending plans with direction,public/planned clocks and last-write version',
            'current ready inventory with observed return clock and original FIFO tie order',
            'all incoming vehicle directions,trips,remaining distances and sample clocks',
            'executed holding prefix in this cycle'],
        input_boundary='No future supply/follower/return/launch identity,time,label,hidden demand/road '
            'draw or unfinished prefix. No ETA,load-based fitted score or new forecast. Native lower '
            'F/B and old prepare clocks remain with the unchanged baseline,not new coordination inputs.',
        obligations='At every native query,next dispatch tick=floor(t/dt)*dt+dt. Allocate eligible '
            'current ready buses once each by original return FIFO/inventory ties to currently due '
            'plans in native pending order. Allocate remaining current ready stock once to future '
            'plans in that order,as snapshot backing only. Separate uncovered due from future '
            'without current ready;the latter alone is NOT a shortage. No vehicle reuse,forecast '
            'return,reservation or schedule-idle slack is created.',
        association='For each destination,order compatible incoming buses by current remaining '
            'distance with original inventory ties. Pair ordinally with unbacked obligations:due '
            'in native order first,then future in native order. Keep excess buses and obligations '
            'explicitly unmatched. Store ordinal,current position,plan ID/version,kind and posted '
            'lead time;lead time is not slack or ETA. Recompute at EVERY query. No assignment '
            'protection or debt survives a changed association;it never funds an upper advance.',
        cycle='Key=(bus,trip,actual observed launch event). Launch opens H=0. Every query copies '
            'current executed H and nonnegative delta since the previous sample,plus current plan '
            'versions and recomputed resource association. Replans,cache reuse,pressure relief, '
            'association changes and station transitions NEVER reset H or mint credit. Delta H '
            'between samples is not attributed to the current obligation:intervening plans/risk '
            'may differ. An observed return closes H at its actual account. Only the next ACTUAL '
            'same-bus launch may create a separate observed parent link and a new H=0 cycle. '
            'Never backfill prior queries;retain open and returned-without-next-launch states.',
        slow_fast='Latest completed direction callback and actual last write per pending trip '
            'govern query state,including cached callbacks. Slow messages do not allocate an '
            'unobserved future bus. Fast-cycle holding remains physical history across messages. '
            'No out-of-cadence upper replan or invented lower permission.',
        future_action_contract='A later separately sealed candidate may retain,increase or decrease '
            'one of the seven currently feasible native bins,not uniformly attenuate or force a '
            'one-bin cut. Compare native and candidate at the SAME state with native later feedback. '
            'Local R/onboard exposure are diagnostics only,not a strict R-nonharm reduction gate '
            'or a fitted weighted score. No selector,threshold,weight or action is evaluated now.',
        evidence_gate='Before proposing an action rule,check this frozen state/association on '
            'consumed development archives under separate read-only authorization. Report all '
            'queries,candidate failures,rank/association changes and mismatches to first actual '
            'same-bus supply;labels never select successful anchors. Prove any later passive '
            'integration preserves original actions,timing,endpoints and old sidecars exactly.',
        efficacy_gate='A separately budgeted later matched-future test needs V107,upper-only,'
            'new-lower-only and joint under original permissions,with memoryless-state ablation '
            'to identify persistence. Freeze choices before independent evaluation futures. '
            'Joint passenger95%CI upper<0 and actual CV2/readiness/original-public total delay/'
            'incomplete/unlaunched CI uppers<=0 versus V107 AND each single-layer control. '
            'Keep strong fixed+bilateral as external control. Use actor/day crossed paired '
            'statistics;no posthoc context replacement,relaxed margins or query-level pseudo-replication.',
        limitations='V231 is two consumed actors by two days,not independent efficacy. Distance '
            'order does not determine return order or dispatch identity. Extra future obligations '
            'lack current ready backing without proving a shortage. Tight observed cycles do '
            'not establish removable holding;action-difference evidence is still absent.',
        next_step='Seek separate authorization for a read-only audit of this state/association on '
            'the eight saved V230 paths;no new capture,replay,training or action admitted.',
        runtime_connected=False, selector_defined=False, archive_analysis_authorized=False,
        replay_authorized=False, training_authorized=False, main_policy_changed=False)


def source_audit(capture_path, contract_path):
    contract = ast.parse(contract_path.read_text())
    payload = next(ast.literal_eval(n.value) for n in contract.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD_FIELDS' for t in n.targets))
    required = {'snapshot_read_s', 'snapshot_event_seq', 'ready', 'active', 'plans', 'executed_holding_before_s'}
    assert required <= set(payload), 'V230 query payload lacks required causal state'
    tree = ast.parse(capture_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResourceCapture')
    query = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'query')
    updates = {k.arg for n in ast.walk(query) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'update' for k in n.keywords}
    assert {'cycle_ref', 'message_refs', 'pending_plan_version_refs'} <= updates
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {'event', 'initialize', 'data'} <= methods
    return dict(required_query_payload=sorted(required), cycle_version_fields_available=True,
                scope='AST only;no capture/runtime imports or trajectory reads.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v231-root', 'capture-source', 'contract-source'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    spec = registration(a.source)
    path = a.out_root/'protocol.json'
    # A completed seal is reused if the scheduler repeats the same launch.
    if path.exists():
        assert json.loads(path.read_text()) == spec
        prior = json.loads((a.out_root/'summary/status.json').read_text())
        assert prior['source'] == a.source and prior['status'] == 'v232_persistent_coordination_protocol_sealed'
        print('V232_EXISTING_SEAL_REUSED_NO_NEW_WORK', flush=True)
        return
    parent = json.loads((a.v231_root/'summary/status.json').read_text())
    assert parent['status'] == 'v231_supply_overlap_complete' and parent['source'] == '2dc70e4bf7'
    assert len(parent['contexts']) == 4 and parent['paths'] == 8
    assert parent['new_replays'] == parent['training_updates'] == parent['shadow_actions_executed'] == 0
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v232_persistent_coordination.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    audit = source_audit(a.capture_source, a.contract_source)
    a.out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v232_persistent_coordination_protocol_sealed', source=a.source,
        parent_source=parent['source'], tests=tests.testsRun, budget=spec['budget'], source_interfaces=audit,
        runtime_connected=False, selector_defined=False, archive_analysis_authorized=False,
        replay_authorized=False, training_authorized=False, main_policy_changed=False,
        protocol_path=str(path), next_step=spec['next_step'])
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V232_PERSISTENT_PROTOCOL_SEALED_NO_RUNTIME '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
