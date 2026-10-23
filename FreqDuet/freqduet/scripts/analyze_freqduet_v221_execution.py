#!/usr/bin/env python3
"""Read-only execution and holding accounting for the completed V221 smoke."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys


ARMS = ('v107', 'upper', 'lower', 'joint')
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
           'public_launch_delay_total_s', 'holding_s_per_trip', 'waiting_person_s', 'onboard_person_s')
PAIRS = (('upper', 'v107'), ('lower', 'v107'), ('joint', 'v107'), ('joint', 'lower'), ('joint', 'upper'))


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as f:
        return json.load(f)


def differences(candidate, reference):
    result = {m: candidate[m]-reference[m] for m in METRICS}
    result['journey_s'] = 60.*result['journey_min']
    return result


def holding_account(data, reference):
    own = {r['trip_id']: r['holding_s'] for r in data['trace']['vehicle_returns']}
    old = {r['trip_id']: r['holding_s'] for r in reference['trace']['vehicle_returns']}
    assert own.keys() == old.keys(), 'do not treat missing service as zero holding'
    assert abs(sum(own.values())-data['ledger']['executed_holding_s']) <= 1e-6
    assert abs(sum(old.values())-reference['ledger']['executed_holding_s']) <= 1e-6
    departures = {(r['trip_id'], r['station_id'], r['direction']): r for r in data['trace']['departures']}
    changed = [r for r in data['responses'] if r['selected_bin'] != r['base_bin']]
    for r in changed:
        key = r['trip_id'], r['station_id'], r['direction']
        assert key in departures, 'changed request has no completed departure'
        assert departures[key]['realized_holding']-r['executed_before_s'] == ACTIONS[r['selected_bin']]
    tasks = {t['trip_id'] for t in data['tasks']}
    assert tasks <= own.keys()
    task_delta = sum(own[t]-old[t] for t in tasks)
    other_delta = sum(own[t]-old[t] for t in own if t not in tasks)
    direct = sum(ACTIONS[r['base_bin']]-ACTIONS[r['selected_bin']] for r in changed)
    return dict(changed_and_executed_requests=len(changed), direct_decrement_vs_live_bilateral_s=direct,
        task_trip_holding_delta_s=task_delta, other_trip_holding_delta_s=other_delta,
        total_holding_delta_s=task_delta+other_delta,
        task_remaining_response_s=task_delta+direct,
        accounting_residual_vs_direct_decrement_s=task_delta+other_delta+direct)


def release_account(data, reference):
    queries = {c['query']['event_seq']: c for c in data['cycles']}
    old = {r['trip']: r for r in reference['trace']['actual_launches']}
    new = {r['trip']: r for r in data['trace']['actual_launches']}
    rows = []
    for task in data['tasks']:
        trip, bus = task['trip_id'], task['reserved_bus_id']
        cycle = queries[task['query_event_seq']]
        q = cycle['query']
        ready = next((b for b in q['ready'] if b['bus_id'] == bus), None)
        assert ready is not None and ready['direction'] == q['direction']
        assert not ready['return_observed'] or ready['last_observed_return_s'] <= q['time_s']
        event = next(e for e in data['events'] if e['kind'] == 'launch' and e['bus_id'] == bus
                     and e['event_seq'] > q['event_seq'])
        current, control = new[trip], old[trip]
        plan = data['launch_plans'][str(trip)]
        expected = task['proposed_s'] if cycle['applied'] else task['native_s']
        rows.append(dict(trip_id=trip, reserved_bus_id=bus, query_s=q['time_s'],
            first_use_matches=event['trip_id'] == trip, actual_bus_matches=current['bus'] == bus,
            reference_bus_matches=control['bus'] == bus, executed_plan_matches=plan == expected,
            native_plan_s=task['native_s'], requested_plan_s=task['proposed_s'], executed_plan_s=plan,
            reference_actual_s=control['actual_s'], actual_s=current['actual_s'],
            actual_advance_s=control['actual_s']-current['actual_s'],
            public_early_s=max(0., current['public_baseline_s']-current['actual_s'])))
    return dict(tasks=len(rows), ready_at_query=len(rows),
        reserved_first_use_matches=sum(r['first_use_matches'] for r in rows),
        actual_bus_matches=sum(r['actual_bus_matches'] for r in rows),
        reference_bus_matches=sum(r['reference_bus_matches'] for r in rows),
        executed_plan_matches=sum(r['executed_plan_matches'] for r in rows),
        actual_advance_histogram_s=dict(Counter(str(r['actual_advance_s']) for r in rows)),
        public_early_total_s=sum(r['public_early_s'] for r in rows), first_task=rows[0] if rows else None)


def audit(root):
    sealed, final = read(root/'protocol.json'), read(root/'summary/status.json')
    assert final['status'] == 'v221_four_arm_execution_complete' and final['started_paths'] == 16
    assert final['authority_gate_passed'] and all(r['passed'] for r in final['authority'])
    assert final['training_updates'] == 0 and not final['main_policy_changed']
    contexts, all_deltas = [], {f'{a}_vs_{b}': [] for a, b in PAIRS}
    for ci, (actor, day) in enumerate(sealed['contexts']):
        for arm in ARMS:
            progress = read(root/'progress'/f'c{ci}_{arm}.json')
            assert progress['completed'] and progress['started']
            if arm == 'v107':
                assert progress['archived_exact'], 'reference exactness was not established'
        data = {a: read(root/'records'/f'c{ci}_{a}.json.gz') for a in ARMS}
        for a, d in data.items():
            assert d['context'] == ci and d['arm'] == a
            assert (d['results']['train_seed'], d['results']['scenario_seed']) == (actor, day)
            assert d['physical_contract'] == data['v107']['physical_contract']
            assert d['results']['trips_unlaunched'] == d['results']['trips_incomplete'] == 0
            for m in METRICS:
                assert d['results'][m] == final['contexts'][ci]['outcomes'][a][m]
        deltas = {f'{a}_vs_{b}': differences(data[a]['results'], data[b]['results']) for a, b in PAIRS}
        for pair, delta in deltas.items():
            all_deltas[pair].append(delta)
        execution = {a: release_account(d, data['v107']) for a, d in data.items()}
        holding = {a: holding_account(data[a], data['v107']) for a in ('lower', 'joint')}
        failures = {}
        for base in ('v107', 'upper', 'lower'):
            diff = deltas[f'joint_vs_{base}']
            failed = ['passenger_person_s'] if diff['passenger_person_s'] >= 0 else []
            failed.extend(m for m in ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s') if diff[m] > 0)
            failures[base] = failed
        contexts.append(dict(context=ci, actor=actor, day=day,
            authority_passed=final['authority'][ci]['passed'], releases=execution, holding=holding,
            live_task_ids_equal=all({t['trip_id'] for t in d['tasks']} ==
                                  {t['trip_id'] for t in data['v107']['tasks']} for d in data.values()),
            deltas=deltas, joint_outcome_failures_vs=failures,
            descriptive_joint_gate=final['contexts'][ci]['descriptive_joint_gate']))
    means = {pair: {m: mean(d[m] for d in rows) for m in rows[0]} for pair, rows in all_deltas.items()}
    return dict(status='v221_read_only_execution_audit_complete',
        analysis_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], controller_source=sealed['source'],
        references_exact=4, completed_paths=16, contexts=contexts, mean_deltas=means,
        authority_gate_passed=True, descriptive_joint_gate=final['descriptive_joint_gate'],
        new_replays=0, training_updates=0, main_policy_changed=False,
        limitations='Four consumed contexts,two actors and two demand days;no independent efficacy or CI. '
            'The first-query release has matched state/ready prefix;later closed-loop differences do not. '
            'Holding residual is matched-trip accounting,not a causal separation of feedback from changed states. '
            'Original-public early releases are explicitly reported. No V216 raw trace was reread.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'keep the original audit result'
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v221_execution_audit.py', '-v'], check=True)
    result = audit(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('V221_READ_ONLY_AUDIT_COMPLETE', json.dumps(result['mean_deltas']), flush=True)


if __name__ == '__main__':
    main()
