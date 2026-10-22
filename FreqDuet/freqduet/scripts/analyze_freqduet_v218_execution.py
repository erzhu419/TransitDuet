#!/usr/bin/env python3
"""Read-only four-arm audit of the frozen V218 engineering paths."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys

from freqduet_v218_cycle_policy import ACTIONS, ARMS

METRICS = ('journey_min', 'cv_squared', 'readiness_delay_s',
           'public_launch_delay_total_s', 'holding_s_per_trip',
           'waiting_person_s', 'onboard_person_s', 'passenger_person_s')


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def close(a, b):
    assert abs(a-b) < 1e-6, (a, b)


def holding_by_stop(departures):
    previous, result = {}, {}
    for row in sorted(departures, key=lambda r: r['time_s']):
        trip = row['trip_id']
        value = row['realized_holding']-previous.get(trip, 0.)
        assert value >= 0
        result[trip, row['station_id'], row['direction']] = value
        previous[trip] = row['realized_holding']
    return result


def index(data):
    trace = data['trace']
    launches = {r['trip']: r for r in trace['actual_launches']}
    returns = {(r['bus_id'], r['trip_id']): r for r in trace['vehicle_returns']}
    next_trip, pending = {}, {}
    for event in sorted(data['events'], key=lambda e: e['event_seq']):
        if event['kind'] == 'return':
            pending[event['bus_id']] = event['trip_id']
        elif event['kind'] == 'launch' and event['bus_id'] in pending:
            next_trip[event['bus_id'], pending.pop(event['bus_id'])] = event['trip_id']
    stops = holding_by_stop(trace['departures'])
    close(sum(stops.values()), data['ledger']['executed_holding_s'])
    return dict(launches=launches, returns=returns, next_trip=next_trip, stops=stops,
                departures={(r['trip_id'], r['station_id'], r['direction']): r
                            for r in trace['departures']})


def chain(data, idx, bus, inbound):
    returned = idx['returns'].get((bus, inbound))
    outbound = idx['next_trip'].get((bus, inbound))
    if returned is None or outbound is None:
        return None
    incoming, launched = idx['launches'][inbound], idx['launches'][outbound]
    assert incoming['bus'] == launched['bus'] == bus
    close(returned['return_s']-returned['launch_s'], returned['holding_s']+returned['other_elapsed_s'])
    close(returned['launch_s'], incoming['actual_s'])
    return dict(bus=bus, inbound=inbound, outbound=outbound,
                direction=launched['direction'], inbound_launch_s=incoming['actual_s'],
                inbound_holding_s=returned['holding_s'], other_elapsed_s=returned['other_elapsed_s'],
                return_s=returned['return_s'], next_launch_s=launched['actual_s'],
                return_to_launch_s=launched['actual_s']-returned['return_s'],
                next_plan_s=data['launch_plans'][str(outbound)])


def chain_delta(candidate, control):
    if control is None:
        return None
    if any(candidate[k] != control[k] for k in ('bus', 'inbound', 'outbound', 'direction')):
        return None
    keys = ('inbound_launch_s', 'inbound_holding_s', 'other_elapsed_s', 'return_s',
            'next_launch_s', 'return_to_launch_s', 'next_plan_s')
    delta = {k: candidate[k]-control[k] for k in keys}
    close(delta['return_s'], delta['inbound_launch_s']+delta['inbound_holding_s']+delta['other_elapsed_s'])
    close(delta['next_launch_s'], delta['return_s']+delta['return_to_launch_s'])
    return delta


def change_pattern(delta):
    plan, actual = delta['next_plan_s'] != 0, delta['next_launch_s'] != 0
    return 'both' if plan and actual else 'plan_only' if plan else 'actual_only' if actual else 'neither'


def binding_evidence(candidate, control, delta, tick):
    return dict(plan_delta_over_roundoff=abs(delta['next_plan_s']) > 1e-6,
                material_plan_and_actual_change=abs(delta['next_plan_s']) > 1e-6 and delta['next_launch_s'] != 0,
                both_plans_before_return=candidate['next_plan_s'] <= candidate['return_s']
                    and control['next_plan_s'] <= control['return_s'],
                both_launch_at_return_plus_tick=candidate['return_to_launch_s'] == tick
                    and control['return_to_launch_s'] == tick)


def signs(values):
    return dict(decreased=sum(v < 0 for v in values), unchanged=sum(v == 0 for v in values),
                increased=sum(v > 0 for v in values))


def means(rows):
    return {k: fmean(r[k] for r in rows) for k in rows[0]}


def audit_context(root, ci, original):
    arms = {a: read(root/'records'/f'c{ci}_{a}.json.gz') for a in ARMS}
    indexes = {a: index(d) for a, d in arms.items()}
    for a, d in arms.items():
        marker = json.loads((root/'progress'/f'c{ci}_{a}.json').read_text())
        assert marker['completed'] and (a != 'v107' or marker['archived_exact'])
        assert d['physical_contract'] == arms['v107']['physical_contract']
        assert d['ledger']['physical_vehicles'] == 12
        assert d['results']['trips_unlaunched'] == d['results']['trips_incomplete'] == 0
        assert d['results']['training_updates'] == d['results']['dual_updates'] == 0
        assert d['results']['base_actors_unchanged'] and not d['main_policy_changed']
        close(d['ledger']['executed_holding_s'], d['ledger']['commanded_holding_s'])
        for metric in METRICS:
            close(d['results'][metric], original['outcomes'][a][metric])
    joint, ji = arms['joint'], indexes['joint']
    changed = [q for q in joint['responses'] if q['base_bin'] != q['selected_bin']]
    grouped = defaultdict(list)
    reproduced_coupled, shadow_reduction = 0, 0.
    for q in changed:
        key = (q['trip_id'], q['station_id'], q['direction'])
        dep = ji['departures'][key]
        close(dep['realized_holding']-q['executed_before_s'], ACTIONS[q['selected_bin']])
        close(ji['stops'][key], ACTIONS[q['selected_bin']])
        shadow_reduction += ACTIONS[q['base_bin']]-ACTIONS[q['selected_bin']]
        c = chain(joint, ji, q['bus_id'], q['trip_id'])
        assert c is not None and c['outbound'] in q['link']['due_trip_ids']
        assert c['direction'] == q['link']['target_direction']
        grouped[q['bus_id'], q['trip_id']].append(q)
        delta = chain_delta(c, chain(arms['lower'], indexes['lower'], q['bus_id'], q['trip_id']))
        reproduced_coupled += delta is not None and change_pattern(delta) == 'both'
    assert len(changed) == original['execution']['executed_responses']
    assert reproduced_coupled == original['execution']['same_chain_planned_and_actual_launch_changed_vs_lower_only']

    comparisons = {}
    for a in ARMS[:-1]:
        rows, unmatched, stop_deltas, examples, binding = [], 0, [], [], Counter()
        for (bus, inbound), queries in grouped.items():
            c = chain(joint, ji, bus, inbound)
            reference = chain(arms[a], indexes[a], bus, inbound)
            delta = chain_delta(c, reference)
            if delta is None:
                unmatched += 1
                continue
            rows.append(delta)
            binding.update({k: int(v) for k, v in binding_evidence(
                c, reference, delta, joint['physical_contract']['dt_s']).items()})
            for q in queries:
                key = (q['trip_id'], q['station_id'], q['direction'])
                stop_deltas.append(ji['stops'][key]-indexes[a]['stops'][key])
            if len(examples) < 2 and change_pattern(delta) == 'both':
                examples.append(dict(bus=bus, inbound=inbound, outbound=c['outbound'],
                                     query_count=len(queries), joint=c, control=reference, delta=delta))
        comparisons[a] = dict(matched_chains=len(rows), unsupported_assignment_or_missing=unmatched,
            planned_actual_patterns=dict(Counter(change_pattern(d) for d in rows)),
            binding_evidence=dict(binding),
            chain_delta_sum={k: sum(d[k] for d in rows) for k in rows[0]} if rows else {},
            chain_delta_signs={k: signs([d[k] for d in rows]) for k in
                               ('inbound_holding_s', 'return_s', 'next_launch_s', 'return_to_launch_s')} if rows else {},
            whole_day_holding_delta_s=joint['ledger']['executed_holding_s']-arms[a]['ledger']['executed_holding_s'],
            other_inbound_trips_holding_delta_s=joint['ledger']['executed_holding_s']
                -arms[a]['ledger']['executed_holding_s']-sum(d['inbound_holding_s'] for d in rows),
            changed_query_actual_stop_holding_delta_sum_s=sum(stop_deltas),
            changed_query_actual_stop_holding_signs=signs(stop_deltas), examples=examples)

    outcomes = {a: {m: d['results'][m] for m in METRICS} for a, d in arms.items()}
    operations = {}
    for a, d in arms.items():
        la, delta_stop = indexes[a]['launches'], indexes[a]['stops']
        base_la, base_stop = indexes['v107']['launches'], indexes['v107']['stops']
        assert la.keys() == base_la.keys() and delta_stop.keys() == base_stop.keys()
        operations[a] = dict(executed_holding_s=d['ledger']['executed_holding_s'],
            applied_upper_shifts=dict(Counter(str(c['applied_shift_s']) for c in d['cycles'])),
            actual_launch_changes_vs_v107=signs([r['actual_s']-base_la[t]['actual_s'] for t, r in la.items()]),
            actual_holding_stop_changes_vs_v107=signs([v-base_stop[k] for k, v in delta_stop.items()]))
    return dict(context=ci, actor=original['seed'], day=original['day'], outcomes=outcomes,
        joint_minus={a: {m: outcomes['joint'][m]-outcomes[a][m] for m in METRICS} for a in ARMS[:-1]},
        upper_minus_v107={m: outcomes['upper'][m]-outcomes['v107'][m] for m in METRICS},
        lower_minus_v107={m: outcomes['lower'][m]-outcomes['v107'][m] for m in METRICS},
        interaction={m: outcomes['joint'][m]-outcomes['upper'][m]-outcomes['lower'][m]+outcomes['v107'][m]
                     for m in METRICS}, operations=operations,
        changed_requests=len(changed), unique_target_chains=len(grouped),
        requested_reduction_vs_own_shadow_s=shadow_reduction, chains=comparisons)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                    str(Path(__file__).resolve().parents[1]/'tests'),
                    '-p', 'test_v218_execution_audit.py', '-v'], check=True)
    old = json.loads((args.root/'summary/status.json').read_text())
    assert old['paths'] == 16 and old['all_baseline_trajectories_archived_exact']
    assert old['execution_gate_passed'] and not old['independent_futures_opened']
    contexts = [audit_context(args.root, c['context'], c) for c in old['contexts']]
    summary = dict(status='v218_read_only_execution_and_outcome_audit_complete',
        archive_equivalence='all four completed original exact comparisons retained; no archive rescan',
        execution_gate_reproduced=True, contexts=contexts,
        arm_means={a: means([c['outcomes'][a] for c in contexts]) for a in ARMS},
        joint_minus_mean={a: means([c['joint_minus'][a] for c in contexts]) for a in ARMS[:-1]},
        upper_minus_v107_mean=means([c['upper_minus_v107'] for c in contexts]),
        lower_minus_v107_mean=means([c['lower_minus_v107'] for c in contexts]),
        interaction_mean=means([c['interaction'] for c in contexts]),
        grouped_descriptive={label: {str(key): {a: means([c['joint_minus'][a] for c in contexts if c[label] == key])
            for a in ARMS[:-1]} for key in sorted({c[label] for c in contexts})} for label in ('actor', 'day')},
        limits=['two consumed actors x two consumed days; descriptive, not independent confirmation',
                'query counts are not independent chains or replications',
                'chain holding deltas include closed-loop propagation, not a local causal decomposition',
                'own-shadow reduction is not the realized between-arm holding reduction'],
        new_replays=0, training_updates=0, main_policy_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(summary['status'], json.dumps(summary['joint_minus_mean']), flush=True)


if __name__ == '__main__':
    main()
