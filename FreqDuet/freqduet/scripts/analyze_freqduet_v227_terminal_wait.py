#!/usr/bin/env python3
"""Read-only clock and terminal-inventory accounting on the frozen V225 paths."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import unittest

import yaml

from analyze_freqduet_v221_execution import read
from analyze_freqduet_v226_coordination import PAIRS, close, supply


WAIT_FIELDS = ('post_eligible_plan_block_s', 'post_eligible_no_ready_s',
               'post_eligible_open_ready_residual_s')
IDLE_FIELDS = ('return_after_dispatch_tick_s', 'idle_plan_block_s', 'idle_plan_open_s')


def frozen_floor(path):
    chain, floors = [], []
    while path is not None:
        cfg = yaml.safe_load(path.read_text())
        chain.append(str(path))
        floors.append(cfg.get('upper', {}).get('timetable_planner', {}).get('terminal_headway_floor', {}))
        parent = cfg.get('_extends')
        if parent:
            # The actual chain ends at config_v2.yaml one directory above configs.
            path = next(p for p in (path.parent/parent, path.parent.parent/parent) if p.exists())
        else:
            path = None
    floor = {}
    for item in reversed(floors):
        floor.update(item)
    assert not floor.get('enable', False), 'active headway floor needs its own archived per-trip labels'
    return dict(chain=chain, resolved=floor)


def plans(data):
    first = data['fresh_upper'][0]
    initial = first['plans']
    rank = {r['trip_id']: i for i, r in enumerate(initial)}
    assert len(rank) == len(initial)
    histories = {r['trip_id']: [((first['time_s'], -1), r['planned_s'])] for r in initial}
    updates = [e for e in data['events'] if e['kind'] == 'plan_update']
    callbacks = data['trace']['upper_callbacks']
    assert len(updates) == len(callbacks), 'plan-update occurrence join failed'
    for event, call in zip(updates, callbacks):
        assert (event['trip_id'], event['time_s']) == (call['trip_id'], call['time_s'])
        point = call['time_s'], rank[call['trip_id']]
        for change in call['changes']:
            history = histories[change['trip_id']]
            assert history[-1][1] == change['before']['planned_s'], 'missing intermediate plan update'
            assert point >= history[-1][0]
            history.append((point, change['after']['planned_s']))
    keys = {trip: [row[0] for row in h] for trip, h in histories.items()}
    return dict(rank=rank, histories=histories, keys=keys, first=first)


def plan_at(history, trip, now):
    point = now, history['rank'][trip]
    i = bisect_right(history['keys'][trip], point)-1
    assert i >= 0, 'plan observation does not cover this dispatch clock'
    return history['histories'][trip][i][1]


def terminal_periods(data, history):
    first, rank = history['first'], history['rank']
    assert not first['active'] and len(first['ready']) == data['physical_contract']['physical_vehicle_count']
    dt = data['physical_contract']['dt_s']
    order = {q['bus_id']: i for i, q in enumerate(first['ready'])}
    open_periods = {q['bus_id']: dict(bus_id=q['bus_id'], direction=q['direction'],
        start_s=first['time_s'], returned_s=None, parent_trip=None) for q in first['ready']}
    returns = {r['trip_id']: r for r in data['trace']['vehicle_returns']}
    launched = {r['trip']: r for r in data['trace']['actual_launches']}
    periods, incoming = [], {}
    for event in data['events']:
        if event['kind'] == 'return':
            r = returns[event['trip_id']]
            bus = r['bus_id']
            assert bus == event['bus_id'] and bus not in open_periods
            assert r['return_s'] == event['time_s']
            open_periods[bus] = dict(bus_id=bus, direction=r['target_direction'],
                returned_s=r['return_s'], start_s=r['return_s']+dt, parent_trip=r['trip_id'])
        elif event['kind'] == 'launch':
            row = launched[event['trip_id']]
            period = open_periods.pop(row['bus'])
            assert row['bus'] == event['bus_id'] and row['direction'] == period['direction']
            assert row['actual_s'] >= period['start_s'], 'return used before the next dispatch pass'
            period.update(end_s=row['actual_s'], end_rank=rank[row['trip']], next_trip=row['trip'])
            periods.append(period)
            incoming[row['trip']] = period
    periods.extend(dict(r, end_s=math.inf, end_rank=0, next_trip=None) for r in open_periods.values())
    return dict(periods=periods, incoming=incoming, order=order)


def ready_at(terminal, now, rank, direction=None):
    point = now, rank
    return [r for r in terminal['periods'] if (r['start_s'], -1) <= point <= (r['end_s'], r['end_rank'])
            and (direction is None or bool(r['direction']) == bool(direction))]


def tick_range(start, end, dt):
    n = round((end-start)/dt)
    assert n >= 0 and n*dt == end-start, 'non-grid observation clock'
    return (start+i*dt for i in range(n))


def audit_path(data):
    history = plans(data)
    terminal = terminal_periods(data, history)
    dt = data['physical_contract']['dt_s']
    launches = data['trace']['actual_launches']
    rank = history['rank']
    assert set(rank) == {r['trip'] for r in launches}
    for query in data['fresh_upper']:
        observed = {(r['bus_id'], bool(r['direction'])) for r in query['ready']}
        reconstructed = {(r['bus_id'], bool(r['direction'])) for r in
                         ready_at(terminal, query['time_s'], rank[query['trip_id']])}
        assert reconstructed == observed, 'query-time terminal inventory mismatch'
    totals, rows = Counter(), {}
    for launch in launches:
        trip, now, direction = launch['trip'], launch['actual_s'], launch['direction']
        eligible, public = launch['eligible_s'], launch['public_baseline_s']
        period = terminal['incoming'][trip]
        waiting = dict.fromkeys(WAIT_FIELDS, 0.)
        overlap = 0.
        for t in tick_range(eligible, now, dt):
            available = bool(ready_at(terminal, t, rank[trip], direction))
            if plan_at(history, trip, t) > t:
                waiting[WAIT_FIELDS[0]] += dt
                overlap += dt if not available else 0.
            elif not available:
                waiting[WAIT_FIELDS[1]] += dt
            else:
                waiting[WAIT_FIELDS[2]] += dt
        close(sum(waiting.values()), now-eligible, 'post-eligibility partition')
        ready = ready_at(terminal, now, rank[trip], direction)
        fifo = min(ready, key=lambda r: (r['returned_s'] if r['returned_s'] is not None
                                        else -math.inf, terminal['order'][r['bus_id']]))
        assert fifo['bus_id'] == launch['bus'], 'native FIFO selection does not reproduce'
        plan = plan_at(history, trip, now)
        assert plan == data['launch_plans'][str(trip)] and plan <= now, 'launch-time plan mismatch'
        plan_tick = math.ceil(plan/dt)*dt
        fields = dict(waiting, post_eligible_joint_plan_supply_block_s=overlap,
            readiness_s=now-eligible, public_delay_s=max(0., now-public),
            public_early_s=max(0., public-now), posted_plan_delta_s=plan-public,
            plan_rounding_s=plan_tick-plan, final_plan_to_actual_s=now-plan_tick)
        close(fields['public_delay_s'], fields['posted_plan_delta_s']+fields['plan_rounding_s']
              +fields['final_plan_to_actual_s']+fields['public_early_s'], 'public delay partition')
        idle = None
        if period['parent_trip'] is not None:
            idle = dict.fromkeys(IDLE_FIELDS, 0.)
            idle[IDLE_FIELDS[0]] = period['start_s']-period['returned_s']
            for t in tick_range(period['start_s'], now, dt):
                idle[IDLE_FIELDS[1] if plan_at(history, trip, t) > t else IDLE_FIELDS[2]] += dt
            close(sum(idle.values()), now-period['returned_s'], 'terminal idle partition')
            fields.update(idle, terminal_idle_s=now-period['returned_s'])
        for name, value in fields.items():
            totals[name] += value
        rows[trip] = dict(trip_id=trip, bus_id=launch['bus'], direction=direction,
            parent_trip=period['parent_trip'], return_s=period['returned_s'],
            available_s=period['start_s'], eligible_s=eligible, final_plan_s=plan,
            final_plan_tick_s=plan_tick, actual_s=now, public_s=public, **fields)
    close(totals['readiness_s'], data['results']['readiness_delay_total_s'], 'readiness endpoint')
    close(totals['public_delay_s'], data['results']['public_launch_delay_total_s'], 'public delay endpoint')
    return dict(totals=dict(totals), rows=rows, query_snapshots_matched=len(data['fresh_upper']),
        launches=len(launches), incoming_chains=sum(r['parent_trip'] is not None for r in rows.values()),
        initial_stock=sum(r['parent_trip'] is None for r in rows.values()))


def compare(current, reference, previous):
    assert current['rows'].keys() == reference['rows'].keys()
    names = current['totals'].keys() | reference['totals'].keys()
    delta = {name: current['totals'].get(name, 0.)-reference['totals'].get(name, 0.) for name in names}
    close(delta['readiness_s'], previous['totals']['readiness_delta_s'], 'V226 readiness')
    close(delta['public_delay_s'], previous['totals']['public_delay_delta_s'], 'V226 public delay')
    # V226 closes only matched inbound chains; preserve any changed source separately.
    counts, idle_delta, clock_groups = Counter(), Counter(), {}
    details = []
    for trip, c in current['rows'].items():
        b = reference['rows'][trip]
        matched = c['bus_id'] == b['bus_id'] and c['parent_trip'] is not None and c['parent_trip'] == b['parent_trip']
        counts['same_inbound_chain'] += matched
        if matched:
            for name in (*IDLE_FIELDS, 'terminal_idle_s'):
                idle_delta[name] += c[name]-b[name]
            launch_delta, return_delta = c['actual_s']-b['actual_s'], c['return_s']-b['return_s']
            sign = lambda x: 'later' if x > 0 else 'earlier' if x < 0 else 'same'
            group = clock_groups.setdefault('launch_'+sign(launch_delta)+'/return_'+sign(return_delta),
                dict(chains=0, launch_delta_s=0., return_delta_s=0., idle_delta_s=0.))
            group['chains'] += 1
            group['launch_delta_s'] += launch_delta
            group['return_delta_s'] += return_delta
            group['idle_delta_s'] += c['terminal_idle_s']-b['terminal_idle_s']
            close(launch_delta-return_delta, c['terminal_idle_s']-b['terminal_idle_s'], 'idle clock identity')
        if any(c.get(k) != b.get(k) for k in ('actual_s', 'final_plan_s', 'return_s', *WAIT_FIELDS, *IDLE_FIELDS)):
            details.append(dict(trip_id=trip, current=c, reference=b, same_inbound_chain=matched))
    close(idle_delta['terminal_idle_s'], previous['totals']['same_chain_terminal_idle_delta_s'], 'V226 terminal idle')
    return dict(delta=delta, matched_idle_delta=dict(idle_delta), counts=dict(counts),
                launch_return_groups=clock_groups), details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-root', type=Path, required=True)
    p.add_argument('--controls-root', type=Path, required=True)
    p.add_argument('--v226-root', type=Path, required=True)
    p.add_argument('--config-file', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                             pattern='test_v227_terminal_wait.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    floor = frozen_floor(a.config_file)
    a.out_root.mkdir(parents=True, exist_ok=True)
    protocol = dict(stage='v227_read_only_terminal_wait', source=os.environ['FREQDUET_ANALYSIS_COMMIT'],
        controller_source='f61f6c9734', config_floor=floor,
        analysis='Observed dispatch-position plan history and FIFO terminal inventory. '
            'Return becomes usable only at the next native tick. Partition recorded readiness '
            'and actual same-bus idle; keep open/ready residual distinct from avoidable slack.',
        new_replays=0, training_updates=0, parameter_changes=0, main_policy_changed=False)
    assert not (a.out_root/'protocol.json').exists(), 'preserve the original V227 audit'
    (a.out_root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    previous = read(a.v226_root/'summary/status.json')
    assert previous['status'] == 'v226_read_only_coordination_complete'
    rows, start = [], time.monotonic()
    (a.out_root/'details').mkdir(exist_ok=True)
    for saved in previous['contexts']:
        ci = saved['context']
        paths = {arm: audit_path(read((a.joint_root if arm == 'joint' else a.controls_root)
                 /'records'/f'c{ci}_{arm}.json.gz')) for arm in ('v107', 'upper', 'joint')}
        comparisons, detail = {}, {}
        for c, b in PAIRS:
            key = c+'_vs_'+b
            comparisons[key], detail[key] = compare(paths[c], paths[b], saved['comparisons'][key]['supply_chain'])
        (a.out_root/'details'/f'c{ci}.json').write_text(json.dumps(detail, indent=2)+'\n')
        rows.append(dict(context=ci, actor=saved['actor'], demand_day=saved['demand_day'],
            paths={arm: {k:v for k,v in path.items() if k != 'rows'} for arm,path in paths.items()},
            comparisons=comparisons))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=4, unit='context',
            elapsed_seconds=time.monotonic()-start)), flush=True)
    summary = dict(protocol, status='v227_terminal_wait_complete', tests=tests.testsRun, contexts=rows,
        v226_totals_reproduced=True,
        limitations='Four consumed contexts;observed-path clock accounting,not counterfactual '
            'dispatch values. Final-plan gap is not pure vehicle shortage. Idle is not free '
            'slack. No alternate departure,passenger trajectory or policy is evaluated.')
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V227_TERMINAL_WAIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
