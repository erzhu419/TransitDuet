#!/usr/bin/env python3
"""Read-only terminal-wait, turnover and lower-response accounting for V130."""

import argparse
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import numpy as np

import audit_freqduet_v129_terminal_recovery as prior

PARTS = ('control_s', 'prior_launch_s', 'prior_holding_s', 'prior_other_s',
    'visibility_lag_s', 'reference_clip_s', 'unmatched_supply_s', 'residual_s')


def launch_state(row, reference):
    visible = reference if row['first_dispatch_visible_s'] is None else row['first_dispatch_visible_s']
    supply = max(0., visible-reference)
    control = max(0., row['eligible_s']-max(reference, visible))
    residual = row['actual_s']-reference-supply-control
    if residual < -1e-6:
        raise AssertionError('actual launch precedes the recorded command or vehicle availability')
    return dict(reference_s=reference, visible_s=visible, supply_s=supply, control_s=control,
        residual_s=residual, delay_s=row['actual_s']-reference)


def clipped_parts(before, after, changes):
    prior.close(sum(changes.values()), after-before, 'unclipped supply change')
    delta = max(0., after)-max(0., before)
    slope = delta/(after-before) if after != before else float(before > 0.)
    parts = {k: slope*v for k, v in changes.items()}
    prior.close(sum(parts.values()), delta, 'clipped supply allocation')
    return parts


def allocate_supply(b, c, changes):
    if changes is None:
        return dict(unmatched_supply_s=c['supply_s']-b['supply_s'])
    return clipped_parts(b['visible_s']-b['reference_s'], c['visible_s']-c['reference_s'], changes)


def stop_actions(arrivals, departures):
    key = lambda r: (int(r['trip_id']), int(r['station_id']), int(r['direction']))
    arrived, departed = [{key(r): r for r in rows} for rows in (arrivals, departures)]
    if arrived.keys() != departed.keys():
        raise AssertionError('completed stop arrivals/departures have different support')
    return {k: dict(arrival_s=r['time_s'], action_s=departed[k]['realized_holding']-r['realized_holding'],
        release_s=r['release'], incoming_holding_s=r['realized_holding'], incoming_other_s=r['other_elapsed'])
        for k, r in arrived.items()}


def read_actions(path):
    with np.load(path) as record:
        fields = record['event_fields'].tolist()
        events = [[dict(zip(fields, r.tolist())) for r in record[k]] for k in ('arrivals', 'departures')]
    return stop_actions(*events)


def mean_rows(rows, fields):
    return {k: fmean(r[k] for r in rows) for k in fields} if rows else None


def analyze(record, baseline, candidate_actions, base_actions, effect):
    clocks = {int(k): v for k, v in record['reference_clocks'].items()}
    old = {r['trip_id']: r for r in baseline['dispatches']}
    cr, br = [{r['trip_id']: r for r in path['returns']} for path in (record, baseline)]
    plans = {r['trip_id']: r for r in record['plans']}
    if candidate_actions.keys() != base_actions.keys():
        raise AssertionError('paired completed stop-action support differs')
    action_changes, held_totals = [], [{}, {}]
    for key, c in candidate_actions.items():
        b = base_actions[key]
        for totals, row in zip(held_totals, (c, b)):
            totals[key[0]] = totals.get(key[0], 0.)+row['action_s']
        if c['action_s'] != b['action_s']:
            action_changes.append(dict(trip_id=key[0], station_id=key[1], direction=key[2],
                base_arrival_s=b['arrival_s'], candidate_arrival_s=c['arrival_s'],
                base_action_s=b['action_s'], candidate_action_s=c['action_s'],
                action_delta_s=c['action_s']-b['action_s'], release_delta_s=c['release_s']-b['release_s'],
                incoming_holding_delta_s=c['incoming_holding_s']-b['incoming_holding_s'],
                incoming_other_delta_s=c['incoming_other_s']-b['incoming_other_s']))
    for totals, returns in zip(held_totals, (cr, br)):
        for trip, returned in returns.items():
            prior.close(totals.get(trip, 0.), returned['holding_s'], 'stop actions recover trip holding')
    rows = []
    for dispatch in record['dispatches']:
        trip = dispatch['trip_id']
        b, c = launch_state(old[trip], old[trip]['eligible_s']), launch_state(dispatch, clocks[trip])
        parts = dict.fromkeys(PARTS, 0.)
        parts.update(control_s=c['control_s']-b['control_s'], residual_s=c['residual_s']-b['residual_s'])
        matched = (dispatch['bus_id'], dispatch['prior_trip_id']) == (old[trip]['bus_id'], old[trip]['prior_trip_id'])
        changes = None
        if matched and dispatch['prior_trip_id'] >= 0:
            predecessor = dispatch['prior_trip_id']
            cb, bb = cr[predecessor], br[predecessor]
            changes = dict(prior_launch_s=cb['launch_s']-bb['launch_s'],
                prior_holding_s=cb['holding_s']-bb['holding_s'],
                prior_other_s=cb['nonholding_elapsed_s']-bb['nonholding_elapsed_s'],
                visibility_lag_s=(c['visible_s']-cb['return_s'])-(b['visible_s']-bb['return_s']),
                reference_clip_s=-(c['reference_s']-b['reference_s']))
        parts.update(allocate_supply(b, c, changes))
        delta = c['delay_s']-b['delay_s']
        prior.close(sum(parts.values()), delta, 'paired total launch delay')
        shift = dispatch['actual_s']-old[trip]['actual_s']
        prior.close(shift, delta+c['reference_s']-b['reference_s'], 'paired reference clocks')
        plan = plans[trip]
        returned_delta = cr[trip]['return_s']-br[trip]['return_s']
        holding_delta = cr[trip]['holding_s']-br[trip]['holding_s']
        other_delta = cr[trip]['nonholding_elapsed_s']-br[trip]['nonholding_elapsed_s']
        prior.close(returned_delta, shift+holding_delta+other_delta, 'current trip return')
        rows.append(dict(trip_id=trip, direction=dispatch['direction'], reference_s=clocks[trip],
            requested_s=plan['command_s']-plan['published_release_s'], ready_at_query=plan['current_ready_same'],
            delay_delta_s=delta, selected_supply_delta_s=c['supply_s']-b['supply_s'],
            actual_shift_s=shift, reference_shift_s=c['reference_s']-b['reference_s'],
            holding_delta_s=holding_delta, other_delta_s=other_delta, return_delta_s=returned_delta,
            source_matched=matched, parts=parts))
    fields = ('requested_s', 'delay_delta_s', 'selected_supply_delta_s', 'actual_shift_s',
        'reference_shift_s', 'holding_delta_s', 'other_delta_s', 'return_delta_s')
    means = mean_rows(rows, fields)
    contributions = {k: fmean(r['parts'][k] for r in rows) for k in PARTS}
    prior.close(means['delay_delta_s'], effect['inclusive_terminal_delay_s'], 'V130 delay endpoint')
    prior.close(means['holding_delta_s'], effect['holding_s_per_trip'], 'V130 holding endpoint')
    prior.close(sum(contributions.values()), means['delay_delta_s'], 'mean delay allocation')
    first = min(action_changes, key=lambda r: min(r['base_arrival_s'], r['candidate_arrival_s'])) if action_changes else None
    commands = sorted((r for r in record['plans'] if r['command_s'] > r['published_release_s']), key=lambda r: r['query_s'])
    detail = dict(dispatches=rows, changed_stop_actions=action_changes)
    summary = dict(effect=effect, means=means, contributions=contributions,
        remapped_vehicle_or_prior=sum(not r['source_matched'] for r in rows),
        paired_returns=prior.paired_returns(record, baseline),
        groups={name: dict(trips=len(chosen), means=mean_rows(chosen, fields)) for name, chosen in
            (('commanded', [r for r in rows if r['requested_s'] > 0]),
             ('not_commanded', [r for r in rows if r['requested_s'] == 0]))},
        changed_stop_actions=len(action_changes), first_changed_stop_action=first, first_commands=commands[:2],
        largest_delay_increases=sorted((r for r in rows if r['delay_delta_s'] > 0), key=lambda r: -r['delay_delta_s'])[:2],
        largest_delay_reductions=sorted((r for r in rows if r['delay_delta_s'] < 0), key=lambda r: r['delay_delta_s'])[:2],
        largest_holding_reductions=sorted(rows, key=lambda r: r['holding_delta_s'])[:2])
    return summary, detail


def run(args):
    parent = prior.read(args.v130_root/'summary/status.json')
    paths, focused_details = [], {}
    for seed, day in prior.ITEMS:
        endpoint = next(r for r in parent['actor_day'] if (r['seed'], r['day']) == (seed, day))
        for lower, method in prior.LOWER.items():
            name = f'{seed}_{day}_{lower}_plan'
            cache = args.v124_root if lower == 'split' else args.v123_root
            summary, detail = analyze(prior.read(args.v130_root/'terminal'/(name+'.json')),
                prior.read(args.v127_root/'turnover'/f'{seed}_{day}_{method}.json'),
                read_actions(args.v130_root/'records'/(name+'.npz')),
                read_actions(cache/'records'/f'{seed}_{day}_{method}_0.npz'),
                endpoint['effects'][lower+'_plan_vs_'+lower])
            paths.append(dict(seed=seed, day=day, lower=lower, **summary))
            write(args.out_root/'details'/(name+'.json'), detail)
            if lower == 'split' and day == 2231004:
                focused_details[seed] = detail
    first, second = [next(r for r in paths if (r['seed'], r['day'], r['lower']) == (s, 2231004, 'split'))
        for s in (12011, 933013)]
    action_tables = [{(r['trip_id'], r['station_id'], r['direction']): r
        for r in focused_details[s]['changed_stop_actions']} for s in (12011, 933013)]
    different = []
    for key in action_tables[0].keys() | action_tables[1].keys():
        a, b = [t.get(key) for t in action_tables]
        if (a['action_delta_s'] if a else 0.) != (b['action_delta_s'] if b else 0.):
            different.append(dict(event=key, actor12011=a, actor933013=b))
    different.sort(key=lambda r: (min(v['base_arrival_s'] for v in
        (r['actor12011'], r['actor933013']) if v), r['event']))
    write(args.out_root/'summary/status.json', dict(status='v131_read_only_plan_propagation_complete', actor_day=paths,
        day2231004_actor_difference=dict(actor_order=[12011, 933013],
            effects={k: first['effect'][k]-second['effect'][k] for k in first['effect']},
            contributions={k: first['contributions'][k]-second['contributions'][k] for k in PARTS},
            first_different_holding_responses=different[:2]),
        consumed_candidate_paths=8, consumed_controls=8, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Read-only realized-path accounting, not per-action causal values. Clipped supply '
            'changes allocate the common positive-part secant slope to matched return components; '
            'unmatched allocations stay separate. Requested and critical control wait are different. '
            'Stop actions are recovered from cumulative holding at arrival/departure; their displayed '
            'clocks are arrivals, not recorded policy-query times. Full per-trip/stop details stay remote. '
            'The two actors and two days are consumed development evidence; no controller or gate tuning.'))
    print('V131_PLAN_PROPAGATION_COMPLETE', dict(paths=len(paths), new_rollouts=0), flush=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v123-root', 'v124-root', 'v127-root', 'v130-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v131_*.py', '-v'], check=True)
    run(args)
