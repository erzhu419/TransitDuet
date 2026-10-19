#!/usr/bin/env python3
"""Read stored V135 paths only; separate request changes and path propagation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import numpy as np

import run_freqduet_v135_handoff_confirmation as frozen
import run_freqduet_v92_elapsed_review as elapsed
import audit_freqduet_v131_plan_propagation as turnover


COMPONENTS = elapsed.COMPONENTS
ENDPOINTS = (*frozen.PRIMARY, 'journey_min', 'waiting_person_s', 'onboard_person_s',
    'holding_person_s', 'holding_s_per_trip', 'dispatch_cv_squared')
CI_FIELDS = ('direct_override_s', 'realized_holding_delta_s', 'continuation_accounting_residual_s',
    'realized_on_handoff_keys_s', 'realized_off_handoff_keys_s',
    *(k+'_allocated' for k in COMPONENTS), 'mean_normalization',
    'delay_prior_holding_s', 'delay_prior_other_s', 'delay_prior_launch_s')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def close(actual, expected, label):
    if not np.isclose(actual, expected, atol=1e-6, rtol=0):
        raise AssertionError(f'{label}: {actual} != {expected}')


def read_path(root, seed, day, mode, endpoint):
    name = f'{seed}_{day}_{mode}'
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        events = {kind: [dict(zip(fields, r.tolist())) for r in record[kind]]
            for kind in ('arrivals', 'departures')}
        launches = [dict(zip(record['launch_fields'].tolist(), r.tolist())) for r in record['launches']]
    split = elapsed.split_stop_time(events)
    series = {kind: elapsed.series(events, split, kind) for kind in ('arrival', 'departure')}
    actions = turnover.stop_actions(events['arrivals'], events['departures'])
    terminal = read(root/'terminal'/(name+'.json'))
    dispatch = {r['trip_id']: r for r in terminal['dispatches']}
    returns = {r['trip_id']: r for r in terminal['returns']}
    clocks = {int(k): v for k, v in terminal['reference_clocks'].items()}
    if dispatch.keys() != returns.keys() or dispatch.keys() != clocks.keys():
        raise AssertionError('complete V135 path lacks a dispatch, return, or reference clock')
    held = dict.fromkeys(dispatch, 0.)
    for k, r in actions.items():
        held[k[0]] += r['action_s']
    for trip, r in returns.items():
        close(held[trip], r['holding_s'], 'realized stop actions recover completed trip holding')
        close(r['return_s']-r['launch_s'], r['holding_s']+r['nonholding_elapsed_s'], 'trip return identity')
    close(series['arrival']['cv_squared'], endpoint['cv_squared'], 'V135 actual arrival CV2')
    close(sum(held.values())/len(dispatch), endpoint['holding_s_per_trip'], 'V135 holding endpoint')
    close(fmean(r['actual_s']-clocks[t] for t, r in dispatch.items()),
        endpoint['inclusive_terminal_delay_s'], 'V135 inclusive delay')
    close(fmean(r['readiness_s'] for r in launches), endpoint['readiness_delay_s'], 'V135 readiness')
    close(endpoint['waiting_person_s']+endpoint['onboard_person_s'],
        endpoint['passenger_person_s'], 'V135 passenger accounting')
    for r in terminal['plans']:
        close(r['command_s'], r['published_release_s'], 'V135 no added terminal command')
    with gzip.open(root/'launch_traces'/(name+'.jsonl.gz'), 'rt') as stream:
        traces = [json.loads(line) for line in stream if line.strip()]
    return dict(events=events, series=series, actions=actions, terminal=terminal,
        dispatch=dispatch, returns=returns, clocks=clocks, traces=traces, held=held,
        launch_times={int(r['trip_id']): r['time_s'] for r in launches})


def direct_accounting(base, candidate):
    b, c = base['actions'], candidate['actions']
    if b.keys() != c.keys():
        raise AssertionError('paired completed stop-action support differs')
    used, details = set(), []
    for r in candidate['traces']:
        if r['online']['status'] != 'launch_handoff_component' or not r['component_used']:
            continue
        k = elapsed.key(r)
        if k in used:
            raise AssertionError('multiple action queries at one stored completed stop')
        used.add(k)
        close(c[k]['action_s'], r['chosen_s'], 'stored action versus handoff receipt')
        close(r['executed_s'], r['chosen_s'], 'handoff executed command')
        delta = r['chosen_s']-r['raw_shadow_s']
        details.append(dict(trip_id=k[0], station_id=k[1], direction=int(k[2]),
            policy_time_s=r['policy_time_s'], old_request_on_candidate_state_s=r['raw_shadow_s'],
            candidate_request_s=r['chosen_s'], direct_override_s=delta,
            actual_old_path_action_s=b[k]['action_s'], realized_delta_s=c[k]['action_s']-b[k]['action_s'],
            incoming_clock_shift_s=c[k]['arrival_s']-b[k]['arrival_s'],
            incoming_holding_shift_s=c[k]['incoming_holding_s']-b[k]['incoming_holding_s'],
            incoming_release_shift_s=c[k]['release_s']-b[k]['release_s'],
            predicted_gap_s=r['online']['gap_s'], raw_gap_s=r['evidence']['eta_prediction_s']))
    changed = [dict(trip_id=k[0], station_id=k[1], direction=int(k[2]),
        base_arrival_s=b[k]['arrival_s'], candidate_arrival_s=c[k]['arrival_s'],
        action_delta_s=c[k]['action_s']-b[k]['action_s'], handoff_key=k in used,
        incoming_release_delta_s=c[k]['release_s']-b[k]['release_s'],
        incoming_holding_delta_s=c[k]['incoming_holding_s']-b[k]['incoming_holding_s'],
        incoming_other_delta_s=c[k]['incoming_other_s']-b[k]['incoming_other_s'])
        for k in b if c[k]['action_s'] != b[k]['action_s']]
    direct = sum(r['direct_override_s'] for r in details)
    realized = sum(r['action_delta_s'] for r in changed)
    on = sum(r['action_delta_s'] for r in changed if r['handoff_key'])
    changed.sort(key=lambda r: min(r['base_arrival_s'], r['candidate_arrival_s']))
    fields = dict(handoff_decisions=len(details), direct_cuts=sum(r['direct_override_s'] < 0 for r in details),
        direct_increases=sum(r['direct_override_s'] > 0 for r in details),
        direct_unchanged=sum(r['direct_override_s'] == 0 for r in details),
        direct_override_s=direct, realized_holding_delta_s=realized,
        realized_on_handoff_keys_s=on, realized_off_handoff_keys_s=realized-on,
        continuation_accounting_residual_s=realized-direct,
        changed_stop_actions=len(changed), changed_off_handoff_keys=sum(not r['handoff_key'] for r in changed))
    if not any(r['direct_override_s'] != 0 for r in details):
        if changed or base['terminal'] != candidate['terminal']:
            raise AssertionError('trajectory changed without a recorded direct request change')
        for kind in ('arrivals', 'departures'):
            if base['events'][kind] != candidate['events'][kind]:
                raise AssertionError('timing changed without a recorded direct request change')
    return fields, dict(direct_requests=details, changed_stop_actions=changed)


def delay_accounting(base, candidate):
    if base['dispatch'].keys() != candidate['dispatch'].keys():
        raise AssertionError('paired dispatch support differs')
    rows = []
    for trip, dispatch in candidate['dispatch'].items():
        old = base['dispatch'][trip]
        b = turnover.launch_state(old, base['clocks'][trip])
        c = turnover.launch_state(dispatch, candidate['clocks'][trip])
        matched = (dispatch['bus_id'], dispatch['prior_trip_id']) == (old['bus_id'], old['prior_trip_id'])
        changes = None
        if matched and dispatch['prior_trip_id'] >= 0:
            bb, cb = [p['returns'][dispatch['prior_trip_id']] for p in (base, candidate)]
            changes = dict(prior_launch_s=cb['launch_s']-bb['launch_s'],
                prior_holding_s=cb['holding_s']-bb['holding_s'],
                prior_other_s=cb['nonholding_elapsed_s']-bb['nonholding_elapsed_s'],
                visibility_lag_s=(c['visible_s']-cb['return_s'])-(b['visible_s']-bb['return_s']),
                reference_clip_s=b['reference_s']-c['reference_s'])
        parts = dict.fromkeys(turnover.PARTS, 0.)
        parts.update(turnover.allocate_supply(b, c, changes),
            control_s=c['control_s']-b['control_s'], residual_s=c['residual_s']-b['residual_s'])
        close(sum(parts.values()), c['delay_s']-b['delay_s'], 'inclusive delay allocation')
        returned = candidate['returns'][trip]['return_s']-base['returns'][trip]['return_s']
        shift = dispatch['actual_s']-old['actual_s']
        holding = candidate['held'][trip]-base['held'][trip]
        other = candidate['returns'][trip]['nonholding_elapsed_s']-base['returns'][trip]['nonholding_elapsed_s']
        close(returned, shift+holding+other, 'return propagation')
        rows.append(dict(trip_id=trip, actual_shift_s=shift,
            reference_shift_s=c['reference_s']-b['reference_s'], return_shift_s=returned,
            holding_delta_s=holding, other_delta_s=other, matched_vehicle_prior=matched, parts=parts))
    fields = {'delay_'+k: fmean(r['parts'][k] for r in rows) for k in turnover.PARTS}
    fields.update({k: fmean(r[k] for r in rows) for k in
        ('actual_shift_s', 'reference_shift_s', 'return_shift_s', 'other_delta_s')})
    fields['remapped_vehicle_prior'] = sum(not r['matched_vehicle_prior'] for r in rows)
    return fields, rows


def pooled_profile(base, candidate, group):
    parts = []
    for p in (base, candidate):
        total, table = p['series']['arrival'], {}
        for k, r in total['edges'].items():
            label = str(group(k))
            table[label] = table.get(label, 0.)+(r['headway']-total['mean'])**2/total['samples']/total['mean']**2
        parts.append(table)
    result = {k: parts[1].get(k, 0.)-parts[0].get(k, 0.) for k in parts[0].keys() | parts[1].keys()}
    close(sum(result.values()), candidate['series']['arrival']['cv_squared']-
        base['series']['arrival']['cv_squared'], 'pooled CV2 profile')
    return result


def analyze_cell(root, out, index):
    seed, day = frozen.ITEMS[index]
    receipt = read(root/'pairs'/f'p{index}.json')
    if (receipt['index'], receipt['seed'], receipt['day']) != (index, seed, day):
        raise AssertionError('V135 pair identity differs from its complete registered grid')
    paths = {m: read_path(root, seed, day, m, receipt['results'][m]) for m in frozen.MODES}
    b, c = paths['old'], paths['handoff']
    actions, action_details = direct_accounting(b, c)
    delay, delay_details = delay_accounting(b, c)
    timing = {k: elapsed.effects(b['series'][k], c['series'][k]) for k in ('arrival', 'departure')}
    effects = receipt['effects'][frozen.CONTRAST]
    close(timing['arrival']['cv_squared'], effects['cv_squared'], 'paired CV2 effect')
    close(actions['realized_holding_delta_s']/len(c['dispatch']), effects['holding_s_per_trip'], 'paired holding effect')
    close(sum(delay['delay_'+k] for k in turnover.PARTS), effects['inclusive_terminal_delay_s'], 'paired delay effect')
    diagnostics = dict(actions, **delay, **timing['arrival'])
    detail = dict(seed=seed, day=day, effects=effects, diagnostics=diagnostics,
        actions=action_details, dispatches=delay_details, timing=timing,
        cv2_by_station=pooled_profile(b, c, lambda k: (k[1], int(k[2]))),
        cv2_by_launch_band=pooled_profile(b, c, lambda k: int(b['launch_times'][k[0]]//14400)))
    write(out/'details'/f'p{index}.json', detail)
    return compact_cell(detail, receipt)


def variance_totals(timing):
    parts = {kind+'_sum': sum(v for k, v in timing.items() if k.endswith('_'+kind))
        for kind in ('diagonal', 'cross')}
    close(sum(parts.values())+timing['mean_normalization'], timing['cv_squared'], 'CV2 diagonal/cross identity')
    return parts


def compact_cell(detail, receipt):
    diagnostics = dict(detail['diagnostics'], **variance_totals(detail['timing']['arrival']))
    action_details = detail['actions']
    return dict(seed=detail['seed'], day=detail['day'], effects=detail['effects'], diagnostics=diagnostics,
        handoff_queries=receipt['use']['handoff']['launch_handoffs'],
        cv2_by_station=detail['cv2_by_station'], cv2_by_launch_band=detail['cv2_by_launch_band'],
        first_direct_changes=sorted((r for r in action_details['direct_requests'] if r['direct_override_s'] != 0),
            key=lambda r: r['policy_time_s'])[:2],
        first_changed_stops=action_details['changed_stop_actions'][:2])


def mean_table(rows, field):
    return {k: fmean(r[field][k] for r in rows) for k in rows[0][field]} if rows else None


def profile_means(rows, field):
    return {k: fmean(r[field].get(k, 0.) for r in rows)
        for k in sorted(set().union(*(r[field].keys() for r in rows)))}


def summarize(rows):
    mechanistic = [dict(seed=r['seed'], day=r['day'], effects={'handoff_vs_old':
        {k: r['diagnostics'][k] for k in CI_FIELDS}}) for r in rows]
    return dict(cells=len(rows), endpoint_means=mean_table(rows, 'effects'),
        diagnostic_means=mean_table(rows, 'diagnostics'),
        diagnostic_ci=frozen.inference.describe(mechanistic, 'effects')['handoff_vs_old'],
        cv2_by_station=profile_means(rows, 'cv2_by_station'),
        cv2_by_launch_band=profile_means(rows, 'cv2_by_launch_band'),
        totals={k: sum(r['diagnostics'][k] for r in rows) for k in
            ('handoff_decisions', 'direct_cuts', 'direct_increases', 'direct_unchanged',
             'changed_stop_actions', 'changed_off_handoff_keys')},
        handoff_queries=sum(r['handoff_queries'] for r in rows),
        no_direct_action_cells=sum(r['diagnostics']['direct_cuts']+r['diagnostics']['direct_increases'] == 0 for r in rows),
        cell_patterns={name: len(chosen) for name, chosen in pattern_groups(rows).items()})


def pattern_groups(rows):
    return {name: [r for r in rows if predicate(r['effects'])] for name, predicate in (
        ('passenger_benefit_cv_harm', lambda e: e['passenger_person_s'] < 0 and e['cv_squared'] > 0),
        ('passenger_harm', lambda e: e['passenger_person_s'] > 0),
        ('joint_improvement', frozen.frozen.joint_improvement),
        ('no_endpoint_change', lambda e: all(e[k] == 0 for k in frozen.PRIMARY)))}


def axes(rows, key):
    return {str(v): dict(cells=len(chosen), endpoints={k: fmean(r['effects'][k] for r in chosen) for k in ENDPOINTS},
        diagnostics=mean_table(chosen, 'diagnostics'))
        for v in sorted({r[key] for r in rows}) for chosen in ([r for r in rows if r[key] == v],)}


def run(args):
    parent = read(args.v135_root/'summary/status.json')
    if parent['status'] != 'v135_frozen_handoff_confirmation_complete' or parent['paired_cells'] != len(frozen.ITEMS):
        raise AssertionError('V136 requires the complete V135 confirmation')
    rows = []
    if args.command == 'report':
        for i, (seed, day) in enumerate(frozen.ITEMS):
            detail = read(args.out_root/'details'/f'p{i}.json')
            if (detail['seed'], detail['day']) != (seed, day):
                raise AssertionError('saved V136 detail identity differs from V135 grid')
            rows.append(compact_cell(detail, read(args.v135_root/'pairs'/f'p{i}.json')))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(analyze_cell, args.v135_root, args.out_root, i) for i in range(len(frozen.ITEMS))]
            for task in as_completed(tasks):
                rows.append(task.result())
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=len(frozen.ITEMS), unit='pair')), flush=True)
    rows.sort(key=lambda r: (r['seed'], r['day']))
    primary = [r for r in rows if r['seed'] in frozen.VALIDATION_SEEDS]
    secondary = [r for r in rows if r['seed'] in frozen.DEVELOPMENT_SEEDS]
    primary_summary = summarize(primary)
    for metric, value in primary_summary['endpoint_means'].items():
        close(value, parent['cohorts']['heldout_actors']['effects'][frozen.CONTRAST][metric]['mean'], 'parent primary mean')
    write(args.out_root/'summary/status.json', dict(status='v136_v135_readonly_propagation_complete',
        primary=primary_summary, development_actor_new_days=summarize(secondary),
        parent_primary_gate=parent['primary_joint_improvement'],
        parent_cell_joint_improvements=parent['cohorts']['heldout_actors']['cell_joint_improvements'],
        parent_endpoint_intervals=parent['cohorts']['heldout_actors']['effects'][frozen.CONTRAST],
        by_actor=axes(primary, 'seed'), by_day=axes(primary, 'day'),
        patterns={name: dict(cells=len(chosen), endpoint_means=mean_table(chosen, 'effects'),
            diagnostics=mean_table(chosen, 'diagnostics')) for name, chosen in pattern_groups(primary).items()},
        examples=[r for identity in ((12071, 2351015), (12143, 2351015), (933059, 2351002),
            (12071, 2351007), (12037, 2351004), (12109, 2351014))
            for r in primary if (r['seed'], r['day']) == identity],
        consumed_pairs=len(rows), consumed_paths=2*len(rows), new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, parameters_changed=False,
        endpoint_reconciliation=True, stop_holding_and_return_reconciliation=True,
        limitations='Read-only accounting of consumed confirmation paths, not causal mediation or a new gate. '
            'Direct override compares the recorded candidate request with the old raw request on that SAME '
            'candidate state at handoff decisions; realized path differences compare DIFFERENT evolving states. '
            'Their arithmetic remainder is continuation/state-distribution accounting, not an isolated effect. '
            'CV2 covariance allocation and clipped supply allocation are identities, not intervention values. '
            'Arrival predecessor changes are retained; pooled profiles use whole-day means. '
            'Nonholding stop includes service/ticks; remaining elapsed is not pure running time. '
            'Sparse launch traces do not provide every follower-ready truth or passenger-level exposure, '
            'so exact ETA bias and per-action passenger causation remain unresolved. '
            'Actor/day intervals are descriptive after consumption, not candidate retuning or promotion.'))
    print('V136_READONLY_PROPAGATION_COMPLETE', dict(pairs=len(rows), new_rollouts=0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'report'))
    p.add_argument('--v135-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v136_*.py', '-v'], check=True)
    run(args)
