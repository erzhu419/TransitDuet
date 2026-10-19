#!/usr/bin/env python3
"""Read V137 only: request, quantization, exposure and turnover accounting."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

import numpy as np

import run_freqduet_v137_passive_ready_exposure as saved
import audit_freqduet_v136_handoff_propagation as timing


PARTS = ('launch_timing_s', 'cumulative_holding_s', 'other_elapsed_s', 'forecast_error_s',
    'discrete_jump_s', 'regime_change_s', 'forward_only_s', 'unresolved_ready_s',
    'follower_identity_change_s', 'follower_not_next_s', 'forward_identity_change_s',
    'own_bus_change_s', 'source_change_s')
ACTIONS = np.asarray((0., 5., 10., 15., 20., 30., 45.))
key = saved.passive.key
read, close = saved.read, saved.close


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def request(row):
    c = row['context']
    if not c['evidence_valid']:
        return 'none', 0., None
    if not c['follower_evidence_valid']:
        return 'forward', .25*(c['target_headway_s']-c['forward_headway_s']), None
    p = row['online']['gap_s'] if row['online']['component_used'] else c['follower_departure_gap_s']
    return 'bilateral', (p-c['forward_headway_s'])/8., p


def clip_slope(before, after):
    b, c = np.clip([before, after], 0., 45.)
    return float((c-b)/(after-before)) if after != before else float(0. < before < 45.)


def timing_response(b, c):
    # Coordinates are launch, cumulative holding, and remaining elapsed.
    result = (c['follower_coordinates']-b['follower_coordinates']
        +c['front_coordinates']-b['front_coordinates']
        -2*(c['own_coordinates']-b['own_coordinates']))/8.
    expected = ((c['true_gap']-b['true_gap'])-(c['forward']-b['forward']))/8.
    close(float(result.sum()), expected, 'realized forward/follower timing coordinates')
    return result


def category(b, c):
    if b['regime'] != c['regime'] or b['regime'] == 'none':
        return 'regime_change'
    if b['regime'] == 'forward':
        return 'forward_only'
    if b['true_gap'] is None or c['true_gap'] is None:
        return 'unresolved_ready'
    if b['follower_id'] != c['follower_id']:
        return 'follower_identity_change'
    if not b['follower_is_next'] or not c['follower_is_next']:
        return 'follower_not_next'
    if b['front_id'] is None or c['front_id'] is None or b['front_id'] != c['front_id']:
        return 'forward_identity_change'
    if b['own_bus'] != c['own_bus']:
        return 'own_bus_change'
    if b['source'] != c['source']:
        return 'source_change'
    return 'stable_matched'


def split_request(b, c):
    parts = dict.fromkeys(PARTS, 0.)
    name, slope = category(b, c), clip_slope(b['u'], c['u'])
    cb, cc = [float(np.clip(r['u'], 0., 45.)) for r in (b, c)]
    parts['discrete_jump_s'] = (c['h']-cc)-(b['h']-cb)
    if name == 'stable_matched':
        physical = timing_response(b, c)
        error = ((c['prediction']-c['true_gap'])-(b['prediction']-b['true_gap']))/8.
        close(float(physical.sum())+error, c['u']-b['u'], 'unbounded request decomposition')
        for k, v in zip(PARTS[:3], physical):
            parts[k] = slope*float(v)
        parts['forecast_error_s'] = slope*error
    else:
        parts[name+'_s'] = cc-cb
    close(sum(parts.values()), c['h']-b['h'], 'executed action decomposition')
    return name, parts, slope


def source_detail(b, c):
    physical = timing_response(b, c)*clip_slope(b['u'], c['u'])
    error = ((c['prediction']-c['true_gap'])-(b['prediction']-b['true_gap']))/8.
    result = dict(zip(PARTS[:3], physical.tolist()))
    result['forecast_error_s'] = error*clip_slope(b['u'], c['u'])
    close(sum(result.values()), float(np.clip(c['u'], 0., 45.)-np.clip(b['u'], 0., 45.)),
        'same-identity source-switch continuous allocation')
    return result


def ready_coordinates(path, identity):
    row, label = path['arrivals'][identity], path['ready'][identity]
    return np.asarray((row['release'], row['realized_holding'],
        label['ready_time_s']-row['release']-row['realized_holding']))


def feature(path, row):
    identity, e, c = key(row), row['evidence'], row['context']
    regime, u, prediction = request(row)
    close(u, .25*row['feedback_error_s'], 'recorded continuous request')
    own_ready = e['ready_time_s']
    front_keys = path['departure_clocks'].get((identity[1], identity[2],
        own_ready-c['forward_headway_s']), []) if c['evidence_valid'] else []
    front_keys = [k for k in front_keys if k != identity]
    front = front_keys[0] if len(front_keys) == 1 else None
    follower = e['predicted_follower_trip_id'], identity[1], identity[2]
    label = path['ready'].get(follower)
    gap, reason = None, 'missing_same_trip_ready'
    if label is not None:
        if label['bus_id'] != e['predicted_follower_bus_id']:
            reason = 'same_trip_bus_mismatch'
        elif label['ready_time_s'] < own_ready:
            reason = 'same_follower_already_ready'
        else:
            gap, reason = label['ready_time_s']-own_ready, None
    h = row['executed_s']
    quantized = float(ACTIONS[np.argmin(np.abs(ACTIONS-np.clip(u, 0., 45.)))])
    result = dict(regime=regime, u=u, prediction=prediction, h=h,
        forward=c['forward_headway_s'], true_gap=gap, unresolved=reason,
        own_bus=e['bus_id'], front_id=None if front is None else
            (front[0], path['dispatch'][front[0]]['bus_id']),
        follower_id=(e['predicted_follower_trip_id'], e['predicted_follower_bus_id']),
        follower_is_next=follower == path['next_departure'].get(identity),
        source=(c['evidence_source'], e['follower_source'], row['online']['status'], row['component_used']),
        unmasked_quantizer_matches=quantized == h, own_coordinates=ready_coordinates(path, identity))
    close(result['own_coordinates'][1], row['executed_before_s'], 'holding before ready/query')
    if front is not None:
        r = path['departures'][front]
        result['front_coordinates'] = np.asarray([r[n] for n in ('release', 'realized_holding', 'other_elapsed')])
        close(float(result['front_coordinates'].sum()), own_ready-c['forward_headway_s'], 'matched front departure')
    if gap is not None:
        result['follower_coordinates'] = ready_coordinates(path, follower)
    return result


def load_path(root, terminal_root, seed, day, mode):
    name = f'{seed}_{day}_{mode}'
    trace = {key(r): r for r in saved.identity.read_trace(root/'traces'/(name+'.jsonl.gz'))}
    ready = {key(r): r for r in saved.identity.read_trace(root/'ready_labels'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        rows = {kind: [dict(zip(fields, r.tolist())) for r in record[kind]] for kind in ('arrivals', 'departures')}
    departures = {key(r): r for r in rows['departures']}
    arrivals = {key(r): r for r in rows['arrivals']}
    clocks = defaultdict(list)
    for k, r in departures.items():
        clocks[k[1], k[2], r['time_s']].append(k)
    terminal = read(terminal_root/'terminal'/(name+'.json'))
    dispatch = {r['trip_id']: r for r in terminal['dispatches']}
    held = dict.fromkeys(dispatch, 0.)
    for k, r in trace.items():
        held[k[0]] += r['executed_s']
    returns = {r['trip_id']: r for r in terminal['returns']}
    for t in held:
        close(held[t], returns[t]['holding_s'], 'issued and returned trip holding')
    return dict(trace=trace, ready=ready, departures=departures, arrivals=arrivals,
        departure_clocks=clocks, next_departure=saved.identity.successors(rows['departures']),
        dispatch=dispatch, returns=returns, held=held,
        clocks={int(k): v for k, v in terminal['reference_clocks'].items()})


def describe(rows):
    changed = [r for r in rows if r['delta_s'] != 0]
    return dict(decisions=len(rows), changed_actions=sum(r['delta_s'] != 0 for r in rows),
        median_changed_clipped_request_s=median(abs(r['clipped_request_delta_s']) for r in changed) if changed else None,
        median_changed_action_s=median(abs(r['delta_s']) for r in changed) if changed else None,
        holding_delta_s=sum(r['delta_s'] for r in rows),
        positive_holding_s=sum(max(0., r['delta_s']) for r in rows),
        negative_holding_s=sum(min(0., r['delta_s']) for r in rows),
        holding_person_delta_s=sum(r['person_delta_s'] for r in rows),
        load_change_person_s=sum(r['load_change_person_s'] for r in rows),
        parts_s={k: sum(r['parts'][k] for r in rows) for k in PARTS},
        parts_at_candidate_load_person_s={k: sum(r['parts'][k]*r['candidate_load'] for r in rows) for k in PARTS})


def clock_bands(rows):
    result = defaultdict(lambda: dict(anchored_holding_s=0., clock_relocation_s=0.,
        issued_holding_delta_s=0., anchored_person_s=0., person_clock_relocation_s=0.,
        issued_person_delta_s=0.))
    for r in rows:
        b, c = int(r['base_time_s']//900), int(r['candidate_time_s']//900)
        result[c]['anchored_holding_s'] += r['delta_s']
        result[c]['anchored_person_s'] += r['person_delta_s']
        for band, sign in ((c, 1.), (b, -1.)):
            result[band]['clock_relocation_s'] += sign*r['base_holding_s']
            result[band]['person_clock_relocation_s'] += sign*r['base_person_s']
        result[c]['issued_holding_delta_s'] += r['base_holding_s']+r['delta_s']
        result[b]['issued_holding_delta_s'] -= r['base_holding_s']
        result[c]['issued_person_delta_s'] += r['base_person_s']+r['person_delta_s']
        result[b]['issued_person_delta_s'] -= r['base_person_s']
    for r in result.values():
        close(r['anchored_holding_s']+r['clock_relocation_s'], r['issued_holding_delta_s'], 'command time-band relocation')
        close(r['anchored_person_s']+r['person_clock_relocation_s'], r['issued_person_delta_s'], 'exposure time-band relocation')
    return result


def analyze(args, seed, day, parent):
    paths = {m: load_path(args.v137_root, args.v135_root, seed, day, m) for m in ('old', 'handoff')}
    b, c = paths['old'], paths['handoff']
    if b['trace'].keys() != c['trace'].keys():
        raise AssertionError('paired decision support differs')
    direct_keys = {key(r) for r in c['trace'].values()
        if r['component_used'] and r['online']['status'] == 'launch_handoff_component'}
    rows, quantizer_mismatches, reasons = [], Counter(), Counter()
    for k, cr in c['trace'].items():
        br = b['trace'][k]
        bf, cf = feature(b, br), feature(c, cr)
        category_name, parts, slope = split_request(bf, cf)
        for mode, f in (('old', bf), ('handoff', cf)):
            quantizer_mismatches[mode] += not f['unmasked_quantizer_matches']
            if f['unresolved'] is not None:
                reasons[mode+':'+f['unresolved']] += 1
        delta = cr['executed_s']-br['executed_s']
        person = cr['holding_person_s']-br['holding_person_s']
        load = br['executed_s']*(cr['query_load']-br['query_load'])
        close(cr['query_load']*sum(parts.values())+load, person, 'request/exposure decomposition')
        rows.append(dict(trip_id=k[0], station_id=k[1], direction=int(k[2]), direct=k in direct_keys,
            base_time_s=br['policy_time_s'], candidate_time_s=cr['policy_time_s'],
            base_holding_s=br['executed_s'], delta_s=delta, candidate_load=cr['query_load'],
            base_person_s=br['holding_person_s'], person_delta_s=person, load_change_person_s=load,
            category=category_name, parts=parts, clip_slope=slope,
            source_transition=repr(bf['source'])+' -> '+repr(cf['source']),
            source_detail=source_detail(bf, cf) if category_name == 'source_change' else None,
            raw_request_delta_s=cf['u']-bf['u'], clipped_request_delta_s=float(np.clip(cf['u'], 0., 45.)-np.clip(bf['u'], 0., 45.)),
            forward_delta_s=None if bf['forward'] is None or cf['forward'] is None else cf['forward']-bf['forward'],
            realized_follower_delta_s=None if bf['true_gap'] is None or cf['true_gap'] is None else cf['true_gap']-bf['true_gap']))
    summary, other = describe(rows), [r for r in rows if not r['direct']]
    effect = parent['effect']
    timing.close(summary['holding_delta_s']/len(c['dispatch']), effect['holding_s_per_trip'], 'parent holding endpoint')
    close(summary['holding_person_delta_s'], effect['holding_person_s'], 'parent holding exposure endpoint')
    delay, dispatches = timing.delay_accounting(b, c)
    close(sum(delay['delay_'+k] for k in timing.turnover.PARTS), effect['inclusive_terminal_delay_s'], 'parent delay endpoint')
    bands = clock_bands(rows)
    exposure = {m: read(args.v137_root/'exposure'/f'{seed}_{day}_{m}.json') for m in paths}
    turnover_bands = defaultdict(lambda: dict(return_shift_s=0., launch_shift_s=0., launch_delay_delta_s=0., trips=0))
    for r in dispatches:
        trip = r['trip_id']
        band = int(c['dispatch'][trip]['actual_s']//900)
        turnover_bands[band]['launch_shift_s'] += r['actual_shift_s']
        turnover_bands[band]['launch_delay_delta_s'] += sum(r['parts'].values())
        turnover_bands[band]['trips'] += 1
        return_band = int(c['returns'][trip]['return_s']//900)
        turnover_bands[return_band]['return_shift_s'] += r['return_shift_s']
    timeline = []
    for band in sorted({int(k) for k in exposure['old']} | {int(k) for k in exposure['handoff']}):
        delta = np.asarray(exposure['handoff'].get(str(band), [0.]*5))-exposure['old'].get(str(band), [0.]*5)
        timeline.append(dict(start_s=900*band, waiting_person_s=float(delta[0]), onboard_person_s=float(delta[1]),
            backlog_trip_s=float(delta[4]), command=bands.get(band), turnover=turnover_bands.get(band)))
    for field, metric in (('waiting_person_s', 'waiting_person_s'), ('onboard_person_s', 'onboard_person_s'), ('backlog_trip_s', 'backlog_trip_s')):
        close(sum(r[field] for r in timeline), effect[metric], 'parent passenger/backlog timeline')
    first = min(r['candidate_time_s'] for r in rows if r['direct'] and
        c['trace'][r['trip_id'], r['station_id'], bool(r['direction'])]['chosen_s'] !=
        c['trace'][r['trip_id'], r['station_id'], bool(r['direction'])]['raw_shadow_s'])
    early = [r for r in timeline if r['start_s']+900 <= first]
    prefix = all(r[k] == 0 for r in early for k in ('waiting_person_s', 'onboard_person_s', 'backlog_trip_s'))
    largest = sorted(timeline, key=lambda r: abs(r['waiting_person_s']+r['onboard_person_s']), reverse=True)[:4]
    full = dict(seed=seed, day=day, effect=effect, all_decisions=summary, direct=describe([r for r in rows if r['direct']]),
        downstream=describe(other),
        downstream_by_category={g: describe([r for r in other if r['category'] == g]) for g in sorted({r['category'] for r in other})},
        downstream_by_sign={g: describe([r for r in other if int(np.sign(r['delta_s'])) == sign])
            for g, sign in (('increases', 1), ('decreases', -1), ('same_command', 0))},
        source_switch_details={g: dict(decisions=len(group),
            holding_delta_s=sum(r['delta_s'] for r in group),
            continuous_parts_s={k: sum(r['source_detail'][k] for r in group) for k in PARTS[:4]},
            continuous_parts_at_candidate_load_person_s={k: sum(r['source_detail'][k]*r['candidate_load'] for r in group) for k in PARTS[:4]},
            discrete_jump_s=sum(r['parts']['discrete_jump_s'] for r in group))
            for g in sorted({r['source_transition'] for r in other if r['source_detail'] is not None})
            for group in [[r for r in other if r['source_detail'] is not None and r['source_transition'] == g]]},
        quantizer_mismatches=dict(quantizer_mismatches), ready_label_unresolved=dict(reasons),
        first_direct_query_s=first, complete_pre_direct_bands_equal=prefix,
        delay_accounting=delay, backlog_minus_sum_launch_delay_delta_s=effect['backlog_trip_s']-
            sum(sum(r['parts'].values()) for r in dispatches),
        nonholding_onboard_delta_person_s=effect['onboard_person_s']-effect['holding_person_s'],
        largest_passenger_change_bands=largest,
        largest_downstream_holding_increases=sorted([r for r in other if r['delta_s'] > 0],
            key=lambda r: -r['person_delta_s'])[:3],
        largest_launch_delay_increases=sorted(dispatches, key=lambda r: -sum(r['parts'].values()))[:3])
    write(args.out_root/'details'/f'{seed}_{day}.json', dict(summary=full, timeline=timeline,
        changed_commands=[r for r in rows if r['delta_s'] != 0], dispatches=dispatches))
    print('V138_READONLY_CASE_DONE', seed, day, flush=True)
    return full


def run(args):
    parent = read(args.v137_root/'summary/status.json')
    if parent['status'] != 'v137_exact_passive_ready_exposure_complete' or not parent['all_original_endpoints_actions_timing_and_terminal_paths_exact']:
        raise AssertionError('V138 requires admitted V137 paths')
    lookup = {(r['seed'], r['day']): r for r in parent['cases']}
    cases = [analyze(args, s, d, lookup[s, d]) for s, d in saved.ITEMS]
    write(args.out_root/'summary/status.json', dict(status='v138_readonly_request_feedback_complete', cases=cases,
        consumed_paths=12, new_rollouts=0, new_demand_days=0, actor_updates=0, critic_updates=0,
        dual_updates=0, parameters_changed=False, main_policy_changed=False,
        limitations='Six outcome-selected consumed pairs, no independent confirmation or tuning. Realized '
            'follower-ready labels include future feedback; timing/error decomposition is an identity, not '
            'pre-decision information or causal mediation. Identity, source and evidence-regime changes '
            'retain separate whole-request buckets. Source-switch details additionally allocate their '
            'same-identity timing/error difference without asserting a same-state source intervention. '
            'Common clipping secant allocates stable request changes; '
            'interior zero-net changes retain cancellation, saturated zero-net changes allocate zero. '
            'Discrete residual also includes any unsupported feasible-mask difference; empirical unmasked '
            'quantizer mismatches are reported. Command exposure bins use issuance clocks, not physical '
            'holding-rate time; clock relocation is separate. Turnover bins anchor matched trips at candidate '
            'launch/return times, not an exact instantaneous queue decomposition. Stop-level passenger waiting '
            'is not saved; system exposure bands cannot establish per-stop passenger causation.'))
    print('V138_READONLY_REQUEST_FEEDBACK_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v138_*.py', '-v'], check=True)
    run(args)
