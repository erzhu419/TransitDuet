#!/usr/bin/env python3
"""Read V244's six saved paths only; no controller or rollout imports."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
import unittest

import numpy as np

from audit_freqduet_v213_coordination import moments, cv_parts


ARMS = ('v107', 'upper_only', 'lower_supply', 'joint_supply', 'lower_blind', 'joint_blind')
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
PARTS = ('release', 'direct_adjustment', 'native_response', 'other_elapsed')
PAIRS = ((a, 'v107') for a in ARMS[1:])
PAIRS = tuple(PAIRS)+(('joint_supply', 'lower_supply'), ('joint_blind', 'lower_blind'),
                     ('lower_supply', 'lower_blind'), ('joint_supply', 'joint_blind'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def close(a, b, name):
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-11, err_msg=name)


def key(row):
    return row['trip_id'], row['station_id'], row['direction']


def unique(rows, identity):
    out = {identity(r): r for r in rows}
    assert len(out) == len(rows), 'ambiguous saved event occurrence'
    return out


def direction_name(station, direction):
    return f'{station}:{int(direction)}'


def group_summary(rows):
    return dict(queries=len(rows), direct_delta_s=sum(r['direct_s'] for r in rows),
        changed=sum(r['direct_s'] != 0 for r in rows),
        predicted_loss_vs_native_s2=sum(r['predicted_loss_s2'] for r in rows if r['predicted_loss_s2'] is not None),
        predicted_loss_supported=sum(r['predicted_loss_s2'] is not None for r in rows),
        restore_local_labels=dict(Counter(r['restore_label'] for r in rows)),
        restored_command_s=sum(-r['direct_s'] for r in rows),
        local_restore_delta_s2=sum(r['restore_delta_s2'] for r in rows if r['restore_delta_s2'] is not None))


def restore_label(extra, front_gap, behind_gap):
    if front_gap is None or behind_gap is None:
        return 'boundary_unresolved', None
    if extra > behind_gap:
        return 'would_change_neighbour_order', None
    value = (front_gap+extra)**2+(behind_gap-extra)**2-front_gap**2-behind_gap**2
    label = 'cut_removed_local_compensation' if value < 0 else 'cut_helped_local_gap' if value > 0 else 'neutral'
    return label, value


def series(events, coordinates):
    previous, edges = {}, {}
    for r in events:
        group = r['station_id'], r['direction']
        if group in previous:
            p = previous[group]
            ident = (*group, p['trip_id'], r['trip_id'])
            assert ident not in edges
            x = coordinates[key(r)]-coordinates[key(p)]
            close(x.sum(), r['time_s']-p['time_s'], 'headway coordinate closure')
            edges[ident] = dict(h=float(x.sum()), x=x, own=key(r), previous=key(p),
                time_s=r['time_s'])
        previous[group] = r
    x = np.asarray([r['x'] for r in edges.values()])
    h = x.sum(1)
    centered = x-x.mean(0)
    covariance = centered.T@centered/len(x)
    close(covariance.sum(), h.var(), 'four-way covariance closure')
    return dict(edges=edges, mean=float(h.mean()), variance=float(h.var()),
        cv2=float(h.var()/h.mean()**2), covariance=covariance)


def four_way_parts(b, c):
    d = (c['covariance']-b['covariance'])/b['mean']**2
    result = {name: float(d[i, i]) for i, name in enumerate(PARTS)}
    result.update({a+'__'+z: float(2*d[i, j]) for i, a in enumerate(PARTS)
                   for j, z in enumerate(PARTS) if j > i})
    result['normalization'] = c['variance']*(1/c['mean']**2-1/b['mean']**2)
    close(sum(result.values()), c['cv2']-b['cv2'], 'direct/native covariance allocation')
    return result


def edge_accounting(b, c):
    be, ce = b['edges'], c['edges']
    nb, nc, mb, mc = len(be), len(ce), b['mean'], c['mean']
    rows = []
    for k in sorted(be.keys() & ce.keys()):
        br, cr = be[k], ce[k]
        contribution = ((cr['h']-mc)**2/nc-(br['h']-mb)**2/nb)/mb**2
        rows.append(dict(station_id=k[0], direction=int(k[1]), previous_trip=k[2], trip_id=k[3],
            candidate_hour=int(cr['time_s']//3600), contribution=contribution,
            headway_delta_s=cr['h']-br['h'],
            timing_delta_s=dict(zip(PARTS, (cr['x']-br['x']).tolist()))))
    added = sum((ce[k]['h']-mc)**2/nc/mb**2 for k in ce.keys()-be.keys())
    removed = -sum((be[k]['h']-mb)**2/nb/mb**2 for k in be.keys()-ce.keys())
    normalization = c['variance']*(1/mc**2-1/mb**2)
    close(sum(r['contribution'] for r in rows)+added+removed+normalization,
          c['cv2']-b['cv2'], 'matched edges plus changed-neighbour remainder')
    groups = defaultdict(lambda: dict(edges=0, cv2_contribution=0., holding_gap_delta_s=0.))
    for r in rows:
        g = groups[direction_name(r['station_id'], r['direction'])]
        g['edges'] += 1
        g['cv2_contribution'] += r['contribution']
        g['holding_gap_delta_s'] += r['timing_delta_s']['direct_adjustment']+r['timing_delta_s']['native_response']
    return dict(matched_edges=len(rows), added_edges=len(ce.keys()-be.keys()),
        removed_edges=len(be.keys()-ce.keys()), added_cv2=added, removed_cv2=removed,
        normalization=normalization, matched_cv2=sum(r['contribution'] for r in rows),
        stop_direction=groups,
        largest_losses=sorted(rows, key=lambda r: r['contribution'], reverse=True)[:8],
        largest_gains=sorted(rows, key=lambda r: r['contribution'])[:8])


def prepare(data, check):
    assert data['source'] == check['source'] == 'b879b53b25' and data['complete'] and check['passed']
    t, requests = data['trace'], data['requests']
    arrivals, departures = (unique(t[name], key) for name in ('arrivals', 'departures'))
    queries = unique(requests, lambda r: key(r['identity']))
    close(sum(q['issued_s'] for q in requests), check['actual_holding_s'], 'saved holding endpoint')
    raw_moments = {name: moments(t[name]) for name in ('arrivals', 'departures')}
    close(raw_moments['arrivals']['cv2'], check['metrics']['cv_squared'], 'saved CV2 endpoint')
    assert set(arrivals) == set(departures)
    by_stop = defaultdict(list)
    for r in t['departures']:
        by_stop[r['station_id'], r['direction']].append(r)
    neighbours = {}
    for rows in by_stop.values():
        for i, r in enumerate(rows):
            neighbours[key(r)] = (None if i == 0 else r['time_s']-rows[i-1]['time_s'],
                None if i+1 == len(rows) else rows[i+1]['time_s']-r['time_s'])
    steps, labels, same_state = defaultdict(list), [], []
    anchor_seq = data['anchor']['fresh_query']['event_seq']
    for q in requests:
        k = key(q['identity'])
        assert q['issued_s'] == ACTIONS[q['selected_bin']]
        direct = q['issued_s']-ACTIONS[q['native_bin']]
        assert direct <= 0
        steps[k[0]].append((departures[k]['time_s'], q['issued_s'], direct))
        if q['native_event_seq'] <= anchor_seq:
            assert direct == 0
            continue
        sr, blind = q['supply_request'], q['blind_request']
        if sr['shadow_bin'] != blind['shadow_bin']:
            same_state.append(dict(key=list(k), supply_s=ACTIONS[sr['shadow_bin']],
                blind_s=ACTIONS[blind['shadow_bin']], delta_s=ACTIONS[sr['shadow_bin']]-ACTIONS[blind['shadow_bin']],
                reason=sr['reason'], exposure=sr['exposure']))
        if direct == 0:
            continue
        selected = sr if q['returned_rule'] == 'v238' else blind
        label, loss = restore_label(-direct, *neighbours[k])
        labels.append(dict(trip_id=k[0], station_id=k[1], direction=int(k[2]),
            hour=int(q['query_s']//3600), native_s=ACTIONS[q['native_bin']],
            selected_s=q['issued_s'], direct_s=direct, exposure=sr['exposure'],
            reason=sr['reason'], predicted_loss_s2=selected['local_gap_delta_vs_native_s2'],
            restore_label=label, restore_delta_s2=loss))
    prefixes = {}
    for trip, rows in steps.items():
        rows.sort()
        prefixes[trip] = ([r[0] for r in rows], np.cumsum([[r[1], r[2]] for r in rows], axis=0))
    coordinates = {}
    for name in ('arrivals', 'departures'):
        out = {}
        for r in t[name]:
            times, values = prefixes.get(r['trip_id'], ([], []))
            i = bisect_right(times, r['time_s'])-1
            held, direct = values[i] if i >= 0 else (0., 0.)
            close(held, r['realized_holding'], 'issued prefix matches actual executed holding')
            out[key(r)] = np.asarray([r['release'], direct, held-direct, r['other_elapsed']])
        coordinates[name] = series(t[name], out)
    launches = unique(t['actual_launches'], lambda r: r['trip'])
    returns = unique(t['vehicle_returns'], lambda r: r['trip_id'])
    next_trip, buses = {}, defaultdict(list)
    for r in t['actual_launches']:
        buses[r['bus']].append(r)
    for rows in buses.values():
        for b, c in zip(rows, rows[1:]):
            assert returns[b['trip']]['return_s'] <= c['actual_s']
            next_trip[b['trip']] = c['trip']
    rows_by_stop = {k: group_summary([r for r in labels if direction_name(r['station_id'], r['direction']) == k])
                   for k in sorted({direction_name(r['station_id'], r['direction']) for r in labels})}
    stats = dict(direct=group_summary(labels),
        actual_holding_s=check['actual_holding_s'],
        native_requested_s=sum(ACTIONS[q['native_bin']] for q in requests),
        by_stop_direction=rows_by_stop,
        by_command={k: group_summary([r for r in labels if f"{r['native_s']}->{r['selected_s']}" == k])
                    for k in sorted({f"{r['native_s']}->{r['selected_s']}" for r in labels})},
        post_anchor_same_state_differences=dict(queries=len(same_state),
            supply_minus_blind_s=sum(r['delta_s'] for r in same_state),
            reasons=dict(Counter(r['reason'] for r in same_state)),
            exposure=dict(Counter(r['exposure'] for r in same_state))),
        metrics=check['metrics'])
    close(stats['native_requested_s']+stats['direct']['direct_delta_s'], stats['actual_holding_s'], 'native/direct holding allocation')
    return dict(queries=queries, raw_moments=raw_moments, series=coordinates, stats=stats,
        launches=launches, returns=returns, next_trip=next_trip)


def paired_holding(b, c):
    bq, cq = b['queries'], c['queries']
    groups = defaultdict(lambda: dict(queries=0, actual_delta_s=0., direct_delta_s=0., native_response_s=0.))
    for k in bq.keys() & cq.keys():
        br, cr = bq[k], cq[k]
        delta = cr['issued_s']-br['issued_s']
        name = 'reduced' if delta < 0 else 'increased' if delta > 0 else 'unchanged'
        g = groups[name]
        g['queries'] += 1
        g['actual_delta_s'] += delta
        native = ACTIONS[cr['native_bin']]-ACTIONS[br['native_bin']]
        g['native_response_s'] += native
        g['direct_delta_s'] += delta-native
    only_c = sum(cq[k]['issued_s'] for k in cq.keys()-bq.keys())
    only_b = -sum(bq[k]['issued_s'] for k in bq.keys()-cq.keys())
    total = c['stats']['actual_holding_s']-b['stats']['actual_holding_s']
    close(sum(g['actual_delta_s'] for g in groups.values())+only_c+only_b, total, 'paired commands including unmatched support')
    return dict(matched=len(bq.keys() & cq.keys()), candidate_only=len(cq.keys()-bq.keys()),
        baseline_only=len(bq.keys()-cq.keys()), candidate_only_s=only_c, baseline_only_s=only_b,
        same_bus=sum(cq[k]['identity']['bus_id'] == bq[k]['identity']['bus_id'] for k in bq.keys() & cq.keys()),
        groups=groups, actual_delta_s=total,
        direct_delta_s=c['stats']['direct']['direct_delta_s']-b['stats']['direct']['direct_delta_s'],
        native_response_s=c['stats']['native_requested_s']-b['stats']['native_requested_s'])


def turnover(b, c):
    out = Counter()
    for trip in b['returns'].keys() & c['returns'].keys():
        br, cr = b['returns'][trip], c['returns'][trip]
        out['common_trips'] += 1
        out['holding_delta_s'] += cr['holding_s']-br['holding_s']
        out['return_shift_s'] += cr['return_s']-br['return_s']
        out['other_elapsed_delta_s'] += cr['other_elapsed_s']-br['other_elapsed_s']
        out['same_bus'] += br['bus_id'] == cr['bus_id']
        bn, cn = b['next_trip'].get(trip), c['next_trip'].get(trip)
        if bn is None or cn is None:
            out['no_matched_next_launch'] += 1
        elif bn != cn or br['bus_id'] != cr['bus_id']:
            out['changed_next_trip_or_bus'] += 1
        else:
            shift = c['launches'][cn]['actual_s']-b['launches'][bn]['actual_s']
            cut = cr['holding_s']-br['holding_s']
            out['same_bus_same_next_trip'] += 1
            out['matched_next_launch_shift_s'] += shift
            out['cut_and_next_launch_earlier'] += cut < 0 and shift < 0
            out['cut_but_next_launch_not_earlier'] += cut < 0 and shift >= 0
    return dict(out)


def analyze(root, out, source):
    parent = json.loads((root/'summary/status.json').read_text())
    assert parent['status'] == 'v244_smoke_engineering_passed_formal_closed'
    paths = {}
    for arm in ARMS:
        with gzip.open(root/'records'/f'smoke_c0_{arm}.pkl.gz', 'rb') as stream:
            data = pickle.load(stream)
        paths[arm] = prepare(data, parent['paths'][arm])
        del data
        print('V245_PATH_PREPARED '+arm, flush=True)
    contrasts = {}
    for candidate, base in PAIRS:
        b, c = paths[base], paths[candidate]
        contrast = dict(metric_delta={k: c['stats']['metrics'][k]-b['stats']['metrics'][k]
                                      for k in b['stats']['metrics']},
            holding=paired_holding(b, c), turnover=turnover(b, c), timing={})
        for kind in ('arrivals', 'departures'):
            contrast['timing'][kind] = dict(
                physical_cv2_parts=cv_parts(b['raw_moments'][kind], c['raw_moments'][kind]),
                direct_native_cv2_parts=four_way_parts(b['series'][kind], c['series'][kind]),
                edge_accounting=edge_accounting(b['series'][kind], c['series'][kind]))
        contrasts[candidate+'-'+base] = contrast
    write(out/'diagnostics.json', dict(source=source, contrasts=contrasts,
        arms={arm: p['stats'] for arm, p in paths.items()}))
    summary = dict(status='v245_readonly_holding_propagation_complete', source=source,
        parent_source='b879b53b25', parent_paths=6, new_rollouts=0, training_updates=0,
        controller_changed=False, v238_changed=False, formal_started=False,
        endpoint_and_timing_closures_passed=True,
        arms={a: {k:p['stats'][k] for k in ('direct','post_anchor_same_state_differences')} for a,p in paths.items()},
        contrasts={name: dict(metric_delta=r['metric_delta'],holding=r['holding'],turnover=r['turnover'],
            arrival_cv2_parts=r['timing']['arrivals']['physical_cv2_parts'],
            arrival_direct_native_parts=r['timing']['arrivals']['direct_native_cv2_parts'],
            neighbour_support={k:r['timing']['arrivals']['edge_accounting'][k] for k in
                ('matched_edges','added_edges','removed_edges','added_cv2','removed_cv2','normalization')},
            top_loss_stops=sorted(r['timing']['arrivals']['edge_accounting']['stop_direction'].items(),
                                 key=lambda v:v[1]['cv2_contribution'],reverse=True)[:3])
            for name,r in contrasts.items()},
        limitations='One consumed smoke context/future. Exact timing/command accounting is not a causal '
            'per-action intervention. Native-response sums are evaluated on each changed trajectory,not '
            'an executable alternative action tape. Local restoring labels freeze realized departure '
            'neighbours;they do not predict closed-loop/full-day gains. Same-state requests are not '
            'both executed. No tuning,efficacy CI,selection or policy promotion.')
    write(out/'summary/status.json', summary)
    print('V245_READONLY_COMPLETE '+json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v244-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v245_holding_propagation.py')
    assert suite.countTestCases() > 0
    assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    analyze(a.v244_root, a.out_root, a.source)


if __name__ == '__main__':
    main()
