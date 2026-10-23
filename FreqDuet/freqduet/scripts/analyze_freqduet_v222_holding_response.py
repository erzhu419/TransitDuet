#!/usr/bin/env python3
"""Frozen, read-only own-trip holding response and proxy audit on V221 records."""

import argparse
from collections import Counter, defaultdict, deque
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from analyze_freqduet_v221_execution import ACTIONS, ARMS, read


PAIRS = (('lower', 'v107'), ('joint', 'v107'), ('joint', 'upper'), ('joint', 'lower'))
TERMS = ('native_s', 'override_s', 'net_s', 'forward_s', 'follower_s',
         'fallback_target_s', 'branch_change_s', 'clipping_s', 'quantization_s')


def identity(row):
    return row['trip_id'], row['station_id'], bool(row['direction'])


def unique(rows):
    result = {identity(r): r for r in rows}
    assert len(result) == len(rows), 'timing event identity is not unique'
    return result


def rule_parts(context):
    if not context['evidence_valid']:
        mode, forward, follower, target = 'no_forward', 0., 0., 0.
    elif context['follower_evidence_valid']:
        mode, forward, follower, target = 'bilateral', -.125*context['forward_headway_s'], \
            .125*context['follower_departure_gap_s'], 0.
    else:
        mode, forward, follower, target = 'fallback', -.25*context['forward_headway_s'], \
            0., .25*context['target_headway_s']
    raw = forward+follower+target
    clipped = min(45., max(0., raw))
    bin_index = min(range(len(ACTIONS)), key=lambda b: abs(ACTIONS[b]-clipped))
    return dict(mode=mode, forward=forward, follower=follower, target=target,
                raw=raw, clipped=clipped, bin=bin_index)


def index_decisions(data):
    """Attach query/prepare stamps and responses without collapsing occurrences."""
    responses = defaultdict(deque)
    for r in data['responses']:
        responses[(*identity(r), r['time_s'])].append(r)
    prepared, queries = {}, []
    for e in data['events']:
        if e['kind'] == 'prepare':
            prepared[(*identity(e), e['bus_id'])] = e
        elif e['kind'] == 'lower_query':
            p = prepared[(*identity(e), e['bus_id'])]
            assert p['event_seq'] < e['event_seq'] and p['time_s'] <= e['time_s']
            queries.append((e, p))
    actions, uniforms = data['trace']['lower_actions'], data['trace']['action_uniforms']
    assert len(actions) == len(queries) == len(uniforms)
    occurrence, result = Counter(), {}
    tasks = {t['trip_id'] for t in data['tasks']}
    for action, (event, prepare), (uniform_key, uniform) in zip(actions, queries, uniforms):
        base = identity(action)
        key = (*base, occurrence[base])
        occurrence[base] += 1
        assert key == tuple(uniform_key) and action['time_s'] == uniform['time_s'] == event['time_s']
        assert identity(event) == base and ACTIONS[action['action']] == uniform['action_s']
        bucket = responses[(*base, action['time_s'])]
        response = bucket.popleft() if bucket else None
        if action['trip_id'] in tasks:
            assert response is not None, 'task native request was not recorded'
        native = response['base_bin'] if response is not None else action['action']
        if response is not None:
            assert response['event_seq'] == event['event_seq'] and response['selected_bin'] == action['action']
        parts = rule_parts(action['context'])
        assert native == parts['bin'], 'frozen unmasked bilateral025 request failed reconstruction'
        result[key] = dict(action, native_s=ACTIONS[native], actual_s=ACTIONS[action['action']],
            override_s=ACTIONS[action['action']]-ACTIONS[native], rule=parts, response=response,
            prepare_s=prepare['time_s'], query_event_seq=event['event_seq'], order=len(result))
    assert not any(responses.values()), 'recorded response has no policy decision'
    return result


def decision_terms(current, reference):
    c, b = current['rule'], reference['rule']
    native = current['native_s']-reference['native_s']
    override = current['override_s']-reference['override_s']
    result = dict.fromkeys(TERMS, 0.)
    result.update(native_s=native, override_s=override, net_s=current['actual_s']-reference['actual_s'],
        clipping_s=(c['clipped']-c['raw'])-(b['clipped']-b['raw']),
        quantization_s=(current['native_s']-c['clipped'])-(reference['native_s']-b['clipped']))
    if c['mode'] == b['mode']:
        result.update(forward_s=c['forward']-b['forward'], follower_s=c['follower']-b['follower'],
                      fallback_target_s=c['target']-b['target'])
    else:
        result['branch_change_s'] = c['raw']-b['raw']
    assert abs(native-sum(result[k] for k in TERMS[3:])) <= 1e-8
    assert result['net_s'] == native+override
    return result


def compare(current, reference, c_index, b_index):
    tasks = {t['trip_id'] for t in reference['tasks']}
    assert tasks == {t['trip_id'] for t in current['tasks']}
    c = {k: r for k, r in c_index.items() if k[0] in tasks}
    b = {k: r for k, r in b_index.items() if k[0] in tasks}
    common = c.keys() & b.keys()
    first = {}
    for key in sorted(common, key=lambda k: c[k]['order']):
        if c[key]['override_s'] or b[key]['override_s']:
            first.setdefault(key[0], c[key]['order'])
    phases, details = {}, []
    for key in sorted(common, key=lambda k: c[k]['order']):
        cr, br = c[key], b[key]
        start = first.get(key[0])
        phase = ('no_override' if start is None else 'before_first' if cr['order'] < start else
                 'first' if cr['order'] == start else 'later_override' if cr['override_s'] or br['override_s']
                 else 'later_native')
        terms = decision_terms(cr, br)
        group = phases.setdefault(phase, dict(decisions=0, **dict.fromkeys(TERMS, 0.)))
        group['decisions'] += 1
        for name in TERMS:
            group[name] += terms[name]
        details.append(dict(key=list(key), phase=phase, current_time_s=cr['time_s'], reference_time_s=br['time_s'],
            current_context=cr['context'], reference_context=br['context'], **terms))
    total = {name: math.fsum(r[name] for r in details) for name in TERMS}
    extra_c = math.fsum(c[k]['actual_s'] for k in c.keys()-b.keys())
    extra_b = math.fsum(b[k]['actual_s'] for k in b.keys()-c.keys())
    c_hold = {r['trip_id']: r['holding_s'] for r in current['trace']['vehicle_returns']}
    b_hold = {r['trip_id']: r['holding_s'] for r in reference['trace']['vehicle_returns']}
    measured = math.fsum(c_hold[t]-b_hold[t] for t in tasks)
    untraced_c = math.fsum(c_hold[t] for t in tasks)-math.fsum(r['actual_s'] for r in c.values())
    untraced_b = math.fsum(b_hold[t] for t in tasks)-math.fsum(r['actual_s'] for r in b.values())
    residual = untraced_c-untraced_b
    assert abs(measured-total['net_s']-extra_c+extra_b-residual) <= 1e-6
    return dict(task_trips=len(tasks), matched_decisions=len(common),
        repeated_occurrences=sum(k[-1] > 0 for k in common), current_only_decisions=len(c.keys()-b.keys()),
        reference_only_decisions=len(b.keys()-c.keys()), current_only_holding_s=extra_c,
        reference_only_holding_s=extra_b, untraced_holding_delta_s=residual,
        task_holding_delta_s=measured, totals=total, phases=phases, details=details)


def loss_delta(gap, target, base, selected):
    return ((gap+selected-target)**2-(gap+base-target)**2)/target**2


def fixed_neighbour_delta(forward, backward, target, extra):
    if backward < extra:
        return None
    return .5*(forward**2+backward**2-(forward+extra)**2-(backward-extra)**2)/target**2


def proxy_audit(data, index):
    arrivals, departures = unique(data['trace']['arrivals']), unique(data['trace']['departures'])
    launches = {r['trip']: r for r in data['trace']['actual_launches']}
    arrival_previous, previous = {}, {}
    for row in data['trace']['arrivals']:
        group = row['station_id'], row['direction']
        arrival_previous[identity(row)] = previous.get(group)
        previous[group] = row
    neighbours, groups = {}, defaultdict(list)
    for row in data['trace']['departures']:
        groups[row['station_id'], row['direction']].append(row)
    for rows in groups.values():
        for i, r in enumerate(rows):
            neighbours[identity(r)] = (rows[i-1] if i else None, rows[i+1] if i+1 < len(rows) else None)
    details = []
    for key, r in index.items():
        if r['override_s'] == 0:
            continue
        response, ctx = r['response'], r['context']
        assert response is not None and ctx['evidence_valid'] and ctx['follower_evidence_valid']
        before = arrival_previous[key[:3]]
        assert before is not None
        predecessor = departures[identity(before)]
        assert predecessor['time_s'] <= r['prepare_s'] <= r['time_s']
        native, chosen, target = r['native_s'], r['actual_s'], ctx['target_headway_s']
        proxy = response['release_holding_gap_s']
        reconstructed = launches[key[0]]['actual_s']-predecessor['release']+response['executed_before_s']-predecessor['realized_holding']
        observed = r['prepare_s']-predecessor['time_s']
        nonholding_gap = r['prepare_s']-launches[key[0]]['actual_s']-response['executed_before_s']-predecessor['other_elapsed']
        assert abs(reconstructed-proxy) <= 1e-6 and abs(observed-ctx['forward_headway_s']) <= 1e-6
        assert abs(observed-proxy-nonholding_gap) <= 1e-6
        p_loss = loss_delta(proxy, target, native, chosen)
        f_loss = loss_delta(observed, target, native, chosen)
        follower = ctx['follower_departure_gap_s']
        b_loss = .5*(f_loss+((max(0., follower-chosen)-target)**2-
                            (max(0., follower-native)-target)**2)/target**2)
        assert p_loss <= 1e-10, 'recorded reduction violates frozen release proxy'
        current = departures[key[:3]]
        assert current['realized_holding']-response['executed_before_s'] == chosen
        prev, following = neighbours[key[:3]]
        fixed = None if prev is None or following is None else fixed_neighbour_delta(
            current['time_s']-prev['time_s'], following['time_s']-current['time_s'], target, native-chosen)
        details.append(dict(key=list(key), query_s=r['time_s'], prepare_s=r['prepare_s'],
            proxy_gap_s=proxy, observed_forward_s=observed, nonholding_gap_s=nonholding_gap,
            query_lag_s=r['time_s']-r['prepare_s'], proxy_loss_delta=p_loss,
            forward_loss_delta=f_loss, estimated_bilateral_loss_delta=b_loss,
            fixed_actual_neighbour_loss_delta=fixed,
            predecessor_changed_before_departure=prev is None or prev['trip_id'] != before['trip_id'],
            departure_clock_residual_s=current['time_s']-r['prepare_s']-chosen))
    stats = dict(changed_decisions=len(details), completed_prefix_reconstruction_exact=True,
        observed_forward_reconstruction_exact=True, proxy_nonharm=sum(r['proxy_loss_delta'] <= 0 for r in details),
        forward_harm=sum(r['forward_loss_delta'] > 0 for r in details),
        estimated_bilateral_harm=sum(r['estimated_bilateral_loss_delta'] > 0 for r in details),
        fixed_actual_neighbour_supported=sum(r['fixed_actual_neighbour_loss_delta'] is not None for r in details),
        fixed_actual_neighbour_harm=sum(r['fixed_actual_neighbour_loss_delta'] is not None
                                      and r['fixed_actual_neighbour_loss_delta'] > 0 for r in details),
        changed_departure_predecessor=sum(r['predecessor_changed_before_departure'] for r in details))
    for name in ('nonholding_gap_s', 'query_lag_s', 'departure_clock_residual_s'):
        values = [r[name] for r in details]
        stats[name] = dict(mean=math.fsum(values)/len(values) if values else None,
                          min=min(values, default=None), max=max(values, default=None))
    stats['details'] = details
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    frozen = a.out_root/'protocol.json'
    assert not frozen.exists(), 'preserve the sealed diagnostic run'
    a.out_root.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        protocol='md/protocol_v6_v222_holding_response.md', pairs=PAIRS,
        new_replays=0, training_updates=0, parameter_changes=0, main_policy_changed=False), indent=2)+'\n')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v222_holding_response.py', '-v'], check=True)
    prior = read(a.root/'read_only_audit_42a75cc2ab/summary/status.json')
    contexts = []
    for ci in range(4):
        data = {arm: read(a.root/'records'/f'c{ci}_{arm}.json.gz') for arm in ARMS}
        indices = {arm: index_decisions(d) for arm, d in data.items()}
        pairs = {f'{c}_vs_{b}': compare(data[c], data[b], indices[c], indices[b]) for c, b in PAIRS}
        proxies = {arm: proxy_audit(data[arm], indices[arm]) for arm in ('lower', 'joint')}
        for arm in ('lower', 'joint'):
            old = prior['contexts'][ci]['holding'][arm]
            terms = pairs[f'{arm}_vs_v107']
            assert terms['task_holding_delta_s'] == old['task_trip_holding_delta_s']
            assert -terms['totals']['override_s'] == old['direct_decrement_vs_live_bilateral_s']
        detail = a.out_root/'details'
        detail.mkdir(exist_ok=True)
        (detail/f'c{ci}.json').write_text(json.dumps(dict(comparisons=pairs, proxies=proxies), indent=2)+'\n')
        row = dict(context=ci, actor=prior['contexts'][ci]['actor'], day=prior['contexts'][ci]['day'],
            comparisons={k: {name: value for name, value in v.items() if name != 'details'} for k, v in pairs.items()},
            proxy_audits={k: {name: value for name, value in v.items() if name != 'details'} for k, v in proxies.items()})
        contexts.append(row)
        print('V222_CONTEXT', json.dumps(dict(context=ci, comparisons={k: v['totals'] for k, v in pairs.items()},
                                             proxy_audits=row['proxy_audits'])), flush=True)
    summary = dict(status='v222_read_only_holding_response_complete', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        contexts=contexts, prior_holding_totals_exact=True, new_replays=0, training_updates=0,
        parameter_changes=0, main_policy_changed=False,
        limitations='Four consumed contexts,two demand days. State/quantization terms are identities on '
            'different closed-loop states,not independent causal effects. Estimated follower identity is '
            'not archived here;actual neighbour labels do not identify same-follower ETA error. Fixed-neighbour '
            'loss is a frozen-tape perturbation,not a rollout of the unexecuted action. No policy is selected.')
    output = a.out_root/'summary'
    output.mkdir(exist_ok=True)
    (output/'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V222_HOLDING_RESPONSE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
