#!/usr/bin/env python3
"""Read-only V225 quantization, native-response and physical supply-chain audit."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np

from analyze_freqduet_v221_execution import METRICS, differences, read
from analyze_freqduet_v222_holding_response import TERMS, decision_terms, rule_parts
from audit_freqduet_v213_coordination import cv_parts, moments
import freqduet_v223_common_reference as law


PAIRS = (('joint', 'upper'), ('upper', 'v107'), ('joint', 'v107'))
TERM_NAMES = (*TERMS, 'feasibility_s')


def close(a, b, label, tolerance=1e-6):
    assert abs(a-b) <= tolerance, (label, a, b)


def identity(row):
    return row['trip_id'], row['station_id'], bool(row['direction'])


def index(data, sidecar):
    prepares = {p['clock']['handoff_order']: p for p in sidecar['prepares']}
    actions, queries = data['trace']['lower_actions'], sidecar['queries']
    assert len(actions) == len(queries) == len(data['trace']['action_uniforms'])
    rows = {}
    for action, q, (key, uniform) in zip(actions, queries, data['trace']['action_uniforms']):
        p = prepares[q['prepare_order']]
        k = (*identity(action), q['query_occurrence'])
        assert k == tuple(key) and k not in rows, 'decision occurrence join failed'
        assert identity(p['current']) == identity(q['identity']) == identity(action)
        assert p['current']['bus_id'] == q['identity']['bus_id']
        assert action['time_s'] == q['query_s'] == uniform['time_s']
        assert action['context'] == p['context']
        ctx = SimpleNamespace(**p['context'])
        native = law.command(ctx, q['feasible_bins'])
        free = law.command(ctx, range(7))
        phases = tuple(p['phases'][role]['phase_s'] if p['phases'][role]['status']
            in ('supported', 'known_zero') else None for role in ('front', 'current', 'follower'))
        corrected = law.command(ctx, q['feasible_bins'], phases)
        assert corrected == q['shadow'] and native['bin'] == q['native_bin']
        chosen = corrected if data['arm'] == 'joint' else native
        actual = law.ACTIONS[action['action']]
        assert chosen['action_s'] == actual == uniform['action_s']
        rows[k] = dict(key=list(k), order=len(rows), time_s=action['time_s'], packet=p,
            rule=rule_parts(p['context']), native_s=native['action_s'], actual_s=actual,
            override_s=actual-native['action_s'], mask_s=native['action_s']-free['action_s'],
            native=native, corrected=corrected, feasible_bins=q['feasible_bins'])
    measured = data['ledger']['executed_holding_s']
    close(math.fsum(r['actual_s'] for r in rows.values()), measured, 'all decision holding')
    close(math.fsum(r['holding_s'] for r in data['trace']['vehicle_returns']), measured, 'trip holding')
    return rows


def response_terms(current, reference):
    parts = decision_terms(current, reference)
    mask = current['mask_s']-reference['mask_s']
    parts['quantization_s'] -= mask
    parts['feasibility_s'] = mask
    close(parts['native_s'], math.fsum(parts[k] for k in TERM_NAMES[3:]), 'native response terms')
    return parts


def response(current, reference, c_rows, b_rows):
    common = c_rows.keys() & b_rows.keys()
    first = next((k for k in c_rows if c_rows[k]['override_s'] != 0.), None)
    first_order = c_rows[first]['order'] if first is not None else None
    groups, trips, details, quantized = {}, {}, [], []
    counts = Counter()
    for key in sorted(common, key=lambda k: c_rows[k]['order']):
        c, b = c_rows[key], b_rows[key]
        t = response_terms(c, b)
        phase = ('no_direct_request' if first_order is None else 'before_first_direct'
                 if c['order'] < first_order else 'first_direct' if key == first
                 else 'later_direct' if c['override_s'] != 0. else 'later_native')
        group = groups.setdefault(phase, dict(decisions=0, **dict.fromkeys(TERM_NAMES, 0.)))
        group['decisions'] += 1
        trip = trips.setdefault(str(key[0]), dict.fromkeys(TERM_NAMES, 0.))
        for name in TERM_NAMES:
            group[name] += t[name]
            trip[name] += t[name]
        counts['same_physical_bus'] += c['packet']['current']['bus_id'] == b['packet']['current']['bus_id']
        counts['feasible_bins_changed'] += c['feasible_bins'] != b['feasible_bins']
        counts['native_increases'] += t['native_s'] > 0
        counts['native_decreases'] += t['native_s'] < 0
        for role in ('front', 'follower'):
            cr, br = c['packet'][role], b['packet'][role]
            supported = cr['status'] == br['status'] == 'supported'
            counts[role+'_both_supported'] += supported
            counts[role+'_identity_changed'] += supported and any(
                cr[k] != br[k] for k in ('trip_id', 'bus_id', 'direction'))
        if t['net_s'] or t['native_s'] or t['override_s']:
            details.append(dict(key=list(key), phase=phase, current_s=c['time_s'], reference_s=b['time_s'],
                current_context=c['packet']['context'], reference_context=b['packet']['context'], **t))
        if c['override_s']:
            raw = c['corrected']['phase_correction_s']
            quantized.append(dict(key=list(key), time_s=c['time_s'], phase=phase,
                role_phases={r: c['packet']['phases'][r]['phase_s'] for r in ('front', 'current', 'follower')},
                native_raw_s=c['native']['raw_s'], correction_s=raw,
                native_clipped_s=c['native']['clipped_s'], corrected_clipped_s=c['corrected']['clipped_s'],
                native_action_s=c['native_s'], selected_action_s=c['actual_s'], action_delta_s=c['override_s']))
    totals = {name: math.fsum(r[name] for r in trips.values()) for name in TERM_NAMES}
    extra = dict(current_only=len(c_rows.keys()-b_rows.keys()), reference_only=len(b_rows.keys()-c_rows.keys()),
        current_only_holding_s=math.fsum(c_rows[k]['actual_s'] for k in c_rows.keys()-b_rows.keys()),
        reference_only_holding_s=math.fsum(b_rows[k]['actual_s'] for k in b_rows.keys()-c_rows.keys()))
    measured = current['ledger']['executed_holding_s']-reference['ledger']['executed_holding_s']
    close(totals['net_s']+extra['current_only_holding_s']-extra['reference_only_holding_s'], measured, 'matched holding')
    raw_changes = [abs(q['correction_s']) for q in quantized]
    summary = dict(matched=len(common), repeated_occurrences=sum(k[-1] > 0 for k in common),
        **extra, counts=dict(counts), total_holding_delta_s=measured, totals=totals, phases=groups,
        direct_changed_commands=len(quantized),
        direct_correction_abs_s=dict(min=min(raw_changes, default=None), max=max(raw_changes, default=None)),
        direct_bin_delta_histogram=dict(Counter(str(q['action_delta_s']) for q in quantized)))
    return summary, dict(decisions=details, direct_quantization=quantized, per_trip=trips)


def supply(data):
    launches = {r['trip']: r for r in data['trace']['actual_launches']}
    returns = {r['trip_id']: r for r in data['trace']['vehicle_returns']}
    parents, last_return = {}, {}
    for event in data['events']:
        if event['kind'] == 'return':
            row = returns[event['trip_id']]
            assert row['bus_id'] == event['bus_id']
            last_return[event['bus_id']] = row['trip_id']
        elif event['kind'] == 'launch':
            trip, bus = event['trip_id'], event['bus_id']
            assert launches[trip]['bus'] == bus and trip not in parents
            parent = last_return.pop(bus, None)
            if parent is not None:
                assert returns[parent]['return_s'] <= launches[trip]['actual_s']
            parents[trip] = parent
    assert parents.keys() == launches.keys(), 'physical supply event join incomplete'
    close(sum(max(0., q['actual_s']-q['public_baseline_s']) for q in launches.values()),
          data['results']['public_launch_delay_total_s'], 'public delay ledger')
    close(sum(q['actual_s']-q['eligible_s'] for q in launches.values()),
          data['results']['readiness_delay_total_s'], 'readiness ledger')
    return dict(launches=launches, returns=returns, parents=parents, plans=data['launch_plans'])


def supply_compare(c, b):
    assert c['launches'].keys() == b['launches'].keys()
    counts, totals, details = Counter(), Counter(), []
    for trip, row in c['launches'].items():
        old = b['launches'][trip]
        actual = row['actual_s']-old['actual_s']
        plan = c['plans'][str(trip)]-b['plans'][str(trip)]
        parent, previous = c['parents'][trip], b['parents'][trip]
        same_bus = row['bus'] == old['bus']
        same_parent = same_bus and parent is not None and parent == previous
        counts['same_physical_bus'] += same_bus
        counts['same_inbound_trip'] += same_parent
        initial = same_bus and parent is None and previous is None
        counts['initial_stock_both'] += initial
        counts['unmatched_supply'] += not same_parent and not initial
        counts['actual_launch_later'] += actual > 0
        counts['actual_launch_earlier'] += actual < 0
        totals['actual_launch_delta_s'] += actual
        totals['eligibility_delta_s'] += row['eligible_s']-old['eligible_s']
        totals['readiness_delta_s'] += (row['actual_s']-row['eligible_s'])-(old['actual_s']-old['eligible_s'])
        totals['public_delay_delta_s'] += max(0., row['actual_s']-row['public_baseline_s'])-max(0., old['actual_s']-old['public_baseline_s'])
        detail = dict(trip_id=trip, current_bus_id=row['bus'], reference_bus_id=old['bus'],
            current_inbound_trip=parent, reference_inbound_trip=previous, actual_launch_delta_s=actual,
            plan_delta_s=plan, plan_to_actual_gap_delta_s=actual-plan,
            readiness_delta_s=(row['actual_s']-row['eligible_s'])-(old['actual_s']-old['eligible_s']))
        if same_parent:
            cr, br = c['returns'][parent], b['returns'][previous]
            terms = dict(parent_launch_delta_s=cr['launch_s']-br['launch_s'],
                parent_holding_delta_s=cr['holding_s']-br['holding_s'],
                parent_other_elapsed_delta_s=cr['other_elapsed_s']-br['other_elapsed_s'],
                terminal_idle_delta_s=(row['actual_s']-cr['return_s'])-(old['actual_s']-br['return_s']))
            close(math.fsum(terms.values()), actual, 'same-bus supply propagation')
            detail.update(terms)
            if actual or any(terms.values()):
                for name, value in terms.items():
                    totals['same_chain_'+name] += value
                totals['same_chain_launch_delta_s'] += actual
        if actual or plan or not same_parent or any(detail.get(n, 0.) for n in
                ('parent_holding_delta_s', 'parent_other_elapsed_delta_s', 'terminal_idle_delta_s')):
            details.append(detail)
    close(totals['actual_launch_delta_s']-totals['eligibility_delta_s'], totals['readiness_delta_s'], 'readiness decomposition')
    return dict(counts=dict(counts), totals=dict(totals)), details


def analyze_context(ci, joint_root, controls_root, saved, out):
    data, sidecars = {}, {}
    for arm in ('v107', 'upper', 'joint'):
        root = joint_root if arm == 'joint' else controls_root
        data[arm] = read(root/'records'/f'c{ci}_{arm}.json.gz')
        sidecars[arm] = read(root/'sidecars'/f'c{ci}_{arm}.json.gz')
        assert data[arm]['context'] == ci and data[arm]['arm'] == arm
        assert all(data[arm]['results'][k] == v for k, v in saved['outcomes'][arm].items())
        assert data[arm]['results']['trips_unlaunched'] == data[arm]['results']['trips_incomplete'] == 0
    indexed = {a: index(d, sidecars[a]) for a, d in data.items()}
    timing = {a: moments(d['trace']['arrivals']) for a, d in data.items()}
    for arm, m in timing.items():
        close(m['cv2'], data[arm]['results']['cv_squared'], 'actual arrival CV2', 1e-10)
    chains = {a: supply(d) for a, d in data.items()}
    summaries, details = {}, {}
    for current, reference in PAIRS:
        name = current+'_vs_'+reference
        holding, h_detail = response(data[current], data[reference], indexed[current], indexed[reference])
        chain, s_detail = supply_compare(chains[current], chains[reference])
        effect = differences(data[current]['results'], data[reference]['results'])
        close(chain['totals'].get('public_delay_delta_s', 0.), effect['public_launch_delay_total_s'], 'launch delay delta')
        if current == 'joint':
            for metric in METRICS:
                close(effect[metric], saved['joint_minus'][reference][metric], 'V225 endpoint '+metric, 0.)
        if name == 'joint_vs_upper':
            close(holding['total_holding_delta_s'], saved['holding_vs_upper']['total_trip_delta_s'], 'V225 total holding')
            close(holding['totals']['override_s'], saved['execution']['direct_vs_same_state_native_s'], 'V225 direct holding')
            close(holding['totals']['native_s'], saved['holding_vs_upper']['native_state_response_s'], 'V225 native response')
        summaries[name] = dict(effects=effect, holding=holding, supply_chain=chain,
                              arrival_cv2_parts=cv_parts(timing[reference], timing[current]))
        details[name] = dict(holding=h_detail, supply_chain=s_detail)
    (out/'details').mkdir(exist_ok=True)
    (out/'details'/f'c{ci}.json').write_text(json.dumps(details, indent=2)+'\n')
    return dict(context=ci, actor=saved['actor'], demand_day=saved['demand_day'], comparisons=summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-root', type=Path, required=True)
    p.add_argument('--controls-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                             pattern='test_v226_coordination.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    a.out_root.mkdir(parents=True, exist_ok=True)
    protocol = dict(stage='v226_read_only_V225_coordination', source=os.environ['FREQDUET_ANALYSIS_COMMIT'],
        controller_source='f61f6c9734', pairs=PAIRS, contexts=[0, 1, 2, 3],
        inputs='Only existing V225 joint and V224 v107/upper records+sidecars and V225 small summary.',
        analysis='Occurrence-matched phase/native/clipping/quantization/feasibility holding identity; '
            'full-arrival CV2 release/holding/other/covariance decomposition;strict native-event '
            'same-bus inbound-return/next-launch chain with initial/unmatched supply separate.',
        no_new_replays=True, training_updates=0, parameter_changes=0, main_policy_changed=False)
    path = a.out_root/'protocol.json'
    assert not path.exists(), 'preserve the original V226 analysis'
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    prior = read(a.joint_root/'summary/status.json')
    assert prior['source'] == 'f61f6c9734' and prior['completed_checks'] == prior['started_paths'] == 4
    assert prior['all_engineering_passed'] and not prior['main_policy_changed']
    rows, start = [], time.monotonic()
    for ci, saved in enumerate(prior['paths']):
        assert ci == saved['context']
        rows.append(analyze_context(ci, a.joint_root, a.controls_root, saved, a.out_root))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=4, unit='context',
            elapsed_seconds=time.monotonic()-start)), flush=True)
    summary = dict(protocol, status='v226_read_only_coordination_complete', tests=tests.testsRun,
        contexts=rows, v225_endpoints_and_holding_reproduced=True,
        limitations='Four consumed contexts,two actors/two demand days;no independent efficacy or CI. '
            'State/quantization/supply terms are observed-path accounting,not isolated causal effects. '
            'Other elapsed includes movement and nonholding service;no exact per-action passenger '
            'exposure or component split is available in these payloads. No new policy or gain selected.')
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V226_READ_ONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
