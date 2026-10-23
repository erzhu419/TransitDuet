#!/usr/bin/env python3
"""Audit frozen V238 requests on accepted V236 states; no controller imports."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import unittest

from preregister_freqduet_v238_persistent_requests import registration, shadow_request


ARMS = ('v107', 'upper')
JOIN = ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt') as stream:
        json.dump(value, stream, separators=(',', ':'))
        stream.write('\n')


def h_band(h):
    return ('zero' if h == 0 else '(0,45)' if h < 45 else '[45,90)' if h < 90
            else '[90,180)' if h < 180 else '[180,inf)')


def prepare_requests(extension, capture):
    queries, extra = capture['queries'], extension['query_extensions']
    assert capture['complete'] and extension['complete'] and len(queries) == len(extra)
    requests = []
    for q, row in zip(queries, extra):
        assert all(q[k] == row[k] for k in JOIN), 'wrong query occurrence join'
        view = q['view']
        r = shadow_request(view, row)
        # This ablates ONLY the rule's memory input,not the physical trajectory.
        memoryless = shadow_request(dict(view, executed_holding_s=0),
                                   dict(row, executed_holding_before_s=0))
        assert memoryless['shadow_bin'] == r['native_bin'], 'H=0 differs from native'
        feasible = [c['bin'] for c in row['shadow_frontier']['candidates']]
        assert r['shadow_bin'] in feasible
        changed = r['shadow_bin'] != r['native_bin']
        if changed:
            assert 0 < r['shadow_s'] < r['native_s'] and r['local_gain_vs_zero_s2'] > 0
        identity = q['identity']
        assert view['cycle_key'][:2] == [identity['bus_id'], identity['trip_id']]
        own = view['own_association']['obligation']
        requests.append(dict(r, key=[identity['trip_id'], identity['station_id'],
            bool(identity['direction']), q['query_occurrence']],
            query_s=view['query_s'], native_event_seq=view['native_event_seq'],
            prepare_order=q['prepare_order'], query_order=q['query_order'],
            native_frontier_status=row['shadow_frontier']['status'], feasible_bins=feasible,
            full_frontier=row['shadow_frontier']['candidates'],
            message_refs=view['message_refs'], pending_plan_version_refs=view['pending_plan_version_refs'],
            own_plan_version=None if own is None else own['last_write_version'],
            position_rank=view['own_association']['distance_rank'],
            memoryless_bin=memoryless['shadow_bin'], changed=changed,
            requested_reduction_s=r['native_s']-r['shadow_s'], H_band=h_band(r['executed_holding_s'])))
    assert len({tuple(r['key']) for r in requests}) == len(requests), 'duplicate request occurrence'
    return requests


def tally(rows):
    changed = [r for r in rows if r['changed']]
    supported = [r for r in rows if r['reason'] == 'persistent_compensation_request']
    loss = math.fsum(r['local_gap_delta_vs_native_s2'] for r in changed)
    retained = math.fsum(r['local_gain_vs_zero_s2'] for r in changed)
    return dict(queries=len(rows), native_positive=sum(r['native_s'] > 0 for r in rows),
        supported=len(supported), changed=len(changed),
        cycles=len({tuple(r['cycle_key']) for r in rows}),
        changed_cycles=len({tuple(r['cycle_key']) for r in changed}),
        native_requests_s=math.fsum(r['native_s'] for r in rows),
        shadow_requests_on_native_states_s=math.fsum(r['shadow_s'] for r in rows),
        requested_reduction_s=math.fsum(r['requested_reduction_s'] for r in rows),
        local_R_loss_vs_native_on_changed_s2=loss,
        retained_local_gain_fraction_on_changed=None if not changed else retained/(retained+loss),
        changed_at_compensation_floor=sum(r['shadow_s'] == r['compensation_floor_s'] for r in changed),
        memoryless_changed=sum(r['memoryless_bin'] != r['native_bin'] for r in rows))


def groups(rows, field):
    grouped = defaultdict(list)
    for r in rows:
        grouped[str(r[field])].append(r)
    return {k: tally(v) for k, v in sorted(grouped.items())}


def label_rows(requests, labels):
    by_key = {tuple(r['key']): r for r in labels['decisions']}
    assert len(by_key) == len(labels['decisions']) == len(requests)
    assert by_key.keys() == {tuple(r['key']) for r in requests}
    result = []
    for r in requests:
        label = by_key[tuple(r['key'])]
        assert (label['bus_id'], label['query_s'], label['H_prefix_s'], label['holding_s'],
                label['own_obligation']) == (r['cycle_key'][0], r['query_s'], r['executed_holding_s'],
                                            r['native_s'], r['obligation_trip']), 'label join mismatch'
        result.append(dict(r, next_clock_label=label['next_clock'],
            native_fixed_neighbour_label=label['fixed']['kind'],
            association_match_label=('unavailable' if label['next_label_matches'] is None else
                                     'match' if label['next_label_matches'] else 'mismatch')))
    return result


def supply_groups(requests, cycles):
    by_cycle = defaultdict(list)
    for r in requests:
        by_cycle[tuple(r['cycle_key'][:2])].append(r)
    labelled_cycles = {(c['bus_id'], c['trip_id']) for c in cycles}
    assert len(labelled_cycles) == len(cycles) and by_cycle.keys() <= labelled_cycles
    grouped = defaultdict(list)
    for c in cycles:
        qs = by_cycle[c['bus_id'], c['trip_id']]
        assert math.fsum(r['native_s'] for r in qs) == c['holding_s']
        grouped[c['clock']['kind']].append((c, qs))
    result = {}
    for name, rows in sorted(grouped.items()):
        labels = [c for c, _ in rows if c['next_trip'] is not None]
        result[name] = dict(cycles=len(rows), next_launch_labels=len(labels),
            changed_cycles=sum(any(r['changed'] for r in qs) for _, qs in rows),
            native_executed_holding_s=math.fsum(c['holding_s'] for c, _ in rows),
            requested_reduction_s=math.fsum(r['requested_reduction_s'] for _, qs in rows for r in qs),
            original_next_public_delay_s=None if not labels else
                math.fsum(c['next_public_delay_s'] for c in labels))
    return result


def read_evaluation(out, label_path):
    seal = read(out/'requests_sealed.json')
    assert seal['arms'] == list(ARMS) and seal['evaluation_labels_read'] == 0
    assert all((out/'requests'/f'c0_{arm}.json.gz').is_file() for arm in ARMS)
    return read(label_path)


def protocol(source):
    return dict(stage='v239_read_only_frozen_shadow_audit', source=source,
        rule_source='b4b1d00203', collection_source='6972500d67', evaluation_source='1f71d9678a',
        actor=12037, demand_day=2151001, arms=list(ARMS),
        budget=dict(query_input_archives=4, saved_label_archives=2, audited_paths=2,
            new_replays=0, new_training=0, model_fits=0, new_futures=0, executed_actions=0),
        order='Compute ALL requests for both paths from accepted V236 resource/cycle query '
            'inputs and save them before reading V237 evaluation details. No label affects requests.',
        strata='All queries and reasons,exposure,seven-bin transitions,H=0,(0,45),[45,90),'
            '[90,180),[180,inf);support and local R loss are descriptive,not tuning criteria.',
        ablation='Same query state and frontier,H=0 only as rule-input ablation;never rewrite '
            'the physical H ledger or apply hypothetical reductions to later queries.',
        labels='Existing same-bus next-launch/timing and native fixed-neighbour-vs-zero '
            'diagnostics only. Count supply cycles once;missing next labels stay null. '
            'Native fixed-neighbour labels are NOT candidate action-difference evidence.',
        decision='Zero supported/changed requests stops progression. Nonzero differences '
            'only admit designing a separately budgeted passive/matched-future protocol; '
            'no tuning or automatic execution. Report compensation loss and any saturation.',
        scope='One consumed actor/day,two reference trajectories. Summed reductions are '
            'requested dose on unchanged native states,NOT realized holding/passenger/CV2/'
            'readiness/launch-delay gains. No efficacy CI or policy promotion.',
        runtime_connected=False, main_policy_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'evaluation-root', 'sealed-rule', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    status = a.out_root/'summary/status.json'
    if status.exists():
        previous = read(status)
        assert previous['source'] == a.source and previous['status'] == 'v239_shadow_audit_complete'
        print('V239_EXISTING_ANALYSIS_REUSED_NO_RESCAN', flush=True)
        return
    assert read(a.sealed_rule) == registration('b4b1d00203'), 'frozen rule changed'
    parent = read(a.collection_root/'summary/status.json')
    assert parent['source'] == '6972500d67' and parent['status'] == 'v236_passive_pair_passed'
    assert parent['started_paths'] == parent['checked_paths'] == 2 and all(r['passed'] for r in parent['paths'])
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v239_shadow_requests.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    a.out_root.mkdir(parents=True, exist_ok=False)
    spec = protocol(a.source)
    write(a.out_root/'protocol.json', spec)
    proposals = {}
    for arm in ARMS:
        name = f'c0_{arm}.json.gz'
        extension = read(a.collection_root/'resource_sidecars'/name)
        capture = read(a.collection_root/'cycle_sidecars'/name)
        proposals[arm] = prepare_requests(extension, capture)
        assert len(proposals[arm]) == 5200
        write(a.out_root/'requests'/name, proposals[arm])
        del extension, capture
        print('V239_REQUESTS_SAVED '+arm, flush=True)
    write(a.out_root/'requests_sealed.json', dict(source=a.source, rule_source='b4b1d00203',
        arms=list(ARMS), counts={k: len(v) for k, v in proposals.items()}, evaluation_labels_read=0))
    evaluated = read(a.evaluation_root/'summary/status.json')
    assert evaluated['source'] == '1f71d9678a' and evaluated['status'] == 'v237_read_only_supply_evidence_complete'
    paths = {}
    for arm in ARMS:
        labels = read_evaluation(a.out_root, a.evaluation_root/'details'/f'c0_{arm}.json.gz')
        rows = label_rows(proposals[arm], labels)
        paths[arm] = dict(totals=tally(rows), reasons=dict(Counter(r['reason'] for r in rows)),
            exposure=groups(rows, 'exposure'), H_bands=groups(rows, 'H_band'),
            transition_counts=dict(Counter(f"{r['native_s']}->{r['shadow_s']}" for r in rows)),
            next_clock_labels=groups(rows, 'next_clock_label'),
            native_fixed_neighbour_labels=groups(rows, 'native_fixed_neighbour_label'),
            association_labels=groups(rows, 'association_match_label'),
            unique_supply=supply_groups(rows, labels['cycles']))
        write(a.out_root/'details'/f'c0_{arm}.json.gz', rows)
    has_support = all(paths[arm]['totals']['supported'] > 0 and paths[arm]['totals']['changed'] > 0 for arm in ARMS)
    summary = dict(status='v239_shadow_audit_complete', source=a.source, rule_source='b4b1d00203',
        tests=tested.testsRun, paths=paths, actor=12037, demand_day=2151001,
        query_archives_read=4, evaluation_archives_read=2, requests_saved_before_evaluation=True,
        new_replays=0, training_updates=0, model_fits=0, shadow_actions_executed=0,
        new_futures=0, runtime_connected=False, main_policy_changed=False,
        nonzero_shadow_support_and_differences=has_support,
        decision='design_only_next_protocol' if has_support else 'stop_no_support_or_difference', scope=spec['scope'])
    write(status, summary)
    compact = {k: v for k, v in summary.items() if k != 'paths'}
    compact['paths'] = {arm: {k: paths[arm][k] for k in
        ('totals', 'reasons', 'transition_counts', 'unique_supply')} for arm in ARMS}
    for arm in ARMS:
        for field in ('exposure', 'H_bands', 'native_fixed_neighbour_labels'):
            compact['paths'][arm][field] = {name: {k: row[k] for k in
                ('queries', 'changed', 'requested_reduction_s', 'changed_at_compensation_floor',
                 'retained_local_gain_fraction_on_changed')} for name, row in paths[arm][field].items()}
    write(a.out_root/'summary/compact.json', compact)
    print('V239_COMPLETE '+str(a.out_root/'summary/compact.json'), flush=True)


if __name__ == '__main__':
    main()
