#!/usr/bin/env python3
"""Read two V249 paths under V251; extract causal prefixes before supply labels."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path
from statistics import mean, median
import unittest

from analyze_freqduet_v233_cycle_state import label_sample
from preregister_freqduet_v229_supply_coordination import ACTIONS
from preregister_freqduet_v251_early_supply import link_prefix, registration


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def save_detail(path, value):
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, separators=(',', ':'))


def distribution(values):
    return dict(n=len(values), sum=sum(values), mean=mean(values) if values else None,
                median=median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def sample(view):
    own = view['own_association']
    return dict(native_event_seq=view['native_event_seq'], query_s=view['query_s'],
        cycle_key=list(view['cycle_key']), H_s=view['executed_holding_s'],
        association=copy.deepcopy(own['obligation']), position_rank=own['distance_rank'],
        incoming_current=[dict(bus_id=b['bus_id'], trip_id=b['inbound_trip_id'])
                          for b in view['target_profile']['position_associations']])


def extract_prefixes(queries, versions):
    """No returns, next-launch identities, endpoint metrics or future doses enter."""
    writes = sorted(v['native_event_seq'] for v in versions if not v['initial'] and v['changes'])
    states, active, cycles = {}, {}, {}
    episodes, intervals, samples = [], [], []
    for q in queries:
        view = q['view']
        key = tuple(view['cycle_key'])
        previous = states.get(key)
        state = link_prefix(previous, view)
        snap = sample(view)
        cycle = cycles.setdefault(key, dict(cycle_key=list(key), queries=0, first_query=snap, first_link=None,
            first_due=None, first_due_episode=None, first_link_episode=None, last=None))
        if state['transition'] in ('lost', 'reassociated'):
            old = active.pop(key)
            old['end_reason'] = state['transition']
            old['closing_query_seq'] = state['query_seq']
        if state['episode'] is not None and state['transition'] in ('opened', 'reassociated'):
            e = dict(episode_id=len(episodes), cycle_key=list(key), start=snap, last=snap,
                first_due=None, queries=0, own_version_changes=0, rank_changes=0,
                crossed_write_intervals=0, internal_H_s=0., end_reason='last_observed_query')
            episodes.append(e)
            active[key] = e
        e = active.get(key)
        if e is not None:
            if previous is not None and state['transition'] not in ('opened', 'reassociated'):
                e['own_version_changes'] += state['transition'] == 'revised'
                e['rank_changes'] += previous['current']['distance_rank'] != state['current']['distance_rank']
            e['last'] = snap
            e['queries'] += 1
            if state['own_due'] and e['first_due'] is None:
                e['first_due'] = snap
        if previous is not None:
            before, seq = previous['query_seq'], state['query_seq']
            nwrites = bisect_left(writes, seq)-bisect_left(writes, before)
            same = previous['episode'] is not None and state['episode'] is not None and state['transition'] not in ('opened', 'reassociated')
            kind = ('same_plan_episode' if same else 'association_boundary'
                    if state['transition'] in ('lost', 'opened', 'reassociated') else 'both_unassociated')
            delta = state['H_s']-previous['H_s']
            row = dict(cycle_key=list(key), previous_query_seq=before, query_seq=seq,
                kind=kind, H_delta_s=delta, writes=nwrites,
                rank_changed=previous['current']['distance_rank'] != state['current']['distance_rank'],
                own_version_changed=state['transition'] == 'revised',
                due_pressure_cleared=previous['target_due_present'] and not state['target_due_present'])
            intervals.append(row)
            if same:
                e['internal_H_s'] += delta
                e['crossed_write_intervals'] += nwrites > 0
        if state['episode'] is not None and cycle['first_link'] is None:
            cycle['first_link'], cycle['first_link_episode'] = snap, e['episode_id']
        if state['own_due'] and cycle['first_due'] is None:
            cycle['first_due'], cycle['first_due_episode'] = snap, e['episode_id']
        cycle.update(queries=cycle['queries']+1, last=snap)
        states[key] = state
        samples.append(snap)
    return dict(cycles=list(cycles.values()), episodes=episodes, intervals=intervals, samples=samples)


def label_tally(labels):
    counts = Counter(r['status'] for r in labels)
    available = counts['match']+counts['mismatch']
    return dict(samples=len(labels), status=dict(counts),
        match_fraction=counts['match']/available if available else None,
        distance_rank_labelled=sum(r['distance_rank_correct'] is not None for r in labels),
        distance_rank_correct=sum(r['distance_rank_correct'] is True for r in labels),
        mismatch_reasons=dict(Counter(r['mismatch_reason'] for r in labels if r['status'] == 'mismatch')))


def evaluate(prefix, capture, resource):
    """Read future labels and actual closed H only after prefix extraction."""
    closed = {tuple(c['cycle_key']): c for c in capture['cycles']}
    cycles = {tuple(c['cycle_key']): c for c in prefix['cycles']}
    assert cycles.keys() <= closed.keys()
    assert not capture['terminal']['open_cycles']
    assert all(c['status'] == 'returned' for c in closed.values())
    next_by_parent = {(r['bus_id'], r['inbound_trip_id']): r['next_trip_id']
                      for r in resource['supply_outcomes'] if r['inbound_trip_id'] is not None}
    observed = {tuple(r['previous']['cycle_key'][:2]): r['current_cycle'][1]
                for r in capture['observed_links']}
    assert next_by_parent == observed, 'first-next same-bus label join differs'
    launches = {r['trip_id']: r for r in resource['cycle_events'] if r['kind'] == 'launch'}
    returns = {(r['bus_id'], r['trip_id']): r for r in resource['cycle_events'] if r['kind'] == 'return'}

    def label(s):
        return label_sample(s, next_by_parent.get(tuple(s['cycle_key'][:2])), launches, returns)

    first = [c['first_link'] for c in cycles.values() if c['first_link'] is not None]
    first_due = [c['first_due'] for c in cycles.values() if c['first_due'] is not None]
    early = [e for e in prefix['episodes'] if e['start']['association']['kind'] == 'future']
    to_due = [e for e in early if e['first_due'] is not None]
    labels = dict(first_cycle=[label(s) for s in first], first_own_due=[label(s) for s in first_due],
        first_future_cycle=[label(s) for s in first if s['association']['kind'] == 'future'],
        all_episode_starts=[label(e['start']) for e in prefix['episodes']],
        future_episode_starts=[label(e['start']) for e in early],
        future_to_due_episode_starts=[label(e['start']) for e in to_due],
        all_query_exposures=[label(s) for s in prefix['samples']])
    due_rows = []
    for c in cycles.values():
        if c['first_due'] is None:
            continue
        a, b, ret = c['first_link'], c['first_due'], closed[tuple(c['cycle_key'])]
        due_rows.append(dict(cycle_key=c['cycle_key'],
            first_link_kind=a['association']['kind'],
            same_episode=c['first_link_episode'] == c['first_due_episode'],
            same_plan=a['association']['trip_id'] == b['association']['trip_id'],
            lead_s=b['query_s']-a['query_s'], H_at_first_link_s=a['H_s'],
            H_until_first_due_s=b['H_s']-a['H_s'], H_after_first_due_s=ret['executed_holding_s']-b['H_s']))
    first_prefix = sum(c['first_query']['H_s'] for c in cycles.values())
    tails = sum(closed[k]['executed_holding_s']-c['last']['H_s'] for k, c in cycles.items())
    zero_query = [c for k, c in closed.items() if k not in cycles]
    total = sum(c['executed_holding_s'] for c in closed.values())
    assert all(closed[k]['executed_holding_s'] >= c['last']['H_s'] for k, c in cycles.items())
    groups = defaultdict(list)
    for r in prefix['intervals']:
        groups[r['kind']].append(r)
    h_intervals = {name: dict(intervals=len(rows), cycles=len({tuple(r['cycle_key']) for r in rows}),
        H_s=sum(r['H_delta_s'] for r in rows), write_intervals=sum(r['writes'] > 0 for r in rows),
        rank_changes=sum(r['rank_changed'] for r in rows), own_version_changes=sum(r['own_version_changed'] for r in rows),
        due_pressure_cleared=sum(r['due_pressure_cleared'] for r in rows))
        for name, rows in sorted(groups.items())}
    assert first_prefix+sum(r['H_s'] for r in h_intervals.values())+tails+sum(c['executed_holding_s'] for c in zero_query) == total
    # V250 already certified every native command against actual execution on these paths.
    assert len(resource['query_extensions']) == len(prefix['samples'])
    doses = defaultdict(lambda: Counter(queries=0, positive=0, certified_executed_H_s=0., local_known=0, local_benefit_vs_zero=0))
    for s, row in zip(prefix['samples'], resource['query_extensions']):
        assert s['native_event_seq'] == row['snapshot_event_seq'] and s['cycle_key'] == row['cycle_ref']
        frontier = row['shadow_frontier']
        h = ACTIONS[frontier['native_bin']]
        group = 'unassociated' if s['association'] is None else s['association']['kind']
        zero = next((c['local_gap_cost_delta_s2'] for c in frontier['candidates'] if c['holding_s'] == 0), None)
        doses[group].update(queries=1, positive=h > 0, certified_executed_H_s=h,
            local_known=h > 0 and frontier['status'] == 'supported',
            local_benefit_vs_zero=h > 0 and zero is not None and zero > 0)
    assert sum(r['certified_executed_H_s'] for r in doses.values()) == total
    episodes = prefix['episodes']
    summary = dict(cycles=dict(total=len(closed), queried=len(cycles), no_query=len(zero_query),
            returned_without_next_launch=sum(tuple(k[:2]) not in next_by_parent for k in closed),
            with_link=len(first), never_link=len(cycles)-len(first), with_own_due=len(first_due),
            never_own_due=len(cycles)-len(first_due), first_link_kind=dict(Counter(s['association']['kind'] for s in first))),
        first_link_to_first_due=dict(cycles=len(due_rows), same_episode=sum(r['same_episode'] for r in due_rows),
            same_plan=sum(r['same_plan'] for r in due_rows), lead_s=distribution([r['lead_s'] for r in due_rows]),
            H_at_first_link_s=sum(r['H_at_first_link_s'] for r in due_rows),
            H_until_first_due_s=sum(r['H_until_first_due_s'] for r in due_rows),
            H_after_first_due_s=sum(r['H_after_first_due_s'] for r in due_rows)),
        future_to_due_same_episode=dict(episodes=len(to_due),
            lead_s=distribution([e['first_due']['query_s']-e['start']['query_s'] for e in to_due]),
            actual_prefix_H_s=distribution([e['first_due']['H_s']-e['start']['H_s'] for e in to_due])),
        episodes=dict(total=len(episodes), future_start=len(early),
            future_never_due=len(early)-len(to_due), ends=dict(Counter(e['end_reason'] for e in episodes)),
            with_own_version_change=sum(e['own_version_changes'] > 0 for e in episodes),
            own_version_changes=sum(e['own_version_changes'] for e in episodes),
            with_rank_change=sum(e['rank_changes'] > 0 for e in episodes),
            rank_changes=sum(e['rank_changes'] for e in episodes),
            with_crossed_write=sum(e['crossed_write_intervals'] > 0 for e in episodes),
            sampled_span_s=distribution([e['last']['query_s']-e['start']['query_s'] for e in episodes])),
        H_ledger=dict(first_query_prefix_s=first_prefix, interquery=h_intervals,
            query_return_tail_s=tails, zero_query_H_s=sum(c['executed_holding_s'] for c in zero_query), total_s=total),
        query_doses=dict(doses), labels={k: label_tally(v) for k, v in labels.items()})
    return summary, dict(labels=labels, cycle_first_to_due=due_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'v250-root', 'v251-root', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source', required=True)
    a = parser.parse_args()
    seal = read(a.v251_root/'summary.json')
    assert seal['status'] == 'v251_early_supply_protocol_sealed' and seal['source'] == '7e9faca963'
    assert read(a.v251_root/'protocol.json') == registration('7e9faca963')
    parent = read(a.collection_root/'summary/status.json')
    assert parent['status'] == 'v249_passive_pair_passed' and parent['source'] == '96960db57a'
    assert parent['started_paths'] == parent['checked_paths'] == 2
    dose_certificate = read(a.v250_root/'summary.json')
    assert dose_certificate['status'] == 'v250_read_only_complete' and dose_certificate['source'] == '5a72961382'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v252_early_supply.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'details').mkdir()
    paths = {}
    for arm in ('v107', 'upper'):
        name = 'c0_'+arm+'.json.gz'
        capture = read(a.collection_root/'cycle_sidecars'/name)
        resource = read(a.collection_root/'resource_sidecars'/name)
        assert capture['complete'] and resource['complete']
        prefix = extract_prefixes(capture['queries'], resource['upper_versions'])
        save_detail(a.out_root/'details'/('c0_'+arm+'_prefix.json.gz'), prefix)
        paths[arm], evaluation = evaluate(prefix, capture, resource)
        assert paths[arm]['H_ledger']['total_s'] == dose_certificate['paths'][arm]['prepares']['H_ledger']['total_s']
        save_detail(a.out_root/'details'/('c0_'+arm+'_labels.json.gz'), evaluation)
        print('V252_READ_ONLY_PATH_COMPLETE '+arm, flush=True)
    summary = dict(status='v252_early_supply_audit_complete', source=a.source,
        protocol_source='7e9faca963', collection_source='96960db57a', dose_certificate_source='5a72961382',
        tests=tested.testsRun, actor=12037, demand_day=2151001, paths=paths,
        analyzed_paths=2, reads_per_required_payload=1, new_replays=0, training_updates=0,
        model_fits=0, new_futures=0, shadow_actions_executed=0, selector_defined=False,
        main_policy_changed=False, v238_changed=False,
        limitations='One consumed actor/day,two arms;no query-level inference. Episodes '
            'are consecutive query observations,not continuously known assignments. '
            'Boundary H and return tails are not assigned to a current obligation. '
            'Lead time,H and matching are not removable slack or action-effect evidence.')
    (a.out_root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V252_COMPLETE '+str(a.out_root/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
