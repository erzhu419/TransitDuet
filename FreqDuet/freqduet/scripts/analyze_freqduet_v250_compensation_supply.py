#!/usr/bin/env python3
"""Read V249 PREPARE evidence; keep next-launch labels in a separate pass."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import unittest

import analyze_freqduet_v231_supply_overlap as accounting
from analyze_freqduet_v237_persistent_supply import next_clock, resource_class, version_intervals
from preregister_freqduet_v246_compensation_supply import (
    ACTIONS, completed_forward_parts, evidence_frontier, forward_parts_available,
    persistent_change)


PARTS = ('release_s', 'executed_holding_gap_s', 'nonholding_elapsed_gap_s')
JOINS = ('prepare_seq', 'prepare_order', 'query_seq', 'query_order', 'query_occurrence')


def component_summary(rows):
    if not rows:
        return dict(samples=0, mean_s=None, variance_s2=None, covariance_with_gap_s2=None)
    n = len(rows)
    means = {k: math.fsum(r[k] for r in rows)/n for k in PARTS}
    gap_mean = math.fsum(r['gap_s'] for r in rows)/n
    covariance = {k: math.fsum((r[k]-means[k])*(r['gap_s']-gap_mean)
                              for r in rows)/n for k in PARTS}
    variance = math.fsum((r['gap_s']-gap_mean)**2 for r in rows)/n
    accounting.close(sum(means.values()), gap_mean, 'mean component closure')
    accounting.close(sum(covariance.values()), variance, 'covariance component closure')
    return dict(samples=n, mean_gap_s=gap_mean, mean_s=means, variance_s2=variance,
                covariance_with_gap_s2=covariance)


def query_tally(rows):
    positive = [r for r in rows if r['native_holding_s'] > 0]
    return dict(queries=len(rows), positive=len(positive),
        native_holding_s=sum(r['native_holding_s'] for r in rows),
        positive_with_nonharm_cut=sum(r['nonharm_cut'] is True for r in positive),
        positive_cut_evidence_unknown=sum(r['nonharm_cut'] is None for r in positive),
        local_benefit_vs_zero=sum(r['zero_loss_s2'] is not None and r['zero_loss_s2'] > 0
                                  for r in positive),
        H_prefix_sum_s=sum(r['H_s'] for r in rows),
        H_prefix_max_s=max((r['H_s'] for r in rows), default=None))


def observable_queries(timing, handoff, resource, capture):
    """This pass receives no returns, realized departures or next-launch identities."""
    lengths = [len(x) for x in (timing['queries'], handoff['queries'],
                               resource['query_extensions'], capture['queries'])]
    assert len(set(lengths)) == 1
    rows, groups, first_due = [], defaultdict(list), {}
    for saved, old, extra, cyc in zip(timing['queries'], handoff['queries'],
                                    resource['query_extensions'], capture['queries']):
        prepared = timing['prepares'][saved['prepare_index']]['mapped']
        join, view = saved['join'], cyc['view']
        expected = dict(prepare_seq=prepared['prepare_seq'], prepare_order=old['prepare_order'],
            query_seq=old['legacy_query_event_seq'], query_order=old['query_order'],
            query_occurrence=old['query_occurrence'])
        assert all(join[k] == expected[k] for k in JOINS)
        assert all(cyc[k] == extra[k] == old[k] for k in
                   ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence'))
        assert prepared['prepare_order'] == old['prepare_order']
        assert prepared['identity'] == old['identity'] == cyc['identity']
        assert join['prepare_s'] == prepared['prepare_s']
        assert join['query_s'] == old['query_s'] == extra['snapshot_read_s']
        assert join['prepare_holding_s'] == prepared['own']['holding_s']
        assert join['query_holding_s'] == view['executed_holding_s']
        query = dict(query_s=old['query_s'], query_seq=old['legacy_query_event_seq'],
            query_order=old['query_order'], native_bin=old['native_bin'],
            terminal_masked=extra['terminal_masked'], onboard=extra['onboard'])
        frontier = evidence_frontier(prepared, query, view, old['feasible_bins'])
        native = ACTIONS[old['native_bin']]
        cuts = [c['local_loss_vs_native_s2'] for c in frontier['candidates'] if c['holding_s'] < native]
        known = bool(cuts) and all(v is not None for v in cuts)
        zero = next((c['local_loss_vs_native_s2'] for c in frontier['candidates']
                     if c['holding_s'] == 0), None)
        own = view['own_association']['obligation']
        row = dict(cycle_key=list(view['cycle_key']), identity=prepared['identity'],
            query_seq=query['query_seq'], query_s=query['query_s'],
            prepare_s=prepared['prepare_s'], prepare_H_s=prepared['own']['holding_s'],
            H_s=view['executed_holding_s'], group=resource_class(view),
            own_obligation_kind=None if own is None else own['kind'],
            own_obligation=None if own is None else dict(own),
            target_due=bool(view['target_profile']['uncovered_due']),
            native_holding_s=native, zero_loss_s2=zero,
            nonharm_cut=any(v <= 0 for v in cuts) if known else None,
            frontier=frontier)
        rows.append(row)
        groups[row['group']].append(row)
        if row['group'] == 'own_due':
            first_due.setdefault(tuple(row['cycle_key']), row)
    summary = dict(all_queries=query_tally(rows),
        groups={k: query_tally(v) for k, v in sorted(groups.items())},
        own_future_with_other_due=sum(r['target_due'] and r['own_obligation_kind'] == 'future'
                                      for r in rows),
        first_own_due_cycles=len(first_due),
        prepare_to_query_H_increases=sum(r['H_s'] > r['prepare_H_s'] for r in rows))
    return summary, rows, first_due


def prepare_account(timing, cycles):
    previous, cycle_prepares = {}, defaultdict(list)
    levels, changes, statuses = [], [], Counter()
    for item in timing['prepares']:
        p = item['mapped']
        key = tuple(p['cycle_key'])
        if forward_parts_available(p):
            levels.append(dict(gap_s=p['forward_s'], **completed_forward_parts(p)))
        change = persistent_change(previous.get(key), p)
        assert change == item['change']
        statuses[change['status']] += 1
        if change['parts'] is not None:
            changes.append(dict(gap_s=p['forward_s']-previous[key]['forward_s'], **change['parts']))
        previous[key] = p
        cycle_prepares[key].append(p)
    prefixes, increments, tails, details = 0., 0., 0., []
    assert set(cycle_prepares) == {tuple(c['cycle_key']) for c in cycles}
    for cycle in cycles:
        rows = cycle_prepares[tuple(cycle['cycle_key'])]
        hs = [p['own']['holding_s'] for p in rows]
        ds = [b-a for a, b in zip(hs, hs[1:])]
        tail = cycle['executed_holding_s']-hs[-1]
        assert hs[0] >= 0 and tail >= 0 and all(d >= 0 for d in ds)
        accounting.close(hs[0]+sum(ds)+tail, cycle['executed_holding_s'], 'PREPARE H ledger')
        prefixes += hs[0]
        increments += sum(ds)
        tails += tail
        details.append(dict(cycle_key=cycle['cycle_key'], prepares=len(rows),
            first_prepare_H_s=hs[0], cross_prepare_increment_s=sum(ds), return_tail_s=tail))
    return dict(prepares=len(timing['prepares']), supported=len(levels),
        transition_status=dict(statuses), levels=component_summary(levels),
        same_front_changes=component_summary(changes),
        H_ledger=dict(cycles=len(cycles), first_prepare_prefix_s=prefixes,
            cross_prepare_increment_s=increments, return_tail_s=tails,
            total_s=prefixes+increments+tails)), details


def supply_labels(rows, model, data, resource):
    """Future labels are attached only after the observable records are complete."""
    launches = {r['trip']: r for r in data['trace']['actual_launches']}
    clocks = {t: next_clock(c, None if c['next_trip'] is None else launches[c['next_trip']],
        None if c['next_trip'] is None else data['launch_plans'][str(c['next_trip'])],
        resource['metadata']['native_dt_s']) for t, c in model['cycles'].items()}
    labels = []
    for r in rows:
        c = model['cycles'][r['cycle_key'][1]]
        assert c['bus_id'] == r['cycle_key'][0]
        p, nxt = r['own_obligation'], c['next_trip']
        status = ('no_next_launch' if nxt is None else 'no_query_association' if p is None
                  else 'match' if p['trip_id'] == nxt else 'mismatch')
        labels.append(dict(cycle_key=r['cycle_key'], query_seq=r['query_seq'], status=status,
            group=r['group'], native_holding_s=r['native_holding_s'], H_prefix_s=r['H_s'],
            later_H_s=c['holding_s']-r['H_s'], actual_next_trip=nxt,
            next_public_delay_s=c['next_public_delay_s'], **clocks[c['trip_id']]))
    return labels


def label_tally(labels):
    available = [r for r in labels if r['actual_next_trip'] is not None]
    return dict(samples=len(labels), status=dict(Counter(r['status'] for r in labels)),
        clocks=dict(Counter(r['kind'] for r in labels)),
        H_already_executed_s=sum(r['H_prefix_s'] for r in labels),
        later_H_s=sum(r['later_H_s'] for r in labels),
        next_public_delay_s=sum(r['next_public_delay_s'] for r in available) if available else None,
        next_readiness_delay_s=sum(r['next_readiness_delay_s'] for r in available) if available else None,
        postreturn_wait_after_native_tick_s=sum(r['postreturn_wait_s'] for r in available) if available else None)


def analyze(data, handoff, resource, capture, timing):
    observable, queries, first_due = observable_queries(timing, handoff, resource, capture)
    prepares, prepare_details = prepare_account(timing, capture['cycles'])
    # Reuse actual-dose and first-next-launch accounting, not new action requests.
    physical, model = accounting.analyze_path(data, handoff, resource)
    accounting.close(observable['all_queries']['native_holding_s'],
                     physical['all_queries']['holding_s'], 'request versus actual total')
    accounting.close(prepares['H_ledger']['total_s'], data['ledger']['executed_holding_s'], 'H total')
    cross_version, intervals = version_intervals(capture['queries'], resource['upper_versions'], capture['cycles'])
    labels = supply_labels(queries, model, data, resource)
    first_labels = supply_labels(list(first_due.values()), model, data, resource)
    overlap = [r for r in labels if r['native_holding_s'] > 0 and r['group'] == 'own_due']
    summary = dict(observable=observable, prepares=prepares, cross_version=cross_version,
        first_own_due_labels=label_tally(first_labels), positive_own_due_query_labels=label_tally(overlap),
        physical_cycles=physical['physical_cycles'],
        initial_stock_launches=physical['initial_stock_launches'], endpoints=physical['outcomes'])
    return summary, model, dict(observable_queries=queries, prepare_cycles=prepare_details,
        future_labels=labels, first_own_due_labels=first_labels, version_intervals=intervals)


def brief_path(path):
    obs, pre = path['observable'], path['prepares']
    return dict(queries=obs['all_queries'], pressure_groups=obs['groups'],
        own_future_with_other_due=obs['own_future_with_other_due'],
        forward_components=pre['levels'], same_front_changes=pre['same_front_changes'],
        H_ledger=pre['H_ledger'], cross_version=path['cross_version'],
        first_own_due=path['first_own_due_labels'],
        positive_own_due=path['positive_own_due_query_labels'], endpoints=path['endpoints'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    parent = accounting.read(args.collection_root/'summary/status.json')
    assert parent['status'] == 'v249_passive_pair_passed' and parent['source'] == '96960db57a'
    assert parent['started_paths'] == parent['checked_paths'] == 2
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v250_compensation_supply.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root/'details').mkdir()
    paths, models = {}, {}
    for arm in ('v107', 'upper'):
        name = 'c0_'+arm+'.json.gz'
        payloads = [accounting.read(args.collection_root/folder/name) for folder in
                    ('records', 'sidecars', 'resource_sidecars', 'cycle_sidecars', 'prepare_timing_sidecars')]
        data = payloads[0]
        assert (data['context'], data['arm']) == (0, arm)
        assert (data['results']['train_seed'], data['results']['scenario_seed']) == (12037, 2151001)
        paths[arm], models[arm], detail = analyze(*payloads)
        with gzip.open(args.out_root/'details'/name, 'wt') as stream:
            json.dump(detail, stream, separators=(',', ':'))
        print('V250_READ_ONLY_PATH_COMPLETE '+arm, flush=True)
    paired, pair_rows = accounting.compare(models['upper'], models['v107'])
    with gzip.open(args.out_root/'details/c0_paired.json.gz', 'wt') as stream:
        json.dump(pair_rows, stream, separators=(',', ':'))
    result = dict(status='v250_read_only_complete', source=args.source, parent_source=parent['source'],
        tests=tested.testsRun, actor=12037, demand_day=2151001, paths=paths, upper_vs_v107=paired,
        archive_passes_per_path=1, analyzed_paths=2, new_replays=0, training_updates=0,
        model_fits=0, shadow_actions_executed=0, selector_defined=False,
        main_policy_changed=False, v238_changed=False,
        limitations='One consumed actor/day,two arms. No query-level independence or causal '
            'necessity/removability claim. Current position associations are not reservations. '
            'Interval H is not attributable to the latest plan. Future labels are evaluation only; '
            'return waiting is not spendable slack. Covariance terms are accounting,not mediation.')
    (args.out_root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    brief = dict(result, paths={k: brief_path(v) for k, v in paths.items()})
    (args.out_root/'brief.json').write_text(json.dumps(brief, separators=(',', ':'))+'\n')
    print('V250_COMPLETE '+str(args.out_root/'brief.json'), flush=True)


if __name__ == '__main__':
    main()
