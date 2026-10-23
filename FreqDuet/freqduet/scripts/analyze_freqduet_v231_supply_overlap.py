#!/usr/bin/env python3
"""Describe saved V230 resource/compensation overlap and same-bus outcomes."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import time
import unittest

from analyze_freqduet_v221_execution import ACTIONS, differences, read
from preregister_freqduet_v224_prepare_handoff import CONTEXTS


def protocol(source):
    return dict(stage='v231_read_only_supply_compensation_overlap', source=source,
        collection_source='eec7a20c71', acceptance_source='67e06526f4', paths=8,
        contexts=list(CONTEXTS), arms=['v107', 'upper'],
        pressure='At query,next native dispatch pass:uncovered posted obligations at '
            'the opposite terminal. Incoming position is not ready stock or promised slack.',
        compensation='Supported native h>0 with R(0)-R(h)>0 is a local predicted benefit '
            'only. Enumerate recorded lower feasible bins with delta R<=0;do not choose one. '
            'Retain unsupported,zero-action and no-nonharm-cut rows separately.',
        outcome='Link first later actual same-bus launch to the then-recorded uncovered,'
            'covered-due,future-posted,or absent obligation. Future links are labels ONLY.',
        accounting='Use actual departure-minus-query holding. Each completed vehicle cycle '
            'and first next launch is counted once;query exposure is not independent service. '
            'Pair upper vs V107 by trip/stop/direction/occurrence and physical supply chain; '
            'retain changed identities,missing matches and final returns separately.',
        comparisons='Four consumed actor/day contexts,not independent confirmation or '
            'a causal mediation test. No CI from treating queries as independent samples.',
        new_replays=0, training_updates=0, model_fits=0, shadow_actions_executed=0,
        new_futures=0, main_policy_changed=False)


def close(a, b, label):
    assert abs(a-b) <= 1e-6, (label, a, b)


def stop_key(row):
    return row['trip_id'], row['station_id'], bool(row['direction'])


def next_relation(next_trip, coverage, plans, target):
    if next_trip is None:
        return 'no_later_same_bus_launch'
    if next_trip in coverage['uncovered'][target]:
        return 'supplied_query_uncovered'
    if next_trip in coverage['due'][target]:
        return 'supplied_query_covered_due'
    if next_trip in {p['trip_id'] for p in plans}:
        return 'supplied_future_posted'
    return 'not_pending_at_query'


def compensation(frontier):
    native = ACTIONS[frontier['native_bin']]
    if frontier['status'] != 'supported':
        return dict(proxy=frontier['status'], nonharm_cut=None, zero_delta_R=None, min_cut_delta_R=None)
    candidates = frontier['candidates']
    zero = next((r['local_gap_cost_delta_s2'] for r in candidates if r['holding_s'] == 0), None)
    cuts = [r['local_gap_cost_delta_s2'] for r in candidates if r['holding_s'] < native]
    return dict(proxy='native_zero' if native == 0 else 'benefit_vs_zero' if zero is not None and zero > 0
        else 'zero_not_feasible' if zero is None else 'no_benefit_vs_zero',
        nonharm_cut=any(v <= 0 for v in cuts), zero_delta_R=zero,
        min_cut_delta_R=min(cuts) if cuts else None)


def tally(rows):
    return dict(queries=len(rows), positive=sum(r['holding_s'] > 0 for r in rows),
        holding_s=math.fsum(r['holding_s'] for r in rows),
        onboard_holding_person_s=math.fsum(r['onboard']*r['holding_s'] for r in rows),
        supported=sum(r['supported'] for r in rows),
        positive_with_nonharm_cut=sum(r['holding_s'] > 0 and r['nonharm_cut'] is True for r in rows))


def cycle_tally(rows):
    labelled = [r for r in rows if r['next_trip'] is not None]
    return dict(cycles=len(rows), next_launch_labels=len(labelled),
        holding_s=math.fsum(r['holding_s'] for r in rows),
        pressure_holding_s=math.fsum(r['pressure_holding_s'] for r in rows),
        matched_pressure_queries=sum(r['matched_pressure_queries'] for r in rows),
        next_public_delay_s=math.fsum(r['next_public_delay_s'] for r in labelled),
        observed_postreturn_wait_s=math.fsum(r['observed_postreturn_wait_s'] for r in labelled))


def analyze_path(data, sidecar, extension):
    prepares = {p['clock']['handoff_order']: p for p in sidecar['prepares']}
    actions = data['trace']['lower_actions']
    queries, extra = sidecar['queries'], extension['query_extensions']
    assert len(actions) == len(queries) == len(extra) == 5200
    assert len({stop_key(q['identity']) for q in queries}) == len(queries), \
        'repeated query occurrence needs an executed-action attribution,not duplicated departure holding'
    departures = {stop_key(r): r for r in data['trace']['departures']}
    launches = {r['trip']: r for r in data['trace']['actual_launches']}
    returns = {r['trip_id']: r for r in data['trace']['vehicle_returns']}
    outcomes = {r['inbound_trip_id']: r for r in extension['supply_outcomes'] if r['inbound_trip_id'] is not None}
    cycles = {}
    for trip, ret in returns.items():
        launch = launches[trip]
        child = outcomes.get(trip)
        nxt = None if child is None else launches[child['next_trip_id']]
        assert ret['bus_id'] == launch['bus']
        if nxt is not None:
            assert child['bus_id'] == nxt['bus'] == launch['bus']
            assert nxt['direction'] == ret['target_direction'] != ret['inbound_direction']
            assert nxt['actual_s'] >= ret['return_s']+extension['metadata']['native_dt_s']
        cycles[trip] = dict(trip_id=trip, bus_id=launch['bus'], launch_s=launch['actual_s'],
            return_s=ret['return_s'], holding_s=ret['holding_s'], other_elapsed_s=ret['other_elapsed_s'],
            next_trip=None if nxt is None else nxt['trip'], next_launch_s=None if nxt is None else nxt['actual_s'],
            next_public_delay_s=None if nxt is None else max(0., nxt['actual_s']-nxt['public_baseline_s']),
            observed_postreturn_wait_s=None if nxt is None else nxt['actual_s']-ret['return_s'],
            queries=0, attributed_holding_s=0., pressure_holding_s=0.,
            pressure_positive_queries=0, matched_pressure_queries=0)
    rows = []
    rank = {b: i for i, b in enumerate(extension['metadata']['native_inventory_order'])}
    for action, q, row in zip(actions, queries, extra):
        key = stop_key(q['identity'])
        assert key == stop_key(action)
        assert all(row[k] == q[k] for k in ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence'))
        assert q['native_bin'] == action['action'] == row['shadow_frontier']['native_bin']
        prepare = prepares[q['prepare_order']]
        age = q['query_s']-prepare['clock']['prepare_s']
        assert age >= 0 and q['query_s'] == row['snapshot_read_s']
        executed = departures[key]['realized_holding']-row['executed_holding_before_s']
        close(executed, ACTIONS[q['native_bin']], 'executed versus requested holding')
        cycle = cycles[key[0]]
        assert cycle['bus_id'] == q['identity']['bus_id'] == row['cycle_ref'][0]
        target = str(not key[2]).lower()
        cov = row['ready_coverage']
        pressure = bool(cov['uncovered'][target])
        relation = next_relation(cycle['next_trip'], cov, row['plans'], target)
        incoming = [b for b in row['active'] if b['direction'] == key[2]]
        incoming.sort(key=lambda b: (b['remaining_distance_m'], rank[b['bus_id']]))
        distance_rank = next(i+1 for i, b in enumerate(incoming) if b['bus_id'] == cycle['bus_id'])
        comp = compensation(row['shadow_frontier'])
        r = dict(key=[*key, q['query_occurrence']], bus_id=cycle['bus_id'], query_s=q['query_s'],
            query_age_s=age, holding_s=executed, onboard=row['onboard'], pressure=pressure,
            uncovered_count=len(cov['uncovered'][target]), supported=row['shadow_frontier']['status'] == 'supported',
            distance_rank=distance_rank, same_bus_next_label=relation, **comp)
        rows.append(r)
        cycle['queries'] += 1
        cycle['attributed_holding_s'] += executed
        cycle['pressure_holding_s'] += executed if pressure else 0.
        cycle['pressure_positive_queries'] += pressure and executed > 0
        cycle['matched_pressure_queries'] += pressure and executed > 0 and relation == 'supplied_query_uncovered'
    for c in cycles.values():
        close(c['attributed_holding_s'], c['holding_s'], 'cycle holding attribution')
    close(math.fsum(r['holding_s'] for r in rows), data['ledger']['executed_holding_s'], 'query holding ledger')
    close(math.fsum(c['holding_s'] for c in cycles.values()), data['ledger']['executed_holding_s'], 'cycle ledger')
    close(math.fsum(max(0., r['actual_s']-r['public_baseline_s']) for r in launches.values()),
          data['results']['public_launch_delay_total_s'], 'public delay ledger')
    assert len({c['next_trip'] for c in cycles.values() if c['next_trip'] is not None}) == len(outcomes)
    groups, links, cycle_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        groups[f"pressure{int(r['pressure'])}/positive{int(r['holding_s'] > 0)}/{r['proxy']}"].append(r)
        if r['pressure'] and r['holding_s'] > 0:
            links[r['same_bus_next_label']].append(r)
    for c in cycles.values():
        label = ('no_next_launch' if c['next_trip'] is None else 'pressure_and_matched_obligation'
            if c['matched_pressure_queries'] else 'pressure_without_matched_obligation'
            if c['pressure_positive_queries'] else 'no_positive_pressure_overlap')
        cycle_groups[label].append(c)
    overlap = [r for r in rows if r['pressure'] and r['holding_s'] > 0]
    assert sum(r['pressure'] for r in rows) == extension['counts']['pressure_queries']
    assert len(overlap) == extension['counts']['pressure_positive_native_overlap']
    assert sum(r['holding_s'] > 0 for r in rows) == extension['counts']['positive_native_queries']
    summary = dict(all_queries=tally(rows), pressure_positive=tally(overlap),
        compensation_groups={k: tally(v) for k, v in sorted(groups.items())},
        next_same_bus_labels={k: tally(v) for k, v in sorted(links.items())},
        overlap_nearest_distance_queries=sum(r['distance_rank'] == 1 for r in overlap),
        query_age_s=dict(max=max(r['query_age_s'] for r in rows), older_than_prepare=sum(r['query_age_s'] > 0 for r in rows)),
        unique_cycle_groups={k: cycle_tally(v) for k, v in sorted(cycle_groups.items())},
        physical_cycles=len(cycles), initial_stock_launches=len(launches)-len(outcomes),
        outcomes={m: data['results'][m] for m in ('passenger_person_s', 'journey_min', 'cv_squared',
            'readiness_delay_s', 'public_launch_delay_total_s')})
    return summary, dict(decisions=rows, cycles=cycles, results=data['results'])


def supply_delta(current, reference):
    assert (current['bus_id'], current['next_trip']) == (reference['bus_id'], reference['next_trip'])
    assert current['next_trip'] is not None
    fields = dict(launch_s='inbound_launch_delta_s', holding_s='holding_delta_s',
        other_elapsed_s='other_elapsed_delta_s', observed_postreturn_wait_s='observed_postreturn_wait_delta_s')
    row = {name: current[k]-reference[k] for k, name in fields.items()}
    nxt = current['next_launch_s']-reference['next_launch_s']
    close(math.fsum(row.values()), nxt, 'next launch one-step supply identity')
    return dict(next_launch_delta_s=nxt, next_public_delay_delta_s=
        current['next_public_delay_s']-reference['next_public_delay_s'],
        pressure_holding_delta_s=current['pressure_holding_s']-reference['pressure_holding_s'], **row)


def compare(current, reference):
    c_rows = {tuple(r['key']): r for r in current['decisions']}
    b_rows = {tuple(r['key']): r for r in reference['decisions']}
    common = c_rows.keys() & b_rows.keys()
    decision_groups = defaultdict(lambda: Counter(queries=0, holding_delta_s=0., same_bus=0))
    for key in common:
        c, b = c_rows[key], b_rows[key]
        name = ('pressure_both' if c['pressure'] and b['pressure'] else 'pressure_new'
                if c['pressure'] else 'pressure_relieved' if b['pressure'] else 'no_pressure_both')
        decision_groups[name].update(dict(queries=1, holding_delta_s=c['holding_s']-b['holding_s'],
                                         same_bus=int(c['bus_id'] == b['bus_id'])))
    unmatched = dict(current_only=len(c_rows.keys()-b_rows.keys()), reference_only=len(b_rows.keys()-c_rows.keys()),
        holding_delta_s=math.fsum(c_rows[k]['holding_s'] for k in c_rows.keys()-b_rows.keys())-
            math.fsum(b_rows[k]['holding_s'] for k in b_rows.keys()-c_rows.keys()))
    holding = math.fsum(r['holding_s'] for r in current['cycles'].values())-math.fsum(
        r['holding_s'] for r in reference['cycles'].values())
    close(math.fsum(r['holding_delta_s'] for r in decision_groups.values())+unmatched['holding_delta_s'],
          holding, 'paired decision holding')
    chain_groups, details = defaultdict(list), []
    excluded = Counter()
    for trip in sorted(current['cycles'].keys() | reference['cycles'].keys()):
        c, b = current['cycles'].get(trip), reference['cycles'].get(trip)
        if c is None or b is None:
            excluded['unmatched_trip'] += 1
        elif c['next_trip'] is None or b['next_trip'] is None:
            excluded['no_next_launch_label'] += 1
        elif (c['bus_id'], c['next_trip']) != (b['bus_id'], b['next_trip']):
            excluded['changed_bus_or_next_trip'] += 1
        else:
            d = supply_delta(c, b)
            sign = 'later' if d['next_launch_delta_s'] > 0 else 'earlier' if d['next_launch_delta_s'] < 0 else 'same'
            chain_groups[sign].append(d)
            details.append(dict(trip_id=trip, bus_id=c['bus_id'], next_trip=c['next_trip'],
                current_pressure_queries=c['pressure_positive_queries'], reference_pressure_queries=b['pressure_positive_queries'],
                current_matched_pressure_queries=c['matched_pressure_queries'],
                reference_matched_pressure_queries=b['matched_pressure_queries'], **d))
    summary = dict(effects=differences(current['results'], reference['results']),
        total_holding_delta_s=holding, matched_decisions=len(common), unmatched_decisions=unmatched,
        pressure_transition_groups=dict(decision_groups), unique_same_chain_next_launches=len(details),
        excluded_cycles=dict(excluded), next_launch_groups={k: dict(cycles=len(v),
            **{name: math.fsum(r[name] for r in v) for name in v[0]}) for k, v in sorted(chain_groups.items())})
    return summary, details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'acceptance-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v231_supply_overlap.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    parent, accepted = read(a.collection_root/'summary/status.json'), read(a.acceptance_root/'status.json')
    assert parent['status'] == 'v230_remaining_six_passed' and parent['all_checked_passed']
    assert parent['started_paths'] == parent['checked_paths'] == 8 and parent['source'] == 'eec7a20c71'
    assert accepted['status'] == 'v230_remaining_six_read_only_acceptance_passed' and accepted['all_eight_passed']
    assert accepted['source'] == '67e06526f4'
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'details').mkdir()
    spec = protocol(a.source)
    (a.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    contexts, start = [], time.monotonic()
    for ci, (actor, day) in enumerate(CONTEXTS):
        summaries, models = {}, {}
        for arm in ('v107', 'upper'):
            name = f'c{ci}_{arm}.json.gz'
            data, sidecar, extra = [read(a.collection_root/folder/name)
                                   for folder in ('records', 'sidecars', 'resource_sidecars')]
            assert (data['context'], data['arm']) == (ci, arm)
            assert (data['results']['train_seed'], data['results']['scenario_seed']) == (actor, day)
            summaries[arm], models[arm] = analyze_path(data, sidecar, extra)
            with gzip.open(a.out_root/'details'/name, 'wt') as stream:
                json.dump(models[arm], stream)
            del data, sidecar, extra
        comparison, detail = compare(models['upper'], models['v107'])
        with gzip.open(a.out_root/'details'/f'c{ci}_paired.json.gz', 'wt') as stream:
            json.dump(detail, stream)
        contexts.append(dict(context=ci, actor=actor, demand_day=day,
                             paths=summaries, upper_vs_v107=comparison))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(contexts), total=4,
            unit='context', elapsed_seconds=time.monotonic()-start)), flush=True)
        del models
    summary = dict(spec, status='v231_supply_overlap_complete', tests=tests.testsRun, contexts=contexts)
    (a.out_root/'summary').mkdir()
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, separators=(',', ':'))+'\n')
    print('V231_SUPPLY_OVERLAP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
