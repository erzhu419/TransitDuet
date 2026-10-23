#!/usr/bin/env python3
"""Read-only pressure, persistent holding and same-bus supply evidence on V236."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import unittest

import analyze_freqduet_v231_supply_overlap as prior


def resource_class(view):
    own = view['own_association']['obligation']
    due = view['target_profile']['uncovered_due']
    if own is not None and own['kind'] == 'due':
        assert own['trip_id'] in due
        return 'own_due'
    if due:
        return 'other_due'
    return 'future_unbacked' if own is not None else 'no_own_obligation'


def fixed_neighbour(holding, forward, backward):
    """Frozen observed neighbours only; not realized no-holding/CV2 outcomes."""
    if holding == 0:
        return dict(kind='native_zero', zero_minus_native_s2=0.)
    if forward is None or backward is None or forward < holding:
        return dict(kind='unavailable', zero_minus_native_s2=None)
    assert forward >= 0 and backward >= 0
    value = 2*holding*(backward-forward+holding)
    return dict(kind='benefit' if value > 0 else 'nonbenefit', zero_minus_native_s2=value)


def next_clock(cycle, launch, posted_s, dt):
    if launch is None:
        return dict(kind='no_next_launch', postreturn_wait_s=None,
                    next_posted_delay_s=None, next_readiness_delay_s=None)
    available = cycle['return_s']+dt
    wait = launch['actual_s']-available
    assert wait >= 0
    late = launch['actual_s'] > math.ceil(posted_s/dt)*dt
    return dict(kind=('return_bound' if wait == 0 else 'postreturn_wait')+
                ('_after_posted_plan' if late else '_not_after_posted_plan'),
        postreturn_wait_s=wait, next_posted_delay_s=max(0., launch['actual_s']-math.ceil(posted_s/dt)*dt),
        next_readiness_delay_s=launch['actual_s']-launch['eligible_s'])


def version_intervals(queries, versions, cycles):
    callbacks = sorted(v['native_event_seq'] for v in versions if not v['initial'])
    writes = sorted(v['native_event_seq'] for v in versions if not v['initial'] and v['changes'])
    previous, groups, rows = {}, defaultdict(list), []
    first_prefix = 0.
    for q in queries:
        view = q['view']
        key, seq = tuple(view['cycle_key']), view['native_event_seq']
        old = previous.get(key)
        if old is None:
            first_prefix += view['executed_holding_s']
        else:
            before = old['native_event_seq']
            ncallback = bisect_left(callbacks, seq)-bisect_left(callbacks, before)
            nwrite = bisect_left(writes, seq)-bisect_left(writes, before)
            delta = view['executed_holding_s']-old['executed_holding_s']
            assert delta == view['delta_executed_since_last_sample_s'] >= 0
            old_p, new_p = (v['own_association']['obligation'] for v in (old, view))
            own_revised = (old_p is not None and new_p is not None and old_p['trip_id'] == new_p['trip_id']
                           and old_p['last_write_version'] != new_p['last_write_version'])
            assert not own_revised or nwrite > 0
            kind = 'actual_write' if nwrite else 'callback_without_write' if ncallback else 'no_callback'
            row = dict(cycle_key=list(key), previous_query_seq=before, query_seq=seq,
                kind=kind, callbacks=ncallback, writes=nwrite,
                prior_H_s=old['executed_holding_s'], interval_delta_H_s=delta, own_plan_revised=own_revised,
                pressure_cleared=bool(old['target_profile']['uncovered_due']) and
                    not view['target_profile']['uncovered_due'])
            rows.append(row)
            groups[kind].append(row)
        previous[key] = view
    tail = sum(c['after_last_query_s'] for c in cycles)
    total = first_prefix+sum(r['interval_delta_H_s'] for r in rows)+tail
    assert total == sum(c['executed_holding_s'] for c in cycles)
    summary = dict(first_query_prefix_s=first_prefix, return_remainder_s=tail,
        closed_holding_s=total, groups={name: dict(intervals=len(values),
            cycles=len({tuple(r['cycle_key']) for r in values}),
            carrying_positive_H=sum(r['prior_H_s'] > 0 for r in values),
            own_plan_revised=sum(r['own_plan_revised'] for r in values),
            own_plan_revised_with_positive_H=sum(r['own_plan_revised'] and r['prior_H_s'] > 0 for r in values),
            max_prior_H_s=max(r['prior_H_s'] for r in values),
            interval_delta_H_s=sum(r['interval_delta_H_s'] for r in values),
            pressure_cleared=sum(r['pressure_cleared'] for r in values))
            for name, values in sorted(groups.items())})
    return summary, rows


def tally(rows):
    return dict(queries=len(rows), positive=sum(r['holding_s'] > 0 for r in rows),
        holding_s=sum(r['holding_s'] for r in rows),
        onboard_exposure_person_s=sum(r['holding_s']*r['onboard'] for r in rows),
        forecast_benefit_holding_s=sum(r['holding_s'] for r in rows if r['proxy'] == 'benefit_vs_zero'),
        fixed_neighbour_benefit_holding_s=sum(r['holding_s'] for r in rows if r['fixed']['kind'] == 'benefit'),
        fixed_neighbour_nonbenefit_holding_s=sum(r['holding_s'] for r in rows if r['fixed']['kind'] == 'nonbenefit'),
        fixed_neighbour_unknown_holding_s=sum(r['holding_s'] for r in rows if r['fixed']['kind'] == 'unavailable'))


def analyze(data, handoff, resource, capture):
    old_summary, model = prior.analyze_path(data, handoff, resource)
    assert capture['complete'] and len(capture['queries']) == len(model['decisions'])
    assert len(capture['cycles']) == len(model['cycles'])
    launches = {r['trip']: r for r in data['trace']['actual_launches']}
    clocks = {}
    for trip, cycle in model['cycles'].items():
        nxt = cycle['next_trip']
        clocks[trip] = next_clock(cycle, None if nxt is None else launches[nxt],
            None if nxt is None else data['launch_plans'][str(nxt)], resource['metadata']['native_dt_s'])
    neighbours, station_rows = {}, defaultdict(list)
    for row in data['trace']['departures']:
        station_rows[row['station_id'], bool(row['direction'])].append(row)
    for rows in station_rows.values():
        for i, row in enumerate(rows):
            neighbours[prior.stop_key(row)] = (None if i == 0 else row['time_s']-rows[i-1]['time_s'],
                None if i+1 == len(rows) else rows[i+1]['time_s']-row['time_s'])
    pressure, exposed, unique_cycles, details = defaultdict(list), defaultdict(list), defaultdict(list), []
    first_due, positive_versions = {}, defaultdict(set)
    for base, old, q in zip(model['decisions'], handoff['queries'], capture['queries']):
        assert all(q[k] == old[k] for k in ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence'))
        assert q['identity'] == old['identity'] and q['view']['query_s'] == base['query_s']
        view, trip = q['view'], base['key'][0]
        assert view['cycle_key'][:2] == [base['bus_id'], trip]
        name = resource_class(view)
        fixed = fixed_neighbour(base['holding_s'], *neighbours[tuple(base['key'][:3])])
        p = view['own_association']['obligation']
        clock = clocks[trip]
        row = dict(base, resource_class=name, fixed=fixed, next_clock=clock['kind'],
            own_obligation=None if p is None else p['trip_id'],
            next_label_matches=None if p is None or model['cycles'][trip]['next_trip'] is None else
                p['trip_id'] == model['cycles'][trip]['next_trip'],
            H_prefix_s=view['executed_holding_s'])
        pressure[name].append(row)
        if row['holding_s'] > 0:
            positive_versions[trip].add(tuple(view['message_refs'][k] for k in ('false', 'true')))
            if clock['kind'] == 'return_bound_after_posted_plan':
                evidence = ('forecast_benefit/' if row['proxy'] == 'benefit_vs_zero' else 'forecast_other/')+fixed['kind']
                exposed[name+'/'+evidence].append(row)
        if name == 'own_due':
            first_due.setdefault(trip, view['executed_holding_s'])
        details.append(row)
    for trip, cycle in model['cycles'].items():
        unique_cycles[clocks[trip]['kind']].append(cycle)
    version_summary, interval_rows = version_intervals(capture['queries'], resource['upper_versions'], capture['cycles'])
    assert version_summary['closed_holding_s'] == data['ledger']['executed_holding_s']
    cycle_lookup = {tuple(c['cycle_key']): c for c in capture['cycles']}
    assert sum(r['holding_s'] for r in details) == version_summary['closed_holding_s']
    for key, cycle in cycle_lookup.items():
        assert cycle['executed_holding_s'] == model['cycles'][key[1]]['holding_s']
    summary = dict(totals=tally(details), pressure_groups={k: tally(v) for k, v in sorted(pressure.items())},
        return_bound_after_plan_evidence={k: tally(v) for k, v in sorted(exposed.items())},
        unique_supply_cycles={k: dict(cycles=len(v), holding_s=sum(c['holding_s'] for c in v),
            next_public_delay_s=sum(c['next_public_delay_s'] for c in v if c['next_trip'] is not None),
            postreturn_wait_after_native_tick_s=sum(clocks[c['trip_id']]['postreturn_wait_s'] for c in v
                if c['next_trip'] is not None)) for k, v in sorted(unique_cycles.items())},
        first_own_due=dict(cycles=len(first_due), H_already_executed_s=sum(first_due.values()),
            later_holding_s=sum(model['cycles'][t]['holding_s']-h for t, h in first_due.items())),
        cross_version=version_summary,
        multi_positive_message_versions=dict(cycles=sum(len(v) > 1 for v in positive_versions.values()),
            holding_s=sum(model['cycles'][t]['holding_s'] for t, v in positive_versions.items() if len(v) > 1)),
        associations=dict(match=sum(r['next_label_matches'] is True for r in details),
            mismatch=sum(r['next_label_matches'] is False for r in details),
            unavailable=sum(r['next_label_matches'] is None for r in details)),
        predicted_local_nonharm_cuts=old_summary['all_queries']['positive_with_nonharm_cut'],
        endpoints=old_summary['outcomes'])
    return summary, model, dict(decisions=details, intervals=interval_rows,
        cycles=[dict(c, clock=clocks[t]) for t, c in model['cycles'].items()])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    status = a.out_root/'summary/status.json'
    if status.exists():
        saved = prior.read(status)
        assert saved['source'] == a.source and saved['status'] == 'v237_read_only_supply_evidence_complete'
        print('V237_EXISTING_COMPLETE_ANALYSIS_REUSED', flush=True)
        return
    parent = prior.read(a.collection_root/'summary/status.json')
    assert parent['status'] == 'v236_passive_pair_passed' and parent['source'] == '6972500d67'
    assert parent['started_paths'] == parent['checked_paths'] == 2
    assert all(r['passed'] and r['legacy_exact'] and r['v224_sidecar_exact'] and r['v230_sidecar_exact'] for r in parent['paths'])
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v237_persistent_supply.py')
    assert suite.countTestCases() == 6
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'details').mkdir()
    paths, models = {}, {}
    for arm in ('v107', 'upper'):
        name = 'c0_'+arm+'.json.gz'
        data, handoff, resource, capture = [prior.read(a.collection_root/folder/name)
            for folder in ('records', 'sidecars', 'resource_sidecars', 'cycle_sidecars')]
        assert (data['context'], data['arm']) == (0, arm)
        assert (data['results']['train_seed'], data['results']['scenario_seed']) == (12037, 2151001)
        paths[arm], models[arm], details = analyze(data, handoff, resource, capture)
        with gzip.open(a.out_root/'details'/name, 'wt') as stream:
            json.dump(details, stream, separators=(',', ':'))
        del data, handoff, resource, capture, details
        print('V237_READ_ONLY_PATH_COMPLETE '+arm, flush=True)
    paired, paired_rows = prior.compare(models['upper'], models['v107'])
    with gzip.open(a.out_root/'details/c0_paired.json.gz', 'wt') as stream:
        json.dump(paired_rows, stream, separators=(',', ':'))
    summary = dict(status='v237_read_only_supply_evidence_complete', source=a.source,
        parent_source='6972500d67', tests=tested.testsRun, actor=12037, demand_day=2151001,
        paths=paths, upper_vs_v107=paired, archive_passes_per_path=1, analyzed_paths=2,
        new_replays=0, training_updates=0, model_fits=0, shadow_actions_executed=0, new_futures=0,
        main_policy_changed=False,
        scope='One consumed actor/day,two arms. Local forecast and fixed-neighbour departure '
            'diagnostics are not necessary/harmful-action labels or actual CV2 gains. '
            'Return-bound timing and supply decomposition are accounting,not mediation. '
            'Inter-query H increments are not attributed to the latest plan/risk;observed wait is not slack.')
    status.parent.mkdir()
    status.write_text(json.dumps(summary, indent=2)+'\n')
    print('V237_SMALL_SUMMARY '+str(status), flush=True)


if __name__ == '__main__':
    main()
