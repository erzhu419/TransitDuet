#!/usr/bin/env python3
"""Frozen V272 day-level gate plus descriptive executed-action diagnostics."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

import run_freqduet_v272_graph_pilot as pilot


PATH_SOURCE = '07e565545a'
BINS = (0, 5, 10, 15, 20, 30, 45)


def describe(row, trace):
    queries = trace['queries']
    counts = Counter(str(q['executed_request_s']) for q in queries)
    actual = {str(h): counts[str(h)] for h in BINS}
    assert actual == row['results']['action_counts']
    assert len(queries) == row['action_chain']['verified_commands']
    delta = np.asarray([q['executed_request_s']-q['action_s'] for q in queries])
    load = np.asarray([q['load'] for q in queries])
    graph = row['graph']
    unsupported = 0 if graph is None else sum(
        n for status,n in graph['statuses'].items() if status.startswith('unsupported'))
    return dict(queries=len(queries), actual_action_counts=actual,
        same_state_native_action_counts={str(h):sum(q['action_s'] == h for q in queries) for h in BINS},
        changed=int(np.count_nonzero(delta)), increased=int(np.sum(delta > 0)), decreased=int(np.sum(delta < 0)),
        signed_requested_h_delta_s=int(delta.sum()), absolute_requested_h_delta_s=int(np.abs(delta).sum()),
        added_requested_h_s=int(delta[delta > 0].sum()), removed_requested_h_s=int(-delta[delta < 0].sum()),
        same_state_load_weighted_h_delta_person_s=float(load @ delta),
        unsupported_queries=unsupported, unsupported_fraction=unsupported/len(queries),
        model_supported_queries=0 if graph is None else len(queries)-unsupported,
        unsupported_reasons={} if graph is None else graph['unsupported_reasons'],
        actual_holding_vehicle_s=row['execution']['executed_s'],
        actual_holding_person_s=row['results']['holding_person_s'],
        unexecuted_command_s=row['execution']['unexecuted_s'],
        callback_p95_s=row['callback_latency_s']['p95'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    a = parser.parse_args()
    units = [pilot.read(a.root/'units'/f'{i}.json') for i in range(8)]
    assert all(u['source'] == PATH_SOURCE and u['status'] == 'complete' and
               not any(u['returncodes'].values()) for u in units)
    pilot.aggregate(SimpleNamespace(root=a.root, source=PATH_SOURCE))
    levels, details = {}, {}
    for index,(day,arm) in enumerate(pilot.ITEMS):
        row = pilot.read(a.root/'pairs'/f'p{index}.json')
        with gzip.open(a.root/'traces'/f'p{index}.pkl.gz', 'rb') as stream:
            trace = pickle.load(stream)
        d = describe(row, trace)
        assert d['changed'] == row['action_chain']['changed_from_native']
        details[str(day)+'_'+arm] = dict(day=day, arm=arm,
            endpoints={m:row['results'][m] for m in pilot.METRICS}, **d)
    for arm in pilot.ARMS:
        cells = [r for r in details.values() if r['arm'] == arm]
        queries = sum(r['queries'] for r in cells)
        unsupported = sum(r['unsupported_queries'] for r in cells)
        changed = sum(r['changed'] for r in cells)
        reasons = Counter()
        for r in cells:
            reasons.update(r['unsupported_reasons'])
        levels[arm] = dict(day_mean={m:float(np.mean([r['endpoints'][m] for r in cells])) for m in pilot.METRICS},
            queries=queries, actual_action_counts={str(h):sum(r['actual_action_counts'][str(h)] for r in cells) for h in BINS},
            changed=changed, changed_fraction=changed/queries,
            increased=sum(r['increased'] for r in cells), decreased=sum(r['decreased'] for r in cells),
            signed_requested_h_delta_s=sum(r['signed_requested_h_delta_s'] for r in cells),
            added_requested_h_s=sum(r['added_requested_h_s'] for r in cells),
            removed_requested_h_s=sum(r['removed_requested_h_s'] for r in cells),
            same_state_load_weighted_h_delta_person_s=sum(r['same_state_load_weighted_h_delta_person_s'] for r in cells),
            unsupported_queries=unsupported, unsupported_fraction=unsupported/queries,
            model_supported_queries=sum(r['model_supported_queries'] for r in cells),
            unsupported_reasons=dict(reasons),
            mean_day_holding_vehicle_s=float(np.mean([r['actual_holding_vehicle_s'] for r in cells])),
            mean_day_holding_person_s=float(np.mean([r['actual_holding_person_s'] for r in cells])),
            total_unexecuted_command_s=sum(r['unexecuted_command_s'] for r in cells))
    result = dict(analysis_source=a.source, path_source=PATH_SOURCE, status='v272_executed_action_diagnostics_complete',
        levels=levels, by_day=list(details.values()), independent_demand_days=4, independent_future_replications=0,
        physical_starts=0, training_updates=0, main_policy_changed=False,
        interpretation='Signed overrides compare the executed request with the bilateral request in THAT SAME visited state; '
            'they are not a counterfactual full-day causal decomposition. Inference uses four paired days, not individual queries.',
        raw_traces_remain_server_side=True)
    pilot.write(a.root/'summary/executed_diagnostics.json', result)
    print('V272_FROZEN_GATE_AND_EXECUTED_DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
