#!/usr/bin/env python3
"""Read saved V271 queries and completed visits; no rollout or policy changes."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle

import numpy as np


def analyze(root):
    accepted = json.loads((root/'summary/noop_fix.json').read_text())
    assert accepted['status'] == 'noop_native_equivalence_passed'
    with gzip.open(root/'traces/graph_queries_noop_fix.pkl.gz', 'rb') as stream:
        saved = pickle.load(stream)
    with gzip.open(root/'traces/noop_fix.pkl.gz', 'rb') as stream:
        native = pickle.load(stream)['queries']
    queries, visits = saved['queries'], saved['completed_visits']
    assert len(queries) == len(native) == accepted['queries']
    groups = defaultdict(list)
    for visit in visits:
        groups[visit.direction, visit.station].append(visit)
    for rows in groups.values():
        rows.sort(key=lambda v: v.arrival_s)
    totals, hours, stops = Counter(), defaultdict(Counter), defaultdict(Counter)
    for profile, query in zip(queries, native):
        assert (profile['time_s'], profile['bus'], profile['trip'], profile['station']) == (
            query['time_s'], query['bus'], query['trip'], query['station'])
        now = profile['time_s']
        completed = {key: [v for v in rows if v.completed_s < now and v.available_s < now]
                     for key, rows in groups.items()}
        assert sum(map(len, completed.values())) == profile['strict_visits']
        missing = {key for key, rows in completed.items() if len(rows) < 3}
        censored = {key for key, rows in completed.items() if len(rows) >= 3 and any(
            v.left_behind or v.incoming_load-v.alighted+v.boarded >= v.capacity for v in rows[-3:])}
        supported = not missing and not censored
        assert supported == (profile['evaluations'] > 0)
        current = query['direction'], query['station']
        row = Counter(queries=1, model_supported=int(supported),
            optimizing=int(profile['evaluations'] > 1),
            shadow_current_change=int(profile['shadow_action_s'] != profile['native_action_s']),
            model_improvement=int(profile['status'] == 'model_improvement'),
            model_improvement_current_unchanged=int(profile['status'] == 'model_improvement' and
                profile['shadow_action_s'] == profile['native_action_s']),
            any_missing_key=int(bool(missing)), any_censored_key=int(bool(censored)),
            censored_without_missing=int(bool(censored) and not missing),
            current_key_censored=int(current in censored),
            current_key_clear_but_other_key_censored=int(bool(censored) and not missing and current not in censored),
            pending_locks=profile['pending_locks'],
            supported_pending_locks=profile['pending_locks'] if supported else 0,
            verified_pending_locks=profile['model_locks_verified'])
        totals.update(row)
        hours[int(now//3600)].update(row)
        for key in groups:
            stops[key].update(queries=1, missing=int(key in missing), censored=int(key in censored),
                queried_here=int(key == current), censored_when_queried=int(key == current and key in censored))
    assert totals['model_supported'] == sum(v for k, v in accepted['statuses'].items() if k != 'unsupported_prefix')
    assert totals['pending_locks'] == accepted['pending_locks']
    assert totals['verified_pending_locks'] == accepted['model_locks_verified']
    supported_times = [q['total_s'] for q in queries if q['evaluations'] > 0]
    return dict(status='saved_prefix_support_reproduced', queries=len(queries), visits=len(visits),
        route_stop_keys=len(groups), totals=dict(totals),
        by_simulation_hour={str(h):dict(c) for h, c in sorted(hours.items())},
        by_direction_station=[dict(direction=d, station=s, **dict(c)) for (d,s),c in
            sorted(stops.items(), key=lambda item: (-item[1]['censored'], item[0]))],
        supported_query_latency_s=dict(mean=float(np.mean(supported_times)),
            p95=float(np.quantile(supported_times,.95)), max=float(max(supported_times))),
        physical_starts=0, training_updates=0, executed_shadow_actions=0,
        main_policy_changed=False, raw_artifacts_remain_server_side=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    path = args.root/'summary/support_audit.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('V271_SAVED_SUPPORT_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
