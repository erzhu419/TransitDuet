#!/usr/bin/env python3
"""Read-only actor/day interpretation of the frozen V177 confirmation."""

import argparse
import json
from math import isclose
from pathlib import Path


SCOPES = ('all', 'active', 'clearance', 'hour2', 'hour13', 'hour13_moving_launch')
METRICS = ('coverage', 'below', 'above', 'finite_fraction', 'width_s', 'score_s',
    'score_delta_s', 'width_delta_s', 'secondary_score_delta_s',
    'secondary_width_delta_s', 'temporal_score_delta_s', 'temporal_width_delta_s',
    'v172_repaired_above', 'v172_upper_penalty_delta_s')


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    source = read(root/'summary/status.json')
    assert source['status'] == 'v177_independent_confirmation_complete'
    assert source['execution_source'] == 'd1454cfc39' and source['map_source'] == '36108d4f49'
    assert source['paths'] == 1536 and source['shards'] == 96
    assert source['confirmation_days'] == list(range(3352001, 3352065))
    assert source['map_updates'] == source['new_training'] == 0
    assert not source['interval_fit'] and not source['main_policy_changed']
    assert all(not source[name+'_changed'] for name in ('v167', 'v172', 'v176', 'v151', 'v146'))
    readouts = {(r['mode'], r['scope']): r for r in source['readouts']}
    assert set(readouts) == {(m, s) for m in ('old', 'handoff') for s in SCOPES}
    assert all(r['complete_actor_day_grid'] and not r['finite_unresolved_n'] and
        not r['missing_reference_n'] for r in readouts.values())
    report = dict(status='v177_readonly_interpretation_complete', gate=source['gate'],
        execution_source=source['execution_source'], map_source=source['map_source'],
        paths=source['paths'], shards=source['shards'], interval_fit=False,
        map_updates=0, main_policy_changed=False, groups=[])
    marginals = {axis: read(root/'details'/('by_'+axis+'.json')) for axis in ('actor', 'day')}
    for (mode, scope), r in readouts.items():
        estimates = r['estimates']
        group = dict(mode=mode, scope=scope, axes={})
        for axis, identities in (('actor', source['actors']), ('day', source['confirmation_days'])):
            rows = [v for v in marginals[axis] if (v['mode'], v['scope']) == (mode, scope)]
            assert len(rows) == len(identities) and {v[axis] for v in rows} == set(identities)
            assert all(v['cells'] == (64 if axis == 'actor' else 12) for v in rows)
            stats = {}
            for metric in METRICS:
                values = [v[metric] for v in rows]
                assert all(v is not None for v in values)
                assert isclose(sum(values)/len(values), estimates[metric]['mean'], abs_tol=1e-8)
                stats[metric] = dict(min=min(values), max=max(values),
                    negative=sum(v < 0 for v in values), positive=sum(v > 0 for v in values))
                if metric in ('coverage', 'finite_fraction', 'below', 'above'):
                    bound = .8 if metric in ('coverage', 'finite_fraction') else .1
                    stats[metric]['point_bound_violations'] = sum(
                        v < bound if bound == .8 else v > bound for v in values)
            group['axes'][axis] = dict(count=len(rows), metrics=stats)
            if scope in ('clearance', 'hour13_moving_launch'):
                fields = (axis, 'coverage', 'below', 'above', 'finite_fraction',
                    'score_delta_s', 'secondary_score_delta_s', 'temporal_score_delta_s',
                    'temporal_width_delta_s')
                worst = sorted(rows, key=lambda v: v['above'], reverse=True)[:3]
                group['axes'][axis]['highest_upper_tail'] = [
                    {k: v[k] for k in fields} for v in worst]
        group['paired_cost'] = {}
        for name, delta in (('coarse', ''), ('v167', 'secondary_'), ('v172', 'temporal_')):
            width = estimates[delta+'width_delta_s']['mean']
            score = estimates[delta+'score_delta_s']['mean']
            lower = estimates[name+'_lower_penalty_delta_s']['mean']
            upper = estimates[name+'_upper_penalty_delta_s']['mean']
            assert isclose(score, width+lower+upper, abs_tol=1e-7)
            group['paired_cost'][name] = dict(width_delta_s=width, score_delta_s=score,
                lower_penalty_delta_s=lower, upper_penalty_delta_s=upper,
                repaired_above=estimates[name+'_repaired_above'],
                new_above=estimates[name+'_new_above'])
            if name == 'v172':
                assert abs(lower) < 1e-8 and width >= 0
                assert estimates[name+'_new_above']['mean'] == 0
        report['groups'].append(group)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    destination = args.root/'interpretation/summary/status.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('V177_READONLY_INTERPRETATION_COMPLETE', result['gate']['decision'], flush=True)
