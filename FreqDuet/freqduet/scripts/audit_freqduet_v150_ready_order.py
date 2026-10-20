#!/usr/bin/env python3
"""Read saved callback insertion order, without issuing predictions or replaying."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v150_service_support as prior


parts, frozen = prior.parts, prior.frozen
GROUPS = ('ready_observed_before_query', 'ready_later_same_tick', 'ready_after_query_time')


def ready_order(rows):
    ranks, clocks, labels = {}, {}, {}
    previous = -float('inf')
    for rank, row in enumerate(rows):
        k, now = parts.key(row), row['ready_time_s']
        clock = k[0], k[2], now
        if now < previous or k in ranks or clock in clocks:
            raise AssertionError('ready archive cannot establish unique chronological callback order')
        previous = now
        ranks[k], clocks[clock], labels[k] = rank, k, row
    return ranks, clocks, labels


def classify(row, order):
    ranks, clocks, labels = order
    own, e, now = parts.key(row), row['evidence'], row['prediction_time_s']
    if labels[own]['ready_time_s'] != now:
        raise AssertionError('prediction does not use its own ready callback')
    same = clocks.get((e['predicted_follower_trip_id'], bool(row['direction']), now))
    if same is None:
        return 'ready_after_query_time'
    if labels[same]['bus_id'] != e['predicted_follower_bus_id']:
        raise AssertionError('same-clock ready label is not the predicted physical follower')
    if same == own:
        raise AssertionError('predicted follower is the querying vehicle')
    return GROUPS[0] if ranks[same] < ranks[own] else GROUPS[1]


def case(root, out, seed, day):
    counts, modes = Counter(), {}
    for mode in frozen.MODES:
        name = f'{seed}_{day}_{mode}.jsonl.gz'
        read_trace = parts.prior.trace
        ready = read_trace(root/'confirmation/ready_labels'/name)
        order = ready_order(ready)
        points = {tuple(r['query']): r for r in read_trace(root/'confirmation/candidate_shadows'/name)}
        groups = defaultdict(list)
        for row in read_trace(root/'confirmation/traces'/name):
            k = parts.key(row)
            if points[k]['status'] != 'no_completed_service_survivor':
                continue
            resolved, truth = parts.prior.target_label(row, order[2])
            if resolved != 'resolved' or points[k]['points']['base'] is None:
                continue
            group = classify(row, order)
            data = dict(episode=[*parts.prior.follower(row), *k[1:]],
                full_bias_s=points[k]['points']['base']-truth,
                full_mae_s=abs(points[k]['points']['base']-truth),
                **{g: float(group == g) for g in GROUPS})
            groups['all_service_fallback'].append(data)
            groups[group].append(data)
            counts[mode+'/'+group] += 1
        expected = prior.read(out/'cells'/f'{seed}_{day}.json')['counts']
        n = sum(v for k, v in expected.items() if k.startswith(mode+'/fallback/') and
            '/survivor_depleted_pool_' in k)
        if len(groups['all_service_fallback']) != n:
            raise AssertionError('order audit changed the already-reconciled service-fallback population')
        modes[mode] = {g: parts.prior.describe(groups[g]) for g in ('all_service_fallback', *GROUPS)}
    prior.write(out/'ready_order_cells'/f'{seed}_{day}.json',
        dict(seed=seed, day=day, modes=modes, counts=dict(counts)))


def run(args):
    if prior.read(args.out_root/'summary/status.json')['status'] != 'v150_readonly_service_support_complete':
        raise AssertionError('requires completed V150 support accounting')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v150_ready_order.py', '-v'], check=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v149_root, args.out_root, s, d) for s, d in frozen.CASES]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 64 == 0:
                print('V150_READY_ORDER_PROGRESS', n, len(jobs), flush=True)
    cells = [prior.read(args.out_root/'ready_order_cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    prior.write(args.out_root/'summary/ready_order.json', dict(
        status='v150_readonly_ready_order_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        support_source=prior.read(args.out_root/'summary/status.json')['execution_source'],
        cells=len(cells), saved_paths=2*len(cells),
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        results={m: parts.prior.aggregate(cells, lambda c: c['modes'][m]) for m in frozen.MODES},
        order_evidence='V145 writes IdentityAudit.ready.items() in insertion order. '
            'IdentityAudit.prepare inserts after the native prepare returns; OnlineAudit.prepare '
            'then updates stream.ready before returning to the sequential bus-update loop. '
            'A prior same-clock ready callback therefore completed before the querying callback.',
        limitations='Ready ordering establishes event observability, not follower-load availability '
            'or the performance of a changed predictor. Future ready labels are diagnostic only.',
        new_rollouts=0, fitted_parameters=0, actor_updates=0, critic_updates=0,
        dual_updates=0, main_policy_changed=False))
    print('V150_READONLY_READY_ORDER_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v149-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
