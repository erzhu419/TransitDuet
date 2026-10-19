"""Fixed interval readouts: keep missing support separate from coverage."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scripts.run_freqduet_protocol_v2_matrix import hierarchical_bootstrap


def mean(values):
    return float(np.mean(values)) if len(values) else None


def describe(rows):
    finite = [r for r in rows if r['rank']['finite']]
    resolved = [r for r in finite if r['label_status'] == 'resolved']
    episodes = defaultdict(list)
    for r in resolved:
        episodes[tuple(r['episode'])].append(r)
    return dict(requests=len(rows), finite_n=len(finite), finite_resolved_n=len(resolved),
        unresolved_n=sum(r['label_status'] != 'resolved' for r in rows),
        finite_unresolved_n=len(finite)-len(resolved),
        label_statuses=dict(Counter(r['label_status'] for r in rows)),
        support_reasons=dict(Counter(r['support_reason'] for r in rows)),
        finite_fraction=None if not rows else len(finite)/len(rows),
        query_coverage=mean([r['covered'] for r in resolved]),
        episode_coverage=mean([np.mean([r['covered'] for r in v]) for v in episodes.values()]),
        finite_resolved_episodes=len(episodes),
        below_rate=mean([r['below'] for r in resolved]),
        above_rate=mean([r['above'] for r in resolved]),
        mean_width_s=mean([r['rank']['upper_s']-r['rank']['lower_s'] for r in finite]))


def strata(rows):
    result = {}
    for kind in ('service', 'ready'):
        selected = [r for r in rows if r['kind'] == kind]
        result[kind] = dict(all=describe(selected),
            hour13=describe([r for r in selected if r['hour'] == 13]),
            rest=describe([r for r in selected if r['hour'] != 13]),
            by_support={s: describe([r for r in selected if r['support_reason'] == s]) for s in
                ('no_completed_history', 'prefix_less_than_nine', 'window_less_than_nine',
                 'elapsed_survivor_depletion', 'finite')})
    return result


def crossed(rows, actors, days):
    expected = {(a, d) for a in actors for d in days}
    actual = [(r['seed'], r['day']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError('validation actor/day grid differs from registration')
    stats = [r['stats'] for r in rows]
    result = dict(cells=len(rows), contributing_cells=sum(s['episode_coverage'] is not None for s in stats),
        requests=sum(s['requests'] for s in stats), finite_n=sum(s['finite_n'] for s in stats),
        finite_resolved_n=sum(s['finite_resolved_n'] for s in stats),
        unresolved_n=sum(s['unresolved_n'] for s in stats),
        finite_unresolved_n=sum(s['finite_unresolved_n'] for s in stats),
        actor_day_finite_fraction=mean([s['finite_fraction'] for s in stats if s['finite_fraction'] is not None]),
        actor_day_mean_width_s=mean([s['mean_width_s'] for s in stats if s['mean_width_s'] is not None]),
        episode_coverage=mean([s['episode_coverage'] for s in stats if s['episode_coverage'] is not None]),
        coverage_ci95=None, coverage_lower_one_sided95=None, coverage_confirmed=False,
        decision='inconclusive_support', label_statuses=dict(sum((Counter(s['label_statuses']) for s in stats), Counter())),
        support_reasons=dict(sum((Counter(s['support_reasons']) for s in stats), Counter())))
    for name in ('query_coverage', 'below_rate', 'above_rate'):
        result['actor_day_'+name] = mean([s[name] for s in stats if s[name] is not None])
    if result['contributing_cells'] != len(rows):
        return result
    frame = pd.DataFrame([dict(train_seed=r['seed'], eval_seed=r['day'],
        coverage=r['stats']['episode_coverage']) for r in rows])
    draws = hierarchical_bootstrap(frame, 'coverage', draws=5000)
    result['coverage_ci95'] = np.percentile(draws, [2.5, 97.5]).tolist()
    result['coverage_lower_one_sided95'] = float(np.percentile(draws, 5.))
    if result['finite_unresolved_n']:
        result['decision'] = 'inconclusive_unresolved_labels'
    elif result['coverage_lower_one_sided95'] >= .8:
        result.update(coverage_confirmed=True, decision='finite_coverage_confirmed')
    else:
        result['decision'] = 'coverage_not_confirmed'
    return result
