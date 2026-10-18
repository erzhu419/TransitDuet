#!/usr/bin/env python3
"""Review V102 and test conditional local-return teachers on disjoint futures."""

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v102_critic_return_refit as refit


source, review = refit.source, refit.review
SYSTEM = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s', 'holding_s_per_trip')
KEYS = ('reward_difference_mse', 'cost_difference_mse', 'combined_difference_mse',
        'reward_difference_mae', 'cost_difference_mae', 'combined_difference_mae',
        'predicted_frozen_objective', 'frozen_pessimism_soft_objective', 'native_soft_objective',
        'soft_reward_gain', 'cost_term_gain', 'entropy_gain', 'pessimism_gain',
        'gain_prediction_error', *SYSTEM)


def teacher_law(rewards, costs, charge, alpha, lam, training_futures):
    reward = np.asarray(rewards, float)[list(training_futures)].mean(0)
    cost = np.asarray(costs, float)[list(training_futures)].mean(0)
    return source.boltzmann(reward - charge - lam * cost, alpha), reward, cost


def gain_error(effects):
    result = dict(effects)
    result['gain_prediction_error'] = result['predicted_frozen_objective'] - result['frozen_pessimism_soft_objective']
    np.testing.assert_allclose(result['gain_prediction_error'],
        result['reward_prediction_error'] + result['cost_term_prediction_error'], atol=1e-10, rtol=0)
    return result


def axes(rows, keys):
    output = {}
    for axis in ('seed', 'day', 'target'):
        groups = defaultdict(list)
        for row in rows:
            groups[row[axis]].append(row['effects'])
        output[axis] = {}
        for name in rows[0]['effects']:
            means = {str(value): {k: float(np.mean([r[name][k] for r in group])) for k in keys}
                     for value, group in sorted(groups.items())}
            output[axis][name] = dict(means=means,
                negative_counts={k: sum(r[k] < 0 for r in means.values()) for k in keys},
                positive_counts={k: sum(r[k] > 0 for r in means.values()) for k in keys})
    return output


def local_conflicts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['seed'], row['day'], row['target'])].append(row['effects'])
    output = {}
    for name in rows[0]['effects']:
        means = [{k: float(np.mean([r[name][k] for r in values]))
                  for k in ('native_soft_objective', 'frozen_pessimism_soft_objective', *SYSTEM)}
                 for values in groups.values()]
        output[name] = dict(contexts=len(means))
        for objective in ('native_soft_objective', 'frozen_pessimism_soft_objective'):
            positive = [r for r in means if r[objective] > 0]
            output[name][objective] = dict(positive_contexts=len(positive),
                passenger_harm=sum(r['passenger_person_s'] > 0 for r in positive),
                cv_harm=sum(r['cv_squared'] > 0 for r in positive),
                readiness_harm=sum(r['readiness_delay_s'] > 0 for r in positive),
                joint_nonharm=sum(all(r[k] <= 0 for k in ('passenger_person_s', 'cv_squared', 'readiness_delay_s'))
                                  for r in positive))
    return output


def v102_review(root):
    old = source.read(root / 'summary/status.json')
    if old['status'] != 'cross_day_critic_refit_complete':
        raise AssertionError('V102 incomplete')
    rows, contrasts = [], []
    for index, (seed, day) in enumerate(refit.audit.ITEMS):
        pair = source.read(root / 'pairs' / f'p{index}.json')
        if (pair['seed'], pair['train_day'], pair['day']) != (seed, day, refit.audit.ITEMS[refit.other_day_index(index)][1]):
            raise AssertionError('V102 cross-day grid changed')
        for j, native in enumerate(pair['results']['native']):
            values = {mode: gain_error(pair['results'][mode][j]['effects']) for mode in refit.MODES}
            row = dict(seed=seed, day=pair['day'], target=native['target'], effects=values)
            rows.append(row)
            contrasts.append(dict(row, effects={a + '_vs_' + b: {k: values[a][k] - values[b][k] for k in KEYS}
                for a, b in (('td', 'native'), ('mc', 'native'), ('mc', 'td'))}))
    return dict(effects=source.confirmation.describe(rows, 'effects'),
        contrasts=source.confirmation.describe(contrasts, 'effects'), axes=axes(contrasts, KEYS),
        local_conflicts=local_conflicts(rows))


def audit_teacher(v101_root):
    old = source.read(v101_root / 'summary/status.json')
    if old['status'] != 'native_action_difference_audit_complete':
        raise AssertionError('V101 incomplete')
    rows = []
    for ci, (ri, target) in enumerate(source.CONTEXTS):
        ref = source.read(v101_root / 'references' / f'r{ri}.json')
        c = ref['captures'][str(target)]
        p, q, qp, cq = [np.asarray(c[k], float) for k in ('p', 'q', 'q_policy', 'cost')]
        charge = q - qp
        arms = [[source.read(v101_root / 'arms' / f'a{source.ITEMS.index((ci, future, a))}.json')
                 for a in range(7)] for future in source.FUTURES]
        rewards, costs = [np.asarray([[r['local'][k] for r in future] for future in arms])
                          for k in ('soft_reward', 'cost_return')]
        for train, held in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            teacher, tr, tc = teacher_law(rewards, costs, charge, c['alpha'], c['lam'], train)
            for future in held:
                values = {}
                for name, b, forecast_r, forecast_c in (
                    ('critic', source.boltzmann(qp - c['lam'] * cq, c['alpha']), q, cq),
                    ('teacher', teacher, tr, tc)):
                    effect = review.decompose(p, b, forecast_r, forecast_r - charge, forecast_c,
                        rewards[future], costs[future], c['alpha'], c['lam'])
                    effect.update({k: float((b - p) @ [r['results'][k] for r in arms[future]]) for k in source.frozen.METRICS})
                    values[name] = gain_error(effect)
                for k in ('native_soft_objective', 'frozen_pessimism_soft_objective', *SYSTEM):
                    np.testing.assert_allclose(values['critic'][k], old['contexts'][ci]['effects'][future][k], atol=1e-8, rtol=1e-7)
                values['teacher_vs_critic'] = {k: values['teacher'][k] - values['critic'][k] for k in values['teacher']}
                rows.append(dict(seed=ref['seed'], day=ref['day'], target=target, effects=values))
    keys = ('native_soft_objective', 'frozen_pessimism_soft_objective', 'gain_prediction_error', *SYSTEM)
    return dict(effects=source.confirmation.describe(rows, 'effects'), axes=axes(rows, keys),
        local_conflicts=local_conflicts(rows),
        scope='two-future conditional teacher evaluated on disjoint two-future labels, then swap; not deployable or a perfect-value upper bound')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v101-root', type=Path, required=True)
    p.add_argument('--v102-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v103_*.py', '-v'], check=True)
    source.pilot.write_json(args.out_root / 'summary/status.json', dict(status='held_future_local_teacher_audit_complete',
        v102=v102_review(args.v102_root), local_teacher=audit_teacher(args.v101_root),
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, controller_admitted=False))
    print('V103_HELD_FUTURE_LOCAL_TEACHER_COMPLETE', flush=True)
