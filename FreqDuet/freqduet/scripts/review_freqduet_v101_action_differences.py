#!/usr/bin/env python3
"""Cached-only attribution of V101 reward, cost, entropy and system effects."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata

import run_freqduet_v101_native_action_differences as source


def rank_correlation(x, y):
    x, y = rankdata(x), rankdata(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def decompose(p, b, q, qp, cq, reward, cost, alpha, lam):
    change = np.asarray(b) - p
    row = dict(soft_reward_gain=float(change @ reward), cost_term_gain=float(-lam * (change @ cost)),
        entropy_gain=alpha * (source.entropy(b) - source.entropy(p)),
        pessimism_gain=float(-change @ (q - qp)),
        reward_prediction_error=float(change @ (q - reward)),
        cost_term_prediction_error=float(-lam * (change @ (cq - cost))))
    row['native_soft_objective'] = sum(row[k] for k in ('soft_reward_gain', 'cost_term_gain', 'entropy_gain'))
    row['frozen_pessimism_soft_objective'] = row['native_soft_objective'] + row['pessimism_gain']
    row['predicted_frozen_objective'] = float(change @ (qp - lam * cq)) + row['entropy_gain']
    np.testing.assert_allclose(row['predicted_frozen_objective'] - row['frozen_pessimism_soft_objective'],
        row['reward_prediction_error'] + row['cost_term_prediction_error'], atol=1e-10, rtol=0)
    return row


def curve_audit(predicted, samples):
    samples = np.asarray(samples)
    differences = samples - samples[:, :1]
    mean, prediction = differences.mean(0), np.asarray(predicted) - predicted[0]
    variance = differences.var(0, ddof=1) / len(samples)
    first, second = differences[:2].mean(0), differences[2:].mean(0)
    signal, correct = 0, 0
    for i in range(7):
        for j in range(i + 1, 7):
            delta = samples[:, i] - samples[:, j]
            if abs(delta.mean()) > delta.std(ddof=1):
                signal += 1
                correct += int((predicted[i] - predicted[j]) * delta.mean() > 0)
    return dict(rank_correlation=rank_correlation(prediction, mean),
        split_half_rank_correlation=rank_correlation(first, second),
        difference_mae=float(np.abs(prediction[1:] - mean[1:]).mean()),
        difference_mse=float(np.mean((prediction[1:] - mean[1:]) ** 2)),
        label_mean_variance=float(variance[1:].mean()),
        squared_error_minus_label_mean_variance=float(np.mean((prediction[1:] - mean[1:]) ** 2 - variance[1:])),
        pairs_signal_above_single_future_sd=signal, such_pairs_with_correct_predicted_sign=correct)


def review(root, output):
    summary = source.read(root / 'summary/status.json')
    if summary['status'] != 'native_action_difference_audit_complete':
        raise AssertionError('V101 incomplete')
    rows, contexts = [], []
    for ci, (ri, target) in enumerate(source.CONTEXTS):
        ref = source.read(root / 'references' / f'r{ri}.json')
        c = ref['captures'][str(target)]
        p, q, qp, cq = [np.asarray(c[k]) for k in ('p', 'q', 'q_policy', 'cost')]
        b = source.boltzmann(qp - c['lam'] * cq, c['alpha'])
        rewards, costs, decompositions = [], [], []
        original = summary['contexts'][ci]
        if (original['seed'], original['day'], original['target']) != (ref['seed'], ref['day'], target):
            raise AssertionError('context grid changed')
        for f in source.FUTURES:
            arms = [source.read(root / 'arms' / f'a{source.ITEMS.index((ci, f, a))}.json') for a in range(7)]
            reward, cost = [np.asarray([r['local'][k] for r in arms]) for k in ('soft_reward', 'cost_return')]
            row = decompose(p, b, q, qp, cq, reward, cost, c['alpha'], c['lam'])
            for k in ('native_soft_objective', 'frozen_pessimism_soft_objective'):
                np.testing.assert_allclose(row[k], original['effects'][f][k], atol=1e-10, rtol=0)
            row.update({k: original['effects'][f][k] for k in source.frozen.METRICS})
            rows.append(dict(seed=ref['seed'], day=ref['day'], effects=dict(boltzmann_vs_actor=row)))
            rewards.append(reward)
            costs.append(cost)
            decompositions.append(row)
        reward, cost = np.asarray(rewards), np.asarray(costs)
        contexts.append(dict(seed=ref['seed'], day=ref['day'], target=target,
            means={k: float(np.mean([r[k] for r in decompositions])) for k in decompositions[0]},
            local_gain_positive_futures=sum(r['frozen_pessimism_soft_objective'] > 0 for r in decompositions),
            plain_reward_gain_positive_futures=sum(r['soft_reward_gain'] > 0 for r in decompositions),
            positive_local_gain_in_both_halves=all(np.mean([r['frozen_pessimism_soft_objective'] for r in half]) > 0
                for half in (decompositions[:2], decompositions[2:])),
            curves=dict(reward=curve_audit(q, reward), cost=curve_audit(cq, cost),
                combined_score=curve_audit(qp - c['lam'] * cq, reward - c['lam'] * cost - (q - qp)))))
    def means(selected):
        return {k: float(np.mean([r['means'][k] for r in selected])) for k in selected[0]['means']}
    curves = {}
    for mode in ('reward', 'cost', 'combined_score'):
        values = [r['curves'][mode] for r in contexts]
        curves[mode] = {k: float(np.mean([v[k] for v in values if v[k] is not None]))
            if any(v[k] is not None for v in values) else None for k in values[0]}
        curves[mode]['signal_pairs_total'] = sum(v['pairs_signal_above_single_future_sd'] for v in values)
        curves[mode]['correct_sign_signal_pairs_total'] = sum(v['such_pairs_with_correct_predicted_sign'] for v in values)
    positive = [r for r in contexts if r['means']['frozen_pessimism_soft_objective'] > 0]
    source.pilot.write_json(output / 'status.json', dict(status='cached_v101_attribution_complete',
        effects=source.confirmation.describe(rows, 'effects'), curves=curves,
        by_seed={str(s): means([r for r in contexts if r['seed'] == s]) for s in source.confirmation.VALIDATION_SEEDS},
        by_day={str(d): means([r for r in contexts if r['day'] == d]) for d in source.audit.prior.DAYS},
        by_time={str(t): means([r for r in contexts if r['target'] == t]) for t in source.TARGETS},
        descriptive_counts=dict(contexts=len(contexts), positive_local_gain=len(positive),
            positive_local_gain_both_halves=sum(r['positive_local_gain_in_both_halves'] for r in contexts),
            positive_local_gain_and_passenger_harm=sum(r['means']['passenger_person_s'] > 0 for r in positive),
            positive_local_gain_and_cv_harm=sum(r['means']['cv_squared'] > 0 for r in positive),
            positive_local_gain_and_readiness_harm=sum(r['means']['readiness_delay_s'] > 0 for r in positive),
            positive_local_gain_and_joint_system_nonharm=sum(all(r['means'][m] <= 0 for m in
                ('passenger_person_s', 'cv_squared', 'readiness_delay_s')) for r in positive)),
        contexts=contexts, new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, controller_admitted=False,
        scope='cached diagnostic decomposition; sign counts and curve correlations do not select a policy'))
    print('V101_CACHED_ATTRIBUTION_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v101-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v101_review*.py', '-v'], check=True)
    review(args.v101_root, args.out_root)
