#!/usr/bin/env python3
"""Cross-day, critic-only TD versus native Monte Carlo target repair."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import review_freqduet_v101_action_differences as review


source, audit = review.source, review.source.audit
pilot, confirmation = source.pilot, source.confirmation
STEPS, BATCH = 64, 512
MODES = ('native', 'td', 'mc')


class NativeBatch:
    def __init__(self, rows, returns, mode, seed):
        self.rows, self.returns, self.mode = rows, returns, mode
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        ids = self.rng.integers(len(self.rows), size=size)
        rows = [self.rows[i] for i in ids]
        state = np.asarray([r['state'] for r in rows], np.float32)
        action = np.asarray([pilot.step.ACTIONS[r['action']] for r in rows], np.float32)[:, None]
        if self.mode == 'mc':
            reward, cost = [a[ids, None] for a in self.returns]
            # The native updater is used as supervised return regression here.
            # These are regression labels, not altered environment transitions.
            following, done = np.zeros_like(state), np.ones((size, 1))
        else:
            reward, cost = [np.asarray([r[k] for r in rows])[:, None] for k in ('reward', 'cost')]
            following = np.asarray([r['next_state'] for r in rows], np.float32)
            done = np.asarray([r['done'] for r in rows])[:, None]
        return state, action, reward, cost, following, done, np.asarray([r['trip'] for r in rows])


def refit(trainer, rows, returns, mode, seed, steps=STEPS, batch=BATCH):
    if mode == 'mc' and (np.min(returns[0]) < -100 or np.max(returns[0]) > 100
                        or np.min(returns[1]) < 0 or np.max(returns[1]) > 50):
        raise AssertionError('MC labels would be clipped by the native updater')
    fitted = copy.deepcopy(trainer)
    policy = copy.deepcopy(fitted.policy_net.state_dict())
    scalars = (float(fitted.alpha), float(fitted.lambda_param), float(fitted.gamma))
    replay = NativeBatch(rows, returns, mode, seed)
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        losses.append(fitted.update(replay, batch_size=batch, reward_scale=audit.REWARD_SCALE,
                                    update_policy=False))
    pilot.joint.assert_same_state(policy, fitted.policy_net.state_dict(), 'v102_frozen_actor')
    if scalars != (float(fitted.alpha), float(fitted.lambda_param), float(fitted.gamma)):
        raise AssertionError('V102 changed alpha, dual or gamma')
    return fitted, dict(updates=steps, batch_size=batch,
        first_loss={k: losses[0][k] for k in ('q_mse', 'cost_q_loss')},
        last_loss={k: losses[-1][k] for k in ('q_mse', 'cost_q_loss')})


class Capture(audit.LearningAudit):
    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.trainer = runner.lower_trainer
        values = audit.native_values(self.trainer, [r['state'] for r in self.transitions])
        logp = values['logp'][np.arange(len(self.transitions)), [r['action'] for r in self.transitions]]
        self.returns = audit.soft_trip_returns(self.transitions, logp,
            self.trainer.gamma, self.trainer.alpha)[:2]


def other_day_index(index):
    seed, day = audit.ITEMS[index]
    other, = [d for d in audit.prior.DAYS if d != day]
    return audit.ITEMS.index((seed, other))


def evaluate(trainer, v101_root, held_index):
    ref = source.read(v101_root / 'references' / f'r{held_index}.json')
    output = []
    for target in source.TARGETS:
        capture = ref['captures'][str(target)]
        values = audit.native_values(trainer, [capture['state']])
        p = np.asarray(capture['p'])
        np.testing.assert_allclose(values['p'][0], p, atol=2e-6, rtol=0)
        q, cq = values['q'][0], values['cost'][0]
        charge = np.asarray(capture['q']) - capture['q_policy']
        qp = q - charge  # Keep the old uncertainty charge fixed across refits.
        b = source.boltzmann(qp - capture['lam'] * cq, capture['alpha'])
        ci = source.CONTEXTS.index((held_index, target))
        rewards, costs, physical = [], [], []
        for future in source.FUTURES:
            arms = [source.read(v101_root / 'arms' / f'a{source.ITEMS.index((ci, future, a))}.json')
                    for a in range(7)]
            rewards.append([r['local']['soft_reward'] for r in arms])
            costs.append([r['local']['cost_return'] for r in arms])
            physical.append({m: float((b - p) @ [r['results'][m] for r in arms]) for m in source.frozen.METRICS})
        rewards, costs = np.asarray(rewards), np.asarray(costs)
        curves = dict(reward=review.curve_audit(q, rewards), cost=review.curve_audit(cq, costs),
            combined=review.curve_audit(qp - capture['lam'] * cq,
                rewards - capture['lam'] * costs - charge))
        for future in source.FUTURES:
            effect = review.decompose(p, b, q, qp, cq, rewards[future], costs[future],
                                      capture['alpha'], capture['lam'])
            effect.update(physical[future])
            effect.update({name + '_' + key: curve[key] for name, curve in curves.items()
                for key in ('difference_mse', 'difference_mae', 'squared_error_minus_label_mean_variance')})
            output.append(dict(target=target, future=future, effects=effect))
    return output


def run_item(root, v100_root, v101_root, index, full_path=False):
    path = root / 'pairs' / f'p{index}.json'
    if path.exists():
        return
    seed, day = audit.ITEMS[index]
    held_index = other_day_index(index)
    rule, credit = Capture(seed), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000 + seed + 100 * day,
        [0.] * 4, f'v102_train{index}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    old = source.read(v100_root / 'pairs' / f'p{index}.json')['results']
    for metric in (*source.frozen.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(old[metric], row[metric], 'v102_cached_native_' + metric)
    if full_path:
        rule.save(root / 'smoke_records/p0.npz')
        with np.load(v101_root / 'smoke_records/reference.npz') as a, np.load(root / 'smoke_records/p0.npz') as b:
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    original = copy.deepcopy([net.state_dict() for net in (rule.trainer.q_net, rule.trainer.cost_q_net)])
    results, training = dict(native=evaluate(rule.trainer, v101_root, held_index)), {}
    for mode in ('td', 'mc'):
        fitted, training[mode] = refit(rule.trainer, rule.transitions, rule.returns, mode, seed + day)
        results[mode] = evaluate(fitted, v101_root, held_index)
    pilot.joint.assert_same_state(original,
        [net.state_dict() for net in (rule.trainer.q_net, rule.trainer.cost_q_net)], 'v102_original_critics')
    native = source.read(v101_root / 'summary/status.json')['contexts']
    for result in results['native']:
        ci = source.CONTEXTS.index((held_index, result['target']))
        for metric in ('native_soft_objective', 'frozen_pessimism_soft_objective', *source.frozen.METRICS):
            np.testing.assert_allclose(result['effects'][metric], native[ci]['effects'][result['future']][metric],
                                       atol=2e-5, rtol=1e-7, err_msg='v102_cached_native_score_' + metric)
    pilot.write_json(path, dict(seed=seed, train_day=day, day=audit.ITEMS[held_index][1],
        training=training, results=results, cached_native_exact=True,
        actor_updates=0, dual_updates=0, main_policy_changed=False))
    print('V102_REFIT_ITEM_DONE ' + str(index), flush=True)


def aggregate(root):
    rows = [source.read(root / 'pairs' / f'p{i}.json') for i in range(len(audit.ITEMS))]
    absolute, contrasts = [], []
    for row in rows:
        if row['train_day'] == row['day']:
            raise AssertionError('training and diagnostic day overlap')
        for j in range(len(row['results']['native'])):
            values = {mode: row['results'][mode][j]['effects'] for mode in MODES}
            absolute.append(dict(seed=row['seed'], day=row['day'], effects=values))
            effects = {a + '_vs_' + b: {m: values[a][m] - values[b][m] for m in values[a]}
                       for a, b in (('td', 'native'), ('mc', 'native'), ('mc', 'td'))}
            contrasts.append(dict(seed=row['seed'], day=row['day'], effects=effects))
    pilot.write_json(root / 'summary/status.json', dict(status='cross_day_critic_refit_complete',
        absolute=confirmation.describe(absolute, 'effects'), contrasts=confirmation.describe(contrasts, 'effects'),
        training=[{k: r[k] for k in ('seed', 'train_day', 'day', 'training')} for r in rows],
        native_replays=len(rows), new_counterfactual_rollouts=0, critic_updates_per_fit=STEPS,
        batch_size=BATCH, actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='cross-day development on consumed V101 days; native continuation, no deployed refit or policy efficacy claim'))
    print('V102_CROSS_DAY_CRITIC_REFIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v100-root', type=Path)
    p.add_argument('--v101-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(audit.ITEMS))
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v102_*.py', '-v'], check=True)
        run_item(args.out_root, args.v100_root, args.v101_root, 0, full_path=True)
        pilot.write_json(args.out_root / 'smoke/done.json', dict(status='native_path_and_frozen_critic_refit_passed'))
    elif args.command == 'aggregate':
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            for task in [pool.submit(run_item, args.out_root, args.v100_root, args.v101_root, i)
                         for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f'shards/{args.start}-{args.end}.json', dict(status='complete'))
