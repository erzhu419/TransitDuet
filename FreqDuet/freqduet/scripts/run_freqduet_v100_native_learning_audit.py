#!/usr/bin/env python3
"""Audit frozen native B200 learning targets, without changing its controller."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v99_objective_authority_audit as prior


benchmark = prior.prior.prior
pilot, frozen, confirmation = prior.pilot, prior.frozen, prior.confirmation
ITEMS = list(itertools.product(confirmation.VALIDATION_SEEDS, prior.DAYS))
REWARD_SCALE = 1.0  # runner_v3's actual update call, not the trainer's default 10.


def errors(prediction, target):
    delta = np.asarray(prediction, float) - np.asarray(target, float)
    return dict(bias=float(delta.mean()), mae=float(np.abs(delta).mean()),
                rmse=float(np.sqrt(np.mean(delta * delta))))


def actor_gap(probability, energy, mask, alpha):
    p, energy, mask = np.asarray(probability), np.asarray(energy), np.asarray(mask, bool)
    logits = np.where(mask, -energy / alpha, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    logz = np.log(np.exp(logits).sum(axis=1, keepdims=True))
    logb = logits - logz
    logp = np.log(np.maximum(p, np.finfo(float).tiny))
    kl = np.where(p > 0, p * (logp - np.where(mask, logb, 0.)), 0.).sum(axis=1)
    return dict(kl_actor_to_boltzmann=float(kl.mean()), soft_objective_gap=float(alpha * kl.mean()),
        total_variation=float(np.abs(p - np.exp(logb)).sum(axis=1).mean() / 2),
        actor_expected_holding_s=float(np.mean(p @ pilot.step.ACTIONS)),
        boltzmann_expected_holding_s=float(np.mean(np.exp(logb) @ pilot.step.ACTIONS)),
        entropy=float(np.mean(-(p * logp).sum(axis=1))))


def soft_trip_returns(rows, logp, gamma, alpha):
    reward, cost = np.zeros(len(rows)), np.zeros(len(rows))
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['trip'], []).append(i)
    for indices in groups.values():
        if not rows[indices[-1]]['done']:
            raise AssertionError('native local-trip return is censored')
        for a, b in zip(indices, indices[1:]):
            if rows[a]['done']:
                raise AssertionError('trip identity reused after terminal transition')
            np.testing.assert_array_equal(rows[a]['next_state'], rows[b]['state'],
                                          err_msg='native replay next-state chain')
        following = None
        for i in reversed(indices):
            reward[i], cost[i] = REWARD_SCALE * rows[i]['reward'], rows[i]['cost']
            if following is not None:
                reward[i] += gamma * (reward[following] - alpha * logp[following])
                cost[i] += gamma * cost[following]
            following = i
    return reward, cost, len(groups)


def prepare(root, v99_root):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v100_*.py', '-v'], check=True)
    rows = [json.loads((v99_root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(prior.ITEMS))]
    paired = []
    for seed, day in ITEMS:
        a, b = [next(r for r in rows if (r['seed'], r['day'], r['mode']) == (seed, day, mode))
                for mode in prior.MODES]
        credit = {k: a['credit'][k] - b['credit'][k] for k in (
            'recorded_shaped_reward_sum', 'regularity_reward_sum', 'load_penalty_sum', 'other_reward_sum')}
        np.testing.assert_allclose(credit['recorded_shaped_reward_sum'], credit['other_reward_sum']
            + credit['regularity_reward_sum'] - credit['load_penalty_sum'], rtol=0, atol=1e-9)
        paired.append(dict(seed=seed, day=day, credit=credit,
            effects={m: a['results'][m] - b['results'][m] for m in frozen.METRICS}))
    def means(values):
        return {k: float(np.mean([r[k] for r in values])) for k in values[0]}
    def conflict(values):
        return sum(r['credit']['recorded_shaped_reward_sum'] > 0 and
                   r['effects']['passenger_person_s'] > 0 for r in values)
    axes = {axis: [dict(credit=means([r['credit'] for r in paired if r[axis] == value]),
                       effects=means([r['effects'] for r in paired if r[axis] == value]))
                   for value in sorted({r[axis] for r in paired})] for axis in ('seed', 'day')}
    pilot.write_json(root / 'prepare/status.json', dict(status='tests_and_v99_reward_join_complete',
        credit_delta=means([r['credit'] for r in paired]), physical_delta=means([r['effects'] for r in paired]),
        absolute_credit={mode: means([{k: r['credit'][k] for k in paired[0]['credit']}
            for r in rows if r['mode'] == mode]) for mode in prior.MODES},
        reward_up_passenger_cost_up=dict(cells=conflict(paired), seeds=conflict(axes['seed']),
                                         days=conflict(axes['day'])), cells=paired,
        scope='two consumed days; reward ownership audit, not independent efficacy confirmation'))
    print('V100_PREPARE_COMPLETE', flush=True)


def native_values(trainer, states):
    import torch
    values = {k: [] for k in ('p', 'logp', 'q', 'q_policy', 'cost', 'target_q', 'target_cost', 'mask')}
    with torch.no_grad():
        for start in range(0, len(states), 512):
            state = torch.as_tensor(np.asarray(states[start:start + 512]), dtype=torch.float32, device=trainer.device)
            p, logp, _ = trainer.policy_net.dist_info(state)
            qall = trainer._discrete_q_values(trainer.q_net, state)
            q_policy, q, _ = trainer._policy_q_value(qall)
            batch = dict(p=p, logp=logp, q=q, q_policy=q_policy,
                cost=trainer._discrete_cost_values(trainer.cost_q_net, state),
                target_q=trainer._aggregate_target_q(trainer._discrete_q_values(trainer.target_q_net, state)),
                target_cost=trainer._discrete_cost_values(trainer.target_cost_q_net, state),
                mask=trainer.policy_net.feasible_action_mask(state))
            for key, value in batch.items():
                values[key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in values.items()}


class LearningAudit(benchmark.BenchmarkAudit):
    def __init__(self, seed):
        super().__init__(seed, 'late')
        self.transitions = []

    def attach(self, runner):
        super().attach(runner)
        self.saved_transition = runner._record_lower_transition
        trainer = runner.lower_trainer
        self.frozen_values = copy.deepcopy([net.state_dict() for net in (
            trainer.q_net, trainer.target_q_net, trainer.cost_q_net, trainer.target_cost_q_net)])
        self.scalars = (float(trainer.alpha), float(trainer.lambda_param), float(trainer.gamma))
        if runner.coupling_mode != 'hiro':
            raise AssertionError('audit targets require the deployed HIRO/no-TAP update contract')
        def record(**event):
            if event['learned_training']:
                raise AssertionError('passive audit must not enable training')
            action = runner._lower_action_scalar(event['action']) if event['action'] is not None else 0.
            state = runner._augment_lower_state(np.asarray(event['raw_state'], np.float32), event['previous_action'])
            next_state = np.zeros_like(state) if event['raw_next_state'] is None else runner._augment_lower_state(
                np.asarray(event['raw_next_state'], np.float32), action)
            bus = event.get('bus')
            if bus is None:
                bus = runner._bus_for_agent(event['key'])
            trip = event.get('trip_id')
            trip = int(bus.trip_id if trip is None else trip)
            reward, cost, applied = self.saved_transition(**event)
            if applied != action:
                raise AssertionError('recorded action changed in passive transition capture')
            self.transitions.append(dict(state=state.copy(), next_state=next_state.copy(),
                action=int(np.flatnonzero(pilot.step.ACTIONS == action).item()), reward=float(reward),
                cost=float(cost), raw_reward=float(event['reward']), raw_cost=float(event['cost']),
                done=bool(event['transition_done']), trip=trip, station=int(event['raw_state'][1])))
            return reward, cost, applied
        runner._record_lower_transition = record

    def detach(self):
        self.runner._record_lower_transition = self.saved_transition
        super().detach()

    def finish(self, runner, records, holds):
        trainer, rows = runner.lower_trainer, self.transitions
        for net, old in zip((trainer.q_net, trainer.target_q_net, trainer.cost_q_net, trainer.target_cost_q_net), self.frozen_values):
            pilot.joint.assert_same_state(net.state_dict(), old, 'frozen_native_critic')
        alpha, lam, gamma = self.scalars
        if self.scalars != (float(trainer.alpha), float(trainer.lambda_param), float(trainer.gamma)):
            raise AssertionError('native learning scalars changed')
        current = native_values(trainer, [r['state'] for r in rows])
        following = native_values(trainer, [r['next_state'] for r in rows])
        action = np.asarray([r['action'] for r in rows])
        index = np.arange(len(rows)), action
        observed = {(r['trip_id'], r['station_id']): r for r in records}
        if len(observed) != len(records) or len(rows) != len(records):
            raise AssertionError('native decision/transition identities are not one-to-one')
        for i, row in enumerate(rows):
            old = observed[(row['trip'], row['station'])]
            if old['action'] != row['action']:
                raise AssertionError('native replay action mismatch')
            np.testing.assert_allclose(current['p'][i], old['probabilities'], atol=2e-6, rtol=0)
        reward, cost = [np.asarray([r[k] for r in rows]) for k in ('reward', 'cost')]
        not_done = 1. - np.asarray([r['done'] for r in rows])
        next_v = (following['p'] * (following['target_q'] - alpha * following['logp'])).sum(1)
        next_c = (following['p'] * following['target_cost']).sum(1)
        rbackup, cbackup = REWARD_SCALE * reward + not_done * gamma * next_v, cost + not_done * gamma * next_c
        mc, cmc, trips = soft_trip_returns(rows, current['logp'][index], gamma, alpha)
        raw = float(sum(r['raw_reward'] for r in rows))
        regularity = float(sum(runner._ep_lower_departure_regularity_rewards))
        load = float(sum(runner._ep_lower_load_hold_penalties))
        residual = float(reward.sum() - raw - regularity + load)
        np.testing.assert_allclose(residual, 0, atol=1e-8, rtol=0,
            err_msg='additional shaping penalties are active: inspect before attributing other reward to raw headway')
        self.learning = dict(decisions=len(rows), completed_trip_chains=trips,
            contract=dict(alpha=alpha, lambda_value=lam, gamma=gamma, reward_scale=REWARD_SCALE,
                reward_backup_clip=[-100, 100], cost_backup_clip=[0, 50],
                native_replay_states_and_probabilities_match=True, local_trip_chains_exact=True),
            actor=actor_gap(current['p'], -current['q_policy'] + lam * current['cost'], current['mask'], alpha),
            calibration=dict(q_mean_vs_train_backup=errors(current['q'][index], np.clip(rbackup, -100, 100)),
                cost_vs_train_backup=errors(current['cost'][index], np.clip(cbackup, 0, 50)),
                q_mean_vs_native_soft_return=errors(current['q'][index], mc),
                q_policy_vs_native_soft_return=errors(current['q_policy'][index], mc),
                cost_vs_native_return=errors(current['cost'][index], cmc)),
            target_clipping=dict(reward_fraction=float(np.mean(rbackup != np.clip(rbackup, -100, 100))),
                                 cost_fraction=float(np.mean(cbackup != np.clip(cbackup, 0, 50)))),
            reward=dict(raw_headway_sum=raw, shaped_sum=float(reward.sum()), regularity_sum=regularity,
                        load_penalty_sum=load, unexplained_shaping_sum=residual))
        super().finish(runner, records, holds)


def run_item(root, v96_root, index, full_path=False):
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        seed, day = ITEMS[index]
        rule, credit = LearningAudit(seed), pilot.CreditObserver()
        row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000 + seed + 100 * day,
            [0.] * 4, f'v100_seed{seed}_day{day}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
        old_index = benchmark.ITEMS.index((seed, day, 0))
        old = json.loads((v96_root / 'pairs' / f'p{old_index}.json').read_text())['results']['late']
        for metric in (*frozen.METRICS, 'action_counts'):
            pilot.joint.assert_same_state(old[metric], row[metric], 'v100_native_cached_' + metric)
        if full_path:
            current = root / 'smoke_records/p0.npz'
            rule.save(current)
            with np.load(v96_root / 'records' / f'p{old_index}_late.npz') as a, np.load(current) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        pilot.write_json(path, dict(seed=seed, day=day, learning=rule.learning, results=row, cached_endpoints_exact=True))
    print('V100_NATIVE_AUDIT_ITEM_DONE ' + str(index), flush=True)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + '/'))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [(r['seed'], r['day']) for r in rows] != ITEMS:
        raise AssertionError('native audit grid changed')
    flat = [flatten(r['learning']) for r in rows]
    pilot.write_json(root / 'summary/status.json', dict(status='native_learning_contract_audit_complete',
        means={k: float(np.mean([r[k] for r in flat])) for k in flat[0]},
        by_seed={str(s): {k: float(np.mean([f[k] for r, f in zip(rows, flat) if r['seed'] == s]))
                         for k in flat[0]} for s in confirmation.VALIDATION_SEEDS}, rows=rows,
        v99_reward_join=json.loads((root / 'prepare/status.json').read_text()),
        rollouts=len(rows), actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        controller_admitted=False, cached_endpoints_exact=all(r['cached_endpoints_exact'] for r in rows),
        scope='native B200 local-trip soft targets on two consumed days; no action-value truth or new efficacy test'))
    print('V100_NATIVE_LEARNING_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--v99-root', type=Path)
    parser.add_argument('--v96-root', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=len(ITEMS))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_root, args.v99_root)
    elif args.command == 'aggregate':
        aggregate(args.out_root)
    elif args.command == 'smoke':
        run_item(args.out_root, args.v96_root, 0, full_path=True)
        pilot.write_json(args.out_root / 'smoke/done.json', dict(status='native_learning_contract_and_cached_timing_exact'))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            for task in [pool.submit(run_item, args.out_root, args.v96_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f'shards/{args.start}-{args.end}.json', dict(status='complete'))
