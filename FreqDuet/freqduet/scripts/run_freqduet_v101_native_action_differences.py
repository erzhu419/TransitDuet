#!/usr/bin/env python3
"""Separate native critic action errors from local/system objective conflict."""

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
import torch

import run_freqduet_v100_native_learning_audit as audit
import run_freqduet_v68_repeated_continuations as repeated


pilot, frozen, confirmation = audit.pilot, audit.frozen, audit.confirmation
TARGETS, FUTURES = (7200, 21600), range(4)
CONTEXTS = list(itertools.product(range(len(audit.ITEMS)), TARGETS))
ITEMS = list(itertools.product(range(len(CONTEXTS)), FUTURES, range(7)))


def read(path):
    return json.loads(path.read_text())


def boltzmann(score, alpha):
    z = np.asarray(score, float) / alpha
    p = np.exp(z - z.max())
    return p / p.sum()


def entropy(p):
    p = np.asarray(p, float)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


def objective_effect(p, b, reward, cost, pessimism, alpha, lam):
    change = np.asarray(b) - p
    entropy_change = alpha * (entropy(b) - entropy(p))
    plain = float(change @ (np.asarray(reward) - lam * np.asarray(cost))) + entropy_change
    return dict(native_soft_objective=plain,
                frozen_pessimism_soft_objective=plain - float(change @ pessimism))


def prepare(root, v100_root):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v101_*.py', '-v'], check=True)
    old = read(v100_root / 'summary/status.json')
    if old['status'] != 'native_learning_contract_audit_complete' or not old['cached_endpoints_exact']:
        raise AssertionError('V100 is not complete')
    seed_rows = old['by_seed']
    fields = ('actor/kl_actor_to_boltzmann', 'actor/total_variation',
              'calibration/q_mean_vs_native_soft_return/bias', 'calibration/cost_vs_native_return/bias')
    pilot.write_json(root / 'prepare/status.json', dict(status='tests_and_v100_review_complete',
        seed_ranges={k: [min(r[k] for r in seed_rows.values()), max(r[k] for r in seed_rows.values())] for k in fields},
        by_day={str(d): {k: float(np.mean([audit.flatten(r['learning'])[k] for r in old['rows'] if r['day'] == d]))
                        for k in fields} for d in audit.prior.DAYS},
        v100_means=old['means'], main_policy_changed=False))
    print('V101_PREPARE_COMPLETE', flush=True)


class ActionAudit(audit.LearningAudit):
    def __init__(self, seed, targets, uniforms, *, reference=None, action=None, future=None):
        super().__init__(seed)
        self.targets, self.uniforms = targets, uniforms
        self.reference, self.action, self.future = reference, action, future
        self.captures, self.upper_path = {}, []

    def attach(self, runner):
        super().attach(runner)
        self.saved_lower, self.saved_upper = runner._lower_action_for_agent, runner._upper_callback_v2
        def lower(obs, key, last_action=0., deterministic=False):
            self.current_state = runner._augment_lower_state(np.asarray(obs, np.float32), last_action).copy()
            return self.saved_lower(obs, key, last_action=last_action, deterministic=deterministic)
        def upper(state, trip):
            value = self.saved_upper(state, trip)
            self.upper_path.append([float(runner.env.current_time), int(trip.launch_turn), float(value)])
            return value
        runner._lower_action_for_agent, runner._upper_callback_v2 = lower, upper

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        runner, now = self.runner, float(self.runner.env.current_time)
        bus = runner._bus_for_agent(runner._pilot_agent)
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        pending = [t for t in self.targets if t not in self.captures and now >= t]
        eligible = context.evidence_valid and context.follower_evidence_valid and not runner._lower_terminal_action_masked(bus)
        if not pending or not eligible:
            return law
        target = pending[0]
        values = audit.native_values(runner.lower_trainer, [self.current_state])
        if not values['mask'].all():
            raise AssertionError('registered seven-bin context contains an infeasible action')
        base = (int(bus.trip_id), int(bus.last_station.station_id), int(bus.direction))
        occurrence = self.uniforms.counts.get(base, 0)
        uniform = float(np.random.default_rng(np.random.SeedSequence([self.uniforms.seed, *base, occurrence])).random())
        p = law[0].numpy().astype(float)
        p /= p.sum()
        trainer = runner.lower_trainer
        capture = dict(time_s=now, trip=base[0], station=base[1], direction=base[2],
            state=self.current_state.tolist(), native_action=pilot.event.categorical_index(p, uniform), p=p.tolist(),
            q=values['q'][0].tolist(), q_policy=values['q_policy'][0].tolist(), cost=values['cost'][0].tolist(),
            alpha=float(trainer.alpha), lam=float(trainer.lambda_param), gamma=float(trainer.gamma),
            prefix_events=[[list(k), v] for k, v in self.uniforms.records.items()], upper_prefix=copy.deepcopy(self.upper_path))
        if self.reference is not None:
            pilot.joint.assert_same_state(capture, self.reference['captures'][str(target)], 'v101_exact_native_prefix')
        self.captures[target] = capture
        if self.future is not None:
            ci = CONTEXTS.index((audit.ITEMS.index((self.seed, self.reference['day'])), target))
            seed = 201000000 + ci * 100 + self.future
            repeated.fork_future(runner, policy, seed)
            self.uniforms.seed = seed + 10000000
        if self.action is None:
            return law
        forced = torch.zeros_like(law)
        forced[0, self.action] = 1.
        return forced

    def detach(self):
        self.runner._lower_action_for_agent, self.runner._upper_callback_v2 = self.saved_lower, self.saved_upper
        super().detach()

    def finish(self, runner, records, holds):
        trainer = runner.lower_trainer
        for net, old in zip((trainer.q_net, trainer.target_q_net, trainer.cost_q_net, trainer.target_cost_q_net), self.frozen_values):
            pilot.joint.assert_same_state(net.state_dict(), old, 'v101_frozen_critic')
        if self.scalars != (float(trainer.alpha), float(trainer.lambda_param), float(trainer.gamma)):
            raise AssertionError('learning scalars changed')
        if set(self.captures) != set(self.targets):
            raise AssertionError('registered observable context not reached')
        self.returns = {}
        for target, capture in self.captures.items():
            rows = [r for r in self.transitions if r['trip'] == capture['trip']]
            start = next(i for i, r in enumerate(rows) if r['station'] == capture['station'])
            rows = rows[start:]
            np.testing.assert_array_equal(rows[0]['state'], np.asarray(capture['state'], np.float32))
            expected = capture['native_action'] if self.action is None else self.action
            if rows[0]['action'] != expected:
                raise AssertionError('executed anchor differs from registered action')
            values = audit.native_values(trainer, [r['state'] for r in rows])
            selected_logp = values['logp'][np.arange(len(rows)), [r['action'] for r in rows]]
            reward, cost, _ = audit.soft_trip_returns(rows, selected_logp, capture['gamma'], capture['alpha'])
            self.returns[target] = dict(soft_reward=float(reward[0]), cost_return=float(cost[0]))
        audit.benchmark.BenchmarkAudit.finish(self, runner, records, holds)


def replay(root, ri, *, reference=None, target=None, action=None, future=None, label):
    seed, day = audit.ITEMS[ri]
    sampler = 187000000 + seed + 100 * day
    uniforms, credit = pilot.event.EventUniforms(sampler), pilot.CreditObserver()
    rule = ActionAudit(seed, TARGETS if target is None else (target,), uniforms,
                       reference=reference, action=action, future=future)
    row, events, _, _ = pilot.rollout(root, seed, day, 'stochastic', sampler, [0.] * 4, label,
        probability_rule=rule, trace_observer=rule, event_uniforms=uniforms, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return dict(seed=seed, day=day, results=row, captures=rule.captures, returns=rule.returns), rule, events


def reference_item(root, v100_root, ri, full_path=False):
    path = root / 'references' / f'r{ri}.json'
    if not path.exists():
        result, rule, _ = replay(root, ri, label=f'reference{ri}')
        old = read(v100_root / 'pairs' / f'p{ri}.json')['results']
        for metric in (*frozen.METRICS, 'action_counts'):
            pilot.joint.assert_same_state(old[metric], result['results'][metric], 'v101_cached_native_' + metric)
        if full_path:
            rule.save(root / 'smoke_records/reference.npz')
        pilot.write_json(path, result)
    print('V101_REFERENCE_DONE ' + str(ri), flush=True)


def arm(root, index):
    path = root / 'arms' / f'a{index}.json'
    if not path.exists():
        ci, future, action = ITEMS[index]
        ri, target = CONTEXTS[ci]
        ref = read(root / 'references' / f'r{ri}.json')
        result, _, events = replay(root, ri, reference=ref, target=target, action=action, future=future, label=f'arm{index}')
        event_path = root / 'events' / f'a{index}.npz'
        event_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(event_path, keys=list(events), uniform=[v['uniform'] for v in events.values()],
            time=[v['time_s'] for v in events.values()], action=[v['action_s'] for v in events.values()])
        pilot.write_json(path, dict(index=index, context=ci, future=future, action=action,
            results=result['results'], local=result['returns'][target], exact_prefix=True))
    print('V101_ARM_DONE ' + str(index), flush=True)


def event_records(root, index):
    with np.load(root / 'events' / f'a{index}.npz') as d:
        return {tuple(k): dict(uniform=float(u), time_s=float(t), action_s=float(a), order=i)
                for i, (k, u, t, a) in enumerate(zip(d['keys'], d['uniform'], d['time'], d['action']))}


def aggregate(root):
    effects, contexts = [], []
    for ci, (ri, target) in enumerate(CONTEXTS):
        ref = read(root / 'references' / f'r{ri}.json')
        c = ref['captures'][str(target)]
        p, q, qp, cq = [np.asarray(c[k]) for k in ('p', 'q', 'q_policy', 'cost')]
        b = boltzmann(qp - c['lam'] * cq, c['alpha'])
        predicted = objective_effect(p, b, q, cq, q - qp, c['alpha'], c['lam'])
        curves, future_effects = [], []
        for f in FUTURES:
            indices = [ITEMS.index((ci, f, a)) for a in range(7)]
            rows = [read(root / 'arms' / f'a{i}.json') for i in indices]
            common = event_records(root, indices[0])
            for i in indices[1:]:
                pilot.event.alignment(common, event_records(root, i))
            reward, cost = [np.asarray([r['local'][k] for r in rows]) for k in ('soft_reward', 'cost_return')]
            local = objective_effect(p, b, reward, cost, q - qp, c['alpha'], c['lam'])
            physical = {m: float((b - p) @ [r['results'][m] for r in rows]) for m in frozen.METRICS}
            values = dict(local, **physical, critic_soft_gain_prediction_error=predicted['frozen_pessimism_soft_objective']
                          - local['frozen_pessimism_soft_objective'])
            effects.append(dict(seed=ref['seed'], day=ref['day'], effects=dict(boltzmann_vs_actor=values)))
            future_effects.append(values)
            curves.append(reward - reward[0])
        means = np.mean(curves, axis=0)
        se = np.std(curves, axis=0, ddof=1) / np.sqrt(len(FUTURES))
        contexts.append(dict(seed=ref['seed'], day=ref['day'], target=target, predicted=predicted,
            effects=future_effects, q_vs_zero=(q - q[0]).tolist(), soft_return_vs_zero=means.tolist(),
            soft_return_diff_se=se.tolist(), action_error_vs_zero=(q - q[0] - means).tolist(),
            first_half_soft_return_vs_zero=np.mean(curves[:2], axis=0).tolist(),
            second_half_soft_return_vs_zero=np.mean(curves[2:], axis=0).tolist()))
    pilot.write_json(root / 'summary/status.json', dict(status='native_action_difference_audit_complete',
        effects=confirmation.describe(effects, 'effects'), contexts=contexts, labelled_arms=len(ITEMS),
        native_references=len(audit.ITEMS), futures_per_context=len(FUTURES), actor_updates=0, critic_updates=0,
        dual_updates=0, main_policy_changed=False, controller_admitted=False,
        exact_prefixes=True, common_event_uniforms_verified=True,
        scope='single-action mechanism audit on consumed days with new post-prefix innovations; not a daily-policy effect'))
    print('V101_ACTION_DIFFERENCE_AUDIT_COMPLETE', flush=True)


def smoke(root, v100_root):
    reference_item(root, v100_root, 0, full_path=True)
    ref = read(root / 'references/r0.json')
    target = TARGETS[0]
    native = ref['captures'][str(target)]['native_action']
    result, rule, _ = replay(root, 0, reference=ref, target=target, action=native, label='forced_native_no_fork')
    for m in (*frozen.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(ref['results'][m], result['results'][m], 'v101_same_action_' + m)
    rule.save(root / 'smoke_records/noop.npz')
    with np.load(root / 'smoke_records/reference.npz') as a, np.load(root / 'smoke_records/noop.npz') as b:
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    for action in (0, 6):
        arm(root, ITEMS.index((0, 0, action)))
    pilot.event.alignment(event_records(root, ITEMS.index((0, 0, 0))), event_records(root, ITEMS.index((0, 0, 6))))
    pilot.write_json(root / 'smoke/done.json', dict(status='native_cached_noop_timing_and_extreme_forks_passed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'references', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--v100-root', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_root, args.v100_root)
    elif args.command == 'smoke':
        smoke(args.out_root, args.v100_root)
    elif args.command == 'aggregate':
        aggregate(args.out_root)
    else:
        end = args.end if args.end is not None else (len(audit.ITEMS) if args.command == 'references' else len(ITEMS))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            pending = [pool.submit(reference_item, args.out_root, args.v100_root, i) if args.command == 'references'
                       else pool.submit(arm, args.out_root, i) for i in range(args.start, end)]
            for task in pending:
                task.result()
        pilot.write_json(args.out_root / f'shards/{args.command}/{args.start}-{end}.json', dict(status='complete'))
