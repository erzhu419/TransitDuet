#!/usr/bin/env python3
"""Complete V107's fixed-upper corners without inventing actor replications."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v107_rule_confirmation as old


prior, pilot = old.prior, old.pilot
ITEMS = tuple(itertools.product(old.DAYS, ('zero', 'bilateral025')))
CONTRASTS = dict(upper_with_rule=('learned_rule', 'fixed_rule'),
    upper_without_holding=('learned_zero', 'fixed_zero'),
    holding_at_fixed=('fixed_rule', 'fixed_zero'),
    holding_at_learned=('learned_rule', 'learned_zero'),
    v107_vs_exact_fixed=('learned_rule', 'fixed_zero'))


def effects(levels):
    result = {name: {m: levels[a][m]-levels[b][m] for m in old.METRICS}
              for name, (a, b) in CONTRASTS.items()}
    result['interaction'] = {m: result['holding_at_learned'][m]-result['holding_at_fixed'][m]
                             for m in old.METRICS}
    return result


class RuleAudit(prior.TradeoffAudit):
    def __init__(self, seed, method, fixed):
        super().__init__(seed, method)
        self.arm = ('fixed_' if fixed else 'learned_') + method
        self.public_delays = []

    def attach(self, runner):
        super().attach(runner)
        native = runner.env.launch_bus

        def launch(trip, actual_launch=None):
            bus = native(trip, actual_launch=actual_launch)
            if bus is not None:
                public = float(getattr(trip, '_freqduet_base_launch',
                    getattr(trip, '_original_launch', trip.launch_time)))
                self.public_delays.append(max(0., float(runner.env.current_time)-public))
            return bus
        runner.env.launch_bus = launch

    def __call__(self, policy, states, features, theta):
        self.count += len(states)
        if self.method == 'zero':
            return prior.factorial.zero_law(policy, states)
        assert self.method == 'bilateral025' and len(states) == 1
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        self.coverage['decisions'] += 1
        self.coverage['missing_forward'] += not context.evidence_valid
        self.coverage['bilateral'] += context.evidence_valid and context.follower_evidence_valid
        self.coverage['forward_fallback'] += context.evidence_valid and not context.follower_evidence_valid
        selected = prior.rule_action(context, 'bilateral', .25, policy.feasible_action_mask(states)[0].numpy())
        law = torch.zeros((1, len(pilot.step.ACTIONS)), dtype=states.dtype)
        law[0, selected] = 1.
        return law


def replay(root, seed, day, method, fixed=True):
    rule, credit = RuleAudit(seed, method, fixed), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v217_{seed}_{day}_{method}_{fixed}', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return dict(seed=seed, day=day, method=method, fixed_upper=fixed, results=row,
        contract=rule.contract, coverage=rule.coverage,
        public_launch_delay_total_s=float(sum(rule.public_delays)), launched=len(rule.public_delays))


def cached(v107, seed, day, method, rep=0):
    index = old.ITEMS.index((seed, day, method, rep))
    row = json.loads((v107/'pairs'/f'p{index}.json').read_text())
    assert (row['seed'], row['day'], row['method'], row['rep']) == (seed, day, method, rep)
    return row


def compare(actual, expected, label):
    for m in (*old.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(actual['results'][m], expected['results'][m], label+'_'+m)
    pilot.joint.assert_same_state(actual['contract'], expected['contract'], label+'_contract')
    pilot.joint.assert_same_state(actual['coverage'], expected['coverage'], label+'_coverage')


def run_item(root, index):
    day, method = ITEMS[index]
    path = root/'fixed'/f'd{day}_{method}.json'
    if not path.exists():
        pilot.write_json(path, replay(root, old.SEEDS[0], day, method))
    print('V217_FIXED_DONE', day, method, flush=True)


def smoke(root, v107, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v217_*.py', '-v'], check=True)
    seed, day = old.SEEDS[0], old.DAYS[0]
    cases = [(seed, day, m, fixed) for fixed in (False, True) for m in ('zero', 'bilateral025')]
    cases.append((old.SEEDS[1], day, 'bilateral025', True))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        tasks = [pool.submit(replay, root, *case) for case in cases]
        ext = pool.submit(prior.factorial.prior.fixed_replay, root, day, prior.factorial.prior.FixedExposure())
        actual = [f.result() for f in tasks]
        external = ext.result()
    for i, method in enumerate(('zero', 'bilateral025')):
        compare(actual[i], cached(v107, seed, day, method), 'V107_exact_learned_'+method)
        pilot.write_json(root/'fixed'/f'd{day}_{method}.json', actual[i+2])
    zero, fixed = actual[2:4]
    compare(actual[4], fixed, 'fixed_actor_invariance')
    for m in old.METRICS:
        np.testing.assert_allclose(zero['results'][m], external['results'][m], atol=1e-6, rtol=0, err_msg=m)
    pilot.joint.assert_same_state(zero['contract'], external['contract'], 'external_exact_fixed_contract')
    assert zero['results']['holding_s_per_trip'] == 0
    pilot.write_json(root/'smoke/done.json', dict(status='cached_V107_actor_invariance_and_external_fixed_passed',
        smoke_paths=6, formal_fixed_paths_reused=2, new_training=0, main_policy_changed=False))


def aggregate(root, v107):
    paired, absolute = [], []
    for seed, day in itertools.product(old.SEEDS, old.DAYS):
        fixed = {m: json.loads((root/'fixed'/f'd{day}_{m}.json').read_text()) for m in ('zero', 'bilateral025')}
        learn = {m: cached(v107, seed, day, m) for m in ('zero', 'bilateral025')}
        for row in (*fixed.values(), *learn.values()):
            pilot.joint.assert_same_state(row['contract'], learn['zero']['contract'], 'V217_matched_physics')
        levels = dict(learned_zero=learn['zero']['results'], learned_rule=learn['bilateral025']['results'],
                      fixed_zero=fixed['zero']['results'], fixed_rule=fixed['bilateral025']['results'])
        for method in ('native', 'preserve'):
            reps = [cached(v107, seed, day, method, r)['results'] for r in range(3)]
            levels[method] = {m: float(np.mean([r[m] for r in reps])) for m in old.METRICS}
        absolute.append(dict(seed=seed, day=day, values={k: {m: r[m] for m in old.METRICS} for k, r in levels.items()}))
        paired.append(dict(seed=seed, day=day, effects=effects(levels)))
    stats = old.confirmation.describe(paired, 'effects')
    pilot.write_json(root/'summary/status.json', dict(status='V107_missing_corners_complete',
        absolute=old.confirmation.describe(absolute, 'values'), effects=stats,
        axes=prior.factorial.prior.prior.axis_summary(paired, 'effects', old.KEY_METRICS),
        joint_vs_fixed_rule=old.joint_gate(stats['upper_with_rule']),
        demand_days=list(old.DAYS), actors=list(old.SEEDS), fixed_paths=32, smoke_extra_paths=4,
        reused_learned_paths=1280, bootstrap='existing 5000 crossed actor/day draws; fixed path shared per day',
        fixed_policy='exact 360s schedule; original fixed pool and native seven holding bins',
        public_delay=dict(fixed={str(d): {m: json.loads((root/'fixed'/f'd{d}_{m}.json').read_text())['public_launch_delay_total_s']
            for m in ('zero', 'bilateral025')} for d in old.DAYS},
            learned=None, reason='original V107 did not save original-public launch clocks; not reconstructed as zero'),
        consumed_days=True, fresh_confirmation=False, new_training=0, main_policy_changed=False,
        interpretation='attribution benchmark, not frequency proof or automatic policy promotion')))
    print('V217_FACTORIAL_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate', 'pipeline'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v107-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.command in ('smoke', 'pipeline'):
        smoke(a.out_root, a.v107_root, a.workers)
    if a.command in ('run', 'pipeline'):
        assert (a.out_root/'smoke/done.json').exists(), 'formal paths require successful equivalence smoke'
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), range(len(ITEMS))))
    if a.command in ('aggregate', 'pipeline'):
        aggregate(a.out_root, a.v107_root)


if __name__ == '__main__':
    main()
