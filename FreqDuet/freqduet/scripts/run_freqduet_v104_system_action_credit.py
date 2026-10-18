#!/usr/bin/env python3
"""Frozen lower: three system-cost heads, independent level/action credit pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_system_action_credit as model
import run_freqduet_v101_native_action_differences as source
from freqduet_system_learner import ArrivalMoments


pilot = source.pilot
DAYS = tuple(range(2041001, 2041005))
SEEDS = (12037, 933083)
REFERENCES = list(itertools.product(SEEDS, DAYS))
CONTEXTS = list(itertools.product(range(len(REFERENCES)), source.TARGETS))
ITEMS = list(itertools.product(range(len(CONTEXTS)), source.FUTURES, range(7)))
MODES = ('separated_a', 'direct_q', 'zero')


class Capture(source.ActionAudit):
    def __init__(self, *args, credit, fork_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit, self.fork_seed = credit, fork_seed
        self.observables, self.prefixes = {}, {}

    def __call__(self, policy, states, features, theta):
        before = set(self.captures)
        law = super().__call__(policy, states, features, theta)
        for target in set(self.captures) - before:
            # Only the existing causal encoder and native law enter the model.
            self.observables[target] = states[0].detach().cpu().numpy().tolist()
            moments = ArrivalMoments()
            moments.observe(self.runner.env.headway_events.headways_s)
            self.prefixes[target] = [float(self.credit.stream.total[:2].sum()), moments.cv_squared, 0.]
            if self.reference is not None and 'observables' in self.reference:
                for key in ('observables', 'prefixes'):
                    pilot.joint.assert_same_state(getattr(self, key)[target], self.reference[key][str(target)], key)
            if self.fork_seed is not None:
                source.repeated.fork_future(self.runner, policy, self.fork_seed)
                self.uniforms.seed = self.fork_seed + 10000000
        return law


def replay(root, seed, day, label, *, reference=None, target=None, action=None, fork_seed=None):
    sampler = 187000000 + seed + 100 * day
    uniforms, credit = pilot.event.EventUniforms(sampler), pilot.CreditObserver()
    rule = Capture(seed, source.TARGETS if target is None else (target,), uniforms,
                   credit=credit, reference=reference, action=action, fork_seed=fork_seed)
    row, events, _, _ = pilot.rollout(root, seed, day, 'stochastic', sampler, [0.] * 4, label,
        probability_rule=rule, trace_observer=rule, event_uniforms=uniforms, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
               backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return dict(seed=seed, day=day, results=row, captures=rule.captures,
                observables=rule.observables, prefixes=rule.prefixes), rule, events


def reference(root, old, index, training):
    directory = 'training_references' if training else 'references'
    path = root / directory / f'r{index}.json'
    if path.exists():
        return
    seed, day = source.audit.ITEMS[index] if training else REFERENCES[index]
    result, rule, _ = replay(root, seed, day, directory + str(index))
    if training:
        previous = source.read(old / 'references' / f'r{index}.json')
        for key in ('captures',):
            pilot.joint.assert_same_state(result[key], previous[key], 'old_native_' + key)
        for metric in (*source.frozen.METRICS, 'action_counts'):
            pilot.joint.assert_same_state(result['results'][metric], previous['results'][metric], metric)
    pilot.write_json(path, result)
    if training and index == 0:
        rule.save(root / 'smoke_records/reference.npz')
    print('V104_REFERENCE_DONE', directory, index, flush=True)


def dataset(root, arm_root, training):
    contexts = source.CONTEXTS if training else CONTEXTS
    grid = source.ITEMS if training else ITEMS
    states, probabilities, outcomes, metadata = [], [], [], []
    physical = []
    for ci, (ri, target) in enumerate(contexts):
        ref = source.read(root / ('training_references' if training else 'references') / f'r{ri}.json')
        p = ref['captures'][str(target)]['p']
        states.append(ref['observables'][str(target)] + p)
        probabilities.append(p)
        rows = [[source.read(arm_root / 'arms' / f'a{grid.index((ci, f, a))}.json')
                 for a in range(7)] for f in source.FUTURES]
        for f in source.FUTURES:
            for a in range(7):
                row = rows[f][a]
                if (row['context'], row['future'], row['action'], row['exact_prefix']) != (ci, f, a, True):
                    raise AssertionError('counterfactual label grid mismatch')
        full = np.asarray([[[row['results'][k] for k in source.frozen.METRICS] for row in future] for future in rows])
        physical.append(full)
        outcomes.append(np.asarray([[[row['results'][k] for k in model.GOALS] for row in future] for future in rows])
                        - np.asarray(ref['prefixes'][str(target)]))
        metadata.append(dict(seed=ref['seed'], day=ref['day'], target=target))
    return tuple(np.asarray(a, float) for a in (states, probabilities, outcomes)), metadata, np.asarray(physical)


def train(root, old):
    arrays, _, _ = dataset(root, old, True)
    artifact = model.fit(*arrays)
    artifact.update(training_days=sorted({day for _, day in source.audit.ITEMS}),
                    held_days=list(DAYS), goals=list(model.GOALS), source='V101 frozen labels')
    path = root / 'fit/model.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    restored = torch.load(path, weights_only=False)
    for key, value in model.predict(artifact, *arrays[:2]).items():
        np.testing.assert_array_equal(value, model.predict(restored, *arrays[:2])[key])
    pilot.write_json(root / 'fit/done.json', dict(status='three_system_heads_frozen',
        prefixes=artifact['prefixes'], updates_per_branch=artifact['steps'],
        goals=list(model.GOALS), final_losses=artifact['final_normalized_losses'],
        actor_updates=0, dual_updates=0, main_policy_changed=False))


def arm(root, index):
    path = root / 'arms' / f'a{index}.json'
    if path.exists():
        return
    ci, future, action = ITEMS[index]
    ri, target = CONTEXTS[ci]
    ref = source.read(root / 'references' / f'r{ri}.json')
    result, _, events = replay(root, *REFERENCES[ri], f'arm{index}', reference=ref,
        target=target, action=action, fork_seed=204000000 + ci * 100 + future)
    path_events = root / 'events' / f'a{index}.npz'
    path_events.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_events, keys=list(events), uniform=[v['uniform'] for v in events.values()],
        time=[v['time_s'] for v in events.values()], action=[v['action_s'] for v in events.values()])
    pilot.write_json(path, dict(index=index, context=ci, future=future, action=action,
                               results=result['results'], exact_prefix=True))
    print('V104_ARM_DONE', index, flush=True)


def summarize(root):
    arrays, metadata, physical = dataset(root, root, False)
    x, p, y = arrays
    artifact = torch.load(root / 'fit/model.pt', weights_only=False)
    pred = model.predict(artifact, x, p)
    errors = model.errors(pred, y, p)
    rows, effects, laws = [], [], []
    for ci, meta in enumerate(metadata):
        for f in source.FUTURES:
            indices = [ITEMS.index((ci, f, a)) for a in range(7)]
            common = source.event_records(root, indices[0])
            for index in indices[1:]:
                pilot.event.alignment(common, source.event_records(root, index))
        measures = {name: {goal + '/' + key: float(value[ci, j])
            for key, value in fields.items() for j, goal in enumerate(model.GOALS)} for name, fields in errors.items()}
        for name in ('direct_q', 'zero'):
            measures['separated_a_vs_' + name] = {k: measures['separated_a'][k] - measures[name][k]
                                                  for k in measures['separated_a']}
        rows.append(dict(meta, effects=measures))
        physical_mean = physical[ci].mean(0)
        outcomes = {}
        for name in MODES:
            b = model.proposal(pred[name][ci], p[ci], artifact['norms']['a_scale'])
            measured = (b - p[ci]) @ physical_mean
            forecast = (b - p[ci]) @ pred[name][ci]
            outcomes[name] = {k: float(measured[j]) for j, k in enumerate(source.frozen.METRICS)}
            outcomes[name].update({goal + '/predicted_change': float(forecast[j]) for j, goal in enumerate(model.GOALS)})
            laws.append(dict(meta, model=name, p=p[ci].tolist(), b=b.tolist()))
        outcomes['separated_a_vs_direct_q'] = {k: outcomes['separated_a'][k] - outcomes['direct_q'][k]
                                              for k in outcomes['separated_a']}
        effects.append(dict(meta, effects=outcomes))
    stats = source.confirmation.describe(rows, 'effects')
    decision = source.confirmation.describe(effects, 'effects')
    contrast_ok = all(stats['separated_a_vs_' + control][goal + '/difference_mse']['ci95'][1] < 0
                      for control in ('direct_q', 'zero') for goal in model.GOALS)
    primary = decision['separated_a']
    joint = primary['passenger_person_s']['ci95'][1] < 0 and all(primary[k]['ci95'][1] <= 0
        for k in ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))
    pilot.write_json(root / 'summary/status.json', dict(status='system_action_credit_pilot_complete',
        contrast_errors=stats, one_action_effects=decision, contrast_gate=contrast_ok, joint_service_gate=joint,
        rows=rows, effects=effects, laws=laws, days=list(DAYS), seeds=list(SEEDS),
        actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='16 fresh-day prefixes, four paired futures; isolated one-action interventions, not a deployable full-day policy'))
    print('V104_COMPLETE', contrast_ok, joint, flush=True)


def smoke(root, old):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v104_*.py', '-v'], check=True)
    reference(root, old, 0, True)
    ref = source.read(root / 'training_references/r0.json')
    seed, day = source.audit.ITEMS[0]
    target = source.TARGETS[0]
    noop, trace, _ = replay(root, seed, day, 'smoke_noop', reference=ref, target=target,
                           action=ref['captures'][str(target)]['native_action'])
    trace.save(root / 'smoke_records/noop.npz')
    for metric in (*source.frozen.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(noop['results'][metric], ref['results'][metric], 'noop_' + metric)
    with np.load(root / 'smoke_records/noop.npz') as actual, np.load(old / 'smoke_records/reference.npz') as previous:
        for key in previous.files:
            np.testing.assert_array_equal(actual[key], previous[key])
    for action in (0, 6):
        result, _, events = replay(root, seed, day, f'smoke_extreme{action}', reference=ref,
            target=target, action=action, fork_seed=201000000)
        index = source.ITEMS.index((0, 0, action))
        cached = source.read(old / 'arms' / f'a{index}.json')
        for metric in (*source.frozen.METRICS, 'action_counts'):
            pilot.joint.assert_same_state(result['results'][metric], cached['results'][metric], 'extreme_' + metric)
        pilot.joint.assert_same_state(events, source.event_records(old, index), 'extreme_timing')
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_native_noop_and_extreme_labels_exact',
        new_demand_days_consumed=0, main_policy_changed=False))
    print('V104_SMOKE_COMPLETE', flush=True)


def item(command, root, old, index):
    if command == 'arms':
        arm(root, index)
    else:
        reference(root, old, index, command == 'training-references')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'training-references', 'fit', 'references', 'arms', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--v101-root', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command in ('smoke', 'fit', 'aggregate'):
        {'smoke': smoke, 'fit': train, 'aggregate': lambda r, o: summarize(r)}[args.command](args.out_root, args.v101_root)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
        list(executor.map(item, itertools.repeat(args.command), itertools.repeat(args.out_root),
                          itertools.repeat(args.v101_root), range(args.start, args.end)))
    pilot.write_json(args.out_root / (args.command + '_shards') / f'{args.start}-{args.end}.json',
                     dict(status='complete', start=args.start, end=args.end, main_policy_changed=False))


if __name__ == '__main__':
    main()
