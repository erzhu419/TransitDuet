#!/usr/bin/env python3
"""V104's fixed models and proposals, evaluated on independent continuations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

import review_freqduet_v104_system_credit as review


prior, model = review.run, review.model
pilot = prior.pilot
FUTURES = tuple(range(32))
COHORTS = {'first16': FUTURES[:16], 'second16': FUTURES[16:], 'pooled32': FUTURES}
ITEMS = list(itertools.product(range(len(prior.CONTEXTS)), FUTURES, range(7)))
METRICS = review.METRICS
METHODS = ('native', 'separated_a', 'direct_q', 'teacher_01', 'teacher_23', 'teacher_mix')


def future_seed(context, future):
    return 205000000 + context * 100 + future


def future_bounds(effects, draws=5000):
    """Conditional MC uncertainty: keep prefixes fixed, resample paired futures."""
    n, f = effects.shape[:2]
    rng = np.random.default_rng(105)
    counts = rng.multinomial(f, np.full(f, 1 / f), size=(draws, n))
    estimates = np.einsum('dcf,cfmk->dmk', counts, effects, optimize=True) / (n * f)
    return np.percentile(estimates, [2.5, 97.5], axis=0)


def evaluate(contexts, physical):
    """No fit, optimizer, proposal selection or new-label-dependent action here."""
    p = np.asarray([c['p'] for c in contexts])
    laws = np.asarray([[c['laws'][name] for name in METHODS] for c in contexts])
    return np.einsum('cma,cfak->cfmk', laws - p[:, None, :], physical)


def contrast_metrics(old, new):
    old_d = model.differences(old).mean(0)
    a, b = [model.differences(new[list(COHORTS[name])]) for name in ('first16', 'second16')]
    ma, mb = a.mean(0), b.mean(0)
    return dict(independent_split_cross_product=(ma * mb).mean(0),
        old_new_cross_product=(old_d * (ma + mb) / 2).mean(0),
        split_difference_mse=((ma - mb) ** 2).mean(0),
        split_difference_noise=a.var(0, ddof=1).mean(0) / len(a) + b.var(0, ddof=1).mean(0) / len(b),
        pooled_mean_noise=model.differences(new).var(0, ddof=1).mean(0) / 32)


def prepare(root, old):
    path = root / 'frozen/payload.json'
    if not path.exists():
        original = prior.source.read(old / 'summary/status.json')
        reviewed = prior.source.read(old / 'review/summary.json')
        if original['status'] != 'system_action_credit_pilot_complete' or reviewed['status'] != 'cached_system_credit_review_complete':
            raise AssertionError('V104 and its cached review must be complete')
        (states, p, y), meta, physical = prior.dataset(old, old, False)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old / 'models/system_credit.pt', root / 'frozen/model.pt')
        shutil.copytree(old / 'references', root / 'frozen/references', dirs_exist_ok=True)
        artifact = torch.load(root / 'frozen/model.pt', weights_only=False)
        predictions = model.predict(artifact, states, p)
        old_laws = {(r['seed'], r['day'], r['target'], r['model']): r['b'] for r in original['laws']}
        contexts, old_teacher = [], []
        for ci, info in enumerate(meta):
            identity = (info['seed'], info['day'], info['target'])
            laws = {name: old_laws[(*identity, name)] for name in ('separated_a', 'direct_q')}
            laws['native'] = p[ci].tolist()
            forecasts = {name: ((np.asarray(laws[name]) - p[ci]) @ predictions[name][ci]).tolist()
                         for name in ('separated_a', 'direct_q')}
            teacher_effects, teacher_forecasts = [], []
            for name, (training, held) in zip(('teacher_01', 'teacher_23'), review.SPLITS):
                b, advantage = review.teacher(y[ci], p[ci], training, artifact['norms']['a_scale'])
                laws[name] = b.tolist()
                forecasts[name] = ((b - p[ci]) @ advantage).tolist()
                teacher_effects.append((b - p[ci]) @ physical[ci, list(held)].mean(0))
                teacher_forecasts.append(forecasts[name])
            laws['teacher_mix'] = ((np.asarray(laws['teacher_01']) + laws['teacher_23']) / 2).tolist()
            forecasts['teacher_mix'] = np.mean(teacher_forecasts, axis=0).tolist()
            forecasts['native'] = [0., 0., 0.]
            old_teacher.append(np.mean(teacher_effects, axis=0))
            contexts.append(dict(info, p=p[ci].tolist(), laws=laws, forecasts=forecasts,
                baseline=prior.source.read(old / 'references' / f'r{prior.CONTEXTS[ci][0]}.json')['prefixes'][str(info['target'])]))
        for k in METRICS:
            j = prior.source.frozen.METRICS.index(k)
            np.testing.assert_allclose(np.mean(old_teacher, axis=0)[j],
                reviewed['effects']['system_teacher'][k]['mean'], atol=1e-8, rtol=1e-10)
        pilot.write_json(path, dict(contexts=contexts, predictions={k: v.tolist() for k, v in predictions.items()},
            old_outcomes=y.tolist(), futures=list(FUTURES), cohorts=COHORTS, model_fitted=False,
            teacher_rule='two original split proposals, fixed equal mixture; never select on new labels'))
    pilot.write_json(root / 'prepare/done.json', dict(status='v104_models_and_proposals_frozen',
        contexts=16, new_futures_per_context=32, labelled_arms=len(ITEMS), updates=0, main_policy_changed=False))
    print('V105_PREPARE_COMPLETE', flush=True)


def rollout(root, ci, action, seed, label):
    ri, target = prior.CONTEXTS[ci]
    ref = prior.source.read(root / 'frozen/references' / f'r{ri}.json')
    return prior.replay(root, *prior.REFERENCES[ri], label, reference=ref,
                        target=target, action=action, fork_seed=seed)


def arm(root, index):
    path = root / 'arms' / f'a{index}.json'
    if path.exists():
        return
    ci, future, action = ITEMS[index]
    result, _, events = rollout(root, ci, action, future_seed(ci, future), f'arm{index}')
    event_path = root / 'events' / f'a{index}.npz'
    event_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(event_path, keys=list(events), uniform=[v['uniform'] for v in events.values()],
        time=[v['time_s'] for v in events.values()], action=[v['action_s'] for v in events.values()])
    pilot.write_json(path, dict(index=index, context=ci, future=future, action=action,
        future_seed=future_seed(ci, future), results=result['results'], exact_prefix=True))
    print('V105_ARM_DONE', index, flush=True)


def smoke(root, old):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v105_*.py', '-v'], check=True)
    prepare(root, old)
    for action in (0, 6):
        result, _, events = rollout(root, 0, action, 204000000, f'smoke_old{action}')
        index = prior.ITEMS.index((0, 0, action))
        cached = prior.source.read(old / 'arms' / f'a{index}.json')
        for k in (*prior.source.frozen.METRICS, 'action_counts'):
            pilot.joint.assert_same_state(result['results'][k], cached['results'][k], 'v104_cached_' + k)
        pilot.joint.assert_same_state(events, prior.source.event_records(old, index), 'v104_cached_events')
    pilot.write_json(root / 'smoke/done.json', dict(status='freeze_tests_and_cached_extreme_replays_passed',
        fresh_labels_generated=0, model_updates=0, main_policy_changed=False))
    print('V105_SMOKE_COMPLETE', flush=True)


def joint_gate(stats):
    return stats['passenger_person_s']['ci95'][1] < 0 and all(stats[k]['ci95'][1] <= 0
        for k in ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))


def aggregate(root):
    payload = prior.source.read(root / 'frozen/payload.json')
    contexts = payload['contexts']
    data = []
    for ci in range(len(contexts)):
        futures = []
        for f in FUTURES:
            indices = [ITEMS.index((ci, f, a)) for a in range(7)]
            rows = [prior.source.read(root / 'arms' / f'a{i}.json') for i in indices]
            common = prior.source.event_records(root, indices[0])
            for i in indices[1:]:
                pilot.event.alignment(common, prior.source.event_records(root, i))
            for a, row in enumerate(rows):
                if (row['context'], row['future'], row['action'], row['future_seed'], row['exact_prefix']) != (ci, f, a, future_seed(ci, f), True):
                    raise AssertionError('independent future label grid mismatch')
            futures.append([[row['results'][k] for k in METRICS] for row in rows])
        data.append(futures)
    physical = np.asarray(data)
    effects = evaluate(contexts, physical)
    summaries = {}
    for cohort, ids in COHORTS.items():
        chosen = effects[:, list(ids)]
        rows = [dict(seed=c['seed'], day=c['day'], target=c['target'],
            effects={name: dict(zip(METRICS, chosen[ci, :, j].mean(0).tolist()))
                for j, name in enumerate(METHODS)}) for ci, c in enumerate(contexts)]
        crossed = prior.source.confirmation.describe(rows, 'effects')
        bounds = future_bounds(chosen)
        conditional = {name: {k: dict(mean=float(chosen[:, :, j, q].mean()), ci95=bounds[:, j, q].tolist())
            for q, k in enumerate(METRICS)} for j, name in enumerate(METHODS)}
        summaries[cohort] = dict(crossed_actor_day=crossed, conditional_future=conditional,
                                axes=review.axis_means(rows))
    goal_y = physical[..., [METRICS.index(k) for k in model.GOALS]] - np.asarray([c['baseline'] for c in contexts])[:, None, None, :]
    predictions = {k: np.asarray(v) for k, v in payload['predictions'].items()}
    p = np.asarray([c['p'] for c in contexts])
    errors = model.errors(predictions, goal_y, p)
    error_rows, signal_rows, curves = [], [], []
    for ci, c in enumerate(contexts):
        measures = {name: {goal + '/' + k: float(values[ci, j]) for k, values in fields.items()
            for j, goal in enumerate(model.GOALS)} for name, fields in errors.items()}
        for control in ('direct_q', 'zero'):
            measures['separated_a_vs_' + control] = {k: measures['separated_a'][k] - measures[control][k]
                                                    for k in measures['separated_a']}
        error_rows.append(dict(seed=c['seed'], day=c['day'], effects=measures))
        signal = contrast_metrics(np.asarray(payload['old_outcomes'][ci]), goal_y[ci])
        signal_rows.append(dict(seed=c['seed'], day=c['day'], effects=dict(replication={
            goal + '/' + k: float(values[j]) for k, values in signal.items() for j, goal in enumerate(model.GOALS)})))
        curves.append(dict(seed=c['seed'], day=c['day'], target=c['target'],
            contrasts={name: dict(mean=model.differences(goal_y[ci, list(ids)]).mean(0).tolist(),
                mcse=(model.differences(goal_y[ci, list(ids)]).std(0, ddof=1) / np.sqrt(len(ids))).tolist())
                for name, ids in COHORTS.items()},
            old_four=model.differences(np.asarray(payload['old_outcomes'][ci])).mean(0).tolist()))
    forecast = np.asarray([[c['forecasts'][name] for name in METHODS] for c in contexts])
    optimism = effects[..., :3] - forecast[:, None, :, :]
    calibration_rows = [dict(seed=c['seed'], day=c['day'], effects={name:
        dict(zip(model.GOALS, optimism[ci, :, j].mean(0).tolist())) for j, name in enumerate(METHODS)})
        for ci, c in enumerate(contexts)]
    pooled = summaries['pooled32']
    gates = {name: dict(crossed_joint=joint_gate(pooled['crossed_actor_day'][name]),
                       conditional_joint=joint_gate(pooled['conditional_future'][name]),
                       both_cohort_point_nonharm=all(
                           summaries[cohort]['crossed_actor_day'][name]['passenger_person_s']['mean'] < 0 and
                           all(summaries[cohort]['crossed_actor_day'][name][k]['mean'] <= 0
                               for k in ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))
                           for cohort in ('first16', 'second16')))
             for name in ('separated_a', 'teacher_mix')}
    pilot.write_json(root / 'summary/status.json', dict(status='frozen_future_replication_complete',
        effects=summaries, gates=gates, contrast_errors=prior.source.confirmation.describe(error_rows, 'effects'),
        contrast_replication=prior.source.confirmation.describe(signal_rows, 'effects'),
        optimism=prior.source.confirmation.describe(calibration_rows, 'effects'),
        action_pairs=prior.pilot.step.ACTIONS[model.PAIRS].tolist(), goals=list(model.GOALS), contrast_curves=curves,
        futures_per_context=32, labelled_arms=len(ITEMS), contexts=16, demand_days=4, archived_actors=2,
        model_updates=0, actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='same 16 V104 prefixes, new post-prefix innovations; fixed proposals only, no new selection or training'))
    print('V105_COMPLETE', gates, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--v104-root', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args.out_root, args.v104_root)
    elif args.command == 'aggregate':
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
            list(executor.map(arm, itertools.repeat(args.out_root), range(args.start, args.end)))
        pilot.write_json(args.out_root / 'shards' / f'{args.start}-{args.end}.json',
                         dict(status='complete', start=args.start, end=args.end, main_policy_changed=False))


if __name__ == '__main__':
    main()
