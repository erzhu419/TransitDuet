#!/usr/bin/env python3
"""Read-only V104 review: system-cost teachers on disjoint cached futures."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v104_system_action_credit as run


model = run.model
SPLITS = (((0, 1), (2, 3)), ((2, 3), (0, 1)))
METRICS = (*model.GOALS, 'journey_min', 'holding_s_per_trip', 'trips_unlaunched', 'trips_incomplete')


def teacher(outcomes, p, training_futures, training_scale):
    advantage = model.center(np.asarray(outcomes)[list(training_futures)].mean(0), p)
    return model.proposal(advantage, p, training_scale), advantage


def axis_means(rows):
    return {axis: {str(value): {name: {k: float(np.mean([r['effects'][name][k] for r in rows if r[axis] == value]))
        for k in rows[0]['effects'][name]} for name in rows[0]['effects']}
        for value in sorted({r[axis] for r in rows})} for axis in ('seed', 'day', 'target')}


def conflicts(rows):
    contexts = {}
    for row in rows:
        contexts.setdefault((row['seed'], row['day'], row['target']), []).append(row)
    output = {}
    for name in rows[0]['effects']:
        means = [{k: float(np.mean([r['effects'][name][k] for r in group])) for k in model.GOALS}
                 for group in contexts.values()]
        output[name] = dict(contexts=len(means),
            joint_nonharm=sum(all(r[k] <= 0 for k in model.GOALS) for r in means),
            harm_counts={k: sum(r[k] > 0 for r in means) for k in model.GOALS})
    return output


def review(root):
    previous = run.source.read(root / 'summary/status.json')
    if previous['status'] != 'system_action_credit_pilot_complete':
        raise AssertionError('V104 incomplete')
    (_, probabilities, outcomes), metadata, physical = run.dataset(root, root, False)
    artifact = torch.load(root / 'models/system_credit.pt', weights_only=False)
    scale = artifact['norms']['a_scale']
    laws = {(r['seed'], r['day'], r['target'], r['model']): np.asarray(r['b']) for r in previous['laws']}
    forecast = {(r['seed'], r['day'], r['target']): r['effects'] for r in previous['effects']}
    columns = [run.source.frozen.METRICS.index(k) for k in METRICS]
    rows = []
    for ci, meta in enumerate(metadata):
        identity = (meta['seed'], meta['day'], meta['target'])
        p = probabilities[ci]
        for training, held in SPLITS:
            b, advantage = teacher(outcomes[ci], p, training, scale)
            for future in held:
                actual = physical[ci, future][:, columns]
                values = {}
                for name in ('separated_a', 'direct_q', 'system_teacher'):
                    law = b if name == 'system_teacher' else laws[(*identity, name)]
                    predicted = ((law - p) @ advantage if name == 'system_teacher' else
                        np.asarray([forecast[identity][name][k + '/predicted_change'] for k in model.GOALS]))
                    measured = (law - p) @ actual
                    result = {k: float(measured[j]) for j, k in enumerate(METRICS)}
                    for j, k in enumerate(model.GOALS):
                        result[k + '/predicted'] = float(predicted[j])
                        result[k + '/optimism'] = float(measured[j] - predicted[j])
                    result.update(native_mass_45=float(p[-1]), proposed_mass_45=float(law[-1]),
                                  anchor_holding_change_s=float((law - p) @ run.pilot.step.ACTIONS))
                    values[name] = result
                rows.append(dict(meta, future=future, effects=values))
    stats = run.source.confirmation.describe(rows, 'effects')
    # Reconstructed controls must remain the frozen V104 estimates.
    for name in ('separated_a', 'direct_q'):
        for k in METRICS:
            np.testing.assert_allclose(stats[name][k]['mean'], previous['one_action_effects'][name][k]['mean'],
                                       atol=1e-8, rtol=1e-10)
    primary = stats['system_teacher']
    joint = primary['passenger_person_s']['ci95'][1] < 0 and all(primary[k]['ci95'][1] <= 0
        for k in ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))
    return dict(status='cached_system_credit_review_complete', effects=stats, axes=axis_means(rows),
        conflicts=conflicts(rows), held_future_teacher_joint_diagnostic=joint,
        original_credit_gate=previous['contrast_gate'], original_service_gate=previous['joint_service_gate'],
        new_rollouts=0, actor_updates=0, dual_updates=0, main_policy_changed=False,
        scope='exploratory two-future conditional system-cost teacher on disjoint two-future labels, then swap; '
              'not observable policy, not a perfect-value bound, not a new confirmation set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v104_review_*.py', '-v'], check=True)
    result = review(args.root)
    run.pilot.write_json(args.root / 'review/summary.json', result)
    print('V104_REVIEW_COMPLETE', result['held_future_teacher_joint_diagnostic'], flush=True)
