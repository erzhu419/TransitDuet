#!/usr/bin/env python3
"""Crossed uncertainty of V158 variance budgets, using saved moments only."""

import argparse
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v158_prior_dependence as frozen


audit = frozen.audit


def weights(size, indices):
    return np.eye(size)[indices].mean(axis=1)


def decomposition(mu, second, actor_weights, day_weights):
    grand = np.einsum('bs,bd,sdi->bi', actor_weights, day_weights, mu, optimize=True)
    raw = np.einsum('bs,bd,sdij->bij', actor_weights, day_weights, second, optimize=True)
    products = np.einsum('sdi,sdj->sdij', mu, mu)
    within = raw-np.einsum('bs,bd,sdij->bij', actor_weights, day_weights, products, optimize=True)
    actor_means = np.einsum('bd,sdi->bsi', day_weights, mu, optimize=True)-grand[:, None, :]
    day_means = np.einsum('bs,sdi->bdi', actor_weights, mu, optimize=True)-grand[:, None, :]
    actor = np.einsum('bs,bsi,bsj->bij', actor_weights, actor_means, actor_means, optimize=True)
    day = np.einsum('bd,bdi,bdj->bij', day_weights, day_means, day_means, optimize=True)
    total = raw-np.einsum('bi,bj->bij', grand, grand)
    return dict(total=total, within=within, actor=actor, day=day, interaction=total-within-actor-day)


def estimate(point, samples):
    finite = np.isfinite(samples)
    return dict(mean=float(point) if np.isfinite(point) else None,
        ci95=np.percentile(samples, [2.5, 97.5]).tolist() if finite.all() else None,
        defined_draws=int(finite.sum()))


def budgets(matrices):
    result = {name+'_variance_s2': m.sum(axis=(1, 2)) for name, m in matrices.items()}
    full = result['total_variance_s2']
    diagonal = np.trace(matrices['total'], axis1=1, axis2=2)
    result['diagonal_variance_s2'] = diagonal
    for name in ('within', 'actor', 'day', 'interaction'):
        result[name+'_fraction'] = np.divide(result[name+'_variance_s2'], full,
            out=np.full_like(full, np.nan), where=full > 0.)
    result['ignore_covariance_excess_fraction'] = np.divide(diagonal-full, full,
        out=np.full_like(full, np.nan), where=full > 0.)
    return result


def grid(cells, mode, group, actors, days):
    entries = [c['modes'][mode]['groups'].get(group) for c in cells]
    if any(e is None for e in entries):
        return None
    fields = list(entries[0]['metrics'])
    values = np.zeros((len(actors), len(days), len(fields)))
    for cell, entry in zip(cells, entries):
        values[actors.index(cell['seed']), days.index(cell['day'])] = [entry['metrics'][f] for f in fields]
    return fields, values


def run(args):
    parent = audit.read(args.v158_root/'summary/status.json')
    if (parent['status'], parent['execution_source']) != ('v158_prior_dependence_complete', 'e5a6705971'):
        raise AssertionError('requires complete original V158')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v158_variance_summary.py', '-v'], check=True)
    cells = [audit.read(p) for p in sorted((args.v158_root/'cells').glob('*.json'))]
    actors, days = sorted({c['seed'] for c in cells}), sorted({c['day'] for c in cells})
    if len(cells) != parent['pairs'] or len(cells) != len(actors)*len(days):
        raise AssertionError('incomplete original crossed cells')
    rng = np.random.RandomState(20260803)
    aw = weights(len(actors), rng.randint(0, len(actors), (5000, len(actors))))
    dw = weights(len(days), rng.randint(0, len(days), (5000, len(days))))
    ap, dp = np.full((1, len(actors)), 1./len(actors)), np.full((1, len(days)), 1./len(days))
    results, priors = {}, {}
    for mode in ('old', 'handoff'):
        results[mode], priors[mode] = {}, {}
        for group, old in parent['dependence'][mode].items():
            values = grid(cells, mode, group, actors, days)
            if values is None:
                results[mode][group] = dict(decision='inconclusive_crossed_support')
                continue
            channels = old['channels']
            fields, table = values
            means = table[:, :, [fields.index('e_'+c) for c in channels]]
            second = np.zeros((*means.shape, len(channels)))
            for i, a in enumerate(channels):
                for j in range(i, len(channels)):
                    second[:, :, i, j] = second[:, :, j, i] = table[:, :, fields.index('m_'+a+'__'+channels[j])]
            point, samples = decomposition(means, second, ap, dp), decomposition(means, second, aw, dw)
            for name, field in (('total', 'covariance_s2'), ('within', 'within_cell_covariance_s2'),
                    ('actor', 'actor_mean_covariance_s2'), ('day', 'day_mean_covariance_s2'),
                    ('interaction', 'actor_day_interaction_covariance_s2')):
                np.testing.assert_allclose(point[name][0], old[field], rtol=0., atol=1e-7)
            p, draws = budgets(point), budgets(samples)
            pairs = {a+'__'+b: {name: estimate(point[name][0, i, j], samples[name][:, i, j])
                for name in point} for (i, a), (j, b) in combinations(enumerate(channels), 2)}
            results[mode][group] = dict(channels=channels, queries=parent['results'][mode][group]['queries'],
                budgets={f: estimate(p[f][0], draws[f]) for f in p}, pair_covariance_s2=pairs)
        for scope in frozen.SCOPES:
            values = grid(cells, mode, 'prior/'+scope, actors, days)
            if values is None:
                priors[mode][scope] = dict(decision='inconclusive_crossed_support')
                continue
            fields, table = values
            def effects(actor, day):
                m = np.einsum('bs,bd,sdi->bi', actor, day, table, optimize=True)
                get = lambda f: m[:, fields.index(f)]
                between, within = get('between_day_variance_s2'), get('within_day_variance_s2')
                return dict(native_minus_frozen_leg_mae_s=get('native_leg_mae_s')-get('frozen_leg_mae_s'),
                    prior_minus_frozen_leg_mae_s=get('prior_leg_mae_s')-get('frozen_leg_mae_s'),
                    frozen_leg_mse_minus_mixture_variance_s2=get('frozen_leg_mse_s2')-get('frozen_mixture_variance_s2'),
                    between_day_fraction_of_mean_prior_variance=between/(between+within))
            p, draws = effects(ap, dp), effects(aw, dw)
            priors[mode][scope] = {f: estimate(p[f][0], draws[f]) for f in p}
    audit.write(args.out_root/'summary/status.json', dict(status='v158_saved_variance_readout_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cells),
        results=results, prior_effects=priors, original_v158_matrices_reproduced=True,
        new_forecasts=0, new_rollouts=0, calibration_refit=False, main_policy_changed=False,
        v151_model_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1],
        limitations='Saved V158 moments only. Variance fractions use crossed actor/day bootstrap '
            'conditional on eight fixed prior days. Within-cell includes stop/state/horizon and '
            'local day-by-context variation; it is not pure conditional noise. Prior between-day '
            'fractions concern leg duration, not the full-ETA day-mean component. Ignoring covariance '
            'excess concerns variance only, not interval coverage. No interval is fitted or tuned.'))
    print('V158_SAVED_VARIANCE_READOUT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v158-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    run(p.parse_args())
