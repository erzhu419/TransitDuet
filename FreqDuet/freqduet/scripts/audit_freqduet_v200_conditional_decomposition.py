#!/usr/bin/env python3
"""Read sealed V199 moments with fixed support, parent centering and V191 weights."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_freqduet_v199_conditional_moments as previous


plan, saved, common = previous.plan, previous.saved, previous.common
read, write, require = previous.read, previous.write, previous.require
DAYS, SCOPES = previous.DAYS, previous.SCOPES
J, Y = (saved.VARS.index(n) for n in ('signed_remaining_s', 'first_running_under_s'))
JY = 1+saved.N+next(i for i, pair in enumerate(zip(*saved.TRI)) if pair == (min(J, Y), max(J, Y)))
PRIMARY = ('clearance', 'hour13_moving_launch')
ENDPOINTS = ('supported_mass', 'unsupported_mass', 'parent_mass', 'supported_cross', 'unsupported_cross',
    'parent_cross', 'supported_conditional_covariance', 'supported_centered_mean_product')
TERMS = ('mix', 'conditional', 'conditional_covariance_change', 'conditional_mean_product_change',
    'mix_covariance', 'mix_mean_product')
FAMILY_ALPHA = .05/8


def centered(m, parent):
    """Center before support exclusion, using the complete base-key/day mean."""
    mass = m[..., 0]
    mu = np.divide(parent[..., [1+J, 1+Y]], parent[..., :1],
        out=np.zeros((*parent.shape[:2], 2)), where=parent[..., :1] > 0.)
    x = m[..., 1+J]-mass*mu[..., 0, None]
    y = m[..., 1+Y]-mass*mu[..., 1, None]
    cross = m[..., JY]-mu[..., 0, None]*m[..., 1+Y]-mu[..., 1, None]*m[..., 1+J]+mass*np.prod(mu, axis=-1)[..., None]
    parent_cross = parent[..., JY]-np.divide(parent[..., 1+J]*parent[..., 1+Y], parent[..., 0],
        out=np.zeros(parent.shape[:2]), where=parent[..., 0] > 0.)
    require(np.allclose(cross.sum(axis=2), parent_cross, rtol=0., atol=1e-8), 'full parent/day centering identity differs')
    require(np.allclose(x.sum(axis=2), 0., rtol=0., atol=1e-10)
        and np.allclose(y.sum(axis=2), 0., rtol=0., atol=1e-10), 'parent/day centered first moments do not vanish')
    return np.stack((mass, x, y, cross), axis=-1)


def evaluate(first, last, keep):
    endpoints, cells = [], []
    for half in (first, last):
        values = half.reshape(*half.shape[:-3], -1, 4)
        selected = values[..., keep.ravel(), :]
        mass, x, y, q = (selected[..., i] for i in range(4))
        # An empty resampled supported cell is unknown; do not remake the mask.
        t = np.divide(q, mass, out=np.full_like(q, np.nan), where=mass > 0.)
        mean_product = np.divide(x*y, mass**2, out=np.full_like(q, np.nan), where=mass > 0.)
        covariance = t-mean_product
        excluded = values[..., ~keep.ravel(), :]
        endpoints.append(np.stack((mass.sum(axis=-1), excluded[..., 0].sum(axis=-1), values[..., 0].sum(axis=-1),
            q.sum(axis=-1), excluded[..., 3].sum(axis=-1), values[..., 3].sum(axis=-1),
            (mass*covariance).sum(axis=-1), (mass*mean_product).sum(axis=-1)), axis=-1))
        cells.append((mass, t, covariance, mean_product))
    a, b = cells
    dm, mm = b[0]-a[0], (b[0]+a[0])/2.
    terms = np.stack(((dm*(a[1]+b[1])/2.).sum(axis=-1), (mm*(b[1]-a[1])).sum(axis=-1),
        (mm*(b[2]-a[2])).sum(axis=-1), (mm*(b[3]-a[3])).sum(axis=-1),
        (dm*(a[2]+b[2])/2.).sum(axis=-1), (dm*(a[3]+b[3])/2.).sum(axis=-1)), axis=-1)
    endpoints = np.stack(endpoints, axis=-2)
    identities(endpoints, terms)
    return endpoints, terms


def identities(endpoints, terms):
    require(np.allclose(endpoints[..., 0]+endpoints[..., 1], endpoints[..., 2], rtol=1e-10, atol=1e-10)
        and np.allclose(endpoints[..., 3]+endpoints[..., 4], endpoints[..., 5], rtol=1e-9, atol=1e-8),
        'supported plus unsupported no longer equals parent')
    valid = np.isfinite(terms).all(axis=-1)
    delta = endpoints[..., 1, 3]-endpoints[..., 0, 3]
    require(np.allclose((terms[..., 0]+terms[..., 1])[valid], delta[valid], rtol=1e-9, atol=1e-8),
        'mix plus conditional does not equal supported delta')
    require(np.allclose(terms[..., 1][valid], (terms[..., 2]+terms[..., 3])[valid], rtol=1e-9, atol=1e-8)
        and np.allclose(terms[..., 0][valid], (terms[..., 4]+terms[..., 5])[valid], rtol=1e-9, atol=1e-8),
        'conditional covariance/mean-product partition differs')


def average_days(values, weights):
    return (weights@values.reshape(32, -1)).reshape(len(weights), *values.shape[1:])


def load_actor(args, actor, mode, scope, support):
    payload = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
    require(payload['source'] == '51a9409ec2', 'V194 source differs')
    parent = plan.base.actor_matrix(payload['rows'], scope, support, actor)
    keys = sorted(tuple(k['key']) for k in support if k['actor'] == actor)
    metadata = read(args.v199_root/'metadata'/f'{actor}_{mode}.json')
    require(metadata['source'] == '71af51ae4d' and metadata['paths'] == 64, 'V199 metadata differs')
    with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
        require(z['days'].tolist() == list(DAYS) and z['keys'].tolist() == [list(k) for k in keys]
            and z['bins'].tolist() == [list(b) for b in previous.BINS], 'sealed day/key/bin order differs')
        m, counts, keep, factors = z['moments'], z['distinct_days'], z['supported_both'], z['parent_balance_factors']
    require(m.shape == (64, len(keys), 9, saved.SIZE) and np.isfinite(m).all(), 'V199 moment schema differs')
    counted, eligible = plan.supported(m[..., 0])
    require(np.array_equal(counted, counts) and np.array_equal(eligible, keep), 'sealed V199 support mask differs')
    # The parent comparison verifies this readout's numerator/centering inputs;
    # V199 already admitted the complete 39-coordinate second-moment block.
    require(np.allclose(m[..., [0, 1+J, 1+Y, JY]].sum(axis=2), parent[..., [0, 1+J, 1+Y, JY]],
        rtol=0., atol=1e-8), 'V194 mass/J/Y/JY input differs')
    values = centered(m, parent)
    for h, expected in enumerate(common.balance(parent[:32], parent[32:])):
        take = slice(h*32, (h+1)*32)
        require(np.allclose(parent[take, :, :1]*factors[h, None, :, None], expected[..., :1],
            rtol=0., atol=1e-12), 'sealed V191 weights differ')
        values[take] *= factors[h, None, :, None, None]
    return values, keep, keys


def summarize(point, boot, mode, scope, counts):
    endpoints, terms = point
    be, bt = boot
    estimates = {}
    for i, name in enumerate(ENDPOINTS):
        p = [*endpoints[:, i], endpoints[:, i].mean(), endpoints[1, i]-endpoints[0, i]]
        b = [be[:, 0, i], be[:, 1, i], be[:, :, i].mean(axis=1), be[:, 1, i]-be[:, 0, i]]
        estimates[name] = {label: saved.prior.interval(v, d) for label, v, d in
            zip(('first32', 'last32', 'all64', 'delta'), p, b)}
    decomposition = {}
    for i, name in enumerate(TERMS):
        result = dict(diagnostic=saved.prior.interval(terms[i], bt[:, i]))
        if scope in PRIMARY and name in ('mix', 'conditional'):
            result['family'] = saved.prior.interval(terms[i], bt[:, i], FAMILY_ALPHA)
        decomposition[name] = result
    return dict(mode=mode, scope=scope, **counts, endpoints=estimates, decomposition=decomposition,
        undefined_bootstrap_decompositions=int((~np.isfinite(bt).all(axis=-1)).sum()),
        no_mass_renormalization=True, complete_parent_day_centering=True)


def run_case(args, actors, mode, scope):
    expected = read(args.v191_root/'details'/f'{mode}_{scope}.json')
    aw, dw = common.weights(len(actors))
    pe, pt = np.zeros((2, len(ENDPOINTS))), np.zeros(len(TERMS))
    be, bt = np.zeros((saved.prior.DRAWS, 2, len(ENDPOINTS))), np.zeros((saved.prior.DRAWS, len(TERMS)))
    support_count, candidate_count, actor_results = 0, 0, []
    for ai, actor in enumerate(actors):
        values, keep, keys = load_actor(args, actor, mode, scope, expected['common_key_support'])
        first, last = (average_days(v, np.full((1, 32), 1./32)) for v in (values[:32], values[32:]))
        e, t = evaluate(first, last, keep)
        require(np.isfinite(t).all(), 'point supported cell has no half exposure')
        pe += e[0]/len(actors)
        pt += t[0]/len(actors)
        support_count += int(keep.sum())
        candidate_count += int(keep.size)
        actor_results.append(dict(actor=actor, supported_cells=int(keep.sum()), endpoints=e[0].tolist(), terms=t[0].tolist()))
        target = args.out_root/'cells'/f'{actor}_{mode}_{scope}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, keys=keys, bins=previous.BINS, supported_both=keep,
            centered_half_sums=np.stack((first[0], last[0])))
        for start in range(0, saved.prior.DRAWS, 256):
            stop = min(start+256, saved.prior.DRAWS)
            a = average_days(values[:32], dw[0, start:stop])
            b = average_days(values[32:], dw[1, start:stop])
            e, t = evaluate(a, b, keep)
            factor = aw[start:stop, ai]
            be[start:stop] += np.where(factor[:, None, None] > 0., factor[:, None, None]*e, 0.)
            bt[start:stop] += np.where(factor[:, None] > 0., factor[:, None]*t, 0.)
        print('V200_CENTERED_MOMENT_ACTOR', mode, scope, actor, flush=True)
    identities(pe, pt)
    identities(be, bt)
    reference = next(r for r in read(args.v199_root/'summary/status.json')['results']
        if (r['mode'], r['scope']) == (mode, scope))
    require(support_count == reference['supported_cells'] and candidate_count == reference['candidate_cells']
        and np.allclose(pe[:, 0], reference['retained_balanced_mass'], rtol=0., atol=1e-12)
        and np.allclose(pe[:, 2], reference['balanced_mass'], rtol=0., atol=1e-12), 'V199 support/exposure readout differs')
    result = summarize((pe, pt), (be, bt), mode, scope, dict(supported_cells=support_count,
        candidate_cells=candidate_count, retained_mass_fraction=(pe[:, 0]/pe[:, 2]).tolist()))
    write(args.out_root/'details'/f'{mode}_{scope}.json', dict(result=result, actors=actor_results,
        endpoint_order=list(ENDPOINTS), term_order=list(TERMS)))
    print('V200_FIXED_SUPPORT_DECOMPOSITION_COMPLETE', mode, scope, flush=True)
    return result


def registration():
    frozen = plan.registration()
    return dict(stage='v200_fixed_support_conditional_decomposition', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), frozen=frozen['frozen'], v199_source='71af51ae4d', v198_source='d347cfadd3',
        readout=frozen['future_readout'], required=frozen['required'],
        signal=saved.VARS[J], target=saved.VARS[Y], centering='Full V194 base-key/day J,Y means BEFORE support exclusion. '
            'Center daily queries algebraically,then pool their centered moments within half/cell. No new residualizer.',
        weights='Saved V199 masks and V191 parent factors fixed across resamples;original query/episode denominators,equal actor/day. '
            'Only divide by cell mass to define t/covariance/mean product;all contributions stay at original nominal mass.',
        terms='For EACH actor/key/supported bin:M=half-mean balanced mass,t=half-mean centered JY sum/M. '
            'mix=DeltaM*average(t),conditional=average(M)*Delta(t);sum cells then average actors. '
            't=conditional covariance+product of conditional centered means;split both contributions by this identity.',
        unsupported='Aggregate mass and centered JY contribution only;no conditional decomposition,mask repair,bin merge or renormalization. '
            'Supported+unsupported must reproduce parent contribution in points and resamples.',
        bootstrap='10000 shared-actor,independent-half-day draws,seed20260803. Center each full original parent/day before resampling. '
            'Recompute pooled cell means/covariances per draw. Zero resampled supported-cell exposure is unknown,not zero; '
            'do not drop/replace a draw or remake masks. Actors with zero bootstrap weight contribute no undefined values.',
        family='8 two-sided mix/conditional contrasts:2 modes x clearance/hour13_moving_launch,alpha=.05/8;other intervals95%. '
            'Hour13 is overlapping diagnostic only. V195 gates unchanged.',
        units='Weighted cross-moment in seconds squared at original nominal mass,not correlation or per-retained-mass covariance.',
        limitations='Reused calibration days;finite bins leave within-bin state mixture and day effects unresolved. '
            'Supported subset cannot explain all parent changes;neither component is causal or held-future control evidence.',
        raw_capture_files_opened=0, interval_fit=False, predictor_fit=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v200_*.py', '-v'], check=True)
    admission = read(args.v199_root/'summary/status.json')
    frozen = read(args.v198_root/'protocol.json')
    require(admission['source'] == '71af51ae4d' and admission['paths'] == 1536 and admission['parent_all9_moments_reproduced']
        and admission['conditional_support_frozen'] and admission['unsupported_moments_retained']
        and admission['protocol']['calibration_days'] == list(DAYS), 'V199 sealed admission differs')
    require(frozen['source'] == 'd347cfadd3' and frozen['future_readout'] == plan.registration()['future_readout'], 'V198 readout differs')
    require((saved.prior.DRAWS, saved.prior.SEED) == (10000, 20260803), 'bootstrap design differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite sealed readout')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    actors = sorted({k['actor'] for k in read(args.v191_root/'details/old_clearance.json')['common_key_support']})
    require(len(actors) == 12, 'actor grid differs')
    print('V200_FIXED_SUPPORT_READOUT_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, actors, mode, scope) for mode, scope in product(('old', 'handoff'), SCOPES)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"fixed_support_readout"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope']))
    contrasts = [dict(mode=r['mode'], scope=r['scope'], term=term,
        estimate=r['decomposition'][term]['family']) for r in results if r['scope'] in PRIMARY for term in ('mix', 'conditional')]
    write(args.out_root/'summary/status.json', dict(status='v200_conditional_decomposition_complete', source=protocol['source'],
        protocol=protocol, results=results, primary_contrasts=contrasts, actors=actors, source_paths=1536,
        v199_masks_and_exposures_reproduced=True, v191_weights_fixed=True, parent_day_centering_checked=True,
        additive_identities_checked=True, low_support_separate=True, no_mass_renormalization=True,
        conditional_associations_estimated=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V200_FIXED_SUPPORT_READOUT_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v198-root', 'v199-root', 'v194-root', 'v191-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
