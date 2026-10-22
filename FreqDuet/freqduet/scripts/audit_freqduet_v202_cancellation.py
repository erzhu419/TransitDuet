#!/usr/bin/env python3
"""Read saved V202 draws only and retain signed covariance cancellation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v202_key_day_covariance as parent


def partition(parts, total, labels):
    parent.require(np.allclose(parts.sum(axis=1), total, rtol=1e-8, atol=1e-8), 'component draws do not reproduce total')
    centered = parts-parts.mean(axis=0)
    covariance = centered.T@centered/len(parts)
    allocation = covariance.sum(axis=1)
    variance = float(total.var())
    parent.require(np.allclose(covariance.sum(), variance, rtol=1e-8, atol=1e-8), 'signed covariance partition differs')
    return dict(labels=labels, variance=variance, covariance=covariance.tolist(), signed_allocation=allocation.tolist(),
        shares=(allocation/variance).tolist() if variance > 0. else None,
        sum_own_variances=float(np.trace(covariance)), cross_covariance_total=float(covariance.sum()-np.trace(covariance)))


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v202_cancellation.py', '-v'], check=True)
    source = parent.read(args.parent_root/'summary/status.json')
    parent.require(source['source'] == '8829065fe1' and source['v201_fixed_key_points_and_variances_reproduced'], 'requires completed V202')
    results = []
    for mode in ('old', 'handoff'):
        for scope in parent.base.PRIMARY:
            with np.load(args.parent_root/'details'/f'{mode}_{scope}_draws.npz') as z:
                boot, point = z['boot'], z['point']
                parent.require(z['keys'].tolist() == [list(k) for k in parent.KEYS], 'fixed keys differ')
            for ki, key in enumerate(parent.KEYS):
                results.append(dict(mode=mode, scope=scope, key=list(key), conditional_point=float(point[ki, 1, 0]),
                    temporal=partition(boot[:, ki, 2:5, 0], boot[:, ki, 1, 0], ['within_day', 'between_day_centered', 'mean_product']),
                    innovation=partition(np.stack((boot[:, ki, 1, 1], -boot[:, ki, 1, 2]), axis=1), boot[:, ki, 1, 0], ['I', 'minus_A'])))
    parent.write(args.out_root/'summary/status.json', dict(status='v202_saved_draw_cancellation_complete',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], parent_source='8829065fe1', results=results,
        interpretation='Signed covariance allocation of matched diagnostic draws,not independent component variances or causal shares.',
        raw_capture_files_opened=0, moment_files_opened=0, masks_and_weights_unchanged=True,
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0, main_policy_changed=False))
    print('V202_SAVED_DRAW_CANCELLATION_COMPLETE_NO_RESCAN', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    run(p.parse_args())
