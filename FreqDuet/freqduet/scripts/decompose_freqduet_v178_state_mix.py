#!/usr/bin/env python3
"""Describe state-mix versus within-state lower-tail variation in saved V178 cells."""

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v178_lower_width as audit


def decomposition(mass, lower):
    pooled_mass = mass.mean(axis=0)
    active = pooled_mass > 0.
    mass, lower = mass[:, active], lower[:, active]
    rates = lower.mean(axis=0)/pooled_mass[active]
    total = lower.sum(axis=1)
    composition = (mass-mass.mean(axis=0))@rates
    within = (lower-mass*rates).sum(axis=1)
    np.testing.assert_allclose(composition+within, total-total.mean(), atol=1e-10)
    variance = float(np.var(total))
    covariance = {name: float(np.mean(value*(total-total.mean())))
        for name, value in (('composition', composition), ('within_state', within))}
    audit.close(sum(covariance.values()), variance)
    return dict(day_variance=variance, covariance=covariance,
        signed_fractions=None if variance == 0. else {k: v/variance for k, v in covariance.items()},
        pooled_rates=rates.tolist(), pooled_mass=pooled_mass[active].tolist())


def run(root):
    summary = audit.read(root/'summary/status.json')
    assert summary['execution_source'] == 'da3857a784' and summary['endpoints_and_crossed_intervals_reproduced']
    rows = [r for p in sorted((root/'cells').glob('*.json')) for r in audit.read(p)['cells']]
    output = []
    for mode, scope in product(('old', 'handoff'), audit.SCOPES):
        selected = [r for r in rows if (r['mode'], r['scope']) == (mode, scope)]
        actors, days = sorted({r['actor'] for r in selected}), sorted({r['day'] for r in selected})
        assert len(selected) == 768 and len(actors) == 12 and len(days) == 64
        index = {(r['actor'], r['day']): r for r in selected}
        assert len(index) == 768
        for axis in ('remaining_stops', 'stage', 'key_days', 'target_stop', 'direction'):
            categories = sorted({str(p['value']) for r in selected for p in r['profiles'] if p['axis'] == axis})
            mass, lower = np.zeros((64, len(categories))), np.zeros((64, len(categories)))
            for i, day in enumerate(days):
                for actor in actors:
                    for p in index[actor, day]['profiles']:
                        if p['axis'] == axis:
                            j = categories.index(str(p['value']))
                            mass[i, j] += p['values'][2]/12.
                            lower[i, j] += p['values'][3+audit.METRICS.index('below')]/12.
            np.testing.assert_allclose(mass.sum(axis=1), 1., atol=1e-10)
            active = mass.mean(axis=0) > 0.
            result = decomposition(mass, lower)
            output.append(dict(mode=mode, scope=scope, axis=axis,
                categories=[c for c, present in zip(categories, active) if present], **result))
    destination = root/'state_mix/summary/status.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(status='v178_readonly_state_mix_complete', results=output,
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], input_source='da3857a784',
        interval_fit=False, map_updates=0, new_replays=0,
        new_training=0, main_policy_changed=False,
        interpretation='Pooled rates are descriptive standardization, not a fitted predictor or deployment rule.'), indent=2)+'\n')
    print('V178_READONLY_STATE_MIX_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v178_state_mix.py', '-v'], check=True)
    run(parser.parse_args().root)
