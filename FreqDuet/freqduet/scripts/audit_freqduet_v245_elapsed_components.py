#!/usr/bin/env python3
"""Split the diagnosed V245 other-elapsed term using saved stop clocks only."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
import unittest

import numpy as np

from audit_freqduet_v245_holding_propagation import ARMS, PAIRS, close, key, series, unique, write


NAMES = ('release', 'holding', 'observed_nonholding_stop', 'remaining_travel_and_clock')


def split(trace):
    departures = unique(trace['departures'], key)
    accumulated, last = defaultdict(float), {}
    coordinates = {}
    for arrival in trace['arrivals']:
        k, trip = key(arrival), arrival['trip_id']
        dep = departures[k]
        assert arrival['time_s'] >= last.get(trip, -np.inf)
        dwell = dep['time_s']-arrival['time_s']-(dep['realized_holding']-arrival['realized_holding'])
        assert dwell >= 0, 'holding exceeds observed stop duration'
        service = accumulated[trip]
        coordinates[k] = np.asarray([arrival['release'], arrival['realized_holding'],
            service, arrival['other_elapsed']-service])
        close(coordinates[k].sum(), arrival['time_s'], 'V92/V109 observed-stop definition')
        accumulated[trip] += dwell
        last[trip] = dep['time_s']
    return series(trace['arrivals'], coordinates)


def effects(b, c):
    delta = (c['covariance']-b['covariance'])/b['mean']**2
    result = {name: float(delta[i, i]) for i, name in enumerate(NAMES)}
    result.update({a+'__'+z: float(2*delta[i, j]) for i, a in enumerate(NAMES)
                   for j, z in enumerate(NAMES) if j > i})
    result['normalization'] = c['variance']*(1/c['mean']**2-1/b['mean']**2)
    close(sum(result.values()), c['cv2']-b['cv2'], 'stop/remaining covariance closure')
    return result


def analyze(root, parent_file, out, source):
    parent = json.loads(parent_file.read_text())
    assert parent['status'] == 'v245_readonly_holding_propagation_complete'
    assert parent['source'] == '39cd832b46'
    paths, dispatches = {}, {}
    for arm in ARMS:
        with gzip.open(root/'records'/f'smoke_c0_{arm}.pkl.gz', 'rb') as stream:
            saved = pickle.load(stream)
        assert saved['source'] == 'b879b53b25'
        paths[arm] = split(saved['trace'])
        dispatches[arm] = {r['trip']: r for r in saved['trace']['actual_launches']}
        close(paths[arm]['cv2'], saved['results']['cv_squared'], 'unchanged original CV2')
        del saved
        print('V245_ELAPSED_PREPARED '+arm, flush=True)
    contrasts = {}
    for candidate, baseline in PAIRS:
        name = candidate+'-'+baseline
        parts = effects(paths[baseline], paths[candidate])
        prior = parent['contrasts'][name]['arrival_cv2_parts']
        close(parts['observed_nonholding_stop']+parts['remaining_travel_and_clock']+
            parts['observed_nonholding_stop__remaining_travel_and_clock'], prior['other'], 'collapsed other variance')
        for short in ('release', 'holding'):
            close(parts[short+'__observed_nonholding_stop']+parts[short+'__remaining_travel_and_clock'],
                  prior[short+'_other'], 'collapsed '+short+'/other covariance')
        b, c = dispatches[baseline], dispatches[candidate]
        shifts = [c[k]['actual_s']-b[k]['actual_s'] for k in b.keys() & c.keys()]
        contrasts[name] = dict(cv2_parts=parts, actual_launches=dict(
            matched=len(shifts), baseline_only=len(b.keys()-c.keys()), candidate_only=len(c.keys()-b.keys()),
            same_bus=sum(b[k]['bus'] == c[k]['bus'] for k in b.keys() & c.keys()),
            shift_counts=dict(Counter('earlier' if d < 0 else 'later' if d > 0 else 'unchanged' for d in shifts)),
            total_shift_s=sum(shifts), minimum_shift_s=min(shifts), maximum_shift_s=max(shifts)))
    result = dict(status='v245_observed_stop_split_complete', source=source,
        parent_analysis_source='39cd832b46', contrasts=contrasts, original_and_parent_closures_passed=True,
        new_rollouts=0, training_updates=0, controller_changed=False, v238_changed=False, formal_started=False,
        definitions='Same V92/V109 definition:observed stop duration less actual holding,accumulated '
            'strictly before each arrival. Remaining includes travel and unobserved terminal/clock effects,'
            'not a separately observed pure road-running duration.',
        limitations='Consumed single smoke context/future;covariance accounting,not causal mediation or efficacy inference.')
    write(out/'summary/status.json', result)
    print('V245_ELAPSED_COMPLETE '+json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v244-root', type=Path, required=True)
    p.add_argument('--parent-summary', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v245_holding_propagation.py')
    assert suite.countTestCases() > 0
    assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    analyze(a.v244_root, a.parent_summary, a.out_root, a.source)


if __name__ == '__main__':
    main()
