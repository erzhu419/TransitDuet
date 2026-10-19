#!/usr/bin/env python3
"""Read V118 only: completed stop clocks, ETA error and holding-sign sensitivity."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v118_follower_identity as prior


JOURNEY = 'same_time_avl_journey_speed_eta'
VARIANTS = ('raw', 'minus_holding', 'minus_service', 'minus_both')
PARTS = ('holding', 'service', 'ticks')


class CompletedPrefix:
    """Index only completed departures; future windows are used for labels only."""

    def __init__(self, departures, arrivals, ready):
        self.departures = sorted(departures, key=lambda r: r['time_s'])
        self.times = [r['time_s'] for r in self.departures]
        self.arrival_times = sorted(r['time_s'] for r in arrivals.values())
        self.values = [np.zeros(3)]
        for row in self.departures:
            identity = prior.audit.key(row)
            value = self.values[-1].copy()
            holding = row['realized_holding']-value[0]
            if identity in arrivals:
                arr = arrivals[identity]
                actual = row['realized_holding']-arr['realized_holding']
                np.testing.assert_allclose(holding, actual, atol=1e-9, rtol=0)
                service = ready[identity]['ready_time_s']-arr['time_s']
                ticks = row['time_s']-ready[identity]['ready_time_s']-actual
                if min(holding, service, ticks) < 0:
                    raise AssertionError('negative completed stop-clock component')
                value += (holding, service, ticks)
            elif holding != 0:
                raise AssertionError('initial departure has unobserved holding')
            self.values.append(value)

    def before(self, now):
        return self.values[bisect_left(self.times, now)].copy()

    def phase_label(self, now, target_arrival):
        # This uses physical event labels, never gates an ETA sensitivity.
        i, j = bisect_left(self.times, now), bisect_left(self.arrival_times, now)
        if ((i < len(self.times) and self.times[i] == now) or
                (j < len(self.arrival_times) and self.arrival_times[j] == now)):
            return 'event_boundary'
        if target_arrival <= now:
            return 'target_already_arrived'
        if i and (not j or self.times[i-1] > self.arrival_times[j-1]):
            return 'in_transit'
        return 'pending_stop'


def clock_projection(evidence, launch, prefix):
    elapsed = evidence['ready_time_s']-launch
    if elapsed <= 0 or prefix.min() < 0 or prefix.sum() > elapsed:
        raise AssertionError('completed clocks do not fit the physical launch clock')
    speed, distance = evidence['speed_mps'], evidence['spatial_gap_m']
    eta = distance/speed
    np.testing.assert_allclose(eta, evidence['eta_prediction_s'], atol=1e-8, rtol=0)
    progress = speed*elapsed
    return distance/progress*prefix


def future_parts(eta, actual_eta, projection, future):
    mismatch = projection-future
    other = (eta-projection.sum())-(actual_eta-future.sum())
    np.testing.assert_allclose(mismatch.sum()+other, eta-actual_eta, atol=1e-8, rtol=0)
    return dict(**{p+'_mismatch_s': float(v) for p, v in zip(PARTS, mismatch)},
        other_error_s=float(other), **{p+'_future_s': float(v) for p, v in zip(PARTS, future)})


def sign_metrics(gaps, truth, forward, executed, cut):
    benefit = prior.prior.pair_restore_change(forward, truth, executed, cut) < 0
    predicted = {v: prior.prior.pair_restore_change(forward, g, executed, cut) < 0
        for v, g in gaps.items()}
    result = dict(same_ready_pair_beneficial=bool(benefit))
    for variant, positive in predicted.items():
        values = dict(sign_disagrees=positive != benefit, false_positive=positive and not benefit,
            false_negative=not positive and benefit, true_positive=positive and benefit,
            lost_benefit_vs_raw=predicted['raw'] and benefit and not positive,
            corrected_false_positive=predicted['raw'] and not benefit and not positive)
        result.update({variant+'_'+k: bool(v) for k, v in values.items()})
    return result


def summarize(rows):
    names = {k for row in rows for k, v in row.items() if isinstance(v, (bool, int, float))}
    return dict(decisions=len(rows), metrics={k: dict(n=sum(k in row for row in rows),
        mean=float(np.mean([row[k] for row in rows if k in row]))) for k in sorted(names)},
        follower_sources=dict(Counter(row['follower_source'] for row in rows)))


def read_cell(root, index):
    seed, day = prior.ITEMS[index]
    name = f'{seed}_{day}_{prior.frozen.PRIMARY}_0'
    trace = prior.read_trace(root/'traces'/(name+'.jsonl.gz'))
    ready = {prior.audit.key(r): r for r in prior.read_trace(root/'ready'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        events = {k: [dict(zip(fields, r.tolist())) for r in record[k]] for k in ('arrivals', 'departures')}
        launches = {int(r[0]): dict(zip(record['launch_fields'].tolist(), r.tolist())) for r in record['launches']}
    arrivals = {prior.audit.key(r): r for r in events['arrivals']}
    departures = {prior.audit.key(r): r for r in events['departures']}
    by_trip_a, by_trip_d = defaultdict(dict), defaultdict(list)
    for k, row in arrivals.items():
        by_trip_a[k[0]][k] = row
    for row in events['departures']:
        by_trip_d[int(row['trip_id'])].append(row)
    prefixes = {trip: CompletedPrefix(rows, by_trip_a[trip], ready) for trip, rows in by_trip_d.items()}
    groups = defaultdict(list)
    for row in trace:
        e, own = row['evidence'], prior.audit.key(row)
        base = dict(follower_eligible=bool(row['follower_valid']), follower_source=e['follower_source'])
        if not row['follower_valid']:
            groups['unlabelled_decisions'].append(base)
            continue
        target = int(e['predicted_follower_trip_id']), own[1], own[2]
        if ready[target]['bus_id'] != e['predicted_follower_bus_id']:
            raise AssertionError('V118 same-object ready label is no longer matched')
        now, eta = e['ready_time_s'], e['eta_prediction_s']
        actual_eta = arrivals[target]['time_s']-now
        truth = ready[target]['ready_time_s']-now
        if truth < 0:
            raise AssertionError('V118 same-object ready label is in the past')
        projection = np.zeros(3)
        if e['follower_source'] == JOURNEY:
            prefix = prefixes[target[0]]
            history = prefix.before(now)
            projection = clock_projection(e, launches[target[0]]['time_s'], history)
            base.update({p+'_completed_s': float(v) for p, v in zip(PARTS, history)})
            base.update({p+'_extrapolated_s': float(v) for p, v in zip(PARTS, projection)})
            phase = prefix.phase_label(now, arrivals[target]['time_s'])
            base['in_transit_label'] = phase == 'in_transit'
            if phase == 'in_transit':
                future = prefix.before(arrivals[target]['time_s'])-history
                np.testing.assert_allclose(history[0],
                    prefix.departures[bisect_left(prefix.times, now)-1]['realized_holding'], atol=1e-9, rtol=0)
                np.testing.assert_allclose(future[0], arrivals[target]['realized_holding']-history[0], atol=1e-9, rtol=0)
                if future.min() < 0 or future.sum() > actual_eta:
                    raise AssertionError('future completed stop clocks do not fit target arrival')
                base.update(future_parts(eta, actual_eta, projection, future))
        else:
            phase = 'other_eta_source'
        corrections = dict(raw=0., minus_holding=projection[0], minus_service=projection[1],
            minus_both=projection[0]+projection[1])
        gaps = {v: e['follower_gap_prediction_s']-c for v, c in corrections.items()}
        for variant, correction in corrections.items():
            eta_error, gap_error = eta-correction-actual_eta, gaps[variant]-truth
            base.update({variant+'_eta_error_s': eta_error, variant+'_eta_mae_s': abs(eta_error),
                variant+'_ready_error_s': gap_error, variant+'_ready_mae_s': abs(gap_error),
                variant+'_eta_mae_delta_s': abs(eta_error)-abs(eta-actual_eta)})
        groups['forecasts'].append(base)
        groups['phase_'+phase].append(base)
        parts = prior.audit.action_parts(row)
        for cause, cut in (('negative', parts['negative_cut_s']), ('quota', -parts['quota'])):
            if cut <= 0:
                continue
            executed = departures[own]['time_s']-now
            signs = sign_metrics(gaps, truth, e['forward_gap_s'], executed, cut)
            values = dict(base, cut_s=cut, **signs)
            departure_truth = departures[target]['time_s']-now
            for variant, gap in gaps.items():
                values[variant+'_departure_sign_disagrees'] = bool(
                    (prior.prior.pair_restore_change(e['forward_gap_s'], gap, executed, cut) < 0) !=
                    (prior.prior.pair_restore_change(e['forward_gap_s'], departure_truth, executed, cut) < 0))
            groups[cause].append(values)
            groups[cause+'_'+phase].append(values)
    return groups


def aggregate(root, out):
    parent = json.loads((root/'summary/status.json').read_text())
    if parent['status'] != 'v115_passive_identity_replay_complete' or not all(
            all(r[k] for k in ('original_endpoints_exact', 'original_decisions_exact',
                'full_timing_arrays_exact', 'v117_coordinates_exact')) for r in parent['receipts']):
        raise AssertionError('V118 exact replay must complete before ETA-clock diagnosis')
    combined, cells = defaultdict(list), []
    ci_fields = ['raw_eta_error_s', 'raw_eta_mae_s', 'holding_extrapolated_s',
        'service_extrapolated_s', 'holding_future_s', 'service_future_s', 'holding_mismatch_s',
        'service_mismatch_s', 'other_error_s']
    for v in VARIANTS:
        ci_fields += [v+'_'+m for m in ('eta_mae_delta_s', 'sign_disagrees', 'false_positive',
            'false_negative', 'lost_benefit_vs_raw', 'corrected_false_positive', 'departure_sign_disagrees')]
    for index, (seed, day) in enumerate(prior.ITEMS):
        groups = read_cell(root, index)
        means = {}
        for group, rows in groups.items():
            combined[group].extend(rows)
            if group in ('forecasts', 'phase_in_transit', 'negative', 'quota'):
                metrics = summarize(rows)['metrics']
                means[group] = {m: metrics[m]['mean'] for m in ci_fields if m in metrics}
        cells.append(dict(seed=seed, day=day, means=means))
        print('V119_READ_ONLY_CELL_DONE', seed, day, flush=True)
    for group, parent_group in (('forecasts', 'forecast_identity_match'), ('negative', 'negative'), ('quota', 'quota')):
        expected = parent['cohorts'][parent_group]
        count = expected['metrics']['follower_eligible']['n'] if group == 'forecasts' else expected['metrics']['same_ready_error_s']['n']
        if len(combined[group]) != count:
            raise AssertionError('read-only cohort differs from V118 resolved forecast support')
    common = {}
    for group in combined:
        if all(group in c['means'] for c in cells):
            fields = set.intersection(*(set(c['means'][group]) for c in cells))
            if fields:
                common[group] = sorted(fields)
    inference_rows = [dict(seed=c['seed'], day=c['day'], values={g: {m: c['means'][g][m] for m in fields}
        for g, fields in common.items()}) for c in cells]
    prior.frozen.pilot.write_json(out/'summary/status.json', dict(status='v118_read_only_eta_clock_complete',
        source_paths=len(prior.ITEMS), cohorts={g: summarize(rows) for g, rows in combined.items()},
        actor_day=cells, cell_mean_intervals=prior.audit.inference.describe(inference_rows, 'values'),
        fitting=False, replayed_paths=0, new_demand_days=0, actor_updates=0, critic_updates=0, dual_updates=0,
        quota_changed=False, main_policy_changed=False,
        limitations='Analytical clock-removal sensitivities, not fitted or executed ETA/controllers. '
            'Strictly completed departure prefixes only; future phases/components are labels, never features. '
            'Future decomposition covers in-transit followers, not pending stops or exact event boundaries. '
            'ETA components are timing identities, not causal error shares. Historical dwell/holding can '
            'predict genuine future stops, so removing them need not help. Holding signs fix other recorded '
            'times and are not changed-action CV2 or system returns. Two archived actors/four consumed days.'))
    print('V119_READ_ONLY_ETA_CLOCK_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v118-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v119_*.py', '-v'], check=True)
    aggregate(a.v118_root, a.out_root)
