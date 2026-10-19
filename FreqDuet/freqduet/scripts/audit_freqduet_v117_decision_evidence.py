#!/usr/bin/env python3
"""Read V115 decision evidence; separate shared residual input from harmful response."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v116_quota_regularity as prior


CI_FIELDS = ('D_s', 'increment_s', 'delta_base_s', 'residual_in_delta_base_s',
    'other_in_delta_base_s', 'base_and_increment_same_sign', 'raw_nonpositive',
    'next_service_gap_error_s', 'predicted_restore_beneficial', 'ready_restore_beneficial',
    'ready_vs_departure_sign_disagrees', 'pre_action_lag_s')


def recover(row, departure, predecessor):
    result = dict(D_s=row['feedback_error_s'], increment_s=row['increment_s'],
        applied_increment_s=row['increment_s'] if row['feedback_error_s'] > 0 else 0.,
        base_request_s=.25*row['feedback_error_s'], raw_request_s=row['raw_request_s'],
        raw_nonpositive=row['raw_request_s'] <= 0, onboard_passengers=row['onboard_passengers'])
    if row['residual_s'] is None:
        return dict(result, regime='missing_forward')
    release = departure['release']-predecessor['release']
    holding = row['executed_before_s']-predecessor['realized_holding']
    forward = release+holding+row['residual_s']
    ready = predecessor['time_s']+forward
    if not (forward >= 0 and departure['arrival_time'] <= ready <= row['policy_time_s']
            and predecessor['time_s'] <= ready <= departure['time_s']-row['executed_s']):
        raise AssertionError('reconstructed forward/ready coordinate is outside its causal interval')
    if row['raw_request_s'] != result['base_request_s']+result['applied_increment_s']:
        raise AssertionError('current raw request differs from the frozen D/innovation law')
    predicted = 2*row['feedback_error_s']+forward if row['follower_valid'] else None
    if predicted is not None and predicted < 0:
        raise AssertionError('reconstructed follower prediction is negative')
    return dict(result, regime='bilateral' if row['follower_valid'] else 'fallback',
        forward_gap_s=forward, release_gap_s=release, holding_gap_s=holding,
        residual_s=row['residual_s'], ready_time_s=ready,
        pre_action_lag_s=departure['time_s']-row['executed_s']-ready,
        policy_capture_lag_s=row['policy_time_s']-ready, predicted_follower_gap_s=predicted,
        target_s=None if row['follower_valid'] else forward+row['feedback_error_s'])


def transition(row, current, old_row, old):
    if row['reset_reason'] != 'continued':
        return dict(transition='reference_reset')
    if (old is None or current['regime'] == 'missing_forward' or old['regime'] == 'missing_forward'
            or row['predecessor_trip_id'] != old_row['predecessor_trip_id']
            or abs(row['station_id']-old_row['station_id']) != 1):
        raise AssertionError('continued innovation lacks its same-predecessor adjacent prefix')
    dr = current['residual_s']-old['residual_s']
    if dr != row['innovation_s'] or -.75*dr != row['increment_s']:
        raise AssertionError('delta_R does not reconstruct the saved incremental request')
    db = .25*(current['D_s']-old['D_s'])
    np.testing.assert_allclose(current['raw_request_s']-old['raw_request_s'],
        db+current['applied_increment_s']-old['applied_increment_s'], atol=1e-9, rtol=0)
    if current['regime'] != old['regime']:
        return dict(transition='regime_changed', delta_R_s=dr)
    bilateral = current['regime'] == 'bilateral'
    coefficient = .125 if bilateral else .25
    parts = dict(residual_in_delta_base_s=-coefficient*dr,
        holding_in_delta_base_s=-coefficient*(current['holding_gap_s']-old['holding_gap_s']),
        release_in_delta_base_s=-coefficient*(current['release_gap_s']-old['release_gap_s']))
    name = 'predicted_follower_gap_s' if bilateral else 'target_s'
    parts['follower_or_target_in_delta_base_s'] = coefficient*(current[name]-old[name])
    np.testing.assert_allclose(sum(parts.values()), db, atol=1e-9, rtol=0)
    return dict(parts, transition='stable_'+current['regime'], delta_R_s=dr, delta_base_s=db,
        other_in_delta_base_s=db-parts['residual_in_delta_base_s'],
        base_and_increment_same_sign=db*current['applied_increment_s'] > 0,
        base_and_increment_opposed=db*current['applied_increment_s'] < 0)


def pair_restore_change(forward, follower_gap, executed, restore):
    return 2*restore*(forward+2*executed-follower_gap)+2*restore**2


def summarize(rows):
    numeric = sorted({k for r in rows for k, v in r.items()
        if isinstance(v, (bool, int, float)) and k != 'ready_time_s'})
    result = {}
    for name in numeric:
        values = np.asarray([float(r[name]) for r in rows if r.get(name) is not None])
        result[name] = dict(n=len(values), mean=float(values.mean()),
            p10=float(np.percentile(values, 10)), median=float(np.median(values)),
            p90=float(np.percentile(values, 90)))
    transitions = {name: sum(r.get('transition') == name for r in rows) for name in
        ('stable_bilateral', 'stable_fallback', 'regime_changed', 'reference_reset')}
    correlation = {}
    for mode in ('stable_bilateral', 'stable_fallback'):
        valid = [r for r in rows if r.get('transition') == mode]
        value = None
        if len(valid) > 1:
            x, y = np.asarray([[r['delta_R_s'], r['delta_base_s']] for r in valid]).T
            if x.std() > 0 and y.std() > 0:
                value = float(np.corrcoef(x, y)[0, 1])
        correlation[mode] = dict(n=len(valid), correlation=value)
    return dict(decisions=len(rows), cut_s=sum(r.get('cut_s', 0.) for r in rows),
        transitions=transitions, regimes={mode: sum(r['regime'] == mode for r in rows) for mode in
            ('bilateral', 'fallback', 'missing_forward')}, delta_R_base_correlation=correlation, metrics=result)


def analyse_path(root, seed, day):
    path = prior.read_path(root, seed, day, prior.frozen.PRIMARY)
    dep = {prior.key(r): r for r in path['departures']}
    evidence = {}
    previous = {}
    groups = defaultdict(list)
    for identity, row in path['decisions'].items():
        before = dep.get((row['predecessor_trip_id'], identity[1], identity[2]))
        if row['residual_s'] is not None and before is None:
            raise AssertionError('saved predecessor has no completed same-stop departure')
        current = recover(row, dep[identity], before)
        old_row, old = previous.get(identity[0], (None, None))
        current.update(transition(row, current, old_row, old))
        evidence[identity] = current
        previous[identity[0]] = row, current
        groups['all_decisions'].append(current)
    stops = defaultdict(list)
    for row in path['departures']:
        stops[(int(row['station_id']), bool(row['direction']))].append(row)
    coverage = defaultdict(int)
    for events in stops.values():
        times = np.asarray([r['time_s'] for r in events])
        for index, end in enumerate(events):
            identity = prior.key(end)
            if identity not in path['decisions']:
                continue
            row, current = path['decisions'][identity], evidence[identity]
            parts = prior.action_parts(row)
            for cause, cut in (('negative', parts['negative_cut_s']), ('quota', -parts['quota'])):
                if cut <= 0:
                    continue
                shifted = times.copy()
                shifted[index] += cut
                geometry = prior.geometry_change(times, shifted, path['departure'])
                cv = geometry['cv_squared_change']
                label = 'useful' if cv < -1e-12 else 'harmful' if cv > 1e-12 else 'neutral'
                record = dict(current, cut_s=cut, geometry_cv2_change=cv)
                internal = index > 0 and index+1 < len(events)
                record['boundary'] = not internal
                if internal and current['regime'] != 'missing_forward':
                    predecessor, follower = events[index-1], events[index+1]
                    same = int(predecessor['trip_id']) == row['predecessor_trip_id']
                    record['completed_predecessor_matches'] = same
                    if same:
                        ready = current['ready_time_s']
                        executed = end['time_s']-ready
                        forward = current['forward_gap_s']
                        departure_gap = follower['time_s']-ready
                        qdep = pair_restore_change(forward, departure_gap, executed, cut)
                        crossing = shifted[index] > times[index+1]
                        record['restore_crosses_follower'] = crossing
                        if not crossing:
                            np.testing.assert_allclose(qdep, geometry['squared_gap_change_s2'], atol=1e-8, rtol=0)
                            record['departure_restore_beneficial'] = qdep < 0
                        predicted = current.get('predicted_follower_gap_s')
                        if predicted is not None:
                            record['predicted_restore_beneficial'] = pair_restore_change(
                                forward, predicted, executed, cut) < 0
                        next_evidence = evidence.get(prior.key(follower))
                        resolved = next_evidence is not None and 'ready_time_s' in next_evidence
                        record['next_service_ready_resolved'] = resolved
                        if resolved:
                            gap = next_evidence['ready_time_s']-ready
                            record['next_service_ready_temporal_valid'] = gap >= 0
                            if gap >= 0:
                                record['next_service_ready_gap_s'] = gap
                                record['next_service_after_ready_s'] = follower['time_s']-next_evidence['ready_time_s']
                                qready = pair_restore_change(forward, gap, executed, cut)
                                record['ready_restore_beneficial'] = qready < 0
                                if not crossing:
                                    record['ready_vs_departure_sign_disagrees'] = (qready < 0) != (qdep < 0)
                                if predicted is not None:
                                    record['next_service_gap_error_s'] = predicted-gap
                                    record['next_service_gap_abs_error_s'] = abs(predicted-gap)
                                    record['predicted_vs_ready_sign_disagrees'] = record['predicted_restore_beneficial'] != (qready < 0)
                groups[cause].append(record)
                groups[cause+'_'+label].append(record)
                coverage[cause+'_decisions'] += 1
                coverage[cause+'_cut_s'] += cut
                coverage[cause+'_'+label+'_decisions'] += 1
                coverage[cause+'_'+label+'_cut_s'] += cut
    print('V117_CONSUMED_CELL_DONE', seed, day, flush=True)
    return groups, dict(coverage)


def review(root, v116, out):
    baseline = json.loads((v116/'summary/status.json').read_text())
    if baseline['status'] != 'v115_readonly_quota_regularity_audit_complete':
        raise AssertionError('V117 requires the completed V116 quota-extension analysis')
    combined, cells = defaultdict(list), []
    for seed in prior.frozen.SEEDS:
        for day in prior.frozen.DAYS:
            groups, coverage = analyse_path(root, seed, day)
            means = {}
            for group, rows in groups.items():
                combined[group].extend(rows)
                summary = summarize(rows)
                means[group] = {m: summary['metrics'][m]['mean'] for m in CI_FIELDS if m in summary['metrics']}
            cells.append(dict(seed=seed, day=day, means=means, coverage=coverage))
    for cause in ('negative', 'quota'):
        rows = combined[cause]
        old = baseline['cohorts'][cause]
        if len(rows) != old['decisions'] or sum(r['cut_s'] for r in rows) != old['cut_s']:
            raise AssertionError('V117 changes V116 cut cohort support')
        for label in ('useful', 'harmful'):
            rows = combined[cause+'_'+label]
            if len(rows) != old[label+'_restores'] or sum(r['cut_s'] for r in rows) != old[label+'_cut_s']:
                raise AssertionError('V117 changes V116 beneficial/harmful geometry labels')
    complete = {}
    for group in combined:
        if all(group in c['means'] for c in cells):
            metrics = set.intersection(*(set(c['means'][group]) for c in cells))
            if metrics:
                complete[group] = sorted(metrics)
    inference_rows = [dict(seed=c['seed'], day=c['day'], values={g: {m: c['means'][g][m] for m in metrics}
        for g, metrics in complete.items()}) for c in cells]
    prior.frozen.pilot.write_json(out/'summary/status.json', dict(
        status='v115_readonly_decision_evidence_audit_complete',
        cohorts={g: summarize(rows) for g, rows in combined.items()}, actor_day=cells,
        cell_mean_intervals=prior.inference.describe(inference_rows, 'values'),
        consumed_paths=len(cells), decisions=len(combined['all_decisions']),
        follower_prediction_identity_saved=False, fitted_models=0, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        exact_coefficients=dict(bilateral_residual_in_quarter=-.125, fallback_residual_in_quarter=-.25,
            additional_increment=-.75),
        limitations='Same information enters a level feedback and an incremental response; algebraic overlap '
            'alone does not prove harmful control. Ready labels belong to the next completed service, not '
            'an identified predicted AVL follower: V115 did not save follower IDs. Label error can mix identity '
            'and ETA errors. Future ready/departure labels are descriptive only, never policy inputs. '
            'Local geometry is not a closed-loop return; consumed two-actor/four-day evidence only. '
            'Intervals are crossed actor/day means where every cell has observed values; unresolved labels '
            'are excluded with observed counts, never imputed as zero.'))
    print('V117_READONLY_DECISION_EVIDENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v115-root', type=Path, required=True)
    p.add_argument('--v116-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v117_*.py', '-v'], check=True)
    review(a.v115_root, a.v116_root, a.out_root)
