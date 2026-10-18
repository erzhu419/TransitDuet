#!/usr/bin/env python3
"""Cross-condition V112 diagnostics without replays or alternative actions."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys

import run_freqduet_v112_decision_trace as trace


def cohorts(row):
    if not row['valid_forward']:
        return {'missing_forward'}
    names = {'valid_forward'}
    label = row['completed_follower']
    matched = (label is not None and label['ready_label_temporal_valid']
               and label['next_arrival_identity_matches'])
    if matched:
        names.add('matched_completed_follower')
    repeated = (row['previous_residual_s'] is not None and row['previous_residual_s'] < 0
        and row['residual_s'] < 0 and row['own_executed_holding_s'] > 0)
    if repeated:
        names.add('repeated_negative_after_holding')
        if row['prior_positive_excess_s'] > 0:
            names.add('repeated_negative_after_shadow_excess')
        if row['feedback_error_s'] <= 0 and row['chosen_s'] > 0:
            names.add('repeat_feedback_nonpositive_positive_hold')
            if row['innovation_s'] >= 0:
                names.add('repeat_no_new_negative_innovation_positive_hold')
            if not row['valid_bilateral']:
                names.add('repeat_positive_hold_without_follower_forecast')
            elif matched:
                names.add('repeat_positive_hold_matched_completed_follower')
                names.add('repeat_positive_hold_completed_feedback_' +
                    ('nonpositive' if label['completed_feedback_error_s'] <= 0 else 'positive'))
    if row['base_shadow_s'] == 0 and row['chosen_s'] > 0:
        names.add('base_zero_positive_override')
        if matched and label['completed_feedback_error_s'] <= 0:
            names.add('base_zero_override_completed_feedback_nonpositive')
    if row['chosen_s'] == 45:
        names.add('cap45')
        if matched:
            names.add('cap45_matched_completed_follower')
            names.add('cap45_' + ('persists' if label['completed_shadow_command_s'] == 45 else 'lower') +
                '_with_completed_follower')
    return names


def accumulate(totals, row):
    for name in cohorts(row):
        cell = totals[name]
        cell['count'] += 1
        cell['command_sum_s'] += row['chosen_s']
        cell['shadow_delta_sum_s'] += row['chosen_s']-row['base_shadow_s']
        for out, field in (('raw_feedforward_sum_s', 'actual_feedforward_s'),
            ('clipping_delta_sum_s', 'clipping_delta_s'), ('quantization_delta_sum_s', 'quantization_delta_s')):
            cell[out] += row[field]
        label = row['completed_follower']
        if label is not None and label['ready_label_temporal_valid'] and label['next_arrival_identity_matches']:
            cell['matched_labels'] += 1
            cell['matched_absolute_forecast_error_sum_s'] += abs(label['prediction_error_s'])
            cell['matched_command_minus_completed_shadow_sum_s'] += row['chosen_s']-label['completed_shadow_command_s']


def review(root):
    parent = json.loads((root / 'summary/status.json').read_text())
    if not parent['all_cached_endpoints_audits_and_timing_exact'] or parent['consumed_path_replays'] != len(trace.ITEMS):
        raise AssertionError('cross-review requires all original endpoints and timing to reproduce')
    combined = {m: defaultdict(lambda: defaultdict(float)) for m in trace.METHODS}
    cells = []
    for index, (seed, day, method) in enumerate(trace.ITEMS):
        pair = json.loads((root / 'pairs' / f'p{index}.json').read_text())
        if (pair['seed'], pair['day'], pair['method']) != (seed, day, method) or not pair['exact_v110_endpoints_and_timing']:
            raise AssertionError('cross-review path identity or endpoint receipt differs')
        local, count = defaultdict(lambda: defaultdict(float)), 0
        with gzip.open(root / 'traces' / f'p{index}.jsonl.gz', 'rt', encoding='ascii') as source:
            for line in source:
                row = json.loads(line)
                accumulate(local, row)
                accumulate(combined[method], row)
                count += 1
        if count != pair['decisions']:
            raise AssertionError('completed trace does not cover the reproduced decisions')
        cells.append(dict(seed=seed, day=day, method=method, cohorts=dict(local)))
    trace.pilot.write_json(root / 'review/status.json', dict(status='v112_cross_condition_review_complete',
        combined={m: dict(v) for m, v in combined.items()}, actor_day=cells,
        consumed_paths_read=len(cells), new_rollouts=0, main_policy_changed=False,
        rollout_source='e8792aa58b', all_original_endpoints_exact=True,
        scope='consumed two-actor/four-day paths; overlapping descriptive cohorts, not independent confirmations',
        limitations='Completed follower shadows use realized continuation, not counterfactual outcomes. '
            'Matched means the predicted physical follower is the next actual arrival. '
            'Clipping and forecast/repeated-level effects overlap; no additive causal attribution of service loss.'))
    print('V112_CROSS_CONDITION_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v112_trace_review.py', '-v'], check=True)
    review(a.out_root)
