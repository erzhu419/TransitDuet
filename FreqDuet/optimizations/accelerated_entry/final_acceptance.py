"""Close the four-day evidence gaps from the original frozen matrix, read-only."""

import argparse
import gzip
import json
from pathlib import Path
import pickle


def failed_metrics(comparison, base):
    strict = {'passenger_person_s'} if base == 'fixed360_bilateral025' else (
        {'readiness_delay_s', 'public_launch_delay_total_s'} if base == 'frozen_v275_apc' else set())
    result = {}
    for metric in ('passenger_person_s', 'cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s'):
        value = comparison['metrics'][metric]
        upper = value['ci95'][1]
        if (upper >= 0 if metric in strict else upper > 0):
            result[metric] = value
    for metric in ('trips_unlaunched', 'trips_incomplete'):
        value = comparison['metrics'][metric]
        if max(value['by_day'].values()) > 0:
            result[metric] = value
    return result


def accept(matrix, entry_check):
    original = json.loads((matrix/'summary/results.json').read_text())
    checks = json.loads(entry_check.read_text())
    if original['status'] != 'twenty_development_paths_complete':
        return dict(status='original_matrix_incomplete', original_status=original['status'],
            missing=original.get('missing', []), main_policy_changed=False, new_traffic_starts=0)
    assert checks['status'] == 'ready_for_new_frozen_evaluations'
    rows = original['by_day']
    cube = {(row['day'], row['arm']):row for row in rows}
    days = sorted({row['day'] for row in rows})
    assert len(rows) == 20 and len(days) == 4
    pairs = []
    fields = ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms')
    for day in days:
        beam, current = cube[day, 'arrival_cv_beam'], cube[day, 'arrival_cv_current']
        traces = []
        for row in (beam, current):
            with gzip.open(matrix/'traces'/f"p{row['index']}.pkl.gz", 'rb') as stream:
                traces.append(pickle.load(stream))
        pairs.append(dict(day=day, endpoints_exact=beam['results'] == current['results'],
            physical_trace_fields_exact={field:traces[0][field] == traces[1][field] for field in fields},
            beam_wall_s=beam['wall_s'], current_wall_s=current['wall_s']))
    comparisons = {}
    for name, value in original['effects'].items():
        _, base = name.split('_vs_', 1)
        failures = failed_metrics(value, base)
        assert value['joint_gate'] == (not failures), 'Frozen gate and its explanation disagree'
        comparisons[name] = dict(joint_gate=value['joint_gate'], failing_metrics=failures,
            metrics={metric:value['metrics'][metric] for metric in (
                'journey_min', 'cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s', 'holding_s_per_trip')})
    return dict(status='four_day_evidence_and_entry_readout_complete',
        gaps=dict(same_metric_four_day_comparison='complete', benefit_preservation_decision='complete',
                  generic_accelerated_entry='server_checks_passed'),
        same_metric_pairs=pairs, current_vs_beam=original['current_vs_beam'],
        frozen_eligible=original['eligible'], frozen_comparisons=comparisons,
        frozen_decision=original['decision'], source=original['source'],
        independent_demand_days=4, independent_futures=0,
        new_traffic_starts=0, training_updates=0, old_results_changed=False,
        running_tasks_changed=False, main_policy_changed=False,
        next_step='Review the original frozen decision; no automatic promotion, confirmation or retuning.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--entry-check', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = accept(args.matrix, args.entry_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FROZEN_FOUR_DAY_ACCEPTANCE_COMPLETE', value['status'], flush=True)
