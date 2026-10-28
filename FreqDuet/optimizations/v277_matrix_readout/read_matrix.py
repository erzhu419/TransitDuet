"""Read existing V277 result JSON only; retain missing cells and frozen gates."""

import argparse
import json
from pathlib import Path


def readout(matrix):
    protocol = json.loads((matrix/'protocol.json').read_text())
    rows = [json.loads(path.read_text()) for path in (matrix/'pairs').glob('p*.json')]
    rows.sort(key=lambda row:row['index'])
    assert len({row['index'] for row in rows}) == len(rows)
    assert all(row['source'] == protocol['source'] for row in rows)
    lookup = {(row['day'], row['arm']):row for row in rows}
    current = 'arrival_cv_current'
    metrics = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
               'public_launch_delay_total_s', 'holding_s_per_trip', 'trips_unlaunched', 'trips_incomplete')
    comparisons = {}
    for arm in ('fixed360_bilateral025', 'frozen_v275_apc', 'frozen_v276_apc', 'arrival_cv_beam'):
        paired = []
        for day in protocol['days']:
            if (day, current) not in lookup or (day, arm) not in lookup:
                continue
            candidate, reference = lookup[day, current], lookup[day, arm]
            delta = {m:candidate['results'][m]-reference['results'][m] for m in metrics}
            delta['journey_s_per_passenger'] = 60*delta['journey_min']
            paired.append(dict(day=day, delta=delta, current_wall_s=candidate['wall_s'],
                reference_wall_s=reference['wall_s'],
                descriptive_wall_speedup=reference['wall_s']/candidate['wall_s']))
        comparisons['current_vs_'+arm] = paired
    return dict(status='descriptive_partial_matrix_readout' if len(rows)<20 else 'complete_matrix_available',
        source=protocol['source'], completed_indices=[row['index'] for row in rows],
        pending_indices=[i for i in range(20) if i not in {row['index'] for row in rows}],
        failures=[path.name for path in (matrix/'failures').glob('*.json')],
        current_days=[d for d in protocol['days'] if (d, current) in lookup],
        same_metric_beam_current_paired_days=[d for d in protocol['days']
            if (d, current) in lookup and (d, 'arrival_cv_beam') in lookup],
        comparisons=comparisons, frozen_gates=protocol['gates'],
        gate_evaluation='reserved_for_original_full_matrix_aggregator', confidence_intervals_computed=False,
        traffic_starts=0, training_updates=0, old_results_changed=False,
        running_tasks_changed=False, main_policy_changed=False,
        limits='Available paired days only; V276 versus current also changes planner CV2 semantics. '
               'No four-day or pure search-simplification claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = readout(args.matrix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('V277_PARTIAL_READOUT_COMPLETE', len(result['completed_indices']),
          len(result['current_days']), flush=True)
