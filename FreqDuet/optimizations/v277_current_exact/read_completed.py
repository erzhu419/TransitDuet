"""Read completed engineering results and paired physical traces on the server."""

import argparse
import gzip
import json
from pathlib import Path
import pickle


def report(results):
    matrix = results/'protocol_v6_v277_metric_alignment/matrix_repaired'
    rows = [json.loads(path.read_text()) for path in (matrix/'pairs').glob('p*.json')]
    cube = {(row['day'], row['arm']):row for row in rows}
    protocol = json.loads((matrix/'protocol.json').read_text())
    optimized = json.loads((results/'protocol_v6_v277_current_exact/summary/results.json').read_text())
    assert optimized['status'] == 'current_complete_day_exact_equivalence_passed'
    assert optimized['day'] == 2077002
    pairs = []
    fields = ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms')
    for day in protocol['days']:
        if (day, 'arrival_cv_beam') not in cube or (day, 'arrival_cv_current') not in cube:
            continue
        beam, current = cube[day, 'arrival_cv_beam'], cube[day, 'arrival_cv_current']
        traces = []
        for row in (beam, current):
            with gzip.open(matrix/'traces'/f"p{row['index']}.pkl.gz", 'rb') as stream:
                traces.append(pickle.load(stream))
        matches = {field:traces[0][field] == traces[1][field] for field in fields}
        item = dict(day=day, endpoints_exact=beam['results'] == current['results'],
            physical_trace_fields_exact=matches,
            physical_trace_counts={field:len(traces[0][field]) for field in fields},
            beam_wall_s=beam['wall_s'], current_wall_s=current['wall_s'],
            descriptive_beam_vs_current_wall_ratio=beam['wall_s']/current['wall_s'])
        if day == optimized['day']:
            item.update(optimized_current_wall_s=optimized['candidate_wall_s'],
                optimized_vs_current_endpoints_exact=optimized['results'] == current['results'],
                optimized_vs_current_saved_traces_exact=optimized['all_saved_trace_records_exact'],
                descriptive_beam_vs_optimized_wall_ratio=beam['wall_s']/optimized['candidate_wall_s'],
                descriptive_current_vs_optimized_wall_ratio=current['wall_s']/optimized['candidate_wall_s'])
        pairs.append(item)
    return dict(status='read_only_completed_pair_report', completed_indices=sorted(row['index'] for row in rows),
        missing_indices=sorted(set(range(20))-{row['index'] for row in rows}),
        failures=[path.name for path in (matrix/'failures').glob('*.json')],
        same_metric_completed_pairs=pairs, traffic_starts=0, training_updates=0,
        old_results_changed=False, running_tasks_changed=False, main_policy_changed=False,
        scope='Completed paired days only; physical execution parity is not equality of all predicted plans. '
              'Archived timings are descriptive; full four-day eligibility remains with the frozen aggregator.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = report(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CURRENT_COMPLETED_READOUT_DONE', len(value['completed_indices']),
          len(value['same_metric_completed_pairs']), flush=True)
