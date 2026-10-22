#!/usr/bin/env python3
"""Read-only V211 selection-archive audit. No controller imports or replay."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ACTIONS = np.array([0., 5., 10., 15., 20., 30., 45.])
LOWER = ((0, 0), (-1, -1), (1, 1), (-1, 1), (1, -1))
UPPER = (0., -30., 30.)
EPS = 1e-8
FIELDS = ('direct_s', 'direct_gross_s', 'total_command_s', 'total_executed_s',
    'rule_response_s', 'raw_upper_delta_s', 'implied_preprojection_ema_delta_s',
    'returned_target_delta_s', 'anchor_plan_delta_s', 'dispatch_mean_delta_s',
    'dispatch_variance_delta_s2', 'inclusive_delay_delta_s', 'readiness_delta_s',
    'journey_delta_s', 'arrival_cv2_delta')
CI_FIELDS = ('direct_s', 'total_command_s', 'rule_response_s', 'dispatch_mean_delta_s',
    'dispatch_variance_delta_s2', 'inclusive_delay_delta_s', 'readiness_delta_s',
    'journey_delta_s', 'arrival_cv2_delta')


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def command_histogram(row):
    counts = {float(k): v for k, v in row['results']['action_counts'].items()}
    require(set(counts) == set(ACTIONS), 'missing/extra action-count bins')
    histogram = np.array([counts[a] for a in ACTIONS], float)
    require(abs(float(histogram @ ACTIONS)-row['ledger']['commanded_holding_s']) <= EPS,
        'action histogram does not reproduce issued holding')
    return histogram


def direct_override(row):
    histogram = np.zeros(7)
    gross = 0.
    for event in row['lower']:
        base, chosen = event['base_bin'], event['selected_bin']
        require(event['actual_command_s'] == ACTIONS[chosen], 'selected bin/command mismatch')
        histogram[chosen] += 1
        histogram[base] -= 1
        gross += abs(float(ACTIONS[chosen]-ACTIONS[base]))
    return histogram, gross


def paired_effect(row, control):
    direct_hist, gross = direct_override(row)
    total_hist = command_histogram(row)-command_histogram(control)
    response_hist = total_hist-direct_hist
    direct = float(direct_hist @ ACTIONS)
    total = row['ledger']['commanded_holding_s']-control['ledger']['commanded_holding_s']
    require(abs(float(total_hist @ ACTIONS)-total) <= EPS, 'paired holding histogram mismatch')
    response = total-direct
    require(abs(float(response_hist @ ACTIONS)-response) <= EPS, 'holding decomposition does not close')
    raw_delta = float(np.asarray(row['anchor_command']['candidate'])[0]-
        np.asarray(control['anchor_command']['candidate'])[0])
    alpha = row['authority']['planner']['action_ema_alpha']
    a, b = row['results'], control['results']
    values = dict(direct_s=direct, direct_gross_s=gross, total_command_s=total,
        total_executed_s=row['ledger']['executed_holding_s']-control['ledger']['executed_holding_s'],
        rule_response_s=response, raw_upper_delta_s=raw_delta,
        implied_preprojection_ema_delta_s=alpha*raw_delta,
        returned_target_delta_s=row['upper']['target_s']-control['upper']['target_s'],
        anchor_plan_delta_s=row['upper']['planned_s']-control['upper']['planned_s'],
        dispatch_mean_delta_s=a['dispatch_gap_mean_s']-b['dispatch_gap_mean_s'],
        dispatch_variance_delta_s2=a['dispatch_gap_variance_s2']-b['dispatch_gap_variance_s2'],
        inclusive_delay_delta_s=a['inclusive_launch_delay_s']-b['inclusive_launch_delay_s'],
        readiness_delta_s=a['readiness_delay_s']-b['readiness_delay_s'],
        journey_delta_s=60*(a['journey_min']-b['journey_min']),
        arrival_cv2_delta=a['cv_squared']-b['cv_squared'])
    return values, np.stack([direct_hist, total_hist, response_hist])


def reduction_class(values):
    if values['direct_s'] >= -EPS:
        return 'no_net_direct_reduction'
    if values['total_command_s'] >= -EPS:
        return 'fully_offset_or_reversed'
    if values['rule_response_s'] > EPS:
        return 'partly_offset'
    if values['rule_response_s'] < -EPS:
        return 'amplified_reduction'
    return 'unchanged_reduction'


def summarized(pairs):
    values = np.array([[p['values'][k] for k in FIELDS] for p in pairs]).reshape(4, 16, len(FIELDS))
    rng = np.random.default_rng(212)
    actors, days = rng.integers(0, 2, (5000, 2)), rng.integers(0, 2, (5000, 2))
    cell_means = values.mean(1).reshape(2, 2, len(FIELDS))
    crossed = cell_means[actors[:, :, None], days[:, None, :]].mean((1, 2))
    counts = rng.multinomial(16, np.full(16, 1/16), size=(5000, 4))
    conditional = np.einsum('bcf,cfk->bk', counts, values, optimize=True)/64
    lo_hi = [np.percentile(v, [2.5, 97.5], axis=0) for v in (crossed, conditional)]
    histograms = np.mean([p['histograms'] for p in pairs], axis=0)
    dispatch_keys = ('dispatch_mean_delta_s', 'dispatch_variance_delta_s2', 'inclusive_delay_delta_s')
    return dict(n_pairs=len(pairs), means=dict(zip(FIELDS, values.mean((0, 1)).tolist())),
        intervals={k: dict(crossed_ci95=lo_hi[0][:, FIELDS.index(k)].tolist(),
            conditional_future_ci95=lo_hi[1][:, FIELDS.index(k)].tolist()) for k in CI_FIELDS},
        context_means=[dict(context=c, **dict(zip(FIELDS, values[c].mean(0).tolist()))) for c in range(4)],
        reduction_classes=dict(Counter(reduction_class(p['values']) for p in pairs)),
        dispatch_changes={k: sum(abs(p['values'][k]) > EPS for p in pairs) for k in dispatch_keys},
        any_saved_dispatch_signature_changed=sum(any(abs(p['values'][k]) > EPS for k in dispatch_keys) for p in pairs),
        nonzero_direct_override_paths=sum(p['values']['direct_gross_s'] > EPS for p in pairs),
        mean_action_count_changes={name: dict(zip(map(lambda a: str(int(a)), ACTIONS), histograms[j].tolist()))
            for j, name in enumerate(('direct_override', 'whole_day', 'rule_response'))})


def scan_inventory(root):
    counts, names, examples, directories = Counter(), Counter(), [], 0
    for directory in sorted((root/'cells').glob('select_*')):
        directories += 1
        for path in directory.rglob('*'):
            if path.is_file():
                counts[path.suffix or '<no suffix>'] += 1
                names[path.name] += 1
                if len(examples) < 8:
                    examples.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size))
    return dict(selection_cell_directories=directories, saved_cell_file_types=dict(counts),
        saved_file_names=dict(names), examples=examples,
        per_trip_dispatch_ledger_persisted_by_v211_writer=False,
        timestamped_full_holding_actions_persisted_by_v211_writer=False)


def audit(root, out):
    protocol = read(root/'protocol.json')
    require(protocol['selection_futures'] == list(range(16)), 'selection bank differs')
    events, pairs, context_counts = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    anomalies, no_effect, candidates = [], Counter(), Counter()
    incomplete, paths, matching_second = [], 0, Counter()
    for context in range(4):
        for future in range(16):
            group = read(root/'select'/f'c{context}_f{future}.json')
            require((group['context'], group['future']) == (context, future), 'selection group mislabeled')
            require([r['candidate'] for r in group['rows']] == list(range(15)), 'selection candidate grid incomplete')
            lookup = {r['candidate']: r for r in group['rows']}
            for candidate, row in lookup.items():
                paths += 1
                u, l = divmod(candidate, 5)
                require(row['continuation'] == 'bilateral025', 'non-selection continuation in input')
                hist = command_histogram(row)
                direct, gross = direct_override(row)
                require(np.isfinite(hist).all(), 'nonfinite action counts')
                if row['ledger']['incomplete_trips'] or row['results']['trips_unlaunched'] or row['results']['trips_incomplete']:
                    incomplete.append(dict(context=context, future=future, candidate=candidate))
                candidates['paths'] += 1
                candidates['paths_with_actual_local_bin_change'] += gross > EPS
                for index, event in enumerate(row['lower']):
                    require(index < 2 and event['offset'] == LOWER[l][index], 'unregistered lower request')
                    offset, base, chosen = event['offset'], event['base_bin'], event['selected_bin']
                    direction = 'decrease' if offset < 0 else 'increase' if offset > 0 else 'zero'
                    for key in ('all/'+direction, f'upper{u}/'+direction, f'decision{index}/'+direction):
                        events[key]['requests'] += 1
                        events[key]['changed'] += base != chosen
                        events[key]['zero_to_zero'] += base == chosen == 0
                        events[key]['unchanged'] += base == chosen
                        events[key][f'base_bin{base}'] += 1
                    context_counts[f'c{context}/upper{u}/lower{l}']['requests'] += offset != 0
                    context_counts[f'c{context}/upper{u}/lower{l}']['changed'] += base != chosen
                    context_counts[f'c{context}/upper{u}/lower{l}']['decrease_zero_to_zero'] += offset < 0 and base == chosen == 0
                if l and gross <= EPS:
                    no_effect['identical_policy_paths'] += 1
                    no_effect[f'upper{u}/paths'] += 1
                    same = row['results'] == lookup[5*u]['results'] and row['ledger'] == lookup[5*u]['ledger']
                    no_effect['exact_saved_outcomes'] += same
                    if not same:
                        anomalies.append(dict(context=context, future=future, candidate=candidate,
                            reason='same upper and no changed lower command but saved outcomes differ'))
                if l:
                    control = lookup[5*u]
                    name = f'lower{l}_at_upper{u}'
                    values, histograms = paired_effect(row, control)
                    pairs[name].append(dict(values=values, histograms=histograms))
                    if row['lower'] and control['lower']:
                        first = row['lower'][0], control['lower'][0]
                        if first[0]['identity'] != first[1]['identity'] or abs(first[0]['time_s']-first[1]['time_s']) > EPS:
                            anomalies.append(dict(context=context, future=future, candidate=candidate,
                                reason='first lower decision differs before any lower override'))
                    if len(row['lower']) == len(control['lower']) == 2:
                        matching_second[name+'/observed'] += 1
                        matching_second[name+'/identity_changed'] += row['lower'][1]['identity'] != control['lower'][1]['identity']
                elif u:
                    values, histograms = paired_effect(row, lookup[0])
                    pairs[f'upper{u}_only'].append(dict(values=values, histograms=histograms))
    require(paths == 960 and all(len(p) == 64 for p in pairs.values()), 'selection archive coverage differs')
    result = dict(status='v212_selection_execution_audit_complete', source_protocol=protocol['stage'],
        selection_paths=paths, contexts=protocol['contexts'], futures=list(range(16)),
        numerical_equality_tolerance=EPS, lower_requests={k: dict(v) for k, v in events.items()},
        context_lower_requests={k: dict(v) for k, v in context_counts.items()},
        candidate_counts=dict(candidates), zero_intervention_equivalence=dict(no_effect),
        paired={k: summarized(p) for k, p in pairs.items()}, second_decision_matching=dict(matching_second),
        incomplete_paths=incomplete, anomalies=anomalies, inventory=scan_inventory(root),
        interpretation=dict(rule_response='Whole-day paired issued holding minus branch-local overrides, '
            'including changed states and event identities;not a pure causal mediation or post-second-only effect.',
            dispatch='Nonzero saved actual-gap/positive public-clock lateness signature proves aggregate execution changed. '
            'Unchanged signatures do not prove identical per-trip launches. Anchor planned_s is not an actual launch.',
            ema='alpha times raw paired coefficient difference is inferred before projection,not a saved executed plan.',
            uncertainty='Descriptive development audit:two actors/two days and original16 selection futures;no promotion test.'),
        new_rollouts=0, training_updates=0, interval_mapping_updates=0, main_policy_changed=False,
        evaluation_files_opened=0, input_files_written=0)
    write(out/'summary/status.json', result)
    print('V212_READ_ONLY_AUDIT_COMPLETE', 'selection_paths', paths, 'anomalies', len(anomalies), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v211-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover',
        '-s', str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v212_*.py', '-v'], check=True)
    audit(a.v211_root, a.out_root)


if __name__ == '__main__':
    main()
