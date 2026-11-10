"""Compare one completed default path with the frozen K4 path; no rollout."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import traceback

import numpy as np

LOG_SUFFIXES = ('/frequency/low_now', '/frequency/low_forecast', '/left_weight', '/right_weight')
AUDIT_PATHS = {
    'results/stream_audit/exposure/lf_waiting_person_s',
    'results/stream_audit/exposure/lf_onboard_person_s',
    'results/stream_audit/prefix_exposure/lf_waiting_person_s',
    'results/stream_audit/prefix_exposure/lf_onboard_person_s',
    'results/stream_audit/conservation_max_error',
}


def read(path):
    return json.loads(Path(path).read_text())


def raw(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def save(path, value, limit=32768):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False,
                         default=lambda v: v.item())+'\n'
    assert len(encoded.encode()) <= limit
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)


def difference(left, right, path, tolerated):
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            return dict(path=path, reason='array_not_exact')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return dict(path=path, reason='dictionary_fields')
        for key in left:
            found = difference(left[key], right[key], path+'/'+str(key), tolerated)
            if found is not None:
                return found
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return dict(path=path, reason='sequence_type_or_length')
        for index, (old, new) in enumerate(zip(left, right)):
            found = difference(old, new, path+'/'+str(index), tolerated)
            if found is not None:
                return found
    elif left != right:
        audit = path in AUDIT_PATHS
        logged = path.endswith(LOG_SUFFIXES)
        if (audit or logged) and np.isclose(left, right, rtol=0 if audit else 1e-12,
                                           atol=1e-7 if audit else 1e-12):
            tolerated.append(dict(path=path, absolute_difference=abs(float(left)-float(right))))
            return None
        return dict(path=path, reason='value', old=repr(left)[:100], new=repr(right)[:100])
    return None


def compare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    old, new = [read(p/'summary/results.json') for p in (args.reference, args.current)]
    assert new['protocol_id'] == 'v352_budget_interface_source01'
    assert new['budget_retain_fraction'] == .5 and new['upper_fourier_k'] == 4
    for row in (old, new):
        assert row['status'] == 'complete_upper_day' and row['day'] == 2077002
        assert row['learner_seed'] == 314011 and row['split'] == 'development'
        assert row['training_updates'] == 0 and row['physical_starts'] == 1
        assert row['native_actors_unchanged'] and row['independent_upper_unchanged']
    assert old['checkpoint'] == new['checkpoint']
    traces = [raw(p/'raw/path.pkl.gz') for p in (args.reference, args.current)]
    saved = [raw(p/'raw/upper_transitions.pkl.gz') for p in (args.reference, args.current)]
    tolerated, checks = [], {}
    for key in ('results', 'contract', 'coverage', 'execution', 'action_chain', 'slow', 'upper', 'action_semantics'):
        checks['summary_'+key] = difference(old[key], new[key], key, tolerated)
    for key in ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms',
                'returns', 'slow_events', 'publish_ticks', 'public_clocks'):
        checks['trace_'+key] = difference(traces[0][key], traces[1][key], key, tolerated)
    checks['actor_states_rewards_transitions'] = difference(
        saved[0]['transitions'], saved[1]['transitions'], 'transitions', tolerated)
    choices = [t['upper_choices'] for t in traces]
    checks['upper_choices'] = None
    pi_error = 0.
    if len(choices[0]) != len(choices[1]):
        checks['upper_choices'] = dict(reason='decision_count')
    else:
        for i, (left, right) in enumerate(zip(*choices)):
            pi_error = max(pi_error, float(np.max(abs(np.asarray(left['pi'])-np.asarray(right['pi'])))))
            found = difference({k:v for k,v in left.items() if k != 'pi'},
                               {k:v for k,v in right.items() if k != 'pi'}, f'choice/{i}', tolerated)
            if found is not None or pi_error != 0:
                checks['upper_choices'] = found or dict(path=f'choice/{i}/pi', error=pi_error)
                break
    passed = all(v is None for v in checks.values())
    save(args.out/'summary/results.json', dict(protocol_id='v352_default_budget_gate',
        completed=True, engineering_passed=passed, default_exact=passed,
        all_compared_fields_bitwise_equal=passed and not tolerated and pi_error == 0,
        checks={k:dict(passed=v is None, first_difference=v) for k,v in checks.items()},
        tolerated_log_and_audit_differences=tolerated,
        numerical_contract=dict(log_rtol=1e-12, log_atol=1e-12, audit_atol=1e-7,
                                actor_probability_atol=0.), actor_probability_max_abs_error=pi_error,
        reference_path=str(args.reference.resolve()), default_path=str(args.current.resolve()),
        budget_retain_fraction=.5, default_day=2077002, candidate_seed=314011,
        reused_physical_starts=1, physical_starts=0, training_updates=0, model_evaluations=0,
        main_policy_changed=False, confirmation_expanded=False, automatic_retry=False,
        decision='admit_eight_budget_sensitivity_paths' if passed else 'stop_default_reproduction_failed'))
    print('V352_DEFAULT_GATE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'current', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        compare(args)
    except Exception:
        save(args.out/'summary/failure.json', dict(completed=False, engineering_passed=False,
            default_exact=False, error=traceback.format_exc()[-6000:], physical_starts=0,
            training_updates=0, model_evaluations=0, automatic_retry=False))
        raise
