"""Server-only isolated equivalence and timing; no traffic replay or training."""

import argparse
import json
from pathlib import Path
import time
import unittest

import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, choose_current
from freqduet_indexed_continuation import IndexedNativeGraph
from check_freqduet_v276_indexed import profile_call, stress_state
from profile_freqduet_v277_solver import load_module

from fast_continuation import FastArrivalMetricGraph, FastNativeGraph


def timed(fn):
    start, cpu = time.perf_counter(), time.process_time()
    value = fn()
    return value, time.perf_counter() - start, time.process_time() - cpu


def run(a):
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_fast_continuation.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Candidate rejected: exact-output tests failed')
    frozen_choose = load_module(a.frozen_flow, 'detached_frozen_flow').choose
    rows = {}
    for workload, seed in (('late', stress_state()),
                           ('midday', stress_state(history_count=2600, now_s=24000., plans_per_direction=13))):
        for name, old_cls, new_cls in (('native', IndexedNativeGraph, FastNativeGraph),
                                      ('arrival_cv', ArrivalMetricGraph, FastArrivalMetricGraph)):
            old, new = old_cls(seed, seed.observed), new_cls(seed, seed.observed)
            expected = old.evaluate()
            assert expected['coverage'] and not expected['failure']
            assert expected == new.evaluate()
            frozen, frozen_s, _ = timed(lambda:frozen_choose(old))
            prior, prior_s, _ = timed(lambda:flow.choose(old))
            new, build_s, _ = timed(lambda:new_cls(seed, seed.observed))
            candidate, candidate_s, cpu_s = timed(lambda:flow.choose(new))
            assert frozen == prior == candidate, 'Full beam output differs; do not deploy'
            profiled = new_cls(seed, seed.observed)
            check, profile = profile_call(a.out/'profiles'/(workload+'_'+name+'.prof'), lambda:flow.choose(profiled))
            assert check == frozen
            row = dict(frozen_s=frozen_s, prior_optimized_s=prior_s, candidate_s=candidate_s,
                build_s=build_s, cpu_s=cpu_s, speedup_vs_frozen=frozen_s/candidate_s,
                speedup_vs_prior=prior_s/candidate_s, complete_solver_output_exact=True,
                evaluations=candidate['evaluations'], action_s=candidate['action_s'],
                native_rule_cache_hits=new.native_rule.hits, native_rule_unique_inputs=len(new.native_rule.values),
                profile=profile)
            if name == 'arrival_cv':
                old = old_cls(seed, seed.observed)
                new = new_cls(seed, seed.observed)
                before, before_s, _ = timed(lambda:choose_current(old))
                after, after_s, _ = timed(lambda:choose_current(new))
                assert before == after
                row['current_only'] = dict(prior_s=before_s, candidate_s=after_s,
                    evaluations=after['evaluations'], complete_solver_output_exact=True)
            rows[workload+'_'+name] = row
            print('DETACHED_EXACT_CASE_COMPLETE', workload, name, row['speedup_vs_frozen'], flush=True)
    summary = dict(source=a.source, tests=result.testsRun, cases=rows,
        traffic_starts=0, training_updates=0, existing_files_changed=False,
        running_tasks_changed=False, main_policy_changed=False, synthetic_workloads=True)
    destination = a.out/'summary/results.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('DETACHED_EXACT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--frozen-flow', type=Path, required=True)
    run(p.parse_args())
