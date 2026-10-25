#!/usr/bin/env python3
"""Bounded whole-day lower-policy development; independent confirmation stays closed."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from multiprocessing import get_context
from pathlib import Path
import pickle
import unittest

import numpy as np
import torch

import freqduet_day_policy_search as learn
import run_freqduet_v267_bilateral_factorial as fixed


SEED, DAYS = fixed.SEEDS[0], fixed.DAYS
ENGINEERING = ('baseline', 'conditional', 'uniform')
SEARCH_PATH_CAP = (learn.MAX_VECTORS - 1) * len(DAYS)
PHYSICAL_START_CAP = len(ENGINEERING) + 2 * SEARCH_PATH_CAP


class DayPolicy(fixed.BenchmarkAudit):
    def __init__(self, theta):
        super().__init__(SEED, 'fixed_bilateral025')
        self.theta = np.asarray(theta, dtype=float)
        assert self.theta.shape == (len(learn.FEATURES),)
        self.changes = dict(decisions=0, increases=0, decreases=0)

    def attach(self, runner):
        super().attach(runner)
        routes = runner.env.routes
        middle = len(routes) // 2
        self.lengths = [sum(r.distance for r in routes[middle:]), sum(r.distance for r in routes[:middle])]

    def __call__(self, policy, states, features, theta):
        base = super().__call__(policy, states, features, theta)
        runner, env = self.runner, self.runner.env
        bus = runner._bus_for_agent(runner._pilot_agent)
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        current = learn.observation(context, len(bus.passengers) / bus.capacity,
            bus.travel_distance / self.lengths[int(bus.direction)],
            env._ready_vehicle_count(not bool(bus.direction)),
            fixed.pilot.due_at_return(env, not bool(bus.direction)))
        assert np.isfinite(current).all()
        selected = learn.choose(context, current, self.theta, policy.feasible_action_mask(states)[0].numpy())
        native = int(torch.argmax(base[0]))
        self.changes['decisions'] += 1
        self.changes['increases'] += selected > native
        self.changes['decreases'] += selected < native
        law = torch.zeros_like(base)
        law[0, selected] = 1.
        return law


def claim(root, source, name):
    allowed = {f'engineering_{name}' for name in ENGINEERING} | {
        f'{family}_v{v}_d{day}' for family in learn.FAMILIES
        for v in range(1, learn.MAX_VECTORS) for day in DAYS}
    assert name in allowed
    fixed.write_new(root / 'started' / (name + '.json'), dict(source=source, path=name, automatic_retry=False))


def run_day(root, source, name, day, theta, save_trace=False):
    claim(root, source, name)
    rule, credit = DayPolicy(theta), fixed.pilot.CreditObserver()
    row, _, _, holds = fixed.pilot.rollout(root, SEED, day, 'stochastic', 187000000 + SEED + 100 * day,
        [0.] * 4, 'v268_' + name, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch, **rule.delays)
    result = dict(source=source, day=day, results=row, theta=list(theta), contract=rule.contract,
        execution=rule.execution, coverage=rule.coverage, same_state_actions=rule.changes,
        base_actor_updates=0, dual_updates=0, main_policy_changed=False)
    fixed.write_new(root / 'paths' / (name + '.json'), result)
    if save_trace:
        path = root / 'traces' / (name + '.pkl.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'xb') as stream:
            pickle.dump(dict(holds=holds, dispatches=rule.dispatches, arrivals=rule.arrivals,
                departures=rule.departures), stream, protocol=pickle.HIGHEST_PROTOCOL)
    print('V268_DAY_DONE ' + name, flush=True)
    return result


def reference(results, day):
    old = results / 'protocol_v6_v267_bilateral_factorial'
    row = fixed.read(old / 'pairs' / (fixed.key((SEED, day, 'fixed_bilateral025')) + '.json'))
    assert row['source'] == '25ed615133'
    assert (row['seed'], row['day'], row['mode']) == (SEED, day, 'fixed_bilateral025')
    return row


def preflight(a):
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1] / 'tests'), pattern='test_v268_day_policy_search.py'))
    assert tests.testsRun and tests.wasSuccessful()
    fixed.write_new(a.root / 'protocol.json', dict(source=a.source, training_days=DAYS,
        upper='Exact 360-second requested schedule; physical readiness still controls launches',
        families=learn.FAMILIES, features=learn.FEATURES, bounds=[-1., 1.],
        population=learn.POPULATION, generations=learn.GENERATIONS, optimizer_seed=learn.SEARCH_SEED,
        vectors_per_family=learn.MAX_VECTORS, new_paths_per_family_cap=SEARCH_PATH_CAP,
        physical_start_cap=PHYSICAL_START_CAP, engineering_paths=len(ENGINEERING),
        original_vehicles=12, action_bins_s=learn.ACTIONS.tolist(), confirmation_started=False,
        main_policy_changed=False, automatic_retry=False, max_concurrent_rollouts=8))
    neutral = run_day(a.root, a.source, 'engineering_baseline', DAYS[0], [0.] * len(learn.FEATURES), True)
    old = reference(a.results_root, DAYS[0])
    for field in ('contract', 'coverage', 'execution'):
        fixed.pilot.joint.assert_same_state(neutral[field], old[field], 'V268_cached_' + field)
    for metric in (*fixed.METRICS, 'action_counts'):
        fixed.pilot.joint.assert_same_state(neutral['results'][metric], old['results'][metric], 'V268_cached_' + metric)
    old_path = a.results_root / 'protocol_v6_v267_bilateral_factorial/traces' / (
        fixed.key((SEED, DAYS[0], 'fixed_bilateral025')) + '.pkl.gz')
    with gzip.open(old_path, 'rb') as stream:
        previous = pickle.load(stream)
    with gzip.open(a.root / 'traces/engineering_baseline.pkl.gz', 'rb') as stream:
        current = pickle.load(stream)
    fixed.pilot.joint.assert_same_state(previous, current, 'V268_zero_policy_full_action_timing_identity')
    probes = (('conditional', [0., -.4, .2, .1, -.2, .2]), ('uniform', [.2, 0., 0., 0., 0., 0.]))
    for family, theta in probes:
        result = run_day(a.root, a.source, 'engineering_' + family, DAYS[0], theta)
        fixed.pilot.joint.assert_same_state(result['contract'], old['contract'], 'V268_same_fleet_clock_demand')
        assert result['same_state_actions']['increases'] + result['same_state_actions']['decreases'] > 0
        assert result['execution']['decisions'] == result['same_state_actions']['decisions']
    fixed.write_new(a.root / 'preflight/done.json', dict(source=a.source, tests=tests.testsRun,
        status='exact_neutral_path_and_nonzero_executed_policies_passed', physical_starts=3,
        outcomes_used_for_release=False, confirmation_started=False, main_policy_changed=False))
    print('V268_PREFLIGHT_COMPLETE', flush=True)


def optimize(a):
    gate = fixed.read(a.root / 'preflight/done.json')
    assert gate['source'] == a.source and gate['status'] == 'exact_neutral_path_and_nonzero_executed_policies_passed'
    references = [reference(a.results_root, day) for day in DAYS]
    fixed.write_new(a.root / 'search' / a.family / 'started.json', dict(source=a.source, automatic_retry=False))
    vectors = []
    with ProcessPoolExecutor(max_workers=len(DAYS), mp_context=get_context('spawn')) as pool:
        def evaluate(values, index):
            theta = learn.coefficients(values, a.family).tolist()
            vectors.append(dict(index=index, theta=theta))
            fixed.write_new(a.root / 'search' / a.family / f'v{index}.json', dict(theta=theta, index=index))
            if not any(theta):
                assert index == 0
                return [r['results'] for r in references]
            jobs = [pool.submit(run_day, a.root, a.source, f'{a.family}_v{index}_d{day}', day, theta) for day in DAYS]
            rows = [job.result() for job in jobs]
            for row, old in zip(rows, references):
                fixed.pilot.joint.assert_same_state(row['contract'], old['contract'], 'V268_matched_development_contract')
            return [r['results'] for r in rows]
        result = learn.search(evaluate, a.family)
    result.update(source=a.source, days=DAYS, vectors=vectors,
        status='bounded_complete_day_search_finished', base_actor_updates=0, dual_updates=0,
        main_policy_changed=False, confirmation_started=False, controller_admitted=False)
    fixed.write_new(a.root / 'search' / a.family / 'done.json', result)
    print('V268_SEARCH_COMPLETE ' + a.family, flush=True)


def aggregate(a):
    results = {family: fixed.read(a.root / 'search' / family / 'done.json') for family in learn.FAMILIES}
    for family, value in results.items():
        assert value['source'] == a.source and value['family'] == family
        assert value['status'] == 'bounded_complete_day_search_finished'
    actual = len(list((a.root / 'started').glob('*.json')))
    expected = 3 + sum((r['measured_vectors'] - 1) * len(DAYS) for r in results.values())
    assert actual == expected <= PHYSICAL_START_CAP
    conditional, uniform = (np.asarray(results[f]['day_cost_changes']) for f in learn.FAMILIES)
    delta = conditional - uniform
    joint_vs_uniform = bool(np.all(delta[:, 0] < 0.) and np.all(delta[:, 1:] <= 0.))
    fixed.write_new(a.root / 'summary/status.json', dict(source=a.source,
        status='complete_day_policy_development_complete', searches=results, physical_starts=actual,
        physical_start_cap=PHYSICAL_START_CAP, conditional_minus_uniform=delta.tolist(),
        conditional_stable_on_development=results['conditional']['per_day_joint_improvement'],
        conditional_joint_over_uniform_on_development=joint_vs_uniform,
        confirmation_eligible=results['conditional']['per_day_joint_improvement'] and joint_vs_uniform,
        confirmation_started=False, main_policy_changed=False, controller_admitted=False,
        scope='Four consumed development days; optimizer fit, not generalization or frequency evidence'))
    print('V268_DEVELOPMENT_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'search', 'aggregate'))
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--family', choices=learn.FAMILIES)
    a = p.parse_args()
    a.root = a.results_root / 'protocol_v6_v268_day_policy_search'
    if a.command == 'preflight':
        preflight(a)
    elif a.command == 'search':
        assert a.family
        optimize(a)
    else:
        aggregate(a)


if __name__ == '__main__':
    main()
