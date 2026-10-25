#!/usr/bin/env python3
"""Read-only endpoint/action review of the completed, frozen V268 search."""

import argparse
from pathlib import Path

import numpy as np

import run_freqduet_v268_day_policy_search as run


METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
           'public_launch_delay_total_s', 'holding_s_per_trip', 'waiting_person_s',
           'onboard_person_s', 'trips_unlaunched', 'trips_incomplete')
GOALS = ('passenger_person_s', *run.learn.CONSTRAINTS)


def summarize(rows, reference, index):
    daily = [{m: row['results'][m] - old['results'][m] for m in METRICS}
             for row, old in zip(rows, reference)]
    means = {m: float(np.mean([d[m] for d in daily])) for m in METRICS}
    changes = {key: sum(row['same_state_actions'][key] for row in rows)
               for key in ('decisions', 'increases', 'decreases')} if index else dict(
                   decisions=sum(row['execution']['decisions'] for row in rows), increases=0, decreases=0)
    return dict(index=index, mean_effects=means, day_effects=daily, same_state_actions=changes,
        mean_executed_holding_change_s=float(np.mean([r['execution']['executed_s'] - old['execution']['executed_s']
                                                    for r, old in zip(rows, reference)])),
        mean_feasible=all(means[m] <= 0. for m in run.learn.CONSTRAINTS),
        mean_joint_improvement=means['passenger_person_s'] < 0. and all(means[m] <= 0. for m in run.learn.CONSTRAINTS),
        per_day_joint_improvement=all(d['passenger_person_s'] < 0. and all(d[m] <= 0. for m in run.learn.CONSTRAINTS)
                                      for d in daily))


def review(a):
    root = a.results_root / 'protocol_v6_v268_day_policy_search'
    saved = run.fixed.read(root / 'summary/status.json')
    preflight = run.fixed.read(root / 'preflight/done.json')
    assert saved['source'] == preflight['source'] == a.source
    assert saved['status'] == 'complete_day_policy_development_complete'
    assert preflight['status'] == 'exact_neutral_path_and_nonzero_executed_policies_passed'
    reference = [run.reference(a.results_root, day) for day in run.DAYS]
    families, expected = {}, {'engineering_' + x for x in run.ENGINEERING}
    all_rows = []
    for family in run.learn.FAMILIES:
        fit = saved['searches'][family]
        records = []
        for index, vector in enumerate(fit['vectors']):
            assert vector['index'] == index
            if not index:
                rows = reference
            else:
                names = [f'{family}_v{index}_d{day}' for day in run.DAYS]
                expected.update(names)
                rows = [run.fixed.read(root / 'paths' / (name + '.json')) for name in names]
                for row, day in zip(rows, run.DAYS):
                    assert row['source'] == a.source and row['day'] == day
                    assert row['theta'] == vector['theta']
                    assert row['base_actor_updates'] == row['dual_updates'] == 0
                    assert not row['main_policy_changed']
                all_rows.extend(rows)
            record = summarize(rows, reference, index)
            np.testing.assert_allclose([record['mean_effects'][m] for m in GOALS],
                fit['evaluations'][index]['mean_cost_changes'], rtol=0, atol=1e-7)
            records.append(record)
        nonzero = records[1:]
        selected = records[fit['selected_vector_index']]
        np.testing.assert_allclose([selected['mean_effects'][m] for m in GOALS], fit['mean_cost_changes'], rtol=0, atol=1e-7)
        assert selected['per_day_joint_improvement'] == fit['per_day_joint_improvement']
        families[family] = dict(selected=selected, vectors=records, evaluated_vectors=len(records),
            nonzero_mean_feasible=sum(r['mean_feasible'] for r in nonzero),
            nonzero_mean_joint_improvements=sum(r['mean_joint_improvement'] for r in nonzero),
            passenger_improvements=sum(r['mean_effects']['passenger_person_s'] < 0. for r in nonzero),
            passenger_improvements_with_cv_harm=sum(r['mean_effects']['passenger_person_s'] < 0.
                and r['mean_effects']['cv_squared'] > 0. for r in nonzero),
            cv_improvements_with_passenger_harm=sum(r['mean_effects']['cv_squared'] < 0.
                and r['mean_effects']['passenger_person_s'] > 0. for r in nonzero),
            vectors_with_both_increases_and_decreases=sum(r['same_state_actions']['increases'] > 0
                and r['same_state_actions']['decreases'] > 0 for r in nonzero),
            solver_message=fit['solver_message'], solver_generations=fit['generations'])
    for subdir in ('started', 'paths'):
        assert {p.stem for p in (root / subdir).glob('*.json')} == expected
    assert len(expected) == saved['physical_starts'] <= run.PHYSICAL_START_CAP
    pairs = []
    for i in range(1, run.learn.POPULATION):
        assert saved['searches']['conditional']['vectors'][i]['theta'][0] == saved['searches']['uniform']['vectors'][i]['theta'][0]
        c, u = (families[f]['vectors'][i] for f in run.learn.FAMILIES)
        pairs.append(dict(index=i, same_initial_bias=True,
            conditional_minus_uniform={m: c['mean_effects'][m] - u['mean_effects'][m] for m in METRICS},
            holding_amount_matched=False))
    relation = [r['results']['public_launch_delay_total_s'] / r['results']['readiness_delay_s']
                for r in [*reference, *all_rows] if r['results']['readiness_delay_s'] > 0.]
    result = dict(source=a.source, review_source=a.review_source,
        status='frozen_v268_endpoint_and_action_review_complete',
        development_task='t211591', development_exit_code=0, physical_starts=len(expected),
        preflight_passed=True, frozen_search_summaries_reproduced=True,
        all_nonzero_paths_service_cleared=all(r['execution']['done_reason'] == 'service_cleared' for r in all_rows),
        max_unexecuted_holding_s=max(r['execution']['unexecuted_s'] for r in all_rows),
        baseline_absolute_means={m: float(np.mean([r['results'][m] for r in reference])) for m in METRICS},
        days=run.DAYS, frozen_runtime_seed=run.SEED, families=families,
        same_bias_initial_pairs=pairs, public_total_over_readiness_mean_range=[min(relation), max(relation)],
        confirmation_eligible=saved['confirmation_eligible'], confirmation_started=False,
        new_rollouts=0, new_model_fits=0, main_policy_changed=False,
        interpretation='Selected zero vectors are literal baseline reuse, not estimated equivalence. '
            'Nonzero vectors and same-bias pairs are descriptive development results, not new winners. '
            'Same-state changes compare the original bilateral rule at each visited state; '
            'they do not match whole trajectories or total holding. No actor-axis inference or post-selection CI.')
    run.fixed.write_new(root / 'review/summary.json', result)
    print('V268_READ_ONLY_REVIEW_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--review-source', required=True)
    review(p.parse_args())


if __name__ == '__main__':
    main()
