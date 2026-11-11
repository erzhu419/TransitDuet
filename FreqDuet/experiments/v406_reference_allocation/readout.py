"""Read completed archived paths without rerunning traffic or selecting policies."""

import argparse
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run
from coupled_model import project_native


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=run.Path, required=True)
    a = p.parse_args()
    missing, results = [], {}
    physical = run.helper(run.ASSETS, 'physical')
    archive = run.helper(run.ASSETS, 'gate_default')
    for regime, days in run.PLAN['days'].items():
        by_day = {}
        for day in days:
            arms, crn = {}, None
            for arm in run.ARMS:
                candidates = [a.root / 'cases' / regime / str(day) / arm]
                if arm in run.PLAN.get('reused_arms', []):
                    candidates.append(a.root.parent / 'run01/cases' / regime / str(day) / arm)
                path = next((v for v in candidates if (v/'summary/results.json').exists()), None)
                if path is None:
                    missing.append([regime, day, arm])
                    continue
                row = run.study.read(path/'summary/results.json')
                trace = archive.raw(path/'raw/path.pkl.gz')
                assert row['physical_account']['passed']
                if crn is None:
                    crn = trace['uniforms']
                assert physical.crn_pair(crn, trace['uniforms'])['exact']
                assert len(trace['uniforms']) == 5200
                arms[arm] = dict(metrics=row['endpoints'], graph=row['graph'],
                    slow=row['slow'], wall_s=row['wall_s'],
                    earliness=row['operating_detail']['early_departures'],
                    unserved=row['passenger_detail']['unserved'])
                if arm != 'cap_10':
                    extra = [(q, profile) for q, profile in zip(
                        trace['queries'], trace['graph_queries'])
                        if profile['shadow_action_s'] != project_native(
                            profile['native_action_s'], q['legal'], 10)]
                    arms[arm]['reference_overrides'] = len(extra)
                    arms[arm]['censored_reference_overrides'] = sum(
                        profile['flow'].get('censored_keys', 0) > 0 for _, profile in extra)
                    arms[arm]['commands_above_reference_cap'] = sum(
                        q['executed_request_s'] > 10 for q in trace['queries'])
            by_day[str(day)] = arms
        comparisons = {}
        for candidate in ('simple_cap', 'single', 'transfer', 'coordinated'):
            baselines = ('cap_10',) if candidate == 'simple_cap' else (
                'simple_cap', 'cap_10', 'single', 'transfer')
            for baseline in baselines:
                if baseline == candidate:
                    continue
                paired = {d:{m:v-arms[baseline]['metrics'][m]
                    for m,v in arms[candidate]['metrics'].items()}
                    for d,arms in by_day.items() if candidate in arms and baseline in arms}
                if paired:
                    comparisons[candidate+'-'+baseline] = dict(days=len(paired), by_day=paired,
                        mean={m:mean(v[m] for v in paired.values()) for m in next(iter(paired.values()))})
        screens = {}
        for method in ('single', 'transfer', 'coordinated'):
            controls = [comparisons.get(method+'-'+name) for name in ('simple_cap', 'cap_10')]
            complete = all(c and c['days'] == len(days) for c in controls)
            screens[method] = (all(c['mean']['journey_s_per_passenger'] < 0 and
                all(v[m] <= 0 for v in c['by_day'].values() for m in
                    ('cv_squared', 'readiness_s_per_trip', 'public_delay_s_per_day'))
                for c in controls) if complete else None)
        results[regime] = dict(by_day=by_day, comparisons=comparisons, screens=screens)
    run.study.write(a.root/'readout/summary/results.json',
        dict(completed=True, matrix_complete=not missing, missing=missing, regimes=results,
             traffic_starts=0, training_updates=0, promotion=False,
             candidate_start_budget=run.PLAN['physical_start_cap'],
             reused_arms=run.PLAN.get('reused_arms', []),
             development_screen={method:all(v['screens'][method] is True for v in results.values())
                                 for method in ('single', 'transfer', 'coordinated')},
             statistical_unit='paired demand day; no incomplete-day substitution or resampling'))


if __name__ == '__main__':
    main()
