"""Read completed archived paths without rerunning traffic or selecting policies."""

import argparse
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run


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
                candidates = [a.root / 'cases' / regime / str(day) / arm,
                    a.root.parent / 'run02/cases' / regime / str(day) / arm]
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
            by_day[str(day)] = arms
        comparisons = {}
        for candidate in ('hold_transfer', 'coordinated'):
            for baseline in ('fixed_bilateral', 'fixed_model', 'cap_10', 'hold_transfer'):
                if baseline == candidate:
                    continue
                paired = {d:{m:v-arms[baseline]['metrics'][m]
                    for m,v in arms[candidate]['metrics'].items()}
                    for d,arms in by_day.items() if candidate in arms and baseline in arms}
                if paired:
                    comparisons[candidate+'-'+baseline] = dict(days=len(paired), by_day=paired,
                        mean={m:mean(v[m] for v in paired.values()) for m in next(iter(paired.values()))})
        results[regime] = dict(by_day=by_day, comparisons=comparisons)
    run.study.write(a.root/'readout/summary/results.json',
        dict(completed=True, matrix_complete=not missing, missing=missing, regimes=results,
             traffic_starts=0, training_updates=0, promotion=False,
             statistical_unit='paired demand day; no incomplete-day substitution or resampling'))


if __name__ == '__main__':
    main()
