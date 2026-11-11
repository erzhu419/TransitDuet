"""Read the sealed four-arm comparison on the server; no new traffic or fitting."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
from coupled_model import project_native


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a = p.parse_args()
    archive,physical = run.helper(run.ASSETS,'gate_default'),run.helper(run.ASSETS,'physical')
    missing,regimes = [],{}
    for regime,days in run.PLAN['days'].items():
        by_day = {}
        for day in days:
            arms,uniforms = {},None
            for arm in run.ARMS:
                path = a.root/'cases'/regime/str(day)/arm
                if not (path/'summary/results.json').exists():
                    missing.append([regime,day,arm])
                    continue
                result = run.study.read(path/'summary/results.json')
                trace = archive.raw(path/'raw/path.pkl.gz')
                assert result['physical_account']['passed'] and len(trace['uniforms']) == 5200
                if uniforms is None:
                    uniforms = trace['uniforms']
                assert physical.crn_pair(uniforms,trace['uniforms'])['exact']
                assert len(trace['queries']) == len(trace['graph_queries']) == 5200
                cap = int(arm.rsplit('_',1)[1]) if arm.startswith('fixed_cap_') else 10
                extra = sum(p['shadow_action_s'] != project_native(p['native_action_s'],q['legal'],cap)
                            for q,p in zip(trace['queries'],trace['graph_queries']))
                windows = [p['service_window'] for p in trace['graph_queries'] if 'service_window' in p]
                arms[arm] = dict(metrics=result['endpoints'],graph=result['graph'],
                    wall_s=result['wall_s'],reference_overrides=extra,
                    executed_bins=dict(Counter(q['executed_request_s'] for q in trace['queries'])),
                    early=result['operating_detail']['early_departures'],
                    unserved=result['passenger_detail']['unserved'],
                    windows=dict(queries=len(windows),
                        pending=sum(w['selected']['pending'] > 0 for w in windows),
                        unfinished=sum(w['selected']['unfinished'] > 0 for w in windows),
                        max_balance_error=max((abs(w['selected']['balance_error']) for w in windows),default=0)))
            by_day[str(day)] = arms
        comparisons = {}
        pairs = [('service_single',base) for base in ('fixed_cap_10','fixed_cap_15','v406_single')]
        pairs += [('v406_single','fixed_cap_10'),('fixed_cap_15','fixed_cap_10')]
        for candidate,baseline in pairs:
            paired = {day:{k:v-arms[baseline]['metrics'][k]
                          for k,v in arms[candidate]['metrics'].items()}
                      for day,arms in by_day.items() if candidate in arms and baseline in arms}
            if paired:
                comparisons[candidate+'-'+baseline] = dict(days=len(paired),by_day=paired,
                    mean={k:mean(d[k] for d in paired.values()) for k in next(iter(paired.values()))})
        primary = comparisons.get('service_single-fixed_cap_10')
        passed = (primary['mean']['journey_s_per_passenger'] < 0 and
            all(d[k] <= 0 for d in primary['by_day'].values() for k in
                ('cv_squared','public_delay_s_per_day','readiness_s_per_trip'))
            if primary and primary['days'] == len(days) else None)
        regimes[regime] = dict(by_day=by_day,comparisons=comparisons,development_screen=passed)
    run.study.write(a.root/'readout/summary/results.json',dict(completed=True,
        matrix_complete=not missing,missing=missing,regimes=regimes,
        passed_development_screen=all(r['development_screen'] is True for r in regimes.values()),
        traffic_starts=0,matrix_start_cap=run.PLAN['physical_start_cap'],
        training_updates=0,promotion=False,statistical_unit='paired demand day; development only'))


if __name__ == '__main__':
    main()
