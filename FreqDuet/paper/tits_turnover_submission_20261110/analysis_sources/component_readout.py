"""Secondary component comparisons; no new policy, weights or primary gate."""

import argparse
from pathlib import Path
import numpy as np
from study import read,write

METRICS=('journey_s_per_passenger','waiting_s_per_passenger','onboard_s_per_passenger',
         'public_delay_s_per_day','cv_squared','readiness_s_per_trip','holding_s_per_trip')
ARMS=('static_7p5','static_10','cap_10','cap_15')


def summarize(days,table,seed):
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(days),(10000,len(days)))
    effects={}
    for arm in ARMS:
        effects[arm]={}
        for metric in METRICS:
            v=np.array([table[d]['pressure'][metric]-table[d][arm][metric] for d in days])
            effects[arm][metric]=dict(mean=float(v.mean()),
                descriptive_ci95=np.quantile(v[draws].mean(1),[.025,.975]).tolist(),
                by_day=dict(zip(days,v.tolist())))
    return dict(days=days,effects=effects)


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
p.add_argument('command',choices=('historical','fresh'));a=p.parse_args()
if a.command=='historical':
    d=read(a.root.parent.parent/'FreqDuet-v402-budget-mechanism/run01/aggregate/summary/results.json')
    result=dict(completed=True,analysis='secondary component readout of consumed V402 confirmation days',
                **summarize(list(d['by_day']),d['by_day'],4042011),primary_gates=d['decisions'])
else:
    plan=read(Path(__file__).parent/'protocol.json');panels={}
    for regime in plan['regimes']:
        days=list(map(str,plan['days'][regime]));table={}
        for day in days:
            table[day]={arm:read(a.root/'cases'/regime/day/arm/'summary/results.json')['endpoints']
                        for arm in ('pressure',*ARMS)}
        panels[regime]=summarize(days,table,4042011)
    result=dict(completed=True,analysis='all frozen component comparisons on the prespecified new panels',
                panels=panels,statistical_unit='four paired demand days per regime')
result.update(physical_starts=0,training_updates=0,primary_gates_unchanged=True,
              multiplicity='Descriptive secondary intervals; no replacement primary decisions.',
              new_policy_selection=False,new_mixture_weights=False)
write(a.root/('component_'+a.command)/'summary/results.json',result)
