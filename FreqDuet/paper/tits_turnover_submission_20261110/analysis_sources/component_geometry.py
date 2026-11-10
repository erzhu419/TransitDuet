"""Show all already-tested simple-control components, without choosing new weights."""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study import read, write
from archive_figures import style, save, COLORS

p=argparse.ArgumentParser();p.add_argument('--root',type=__import__('pathlib').Path,required=True)
p.add_argument('--assets',type=__import__('pathlib').Path,required=True);a=p.parse_args()
old=a.root.parent.parent/'FreqDuet-v402-budget-mechanism/run01'
data=read(old/'aggregate/summary/results.json');sealed=read(old/'selection/summary/results.json')
metrics=('cv_squared','public_delay_s_per_day','journey_s_per_passenger')
means={arm:{m:float(np.mean([day[arm][m] for day in data['by_day'].values()])) for m in metrics}
       for arm in next(iter(data['by_day'].values()))}
out=a.root/'components/summary';out.mkdir(parents=True,exist_ok=True)
style();fig,axes=plt.subplots(1,2,figsize=(7.15,2.7))
mixtures={}
for family,marker,label in (('static','s','Static trip credit'),('cap','o','Stop cap')):
    components=sealed['families'][family]['components']
    mixtures[family]={m:sum(c['probability']*means[c['arm']][m] for c in components) for m in metrics}
    for i,(ax,metric) in enumerate(zip(axes,('public_delay_s_per_day','journey_s_per_passenger'))):
        scale=60 if i==0 else 1
        ax.plot([means[c['arm']]['cv_squared'] for c in components],
            [means[c['arm']][metric]/scale for c in components],color=COLORS[family],lw=.8,
            ls='--',marker=marker,ms=4,mfc='white',label=label+' components')
        ax.scatter(mixtures[family]['cv_squared'],mixtures[family][metric]/scale,
            color=COLORS[family],marker=marker,s=40,zorder=4)
        for c in components:
            ax.annotate(str(c['parameter']),
                (means[c['arm']]['cv_squared'],means[c['arm']][metric]/scale),
                xytext=(4,4),textcoords='offset points',fontsize=6,color=COLORS[family])
for i,(ax,metric) in enumerate(zip(axes,('public_delay_s_per_day','journey_s_per_passenger'))):
    scale=60 if i==0 else 1
    ax.scatter(means['pressure']['cv_squared'],means['pressure'][metric]/scale,
        color=COLORS['pressure'],marker='*',s=85,label='Pressure credit',zorder=5)
    ax.set_xlabel('Actual arrival CV$^2$')
    ax.set_ylabel('Public delay (min/day)' if i==0 else 'Journey (s/passenger)')
    ax.text(-.15,1.03,chr(97+i),weight='bold',transform=ax.transAxes)
axes[0].legend(fontsize=6)
fig.tight_layout();save(fig,out,'independent_tradeoff')
write(out/'source_data.json',dict(component_means=means,frozen_mixture_means=mixtures,
    days=list(data['by_day']),weights=sealed['families'],selected_new_policy=False,
    geometry='Dashed lines span expected daily randomizations between the two evaluated whole-day components; filled points are the original frozen weights.'))
write(out/'results.json',dict(completed=True,passed=True,physical_starts=0,
    training_updates=0,statistical_resamples=0,new_policy_selection=False,old_gate=data['decisions']))
