"""Plot the already frozen retention comparison; no new resampling or selection."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import read,write
from archive_figures import style,save

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
a=p.parse_args()
d=read(a.root.parent.parent/'FreqDuet-v400-independent-confirmation/run01/aggregate/summary/results.json')
metrics=('journey_s_per_passenger','readiness_s_per_trip','holding_s_per_trip','cv_squared')
points={arm:{m:float(np.mean([day[arm][m] for day in d['by_day'].values()])) for m in metrics}
        for arm in ('manual_r050','manual_r025','K4_r050','K4_r025')}
style()
fig,axes=plt.subplots(1,3,figsize=(7.15,2.3))
for ax,metric,label in zip(axes,metrics[:3],
        ('Journey (s/person)','Readiness delay (s/trip)','Holding (s/trip)')):
    for family,color in (('manual','#287C8E'),('K4','#B77942')):
        lo,hi=points[family+'_r050'],points[family+'_r025']
        ax.plot([lo['cv_squared'],hi['cv_squared']],[lo[metric],hi[metric]],color=color,lw=.9,
                label='Manual joint' if family=='manual' else 'K4 permission')
        for row,marker,rho in ((lo,'o','.50'),(hi,'s','.25')):
            ax.scatter(row['cv_squared'],row[metric],color=color,s=23,marker=marker)
            ax.annotate(rho,(row['cv_squared'],row[metric]),xytext=(4,4),textcoords='offset points',fontsize=6)
    ax.set(xlabel='Actual arrival CV$^2$',ylabel=label)
axes[0].legend(loc='best',fontsize=6)
for i,ax in enumerate(axes):
    ax.text(-.17,1.03,chr(97+i),transform=ax.transAxes,weight='bold')
fig.tight_layout(w_pad=1.1)
out=a.root/'budget_figure/summary';out.mkdir(parents=True,exist_ok=True)
save(fig,out,'budget_retention')
write(out/'source_data.json',dict(points=points,days=d['days'],
    paired_effects={k:d['effects'][k] for k in ('manual_tight_minus_default','K4_tight_minus_default')},
    interpretation='Two prespecified retentions on the same 16 days, not a continuous optimized frontier.'))
write(out/'results.json',dict(completed=True,physical_starts=0,resamples=0,policy_selection=False))
