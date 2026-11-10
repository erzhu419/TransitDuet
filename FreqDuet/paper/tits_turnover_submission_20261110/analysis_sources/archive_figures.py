"""Read old compact evidence on the server and export publication figure assets."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study import read, write

COLORS = dict(pressure='#287C8E', static='#59616A', cap='#B77942', candidate='#8A668B')
LABELS = dict(pressure='Pressure credit', static='Static trip credit', cap='Stop cap',
              candidate='Common-reference credit')


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 8, 'legend.fontsize': 7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .7})


def save(fig, dest, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(dest / (name + '.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)


def archive(args):
    style()
    base = args.root.parent.parent
    root402 = base / 'FreqDuet-v402-budget-mechanism/run01'
    selection = read(root402 / 'selection/summary/results.json')
    old = read(root402 / 'aggregate/summary/results.json')
    last = read(base / 'FreqDuet-v403-joint-credit/run01/aggregate/summary/results.json')
    out = args.root / 'archive/summary'
    out.mkdir(parents=True, exist_ok=True)
    metrics = ('journey_s_per_passenger', 'public_delay_s_per_day', 'cv_squared', 'holding_s_per_trip')
    calibration = selection['calibration']
    arms = list(next(iter(calibration.values())))
    means = {a: {m: float(np.mean([d[a][m] for d in calibration.values()])) for m in metrics}
             for a in arms}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.2))
    for family in ('static', 'cap'):
        points = sorted(selection['families'][family]['cases'], key=lambda r: r['parameter'])
        for ax, y in zip(axes, ('public_delay_s_per_day', 'journey_s_per_passenger', 'holding_s_per_trip')):
            x = [means[p['arm']]['cv_squared'] for p in points]
            v = [means[p['arm']][y] / (60 if y == 'public_delay_s_per_day' else 1) for p in points]
            ax.plot(x, v, marker='o', ms=3.5, lw=1, color=COLORS[family], label=LABELS[family])
    for i, (ax, y, label) in enumerate(zip(axes, metrics[:2] + ('holding_s_per_trip',),
            ('Dispatch delay (min/day)', 'Journey (s/passenger)', 'Holding (s/trip)'))):
        # Explicit coordinates avoid mixing the plotted endpoint order.
        endpoint = ('public_delay_s_per_day', 'journey_s_per_passenger', 'holding_s_per_trip')[i]
        value = means['pressure'][endpoint] / (60 if i == 0 else 1)
        ax.scatter(means['pressure']['cv_squared'], value, marker='*', s=75,
                   color=COLORS['pressure'], zorder=5, label=LABELS['pressure'])
        ax.set(xlabel='Actual arrival CV$^2$', ylabel=label)
        ax.text(-.2, 1.04, chr(97+i), transform=ax.transAxes, weight='bold')
    axes[0].legend(loc='best')
    fig.tight_layout(w_pad=1.2)
    save(fig, out, 'calibration_tradeoff')
    # Independent points are not connected into a claimed optimal frontier.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    independent = {a: {m: float(np.mean([d[a][m] for d in old['by_day'].values()]))
                       for m in metrics} for a in next(iter(old['by_day'].values()))}
    points = {'pressure': independent['pressure']}
    for family in ('static', 'cap'):
        points[family] = {m: sum(c['probability'] * independent[c['arm']][m]
            for c in selection['families'][family]['components']) for m in metrics}
    for ax, metric in zip(axes, ('public_delay_s_per_day', 'journey_s_per_passenger')):
        for family, p in points.items():
            ax.scatter(p['cv_squared'], p[metric] / (60 if metric.startswith('public') else 1),
                color=COLORS[family], s=50, marker={'pressure': '*', 'static': 's', 'cap': 'o'}[family],
                label=LABELS[family])
        ax.set_xlabel('Actual arrival CV$^2$')
        ax.set_ylabel('Dispatch delay (min/day)' if metric.startswith('public') else 'Journey (s/passenger)')
    axes[0].legend(loc='best')
    for i, ax in enumerate(axes):
        ax.text(-.15, 1.03, chr(97+i), transform=ax.transAxes, weight='bold')
    fig.tight_layout()
    save(fig, out, 'independent_tradeoff')
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.3))
    labels = ('Static', 'Stop cap')
    for i, (ax, metric, unit, scale) in enumerate(zip(axes,
        ('journey_s_per_passenger', 'public_delay_s_per_day', 'cv_squared'),
        ('Journey difference (s/person)', 'Dispatch difference (min/day)', 'Actual CV$^2$ difference'),
        (1, 60, 1))):
        for j, study in enumerate((old, last)):
            for k, family in enumerate(('static', 'cap')):
                effect = study['effects'][family][metric]
                ci = effect.get('ci95', effect.get('descriptive_ci95'))
                mean = effect['mean'] / scale
                ax.errorbar(k + (j-.5)*.16, mean, yerr=[[mean-ci[0]/scale], [ci[1]/scale-mean]],
                    fmt='o' if j == 0 else 's', color=COLORS['pressure' if j == 0 else 'candidate'],
                    ms=4, capsize=2, label=('Pressure, 16 new days' if j == 0 else
                    'Common reference, 4 development days') if k == 0 else None)
        ax.axhline(0, color='.7', lw=.7)
        ax.set(xticks=[0, 1], xticklabels=labels, ylabel=unit)
        ax.text(-.22, 1.04, chr(97+i), transform=ax.transAxes, weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2,
               bbox_to_anchor=(.5, 1.12), fontsize=7)
    fig.tight_layout()
    save(fig, out, 'reference_tradeoff')
    write(out / 'source_data.json', dict(calibration_days=list(calibration),
        calibration_means=means, independent_days=list(old['by_day']), independent_points=points,
        old_gates=old['decisions'], common_reference_gate=last['decisions'],
        no_test_reweighting=True, no_new_traffic=True))
    write(out / 'results.json', dict(completed=True, passed=True, physical_starts=0,
        figures=['calibration_tradeoff', 'independent_tradeoff', 'reference_tradeoff'],
        calibration_paths=64, independent_paths=80, statistical_resamples=0,
        note='Calibration curves describe a finite grid; independent frozen points are not an optimal frontier.'))


def supplement(args):
    style()
    data = read(args.root / 'aggregate/summary/results.json')
    out = args.root / 'figures/summary'
    out.mkdir(parents=True, exist_ok=True)
    regimes = list(data['effects'])
    contrasts = ('joint_vs_bilateral', 'pressure_vs_static', 'pressure_vs_cap')
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.1))
    for ax, metric, label, scale in zip(axes,
        ('journey_s_per_passenger', 'public_delay_s_per_day', 'cv_squared'),
        ('Journey difference (s/person)', 'Dispatch difference (min/day)', 'Actual CV$^2$ difference'), (1,60,1)):
        for j, contrast in enumerate(contrasts):
            for k, regime in enumerate(regimes):
                e = data['effects'][regime][contrast][metric]
                mean = e['mean']/scale; lo,hi = np.array(e['descriptive_ci95'])/scale
                ax.errorbar(mean, k+(j-1)*.19, xerr=[[mean-lo],[hi-mean]], fmt=('o','s','^')[j],
                    color=('#287C8E','#59616A','#B77942')[j], capsize=2, ms=3,
                    label=('vs bilateral','vs static','vs stop cap')[j] if k == 0 else None)
        ax.axvline(0,color='.7',lw=.7)
        ax.set(xlabel=label, yticks=range(4), yticklabels=regimes)
        ax.invert_yaxis()
    axes[0].legend(loc='best',fontsize=6)
    fig.tight_layout()
    save(fig,out,'regime_robustness')
    fig, axes = plt.subplots(1,3,figsize=(7.15,2.6))
    for ax,metric,label,scale in zip(axes,
        ('journey_s_per_passenger','public_delay_s_per_day','cv_squared'),
        ('Journey (s/person)','Dispatch (min/day)','Actual CV$^2$'),(1,60,1)):
        for arm,color,marker in (('fixed_model','#777777','o'),('plan','#B77942','s'),
                                 ('budget','#668A73','^'),('pressure','#287C8E','D')):
            ax.plot(range(4),[data['means'][r][arm][metric]/scale for r in regimes],
                color=color,marker=marker,ms=3,lw=1,label=arm.replace('_model','').replace('pressure','joint'))
        ax.set(ylabel=label,xticks=range(4),xticklabels=regimes)
        ax.tick_params(axis='x',labelrotation=25)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig,out,'factorial_coordination')
    write(out/'results.json',dict(completed=True,passed=True,
        figures=['regime_robustness','factorial_coordination'],resamples_performed_here=0))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('command',choices=('archive','supplement'))
    a=p.parse_args();globals()[a.command](a)
