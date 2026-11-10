"""Re-export the saved cohort estimates with readable publication margins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--assets')
    args = parser.parse_args()
    source = args.root / 'extra/summary/cohort_source_data.json'
    cohorts = json.loads(source.read_text())['cohorts']['nominal']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax, axis, label in zip(axes, ('generation_hour', 'origin_station'),
            ('Generation hour since service start', 'Recorded origin station ID')):
        for family, marker, color, name in (
                ('bilateral', 'o', '#258194', 'vs bilateral'),
                ('static', 's', '#59616b', 'vs static'),
                ('cap', '^', '#bd7a3d', 'vs cap')):
            values = cohorts[family][axis]
            keys = sorted(map(int, values))
            ax.plot(keys, [values[str(k)]['timestamp_journey_difference_s_per_person']
                for k in keys], marker=marker, color=color, label=name, lw=1.2, ms=4)
        ax.axhline(0, color='0.65', lw=.8)
        ax.set(xlabel=label, ylabel='Journey difference (s/person)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.96, wspace=.31)
    out = args.root / 'extra/summary'
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'cohort_exposure.{extension}', dpi=180)
    plt.close(fig)
    (args.root/'figure_layout/summary').mkdir(parents=True, exist_ok=True)
    (args.root/'figure_layout/summary/results.json').write_text(json.dumps(
        dict(completed=True, estimates_changed=False, physical_starts=0,
             statistical_resamples=0))+'\n')


if __name__ == '__main__':
    main()
