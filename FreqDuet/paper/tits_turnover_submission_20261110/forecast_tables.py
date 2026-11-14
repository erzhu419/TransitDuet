"""Format the archived running-speed, forecast, and control follow-ups."""

import argparse
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCES = {
    'v412': 'v412_running_speed_transfer/review/aggregate/results.json',
    'v414': 'v413_public_slot_observer/review/running_shadow_readout/results.json',
    'v415': 'v415_running_closed_loop/review/aggregate/results.json',
    'v415_day': 'v415_running_closed_loop/review/divergence/results.json',
    'v416': 'v416_dispatch_spacing/review/aggregate/results.json',
    'v417': 'v417_pooled_spacing/review/aggregate/results.json',
}
METRICS = (
    ('Journey', 'journey_s_per_passenger', 2),
    ('Readiness', 'readiness_s_per_trip', 2),
    ('Public delay', 'public_delay_s_per_day', 1),
    ('Arrival CV$^2$', 'cv_squared', 5),
    ('Dispatch CV$^2$', 'dispatch_cv_squared', 5),
)


def main(snapshot=False):
    data, manifest = {}, {}
    for key, source in SOURCES.items():
        source = 'experiments/' + source
        copy = HERE / 'evidence' / f'{key}.json'
        path = PROJECT / source
        if snapshot:
            copy.write_bytes(path.read_bytes())
        data[key] = json.loads(copy.read_text())
        if path.is_file():
            assert data[key] == json.loads(path.read_text()), source
        assert data[key]['complete'] and not data[key]['main_policy_changed']
        manifest[key] = dict(archive=source, bundled=str(copy.relative_to(HERE)))

    v412, v414, v415, v416, v417 = (data[k] for k in ('v412', 'v414', 'v415', 'v416', 'v417'))
    assert v412['regimes']['normal']['passed'] and not v412['regimes']['slow']['passed']
    slow = v412['regimes']['slow']
    assert all(slow['matching'].values())
    assert slow['outcomes'] == dict(journey_s_per_passenger=True, cv_squared=True,
                                   readiness_s_per_trip=False, public_delay_s_per_day=False)
    assert v414['primary_gate']['passed'] and v414['paid_confirmation_paths'] == 8
    assert not v415['passed'] and len(v415['checks']) == 4 and not any(v415['checks'].values())
    assert not v416['development_screen_passed'] and v416['physical_starts'] == 13
    assert [k for k, passed in v416['checks'].items() if not passed] == ['arrival_cv2_not_higher_than_either']
    assert v417['physical_starts'] == 4 and v417['reused_paths'] == 12 and not v417['v416_gate']
    assert v417['new_arm_by_day'].keys() == v416['by_day'].keys()
    cells = []

    def estimate(key, pointer, precision):
        row = data[key]
        for part in pointer.split('/'):
            row = row[part]
        lo, hi = row['ci95']
        cells.append(dict(study=key, pointer='/' + pointer, mean=row['mean'], interval=row['ci95']))
        return (r'\shortstack[r]{$' + f"{row['mean']:+.{precision}f}" + r'$\\{\scriptsize $['
                + f'{lo:+.{precision}f}, {hi:+.{precision}f}' + r']$}}')

    def table(filename, label, caption, rows, metrics):
        lines = [r'\begin{table}[htbp]', r'\centering\small', r'\caption{' + caption + '}',
                 r'\label{' + label + '}', r'\setlength{\tabcolsep}{3pt}',
                 r'\begin{tabular}{@{}l' + 'r'*len(metrics) + r'@{}}\toprule',
                 'Comparison & ' + ' & '.join(m[0] for m in metrics) + r' \\ \midrule']
        for label, source, pointer in rows:
            values = [estimate(source, f'{pointer}/{k}', precision) for _, k, precision in metrics]
            lines.append(label + ' & ' + ' & '.join(values) + r' \\[3pt]')
        lines += [r'\bottomrule\end{tabular}', r'\end{table}']
        (HERE / 'tables' / filename).write_text('\n'.join(lines) + '\n')

    table('running_transfer.tex', 'tab:running-transfer',
          'Running-speed transfer and the separate hourly-predictor deployment test. '
          'Entries are paired day means and archived 95\% intervals. Journey is in s/person, '
          'readiness in s/trip and public delay in s/day; CV$^2$ is unitless.', [
              ('V412 normal', 'v412', 'regimes/normal/effects/v406_cap_10_minus_transition'),
              ('V412 slower', 'v412', 'regimes/slow/effects/v406_cap_10_minus_transition'),
              ('V415 hourly minus pooled', 'v415', 'effects'),
          ], METRICS[:4])
    table('forecast_spacing.tex', 'tab:forecast-spacing',
          'Forecast-by-spacing development comparison. All contrasts use the same four days. '
          'Units match Table~\\ref{tab:running-transfer}; Dep. is departure. Intervals are descriptive.', [
              ('Hourly, no guard', 'v417', 'effects/hourly_unguarded'),
              ('Guard with pooled', 'v417', 'effects/guard_pooled'),
              ('Guard with hourly', 'v417', 'effects/guard_hourly'),
              ('Hourly, both guarded', 'v417', 'effects/hourly_guarded'),
              ('Interaction', 'v417', 'effects/interaction'),
              ('V416 vs. pooled', 'v416',
               'effects/v406_cap_10_hourly_spacing_minus_v406_cap_10'),
          ], (*METRICS[:4], ('Dep. CV$^2$', *METRICS[4][1:])))

    macros = []
    scalar_sources = {}

    def scalar(name, key, pointer, precision=0, magnitude=False):
        value = data[key]
        for part in pointer.split('/'):
            value = value[part.replace('~1', '/')]
        macros.append('\\newcommand{\\' + name + '}{' +
                      f'{abs(value) if magnitude else value:.{precision}f}' + '}')
        scalar_sources[name] = dict(study=key, pointer='/' + pointer, value=value)

    totals = 'totals/300~1scenario_0/launch/all'
    for arm, prefix in (('v413', 'Pooled'), ('v414', 'Hourly')):
        rows = [row['300/scenario_0']['launch']['all'][arm] for row in v414['by_day'].values()]
        mae = mean(row['absolute_s']/row['n'] for row in rows)
        macros.append(r'\newcommand{\Forecast' + prefix + 'MAE}{' + f'{mae:.2f}' + '}')
        scalar_sources['Forecast'+prefix+'MAE'] = dict(study='v414', value=mae,
            derivation=f'Mean across days of absolute_s/n at /by_day/<day>/300~1scenario_0/launch/all/{arm}')
        scalar('Forecast'+prefix+'Misses', 'v414', f'{totals}/{arm}/misses')
        scalar('Forecast'+prefix+'Alarms', 'v414', f'{totals}/{arm}/false_alarms')
    scalar('ForecastSlotPairs', 'v414', f'{totals}/v413/n')
    scalar('ForecastMAEGain', 'v414', 'primary_gate/mean_day_mae_delta_s', 2, magnitude=True)
    scalar('FocalLaunchComponent', 'v415_day', 'cv_parts_delta/launch', 5)
    scalar('FocalWaitingLoss', 'v415_day', 'passenger/official_wait_delta_s_per_person', 2)
    scalar('SpacingJourneyGain', 'v417', 'effects/guard_hourly/journey_s_per_passenger/mean', 2, magnitude=True)
    scalar('GuardedHourlyReadinessGain', 'v417', 'effects/hourly_guarded/readiness_s_per_trip/mean', 2, magnitude=True)
    scalar('GuardedHourlyArrivalLoss', 'v417', 'effects/hourly_guarded/cv_squared/mean', 5)
    (HERE/'forecast_values.tex').write_text('\n'.join(macros)+'\n')

    path = HERE/'evidence_manifest.json'
    index = json.loads(path.read_text())
    index['forecast_followup'] = dict(sources=manifest, cells=cells, scalars=scalar_sources,
        confirmed_forecast_only=True, v415_closed_loop_gate=False, v416_development_gate=False,
        v417_reuses_v416_days=True, local_statistical_resamples=0)
    path.write_text(json.dumps(index, indent=2)+'\n')
    print('FORECAST_EVIDENCE_OK', len(cells), 'archived estimates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true', help='copy compact archived summaries into the source package')
    main(parser.parse_args().snapshot)
