"""Format server-computed supplement estimates without local statistical work."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cell(effect, precision, scale=1):
    mean = effect['mean'] / scale
    lo, hi = [x / scale for x in effect['descriptive_ci95']]
    return (r'\shortstack[r]{$' + f'{mean:+.{precision}f}' + r'$\\{\scriptsize $[' +
            f'{lo:+.{precision}f},{hi:+.{precision}f}' + r']$}}')


def main():
    data = json.loads((HERE / 'evidence/v404.json').read_text())
    assert data['completed'] and data['historical_gates_unchanged']
    assert data['physical_pairing_checks'] == 144 and data['physical_starts'] == 146
    extra=json.loads((HERE/'evidence/v404_extra.json').read_text())
    assert extra['charged_start_account']==dict(formal=144,engineering=2,total=146,
        failed_observation_capture_retained=True)
    metrics = [('journey_s_per_passenger', 3, 1), ('public_delay_s_per_day', 2, 60),
               ('readiness_s_per_trip',2,1),('cv_squared', 6, 1)]
    component_metrics=[v for v in metrics if v[0]!='readiness_s_per_trip']+[
        ('holding_s_per_trip',2,1)]
    rows = [r'\begin{table*}[t]', r'\centering\footnotesize',
        r'\caption{Conditional timetable and budget effects on four new paired demand days per regime. '
        r'Timetable means joint minus budget-only; budget means joint minus timetable-only. '
        r'Day-bootstrap intervals are descriptive, not a replacement for the original confirmation gates.}',
        r'\label{tab:factorial}', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabularx}{\textwidth}{@{}llYYYY@{}}\toprule',
        r'Regime & Added component & Journey (s/person) & Public delay (min/day) & Readiness (s/trip) & Actual CV$^2$ \\ \midrule']
    cells = []
    for regime, contrasts in data['effects'].items():
        for name, label in (('plan_at_budget', 'Timetable given budget'),
                            ('budget_at_plan', 'Budget given timetable')):
            rows.append(regime + ' & ' + label + ' & ' + ' & '.join(
                cell(contrasts[name][m], p, s) for m, p, s in metrics) + r' \\[2pt]')
    rows += [r'\bottomrule\end{tabularx}', r'\end{table*}']
    (HERE / 'tables/factorial.tex').write_text('\n'.join(rows) + '\n')
    detail = []
    for regime, contrasts in data['effects'].items():
        detail += [r'\subsection{' + regime.capitalize() + r'}', r'\begin{center}\small',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Contrast & Journey (s/person) & Delay (min/day) & Readiness (s/trip) & Actual CV$^2$ \\ \midrule']
        for name, values in contrasts.items():
            detail.append(name.replace('_', r'\_') + ' & ' + ' & '.join(
                cell(values[m], p, s) for m, p, s in metrics) + r' \\[3pt]')
            cells.extend(dict(source='evidence/v404.json', pointer=f'/effects/{regime}/{name}/{m}',
                estimate=values[m]['mean'], interval=values[m]['descriptive_ci95'],
                displayed_divisor=s) for m,p,s in metrics)
        detail += [r'\bottomrule\end{tabular}\end{center}']
    (HERE / 'tables/factorial_all.tex').write_text('\n'.join(detail) + '\n')
    historical = json.loads((HERE / 'evidence/components_historical.json').read_text())
    fresh = json.loads((HERE / 'evidence/components_fresh.json').read_text())
    component_rows = []
    for label, panel, source, prefix in [('Original 16-day panel', historical,
            'evidence/components_historical.json',''), *(
            (regime.capitalize() + ' supplement', panel,'evidence/components_fresh.json',f'/panels/{regime}')
            for regime, panel in fresh['panels'].items())]:
        component_rows += [r'\subsection{' + label + '}', r'\begin{center}\small',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Pressure minus & Journey (s/person) & Delay (min/day) & Actual CV$^2$ & H (s/trip) \\ \midrule']
        for arm, values in panel['effects'].items():
            component_rows.append(arm.replace('_',r'\_') + ' & ' + ' & '.join(
                cell(values[m],p,s) for m,p,s in component_metrics) + r' \\[3pt]')
            cells.extend(dict(source=source,pointer=prefix+f'/effects/{arm}/{m}',
                estimate=values[m]['mean'],interval=values[m]['descriptive_ci95'],
                displayed_divisor=s) for m,p,s in component_metrics)
        component_rows += [r'\bottomrule\end{tabular}\end{center}']
    (HERE/'tables/components_all.tex').write_text('\n'.join(component_rows)+'\n')
    check = dict(passed=True, source='evidence/v404.json',
                 mean_interval_cells=32, distinct_factorial_and_component_cells=len(cells),
                 statistical_resamples=0, local_traffic=0)
    (HERE / 'supplement_evidence_check.json').write_text(json.dumps(check, indent=2) + '\n')
    manifest=json.loads((HERE/'evidence_manifest.json').read_text())
    manifest['supplemental_sources']=dict(v404='experiments/v404_publication_closure/review/aggregate/results.json',
        service='experiments/v404_publication_closure/review/extra/results.json',
        figures='experiments/v404_publication_closure/archive_figures.py')
    manifest['supplemental_cells']=cells
    manifest['figure_source_data']=dict(archive='figures/source_data.json',
        retention='figures/retention_source_data.json',
        components='figures/component_source_data.json',cohorts='figures/cohort_source_data.json',
        mechanism='figures/vehicle_source_data.json', operating_conditions='evidence/v404.json')
    manifest['runtime']=dict(source_inventory='evidence/runtime_source_manifest.json',
        server_bundle='evidence/runtime_packaging.json',environment='evidence/runtime_environment.json',
        sealed_release='evidence/runtime_release.json',
        installable_version_pins='requirements-runtime.txt',
        source_snapshot='FreqDuet/runtime_release/v404',clean_reinstall_tested=False)
    manifest['aggregation_acceptance']='evidence/v404_aggregation_acceptance.json'
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
