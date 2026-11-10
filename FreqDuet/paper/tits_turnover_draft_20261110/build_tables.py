"""Format archived estimates; never resample days or recompute an effect."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCES = {
    'v400': 'experiments/v400_independent_confirmation/review01/aggregate/results.json',
    'v401': 'experiments/v401_budget_conditioned_upper/review02/aggregate/results.json',
    'v401_readout': 'experiments/v401_budget_conditioned_upper/review02/readout01/results.json',
    'v402': 'experiments/v402_budget_mechanism/review/aggregate/results.json',
    'v402_selection': 'experiments/v402_budget_mechanism/review/selection/results.json',
    'v403': 'experiments/v403_joint_credit/review/aggregate/results.json',
}
DATA = {key: json.loads((HERE / 'evidence' / f'{key}.json').read_text()) for key in SOURCES}
CELLS = []


def number(value, precision):
    return f'{value:+.{precision}f}' if value else f'{0:.{precision}f}'


def estimate(study, contrast, metric, precision):
    value = DATA[study]['effects'][contrast][metric]
    interval_key = 'descriptive_ci95' if study == 'v403' else 'ci95'
    CELLS.append(dict(study=study, pointer=f'/effects/{contrast}/{metric}',
                      estimate=value['mean'], interval_key=interval_key,
                      interval=value[interval_key]))
    low, high = value[interval_key]
    return (r'\shortstack[r]{$' + number(value['mean'], precision) + r'$\\'
            r'{\scriptsize $[' + number(low, precision) + ',' + number(high, precision) + ']$}}')


def table(study, filename, label, caption, rows, metrics):
    lines = [r'\begin{table*}[t]', r'\centering', r'\footnotesize',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabularx}{\textwidth}{@{}l' + 'Y' * len(metrics) + '@{}}',
             r'\toprule', 'Comparison & ' + ' & '.join(m[0] for m in metrics) + r' \\', r'\midrule']
    for name, contrast in rows:
        lines.append(name + ' & ' + ' & '.join(estimate(study, contrast, key, precision)
                                               for _, key, precision in metrics) + r' \\[3pt]')
    lines.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table*}'])
    (HERE / 'tables' / filename).write_text('\n'.join(lines) + '\n')


def main():
    originals = {}
    for key, path in SOURCES.items():
        source = PROJECT / path
        originals[key] = source.exists()
        if source.exists():
            assert DATA[key] == json.loads(source.read_text()), path
    assert DATA['v400']['primary_decisions'] == dict(budget_service_improvement=True,
                                                   harmonic_journey_improvement=False)
    assert len(DATA['v400']['days']) == len(DATA['v402']['by_day']) == 16
    assert len(DATA['v403']['by_day']) == 4
    assert DATA['v401']['screen_passed'] is DATA['v402']['passed'] is False
    assert DATA['v403']['development_screen_passed'] is False
    learned = [r for r in DATA['v401_readout']['development'] if r['arm'] in ('explicit', 'implicit')]
    assert len(learned) == 16
    assert all(r['mode_counts']['joint'] == r['publications'] == 47 for r in learned)
    for study in ('v400', 'v401', 'v402', 'v403'):
        assert DATA[study]['main_policy_changed'] is False
    (HERE / 'tables').mkdir(exist_ok=True)
    table('v400', 'budget.tex', 'tab:budget',
          'Independent budget study: tighter retention ($.25-.50$), 16 paired demand days. '
          'Estimates and archived 95\% day-bootstrap intervals. Negative means lower.',
          [('Learned supervisor', 'K4_tight_minus_default'), ('Manual joint', 'manual_tight_minus_default')],
          [('Journey (s/person)', 'journey_s_per_passenger', 3),
           ('Readiness (s/trip)', 'readiness_s_per_trip', 3),
           ('Public delay (s/day)', 'public_delay_s_per_day', 2), ('Actual CV$^2$', 'cv_squared', 6)])
    table('v400', 'supervisor.tex', 'tab:supervisor',
          'Independent representation and supervisor comparisons, 16 paired demand days. '
          'K4 retains harmonics; K0 retains trend, online adaptation and innovation statistics. '
          'Intervals crossing zero are not equivalence tests.',
          [('K4$-$manual, $\\rho_B=.50$', 'K4_minus_manual_r050'),
           ('K4$-$manual, $\\rho_B=.25$', 'K4_minus_manual_r025'),
           ('K4$-$K0, $\\rho_B=.50$', 'K4_minus_K0_r050')],
          [('Journey (s/person)', 'journey_s_per_passenger', 3),
           ('Readiness (s/trip)', 'readiness_s_per_trip', 3), ('Actual CV$^2$', 'cv_squared', 6)])
    table('v402', 'simple_limits.tex', 'tab:limits',
          'Pressure credit minus frozen simple-limit mixtures on 16 new paired demand days. '
          'Weights matched development holding exposure, not test-day holding or regularity. '
          'Both delay tests pass and both CV$^2$ protections fail.',
          [('Static trip credit', 'static'), ('Per-stop cap', 'cap')],
          [('Public delay (s/day)', 'public_delay_s_per_day', 2), ('Actual CV$^2$', 'cv_squared', 6),
           ('Journey (s/person)', 'journey_s_per_passenger', 3),
           ('Readiness (s/trip)', 'readiness_s_per_trip', 3)])
    table('v403', 'joint_credit.tex', 'tab:joint',
          'Common-reference joint-credit candidate minus each comparator on four NEW development days. '
          'Intervals are descriptive. This is not independent confirmation. '
          'Both simple-control delay screens fail; both mean CV$^2$ protections pass.',
          [('Static mixture', 'static'), ('Stop-cap mixture', 'cap'), ('Old pressure', 'pressure')],
          [('Public delay (s/day)', 'public_delay_s_per_day', 2), ('Actual CV$^2$', 'cv_squared', 6),
           ('Journey (s/person)', 'journey_s_per_passenger', 3),
           ('Holding (s/trip)', 'holding_s_per_trip', 3)])
    macros = {
        'ManualBudgetGain': ('v400', 'manual_tight_minus_default', 'journey_s_per_passenger', 3, True),
        'ManualReadinessGain': ('v400', 'manual_tight_minus_default', 'readiness_s_per_trip', 3, True),
        'ManualBudgetCV': ('v400', 'manual_tight_minus_default', 'cv_squared', 6, False),
        'LearnedBudgetGain': ('v400', 'K4_tight_minus_default', 'journey_s_per_passenger', 3, True),
        'HarmonicJourney': ('v400', 'K4_minus_K0_r050', 'journey_s_per_passenger', 3, False),
        'HarmonicCV': ('v400', 'K4_minus_K0_r050', 'cv_squared', 6, False),
        'PressureStaticGain': ('v402', 'static', 'public_delay_s_per_day', 2, True),
        'PressureCapGain': ('v402', 'cap', 'public_delay_s_per_day', 2, True),
        'JointStaticLoss': ('v403', 'static', 'journey_s_per_passenger', 3, True),
        'JointCapLoss': ('v403', 'cap', 'journey_s_per_passenger', 3, True),
    }
    text = []
    for name, (study, contrast, metric, precision, magnitude) in macros.items():
        value = DATA[study]['effects'][contrast][metric]['mean']
        rendered = f'{abs(value):.{precision}f}' if magnitude else number(value, precision)
        text.append('\\newcommand{\\' + name + '}{' + rendered + '}')
    (HERE / 'numeric_values.tex').write_text('\n'.join(text) + '\n')
    manifest = dict(sources=SOURCES, cells=CELLS,
                    protocol_sources={key: path.split('/review')[0] + '/PROTOCOL.md'
                                      for key, path in SOURCES.items() if key in ('v400','v401','v402','v403')},
                    entrypoints=dict(v400='experiments/v400_independent_confirmation/evaluation.py',
                                     v401='experiments/v401_budget_conditioned_upper/worker.py',
                                     v402='experiments/v402_budget_mechanism/worker.py',
                                     v403='experiments/v403_joint_credit/worker.py'),
                    runtime_assets='experiments/v400_independent_confirmation/deployment.json',
                    original_supervisor='K4/V326, seed314011, batch8; K0 from matched V350',
                    units=dict(journey_s_per_passenger='restricted seconds per generated passenger',
                               readiness_s_per_trip='seconds per scheduled trip',
                               public_delay_s_per_day='positive lateness seconds per day',
                               cv_squared='dimensionless actual nonterminal arrival CV squared'),
                    study_units=dict(v400='16 independent demand days, one frozen learner seed',
                                     v401='4 development days, two matched learners',
                                     v402='16 independent demand days, fixed daily-mixture expectations',
                                     v403='4 development days, fixed daily-mixture expectations'),
                    original_copies_equal_when_available=originals,
                    formatting_only=True, new_statistical_resamples=0, new_traffic_paths=0)
    (HERE / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (HERE / 'evidence_check.json').write_text(json.dumps(dict(
        passed=True, estimate_cells=len(CELLS), original_copies_checked=sum(originals.values()),
        frozen_gate_values_preserved=True, new_statistical_resamples=0, new_traffic_paths=0), indent=2) + '\n')
    print('PAPER_EVIDENCE_OK', len(CELLS), 'archived estimates; no statistical recomputation')


if __name__ == '__main__':
    main()
