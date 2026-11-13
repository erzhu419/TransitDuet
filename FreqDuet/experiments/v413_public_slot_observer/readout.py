"""Frozen V413 forecast readout; scans only the eight paid server archives."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_confirmation', HERE.parent / 'confirmation01/confirmation.py')
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)
run = confirmation.frozen_run
PLAN, read, write = run.PLAN, run.read, run.write


def clock_score(predicted_late, actual_late, observed_ready):
    return dict(n=1, actual_late=int(actual_late), predicted_late=int(predicted_late),
        correct=int(predicted_late == actual_late),
        misses=int(actual_late and not predicted_late), false_alarms=int(predicted_late and not actual_late),
        ready_only_correct=int((not observed_ready) == actual_late),
        ready_only_misses=int(actual_late and observed_ready),
        ready_only_false_alarms=int(not actual_late and not observed_ready))


def return_order(predictions, actual, gap_s):
    estimated_first = min(predictions.values())
    realized_first = min(actual.values())
    predicted = {b for b, t in predictions.items() if t == estimated_first}
    realized = {b for b, t in actual.items() if t == realized_first}
    clocks = sorted(predictions.values())
    band = 'single' if len(clocks) == 1 else ('gap_le_30' if clocks[1]-clocks[0] <= gap_s else 'gap_gt_30')
    return band, dict(n=1, wrong=int(not predicted & realized), predicted_tie=int(len(predicted) > 1),
                     actual_tie=int(len(realized) > 1))


def assess(trace, slots):
    dispatch = {r['trip']: r for r in trace['dispatches']}
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    groups, examples = {}, []
    for slot in slots:
        lead, trip, due = str(slot['lead_s']), slot['trip'], slot['public_s']
        group = groups.setdefault(lead, dict(coverage=Counter(), scenarios={}))
        coverage = group['coverage']
        coverage.update(snapshots=1, already_ready=int(bool(slot['ready'])),
                        earlier_queued=int(bool(slot['earlier'])))
        if not slot['scenarios']:
            coverage['unsupported'] += 1
            continue
        coverage['with_model'] += 1
        cohort = slot['cohort']
        actual = {bus: returns[bus, active] for bus, active in cohort.items()}
        assert all(t >= slot['query_s'] for t in actual.values())
        for model in slot['scenarios']:
            output = group['scenarios'].setdefault(str(model['scenario']), dict(
                coverage=Counter(), launch=Counter(), launch_no_earlier=Counter(),
                launch_earlier_queued=Counter(), launch_errors=Counter(),
                rank={k:Counter() for k in ('all', 'gap_le_30', 'gap_gt_30', 'single')},
                first_return_clock=Counter()))
            output['coverage']['snapshots'] += 1
            if model['failure']:
                output['coverage']['model_failed'] += 1
                continue
            forecast = model['launches'].get(trip)
            if forecast is None:
                output['coverage']['target_launch_missing'] += 1
            else:
                estimate, _ = forecast
                event = clock_score(estimate > due, dispatch[trip]['actual_s'] > due, bool(slot['ready']))
                for k in ('launch', 'launch_earlier_queued' if slot['earlier'] else 'launch_no_earlier'):
                    output[k].update(event)
                error = estimate-dispatch[trip]['actual_s']
                output['launch_errors'].update(n=1, signed_s=error, absolute_s=abs(error))
            if slot['earlier']:
                output['coverage']['rank_excluded_earlier'] += 1
                continue
            if slot['ready']:
                output['coverage']['rank_excluded_ready'] += 1
                continue
            if not actual or not actual.keys() <= model['returns'].keys():
                output['coverage']['rank_incomplete_cohort'] += 1
                continue
            predicted = {bus: model['returns'][bus] for bus in actual}
            band, row = return_order(predicted, actual, PLAN['near_tie_s'])
            output['rank']['all'].update(row)
            output['rank'][band].update(row)
            clock = clock_score(min(predicted.values()) >= due, min(actual.values()) >= due, False)
            output['first_return_clock'].update(clock)
            if row['wrong'] and model['scenario'] == 0 and len(examples) < 3:
                examples.append(dict(trip=trip, lead_s=int(lead), query_s=slot['query_s'], band=band,
                    predicted_returns=predicted, actual_returns=actual))
    return dict(leads=groups, rank_error_examples=examples)


def add_totals(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            add_totals(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def synthetic_checks():
    missed = clock_score(False, True, False)
    assert missed['misses'] == 1 and missed['ready_only_correct'] == 1
    alarm = clock_score(True, False, True)
    assert alarm['false_alarms'] == 1 and alarm['ready_only_correct'] == 1
    band, score = return_order({1:100, 2:120}, {1:150, 2:110}, 30)
    assert band == 'gap_le_30' and score['wrong'] == 1
    band, score = return_order({1:100, 2:100}, {1:150, 2:110}, 30)
    assert score['predicted_tie'] == 1 and score['wrong'] == 0
    trace = dict(dispatches=[dict(trip=7,actual_s=110)],
                 returns=[dict(bus_id=1,trip_id=4,physical_time_s=120),
                          dict(bus_id=2,trip_id=5,physical_time_s=105)])
    slot = dict(trip=7, lead_s=300, query_s=0, public_s=100, ready=[], earlier=[6], cohort={1:4,2:5},
        scenarios=[dict(scenario=0,failure=None,launches={7:[110,2]},returns={1:120,2:105})])
    score = assess(trace, [slot])['leads']['300']['scenarios']['0']
    assert score['launch_earlier_queued']['n'] == 1 and not score['rank']['all']


def readout(root):
    synthetic_checks()
    archive = run.old.frozen.helper(run.old.frozen.ASSETS, 'gate_default')
    by_day, totals, audits, walls = {}, {}, {}, []
    for day in PLAN['confirmation_days']:
        folder = root / 'confirmation' / str(day)
        receipt = read(folder / 'summary/results.json')
        assert receipt['completed'] and receipt['day'] == day and receipt['physical_starts'] == 1
        assert not receipt['training_updates'] and not receipt['main_policy_changed']
        case = folder / 'cases' / PLAN['regime'] / str(day) / PLAN['arm']
        trace = archive.raw(case / 'raw/path.pkl.gz')
        slots = read(case / 'raw/public_slots.json')
        for slot in slots:
            slot['cohort'] = {int(k):v for k,v in slot['cohort'].items()}
            for model in slot['scenarios']:
                for key in ('returns', 'launches'):
                    model[key] = {int(k):v for k,v in model[key].items()}
        audit = confirmation.slot_audit(trace, slots)
        assert audit == receipt['slots']
        assert len(trace['queries']) == len(trace['graph_queries']) == len(trace['uniforms']) == 5200
        score = assess(trace, slots)
        by_day[day] = score
        add_totals(totals, score['leads'])
        audits[day] = dict(schedule_exact=audit['schedule_exact'], cohort_labels_complete=True,
            snapshot_counts={k:v['snapshots'] for k,v in audit['leads'].items()})
        walls.append(receipt['wall_s'])
    daily_effects = {}
    for lead, total in totals.items():
        primary = total['scenarios']['0']
        for subset in ('launch', 'launch_no_earlier', 'launch_earlier_queued'):
            differences = {}
            for day, row in by_day.items():
                counts = row['leads'][lead]['scenarios']['0'][subset]
                if counts.get('n'):
                    differences[day] = (counts['correct']-counts['ready_only_correct'])/counts['n']
            daily_effects[f'{lead}/{subset}'] = dict(by_day=differences,
                mean_accuracy_delta=sum(differences.values())/len(differences),
                days_improved=sum(v>0 for v in differences.values()), days_worse=sum(v<0 for v in differences.values()))
        assert primary['launch']['n'] == total['coverage']['with_model']
    write(root / 'readout/summary/results.json', dict(complete=True, physical_starts=0,
        paid_confirmation_paths=len(by_day), training_updates=0, main_policy_changed=False,
        audits=audits, totals=totals, by_day=by_day, daily_effects=daily_effects,
        wall_min_s=min(walls), wall_max_s=max(walls),
        interpretation='Forecast validation only. Counts describe slots; independent units are eight demand days.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    readout(p.parse_args().root)
