"""Explain the 81 frozen advance forecast errors from existing V413 archives."""

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_readout', HERE.parent / 'readout01/readout.py')
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
run, PLAN, read, write = readout.run, readout.PLAN, readout.read, readout.write


def supply_clock(slot, returns):
    """Diagnostic first-use supply bound, preserving all earlier obligations."""
    stock = [(slot['query_s'], b) for b in slot['ready']]
    stock += [(returns[b]+1, b) for b in slot['cohort']]
    stock.sort()
    index = len(slot['earlier'])
    if index >= len(stock):
        return None
    clock, bus = stock[index]
    return dict(launch_s=max(slot['public_s'], clock), bus=bus,
                ordinal=index+1, first_supply_s=stock[0][0], marginal_supply_s=clock,
                supplies_by_public=sum(t <= slot['public_s'] for t, _ in stock))


def ledger(trace):
    launches = {r['trip']: r for r in trace['dispatches']}
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    previous, predecessors = {}, {}
    for r in sorted(launches.values(), key=lambda v: (v['actual_s'], v['trip'])):
        predecessors[r['trip']] = previous.get(r['bus'])
        previous[r['bus']] = r['trip']
    return launches, returns, predecessors


def explain(slot, model, launches, returns, predecessors):
    trip, due, now = slot['trip'], slot['public_s'], slot['query_s']
    predicted_launch, predicted_bus = model['launches'][trip]
    actual = launches[trip]
    kind = 'miss' if predicted_launch <= due else 'false_alarm'
    assert (predicted_launch > due) != (actual['actual_s'] > due)
    assert actual['public_s'] == actual['scheduled_s'] == actual['eligible_s'] == due
    real_returns = {b:returns[b, t] for b, t in slot['cohort'].items()}
    complete = slot['cohort'].keys() <= model['returns'].keys()
    predicted_returns = {b:model['returns'][b] for b in slot['cohort']} if complete else {}
    real_stock = supply_clock(slot, real_returns)
    predicted_stock = supply_clock(slot, predicted_returns) if complete else None
    stats = dict(n=1, queued=int(bool(slot['earlier'])), already_ready=int(bool(slot['ready'])),
        assigned_bus_changed=int(predicted_bus != actual['bus']),
        actual_delay_s=actual['actual_s']-due, predicted_delay_s=predicted_launch-due,
        absolute_clock_error_s=abs(predicted_launch-actual['actual_s']))
    row = dict(trip=trip, query_s=now, public_s=due, kind=kind, earlier=slot['earlier'],
        ready=slot['ready'], cohort=slot['cohort'], predicted_launch_s=predicted_launch,
        actual_launch_s=actual['actual_s'], predicted_bus=predicted_bus, actual_bus=actual['bus'],
        actual_predecessor=predecessors[trip],
        predicted_stock=predicted_stock, actual_stock=real_stock)
    if predicted_stock and real_stock:
        pe = predicted_launch-predicted_stock['launch_s']
        ae = actual['actual_s']-real_stock['launch_s']
        matches = pe == 0 and ae == 0
        stats.update(stock_pair=1, model_stock_exact=int(pe == 0), actual_stock_exact=int(ae == 0),
            both_stock_exact=int(matches),
            actual_first_ready_but_queue_late=int(real_stock['first_supply_s'] <= due < real_stock['launch_s']),
            model_first_ready_but_queue_late=int(predicted_stock['first_supply_s'] <= due < predicted_stock['launch_s']))
        row['stock_residuals'] = dict(predicted_s=pe, actual_s=ae,
            marginal_clock_error_s=predicted_stock['launch_s']-real_stock['launch_s'])
        assert predicted_launch-actual['actual_s'] == (
            row['stock_residuals']['marginal_clock_error_s'] + pe-ae)
    bus = actual['bus']
    active_trip = slot['cohort'].get(bus)
    first_use = not any(b == bus and t < predicted_launch
        for other, (t, b) in model['launches'].items() if other != trip)
    matched = (predicted_bus == bus and active_trip is not None
               and active_trip == predecessors[trip] and first_use
               and bus in model['returns'])
    stats['same_current_cycle_decomposable'] = int(matched)
    if matched:
        pr, ar = model['returns'][bus], real_returns[bus]
        error = predicted_launch-actual['actual_s']
        return_error = pr-ar
        offset_error = (predicted_launch-pr)-(actual['actual_s']-ar)
        assert error == return_error+offset_error
        row['same_cycle'] = dict(trip=active_trip, predicted_return_s=pr, actual_return_s=ar,
            launch_error_s=error, return_error_s=return_error, post_return_offset_error_s=offset_error,
            actual_after_return_s=actual['actual_s']-ar, predicted_after_return_s=predicted_launch-pr)
        stats.update(return_error_s=return_error, post_return_offset_error_s=offset_error,
            actual_return_binding=int(actual['actual_s'] == ar+1),
            model_return_binding=int(predicted_launch == pr+1))
    return row, stats


def checks():
    slot = dict(trip=7, query_s=0, public_s=300, ready=[], earlier=[1], cohort={2:4,3:5})
    stock = supply_clock(slot, {2:200,3:350})
    assert stock['launch_s'] == 351 and stock['ordinal'] == 2 and stock['first_supply_s'] == 201
    slot['earlier'] = []
    assert supply_clock(slot, {2:300,3:350})['launch_s'] == 301
    model = dict(launches={7:[300,2]}, returns={2:250,3:350})
    launches = {7:dict(bus=2, public_s=300, scheduled_s=300, eligible_s=300, actual_s=331)}
    row, s = explain(slot, model, launches, {(2,4):330,(3,5):350}, {7:4})
    assert s['same_current_cycle_decomposable'] == 1 and row['kind'] == 'miss'
    assert row['same_cycle']['launch_error_s'] == -31
    assert row['same_cycle']['return_error_s'] == -80
    assert row['same_cycle']['post_return_offset_error_s'] == 49
    model['launches'][6] = [100,2]
    assert explain(slot, model, launches, {(2,4):330,(3,5):350}, {7:4})[1]['same_current_cycle_decomposable'] == 0


def audit(root):
    checks()
    frozen = read(root / 'readout/summary/results.json')
    assert frozen['complete'] and frozen['paid_confirmation_paths'] == 8
    archive = run.old.frozen.helper(run.old.frozen.ASSETS, 'gate_default')
    by_day, totals, all_details = {}, defaultdict(Counter), {}
    for day in PLAN['confirmation_days']:
        directory = root / 'confirmation' / str(day) / 'cases' / PLAN['regime'] / str(day) / PLAN['arm']
        trace = archive.raw(directory / 'raw/path.pkl.gz')
        slots = read(directory / 'raw/public_slots.json')
        launches, returns, predecessors = ledger(trace)
        counts, details = defaultdict(Counter), []
        candidates = Counter()
        for slot in slots:
            if slot['lead_s'] != 300:
                continue
            slot['cohort'] = {int(k):v for k,v in slot['cohort'].items()}
            for model in slot['scenarios']:
                if model['scenario'] != 0 or model['failure']:
                    continue
                for k in ('returns','launches'):
                    model[k] = {int(b):v for b,v in model[k].items()}
                pair = model['launches'].get(slot['trip'])
                if pair is None:
                    continue
                predicted = pair[0] > slot['public_s']
                actual = launches[slot['trip']]['actual_s'] > slot['public_s']
                candidates.update(readout.clock_score(predicted, actual, bool(slot['ready'])))
                if predicted == actual:
                    continue
                row, stat = explain(slot, model, launches, returns, predecessors)
                row['day'] = day
                details.append(row)
                counts[row['kind']].update(stat)
                counts[row['kind']+'/' + ('queued' if slot['earlier'] else 'no_earlier')].update(stat)
        reference = frozen['by_day'][str(day)]['leads']['300']['scenarios']['0']['launch']
        assert dict(candidates) == reference
        assert len(details) == reference['misses']+reference['false_alarms']
        all_details[day] = details
        by_day[day] = counts
        for key, values in counts.items():
            totals[key].update(values)
    original = frozen['totals']['300']['scenarios']['0']['launch']
    assert totals['miss']['n'] == original['misses'] == 30
    assert totals['false_alarm']['n'] == original['false_alarms'] == 51
    examples = {kind:sorted((r for rows in all_details.values() for r in rows if r['kind'] == kind),
        key=lambda r:abs(r['predicted_launch_s']-r['actual_launch_s']), reverse=True)[:3]
        for kind in ('miss', 'false_alarm')}
    write(root / 'error_audit/details.json', all_details)
    write(root / 'error_audit/summary/results.json', dict(complete=True, physical_starts=0,
        training_updates=0, main_policy_changed=False, totals=totals, by_day=by_day, examples=examples,
        scope='81 fixed first-scenario advance errors; accounting, not counterfactual attribution',
        detail_file=str(root / 'error_audit/details.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
