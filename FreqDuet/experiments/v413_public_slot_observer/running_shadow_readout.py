"""Score V414 on the eight paid archives without running a model or policy."""

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v414_confirmation',
    HERE.parent / 'running_confirmation01/running_shadow_confirmation.py')
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)
run, read, write, PLAN = confirmation.run, confirmation.read, confirmation.write, confirmation.PLAN
readout = confirmation.parity.prior.readout
FIELDS = confirmation.parity.prior.FIELDS
from components import actual_components


def launch_pair(slot, original, shadow, actual_s):
    trip, due = slot['trip'], slot['public_s']
    base = original['launches'].get(trip)
    if original['failure'] or base is None:
        return None
    candidate = shadow['launches'].get(trip)
    fallback = bool(shadow['failure'] or candidate is None)
    effective = base if fallback else candidate
    scores = {}
    for key, (estimate, _) in (('v413', base), ('v414', effective)):
        score = readout.clock_score(estimate > due, actual_s > due, bool(slot['ready']))
        score.update(signed_s=estimate-actual_s, absolute_s=abs(estimate-actual_s))
        scores[key] = score
    return scores, base, effective, fallback


def score_day(trace, slots, sidecar):
    launches = {r['trip']: r for r in trace['dispatches']}
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    last, predecessors = {}, {}
    for row in sorted(launches.values(), key=lambda r: (r['actual_s'], r['trip'])):
        predecessors[row['trip']] = last.get(row['bus'])
        last[row['bus']] = row['trip']
    indexed = {tuple(key): sample for sample in sidecar for key in sample['slots']}
    groups, cached = {}, {}
    for slot in slots:
        trip, lead, query = (slot[k] for k in ('trip', 'lead_s', 'query_s'))
        sample = indexed[trip, lead]
        actual = launches[trip]
        cohort = slot['cohort']
        realized = {b: returns[b, t] for b, t in cohort.items()}
        for base, candidate in zip(slot['scenarios'], sample['scenarios']):
            key = f'{lead}/scenario_{base["scenario"]}'
            group = groups.setdefault(key, dict(launch=defaultdict(lambda: defaultdict(Counter)),
                rank=defaultdict(lambda: defaultdict(Counter)), components=defaultdict(Counter), coverage=Counter()))
            coverage = group['coverage']
            if (not base['failure'] and not slot['earlier'] and not slot['ready'] and realized
                    and realized.keys() <= base['returns'].keys()):
                base_returns = {b: base['returns'][b] for b in realized}
                rank_fallback = bool(candidate['failure'] or not realized.keys() <= candidate['returns'].keys())
                candidate_returns = base_returns if rank_fallback else {b: candidate['returns'][b] for b in realized}
                band, old = readout.return_order(base_returns, realized, run.PLAN['near_tie_s'])
                _, new = readout.return_order(candidate_returns, realized, run.PLAN['near_tie_s'])
                coverage.update(rank_pairs=1, rank_fallback=int(rank_fallback))
                for stratum in ('all', band):
                    group['rank'][stratum]['v413'].update(old)
                    group['rank'][stratum]['v414'].update(new)

            pair = launch_pair(slot, base, candidate, actual['actual_s'])
            if pair is None:
                coverage['baseline_launch_unavailable'] += 1
                continue
            scores, bp, cp, fallback = pair
            coverage.update(launch_pairs=1, launch_fallback=int(fallback), launch_clock_changed=int(bp[0] != cp[0]),
                            launch_bus_changed=int(bp[1] != cp[1]))
            queue = 'earlier_queued' if slot['earlier'] else 'no_earlier'
            for stratum in ('all', queue):
                for name, score in scores.items():
                    group['launch'][stratum][name].update(score)

            bus = actual['bus']
            if fallback or bp[1] != bus or cp[1] != bus or bus not in cohort or cohort[bus] != predecessors[trip]:
                continue
            if any(any(t < clock and b == bus for other, (t, b) in model['launches'].items() if other != trip)
                   for model, (clock, _) in ((base, bp), (candidate, cp))):
                continue
            active = cohort[bus]
            components = [{(r['bus'], r['trip']): r for r in candidate[name] if r['supported']}.get((bus, active))
                          for name in ('baseline', 'candidate')]
            if any(row is None for row in components):
                continue
            cache_key = bus, active, query
            if cache_key not in cached:
                cached[cache_key] = actual_components(trace, bus, active, query)
            observed = cached[cache_key]
            if not observed['supported']:
                coverage['actual_component_unsupported'] += 1
                continue
            coverage['matched_cycle_pairs'] += 1
            for name, model in zip(('v413', 'v414'), components):
                errors = {k: model['components'][k] - observed['components'][k] for k in FIELDS}
                total = model['return_s'] - observed['return_s']
                assert sum(errors.values()) == total
                group['components'][name].update(n=1, signed_return_s=total, absolute_return_s=abs(total),
                    **errors, future_requested_h_error_s=model['future_requested_h_s']-observed['future_requested_h_s'])
    return groups


def gate(day_rows):
    deltas, old, new = {}, Counter(), Counter()
    for day, groups in day_rows.items():
        b, c = (groups['300/scenario_0']['launch']['all'][arm] for arm in ('v413', 'v414'))
        assert b['n'] == c['n'] and b['n'] > 0
        deltas[day] = (c['absolute_s'] - b['absolute_s']) / b['n']
        old.update(b)
        new.update(c)
    delta = mean(deltas.values())
    improved = sum(x < 0 for x in deltas.values())
    checks = dict(mean_day_mae_improved=delta < 0, majority_days_improved=2*improved > len(deltas),
                  misses_not_increased=new['misses'] <= old['misses'],
                  false_alarms_not_increased=new['false_alarms'] <= old['false_alarms'])
    return dict(passed=all(checks.values()), checks=checks, day_mae_delta_s=deltas,
                mean_day_mae_delta_s=delta, days_improved=improved, days_worse=sum(x > 0 for x in deltas.values()))


def check():
    slot = dict(trip=1, public_s=300, ready=[])
    original = dict(launches={1: [300, 1]}, failure=None)
    candidate = dict(launches={1: [320, 2]}, failure=None)
    scores, _, _, fallback = launch_pair(slot, original, candidate, 321)
    assert not fallback and scores['v413']['misses'] == 1 and scores['v414']['absolute_s'] == 1
    for absent in (dict(launches={}, failure=None), dict(launches=candidate['launches'], failure='unsupported')):
        scores, _, _, fallback = launch_pair(slot, original, absent, 299)
        assert fallback and scores['v413'] == scores['v414']
    assert launch_pair(slot, dict(launches={}, failure=None), candidate, 300) is None


def review(root):
    check()
    confirmation.check_release(root)
    frozen = run.old.frozen
    archive = frozen.helper(frozen.ASSETS, 'gate_default')
    by_day, totals, execution, calls = {}, {}, {}, Counter()
    for day in PLAN['confirmation_days']:
        folder = root / confirmation.PHASE / str(day)
        start, receipt = read(folder / 'started.json'), read(folder / 'summary/results.json')
        assert start == dict(day=day, physical_starts=1, automatic_retry=False)
        assert receipt['completed'] and receipt['day'] == day and receipt['physical_starts'] == 1
        assert not any(receipt[k] for k in ('training_updates', 'main_policy_changed', 'automatic_retry'))
        assert receipt['paired_events'] == 5200 and receipt['slots']['schedule_exact']
        raw = folder / 'cases' / run.REGIME / str(day) / run.CAP / 'raw'
        trace = archive.raw(raw / 'path.pkl.gz')
        slots, sidecar = read(raw / 'public_slots.json'), read(raw / 'running_shadow.json')
        for slot in slots:
            slot['cohort'] = {int(k): v for k, v in slot['cohort'].items()}
            for model in slot['scenarios']:
                for key in ('returns', 'launches'):
                    model[key] = {int(k): v for k, v in model[key].items()}
        for row in sidecar:
            for model in row['scenarios']:
                for key in ('returns', 'launches'):
                    model[key] = {int(k): v for k, v in model[key].items()}
        assert confirmation.validate_sidecar(slots, sidecar) == receipt['shadow']
        assert confirmation.slot_audit(trace, slots) == receipt['slots']
        result = by_day[day] = score_day(trace, slots, sidecar)
        for lead in PLAN['lead_s']:
            expected = receipt['slots']['leads'][str(lead)]
            group = result[f'{lead}/scenario_0']
            assert sum(group['coverage'].get(k, 0) for k in ('launch_pairs', 'baseline_launch_unavailable')) == expected['model_available']
        readout.add_totals(totals, result)
        calls.update(receipt['shadow'])
        execution[day] = dict(wall_s=receipt['wall_s'], paired_events=receipt['paired_events'],
                              snapshots={lead: row['snapshots'] for lead, row in receipt['slots']['leads'].items()},
                              link_calls=receipt['shadow'])
    write(root / 'running_shadow_readout/summary/results.json', dict(complete=True, physical_starts=0,
        paid_confirmation_paths=len(by_day), training_updates=0, main_policy_changed=False,
        primary_gate=gate(by_day), totals=totals, by_day=by_day, execution=execution, link_calls=calls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    review(parser.parse_args().root)
