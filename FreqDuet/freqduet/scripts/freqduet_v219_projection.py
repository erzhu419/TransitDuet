"""Pure native-planner previews from archived, query-time timetable state."""

import copy
from math import ceil

import numpy as np

from env.timetable import Timetable
import freqduet_v219_service_task as law


def ema_action(raw, previous, alpha):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    return ((alpha*raw) if previous is None else
            (alpha*raw+(1.-alpha)*previous)).astype(np.float32)


def query_timetable(query, observed_launches):
    trips = []
    for r in observed_launches:
        t = Timetable(r['public_baseline_s'], r['trip'], r['direction'])
        t.launched = True
        t._freqduet_actual_launch = r['actual_s']
        trips.append(t)
    for r in query['plans']:
        t = Timetable(r['public_s'], r['trip_id'], r['direction'])
        t._freqduet_base_launch = r['public_s']
        t._freqduet_base_target_headway = t.target_headway
        t.target_headway = r['target_s']
        t._freqduet_scheduled_launch = r['planned_s']
        t._freqduet_projected_target_headway = r['target_s']
        trips.append(t)
    return trips


def preview(planner, trips, query, effective_action):
    copied = copy.deepcopy(trips)
    current = next(t for t in copied if t.launch_turn == query['trip_id'])
    planner.apply(copied, current, effective_action, origin_launch_s=current.launch_time,
                  write_scheduled_launch=True, plan_id=current.launch_turn)
    return {int(t.launch_turn): dict(planned_s=float(t._freqduet_scheduled_launch),
                                     target_s=float(t.target_headway))
            for t in copied if not t.launched}


def verify_native(actual, expected, tick_s):
    assert actual.keys() == expected.keys(), 'native preview omitted pending trips'
    errors = []
    for trip in actual:
        for key in ('planned_s', 'target_s'):
            error = abs(actual[trip][key]-expected[trip][key])
            assert error <= 1e-6, ('native preview mismatch', trip, key, actual[trip], expected[trip])
            errors.append(error)
        assert ceil(actual[trip]['planned_s']/tick_s) == ceil(expected[trip]['planned_s']/tick_s), (
            'native executable clock mismatch', trip, actual[trip], expected[trip])
    return max(errors, default=0.)


def inspect_proposal(query, native, proposed, tick_s):
    before = {t: r['planned_s'] for t, r in native.items()}
    after = {t: r['planned_s'] for t, r in proposed.items()}
    admission = law.admit_preview(query, before, after, tick_s)
    resource = law.ready_prefix(query)
    rows = []
    for p in query['plans']:
        trip = p['trip_id']
        old = max(query['time_s'], ceil(before[trip]/tick_s)*tick_s)
        new = max(query['time_s'], ceil(after[trip]/tick_s)*tick_s)
        if before[trip] == after[trip]:
            continue
        rows.append(dict(trip_id=trip, direction=p['direction'], native_s=before[trip], proposed_s=after[trip],
            continuous_delta_s=after[trip]-before[trip], executable_delta_s=new-old,
            within_cycle=new < resource['expiry_public_s'],
            ready_bus_id=resource['ready_bus_by_trip'].get(trip)))
    continuous_later = [r['trip_id'] for r in rows if r['continuous_delta_s'] > 1e-6]
    clock_later = [r['trip_id'] for r in rows if r['executable_delta_s'] > 0]
    advances = [r for r in rows if r['executable_delta_s'] < 0 and r['within_cycle']]
    unfunded = [r['trip_id'] for r in advances if r['ready_bus_id'] is None]
    supported = admission['admitted'] and not continuous_later and not clock_later and not unfunded
    return dict(law_admission=admission, supported=bool(supported),
        rejection='continuous_plan_delay' if continuous_later else admission['reason'],
        continuous_later_trip_ids=continuous_later, clock_later_trip_ids=clock_later,
        unfunded_within_cycle_trip_ids=unfunded,
        advanced_by_direction=[sum(r['direction'] == d for r in advances) for d in (False, True)],
        outside_cycle_advanced=sum(r['executable_delta_s'] < 0 and not r['within_cycle'] for r in rows),
        changes=rows)
