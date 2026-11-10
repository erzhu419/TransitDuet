"""V402 copy of the V401 law; budget issuance is factored for three policies.

Only completed physical trips and current telemetry form supply estimates.
Those estimates do not reserve vehicles or certify counterfactual service gains.
"""

import math
from statistics import median

from freqduet_fleet_observation import capture_fleet
from freqduet_native_event_graph import native_routes

START_S, PERIOD_S, LOOKAHEAD_S, SHIFT_S = 7200., 900., 2700., 30.
MIN_HISTORY = 3


def quota(normal_s, pressure_s, retain_fraction):
    """Apply the unchanged allowance formula at the declared retention."""
    assert retain_fraction in (.25, .5)
    return int(math.ceil(max(retain_fraction*normal_s, normal_s-pressure_s)))


def terminal_time(now_s, previous_actual_s, first_public_s, closing_s,
                  low_now, low_forecast, supported):
    if supported and low_now+low_forecast > 0:
        weight = min(1., max(0., (closing_s-now_s)/1800.))
        left = max(0., low_now)
        right = max(0., (1-weight)*low_now+weight*low_forecast)
    else:
        left = right = 1.
    ideal = (left*previous_actual_s+right*closing_s)/(left+right)
    offset = min(SHIFT_S, max(-SHIFT_S, ideal-first_public_s))
    return first_public_s+offset, left, right


class SlowManager:
    def __init__(self, arm, env, credit, retain_fraction):
        if arm not in ('fixed', 'plan', 'budget', 'joint'):
            raise ValueError(arm)
        self.arm, self.env, self.credit = arm, env, credit
        assert retain_fraction in (.25, .5)
        self.retain_fraction = float(retain_fraction)
        self.remaining, self.events = {}, []
        self.last_model = None
        self.active, self.completed = {}, {False: [], True: []}
        self.last_actual = {False: None, True: None}
        self.public = None
        self.last_cycle = -1
        self.budgeted_commands = self.budgeted_command_s = 0
        self.budgeted_residence_s = 0

    def on_launch(self, trip, bus, time_s):
        # New buses have not yet received this mode from env.step at launch;
        # the environment is the source assigned before every physical drive.
        assert self.env.unobserved_action_mode == 'zero'
        key = int(bus.bus_id), int(trip.launch_turn)
        assert key not in self.active
        direction = bool(trip.direction)
        self.active[key] = dict(direction=direction, launch_s=float(time_s),
            command_s=0., residence_s=0., commands=0)
        self.last_actual[direction] = dict(trip=key[1], actual_s=float(time_s),
            release_s=float(trip._freqduet_scheduled_launch))

    def commit(self, bus, trip, station, time_s, action):
        key, action = (int(bus), int(trip)), int(action)
        row = self.active[key]
        row['command_s'] += action
        row['residence_s'] += max(1, action)
        row['commands'] += 1
        if key in self.remaining:
            assert action <= self.remaining[key], (key, action, self.remaining[key])
            self.remaining[key] -= action
            self.budgeted_commands += 1
            self.budgeted_command_s += action
            self.budgeted_residence_s += max(1, action)

    def on_return(self, bus, trip, time_s):
        key = int(bus), int(trip)
        row = self.active.pop(key)
        duration = float(time_s)-row['launch_s']
        direction = row['direction']
        controls = [s.station for s in native_routes(self.env)[direction] if s.control]
        requested = [h for h in self.credit.holds
                     if (int(h['bus_id']),int(h['trip_id'])) == key]
        requested_stations = {int(h['station_id']) for h in requested}
        assert len(requested_stations) == len(requested) == row['commands']
        assert requested_stations <= set(controls)
        assert sum(h['action_s'] for h in requested) == row['command_s']
        assert sum(max(1.,h['action_s']) for h in requested) == row['residence_s']
        # The frozen native route visits every control once before a physical
        # return. With unobserved_action_mode='zero', every unrequested control
        # takes one automatic zero tick. Credit.holds contains all requested
        # actions, including genuine zero requests, and is execution-checked.
        automatic = set(controls)-requested_stations
        residence = row['residence_s']+len(automatic)
        assert duration >= residence >= row['residence_s']
        # First trips can have no valid forward headway and therefore no policy
        # requests. Their automatic zeros are physical one-tick residences, not
        # submitted holding commands or evidence of a zero policy mean.
        eligible = row['commands'] > 0
        self.completed[row['direction']].append(dict(
            trip=key[1], bus=key[0], return_s=float(time_s), duration_s=duration,
            nonholding_s=duration-residence, policy_commands=row['commands'],
            physical_controls=len(controls), automatic_zero_controls=len(automatic),
            physical_residence_s=residence, history_eligible=eligible,
            command_per_stop_s=row['command_s']/len(controls) if eligible else None,
            residence_per_stop_s=residence/len(controls) if eligible else None))
        if key in self.remaining:
            self.events.append(dict(kind='budget_return', time_s=float(time_s),
                bus=key[0], trip=key[1], unused_s=self.remaining.pop(key)))

    def _history(self, direction, now_s):
        rows = [r for r in self.completed[direction]
                if r['return_s'] < now_s and r['history_eligible']]
        if len(rows) < MIN_HISTORY:
            return None
        return dict(samples=len(rows), nonholding_s=median(r['nonholding_s'] for r in rows),
            command_per_stop_s=median(r['command_per_stop_s'] for r in rows),
            residence_per_stop_s=median(r['residence_per_stop_s'] for r in rows))

    def _estimates(self, env, captured, now_s):
        routes = native_routes(env)
        split = len(env.routes)//2
        coordinates = {}
        for direction, links in ((True, env.routes[:split]), (False, env.routes[split:])):
            distance, values = 0., []
            assert len(links) == len(routes[direction])
            for link, stop in zip(links, routes[direction]):
                distance += float(link.distance)
                values.append((distance, stop))
            coordinates[direction] = values
        pending = {(int(c.bus_id), int(c.trip_id), int(c.station_id)):float(c.action_s)
                   for c in self.credit.stream.pending_commands}
        histories = {d:self._history(d, now_s) for d in (False, True)}
        estimates = {}
        for row in captured['fleet']:
            if not row['on_route']:
                continue
            direction, bus, trip = row['direction'], row['bus_id'], row['active_trip_id']
            history = histories[direction]
            if history is None:
                continue
            progress = row['progress_m']
            remaining_stops = []
            committed_now = 0.
            for position, stop in coordinates[direction]:
                key = bus, trip, stop.station
                if key in pending:
                    committed_now += max(1., pending[key])
                current_uncommitted = (abs(position-progress) < 1e-6 and
                    row['state'] in ('HOLDING', 'WAITING_ACTION') and key not in pending)
                if stop.control and (position > progress+1e-6 or current_uncommitted) and key not in pending:
                    remaining_stops.append(stop.station)
            n = len(remaining_stops)
            fraction = min(1., max(0., 1-progress/captured['route_lengths_m'][int(direction)]))
            normal = n*history['command_per_stop_s']
            # Keep command seconds distinct from residence: zero commands still
            # require a tick. The observed ordinary overhead is a forecast only;
            # budget cuts may change its future value, so no return guarantee.
            overhead = n*max(0., history['residence_per_stop_s']-history['command_per_stop_s'])
            irreversible = row['remaining_command_hold_s']+row['remaining_service_s']+committed_now
            base_return = now_s+fraction*history['nonholding_s']+irreversible+overhead
            estimates[bus,trip] = dict(bus=bus, trip=trip, direction=bool(direction),
                supply_direction=not bool(direction), remaining_controls=n,
                normal_command_s=normal, base_return_s=base_return,
                ordinary_return_s=base_return+normal, irreversible_s=irreversible,
                ordinary_zero_tick_overhead_s=overhead, history_samples=history['samples'])
        return estimates, histories

    def _match(self, captured, estimates, now_s, use_budget):
        supplies = {False: [], True: []}
        for row in captured['fleet']:
            if not row['on_route']:
                returned = row['terminal_return_s']
                ready_s = now_s if returned is None else max(now_s, returned+1)
                supplies[row['direction']].append((ready_s, row['bus_id'], None))
        for key, item in estimates.items():
            allowance = item['normal_command_s']
            if use_budget and key in self.remaining:
                allowance = min(allowance, self.remaining[key])
            supplies[item['supply_direction']].append((item['base_return_s']+allowance+1,
                                                     item['bus'], item['trip']))
        matched = {}
        for direction in (False, True):
            stock = sorted(supplies[direction], key=lambda r:(r[0],r[1]))
            plans = sorted((p for p in captured['plans'] if not p['launched'] and
                p['direction'] == direction and self.public[p['trip_id']] <= now_s+LOOKAHEAD_S),
                key=lambda p:(self.public[p['trip_id']],p['trip_id']))
            for plan, supply in zip(plans, stock):
                available, bus, active_trip = supply
                matched[plan['trip_id']] = dict(available_s=available, bus=bus,
                    active_trip=active_trip, public_s=self.public[plan['trip_id']])
        return matched

    def issue_budgets(self, estimates, matched):
        issued = []
        if self.arm in ('budget','joint'):
            for target, match in matched.items():
                key = match['bus'], match['active_trip']
                if key not in estimates:
                    continue
                item = estimates[key]
                normal = item['normal_command_s']
                pressure = max(0., item['ordinary_return_s']+1-match['public_s'])
                if normal <= 0 or pressure <= 0 or not item['remaining_controls']:
                    continue
                proposed = quota(normal, pressure, self.retain_fraction)
                old = self.remaining.get(key)
                value = proposed if old is None else min(old, proposed)
                self.remaining[key] = value
                if old is None or value < old:
                    issued.append(dict(item, target_trip=target,
                        old_remaining_s=old, remaining_s=value, pressure_s=pressure))
        return issued

    def publish(self, env, bridge):
        now = float(env.current_time)
        if now < START_S:
            return
        cycle = int((now-START_S)//PERIOD_S)
        if cycle == self.last_cycle:
            return
        self.last_cycle = cycle
        if self.public is None:
            self.public = {int(t.launch_turn):float(getattr(t, '_freqduet_base_launch',
                getattr(t, '_original_launch', t.launch_time))) for t in env.timetables}
        captured = capture_fleet(env)
        estimates, histories = self._estimates(env, captured, now)
        matched = self._match(captured, estimates, now, use_budget=False)
        issued = self.issue_budgets(estimates, matched)
        # In the joint arm the plan may use its newly published allowance, but
        # the estimate remains a prediction, never a physical vehicle reserve.
        plan_supply = self._match(captured, estimates, now, use_budget=self.arm == 'joint')
        frequency = self.upper_summary
        low = max(0., float(frequency['freq_low_demand']))
        future = max(0., float(frequency['freq_low_forecast']))
        frequency_supported = int(frequency['freq_updates']) > 0 and low+future > 0
        changes, rejections = [], []
        if self.arm in ('plan','joint'):
            by_id = {int(t.launch_turn):t for t in env.timetables}
            for direction in (False, True):
                pending = sorted((p for p in captured['plans'] if not p['launched'] and
                    p['direction'] == direction), key=lambda p:(self.public[p['trip_id']],p['trip_id']))
                previous = self.last_actual[direction]
                if previous is None or len(pending) < 2 or histories[direction] is None:
                    rejections.append(dict(direction=direction, reason='history_or_anchor_missing'))
                    continue
                first, closing = pending[:2]
                p1, p2 = self.public[first['trip_id']], self.public[closing['trip_id']]
                mutable = lambda p: (not p['upper_queried'] and p['first_eligible_s'] is None
                    and p['terminal_mode'] and not p['offset_mode'] and p['min_gap_s'] is None)
                if not (now < p1 < p2 <= now+PERIOD_S and mutable(first) and mutable(closing)):
                    rejections.append(dict(direction=direction, reason='first_two_not_mutable_future'))
                    continue
                desired, left, right = terminal_time(now, previous['actual_s'], p1, p2,
                                                     low, future, frequency_supported)
                desired = min(p2-180., max(now+1., previous['release_s']+180., desired))
                executable = float(math.ceil(desired))
                if executable == first['published_release_s']:
                    continue
                supply = plan_supply.get(first['trip_id'])
                if supply is None or supply['available_s'] > executable:
                    rejections.append(dict(direction=direction, trip=first['trip_id'], reason='supply_estimate_not_ready'))
                    continue
                changed, anchor = by_id[first['trip_id']], by_id[closing['trip_id']]
                assert anchor._freqduet_scheduled_launch == p2
                old = float(changed._freqduet_scheduled_launch)
                changed._freqduet_scheduled_launch = executable
                changed.target_headway = executable-previous['release_s']
                anchor.target_headway = p2-executable
                changes.append(dict(direction=direction, trip=first['trip_id'], closing_trip=closing['trip_id'],
                    public_s=p1, old_release_s=old, release_s=executable, shift_s=executable-p1,
                    previous_actual_s=previous['actual_s'], previous_release_s=previous['release_s'],
                    target_s=changed.target_headway, closing_target_s=anchor.target_headway,
                    supply=supply, left_weight=left, right_weight=right))
        self.events.append(dict(kind='publish', time_s=now, cycle=cycle,
            history_samples={str(d):0 if h is None else h['samples'] for d,h in histories.items()},
            frequency=dict(updates=int(frequency['freq_updates']), low_now=low,
                low_forecast=future, supported=frequency_supported),
            budgets=issued, plans=changes, rejections=rejections))

    def summary(self):
        publications = [e for e in self.events if e['kind'] == 'publish']
        plans = [p for e in publications for p in e['plans']]
        budgets = [b for e in publications for b in e['budgets']]
        return dict(arm=self.arm, publications=len(publications), planned_changes=len(plans),
            advances=sum(p['shift_s'] < 0 for p in plans), delays=sum(p['shift_s'] > 0 for p in plans),
            plan_trip_ids=sorted({p['trip'] for p in plans}),
            budget_issues=sum(b['old_remaining_s'] is None for b in budgets),
            budget_tightenings=sum(b['old_remaining_s'] is not None for b in budgets),
            budgeted_commands=self.budgeted_commands, budgeted_command_s=self.budgeted_command_s,
            budgeted_residence_s=self.budgeted_residence_s,
            closed_budgets=sum(e['kind'] == 'budget_return' for e in self.events),
            remaining_budget_count=len(self.remaining),
            frequency_fallback_publications=sum(not e['frequency']['supported'] for e in publications),
            completed_history={str(d):len(rows) for d,rows in self.completed.items()},
            eligible_history={str(d):sum(r['history_eligible'] for r in rows)
                              for d,rows in self.completed.items()},
            completed_without_policy_requests=sum(not r['history_eligible']
                for rows in self.completed.values() for r in rows),
            automatic_zero_controls=sum(r['automatic_zero_controls']
                for rows in self.completed.values() for r in rows))
