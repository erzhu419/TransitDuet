"""Exercise real frozen model calls without starting a simulator."""

from dataclasses import replace
from types import SimpleNamespace as NS

from freqduet_apc_flow_evidence import APCArrival
from freqduet_causal_event_graph import Stop, Visit
from freqduet_fleet_observation import capture_fleet
from freqduet_flow_controller import assemble
from freqduet_native_continuation import inputs, native_request
from freqduet_native_event_graph import snapshot
from frozen_v406 import ReferenceGraph
from observer import PublicSlots, forecast, slot_snapshot
from plan_model import observe_plans


def fixture():
    now = 5000.
    routes = {d: (Stop(10+int(d)), Stop(20+int(d), False, True)) for d in (False, True)}
    links = [NS(distance=100., speed_limit=10.) for _ in range(4)]

    def vehicle(bus, state, direction=False):
        active = state != 'TERMINAL'
        route = routes[direction]
        return NS(bus_id=bus, trip_id=bus+30 if active else -1, on_route=active, direction=direction,
            state=NS(name=state), travel_distance=100., current_speed=10. if state == 'TRAVEL' else 0.,
            passengers=[None]*6 if active else [], capacity=50, launch_time=now-40,
            back_to_terminal_time=-1., dwelling_time=0., holding_time=0., last_service_dwell_s=10.,
            last_station=NS(station_id=route[0].station), next_station=NS(station_id=route[1].station),
            next_station_dis=50., current_route=links[1], episode_hold_vehicle_seconds=15.,
            forward_predecessor_trip_id=20, _target_headway=360.)

    buses = [vehicle(0, 'WAITING_ACTION'), vehicle(1, 'WAITING_ACTION'), vehicle(2, 'TRAVEL', True)]
    buses += [vehicle(i, 'TERMINAL', bool(i % 2)) for i in range(3, 12)]
    trips = [NS(launch_turn=80+i, direction=bool(i % 2), launched=False, target_headway=360.,
        launch_time=now+300+i*180, _freqduet_base_launch=now+300+i*180,
        _freqduet_terminal_dispatch=True, _freqduet_scheduled_launch=now+300+i*180)
        for i in range(5)]
    env = NS(current_time=now, bus_all=buses, bus_in_terminal=buses[3:], timetables=trips,
        routes=links, stations=[], max_agent_num=30, protocol=NS(evaluation_end_time_s=12000.),
        service_start_hour=5, service_end_hour=19, _last_dispatch_time={d:now-360 for d in routes})
    visits, events = [], []
    for i in range(4):
        for d, stops in routes.items():
            for stop in stops:
                t = now-1600+i*360
                boarded = 0. if stop.terminal else 6.
                visits.append(Visit(d, stop.station, t, t+10, t, 10., 8., boarded, 2., 360.))
                events.append(APCArrival(len(events), 8, 20+i, d, stop.station, t, t, t+10,
                                         8., 2., boarded, 50.))
    # A visit finishing on the query clock is not completed input.
    visits.append(replace(visits[0], completed_s=now, run_s=900.))
    packets, current = {}, {}
    for bus in buses[:2]:
        b, t, station = bus.bus_id, bus.trip_id, bus.last_station.station_id
        arrival = now-20+b
        current[b, t, station, False] = dict(arrival_s=arrival, boarded=4.)
        events.append(APCArrival(len(events), b, t, False, station, arrival, arrival, None, 4., 2., 4., 50.))
        packets[b, t, station] = dict(bus=b, trip=t, station=station, direction=False,
            prepare_s=now-1, progress_m=100., service_s=10., target_s=360.,
            predecessor=20, departure=dict(trip_id=20,time_s=now-270), avl=())
    prefix = NS(visits=visits, flow_events=events, current=current,
        launches={bus.trip_id:dict(holding_counter_at_launch=10.) for bus in buses[:3]},
        strict=lambda t: tuple(v for v in visits if v.completed_s < t and v.available_s < t))
    pending = NS(time_s=now, bus_id=1, trip_id=31, station_id=10, action_s=5)
    mask = (0, 5, 10)
    bridge = NS(routes=routes, prefix=prefix, evidence=NS(packets=packets),
        masks={(0, 30, 10):mask, (1, 31, 10):mask}, credit=NS(stream=NS(pending_commands=[pending])))
    return env, bridge, mask


def checks():
    env, bridge, mask = fixture()
    pending = bridge.credit.stream.pending_commands
    now = env.current_time
    native = native_request(bridge.evidence.packets[0, 30, 10], mask)[0]
    reference, locks = snapshot(env, bridge.prefix, bridge.routes, 0, mask, native, pending, bridge.masks)
    detached = slot_snapshot(env, bridge, capture_fleet(env))
    assert detached == replace(reference, current_bus=-1, native_action_s=0)
    assert locks == {(1, 31, 10):now+5} and detached.vehicles[1].remaining_s == 6
    assert detached.vehicles[2].remaining_fraction == .5

    base, _ = assemble(reference, bridge.prefix.strict(now), bridge.prefix.flow_events)
    plans, last = observe_plans(env)
    expected = ReferenceGraph(base, inputs(env, bridge.prefix, bridge.evidence), plans, last, {}).evaluate()
    forecasted, censored = forecast(env, bridge, capture_fleet(env))
    assert len(forecasted) == 1 and not censored and not forecasted[0]['failure']
    assert expected['returns'] == forecasted[0]['returns'] and len(expected['returns']) == 3
    assert {t:list(pair) for t, pair in expected['launches'].items()} == forecasted[0]['launches']

    recorder = PublicSlots([300, 0])
    recorder.before_step(env, bridge)
    assert len(recorder.records) == 1 and recorder.records[0]['lead_s'] == 300
    assert recorder.records[0]['cohort'] == {2:32} and recorder.records[0]['ready'] == [4, 6, 8, 10]
    env.current_time += 1
    recorder.before_step(env, bridge)
    recorder.finish(now, now+1)
    assert pending[0].action_s == 5 and env.bus_all[1].state.name == 'WAITING_ACTION'

    pending.clear()
    for second in range(2, 301):
        env.current_time = now+second
        recorder.before_step(env, bridge)
    assert recorder.records[-1]['trip'] == 80 and recorder.records[-1]['lead_s'] == 0
    assert recorder.records[-1]['scenarios']
    recorder.finish(now, now+300)

    env, bridge, _ = fixture()
    bridge.evidence.packets.clear()
    recorder = PublicSlots([300, 0])
    recorder.before_step(env, bridge)
    assert recorder.records[0]['unavailable'] == 'missing strictly prior PREPARE evidence'
    return ['native snapshot and issued-command parity', 'same-state frozen forecast parity',
            'public-slot timing and full compatible cohort', 'unsupported query abstains without action']
