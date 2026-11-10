"""Once-only system-time credit on the chronological lower decision stream."""

from dataclasses import dataclass, replace

import numpy as np


EXPOSURES = ("waiting_person_s", "onboard_person_s", "lf_waiting_person_s",
             "lf_onboard_person_s", "dispatch_backlog_trip_s")


@dataclass(frozen=True)
class Command:
    time_s: float
    bus_id: int
    trip_id: int
    station_id: int
    action_s: float


@dataclass(frozen=True)
class Decision:
    time_s: float
    bus_id: int
    trip_id: int
    station_id: int
    direction: bool
    local_state: tuple
    target_headway_s: float
    ready_origin: int
    ready_opposite: int
    pending_commands: tuple = ()


@dataclass(frozen=True)
class SystemTransition:
    state: Decision
    action_s: float
    next_state: Decision | None
    end_time_s: float
    exposure: np.ndarray

    @property
    def done(self):
        return self.next_state is None

    @property
    def bootstrap_discount(self):
        return 0.0 if self.done else 1.0

    @property
    def reward(self):
        return -float(self.exposure[0] + self.exposure[1]) / 3600.0


def observable_decision(runner, obs, key, previous_action):
    bus = runner._bus_for_agent(key)
    env = runner.env
    context = runner._pending_lower_action_context[int(key)]
    return Decision(
        time_s=float(env.current_time), bus_id=int(key), trip_id=int(bus.trip_id),
        station_id=int(obs[1]), direction=bool(bus.direction),
        local_state=tuple(float(x) for x in runner._augment_lower_state(obs, previous_action)),
        target_headway_s=float(context.target_headway_s),
        ready_origin=int(env._ready_vehicle_count(bool(bus.direction))),
        ready_opposite=int(env._ready_vehicle_count(not bool(bus.direction))))


class SystemCreditStream:
    """A finite-day, undiscounted SMDP chain; trip returns are nonterminal events."""

    def __init__(self):
        self.clock_s = 0.0
        self.total = np.zeros(len(EXPOSURES), dtype=float)
        self.prefix = self.total.copy()
        self.prefix_duration_s = 0.0
        self.transitions = []
        self.pending_commands = []
        self.returns = []
        self.active = None
        self.closed = False

    def decide(self, observation, action_s):
        if self.closed or observation.time_s != self.clock_s:
            raise ValueError("decision is not at the current open system clock")
        state = replace(observation, pending_commands=tuple(self.pending_commands))
        if self.active is None:
            self.prefix = self.total.copy()
            self.prefix_duration_s = self.clock_s
        else:
            self._close(state, self.clock_s)
        action_s = float(action_s)
        self.active = (state, action_s, self.total.copy())
        self.pending_commands.append(Command(self.clock_s, state.bus_id, state.trip_id,
                                             state.station_id, action_s))
        return state

    def advance(self, start_s, dt_s, waiting, onboard, lf_waiting, lf_onboard, backlog):
        if self.closed or float(start_s) != self.clock_s or float(dt_s) <= 0:
            raise ValueError("physical intervals must be positive, contiguous and recorded once")
        rates = np.array([waiting, onboard, lf_waiting, lf_onboard, backlog], dtype=float)
        if (not np.isfinite(rates).all() or (rates < 0).any()
                or lf_waiting > waiting + 1e-8 or lf_onboard > onboard + 1e-8):
            raise ValueError("invalid total/LF passenger exposure")
        self.total += float(dt_s) * rates
        self.clock_s += float(dt_s)

    def acknowledge(self, executed):
        key = lambda c: (c.time_s, c.bus_id, c.trip_id, c.station_id)
        if sorted(executed, key=key) != sorted(self.pending_commands, key=key):
            raise ValueError("proposed lower commands do not match physical execution")
        self.pending_commands.clear()

    def returned(self, bus_id, trip_id, physical_time_s):
        self.returns.append(dict(observed_time_s=self.clock_s, bus_id=int(bus_id),
                                 trip_id=int(trip_id), physical_time_s=float(physical_time_s)))

    def _close(self, next_state, end_time_s):
        state, action, start_exposure = self.active
        self.transitions.append(SystemTransition(state, action, next_state, float(end_time_s),
                                                  self.total - start_exposure))

    def finish(self, horizon_s, *, service_cleared, unserved, lf_unserved):
        if self.closed or self.pending_commands:
            raise ValueError("cannot finish a closed stream or leave unexecuted commands")
        horizon_s = float(horizon_s)
        if horizon_s < self.clock_s:
            raise ValueError("system clock exceeds the fixed evaluation horizon")
        if horizon_s > self.clock_s:
            if not service_cleared:
                raise ValueError("a truncated service cannot be extended as a cleared tail")
            self.advance(self.clock_s, horizon_s - self.clock_s,
                         unserved, 0, lf_unserved, 0, 0)
        if self.active is None:
            self.prefix = self.total.copy()
            self.prefix_duration_s = self.clock_s
        else:
            self._close(None, self.clock_s)
            self.active = None
        self.closed = True

    def audit(self):
        if not self.closed:
            raise ValueError("system credit audit requires a completed day")
        assigned = self.prefix + sum((t.exposure for t in self.transitions), np.zeros(len(EXPOSURES)))
        duration = self.prefix_duration_s + sum(t.end_time_s - t.state.time_s for t in self.transitions)
        np.testing.assert_allclose(assigned, self.total, rtol=0, atol=1e-7)
        if duration != self.clock_s:
            raise AssertionError("credit intervals do not conserve elapsed time")
        links = [t for t in self.transitions if not t.done]
        return dict(
            decisions=len(self.transitions), terminal_transitions=sum(t.done for t in self.transitions),
            cross_bus_bootstraps=sum(t.state.bus_id != t.next_state.bus_id for t in links),
            cross_trip_bootstraps=sum(t.state.trip_id != t.next_state.trip_id for t in links),
            same_clock_bootstraps=sum(t.state.time_s == t.end_time_s for t in links),
            pending_state_count=sum(bool(t.state.pending_commands) for t in self.transitions),
            returns=len(self.returns),
            return_crossing_bootstraps=sum(any(t.state.time_s < r["observed_time_s"] <= t.end_time_s
                                               for r in self.returns) for t in links),
            prefix_duration_s=self.prefix_duration_s, horizon_s=self.clock_s,
            exposure=dict(zip(EXPOSURES, self.total.tolist())),
            prefix_exposure=dict(zip(EXPOSURES, self.prefix.tolist())),
            reward_sum=sum(t.reward for t in self.transitions),
            conservation_max_error=float(np.max(np.abs(assigned - self.total))),
            finite_day_discount=1.0)
