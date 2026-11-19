#!/usr/bin/env python3
"""V500 Gate 1 (Rev 02): Zero-training multi-timescale headroom & leakage study.

Incorporates the three physical calibrations uncovered by Rev 01 on dev days:
  1. Ceder-calibrated fleet pool & standard 12m urban bus capacity (75 pax on
     tidal_bimodal / mbta_route111_wkdy; legacy 12-bus / 50-pax on flat_sanity).
  2. Causal headway-normalized APC arrival rate feed (B / (Delta_t / 60)),
     eliminating intermittent zero-bin log1p Jensen bias in DemandFrequencyTracker.
  3. Exact Newell-Pontryagin remaining-quota square-root dispatch allocation
     with 20-min corridor travel-time lead forecast.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
FREQDUET_ROOT = HERE / "freqduet_src"
if not FREQDUET_ROOT.exists():
    FREQDUET_ROOT = HERE.parents[1] / "freqduet"
if str(FREQDUET_ROOT) not in sys.path:
    sys.path.insert(0, str(FREQDUET_ROOT))

from env.bus import Bus, BusState  # noqa: E402
from env.passenger import Passenger  # noqa: E402
from env.sim import env_bus  # noqa: E402
from env.station import Station  # noqa: E402
from lower.causal_departure_regularity import (  # noqa: E402
    causal_two_sided_holding_target_s,
)


def install_variable_link_patch() -> None:
    """Fix hardcoded 500m-per-link assumptions and support configurable bus capacity."""
    if getattr(Bus, "_v500_variable_link_patched", False):
        return

    orig_init = Bus.__init__
    orig_reset = Bus.reset_bus

    def _corridor_len_m(bus_obj: Bus) -> float:
        return float(sum(float(r.distance) for r in bus_obj.effective_route))

    def patched_init(self, bus_id, trip_id, launch_time, direction, routes, stations):
        orig_init(self, bus_id, trip_id, launch_time, direction, routes, stations)
        self.capacity = int(getattr(Bus, "_v500_default_capacity", 50))
        self.absolute_distance = 0.0 if self.direction else _corridor_len_m(self)

    def patched_reset(self, trip_num, launch_time):
        orig_reset(self, trip_num, launch_time)
        self.capacity = int(getattr(Bus, "_v500_default_capacity", 50))
        self.absolute_distance = 0.0 if self.direction else _corridor_len_m(self)

    Bus.__init__ = patched_init
    Bus.reset_bus = patched_reset
    Bus._v500_variable_link_patched = True


def install_exogenous_burst_patch() -> None:
    """Allow policy-independent ScenarioTape high-frequency station surge windows."""
    if getattr(Station, "_v500_burst_patched", False):
        return

    orig_schedule = Station.schedule_passenger_window

    def patched_schedule(
        self,
        current_time,
        stations,
        passenger_update_interval=1,
        demand_multipliers=None,
        demand_scale=1.0,
        od_multipliers=None,
        peak_shift=0,
        service_start_hour=6,
        service_end_hour=19,
        scenario_tape=None,
    ):
        base_scheduled = orig_schedule(
            self,
            current_time,
            stations,
            passenger_update_interval=passenger_update_interval,
            demand_multipliers=demand_multipliers,
            demand_scale=demand_scale,
            od_multipliers=od_multipliers,
            peak_shift=peak_shift,
            service_start_hour=service_start_hour,
            service_end_hour=service_end_hour,
            scenario_tape=scenario_tape,
        )
        bursts = getattr(self, "_v500_burst_windows", None)
        if not bursts or self.od is None or scenario_tape is None:
            return base_scheduled

        now_s = float(current_time)
        active_boost = 0.0
        for b_start, b_end, b_boost in bursts:
            if b_start <= now_s < b_end:
                active_boost += float(b_boost)
        if active_boost <= 1e-9:
            return base_scheduled

        interval_s = float(passenger_update_interval)
        hour = max(int(service_start_hour), min(int(service_end_hour), int(service_start_hour) + int(now_s // 3600)))
        lookup_hour = max(int(service_start_hour), min(int(service_end_hour), hour + int(peak_shift)))
        period_od = self.od.get(f"{lookup_hour:02d}:00:00", {})

        extra_scheduled = 0
        for destination_name, demand in period_od.items():
            if float(demand) <= 0.0:
                continue
            extra_rate = float(demand) * active_boost * float(demand_scale) / 3600.0 * interval_s
            count = scenario_tape.poisson(
                extra_rate,
                "v500_hf_burst_count",
                int(self.station_id),
                bool(self.direction),
                str(destination_name),
                now_s,
            )
            if count <= 0:
                continue
            destination = next(
                st for st in stations
                if st.station_name == destination_name and st.direction == self.direction
            )
            for idx in range(count):
                offset_s = scenario_tape.uniform(
                    0.0,
                    interval_s,
                    "v500_hf_burst_offset",
                    int(self.station_id),
                    bool(self.direction),
                    str(destination_name),
                    now_s,
                    int(idx),
                )
                offset_s = max(offset_s, float(np.nextafter(0.0, 1.0)))
                self._pending_passengers.append(Passenger(now_s + offset_s, self, destination))
            extra_scheduled += int(count)
        if extra_scheduled:
            self._pending_passengers.sort(key=lambda item: item.appear_time)
        return base_scheduled + extra_scheduled

    Station.schedule_passenger_window = patched_schedule
    Station._v500_burst_patched = True


def attach_day_bursts(env: env_bus, corridor_name: str) -> List[dict]:
    """Sample 4 exogenous 15-min localized surge events per day from ScenarioTape."""
    if corridor_name == "flat_sanity":
        return []
    tape = env.scenario_tape
    by_dir: Dict[bool, List[Station]] = {True: [], False: []}
    for st in env.stations:
        if st.station_type != 0 and st.od is not None:
            by_dir[bool(st.direction)].append(st)

    specs = []
    for k in range(4):
        direction = bool(k % 2 == 0)
        cand_stations = by_dir[direction]
        if len(cand_stations) < 4:
            continue
        start_s = float(int(tape.uniform(3600.0, 42000.0, "v500_burst_start", k) // 60) * 60)
        dur_s = 900.0
        center_idx = int(tape.choice(list(range(1, max(2, len(cand_stations) - 2))), "v500_burst_center", k))
        boost = float(tape.uniform(1.5, 2.4, "v500_burst_boost", k))
        affected = cand_stations[max(0, center_idx - 1): min(len(cand_stations), center_idx + 2)]
        for st in affected:
            if not hasattr(st, "_v500_burst_windows"):
                st._v500_burst_windows = []
            st._v500_burst_windows.append((start_s, start_s + dur_s, boost))
        specs.append({
            "burst_id": k,
            "direction": direction,
            "start_s": start_s,
            "end_s": start_s + dur_s,
            "boost": round(boost, 3),
            "stations": [int(s.station_id) for s in affected],
        })
    return specs


class CausalHeadwayAPCRateFeed:
    """Convert intermittent bus APC boarding counts into causal per-minute arrival rates.

    Eliminates the severe downward Jensen bias of feeding zero-padded 1-min APC
    counts into log1p(rate) between bus arrivals while remaining 100% causal and
    strictly deployable from onboard APC + stop arrival timestamps.
    """

    def __init__(self, env: env_bus):
        self.env = env
        self.rate_per_min: Dict[Tuple[int, bool], float] = {}
        self.last_board_seen: Dict[int, float | None] = {}
        self.seen_stop_arrival: set[Tuple[int, bool]] = set()
        self._orig_update = env.frequency_tracker.update

        # Initialize each stop's sample-and-hold rate from its t=0 historical prior
        tracker = env.frequency_tracker
        for st in env.stations:
            if st.station_type == 0 or st.od is None:
                continue
            key = (int(st.station_id), bool(st.direction))
            state = tracker._get_local_state(key)
            self.rate_per_min[key] = max(0.0, float(state.low))

        def wrapped_update(_raw_apc_bin, od_arrivals=None):
            scale_counts = float(tracker.bin_interval_s) / 60.0
            held_counts = {k: v * scale_counts for k, v in self.rate_per_min.items()}
            return self._orig_update(held_counts, od_arrivals or {})

        env.frequency_tracker.update = wrapped_update

    def observe_tick(self) -> None:
        env = self.env
        now_s = float(env.current_time)
        for bus in env.bus_all:
            if not bus.on_route:
                continue
            b_id = int(bus.bus_id)
            b_time = bus.last_board_time
            if b_time is None or b_time == self.last_board_seen.get(b_id):
                continue
            self.last_board_seen[b_id] = b_time
            st_id = int(bus.last_board_station_id)
            direction = bool(bus.direction)
            key = (st_id, direction)
            if key not in self.rate_per_min:
                continue

            # Causal elapsed service interval at this stop:
            # For the first bus of the day at this stop, passengers accumulated since t=0.0!
            hw_s = getattr(bus, "forward_headway", None)
            src = getattr(bus, "forward_headway_source", "")
            if key not in self.seen_stop_arrival or src != "arrival_event" or hw_s is None or float(hw_s) <= 0.0:
                hw_s = max(60.0, float(b_time))
            self.seen_stop_arrival.add(key)
            dt_min = float(np.clip(float(hw_s), 60.0, 1800.0)) / 60.0
            boarded = float(bus.last_board_count)
            if len(bus.passengers) >= int(bus.capacity):
                boarded *= 1.15  # causal APC full-load right-censoring adjustment
            inst_rate = boarded / dt_min
            self.rate_per_min[key] = float(0.75 * inst_rate + 0.25 * self.rate_per_min[key])


def plan_consistent_two_sided_target_s(
    *,
    forward_headway_s: float | None,
    follower_departure_gap_s: float | None,
    own_target_s: float,
    follower_target_s: float,
    action_cap_s: float,
) -> float | None:
    """Plan-normalized two-sided holding target under non-stationary headways."""
    cap = float(action_cap_s)
    if not np.isfinite(cap) or cap <= 0.0:
        return 0.0
    if forward_headway_s is None or follower_departure_gap_s is None:
        return None
    fwd = float(forward_headway_s)
    bwd = float(follower_departure_gap_s)
    h_own = max(float(own_target_s), 60.0)
    h_fol = max(float(follower_target_s), 60.0)
    if not (np.isfinite(fwd) and np.isfinite(bwd)) or fwd < 0.0 or bwd < 0.0:
        return None
    w_own = 1.0 / (h_own * h_own)
    w_fol = 1.0 / (h_fol * h_fol)
    u_star = ((bwd - h_fol) * w_fol - (fwd - h_own) * w_own) / (w_own + w_fol)
    return float(np.clip(u_star, 0.0, cap))


def causal_lf_allocation_stats(env: env_bus, now_s: float) -> dict:
    """Compute causal directional LF lead rates and remaining-day root-mean."""
    tracker = env.frequency_tracker
    bin_sec = float(tracker.bin_interval_s)
    cur_step = int(now_s // max(bin_sec, 1.0))
    end_step = int(46800.0 // max(bin_sec, 1.0))
    # Sample remaining horizon with 20-min corridor lead alignment
    lead_bins = 20
    rem_steps = list(range(cur_step + lead_bins, max(cur_step + lead_bins + 1, end_step + 1), 5))

    cur_dir = {True: 0.0, False: 0.0}
    step_totals = np.zeros(len(rem_steps), dtype=np.float64)

    for st in env.stations:
        if st.station_type == 0 or st.od is None:
            continue
        d = bool(st.direction)
        key = (int(st.station_id), d)
        state = tracker._get_local_state(key)
        # 20-minute lead forecast matches mid-corridor bus arrival time
        lead_est = 0.30 * float(state.low) + 0.70 * float(state.forecast(lead_bins))
        cur_dir[d] += max(0.0, lead_est)
        if hasattr(state, "_features") and hasattr(state, "_rate_from_log"):
            for idx, s in enumerate(rem_steps):
                step_totals[idx] += state._rate_from_log(float(state._features(s) @ state.theta))
        else:
            step_totals += max(0.0, lead_est)

    cur_tot = max(cur_dir[True] + cur_dir[False], 0.1)
    root_rem_mean = float(np.mean(np.sqrt(np.maximum(step_totals, 0.1))))
    return {
        "cur_up": max(cur_dir[True], 0.05),
        "cur_dn": max(cur_dir[False], 0.05),
        "cur_tot": cur_tot,
        "root_cur_tot": math.sqrt(cur_tot),
        "root_rem_mean": max(root_rem_mean, 0.1),
    }


def remaining_quota_headway_s(env: env_bus, now_s: float, direction: bool) -> float:
    """Return remaining average available headway (closing_s - anchor_s) / K_rem."""
    demand_end_s = float(getattr(env.protocol, "demand_end_time_s", 46980.0))
    dir_trips = sorted(
        (tt for tt in env.timetables if bool(tt.direction) == bool(direction)),
        key=lambda tt: float(getattr(tt, "_freqduet_base_launch", tt.launch_time)),
    )
    committed = [
        tt for tt in dir_trips
        if tt.launched or getattr(tt, "_upper_queried", False)
        or getattr(tt, "_freqduet_first_eligible_time", None) is not None
    ]
    mutable = [tt for tt in dir_trips if tt not in committed]
    if not mutable:
        return 360.0
    closing_s = max(float(dir_trips[-1]._freqduet_base_launch), demand_end_s)
    if not committed:
        anchor_s = max(float(mutable[0]._freqduet_base_launch), now_s)
        k_rem = max(1, len(mutable) - 1)
    else:
        last_c = committed[-1]
        anchor_s = float(getattr(
            last_c, "_actual_launch_time", getattr(last_c, "_freqduet_scheduled_launch", last_c.launch_time)
        ))
        k_rem = len(mutable)
    return float(np.clip((closing_s - anchor_s) / max(k_rem, 1), 240.0, 500.0))


def apply_global_quota_timetable(
    env: env_bus,
    now_s: float,
    desired_headway_by_dir: Dict[bool, float],
    active_window_s: float = 1800.0,
    h_bounds: Tuple[float, float] = (240.0, 520.0),
    tail_bounds: Tuple[float, float] = (225.0, 560.0),
) -> Dict[bool, dict]:
    """Unanchored recursive launch projection conserving exact daily trip count."""
    h_min, h_max = float(h_bounds[0]), float(h_bounds[1])
    t_min, t_max = float(tail_bounds[0]), float(tail_bounds[1])
    summary: Dict[bool, dict] = {}

    for direction in (True, False):
        dir_trips = sorted(
            (tt for tt in env.timetables if bool(tt.direction) == direction),
            key=lambda tt: float(getattr(tt, "_freqduet_base_launch", tt.launch_time)),
        )
        for tt in dir_trips:
            if not hasattr(tt, "_freqduet_base_launch"):
                tt._freqduet_base_launch = float(tt.launch_time)

        committed = [
            tt for tt in dir_trips
            if tt.launched or getattr(tt, "_upper_queried", False)
            or getattr(tt, "_freqduet_first_eligible_time", None) is not None
        ]
        mutable = [tt for tt in dir_trips if tt not in committed]
        if not mutable:
            summary[direction] = {"mutable": 0}
            continue

        closing_s = max(
            float(dir_trips[-1]._freqduet_base_launch),
            float(getattr(env.protocol, "demand_end_time_s", 46980.0)),
        )
        if not committed:
            first_tt = mutable[0]
            anchor_s = max(float(first_tt._freqduet_base_launch), now_s)
            first_tt._freqduet_scheduled_launch = anchor_s
            first_tt._freqduet_projected_target_headway = 360.0
            first_tt.target_headway = 360.0
            first_tt._freqduet_terminal_dispatch = True
            rest = mutable[1:]
        else:
            last_c = committed[-1]
            anchor_s = float(getattr(
                last_c,
                "_actual_launch_time",
                getattr(last_c, "_freqduet_scheduled_launch", last_c.launch_time),
            ))
            rest = mutable

        k_total = len(rest)
        if k_total == 0:
            summary[direction] = {"mutable": 1, "h_act": 360.0, "h_tail": 360.0}
            continue

        span_s = max(closing_s - anchor_s, k_total * t_min)
        h_des = float(np.clip(desired_headway_by_dir[direction], h_min, h_max))
        if k_total <= 3:
            h_act = float(max(t_min, span_s / k_total))
            prev = anchor_s
            for idx, tt in enumerate(rest, start=1):
                sched = max(prev + h_act, closing_s) if idx == k_total else (prev + h_act)
                step_h = sched - prev
                tt._freqduet_scheduled_launch = float(sched)
                tt._freqduet_projected_target_headway = float(step_h)
                tt.target_headway = float(step_h)
                tt._freqduet_terminal_dispatch = True
                prev = sched
            summary[direction] = {"mutable": k_total, "h_act": round(h_act, 2), "h_tail": round(h_act, 2)}
            continue

        k_win = min(k_total - 1, max(2, int(round(active_window_s / max(h_des, 180.0)))))
        k_tail = k_total - k_win

        lo_from_tail_max = (span_s - k_tail * t_max) / k_win
        hi_from_tail_min = (span_s - k_tail * t_min) / k_win
        lo_from_clock = max(0.0, (now_s + 1.0) - anchor_s)
        low_bound = max(h_min, lo_from_tail_max, lo_from_clock)
        high_bound = min(h_max, hi_from_tail_min)
        if low_bound > high_bound:
            h_act = float(np.clip(span_s / k_total, t_min, t_max))
            h_tail = h_act
        else:
            h_act = float(np.clip(h_des, low_bound, high_bound))
            h_tail = (span_s - k_win * h_act) / k_tail

        prev = anchor_s
        for idx, tt in enumerate(rest):
            if idx == k_total - 1:
                # Final sweeper trip never launches before demand_end_time_s so no late arrivals are stranded
                sched = max(prev + h_tail, closing_s)
                step_h = sched - prev
            else:
                step_h = h_act if idx < k_win else h_tail
                sched = prev + step_h
            tt._freqduet_scheduled_launch = float(sched)
            tt._freqduet_projected_target_headway = float(step_h)
            tt.target_headway = float(step_h)
            tt._freqduet_terminal_dispatch = True
            prev = sched

        summary[direction] = {
            "mutable": k_total,
            "k_win": k_win,
            "h_des": round(h_des, 2),
            "h_act": round(h_act, 2),
            "h_tail": round(h_tail, 2),
        }
    return summary


class TurnaroundBudgetLedger:
    """Track non-holding trip durations and issue causal turnaround-aware holding budgets."""

    def __init__(self, corridor_len_m: float):
        self.corridor_len_m = float(corridor_len_m)
        self.active_trips: Dict[Tuple[int, int], dict] = {}
        self.completed_nonhold_s: Dict[bool, List[float]] = {True: [], False: []}
        self.remaining_budget_s: Dict[Tuple[int, int], float] = {}
        self.slack_s: Dict[Tuple[int, int], float] = {}

    def sync_vehicles(self, env: env_bus, now_s: float) -> bool:
        live_keys = set()
        new_launched = False
        for bus in env.bus_all:
            if bus.on_route and int(bus.trip_id) >= 0:
                key = (int(bus.bus_id), int(bus.trip_id))
                live_keys.add(key)
                if key not in self.active_trips:
                    self.active_trips[key] = {
                        "direction": bool(bus.direction),
                        "launch_s": float(bus.launch_time),
                        "hold_s": 0.0,
                    }
                    new_launched = True
        finished = [k for k in self.active_trips if k not in live_keys]
        for key in finished:
            rec = self.active_trips.pop(key)
            dur = max(60.0, float(now_s) - float(rec["launch_s"]))
            nonhold = max(60.0, dur - float(rec["hold_s"]))
            self.completed_nonhold_s[rec["direction"]].append(nonhold)
            self.remaining_budget_s.pop(key, None)
            self.slack_s.pop(key, None)
        return new_launched

    def median_nonhold_s(self, direction: bool) -> float:
        hist = self.completed_nonhold_s[bool(direction)]
        if len(hist) >= 2:
            return float(np.median(hist[-8:]))
        return self.corridor_len_m / 6.8 + 230.0

    def cycle_headway_floor_s(self, n_fleet: int) -> float:
        cycle_s = self.median_nonhold_s(True) + self.median_nonhold_s(False) + 60.0
        return max(275.0, 1.14 * cycle_s / max(int(n_fleet), 1))

    def refresh_budgets(self, env: env_bus, now_s: float, lf_stats: dict) -> None:
        supplies: Dict[bool, List[Tuple[float, int, int | None]]] = {True: [], False: []}

        for bus in env.bus_all:
            if not bus.on_route:
                raw_ret = getattr(bus, "back_to_terminal_time", None)
                ready_s = (
                    now_s
                    if (raw_ret is None or not math.isfinite(float(raw_ret)))
                    else max(now_s, float(raw_ret) + 1.0)
                )
                supplies[bool(bus.direction)].append((ready_s, int(bus.bus_id), None))
            elif int(bus.trip_id) >= 0:
                d = bool(bus.direction)
                opp_d = not d
                prog = float(np.clip(bus.travel_distance / max(self.corridor_len_m, 1.0), 0.0, 1.0))
                rem_nonhold = (
                    (1.0 - prog) * self.median_nonhold_s(d)
                    + float(max(0.0, bus.holding_time))
                    + float(max(0.0, bus.dwelling_time))
                )
                eta_s = now_s + rem_nonhold
                supplies[opp_d].append((eta_s + 1.0, int(bus.bus_id), int(bus.trip_id)))

        matched_keys = set()
        for opp_d in (True, False):
            stock = sorted(supplies[opp_d], key=lambda item: (item[0], item[1]))
            upcoming = sorted(
                (
                    tt for tt in env.timetables
                    if not tt.launched and bool(tt.direction) == opp_d
                ),
                key=lambda tt: float(getattr(tt, "_freqduet_scheduled_launch", tt.launch_time)),
            )
            cur_opp_lf = lf_stats["cur_up"] if opp_d else lf_stats["cur_dn"]
            cur_own_lf = lf_stats["cur_dn"] if opp_d else lf_stats["cur_up"]
            backhaul_express = cur_opp_lf > 1.35 * max(cur_own_lf, 0.1)

            for tt, sup in zip(upcoming, stock):
                eta_ready, b_id, active_tid = sup
                if active_tid is None:
                    continue
                key = (b_id, active_tid)
                matched_keys.add(key)
                sched_s = float(getattr(tt, "_freqduet_scheduled_launch", tt.launch_time))
                slack = sched_s - eta_ready - 20.0
                self.slack_s[key] = float(slack)
                if backhaul_express and slack < 75.0:
                    proposed = float(np.clip(0.35 * max(0.0, slack), 0.0, 30.0))
                else:
                    proposed = float(np.clip(0.80 * max(0.0, slack), 0.0, 85.0))
                prev = self.remaining_budget_s.get(key)
                self.remaining_budget_s[key] = proposed if prev is None else min(prev, proposed)

        for key in self.active_trips:
            if key not in matched_keys and key not in self.remaining_budget_s:
                self.remaining_budget_s[key] = 75.0
                self.slack_s.setdefault(key, 180.0)

    def debit(self, bus_id: int, trip_id: int, hold_s: float) -> None:
        key = (int(bus_id), int(trip_id))
        if key in self.remaining_budget_s:
            self.remaining_budget_s[key] = max(0.0, self.remaining_budget_s[key] - float(hold_s))
        if key in self.active_trips:
            self.active_trips[key]["hold_s"] += float(hold_s)


def compute_headway_metrics(events: List[dict]) -> dict:
    """Compute raw CV2, plan-normalized CV2, and TfL Excess Waiting Time (EWT)."""
    raw_h = []
    norm_sq = []
    by_sd: Dict[Tuple[int, bool], List[Tuple[float, float]]] = defaultdict(list)

    for ev in events:
        h = ev.get("headway_s")
        t = ev.get("target_headway_s")
        if h is None or t is None:
            continue
        hf, tf = float(h), max(float(t), 1.0)
        if not (math.isfinite(hf) and math.isfinite(tf)):
            continue
        raw_h.append(hf)
        norm_sq.append(((hf - tf) / tf) ** 2)
        by_sd[(int(ev["station_id"]), bool(ev["direction"]))].append((hf, tf))

    if not raw_h:
        return {"raw_cv2": 0.0, "plan_cv2": 0.0, "ewt_s": 0.0, "awt_s": 0.0, "swt_s": 0.0}

    arr = np.asarray(raw_h, dtype=np.float64)
    mean_h = max(float(arr.mean()), 1.0)
    raw_cv2 = float(arr.var() / (mean_h ** 2))
    plan_cv2 = float(np.mean(norm_sq))

    awt_list, swt_list, ewt_list = [], [], []
    for pairs in by_sd.values():
        hs = np.asarray([p[0] for p in pairs], dtype=np.float64)
        ts = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if hs.sum() <= 1.0 or ts.sum() <= 1.0:
            continue
        awt = float((hs @ hs) / (2.0 * hs.sum()))
        swt = float((ts @ ts) / (2.0 * ts.sum()))
        awt_list.append(awt)
        swt_list.append(swt)
        ewt_list.append(awt - swt)

    return {
        "raw_cv2": raw_cv2,
        "plan_cv2": plan_cv2,
        "awt_s": float(np.mean(awt_list)) if awt_list else 0.0,
        "swt_s": float(np.mean(swt_list)) if swt_list else 0.0,
        "ewt_s": float(np.mean(ewt_list)) if ewt_list else 0.0,
    }


def _follower_target_headway_s(env: env_bus, bus: Bus) -> float:
    f_bid = getattr(bus, "pre_action_follower_bus_id", None)
    if f_bid is not None:
        for cand in env.bus_all:
            if int(cand.bus_id) == int(f_bid) and cand.on_route:
                return float(env._get_target_headway_for_bus(cand))
    return float(bus._target_headway)


def run_episode(corridor_dir: Path, corridor_meta: dict, day_seed: int, arm: str) -> dict:
    """Execute one complete service day for the given arm under exact CRN."""
    install_variable_link_patch()
    install_exogenous_burst_patch()
    Bus._v500_default_capacity = int(corridor_meta.get("bus_capacity", 50))

    n_fleet = int(corridor_meta["n_fleet"])
    env_cfg = {
        "service_start_hour": 6,
        "service_end_hour": 19,
        "clearance_time_s": 14400,
        "include_terminal_headways": False,
        "fleet_inventory_mode": "fixed_pool",
        "upper_fleet_state_mode": "fixed_pool_readiness_v4",
        "observation_contract": "deployable_apc_avl_v4",
        "headway_reward_mode": "forward_event_only",
        "scenario_seed": int(day_seed),
        "max_agent_num": 30,
    }
    env = env_bus(str(corridor_dir), debug=False, render=False, route_sigma=1.5, env_config=env_cfg)
    env.enable_plot = False
    env._n_fleet_target = n_fleet
    env.demand_noise = 0.15
    env.lower_state_input_schema = "causal_forward_v4"
    env.unobserved_action_mode = "zero"
    env.holding_action_trace_mode = "all_decisions"

    env.configure_frequency_features({
        "enable": True,
        "method": "harmonic",
        "observation_source": "apc_boardings",
        "use_historical_prior": True,
        "bin_sec": 60,
        "harmonic_period_s": 50400.0,
        "fourier_K": 4,
        "harmonic_forgetting": 0.998,
        "harmonic_prior_var": 0.05,
        "forecast_horizon_s": 1800.0,
        "forecast_mode": "causal",
        "upper_features": True,
        "lower_features": True,
        "upper_mode": "low",
        "lower_mode": "high",
        "promotion": {
            "enable": True,
            "window_min": 10.0,
            "cooldown_min": 15.0,
            "residual_threshold": 1.25,
            "residual_mode": "positive",
            "persistence_ratio": 0.55,
            "adapt_low": True,
            "adapt_gain": 0.08,
            "adapt_strength_min": 0.2,
            "adapt_local": True,
        },
        "lower_context": {
            "enable": True,
            "features": ["load", "capacity", "queue", "speed_residual", "shock_age", "schedule_slack"],
        },
    })

    use_lf_schedule = arm in {
        "A2_lf_quota_nohold",
        "A3_lf_quota_unbudgeted_hold",
        "A4_lf_quota_turnaround_hf_hold",
    }
    env._upper_policy_callback = lambda _s, trip: float(getattr(trip, "target_headway", 360.0))

    env.reset()
    apc_feed = CausalHeadwayAPCRateFeed(env)
    for tt in env.timetables:
        tt._freqduet_base_launch = float(tt.launch_time)
        tt._freqduet_scheduled_launch = float(tt.launch_time)
        tt._freqduet_projected_target_headway = 360.0
        tt.target_headway = 360.0
        tt._freqduet_terminal_dispatch = True
        tt._freqduet_min_dispatch_headway = 120.0
    bursts = attach_day_bursts(env, corridor_meta["name"])
    corridor_len_m = float(sum(float(r.distance) for r in env.routes[: len(env.routes) // 2]))
    ledger = TurnaroundBudgetLedger(corridor_len_m=corridor_len_m)

    last_replan_cycle = -1
    last_budget_refresh_s = -999.0
    cached_lf_stats = causal_lf_allocation_stats(env, 0.0)
    replan_records = []
    total_hold_s = 0.0
    action_bins = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0], dtype=np.float64)

    t0 = time.monotonic()
    while not env.done:
        now_s = float(env.current_time)
        new_trip_launched = ledger.sync_vehicles(env, now_s)

        # Slow clock every 900s during active service
        if now_s <= 45900.0:
            cycle = int(now_s // 900.0)
            if cycle != last_replan_cycle:
                last_replan_cycle = cycle
                cached_lf_stats = causal_lf_allocation_stats(env, now_s)
                if use_lf_schedule:
                    h_floor = ledger.cycle_headway_floor_s(n_fleet)
                    ratio_root = cached_lf_stats["root_rem_mean"] / cached_lf_stats["root_cur_tot"]
                    desired_hw = {}
                    for d in (True, False):
                        h_avail = remaining_quota_headway_s(env, now_s, d)
                        h_loop = h_avail * ratio_root
                        cur_d = cached_lf_stats["cur_up"] if d else cached_lf_stats["cur_dn"]
                        opp_d = cached_lf_stats["cur_dn"] if d else cached_lf_stats["cur_up"]
                        ready_d = env._ready_vehicle_count(d)
                        ready_opp = env._ready_vehicle_count(not d)
                        dir_share_centered = (cur_d / cached_lf_stats["cur_tot"]) - 0.5
                        inv_guard = float(np.clip(0.5 + 0.35 * (ready_d - ready_opp), 0.0, 1.0))
                        tilt_s = -35.0 * dir_share_centered * inv_guard
                        raw_h = h_loop + tilt_s
                        if opp_d > 1.45 * max(cur_d, 0.05):
                            raw_h = min(raw_h, 440.0)
                        desired_hw[d] = float(np.clip(raw_h, h_floor, 500.0))
                    plan_info = apply_global_quota_timetable(
                        env, now_s, desired_hw, h_bounds=(h_floor, 500.0), tail_bounds=(max(225.0, h_floor - 15.0), 550.0)
                    )
                    replan_records.append({
                        "time_s": now_s,
                        "h_floor": round(h_floor, 1),
                        "ratio_root": round(ratio_root, 3),
                        "plan": plan_info,
                    })

        if arm == "A4_lf_quota_turnaround_hf_hold" and (
            new_trip_launched or (now_s - last_budget_refresh_s) >= 60.0
        ):
            cached_lf_stats = causal_lf_allocation_stats(env, now_s)
            ledger.refresh_budgets(env, now_s, cached_lf_stats)
            last_budget_refresh_s = now_s

        act_dict = {bus.bus_id: None for bus in env.bus_all}
        for bus in env.bus_all:
            if not bus.on_route or bus.state != BusState.WAITING_ACTION:
                continue
            fwd_valid = (
                bus.pre_action_forward_headway_source == "matched_departure_event"
                and bus.pre_action_forward_headway is not None
            )
            bwd_valid = (
                str(bus.pre_action_follower_source).startswith("same_time_avl_")
                and bus.pre_action_follower_departure_gap is not None
            )
            fwd = bus.pre_action_forward_headway if fwd_valid else None
            bwd = bus.pre_action_follower_departure_gap if bwd_valid else None

            if arm in {"A0_fixed_nohold", "A2_lf_quota_nohold"}:
                chosen = 0.0
            elif arm == "A1_fixed_bilateral":
                req = causal_two_sided_holding_target_s(
                    forward_headway_s=fwd,
                    follower_departure_gap_s=bwd,
                    action_cap_s=15.0,
                )
                bins15 = action_bins[action_bins <= 15.0]
                chosen = float(bins15[np.argmin(np.abs(bins15 - req))]) if req is not None else 0.0
            elif arm == "A3_lf_quota_unbudgeted_hold":
                fol_hw = _follower_target_headway_s(env, bus)
                req = plan_consistent_two_sided_target_s(
                    forward_headway_s=fwd,
                    follower_departure_gap_s=bwd,
                    own_target_s=float(bus._target_headway),
                    follower_target_s=fol_hw,
                    action_cap_s=30.0,
                )
                bins30 = action_bins[action_bins <= 30.0]
                chosen = float(bins30[np.argmin(np.abs(bins30 - req))]) if req is not None else 0.0
            elif arm == "A4_lf_quota_turnaround_hf_hold":
                key = (int(bus.bus_id), int(bus.trip_id))
                rem_b = float(ledger.remaining_budget_s.get(key, 30.0))
                slack = float(ledger.slack_s.get(key, 120.0))
                cap_s = min(20.0, max(0.0, rem_b))
                if cap_s < 5.0 or fwd is None or bwd is None:
                    chosen = 0.0
                else:
                    h_own = max(float(bus._target_headway), 60.0)
                    fol_hw = max(_follower_target_headway_s(env, bus), 60.0)
                    cur_idx = bus.effective_station.index(bus.last_station) if bus.last_station in bus.effective_station else 1
                    # Causal downstream LF + positive HF passenger arrival rate [pax/s]
                    ds_rate_per_min = 0.0
                    for s_obj in bus.effective_station[cur_idx + 1 : -1]:
                        st_state = env.frequency_tracker._get_local_state((int(s_obj.station_id), bool(bus.direction)))
                        ds_rate_per_min += max(0.0, float(st_state.low)) + max(0.0, float(st_state.high))
                    ds_rate_per_s = ds_rate_per_min / 60.0
                    onboard_pax = float(len(bus.passengers))
                    slack_shadow_pax = 12.0 * max(0.0, (45.0 - slack) / 45.0)

                    # Evaluate exact Osuna-Newell net person-seconds saved across legal bins
                    legal_bins = action_bins[action_bins <= cap_s + 1e-6]
                    best_u = 0.0
                    best_net = 0.0
                    gap_term = (float(bwd) / fol_hw) - (float(fwd) / h_own)
                    curv_term = (1.0 / h_own) + (1.0 / fol_hw)
                    for cand_u in legal_bins:
                        u_f = float(cand_u)
                        if u_f <= 0.0:
                            continue
                        wait_saved_pax_s = ds_rate_per_s * 0.5 * (h_own + fol_hw) * u_f * (gap_term - 0.5 * u_f * curv_term)
                        delay_cost_pax_s = (onboard_pax + slack_shadow_pax) * u_f
                        net_gain = wait_saved_pax_s - delay_cost_pax_s
                        if net_gain > best_net:
                            best_net = net_gain
                            best_u = u_f
                    chosen = best_u
            else:
                raise ValueError(f"Unknown arm: {arm}")

            act_dict[bus.bus_id] = chosen
            if chosen > 0.0:
                total_hold_s += chosen
            ledger.debit(bus.bus_id, bus.trip_id, chosen)

        env.step(act_dict, debug=False, render=False)
        apc_feed.observe_tick()

    wall_s = time.monotonic() - t0
    details = dict(env._measurement_details)
    hw_metrics = compute_headway_metrics(env.headway_events.events)
    launched_count = sum(1 for tt in env.timetables if tt.launched)
    completed_count = len(env._completed_trip_ids)
    readiness_mean = float(np.mean(env._fleet_readiness_delays_s)) if env._fleet_readiness_delays_s else 0.0
    readiness_p95 = float(np.percentile(env._fleet_readiness_delays_s, 95)) if env._fleet_readiness_delays_s else 0.0

    return {
        "corridor": corridor_meta["name"],
        "day": int(day_seed),
        "arm": arm,
        "scenario_tape_id": env.scenario_tape.identifier,
        "burst_count": len(bursts),
        "wall_s": round(wall_s, 3),
        "metrics": {
            "passengers_generated": int(details["passengers_generated"]),
            "passenger_unserved_rate": float(details["passenger_unserved_rate"]),
            "restricted_journey_s_per_pax": float(details["restricted_total_journey_horizon_min"]) * 60.0,
            "restricted_wait_s_per_pax": float(details["restricted_wait_horizon_min"]) * 60.0,
            "restricted_in_vehicle_s_per_pax": float(details["restricted_in_vehicle_horizon_min"]) * 60.0,
            "ewt_s": hw_metrics["ewt_s"],
            "plan_cv2": hw_metrics["plan_cv2"],
            "raw_cv2": hw_metrics["raw_cv2"],
            "awt_s": hw_metrics["awt_s"],
            "swt_s": hw_metrics["swt_s"],
            "readiness_delay_s_per_trip": readiness_mean,
            "readiness_delay_p95_s": readiness_p95,
            "hold_s_per_trip": total_hold_s / max(launched_count, 1),
            "trips_launched": int(launched_count),
            "trips_completed": int(completed_count),
            "physical_buses": len(env.bus_all),
        },
        "sample_replans": replan_records[:12],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def preflight(root: Path) -> None:
    proto = json.loads((HERE / "protocol.json").read_text())
    manifest = json.loads((HERE / "corridors" / "manifest.json").read_text())
    assert set(proto["corridors"]) <= set(manifest.keys())
    for c in proto["corridors"]:
        c_dir = HERE / "corridors" / c
        assert (c_dir / "config.json").exists()
        for fn in ("passenger_OD.xlsx", "route_news.xlsx", "stop_news.xlsx", "time_table.xlsx"):
            assert (c_dir / "data" / fn).exists(), (c, fn)
    install_variable_link_patch()
    install_exogenous_burst_patch()
    out = {
        "passed": True,
        "physical_starts": 0,
        "corridors": proto["corridors"],
        "days": proto["days"],
        "arms": proto["arms"],
    }
    write_json(root / "preflight" / "summary" / "results.json", out)
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "check", "elapsed_seconds": 0.1}), flush=True)


def run_shard(root: Path, corridor: str, day: int) -> None:
    proto = json.loads((HERE / "protocol.json").read_text())
    c_dir = HERE / "corridors" / corridor
    c_meta = json.loads((c_dir / "corridor_meta.json").read_text())
    arms = proto["arms"]
    started = time.monotonic()
    print("SCHEDULEURM_PROGRESS " + json.dumps({
        "current": 0, "total": len(arms), "unit": "arm", "elapsed_seconds": 0.0,
    }), flush=True)

    results = {}
    base_pax = None
    base_tape = None
    for idx, arm in enumerate(arms, start=1):
        row = run_episode(c_dir, c_meta, day, arm)
        if base_pax is None:
            base_pax = row["metrics"]["passengers_generated"]
            base_tape = row["scenario_tape_id"]
        else:
            assert row["metrics"]["passengers_generated"] == base_pax, (
                "CRN violation: generated passenger count differed!", corridor, day, arm,
                row["metrics"]["passengers_generated"], base_pax,
            )
            assert row["scenario_tape_id"] == base_tape
        assert row["metrics"]["physical_buses"] == int(c_meta["n_fleet"])
        assert row["metrics"]["trips_launched"] == 262
        results[arm] = row
        print("SCHEDULEURM_PROGRESS " + json.dumps({
            "current": idx, "total": len(arms), "unit": "arm",
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }), flush=True)

    summary = {
        "completed": True,
        "corridor": corridor,
        "day": int(day),
        "scenario_tape_id": base_tape,
        "passengers_generated": base_pax,
        "arms": results,
    }
    write_json(root / "shards" / f"{corridor}_{day}" / "summary" / "results.json", summary)


def aggregate(root: Path) -> None:
    proto = json.loads((HERE / "protocol.json").read_text())
    corridors = proto["corridors"]
    days = proto["days"]
    arms = proto["arms"]
    rng = np.random.default_rng(int(proto["bootstrap"]["seed"]))
    draws = rng.integers(0, len(days), size=(int(proto["bootstrap"]["draws"]), len(days)))

    by_corridor: Dict[str, Any] = {}
    for c in corridors:
        shard_rows = [
            json.loads((root / "shards" / f"{c}_{d}" / "summary" / "results.json").read_text())
            for d in days
        ]
        means = {}
        for arm in arms:
            metric_keys = list(shard_rows[0]["arms"][arm]["metrics"].keys())
            means[arm] = {
                k: round(float(np.mean([r["arms"][arm]["metrics"][k] for r in shard_rows])), 6)
                for k in metric_keys
            }

        contrasts_to_compute = [
            ("A4_minus_A1_fixed_bilateral", "A4_lf_quota_turnaround_hf_hold", "A1_fixed_bilateral"),
            ("A4_minus_A0_fixed_nohold", "A4_lf_quota_turnaround_hf_hold", "A0_fixed_nohold"),
            ("A2_minus_A0_pure_lf_gain", "A2_lf_quota_nohold", "A0_fixed_nohold"),
            ("A3_minus_A2_unbudgeted_leakage", "A3_lf_quota_unbudgeted_hold", "A2_lf_quota_nohold"),
            ("A4_minus_A3_budget_repair", "A4_lf_quota_turnaround_hf_hold", "A3_lf_quota_unbudgeted_hold"),
        ]
        contrasts = {}
        for label, a, b in contrasts_to_compute:
            c_dict = {}
            for k in (
                "restricted_journey_s_per_pax",
                "restricted_wait_s_per_pax",
                "restricted_in_vehicle_s_per_pax",
                "ewt_s",
                "plan_cv2",
                "raw_cv2",
                "readiness_delay_s_per_trip",
                "hold_s_per_trip",
                "passenger_unserved_rate",
            ):
                deltas = np.asarray([r["arms"][a]["metrics"][k] - r["arms"][b]["metrics"][k] for r in shard_rows], dtype=np.float64)
                boot = deltas[draws].mean(axis=1)
                c_dict[k] = {
                    "mean": round(float(deltas.mean()), 6),
                    "ci95": [round(float(x), 6) for x in np.quantile(boot, [0.025, 0.975])],
                    "days_improved": int((deltas < 0.0).sum()),
                }
            contrasts[label] = c_dict

        by_corridor[c] = {
            "n_days": len(days),
            "arm_means": means,
            "paired_contrasts": contrasts,
        }

    out = {
        "completed": True,
        "experiment": proto["experiment"] + "_rev02",
        "by_corridor": by_corridor,
    }
    write_json(root / "aggregate" / "summary" / "results.json", out)
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "aggregate", "elapsed_seconds": 0.5}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "shard", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corridor", type=str)
    parser.add_argument("--day", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight(args.root)
    elif args.phase == "shard":
        run_shard(args.root, args.corridor, args.day)
    elif args.phase == "aggregate":
        aggregate(args.root)


if __name__ == "__main__":
    main()
