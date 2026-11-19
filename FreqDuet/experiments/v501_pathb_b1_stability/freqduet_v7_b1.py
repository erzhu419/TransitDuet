#!/usr/bin/env python3
"""FreqDuet-v7 (Paradigm B1): Pure Bi-Level Constrained Soft Actor-Critic.

Implements genuine frequency-separated hierarchical RL for high-frequency
urban bus corridors under fixed fleet pools and exact daily trip quotas:
  - Upper Slow-Clock Ensemble SAC (Delta_T = 900s, 30 discrete plans over
    remaining-quota loop headway factor x inventory-guarded directional tilt
    x turnaround holding budget retention tier).
  - Lower Stop-Event Constrained SAC (zero-hold-advantage ensemble critic +
    hard turnaround-budget action masking via action_limit_feature_index=0 +
    zero-hold-regret CMDP dual).
  - 100% estimator-free physical rewards at both levels; frequency separation
    enters strictly through causal observation channels (`aligned`, `raw`,
    `all_band`, `swapped`, `nofreq`).
  - Full atomic checkpoint save/resume (`--ckpt-dir` + `--resume`) for
    scheduleurm cluster execution on node001-node006.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
FREQDUET_ROOT = HERE / "freqduet_src"
if not FREQDUET_ROOT.exists():
    FREQDUET_ROOT = HERE.parents[1] / "freqduet"
if str(FREQDUET_ROOT) not in sys.path:
    sys.path.insert(0, str(FREQDUET_ROOT))

torch.set_num_threads(1)

from env.bus import Bus, BusState  # noqa: E402
from env.passenger import Passenger  # noqa: E402
from env.sim import env_bus  # noqa: E402
from env.station import Station  # noqa: E402
from lower.causal_departure_regularity import (  # noqa: E402
    causal_two_sided_holding_target_s,
)
from lower.cost_replay_buffer import CostReplayBuffer  # noqa: E402
from lower.resac_lagrangian import RESACLagrangianTrainer  # noqa: E402
from upper.resac_upper import RESACUpperTrainer  # noqa: E402


UPPER_STATE_DIM = 21
LOWER_STATE_DIM = 20
LOWER_ACTION_BINS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
LOWER_MAX_HOLD_S = 25.0
UPPER_GAMMA = 0.996
UPPER_N_STEP = 6


def build_upper_action_candidates() -> np.ndarray:
    """30 discrete joint plans: 5 loop headway factors x 3 tilts x 2 budget tiers."""
    rows = []
    for loop_factor in (0.78, 0.89, 1.00, 1.12, 1.24):
        for dir_tilt_s in (-36.0, 0.0, 36.0):
            for budget_tier in (0.50, 0.90):
                rows.append([loop_factor, dir_tilt_s, budget_tier])
    return np.asarray(rows, dtype=np.float32)


def install_simulator_patches() -> None:
    """Apply variable-link GTFS geometry, configurable capacity, and HF burst patches."""
    if not getattr(Bus, "_v7_patched", False):
        orig_init = Bus.__init__
        orig_reset = Bus.reset_bus

        def _corridor_len_m(bus_obj: Bus) -> float:
            return float(sum(float(r.distance) for r in bus_obj.effective_route))

        def patched_init(self, bus_id, trip_id, launch_time, direction, routes, stations):
            orig_init(self, bus_id, trip_id, launch_time, direction, routes, stations)
            self.capacity = int(getattr(Bus, "_v7_default_capacity", 50))
            self.absolute_distance = 0.0 if self.direction else _corridor_len_m(self)

        def patched_reset(self, trip_num, launch_time):
            orig_reset(self, trip_num, launch_time)
            self.capacity = int(getattr(Bus, "_v7_default_capacity", 50))
            self.absolute_distance = 0.0 if self.direction else _corridor_len_m(self)

        Bus.__init__ = patched_init
        Bus.reset_bus = patched_reset
        Bus._v7_patched = True

    if not getattr(Station, "_v7_burst_patched", False):
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
            bursts = getattr(self, "_v7_burst_windows", None)
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
        Station._v7_burst_patched = True


def attach_day_bursts(env: env_bus, corridor_name: str, burst_scale: float = 1.0) -> List[dict]:
    """Attach ScenarioTape-paired exogenous 15-min localized station surges."""
    if corridor_name == "flat_sanity" or burst_scale <= 0.0:
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
        boost = float(tape.uniform(1.5, 2.4, "v500_burst_boost", k)) * float(burst_scale)
        affected = cand_stations[max(0, center_idx - 1): min(len(cand_stations), center_idx + 2)]
        for st in affected:
            if not hasattr(st, "_v7_burst_windows"):
                st._v7_burst_windows = []
            st._v7_burst_windows.append((start_s, start_s + dur_s, boost))
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
    """Deployable causal APC boarding rate estimator + raw APC history tracker."""

    def __init__(self, env: env_bus):
        self.env = env
        self.rate_per_min: Dict[Tuple[int, bool], float] = {}
        self.raw_rate_hist: Dict[Tuple[int, bool], deque] = {}
        self.dir_raw_hist: Dict[bool, deque] = {True: deque(maxlen=30), False: deque(maxlen=30)}
        self.last_board_seen: Dict[int, float | None] = {}
        self.seen_stop_arrival: set[Tuple[int, bool]] = set()
        self._orig_update = env.frequency_tracker.update
        tracker = env.frequency_tracker

        for st in env.stations:
            if st.station_type == 0 or st.od is None:
                continue
            key = (int(st.station_id), bool(st.direction))
            state = tracker._get_local_state(key)
            init_r = max(0.0, float(state.low))
            self.rate_per_min[key] = init_r
            self.raw_rate_hist[key] = deque([init_r] * 6, maxlen=16)

        for d in (True, False):
            tot_d = sum(v for (_, kd), v in self.rate_per_min.items() if kd == d)
            for _ in range(16):
                self.dir_raw_hist[d].append(tot_d)

        def wrapped_update(_raw_apc_bin, od_arrivals=None):
            scale_counts = float(tracker.bin_interval_s) / 60.0
            held_counts = {k: v * scale_counts for k, v in self.rate_per_min.items()}
            for d in (True, False):
                self.dir_raw_hist[d].append(
                    sum(v for (_, kd), v in self.rate_per_min.items() if kd == d)
                )
            return self._orig_update(held_counts, od_arrivals or {})

        env.frequency_tracker.update = wrapped_update

    def observe_tick(self) -> None:
        env = self.env
        for bus in env.bus_all:
            if not bus.on_route:
                continue
            b_id = int(bus.bus_id)
            b_time = bus.last_board_time
            if b_time is None or b_time == self.last_board_seen.get(b_id):
                continue
            self.last_board_seen[b_id] = b_time
            key = (int(bus.last_board_station_id), bool(bus.direction))
            if key not in self.rate_per_min:
                continue

            hw_s = getattr(bus, "forward_headway", None)
            src = getattr(bus, "forward_headway_source", "")
            if key not in self.seen_stop_arrival or src != "arrival_event" or hw_s is None or float(hw_s) <= 0.0:
                hw_s = max(60.0, float(b_time))
            self.seen_stop_arrival.add(key)
            dt_min = float(np.clip(float(hw_s), 60.0, 1800.0)) / 60.0
            boarded = float(bus.last_board_count)
            if len(bus.passengers) >= int(bus.capacity):
                boarded *= 1.15
            inst_rate = boarded / dt_min
            self.raw_rate_hist[key].append(inst_rate)
            self.rate_per_min[key] = float(0.75 * inst_rate + 0.25 * self.rate_per_min[key])


def plan_consistent_two_sided_loss_and_target(
    *,
    forward_headway_s: float | None,
    follower_departure_gap_s: float | None,
    own_target_s: float,
    follower_target_s: float,
    action_s: float,
    action_cap_s: float,
) -> Tuple[float, float, float, bool]:
    """Return (loss_zero, loss_action, u_star_s, valid_two_sided)."""
    h_own = max(float(own_target_s), 60.0)
    h_fol = max(float(follower_target_s), 60.0)
    u = max(float(action_s), 0.0)
    cap = max(float(action_cap_s), 0.0)

    if forward_headway_s is None or not math.isfinite(float(forward_headway_s)):
        return 0.0, 0.0, 0.0, False
    fwd = max(0.0, float(forward_headway_s))

    if follower_departure_gap_s is None or not math.isfinite(float(follower_departure_gap_s)):
        # One-sided causal forward bunching relief when follower has not yet launched
        l0 = 0.5 * (max(0.0, (h_own - fwd) / h_own) ** 2)
        lu = 0.5 * (((fwd + u - h_own) / h_own) ** 2 if (fwd + u > h_own) else (max(0.0, (h_own - (fwd + u)) / h_own) ** 2))
        u_star = float(np.clip(0.5 * max(0.0, h_own - fwd), 0.0, cap))
        return float(l0), float(lu), u_star, False

    bwd = max(0.0, float(follower_departure_gap_s))
    w_own = 1.0 / (h_own * h_own)
    w_fol = 1.0 / (h_fol * h_fol)
    u_star = float(np.clip(((bwd - h_fol) * w_fol - (fwd - h_own) * w_own) / (w_own + w_fol), 0.0, cap))
    l0 = 0.5 * (((fwd - h_own) / h_own) ** 2 + ((bwd - h_fol) / h_fol) ** 2)
    lu = 0.5 * (((fwd + u - h_own) / h_own) ** 2 + ((max(0.0, bwd - u) - h_fol) / h_fol) ** 2)
    return float(l0), float(lu), u_star, True


def fast_directional_lf_now(env: env_bus) -> Dict[str, float]:
    """Fast O(N_stations) lookup of current directional LF lead rates for 60s budget refresh."""
    tracker = env.frequency_tracker
    up_v, dn_v = 0.0, 0.0
    for st in env.stations:
        if st.station_type == 0 or st.od is None:
            continue
        d = bool(st.direction)
        state = tracker._get_local_state((int(st.station_id), d))
        est = max(0.0, 0.30 * float(state.low) + 0.70 * float(state.forecast(20)))
        if d:
            up_v += est
        else:
            dn_v += est
    return {"cur_up": max(up_v, 0.05), "cur_dn": max(dn_v, 0.05)}


def causal_lf_allocation_stats(env: env_bus, now_s: float) -> dict:
    """Compute causal directional LF lead rates, slopes, HF summaries, and remaining root-mean."""
    tracker = env.frequency_tracker
    bin_sec = float(tracker.bin_interval_s)
    cur_step = int(now_s // max(bin_sec, 1.0))
    end_step = int(46800.0 // max(bin_sec, 1.0))
    lead_bins = 20
    rem_steps = list(range(cur_step + lead_bins, max(cur_step + lead_bins + 1, end_step + 1), 5))

    cur_dir = {True: 0.0, False: 0.0}
    now_dir = {True: 0.0, False: 0.0}
    slope_dir = {True: 0.0, False: 0.0}
    hf_pos_dir = {True: 0.0, False: 0.0}
    hf_energy_dir = {True: 0.0, False: 0.0}
    hf_max_dir = {True: 0.0, False: 0.0}
    hf_active_cnt = {True: 0, False: 0}
    st_cnt = {True: 0, False: 0}
    step_totals = np.zeros(len(rem_steps), dtype=np.float64)

    for st in env.stations:
        if st.station_type == 0 or st.od is None:
            continue
        d = bool(st.direction)
        st_cnt[d] += 1
        key = (int(st.station_id), d)
        state = tracker._get_local_state(key)
        low_v = max(0.0, float(state.low))
        fc_v = max(0.0, float(state.forecast(lead_bins)))
        lead_est = 0.30 * low_v + 0.70 * fc_v
        now_dir[d] += low_v
        cur_dir[d] += lead_est
        slope_dir[d] += float(state.low_slope)

        pos_h = max(0.0, float(state.high))
        hf_pos_dir[d] += pos_h
        hf_energy_dir[d] += math.sqrt(max(0.0, float(state.high_energy)))
        hf_max_dir[d] = max(hf_max_dir[d], pos_h)
        if pos_h > 0.35 * max(low_v, 0.5):
            hf_active_cnt[d] += 1

        if hasattr(state, "_features") and hasattr(state, "_rate_from_log"):
            for idx, s in enumerate(rem_steps):
                step_totals[idx] += state._rate_from_log(float(state._features(s) @ state.theta))
        else:
            step_totals += lead_est

    cur_tot = max(cur_dir[True] + cur_dir[False], 0.1)
    rem_mean = float(np.mean(np.maximum(step_totals, 0.1)))
    root_rem_mean = float(np.mean(np.sqrt(np.maximum(step_totals, 0.1))))
    return {
        "now_up": max(now_dir[True], 0.05),
        "now_dn": max(now_dir[False], 0.05),
        "cur_up": max(cur_dir[True], 0.05),
        "cur_dn": max(cur_dir[False], 0.05),
        "slope_up": slope_dir[True],
        "slope_dn": slope_dir[False],
        "hf_pos_up": hf_pos_dir[True],
        "hf_pos_dn": hf_pos_dir[False],
        "hf_energy_up": hf_energy_dir[True],
        "hf_energy_dn": hf_energy_dir[False],
        "hf_max_up": hf_max_dir[True],
        "hf_max_dn": hf_max_dir[False],
        "hf_frac_up": hf_active_cnt[True] / max(st_cnt[True], 1),
        "hf_frac_dn": hf_active_cnt[False] / max(st_cnt[False], 1),
        "cur_tot": cur_tot,
        "rem_mean": max(rem_mean, 0.1),
        "root_cur_tot": math.sqrt(cur_tot),
        "root_rem_mean": max(root_rem_mean, 0.1),
    }


def remaining_quota_info(env: env_bus, now_s: float, direction: bool) -> Tuple[float, float]:
    """Return (remaining_avail_headway_s, remaining_mutable_fraction)."""
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
        return 360.0, 0.0
    closing_s = max(
        float(dir_trips[-1]._freqduet_base_launch),
        float(getattr(env.protocol, "demand_end_time_s", 46980.0)),
    )
    if not committed:
        anchor_s = max(float(mutable[0]._freqduet_base_launch), now_s)
        k_rem = max(1, len(mutable) - 1)
    else:
        last_c = committed[-1]
        anchor_s = float(getattr(
            last_c, "_actual_launch_time", getattr(last_c, "_freqduet_scheduled_launch", last_c.launch_time)
        ))
        k_rem = len(mutable)
    h_avail = float(np.clip((closing_s - anchor_s) / max(k_rem, 1), 240.0, 500.0))
    return h_avail, float(len(mutable) / max(len(dir_trips), 1))


def quota_future_wait_potential_pax_s(env: env_bus, now_s: float, daily_pax_prior: float) -> float:
    """Terminal-vanishing Ng-Harada-Russell (1999) potential Phi(s) = -sum_d 0.5 * lam_dir * S_rem_d * H_avail_d."""
    lam_dir_s = 0.5 * float(daily_pax_prior) / 46980.0
    tot_fut_wait = 0.0
    for direction in (True, False):
        dir_trips = sorted(
            (tt for tt in env.timetables if bool(tt.direction) == direction),
            key=lambda tt: float(getattr(tt, "_freqduet_base_launch", tt.launch_time)),
        )
        committed = [
            tt for tt in dir_trips
            if tt.launched or getattr(tt, "_upper_queried", False)
            or getattr(tt, "_freqduet_first_eligible_time", None) is not None
        ]
        mutable = [tt for tt in dir_trips if tt not in committed]
        if not mutable:
            continue
        closing_s = max(
            float(dir_trips[-1]._freqduet_base_launch),
            float(getattr(env.protocol, "demand_end_time_s", 46980.0)),
        )
        if not committed:
            anchor_s = max(float(mutable[0]._freqduet_base_launch), now_s)
            k_rem = max(1, len(mutable) - 1)
        else:
            last_c = committed[-1]
            anchor_s = float(getattr(
                last_c, "_actual_launch_time", getattr(last_c, "_freqduet_scheduled_launch", last_c.launch_time)
            ))
            k_rem = len(mutable)
        span_s = max(0.0, closing_s - anchor_s)
        h_av = float(np.clip(span_s / max(k_rem, 1), 220.0, 600.0))
        tot_fut_wait += 0.5 * lam_dir_s * span_s * h_av
    return -tot_fut_wait


def apply_global_quota_timetable(
    env: env_bus,
    now_s: float,
    desired_headway_by_dir: Dict[bool, float],
    active_window_s: float = 1800.0,
    h_bounds: Tuple[float, float] = (250.0, 455.0),
    tail_bounds: Tuple[float, float] = (255.0, 445.0),
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
                last_c, "_actual_launch_time", getattr(last_c, "_freqduet_scheduled_launch", last_c.launch_time)
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
            h_act = float(max(t_min, span_s / k_total))
            h_tail = h_act
        else:
            h_act = float(np.clip(h_des, low_bound, high_bound))
            h_tail = (span_s - k_win * h_act) / k_tail

        prev = anchor_s
        for idx, tt in enumerate(rest):
            if idx == k_total - 1:
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
    """Causal turnaround slack ledger with upper-policy budget retention control."""

    def __init__(self, corridor_len_m: float, disable_budget: bool = False):
        self.corridor_len_m = float(corridor_len_m)
        self.disable_budget = bool(disable_budget)
        self.budget_retention = 0.85
        self.active_trips: Dict[Tuple[int, int], dict] = {}
        self.completed_nonhold_s: Dict[bool, List[float]] = {True: [], False: []}
        self.completed_usage_hist: Dict[bool, deque] = {
            True: deque([0.25], maxlen=8),
            False: deque([0.25], maxlen=8),
        }
        self.remaining_budget_s: Dict[Tuple[int, int], float] = {}
        self.initial_budget_s: Dict[Tuple[int, int], float] = {}
        self.slack_s: Dict[Tuple[int, int], float] = {}

    def sync_vehicles(self, env: env_bus, now_s: float) -> Tuple[bool, List[Tuple[Tuple[int, int], dict]]]:
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
        finished_keys = [k for k in self.active_trips if k not in live_keys]
        finished_records = []
        for key in finished_keys:
            rec = self.active_trips.pop(key)
            dur = max(60.0, float(now_s) - float(rec["launch_s"]))
            nonhold = max(60.0, dur - float(rec["hold_s"]))
            self.completed_nonhold_s[rec["direction"]].append(nonhold)
            init_b = max(float(self.initial_budget_s.pop(key, 60.0)), 5.0)
            self.completed_usage_hist[rec["direction"]].append(min(1.5, float(rec["hold_s"]) / init_b))
            rec["final_slack_s"] = float(self.slack_s.pop(key, 60.0))
            self.remaining_budget_s.pop(key, None)
            finished_records.append((key, rec))
        return new_launched, finished_records

    def median_nonhold_s(self, direction: bool) -> float:
        hist = self.completed_nonhold_s[bool(direction)]
        if len(hist) >= 2:
            return float(np.median(hist[-8:]))
        return self.corridor_len_m / 6.8 + 230.0

    def cycle_headway_floor_s(self, n_fleet: int, env: env_bus | None = None) -> float:
        cycle_s = self.median_nonhold_s(True) + self.median_nonhold_s(False) + 40.0
        steady_h = cycle_s / max(int(n_fleet), 1)
        if env is not None:
            ready_tot = env._ready_vehicle_count(True) + env._ready_vehicle_count(False)
            buf_factor = 0.85 if ready_tot >= 3 else (0.90 if ready_tot == 2 else 0.97)
        else:
            buf_factor = 0.92
        return float(np.clip(buf_factor * steady_h, 250.0, 335.0))

    def refresh_budgets(self, env: env_bus, now_s: float, lf_stats: dict) -> None:
        if self.disable_budget:
            for key in self.active_trips:
                self.remaining_budget_s[key] = 999.0
                self.initial_budget_s.setdefault(key, 999.0)
                self.slack_s[key] = 180.0
            return

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
        ret = float(self.budget_retention)
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
                    proposed = float(np.clip(0.40 * ret * max(0.0, slack), 0.0, 30.0))
                else:
                    proposed = float(np.clip(ret * max(0.0, slack), 0.0, 85.0))
                self.initial_budget_s.setdefault(key, max(proposed, 5.0))
                prev = self.remaining_budget_s.get(key)
                self.remaining_budget_s[key] = proposed if prev is None else min(prev, proposed)

        for key in self.active_trips:
            if key not in matched_keys and key not in self.remaining_budget_s:
                base_b = 75.0 * ret
                self.remaining_budget_s[key] = base_b
                self.initial_budget_s.setdefault(key, max(base_b, 5.0))
                self.slack_s.setdefault(key, 180.0)

    def debit(self, bus_id: int, trip_id: int, hold_s: float) -> None:
        key = (int(bus_id), int(trip_id))
        if not self.disable_budget and key in self.remaining_budget_s:
            self.remaining_budget_s[key] = max(0.0, self.remaining_budget_s[key] - float(hold_s))
        if key in self.active_trips:
            self.active_trips[key]["hold_s"] += float(hold_s)


def build_upper_state(
    env: env_bus,
    now_s: float,
    n_fleet: int,
    ledger: TurnaroundBudgetLedger,
    apc_feed: CausalHeadwayAPCRateFeed,
    lf_stats: dict,
    prev_interval_readiness_s: float,
    band_mode: str,
    daily_pax_prior: float = 15660.0,
) -> np.ndarray:
    """Build matched 21-dim upper slow-clock state vector."""
    phase = 2.0 * math.pi * (now_s / 50400.0)
    rem_day_frac = float(np.clip((46800.0 - now_s) / 46800.0, 0.0, 1.0))
    h_av_up, rem_tr_up = remaining_quota_info(env, now_s, True)
    h_av_dn, rem_tr_dn = remaining_quota_info(env, now_s, False)

    ready_up = env._ready_vehicle_count(True) / max(n_fleet, 1)
    ready_dn = env._ready_vehicle_count(False) / max(n_fleet, 1)
    on_up = sum(1 for b in env.bus_all if b.on_route and bool(b.direction)) / max(n_fleet, 1)
    on_dn = sum(1 for b in env.bus_all if b.on_route and not bool(b.direction)) / max(n_fleet, 1)
    h_floor_norm = float(np.clip((ledger.cycle_headway_floor_s(n_fleet, env) - 240.0) / 180.0, 0.0, 2.0))
    hold_use_up = float(np.mean(ledger.completed_usage_hist[True]))
    hold_use_dn = float(np.mean(ledger.completed_usage_hist[False]))
    readiness_norm = float(np.clip(prev_interval_readiness_s / 60.0, 0.0, 3.0))

    base = [
        math.sin(phase),
        math.cos(phase),
        rem_day_frac,
        rem_tr_up - rem_day_frac,
        rem_tr_dn - rem_day_frac,
        (h_av_up - 360.0) / 180.0,
        (h_av_dn - 360.0) / 180.0,
        ready_up,
        ready_dn,
        on_up + on_dn,
        h_floor_norm,
        0.5 * (hold_use_up + hold_use_dn),
        readiness_norm,
    ]  # 13 dims

    ref_half = max(0.5 * float(daily_pax_prior) / 783.0, 0.2)
    if band_mode == "aligned":
        band = [
            np.clip(lf_stats["now_up"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["now_dn"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["cur_up"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["cur_dn"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["slope_up"] / ref_half * 10.0, -2.0, 2.0),
            np.clip(lf_stats["slope_dn"] / ref_half * 10.0, -2.0, 2.0),
            np.clip(lf_stats["root_rem_mean"] / lf_stats["root_cur_tot"], 0.4, 2.2),
            np.clip((lf_stats["cur_up"] - lf_stats["cur_dn"]) / lf_stats["cur_tot"], -1.0, 1.0),
        ]
    elif band_mode == "raw":
        h_up = list(apc_feed.dir_raw_hist[True])
        h_dn = list(apc_feed.dir_raw_hist[False])
        band = [
            np.clip(h_up[-1] / ref_half, 0.0, 3.5),
            np.clip(h_dn[-1] / ref_half, 0.0, 3.5),
            np.clip(float(np.mean(h_up[-5:])) / ref_half, 0.0, 3.5),
            np.clip(float(np.mean(h_dn[-5:])) / ref_half, 0.0, 3.5),
            np.clip((h_up[-1] - h_up[-4]) / ref_half, -2.0, 2.0),
            np.clip((h_dn[-1] - h_dn[-4]) / ref_half, -2.0, 2.0),
            np.clip(float(np.mean(h_up[-15:])) / ref_half, 0.0, 3.5),
            np.clip(float(np.mean(h_dn[-15:])) / ref_half, 0.0, 3.5),
        ]
    elif band_mode == "swapped":
        # Upper receives only high-frequency residual statistics
        band = [
            np.clip(lf_stats["hf_pos_up"] / ref_half, 0.0, 3.0),
            np.clip(lf_stats["hf_pos_dn"] / ref_half, 0.0, 3.0),
            np.clip(lf_stats["hf_energy_up"] / ref_half, 0.0, 3.0),
            np.clip(lf_stats["hf_energy_dn"] / ref_half, 0.0, 3.0),
            np.clip(lf_stats["hf_max_up"] / max(ref_half * 0.2, 0.2), 0.0, 3.0),
            np.clip(lf_stats["hf_max_dn"] / max(ref_half * 0.2, 0.2), 0.0, 3.0),
            float(lf_stats["hf_frac_up"]),
            float(lf_stats["hf_frac_dn"]),
        ]
    elif band_mode == "all_band":
        h_up = list(apc_feed.dir_raw_hist[True])
        h_dn = list(apc_feed.dir_raw_hist[False])
        band = [
            np.clip(lf_stats["cur_up"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["cur_dn"] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["hf_pos_up"] / ref_half, 0.0, 3.0),
            np.clip(lf_stats["hf_pos_dn"] / ref_half, 0.0, 3.0),
            np.clip(h_up[-1] / ref_half, 0.0, 3.5),
            np.clip(h_dn[-1] / ref_half, 0.0, 3.5),
            np.clip(lf_stats["root_rem_mean"] / lf_stats["root_cur_tot"], 0.4, 2.2),
            np.clip((lf_stats["hf_pos_up"] - lf_stats["hf_pos_dn"]) / ref_half, -2.0, 2.0),
        ]
    elif band_mode == "nofreq":
        h_up = list(apc_feed.dir_raw_hist[True])
        h_dn = list(apc_feed.dir_raw_hist[False])
        load_up = [len(b.passengers) / max(b.capacity, 1) for b in env.bus_all if b.on_route and bool(b.direction)]
        load_dn = [len(b.passengers) / max(b.capacity, 1) for b in env.bus_all if b.on_route and not bool(b.direction)]
        band = [
            np.clip(float(np.mean(h_up[-15:])) / ref_half, 0.0, 3.5),
            np.clip(float(np.mean(h_dn[-15:])) / ref_half, 0.0, 3.5),
            float(np.mean(load_up)) if load_up else 0.0,
            float(np.mean(load_dn)) if load_dn else 0.0,
            on_up,
            on_dn,
            0.0,
            0.0,
        ]
    else:
        raise ValueError(f"Unknown band_mode: {band_mode}")

    vec = np.asarray(base + band, dtype=np.float32)
    assert vec.shape == (UPPER_STATE_DIM,)
    return vec


def _follower_target_headway_s(env: env_bus, bus: Bus) -> float:
    f_bid = getattr(bus, "pre_action_follower_bus_id", None)
    if f_bid is not None:
        for cand in env.bus_all:
            if int(cand.bus_id) == int(f_bid) and cand.on_route:
                return float(env._get_target_headway_for_bus(cand))
    return float(bus._target_headway)


def build_lower_state(
    env: env_bus,
    bus: Bus,
    now_s: float,
    corridor_len_m: float,
    ledger: TurnaroundBudgetLedger,
    apc_feed: CausalHeadwayAPCRateFeed,
    band_mode: str,
) -> Tuple[np.ndarray, dict]:
    """Build matched 20-dim lower stop-event state vector (index 0 = budget limit ratio)."""
    key = (int(bus.bus_id), int(bus.trip_id))
    rem_b = float(ledger.remaining_budget_s.get(key, 30.0))
    slack = float(ledger.slack_s.get(key, 120.0))
    cap_s = min(LOWER_MAX_HOLD_S, max(0.0, rem_b))
    limit_ratio = float(np.clip(cap_s / LOWER_MAX_HOLD_S, 0.0, 1.0))

    cur_idx = bus.effective_station.index(bus.last_station) if bus.last_station in bus.effective_station else 1
    rem_stops = max(1, len(bus.effective_station) - 1 - cur_idx)

    opp_d = not bool(bus.direction)
    overdue_opp = [
        max(0.0, now_s - float(getattr(tt, "_freqduet_scheduled_launch", tt.launch_time)))
        for tt in env.timetables
        if not tt.launched and bool(tt.direction) == opp_d
        and float(getattr(tt, "_freqduet_scheduled_launch", tt.launch_time)) <= now_s
    ]
    opp_overdue_norm = float(np.clip(max(overdue_opp, default=0.0) / 180.0, 0.0, 2.0))

    fwd_valid = (
        bus.pre_action_forward_headway_source == "matched_departure_event"
        and bus.pre_action_forward_headway is not None
    )
    bwd_valid = (
        str(bus.pre_action_follower_source).startswith("same_time_avl_")
        and bus.pre_action_follower_departure_gap is not None
    )
    fwd = float(bus.pre_action_forward_headway) if fwd_valid else float(bus._target_headway)
    bwd = float(bus.pre_action_follower_departure_gap) if bwd_valid else None
    h_own = max(float(bus._target_headway), 60.0)
    h_fol = _follower_target_headway_s(env, bus)

    _, _, u_star, two_sided_ok = plan_consistent_two_sided_loss_and_target(
        forward_headway_s=(fwd if fwd_valid else None),
        follower_departure_gap_s=bwd,
        own_target_s=h_own,
        follower_target_s=h_fol,
        action_s=0.0,
        action_cap_s=LOWER_MAX_HOLD_S,
    )

    fwd_rel = float(np.clip((fwd - h_own) / h_own, -1.0, 2.0))
    bwd_rel = float(np.clip((bwd - h_fol) / h_fol, -1.0, 2.0)) if bwd is not None else 0.0
    load_ratio = float(len(bus.passengers) / max(float(bus.capacity), 1.0))
    left_behind_norm = float(np.clip(float(bus.last_left_behind_count) / 20.0, 0.0, 2.0))
    prog = float(np.clip(bus.travel_distance / max(corridor_len_m, 1.0), 0.0, 1.0))
    phase = 2.0 * math.pi * (now_s / 50400.0)

    base16 = [
        limit_ratio,  # [0] HARD ACTION MASK FEATURE INDEX!
        float(np.clip(rem_b / 75.0, 0.0, 1.5)),
        float(np.clip((rem_b / rem_stops) / 12.0, 0.0, 2.0)),
        float(np.clip(slack / 120.0, -1.0, 2.0)),
        opp_overdue_norm,
        fwd_rel,
        bwd_rel,
        1.0 if two_sided_ok else 0.0,
        float(np.clip(u_star / LOWER_MAX_HOLD_S, 0.0, 1.0)),
        (h_own - 360.0) / 180.0,
        (h_fol - 360.0) / 180.0,
        load_ratio,
        left_behind_norm,
        prog,
        math.sin(phase),
        math.cos(phase),
    ]

    # Downstream 2 stops for localized lookahead along route
    direction = bool(bus.direction)
    st_here = int(bus.last_station.station_id)
    ds_ids = [
        int(bus.effective_station[i].station_id)
        for i in range(cur_idx + 1, min(len(bus.effective_station) - 1, cur_idx + 3))
    ]
    tracker = env.frequency_tracker
    loc_state = tracker._get_local_state((st_here, direction))
    ref_st = max(float(loc_state.low), 0.5)

    if band_mode == "aligned":
        hf_here = max(0.0, float(loc_state.high)) / ref_st
        hf_en = math.sqrt(max(0.0, float(loc_state.high_energy))) / ref_st
        ds_hf = [
            max(0.0, float(tracker._get_local_state((sid, direction)).high))
            / max(float(tracker._get_local_state((sid, direction)).low), 0.5)
            for sid in ds_ids
        ]
        promo_age = float(tracker.local_promotion_summary(st_here, direction).get("age", 0.0))
        band4 = [
            float(np.clip(hf_here, 0.0, 3.0)),
            float(np.clip(hf_en, 0.0, 3.0)),
            float(np.clip(max(ds_hf, default=0.0), 0.0, 3.0)),
            float(np.clip(promo_age, 0.0, 1.0)),
        ]
    elif band_mode == "raw":
        rh = list(apc_feed.raw_rate_hist.get((st_here, direction), [1.0, 1.0]))
        ds_raw = [
            float(np.mean(list(apc_feed.raw_rate_hist.get((sid, direction), [1.0]))[-2:]))
            for sid in ds_ids
        ]
        band4 = [
            float(np.clip(rh[-1] / 3.0, 0.0, 3.0)),
            float(np.clip(float(np.mean(rh[-3:])) / 3.0, 0.0, 3.0)),
            float(np.clip(max(ds_raw, default=0.0) / 3.0, 0.0, 3.0)),
            float(np.clip((rh[-1] - rh[-2]) / 3.0, -2.0, 2.0)),
        ]
    elif band_mode == "swapped":
        ds_lf = [float(tracker._get_local_state((sid, direction)).low) / 3.0 for sid in ds_ids]
        band4 = [
            float(np.clip(float(loc_state.low) / 3.0, 0.0, 3.0)),
            float(np.clip(float(loc_state.low_slope) * 5.0, -2.0, 2.0)),
            float(np.clip(max(ds_lf, default=0.0), 0.0, 3.0)),
            float(np.clip(float(loc_state.forecast(15)) / max(ref_st, 0.5), 0.0, 3.0)),
        ]
    elif band_mode == "all_band":
        rh = list(apc_feed.raw_rate_hist.get((st_here, direction), [1.0]))
        band4 = [
            float(np.clip(float(loc_state.low) / 3.0, 0.0, 3.0)),
            float(np.clip(max(0.0, float(loc_state.high)) / ref_st, 0.0, 3.0)),
            float(np.clip(rh[-1] / 3.0, 0.0, 3.0)),
            float(np.clip(float(loc_state.forecast(15)) / 3.0, 0.0, 3.0)),
        ]
    elif band_mode == "nofreq":
        band4 = [
            float(np.clip(float(bus.last_board_count) / 20.0, 0.0, 2.0)),
            float(np.clip(float(bus.last_service_dwell_s) / 30.0, 0.0, 2.0)),
            0.0,
            0.0,
        ]
    else:
        raise ValueError(f"Unknown band_mode: {band_mode}")

    vec = np.asarray(base16 + band4, dtype=np.float32)
    assert vec.shape == (LOWER_STATE_DIM,)
    ctx = {
        "fwd": fwd if fwd_valid else None,
        "bwd": bwd,
        "h_own": h_own,
        "h_fol": h_fol,
        "load_ratio": load_ratio,
        "onboard_pax": float(len(bus.passengers)),
        "rem_stops": float(rem_stops),
        "slack_s": slack,
        "board_here": float(bus.last_board_count),
    }
    return vec, ctx


def compute_lower_reward_and_cost(
    ctx: dict,
    action_s: float,
    ds_rate_s: float,
    terminal_slack_s: float | None = None,
) -> Tuple[float, float]:
    """100% frequency-tracker-free physical person-seconds step reward and CMDP cost."""
    u = float(action_s)
    if u <= 1e-6:
        return 0.0, 0.0

    l0, lu, _, two_sided_ok = plan_consistent_two_sided_loss_and_target(
        forward_headway_s=ctx["fwd"],
        follower_departure_gap_s=ctx["bwd"],
        own_target_s=ctx["h_own"],
        follower_target_s=ctx["h_fol"],
        action_s=u,
        action_cap_s=LOWER_MAX_HOLD_S,
    )

    h_own = float(ctx["h_own"])
    h_fol = float(ctx["h_fol"])
    if two_sided_ok and ctx["fwd"] is not None and ctx["bwd"] is not None:
        gap_term = (float(ctx["bwd"]) / h_fol) - (float(ctx["fwd"]) / h_own)
        curv_term = (1.0 / h_own) + (1.0 / h_fol)
        wait_saved_pax_s = float(ds_rate_s) * 0.5 * (h_own + h_fol) * u * (gap_term - 0.5 * u * curv_term)
    elif ctx["fwd"] is not None:
        wait_saved_pax_s = float(ds_rate_s) * u * (0.75 * (h_own - float(ctx["fwd"])) - 0.5 * u)
    else:
        wait_saved_pax_s = 0.0

    onboard_delay_pax_s = float(ctx["onboard_pax"]) * u
    slack_shadow_pax_s = 10.0 * max(0.0, (40.0 - float(ctx["slack_s"])) / 40.0) * u
    term_pen_pax_s = 0.0
    if terminal_slack_s is not None and terminal_slack_s < 0.0:
        term_pen_pax_s = 16.0 * min(2.0, abs(float(terminal_slack_s)) / 30.0) * u

    net_pax_s = wait_saved_pax_s - onboard_delay_pax_s - slack_shadow_pax_s - term_pen_pax_s
    reward = float(np.clip(net_pax_s / 150.0, -2.5, 2.5))
    overshoot_regret = max(0.0, lu - l0)
    load_excess_cost = 0.25 * max(0.0, float(ctx["load_ratio"]) - 0.55) * (u / LOWER_MAX_HOLD_S)
    tight_slack_cost = 0.35 * (1.0 if float(ctx["slack_s"]) < 15.0 else 0.0) * (u / LOWER_MAX_HOLD_S)
    cost = float(min(8.0 * overshoot_regret + load_excess_cost + tight_slack_cost, 2.0))
    return reward, cost


def flush_trip_lower_transitions(
    traj: List[dict],
    final_slack_s: float,
    lower_replay: CostReplayBuffer | None,
) -> Tuple[List[float], List[float]]:
    """Settle a completed trip's stop transitions using exact realized downstream arrival rates."""
    r_list: List[float] = []
    c_list: List[float] = []
    if not traj or lower_replay is None:
        return r_list, c_list
    m_len = len(traj)
    for idx, step in enumerate(traj):
        ds_rate_s = sum(
            float(st_next["board_here"]) / float(np.clip(st_next["arr_hw"], 90.0, 900.0))
            for st_next in traj[idx + 1 :]
        )
        is_last = (idx + 1) == m_len
        r_l, c_l = compute_lower_reward_and_cost(
            step["ctx"],
            step["action_s"],
            ds_rate_s,
            terminal_slack_s=(float(final_slack_s) if is_last else None),
        )
        if is_last:
            next_s = step["state"].copy()
            next_s[0] = 0.0
        else:
            next_s = traj[idx + 1]["state"]
        lower_replay.push(
            step["state"],
            step["action_s"],
            r_l,
            c_l,
            next_s,
            bool(is_last),
            trip_id=int(step["trip_id"]),
        )
        r_list.append(r_l)
        c_list.append(c_l)
    return r_list, c_list


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


def run_b1_episode(
    *,
    corridor_dir: Path,
    corridor_meta: dict,
    day_seed: int,
    upper_trainer: RESACUpperTrainer | None,
    lower_trainer: RESACLagrangianTrainer | None,
    lower_replay: CostReplayBuffer | None,
    eval_mode: bool,
    band_mode: str = "aligned",
    ablation: str = "none",
    baseline_arm: str | None = None,
    speed_delta_mps: float = 0.0,
    peak_shift_override: int | None = None,
    burst_scale: float = 1.0,
) -> dict:
    """Run one complete service day (either RL train/eval or matched baseline arm)."""
    install_simulator_patches()
    Bus._v7_default_capacity = int(corridor_meta.get("bus_capacity", 50))
    n_fleet = int(corridor_meta["n_fleet"])
    disable_budget = ablation == "no_budget"
    enable_promotion = ablation != "no_promotion"

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
    env.peak_shift_choices = [int(peak_shift_override)] if peak_shift_override is not None else [0]
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
        "harmonic_prior_var": 0.35,
        "forecast_horizon_s": 1800.0,
        "forecast_mode": "causal",
        "upper_features": True,
        "lower_features": True,
        "upper_mode": "low",
        "lower_mode": "high",
        "promotion": {
            "enable": bool(enable_promotion),
            "window_min": 10.0,
            "cooldown_min": 15.0,
            "residual_threshold": 1.25,
            "residual_mode": "positive",
            "persistence_ratio": 0.55,
            "adapt_low": bool(enable_promotion),
            "adapt_gain": 0.16,
            "adapt_strength_min": 0.2,
            "adapt_local": bool(enable_promotion),
        },
        "lower_context": {
            "enable": True,
            "features": ["load", "capacity", "queue", "speed_residual", "shock_age", "schedule_slack"],
        },
    })
    env._upper_policy_callback = lambda _s, trip: float(getattr(trip, "target_headway", 360.0))

    env.reset()
    if abs(speed_delta_mps) > 1e-6:
        for route in env.routes:
            route.speed_history = (route.speed_history.astype(float) + float(speed_delta_mps)).clip(lower=3.5)

    apc_feed = CausalHeadwayAPCRateFeed(env)
    for tt in env.timetables:
        tt._freqduet_base_launch = float(tt.launch_time)
        tt._freqduet_scheduled_launch = float(tt.launch_time)
        tt._freqduet_projected_target_headway = 360.0
        tt.target_headway = 360.0
        tt._freqduet_terminal_dispatch = True
        tt._freqduet_min_dispatch_headway = 120.0
    bursts = attach_day_bursts(env, corridor_meta["name"], burst_scale=burst_scale)

    corridor_len_m = float(sum(float(r.distance) for r in env.routes[: len(env.routes) // 2]))
    ledger = TurnaroundBudgetLedger(corridor_len_m=corridor_len_m, disable_budget=disable_budget)

    daily_pax_prior = float(corridor_meta.get("daily_pax", 18000.0))
    p_ref_interval = max(daily_pax_prior / 52.0, 50.0)
    n_active_stops = max(sum(1 for s in env.stations if s.station_type != 0 and s.od is not None), 10)
    mean_stop_rate = max(daily_pax_prior / (14.0 * 60.0 * n_active_stops), 0.25)

    last_replan_cycle = -1
    last_budget_refresh_s = -999.0
    cached_lf_stats = causal_lf_allocation_stats(env, 0.0)

    # Interval accumulators for estimator-free Upper reward with Ng-Harada-Russell quota potential
    pending_upper = None
    ep_upper_steps: List[dict] = []
    int_wait_pax_s = 0.0
    int_hold_pax_s = 0.0
    int_ev_start_idx = 0
    int_ready_start_idx = 0
    prev_int_readiness_s = 0.0
    upper_rewards = []

    # Per-(bus_id, trip_id) trajectory buffer for exact realized downstream arrival rate settlement
    trip_trajs: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    lower_rewards = []
    lower_costs = []
    total_hold_s = 0.0
    bins15 = np.asarray([0.0, 5.0, 10.0, 15.0], dtype=np.float64)

    t0 = time.monotonic()
    while not env.done:
        now_s = float(env.current_time)
        new_launched, finished_trips = ledger.sync_vehicles(env, now_s)

        # Settle all stop transitions for trips that just arrived at terminal
        if baseline_arm is None and not eval_mode and lower_replay is not None:
            for t_key, rec in finished_trips:
                traj = trip_trajs.pop(t_key, [])
                rs, cs = flush_trip_lower_transitions(
                    traj, float(rec.get("final_slack_s", 60.0)), lower_replay
                )
                lower_rewards.extend(rs)
                lower_costs.extend(cs)

        # Slow clock every 900s during active service
        if baseline_arm is None and now_s <= 45900.0:
            cycle = int(now_s // 900.0)
            if cycle != last_replan_cycle:
                last_replan_cycle = cycle
                cached_lf_stats = causal_lf_allocation_stats(env, now_s)

                # Settle previous 900s upper interval
                new_ready = env._fleet_readiness_delays_s[int_ready_start_idx:]
                prev_int_readiness_s = float(np.mean(new_ready)) if new_ready else 0.0
                new_evs = env.headway_events.events[int_ev_start_idx:]
                int_ewt = compute_headway_metrics(new_evs)["ewt_s"] if new_evs else 0.0
                phi_now = quota_future_wait_potential_pax_s(env, now_s, daily_pax_prior)

                s_u = build_upper_state(
                    env, now_s, n_fleet, ledger, apc_feed, cached_lf_stats,
                    prev_int_readiness_s, band_mode, daily_pax_prior,
                )
                if pending_upper is not None:
                    shaped_pax_s = -(int_wait_pax_s + int_hold_pax_s) + (UPPER_GAMMA * phi_now - pending_upper["phi"])
                    r_u = float(np.clip(
                        shaped_pax_s / (360.0 * p_ref_interval)
                        - 0.08 * (float(np.clip(int_ewt, -20.0, 120.0)) / 60.0)
                        - 0.16 * (float(np.clip(prev_int_readiness_s, 0.0, 240.0)) / 60.0),
                        -2.5,
                        2.5,
                    ))
                    upper_rewards.append(r_u)
                    ep_upper_steps.append({
                        "state": pending_upper["state"],
                        "action": pending_upper["action"],
                        "reward": r_u,
                    })

                int_wait_pax_s = 0.0
                int_hold_pax_s = 0.0
                int_ev_start_idx = len(env.headway_events.events)
                int_ready_start_idx = len(env._fleet_readiness_delays_s)

                a_u = upper_trainer.policy_net.get_action(s_u, deterministic=eval_mode)
                loop_factor = float(a_u[0])
                dir_tilt_s = float(a_u[1])
                ledger.budget_retention = float(a_u[2])

                h_floor = ledger.cycle_headway_floor_s(n_fleet, env)
                desired_hw = {}
                for d in (True, False):
                    h_avail, _ = remaining_quota_info(env, now_s, d)
                    ready_d = env._ready_vehicle_count(d)
                    ready_opp = env._ready_vehicle_count(not d)
                    inv_guard = float(np.clip(0.55 + 0.35 * (ready_d - ready_opp), 0.15, 1.0))
                    sign_d = 1.0 if d else -1.0
                    raw_h = h_avail * loop_factor + sign_d * dir_tilt_s * inv_guard
                    desired_hw[d] = float(np.clip(raw_h, h_floor, 455.0))

                apply_global_quota_timetable(
                    env, now_s, desired_hw, h_bounds=(h_floor, 455.0), tail_bounds=(max(245.0, h_floor - 10.0), 445.0)
                )
                ledger.refresh_budgets(env, now_s, cached_lf_stats)
                last_budget_refresh_s = now_s
                pending_upper = {"state": s_u, "action": a_u, "phi": phi_now}

        if baseline_arm is None and (new_launched or (now_s - last_budget_refresh_s) >= 60.0):
            ledger.refresh_budgets(env, now_s, fast_directional_lf_now(env))
            last_budget_refresh_s = now_s

        # Stop-event lower decisions
        act_dict = {bus.bus_id: None for bus in env.bus_all}
        for bus in env.bus_all:
            if not bus.on_route or bus.state != BusState.WAITING_ACTION:
                continue

            if baseline_arm == "A0_fixed_nohold":
                chosen = 0.0
            elif baseline_arm == "A1_fixed_bilateral":
                fwd_ok = (
                    bus.pre_action_forward_headway_source == "matched_departure_event"
                    and bus.pre_action_forward_headway is not None
                )
                bwd_ok = (
                    str(bus.pre_action_follower_source).startswith("same_time_avl_")
                    and bus.pre_action_follower_departure_gap is not None
                )
                req = causal_two_sided_holding_target_s(
                    forward_headway_s=(bus.pre_action_forward_headway if fwd_ok else None),
                    follower_departure_gap_s=(bus.pre_action_follower_departure_gap if bwd_ok else None),
                    action_cap_s=15.0,
                )
                chosen = float(bins15[np.argmin(np.abs(bins15 - req))]) if req is not None else 0.0
            else:
                t_key = (int(bus.bus_id), int(bus.trip_id))
                s_l, ctx = build_lower_state(env, bus, now_s, corridor_len_m, ledger, apc_feed, band_mode)
                chosen = float(lower_trainer.policy_net.get_action(s_l, deterministic=eval_mode))
                if not eval_mode and lower_replay is not None:
                    raw_arr = getattr(bus, "forward_headway", None)
                    arr_hw = (
                        float(raw_arr)
                        if (raw_arr is not None and math.isfinite(float(raw_arr)) and float(raw_arr) > 0.0)
                        else float(bus._target_headway)
                    )
                    trip_trajs[t_key].append({
                        "state": s_l,
                        "action_s": chosen,
                        "ctx": ctx,
                        "board_here": float(bus.last_board_count),
                        "arr_hw": arr_hw,
                        "trip_id": int(t_key[1]),
                    })

            act_dict[bus.bus_id] = chosen
            if chosen > 0.0:
                total_hold_s += chosen
            ledger.debit(bus.bus_id, bus.trip_id, chosen)

        # Accumulate physical 1-second waiting & discretionary holding person-seconds
        int_wait_pax_s += float(sum(len(st.waiting_passengers) for st in env.stations))
        int_hold_pax_s += float(sum(
            len(b.passengers) for b in env.bus_all
            if b.on_route and b.state == BusState.DWELLING and (b.dwelling_time or 0.0) > 0.0
        ))

        env.step(act_dict, debug=False, render=False)
        apc_feed.observe_tick()

    # Flush any residual lower trajectories at episode end
    if baseline_arm is None and not eval_mode and lower_replay is not None:
        for t_key, traj in list(trip_trajs.items()):
            rs, cs = flush_trip_lower_transitions(traj, 60.0, lower_replay)
            lower_rewards.extend(rs)
            lower_costs.extend(cs)
        trip_trajs.clear()

    # Settle final upper transition & push exact N-step corridor-cycle Bellman transitions
    if baseline_arm is None and pending_upper is not None:
        new_ready = env._fleet_readiness_delays_s[int_ready_start_idx:]
        fin_readiness_s = float(np.mean(new_ready)) if new_ready else 0.0
        new_evs = env.headway_events.events[int_ev_start_idx:]
        fin_ewt = compute_headway_metrics(new_evs)["ewt_s"] if new_evs else 0.0
        shaped_fin = -(int_wait_pax_s + int_hold_pax_s) + (0.0 - pending_upper["phi"])
        r_u = float(np.clip(
            shaped_fin / (360.0 * p_ref_interval)
            - 0.08 * (float(np.clip(fin_ewt, -20.0, 120.0)) / 60.0)
            - 0.16 * (float(np.clip(fin_readiness_s, 0.0, 240.0)) / 60.0),
            -2.5,
            2.5,
        ))
        upper_rewards.append(r_u)
        ep_upper_steps.append({
            "state": pending_upper["state"],
            "action": pending_upper["action"],
            "reward": r_u,
        })
        if not eval_mode:
            s_term = np.zeros(UPPER_STATE_DIM, dtype=np.float32)
            n_u = len(ep_upper_steps)
            for idx_u in range(n_u):
                m_steps = min(UPPER_N_STEP, n_u - idx_u)
                r_mult = sum(
                    (UPPER_GAMMA ** j) * float(ep_upper_steps[idx_u + j]["reward"])
                    for j in range(m_steps)
                )
                if idx_u + m_steps < n_u:
                    ns_u = ep_upper_steps[idx_u + m_steps]["state"]
                    done_u = 0.0
                else:
                    ns_u = s_term
                    done_u = 1.0
                upper_trainer.replay_buffer.push(
                    ep_upper_steps[idx_u]["state"],
                    ep_upper_steps[idx_u]["action"],
                    float(r_mult),
                    ns_u,
                    done_u,
                    float(m_steps),
                )

    wall_s = time.monotonic() - t0
    details = dict(env._measurement_details)
    hw_metrics = compute_headway_metrics(env.headway_events.events)
    launched_count = sum(1 for tt in env.timetables if tt.launched)
    completed_count = len(env._completed_trip_ids)
    readiness_mean = float(np.mean(env._fleet_readiness_delays_s)) if env._fleet_readiness_delays_s else 0.0
    readiness_p95 = float(np.percentile(env._fleet_readiness_delays_s, 95)) if env._fleet_readiness_delays_s else 0.0

    return {
        "day": int(day_seed),
        "scenario_tape_id": env.scenario_tape.identifier,
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
        "rl_stats": {
            "upper_return": round(float(np.sum(upper_rewards)), 4) if upper_rewards else 0.0,
            "lower_reward_mean": round(float(np.mean(lower_rewards)), 5) if lower_rewards else 0.0,
            "lower_cost_mean": round(float(np.mean(lower_costs)), 5) if lower_costs else 0.0,
        },
    }


def make_trainers(seed: int) -> Tuple[RESACUpperTrainer, RESACLagrangianTrainer, CostReplayBuffer]:
    candidates = build_upper_action_candidates()
    upper = RESACUpperTrainer(
        state_dim=UPPER_STATE_DIM,
        action_dim=3,
        hidden_dim=96,
        action_candidates=candidates,
        discrete_critic="indexed",
        ensemble_size=6,
        beta=-1.0,
        beta_ood=0.01,
        weight_reg=1e-4,
        weight_reg_mode="mean",
        lr=3e-4,
        gamma=UPPER_GAMMA,
        soft_tau=2e-2,
        auto_entropy=True,
        temperature_contract="bounded_log_parameter_v4",
        initial_alpha=0.06,
        minimum_alpha=0.004,
        maximum_alpha=0.20,
        policy_sample_seed=seed * 10 + 1,
        replay_seed=seed * 10 + 2,
        replay_capacity=20000,
        device="cpu",
    )
    lower = RESACLagrangianTrainer(
        state_dim=LOWER_STATE_DIM,
        action_dim=1,
        hidden_dim=96,
        action_range=LOWER_MAX_HOLD_S,
        action_bins=LOWER_ACTION_BINS,
        action_limit_feature_index=0,
        discrete_critic="zero_hold_advantage",
        cost_limit=0.020,
        ensemble_size=6,
        beta=-1.0,
        beta_ood=0.01,
        weight_reg=1e-4,
        weight_reg_mode="mean",
        lr=3e-4,
        lambda_lr=1e-3,
        gamma=0.92,
        soft_tau=1e-2,
        auto_entropy=True,
        temperature_contract="bounded_log_parameter_v4",
        initial_alpha=0.06,
        minimum_alpha=0.003,
        maximum_alpha=0.20,
        policy_sample_seed=seed * 10 + 3,
        device="cpu",
    )
    upper.target_entropy = 0.35 * float(math.log(len(candidates)))
    lower.target_entropy = 0.35 * float(math.log(len(LOWER_ACTION_BINS)))
    lower.log_lambda.data.fill_(float(math.log(0.05)))
    lower_replay = CostReplayBuffer(capacity=180000, seed=seed * 10 + 4)
    return upper, lower, lower_replay


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def evaluate_frozen(
    corridor_dir: Path,
    corridor_meta: dict,
    val_days: List[int],
    upper_trainer: RESACUpperTrainer,
    lower_trainer: RESACLagrangianTrainer,
    band_mode: str,
    ablation: str,
) -> dict:
    per_day = []
    for d in val_days:
        row = run_b1_episode(
            corridor_dir=corridor_dir,
            corridor_meta=corridor_meta,
            day_seed=int(d),
            upper_trainer=upper_trainer,
            lower_trainer=lower_trainer,
            lower_replay=None,
            eval_mode=True,
            band_mode=band_mode,
            ablation=ablation,
        )
        per_day.append(row)
    keys = list(per_day[0]["metrics"].keys())
    means = {k: round(float(np.mean([r["metrics"][k] for r in per_day])), 6) for k in keys}
    return {"means": means, "per_day": per_day}


def train_seed(
    *,
    corridor_dir: Path,
    corridor_: str,
    seed: int,
    episodes: int,
    val_days: List[int],
    band_mode: str,
    ablation: str,
    ckpt_dir: Path,
    result_dir: Path,
    resume_path: Path | None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    corridor_meta = json.loads((corridor_dir / "corridor_meta.json").read_text())
    upper, lower, lower_replay = make_trainers(seed)

    start_ep = 1
    train_curve: List[dict] = []
    checkpoint_evals: Dict[str, dict] = {}
    latest_ckpt = ckpt_dir / "latest.pt"
    load_target = resume_path if (resume_path is not None and resume_path.exists()) else (latest_ckpt if (resume_path is not None and latest_ckpt.exists()) else None)

    if load_target is not None:
        state = torch.load(load_target, map_location="cpu", weights_only=False)
        start_ep = int(state["episode"]) + 1
        upper.load_training_state_dict(state["upper"])
        lower.load_training_state_dict(state["lower"])
        lower_replay.load_state_dict(state["lower_replay"])
        random.setstate(state["py_rng"])
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        train_curve = list(state.get("train_curve", []))
        checkpoint_evals = dict(state.get("checkpoint_evals", {}))
        print(f"Resumed {corridor_} seed={seed} from episode {start_ep}", flush=True)

    t_start = time.monotonic()
    eval_milestones = {40, 100, episodes}

    for ep in range(start_ep, episodes + 1):
        day_seed = 1_000_000 + int(seed) * 10_000 + int(ep)
        ep_out = run_b1_episode(
            corridor_dir=corridor_dir,
            corridor_meta=corridor_meta,
            day_seed=day_seed,
            upper_trainer=upper,
            lower_trainer=lower,
            lower_replay=lower_replay,
            eval_mode=False,
            band_mode=band_mode,
            ablation=ablation,
        )

        u_upd, l_upd = {}, {}
        if ep >= 3 and len(upper.replay_buffer) >= 48:
            for _ in range(32):
                u_upd = upper.update(batch_size=48)
        if ep >= 3 and len(lower_replay) >= 256:
            for _ in range(64):
                l_upd = lower.update(lower_replay, batch_size=128, reward_scale=2.0)

        train_curve.append({
            "episode": ep,
            "journey_s": round(ep_out["metrics"]["restricted_journey_s_per_pax"], 3),
            "wait_s": round(ep_out["metrics"]["restricted_wait_s_per_pax"], 3),
            "ewt_s": round(ep_out["metrics"]["ewt_s"], 3),
            "plan_cv2": round(ep_out["metrics"]["plan_cv2"], 5),
            "readiness_s": round(ep_out["metrics"]["readiness_delay_s_per_trip"], 3),
            "hold_s": round(ep_out["metrics"]["hold_s_per_trip"], 3),
            "upper_return": ep_out["rl_stats"]["upper_return"],
            "lower_lambda": round(float(l_upd.get("lambda", lower.log_lambda.exp().item())), 5),
            "upper_alpha": round(float(upper.alpha), 5),
            "lower_alpha": round(float(lower.alpha), 5),
        })

        if ep in eval_milestones:
            tag = f"ep{ep}"
            frozen_dir = result_dir.parent / "frozen_models"
            frozen_dir.mkdir(parents=True, exist_ok=True)
            upper.save(frozen_dir / f"upper_{tag}.pt")
            lower.save(frozen_dir / f"lower_{tag}.pt")
            checkpoint_evals[tag] = evaluate_frozen(
                corridor_dir, corridor_meta, val_days, upper, lower, band_mode, ablation
            )

        if ep % 10 == 0 or ep in eval_milestones:
            atomic_torch_save(latest_ckpt, {
                "episode": ep,
                "upper": upper.training_state_dict(),
                "lower": lower.training_state_dict(),
                "lower_replay": lower_replay.state_dict(),
                "py_rng": random.getstate(),
                "np_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "train_curve": train_curve,
                "checkpoint_evals": checkpoint_evals,
            })

        print("SCHEDULEURM_PROGRESS " + json.dumps({
            "current": ep,
            "total": episodes,
            "unit": "episode",
            "elapsed_seconds": round(time.monotonic() - t_start, 2),
        }), flush=True)

    summary = {
        "completed": True,
        "corridor": corridor_,
        "seed": int(seed),
        "band_mode": band_mode,
        "ablation": ablation,
        "episodes": int(episodes),
        "val_days": val_days,
        "checkpoint_evals": checkpoint_evals,
        "train_curve": train_curve,
    }
    write_json(result_dir / "results.json", summary)
    print(f"Training complete: {corridor_} seed={seed} finished {episodes} episodes.", flush=True)


def run_baselines(corridor_dir: Path, corridor_: str, val_days: List[int], result_dir: Path) -> None:
    corridor_meta = json.loads((corridor_dir / "corridor_meta.json").read_text())
    arms = ("A0_fixed_nohold", "A1_fixed_bilateral")
    t_start = time.monotonic()
    out_arms = {}
    step = 0
    total = len(arms) * len(val_days)
    for arm in arms:
        rows = []
        for d in val_days:
            rows.append(run_b1_episode(
                corridor_dir=corridor_dir,
                corridor_meta=corridor_meta,
                day_seed=int(d),
                upper_trainer=None,
                lower_trainer=None,
                lower_replay=None,
                eval_mode=True,
                baseline_arm=arm,
            ))
            step += 1
            print("SCHEDULEURM_PROGRESS " + json.dumps({
                "current": step,
                "total": total,
                "unit": "episode",
                "elapsed_seconds": round(time.monotonic() - t_start, 2),
            }), flush=True)
        keys = list(rows[0]["metrics"].keys())
        out_arms[arm] = {
            "means": {k: round(float(np.mean([r["metrics"][k] for r in rows])), 6) for k in keys},
            "per_day": rows,
        }
    write_json(result_dir / "results.json", {
        "completed": True,
        "corridor": corridor_,
        "val_days": val_days,
        "baselines": out_arms,
    })


def preflight(corridors_root: Path, result_dir: Path) -> None:
    install_simulator_patches()
    u, l, buf = make_trainers(101)
    assert u.discrete_actions.shape == (30, 3)
    assert l.discrete_actions.shape == (len(LOWER_ACTION_BINS), 1)
    for c in ("tidal_bimodal", "mbta_route111_wkdy"):
        assert (corridors_root / c / "corridor_meta.json").exists(), c
    write_json(result_dir / "results.json", {"passed": True, "physical_starts": 0, "len_buf": len(buf)})
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "check", "elapsed_seconds": 0.1}), flush=True)


def aggregate(root: Path, corridors: List[str], seeds: List[int], val_days: List[int], result_dir: Path) -> None:
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(val_days), size=(10000, len(val_days)))
    by_corridor = {}

    for c in corridors:
        base_json = json.loads((root / f"baselines_{c}" / "summary" / "results.json").read_text())["baselines"]
        seed_jsons = [
            json.loads((root / f"train_{c}_s{s}" / "summary" / "results.json").read_text())
            for s in seeds
        ]

        milestone_summary = {}
        for tag in ("ep40", "ep100", "ep200"):
            day_seed_means = []
            for d_idx, _ in enumerate(val_days):
                m_dict = {}
                for k in ("restricted_journey_s_per_pax", "restricted_wait_s_per_pax", "restricted_in_vehicle_s_per_pax",
                          "ewt_s", "plan_cv2", "raw_cv2", "readiness_delay_s_per_trip", "hold_s_per_trip"):
                    m_dict[k] = float(np.mean([
                        sj["checkpoint_evals"][tag]["per_day"][d_idx]["metrics"][k]
                        for sj in seed_jsons
                    ]))
                day_seed_means.append(m_dict)
            milestone_summary[tag] = {
                "across_seed_mean": {
                    k: round(float(np.mean([d[k] for d in day_seed_means])), 6)
                    for k in day_seed_means[0]
                },
                "across_seed_std": {
                    k: round(float(np.std([sj["checkpoint_evals"][tag]["means"][k] for sj in seed_jsons], ddof=1)), 6)
                    for k in day_seed_means[0]
                },
                "per_day_seed_mean": day_seed_means,
            }

        # Paired day bootstrap: ep200 vs A1, ep200 vs A0, and stability ep200 - ep40
        def _boot_contrast(a_days: List[dict], b_days: List[dict]) -> dict:
            out_c = {}
            for k in ("restricted_journey_s_per_pax", "restricted_wait_s_per_pax", "ewt_s", "plan_cv2", "readiness_delay_s_per_trip"):
                deltas = np.asarray([a_days[i][k] - b_days[i][k] for i in range(len(val_days))], dtype=np.float64)
                b_means = deltas[draws].mean(axis=1)
                out_c[k] = {
                    "mean": round(float(deltas.mean()), 6),
                    "ci95": [round(float(x), 6) for x in np.quantile(b_means, [0.025, 0.975])],
                    "days_improved": int((deltas < 0.0).sum()),
                }
            return out_c

        ep200_days = milestone_summary["ep200"]["per_day_seed_mean"]
        ep40_days = milestone_summary["ep40"]["per_day_seed_mean"]
        a1_days = [r["metrics"] for r in base_json["A1_fixed_bilateral"]["per_day"]]
        a0_days = [r["metrics"] for r in base_json["A0_fixed_nohold"]["per_day"]]

        contrasts = {
            "ep200_minus_A1_fixed_bilateral": _boot_contrast(ep200_days, a1_days),
            "ep200_minus_A0_fixed_nohold": _boot_contrast(ep200_days, a0_days),
            "stability_ep200_minus_ep40": _boot_contrast(ep200_days, ep40_days),
        }
        for tag in ("ep40", "ep100", "ep200"):
            milestone_summary[tag].pop("per_day_seed_mean", None)

        by_corridor[c] = {
            "n_seeds": len(seeds),
            "n_val_days": len(val_days),
            "baselines": {k: v["means"] for k, v in base_json.items()},
            "b1_milestones": milestone_summary,
            "paired_contrasts": contrasts,
        }

    write_json(result_dir / "results.json", {
        "completed": True,
        "experiment": "v501_pathb_b1_stability",
        "by_corridor": by_corridor,
    })
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "aggregate", "elapsed_seconds": 0.4}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "baselines", "aggregate"))
    parser.add_argument("--corridors-root", type=Path, default=HERE / "corridors")
    parser.add_argument("--corridor", type=str, default="tidal_bimodal")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=10)
    parser.add_argument("--band-mode", type=str, default="aligned")
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/tmp/v501_ckpt"))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/tmp/v501_root"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=Path, default=None)
    args = parser.parse_args()

    val_days = list(range(2650101, 2650111))[: int(args.val_limit)]
    if args.phase == "preflight":
        preflight(args.corridors_root, args.result_dir)
    elif args.phase == "train":
        res_p = args.resume_from if args.resume_from is not None else ((args.ckpt_dir / "latest.pt") if args.resume else None)
        train_seed(
            corridor_dir=args.corridors_root / args.corridor,
            corridor_=args.corridor,
            seed=args.seed,
            episodes=args.episodes,
            val_days=val_days,
            band_mode=args.band_mode,
            ablation=args.ablation,
            ckpt_dir=args.ckpt_dir,
            result_dir=args.result_dir,
            resume_path=res_p,
        )
    elif args.phase == "baselines":
        run_baselines(args.corridors_root / args.corridor, args.corridor, val_days, args.result_dir)
    elif args.phase == "aggregate":
        aggregate(args.root, ["tidal_bimodal", "mbta_route111_wkdy"], list(range(101, 107)), val_days, args.result_dir)


if __name__ == "__main__":
    main()
