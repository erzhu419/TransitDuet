#!/usr/bin/env python3
"""V502 Gate 3: Full TR-C Confirmation Matrix, Literature Baselines, 4-Way Frequency Ablation & OOD Suite.

Imports the unified physical environment, turnaround ledger, and B1 trainers
from `freqduet_v7_b1.py` so every method shares 100% identical simulator
dynamics, ScenarioTape CRN pairing, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if not (HERE / "freqduet_v7_b1.py").exists():
    _v501_dir = HERE.parent / "v501_pathb_b1_stability"
    if _v501_dir.exists() and str(_v501_dir) not in sys.path:
        sys.path.insert(0, str(_v501_dir))

torch.set_num_threads(1)

import freqduet_v7_b1 as v7  # noqa: E402
from env.bus import Bus, BusState  # noqa: E402
from env.sim import env_bus  # noqa: E402
from lower.causal_departure_regularity import (  # noqa: E402
    causal_two_sided_holding_target_s,
)


PRIMARY_CORRIDORS = ("tidal_bimodal", "mbta_route111_wkdy")
CONFIRM_DAYS = list(range(2700101, 2700117))  # 16 untouched confirmation CRN days
OOD_DAYS = list(range(2710101, 2710109))      # 8 untouched OOD CRN days per perturbation

CLASSICAL_ARMS = (
    "B0_Fixed_NoHold",
    "B1_Daganzo2009",
    "B2_Berrebi2018",
    "B3_EconMPC",
)

LEARNED_SPECS: Dict[str, dict] = {
    "FreqDuet_Aligned": {"kind": "bilevel", "band_mode": "aligned", "ablation": "none", "reuse_v501": True},
    "B4_FlatSAC": {"kind": "flat_sac", "band_mode": "raw", "ablation": "no_budget", "reuse_v501": False},
    "B5_TransitDuet_NoFreq": {"kind": "bilevel", "band_mode": "nofreq", "ablation": "none", "reuse_v501": False},
    "Abl_RawHistory": {"kind": "bilevel", "band_mode": "raw", "ablation": "none", "reuse_v501": False},
    "Abl_AllBand": {"kind": "bilevel", "band_mode": "all_band", "ablation": "none", "reuse_v501": False},
    "Abl_Swapped": {"kind": "bilevel", "band_mode": "swapped", "ablation": "none", "reuse_v501": False},
    "Abl_NoBudget": {"kind": "bilevel", "band_mode": "aligned", "ablation": "no_budget", "reuse_v501": False},
    "Abl_NoPromotion": {"kind": "bilevel", "band_mode": "aligned", "ablation": "no_promotion", "reuse_v501": False},
}


def run_classical_or_flat_episode(
    *,
    corridor_dir: Path,
    corridor_meta: dict,
    day_seed: int,
    method: str,
    lower_trainer: v7.RESACLagrangianTrainer | None = None,
    lower_replay: v7.CostReplayBuffer | None = None,
    eval_mode: bool = True,
    speed_delta_mps: float = 0.0,
    peak_shift_override: int | None = None,
    burst_scale: float = 1.0,
) -> dict:
    """Execute classical baselines (B0..B3) or single-level B4_FlatSAC under exact CRN."""
    v7.install_simulator_patches()
    Bus._v7_default_capacity = int(corridor_meta.get("bus_capacity", 50))
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
        "promotion": {"enable": False},
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

    apc_feed = v7.CausalHeadwayAPCRateFeed(env)
    for tt in env.timetables:
        tt._freqduet_base_launch = float(tt.launch_time)
        tt._freqduet_scheduled_launch = float(tt.launch_time)
        tt._freqduet_projected_target_headway = 360.0
        tt.target_headway = 360.0
        tt._freqduet_terminal_dispatch = True
        tt._freqduet_min_dispatch_headway = 120.0
    v7.attach_day_bursts(env, corridor_meta["name"], burst_scale=burst_scale)

    corridor_len_m = float(sum(float(r.distance) for r in env.routes[: len(env.routes) // 2]))
    disable_budget = method != "B3_EconMPC"
    ledger = v7.TurnaroundBudgetLedger(corridor_len_m=corridor_len_m, disable_budget=disable_budget)

    daily_pax_prior = float(corridor_meta.get("daily_pax", 18000.0))
    n_active_stops = max(sum(1 for s in env.stations if s.station_type != 0 and s.od is not None), 10)
    mean_stop_rate = max(daily_pax_prior / (14.0 * 60.0 * n_active_stops), 0.25)
    raw_daily_mean_rate = max(daily_pax_prior / (14.0 * 60.0), 1.0)

    last_replan_cycle = -1
    last_budget_refresh_s = -999.0
    trip_trajs: Dict[tuple, List[dict]] = {}
    total_hold_s = 0.0
    bins15 = np.asarray([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
    bins20 = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)

    t0 = time.monotonic()
    while not env.done:
        now_s = float(env.current_time)
        new_launched, finished_trips = ledger.sync_vehicles(env, now_s)

        if method == "B4_FlatSAC" and not eval_mode and lower_replay is not None:
            for t_key, rec in finished_trips:
                traj = trip_trajs.pop(t_key, [])
                v7.flush_trip_lower_transitions(
                    traj, float(rec.get("final_slack_s", 60.0)), lower_replay
                )

        # Rolling-horizon Economic MPC slow clock (uses trailing unfiltered APC counts, no harmonic-RLS/promotion)
        if method == "B3_EconMPC" and now_s <= 45900.0:
            cycle = int(now_s // 900.0)
            if cycle != last_replan_cycle:
                last_replan_cycle = cycle
                h_floor = ledger.cycle_headway_floor_s(n_fleet, env)
                raw_up = max(float(np.mean(list(apc_feed.dir_raw_hist[True])[-10:])), 0.1)
                raw_dn = max(float(np.mean(list(apc_feed.dir_raw_hist[False])[-10:])), 0.1)
                raw_tot = raw_up + raw_dn
                ratio_mpc = math.sqrt(raw_daily_mean_rate / max(raw_tot, 0.1))
                desired_hw = {}
                for d in (True, False):
                    h_avail, _ = v7.remaining_quota_info(env, now_s, d)
                    desired_hw[d] = float(np.clip(h_avail * ratio_mpc, h_floor, 455.0))
                v7.apply_global_quota_timetable(
                    env, now_s, desired_hw, h_bounds=(h_floor, 455.0), tail_bounds=(max(245.0, h_floor - 10.0), 445.0)
                )
                ledger.budget_retention = 0.75
                ledger.refresh_budgets(env, now_s, {"cur_up": raw_up, "cur_dn": raw_dn})
                last_budget_refresh_s = now_s

        if method == "B3_EconMPC" and (new_launched or (now_s - last_budget_refresh_s) >= 60.0):
            raw_up = max(float(np.mean(list(apc_feed.dir_raw_hist[True])[-5:])), 0.1)
            raw_dn = max(float(np.mean(list(apc_feed.dir_raw_hist[False])[-5:])), 0.1)
            ledger.refresh_budgets(env, now_s, {"cur_up": raw_up, "cur_dn": raw_dn})
            last_budget_refresh_s = now_s
        elif method == "B4_FlatSAC" and new_launched:
            ledger.refresh_budgets(env, now_s, {})

        act_dict = {bus.bus_id: None for bus in env.bus_all}
        for bus in env.bus_all:
            if not bus.on_route or bus.state != BusState.WAITING_ACTION:
                continue
            fwd_ok = (
                bus.pre_action_forward_headway_source == "matched_departure_event"
                and bus.pre_action_forward_headway is not None
            )
            bwd_ok = (
                str(bus.pre_action_follower_source).startswith("same_time_avl_")
                and bus.pre_action_follower_departure_gap is not None
            )
            fwd = float(bus.pre_action_forward_headway) if fwd_ok else None
            bwd = float(bus.pre_action_follower_departure_gap) if bwd_ok else None

            if method == "B0_Fixed_NoHold":
                chosen = 0.0
            elif method == "B1_Daganzo2009":
                if fwd is None:
                    chosen = 0.0
                else:
                    req = float(np.clip(0.60 * (float(bus._target_headway) - fwd), 0.0, 20.0))
                    chosen = float(bins20[np.argmin(np.abs(bins20 - req))])
            elif method == "B2_Berrebi2018":
                req = causal_two_sided_holding_target_s(
                    forward_headway_s=fwd,
                    follower_departure_gap_s=bwd,
                    action_cap_s=15.0,
                )
                chosen = float(bins15[np.argmin(np.abs(bins15 - req))]) if req is not None else 0.0
            elif method == "B3_EconMPC":
                key = (int(bus.bus_id), int(bus.trip_id))
                rem_b = float(ledger.remaining_budget_s.get(key, 25.0))
                cap_s = min(20.0, max(0.0, rem_b))
                if cap_s < 5.0 or fwd is None or bwd is None:
                    chosen = 0.0
                else:
                    h_own = max(float(bus._target_headway), 60.0)
                    fol_hw = max(v7._follower_target_headway_s(env, bus), 60.0)
                    cur_idx = bus.effective_station.index(bus.last_station) if bus.last_station in bus.effective_station else 1
                    ds_raw = 0.0
                    for s_obj in bus.effective_station[cur_idx + 1 : -1]:
                        rh = apc_feed.raw_rate_hist.get((int(s_obj.station_id), bool(bus.direction)))
                        ds_raw += float(rh[-1]) if rh else 1.0
                    ds_rate_s = ds_raw / 60.0
                    onboard_pax = float(len(bus.passengers))
                    legal = bins20[bins20 <= cap_s + 1e-6]
                    best_u, best_net = 0.0, 0.0
                    gap_term = (bwd / fol_hw) - (fwd / h_own)
                    curv_term = (1.0 / h_own) + (1.0 / fol_hw)
                    for cand_u in legal:
                        u_f = float(cand_u)
                        if u_f <= 0.0:
                            continue
                        net = ds_rate_s * 0.5 * (h_own + fol_hw) * u_f * (gap_term - 0.5 * u_f * curv_term) - onboard_pax * u_f
                        if net > best_net:
                            best_net, best_u = net, u_f
                    chosen = best_u
            elif method == "B4_FlatSAC":
                t_key = (int(bus.bus_id), int(bus.trip_id))
                s_l, ctx = v7.build_lower_state(env, bus, now_s, corridor_len_m, ledger, apc_feed, "raw")
                chosen = float(lower_trainer.policy_net.get_action(s_l, deterministic=eval_mode))
                if not eval_mode and lower_replay is not None:
                    raw_arr = getattr(bus, "forward_headway", None)
                    arr_hw = (
                        float(raw_arr)
                        if (raw_arr is not None and math.isfinite(float(raw_arr)) and float(raw_arr) > 0.0)
                        else float(bus._target_headway)
                    )
                    trip_trajs.setdefault(t_key, []).append({
                        "state": s_l,
                        "action_s": chosen,
                        "ctx": ctx,
                        "board_here": float(bus.last_board_count),
                        "arr_hw": arr_hw,
                        "trip_id": int(t_key[1]),
                    })
            else:
                raise ValueError(f"Unknown method: {method}")

            act_dict[bus.bus_id] = chosen
            if chosen > 0.0:
                total_hold_s += chosen
            ledger.debit(bus.bus_id, bus.trip_id, chosen)

        env.step(act_dict, debug=False, render=False)
        apc_feed.observe_tick()

    if method == "B4_FlatSAC" and not eval_mode and lower_replay is not None:
        for t_key, traj in list(trip_trajs.items()):
            v7.flush_trip_lower_transitions(traj, 60.0, lower_replay)
        trip_trajs.clear()

    wall_s = time.monotonic() - t0
    details = dict(env._measurement_details)
    hw_metrics = v7.compute_headway_metrics(env.headway_events.events)
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
    }


def eval_suite_for_agent(
    *,
    corridors_root: Path,
    corridor_: str,
    method: str,
    spec: dict,
    upper: v7.RESACUpperTrainer | None,
    lower: v7.RESACLagrangianTrainer | None,
) -> dict:
    """Evaluate a frozen agent across the 16 confirmation days + OOD/Null suites."""
    c_dir = corridors_root / corridor_
    c_meta = json.loads((c_dir / "corridor_meta.json").read_text())

    def _run_days(target_dir: Path, target_meta: dict, days: List[int], **kw) -> dict:
        rows = []
        for idx, d in enumerate(days):
            ps = kw.get("peak_shift_override")
            if kw.get("alternating_shift"):
                ps = -1 if (idx % 2 == 0) else 1
            if spec["kind"] == "flat_sac":
                row = run_classical_or_flat_episode(
                    corridor_dir=target_dir,
                    corridor_meta=target_meta,
                    day_seed=int(d),
                    method="B4_FlatSAC",
                    lower_trainer=lower,
                    eval_mode=True,
                    speed_delta_mps=kw.get("speed_delta_mps", 0.0),
                    peak_shift_override=ps,
                    burst_scale=kw.get("burst_scale", 1.0),
                )
            else:
                row = v7.run_b1_episode(
                    corridor_dir=target_dir,
                    corridor_meta=target_meta,
                    day_seed=int(d),
                    upper_trainer=upper,
                    lower_trainer=lower,
                    lower_replay=None,
                    eval_mode=True,
                    band_mode=spec["band_mode"],
                    ablation=spec["ablation"],
                    speed_delta_mps=kw.get("speed_delta_mps", 0.0),
                    peak_shift_override=ps,
                    burst_scale=kw.get("burst_scale", 1.0),
                )
            rows.append(row)
        keys = list(rows[0]["metrics"].keys())
        return {
            "means": {k: round(float(np.mean([r["metrics"][k] for r in rows])), 6) for k in keys},
            "per_day": rows,
        }

    out = {
        "confirm_16d": _run_days(c_dir, c_meta, CONFIRM_DAYS),
        "ood_peak_shift": _run_days(c_dir, c_meta, OOD_DAYS, alternating_shift=True),
        "ood_hf_surge": _run_days(c_dir, c_meta, OOD_DAYS, burst_scale=1.8),
        "ood_congestion": _run_days(c_dir, c_meta, OOD_DAYS, speed_delta_mps=-0.8),
    }
    if corridor_ == "tidal_bimodal":
        flat_dir = corridors_root / "flat_sanity"
        flat_meta = json.loads((flat_dir / "corridor_meta.json").read_text())
        out["null_flat_sanity"] = _run_days(flat_dir, flat_meta, OOD_DAYS, burst_scale=0.0)
    return out


def run_train_and_confirm(
    *,
    corridors_root: Path,
    corridor_: str,
    method: str,
    seed: int,
    episodes: int,
    ckpt_dir: Path,
    result_dir: Path,
    resume_path: Path | None,
    v501_root: Path,
) -> None:
    spec = LEARNED_SPECS[method]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    upper, lower, lower_replay = v7.make_trainers(seed)
    c_dir = corridors_root / corridor_
    c_meta = json.loads((c_dir / "corridor_meta.json").read_text())
    t_start = time.monotonic()

    if spec["reuse_v501"]:
        frozen_dir = v501_root / f"train_{corridor_}_s{seed}" / "frozen_models"
        u_path = frozen_dir / "upper_ep200.pt"
        l_path = frozen_dir / "lower_ep200.pt"
        assert u_path.exists() and l_path.exists(), f"Missing V501 frozen ep200 models at {frozen_dir}"
        upper.load(u_path)
        lower.load(l_path)
        suites = eval_suite_for_agent(
            corridors_root=corridors_root, corridor_=corridor_, method=method,
            spec=spec, upper=upper, lower=lower,
        )
        v7.write_json(result_dir / "results.json", {
            "completed": True,
            "corridor": corridor_,
            "method": method,
            "seed": int(seed),
            "reused_v501_ep200": True,
            "suites": suites,
        })
        print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "eval_suite", "elapsed_seconds": round(time.monotonic() - t_start, 2)}), flush=True)
        print(f"Training complete: evaluated reused V501 {method} {corridor_} s={seed}", flush=True)
        return

    start_ep = 1
    train_curve: List[dict] = []
    latest_ckpt = ckpt_dir / "latest.pt"
    load_target = resume_path if (resume_path is not None and resume_path.exists()) else None
    if load_target is not None:
        state = torch.load(load_target, map_location="cpu", weights_only=False)
        start_ep = int(state["episode"]) + 1
        if spec["kind"] != "flat_sac":
            upper.load_training_state_dict(state["upper"])
        lower.load_training_state_dict(state["lower"])
        lower_replay.load_state_dict(state["lower_replay"])
        random.setstate(state["py_rng"])
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        train_curve = list(state.get("train_curve", []))

    for ep in range(start_ep, episodes + 1):
        day_seed = 1_000_000 + int(seed) * 10_000 + int(ep)
        if spec["kind"] == "flat_sac":
            ep_out = run_classical_or_flat_episode(
                corridor_dir=c_dir, corridor_meta=c_meta, day_seed=day_seed,
                method="B4_FlatSAC", lower_trainer=lower, lower_replay=lower_replay, eval_mode=False,
            )
        else:
            ep_out = v7.run_b1_episode(
                corridor_dir=c_dir, corridor_meta=c_meta, day_seed=day_seed,
                upper_trainer=upper, lower_trainer=lower, lower_replay=lower_replay,
                eval_mode=False, band_mode=spec["band_mode"], ablation=spec["ablation"],
            )

        if spec["kind"] != "flat_sac" and ep >= 3 and len(upper.replay_buffer) >= 48:
            for _ in range(32):
                upper.update(batch_size=48)
        if ep >= 3 and len(lower_replay) >= 256:
            for _ in range(64):
                lower.update(lower_replay, batch_size=128, reward_scale=2.0)

        train_curve.append({
            "episode": ep,
            "journey_s": round(ep_out["metrics"]["restricted_journey_s_per_pax"], 3),
            "ewt_s": round(ep_out["metrics"]["ewt_s"], 3),
            "readiness_s": round(ep_out["metrics"]["readiness_delay_s_per_trip"], 3),
        })

        if ep % 10 == 0 or ep == episodes:
            v7.atomic_torch_save(latest_ckpt, {
                "episode": ep,
                "upper": (upper.training_state_dict() if spec["kind"] != "flat_sac" else None),
                "lower": lower.training_state_dict(),
                "lower_replay": lower_replay.state_dict(),
                "py_rng": random.getstate(),
                "np_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "train_curve": train_curve,
            })

        print("SCHEDULEURM_PROGRESS " + json.dumps({
            "current": ep, "total": episodes, "unit": "episode",
            "elapsed_seconds": round(time.monotonic() - t_start, 2),
        }), flush=True)

    frozen_dir = result_dir.parent / "frozen_models"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    if spec["kind"] != "flat_sac":
        upper.save(frozen_dir / "upper_ep200.pt")
    lower.save(frozen_dir / "lower_ep200.pt")

    suites = eval_suite_for_agent(
        corridors_root=corridors_root, corridor_=corridor_, method=method,
        spec=spec, upper=upper, lower=lower,
    )
    v7.write_json(result_dir / "results.json", {
        "completed": True,
        "corridor": corridor_,
        "method": method,
        "seed": int(seed),
        "episodes": int(episodes),
        "suites": suites,
        "train_curve": train_curve,
    })
    print(f"Training complete: {method} {corridor_} seed={seed}", flush=True)


def run_classical_suite(corridors_root: Path, corridor_: str, method: str, result_dir: Path) -> None:
    c_dir = corridors_root / corridor_
    c_meta = json.loads((c_dir / "corridor_meta.json").read_text())
    t_start = time.monotonic()

    def _run_days(target_dir: Path, target_meta: dict, days: List[int], **kw) -> dict:
        rows = []
        for idx, d in enumerate(days):
            ps = kw.get("peak_shift_override")
            if kw.get("alternating_shift"):
                ps = -1 if (idx % 2 == 0) else 1
            rows.append(run_classical_or_flat_episode(
                corridor_dir=target_dir, corridor_meta=target_meta, day_seed=int(d),
                method=method, eval_mode=True,
                speed_delta_mps=kw.get("speed_delta_mps", 0.0),
                peak_shift_override=ps,
                burst_scale=kw.get("burst_scale", 1.0),
            ))
        keys = list(rows[0]["metrics"].keys())
        return {
            "means": {k: round(float(np.mean([r["metrics"][k] for r in rows])), 6) for k in keys},
            "per_day": rows,
        }

    suites = {
        "confirm_16d": _run_days(c_dir, c_meta, CONFIRM_DAYS),
        "ood_peak_shift": _run_days(c_dir, c_meta, OOD_DAYS, alternating_shift=True),
        "ood_hf_surge": _run_days(c_dir, c_meta, OOD_DAYS, burst_scale=1.8),
        "ood_congestion": _run_days(c_dir, c_meta, OOD_DAYS, speed_delta_mps=-0.8),
    }
    if corridor_ == "tidal_bimodal":
        flat_dir = corridors_root / "flat_sanity"
        flat_meta = json.loads((flat_dir / "corridor_meta.json").read_text())
        suites["null_flat_sanity"] = _run_days(flat_dir, flat_meta, OOD_DAYS, burst_scale=0.0)

    v7.write_json(result_dir / "results.json", {
        "completed": True,
        "corridor": corridor_,
        "method": method,
        "suites": suites,
    })
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "classical_suite", "elapsed_seconds": round(time.monotonic() - t_start, 2)}), flush=True)


def aggregate(root: Path, seeds: List[int], result_dir: Path) -> None:
    rng = np.random.default_rng(20261120)
    draws16 = rng.integers(0, len(CONFIRM_DAYS), size=(10000, len(CONFIRM_DAYS)))
    draws8 = rng.integers(0, len(OOD_DAYS), size=(10000, len(OOD_DAYS)))

    metric_keys = (
        "restricted_journey_s_per_pax",
        "restricted_wait_s_per_pax",
        "restricted_in_vehicle_s_per_pax",
        "ewt_s",
        "plan_cv2",
        "raw_cv2",
        "readiness_delay_s_per_trip",
        "hold_s_per_trip",
        "passenger_unserved_rate",
    )

    by_corridor = {}
    for c in PRIMARY_CORRIDORS:
        suite_names = ["confirm_16d", "ood_peak_shift", "ood_hf_surge", "ood_congestion"]
        if c == "tidal_bimodal":
            suite_names.append("null_flat_sanity")

        # Load classical arms
        method_day_tables: Dict[str, Dict[str, List[dict]]] = {}
        method_seed_stds: Dict[str, Dict[str, Dict[str, float]]] = {}
        for m in CLASSICAL_ARMS:
            js = json.loads((root / f"classical_{c}_{m}" / "summary" / "results.json").read_text())
            method_day_tables[m] = {
                sn: [r["metrics"] for r in js["suites"][sn]["per_day"]]
                for sn in suite_names
            }
            method_seed_stds[m] = {sn: {k: 0.0 for k in metric_keys} for sn in suite_names}

        # Load learned arms (averaged per CRN day across seeds, plus across-seed SD)
        for m in LEARNED_SPECS:
            s_jsons = [
                json.loads((root / f"learned_{c}_{m}_s{s}" / "summary" / "results.json").read_text())
                for s in seeds
            ]
            method_day_tables[m] = {}
            method_seed_stds[m] = {}
            for sn in suite_names:
                n_d = len(s_jsons[0]["suites"][sn]["per_day"])
                day_list = []
                for di in range(n_d):
                    day_list.append({
                        k: float(np.mean([sj["suites"][sn]["per_day"][di]["metrics"][k] for sj in s_jsons]))
                        for k in metric_keys
                    })
                method_day_tables[m][sn] = day_list
                method_seed_stds[m][sn] = {
                    k: round(float(np.std([sj["suites"][sn]["means"][k] for sj in s_jsons], ddof=1)), 6)
                    for k in metric_keys
                }

        suites_out = {}
        for sn in suite_names:
            draws = draws16 if sn == "confirm_16d" else draws8
            means_table = {}
            for m, sdict in method_day_tables.items():
                rows_m = sdict[sn]
                means_table[m] = {
                    "mean": {k: round(float(np.mean([r[k] for r in rows_m])), 6) for k in metric_keys},
                    "seed_std": method_seed_stds[m][sn],
                }

            # Paired contrasts of FreqDuet_Aligned against EVERY other arm
            aligned_days = method_day_tables["FreqDuet_Aligned"][sn]
            contrasts = {}
            for other_m in [*CLASSICAL_ARMS, *(k for k in LEARNED_SPECS if k != "FreqDuet_Aligned")]:
                other_days = method_day_tables[other_m][sn]
                c_entry = {}
                for k in ("restricted_journey_s_per_pax", "restricted_wait_s_per_pax", "ewt_s", "plan_cv2", "readiness_delay_s_per_trip"):
                    deltas = np.asarray([aligned_days[i][k] - other_days[i][k] for i in range(len(aligned_days))], dtype=np.float64)
                    b_means = deltas[draws].mean(axis=1)
                    c_entry[k] = {
                        "mean": round(float(deltas.mean()), 6),
                        "ci95": [round(float(x), 6) for x in np.quantile(b_means, [0.025, 0.975])],
                        "days_improved": int((deltas < 0.0).sum()),
                    }
                contrasts[f"FreqDuet_minus_{other_m}"] = c_entry

            suites_out[sn] = {
                "n_days": len(aligned_days),
                "methods": means_table,
                "paired_contrasts_vs_FreqDuet": contrasts,
            }

        by_corridor[c] = suites_out

    v7.write_json(result_dir / "results.json", {
        "completed": True,
        "experiment": "v502_pathb_trc_matrix",
        "seeds": seeds,
        "by_corridor": by_corridor,
    })
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 1, "total": 1, "unit": "aggregate", "elapsed_seconds": 0.6}), flush=True)


def preflight(corridors_root: Path, result_dir: Path) -> None:
    v7.install_simulator_patches()
    test_env = env_bus(
        str(corridors_root / "mbta_route111_wkdy"),
        debug=False,
        render=False,
        route_sigma=1.5,
        env_config={"service_start_hour": 6, "service_end_hour": 19, "scenario_seed": 2710101},
    )
    test_env.reset()
    for route in test_env.routes:
        route.speed_history = (route.speed_history.astype(float) - 0.8).clip(lower=3.5)
    test_env.routes[0].route_update(0, test_env._route_effective_period, scenario_tape=test_env.scenario_tape)
    assert len(test_env.bus_all) == 0
    v7.preflight(corridors_root, result_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "classical", "learned", "aggregate"))
    parser.add_argument("--corridors-root", type=Path, default=HERE / "corridors")
    parser.add_argument("--corridor", type=str, default="tidal_bimodal")
    parser.add_argument("--method", type=str, default="B0_Fixed_NoHold")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/tmp/v502_ckpt"))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/tmp/v502_root"))
    parser.add_argument("--v501-root", type=Path, default=Path("/home/zhengliang01/scheduleurm_work/FreqDuet-v501-pathb-b1/run01"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=Path, default=None)
    args = parser.parse_args()

    if args.phase == "preflight":
        preflight(args.corridors_root, args.result_dir)
    elif args.phase == "classical":
        run_classical_suite(args.corridors_root, args.corridor, args.method, args.result_dir)
    elif args.phase == "learned":
        res_p = args.resume_from if args.resume_from is not None else ((args.ckpt_dir / "latest.pt") if args.resume else None)
        run_train_and_confirm(
            corridors_root=args.corridors_root,
            corridor_=args.corridor,
            method=args.method,
            seed=args.seed,
            episodes=args.episodes,
            ckpt_dir=args.ckpt_dir,
            result_dir=args.result_dir,
            resume_path=res_p,
            v501_root=args.v501_root,
        )
    elif args.phase == "aggregate":
        aggregate(args.root, list(range(101, 105)), args.result_dir)


if __name__ == "__main__":
    main()
