#!/usr/bin/env python3
"""Generate same-state dispatch counterfactual labels for FreqDuet.

The fixed-action CRN matrix can say which terminal/headway candidate wins on
an entire episode, but it cannot label a single dispatch decision under the
same simulator state.  This audit wraps the live upper callback, deep-copies the
environment before the real action is applied, and replays a short horizon for
several candidate actions from that identical snapshot.

This is intentionally offline-only.  It writes candidate labels and summary
metadata, but it does not alter the training code or promote a value selector.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_csv(value: str, cast=str) -> list:
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def resolve_config(config: str) -> Path:
    path = Path(config)
    if path.exists():
        return path
    filename = config if config.endswith(".yaml") else f"{config}.yaml"
    candidates = [
        ROOT / "configs_freqduet" / filename,
        ROOT / "configs_freqduet" / "paper_generalization" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"config not found: {config}")


def set_worker_threads(n_threads: int | None) -> None:
    if n_threads is None:
        return
    n = str(max(1, int(n_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "FREQDUET_TORCH_THREADS",
    ):
        os.environ[key] = n


def import_runner():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runner_v3 import DiagnosticLog, TransitDuetV2Runner, load_config

    return DiagnosticLog, TransitDuetV2Runner, load_config


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            state["torch"] = torch_mod.get_rng_state()
        except Exception:
            pass
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and "torch" in state:
        try:
            torch_mod.set_rng_state(state["torch"])
        except Exception:
            pass


def trip_matches(candidate, reference) -> bool:
    return (
        int(getattr(candidate, "launch_turn", -1))
        == int(getattr(reference, "launch_turn", -2))
        and bool(getattr(candidate, "direction", True))
        == bool(getattr(reference, "direction", True))
    )


def find_trip(env, reference_trip):
    for trip in getattr(env, "timetables", []):
        if trip_matches(trip, reference_trip):
            return trip
    ref_launch = float(getattr(reference_trip, "launch_time", -1.0))
    ref_dir = bool(getattr(reference_trip, "direction", True))
    for trip in getattr(env, "timetables", []):
        if (
            bool(getattr(trip, "direction", True)) == ref_dir
            and abs(float(getattr(trip, "launch_time", -9999.0)) - ref_launch) < 1e-6
        ):
            return trip
    raise RuntimeError(
        "snapshot trip not found: "
        f"tid={getattr(reference_trip, 'launch_turn', None)} "
        f"dir={getattr(reference_trip, 'direction', None)}"
    )


def action_name(mode: str, delta_s: float) -> str:
    prefix = "term45" if mode == "terminalhold45" else "target"
    delta = int(round(float(delta_s)))
    if delta == 0:
        suffix = "0"
    elif delta < 0:
        suffix = f"m{abs(delta)}"
    else:
        suffix = f"p{delta}"
    return f"{prefix}_{suffix}"


def launched_count(env) -> int:
    return int(sum(1 for trip in getattr(env, "timetables", []) if getattr(trip, "launched", False)))


def waiting_total(env) -> int:
    return int(sum(len(getattr(station, "waiting_passengers", [])) for station in getattr(env, "stations", [])))


def fleet_concurrent(env) -> int:
    return int(sum(1 for bus in getattr(env, "bus_all", []) if getattr(bus, "on_route", False)))


def active_headway_cv(env) -> float:
    values = []
    for bus in getattr(env, "bus_all", []):
        if not getattr(bus, "on_route", False):
            continue
        for attr in ("forward_headway", "backward_headway"):
            value = float(getattr(bus, attr, 0.0) or 0.0)
            if value > 0.0 and np.isfinite(value):
                values.append(value)
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std() / max(arr.mean(), 1.0))


def target_actual_launch(env, reference_trip):
    try:
        trip = find_trip(env, reference_trip)
    except RuntimeError:
        return None
    return getattr(trip, "_actual_launch_time", None)


def frequency_context(env) -> dict[str, float]:
    try:
        summary = env.frequency_summary()
    except Exception:
        summary = {}
    keys = (
        "freq_low_demand",
        "freq_low_forecast",
        "freq_high_energy",
        "freq_middle_energy",
        "freq_od_entropy",
        "freq_promotion_strength",
        "freq_promotion_active",
    )
    return {key: float(summary.get(key, 0.0)) for key in keys}


def infer_domain(config_name: str) -> str:
    if "gen_highnoise" in config_name:
        return "highnoise"
    if "gen_odshift" in config_name:
        return "odshift"
    if "gen_rushshift" in config_name:
        return "rushshift"
    if "terminal" in config_name:
        return "terminal"
    return "unknown"


def clear_candidate_window(env, trip, planner, plan_all_directions: bool) -> None:
    horizon_s = float(getattr(planner, "horizon_s", 0.0) if planner is not None else 0.0)
    origin = float(getattr(trip, "launch_time", getattr(env, "current_time", 0.0)))
    directions = {True, False} if plan_all_directions else {bool(getattr(trip, "direction", True))}
    attrs = (
        "_freqduet_scheduled_launch",
        "_freqduet_terminal_dispatch",
        "_freqduet_min_dispatch_headway",
        "_freqduet_planned_by",
        "_freqduet_plan_offset_s",
        "_delta_t",
    )
    for candidate in getattr(env, "timetables", []):
        if getattr(candidate, "launched", False):
            continue
        if bool(getattr(candidate, "direction", True)) not in directions:
            continue
        offset = float(getattr(candidate, "launch_time", origin)) - origin
        if planner is not None and (offset < -1e-6 or offset > horizon_s):
            continue
        for attr in attrs:
            if hasattr(candidate, attr):
                delattr(candidate, attr)


def noop_upper_callback(_s_upper, trip):
    if not hasattr(trip, "_original_launch"):
        trip._original_launch = getattr(trip, "launch_time", 0)
    if not hasattr(trip, "_freqduet_base_target_headway"):
        trip._freqduet_base_target_headway = float(getattr(trip, "target_headway", 360.0))
    return float(getattr(trip, "target_headway", getattr(trip, "_freqduet_base_target_headway", 360.0)))


def apply_candidate(
    env,
    planner,
    trip,
    action_dim: int,
    mode: str,
    delta_s: float,
    terminal_hold_s: float,
    terminal_min_s: float,
    terminal_floor_ratio: float,
    terminal_floor_min_s: float,
    plan_all_directions: bool,
) -> dict[str, float]:
    clear_candidate_window(env, trip, planner, plan_all_directions)
    if not hasattr(trip, "_original_launch"):
        trip._original_launch = getattr(trip, "launch_time", 0)
    if not hasattr(trip, "_freqduet_base_target_headway"):
        trip._freqduet_base_target_headway = float(getattr(trip, "target_headway", 360.0))

    write_terminal = mode == "terminalhold45"
    action_vec = np.full(max(1, int(action_dim)), float(delta_s), dtype=np.float32)

    if planner is not None:
        summary = planner.apply(
            getattr(env, "timetables", []),
            trip,
            action_vec,
            origin_launch_s=float(getattr(trip, "launch_time", getattr(env, "current_time", 0.0))),
            write_scheduled_launch=write_terminal,
            terminal_shift_min_s=float(terminal_min_s) if write_terminal else None,
            terminal_shift_max_s=float(terminal_hold_s) if write_terminal else None,
            terminal_shift_bias_s=0.0,
            terminal_headway_floor_ratio=float(terminal_floor_ratio) if write_terminal else 0.0,
            terminal_headway_floor_min_s=float(terminal_floor_min_s) if write_terminal else 0.0,
        )
        effective_delta = float(summary.get("effective_delta", delta_s))
        base_headway = float(summary.get("base_headway", getattr(trip, "_freqduet_base_target_headway", 360.0)))
        target_headway = float(summary.get("target_headway", getattr(trip, "target_headway", base_headway)))
    else:
        base_headway = float(getattr(trip, "_freqduet_base_target_headway", getattr(trip, "target_headway", 360.0)))
        target_headway = float(np.clip(base_headway + float(delta_s), 180.0, 720.0))
        trip.target_headway = target_headway
        effective_delta = target_headway - base_headway
        summary = {
            "target_headway": target_headway,
            "base_headway": base_headway,
            "effective_delta": effective_delta,
            "planned_n": 1,
        }

    trip._delta_t = 0
    if write_terminal:
        trip._freqduet_terminal_dispatch = True
        if not hasattr(trip, "_freqduet_scheduled_launch"):
            trip._freqduet_scheduled_launch = int(round(float(getattr(trip, "launch_time", 0.0))))
    elif hasattr(trip, "_freqduet_terminal_dispatch"):
        delattr(trip, "_freqduet_terminal_dispatch")

    trip._upper_queried = True
    if hasattr(env, "_compute_dispatch_proxy_reward"):
        try:
            env._compute_dispatch_proxy_reward(trip)
        except Exception:
            pass

    return {
        "base_headway": base_headway,
        "target_headway": target_headway,
        "effective_delta": effective_delta,
        "planned_n": float(summary.get("planned_n", 0)),
        "scheduled_n": float(summary.get("scheduled_n", 0)),
        "scheduled_launch": float(getattr(trip, "_freqduet_scheduled_launch", getattr(trip, "launch_time", 0.0))),
    }


def truncate_ep_lists(runner, lengths: dict[str, int]) -> None:
    for name, size in lengths.items():
        value = getattr(runner, name, None)
        if isinstance(value, list) and len(value) > size:
            del value[size:]


def lower_action_for_env(runner, env, obs, bus, last_action: float, deterministic: bool):
    lengths = {
        name: len(value)
        for name, value in runner.__dict__.items()
        if name.startswith("_ep_") and isinstance(value, list)
    }
    original_env = runner.env
    try:
        runner.env = env
        action = runner._lower_policy_action(
            obs,
            last_action=last_action,
            deterministic=deterministic,
        )
        action = runner._apply_lower_fleet_noharm(action, bus)
        return action
    finally:
        runner.env = original_env
        truncate_ep_lists(runner, lengths)


def bus_by_id(env, bus_id: int):
    for bus in getattr(env, "bus_all", []):
        if int(getattr(bus, "bus_id", -1)) == int(bus_id):
            return bus
    return None


def replay_horizon(env, runner, horizon_s: float, deterministic_lower: bool) -> dict[str, float]:
    start_time = float(getattr(env, "current_time", 0.0))
    end_time = start_time + float(horizon_s)
    start_launched = launched_count(env)
    env._upper_policy_callback = noop_upper_callback

    state_dict = copy.deepcopy(getattr(env, "state", {}))
    reward_dict = copy.deepcopy(getattr(env, "reward", {}))
    action_dict = {key: None for key in range(int(getattr(env, "max_agent_num", 0)))}
    lower_last_action = {key: 0.0 for key in range(int(getattr(env, "max_agent_num", 0)))}
    steps = 0

    while not getattr(env, "done", False) and float(getattr(env, "current_time", 0.0)) < end_time:
        for key in list(state_dict.keys()):
            if key not in action_dict:
                action_dict[key] = None
            if key not in lower_last_action:
                lower_last_action[key] = 0.0
            states = state_dict.get(key, [])
            if len(states) == 1:
                if action_dict[key] is None:
                    bus = bus_by_id(env, int(key))
                    action_dict[key] = lower_action_for_env(
                        runner,
                        env,
                        states[0],
                        bus,
                        lower_last_action.get(key, 0.0),
                        deterministic=deterministic_lower,
                    )
            elif len(states) >= 2:
                if states[0][1] != states[1][1]:
                    act_val = runner._lower_action_scalar(action_dict[key]) if action_dict[key] is not None else 0.0
                    lower_last_action[key] = float(act_val)
                state_dict[key] = states[1:]
                bus = bus_by_id(env, int(key))
                action_dict[key] = lower_action_for_env(
                    runner,
                    env,
                    state_dict[key][0],
                    bus,
                    lower_last_action.get(key, 0.0),
                    deterministic=deterministic_lower,
                )

        state_dict, reward_dict, _cost_dict, _done = env.step(action_dict, render=False)
        steps += 1

    concurrent = fleet_concurrent(env)
    n_fleet = float(getattr(env, "_n_fleet_target", 25))
    overshoot = max(0.0, float(concurrent) - n_fleet)
    wait = float(waiting_total(env))
    cv = active_headway_cv(env)
    proxy_cost = wait / 500.0 + (overshoot * overshoot) / max(n_fleet, 1.0) + cv
    return {
        "replay_start_time_s": start_time,
        "replay_end_time_s": float(getattr(env, "current_time", 0.0)),
        "replay_steps": float(steps),
        "waiting_total": wait,
        "fleet_concurrent": float(concurrent),
        "fleet_target": n_fleet,
        "fleet_overshoot": overshoot,
        "headway_cv_active": cv,
        "proxy_cost": float(proxy_cost),
        "launched_delta": float(launched_count(env) - start_launched),
    }


def evaluate_candidate(
    snapshot_env,
    runner,
    reference_trip,
    mode: str,
    delta_s: float,
    args,
) -> dict[str, float | str]:
    env = copy.deepcopy(snapshot_env)
    trip = find_trip(env, reference_trip)
    planner = getattr(runner, "timetable_planner", None)
    plan_all_directions = bool(
        getattr(runner, "timetable_plan_all_directions", False)
        or getattr(planner, "plan_all_directions", False)
    )
    plan = apply_candidate(
        env,
        planner,
        trip,
        int(getattr(runner, "upper_action_dim", 1)),
        mode,
        float(delta_s),
        float(args.terminal_hold_s),
        float(args.terminal_min_s),
        float(args.terminal_floor_ratio),
        float(args.terminal_floor_min_s),
        plan_all_directions,
    )
    replay = replay_horizon(
        env,
        runner,
        horizon_s=float(args.horizon_s),
        deterministic_lower=not bool(args.stochastic_lower),
    )
    actual = target_actual_launch(env, reference_trip)
    launch_time = float(getattr(trip, "launch_time", 0.0))
    out: dict[str, float | str] = {
        "candidate_method": action_name(mode, delta_s),
        "candidate_mode": mode,
        "candidate_delta_s": float(delta_s),
        "target_actual_launch_s": float(actual) if actual is not None else np.nan,
        "target_launch_delay_s": float(actual - launch_time) if actual is not None else np.nan,
    }
    out.update(plan)
    out.update(replay)
    return out


def build_context_row(ep: int, dispatch_index: int, env, trip) -> dict[str, float | int | str]:
    hour = 6 + int(getattr(trip, "launch_time", 0)) // 3600
    period = "peak" if (7 <= hour <= 9 or 17 <= hour <= 19) else ("off" if 9 < hour < 17 else "trans")
    row: dict[str, float | int | str] = {
        "ep": int(ep),
        "dispatch_index": int(dispatch_index),
        "snapshot_time_s": float(getattr(env, "current_time", 0.0)),
        "tid": int(getattr(trip, "launch_turn", -1)),
        "dir": int(bool(getattr(trip, "direction", True))),
        "scheduled_launch_s": float(getattr(trip, "launch_time", 0.0)),
        "hour": int(hour),
        "period": period,
        "base_target_headway_s": float(getattr(trip, "target_headway", 360.0)),
        "waiting_total_pre": float(waiting_total(env)),
        "fleet_concurrent_pre": float(fleet_concurrent(env)),
        "fleet_target_pre": float(getattr(env, "_n_fleet_target", 25)),
        "headway_cv_active_pre": float(active_headway_cv(env)),
    }
    row.update(frequency_context(env))
    return row


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args) -> tuple[Path, dict[str, object]]:
    set_worker_threads(args.worker_threads)
    DiagnosticLog, TransitDuetV2Runner, load_config = import_runner()

    cfg_path = resolve_config(args.config)
    cfg = load_config(str(cfg_path))
    cfg["seed"] = int(args.seed)
    cfg.setdefault("coupling", {})["upper_warmup_eps"] = int(args.upper_warmup_eps)
    cfg.setdefault("output", {})["enable_trip_details"] = False
    cfg.setdefault("output", {})["enable_trace_logger"] = False

    runner = TransitDuetV2Runner(cfg, device="cpu")
    if getattr(runner, "diag", None) is None:
        runner.diag = DiagnosticLog(runner.log_dir, resume=False)
    modes = parse_csv(args.modes, str)
    deltas = parse_csv(args.deltas_s, float)
    rows: list[dict] = []
    dispatch_count = 0
    audit_count = 0
    original_callback = runner._upper_callback_v2
    start_wall = time.time()

    def wrapped_callback(s_upper_v1, trip):
        nonlocal dispatch_count, audit_count
        dispatch_count += 1
        should_sample = (
            audit_count < int(args.max_snapshots)
            and dispatch_count >= int(args.start_dispatch)
            and (dispatch_count - int(args.start_dispatch)) % max(1, int(args.snapshot_stride)) == 0
        )
        if should_sample:
            try:
                snapshot_env = copy.deepcopy(runner.env)
                snapshot_trip = find_trip(snapshot_env, trip)
                rng_state = capture_rng_state()
                context = build_context_row(
                    int(getattr(runner, "_current_ep", 0)),
                    dispatch_count,
                    snapshot_env,
                    snapshot_trip,
                )
                context["config"] = cfg_path.stem
                context["seed"] = int(args.seed)
                context["domain"] = infer_domain(cfg_path.stem)
                for mode in modes:
                    for delta_s in deltas:
                        restore_rng_state(rng_state)
                        candidate = evaluate_candidate(
                            snapshot_env,
                            runner,
                            snapshot_trip,
                            mode,
                            delta_s,
                            args,
                        )
                        row = dict(context)
                        row.update(candidate)
                        rows.append(row)
                restore_rng_state(rng_state)
                audit_count += 1
                print(
                    f"SNAPSHOT ep={context['ep']} dispatch={dispatch_count} "
                    f"tid={context['tid']} rows={len(rows)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"WARN snapshot failed ep={getattr(runner, '_current_ep', None)} "
                    f"dispatch={dispatch_count}: {exc}",
                    flush=True,
                )
        return original_callback(s_upper_v1, trip)

    for burn_ep in range(int(args.burn_in_episodes)):
        runner.run_episode(ep=burn_ep, training=True)

    dispatch_count = 0
    audit_count = 0
    runner._upper_callback_v2 = wrapped_callback
    start_ep = max(int(args.upper_warmup_eps), int(args.burn_in_episodes))
    for offset in range(int(args.episodes)):
        ep = start_ep + offset
        runner.run_episode(ep=ep, training=True)
        if audit_count >= int(args.max_snapshots):
            break

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_name = (
        f"{cfg_path.stem}_seed{int(args.seed)}_ep{int(args.episodes)}"
        f"_snap{int(args.max_snapshots)}_h{int(args.horizon_s)}"
    )
    run_dir = out_dir / run_name
    csv_path = run_dir / "snapshot_counterfactual_labels.csv"
    write_rows(csv_path, rows)
    meta = {
        "config": str(cfg_path),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "burn_in_episodes": int(args.burn_in_episodes),
        "upper_warmup_eps": int(args.upper_warmup_eps),
        "horizon_s": float(args.horizon_s),
        "modes": modes,
        "deltas_s": deltas,
        "max_snapshots": int(args.max_snapshots),
        "snapshots_collected": int(audit_count),
        "dispatches_seen": int(dispatch_count),
        "rows": int(len(rows)),
        "elapsed_s": float(time.time() - start_wall),
        "csv": str(csv_path),
        "note": (
            "Offline audit labels from env deepcopy inside the upper dispatch "
            "callback. Replay uses deterministic lower policy unless "
            "--stochastic-lower is set; pending lower last-action state is "
            "reinitialized at the snapshot."
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshot_counterfactual_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )
    return csv_path, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="F_freqduet_terminal_main_hiro")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--burn-in-episodes", type=int, default=0)
    parser.add_argument("--upper-warmup-eps", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=12)
    parser.add_argument("--start-dispatch", type=int, default=1)
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--horizon-s", type=float, default=900.0)
    parser.add_argument("--modes", default="target,terminalhold45")
    parser.add_argument("--deltas-s", default="-20,0,20")
    parser.add_argument("--terminal-hold-s", type=float, default=45.0)
    parser.add_argument("--terminal-min-s", type=float, default=0.0)
    parser.add_argument("--terminal-floor-ratio", type=float, default=0.0)
    parser.add_argument("--terminal-floor-min-s", type=float, default=0.0)
    parser.add_argument("--stochastic-lower", action="store_true")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default="results_freqduet/snapshot_counterfactual",
        help="Output directory relative to the FreqDuet/freqduet root unless absolute.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_path, meta = run_audit(args)
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(f"WROTE {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
