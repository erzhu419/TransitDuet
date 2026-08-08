#!/usr/bin/env python3
"""Run classical external baselines with the FreqDuet paper metric format."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        runtime_environment,
        validate_csv_artifact,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from analysis_provenance import (
        canonical_json_sha256,
        csv_artifact_record,
        runtime_environment,
        validate_csv_artifact,
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sim import env_bus
from env.evaluation import (
    composite_service_cost,
    normalize_wait_metric,
    service_cost_views,
)
from runner_v3 import load_config
from run_baseline_rule import hour_to_slot, mpc_plan, rule_holding_action
from scripts.run_freqduet_protocol_v2_matrix import (
    V4_SAFETY_METRICS,
    V5_SAFETY_METRICS,
    config_fingerprint,
    git_provenance,
    protocol_version_for_config,
    scenario_contract,
    source_fingerprint,
)
from scripts import run_freqduet_protocol_v2_matrix as matrix_protocol


BASELINE_VARIANTS = ("fixed_headway", "rule_holding", "rule_mpc")
ROUTE_HEADWAY_POLICY_PREFIXES = (
    "route_value_",
    "route_headway_",
    "route_oracle_",
)
DEFAULT_FIXED_HEADWAY_S = 360.0
DEFAULT_DOMAIN_CONFIGS = {
    "terminal": "F_freqduet_terminal_main_hiro",
    "highnoise": "F_freqduet_gen_highnoise_main_hiro",
    "odshift": "F_freqduet_gen_odshift_main_hiro",
    "rushshift": "F_freqduet_gen_rushshift_main_hiro",
}
EXTERNAL_BASELINE_MANIFEST = "external_baseline_manifest.json"
V6_PROTOCOL_VERSION = "freqduet-eval-v6"
V6_RUN_MANIFEST_VERSION = "freqduet-external-baseline-v6"
V6_SUMMARY_MANIFEST_VERSION = "freqduet-external-summary-v6"
V6_SCENARIO_CONTRACT_VERSION = "freqduet-scenario-contract-v1"
V6_COMMAND_CONTRACT_VERSION = "freqduet-external-command-v1"
V6_DIAGNOSTICS_PRIMARY_KEY = ["config", "variant", "eval_seed"]
V6_PER_SEED_PRIMARY_KEY = ["config", "method", "eval_seed"]
V6_SUMMARY_PRIMARY_KEY = ["config", "domain", "method"]
V6_EXECUTION_METRICS_FALLBACK = [
    "fleet_denied_retry_trip_seconds",
    "commanded_holding_vehicle_seconds",
    "commanded_holding_passenger_seconds",
    "commanded_holding_passenger_min_per_generated",
    "terminal_actual_dispatch_gap_mean_s",
    "terminal_dispatch_execution_error_mean_s",
    "terminal_dispatch_execution_error_abs_mean_s",
]


def _ordered_union(*groups: list[str] | tuple[str, ...]) -> list[str]:
    result = []
    for group in groups:
        for value in group:
            if value not in result:
                result.append(value)
    return result


V6_EXECUTION_METRICS = _ordered_union(
    list(getattr(matrix_protocol, "V6_EXECUTION_METRICS", [])),
    V6_EXECUTION_METRICS_FALLBACK,
)
V6_SAFETY_METRICS = _ordered_union(
    V5_SAFETY_METRICS,
    list(getattr(matrix_protocol, "V6_SAFETY_METRICS", [])),
    V6_EXECUTION_METRICS,
)


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def parse_csv_file(path: str | Path, cast=str) -> list:
    items = []
    with open(path, "r") as f:
        for line in f:
            items.extend(parse_csv_list(line, cast=cast))
    return items


def fixed_headway_target_s(variant: str) -> float | None:
    """Return the target seconds for fixed-headway variants.

    ``fixed_headway`` keeps the historical 360 s baseline.  Dynamic names such
    as ``fixed_headway_330`` and ``fixed_headway_h330`` are used for route-day
    counterfactual candidate sweeps.
    """
    text = str(variant).strip()
    if text == "fixed_headway":
        return DEFAULT_FIXED_HEADWAY_S
    for prefix in ("fixed_headway_", "fixed_h"):
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix):]
        if suffix.startswith("h"):
            suffix = suffix[1:]
        try:
            target = float(suffix)
        except ValueError:
            return None
        if target <= 0:
            return None
        return target
    return None


def is_route_headway_policy_variant(variant: str) -> bool:
    text = str(variant).strip()
    return any(text.startswith(prefix) for prefix in ROUTE_HEADWAY_POLICY_PREFIXES)


def is_known_variant(variant: str) -> bool:
    return (
        str(variant) in BASELINE_VARIANTS
        or fixed_headway_target_s(variant) is not None
        or is_route_headway_policy_variant(variant)
    )


def config_path(name: str) -> Path:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return ROOT / "configs_freqduet" / filename


def resolve_under_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def run_dir_for(config: str, variant: str, seed: int, logs_dir: Path) -> Path:
    return logs_dir / f"{config}_{variant}_seed{seed}"


def external_evaluator_fingerprint() -> dict[str, object]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "run_baseline_rule.py",
        ROOT / "scripts/analysis_provenance.py",
        ROOT / "scripts/run_freqduet_protocol_v2_matrix.py",
    ]
    digest = hashlib.sha256()
    files = []
    for path in paths:
        payload = path.read_bytes()
        label = str(path.relative_to(ROOT))
        file_sha = hashlib.sha256(payload).hexdigest()
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha))
        files.append({
            "path": label,
            "sha256": file_sha,
            "size_bytes": len(payload),
        })
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _fingerprint_sha(record: object, label: str) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"{label} fingerprint is missing")
    value = str(record.get("sha256", ""))
    if len(value) != 64:
        raise ValueError(f"{label} fingerprint is invalid")
    return value


def scenario_contract_fingerprint(config: str) -> dict[str, object]:
    """Use the learned-matrix scenario contract without reimplementation."""
    record = scenario_contract(config)
    if record.get("version") != V6_SCENARIO_CONTRACT_VERSION:
        raise ValueError("matrix scenario contract version mismatch")
    _validate_self_hashed_record(
        record, payload_key="payload", label="scenario contract")
    return record


def locked_runtime_environment() -> dict[str, object]:
    payload = runtime_environment()
    return {
        "sha256": canonical_json_sha256(payload),
        "environment": payload,
    }


def external_command_contract(
    *,
    config: str,
    variant: str,
    seed: int,
    episodes: int,
    direct_scenario_seed: bool,
    worker_threads: int | None,
    route_headway_target_s: float | None,
) -> dict[str, object]:
    payload = {
        "version": V6_COMMAND_CONTRACT_VERSION,
        "entrypoint": "scripts/run_freqduet_external_baselines.py",
        "config_name": str(config),
        "variant": str(variant),
        "method": str(variant),
        "seed": int(seed),
        "episodes": int(episodes),
        "direct_scenario_seed": bool(direct_scenario_seed),
        "worker_threads": (
            None if worker_threads is None else int(worker_threads)),
        "route_headway_target_s": (
            None if route_headway_target_s is None
            else float(route_headway_target_s)),
    }
    return {
        "sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def _validate_self_hashed_record(
    record: object,
    *,
    payload_key: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    payload = record.get(payload_key)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload is missing")
    observed = canonical_json_sha256(payload)
    if observed != str(record.get("sha256", "")):
        raise ValueError(f"{label} hash does not match its payload")
    return payload


def install_exact_dispatch_schedule(env, target_fn, projection_mode: str) -> None:
    """Write one executable launch sequence per direction before simulation."""
    mode = str(projection_mode).strip()
    if not mode:
        raise ValueError("projection_mode must be non-empty")
    for direction in (False, True):
        trips = sorted(
            (trip for trip in env.timetables
             if bool(trip.direction) == direction),
            key=lambda trip: (float(trip.launch_time), int(trip.launch_turn)),
        )
        if not trips:
            continue
        previous_launch = None
        for index, trip in enumerate(trips):
            base_launch = float(trip.launch_time)
            target = float(target_fn(
                direction, base_launch if previous_launch is None
                else previous_launch, trip))
            if not np.isfinite(target) or target <= 0.0:
                raise ValueError("baseline target headway must be finite and positive")
            scheduled_launch = (
                base_launch if index == 0
                else float(previous_launch) + target
            )
            trip._freqduet_base_launch = base_launch
            trip._freqduet_base_target_headway = float(
                getattr(trip, "target_headway", target))
            trip._freqduet_scheduled_launch = float(scheduled_launch)
            trip._freqduet_predecessor_scheduled_launch = previous_launch
            trip._freqduet_desired_headway_s = target
            trip._freqduet_projected_target_headway = target
            trip._freqduet_phase_displacement_s = scheduled_launch - base_launch
            trip._freqduet_projection_mode = mode
            trip._freqduet_terminal_dispatch = True
            trip._freqduet_planned_by = -1
            trip.target_headway = target
            previous_launch = float(scheduled_launch)


def diagnostics_complete(run_dir: Path, episodes: int) -> bool:
    csv_path = run_dir / "diagnostics.csv"
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if "ep" in df.columns:
        df = df[df["ep"] < 9000]
    return len(df) >= int(episodes)


def apply_worker_threads(worker_threads: int | None) -> None:
    if worker_threads is None:
        return
    n = str(max(1, int(worker_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[key] = n


def infer_domain(config: str) -> str:
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "unknown"


def load_headway_policy(path: str | Path | None) -> dict[tuple[str, str, int | None], float]:
    """Load route-day policy targets keyed by (variant, config, seed-or-None)."""
    if not path:
        return {}
    policy_path = resolve_under_root(path)
    if not policy_path.exists():
        raise SystemExit(f"headway policy csv does not exist: {policy_path}")
    df = pd.read_csv(policy_path)
    if df.empty:
        raise SystemExit(f"headway policy csv is empty: {policy_path}")
    if "config" not in df.columns:
        raise SystemExit("headway policy csv must include a config column")
    target_col = None
    for col in ("target_headway_s", "selected_target_headway_s", "headway_s"):
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        if "selected_method" in df.columns:
            df["target_headway_s"] = df["selected_method"].map(fixed_headway_target_s)
            target_col = "target_headway_s"
        else:
            raise SystemExit(
                "headway policy csv must include target_headway_s or selected_method")
    variant_col = None
    for col in ("policy_variant", "variant", "policy_name"):
        if col in df.columns:
            variant_col = col
            break
    if variant_col is None:
        df["policy_variant"] = "route_value_policy"
        variant_col = "policy_variant"

    policy: dict[tuple[str, str, int | None], float] = {}
    for _, row in df.iterrows():
        variant = str(row[variant_col]).strip()
        config = str(row["config"]).strip()
        if not variant or not config:
            continue
        target = pd.to_numeric(row[target_col], errors="coerce")
        if not np.isfinite(target) or float(target) <= 0:
            continue
        seed_key = None
        if "seed" in df.columns and not pd.isna(row.get("seed")):
            seed_val = pd.to_numeric(row.get("seed"), errors="coerce")
            if np.isfinite(seed_val):
                seed_key = int(seed_val)
        policy[(variant, config, seed_key)] = float(target)
    if not policy:
        raise SystemExit(f"no usable rows in headway policy csv: {policy_path}")
    return policy


def route_policy_target_s(
    policy: dict[tuple[str, str, int | None], float],
    variant: str,
    config: str,
    seed: int,
    default_headway_s: float | None,
) -> float:
    keys = [
        (variant, config, int(seed)),
        (variant, config, None),
        ("*", config, int(seed)),
        ("*", config, None),
    ]
    for key in keys:
        if key in policy:
            return float(policy[key])
    if default_headway_s is not None:
        return float(default_headway_s)
    raise KeyError(
        f"no route headway policy target for variant={variant} config={config} seed={seed}")


def make_env_from_config(config_name: str):
    cfg = load_config(str(config_path(config_name)))
    env_cfg = cfg.get("env", {})
    upper_cfg = cfg.get("upper", {})
    lower_cfg = cfg.get("lower", {})
    env_path = env_cfg.get("path", "env")
    env = env_bus(
        str(resolve_under_root(env_path)),
        route_sigma=env_cfg.get("route_sigma", cfg.get("env", {}).get("route_sigma", 1.5)),
        env_config=env_cfg,
    )
    env.configure_frequency_features(cfg.get("frequency", {}))
    env.enable_plot = False
    env._n_fleet_target = upper_cfg.get("N_fleet", 12)
    env.demand_noise = env_cfg.get("demand_noise", 0.0)
    env.demand_scale = env_cfg.get("demand_scale", 1.0)
    env.demand_hourly_multipliers = env_cfg.get("demand_hourly_multipliers", None)
    env.service_start_hour = env_cfg.get("service_start_hour", 6)
    env.service_end_hour = env_cfg.get("service_end_hour", 19)
    env.od_noise = env_cfg.get("od_noise", 0.0)
    env.od_noise_clip = env_cfg.get("od_noise_clip", [0.3, 2.0])
    env.peak_shift_choices = env_cfg.get("peak_shift_choices", None)
    env.peak_shift_probs = env_cfg.get("peak_shift_probs", None)
    lower_encoder = lower_cfg.get("state_encoder", {}) or {}
    env.lower_state_input_schema = str(lower_encoder.get(
        "input_schema", "legacy_headway_deviation")).strip().lower()
    env.lower_observation_contract = str(lower_cfg.get(
        "observation_contract", env.lower_observation_contract)).strip().lower()
    env.headway_reward_mode = str(lower_cfg.get(
        "headway_reward_mode", env.headway_reward_mode)).strip().lower()
    env.headway_state_mode = str(lower_cfg.get(
        "headway_state_mode", "arrival_event")).strip().lower()
    env.unobserved_action_mode = str(lower_cfg.get(
        "unobserved_action_mode", "legacy_carry_forward")).strip().lower()
    env.holding_action_trace_mode = str(lower_cfg.get(
        "holding_action_trace_mode", "observed_only")).strip().lower()
    return env, cfg


def composite(
    wait: float,
    cv: float,
    peak_fleet: float,
    n_fleet: int,
    passenger_unserved_rate: float = 0.0,
    trip_completion_rate: float = 1.0,
    weights: dict | None = None,
) -> float:
    value, _ = composite_service_cost(
        avg_wait_min=wait,
        peak_fleet=peak_fleet,
        headway_cv=cv,
        n_fleet=n_fleet,
        passenger_unserved_rate=passenger_unserved_rate,
        trip_completion_rate=trip_completion_rate,
        weights=weights,
    )
    return value


def mpc_candidates() -> list[tuple[float, float, float]]:
    triples = []
    for hp in [240, 300, 360, 420, 480]:
        for ho in [360, 480, 600, 720]:
            for ht in [300, 360, 420]:
                triples.append((float(hp), float(ho), float(ht)))
    return triples


def run_episode_external(
    env,
    variant: str,
    n_fleet: int,
    rng: np.random.RandomState,
    demand_noise: float,
    route_headway_target_s: float | None = None,
    objective_wait_metric: str = "observed",
    objective_weights: dict | None = None,
    exact_schedule: bool = False,
    schedule_contract_version: str = "v4",
):
    env._n_fleet_target = int(n_fleet)
    fixed_target = fixed_headway_target_s(variant)
    if fixed_target is None and route_headway_target_s is not None:
        fixed_target = float(route_headway_target_s)
    base_target = DEFAULT_FIXED_HEADWAY_S if fixed_target is None else fixed_target
    chosen_triple = (base_target, base_target, base_target)
    scheduled_targets = {0: [], 1: [], 2: []}
    candidates = None
    demand_proxy = 1.0
    if variant == "rule_mpc":
        candidates = mpc_candidates()
        demand_proxy = float(np.clip(
            rng.normal(1.0, max(float(demand_noise), 1e-9)), 0.3, 2.0))

    def schedule_target(direction, predecessor_launch, trip):
        nonlocal chosen_triple
        hour = int(
            getattr(env, "_service_start_hour", 6)
            + float(predecessor_launch) // 3600)
        slot = hour_to_slot(hour)
        if variant == "rule_mpc":
            chosen_triple = mpc_plan(
                current_hour=hour,
                last_dispatch_time_per_dir=None,
                episode_budget_n=n_fleet,
                current_demand_proxy=demand_proxy,
                candidates=candidates,
            )
            target = float(chosen_triple[slot])
        else:
            target = float(base_target)
        scheduled_targets[slot].append(target)
        return target

    def upper_cb(s_upper, trip):
        nonlocal chosen_triple
        trip._upper_queried = True
        if exact_schedule:
            return float(trip.target_headway)
        if variant == "rule_mpc":
            hour = int(
                getattr(env, "_service_start_hour", 6)
                + trip.launch_time // 3600)
            chosen_triple = mpc_plan(
                current_hour=hour,
                last_dispatch_time_per_dir=None,
                episode_budget_n=n_fleet,
                current_demand_proxy=demand_proxy,
                candidates=candidates,
            )
            return float(chosen_triple[hour_to_slot(hour)])
        return float(base_target)

    env.reset()
    projection_mode = "legacy_headway_enforcement"
    if exact_schedule:
        contract_label = str(schedule_contract_version).strip().lower()
        if contract_label not in {"v4", "v5", "v6"}:
            raise ValueError("exact schedule contract must be v4, v5, or v6")
        if variant == "rule_mpc":
            projection_mode = f"exact_rule_mpc_schedule_{contract_label}"
        elif route_headway_target_s is not None:
            projection_mode = f"exact_route_headway_schedule_{contract_label}"
        else:
            projection_mode = f"exact_fixed_headway_schedule_{contract_label}"
        install_exact_dispatch_schedule(
            env, schedule_target, projection_mode=projection_mode)
    env._upper_policy_callback = upper_cb
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: 0.0 for k in range(env.max_agent_num)}
    last_target = {k: 360.0 for k in range(env.max_agent_num)}

    while not env.done:
        for key in state_dict:
            obs_list = state_dict[key]
            if not obs_list:
                continue
            obs = np.array(obs_list[0], dtype=np.float32)
            for bus in env.bus_all:
                if bus.bus_id == int(obs[0]) and bus.on_route:
                    last_target[key] = float(getattr(
                        bus, "_freqduet_dispatch_target_headway", 360.0))
                    break
            if fixed_target is not None:
                action_dict[key] = 0.0
            else:
                action_dict[key] = rule_holding_action(obs, last_target[key])
            if len(obs_list) == 2:
                state_dict[key] = state_dict[key][1:]
        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    z = env.measurement_vector
    details = env.measurement_details
    wait = float(z[0])
    peak_fleet = float(z[1])
    cv = float(z[2])
    overshoot = max(0.0, peak_fleet - float(n_fleet))
    service_cost_by_wait = service_cost_views(
        details,
        peak_fleet=peak_fleet,
        headway_cv=cv,
        n_fleet=n_fleet,
        weights=objective_weights,
    )
    observed_service_cost = service_cost_by_wait["observed"]
    restricted_service_cost = service_cost_by_wait["restricted"]
    wait_metric = normalize_wait_metric(objective_wait_metric)
    selected_service_cost = service_cost_by_wait[wait_metric]
    target_summary = []
    for slot, fallback in enumerate(chosen_triple):
        values = scheduled_targets[slot]
        target_summary.append(
            float(np.mean(values)) if values else float(fallback))
    return {
        "avg_wait_min": wait,
        "peak_fleet": peak_fleet,
        "headway_cv": cv,
        "avg_wait_observed_min": float(details["avg_wait_observed_min"]),
        "restricted_wait_horizon_min": float(
            details["restricted_wait_horizon_min"]),
        "avg_in_vehicle_observed_min": float(
            details["avg_in_vehicle_observed_min"]),
        "restricted_in_vehicle_horizon_min": float(
            details["restricted_in_vehicle_horizon_min"]),
        "avg_total_journey_observed_min": float(
            details["avg_total_journey_observed_min"]),
        "restricted_total_journey_horizon_min": float(
            details["restricted_total_journey_horizon_min"]),
        "passengers_generated": int(details["passengers_generated"]),
        "passengers_unserved": int(details["passengers_unserved"]),
        "passenger_unserved_rate": float(details["passenger_unserved_rate"]),
        "headway_sample_count": int(details["headway_sample_count"]),
        "trips_unlaunched": int(details["trips_unlaunched"]),
        "trip_launch_rate": float(details["trip_launch_rate"]),
        "trips_completed": int(details["trips_completed"]),
        "trips_incomplete": int(details["trips_incomplete"]),
        "trip_completion_rate": float(details["trip_completion_rate"]),
        "done_reason": str(details["done_reason"]),
        "scenario_tape_id": str(details["scenario_tape_id"]),
        "N_fleet": int(n_fleet),
        "physical_vehicle_count": int(details.get(
            "physical_vehicle_count", len(env.bus_all))),
        "fleet_inventory_mode": str(details.get(
            "fleet_inventory_mode", env.fleet_inventory_mode)),
        "fleet_denied_dispatch_events": int(details.get(
            "fleet_denied_dispatch_events", 0)),
        "fleet_denied_retry_trip_seconds": float(details.get(
            "fleet_denied_retry_trip_seconds", 0.0)),
        "fleet_denied_trips": int(details.get("fleet_denied_trips", 0)),
        "fleet_denied_trip_rate": float(details.get(
            "fleet_denied_trip_rate", 0.0)),
        "fleet_readiness_delay_mean_s": float(details.get(
            "fleet_readiness_delay_mean_s", 0.0)),
        "fleet_readiness_delay_max_s": float(details.get(
            "fleet_readiness_delay_max_s", 0.0)),
        "holding_vehicle_seconds": float(details.get(
            "holding_vehicle_seconds", 0.0)),
        "holding_vehicle_seconds_per_launched_trip": float(details.get(
            "holding_vehicle_seconds_per_launched_trip", 0.0)),
        "holding_passenger_seconds": float(details.get(
            "holding_passenger_seconds", 0.0)),
        "holding_passenger_min_per_generated": float(details.get(
            "holding_passenger_min_per_generated", 0.0)),
        "commanded_holding_vehicle_seconds": float(details.get(
            "commanded_holding_vehicle_seconds", 0.0)),
        "commanded_holding_passenger_seconds": float(details.get(
            "commanded_holding_passenger_seconds", 0.0)),
        "commanded_holding_passenger_min_per_generated": float(details.get(
            "commanded_holding_passenger_min_per_generated", 0.0)),
        "terminal_actual_dispatch_gap_mean_s": float(details.get(
            "terminal_actual_dispatch_gap_mean_s", 0.0)),
        "terminal_dispatch_execution_error_mean_s": float(details.get(
            "terminal_dispatch_execution_error_mean_s", 0.0)),
        "terminal_dispatch_execution_error_abs_mean_s": float(details.get(
            "terminal_dispatch_execution_error_abs_mean_s", 0.0)),
        "fleet_overshoot": overshoot,
        "composite": selected_service_cost,
        "service_cost_wait_metric": wait_metric,
        "service_cost_observed": observed_service_cost,
        "service_cost_restricted": restricted_service_cost,
        "target_peak": target_summary[0],
        "target_offpeak": target_summary[1],
        "target_transition": target_summary[2],
        "baseline_schedule_projection_mode": projection_mode,
        "lower_observation_contract": str(details.get(
            "lower_observation_contract", env.lower_observation_contract)),
        "headway_reward_mode": str(details.get(
            "headway_reward_mode", env.headway_reward_mode)),
        "frequency_share_max_error": float(details.get(
            "frequency_share_max_error", 0.0) or 0.0),
    }


def run_one(
    config: str,
    variant: str,
    seed: int,
    episodes: int,
    logs_dir: Path,
    worker_threads: int | None = None,
    headway_policy: dict[tuple[str, str, int | None], float] | None = None,
    policy_default_headway_s: float | None = None,
    direct_scenario_seed: bool = False,
) -> Path:
    if direct_scenario_seed and int(episodes) != 1:
        raise ValueError("direct scenario-seed mode requires episodes=1")
    protocol_version = protocol_version_for_config(config)
    if protocol_version == V6_PROTOCOL_VERSION and not direct_scenario_seed:
        raise ValueError(
            "V6 external evidence requires direct scenario-seed mode")
    apply_worker_threads(worker_threads)
    run_dir = run_dir_for(config, variant, seed, logs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env, cfg = make_env_from_config(config)
    upper_cfg = cfg.get("upper", {})
    env_cfg = cfg.get("env", {})
    rng = np.random.RandomState(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    route_headway_target = None
    if is_route_headway_policy_variant(variant):
        route_headway_target = route_policy_target_s(
            headway_policy or {},
            variant=variant,
            config=config,
            seed=int(seed),
            default_headway_s=policy_default_headway_s,
        )
    rows = []
    t_start = time.time()
    loaded_protocol_version = str(
        (cfg.get("protocol", {}) or {}).get("version", "legacy"))
    if loaded_protocol_version != protocol_version:
        raise ValueError("resolved and loaded protocol versions differ")
    frozen_protocol = protocol_version.startswith("freqduet-eval-v")
    objective_cfg = cfg.get("objective", {}) or {}
    for ep in range(int(episodes)):
        scenario_seed = (
            int(seed) if direct_scenario_seed
            else int(seed) * 1000003 + int(ep)
        )
        env.scenario_seed = scenario_seed
        if (upper_cfg.get("fleet_mode", "fixed") == "elastic"
                and not frozen_protocol):
            n_fleet = int(rng.randint(int(upper_cfg.get("fleet_min", 8)), int(upper_cfg.get("fleet_max", 16)) + 1))
        else:
            n_fleet = int(upper_cfg.get("N_fleet", 12))
        t0 = time.time()
        row = run_episode_external(
            env,
            variant=variant,
            n_fleet=n_fleet,
            rng=rng,
            demand_noise=float(env_cfg.get("demand_noise", 0.0)),
            route_headway_target_s=route_headway_target,
            objective_wait_metric=str(
                objective_cfg.get("wait_metric", "observed")),
            objective_weights=dict(objective_cfg.get("weights", {}) or {}),
            exact_schedule=(protocol_version in {
                "freqduet-eval-v4", "freqduet-eval-v5",
                V6_PROTOCOL_VERSION}),
            schedule_contract_version=(
                "v6" if protocol_version == V6_PROTOCOL_VERSION
                else "v5" if protocol_version == "freqduet-eval-v5"
                else "v4"),
        )
        row.update({
            "ep": ep,
            "variant": variant,
            "config": config,
            "domain": infer_domain(config),
            "seed": int(seed),
            "scenario_seed": scenario_seed,
            "eval_seed": scenario_seed if direct_scenario_seed else "",
            "protocol_version": protocol_version,
            "wall_s": round(time.time() - t0, 3),
        })
        rows.append(row)
        if ep % 5 == 0 or ep == int(episodes) - 1:
            print(
                f"{variant} {config} seed={seed} ep={ep:03d} "
                f"N={row['N_fleet']:2d} wait={row['avg_wait_min']:.2f} "
                f"cv={row['headway_cv']:.3f} over={row['fleet_overshoot']:.1f} "
                f"comp={row['composite']:.3f}"
            )

    diag_path = run_dir / "diagnostics.csv"
    with diag_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "summary.json").open("w") as f:
        json.dump({
            "config": config,
            "domain": infer_domain(config),
            "variant": variant,
            "seed": int(seed),
            "episodes": int(episodes),
            "wall_s": round(time.time() - t_start, 3),
        }, f, indent=2)
    manifest_version = (
        V6_RUN_MANIFEST_VERSION
        if protocol_version == V6_PROTOCOL_VERSION
        else "freqduet-external-baseline-v5"
        if protocol_version == "freqduet-eval-v5"
        else "freqduet-external-baseline-v4"
    )
    manifest = {
        "manifest_version": manifest_version,
        "protocol_version": protocol_version,
        "config_name": config,
        "config_fingerprint": config_fingerprint(config),
        "core_source_fingerprint": source_fingerprint(),
        "evaluator_source_fingerprint": external_evaluator_fingerprint(),
        "variant": variant,
        "seed": int(seed),
        "direct_scenario_seed": bool(direct_scenario_seed),
        "scenario_seeds": [int(row["scenario_seed"]) for row in rows],
        "episodes": int(episodes),
    }
    if protocol_version == V6_PROTOCOL_VERSION:
        diagnostics_frame = pd.read_csv(diag_path)
        diagnostics_key = (
            V6_DIAGNOSTICS_PRIMARY_KEY if direct_scenario_seed
            else ["config", "variant", "seed", "ep"])
        argv = [str(value) for value in sys.argv]
        manifest.update({
            "method": variant,
            "scenario_seed": (
                int(rows[0]["scenario_seed"]) if len(rows) == 1 else None),
            "scenario_contract": scenario_contract_fingerprint(config),
            "runtime_environment": locked_runtime_environment(),
            "git": git_provenance(),
            "launch_argv": {
                "sha256": canonical_json_sha256(argv),
                "argv": argv,
            },
            "command_contract": external_command_contract(
                config=config,
                variant=variant,
                seed=seed,
                episodes=episodes,
                direct_scenario_seed=direct_scenario_seed,
                worker_threads=worker_threads,
                route_headway_target_s=route_headway_target,
            ),
            "diagnostics_artifact": csv_artifact_record(
                diag_path, diagnostics_frame, diagnostics_key),
        })
    with (run_dir / EXTERNAL_BASELINE_MANIFEST).open("w") as f:
        json.dump(manifest, f, indent=2)
    return run_dir


def selected_jobs(
    configs: list[str],
    variants: list[str],
    seeds: list[int],
    job_start: int | None = None,
    job_end: int | None = None,
) -> list[tuple[str, str, int]]:
    jobs = []
    for config in configs:
        for variant in variants:
            for seed in seeds:
                jobs.append((config, variant, seed))
    total = len(jobs)
    if job_start is None and job_end is None:
        return jobs
    start = 0 if job_start is None else max(0, int(job_start))
    end = total if job_end is None else min(total, int(job_end))
    if end < start:
        end = start
    print(f"Shard jobs [{start},{end}) of {total}")
    return jobs[start:end]


def run_jobs(
    configs: list[str],
    variants: list[str],
    seeds: list[int],
    episodes: int,
    logs_dir: Path,
    workers: int,
    skip_existing: bool = False,
    worker_threads: int | None = None,
    job_start: int | None = None,
    job_end: int | None = None,
    headway_policy: dict[tuple[str, str, int | None], float] | None = None,
    policy_default_headway_s: float | None = None,
    direct_scenario_seeds: bool = False,
) -> None:
    jobs = []
    for config, variant, seed in selected_jobs(
        configs, variants, seeds, job_start=job_start, job_end=job_end
    ):
        run_dir = run_dir_for(config, variant, seed, logs_dir)
        if skip_existing and diagnostics_complete(run_dir, episodes):
            if direct_scenario_seeds:
                validate_direct_baseline_run(
                    run_dir, pd.read_csv(run_dir / "diagnostics.csv"))
            print(
                f"SKIP {config} {variant} seed={seed}: "
                f"diagnostics already has >= {episodes} rows"
            )
            continue
        jobs.append((config, variant, seed))
    if not jobs:
        return

    workers = max(1, int(workers))
    if workers == 1:
        for config, variant, seed in jobs:
            run_dir = run_one(
                config, variant, seed, episodes, logs_dir,
                worker_threads=worker_threads,
                headway_policy=headway_policy,
                policy_default_headway_s=policy_default_headway_s,
                direct_scenario_seed=direct_scenario_seeds,
            )
            print(f"DONE {config} {variant} seed={seed}: {run_dir}")
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_one, config, variant, seed, episodes, logs_dir,
                worker_threads,
                headway_policy,
                policy_default_headway_s,
                direct_scenario_seeds,
            )
            for config, variant, seed in jobs
        ]
        for fut in as_completed(futures):
            run_dir = fut.result()
            print(f"DONE {run_dir.name}: {run_dir}")


def validate_direct_baseline_run(run_dir: Path, frame: pd.DataFrame) -> dict:
    manifest_path = run_dir / EXTERNAL_BASELINE_MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"{run_dir}: missing {EXTERNAL_BASELINE_MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    required_columns = {
        "config", "variant", "seed", "scenario_seed", "eval_seed",
        "protocol_version", "scenario_tape_id", "fleet_inventory_mode",
        "lower_observation_contract", "headway_reward_mode",
        "baseline_schedule_projection_mode", "physical_vehicle_count",
        "N_fleet", "peak_fleet", "frequency_share_max_error",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{run_dir}: missing direct columns {missing_columns}")
    versions = set(frame["protocol_version"].astype(str))
    if len(versions) != 1:
        raise ValueError(f"{run_dir}: mixed protocol versions")
    protocol_version = next(iter(versions))
    if protocol_version not in {
            "freqduet-eval-v4", "freqduet-eval-v5",
            V6_PROTOCOL_VERSION}:
        raise ValueError(f"{run_dir}: unsupported frozen protocol")
    expected_manifest_version = (
        V6_RUN_MANIFEST_VERSION
        if protocol_version == V6_PROTOCOL_VERSION
        else "freqduet-external-baseline-v5"
        if protocol_version == "freqduet-eval-v5"
        else "freqduet-external-baseline-v4")
    if manifest.get("manifest_version") != expected_manifest_version:
        raise ValueError(f"{manifest_path}: manifest version mismatch")
    if len(frame) != 1 or frame["eval_seed"].duplicated().any():
        raise ValueError(
            f"{run_dir}: direct mode requires exactly one unique eval seed")
    expected_seeds = frame["scenario_seed"].astype(int).tolist()
    if manifest.get("scenario_seeds") != expected_seeds:
        raise ValueError(f"{manifest_path}: scenario seeds do not match rows")
    if not bool(manifest.get("direct_scenario_seed")):
        raise ValueError(f"{manifest_path}: direct scenario mode is not locked")
    config = str(frame["config"].iloc[0])
    variant = str(frame["variant"].iloc[0])
    run_seed = int(frame["seed"].iloc[0])
    eval_seed = int(pd.to_numeric(frame["eval_seed"], errors="raise").iloc[0])
    scenario_seed = int(frame["scenario_seed"].iloc[0])
    if eval_seed != scenario_seed or run_seed != scenario_seed:
        raise ValueError(
            f"{run_dir}: direct seed, scenario seed, and eval seed differ")
    expected_manifest_values = {
        "protocol_version": protocol_version,
        "config_name": config,
        "variant": variant,
        "seed": run_seed,
        "episodes": 1,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{manifest_path}: {key} does not match diagnostics")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError(f"{manifest_path}: config fingerprint mismatch")
    if manifest.get("core_source_fingerprint") != source_fingerprint():
        raise ValueError(f"{manifest_path}: core source fingerprint mismatch")
    if manifest.get(
            "evaluator_source_fingerprint") != external_evaluator_fingerprint():
        raise ValueError(f"{manifest_path}: evaluator fingerprint mismatch")
    if protocol_version == V6_PROTOCOL_VERSION:
        if manifest.get("method") != variant:
            raise ValueError(f"{manifest_path}: method/variant mismatch")
        if manifest.get("scenario_seed") != scenario_seed:
            raise ValueError(f"{manifest_path}: scenario seed mismatch")
        expected_scenario = scenario_contract_fingerprint(config)
        scenario_payload = _validate_self_hashed_record(
            manifest.get("scenario_contract"),
            payload_key="payload",
            label="scenario contract",
        )
        if scenario_payload.get(
                "contract_version") != V6_SCENARIO_CONTRACT_VERSION:
            raise ValueError(f"{manifest_path}: scenario contract version mismatch")
        if manifest.get("scenario_contract") != expected_scenario:
            raise ValueError(f"{manifest_path}: scenario contract mismatch")
        runtime_payload = _validate_self_hashed_record(
            manifest.get("runtime_environment"),
            payload_key="environment",
            label="runtime environment",
        )
        if not {"python", "python_executable", "platform", "packages"}.issubset(
                runtime_payload):
            raise ValueError(f"{manifest_path}: runtime environment is incomplete")
        launch_argv = manifest.get("launch_argv")
        if not isinstance(launch_argv, dict) or not isinstance(
                launch_argv.get("argv"), list):
            raise ValueError(f"{manifest_path}: launch argv is missing")
        if (not launch_argv["argv"]
                or not all(isinstance(value, str)
                           for value in launch_argv["argv"])):
            raise ValueError(f"{manifest_path}: launch argv is invalid")
        if canonical_json_sha256(launch_argv["argv"]) != str(
                launch_argv.get("sha256", "")):
            raise ValueError(f"{manifest_path}: launch argv hash mismatch")
        command_payload = _validate_self_hashed_record(
            manifest.get("command_contract"),
            payload_key="payload",
            label="command contract",
        )
        expected_command_values = {
            "version": V6_COMMAND_CONTRACT_VERSION,
            "entrypoint": "scripts/run_freqduet_external_baselines.py",
            "config_name": config,
            "variant": variant,
            "method": variant,
            "seed": run_seed,
            "episodes": 1,
            "direct_scenario_seed": True,
        }
        for key, expected in expected_command_values.items():
            if command_payload.get(key) != expected:
                raise ValueError(
                    f"{manifest_path}: command contract {key} mismatch")
        git = manifest.get("git")
        git_commit = str(git.get("commit", "")) if isinstance(git, dict) else ""
        if (not isinstance(git, dict)
                or len(git_commit) != 40
                or any(char not in "0123456789abcdef" for char in git_commit)
                or not isinstance(git.get("tracked_dirty"), bool)):
            raise ValueError(f"{manifest_path}: Git provenance is incomplete")
        locked_frame = validate_csv_artifact(
            run_dir / "diagnostics.csv",
            manifest.get("diagnostics_artifact"),
            expected_primary_key=V6_DIAGNOSTICS_PRIMARY_KEY,
        )
        if len(locked_frame) != 1:
            raise ValueError(f"{manifest_path}: diagnostics artifact is not direct")
        safety_metrics = V6_SAFETY_METRICS
    elif protocol_version == "freqduet-eval-v5":
        safety_metrics = V5_SAFETY_METRICS
    else:
        safety_metrics = V4_SAFETY_METRICS
    missing_safety = sorted(set(safety_metrics) - set(frame.columns))
    if missing_safety:
        raise ValueError(
            f"{run_dir}: missing frozen safety metrics {missing_safety}")
    for metric in safety_metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{run_dir}: non-finite safety metric {metric}")
    required_values = {
        "protocol_version": protocol_version,
        "fleet_inventory_mode": "fixed_pool",
        "lower_observation_contract": "deployable_apc_avl_v4",
        "headway_reward_mode": "forward_event_only",
    }
    for column, expected in required_values.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"{run_dir}: {column} contract mismatch")
    if not frame["baseline_schedule_projection_mode"].astype(
            str).str.startswith("exact_").all():
        raise ValueError(f"{run_dir}: external schedule is not exact")
    if (protocol_version == V6_PROTOCOL_VERSION
            and not frame["baseline_schedule_projection_mode"].astype(
                str).str.endswith("_v6").all()):
        raise ValueError(f"{run_dir}: external schedule is not V6 exact")
    physical = pd.to_numeric(frame["physical_vehicle_count"], errors="raise")
    fleet = pd.to_numeric(frame["N_fleet"], errors="raise")
    peak = pd.to_numeric(frame["peak_fleet"], errors="raise")
    if not physical.eq(fleet).all() or not peak.le(physical).all():
        raise ValueError(f"{run_dir}: physical fleet conservation failed")
    share_error = pd.to_numeric(
        frame["frequency_share_max_error"], errors="raise")
    if not share_error.le(1e-9).all():
        raise ValueError(f"{run_dir}: frequency-share conservation failed")
    if frame["scenario_tape_id"].isna().any():
        raise ValueError(f"{run_dir}: missing scenario tape identifier")
    if protocol_version == V6_PROTOCOL_VERSION:
        for realized, commanded in [
            ("holding_vehicle_seconds", "commanded_holding_vehicle_seconds"),
            ("holding_passenger_seconds", "commanded_holding_passenger_seconds"),
            ("holding_passenger_min_per_generated",
             "commanded_holding_passenger_min_per_generated"),
        ]:
            realized_values = pd.to_numeric(frame[realized], errors="raise")
            commanded_values = pd.to_numeric(frame[commanded], errors="raise")
            if not realized_values.le(commanded_values + 1e-9).all():
                raise ValueError(
                    f"{run_dir}: realized {realized} exceeds commanded value")
        signed_error = pd.to_numeric(
            frame["terminal_dispatch_execution_error_mean_s"], errors="raise")
        absolute_error = pd.to_numeric(
            frame["terminal_dispatch_execution_error_abs_mean_s"],
            errors="raise")
        if not signed_error.abs().le(absolute_error + 1e-9).all():
            raise ValueError(
                f"{run_dir}: terminal execution error moments are inconsistent")
    return manifest


def summarize_run(run_dir: Path, last_k: int) -> dict:
    df = pd.read_csv(run_dir / "diagnostics.csv")
    direct_eval = (
        "eval_seed" in df
        and pd.to_numeric(df["eval_seed"], errors="coerce").notna().all()
    )
    run_manifest = None
    if direct_eval:
        run_manifest = validate_direct_baseline_run(run_dir, df)
    tail = df.iloc[-min(int(last_k), len(df)):]
    row = {
        "config": str(tail["config"].iloc[0]),
        "domain": str(tail["domain"].iloc[0]),
        "method": str(tail["variant"].iloc[0]),
        "seed": int(tail["seed"].iloc[0]),
        "episodes": int(len(df)),
        "logs_dir": str(run_dir.parent),
        "protocol_version": str(tail["protocol_version"].iloc[0]),
    }
    if "eval_seed" in tail and pd.to_numeric(
            tail["eval_seed"], errors="coerce").notna().all():
        row["eval_seed"] = int(pd.to_numeric(
            tail["eval_seed"], errors="raise").iloc[0])
    if "scenario_tape_id" in tail and tail["scenario_tape_id"].nunique() == 1:
        row["scenario_tape_id"] = str(tail["scenario_tape_id"].iloc[0])
    if (run_manifest is not None
            and row["protocol_version"] == V6_PROTOCOL_VERSION):
        row["scenario_contract_sha256"] = _fingerprint_sha(
            run_manifest.get("scenario_contract"), "scenario contract")
    for out_col, src_col in [
        ("wait", "avg_wait_min"),
        ("cv", "headway_cv"),
        ("overshoot", "fleet_overshoot"),
        ("composite", "composite"),
        ("avg_wait_observed_min", "avg_wait_observed_min"),
        ("restricted_wait_horizon_min", "restricted_wait_horizon_min"),
        ("avg_in_vehicle_observed_min", "avg_in_vehicle_observed_min"),
        ("restricted_in_vehicle_horizon_min", "restricted_in_vehicle_horizon_min"),
        ("avg_total_journey_observed_min", "avg_total_journey_observed_min"),
        ("restricted_total_journey_horizon_min", "restricted_total_journey_horizon_min"),
        ("headway_cv", "headway_cv"),
        ("fleet_overshoot", "fleet_overshoot"),
        ("target_peak", "target_peak"),
        ("target_offpeak", "target_offpeak"),
        ("target_transition", "target_transition"),
        ("service_cost_observed", "service_cost_observed"),
        ("service_cost_restricted", "service_cost_restricted"),
        ("passenger_unserved_rate", "passenger_unserved_rate"),
        ("trip_launch_rate", "trip_launch_rate"),
        ("trip_completion_rate", "trip_completion_rate"),
        ("physical_vehicle_count", "physical_vehicle_count"),
        ("fleet_denied_dispatch_events", "fleet_denied_dispatch_events"),
        ("fleet_denied_retry_trip_seconds",
         "fleet_denied_retry_trip_seconds"),
        ("fleet_denied_trips", "fleet_denied_trips"),
        ("fleet_denied_trip_rate", "fleet_denied_trip_rate"),
        ("fleet_readiness_delay_mean_s", "fleet_readiness_delay_mean_s"),
        ("fleet_readiness_delay_max_s", "fleet_readiness_delay_max_s"),
        ("holding_vehicle_seconds", "holding_vehicle_seconds"),
        ("holding_vehicle_seconds_per_launched_trip",
         "holding_vehicle_seconds_per_launched_trip"),
        ("holding_passenger_seconds", "holding_passenger_seconds"),
        ("holding_passenger_min_per_generated",
         "holding_passenger_min_per_generated"),
        ("commanded_holding_vehicle_seconds",
         "commanded_holding_vehicle_seconds"),
        ("commanded_holding_passenger_seconds",
         "commanded_holding_passenger_seconds"),
        ("commanded_holding_passenger_min_per_generated",
         "commanded_holding_passenger_min_per_generated"),
        ("terminal_actual_dispatch_gap_mean_s",
         "terminal_actual_dispatch_gap_mean_s"),
        ("terminal_dispatch_execution_error_mean_s",
         "terminal_dispatch_execution_error_mean_s"),
        ("terminal_dispatch_execution_error_abs_mean_s",
         "terminal_dispatch_execution_error_abs_mean_s"),
    ]:
        row[out_col] = (
            float(pd.to_numeric(tail[src_col], errors="coerce").mean())
            if src_col in tail else float("nan")
        )
    for column in [
        "fleet_inventory_mode",
        "baseline_schedule_projection_mode",
        "lower_observation_contract",
        "headway_reward_mode",
    ]:
        if column in tail and tail[column].nunique() == 1:
            row[column] = str(tail[column].iloc[0])
    return row


def aggregate(
    logs_dirs: list[Path] | Path,
    out_dir: Path,
    last_k: int,
    *,
    configs: list[str] | None = None,
    variants: list[str] | None = None,
    eval_seeds: list[int] | None = None,
) -> None:
    if isinstance(logs_dirs, (str, Path)):
        logs_dirs = [Path(logs_dirs)]
    rows = []
    manifests = []
    diagnostics_paths = []
    for logs_dir in logs_dirs:
        for diag in sorted(Path(logs_dir).glob("*/diagnostics.csv")):
            rows.append(summarize_run(diag.parent, last_k=last_k))
            diagnostics_paths.append(diag.resolve())
            manifest_path = diag.parent / EXTERNAL_BASELINE_MANIFEST
            if manifest_path.exists():
                manifests.append(json.loads(manifest_path.read_text()))
    if not rows:
        roots = ", ".join(str(p) for p in logs_dirs)
        raise SystemExit(f"No diagnostics.csv found under {roots}")
    per_seed = pd.DataFrame(rows)
    direct_mode = "eval_seed" in per_seed and per_seed["eval_seed"].notna().all()
    if direct_mode and len(manifests) != len(per_seed):
        raise ValueError("direct external matrix has missing run manifests")
    protocol_versions = sorted(set(per_seed["protocol_version"].astype(str)))
    is_v6 = V6_PROTOCOL_VERSION in protocol_versions
    if is_v6 and protocol_versions != [V6_PROTOCOL_VERSION]:
        raise ValueError("V6 external matrix mixes protocol versions")
    if is_v6:
        if not direct_mode:
            raise ValueError("V6 external aggregation requires direct seed mode")
        if len(manifests) != len(per_seed):
            raise ValueError("V6 external matrix has missing run manifests")
        if configs is None or variants is None or eval_seeds is None:
            raise ValueError(
                "V6 aggregation requires exact configs, variants, and eval seeds")
        expected_configs = [Path(value).stem for value in configs]
        expected_methods = [str(value) for value in variants]
        expected_eval_seeds = [int(value) for value in eval_seeds]
        for label, values in [
            ("configs", expected_configs),
            ("methods", expected_methods),
            ("eval seeds", expected_eval_seeds),
        ]:
            if len(values) != len(set(values)):
                raise ValueError(f"V6 aggregation {label} must be unique")
        observed_configs = set(per_seed["config"].astype(str))
        observed_methods = set(per_seed["method"].astype(str))
        observed_eval_seeds = set(per_seed["eval_seed"].astype(int))
        if observed_configs != set(expected_configs):
            raise ValueError(
                "V6 external configs do not match the exact requested set")
        if observed_methods != set(expected_methods):
            raise ValueError(
                "V6 external methods do not match the exact requested set")
        if observed_eval_seeds != set(expected_eval_seeds):
            raise ValueError(
                "V6 external eval seeds do not match the exact requested set")
        if per_seed.duplicated(V6_PER_SEED_PRIMARY_KEY).any():
            raise ValueError("V6 external matrix has duplicate config/method/eval seed")
        expected_rows = (
            len(expected_configs) * len(expected_methods)
            * len(expected_eval_seeds))
        if len(per_seed) != expected_rows:
            raise ValueError(
                f"V6 external matrix has {len(per_seed)} rows, "
                f"expected {expected_rows}")
        expected_grid = {
            (config, method, seed)
            for config in expected_configs
            for method in expected_methods
            for seed in expected_eval_seeds
        }
        observed_grid = set(zip(
            per_seed["config"].astype(str),
            per_seed["method"].astype(str),
            per_seed["eval_seed"].astype(int),
        ))
        if observed_grid != expected_grid:
            raise ValueError("V6 external matrix is not a complete requested grid")
    core_hashes = {
        item["core_source_fingerprint"]["sha256"] for item in manifests
    }
    evaluator_hashes = {
        item["evaluator_source_fingerprint"]["sha256"] for item in manifests
    }
    if len(core_hashes) > 1 or len(evaluator_hashes) > 1:
        raise ValueError("external matrix mixes source fingerprints")
    input_diagnostics = []
    config_fingerprints: dict[str, dict[str, object]] = {}
    scenario_contracts: dict[str, dict[str, object]] = {}
    runtime_records: dict[str, dict[str, object]] = {}
    run_git_records: dict[tuple[str, bool], dict[str, object]] = {}
    if is_v6:
        manifest_by_key = {}
        for diag_path, manifest in zip(diagnostics_paths, manifests):
            config = str(manifest.get("config_name", ""))
            method = str(manifest.get("method", ""))
            scenario_seed = int(manifest.get("scenario_seed", -1))
            key = (config, method, scenario_seed)
            if key in manifest_by_key:
                raise ValueError(f"duplicate V6 external run manifest for {key}")
            manifest_by_key[key] = manifest
            artifact = manifest.get("diagnostics_artifact")
            validate_csv_artifact(
                diag_path,
                artifact,
                expected_primary_key=V6_DIAGNOSTICS_PRIMARY_KEY,
            )
            config_record = manifest.get("config_fingerprint")
            if config in config_fingerprints \
                    and config_fingerprints[config] != config_record:
                raise ValueError(f"V6 external config {config} mixes fingerprints")
            config_fingerprints[config] = config_record
            scenario_record = manifest.get("scenario_contract")
            if config in scenario_contracts \
                    and scenario_contracts[config] != scenario_record:
                raise ValueError(
                    f"V6 external config {config} mixes scenario contracts")
            scenario_contracts[config] = scenario_record
            runtime_record = manifest.get("runtime_environment")
            runtime_sha = _fingerprint_sha(runtime_record, "runtime environment")
            runtime_records[runtime_sha] = runtime_record
            git_record = manifest.get("git") or {}
            git_key = (
                str(git_record.get("commit", "")),
                bool(git_record.get("tracked_dirty")),
            )
            run_git_records[git_key] = git_record
            input_diagnostics.append({
                "path": str(diag_path),
                "config": config,
                "method": method,
                "eval_seed": scenario_seed,
                "artifact": artifact,
            })
        if set(manifest_by_key) != expected_grid:
            raise ValueError("V6 run manifests do not match the requested grid")
        if len(runtime_records) != 1:
            raise ValueError("V6 external matrix mixes runtime environments")
        if len(run_git_records) != 1:
            raise ValueError("V6 external matrix mixes run Git provenance")
        common_run_git = next(iter(run_git_records.values()))
        aggregate_git = git_provenance()
        if str(common_run_git.get("commit", "")) != str(
                aggregate_git.get("commit", "")):
            raise ValueError(
                "V6 external aggregation Git commit differs from its runs")
        scenario_hashes = {
            _fingerprint_sha(record, "scenario contract")
            for record in scenario_contracts.values()
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "external_baselines_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows = []
    for (config, domain, method), group in per_seed.groupby(
            ["config", "domain", "method"], sort=False):
        row = {
            "config": config,
            "domain": domain,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in [
            "wait", "cv", "overshoot", "composite",
            "avg_wait_observed_min", "restricted_wait_horizon_min",
            "avg_in_vehicle_observed_min",
            "restricted_in_vehicle_horizon_min",
            "avg_total_journey_observed_min",
            "restricted_total_journey_horizon_min",
            "headway_cv", "fleet_overshoot",
            "service_cost_observed", "service_cost_restricted",
            "passenger_unserved_rate", "trip_launch_rate",
            "trip_completion_rate", "fleet_denied_dispatch_events",
            "fleet_denied_retry_trip_seconds",
            "fleet_denied_trips", "fleet_readiness_delay_mean_s",
            "fleet_readiness_delay_max_s", "holding_vehicle_seconds",
            "holding_vehicle_seconds_per_launched_trip",
            "holding_passenger_seconds",
            "holding_passenger_min_per_generated", "fleet_denied_trip_rate",
            "commanded_holding_vehicle_seconds",
            "commanded_holding_passenger_seconds",
            "commanded_holding_passenger_min_per_generated",
            "terminal_actual_dispatch_gap_mean_s",
            "terminal_dispatch_execution_error_mean_s",
            "terminal_dispatch_execution_error_abs_mean_s",
        ]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "external_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)
    with (out_dir / "external_baselines_summary.json").open("w") as f:
        payload = {
            "manifest_version": (
                V6_SUMMARY_MANIFEST_VERSION if is_v6 else None),
            "protocol_version": (
                protocol_versions[0] if len(protocol_versions) == 1 else None),
            "logs_dirs": [str(p) for p in logs_dirs],
            "last_k": int(last_k),
            "n_rows": int(len(per_seed)),
            "direct_scenario_seeds": bool(direct_mode),
            "run_manifests_verified": bool(manifests),
            "protocol_versions": protocol_versions,
            "core_source_sha256": (
                next(iter(core_hashes)) if core_hashes else None),
            "evaluator_source_sha256": (
                next(iter(evaluator_hashes)) if evaluator_hashes else None),
        }
        if is_v6:
            payload.update({
                "strict_complete": True,
                "configs": expected_configs,
                "methods": expected_methods,
                "eval_seeds": expected_eval_seeds,
                "expected_rows": expected_rows,
                "primary_key": V6_PER_SEED_PRIMARY_KEY,
                "core_source_fingerprint": manifests[0][
                    "core_source_fingerprint"],
                "evaluator_source_fingerprint": manifests[0][
                    "evaluator_source_fingerprint"],
                "config_fingerprints": config_fingerprints,
                "scenario_contract_fingerprints": scenario_contracts,
                "scenario_contract": (
                    next(iter(scenario_contracts.values()))
                    if len(scenario_hashes) == 1 else None),
                "runtime_environment": next(iter(runtime_records.values())),
                "run_git_provenance": {
                    "commit": str(common_run_git["commit"]),
                    "tracked_dirty": bool(common_run_git["tracked_dirty"]),
                },
                "git": aggregate_git,
                "input_diagnostics": input_diagnostics,
                "artifacts": {
                    "external_baselines_per_seed.csv": csv_artifact_record(
                        per_seed_path, per_seed, V6_PER_SEED_PRIMARY_KEY),
                    "external_baselines_summary.csv": csv_artifact_record(
                        summary_path, summary, V6_SUMMARY_PRIMARY_KEY),
                },
            })
        json.dump(payload, f, indent=2)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT_DOMAIN_CONFIGS.values()))
    ap.add_argument("--configs-file", default=None,
                    help="file containing comma- or newline-separated config names")
    ap.add_argument("--variants", default="fixed_headway,rule_holding,rule_mpc")
    ap.add_argument("--seeds", default="42,123,456,789,2026")
    ap.add_argument(
        "--direct-scenario-seeds",
        action="store_true",
        help=(
            "interpret --seeds as frozen scenario-tape seeds directly; "
            "requires --episodes 1"
        ),
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--last-k", type=int, default=20)
    ap.add_argument("--logs-dir", default="logs_external_baselines")
    ap.add_argument("--aggregate-logs-dirs", default=None)
    ap.add_argument("--out-dir", default="results_freqduet/external_baselines")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="run selected jobs but skip summary writing; useful for scheduler shards")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel baseline processes")
    ap.add_argument("--worker-threads", type=int, default=None,
                    help="numeric-library threads per baseline process")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose diagnostics already has enough rows")
    ap.add_argument("--headway-policy-csv", default=None,
                    help="CSV mapping route policy variants/configs[/seeds] to target headways")
    ap.add_argument("--policy-default-headway", type=float, default=None,
                    help="fallback target for route policy variants missing from the CSV")
    ap.add_argument("--job-start", type=int, default=None,
                    help="flattened config x variant x seed start index for scheduler shards")
    ap.add_argument("--job-end", type=int, default=None,
                    help="flattened config x variant x seed end index for scheduler shards")
    args = ap.parse_args()

    configs = (
        parse_csv_file(args.configs_file)
        if args.configs_file else parse_csv_list(args.configs)
    )
    variants = parse_csv_list(args.variants)
    seeds = parse_csv_list(args.seeds, int)
    protocol_versions = {
        protocol_version_for_config(config) for config in configs}
    if (V6_PROTOCOL_VERSION in protocol_versions
            and not args.direct_scenario_seeds):
        raise SystemExit(
            "V6 external evidence requires --direct-scenario-seeds")
    if args.direct_scenario_seeds and int(args.episodes) != 1:
        raise SystemExit("--direct-scenario-seeds requires --episodes 1")
    unknown = sorted(v for v in set(variants) if not is_known_variant(v))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    if (
        not args.aggregate_only
        and any(is_route_headway_policy_variant(v) for v in variants)
        and not args.headway_policy_csv
    ):
        raise SystemExit(
            "route headway policy variants require --headway-policy-csv")
    headway_policy = load_headway_policy(args.headway_policy_csv)

    logs_dir = resolve_under_root(args.logs_dir)
    out_dir = resolve_under_root(args.out_dir)
    env_job_start = (
        os.environ.get("SCHEDULEURM_CPU_START")
        or os.environ.get("SCHEDULEURM_CPU_SHARD_START")
    )
    env_job_end = (
        os.environ.get("SCHEDULEURM_CPU_END")
        or os.environ.get("SCHEDULEURM_CPU_SHARD_END")
    )
    job_start = args.job_start
    job_end = args.job_end
    if job_start is None and env_job_start is not None:
        job_start = int(env_job_start)
    if job_end is None and env_job_end is not None:
        job_end = int(env_job_end)
    if not args.aggregate_only:
        run_jobs(
            configs=configs,
            variants=variants,
            seeds=seeds,
            episodes=args.episodes,
            logs_dir=logs_dir,
            workers=args.workers,
            skip_existing=args.skip_existing,
            worker_threads=args.worker_threads,
            job_start=job_start,
            job_end=job_end,
            headway_policy=headway_policy,
            policy_default_headway_s=args.policy_default_headway,
            direct_scenario_seeds=args.direct_scenario_seeds,
        )
    if args.no_aggregate:
        print("Skipped aggregation (--no-aggregate).")
        return

    if args.aggregate_logs_dirs:
        aggregate_logs_dirs = [
            resolve_under_root(p) for p in parse_csv_list(args.aggregate_logs_dirs)
        ]
    else:
        aggregate_logs_dirs = [logs_dir]
    aggregate(
        aggregate_logs_dirs,
        out_dir,
        args.last_k,
        configs=configs,
        variants=variants,
        eval_seeds=seeds,
    )


if __name__ == "__main__":
    main()
