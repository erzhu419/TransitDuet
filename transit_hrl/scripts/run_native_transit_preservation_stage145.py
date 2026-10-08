#!/usr/bin/env python3
"""Compare the preserved native runner before/after count-core extraction."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_preservation_stage145_v1"
NATIVE = ROOT / "native_freqduet"
ROOTS = (37, 49)


def contract(preflight):
    return {
        "purpose": "native_extraction_equivalence_not_performance_claim",
        "reference": "preserved_native_original_count_estimator",
        "candidate": "same_runner_using_freq_hrl_count_harmonic",
        "config": "F_freqduet_harmonic_hiro",
        "train_episodes": 2 if preflight else 32,
        "upper_warmup": 1 if preflight else 30,
        "eval_episodes": 1 if preflight else 2,
        "preflight_overrides": {
            "effective_trip_num": 24, "demand_end_time_s": 3600,
            "evaluation_end_time_s": 5400, "service_start_hour": 6,
            "service_end_hour": 7, "upper_batch_size": 8,
            "lower_batch_size": 32, "updates_per_episode_each_level": 2,
        } if preflight else {},
        "unchanged": ["physical_actions", "reward", "RE-SAC_backend",
                      "historical_OD_prior", "log_count_RLS", "feature_layout"],
        "comparison": "all_non_wall_episode_fields_actions_and_final_network_arrays_exact",
        "require_learning": "both_level_update_calls_and_actor_parameter_changes",
        "artifacts": "compact_JSON_only_temporary_arrays_removed_no_checkpoints",
    }


def model_arrays(runner):
    arrays = {}
    for level, trainer in (("lower", runner.lower_trainer), ("upper", runner.upper_trainer)):
        for name in ("policy_net", "q_net", "target_q_net", "cost_q_net", "target_cost_q_net"):
            module = getattr(trainer, name, None)
            if module is not None:
                for key, tensor in module.state_dict().items():
                    arrays[f"{level}/{name}/{key}"] = tensor.detach().cpu().numpy().copy()
    return arrays


def worker(args):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    from frequency import demand_frequency, intensity_estimator
    if args.worker == "original":
        demand_frequency.CausalHarmonicBandState = intensity_estimator.CausalHarmonicBandState
        demand_frequency.CausalNegativeBinomialHarmonicBandState = intensity_estimator.CausalNegativeBinomialHarmonicBandState
        frequency.fit_harmonic_prior = intensity_estimator.fit_harmonic_prior
    from runner_v3 import TransitDuetV2Runner, load_config

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml"))
    cfg["seed"] = args.seed
    cfg["logging"] = {"logs_dir": str(args.output.parent / "logs")}
    spec = contract(args.preflight)
    if args.preflight:
        overrides = spec["preflight_overrides"]
        cfg["env"].update({k: v for k, v in overrides.items() if k in {
            "effective_trip_num", "demand_end_time_s", "evaluation_end_time_s",
            "service_start_hour", "service_end_hour"}})
        cfg["coupling"]["upper_warmup_eps"] = spec["upper_warmup"]
        for level in ("upper", "lower"):
            cfg[level]["batch_size"] = overrides[f"{level}_batch_size"]
            cfg[level]["updates_per_episode"] = overrides["updates_per_episode_each_level"]
    runner = TransitDuetV2Runner(cfg, device="cpu")
    initial = model_arrays(runner)
    updates = {"upper": 0, "lower": 0}
    for level in updates:
        trainer = getattr(runner, f"{level}_trainer")
        original_update = trainer.update
        def counted_update(*pos, _level=level, _update=original_update, **kw):
            updates[_level] += 1
            return _update(*pos, **kw)
        trainer.update = counted_update
    arrays, rows = {}, []
    total_steps = 0
    for ep in range(spec["train_episodes"] + spec["eval_episodes"]):
        training = ep < spec["train_episodes"]
        # Native training seeds are untouched; held-out evaluation seeds are disjoint.
        options = {} if training else {"scenario_seed": 900000000 + args.seed * 100 + ep}
        row = runner.run_episode(ep, training=training, **options)
        rows.append({k: v for k, v in row.items() if k not in {"wall_env_s", "wall_train_s"}})
        total_steps += int(row["simulation_end_time_s"] / runner.env.time_step)
        arrays[f"actions/{ep}/upper"] = np.asarray(runner._ep_upper_deltas)
        arrays[f"actions/{ep}/lower"] = np.asarray(runner._ep_lower_actions)
        print(f"{args.worker} seed={args.seed} ep={ep} stage={row['stage']} "
              f"reward={row['ep_reward']} wait={row['avg_wait_min']} "
              f"updates={updates}", flush=True)
    final = model_arrays(runner)
    arrays.update(final)
    actor_change = {
        level: max(float(np.max(np.abs(final[k] - initial[k])))
                   for k in final if k.startswith(f"{level}/policy_net/"))
        for level in updates
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    write_json(args.output, {
        "rows": rows, "updates": updates, "actor_change_max_abs": actor_change,
        "native_steps": total_steps, "state_dims": {
            "upper": runner.upper_state_dim, "lower": runner.lower_state_dim},
        "estimator": type(runner.env.frequency_tracker.global_state).__module__,
        "endpoints": [{k: row[k] for k in ("ep", "stage", "ep_reward", "avg_wait_min",
                                            "headway_cv", "peak_fleet")} for row in rows],
    })


def compare(reference, candidate, reference_arrays, candidate_arrays):
    failures = []
    for key in ("rows", "updates", "actor_change_max_abs", "native_steps", "state_dims"):
        if reference[key] != candidate[key]:
            failures.append(key)
    if set(reference_arrays) != set(candidate_arrays):
        failures.append("array_keys")
    max_difference = 0.0
    for key in reference_arrays.keys() & candidate_arrays.keys():
        a, b = reference_arrays[key], candidate_arrays[key]
        if a.shape != b.shape or not np.array_equal(a, b):
            failures.append(key)
        if a.shape == b.shape and a.size:
            max_difference = max(max_difference, float(np.max(np.abs(a - b))))
    for level in ("upper", "lower"):
        if candidate["updates"][level] <= 0 or candidate["actor_change_max_abs"][level] <= 0:
            failures.append(f"no_{level}_learning")
    if candidate["estimator"] != "freq_hrl.encoders.count_harmonic":
        failures.append("shared_core_not_used")
    if reference["estimator"] != "frequency.intensity_estimator":
        failures.append("reference_core_not_used")
    return {
        "passed": not failures, "differences": failures,
        "max_action_or_network_abs_difference": max_difference,
        "arrays_compared": len(reference_arrays), "episode_fields_compared": len(candidate["rows"][0]),
        "updates_per_implementation": candidate["updates"],
        "actor_change_max_abs": candidate["actor_change_max_abs"],
        "state_dims": candidate["state_dims"],
        "native_steps_both_implementations": reference["native_steps"] + candidate["native_steps"],
        "endpoints": candidate["endpoints"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", choices=("original", "shared"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native_preservation_", dir=args.output.parent) as temp:
        paths = {}
        for implementation in ("original", "shared"):
            paths[implementation] = Path(temp) / implementation / "result.json"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--seed", str(args.seed), "--output", str(paths[implementation]),
                       "--worker", implementation]
            if args.preflight:
                command.append("--preflight")
            subprocess.run(command, cwd=ROOT, check=True)
        reference = json.loads(paths["original"].read_text())
        candidate = json.loads(paths["shared"].read_text())
        with np.load(paths["original"].with_suffix(".npz")) as a, np.load(paths["shared"].with_suffix(".npz")) as b:
            result = compare(reference, candidate, dict(a), dict(b))
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "seed": args.seed,
                             "preflight": args.preflight, "contract": contract(args.preflight), **result})
    if not result["passed"]:
        raise SystemExit(f"native extraction failed: {result['differences']}")
    print("NATIVE_PRESERVATION_PASS", flush=True)


if __name__ == "__main__":
    main()
