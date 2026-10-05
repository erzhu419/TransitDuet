#!/usr/bin/env python3
"""Compare independent-noise actor scores before any Stage121 policy update."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_joint_reference as experiment
from freq_hrl.experiments.pointmaze_actor_credit import cosine
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from scripts import pointmaze_joint_reference_stage121_spec as spec

PROTOCOL = "pointmaze_joint_reference_credit_stage122_v1"
ROOTS = tuple(spec.roots(preflight=False)[:2])


def phase_baseline(returns):
    """Exclude the current scenario; never mix the independent noise folds."""
    returns = np.asarray(returns, dtype=np.float64)
    baseline = np.empty_like(returns)
    for fold in (0, 1):
        rows = returns[fold::2]
        baseline[fold::2] = (rows.sum(axis=0) - rows) / (len(rows) - 1)
    return baseline


def loss_gradient(actor, batch, advantage, *, clip_ratio):
    parameters = [p for p in actor.parameters() if p.requires_grad]
    gradient = [torch.zeros_like(p) for p in parameters]
    advantage = experiment.FrequencySeparatedActorCriticPPO._normalize(advantage)
    error = 0.
    for start in range(0, batch.size, spec.MINIBATCH):
        stop = start + spec.MINIBATCH
        state = torch.as_tensor(batch.state[start:stop])
        action = torch.as_tensor(batch.action[start:stop])
        logp, _ = actor.log_prob_entropy(state, action)
        old_logp = torch.as_tensor(batch.old_logp[start:stop])
        error = max(error, float((logp.detach() - old_logp).abs().max()))
        ratio = torch.exp(logp - old_logp)
        clipped = ratio.clamp(1. - clip_ratio, 1. + clip_ratio)
        signal = torch.as_tensor(advantage[start:stop])
        loss = -torch.minimum(ratio * signal, clipped * signal).sum() / batch.size
        parts = torch.autograd.grad(loss, parameters)
        for total, part in zip(gradient, parts):
            total.add_(part.detach())
    if error > spec.LOGP_REPLAY_TOLERANCE:
        raise ValueError("Stage122 score does not reproduce the collected policy")
    return np.concatenate([g.double().numpy().reshape(-1) for g in gradient]), error


def diagnose_level(trainer, batches, level):
    levels = [getattr(b, level) for b in batches]
    if not levels[0].size:
        return None
    actor = getattr(trainer, level + "_actor")
    returns = np.stack([trainer._gae(b.reward, b.done, b.duration, b.old_value)[1] for b in levels])
    values = np.stack([b.old_value for b in levels])
    baseline = phase_baseline(returns)
    signals = {"source_critic": returns - values, "noise_fold_LOO_phase": returns - baseline}
    gradients, reports = {}, {}
    for name, signal in signals.items():
        gradients[name] = []
        replay = 0.
        for fold in (0, 1):
            batch = getattr(concat_hierarchical_batches(batches[fold::2]), level)
            gradient, error = loss_gradient(actor, batch, signal[fold::2].reshape(-1),
                clip_ratio=trainer.config.clip_ratio)
            gradients[name].append(gradient)
            replay = max(replay, error)
        variance = np.var(signal)
        temporal = np.var(signal.mean(axis=0)) / variance if variance > 0 else 0.
        reports[name] = {"advantage_std": float(signal.std()),
            "phase_mean_variance_fraction": float(temporal),
            "noise_fold_gradient_cosine": cosine(*gradients[name]),
            "noise_fold_gradient_norms": [float(np.linalg.norm(g)) for g in gradients[name]],
            "max_logp_replay_error": replay}
    pooled = {k: np.mean(v, axis=0) for k, v in gradients.items()}
    return {"episodes": len(batches), "decisions_per_episode": levels[0].size,
        "critic_MSE": float(np.mean(np.square(returns - values))), "signals": reports,
        "pooled_baseline_direction_cosine": cosine(*pooled.values())}


def run(root, output):
    if root not in ROOTS:
        raise ValueError("Stage122 uses the first two registered roots, not selected winners")
    args = spec.arguments(root, preflight=False)
    models, predictor, _, calibrations = experiment.source.load_source(root)
    roster = spec.seed_roles(root, preflight=False)["training_rounds"][0]
    groups = {}
    episodes = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
            initializer=experiment.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            teacher = experiment.base.load_lower_state(root, period, protocol=spec)
            trainers = {m: experiment.make_trainer(model, teacher, args) for m in spec.METHODS}
            states = {m: experiment.weights(t) for m, t in trainers.items()}
            snapshots = {m: copy.deepcopy(t.state_dict()) for m, t in trainers.items()}
            jobs = [(experiment.weights(model), teacher, states, r["scenario_seed"], r["noise_seeds"],
                period, predictor, calibrations[str(period)]["envelope"], True) for r in roster]
            rows = {m: [] for m in spec.METHODS}
            for registered, group in zip(roster, pool.map(experiment.worker_group, jobs)):
                if list(group["pairing"]) != registered["noise_seeds"]:
                    raise ValueError("Stage122 noise fold roster changed")
                for method, batch, row in group["outputs"]:
                    if row["seed"] != registered["scenario_seed"]:
                        raise ValueError("Stage122 scenario roster changed")
                    rows[method].append((batch, row))
                    episodes += 1
            groups[str(period)] = {}
            for method, trainer in trainers.items():
                batches = [b for b, _ in rows[method]]
                groups[str(period)][method] = {level: diagnose_level(trainer, batches, level)
                    for level in ("upper", "lower")}
                after = trainer.state_dict()
                if after.pop("config") != snapshots[method].pop("config"):
                    raise ValueError("Stage122 trainer configuration changed")
                torch.testing.assert_close(after, snapshots[method], atol=0, rtol=0)
            groups[str(period)]["initial_sampled_joint_minus_forecast"] = float(np.mean([
                a["episode_return"] - b["episode_return"]
                for (_, a), (_, b) in zip(rows["joint"], rows["forecast"])]))
            print(f"root={root} period={period}: frozen credit probe complete", flush=True)
    expected = len(spec.PERIODS) * len(spec.METHODS) * len(roster) * 2
    if episodes != expected:
        raise ValueError("Stage122 native path budget changed")
    result = {"status": "complete", "protocol": PROTOCOL, "root": root,
        "kind": "frozen_first_round_credit_diagnosis_not_native_gain_confirmation",
        "source_protocol": spec.EXPERIMENT_PROTOCOL, "source_training_round": 1,
        "roster": roster, "groups": groups, "native_episodes": episodes,
        "native_steps": episodes * args.horizon, "optimizer_steps": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0,
        "actor_critic_and_teacher_unchanged": "passed"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": PROTOCOL})
    print("Eval complete: frozen joint credit probe written", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.optimizer_seed, args.output)


if __name__ == "__main__":
    main()
