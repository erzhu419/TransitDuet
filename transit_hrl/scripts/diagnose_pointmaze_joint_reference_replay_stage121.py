#!/usr/bin/env python3
"""Separate scalar replay from batch-kernel error on the first full native round."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freq_hrl.experiments import pointmaze_joint_reference as experiment
from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from freq_hrl.experiments.pointmaze_root_response import write_json
from scripts import pointmaze_joint_reference_stage121_spec as spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()
    torch.set_num_threads(1)
    root, period = args_cli.optimizer_seed, 50
    args = spec.arguments(root, preflight=False)
    models, predictor, _, calibration = experiment.source.load_source(root)
    model = models[str(period)]
    teacher = experiment.base.load_lower_state(root, period, protocol=spec)
    trainers = {method: experiment.make_trainer(model, teacher, args) for method in spec.METHODS}
    states = {method: experiment.weights(trainer) for method, trainer in trainers.items()}
    roster = spec.seed_roles(root, preflight=False)["training_rounds"][0]
    jobs = [(experiment.weights(model), teacher, states, r["scenario_seed"], r["noise_seeds"],
        period, predictor, calibration[str(period)]["envelope"], True) for r in roster]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
            initializer=experiment.source.native.init_worker, initargs=(model.config, args)) as pool:
        outputs = [out for group in pool.map(experiment.worker_group, jobs) for out in group["outputs"]]
    report = {}
    for method, trainer in trainers.items():
        joined = concat_hierarchical_batches([batch for name, batch, _ in outputs if name == method])
        for level, actor in (("upper", trainer.upper_actor), ("lower", trainer.lower_actor)):
            batch = getattr(joined, level)
            if not batch.size:
                continue
            largest = None
            for start in range(0, batch.size, spec.MINIBATCH):
                state = torch.as_tensor(batch.state[start:start + spec.MINIBATCH])
                action = torch.as_tensor(batch.action[start:start + spec.MINIBATCH])
                with torch.no_grad():
                    dist = actor.distribution(state)
                    lp = dist.log_prob(action).sum(-1)
                    error = np.abs(lp.numpy() - batch.old_logp[start:start + spec.MINIBATCH])
                    index = int(error.argmax())
                    scalar = actor.distribution(state[index:index + 1])
                    scalar_lp = scalar.log_prob(action[index:index + 1]).sum().item()
                    candidate = {"max_batch_saved_logp_error": float(error[index]),
                        "scalar_saved_logp_error": abs(scalar_lp - float(batch.old_logp[start + index])),
                        "batch_scalar_logp_error": abs(lp[index].item() - scalar_lp),
                        "batch_scalar_mean_error": float((dist.mean[index] - scalar.mean[0]).abs().max()),
                        "minimum_std": float(scalar.stddev.min()), "index": start + index}
                if largest is None or candidate["max_batch_saved_logp_error"] > largest["max_batch_saved_logp_error"]:
                    largest = candidate
            report[f"{method}/{level}"] = largest
    result = {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
        "kind": "native_likelihood_numerical_probe_only", "native_episodes": len(outputs),
        "native_steps": len(outputs) * args.horizon, "optimizer_steps": 0, "diagnostics": report}
    write_json(args_cli.output, result)
    write_json(args_cli.output.parent / "completion" / "ready.json", {"status": "complete"})
    print(json.dumps(result, sort_keys=True), flush=True)
    print("Eval complete: likelihood numerical probe written", flush=True)


if __name__ == "__main__":
    main()
