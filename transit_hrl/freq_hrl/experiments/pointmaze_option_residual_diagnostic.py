"""Explain the Stage112 learned-versus-forecast failure without retraining."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from . import pointmaze_option_residual as source
from . import pointmaze_option_residual_train as trainer
from .pointmaze_root_response import write_json
from scripts import pointmaze_option_residual_diagnostic_spec as spec


def load_weights(root, period, arm):
    payload = torch.load(spec.checkpoint(root, period, arm), map_location="cpu")
    if payload.get("protocol") != "pointmaze_option_residual_train_stage112_v1":
        raise ValueError("diagnostic checkpoint protocol changed")
    return payload["weights"]


def branch_metrics(model, state):
    actor = trainer.branch(model)
    tensor = torch.as_tensor(state, dtype=torch.float32)
    with torch.inference_mode():
        output = actor.distribution(tensor).mean
        base = actor.base.distribution(tensor).mean
    correction = (output - base).numpy()
    advice = np.asarray(state[:, 392:396], dtype=np.float64)
    norms = np.linalg.norm(advice, axis=1)
    correction_norms = np.linalg.norm(correction, axis=1)
    return {"advice_norm": float(np.mean(norms)),
        "residual_correction_norm": float(np.mean(correction_norms)),
        "advice": advice}


def diagnostic_episode(job):
    source_weights, learned_weights, forecast_weights, seed, period, predictor, calibration, args = job
    learned_batch, learned_row, _ = trainer.native_episode(source_weights, learned_weights,
        seed=seed, noise_seed=seed, arm="learned", period=period, predictor=predictor,
        calibration=calibration, args=args, collect=True)
    forecast_batch, forecast_row, _ = trainer.native_episode(source_weights, forecast_weights,
        seed=seed, noise_seed=seed, arm="forecast", period=period, predictor=predictor,
        calibration=calibration, args=args, collect=True)
    learned = branch_metrics(source.native._WORKER[0], learned_batch.state)
    forecast = branch_metrics(source.native._WORKER[0], forecast_batch.state)
    learned_advice = learned.pop("advice")
    forecast_advice = forecast.pop("advice")
    cosine = np.sum(learned_advice * forecast_advice, axis=1) / np.maximum(
        np.linalg.norm(learned_advice, axis=1) * np.linalg.norm(forecast_advice, axis=1), 1e-12)
    return {"seed": seed, "learned_return": learned_row["episode_return"],
        "forecast_return": forecast_row["episode_return"],
        "return_difference": learned_row["episode_return"] - forecast_row["episode_return"],
        "learned": learned, "forecast": forecast,
        "advice_cosine": float(np.mean(cosine)),
        "paired_noise": "same_policy_and_lower_seed"}


def bootstrap(values, *, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(spec.BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]


def aggregate(rows, period):
    metrics = {"return_difference": [row["return_difference"] for row in rows]}
    for arm in ("learned", "forecast"):
        for metric in ("advice_norm", "residual_correction_norm"):
            metrics[f"{arm}_{metric}"] = [row[arm][metric] for row in rows]
    metrics["advice_cosine"] = [row["advice_cosine"] for row in rows]
    result = {"episodes": len(rows), "seed_roster": [row["seed"] for row in rows]}
    for index, (name, values) in enumerate(metrics.items()):
        result[name] = {"mean": float(np.mean(values)), "ci": bootstrap(values, seed=spec.BOOTSTRAP_SEED + period + index)}
    return result


def run(root, *, output):
    if root not in spec.roots():
        raise ValueError("diagnostic root changed")
    source_cell = json.loads(spec.source_result(root).read_text())
    source.qualify(source_cell, preflight=False)
    models, predictor, _, calibrations = source.load_source(root)
    args = spec.arguments(root)
    jobs = []
    for period in spec.PERIODS:
        learned = load_weights(root, period, "learned")
        forecast = load_weights(root, period, "forecast")
        source_weights = source.native.joint.inference_weights(models[str(period)])
        for seed in spec.diagnostic_seeds(root):
            jobs.append((source_weights, learned, forecast, seed, period, predictor,
                calibrations[str(period)], args))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        rows = list(pool.map(diagnostic_episode, jobs))
    groups = {}
    offset = 0
    for period in spec.PERIODS:
        groups[str(period)] = aggregate(rows[offset:offset + spec.EPISODES_PER_PERIOD], period)
        offset += spec.EPISODES_PER_PERIOD
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "root": root,
        "contract": spec.contract(), "budget": spec.budget(), "groups": groups,
        "raw_artifacts": "server_only"}
    write_json(output, result)
    write_json(Path(output).parent / "completion" / "ready.json",
        {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root})
    return result
