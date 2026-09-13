#!/usr/bin/env python3
"""Controlled same-state projection targets; no policy or performance evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.core.causal_terminal_reserve_projector import (  # noqa: E402
    CausalTerminalReserveProjector,
)

PROTOCOL = "mujoco_v25_same_state_target_noise_v1"
ROOTS = (3837595251, 1475821288, 4094869574, 449311583)
DIMENSIONS = (3, 6)
MEAN_AMPLITUDES = (0.0, 1.0, 2.5)
STANDARD_DEVIATIONS = (0.2, 0.5)
PREFIX_STEPS = (0, 32, 64)
UPPER_DRAWS = 8
LOWER_DRAWS = 16


def target_variance(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=(0, 1), keepdims=True)
    upper = values.mean(axis=1, keepdims=True) - mean
    lower = values.mean(axis=0, keepdims=True) - mean
    interaction = values - mean - upper - lower
    return {
        "total": float(np.mean((values - mean) ** 2)),
        "upper_sampling": float(np.mean(upper ** 2)),
        "lower_sampling": float(np.mean(lower ** 2)),
        "interaction": float(np.mean(interaction ** 2)),
    }


def raw_target(values):
    bounded = np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.arctanh(np.clip(bounded, -1.0 + 1e-6, 1.0 - 1e-6))


def gradient_summary(gradient):
    mean = gradient.mean(axis=(0, 1))
    return {
        "mean_squared_norm": float(np.mean(np.sum(gradient ** 2, axis=-1))),
        "variance_trace": float(np.mean(np.sum((gradient - mean) ** 2, axis=-1))),
        "mean": mean.tolist(),
    }


def loss_gradients(mean, samples, bounded_targets):
    dimension = samples.shape[-1]
    targets = raw_target(bounded_targets)
    bounded_samples = np.tanh(samples)
    return {
        "raw_mean": 2.0 * (mean - targets) / dimension,
        "raw_sample": 2.0 * (samples - targets) / dimension,
        "action_sample": (
            2.0 * (bounded_samples - bounded_targets)
            * (1.0 - bounded_samples ** 2) / dimension
        ),
    }


def snapshot(projector, upper_mean, lower_mean, upper_noise, lower_noise):
    upper_samples = upper_mean + upper_noise
    lower_samples = lower_mean + lower_noise
    shape = (len(upper_samples), len(lower_samples), len(upper_mean))
    upper_targets = np.empty(shape)
    lower_targets = np.empty(shape)
    plugin_targets = np.empty((len(upper_samples), len(upper_mean)))
    before = projector.policy_context
    for i, upper in enumerate(upper_samples):
        bounded_upper = np.tanh(upper)
        plugin_targets[i] = projector.preview(
            bounded_upper, np.tanh(lower_mean)
        )["upper"]
        for j, lower in enumerate(lower_samples):
            row = projector.preview(bounded_upper, np.tanh(lower))
            upper_targets[i, j] = row["upper"]
            lower_targets[i, j] = row["lower"]
    after = projector.policy_context
    context_unchanged = (
        before[1] == after[1]
        and all(np.array_equal(a, b) for a, b in zip(before[0], after[0]))
    )
    if not context_unchanged:
        raise RuntimeError("same-state previews advanced projector history")
    levels = {}
    for level, mean, samples, targets in (
        ("upper", upper_mean, upper_samples[:, None, :], upper_targets),
        ("lower", lower_mean, lower_samples[None, :, :], lower_targets),
    ):
        samples = np.broadcast_to(samples, shape)
        gradients = loss_gradients(mean, samples, targets)
        identity = loss_gradients(mean, samples, np.tanh(samples))
        levels[level] = {
            "raw_target_variance": target_variance(raw_target(targets)),
            "action_target_variance": target_variance(targets),
            "gradients": {k: gradient_summary(v) for k, v in gradients.items()},
            "identity_projection_gradients": {
                k: gradient_summary(v) for k, v in identity.items()
            },
            "component_correction_mse": float(np.mean((np.tanh(samples) - targets) ** 2)),
        }
    plugin = {}
    for name, values, point in (
        ("raw", raw_target(upper_targets), raw_target(plugin_targets)),
        ("action", upper_targets, plugin_targets),
    ):
        plugin[name] = {
            "conditional_mean_displacement_mse": float(np.mean((point - values.mean(axis=1)) ** 2)),
            "reference_mean_sampling_variance": float(np.mean(values.var(axis=1, ddof=1)) / len(lower_samples)),
        }
    return {"levels": levels, "lower_mean_plugin": plugin, "context_unchanged": context_unchanged}


def run_cell(dimension, amplitude, std, seed):
    history_rng, sample_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    phase = history_rng.uniform(-np.pi, np.pi)
    angles = phase + 2.0 * np.pi * np.arange(dimension) / dimension
    upper_mean = amplitude * np.sin(angles)
    lower_mean = amplitude * np.cos(angles)
    upper_noise = std * sample_rng.normal(size=(UPPER_DRAWS, dimension))
    lower_noise = std * sample_rng.normal(size=(LOWER_DRAWS, dimension))
    projector = CausalTerminalReserveProjector()
    projector.reset(dimension)
    snapshots = []
    upper = np.zeros(dimension)
    started = time.monotonic()
    for step in range(max(PREFIX_STEPS) + 1):
        if step in PREFIX_STEPS:
            snapshots.append({"prefix_steps": step, **snapshot(
                projector, upper_mean, lower_mean, upper_noise, lower_noise
            )})
            print(f"prefix={step} elapsed={time.monotonic() - started:.2f}s", flush=True)
        if step < max(PREFIX_STEPS):
            if step % 16 == 0:
                upper = np.tanh(upper_mean + std * history_rng.normal(size=dimension))
            lower = np.tanh(lower_mean + std * history_rng.normal(size=dimension))
            projector.project(upper, lower)
    return {
        "protocol": PROTOCOL,
        "evidence_role": "controlled_same_state_mechanism_diagnostic",
        "dimension": dimension, "mean_amplitude": amplitude,
        "standard_deviation": std, "seed": seed,
        "upper_draws": UPPER_DRAWS, "lower_draws": LOWER_DRAWS,
        "elapsed_seconds": time.monotonic() - started,
        "snapshots": snapshots,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS, required=True)
    parser.add_argument("--amplitude", type=float, choices=MEAN_AMPLITUDES, required=True)
    parser.add_argument("--std", type=float, choices=STANDARD_DEVIATIONS, required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_cell(args.dimension, args.amplitude, args.std, args.seed)
    result["code_revision"] = args.code_revision
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("same-state diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
