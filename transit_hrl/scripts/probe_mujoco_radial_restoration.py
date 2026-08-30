#!/usr/bin/env python3
"""Probe deployment-aligned radial actor restoration on frozen MuJoCo paths."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from freq_hrl.experiments.mujoco.control_validation import (
    _model_parameter_sha256,
    paired_closed_loop_guard_snapshot,
    rollout_hierarchical,
    summarize,
)
from freq_hrl.rl.smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO,
    SMDPPPOConfig,
)


PROBE_VERSION = "mujoco_radial_restoration_probe_v1"


def _parse_gains(value: str) -> tuple[float, ...]:
    gains = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not gains or any(
        not np.isfinite(gain) or not 0.0 < gain <= 1.0 for gain in gains
    ):
        raise ValueError("radial gains must be finite and in (0, 1]")
    if len(set(gains)) != len(gains):
        raise ValueError("radial gains must be unique")
    return gains


def _parse_router_strengths(value: str) -> tuple[float, ...]:
    strengths = tuple(
        float(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not strengths or any(
        not np.isfinite(strength) or not 0.0 <= strength <= 1.0
        for strength in strengths
    ):
        raise ValueError("router strengths must be finite and in [0, 1]")
    if len(set(strengths)) != len(strengths):
        raise ValueError("router strengths must be unique")
    return strengths


def _actor_output_head(actor: nn.Module) -> nn.Linear:
    network = getattr(actor, "net", None)
    if network is None or len(network) < 1 or not isinstance(network[-1], nn.Linear):
        raise TypeError("radial restoration requires an MLP Gaussian actor")
    return network[-1]


def scale_actor_output_head(actor: nn.Module, gain: float) -> None:
    value = float(gain)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("radial gain must be finite and in (0, 1]")
    head = _actor_output_head(actor)
    with torch.no_grad():
        head.weight.mul_(value)
        if head.bias is not None:
            head.bias.mul_(value)


def _load_model(checkpoint: dict[str, Any]) -> FrequencySeparatedActorCriticPPO:
    state = copy.deepcopy(checkpoint["model_state_dict"])
    config = dict(state["config"])
    config["device"] = "cpu"
    model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**config))
    model.load_state_dict(state)
    return model


def _guard_paths(summary: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = dict(summary.get("deployment_frequency_closed_loop_guard_baseline") or {})
    paths = [dict(item) for item in baseline.get("paths", [])]
    if (
        not paths
        or len(paths) != int(baseline.get("row_count", -1))
        or len({(str(item["disturbance_mode"]), int(item["seed"])) for item in paths})
        != len(paths)
    ):
        raise ValueError("summary does not contain a unique closed-loop guard registry")
    return paths


def _evaluate_rows(
    model: FrequencySeparatedActorCriticPPO,
    *,
    summary: dict[str, Any],
    paths: Iterable[dict[str, Any]],
    episode_horizon: int,
    leakage_cost_mode: str,
    router_strength: float,
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(rollout_hierarchical(
            model,
            seed=int(path["seed"]),
            env_id=str(summary["environment"]),
            disturbance_mode=str(path["disturbance_mode"]),
            steps=int(summary["steps"]),
            upper_period=int(summary["upper_period"]),
            frequency_routing=True,
            leakage_constraint=True,
            lower_lf_rms_budget=float(summary["lower_lf_rms_budget"]),
            leakage_constraint_scope=str(summary["leakage_constraint_scope"]),
            upper_hf_rms_budget=float(summary["upper_hf_rms_budget"]),
            upper_hf_penalty_coef=float(summary["upper_hf_penalty_coef"]),
            upper_constraint_mode=str(summary["upper_constraint_mode"]),
            lower_lf_alpha=0.04,
            upper_action_scale=float(summary["upper_action_scale"]),
            lower_action_scale=float(summary["lower_action_scale"]),
            responsibility_mode=str(summary["responsibility_mode"]),
            leakage_cost_mode=str(leakage_cost_mode),
            lower_action_router_mode=str(summary["lower_action_router_mode"]),
            lower_action_router_alpha=float(summary["lower_action_router_alpha"]),
            lower_action_router_strength=float(router_strength),
            lower_action_router_observe_strength=bool(
                summary["lower_action_router_observe_strength"]
            ),
            sample=False,
            collect_trajectory=False,
            method="freq_hrl",
            episode_horizon=int(episode_horizon),
        )[1])
    return rows


def run_probe(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    output_path: Path,
    gains: tuple[float, ...],
    router_strengths: tuple[float, ...],
    episode_horizon: int,
    leakage_cost_mode: str,
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    paths = _guard_paths(summary)
    baseline_model = _load_model(checkpoint)
    baseline_parameter_sha256 = _model_parameter_sha256(baseline_model)
    registered_baseline = str(
        summary["deployment_frequency_closed_loop_guard_baseline"][
            "parameter_sha256"
        ]
    )
    if baseline_parameter_sha256 != registered_baseline:
        raise RuntimeError("checkpoint does not reconstruct the registered guard baseline")
    baseline_rows = _evaluate_rows(
        baseline_model,
        summary=summary,
        paths=paths,
        episode_horizon=episode_horizon,
        leakage_cost_mode=leakage_cost_mode,
        router_strength=float(summary["lower_action_router_strength"]),
    )
    modes = tuple(dict.fromkeys(str(item["disturbance_mode"]) for item in paths))

    def snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return paired_closed_loop_guard_snapshot(
            rows,
            baseline_rows=baseline_rows,
            expected_modes=modes,
            lower_reduction_fraction=float(
                summary["lower_deployment_frequency_reference_reduction_fraction"]
            ),
            upper_reduction_fraction=float(
                summary["upper_deployment_frequency_reference_reduction_fraction"]
            ),
            lower_power_floor=float(
                summary["lower_deployment_frequency_rms_budget"]
            ) ** 2,
            upper_power_floor=float(
                summary["upper_deployment_frequency_rms_budget"]
            ) ** 2,
            risk_mode=str(summary["deployment_frequency_closed_loop_risk_mode"]),
            cvar_alpha=float(summary["deployment_frequency_closed_loop_cvar_alpha"]),
        )

    candidates = []
    registered_strength = float(summary["lower_action_router_strength"])
    for router_strength in router_strengths:
        for upper_gain in gains:
            for lower_gain in gains:
                model = _load_model(checkpoint)
                scale_actor_output_head(model.upper_actor, upper_gain)
                scale_actor_output_head(model.lower_actor, lower_gain)
                rows = (
                    baseline_rows
                    if (
                        upper_gain == 1.0
                        and lower_gain == 1.0
                        and router_strength == registered_strength
                    )
                    else _evaluate_rows(
                        model,
                        summary=summary,
                        paths=paths,
                        episode_horizon=episode_horizon,
                        leakage_cost_mode=leakage_cost_mode,
                        router_strength=router_strength,
                    )
                )
                guard = snapshot(rows)
                candidates.append({
                    "upper_gain": float(upper_gain),
                    "lower_gain": float(lower_gain),
                    "router_strength": float(router_strength),
                    "parameter_sha256": _model_parameter_sha256(model),
                    "rank": list(guard["rank"]),
                    "reward_violation_count": int(guard["reward_violation_count"]),
                    "frequency_violation_count": int(
                        guard["frequency_violation_count"]
                    ),
                    "frequency_violation_merit": float(
                        guard["frequency_violation_merit"]
                    ),
                    "worst_frequency_violation": float(
                        guard["worst_frequency_violation"]
                    ),
                    "worst_constraint": dict(guard["worst_constraint"]),
                    "summary": summarize(rows),
                })
    candidates.sort(key=lambda item: tuple(item["rank"]), reverse=True)
    feasible = [
        item for item in candidates
        if item["reward_violation_count"] == 0
        and item["frequency_violation_count"] == 0
    ]
    payload = {
        "probe_version": PROBE_VERSION,
        "checkpoint": str(Path(checkpoint_path)),
        "summary": str(Path(summary_path)),
        "checkpoint_code_revision": str(checkpoint.get("code_revision", "")),
        "checkpoint_source_manifest_sha256": str(
            checkpoint.get("source_manifest_sha256", "")
        ),
        "baseline_parameter_sha256": baseline_parameter_sha256,
        "guard_path_count": len(paths),
        "gains": list(gains),
        "router_strengths": list(router_strengths),
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "best_candidate": candidates[0],
        "best_feasible_candidate": feasible[0] if feasible else None,
        "candidates": candidates,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", default="1.0,0.99,0.98,0.97,0.95")
    parser.add_argument("--router-strengths", default="0.5")
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--leakage-cost-mode", default="power_excess")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_probe(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        output_path=args.output,
        gains=_parse_gains(args.gains),
        router_strengths=_parse_router_strengths(args.router_strengths),
        episode_horizon=args.episode_horizon,
        leakage_cost_mode=args.leakage_cost_mode,
    )
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": payload["candidate_count"],
        "feasible_candidate_count": payload["feasible_candidate_count"],
        "best_candidate": payload["best_candidate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
