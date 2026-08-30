#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.23 frequency-horizon critic preflight."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as spec
from scripts.analyze_mujoco_v14_23_frequency_horizon_critic_preflight import analyze_run
from scripts import submit_mujoco_v14_18_router_probe_scheduleurm as base


LAUNCHER_PATH = Path(__file__).resolve()
SIGNATURE_VERSION = "mujoco-v14-23-frequency-horizon-critic-preflight-v1"


def _build_probe_command(args, environment: str, seed: int) -> str:
    anchor = base.anchor_relative_dir(args.anchor_run_name, environment, seed)
    output = base.cell_relative_dir(args.run_name, environment, seed) / "probe.json"
    command = [
        str(args.python_executable),
        "scripts/probe_mujoco_action_cost_critic_restoration.py",
        "--checkpoint", str(anchor / "checkpoint.pt"),
        "--summary", str(anchor / "cell_summary.json"),
        "--output", str(output),
        "--critic-train-roots", ",".join(map(str, spec.CRITIC_TRAIN_ROOTS)),
        "--critic-holdout-roots", ",".join(map(str, spec.CRITIC_HOLDOUT_ROOTS)),
        "--design-roots", ",".join(map(str, spec.DESIGN_ROOTS)),
        "--validation-roots", ",".join(map(str, spec.VALIDATION_ROOTS)),
        "--critic-seeds", ",".join(map(str, spec.CRITIC_ENSEMBLE_SEEDS)),
        "--critic-hidden-dim", str(spec.CRITIC_HIDDEN_DIM),
        "--critic-epochs", str(spec.CRITIC_EPOCHS),
        "--critic-minibatch-size", str(spec.CRITIC_MINIBATCH_SIZE),
        "--critic-learning-rate", str(spec.CRITIC_LEARNING_RATE),
        "--critic-minimum-holdout-r2", str(spec.CRITIC_MINIMUM_HOLDOUT_R2),
        "--critic-minimum-action-permutation-mse-increase",
        str(spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE),
        "--minimum-gradient-median-cosine",
        str(spec.MINIMUM_GRADIENT_MEDIAN_COSINE),
        "--upper-cost-return-horizon-decisions",
        str(spec.UPPER_COST_RETURN_HORIZON_DECISIONS),
        "--lower-cost-return-horizon-decisions",
        str(spec.LOWER_COST_RETURN_HORIZON_DECISIONS),
        "--actor-state-limit", str(spec.ACTOR_STATE_LIMIT_PER_LEVEL),
        "--actor-step-rms-values", ",".join(map(str, spec.ACTOR_STEP_RMS_VALUES)),
        "--minimum-reduction", str(spec.MINIMUM_REDUCTION),
        "--funnel-multiplier", str(spec.FUNNEL_MULTIPLIER),
        "--workers", str(spec.WORKERS),
        "--risk-mode", spec.RISK_MODE,
        "--cvar-alpha", str(spec.CVAR_ALPHA),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--leakage-cost-mode", spec.LEAKAGE_COST_MODE,
        "--probe-version", spec.PROBE_VERSION,
    ]
    environment_variables = [
        "MUJOCO_GL=egl", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=",
    ]
    return " ".join([*environment_variables, shlex.join(command)]) + " && echo DONE"


def _write_preregistration(args) -> None:
    target = base.ROOT / "results" / args.run_name
    target.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=base.ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    payload = {
        "status": "frozen_before_v14_23_preflight_outcome_access",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "source_revision": revision,
        "anchor_run_name": args.anchor_run_name,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "critic_train_roots": list(spec.CRITIC_TRAIN_ROOTS),
        "critic_holdout_roots": list(spec.CRITIC_HOLDOUT_ROOTS),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "critic_ensemble_seeds": list(spec.CRITIC_ENSEMBLE_SEEDS),
        "critic_hidden_dim": spec.CRITIC_HIDDEN_DIM,
        "critic_epochs": spec.CRITIC_EPOCHS,
        "critic_minibatch_size": spec.CRITIC_MINIBATCH_SIZE,
        "critic_learning_rate": spec.CRITIC_LEARNING_RATE,
        "upper_cost_return_horizon_decisions": (
            spec.UPPER_COST_RETURN_HORIZON_DECISIONS
        ),
        "lower_cost_return_horizon_decisions": (
            spec.LOWER_COST_RETURN_HORIZON_DECISIONS
        ),
        "critic_minimum_holdout_r2": spec.CRITIC_MINIMUM_HOLDOUT_R2,
        "critic_minimum_action_permutation_mse_increase": (
            spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
        ),
        "minimum_gradient_median_cosine": spec.MINIMUM_GRADIENT_MEDIAN_COSINE,
        "actor_state_limit_per_level": spec.ACTOR_STATE_LIMIT_PER_LEVEL,
        "actor_step_rms_values": list(spec.ACTOR_STEP_RMS_VALUES),
        "risk_mode": spec.RISK_MODE,
        "cvar_alpha": spec.CVAR_ALPHA,
        "selection_contract": spec.SELECTION_CONTRACT,
        "scheduler_contract": {
            "scheduler": "scheduleurm", "allowed_nodes": list(args.nodes),
            "require_node": None, "cpu_per_task": spec.CPU_PER_TASK,
            "ram_mb_per_task": spec.RAM_MB_PER_TASK, "slurm_used": False,
        },
    }
    (target / "preregistration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _overrides() -> dict[str, Any]:
    return {
        "spec": spec,
        "analyze_run": analyze_run,
        "build_probe_command": _build_probe_command,
        "_write_preregistration": _write_preregistration,
        "SIGNATURE_VERSION": SIGNATURE_VERSION,
        "LAUNCHER_PATH": LAUNCHER_PATH,
    }


@contextmanager
def configured_base() -> Iterator[Any]:
    overrides = _overrides()
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield base
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def build_parser():
    with configured_base() as launcher:
        return launcher.build_parser()


def normalize_args(args):
    with configured_base() as launcher:
        return launcher.normalize_args(args)


def selected_cells():
    with configured_base() as launcher:
        return launcher.selected_cells()


def build_probe_command(args, environment, seed):
    return _build_probe_command(args, environment, seed)


def build_scheduler_spec(args, environment, seed):
    with configured_base() as launcher:
        return launcher.build_scheduler_spec(args, environment, seed)


def main() -> None:
    for name, value in _overrides().items():
        setattr(base, name, value)
    base.main()


if __name__ == "__main__":
    main()
