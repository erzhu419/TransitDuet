#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.21 distributional actor preflight."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_21_distributional_actor_preflight_spec as spec
from scripts.analyze_mujoco_v14_21_distributional_actor_preflight import analyze_run
from scripts import submit_mujoco_v14_18_router_probe_scheduleurm as base


LAUNCHER_PATH = Path(__file__).resolve()
SIGNATURE_VERSION = "mujoco-v14-21-distributional-actor-preflight-v1"


def _build_probe_command(args, environment: str, seed: int) -> str:
    anchor = base.anchor_relative_dir(args.anchor_run_name, environment, seed)
    output = base.cell_relative_dir(args.run_name, environment, seed) / "probe.json"
    command = [
        str(args.python_executable),
        "scripts/probe_mujoco_zeroth_order_actor_restoration.py",
        "--checkpoint", str(anchor / "checkpoint.pt"),
        "--summary", str(anchor / "cell_summary.json"),
        "--output", str(output),
        "--direction-count", str(spec.DIRECTION_COUNT),
        "--direction-seed", str(spec.DIRECTION_SEED),
        "--perturb-rms", str(spec.PERTURB_RMS),
        "--step-rms-values", ",".join(map(str, spec.STEP_RMS_VALUES)),
        "--design-roots", ",".join(map(str, spec.DESIGN_ROOTS)),
        "--validation-roots", ",".join(map(str, spec.VALIDATION_ROOTS)),
        "--minimum-reduction", str(spec.MINIMUM_REDUCTION),
        "--funnel-multiplier", str(spec.FUNNEL_MULTIPLIER),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--leakage-cost-mode", spec.LEAKAGE_COST_MODE,
        "--workers", str(spec.WORKERS),
        "--risk-mode", spec.RISK_MODE,
        "--cvar-alpha", str(spec.CVAR_ALPHA),
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
        "status": "frozen_before_v14_21_preflight_outcome_access",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "source_revision": revision,
        "anchor_run_name": args.anchor_run_name,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "direction_count": spec.DIRECTION_COUNT,
        "direction_seed": spec.DIRECTION_SEED,
        "perturb_rms": spec.PERTURB_RMS,
        "step_rms_values": list(spec.STEP_RMS_VALUES),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
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
