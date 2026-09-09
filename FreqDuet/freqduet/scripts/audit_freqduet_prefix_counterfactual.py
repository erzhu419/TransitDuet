#!/usr/bin/env python3
"""Generate exact-prefix, single-decision counterfactual labels.

Unlike the legacy environment-deepcopy audit, every branch reconstructs the
entire controller and simulator state by replaying from the same frozen
checkpoint and scenario seed. Labels are written only after an unchanged
duplicate and every candidate pass the causal-prefix equality contract.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_freqduet_snapshot_counterfactual import (
    build_context_row,
    parse_csv,
    resolve_config,
    set_worker_threads,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


PROTOCOL_VERSION = "freqduet-v28-exact-prefix-counterfactual-v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_ROW_FIELDS = {"wall_env_s", "wall_train_s"}
OUTCOME_FIELDS = (
    "service_cost",
    "service_cost_observed",
    "service_cost_restricted",
    "restricted_total_journey_horizon_min",
    "avg_total_journey_observed_min",
    "headway_cv",
    "peak_fleet",
    "fleet_overshoot",
    "holding_vehicle_seconds",
    "holding_passenger_seconds",
    "trip_completion_rate",
    "passenger_unserved_rate",
)


def import_runner():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runner_v3 import TransitDuetV2Runner, load_config

    return TransitDuetV2Runner, load_config


def reset_global_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod.manual_seed(int(seed))


def capture_rng_state(runner) -> dict[str, Any]:
    torch_mod = sys.modules.get("torch")
    global_state: dict[str, Any] = {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
    }
    if torch_mod is not None:
        global_state["torch"] = (
            torch_mod.random.get_rng_state().detach().cpu().numpy().copy()
        )
    runtime = {
        str(name): copy.deepcopy(rng.get_state())
        for name, rng in runner._runtime_numpy_streams().items()
    }
    return {"global": global_state, "runtime": runtime}


def first_difference(left: Any, right: Any, path: str = "root") -> str | None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            lhs = np.asarray(left)
            rhs = np.asarray(right)
        except Exception:
            return f"{path}: array conversion failed"
        if lhs.shape != rhs.shape:
            return f"{path}: shape {lhs.shape} != {rhs.shape}"
        if not np.array_equal(lhs, rhs, equal_nan=True):
            try:
                equal = np.equal(lhs, rhs)
                if np.issubdtype(lhs.dtype, np.floating):
                    equal = equal | (np.isnan(lhs) & np.isnan(rhs))
                mismatch = np.argwhere(~equal)
            except (TypeError, ValueError):
                mismatch = np.empty((0, lhs.ndim), dtype=np.int64)
            index = tuple(mismatch[0]) if mismatch.size else ()
            return f"{path}{index}: {lhs[index]!r} != {rhs[index]!r}"
        return None
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return f"{path}: mapping type mismatch"
        if set(left) != set(right):
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in sorted(left, key=str):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return f"{path}: sequence type mismatch"
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            difference = first_difference(lhs, rhs, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        lhs = float(left)
        rhs = float(right)
        if np.isnan(lhs) and np.isnan(rhs):
            return None
        return None if lhs == rhs else f"{path}: {lhs!r} != {rhs!r}"
    if isinstance(left, (int, np.integer, bool, np.bool_)) or isinstance(
            right, (int, np.integer, bool, np.bool_)):
        return None if left == right else f"{path}: {left!r} != {right!r}"
    return None if left == right else f"{path}: {left!r} != {right!r}"


def assert_equal(label: str, left: Any, right: Any) -> None:
    difference = first_difference(left, right)
    if difference is not None:
        raise RuntimeError(f"{label} mismatch: {difference}")


def non_runtime_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in RUNTIME_ROW_FIELDS
    }


def action_token(offset_s: float) -> str:
    value = int(round(float(offset_s)))
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "0"


def candidate_name(offset_s: float) -> str:
    return f"actor_firstknot_{action_token(offset_s)}"


def perturb_direction_first_knot(
    action: np.ndarray,
    offset_s: float,
    direction: bool,
    planner,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, int]:
    candidate = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if planner is None:
        raise RuntimeError("V28 prefix intervention requires a timetable planner")
    if str(getattr(planner, "coefficient_parameterization", "full")) == (
            "antisymmetric_linear_v5"):
        index = 0 if bool(getattr(planner, "shared_directions", False)) else (
            0 if bool(direction) else 1)
    else:
        basis = int(getattr(planner, "basis_per_direction", 0))
        if basis <= 0:
            raise RuntimeError("planner has no positive basis_per_direction")
        index = 0 if bool(getattr(planner, "shared_directions", False)) else (
            0 if bool(direction) else basis)
    if not 0 <= index < candidate.size:
        raise RuntimeError(
            f"first-knot index {index} outside action size {candidate.size}")
    candidate[index] = np.clip(
        float(candidate[index]) + float(offset_s),
        float(action_low[index]),
        float(action_high[index]),
    )
    return candidate.astype(np.float32), int(index)


@dataclass
class BranchResult:
    label: str
    candidate_offset_s: float | None
    target_identity: dict[str, Any]
    context: dict[str, Any]
    actor_action: np.ndarray
    executed_action: np.ndarray
    changed_index: int | None
    terminal_dispatch: bool
    upper_prefix: list[dict[str, Any]]
    lower_prefix: list[dict[str, Any]]
    rng_state: dict[str, Any]
    policy_digest: str
    episode_row: dict[str, Any]


def run_branch(
    *,
    cfg_path: Path,
    train_seed: int,
    checkpoint_dir: Path,
    checkpoint_ep: int,
    eval_episode: int,
    scenario_seed: int,
    replay_seed: int,
    decision_index: int,
    candidate_offset_s: float | None,
    branch_label: str,
    branch_log_root: Path,
) -> BranchResult:
    TransitDuetV2Runner, load_config = import_runner()
    reset_global_rng(replay_seed)
    cfg = load_config(str(cfg_path))
    cfg["seed"] = int(train_seed)
    cfg.setdefault("logging", {})["logs_dir"] = str(branch_log_root / branch_label)
    runner = TransitDuetV2Runner(cfg, device="cpu")
    loaded_ep = runner.load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        ep=int(checkpoint_ep),
        require_deployment_state=True,
    )
    if int(loaded_ep) != int(checkpoint_ep):
        raise RuntimeError(f"loaded checkpoint {loaded_ep} != {checkpoint_ep}")
    policy_digest = str(runner._policy_digest())

    target_seen = False
    upper_prefix: list[dict[str, Any]] = []
    lower_prefix: list[dict[str, Any]] = []
    target_identity: dict[str, Any] | None = None
    target_context: dict[str, Any] | None = None
    actor_action: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    target_rng: dict[str, Any] | None = None
    changed_index: int | None = None
    decision_count = 0

    original_lower = runner._lower_action_for_agent

    def traced_lower(obs, key, last_action=0.0, deterministic=False):
        action = original_lower(
            obs,
            key,
            last_action=last_action,
            deterministic=deterministic,
        )
        if not target_seen:
            lower_prefix.append({
                "time_s": float(getattr(runner.env, "current_time", 0.0)),
                "agent": int(key),
                "state": np.asarray(obs, dtype=np.float32).reshape(-1).copy(),
                "previous_action_s": float(last_action),
                "action": np.asarray(action, dtype=np.float32).reshape(-1).copy(),
                "deterministic": bool(deterministic),
            })
        return action

    runner._lower_action_for_agent = traced_lower

    def intervention(*, action_vec, write_terminal_dispatch, s_upper, trip,
                     decision_time_s):
        nonlocal target_seen, target_identity, target_context
        nonlocal actor_action, executed_action, target_rng, changed_index
        nonlocal decision_count
        if target_seen:
            return None
        decision_count += 1
        action = np.asarray(action_vec, dtype=np.float32).reshape(-1).copy()
        event = {
            "decision_index": int(decision_count),
            "time_s": float(decision_time_s),
            "trip_id": int(getattr(trip, "launch_turn", -1)),
            "direction": bool(getattr(trip, "direction", True)),
            "scheduled_launch_s": float(getattr(trip, "launch_time", 0.0)),
            "s_upper": np.asarray(s_upper, dtype=np.float32).reshape(-1).copy(),
            "actor_action": action.copy(),
            "write_terminal_dispatch": bool(write_terminal_dispatch),
        }
        upper_prefix.append(event)
        if decision_count != int(decision_index):
            return None

        context = build_context_row(
            int(eval_episode), int(decision_count), runner.env, trip)
        state = np.asarray(s_upper, dtype=np.float32).reshape(-1)
        for index, value in enumerate(state):
            context[f"upper_state_{index:03d}"] = float(value)
        context["upper_state_dim"] = int(state.size)
        context["train_seed"] = int(train_seed)
        context["eval_episode"] = int(eval_episode)
        context["scenario_seed"] = int(scenario_seed)

        target_seen = True
        target_identity = {
            key: copy.deepcopy(event[key])
            for key in (
                "decision_index", "time_s", "trip_id", "direction",
                "scheduled_launch_s", "write_terminal_dispatch",
            )
        }
        target_context = context
        actor_action = action.copy()
        target_rng = capture_rng_state(runner)
        if candidate_offset_s is None:
            executed_action = action.copy()
            return None
        candidate, index = perturb_direction_first_knot(
            action=action,
            offset_s=float(candidate_offset_s),
            direction=bool(getattr(trip, "direction", True)),
            planner=runner.timetable_planner,
            action_low=runner.upper_action_low,
            action_high=runner.upper_action_high,
        )
        executed_action = candidate.copy()
        changed_index = int(index)
        return {"action_vec": candidate}

    runner._offline_upper_action_intervention = intervention
    episode_row = runner.run_episode(
        ep=int(eval_episode),
        training=False,
        scenario_seed=int(scenario_seed),
        record_diagnostics=False,
    )
    if not target_seen:
        raise RuntimeError(
            f"episode reached only {decision_count} upper decisions; "
            f"target was {decision_index}")
    if any(value is None for value in (
            target_identity, target_context, actor_action, executed_action,
            target_rng)):
        raise RuntimeError("target capture is incomplete")
    result = BranchResult(
        label=str(branch_label),
        candidate_offset_s=(
            None if candidate_offset_s is None else float(candidate_offset_s)),
        target_identity=target_identity,
        context=target_context,
        actor_action=actor_action,
        executed_action=executed_action,
        changed_index=changed_index,
        terminal_dispatch=bool(target_identity["write_terminal_dispatch"]),
        upper_prefix=upper_prefix,
        lower_prefix=lower_prefix,
        rng_state=target_rng,
        policy_digest=policy_digest,
        episode_row=episode_row,
    )
    runner._offline_upper_action_intervention = None
    runner._lower_action_for_agent = original_lower
    del runner
    gc.collect()
    return result


def assert_prefix_equal(reference: BranchResult, candidate: BranchResult) -> None:
    assert_equal("policy checkpoint", reference.policy_digest, candidate.policy_digest)
    assert_equal("target identity", reference.target_identity, candidate.target_identity)
    assert_equal("causal upper context", reference.context, candidate.context)
    assert_equal("pre-intervention actor action", reference.actor_action, candidate.actor_action)
    assert_equal("upper decision prefix", reference.upper_prefix, candidate.upper_prefix)
    assert_equal("lower action prefix", reference.lower_prefix, candidate.lower_prefix)
    assert_equal("random streams", reference.rng_state, candidate.rng_state)


def scalar_or_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return json.dumps(value, sort_keys=True, default=str)


def label_row(branch: BranchResult, reference: BranchResult) -> dict[str, Any]:
    row = dict(branch.context)
    row.update({
        "candidate_method": (
            "actor" if branch.candidate_offset_s is None
            else candidate_name(branch.candidate_offset_s)),
        "candidate_offset_s": (
            0.0 if branch.candidate_offset_s is None
            else float(branch.candidate_offset_s)),
        "candidate_changed_index": (
            -1 if branch.changed_index is None else int(branch.changed_index)),
        "candidate_terminal_dispatch": int(branch.terminal_dispatch),
        "actor_action_json": json.dumps(branch.actor_action.tolist()),
        "candidate_action_json": json.dumps(branch.executed_action.tolist()),
        "candidate_action_linf_delta_s": float(np.max(np.abs(
            branch.executed_action.astype(np.float64)
            - branch.actor_action.astype(np.float64)))),
    })
    for key, value in branch.episode_row.items():
        row[f"episode_{key}"] = scalar_or_json(value)
    for field in OUTCOME_FIELDS:
        if field not in branch.episode_row or field not in reference.episode_row:
            continue
        try:
            row[f"episode_{field}_delta_vs_actor"] = (
                float(branch.episode_row[field])
                - float(reference.episode_row[field]))
        except (TypeError, ValueError):
            continue
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_source(expected_commit: str | None = None) -> dict[str, Any]:
    provenance = git_provenance()
    commit = str(provenance.get("commit", "")).strip().lower()
    if not COMMIT_RE.fullmatch(commit):
        raise RuntimeError(
            f"prefix audit requires an identified full source commit: {commit!r}")
    if provenance.get("tracked_dirty") is not False:
        raise RuntimeError("prefix audit requires a clean tracked source snapshot")
    if expected_commit and commit != str(expected_commit).strip().lower():
        raise RuntimeError(
            f"source commit {commit} != expected {expected_commit}")
    return provenance


def run_audit(args) -> tuple[Path, dict[str, Any]]:
    set_worker_threads(args.worker_threads)
    source = validate_source(args.expected_source_commit)
    protocol_version = str(args.protocol_version).strip()
    if not protocol_version:
        raise RuntimeError("prefix audit requires a nonempty protocol version")
    cfg_path = resolve_config(args.config).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    offsets = parse_csv(args.offsets_s, float)
    if 0.0 not in offsets:
        raise RuntimeError("prefix audit requires a zero-offset identity candidate")
    if int(args.decision_index) < 1:
        raise RuntimeError("decision-index is one-based and must be positive")

    out_dir = Path(args.out_dir).resolve()
    for filename in (
            "prefix_counterfactual_labels.csv",
            "prefix_counterfactual_meta.json"):
        if (out_dir / filename).exists():
            raise FileExistsError(
                f"refusing to mix prefix evidence in existing output: {out_dir}")
    branch_log_root = out_dir / "branch_logs"
    started = time.time()
    common = {
        "cfg_path": cfg_path,
        "train_seed": int(args.train_seed),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_ep": int(args.checkpoint_ep),
        "eval_episode": int(args.eval_episode),
        "scenario_seed": int(args.scenario_seed),
        "replay_seed": int(args.replay_seed),
        "decision_index": int(args.decision_index),
        "branch_log_root": branch_log_root,
    }
    reference = run_branch(
        **common, candidate_offset_s=None, branch_label="actor_reference")
    repeat = run_branch(
        **common, candidate_offset_s=None, branch_label="actor_repeat")
    assert_prefix_equal(reference, repeat)
    assert_equal(
        "unchanged actor episode",
        non_runtime_episode_row(reference.episode_row),
        non_runtime_episode_row(repeat.episode_row),
    )

    candidates = []
    zero_branch = None
    for offset in offsets:
        branch = run_branch(
            **common,
            candidate_offset_s=float(offset),
            branch_label=candidate_name(float(offset)),
        )
        assert_prefix_equal(reference, branch)
        candidates.append(branch)
        if float(offset) == 0.0:
            zero_branch = branch
    if zero_branch is None:
        raise RuntimeError("zero-offset identity branch was not executed")
    assert_equal(
        "zero-offset episode",
        non_runtime_episode_row(reference.episode_row),
        non_runtime_episode_row(zero_branch.episode_row),
    )
    nonzero_response = any(
        np.max(np.abs(
            branch.executed_action.astype(np.float64)
            - branch.actor_action.astype(np.float64))) > 0.0
        for branch in candidates
        if float(branch.candidate_offset_s or 0.0) != 0.0
    )
    if not nonzero_response:
        raise RuntimeError("all nonzero interventions were clipped to the actor action")

    rows = [label_row(reference, reference)] + [
        label_row(branch, reference) for branch in candidates
    ]
    csv_path = out_dir / "prefix_counterfactual_labels.csv"
    write_csv(csv_path, rows)
    meta = {
        "protocol_version": protocol_version,
        "status": "mechanical_pass",
        "effect_evidence": False,
        "source": source,
        "config": str(cfg_path),
        "train_seed": int(args.train_seed),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_ep": int(args.checkpoint_ep),
        "policy_digest": reference.policy_digest,
        "eval_episode": int(args.eval_episode),
        "scenario_seed": int(args.scenario_seed),
        "replay_seed": int(args.replay_seed),
        "decision_index": int(args.decision_index),
        "target_identity": reference.target_identity,
        "offsets_s": [float(value) for value in offsets],
        "candidate_parameterization": "same_direction_first_bernstein_knot_v1",
        "terminal_dispatch_preserved": bool(reference.terminal_dispatch),
        "checks": {
            "actor_repeat_prefix_exact": True,
            "actor_repeat_episode_exact": True,
            "candidate_prefixes_exact": True,
            "zero_offset_episode_exact": True,
            "policy_checkpoint_exact": True,
            "global_and_isolated_rng_exact": True,
            "nonzero_action_response": True,
        },
        "rows": len(rows),
        "labels": str(csv_path),
        "branch_logs_retained": False,
        "elapsed_s": float(time.time() - started),
    }
    if branch_log_root.exists():
        shutil.rmtree(branch_log_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "prefix_counterfactual_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return csv_path, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-ep", type=int, required=True)
    parser.add_argument("--eval-episode", type=int, default=100000)
    parser.add_argument("--scenario-seed", type=int, required=True)
    parser.add_argument("--replay-seed", type=int, default=28001)
    parser.add_argument("--decision-index", type=int, required=True)
    parser.add_argument("--offsets-s", default="-20,0,20")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    try:
        csv_path, meta = run_audit(args)
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        invalid = {
            "protocol_version": str(args.protocol_version).strip(),
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
            "labels_written": False,
        }
        (out_dir / "prefix_counterfactual_invalid.json").write_text(
            json.dumps(invalid, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE prefix protocol={meta['protocol_version']} "
        f"gate={meta['status']} rows={meta['rows']} "
        f"labels={csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
