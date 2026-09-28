"""Keep lower on-policy while aligning frozen levels with deployed execution."""

import numpy as np
import torch

from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_calibration as calibration
from scripts import pointmaze_frozen_execution_stage40_spec as spec


def worker_rollout(job, *, specification=spec, lower_value_context_builder=None):
    spec = specification
    weights, seed, phase, mode, path, method = job
    model, args, native, credit = joint._WORKER
    model.load_state_dict(weights)
    kwargs = spec.rollout_arguments(args.optimizer_seed, method, seed, phase=phase, mode=mode)
    policy_seed = int(seed) + args.optimizer_seed if kwargs["sample"] else spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    batch, row, raw = joint.rollout(model, args, native, seed=seed, capture=path is not None,
                                  lower_credit=credit, lower_value_context_builder=lower_value_context_builder, **kwargs)
    row.update(policy_seed=policy_seed, deployment_mode=mode)
    if path is not None:
        np.savez_compressed(path, **raw)
    return batch, row


def audit_result(result, *, raw_path, specification=spec):
    spec = specification
    audited = calibration.audit_result(result, raw_path=raw_path, specification=spec)
    root, method, preflight = result["root"], result["method"], result["preflight"]
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    for row in result["training"]:
        offset = (row["iteration"] - 1) * opt["rollouts_per_iteration"]
        seeds = roles["training"][offset:offset + opt["rollouts_per_iteration"]]
        mode = "warmup" if row["iteration"] <= opt["warmup_iterations"] else "learning"
        expected = []
        for seed in seeds:
            kwargs = spec.rollout_arguments(root, method, seed, phase="train", mode=mode)
            expected.append({"seed": seed, "policy_seed": seed + root,
                             **{k: v for k, v in kwargs.items() if k != "sample"}})
        if row["rollout_sampling"] != expected:
            raise ValueError("frozen execution training sampling or paired lower stream changed")
    for stage in result["snapshots"].values():
        for mode, rows in stage["evaluation_rows"].items():
            for row in rows:
                kwargs = spec.rollout_arguments(root, method, row["seed"], phase="eval", mode=mode)
                if any(row[k] != v for k, v in kwargs.items() if k != "sample"):
                    raise ValueError("frozen execution deployment sampling changed")
    return audited
