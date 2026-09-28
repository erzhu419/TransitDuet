"""Give only the lower reward critic causal control-clock information."""

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO
from . import pointmaze_joint_renewal as joint
from . import pointmaze_frozen_execution as execution
from .pointmaze_warmup_alignment import audit_warmup_pair
from scripts import pointmaze_critic_clock_stage42_spec as spec


def make_model(controller, method, *, root):
    original = joint.make_model(controller, method, root=root)
    if original.config.state_encoder != "mlp":
        raise ValueError("Stage-42 inherits the registered MLP controller")
    model = FrequencySeparatedActorCriticPPO(replace(
        original.config, lower_value_state_dim=original.config.lower_state_dim + 2))
    # Construct the original gate first: expanding a layer changes RNG consumption.
    for name, weights in joint.inference_weights(original).items():
        if name == "lower_value":
            weight = weights["net.0.weight"]
            weights["net.0.weight"] = torch.cat((weight, weight.new_zeros((weight.shape[0], 2))), dim=1)
        getattr(model, name).load_state_dict(weights)
    return model


def time_context(*, age, step, horizon, clock):
    return np.asarray([age / joint.spec.MAX_AGE_STEPS, (horizon - step) / horizon]
                      if clock else [0., 0.], dtype=np.float32)


def context_builder(method):
    return partial(time_context, clock=spec.VALUE_CLOCK[method])


def worker_rollout(job):
    return execution.worker_rollout(job, specification=spec,
                                   lower_value_context_builder=context_builder(job[-1]))


def audit_context(batch, row, context, *, clock):
    horizon, decisions = row["episode_length"], row["decision_steps"]
    step = np.arange(horizon)
    age = step - np.repeat(decisions, np.diff([*decisions, horizon]))
    expected = np.column_stack((age / joint.spec.MAX_AGE_STEPS, (horizon - step) / horizon)).astype(np.float32)
    if not clock:
        expected[:] = 0.
    np.testing.assert_array_equal(context, expected, err_msg="lower critic clock differs from causal current-option age")
    if batch is not None:
        np.testing.assert_array_equal(batch.value_state[:, :-2], batch.state, err_msg="critic context changed actor state")
        np.testing.assert_array_equal(batch.value_state[:, -2:], expected, err_msg="critic clock missing from training batch")


def audit_pair(left, right, learning_left, learning_right):
    detail = audit_warmup_pair(left, right, learning_left, learning_right)
    np.testing.assert_array_equal(learning_left.value_state[:, -2:], np.zeros_like(learning_left.value_state[:, -2:]),
                                  err_msg="sham critic received time information")
    for batch in (learning_left, learning_right):
        np.testing.assert_array_equal(batch.value_state[:, :-2], batch.state)
    return detail


def audit_result(result, *, raw_path):
    audited = execution.audit_result(result, raw_path=raw_path, specification=spec)
    for iteration, stage in result["snapshots"].items():
        for mode, rows in stage["evaluation_rows"].items():
            for row in rows:
                path = Path(raw_path) / f"iteration_{iteration}" / mode / f"episode_{row['seed']}.npz"
                with np.load(path) as raw:
                    audit_context(None, row, raw["lower_value_context"], clock=spec.VALUE_CLOCK[result["method"]])
    return audited
