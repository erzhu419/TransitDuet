"""Change critic warmup execution while holding lower learning execution fixed."""

import numpy as np
import torch

from . import pointmaze_frozen_execution as execution
from .pointmaze_update_isolation import change_norms
from scripts import pointmaze_warmup_alignment_stage41_spec as spec


def worker_rollout(job):
    return execution.worker_rollout(job, specification=spec)


def audit_warmup_pair(left, right, learning_left, learning_right):
    for name in ("upper_actor", "lower_actor", "upper_value", "promotion_actor", "promotion_value"):
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    for name in ("state", "action", "reward", "duration", "done", "old_logp"):
        np.testing.assert_array_equal(getattr(learning_left, name), getattr(learning_right, name),
                                      err_msg=f"first learning episode {name} differs between warmup arms")
    delta = learning_left.old_value.astype(np.float64) - learning_right.old_value
    return {"first_learning_transitions": learning_left.size,
            "first_learning_value_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "lower_value_parameter_distance": change_norms(
                {"lower_value": left["lower_value"]}, {"lower_value": right["lower_value"]})["lower_value"]}
