import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_full_plan_train as experiment
from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts import pointmaze_upper_full_plan_train_stage116_spec as spec
from scripts.submit_pointmaze_upper_full_plan_train_stage116_scheduleurm import (
    qualification_task, task_specification,
)


def test_stage116_zero_full_readout_preserves_donor_and_zero_extra_plan_means():
    donor = GaussianActor(3, 4, 8, -0.5)
    branch = experiment.FullPlanCoordinateActor(donor)
    state = torch.randn(5, 3)
    with torch.inference_mode():
        donor_dist = donor.distribution(state)
        branch_dist = branch.distribution(state)
    torch.testing.assert_close(branch_dist.mean[..., :4], donor_dist.mean, atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.mean[..., 4:], torch.zeros(5, 4), atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.stddev[..., :4], donor_dist.stddev, atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.stddev[..., 4:], donor_dist.stddev, atol=0, rtol=0)
    assert all(not value.requires_grad for value in branch.base.parameters())


def test_stage116_scheduler_is_dynamic_and_has_qualification_dependency():
    task = task_specification("stage116_test", spec.roots(preflight=True)[0], preflight=True)
    qualification = qualification_task("stage116_test", preflight=True)
    assert task["require_node"] is None
    assert set(task["allowed_nodes"]) == {f"node00{i}" for i in range(1, 7)}
    assert len(qualification["wait_for_files"]) == 1

