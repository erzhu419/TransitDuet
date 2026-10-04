import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_wide_plan_train as experiment
from freq_hrl.experiments import pointmaze_option_residual_train as residual_train
from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts import pointmaze_upper_wide_plan_train_stage115_spec as spec
from scripts.submit_pointmaze_upper_wide_plan_train_stage115_scheduleurm import (
    qualification_task, task_specification,
)


def test_stage115_zero_extra_coordinates_preserve_donor_distribution():
    donor = GaussianActor(3, 4, 8, -0.5)
    branch = experiment.WidePlanResidualActor(donor)
    state = torch.randn(5, 3)
    with torch.inference_mode():
        donor_dist = donor.distribution(state)
        branch_dist = branch.distribution(state)
    torch.testing.assert_close(branch_dist.mean[..., :4], donor_dist.mean, atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.stddev[..., :4], donor_dist.stddev, atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.mean[..., 4:], torch.zeros(5, 4), atol=0, rtol=0)
    torch.testing.assert_close(branch_dist.stddev[..., 4:], donor_dist.stddev, atol=0, rtol=0)
    assert all(not value.requires_grad for value in branch.base.parameters())


def test_stage115_bernstein_head_has_eight_anchored_coordinates():
    plan = experiment.WideBernsteinPlan(None, 50, 1.0, {})
    assert plan.mapper.action_dim == 8
    latent = np.asarray([.1, .2, .3, .4, .5, .6, .7, .8], dtype=np.float64)
    coefficients = plan.plan_coefficients(latent).reshape(2, 5)
    np.testing.assert_array_equal(coefficients[:, 0], np.zeros(2))
    np.testing.assert_allclose(coefficients[0, 1:3], np.tanh(latent[:2]))
    np.testing.assert_allclose(coefficients[0, 3:5], np.tanh(latent[4:6]))
    np.testing.assert_allclose(coefficients[1, 1:3], np.tanh(latent[2:4]))
    np.testing.assert_allclose(coefficients[1, 3:5], np.tanh(latent[6:8]))


def test_stage115_residual_update_uses_full_kl_and_extra_coordinate_fisher():
    donor = GaussianActor(3, 4, 8, -0.5)
    actor = experiment.WidePlanResidualActor(donor)
    parameter_count = actor.readout.weight.numel() + actor.readout.bias.numel()
    score = {
        "gradients": {"A": np.ones(parameter_count), "B": np.full(parameter_count, .5)},
        "states": np.zeros((8, 3), dtype=np.float32),
        "score_costs": {"A": {}, "B": {}},
        "signal_rms": {"A": 1., "B": 1.},
    }
    cost = {"residual_fisher_batches": 0, "residual_kl_checks": 0,
        "residual_parameter_updates": 0, "training_freeze_checks": 0}
    before = {key: value.detach().clone() for key, value in actor.base.state_dict().items()}
    history = residual_train.residual_update(actor, score, cost=cost)
    for key, value in before.items():
        torch.testing.assert_close(actor.base.state_dict()[key], value, atol=0, rtol=0)
    assert history["geometry"]["radius_check"] == "passed"
    assert cost["residual_kl_checks"] == 1


def test_stage115_scheduler_is_dynamic_and_has_qualification_dependency():
    task = task_specification("stage115_test", spec.roots(preflight=True)[0], preflight=True)
    qualification = qualification_task("stage115_test", preflight=True)
    assert task["require_node"] is None
    assert set(task["allowed_nodes"]) == {f"node00{i}" for i in range(1, 7)}
    assert len(qualification["wait_for_files"]) == 1
