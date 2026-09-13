import numpy as np

from freq_hrl.core.causal_terminal_reserve_projector import CausalTerminalReserveProjector
from scripts.diagnose_mujoco_v25_projection_target_noise import (
    loss_gradients, snapshot, target_variance,
)
from scripts.submit_mujoco_v25_target_noise_scheduleurm import cells, task_spec


def test_crossed_variance_separates_upper_lower_and_interaction():
    upper = np.array([-1.0, 1.0])[:, None, None]
    lower = np.array([-2.0, 2.0])[None, :, None]
    parts = target_variance(upper + lower + upper * lower)
    assert parts == {"total": 9.0, "upper_sampling": 1.0, "lower_sampling": 4.0, "interaction": 4.0}


def test_action_sample_has_zero_gradient_when_projection_does_not_change_action():
    samples = np.array([[[-1.5, 0.4]], [[1.5, -0.4]]])
    gradients = loss_gradients(np.zeros(2), samples, np.tanh(samples))
    np.testing.assert_array_equal(gradients["action_sample"], np.zeros_like(samples))
    assert np.mean(gradients["raw_mean"] ** 2) > 0.1


def test_action_sample_gradient_matches_fixed_noise_finite_difference():
    mean = np.array([0.3, -0.2])
    sample = np.array([[[0.8, -1.2]]])
    target = np.array([[[0.2, -0.3]]])
    analytical = loss_gradients(mean, sample, target)["action_sample"][0, 0]
    delta = 1e-6
    numerical = []
    for k in range(2):
        offset = np.zeros(2)
        offset[k] = delta
        plus = np.mean((np.tanh(sample + offset) - target) ** 2)
        minus = np.mean((np.tanh(sample - offset) - target) ** 2)
        numerical.append((plus - minus) / (2 * delta))
    np.testing.assert_allclose(analytical, numerical, atol=1e-10)


def test_snapshot_uses_one_history_for_all_counterfactuals():
    projector = CausalTerminalReserveProjector()
    projector.reset(3)
    projector.project(np.array([0.2, -0.1, 0.3]), np.array([0.1, 0.2, -0.1]))
    result = snapshot(projector, np.zeros(3), np.zeros(3),
                      np.array([[0.4, 0.2, -0.3], [-0.4, -0.2, 0.3]]),
                      np.array([[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4]]))
    assert result["context_unchanged"]
    for level in result["levels"].values():
        parts = level["raw_target_variance"]
        np.testing.assert_allclose(parts["total"], sum(parts[k] for k in ("upper_sampling", "lower_sampling", "interaction")))


def test_fixed_matrix_stays_small_and_unpinned():
    matrix = cells()
    assert len(matrix) == len(set(matrix)) == 48
    task = task_spec("test", "revision", matrix[0])
    assert task["cpu"] == 1
    assert task["require_node"] is None
    assert len(task["allowed_nodes"]) == 6
    assert task["allow_duplicate"] is False
