from types import SimpleNamespace

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.responsibility_distillation import (
    causal_macro_responsibility_targets,
    distill_hierarchical_actor_heads,
    fit_actor_output_head,
)


def _raw(action: np.ndarray) -> np.ndarray:
    return np.arctanh(np.asarray(action, dtype=np.float64))


def _low_power(values: np.ndarray, window: int) -> float:
    rows = np.asarray(values, dtype=np.float64)
    low = np.asarray([
        np.mean(rows[max(0, index - window + 1):index + 1], axis=0)
        for index in range(rows.shape[0])
    ])
    return float(np.mean(np.square(low)))


def _high_power(values: np.ndarray, window: int) -> float:
    rows = np.asarray(values, dtype=np.float64)
    low = np.asarray([
        np.mean(rows[max(0, index - window + 1):index + 1], axis=0)
        for index in range(rows.shape[0])
    ])
    return float(np.mean(np.square(rows - low)))


def test_causal_targets_reconstruct_and_ignore_future_lower_actions():
    upper_action = np.asarray([[0.2], [-0.1], [0.3]], dtype=np.float64)
    lower_action = np.asarray(
        [[0.1], [0.2], [0.15], [0.05], [-0.2], [0.1]],
        dtype=np.float64,
    )
    durations = np.asarray([2, 2, 2], dtype=np.float64)
    baseline = causal_macro_responsibility_targets(
        _raw(upper_action),
        _raw(lower_action),
        durations,
        slow_alpha=0.5,
        transfer_strength=1.0,
    )
    changed = lower_action.copy()
    changed[4:] = np.asarray([[0.4], [-0.4]])
    future_changed = causal_macro_responsibility_targets(
        _raw(upper_action),
        _raw(changed),
        durations,
        slow_alpha=0.5,
        transfer_strength=1.0,
    )

    np.testing.assert_allclose(
        baseline.upper_action, future_changed.upper_action, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        baseline.lower_action[:4],
        future_changed.lower_action[:4],
        rtol=0.0,
        atol=0.0,
    )
    original_total = upper_action[baseline.macro_index] + lower_action
    np.testing.assert_allclose(
        baseline.upper_action[baseline.macro_index] + baseline.lower_action,
        original_total,
        rtol=0.0,
        atol=1e-12,
    )
    assert baseline.reconstruction_max_abs <= 1e-12


def test_causal_targets_reduce_both_bands_for_compensating_actions():
    macro_count = 10
    duration = 4
    upper_action = np.asarray(
        [[0.4 if index % 2 == 0 else -0.4] for index in range(macro_count)],
        dtype=np.float64,
    )
    repeated_upper = np.repeat(upper_action, duration, axis=0)
    lower_action = 0.2 - repeated_upper
    targets = causal_macro_responsibility_targets(
        _raw(upper_action),
        _raw(lower_action),
        np.full(macro_count, duration),
        slow_alpha=1.0,
        transfer_strength=1.0,
    )
    repeated_target_upper = targets.upper_action[targets.macro_index]

    assert _high_power(repeated_target_upper, 8) < _high_power(
        repeated_upper, 8
    )
    assert _low_power(targets.lower_action, 32) < _low_power(lower_action, 32)


def test_actor_output_head_ridge_fit_reduces_target_error():
    torch.manual_seed(17)
    rng = np.random.default_rng(19)
    actor = GaussianActor(2, 1, hidden_dim=0, init_log_std=-1.0)
    states = rng.normal(size=(64, 2))
    target = 0.7 * states[:, :1] - 0.3 * states[:, 1:] + 0.2
    diagnostics = fit_actor_output_head(
        actor, states, target, ridge=0.0, blend=1.0
    )

    assert diagnostics["target_mse_after"] < 1e-12
    assert diagnostics["target_mse_after"] < diagnostics["target_mse_before"]


def test_actor_output_head_parameter_trust_region_caps_the_update():
    torch.manual_seed(31)
    actor = GaussianActor(2, 1, hidden_dim=0, init_log_std=-1.0)
    states = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    target = np.full((4, 1), 20.0, dtype=np.float64)
    diagnostics = fit_actor_output_head(
        actor,
        states,
        target,
        ridge=0.0,
        blend=1.0,
        parameter_delta_rms_limit=0.05,
    )

    assert diagnostics["requested_parameter_delta_rms"] > 0.05
    assert diagnostics["parameter_delta_rms"] <= 0.05 + 1e-12
    assert diagnostics["trust_region_scale"] < 1.0


def test_causal_targets_limit_saturated_raw_logits():
    upper_action = np.asarray([[0.9999]], dtype=np.float64)
    lower_action = np.asarray([[0.0]], dtype=np.float64)
    targets = causal_macro_responsibility_targets(
        _raw(upper_action),
        _raw(lower_action),
        np.asarray([1.0]),
        raw_target_limit=2.5,
    )

    assert float(np.max(np.abs(targets.upper_raw))) == 2.5
    assert targets.raw_target_clip_fraction > 0.0


def test_hierarchical_distillation_updates_both_heads():
    torch.manual_seed(23)
    model = SimpleNamespace(
        upper_actor=GaussianActor(2, 1, hidden_dim=4, init_log_std=-1.0),
        lower_actor=GaussianActor(3, 1, hidden_dim=4, init_log_std=-1.0),
    )
    rng = np.random.default_rng(29)
    lower_state = rng.normal(size=(6, 3))
    lower_state[:, 1] = np.repeat([0.2, -0.2, 0.1], 2)
    trajectory = SimpleNamespace(
        upper=SimpleNamespace(
            state=rng.normal(size=(3, 2)),
            action=_raw(np.asarray([[0.2], [-0.2], [0.1]])),
            duration=np.asarray([2.0, 2.0, 2.0]),
        ),
        lower=SimpleNamespace(
            state=lower_state,
            action=_raw(np.asarray(
                [[0.1], [0.2], [0.3], [0.1], [-0.2], [0.0]]
            )),
        ),
    )
    diagnostics = distill_hierarchical_actor_heads(
        model,
        [trajectory],
        upper_action_scale=1.0,
        lower_action_scale=1.0,
        slow_alpha=0.5,
        transfer_strength=0.75,
        ridge=1e-3,
        blend=0.5,
        lower_action_context_start=1,
    )

    assert diagnostics["target_reconstruction_max_abs"] <= 1e-12
    assert diagnostics["upper_fit"]["target_mse_after"] < diagnostics[
        "upper_fit"
    ]["target_mse_before"]
    assert diagnostics["lower_fit"]["target_mse_after"] < diagnostics[
        "lower_fit"
    ]["target_mse_before"]
    assert diagnostics["lower_action_context_counterfactual"] is True
    assert diagnostics["lower_action_context_shift_rms"] > 0.0
    assert diagnostics["student_target_reconstruction_max_abs"] <= 1e-12
