import numpy as np

from freq_hrl.experiments.mujoco.causal_actor_adapter import (
    apply_causal_actor_adapter,
    causal_actor_features,
    fit_causal_actor_adapter,
)


def test_actor_features_are_strictly_prefix_causal():
    total = np.arange(24, dtype=np.float64).reshape(8, 3) / 24.0
    upper = 0.6 * total
    reference = causal_actor_features(total, upper, window=3)
    changed_total = total.copy()
    changed_upper = upper.copy()
    changed_total[5:] = -9.0
    changed_upper[5:] = 7.0
    changed = causal_actor_features(
        changed_total, changed_upper, window=3
    )
    assert np.array_equal(reference[:5], changed[:5])


def test_path_balanced_weighted_fit_recovers_causal_residual_mapping():
    generator = np.random.default_rng(171301)
    coefficient = np.array([
        [[0.10], [-0.05]],
        [[-0.03], [0.02]],
    ])
    totals = [
        generator.normal(scale=0.3, size=(length, 1))
        for length in (20, 80, 35)
    ]
    uppers = [
        generator.normal(scale=0.2, size=total.shape) for total in totals
    ]
    targets = [
        causal_actor_features(total, upper, window=2)
        @ coefficient.reshape(4, 1)
        for total, upper in zip(totals, uppers, strict=True)
    ]
    model = fit_causal_actor_adapter(
        totals,
        uppers,
        targets,
        [1.0, 3.0, 2.0],
        window=2,
        ridge_penalty=1e-10,
        feature_scale_floor=1e-12,
    )
    assert np.allclose(model["coefficients"], coefficient, atol=1e-7)


def test_adapter_enforces_trust_region_and_reports_executed_change():
    total = np.array([[1.2], [0.5], [-1.4]], dtype=np.float64)
    upper = 0.5 * total
    model = {
        "window": 1,
        "action_dimension": 1,
        "proposal_dimension": 2,
        "coefficients": np.array([[[2.0], [2.0]]]),
    }
    result = apply_causal_actor_adapter(
        total,
        upper,
        model,
        output_gain=1.0,
        correction_abs_limit=0.04,
    )
    assert np.max(np.abs(result["correction"])) <= 0.04 + 1e-12
    assert np.max(np.abs(result["corrected_total"])) <= 2.0
    assert result["executed_correction"][0, 0] == 0.0
    assert result["executed_correction"][1, 0] != 0.0
    assert result["executed_correction"][2, 0] == 0.0


def test_zero_targets_produce_exact_identity_adapter():
    total = np.linspace(-0.8, 0.8, 18, dtype=np.float64).reshape(9, 2)
    upper = 0.25 * total
    model = fit_causal_actor_adapter(
        [total],
        [upper],
        [np.zeros_like(total)],
        [1.0],
        window=3,
        ridge_penalty=1e-3,
        feature_scale_floor=1e-8,
    )
    result = apply_causal_actor_adapter(
        total,
        upper,
        model,
        output_gain=2.0,
        correction_abs_limit=0.05,
    )
    assert np.array_equal(result["corrected_total"], total)
    assert np.array_equal(result["correction"], np.zeros_like(total))
