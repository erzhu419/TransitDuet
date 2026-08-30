import numpy as np

from freq_hrl.experiments.mujoco.state_conditioned_actor import (
    apply_state_conditioned_actor,
    causal_state_actor_features,
    fit_state_conditioned_actor,
)


def test_state_actor_features_never_read_future_state_or_proposals():
    state = np.arange(12, dtype=np.float64).reshape(4, 3)
    total = np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0
    upper = total * 0.4
    features = causal_state_actor_features(
        state, total, upper, proposal_window=2
    )
    changed_state = state.copy()
    changed_total = total.copy()
    changed_upper = upper.copy()
    changed_state[3] = 999.0
    changed_total[3] = 999.0
    changed_upper[3] = -999.0
    changed = causal_state_actor_features(
        changed_state, changed_total, changed_upper, proposal_window=2
    )
    assert np.array_equal(features[:3], changed[:3])
    assert features.shape == (4, 11)


def test_state_actor_fits_nonlinear_state_target_and_preserves_bounds():
    generator = np.random.default_rng(18021)
    states = []
    totals = []
    uppers = []
    targets = []
    for _ in range(6):
        state = generator.normal(size=(48, 4))
        total = 0.2 * generator.normal(size=(48, 2))
        upper = 0.5 * total
        target = 0.04 * np.stack(
            (np.tanh(state[:, 0]), np.tanh(state[:, 1] * state[:, 2])),
            axis=1,
        )
        states.append(state)
        totals.append(total)
        uppers.append(upper)
        targets.append(target)
    model = fit_state_conditioned_actor(
        states,
        totals,
        uppers,
        targets,
        [1.0] * len(states),
        proposal_window=1,
        hidden_dim=32,
        hidden_layers=2,
        correction_abs_limit=0.05,
        learning_rate=3e-3,
        weight_decay=1e-5,
        epochs=160,
        random_seed=18023,
    )
    predicted = np.concatenate([
        apply_state_conditioned_actor(state, total, upper, model)[
            "correction"
        ]
        for state, total, upper in zip(states, totals, uppers, strict=True)
    ])
    expected = np.concatenate(targets)
    assert np.mean(np.square(predicted - expected)) < 1.5e-4
    assert np.max(np.abs(predicted)) <= 0.05 + 1e-12


def test_state_actor_path_balancing_handles_unequal_trajectory_lengths():
    short_state = np.ones((8, 2), dtype=np.float64)
    long_state = -np.ones((80, 2), dtype=np.float64)
    short_total = np.zeros((8, 1), dtype=np.float64)
    long_total = np.zeros((80, 1), dtype=np.float64)
    model = fit_state_conditioned_actor(
        [short_state, long_state],
        [short_total, long_total],
        [short_total, long_total],
        [np.full((8, 1), 0.04), np.zeros((80, 1))],
        [10.0, 1.0],
        proposal_window=1,
        hidden_dim=16,
        hidden_layers=1,
        correction_abs_limit=0.05,
        learning_rate=5e-3,
        weight_decay=0.0,
        epochs=120,
        random_seed=18029,
    )
    short_prediction = apply_state_conditioned_actor(
        short_state, short_total, short_total, model
    )["correction"]
    long_prediction = apply_state_conditioned_actor(
        long_state, long_total, long_total, model
    )["correction"]
    assert float(np.mean(short_prediction)) > 0.03
    assert abs(float(np.mean(long_prediction))) < 0.01
    assert model["fit_total_path_weight"] == 11.0
