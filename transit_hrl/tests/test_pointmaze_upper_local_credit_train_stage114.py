from types import SimpleNamespace

import numpy as np

from freq_hrl.experiments import pointmaze_upper_local_credit_train as experiment


def test_local_credit_is_aligned_to_upper_decisions(monkeypatch):
    class Batch:
        def __init__(self, reward, episode):
            self.reward = np.asarray(reward, dtype=np.float32)
            self.episode = float(episode)

    pairs = [{"lower_batches": [Batch([1, 2], 3), Batch([3, 5], 8)],
        "upper_batches": [Batch([1, 2], 3), Batch([3, 5], 8)],
        "rows": [{"episode_return": 3}, {"episode_return": 8}]}]
    captured = []

    monkeypatch.setattr(experiment.base.independent, "exact_returns",
        lambda batch, gamma: np.asarray([batch.episode], dtype=np.float64))
    monkeypatch.setattr(experiment.base, "concat_level_batches",
        lambda batches: SimpleNamespace(state=np.zeros((4, 390), dtype=np.float32)))

    def fake_score(actor, upper, signals, *, clip_ratio, chunk_size):
        captured.append(np.asarray(signals["scenario"]))
        return {"scenario": np.ones(3, dtype=np.float64)}, {
            "actor_score_forward_batches": 1, "actor_score_backward_batches": 2}

    monkeypatch.setattr(experiment.base.lower_training, "residual_actor_gradients", fake_score)
    cost = {"objective_checks": 0, "mc_calls": 0,
        "actor_score_forward_batches": 0, "actor_score_backward_batches": 0}
    score = experiment.score_upper_local(object(), {"A": pairs, "B": pairs},
        horizon=4, period=2, cost=cost)

    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], np.asarray([-2, -3, 2, 3], dtype=np.float64))
    assert all(signal.shape == (4,) for signal in captured)
    assert score["gradients"]["A"].shape == (3,)
    assert cost["objective_checks"] == 4
    assert cost["mc_calls"] == 4
