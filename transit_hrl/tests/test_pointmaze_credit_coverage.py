import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_credit_coverage as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


def test_label_uses_one_complete_noise_path_and_full_suffix(source_data):
    models, predictor, cal, args, teachers = source_data
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)))
        trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
        noise, seed = 127110001, 127100001
        _, baseline, _ = experiment.joint.native_episode(trainer, args=args, seed=seed, noise_seed=noise,
            arm="forecast", period=50, predictor=predictor, envelope=cal["50"]["envelope"], collect=False)
        for start in (0, 50):
            query = {"scenario_seed": seed, "noise_seed": noise, "panel": "A", "start": start}
            row = experiment.worker_label((experiment.joint.weights(models["50"]), teachers["50"], query,
                50, predictor, cal["50"]["envelope"]))
            np.testing.assert_allclose(row["zero_episode_return"], baseline["episode_return"], atol=1e-9, rtol=0)
            assert row["pairing_and_freeze"] == "passed" and len(row["gradient"]) == 8
            assert row["native_episodes"] == 17 and row["native_steps"] == 1700
            assert row["max_reference_peak"] <= .05 + 1e-8
            if start == 0:
                assert row["zero_suffix_return"] == row["zero_episode_return"]


def test_controls_share_fisher_states_and_sparse_labels_are_inverse_weighted(source_data):
    models, _, _, args, teachers = source_data
    actor = experiment.joint.make_trainer(models["50"], teachers["50"], args).upper_actor
    before = copy.deepcopy(actor.state_dict())
    labels = [{"state": np.arange(392, dtype=np.float32) * .0001 + i * .01,
        "gradient": np.arange(1., 9.) * (i + 1),
        "query": {"scenario_seed": 1, "panel": panel, "start": start}}
        for i, (panel, start) in enumerate((p, s) for p in ("A", "B") for s in (0, 50))]
    captured = {}
    original = experiment.native_mean_directions
    def record(states, signals, std, **kw):
        captured.update(states=states.copy(), signals=signals)
        return original(states, signals, std, **kw)
    with patch.object(experiment.spec, "arguments", return_value=args), \
            patch.object(experiment.spec, "COARSE_STARTS", (0,)), \
            patch.object(experiment, "native_mean_directions", side_effect=record):
        candidates, summary, work = experiment.learn_directions(actor, labels, 50)
    np.testing.assert_array_equal(captured["states"], [r["state"] for r in labels])
    for panel, indices in (("A", (0, 1)), ("B", (2, 3))):
        complete, coarse = captured["signals"]["complete_" + panel], captured["signals"]["coarse_" + panel]
        for index in indices:
            np.testing.assert_array_equal(complete[index], 2 * labels[index]["gradient"])
        np.testing.assert_array_equal(coarse[indices[0]], 4 * labels[indices[0]]["gradient"])
        assert not coarse[indices[1]].any()
    assert summary["shared_geometry"]["training_rows"] == 4
    assert summary["directions"]["coarse"]["credit_rows_used"] == 2
    assert summary["directions"]["complete"]["credit_rows_used"] == 4
    assert work == {"empirical_fisher_solves": 1, "fisher_jvp_batches": 2, "exact_kl_forward_batches": 4}
    for method in candidates:
        plus = candidates[method]["plus"]
        means = torch.nn.functional.linear(torch.as_tensor(captured["states"]),
            plus["net.0.weight"], plus["net.0.bias"]).detach().double().numpy()
        expected = np.sum(means * np.asarray([r["gradient"] for r in labels])) / 2
        np.testing.assert_allclose(summary["directions"][method]["predicted_episode_directional_derivative"], expected, atol=1e-12)
        for sign, state in candidates[method].items():
            torch.testing.assert_close(state["log_std"], before["log_std"], atol=0, rtol=0)
            np.testing.assert_allclose(summary["directions"][method]["radius"]["exact_kl"][sign],
                experiment.spec.FISHER_RADIUS, rtol=2e-6)
    torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)


def test_reduced_run_has_exact_budget_training_only_fits_and_no_state_exports(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    before = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    spec = experiment.spec
    fit_seeds, original_fit = [], experiment.source.fit_method
    def record_fit(rows, method):
        fit_seeds.extend(r["zero"]["seed"] for r in rows)
        return original_fit(rows, method)
    with ExitStack() as stack:
        native_task(stack)
        for name, value in (("LABEL_SCENARIOS", 1), ("TRAINING_SCENARIOS", 1), ("EVALUATION_EPISODES", 1), ("COARSE_STARTS", (0,))):
            stack.enter_context(patch.object(spec, name, value))
        stack.enter_context(patch.object(spec, "arguments", return_value=args))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(experiment.source, "fit_method", side_effect=record_fit))
        result = experiment.run(410011, tmp_path / "result.json")
        assert result["cost"] == spec.budget()
        assert set(fit_seeds) == {r["scenario_seed"] for r in spec.training_roles(410011)}
    assert result["cost"]["native_episodes"] == 162 and result["cost"]["native_steps"] == 16200
    assert len(list(tmp_path.rglob("*.pt"))) == 4 and not list(tmp_path.rglob("*.npz"))
    for p, group in result["groups"].items():
        assert all("state" not in r and r["pairing_and_freeze"] == "passed" for r in group["native_labels"])
        assert group["effects"]["complete_minus_source_forecast"] == group["effects"]["complete_minus_complete_blinded"]
        assert group["lower_and_critics_frozen"] == "passed"
        assert set(group["gradient_consistency_audit"]) == {"coarse", "complete"}
        experiment.joint.source.native.curves.support.assert_frozen(models[p], before[p])
    # With one decision at period100, the coarse and complete controls coincide.
    assert result["groups"]["100"]["effects"]["complete_minus_coarse"]["mean"] == 0.


def test_frozen_budgets_temporal_coverage_and_disjoint_rosters():
    spec = experiment.spec
    assert spec.budget()["label_queries"] == 288 and spec.budget()["label_episodes"] == 4896
    assert spec.budget()["native_episodes"] == 5856 and spec.budget()["native_steps"] == 7027200
    assert spec.FISHER_RADIUS == .005**2/(2*.15**2) and spec.MINIMUM_GAIN == .5
    assert spec.PERIODS == (50, 100) and spec.AUDIT_SCALE == .1
    seen = set()
    for root in spec.ROOTS:
        label = {s for r in spec.label_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        train = {s for r in spec.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        evaluation = set(spec.evaluation_seeds(root))
        old = set(spec.source.evaluation_seeds(root)) | {s for r in spec.source.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        assert len(label) == 12 and len(train) == 48 and len(evaluation) == 32
        assert not label & train and not (label | train) & evaluation
        assert not (label | train | evaluation) & (old | seen)
        for period in spec.PERIODS:
            for role in spec.label_roles(root):
                for panel in spec.PANELS:
                    rows = [q for q in spec.queries(root, period) if q["scenario_seed"] == role["scenario_seed"] and q["panel"] == panel]
                    assert [q["start"] for q in rows] == list(range(0, 1200, period))
                    assert {q["noise_seed"] for q in rows} == {role["noise_seeds"][panel]}
            assert all(s % period == 0 for s in spec.COARSE_STARTS)
        seen.update(label | train | evaluation)
