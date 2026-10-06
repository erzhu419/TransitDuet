import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from freq_hrl.experiments import pointmaze_native_geometry as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


def test_dual_native_geometry_matches_primal_damped_fisher_with_constant_columns():
    states = np.array([[1., 2., 7.], [2., -1., 7.], [-1., 0., 7.], [0., 3., 7.]])
    signals = {"A": np.array([[1., 2.], [0., -1.], [-1., 1.], [2., 0.]]), "B": np.ones((4, 2))}
    std = np.array([.15, .3])
    directions, geometry = native_mean_directions(states, signals, std, damping=1.)
    normalized = (states[:, :2] - states[:, :2].mean(0)) / states[:, :2].std(0) / np.sqrt(2)
    design = np.c_[normalized, np.ones(4)]
    for panel in signals:
        coefficient = np.linalg.solve(design.T @ design + np.eye(3), design.T @ (signals[panel] * std ** 2))
        predicted = states @ directions[panel]["weight"].T + directions[panel]["bias"]
        np.testing.assert_allclose(predicted, design @ coefficient, atol=1e-14, rtol=0)
        assert not directions[panel]["weight"][:, 2].any()
    assert geometry["active_state_dimensions"] == 2 and geometry["design_rank"] == 3
    changed = states * np.array([1000., .01, 2.]) + np.array([50., -20., 1.])
    alternate, _ = native_mean_directions(changed, signals, std, damping=1.)
    for panel in signals:
        np.testing.assert_allclose(changed @ alternate[panel]["weight"].T + alternate[panel]["bias"],
            states @ directions[panel]["weight"].T + directions[panel]["bias"], atol=1e-12, rtol=0)


def test_preconditioned_actor_has_matched_radius_and_frozen_source(source_data):
    models, _, _, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    snapshot = experiment.joint.weights(trainer)
    states = np.stack([np.linspace(-.2, .3, 392), np.linspace(.1, -.1, 392), np.ones(392) * .05]).astype(np.float32)
    labels = [{"gradients": {p: (np.arange(1., 9.) * value).tolist() for p, value in (("A", i + 1.), ("B", i + 1.1))}}
        for i in range(3)]
    candidates, geometry, cost = experiment.natural_candidates(trainer.upper_actor, states, labels)
    assert geometry["preconditioned_panel_cosine"] > .9
    assert cost == {"fisher_jvp_batches": 1, "exact_kl_forward_batches": 2}
    for sign, state in candidates.items():
        torch.testing.assert_close(state["log_std"], snapshot["upper_actor"]["log_std"], atol=0, rtol=0)
        np.testing.assert_allclose(geometry["radius"]["exact_kl"][sign], experiment.spec.FISHER_RADIUS, rtol=2e-6)
    torch.testing.assert_close(experiment.joint.weights(trainer), snapshot, atol=0, rtol=0)


def test_reduced_run_matches_calibration_budget_and_never_fits_evaluation(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    before = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    cache = tmp_path / "labels" / "result.json"
    source_path = tmp_path / "euclidean" / "result.json"
    label_spec = experiment.spec.source.source.source
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(label_spec, "arguments", return_value=args))
        stack.enter_context(patch.object(label_spec, "STARTS", (0,)))
        stack.enter_context(patch.object(label_spec, "SCENARIOS_PER_START", 1))
        stack.enter_context(patch.object(experiment.spec, "TRAINING_SCENARIOS", 1))
        stack.enter_context(patch.object(experiment.spec, "EVALUATION_EPISODES", 1))
        stack.enter_context(patch.object(experiment.spec, "label_result", return_value=cache))
        stack.enter_context(patch.object(experiment.source.spec, "source_result", return_value=source_path))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state",
            side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        groups = {}
        for period in experiment.spec.PERIODS:
            p = str(period)
            trainer = experiment.joint.make_trainer(models[p], teachers[p], args)
            initial = trainer.upper_actor.state_dict()
            query = label_spec.queries(410011)[0]
            row, _ = experiment.source.source.source.intervention_episode(trainer, args=args, query=query, panel="A",
                action_delta=np.zeros(8), period=period, predictor=predictor, envelope=cal[p]["envelope"])
            groups[p] = {"queries": [{"query": query, "gradients": {panel: np.arange(1., 9.).tolist() for panel in ("A", "B")},
                "coordinate_suffix_returns": {"A": {"zero": row["suffix_return"]}}}]}
            for sign, value in (("plus", .00001), ("minus", -.00001)):
                weights = {k: v.clone() if k == "log_std" else v + value for k, v in initial.items()}
                path = source_path.parent / "final_weights" / f"period_{period}_{sign}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": experiment.spec.source.source.PROTOCOL, "root": 410011, "period": period,
                    "sign": sign, "fisher_radius": experiment.spec.FISHER_RADIUS, "weights": weights}, path)
        experiment.write_json(cache, {"status": "complete", "protocol": label_spec.PROTOCOL, "root": 410011,
            "cost": label_spec.budget(), "groups": groups})
        real_fit, fit_seeds = experiment.fit_method, []
        def record_fit(rows, method):
            fit_seeds.extend(r["zero"]["seed"] for r in rows)
            return real_fit(rows, method)
        stack.enter_context(patch.object(experiment, "fit_method", side_effect=record_fit))
        result = experiment.run(410011, tmp_path / "candidate" / "result.json")
        assert result["cost"] == experiment.spec.budget()
        assert set(fit_seeds) == {r["scenario_seed"] for r in experiment.spec.training_roles(410011)}
    assert result["cost"]["native_episodes"] == 46 and result["cost"]["native_steps"] == 4600
    assert result["cost"]["upper_candidate_weight_steps"] == 18
    assert len(list((tmp_path / "candidate").rglob("*.pt"))) == 2
    for p, group in result["groups"].items():
        assert set(group["training_native_return_fits"]) == {"euclidean", "natural"}
        assert group["effects"]["natural_minus_source_forecast"] == group["effects"]["natural_minus_natural_blinded"]
        assert group["lower_and_critics_frozen"] == "passed"
        experiment.joint.source.native.curves.support.assert_frozen(models[p], before[p])


def test_frozen_rosters_budget_and_no_seed_reuse():
    spec = experiment.spec
    assert spec.MINIMUM_GAIN == .5 and spec.DAMPING == 1. and spec.PERIODS == (50, 100)
    assert spec.budget()["native_episodes"] == 912 and spec.budget()["native_steps"] == 1094400
    seen = set()
    for root in spec.ROOTS:
        old = set(spec.source.evaluation_seeds(root)) | set(spec.source.source.evaluation_seeds(root))
        old.update(s for q in spec.source.training_roles(root) for s in (q["scenario_seed"], *q["noise_seeds"].values()))
        labels = spec.source.source.source
        old.update(s for q in labels.queries(root) for s in (q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values()))
        roles = labels.source.seed_roles(root, preflight=False)
        old.update(s for batch in roles["training_rounds"] for q in batch for s in (q["scenario_seed"], *q["noise_seeds"]))
        old.update(roles["native_evaluation"])
        train = [s for q in spec.training_roles(root) for s in (q["scenario_seed"], *q["noise_seeds"].values())]
        evaluation = spec.evaluation_seeds(root)
        assert len(train) == len(set(train)) == 48 and len(evaluation) == len(set(evaluation)) == 32
        assert not set(train).intersection(old | seen | set(evaluation))
        assert not set(evaluation).intersection(old | seen)
        seen.update(train + evaluation)
