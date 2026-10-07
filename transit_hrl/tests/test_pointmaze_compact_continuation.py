import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_compact_continuation as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


def test_refreshed_and_stale_updates_use_current_policy_kl_and_incremental_prediction(source_data):
    models, _, _, args, teachers = source_data
    actor = experiment.joint.make_trainer(models["50"], teachers["50"], args).upper_actor
    origin = copy.deepcopy(actor.state_dict())
    with torch.no_grad():
        actor.net[0].bias.fill_(.25)
    current = copy.deepcopy(actor.state_dict())
    rng = np.random.default_rng(130)
    labels = [{"state": rng.normal(size=392).astype(np.float32), "gradient": np.arange(1., 9.),
        "query": {"scenario_seed": i//2, "panel": ("A", "B")[i%2]}} for i in range(8)]
    candidates, learning, cost = experiment.learn_continuations(actor, origin, labels)
    states = torch.as_tensor(np.asarray([r["state"] for r in labels]))
    baseline = actor.distribution(states)
    for m, directions in candidates.items():
        candidate = copy.deepcopy(actor)
        for sign, weights in directions.items():
            candidate.load_state_dict(weights)
            dist = candidate.distribution(states)
            old = torch.distributions.Normal(baseline.mean.double(), baseline.stddev.double())
            new = torch.distributions.Normal(dist.mean.double(), dist.stddev.double())
            kl = torch.distributions.kl_divergence(old, new).sum(-1).mean().item()
            np.testing.assert_allclose(kl, experiment.spec.FISHER_RADIUS, rtol=2e-5)
            torch.testing.assert_close(weights["log_std"], current["log_std"], atol=0, rtol=0)
        candidate.load_state_dict(directions["plus"])
        delta = (candidate.distribution(states).mean - baseline.mean).detach().double().numpy()
        expected = (delta*np.asarray([r["gradient"] for r in labels])).sum(1).mean()
        np.testing.assert_allclose(learning[m]["predicted_incremental_episode_derivative"], expected, atol=1e-9)
        assert learning[m]["policy_mean_rms"] > 50*learning[m]["incremental_mean_rms"]
    torch.testing.assert_close(actor.state_dict(), current, atol=0, rtol=0)
    assert cost == {"empirical_fisher_solves": 1, "fisher_jvp_batches": 2, "exact_kl_forward_batches": 4}


def test_native_label_runs_nonzero_upper_and_its_actual_prefix(source_data):
    models, predictor, cal, args, teachers = source_data
    joint = experiment.joint
    trainer = joint.make_trainer(models["50"], teachers["50"], args)
    with torch.no_grad():
        trainer.upper_actor.net[0].bias.copy_(torch.arange(1., 9.)*.001)
    upper = copy.deepcopy(trainer.upper_actor.state_dict())
    q = {"scenario_seed": 130100001, "noise_seed": 130110001, "panel": "A", "start": 50}
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(joint.source.native, "_WORKER", (models["50"], args)))
        row = experiment.worker_label((joint.weights(models["50"]), teachers["50"], upper, q, 50, predictor, cal["50"]["envelope"]))
        _, expected, _ = joint.native_episode(trainer, args=args, seed=q["scenario_seed"], noise_seed=q["noise_seed"],
            arm="joint", period=50, predictor=predictor, envelope=cal["50"]["envelope"], collect=False)
    np.testing.assert_allclose(row["baseline_episode_return"], expected["episode_return"], atol=1e-12, rtol=0)
    assert row["pairing_and_freeze"] == "passed" and row["native_episodes"] == 17
    assert row["max_reference_peak"] <= .05+1e-8


def test_reduced_continuation_keeps_learned_baseline_and_fits_training_only(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    cache = tmp_path/"source"/"result.json"
    source_groups = {}
    for period in (50, 100):
        actor = experiment.joint.make_trainer(models[str(period)], teachers[str(period)], args).upper_actor
        with torch.no_grad():
            actor.net[0].bias.copy_(torch.arange(1., 9.)*.001)
        fit = {"scale": 1.}
        path = cache.parent/"final_weights"/f"period_{period}_compact_upper.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": experiment.spec.source.PROTOCOL, "root": 410011, "period": period,
            "method": "compact", "fit": fit, "weights": actor.state_dict()}, path)
        source_groups[str(period)] = {"training_native_return_fits": {"compact": {"pooled": fit}}}
    experiment.write_json(cache, {"status": "complete", "protocol": experiment.spec.source.PROTOCOL,
        "root": 410011, "cost": experiment.spec.source.budget(), "groups": source_groups})
    seen, original_fit = [], experiment.fit_method
    def record_fit(rows, method):
        seen.extend(r["zero"]["seed"] for r in rows)
        assert all(r["zero"]["upper_mean_rms"] > 0. for r in rows)
        return original_fit(rows, method)
    before = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    with ExitStack() as stack:
        native_task(stack)
        for name in ("LABEL_SCENARIOS", "TRAINING_SCENARIOS", "EVALUATION_EPISODES"):
            stack.enter_context(patch.object(experiment.spec, name, 1))
        stack.enter_context(patch.object(experiment.spec, "arguments", return_value=args))
        stack.enter_context(patch.object(experiment.spec, "source_result", return_value=cache))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(experiment, "fit_method", side_effect=record_fit))
        result = experiment.run(410011, tmp_path/"new"/"result.json")
        assert result["cost"] == experiment.spec.budget()
        assert set(seen) == {r["scenario_seed"] for r in experiment.spec.training_roles(410011)}
    assert result["cost"]["native_episodes"] == 148 and result["cost"]["native_steps"] == 14800
    assert len(list((tmp_path/"new").rglob("*.pt"))) == 4 and not list(tmp_path.rglob("*.npz"))
    for p, g in result["groups"].items():
        assert g["effects"]["refresh_minus_source_forecast"] == g["effects"]["refresh_minus_refresh_blinded"]
        assert g["mean_metrics"]["single"]["upper_mean_rms"] > 0.
        assert all("state" not in r for r in g["native_labels"])
        experiment.joint.source.native.curves.support.assert_frozen(models[p], before[p])


def test_fixed_rosters_and_incremental_budget_do_not_reuse_old_scenes():
    spec = experiment.spec
    assert spec.budget()["native_episodes"] == 5856 and spec.budget()["native_steps"] == 7027200
    assert spec.budget()["label_episodes"] == 4896 and spec.budget()["evaluation_episodes"] == 448
    seen = set()
    for root in spec.ROOTS:
        labels = {s for r in spec.label_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        train = {s for r in spec.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        evaluation = set(spec.evaluation_seeds(root))
        old = {s for r in spec.source.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(spec.source.evaluation_seeds(root))
        assert not labels & train and not (labels|train) & evaluation
        assert not (labels|train|evaluation) & (seen|old)
        seen.update(labels|train|evaluation)
