import copy
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch
from scripts import diagnose_pointmaze_joint_reference_credit_stage122 as probe
from test_pointmaze_joint_reference import TrackingDonor, source_data
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def test_phase_baseline_excludes_own_scenario_and_other_noise_fold():
    returns = np.arange(24, dtype=np.float64).reshape(6, 4)
    baseline = probe.phase_baseline(returns)
    np.testing.assert_array_equal(baseline[0], returns[[2, 4]].mean(0))
    changed = returns.copy()
    changed[0] += 100.
    np.testing.assert_array_equal(probe.phase_baseline(changed)[0], baseline[0])
    changed = returns.copy()
    changed[1::2] += 100.
    np.testing.assert_array_equal(probe.phase_baseline(changed)[::2], baseline[::2])


def test_phase_baseline_removes_deterministic_time_to_go():
    returns = np.tile(np.array([100., 75., 50., 25.]), (16, 1))
    np.testing.assert_array_equal(returns - probe.phase_baseline(returns), 0.)


def test_score_excludes_frozen_teacher_and_matches_direct_surrogate():
    torch.set_num_threads(1)
    teacher = OptionalActionResidual(TrackingDonor(), feedback_dim=392, advice_dim=4)
    actor = probe.experiment.ReferenceResidualActor(teacher)
    state = np.zeros((6, 402), dtype=np.float32)
    state[:, -1] = np.linspace(0., 1., 6)
    action = np.arange(12, dtype=np.float32).reshape(6, 2) / 12
    with torch.no_grad():
        old_logp = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))[0].numpy()
    batch = LevelTrajectoryBatch(state=state, action=action, old_logp=old_logp,
        reward=np.ones(6, dtype=np.float32), duration=np.ones(6, dtype=np.int64),
        done=np.array([0., 0., 0., 0., 0., 1.], dtype=np.float32), old_value=np.zeros(6, dtype=np.float32))
    signal = np.arange(6, dtype=np.float32)
    snapshot = copy.deepcopy(actor.state_dict())
    gradient, error = probe.loss_gradient(actor, batch, signal, clip_ratio=.2)
    logp, _ = actor.log_prob_entropy(torch.as_tensor(state), torch.as_tensor(action))
    loss = -(torch.exp(logp - torch.as_tensor(old_logp)) * torch.as_tensor(probe.experiment.FrequencySeparatedActorCriticPPO._normalize(signal))).mean()
    expected = torch.autograd.grad(loss, actor.readout.parameters())
    np.testing.assert_allclose(gradient, np.concatenate([g.numpy().reshape(-1) for g in expected]), atol=1e-8, rtol=0)
    assert error < 1e-6 and np.linalg.norm(gradient) > 0
    assert all(p.grad is None for p in actor.parameters())
    torch.testing.assert_close(actor.state_dict(), snapshot, atol=0, rtol=0)


def test_reduced_probe_keeps_teacher_and_optimizers_frozen(source_data, tmp_path):
    models, pred, cal, args, teachers = source_data
    roles = probe.spec.seed_roles(410011, preflight=False)
    roles["training_rounds"] = [roles["training_rounds"][0][:2]]
    with patch.object(probe.spec, "arguments", return_value=args), \
            patch.object(probe.spec, "seed_roles", return_value=roles), \
            patch.object(probe.experiment.source, "load_source", return_value=(models, pred, {}, cal)), \
            patch.object(probe.experiment.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]), \
            patch.object(probe, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(probe.experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(probe.experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        result = probe.run(410011, tmp_path / "result.json")
    assert result["native_episodes"] == 24 and result["native_steps"] == 2400
    assert result["optimizer_steps"] == result["checkpoint_writes"] == result["native_trace_writes"] == 0
    assert result["actor_critic_and_teacher_unchanged"] == "passed"
    assert result["groups"]["50"]["forecast"]["upper"] is None
    assert result["groups"]["100"]["joint"]["upper"]["decisions_per_episode"] == 1
    assert (tmp_path / "completion" / "ready.json").is_file()
