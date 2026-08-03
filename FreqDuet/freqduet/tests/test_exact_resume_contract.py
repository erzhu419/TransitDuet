import tempfile
import unittest

import numpy as np

from runner_v3 import TransitDuetV2Runner, load_config


class ExactResumeContractTest(unittest.TestCase):
    def _config(self, logs_dir):
        config = load_config(
            "configs_freqduet/F_freqduet_protocol_v4_main_hiro.yaml")
        config["seed"] = 8123
        config["env"]["effective_trip_num"] = 2
        config["logging"] = {"logs_dir": logs_dir}
        return config

    def test_exact_resume_restores_replay_optimizers_and_sampling_streams(self):
        with tempfile.TemporaryDirectory(prefix="freqduet-exact-resume-") as tmp:
            original = TransitDuetV2Runner(self._config(tmp), device="cpu")
            lower_state = np.zeros(original.lower_state_dim, dtype=np.float32)
            upper_state = np.zeros(original.upper_state_dim, dtype=np.float32)
            for idx in range(8):
                original.replay_buffer.push(
                    lower_state + idx / 10.0,
                    float(idx % 2) * 5.0,
                    -0.1,
                    0.2,
                    lower_state,
                    True,
                    idx,
                )
                original.upper_trainer.replay_buffer.push(
                    upper_state + idx / 10.0,
                    np.zeros(original.upper_action_dim, dtype=np.float32),
                    -0.2,
                    upper_state,
                    True,
                )
            original.lower_trainer.update(
                original.replay_buffer, 8, reward_scale=1.0)
            original.upper_trainer.update(8)
            original._save_checkpoint(3)

            expected_lower_action = original.lower_trainer.policy_net.get_action(
                lower_state, deterministic=False)
            expected_upper_action = original.upper_trainer.policy_net.get_action(
                upper_state, deterministic=False)
            expected_lower_batch = original.replay_buffer.sample(4)[-1]
            expected_upper_batch = original.upper_trainer.replay_buffer.sample(4)[1]

            restored = TransitDuetV2Runner(self._config(tmp), device="cpu")
            self.assertEqual(restored.maybe_resume(), 4)
            self.assertEqual(len(restored.replay_buffer), 8)
            self.assertEqual(len(restored.upper_trainer.replay_buffer), 8)
            np.testing.assert_allclose(
                restored.lower_trainer.policy_net.get_action(
                    lower_state, deterministic=False),
                expected_lower_action,
            )
            np.testing.assert_allclose(
                restored.upper_trainer.policy_net.get_action(
                    upper_state, deterministic=False),
                expected_upper_action,
            )
            np.testing.assert_array_equal(
                restored.replay_buffer.sample(4)[-1], expected_lower_batch)
            np.testing.assert_array_equal(
                restored.upper_trainer.replay_buffer.sample(4)[1],
                expected_upper_batch,
            )


if __name__ == "__main__":
    unittest.main()
