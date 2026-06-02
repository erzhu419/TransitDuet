# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 24x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.92
- gate guard: strength>=0.95 age>=1.0 min_elapsed_s=900.0 cooldown_s=900.0 preselect_action=True plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.1 max_hf_to_lf=8.0 max_replans=1 max_total_replans=0
- promotion replan policy: learned_wait_aware wait_gain_s=8.0 max_shift_s=4.0
- lower HF wait action prior: gain_s=45.0 offset=11
- off-policy replay updates per native batch: 1
- mean wait: 34.9770
- mean headway CV: 0.5507
- mean shared-PPO score: -36.0784
- mean gate value: 0.9820
- mean wait-aware replan pressure: 0.4375
- mean wait-aware replan shift: -1.9997s
- mean learned replan base delta: 3.5888s
- mean learned replan final delta: 2.5890s
- native boarded pax: 25740.0
- native alighted pax: 25740.0
- native onboard load: avg=0.6629, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34.9770 | 0.5507 | -17962.5900 | 25740 | 25740 | 0.6629 | 4970 | 66 | 1 | 5232 | 14957.3525 |
