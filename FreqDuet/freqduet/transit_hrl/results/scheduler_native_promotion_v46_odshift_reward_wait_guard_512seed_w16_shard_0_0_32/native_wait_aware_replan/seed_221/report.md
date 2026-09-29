# Native Transit Shared-PPO Episode Loop

- status: supported_native_episode_loop
- episodes: 1
- shared core: `freq_hrl.rl.DualActorCriticPPO`
- upper contract: 24x4
- upper model action dim: 5
- lower contract: 43x1
- learned promotion gate: True threshold=0.2
- gate guard: strength>=0.3 age>=0.0 min_elapsed_s=0.0 cooldown_s=450.0 preselect_action=True plan_blend=0.0
- gate LF/HF guard: low_signal_min=0.0 max_hf_to_lf=0.0 max_replans=1 max_total_replans=0
- replan target-headway guard: max_s=349.0
- replan target-headway projection: enabled=True margin_s=0.25
- replan throughput/reward floor: throughput_min=0.05 floor_min=0.12 reward_floor=0.025 target_min_s=341.0
- adaptive drift penalty: gain=0.1 min_scale=0.78
- replan final-delta guard: min_s=0.04 max_s=1.1
- promotion replan policy: wait_aware wait_gain_s=4.0 max_shift_s=1.05
- lower HF wait action prior: gain_s=45.0 offset=11 context_dim=0 min_scale=0.0 max_scale=1.0
- lower HF boarding rescue: gain_s=0.0 max_s=0.0 queue_min=0.0 load_max=0.0
- adaptive lower drift penalty: gain=0.0 min_scale=0.25
- off-policy replay updates per native batch: 1
- mean wait: 62.3900
- mean headway CV: 0.3713
- mean shared-PPO score: -63.1326
- mean gate value: 0.0000
- mean wait-aware replan pressure: 0.0000
- mean adaptive drift scale: 1.0000
- mean throughput proxy score: 0.0000
- mean throughput floor delta fraction: 1.0000
- mean reward-floor score: 0.0000
- mean value-guard score: 0.0000
- mean value-guard selected scale: 0.0000
- mean adaptive lower drift scale: 1.0000
- mean lower prior scale: 1.0000
- mean lower boarding rescue: 0.0000s
- mean wait-pressure override count: 0.0000
- mean wait-aware replan shift: 0.0000s
- mean learned replan base delta: 0.0000s
- mean learned replan final delta: 0.0000s
- native boarded pax: 26309.0
- native alighted pax: 26309.0
- native onboard load: avg=0.6520, peak=1.0000

| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62.3900 | 0.3713 | 15926.2940 | 26309 | 26309 | 0.6520 | 4970 | 66 | 0 | 5232 | 13967.4460 |
