# Stage-36 Preflight

Implementation `4b7af62fd9`; tasks `t102161`-`t102165` complete five native
treatments. Five raw-trajectory audits and ten final/selected checkpoint
replays pass. Enabled actor/critic weights update; frozen ones are unchanged.
Twenty-seven focused tests pass, including equivalence to existing PPO.
NumPy minibatch shuffles are now explicitly seeded alongside Torch.

Method cost: 19500 steps, 308 upper calls, 448 gate calls, no previews.
Verification: 3000 steps, 36 upper calls, 72 gate calls. Only compact JSON
returns locally. This authorizes the frozen 40-cell experiment; preflight
performance is not a scientific result or a selection criterion.
