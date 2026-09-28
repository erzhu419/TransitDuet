# Stage-33: independent optimizer-root response qualification

Stage-32 stopped the kernel correction. This experiment freezes the original
Stage-31 linear forecast-conditioned response, without carrying over controller
weights, motion fits, response fits, or paths from the two development roots.

Full optimizer roots: 310011, 310023, 310037, 310049, 310061, 310073, 310089,
310101. Separate infrastructure preflight: 310001. No root extension or filtering.
The executable specification freezes all disjoint episode roles and budgets.

Each controller is trained from scratch using the unchanged Stage-12 PPO core:
384 iterations, eight rollout roots, horizon 1200, balanced jitter, MLP-128,
learning rate 0.0003, 64-step history and 50-step nominal upper period.
Eight selection paths rank checkpoints every eight iterations by negative ISE,
then dense return. Twenty-four diagnostic paths never select checkpoints.
Each root costs 4,243,200 controller-training/selection/diagnostic steps.

Sixteen independent motion-fit tapes and eight motion-evaluation tapes use the
unchanged causal forecast features and unit ridge. Motion gates are diagnostic;
failing roots remain in the complete roster. Sixteen response-fit paths use the
original twenty opportunities/path, excluding check <64 before replay. Seven
linear response heads are saved before collecting 120 queries on eight fresh
paths. Keep holds 100 steps; both arms execute one upper call before the 150-step
settlement; lower feedback remains continuous. All views share causal proposals.

Primary endpoints are eight settled ISE benefits and seven settled-rate MSE
reductions versus the original controls. Equally weighted optimizer roots are
the statistical unit: 65,536 paired root bootstrap draws, fixed seed (33,33039),
nominal two-sided percentile intervals with Bonferroni familywise alpha 0.05 over all
15 endpoints. Joint qualification requires every lower bound >0. Individual
point gates are descriptive. No tuning, outcome-based exclusion or added roots.

Scheduler allocates one CPU/3 GiB per training root, then 17 CPUs/24 GiB per
16-worker qualification root, dynamically across node001-node006. Only compact
JSON is pulled; checkpoints and raw arrays remain on the server. This qualifies
local response credit, not an episode-deployed trigger or planning-cost savings.
