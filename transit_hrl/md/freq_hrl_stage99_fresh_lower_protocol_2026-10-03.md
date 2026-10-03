# Stage99: Fresh Matched Lower Donors

Stage98 confirmed four joint-call endpoints on new teachers. Rebuild the four Stage93 lower controls with these sources before testing staged upper learning.

- Sources: full Stage96 L0/U0 clones, full Stage97 fixed decoders/forecasters and final update8 Stage98 joint-call UJ. Use all eight roots, including root410011's full sources for preflight. No old joint/lower weights, re-cloning or decoder adjustment.
- Four learners start from the same L0: U0 or UJ fixed, each with independent upper/lower noise or common upper innovations and independent lower noise. Replay affects only the second training replica's upper innovations, not its actions; both streams remain independent in final evaluation.
- Keep the Stage93 MC credit, all64 paths per full update, eight updates, per-learner lower KL 0.00099 at50 / 0.000995 at100, fixed std/values/upper/forecaster and no Adam/critic fits or intermediate evaluation.
- Preflight: two updates x eight credit paths per learner/period and four final eval paths per each of12 compositions, H300; 224 episodes / 67,200 steps; 16 mean updates / 144 upper replay forwards; no checkpoint writes.
- Full: 4,864 episodes / 5,836,800 steps and eight fixed-final checkpoints per root; cohort 38,912 episodes / 46,694,400 steps, 64 server-only checkpoints and 147,456 replay forwards. Count replay as extra compute.
- Fresh Stage99 rosters; unchanged Stage80 noise mapping and Stage93 26-contrast equal-root Bonferroni family, 65,536 draws and bootstrap seed. Four primary common-versus-independent contrasts must all have positive lower CI bounds for a global conditioning claim.
- Keep all final lower donors even if conditioning fails. The subsequent staged-upper test remains separate, with zero/U0/fixed-UJ/matched-independent controls and no Stage94/95 pooling. Frequency-superiority and Stage67 critic-credit HOLD are unchanged.
- Scheduler: dynamic node001-006; 3 CPU/3GB preflight, 9 CPU/8GB full. Pull compact JSON/completion only. Full launch follows native preflight mechanical qualification, not a reward admission rule.

Scope: fixed-upper lower MC mean learning on the new teacher cohort, not full actor-critic or unseen-task generalization.
