# W Coverage And Complexity

Freeze a development 2x2 factorial on the four existing task/root groups:
8 prefixes x 4 fitting futures/state versus 32 prefixes x 1 fitting future/state;
each at slope effective DF 2 and 8. Three states/prefix give 96 fitting
state-future pairs/arm. All arms retain the 39 causal features, original history
continuation and 0.75 training-path-OOF threshold quantile. No holdout selection.

Reuse the server's previous training bank. Dense uses original A0:4; wide uses
A0:1 on the same eight prefixes plus 24 new prefixes, root+8000 through+8023.
Independent B uses four futures/state. New held paths are root+7000 through+7007;
neither revealed root+5000 nor root+6000 evaluation paths are fitted or evaluated.
New seed roles exclude all registered roles and resolved primary training seeds.

Four native prechecks reproduce the old DF8 W head, original history episodes,
paired prefixes and factual replay, touching only a new TRAINING prefix. After
all pass, main jobs acquire new training data, fit and save all four rules before
independent-B readout and held-prefix access. Newly sampled banks are saved per
prefix on the server. Only W/G branch credit is acquired; no unnecessary L branch
or controller/reference training is repeated.

Report training-A, independent-B and path-OOF-B W errors; held A/B W and G errors;
and complete-episode ISE with original history, L, threshold and termination.
Show both episode halves, within-root path CIs, coverage effects at each DF,
complexity effects at each coverage, and their paired interaction. No automatic
winner or promotion: parent HOLD and failed termination eligibility remain.

Limits: fitting-future counts match, not total CPU or B-label count. Dense and
wide are nested training populations, so coverage and reduced future averaging
are traded together. Marginal acquisition costs and reused supervision remain
explicit; old unknown lifecycle costs are not repaired. Two roots/task provide
development evidence, not confirmatory cross-root inference. No new feature set,
neural model, threshold search or change of HRL applicability scope is introduced.

Execution: four one-CPU/2-GiB prechecks, then four main jobs; dynamic scheduler
placement on node001-node006, no Slurm or login-node installs. Full caches and
models stay server-side; only compact public outputs are retrieved.
