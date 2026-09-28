# Stage-34: full-episode frozen-response deployment

Stage-33 qualified local 150-step response utility. Freeze its original eight
controllers, three motion heads and seven linear response heads. No retraining,
refitting, kernel correction, threshold changes, exclusions or seed extensions.
The executable specification allocates 32 unused paths per optimizer root;
separate infrastructure preflight uses root 310001 and two unused paths.

Each 1200-step episode runs its own closed loop. Warm up for 100 steps with the
original balanced-jitter schedule. At steps 100, 250, 400, 550, 700, 850 and 1000,
positive predicted settled response selects renewal now; otherwise renew at
check+100 using a fresh policy call at the then-current state. No other upper
action occurs until the 150-step block ends. Lower feedback runs every step.
The final 50-step tail resumes the fixed schedule. All seven response views,
always-keep and always-renew use the same executed-plan budget. Fixed50 is the
original-frequency reference, outside the primary joint gate.

Candidate previews are actual upper-policy calls. Immediate renewal reuses its
preview; keep discards it and pays another call at check+100. Constants do not
evaluate unused previews. Record all calls, executed plans, discarded previews,
upper/lower/response inference time and wall time. No dummy calls equalize costs;
timing is descriptive, and fewer executed plans alone is not compute savings.

Primary outcomes: complete-episode ISE reduction and dense-return increase
against the original eight controls (16 endpoints). Equal-weight optimizer roots
are the statistical unit. Freeze 65536 paired root-bootstrap draws, seed
(34,34039), nominal two-sided percentile Bonferroni familywise alpha .05.
Joint deployment success requires all 16 adjusted lower bounds >0. Report the
fixed50 comparison and actual cost separately. This is fresh-path transfer
conditional on qualified models, not a new independent optimizer-root test.

Full budget: 3081600 primitive steps; preflight: 6300, accounted separately.
Scheduler dynamically uses node001-node006, 17 CPUs/16 GiB per full root,
two CPUs/3 GiB for preflight. Raw trajectories and weights remain remote.
