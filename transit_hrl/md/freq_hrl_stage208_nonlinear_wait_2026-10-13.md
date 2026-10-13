# Stage208: nonlinear causal state x plan wait head

Freeze Stage206 GLMs,cost heads,600/240 states,47 folds and the candidate menu.
Two zero-output16-tanh residual heads start exactly at the symmetric GLM:
paired198 inputs,or paired198 plus12 absolute causal conditions. New columns
start zero and common columns share initialization. Use unchanged symmetric
mass loss,draw counts8/1,old GLM scales plus training-only context scales;
600 AdamW steps,seed208,lr0.003,decay0.01,float64,fixed final checkpoint.
No root/regime/future inputs or test-time threshold tuning;share<=0.5.

Run: `native_transit_nonlinear_wait_stage208_development_20261013_r1`.
47 fits/two new heads each,one report;94 models/56400 wait optimizer steps.
1CPU/3072MB unbound node001-node006,no new cost fitting or native episodes.
Keep raw artifacts remote. Report training mean8 boundary/excess loss against
the frozen symmetric GLM,mean8 root/regime performance and both future halves.
Primary:state-conditioned head;accept only improved training fit and complete
cost/wait development gate,retaining all negative and baseline-relative results.

Results pending. Next decision follows training fit and held-out outcomes.
Limits: new capacities3201/3393 and optimizer/regularization differ from199-
parameter GLMs;not a matched-capacity isolation. Same40 prefixes and noisy8-
draw labels remain supervised development,not independent CI or full HRL.
