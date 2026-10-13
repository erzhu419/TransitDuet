# Stage208: nonlinear causal state x plan wait head

Freeze Stage206 GLMs,costs,600/240 states,47 folds and the menu. Two zero-
output16-tanh residuals use paired198 inputs,or add12 absolute causal states.
Same symmetric mass loss,count8/1,train-only scales;600 AdamW steps,seed208,
lr0.003,decay0.01,float64,fixed final checkpoint/threshold0.5. Common weights
match initially;added columns start zero. No root/regime/future inputs.

Run: `native_transit_nonlinear_wait_stage208_development_20261013_r1`.
47 fits/two new heads each,one report;94 models/56400 wait optimizer steps.
1CPU/3072MB unbound node001-node006,no new cost fitting or native episodes.
Keep raw artifacts remote. Report training mean8 boundary/excess loss against
the frozen symmetric GLM,mean8 root/regime performance and both future halves.
Primary:state-conditioned head;accept only improved training fit and complete
cost/wait development gate,retaining all negative and baseline-relative results.

All47 fits`t148097-t148143` and report`t148148` completed with exit0;
source`6b2cd642df`;8 tests passed. One560592-byte report pull,no raw/models.
Both new heads pass the training-fit gate but fail cost and wait;no promotion.
Primary scene training boundary error40.106% ->18.367%,held41.304% ->43.227%.
Primary scene excess loss:train0.026474,held0.513596 nats (GLM held0.280190).
Regime primary wait is negative for both roots,but scene/root397 cost is
+0.001558033 and cross-lower/root397 wait +0.000451042 min. Old failing
scopes improve while new scopes fail;neither future-half partition closes.

Next: grouped inner-scene checkpoint validation/early stopping,not more
capacity or test-time margins. Refit the inner GLM on inner-training groups
with its own scales;the outer GLM has seen inner-validation labels and cannot
serve as a leakage-free inner initializer. Choose steps using inner data,
then refit on outer training while preserving outer costs and held-out gate.
Limits: new capacities3201/3393 and optimizer/regularization differ from199-
parameter GLMs;not a matched-capacity isolation. Same40 prefixes and noisy8-
draw labels remain supervised development,not independent CI or full HRL.
