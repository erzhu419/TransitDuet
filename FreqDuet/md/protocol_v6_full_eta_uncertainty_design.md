# Full-ETA uncertainty after V158

Design only. V151/main,V154 moments/weights,V146(3s,0s),3600s window and .1/.1
tails stay frozen. V151 remains reference;the eight-day candidate is unpromoted.

Chosen direction:state-conditional empirical joint residual blocks. Keep running,
service and post-ready errors from the same prediction together. Inseparable
current-ready residence stays joint;opaque fallback gets its own full-residual
stratum or abstention,not an invented split. Avoid independent component bands,
a dominant all-day bias or a constant global Gaussian covariance.

Use query-time phase,route/stop,current service/source and frozen predicted
horizon/route clocks as context. Retain day blocks and equal-day/episode weights;
daily-mean ranges or standard errors omit future within-day variation.
For e=prediction-truth,T=m-e:fixed-tail bounds would be
[m-Q_0.9(e|context),m-Q_0.1(e|context)]. No quantiles are estimated here.

V159 extraction/readback now reproduces V158 on all 768 archived paths,including
query time,same-follower target-ready/label-available time and observable context.
V160 completes read-only sidecars for follower-validity/provenance,
joint-current-residence conditional support and separate shadow service support.
Late policy-time load is unavailable as query-time context and stays excluded.
Independent calibration/confirmation cohorts of 32 new days each are reserved,
not collected. Next:seal the interval estimator and final inference gates before
fresh collection;day reservation alone is not a full interval-analysis protocol.
Online updates require labels strictly completed before the query;keep signed
asymmetry and unsupported states,without changing the point or filling zeros.
An eventual fixed interval protocol needs calibration/confirmation days disjoint
from prior fitting and one another. Consumed V145/V151 days are development,
not fresh confirmation. V146 stage coverage does not certify full-ETA coverage.

Limitations:within-cell still mixes stop/state/horizon and local day-by-context
shifts;moments do not identify conditional tails. Eight prior days mix latent
shifts and sampling noise. No interval fitted,policy changed,replay or new data.
