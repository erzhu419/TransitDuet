# Conclusions

FreqDuet implements a causal frequency-to-authority interface for asynchronous
bus control: historical and completed-bin demand information defines a smooth
state for executable terminal headway planning, while station-local innovations
and same-time APC/AVL context inform discrete intermediate-stop holding. The
frozen controller met its independent short-training headway-regularity gate
without violating the registered passenger-journey no-harm condition. In the
separate long-training test, passenger journey improved relative to the Protocol
V6 reference, but the registered regularity magnitude was not confirmed.

The external comparison clarifies the practical meaning of these results.
FreqDuet was more regular than fixed headway and improved passenger journey over
the weaker rule-holding and rule-MPC baselines, yet fixed headway remained better
for passenger journey, holding burden, and fleet readiness. The study therefore
supports a registered gate-positive short-horizon result for the complete
selected controller. Because the Holm-adjusted sign-flip result was $p=0.125$,
it does not establish a familywise-significant effect, long-run robustness, or
universal superiority. A fresh factorial confirmation and matched route-day
evaluation are required to isolate frequency allocation and determine whether
its regularity benefit can be converted into a passenger-service benefit.
