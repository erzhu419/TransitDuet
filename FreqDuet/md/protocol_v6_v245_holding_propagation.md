# V245: read-only V244 holding propagation

Read the six V244 stored paths only. Reproduce CV2 and actual holding endpoints,
then close event times and headway covariance using launch,direct adjustment,
native state response,and remaining elapsed time. Keep changed-neighbour support
and CV mean normalization explicit. Compare both supply/blind closed-loop pairs
and the two saved requests on each identical post-anchor state.

Local restoration labels freeze completed departure neighbours;they are geometry,
not replayed action values. No per-query independence/CI or causal promotion claim.
No V238/controller edits,tuning,new futures,replays,training or formal matrix.
Run synthetic accounting tests and analysis through scheduler high/manual on the
server. Detailed diagnostics remain there;return only a small summary.
