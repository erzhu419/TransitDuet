# V76: frozen response-shape and extreme-label source diagnosis

Use V75's saved expanded44 original/structured predictors and cached train/now
consumed-validation labels. Reproduce the cached validation scores first.
Project each known seven-action response onto the native-anchored two-column
quadratic action basis, without fitting a deployable state model. Orthogonality
splits nonnative MSE exactly into response-shape and state-to-curve error.
Inspect actual selected actions under both components, and report projected
versus unrestricted mean32-label choices as optimistic diagnostics only.

For the frozen primary ridge, solve source contributions with its saved scaler,
training weights and ridge: per training day and fixed top 1/5/10 percent of
contexts by squared label mass. Require day contributions to reconstruct its
cached coefficients and selected predictions. This is linear label attribution,
not deleting a day and refitting; large contribution does not prove bad data.
Report longer/shorter label distributions without changing the passenger-time
objective. Keep individual contexts server-side; pull only summary JSON.

Run tests and all diagnosis through scheduler on node001-node006. Actor, dual,
main and admission gates remain frozen; no candidate fitting or simulation.
Next choose a structural repair only after separating shape failure from curve
estimation and extreme-label influence. No clipping or gate tuning from this audit.
