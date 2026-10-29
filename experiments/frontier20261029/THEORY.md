# Sample-level objectives: mechanisms and falsifiable predictions

All candidates use the same 51,097,134 trainable parameters. A minibatch only averages independent video-window losses. No batch moments, cross-sample negatives, BatchNorm, target network or stopped target gradient enters these L2 objectives. The current screen is 600-update adaptation of paired existing two-epoch models, not from-scratch evidence or a novelty claim.

## Identity-carry rate and posterior noise

Write the prior as F(s,a)=s+r(s,a), s=mu+epsilon, epsilon~N(0,sigma^2 I), and Delta=mu_next-mu. Its variable transition rate is E||Delta-epsilon-r(mu+epsilon,a)||^2/(2 sigma^2). Under local linearization r(mu+epsilon)=r(mu)+J_r epsilon, this is

    ||Delta-r(mu)||^2/(2 sigma^2) + 0.5 ||I+J_r||_F^2.

This derivation conditions on a fixed history/action and deterministic prior realization. Nonlinear curvature and attention to earlier states add terms; it is a local mechanism, not an exact formula for the full stochastic transformer.

* The original rate prices posterior-noise sensitivity of the entire carried state. Identity dynamics alone produce positive variable rate even for constant encoded means; this explains the independently documented active-dimension counterexample.
* clean-prior removes noisy prior inputs and changes this objective.
* innovation-corrected subtracts the known carried epsilon before pricing the residual. In the local single-state expression its sensitivity term becomes 0.5||J_r||_F^2. This is a surrogate and does not preserve the original KL.
* marginal-prior averages F(mu+epsilon) and F(mu-epsilon) before squaring. First-order perturbations cancel; curvature remains and Jensen changes the expected objective.
* antithetic-rate averages the two squared errors instead. Its expectation matches the original rate under symmetric noise and matching marginal predictor randomness, but lower estimator variance is a hypothesis. Separate dropout draws need not achieve full antithetic cancellation.
* overshoot changes the training horizon to expose accumulated dynamics error. This adapts known latent-overshooting principles; it is not claimed as a new invention.
* window-free-bits floors each sample/dimension's variable rate. It avoids cross-sample aggregation, differs from batch-expected free-bits, and may have little effect if few sample rates reach the floor.

## Motion objectives

With renderer residual e_t=y_t-epsilon-v_t, sharing epsilon and flow time inside one video gives

    e_t-e_(t-1) = (y_t-y_(t-1)) - (v_t-v_(t-1)).

The proposed differential-flow term penalizes this difference. It increases the price of temporal residual variation; it can also emphasize jitter. The shared-flow-noise control isolates the effect of changing the estimator from that of adding the difference penalty. Both are sample-level. Motion-weighted flow uses target changes and normalization only within the same training window. These targets do not enter inference. Weighted loss magnitudes and the resulting rate/reconstruction balance cannot be compared directly with unweighted raw training losses.

## Decision before more compute

Numerically finite execution and active_dims=64 alone do not establish useful state learning. Inspect posterior reconstruction, zero/permutation interventions, paired planning success, generated motion and action counterfactuals. Compare against the seed-matched original continuation, and compare differential flow additionally against shared noise. Preserve all diagnostic-seed failures. Consider cost per update as well as parameter count.

Continue a small number of supported mechanism families for another bounded stage. Short adaptation cannot establish stable from-scratch learning; promising results require untouched evaluation cases and reserved training seeds. No 89% or novelty claim is authorized by this screen alone.
