# Independent review: relative response loss and posterior grounding

Scope: mathematical review of the focused followup to the failed first response-query gate. No main-code changes, training, accelerator use, or new-method claim. This reviews the current rank-normalized implementation and the proposed second route: half posterior-grounded query loss plus half causal-forecast query loss. The accompanying pure-Python checks are constructive examples, not task evidence.

## 1. What the first gate establishes

The action-annihilator's held-out projected MSE essentially equals the zero projected-prediction target energy: approximately 0.0008805 on TwoRoom and 0.003409 on Cube. Initial-state/action interventions have negligible effect on that auxiliary output. Its low absolute loss therefore does not establish learned response prediction. Shared initial-image grounding has much greater error, approximately 0.149/0.214 on the two TwoRoom seeds. This motivates a scale diagnostic; it does not establish that scale is the only obstruction. See [the first-gate report](../results/response600/RESULTS.md).

The proposed followup changes three separable things: target-relative weighting, temporal operator, and where the response loss enters the encoder. It should remain a bounded mechanism experiment, with no normalization variant counted as an additional research idea.

## 2. Relative normalization changes weighting, not per-clip stationary points

For one clip let Y and Yhat be T-by-D target/prediction matrices, P an orthogonal temporal projector of rank r>0, and delta=1/127.5. The implemented reduction is rank-normalized, rather than divided by T:

    e = ||PY||_F^2 / (r D)
    L = ||P(Yhat-Y)||_F^2 / (r D)
    Q = L / (e + delta^2).

The target and denominator are detached. Consequently every parameter gradient and Hessian of this clip's Q equals the corresponding derivative of L times the positive scalar 1/(e+delta^2). Exact gradient zeros, cancellation, and per-clip minimizers are unchanged. Summing differently weighted clips changes the overall optimum and descent direction; adding the unchanged anchor/native objectives also changes their relative influence. Thus this is more than a single global learning-rate change, but it is not a structural escape from collapse.

At zero projected prediction, Q=e/(e+delta^2), not exactly one. Substituting the reported *mean* energies gives 0.93470 and 0.98228, and gradient multipliers approximately 1061.5 and 288.1. These are scale illustrations, not the mean of the actual per-clip ratios; that mean must be measured directly.

The raw output-gradient norm at zero projected prediction is

    ||grad_Yhat Q||_F = 2 sqrt(e) / [sqrt(r D) (e+delta^2)].

It peaks at e=delta^2. Equalizing relative loss therefore does not equalize gradients. Tiny but nonzero residuals near the floor can have particularly large gradients. Same-rank action/random operators control the explicit r factor; identity has r=T and retains image background. An adaptive optimizer can also cancel a uniform gradient rescaling through its moment normalization, so a larger gradient norm alone does not imply proportionally larger useful parameter updates. Here nonuniform clip weights and the unchanged competing losses are material.

## 3. Exact initialization obstruction in the forecast-only route

The current carry predictor has zero output-projection weight and bias at initialization. Hence its rollout is z_t=z_0 at every time. The query head is affine, Yhat_t=W z_t+b, so its predictions are temporally constant. Both action and random projectors satisfy P1=0 and P time=0.

Let R=P(Yhat-Y). Since 1^T R=0:

* The projected query's gradient with respect to initial state z_0 is zero: W^T sum_t R_t=0.
* The query-head weight and bias gradients are zero: the common state and bias are multiplied by that same residual sum.
* With zero predictor output weights, action-embedding and internal predictor gradients from this query are also blocked by that zero matrix.
* A predictor output bias would produce z_t=z_0+t b_pred to first order. Its query gradient is zero because time^T R=0.

Output-projection **weights** can nevertheless receive a gradient. Their first-order effect uses time-varying hidden features, including recorded actions; these need not lie in P's nullspace. Later updates can create nonconstant state Jacobians, allowing an encoder gradient to appear. This route can learn; it just cannot obtain a direct response-query encoder gradient on its first step through the common carried initial state. Relative normalization leaves these exact cancellations intact. Floating-point projector residuals may yield tiny nonzero values and should not be mistaken for substantive signal.

The anchor term and native flow/rate still supply other gradients. A test showing nonzero gradients for the *sum* of these terms does not test the claim above.

## 4. What posterior grounding can and cannot change

Define Q_post using Yhat_t=W E_theta(x_t)+b, and Q_fore using the causal rollout from E_theta(x_0). Use

    Q_mix = 0.5 Q_post + 0.5 Q_fore,

with the identical projector, target, denominator and shared head, and keep the initial-image anchor term unchanged. Future frames are training targets/encoder inputs for Q_post only; they must never enter the forecast rollout. No optimal-action label is involved.

For posterior grounding, the encoder gradient contains

    sum_t J_theta E(x_t)^T W^T R_t.

P1=0 cancels this expression only if the relevant frame Jacobians are identical or otherwise lie in the projector's nullspace. Equal current state **values** do not imply equal parameter Jacobians. Constructively, E_theta(x_t)=z_bar+theta*x_t has identical states at theta=0 but different Jacobians x_t. If P x is aligned with P Y and W is nonzero, the posterior query has a nonzero theta gradient while the forecast-only carried-state query has zero gradient. The executable example gives -1.39948 versus zero.

This does not prove noncollapse. If W=0 and all encoded states are constant, both the encoder and head gradients of the projected query can vanish. They can also vanish when every available frame-dependent Jacobian is affine in time/controls, hence annihilated by P, or is orthogonal to the target residual. A learned posterior reconstruction may improve while the causal predictor or goal geometry remains poor. The unchanged random nonzero head avoids one particular zero-head initialization obstruction; it is not a universal guarantee.

Mix **losses**, not predictions. The exact identity is

    0.5 Q_post + 0.5 Q_fore
      = Q((Yhat_post+Yhat_fore)/2)
        + ||P(Yhat_post-Yhat_fore)||_F^2 / [4 r D (e+delta^2)].

A loss on averaged predictions omits the positive disagreement term and permits opposite errors to cancel. The chosen mixture also halves the forecast query's coefficient relative to the forecast-only arm. An improvement therefore establishes the usefulness of this allocation/gradient route as a package, not a clean causal attribution to one derivative alone.

## 5. Target-dependent weighting changes the statistical estimand

Even with a detached denominator, weights depend on the realized target. Under ambiguous inputs or stochastic outcomes, weighted squared regression estimates

    f*(input) = E[w(Y)Y | input] / E[w(Y) | input],

not E[Y|input]. Scalar counterexample: the same input has Y=1 or 2 with equal probability, and w=1/(Y^2+delta^2). The unweighted optimum is 1.5; the relative optimum is 1.200007. The check explicitly verifies its lower weighted risk. Symmetric zero-mean targets with symmetric weights can be unbiased, and a realizable deterministic target permits zero error under both losses; bias is not automatic in every setting.

Target energy is not a measured uncertainty or noise variance. Relative weighting may downweight informative large contact responses and amplify small unpredictable residuals. It can also help avoid spending nearly all auxiliary effort on high-energy but easy clips. Which effect occurs is empirical. Report raw and relative held-out errors, zero-prediction baselines, energy distributions and the fraction below the fixed floor, rather than interpreting relative error as a proper predictive likelihood.

## 6. The floor is a fixed image-unit convention, not universal noise variance

For uint8 rendered RGB mapped to [-1,1], delta=2/255 is exactly one original intensity step. Thus delta^2=6.15148e-5 is a clear task-independent reference in these input units. The implementation's comment appropriately says it is **not** estimated noise variance after pooling.

The target averages each 8-by-8 block of a 64-by-64 frame. Under the illustrative assumptions of independent, uniform rounding errors, source-pixel error variance would be delta^2/12, and the 64-pixel average would have variance delta^2/(12*64). Orthogonal temporal projection with rank-normalized energy preserves this expected per-retained-dimension variance. The proposed floor is 768 times that hypothetical variance. Real rendered quantization errors need not be independent or uniform, and other stochasticity can be much larger. This calculation is a counterexample to any universal noise-variance interpretation, not a recommendation to retune the floor.

Spatial averaging, observation scaling, correlated noise or different target features change the appropriate measurement model. The floor is therefore a declared fixed numerical/measurement-resolution convention for this specific RGB target. It is not invariant to arbitrary feature rescaling unless the floor is transformed as well.

## 7. What the random operator controls

The inspected implementation draws one private-seed orthonormal basis within the complement of constants/time, then retains its first r directions. This gives a projector of exactly the action projector's rank, leaves the outer RNG untouched, and depends on actions only through r. Rank-zero clips contribute no query in either route. Same rank also matches expected energy of isotropic temporal noise when divided by rD.

It does not match every frequency, boundary, or temporal-smoothness property of the action projector. The fixed orientation is shared across clips/training seeds, so a difference against this one random control does not average over all possible random subspaces. Rank itself can reveal action complexity. It is a useful control for action alignment versus generic equal-rank temporal contrast, with these explicit limits.

## 8. Decisive checks for this bounded gate

Before training, inspect Q alone, excluding the shared anchor/native losses. Verify forecast initialization cancellation separately from posterior-grounded gradients; distinguish encoder/pool, query head, predictor output weight/bias, and predictor internals. Do not require the exact toy posterior result on every arbitrary real batch: nonzero alignment is data-dependent. Preserve the full loss/gradient/RNG default regression and equal initialization/compute checks.

At the held-out endpoint, require genuine reduction below each arm's zero-prediction query baseline, together with causal-forecast degradation when initial state or actions are replaced while **holding target/operator fixed**. A posterior-only improvement is not a better learned transition. Report native planning separately. Use both weighted objective improvement and unweighted error/energy to expose success driven only by low-energy clips.

A favorable action-null result against both identity and equal-rank random contrast would support a useful role for the action-aligned target in this budget. Grounding better than forecast-only would support the gradient-entry hypothesis, with the allocation caveat above. No advantage, or posterior-only fit without forecast/state use, would falsify the intended short-budget channel-learning mechanism. None of these outcomes alone establishes mature superiority over LeWM or a new general theorem.

Reproduction: `python3 experiments/ideas10_20261030/theory/check_relative_response.py`. Output: [exact checks](ideas10-relative-response-checks-20261031.json). All checks pass; no GPU or task training was used.
