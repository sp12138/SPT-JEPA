# Candidate F: learn dynamics coordinates without deleting the measured carrier

2026-10-31. A concrete representation-learning hypothesis, not a benchmark result or a claim to have invented invertible networks. This follows the observation that TwoRoom position is already strongly readable in initialized spatialRP64 features and is poorly exposed by the learned QueryPool. The question is whether learning can improve coordinates while being forbidden to destroy that existing information.

## Update rule

Fix the already defined 64-dimensional spatial random-projection readout of a frozen initialized image encoder, `u(x)`. Use exactly the same initialization, spatial readout and training-only fixed normalization for every control. This uses no pretrained external semantic model and no simulator labels. Replace freely learned state coordinates by

\[
z(x)=T_\theta(u(x)),\qquad T_\theta(u)=u+g_\theta(u),\qquad
\operatorname{Lip}(g_\theta)\le c<1.
\]

Train T and the ordinary action-conditioned predictor jointly. A basic sample-level latent objective is

\[
L_{\rm pred}=\sum_h\|F_\phi^{(h)}(T_\theta(u_t),a_{t:t+h-1})
                         -T_\theta(u_{t+h})\|^2.
\]

For the first AS-JEPA comparison, retain its existing forecast-render/rate objective and replace only the state construction; no new semantic head or policy target is required. Both branches of a target must be specified consistently with the native implementation rather than silently adding EMA/stop-gradient. The predictor remains nonlinear; the proposal does not require Koopman-linear dynamics. Use a bounded residual network for g with an actual operator-norm bound, not an unchecked few-step spectral estimate advertised as a certificate. A small residual MLP is enough for a first gate; exact singular values of its small matrices are feasible. A single globally chosen c, for example .5, defines the permitted distortion and must not be tuned separately per task.

This deliberately restricts *what representation learning may erase*. Compression occurs in the fixed measurement u. Subsequent world learning may reorganize its coordinates to make controlled prediction easier, but cannot remove a difficult factor merely to lower the predictive loss. It is a hypothesis about an information-preserving parameterization, not a generic spread penalty or a change of posterior-noise geometry.

## What is guaranteed, and what is not

By the triangle and reverse-triangle inequalities,

\[
(1-c)\|u-v\|\le\|T(u)-T(v)\|\le(1+c)\|u-v\|.
\]

The lower bound gives injectivity. For any z, the map `u -> z−g(u)` is a contraction, so Banach's fixed-point theorem gives a unique inverse on Euclidean space. Thus every function recoverable from u is recoverable from T(u), using the inverse; the generated sigma-algebras are the same. This does not guarantee equally good **linear** probes. Existing i-ResNet theory already establishes this construction.

With the original isotropic Gaussian state noise of scale σ, distinguishability of two code distributions obeys

\[
\operatorname{TV}\{N(T(u),\sigma^2I),N(T(v),\sigma^2I)\}
 =2\Phi(\|T(u)-T(v)\|/(2\sigma))-1
 \ge2\Phi((1-c)\|u-v\|/(2\sigma))-1.
\]

Hence learning cannot make a carrier distinction arbitrarily smaller than the noise floor. This is not a claim that noisy mutual information or the original rate is invariant under T. An arbitrary invertible adapter can shrink one direction toward zero; the quantitative lower bound is essential.

The guarantee is only relative to u. Information missing from the frozen carrier cannot be created. Retaining nuisance variation can consume capacity or make prediction harder. A faithful coordinate map does not specify object-only goal semantics. At sufficiently large predictor capacity, a frozen-u predictor can absorb `T^{-1} F T`, so there is no universal expressive advantage over all frozen-feature models; the proposed benefit is a finite-budget inductive bias and a prevention of destructive representation updates.

## Constructive separation from the nearest alternatives

Let the true state be s=(s1,s2), with bounded s1, and let the carrier be a sheared observation

\[
u=(s_1,s_2+\alpha h(s_1)),\qquad
s'_1=.7s_1+.25a,\qquad s'_2=.85s_2+.3s_1,
\]

where α=.1 and h(r)=r² for |r|≤1, continued as `2|r|−1` outside. Then h is globally 2-Lipschitz. The chart `T(u)=(u1,u2−αh(u1))` has residual Lipschitz constant .2 and recovers s exactly.

- **Versus a frozen carrier:** a linear predictor of u cannot generally represent the quadratic and state-action cross terms in u'. The bounded chart admits an exact linear predictor. This is a constructive finite-capacity difference; a sufficiently expressive nonlinear frozen-u predictor removes it.
- **Versus unconstrained JEPA prediction:** the lossy encoding `E(u)=u1` also has exact latent prediction and discards s2. The proposed parameterization cannot realize that encoding. The distinction is not merely achieving nonzero variance.
- **Versus offline semantic alignment:** a semantic identity teacher u makes `||T(u)−u||²` minimal at T=identity, while the controlled prediction objective above favors the inverse shear. The training rules therefore need not choose the same chart even with identical adapter architecture.
- **Versus density-estimation flows:** this rule has no log-determinant likelihood objective. Its coordinate preference comes from the action-conditioned prediction task, while invertibility is a structural constraint.

[check_information_carrier.py](check_information_carrier.py) fits the scalar chart parameter and a linear predictor using 8,192 training and 8,192 independent test transitions. It recovers theta=.1000000003: frozen-carrier test MSE is .0001240, versus 1.30e−21 with the fitted chart. The lossy one-coordinate encoder also has zero prediction MSE but discards a coordinate with variance .3345. It also checks the global distance bounds numerically, exact inversion and the identity-teacher disagreement. [Full CPU output](ideas10-information-carrier-20261031.json). These synthetic checks establish separations, not performance on either benchmark.

## Why test it on both tasks

TwoRoom provides unusually direct motivation: the initialized carrier exposes agent x/y with R² around .94/.94 and .93/.89, while the learned QueryPool remains around .14/.25 and .11/.29 at update 2000. The lower-distortion constraint would prevent subsequent training from eliminating distinctions already present in that carrier. It does not guarantee that the DiT will read them or that the predictor learns their action effects.

Cube is a harder test of the hypothesis, not an equally established opportunity. Initialized spatialRP64 has average block-position R² about .60/.62, below the mature learned representation. An invertible chart might make existing nonlinear information easier to use and constrain destructive loss minimization, but cannot manufacture missing block/height information. Failure on Cube would show that a protected initial carrier is insufficient; the method must not be rescued by silently switching to privileged or unmatched pretrained features.

## Closest primary work and novelty boundary

1. [Invertible Residual Networks](https://proceedings.mlr.press/v97/behrmann19a.html) supplies the architecture, contraction inverse and bi-Lipschitz reasoning. None is claimed as new here.
2. [Semantic Latent Alignment with Invertible Bi-Lipschitz Adapters](https://arxiv.org/abs/2610.21332) is a particularly close architectural precedent: it keeps a pretrained encoder/decoder fixed, aligns an adapter to a semantic teacher offline, then freezes it during flow matching. The proposed rule instead jointly learns coordinates from recorded-action dynamics on an untrained but audited spatial carrier. This is a functional training difference, not a claim that reversible adaptation itself is novel.
3. [DINO-WM](https://arxiv.org/abs/2411.04983) provides the frozen-feature world-model reference. Its pretraining and patch-feature setting differ, and simply comparing against its published score would not isolate the proposed coordinate-learning constraint.

Random-feature world models also have established precedents, including [Locality Sensitive Sparse Encoding](https://arxiv.org/abs/2401.13034). Therefore the potential contribution is specifically the tested interaction between a preserved observational carrier, bounded learnable coordinates and action-conditioned prediction. A targeted search has not proven that this full combination is new; it is an admissible research hypothesis rather than an accepted novelty claim.

## A bounded falsification

Use the same frozen carrier and predictor in three short controls: no chart, an unconstrained residual chart, and the bounded chart with identical parameters and initialization. Preserve native data exposure, action inputs, noise and training objective. Record all trainable/frozen parameters and the adapter's compute; total parameter matching alone is inadequate when one backbone is frozen. Compare against the held native temporal-forecast and LeWM controls separately.

The minimum acceptable result is that the bounded chart retains noisy carrier distinctions while improving actual action-conditioned prediction over no chart, and improves planning over the unconstrained chart under the same evaluator. Merely preserving a clean probe is not sufficient. Reject the mechanism if the carrier omits required factors, if the no-chart predictor matches it at the same compute, or if it only improves geometry while physical outcome prediction is unchanged. This note launches no such training.
