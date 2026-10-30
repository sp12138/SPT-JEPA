# Final spatial design pass: zero additional admitted methods

2026-10-31. Two concrete mechanisms were specified in `FINAL_DESIGN_FIRST.md` before the targeted primary-literature audit, then checked by `final_design_checks.py`. Both are rejected for admission as new methods. This is a constructive failure analysis, not a claim that no useful algorithm exists. No GPU, training, planner change or main-source edit was performed.

The latest actual gate motivates the question precisely: observed Cube query reconstruction and block-height readout improved under posterior grounding, but the corresponding causal query forecast remained approximately the zero-response baseline, and native planning did not improve. Better information in an observed code is not enough; that information must drive an accurate transition. The two attempted repairs below fail for different reasons.

## A. A transition that is forbidden to erase the initial code

The proposed operation was F_a(z)=z+g(z,a), with Lip_z(g)≤c<1 and all original objectives retained. It gives the useful mathematical inequality

\[
\|F_a(z)-F_a(z')\|\ge(1-c)\|z-z'\|.
\]

This prevents one transition from replacing every grounded initial state by a common code. However, it also forbids legitimate many-to-one transitions when the full predicted code must match the encoder of the next observation. A wall-clamped family of initial positions is a simple example.

For any random encoded initial state Z and fixed encoded post-contact target z_g, an m-co-Lipschitz transition satisfies

\[
\mathbb E\|F(Z)-z_g\|^2
\ge \operatorname{tr}\operatorname{Cov}(F(Z))
=\tfrac12\mathbb E\|F(Z)-F(Z')\|^2
\ge m^2\operatorname{tr}\operatorname{Cov}(Z),
\]

where Z' is an independent copy. This bound is nonzero whenever the initial code is informative about a direction that contact genuinely removes. An encoder can evade it by collapsing that direction beforehand, but that defeats the stated purpose. The bound concerns this fully observed/fully compared state contract; it is not a claim that microscopic physical dynamics must be noninvertible.

**Executed check.** With m=.5 and Z uniformly distributed over [0,1], the continuous bound is 0.02083333333. A 100,000-point midpoint approximation gives 0.02083333333125, attained by the centered linear predictor. The pairwise endpoint version gives a minimum mean squared error of 0.0625 for the two initial endpoints. A separate 10,000-pair numerical check verifies the lower-distance bound for a norm-bounded residual map.

Putting the erased information in extra latent coordinates and projecting it out of the goal metric can avoid the contradiction. That changes the state/goal observation contract, introduces a hidden-memory interpretation, and overlaps the already discussed observation-fibre proposal. A learned contact-dependent contraction gate is another mechanism, not an automatic consequence of the residual bound. Neither should be silently added to claim that this attempt succeeded.

**Primary audit.** The core operator and its invertibility/Lipschitz guarantee are already [Invertible Residual Networks](https://proceedings.mlr.press/v97/behrmann19a.html). [Augmented Neural ODEs](https://papers.neurips.cc/paper_files/paper/2019/hash/21be9a4bd4f81549a9d1d241981cec3c-Abstract.html) explicitly examines representational restrictions from topology preservation and the use of extra dimensions. [Approximation Capabilities of Neural ODEs and Invertible Residual Networks](https://proceedings.mlr.press/v119/zhang20h.html) establishes the importance of augmentation/output projection for representing noninvertible functions. Applying the existing constraint to the predictor would be a useful architectural control at most, not a distinct newly proposed principle.

**Verdict:** reject as a new method, and reject the universal claim that preserving all state differences through contact is the right fix. What should be preserved is the information required by subsequent outcomes, which is not identical to all initial information.

## B. A path-area target that sees what the endpoint misses

The proposed sample-level target used fixed addressed image features y_t, their displacement r, and antisymmetric second-level area A:

\[
r=\sum_t d_t,\qquad
A=\tfrac12\sum_{i<j}(d_i d_j^\top-d_j d_i^\top),\qquad d_t=y_{t+1}-y_t.
\]

The clean rollout's decoded features would match r and A while retaining the original objective. It requires no optimal action, branch experiment, contact annotation or new image access. It differs algebraically from the linear action-conditioned S1 projector, but its compositional-path motivation is close to S3 and established signature learning.

The exact concatenation identity is

\[
A(XY)=A(X)+A(Y)+\tfrac12\{r(X)r(Y)^\top-r(Y)r(X)^\top\}.
\]

**Executed check.** Across 100 random three-dimensional paths the maximum identity error is 1.78×10⁻¹⁵. Two L-shaped paths with common endpoint (1,1) have signed areas +.5 and −.5. Thus the target truly distinguishes some endpoint-equivalent paths; the feature is not empty terminology.

The learning obstruction is sharper than the feature property. At an exactly constant predicted path, all predicted increments vanish. Area is quadratic in those increments, so its entire Jacobian is zero. For a target unit square loop, r_target=0 and A_target,12=1. Therefore the displacement-plus-area loss is 1 while its gradient to **every predicted frame is exactly zero**. In the same construction, ordinary pointwise prediction has loss .4 and gradient norm .4. Small nonconstant perturbations of sizes 10⁻²,10⁻³,10⁻⁴,10⁻⁵ give area-loss gradient norms approximately .021677,.00216769,.000216769,.0000216769: a real but progressively weak signal near the stationary configuration.

This does not show that signature models cannot learn paths. A monotone time coordinate changes the stationary-point analysis: time–space area includes a linear integral of the spatial path, so it can supply a first-order gradient. Likewise, random initial variation or additional grounded objectives may escape the construction. Those are standard variants requiring their own accounting; the check falsifies the proposed observation-only area repair as a guaranteed escape from carry, not the entire signature literature.

**Primary audit.** [Conditional Sig-Wasserstein GANs](https://arxiv.org/abs/2006.05421) already trains conditional future-path models with signature-based discrepancies. [Neural Rough Differential Equations](https://proceedings.mlr.press/v139/morrill21b.html) uses log-signatures to encode how path segments drive a dynamical system. More directly, [Time series generation with spectrally aligned latent flow matching](https://arxiv.org/html/2609.21989v1) includes a squared signature-consistency loss for decoded latent trajectories. Their complete systems are not identical to the proposed JEPA auxiliary, but the operation is established. Moving it into this repository is insufficient substantive novelty, especially without solving the observed forecast failure.

**Verdict:** reject as an additional new method. The retained lesson is that a target can distinguish the right paths while failing to provide a useful gradient at the model's actual starting behavior. Distinguish representational sufficiency from optimization at initialization.

## Relation to the current empirical gap

[The Intervention Gap in Latent World Models](https://arxiv.org/abs/2608.29998) is a close primary diagnostic reference discovered during this audit. It reports cases where current task variables and real intervention effects are decodable, while imagined effects are poor. Its reported direction/gain failures are not our near-zero clean response, and its tasks/results are not matched Cube comparisons. It does establish that the broad distinction between captured state information and correct imagined intervention effects must not be claimed as our new discovery.

This pass adds two tested constraints on future designs, not two innovations: a predictor must be allowed to erase physically irrelevant distinctions, and an informative path target must also have the needed optimization signal. There is no justification here for another GPU grid or a new method count.
