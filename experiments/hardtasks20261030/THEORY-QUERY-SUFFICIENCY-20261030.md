# Mechanism hypotheses: conditional generation versus planning

Research note, 2026-10-30. This is analysis of existing evidence and a diagnostic proposal, not a new performance result or novelty claim. No training is launched by this note. The current setting is action-conditioned world-model learning with sample-level objectives and no optimal-action oracle.

## What the evidence establishes

Temporal-forecast improves matched PushT planning over AS-JEPA controls, but that advantage has not transferred at the budgets tested. Cube first-window-epoch success is 24/26%, compared with lower-beta 22/26%; its denoising materially depends on encoded states and actions. TwoRoom at 2000 updates has 2/6% success, nearly constant encoded states, and sometimes action-sensitive forecast denoising. Cutting the anchor for 600 further updates suppresses that action effect without reliably restoring state use. Sharing the forecast prior between rendering and rate does not provide a consistent gain. See the existing `results/learn2000`, `results/anchor2600`, `results/sharedprior2600`, and `results/cube-epoch1` reports.

These observations require separating four properties:

1. State information exists in the representation.
2. Predicted outcomes distinguish candidate action sequences correctly.
3. The goal cost ranks those outcomes according to the evaluation task.
4. Search visits sufficiently good candidates.

Passing property 1 does not establish 2–4. Increasing denoising state/action sensitivity establishes dependence in a particular prediction task, not correct action ranking for planning.

## Actual graph and what the loss rewards

`as_jepa_lewm/robust.py` encodes each clean frame into a mean `mu`. The posterior renderer condition is the corresponding noisy state. For half of frame conditions, temporal-forecast instead supplies a free-running predictor output, initialized from the first noisy state and conditioned on recorded actions. The renderer also receives the clean anchor and a noisy version of the target frame. It receives no raw actions directly. The one-step predictive rate matches a temporally correlated posterior target; forecast rendering has its own causal predictor rollout. The tested shared-prior variant changes the rate predictor to that rollout.

For a fixed representation, let V be the flow velocity target, U be the noisy target, corruption time and anchor, and Z be the renderer condition. For unrestricted squared-error predictors, the exact improvement obtained by revealing Z is

\[
 \inf_f E\|V-f(U)\|^2-\inf_g E\|V-g(U,Z)\|^2
 =E\|E[V\mid U,Z]-E[V\mid U]\|^2.
\]

This follows from orthogonal projection in L2. It is not an assumption that flow matching equals a normalized likelihood, and it is not generally equal to mutual information. It describes what the best renderer could gain from a fixed condition; the finite trained renderer need not attain either infimum.

The representation is rewarded for explaining what the noisy target and anchor do not already explain. A physical quantity can be crucial to control and have a small value under this conditional prediction problem. Conversely, a quantity can greatly reduce pixel error without correctly distinguishing task outcomes. This is the central hypothesis connecting the two failures.

The flow target in the implementation is `y - eps`, with t=1 denoting pure noise. Higher corruption reduces the direct target-image information but does not remove the anchor or change the relative semantic value of pixel regions. It can increase the value of state information, but provides no theorem that agent position, contact state, or object pose must be encoded.

Both flow error and rate are normalized by image elements: flow is a pixel mean and rate is divided by `(window-1)*FRAME_ELEMS`. Therefore, blaming resolution for reducing only the flow term would be incorrect. The relevant quantities are the total residual energy that a physical factor can explain, its encoding cost, and the optimization state. Region diagnostics should report both total error contribution and error per pixel.

## A precise gap in the original identifiability argument

`docs/lingjing/anchored-state-jepa-design.md` assumes at line 212 that every state function shares information with other frames conditional on the anchor and actions:

\[
 I(\phi(S_t);X_{\ne t}\mid X_0,A_{0:T-1})>0.
\]

If X0 identifies the initial physical state and dynamics/rendering are deterministic given the full action sequence, every later observation is already a function of the conditioning variables. The conditional mutual information above is zero. Thus this assumption excludes an important limiting case of the navigation setting; it cannot guarantee preservation of state information there. This is a counterexample to applicability of that argument, not a proof that the implemented objective must collapse.

The graph makes the practical bypass more specific. The posterior branch can exploit the noisy target and anchor. In the forecast branch, the predictor condition can carry action-sequence information while the renderer obtains initial configuration from the anchor. An action-sensitive renderer is consequently compatible with an almost state-independent encoder. Existing interventions support this possibility, but have not yet measured every causal route separately.

There is also a coordinate ambiguity: an encoder that sees only a current image can still see its context c and encode a context-dependent transform h_c(s). A decoder with an anchor containing c can invert that transform. The absence of a direct anchor input to the encoder does not by itself force one context-independent state coordinate system. For example, in an isotropic context-independent stochastic process, context-dependent orthogonal rotations preserve isotropic Gaussian entry and transition distributions while the decoder can undo them. This toy process illustrates a remaining symmetry; it does not assert that arbitrary such rotations preserve the actual action-conditioned AS-JEPA objective.

## Why compression can remove a useful factor

Consider a scalar residual factor X of variance lambda, a noisy code Z=kX+sigma*epsilon, an optimal linear Gaussian decoder, and a quadratic entry cost. Write v=k^2*lambda. A local surrogate objective is

\[
 J(v)=\frac{\lambda\sigma^2}{v+\sigma^2}
       +\frac{\beta_{\rm eff}}{2}v,\qquad v\ge0.
\]

Its optimum is

\[
 v^*=\max\{0,\sqrt{2\lambda\sigma^2/\beta_{\rm eff}}-\sigma^2\}.
\]

A factor is switched off when its residual variance is below `beta_eff*sigma^2/2`. This is an exact calculation for the surrogate only. In the real model, predictable factors also pay a temporal surprise term, renderer capacity is finite, posterior noise is correlated, and the target includes denoising side information. Those differences prevent substituting raw empirical image variance into this formula and claiming a predicted collapse threshold.

The useful conclusion is narrower: the loss allocates representational resolution according to reduction of its chosen prediction error per unit of rate, not according to a factor's eventual control importance. This is consistent with rate-distortion perspectives, and with analyses connecting latent collapse to likelihood/mean-regularization competition. It is not a new discovery of those principles. [Alemi et al., 2018](https://proceedings.mlr.press/v80/alemi18a.html), [Wang and Ziyin, 2022](https://arxiv.org/abs/2205.04009).

Optimization history is a distinct explanation. Linear-VAE analysis shows that collapse can also reflect local maxima of marginal likelihood, rather than being caused solely by an ELBO approximation. Consequently, removing rate from an already collapsed model and training briefly need not reveal the optimum of the rate-free objective. [Lucas et al., 2019](https://arxiv.org/abs/1911.02469).

## Cube: information sufficiency does not determine a useful metric

Let the true state contain robot pose r and object pose o. An injective representation `E(r,o)=(c*r, o/c)` can preserve all state information. With exact transformed dynamics and a matching decoder, prediction can remain exact while Euclidean latent goal cost weights robot error by c^2 and object error by c^-2. This can produce poor object-goal choices even with perfect rollouts.

This example proves that invertibility or predictability alone does not establish suitability of Euclidean latent planning. It does **not** prove that all such rescalings leave the actual AS-JEPA objective invariant: fixed isotropic posterior noise, the prior family and the rate explicitly constrain coordinates. The question is whether the coordinates selected by those constraints align with the goal objective.

A useful diagnostic ladder evaluates the same candidate action bank using: (a) actual physical endpoint error, (b) encoded actual-endpoint distance to encoded goal, and (c) predicted-endpoint distance to encoded goal. Failure between (a) and (b) isolates representation/metric mismatch; failure between (b) and (c) isolates rollout-based ranking; absence of a good candidate under (a) implicates candidate coverage or horizon. All physical-state use here is diagnostic, not a training target or deployable controller.

The same ladder also distinguishes contact effects: small physical changes near contact can have large downstream consequences even if local image error is small. One-step average prediction accuracy can obscure these rare branches. This is a testable hypothesis, not yet an observed cause of the Cube failures.

For a finite action bank, if a surrogate cost approximates true task cost uniformly within epsilon, choosing its minimizer incurs at most 2*epsilon regret relative to the best candidate in that bank. When costs have been put on a common scale, epsilon can be bounded by a metric error plus a rollout error via the triangle inequality. A separate coverage gap remains between the bank's best candidate and the best feasible action sequence. In practice, monotone changes of cost do not affect ranking, so rank and selection regret are more robust diagnostics than comparing raw latent-MSE magnitudes across encoders. No small uniform error bound has yet been demonstrated for our models.

## Why LeWM may do better without proving its universal superiority

LeWM directly constrains representation spread through SIGReg, rather than requiring every represented difference to earn its cost by reducing a conditional renderer's error. This gives an additional source of pressure against near-constant representations. It does not guarantee a task-correct metric or identify every state factor. Its own results report limitations in low-complexity TwoRoom and suggest a mismatch between an isotropic high-dimensional latent distribution and low intrinsic dimensionality. [LeWM](https://arxiv.org/html/2603.19312v3).

The useful contrast is therefore between mechanisms for allocating representation, not an unconditional assertion that distribution matching is better than generation. In our present observations, LeWM's released checkpoint solves many TwoRoom cases while AS-JEPA's states nearly collapse. Model size, exposure and training budget still differ, so a controlled experiment is needed to attribute that gap.

A particularly close precedent appears in PLDM Appendix G/Table 6: in its own TwoRoom protocol, PLDM obtains 97.4 +/- 1.3% while a matched encoder/predictor trained with reconstruction instead of VICReg obtains 26.2 +/- 13.9%. Its reward/actor-removed DreamerV3 variant obtains 24.0 +/- 6.9%; the authors explicitly acknowledge that this is not Dreamer's intended setup. A reward-removed TD-MPC2 variant collapses, and adding inverse dynamics reaches 35% in a single seed. These results are not numerically comparable to our custom cases, but demonstrate that poor latent-goal planning under reconstruction-based learning is an established empirical issue in this environment family. PLDM itself also uses inverse dynamics as an auxiliary objective; this predicts executed actions without asserting they are optimal. [PLDM](https://arxiv.org/html/2502.14819v4).

## Three bounded diagnostics and their interpretation

1. **Rate intervention with history control.** For each task/seed, compare beta=.25 and beta=0 from exactly matched fresh initial states and separately from exactly matched 2000-update full states, stopping after 600 updates. Track fixed-case state variance, semantic probes, actual-goal metric alignment, and posterior/forecast state use at early checkpoints as well as the endpoint. Beta=0 removes both entry compression and latent dynamics consistency; it is a diagnostic, not an already justified method. A representation recovery without planning recovery is compatible with loss of consistency. Within each starting point the beta comparison is causal; fresh versus resumed differences combine weights, optimizer history, LR and data position and should be described as history dependence, not proof of an irreversible attractor.
2. **Loss-source audit.** At a fixed batch and fixed random draws, separate encoder gradients from flow and rate, report norm and cosine, and measure the directional derivative for expanding centered state differences. Separate posterior and forecast renderer branches, and report spatial total/per-pixel error contributions. A physical mask can diagnose agent/object/robot contributions but must not become case-specific training supervision. If beta=0 does not recover useful state and flow gradients remain insensitive to position, insufficient prediction demand is more plausible than rate alone.
3. **Frozen candidate-ranking ladder.** Use a shared action bank and identical starts/goals across checkpoints for the physical/encoded/predicted cost decomposition above. Report rank correlation, selected-candidate physical regret and candidate coverage, not only average latent MSE. Split by contact and horizon when those categories are determined from simulator logs for diagnosis. Do not label an expert trajectory the optimum; use it only as one known feasible candidate where applicable.

These diagnostics reuse existing models and data, make few causal changes, and separate explanations before proposing another long training grid.

## The abstraction to pursue, and its limits

A useful state is sufficient for a family of future questions. For reward-free planning, those questions include how outcomes differ under alternative action sequences and whether those outcomes satisfy a supplied goal. A state sufficient for conditional frame denoising is not automatically sufficient for this larger family. Enlarging the question family can require preserving directions that contribute little to pixel prediction.

This connects to established predictive-state representations, which use multi-step action-conditioned observation predictions, and value-equivalence, which asks whether models agree on the updates needed by a specified family of values/policies. It is not a newly invented theory. [Littman, Sutton and Singh, 2001](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html), [Grimm et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html).

There are two unavoidable limits. First, data from one behavior policy cannot identify consequences of arbitrary unsupported actions; a new loss cannot create counterfactual evidence. Second, without specifying a goal/reward family, there is no general reason to discard one predictable physical factor as irrelevant to every future task. Task-specific bisimulation methods address a more specified problem and cannot be imported unchanged while claiming a fully reward-free objective. [Zhang et al., 2021](https://arxiv.org/abs/2006.10742).

The research opportunity is to construct and validate a simple sample-level objective whose representation is sufficient for the relevant action-conditioned questions, while preserving a useful goal metric under the same planning interface. The current evidence identifies the missing property; it does not yet supply a proven new objective. Any proposal should first predict a distinguishable outcome in the diagnostics above and then pass the short training gates.
