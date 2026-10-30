# Independent design record before nearest-work search

2026-10-30. These are candidates, not accepted ideas or novelty claims.

## R1: Common-context decoder differences

Prior evidence: the anchor and noisy-target route can explain much of rendering while the clean state has readable position. An arbitrary renderer bias/context prediction can improve the usual objective without making state distinctions consequential.

For an observed transition i→j with executed action sequence a, propose using the same decoder context U and same generative random variable in two evaluations:

    delta_hat = D(U, F^k(E(o_i), a)) - D(U, E(o_i))
    delta_target = psi(o_j) - psi(o_i)

U must be identical in both branches, contain no target image or action sequence, and have exactly declared observation access. psi is observation-derived and fixed, not privileged state. A posterior analogue replaces F^k(E(o_i),a) with E(o_j). This is a paired, within-trajectory loss. The intended mechanism is cancellation of every state-independent decoder/context contribution, making the *difference* require the state/action route.

Exact elementary property: if both condition vectors coincide, delta_hat is identically zero, independent of decoder capacity and context. On a clip with nonzero delta_target, its loss has a positive floor. This excludes a zero-loss collapsed state under an exact posterior contrast but does not guarantee that optimization escapes collapse, that forecast conditions cannot simply encode actions, or that retained differences have good goal geometry. With an observed anchor, the forecast branch can still render context-dependent action consequences; therefore the posterior condition and cross-context portability need separate checks.

Possible non-equivalence test: D(U,z)=b(U)+Az. The proposed difference removes b(U) exactly, whereas anchored reconstruction may fit via b(U). However, this may be equivalent to known temporal-difference reconstruction or a residual autoencoder; literature audit decides whether it survives. It also does not by itself solve Cube's robot/object allocation. A variant that merely reweights moving patches would duplicate earlier I07 and must not count separately.

Planned falsification: construct an anchor-only renderer and a learned linear encoder/decoder with a small moving factor. Compare full reconstruction, paired differences and an ordinary difference-prediction baseline. If the new loss is algebraically identical to the baseline under the stated architecture, reject it as a distinct idea. Check whether latent collapse remains stationary even though its loss is nonzero.

**Audit outcome: reject R1 as a new mechanism.** With e_i=D(U,z_i)-psi(o_i), the complete pairwise objective is exactly sum(i<j)||e_i-e_j||²=n sum_i||e_i-mean(e)||². It is centered reconstruction with an integrable shared decoder, not a sufficiently new learning principle. A nonzero collapsed loss is also not a guarantee of an escaping gradient. Close precedents include difference reconstruction in [Universal World Model](https://openreview.net/pdf?id=lw5GlytIY5), latent increments in [DeltaWorld](https://arxiv.org/abs/2610.02691), and learned temporal shifts in [TDV](https://temporal-difference-vision.github.io/static/pdfs/tdv.pdf). Their complete implementations are not identical to R1; the rejection concerns insufficient substantive differentiation, not a claim of literal source-code equivalence.

## R2: Conditional action-response coordinates through a Stein operator

Designed after the command-only subtraction candidate failed its actual-data independence assumption. Let x be the current observation/history, a an executed action and y its observed outcome. Fit the behavior density pi(a|x) only as a training nuisance; it predicts recorded actions without claiming their optimality. For a fixed observation query psi and smooth test vector v that vanishes at the action-support boundary, integration by parts gives

    E[-psi(Y) (div v(A) + v(A)·grad_a log pi(A|x)) | x]
        = E[v(A)·grad_a E[psi(Y)|x,A] | x].

Regressing a state-only head onto the left-hand single-transition product estimates a response coordinate without differentiating a learned forward model. For scalar a, v>=0, divide by E[v(A)|x] to obtain a weighted-average response. If the physical response is affine with constant derivative, this coordinate is independent of the policy's state-dependent action mean; clipping/contact can produce state-dependent response. This differs from old I10's use of a model Jacobian to weight a planning cost, but its underlying Stein identity and score-feature principle are established, not newly proved theory.

For bounded a in [-1,1], use v=(1-a²)²; both v and v' vanish at the endpoints. Ordinary unbounded-Gaussian identities are invalid for clipped or atomic action distributions. Even this boundary-safe field receives zero weight on endpoint atoms; all-atomic policies provide no continuous-response information. A mixed/discrete Stein operator would be a different necessary implementation, not an automatic extension.

Critical limits: (1) constant response does not encode absolute location, so a pure response state fails free-space goal navigation; (2) multi-step feedback action sequences contain information about intervening outcomes, so conditioning on a complete action sequence does not automatically identify its interventional response; (3) learned density/score errors bias these targets, with no automatic Neyman orthogonality; (4) observations must suffice to block action/outcome confounding for a causal interpretation; (5) good response coordinates do not determine an object-only goal metric. A grounded zeroth-order state and supported open-loop queries are required, but their combination may collapse into ordinary predictive-state learning and requires separate novelty scrutiny.

Closest mathematical source: [Janzamin, Sedghi and Anandkumar, Score Function Features, ICLR2015](https://arxiv.org/abs/1412.6514). Its score-weighted outcome moments already recover derivatives; the candidate distinction is conditional executed-action scores as a reward-free world-state training target. This remains a research hypothesis until action support and the indispensable zeroth-order branch are validated on both datasets.
