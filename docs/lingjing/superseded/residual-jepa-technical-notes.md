# Residual SPT: technical clarifications

Updated 2026-10-08. These notes support the [concise author message](residual-jepa-spt-proposal.md). The current proposal uses a globally fixed selected subset and a full-state residual, replacing the earlier clip-dependent mask and complementary-only carry. No new model has been implemented or evaluated.

## 1. Stable latent variables across the distribution

For the intended causal interpretation, use the same selected latent coordinates or slots J across examples, time, training and deployment. A different random mask per clip would test robustness to missing features rather than identify a common predictive subset. The encoder learns to place relevant information in the selected locations.

Stable latent identity is distinct from a fixed image location. A patch at the top left can contain different objects in different scenes. A selector that tracks the same variables through changing image locations could preserve semantic identity despite changing patch indices; random mask resampling does not provide that correspondence. Global ViT attention can route information into fixed selected patch slots, so these slots can function as a shared bottleneck, but they do not automatically become individually identified physical variables. Dedicated learned summary slots or a fixed channel subspace make the intended roles clearer at the cost of an encoder/interface change. Channel selection generally does not reduce the number of transformer tokens.

For a first test, fix a set of post-ViT token indices once and record it with the model. Train the encoder jointly so it can route information into those slots. A learned global selector could be compared later. With an unrestricted learned encoder, selecting the first k coordinates can be equivalent to selecting any other global k-coordinate subset after a permutation; mask learning is not inherently the contribution.

The supplied example distinguishes environment-specific support from stable variable identity. These can coexist: one shared causal variable set can have different active interactions in different scenes. If different environments require different parents, a global selected set must include their union. Sparse parent sets for each output do not ensure that this union is small. For example, a diagonal dynamics graph has few edges but uses every state variable. Our small shared J is a stronger hypothesis than ordinary edge sparsity.

## 2. A single predictor and a full-state skip

The revised formulation is:

\[
z_t=E(x_t),\quad s_t=z_t[J],\quad
\Delta z_{t+1}=P(s_{\leq t},a_{\leq t}),\quad
\hat z_{t+1}=z_t+\Delta z_{t+1}.
\]

All output entries now have the same role: they are increments. The selected state has a closed recurrence:

\[
\hat s_{t+1}=s_t+P(s_{\leq t},a_{\leq t})[J].
\]

There is no need to name a separate G. P includes whatever output readout maps its sparse internal tokens to the complete state's shape. This simplifies the interface, not necessarily the computation.

The supplied SPT manuscript (*Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal History*), Section 3.2, returns M retained tokens from the causal transformer, while its DiT predicts a dense future chunk in pixels or fixed-codec coordinates. Thus the original causal-transformer output is not already the full S-token encoder grid. A full output head is still needed inside P. Dense positional output queries attending to sparse causal features are one option. Simply retaining all tokens throughout the transformer would remove the intended sparse temporal computation.

For encoder-grid shape [S,d], the head must produce [S,d], not merely use the same channel width d. The residual addition itself is cheap, but full output generation, dense DiT conditioning and full-state storage remain. No speedup is assumed without measuring these costs.

For an SPT-preserving first experiment, condition the existing DiT on the projected sum and keep the fixed-space flow target. Under this objective the sum is a learned condition, not necessarily E(x_{t+1}). Generate the next chunk, re-encode it, and apply the same J for continued generation. SPT's unmasked continuation fine-tuning would change this state interface, so keep it as a separately reported comparison.

To make latent-only rollout meaningful, supervise at least the selected coordinates against E(x_{t+1})[J], with a specified target-gradient and anti-collapse recipe. To claim that the entire residual output predicts the next encoder state, supervise the full grid. In either case compare z_t+Delta z with the next-state target, or compare Delta z with the next-minus-current target. Do not train Delta z against the absolute future encoding and then add z_t again.

For multi-step latent rollout, copy the predicted state from the preceding step, never a ground-truth future state. For a direct k-step forecast, copy the observed origin z_t and condition P on k. Unobserved intervening features must not enter the carry.

## 3. Causal motivation and the JEPA connection

The graph in the author message represents the proposed dependency of the residual update on selected variables. Every coordinate still influences its own complete successor through the identity path. Consequently J is the candidate shared parent set of changes, not the set of all causes of the next observation.

The bottleneck directly supports the JEPA motivation: give temporal prediction a compact representation and allow rendering information to take another path. However, three claims remain distinct: a small computational interface, sufficient information for the intended predictions or tasks, and identified causal variables. Architecture supplies the first and makes the latter two testable.

A small update-generating state need not be a sufficient planning state. For free motion, Delta position = velocity: velocity suffices to calculate the increment, but reaching a specified location also requires current position. If the planner operates only on the selected state, J must preserve information needed to evaluate its goals as well as to calculate changes. Alternatively, a cost function can read necessary carried state, but that is a different planning claim.

Likewise, fixed J does not imply that the selected values are static, or that unselected values never change. Selected information can update every output coordinate. Static facts such as obstacle geometry can affect increments and therefore need representation in J when relevant.

The residual path provides an easy rendering route but can also weaken supervision of the selected state. A powerful decoder may learn future change from the full carried encoding, leaving P uninformative. An explicit next-selected-state loss can itself collapse jointly with its targets. Retain a defined stabilization treatment when testing that loss, and test any claim that reconstruction removes the need for stabilization separately. Feature variance, history/action interventions, nuisance changes and selected-state downstream performance answer different parts of this question.

## 4. What MC-JEPA actually reconstructs

MC-JEPA estimates optical flow from both observed consecutive frames. It uses that flow to warp an observed image and compares the result with the other image. In the paper's abstract warp notation:

\[
\widehat I_{t+1}=f_{t,t+1}(I_t),\qquad
\mathcal L_{\rm rec}=d(I_{t+1},\widehat I_{t+1}),
\]

where d mixes L1, L2 and SSIM losses. Thus the reconstruction teaches correspondence and motion estimation; it does not generate an unseen future from the past alone. Numerical mixture weights are not specified in the inspected text. The abstract warp notation is not an implementation specification for forward versus backward sampling. [MC-JEPA, Section 3.1, Equation 3 and Appendix B](https://arxiv.org/html/2307.12698v1#S3.SS1)

## 5. Narrow implementation comparison

Use the same data, future horizon, optimization and output head to compare: a dense-input reference, independent sparse masks, fixed J without the full skip, and fixed J with the full skip. Match retained count across the sparse variants; the dense reference receives every token and reports its additional computation. The fixed-versus-variable comparison must hold the skip policy constant. Compare variants before any unmasked fine-tuning as well as under a separately declared SPT continuation protocol. Record actual attention-token counts, generation latency and memory.

Use fixed-slot probes to avoid mistaking spatial position variation for information about scene state. Add a matched ablation that neutralizes P while retaining the full carry; if prediction barely changes, the decoder may be doing the temporal modeling. For an explicit JEPA variant, check selected latent-only rollout and task performance with the rendering decoder absent. These are proposed diagnostics, not experimental results.

The [prior-work report](residual-jepa-prior-work.md) separates sparse node selection, sparse interaction edges, sparse mechanism changes and content/dynamics separation. Their contributions overlap different parts of this proposal and should not be treated as equivalent assumptions.
