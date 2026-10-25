# SPT extension: separate dynamics from visual detail

Please start with the residual model below, trained end to end on PushT. The first goal is to generate futures that follow the supplied actions. Once that works, compare a more flexible transition with explicit control over latent capacity. Planning evaluation comes afterward.

The three essential additions to SPT are **a stable latent partition, a residual next-state prediction, and a JEPA-style latent prediction loss**. The second model replaces the residual restriction with two asymmetric transitions and a capacity penalty.

## Motivation and shared interface

We want a compact **high-level state `h`** to hold information needed to predict dynamics, while a **low-level state `l`** preserves visual detail for generation. In PushT, positions, velocities, and contact geometry may belong in `h`; colors and background appearance are candidates for `l`. These are intended learned roles, not supervised labels.

This lets dynamics prediction focus on change while retaining detail for reconstruction. Our hypothesis is that reconstruction, transition asymmetry, and suitable capacity can support learning without SIGReg, EMA, or stop-gradient. **Reconstruction alone does not ensure that both partitions remain useful.**

Keep SPT's ViT, causal history processing, conditioning projection, and DiT flow generator. Use next-chunk prediction first, with `a_t` denoting the action or action sequence taking chunk `t` to `t+1`.

- `z_t = E(x_t)` has shape `[N, d]`, omitting the batch dimension. `N` is the number of post-ViT feature tokens; `d` is channels per token. `E` encodes chunks independently, without access to future chunks.
- Choose fixed token indices `J`, with complement `Jc`: `h_t = z_t[J]`, `l_t = z_t[Jc]`. Keep their identity across examples, time, and inference; train the encoder to route information into them.
- Dimensionality primarily means **token count**, including each partition's count; `N*d` is the scalar width. Reducing only temporal-transformer inputs does not constrain reconstruction capacity.
- Split after the full-chunk ViT. The slots can contain information from across the image; they are not fixed physical objects.

The SPT manuscript (*Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal History*), Sections 3.2–3.4, uses `ViT -> randomly retained tokens -> causal transformer -> projection -> DiT`, with future flow matching. The boxes below specify changes to those components, rather than exact edits to an available SPT implementation.

## Algorithm 1: residual transition, implement first

The high-level history and actions compute a **full-state increment**. The previous full state supplies an identity carry:

```python
# Input: observed chunks x[:t+1], actions through a_t, target x[t+1].
# NEW marks the three architectural/objective additions.
z_hist = E(x[:t+1])                          # [B, T, N, d]
h_hist = z_hist[:, :, J, :]                  # NEW: fixed high-level partition

u = CausalTransformer(h_hist, actions)[:, -1] # latest origin; actions enter here only
delta = FullGridHead(u)                     # NEW: [B, N, d], from h/actions only
z_next = z_hist[:, -1] + delta               # NEW: full-state residual

z_target = E(x[t+1])                        # shared encoder; gradients stay attached
L_pred = partition_mse(z_next, z_target, J)  # NEW: explicit next-latent prediction
condition = Project(z_next)
L_rec = flow_loss(DiT, x[t+1], condition)    # retain SPT's generative objective
loss = L_rec + lambda_pred * L_pred
update(E, CausalTransformer, FullGridHead, Project, DiT, loss)
```

**Both partitions may change. Only `h` and actions compute those changes.** The low-level state enters its successor through addition, rather than through a learned nonlinear transition. This is the intended restriction, and it is deliberately strong.

`FullGridHead` must expand SPT's sparse causal output to `[N, d]`, for example with fixed positional queries attending to `u`. Low-level features must not enter this head or earlier history-fusion layers.

For clarity, use the following shared loss definitions in both algorithms:

- `partition_mse = 0.5 * (MSE(h_next, h_target) + MSE(l_next, l_target))`, averaging within each partition. During soft-mask training, keep the denominators fixed at the candidate partition sizes, including masked coordinates. Supervise both partitions for full-state latent rollout; high-level-only prediction is a later ablation.
- `flow_loss` retains SPT's target in pixels or fixed-codec coordinates. With noise fraction `s`, noise `eps`, and target `y`, the DiT receives `(1-s)*y + s*eps` and regresses `eps-y`, conditioned on `Project(z_next)`. Here `L_rec` names this visual generation loss, not an additional autoencoder loss.

Train both encoder branches jointly. Disable SIGReg, target detachment, teacher EMA, and checkpoint-weight EMA for the proposed experiment. Keep latent normalization fixed across comparisons and monitor latent norms: shrinking representations can lower MSE without improving predictions.

## Algorithm 2: flexible transitions with capacity control

Allow the low-level state to evolve through a second causal transformer. Only the high-level transition receives actions. I suggest also letting the low-level transition read the **predicted next high-level state**, so it can express immediate action effects through that route.

**The capacity idea is to retain as few representation tokens as possible while preserving reconstruction and prediction quality.** Reconstruction gives the model a reason to retain information; a sparsity penalty gives it a reason to remove unnecessary tokens. As capacity becomes tight, we expect the remaining tokens to carry useful information. Apply this pressure to the whole representation, with the same penalty for high-level and low-level tokens.

One simple implementation is a learned global soft mask: one scalar `g_i` per token, shared across examples and time, with `0 < g_i < 1`. Multiply each token by its scalar and add `lambda_dim * mean(abs(g))` to the loss. This is an L1 penalty on mask values, a practical proxy for reducing token count.

```python
# mask_logits: N learned parameters, initialized once so gates start near 1.
g = sigmoid(mask_logits)[:, None]           # [N, 1]; shared across batch/time/target
z_hist = g * E(x[:t+1])                     # apply to EVERY state-carrying path
h_hist, l_hist = split(z_hist, J)

h_next = g[J] * HighTransition(h_hist, actions)
l_next = g[Jc] * LowTransition(h_hist, l_hist, h_next)
# LowTransition has no action input; h_next is its only current-action route.
z_next = merge_in_original_slot_order(h_next, l_next)
z_target = g * E(x[t+1])                     # same gate, attached target

L_pred = partition_mse(z_next, z_target, J)   # fixed denominators during soft training
L_rec = flow_loss(DiT, x[t+1], Project(z_next))
L_dim = g.abs().mean()                      # global L1 pressure on ALL token gates
loss = L_rec + lambda_pred * L_pred + lambda_dim * L_dim
update(E, HighTransition, LowTransition, Project, DiT, mask_logits, loss)
```

Both transitions are causal transformers returning next-state tokens. Neither the low-level transformer nor the DiT receives a separate action input. A past-only variant `LowTransition(h_hist, l_hist)` is also worth testing, but cannot immediately respond to the current action.

Start with gates near one and introduce the penalty gradually after generation begins to learn. Increase `lambda_dim` to explore how much representation can be removed while maintaining validation quality. Keep masks independent of images and actions so mask choices cannot carry sample-specific information. A separate low-level budget is an optional later experiment, not part of this initial test of total capacity.

There are three implementation checks:

1. **Verify removal, not just small gates.** Encoder or decoder weights can compensate for smaller multipliers. A sigmoid also never produces an exact zero. Choose a pruning threshold on validation data, remove low-gate tokens from every state path, and evaluate before any recovery fine-tuning. Report the actual retained count as well as the L1 value. L1 scaling followed by pruning has a precedent in [Network Slimming, Section 3](https://arxiv.org/html/1708.06519v1#S3); applying it to these latent tokens is our proposed adaptation.
2. **Check normalization placement.** Per-token LayerNorm can approximately cancel a positive scalar mask: `LayerNorm(g_i * z_i)` can be nearly unchanged as `g_i` shrinks. Apply the gate after the relevant normalization or to the token's attention contribution, and check for ungated bypasses.
3. **Check decoded predictions.** Gating both sides also reduces latent MSE as gates shrink, even without better predictions. SPT's DiT receives a noisy future target during training, so low flow loss alone also does not establish that retained tokens carry the needed information. Judge generation from noise, reconstruction fidelity, and action following after pruning.

The hypothesis remains that **total capacity pressure, together with routing actions through `h`, encourages useful high-level dynamics**. A compact solution could still store scene information in `l`, while `h` is unused or merely passes along an action code. Test whether removing retained tokens harms reconstruction, and whether replacing high-level history while holding actions fixed harms predicted motion. These test economical use of the representation and useful high-level state separately. If a whole partition disappears, report that as a failure of the intended partition, even if images still look good.

## Focused tuning and implementation order

These are starting ranges, not established optima. Keep resolution, data, action timing, history length, and optimizer fixed while testing an axis.

| Component | Controls to vary | Initial comparison |
| --- | --- | --- |
| Partition and capacity | Encoder token count `N`; candidate high-level fraction; global mask penalty `lambda_dim` | Use the existing legal patch grid `N0`, then grids near `N0/2` and `N0/4`, with channel width `d` fixed. Try high-level fractions `1/8`, `1/4`, `1/2`; report the retained fractions after pruning. |
| Transition asymmetry | Residual versus the two-transition model; global capacity pressure for the latter | Compare the same two-transition architecture with zero, weak, and stronger L1 penalties. Select penalty strength and pruning threshold using retained token count versus validation reconstruction and action-following quality. |
| Latent prediction | `lambda_pred` relative to `L_rec` | With the loss normalization above, start at `1`; try `0.1` and `10` if needed. Use `0` only as the loss-removal ablation. Log both losses and encoder gradient contributions. |

Changing patch size and pruning post-ViT features are different interventions; record both. Match training exposure and report parameters, throughput, memory, and decoder sampling steps.

Implement Algorithm 1 first. Verify that targets receive gradients, future observations cannot change predictions, and perturbing `l` leaves `delta` unchanged. Then tune partition size and loss balance before adding Algorithm 2. Use one seed for screening and initially three for selected comparisons; unstable differences need more evidence. Profile training cost before setting a sweep budget.

Compare action-conditioned original SPT, Algorithm 1, its no-latent-loss and no-carry ablations, and Algorithm 2 with versus without capacity control. The no-carry head predicts the absolute next state. Match tuning opportunities and avoid a full Cartesian sweep.

## Evaluate generation and action following before planning

Train on ordinary PushT trajectories with recorded actions; **paired counterfactual training data are not required**. Split by episode before constructing windows. Verify action normalization and frame skipping. Include varied actions and both contact and non-contact motion.

Encode a prefix once, recursively predict the full latent, and decode each prediction. Carry predicted states forward, never ground-truth futures. Keep the partition and gate fixed. Report generate-then-re-encode continuation separately.

| Test | What to show or measure | What failure tells us |
| --- | --- | --- |
| Held-out generation | Side-by-side videos and image error at 1, 4, and 8 model steps, with elapsed time reported | Good one-step output with rapid drift indicates a rollout problem. Low latent loss alone is insufficient. |
| Action following | From the same simulator state and observed prefix, compare different action sequences with their actual simulator futures. Reuse decoder noise across alternatives. Measure generated agent position, block position, and block angle errors. | Futures must change in the correct direction and magnitude, not merely look different. Wrong contact outcomes expose missing dynamics. |
| Use of the high-level state | Replace high-level history with its dataset mean while preserving `l` and holding actions fixed; separately change or shuffle actions. Inspect generated motion and errors. | Little effect from replacing history suggests that `h` carries little useful scene state, even if actions affect outputs. Interpret alongside normal held-out performance. |
| Use of retained capacity | Mask L1, actual retained count, latent norms and variation; generation before/after pruning; remove retained tokens one at a time without retraining | If many retained tokens can still be removed harmlessly, capacity remains redundant. If pruning low-gate tokens damages generation, small masks did not identify dispensable tokens. |

Same-state comparisons can be **evaluation-only** simulator rollouts. Restore velocities as well as positions, and define a hold action using the actual control convention. Use the same image-based pose extractor for all models, calibrated on real held-out renders. Simulator states provide reference poses, not model inputs.

Start with the repository's LeWM as an end-to-end JEPA baseline; PreJEPA/DINO-WM is an optional frozen-feature comparison. Retain their representation-training recipes, then freeze the encoder and predictor and train a decoder on encoded training frames. Decode both real-frame encodings and predicted future latents to distinguish decoder limitations from transition errors. Match decoder capacity, data, and sampling effort where feasible; report the extra decoder training.

Please return videos, metrics with seed variability, and reproducible configurations/checkpoints under `experimental_results/`. Move to planning after consistent action following in free motion and contact. If `h` is ignored, inspect action diversity and bypasses, then assess whether global compression changes its usefulness. If motion remains correct but appearance deteriorates, weaken the penalty or retain more tokens. Report any added stabilization as a separate variant.

## What is open for your input

Please start by testing the global soft mask with L1, and suggest the best gate placement and pruning schedule in your SPT code. Can we reduce total retained tokens while preserving appearance and making high-level history more useful for action-conditioned prediction? Fixed token counts or a channel bottleneck are useful alternatives. Use the residual model as our common reference.

For background, the [image-translation paper](https://openreview.net/pdf?id=U2g8OGONA_V) and its [mask implementation](https://github.com/Mid-Push/i-stylegan/blob/main/training/networks.py#L263) motivate selecting domain-dependent changes, rather than limiting total latent information. [Mechanism sparsity, Sections 3.4–3.8 and 5](https://arxiv.org/html/2401.04890v2) motivates restricted dependencies and varied influences under additional assumptions. Neither establishes collapse prevention here; the practical tests above are our evidence.
