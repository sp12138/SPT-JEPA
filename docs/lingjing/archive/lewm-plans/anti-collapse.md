# Anti-collapse exploration plan

Branch: `codex/plan-anti-collapse`

Base: `8b2e8c685dd7f11d4189f5c09d9b6bfffd5cdca4`

Status: plan only; no method implementation or experiment launch is authorized by this document.

Shared controls, data, evaluation, and resource policy: [common protocol](common-protocol.md).

## Decision and scope

Study the three requested families in sequence within this branch: pooled SIGReg, pooled SIGReg through a bounded invertible normalizing map, and pooled SIGReg through a bounded continuous transport trained with a flow-matching auxiliary loss. Keep ordinary CLS representations, dense past inputs, next-state prediction, and the original one-step planner interface fixed. These changes remain separate from sparse-past and multiple-future experiments until their individual effects are understood.

Here, **time pooling means pooling samples**: reshape clean embeddings from `[B,T,D]` to `[B*T,D]`. It does not mean averaging the embeddings over time. A time average could remove exactly the motion information we want to retain. The linked discussion motivates the sample-pooling interpretation.

The key question is whether relaxing the distribution imposed on individual temporal marginals improves useful dynamics representations. Distribution matching alone does not specify which information should occupy those representations. A static episode identifier can satisfy either original or pooled SIGReg while omitting moving objects. Consequently, planning and physical-state probes, especially under static distractors, are required alongside anti-collapse diagnostics.

The flow families are feasibility investigations. A learned normalizing flow or a flow-matching loss is not automatically an anti-collapse objective. The explicit construction below keeps a Gaussian endpoint penalty and limits how much a transport may amplify small encoder differences. This is a proposed implementation, not an established guarantee of useful representations.

## What the current code does

`scripts/train/lewm.py::lejepa_forward` encodes all four observed frames for the default history length three and target offset one. It predicts the next three states from the three causal history positions and calls:

```python
sigreg_loss = self.sigreg(emb.transpose(0, 1))  # [T,B,D]
loss = pred_loss + 0.09 * sigreg_loss
```

`stable_worldmodel/wm/loss.py::SIGReg.forward` averages the empirical characteristic function across `B` independently at each relative clip position, then multiplies its discrepancy by `B`. It uses 1,024 random unit projections and 17 quadrature knots. This constrains batch marginals at each relative position; it neither asserts independence across time nor directly regularizes predicted states. Preserve clean encoder targets and the existing target-gradient behavior.

The default encoder is shared across frames and does not receive a relative clip index. Relative positions in randomly sampled clips may already have similar distributions. Measure that before attributing a benefit to nonstationarity. Projector and prediction-head BatchNorm, optimizer settings, and observed clip length stay unchanged in this branch.

## Claims and competing explanations

| Claim | Required evidence | Important alternative |
|---|---|---|
| AC1: pooling improves useful representations by relaxing temporal-marginal constraints | Pooled versus per-time SIGReg at matched normalization and tuning budget, evaluated on planning and state probes | A larger effective regularization coefficient or a different estimator sample count caused the gain |
| AC2: a bounded learned normalizer improves the raw latent geometry relative to pooled Gaussianity | Bounded affine and nonlinear maps versus identity, with raw-state probes and unchanged raw-space planning | Extra parameters fit the distribution penalty while hiding weak raw representations |
| AC3: flow matching makes a bounded transport easier to optimize | Identical continuous transport with and without the flow-matching auxiliary, equal tuning opportunity | The transport architecture or extra training compute caused the gain |
| AC4: a method retains temporal information in the presence of static distractors | Separate dynamic-state and static-appearance probes, paired distractor conditions, and rollout/planning results | Increased variance belongs to episode identity or nuisance appearance |

## Family A: pooled SIGReg with normalization controls

Let `d(X)` be the SIGReg characteristic-function discrepancy before multiplying by the number of samples. For equal-length clips, compare:

```text
per_time = B * mean_t d(z[:, t, :])
pooled   = B * d(z.reshape(B*T, D))
loss     = prediction_MSE + lambda * selected_regularizer
```

Using the existing implementation, the pooled expression is exactly:

```python
pooled = sigreg(emb.reshape(1, B * T, D)) / T
```

For `T=4`, omitting `/T` multiplies the pooled penalty by four at the same configured lambda. Include that unnormalized variant as an explicit diagnostic, not as the primary grouping comparison. Both implementations retain the baseline convention that the multiplier is the current batch size; do not introduce a separate batch-size change.

With the same projection directions and quadrature, the difference between the matched per-time and pooled penalties is exactly the average squared difference between each temporal empirical characteristic function and their average, multiplied by `B`. This makes the intervention precise: pooling removes that marginal-consistency component. It still asks the pooled distribution to resemble a Gaussian.

That empirical difference also contains sampling noise. Even identical population marginals need not give identical finite-batch characteristic functions. Use stationary synthetic samples and repeated held-out batches to estimate this noise floor before interpreting the difference as actual temporal nonstationarity.

Use these controls:

| ID | Regularizer | Purpose |
|---|---|---|
| A0 | Original per-time SIGReg | Shared reference |
| A1 | Pooled SIGReg divided by `T` | Main grouping intervention |
| A2 | Pooled SIGReg without division by `T` | Reproduce the easily confounded implementation |
| A3 | One randomly selected time position per clip, then SIGReg over the resulting `B` embeddings | Pool temporal positions without increasing samples or counting correlated frames repeatedly |
| A4, optional | Per-time SIGReg on `z - temporal_mean(z)` | Dynamics-oriented comparison to TC-LeWM; a different hypothesis from sample pooling |

A3 samples a time independently for each clip, uniformly over the same `T` encoded positions. Across training it has the same mixture target as A1 but different finite-sample bias and variance. It is a diagnostic, not an exactly equal estimator. Track temporal correlation and use episodes, not frames, as independent units for uncertainty estimates.

Primary sensitivity: `lambda / 0.09 ∈ {0.25, 0.5, 1, 2, 4}` for A0 and A1 with equal tuning opportunity. Screen the center and endpoints first, then refine around promising stable values. Evaluate A2 and A3 first at the center setting, expanding only if they change the interpretation. For A2, report the effective coefficient after accounting for `T`.

Keep `T=4` for the essential comparison. A later sample-grouping sensitivity can select one, two, or four regularization positions from the same encoded clip, using the same reference multiplier `B`. Changing history length or frame skip also changes the prediction problem and belongs in a separately labeled transfer check.

## Family B: pooled Gaussianity through a bounded normalizing flow

The encoder and predictor continue to operate in `z` space. A deterministic, shared, train-only map `g_phi: R^D -> R^D` transforms clean states solely for regularization:

```text
u = g_phi(z)
loss = prediction_MSE(z) + lambda * B * d(flatten(u))
```

Do not add flow-transformed targets to the predictor or planner. Do not condition the map on clip ID, frame position, batch statistics, or independent random noise. Such conditioning would let the regularizer obtain diversity unavailable in the encoder. Prediction and endpoint losses both backpropagate to the encoder; the endpoint term trains the map. The map remains in training checkpoints but is unnecessary for ordinary inference.

This is a normalizing flow trained by distribution discrepancy rather than maximum likelihood. Do not add a log-determinant likelihood term in the initial experiment. A flexible density estimator trained jointly on a learned encoder can accommodate contracting embeddings and does not isolate the desired mechanism.

Use bounded residual blocks `g_l(x) = x + r_l(x)` with `Lip(r_l) <= rho_l < 1`. They are invertible, with explicit lower and upper distance bounds. The total amplification is at most `G = product_l(1 + rho_l)`. Use `tanh`, a 1-Lipschitz activation, and normalize each linear matrix with the certified upper bound `min(Frobenius_norm(W), sqrt(induced_1_norm(W)*induced_infinity_norm(W)))`. Multiplying two matrices bounded by one with the block's `rho_l` gives the required residual bound. A finite number of power iterations is an estimate, not a certified upper bound; log numerical Jacobian audits separately. Avoid BatchNorm, stochastic dropout, and unconstrained affine scale inside this map.

Start with two residual blocks, hidden width `D`, total upper bound `G=2`, and near-identity initialization. Equal block limits then give `rho = sqrt(2)-1`. The total lower distance bound is `product_l(1-rho_l)`, which must also be logged. This restriction intentionally limits how much Gaussianization can distort the raw planning geometry.

Why the bound matters: if `u=g(z)` and `g` is `G`-Lipschitz, then `trace Cov(u) <= G^2 trace Cov(z)`. A nondegenerate endpoint therefore rules out arbitrarily small total source variance at a fixed `G`. This statement concerns a population variance and an actual Lipschitz bound. Finite-projection SIGReg only approximates the endpoint constraint, and the bound does not establish full-dimensional information, temporal variation, semantics, or reliable optimization.

Controls and staged sensitivity:

1. Identity map must reproduce A1 numerically, including gradients under common random projections.
2. A bounded diagonal affine map, with scales in `[1/G,G]`, tests whether simple rescaling and centering suffice.
3. The bounded nonlinear residual map tests the extra value of nonlinear distribution shaping.
4. After a stable default, vary total amplification `G ∈ {1.5,2,3}`, then residual depth `{2,4}` at fixed `G`, then width `{D,2D}`. Do not cross the full grid. Reuse the same lambda sensitivity range and equal tuning allocation as the identity reference.
5. An unbounded map is a small synthetic failure demonstration only. It is not a candidate to promote simply because its endpoint discrepancy is low.

If a flow wins only when its gain bound is relaxed, while raw variance or dynamic probes degrade, reject the interpretation that it stabilizes useful representations. If the affine control matches the nonlinear map, prefer the simpler mechanism.

## Family C: pooled continuous transport with flow-matching stabilization

Treat this as a gated experiment after the preceding controls are credible. An ordinary flow-matching regression loss alone is insufficient. For example, with all source embeddings equal to zero, interpolation to Gaussian noise has `x_s=s*epsilon`; the velocity `v_s(x)=x/s` fits the interior target velocity perfectly for `s>0`. Its singular behavior at zero prevents a regular deterministic flow from creating a Gaussian distribution from a point. A small training regression loss therefore does not certify a valid, noncollapsed endpoint transport.

Define a deterministic transport by integrating `dx/ds = v_phi(x,s)` from `x_0=z` to `x_1=u`. Time `s` is transport time, not video time. Bound the vector field's Lipschitz constant with respect to `x` by `L`, uniformly for `s∈[0,1]`. Use `L=log(G)` with default `G=2`, giving a continuous-flow amplification bound of `exp(L)=G`. The vector field may receive transport time through additive features but no observation identifier or random seed.

The encoder objective remains:

```text
L_encoder = prediction_MSE(z) + lambda * B * d(flatten(T_phi(z)))
```

There is no noise injected when calculating `T_phi(z)`. Gaussian noise appears only in the auxiliary velocity fitting task:

```text
source = stop_gradient(z)
epsilon ~ Normal(0, I), s ~ Uniform(0, 1)
x_s = (1-s)*source + s*epsilon
L_FM = mean(||v_phi(x_s, s) - (epsilon-source)||^2)
```

Implement one objective, `L = prediction_MSE + lambda*endpoint_SIGReg + beta*L_FM`, and one ordinary backward pass. Keep the actual endpoint `T_phi(z)` differentiable with respect to both the encoder and flow. Detach the source everywhere in the auxiliary FM interpolation and velocity target. The encoder then receives only prediction and endpoint gradients, the flow receives endpoint and FM gradients, and the predictor receives the unchanged prediction gradient. No duplicate endpoint evaluation or alternating optimizer cycle is necessary. Test these routes explicitly so the auxiliary cannot pull the encoder toward an easier source distribution.

The FM term guides the flow's optimization; the deterministic endpoint penalty and gain constraint carry the anti-contraction argument. Independent random pairings are a simple specified baseline, not a claim that they provide optimal transport. If their desired velocity violates the bounded model class, persistent FM error is possible and must not be hidden by weakening the bound.

Start with a fixed-step differentiable solver, eight Euler steps, and a bounded velocity MLP of width `D`. For each step size `h`, require `h*L<1`; every discrete residual update is then invertible and the implemented upper amplification is bounded by `product(1+h*L)`. Record this implemented bound rather than relying only on the continuous theorem. Compare 4, 8, and 16 integration steps on the same saved embeddings before full runs; endpoint outputs and encoder gradients must converge sufficiently to leave the model ranking unchanged. Do not use an opaque adjoint implementation before checking its gradients against explicit backpropagation.

Required controls:

| ID | Transport and losses | Interpretation |
|---|---|---|
| C0 | Identity transport and pooled endpoint SIGReg | A1 reference |
| C1 | Bounded continuous transport and endpoint SIGReg, `beta=0` | Transport architecture control |
| C2 | Identical transport plus detached-source FM auxiliary | Isolates flow-matching stabilization |

After verifying that `beta=0` exactly recovers C1, use `beta ∈ {0.1,0.3,1}`. Average FM squared error over coordinates and samples; retain the explicitly normalized endpoint SIGReg defined above. These are pilot values, not presumed comparable magnitudes. Log flow-parameter gradient norms from each term, then refine around a stable ratio if one term dominates. Keep encoder endpoint lambda tuning independent of beta. Vary `G ∈ {1.5,2,3}` only after a stable solver and auxiliary setting exist. Report vector-field evaluations, memory, parameters, and training time. The flow is discarded for inference.

## Stages, evidence gates, and implementation surfaces

All work below is future implementation by the branch owner. The common protocol owns data access, episode splitting, normalizers, held-out evaluation, run manifests, and the allocated compute launcher.

| Stage | Work | Passing evidence and next action |
|---|---|---|
| 0: arithmetic and gradient audit | Add grouping controls and a standalone transport module; run small CPU unit checks and analytical counterexamples | Exact baseline/identity parity, correct sample multiplier, no auxiliary encoder gradient, and valid gain bounds. Fix defects before any training |
| 1: pooled pilot | Profile A0/A1, screen matched lambda values, inspect A2/A3 normalization controls | Stable optimization and interpretable raw-state diagnostics. If gains disappear after normalization, record a strength effect instead of a pooling effect |
| 2: pooled confirmation | Repeat selected A0/A1 settings under shared seeds and distractor evaluation | Planning/probe improvement or a precise null result under the common uncertainty criterion. A negative result does not forbid transport feasibility, but caps its initial budget |
| 3: bounded NF pilot | Compare identity, bounded affine, and residual maps; then targeted gain/depth sensitivity | No hidden raw contraction, no substantial planner-geometry degradation, and benefit beyond affine normalization. Otherwise retain the simpler method |
| 4: bounded FM feasibility | Verify solver, gradient routes, collapse counterexample, C1/C2 matched architecture | Stable endpoint integration and an identifiable gain from the auxiliary. Failure ends the FM line at a documented negative feasibility result |
| 5: confirmation | Evaluate only the supported candidates with untouched test episodes | Report raw-space planning, factor probes, cost, and uncertainty. Combine with other branches only in a separately designed interaction study |

The initial essential screen contains eight configurations: three lambda values each for A0 and A1, plus A2 and A3 at the center. At most four local lambda refinements follow if needed. Reuse valid common-baseline runs. For flows, begin with two additional NF configurations (affine and nonlinear) and two continuous-transport configurations (C1 and C2 at `beta=0.3`). These are short stability screens, not full confirmation runs. Only supported families receive sensitivity studies: at most four initial NF extensions (two gain settings, one extra depth, one extra width), and at most four FM extensions (two beta settings and two gain settings). Solver convergence uses fixed saved embeddings before model training. Final confirmation starts with three paired seeds for each selected contrast, then follows the common precision rule. These configuration counts are staged ceilings before profiling and budget agreement, not authorization for all runs. A4, a second environment, and cross-branch combinations are optional extensions.

Affected files and concrete contracts:

- `stable_worldmodel/wm/loss.py`: preserve default `SIGReg` behavior. Expose or internally reuse the discrepancy before the sample-count factor, and support grouping plus a consistent reference multiplier. An optional explicit projection matrix in tests allows exact arithmetic comparisons without changing training defaults.
- `scripts/train/lewm.py::lejepa_forward`: select regularizer grouping; use clean `[B,T,D]` embeddings; instantiate the optional transport; log raw and transformed diagnostics. Preserve original next-state loss and encoder target gradients. The current optimizer specifies `modules: model`, so a transport attached separately to the training module will otherwise be excluded. Explicitly include every trainable transport parameter in the optimizer and test parameter-ID coverage.
- `scripts/train/config/lewm.yaml`: retain existing defaults. Add a clearly grouped experimental regularizer configuration for grouping, normalization, map type, gain, depth/width, solver steps, and FM weight. Configurations must record every nondefault setting.
- Proposed `stable_worldmodel/wm/regularization_transport.py`: implement identity, bounded affine, bounded residual, and bounded continuous maps behind `forward(z) -> u` with shape preservation. Keep the flow's gradient-routing helper explicit, without extending the rollout API.
- Proposed `tests/wm/test_sigreg_grouping.py` and `tests/wm/test_regularization_transport.py`: test the substantive numerical and gradient contracts below. Extend existing `tests/wm/test_lewm.py` only for checkpoint/inference compatibility.

The existing model export is not sufficient to resume transport training by itself. Save a full training checkpoint containing encoder, predictor, transport, optimizer and scheduler state, random-generator state, and resolved configuration. Test interrupted-versus-uninterrupted continuation. Keep a separate model-only deployment export so raw-space inference remains compatible and does not require the transport.

Necessary checks include pooled scaling by `T`, the characteristic-function decomposition, identity parameter/gradient parity, exact collapse producing positive discrepancy, near-collapse with bounded versus unbounded amplification, independent time-varying and static episode codes, finite gradients through the integrated endpoint, no FM gradient to encoder parameters, and checkpoint reload preserving the ordinary planner output. Positive loss at exact collapse does not imply a nonzero gradient there; do not write a false recovery test.

Implementation verification commands, once the named tests exist, should use the project's working Python environment:

```bash
python -m pytest tests/wm/test_sigreg_grouping.py tests/wm/test_regularization_transport.py tests/wm/test_lewm.py
```

Training invocations retain `python scripts/train/lewm.py` through the allocation-aware launcher established by the common protocol, with `wm.history_size=3 wm.num_preds=1` and the appropriate recorded experimental overrides. The new override names are not runnable until the implementation stage; generate the exact launch matrix from resolved configs rather than documenting speculative commands as working ones.

## Measurements, cost, and decision rules

Use the common protocol's planning and probe metrics as the primary decision evidence. Add:

- Raw and transformed mean, covariance trace, eigenvalue spectrum, effective rank, per-time variance, and within-episode versus between-episode variance.
- The per-time and pooled discrepancies, plus their marginal-consistency difference, evaluated with shared diagnostic projections.
- Physical-state probe errors separated by factor, temporal change magnitude, action sensitivity, and performance under static versus changing distractors. More temporal variance alone can be noise.
- Raw-space one-step and rollout prediction errors with variance-normalized reporting, goal-distance ranking against physical task distance where available, and latent trajectory curvature as a descriptive diagnostic. Low latent MSE can accompany contraction.
- Transport gain bounds and observed Jacobian singular values, parameter-gradient norms by objective, solver convergence, wall time per optimizer step, peak memory, and flow evaluations. Empirical Jacobians do not replace the bound.

Use the same data batches, environment evaluation seeds, and tuning allocation across paired controls. Screen broad settings cheaply, then repeat finalists according to pilot variability and the common precision target. One successful seed and a fixed nominal seed count are not sufficient evidence. Keep the test set untouched during selection.

There is no measured GPU-hour estimate yet. On an allocated worker, profile baseline, pooled, two-block residual flow, and eight-step continuous flow after warm-up at the actual batch size. Estimate each planned stage from measured seconds per step, training steps, repeated seeds, and evaluation cost. The flow's additional work is likely small compared with the image encoder in parameter count, but multiple differentiable solver steps and norm calculations can still be costly. Do not claim negligible overhead before measuring it. No training or substantial synthetic sweep runs on a login node.

Save configs, source revision, splits, diagnostics, and results under `experimental_results/ablations/anti_collapse/<timestamp>_<run_id>/`, following the shared protocol. Keep a table mapping claim IDs to supported, rejected, or unresolved conclusions. A successful endpoint Gaussian fit with poor raw dynamic probes is a failed representation result. A simpler pooled or affine method matching a complex flow is evidence to keep the simpler method. An FM auxiliary that does not improve over C1 at matched resources does not justify the extra optimization machinery.

## Prior work and the resulting contribution boundary

[MotionJEPA](https://arxiv.org/html/2609.23881v1) already evaluates LeWM-Flat, which flattens batch and time. Its static-distractor planning comparison is encouraging, but the reported baseline uses SIGReg weight 0.09 while Flat uses 1.0. It therefore does not isolate grouping at matched effective strength. This branch's first value is a controlled normalization and mechanism analysis, not a claim to invent pooled SIGReg.

[TC-LeWM, version 3](https://arxiv.org/html/2607.26924v3) regularizes temporal residuals after subtracting a local temporal mean and still evaluates SIGReg across the batch separately at each temporal position. Its downstream evaluation emphasizes robot policy learning. It is a distinct optional comparator for motion retention, not the same operation as flattening time and batch.

[Invertible Residual Networks](https://arxiv.org/abs/1811.00995) provides the bounded residual construction, and [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) provides the conditional velocity-regression approach. Neither establishes that jointly learned LeWM states retain meaningful dynamics under this proposed auxiliary. Novelty and empirical usefulness of the bounded transport combination remain open until a targeted follow-up literature check and experiments.
