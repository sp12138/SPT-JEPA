# Sparse past in LeWM: parallel exploration plan

Status: research and implementation plan, not an implemented method or an experiment launch.

Branch: `codex/plan-sparse-past`. Base revision: `8b2e8c685dd7f11d4189f5c09d9b6bfffd5cdca4`.

This branch asks whether restricting the information available in past representations improves the representations and dynamics that LeWM learns. It preserves one-step prediction and the original SIGReg objective. The anti-collapse and multiple-future branches remain independent until each has a result that warrants an interaction experiment.

The [common protocol](common-protocol.md) owns data, splits, normalization, baselines, evaluation, uncertainty, and compute gates. No dataset or training allocation is assumed available.

## 1. Recommendation and the claims to test

Start with two inexpensive families in parallel: coordinate masking of LeWM's existing CLS representation, and masking spatial patch tokens before mean pooling. Give each its own unmasked baseline. Establish what information is actually removed, then compare mask geometry, temporal persistence, and recency allocation. Develop dense patch-to-patch prediction as a later, explicitly separate family with its own unmasked architecture control.

This order answers three different questions without conflating them:

| Claim | Evidence that would support it | Competing explanation or falsification |
| --- | --- | --- |
| SP1: incomplete past features improve a compact LeWM state | Masked CLS or pooled models improve held-out planning and state probes over their matching unmasked model | Lower latent error alone may reflect easier or collapsed targets; gains only over CLS may come entirely from changing pooling |
| SP2: the geometry and temporal pattern of information removal matter | At matched visible-token budgets, repeatable differences appear between independent masks, persistent masks, and recency schedules | A nominally better mask may simply expose more recent tokens, different batch statistics, or more total information |
| SP3: preserving a spatial grid improves control or makes sparsity more effective | Dense prediction improves over a dense unmasked control or changes the mask response consistently | Additional parameters, a stronger decoder, different regularizer grouping, or additional compute can explain a dense-versus-pooled difference |
| SP4: a useful mask discourages irrelevant appearance while preserving transition-relevant information | Controlled appearance changes impair state probes or planning less, while geometry/contact information remains decodable | Masks can erase a small moving object, conceal a static obstacle, or encourage the encoder to redundantly copy information into every token |
| SP5: sparse temporal integration can save computation | Measured predictor FLOPs, memory, and step time decrease in the dense token family | Pooling before prediction leaves the temporal predictor sequence length unchanged and does not establish this claim |

Masking and SIGReg do not guarantee causal recovery, exogenous-noise removal, disentanglement, or minimal sufficient states. Investigate these through interventions and probes.

## 2. What the inspected code currently does

At this revision, [`LeWM.encode`](../../stable_worldmodel/wm/lewm/lewm.py) runs the image encoder independently on each frame, selects `last_hidden_state[:, 0]`, applies the projector, and returns `emb` with shape `[B,T,D]`. With the default configuration, `D=192`, image size is 224, patch size is 14, and the encoder also produces 256 spatial patch tokens that are discarded by this CLS readout.

[`Predictor.forward`](../../stable_worldmodel/wm/lewm/module.py) receives `[B,H,D]`, adds temporal positions, and applies action-conditioned causal attention. There are `H=3` state tokens in the default predictor, not `3*256` patch tokens. Its existing `emb_dropout` acts after temporal positions are added; it is not an adequate controlled implementation of the masks proposed here.

[`lejepa_forward`](../../scripts/train/lewm.py) supervises each available causal prefix against its next encoded state, computes MSE over the full clean target embedding, and applies SIGReg to `emb.transpose(0,1)`. The original regularizer pools over the batch at each relative time, then averages across times. Targets are learned and differentiable; do not introduce an EMA encoder or stop-gradient in this branch.

The projector contains BatchNorm. Two separate clean and masked projector calls change running statistics and confound the intended context-only intervention. Section 7 specifies a paired implementation.

## 3. What transfers from the literature

The uploaded SPT paper, Sections 3.2 and 4.3, drops rows of full-width features after its within-chunk encoder, independently across chunks. Retained rows may already contain information from the entire chunk. Its causal transformer processes the surviving rows and its future target stays dense. This motivates post-encoder token selection, but it does not make coordinate dropout equivalent to token removal. Its mask-ratio ablation uses a different causal ViT-DiT architecture from the main split encoder and evaluates after dense-history, next-chunk fine-tuning. Consequently, its best 90% rate is a useful high-sparsity probe, not a default to copy uncritically. Source: supplied *Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal Histories*, Sections 3.2, 4.3 and Figure 3.

The literature does not establish that every good mask must consist of contiguous blocks:

- Original MAE reports that high-ratio random patch sampling worked best in its image experiments; block masks became harder and degraded at its higher masking ratio. [MAE, Section 4.1](https://arxiv.org/html/2111.06377v3).
- VideoMAE repeats a random spatial mask across the temporal axis. Its tube versus random versus frame comparison supports temporal persistence for its video reconstruction setting, not a universal requirement for contiguous spatial rectangles. [VideoMAE, Sections 3.3 and 4.2, Table 1b](https://arxiv.org/html/2203.12602v3).
- V-JEPA uses unions of large spatial blocks extended through the clip and studies spatial coverage, temporal coverage, and block count. This is strong motivation to test blocks, although its bidirectional masked-video task differs from causal action-conditioned next-state prediction. [V-JEPA, Section 4.4 and Appendix E.4](https://arxiv.org/html/2404.08471v1).

Dense latent prediction and recursive planning have precedents in [DINO-WM](https://arxiv.org/html/2411.04983v2), which uses pretrained visual patch features. The local PreJEPA implementation also carries `[B,T,P,D]` tensors and block-causal attention. Neither establishes the outcome of end-to-end LeWM training with sparse contexts.

Coordinate truncation and full-target prediction overlap with [ALeWM](https://arxiv.org/html/2609.32921v1), which learns ordered prefixes and uses MixSIGReg. Object and trajectory masks overlap with [Causal-JEPA](https://arxiv.org/html/2602.11389v2). Broad masking novelty claims are unsupported. The prospective contribution is controlled evidence about information availability through time and its effects on end-to-end planning.

## 4. Three representation families

Use `rho` for the fraction hidden. The clean encoder and target path always sees the complete image in the initial experiments. Mask only predictor context. Never apply the mask to the target or to the inputs of SIGReg.

### C: CLS coordinate masking

Keep the existing clean state `z_t` with shape `[D]`. Apply `z_ctx = mask * z` before predictor temporal positions and action conditioning. The model still receives a width-D vector; hidden coordinates are zero, rather than physically removing columns and changing linear-layer dimensions. Start without inverted-dropout scaling, since that otherwise changes the retained values by `1/(1-rho)`. Record this choice and run a scaling control only if the result depends strongly on amplitude or train-test mismatch.

The predictor must estimate all D coordinates of the next clean state. Coordinates have no established spatial adjacency, so a contiguous interval of coordinate indices is not a spatial block. Ordered-prefix masks are an optional comparison to ALeWM, not a substitute for the requested random-coordinate test.

Initial masks are exact-count, uniform, independent across history states. At a surviving rate, compare independently sampled coordinate sets with one shared set across the clip. Actions and positional embeddings remain intact. The regularizer uses clean `[T,B,D]` exactly as in the common baseline.

### P: spatial selection followed by mean pooling

Let `E_t` have shape `[P,D_encoder]` and contain patch tokens, excluding CLS. Define the clean state and corrupted context as:

```text
full_pool_t = mean(E_t[p] for every patch p)
masked_pool_t = mean(E_t[p] for visible patches p)
z_t, z_ctx_t = shared_projector_pair(full_pool_t, masked_pool_t)
```

Use the mean over retained tokens, not a zero-filled sum divided by the original patch count. No empty retained set is permitted. Full-patch mean pooling with the same projector and predictor is the mandatory P0 control. A masked pooled model is compared with P0 before comparing it with CLS.

This family preserves LeWM's `[B,H,D]` predictor and ordinary rollout. It tests whether a state assembled from a restricted set of encoded spatial features trains a better encoder and predictor. It does not reproduce SPT's sparse multi-token temporal integration or its compute savings. Because the image encoder already mixed patches, hiding a token row is not equivalent to removing every visual detail from that location.

Training loss remains next-clean-state MSE plus clean-state SIGReg. At inference, compute the full spatial mean with no mask. Initially there is no dense-history adaptation stage, no CLS-plus-pool hybrid, and no simultaneous future-horizon change. If full-context inference is materially worse than matched masked-context inference, diagnose the distribution shift before deciding whether a separately labeled adaptation experiment is useful.

### D: later dense patch-to-patch prediction

The clean state is a complete grid `Z_t` with shape `[P,D]`. Project patch features with shared weights. Sparse context consists of the selected patch rows plus their original spatial and temporal positions. The predictor returns the complete next grid, including positions absent from the history.

A suitable contract is a block-causal context transformer followed by P spatial target queries for each origin. Each context frame j supplies its outgoing action embedding `a_j` to its retained patch tokens before the action-conditioned context attention, preserving the original historical action inputs. A query at origin i may attend only to context states through i, and uses the outgoing action block a_i. It predicts `Z_(i+1)` without seeing its encoded target. Start with the existing six-layer context-transformer dimensions and a two-layer cross-attention query decoder at width D; profile this definite starting design before changing capacity. Batched queries for different origins must be isolated from one another or masked so later-origin information cannot leak backward. Never renumber surviving patch positions as if they formed a smaller contiguous image.

The all-kept D0 model includes this same decoder and parameter count. Compare sparse dense models to D0. Pooling versus dense results are architecture comparisons, not a pure masking ablation.

Recursive inference is well-defined: append the predicted complete grid, truncate to the latest H grids, and repeat. Goal encoding produces the same grid shape, and the planning cost is mean squared error over patches and embedding coordinates. The practical costs are larger predictor sequences, decoder queries, and candidate-rollout memory. Sparse context with dense output has a complete feedback state; there is no inherent obstacle to recurrence.

For this family, apply the original SIGReg construction independently to each `(relative_time, patch_position)` group across B, then average over those groups. This is `SIGReg(rearrange(Z, 'b t p d -> (t p) b d'))`, not flattening time or patches into the sample axis. It preserves the per-group objective and coefficient, but introduces more groups. Report this explicit extension when comparing with pooled representations. D0 and all dense masking arms must use exactly the same extension. The dense branch does not test alternative anti-collapse objectives.

Profile the regularizer as well as attention: at the defaults, its projection-frequency tensor has `4*256*128*1024*17`, about 2.28 billion entries, over 9 GB for a single float32 tensor before autograd intermediates. Implement exact averaging in bounded groups with activation checkpointing and the same sampled projection directions across groups and recomputation. Merely summing chunk losses and calling backward at the end can retain every chunk's graph. Verify chunked loss and gradients against the full calculation on a small fixture. Do not silently reduce projections, sample patches, or change batch size only in masked arms to solve memory pressure.

Reuse grid layout and block-causal masking ideas from [`PreJEPA`](../../stable_worldmodel/wm/prejepa/prejepa.py) and [`generate_mask_matrix`](../../stable_worldmodel/wm/prejepa/module.py). Do not inherit PreJEPA's detached image features; gradients must reach LeWM's image encoder through prediction and regularization.

## 5. A mask specification that separates the requested axes

A sampler returns a Boolean mask, original token indices, retained counts per state, and the realized total budget. Record a visualization and these counts for every configuration before training. The main mask is independent of image content and never sees the prediction target.

| Axis | Initial and follow-up values | Question answered |
| --- | --- | --- |
| Hidden fraction | C: `0, .25, .50, .75`; P: `0, .25, .50, .75, .90`; dense rates chosen after a profile, including `0` | Does information restriction help, and where does evidence become insufficient? |
| Spatial geometry, P and D | Individual patches; contiguous `2x2` tiles; contiguous `4x4` tiles on the default `16x16` grid | Does a local spatial hole matter beyond the number of visible features? |
| Temporal persistence | Independently resampled each state; one mask shared through H states; intermediate runs of length 2 | Can the model recover the same feature from another state, and does preventing this help? |
| Recency allocation | Uniform; recent-heavy and old-heavy with two strengths each | Is equal allocation optimal, or is one end of history more useful? |
| Mask location in computation | Post-encoder first; pre-encoder only after a separate feasibility/control stage | Is any effect caused by temporal access restriction or actual visual occlusion? |
| Whole-state temporal gaps | Optional dedicated experiment after feature masks; contiguous versus dispersed missing states with sufficient history | Does losing an entire interval differ from losing the same amount of information spatially? |

For tile masks, select entire tiles uniformly. Use budget levels divisible by the tile area so geometry is not confounded by a partial random remainder. For example, `.25`, `.50`, and `.75` are exactly representable on a 16x16 grid for both tile sizes. If `.90` wins the first screen, first compare it with the nearest agreed exactly representable budget before the tile study; do not silently call rounded unequal counts a matched-budget comparison. A union-of-rectangles V-JEPA-style sampler can be a later geometry check; record its realized coverage rather than equating nominal block area with hidden fraction.

For the intermediate temporal length, sample a phase for length-2 segments on a virtual timeline and intersect the resulting segments with the H-state clip. Draw a new spatial mask per segment. This avoids always assigning the middle or last state a special mask boundary. Whole-clip tubes and fully independent masks are explicit endpoints, not accidental consequences of this phase scheme.

For recency schedules, index the oldest to newest context states by `t=0,...,H-1`. Set relative keep weights `w_t = 1 + alpha * (2*t/(H-1)-1)` with `alpha` in `{-0.75,-0.5,0.5,0.75}`. Allocate counts in units of the chosen tile area. Solve for a scale c such that `sum_t clip(c*w_t, 1, units_per_grid)` equals the fixed total visible-unit budget, then apply largest-remainder rounding only among states with remaining capacity. This redistributes excess after saturation and preserves the total exactly. Do not normalize first and then cap without redistribution. At H=3 and alpha=.5, unsaturated relative budgets are `.5:1:1.5`; negative alpha reverses this. Require at least one retained unit per state and log all realized counts and saturation. Start with independent masks to isolate count allocation. A later persistent-rank version uses shared random ranks and selects each state's budget from them; its nested supports are not identical tubes when budgets differ.

Because LeWM supervises three causal prefixes in one pass, this initial schedule is relative to the full clip. It gives the final origin a recent-heavy history, but does not recenter the schedule for every earlier origin. Report retained counts and errors per origin; matched total input tokens do not imply matched cumulative exposure across prefix queries. If a schedule is promising, confirm it using separate origin queries with the schedule recentered and the visible budget matched within each prefix. Use an origin-isolated uniform control and gather all outputs before the shared prediction projector so its BatchNorm grouping does not change.

At H=3, whole-frame removal offers few distinct geometries and easily confounds gap length with which state remains. Do not claim a temporal-block law from that setting. If this question remains important, introduce an explicit H=6 sub-study, including a matched H=6 no-mask model and equal-count contiguous versus dispersed frame gaps. Represent an absent frame with a designated missing-state token while retaining timestamps and outgoing action slots; otherwise compressing the sequence incorrectly makes a multi-step gap look like a one-step transition. Forbid an entirely empty clip in the main experiments. An action-only endpoint, if used, is a separate diagnostic.

Explore these dimensions in stages, not as a Cartesian product.

## 6. Staged experiment matrix and decisions

One pilot seed screens configurations for correctness, collapse, and obviously uncompetitive behavior. It does not establish efficacy. Finalists and their exact controls receive paired repeats under the common uncertainty protocol. Use validation episodes for selection and reserve the held-out test set for the selected comparisons.

| Stage | Configurations and work | Passing evidence and next action |
| --- | --- | --- |
| S0: faithful controls | Reuse validated common CLS baseline C0; establish P0 full-patch mean; verify the mask code is an identity at rho=0 | C0 reproduces the common baseline and P0 is stable. If either fails, fix implementation or explain the representation change before interpreting masks |
| S1: information amount | Three C rates `.25,.50,.75`; four P rates `.25,.50,.75,.90`, initially independent masks | Compare each family to its own control on planning, state probes, and collapse diagnostics. Keep at most two informative rates per family. A negative result can still justify a mechanistic diagnostic, but not a broad tuning expansion |
| S2: geometry and persistence | At one viable, exactly representable P rate: individual, 2x2, and 4x4 masks crossed with independent and whole-clip persistent masks, six configurations with existing arms reused; add length-2 persistence only for the most informative geometry | Establish whether budget-matched geometry or persistence matters. If nothing separates beyond noise, retain the simplest independent mask rather than multiplying shapes |
| S3: where history information is allocated | At the selected P geometry and budget: alpha `-.75,-.5,.5,.75`, plus the existing uniform arm | Keep only schedules that survive paired confirmation. If newest-state dominance explains the result, report that result rather than attributing it to a general causal prior |
| S4: coordinate interpretation | At the most informative C rate, compare independent versus shared-across-time coordinate masks; test the alternative amplitude convention if latent-norm diagnostics implicate scaling | Determines whether coordinate robustness or a particular dropout convention explains the CLS result; do not infer spatial selectivity from coordinate masking |
| S5: dense family, conditional on resources and information value | D0, one independently masked dense model, and one informative structured mask at equal budgets; then two neighboring rates only if needed | Verify full-grid rollout and planning cost first. Promote dense only if benefit or compute-quality tradeoff exceeds its extra complexity. A pooled negative result does not logically rule out dense sparsity |
| S6: mechanism and generalization | Selected models on controlled appearance/occlusion changes, action permutations, temporal-order checks, and a second available environment; optional H=6 temporal-gap and pre-encoder studies | Determine what information was retained and whether the finding transfers. Report narrow evidence if only the initial task improves |

The first decision uses nine configurations including the reusable C0 and the P0 control. S2 adds at most six new configurations when the independent-patch arm is reused, and S3 adds four. Subsequent stages are conditional, not a commitment to train every listed variant. No full run count or GPU-hour promise can be made before data access, throughput, and the resource ceiling are established.

For every promoted comparison, use the same training updates, sampled episodes, optimizer schedule, augmentation, prediction origins, and evaluation task list. Report equal-example results first, then a separate equal-compute comparison where dense sparsity materially changes step cost. Do not give one family a much larger tuning budget and compare only its best checkpoint with an untuned baseline.

A lower training prediction loss is not a promotion criterion. The primary decision follows the common planning metric and uncertainty criterion, with frozen state probes and noncollapse diagnostics as necessary supporting evidence. Record all attempted settings so an apparent gain can be interpreted against the search performed.

## 7. Implementation contract and code seams

These are proposed additions, not commands or configuration keys already available in the repository.

| File or surface | Required change | Constraint |
| --- | --- | --- |
| [`stable_worldmodel/wm/lewm/lewm.py`](../../stable_worldmodel/wm/lewm/lewm.py) | Expose clean CLS and patch encoder features; select `cls`, `patch_mean`, or later `patch_grid`; provide clean and masked context states from one image-encoder pass | Keep existing `emb` and rollout behavior backward compatible for compact states; no future frame enters a past image encoder |
| [`stable_worldmodel/wm/lewm/module.py`](../../stable_worldmodel/wm/lewm/module.py) | Add a paired projector helper and, only for D, an explicitly grid-aware predictor/query decoder | Apply masking before temporal positions; preserve action order and original spatial indices |
| Proposed `stable_worldmodel/wm/lewm/masking.py` | One sampler for exact budgets, geometry, temporal persistence, and recency counts | Return masks and metadata, not a new training framework; deterministic under the recorded generator seed |
| [`scripts/train/lewm.py`](../../scripts/train/lewm.py) | Build corrupted context from clean encoder features; predict the full next target; log mask metadata and representation diagnostics | Original next-state indexing and original clean SIGReg remain unchanged for C/P |
| [`scripts/train/config/lewm.yaml`](../../scripts/train/config/lewm.yaml) and proposed experiment configs | Add explicit representation and masking fields with identity defaults; keep run configurations resolved in outputs | `wm.num_preds=1`; alternative horizons belong to the other branch |
| Existing planning and rollout surface | For compact models, no mask in ordinary inference; for D, preserve grid state and reduce cost over P and D | All models use the same explicit context length and task manifest from the common protocol |

Suggested compact configuration fields:

```yaml
representation: patch_mean
context_mask:
  enabled: true
  domain: patch
  fraction: 0.5
  geometry: individual
  tile_shape: [1, 1]
  temporal: independent
  segment_length: 1
  recency_alpha: 0.0
  location: post_encoder
  coordinate_rescale: none
```

For the pooled pair, each BatchNorm layer computes its moments from the clean full-pool path, applies those same moments and affine parameters to both paths, and updates its running statistics once. Keep the moments differentiable so the all-kept paired computation can match the original output and gradients. This makes clean targets and clean SIGReg independent of which context mask was sampled for a fixed encoder and batch. Do not approximate this by switching the masked pass to running-stat evaluation during early training. If the paired implementation cannot pass the identity and statistics checks, resolve that before launching the P series; silently replacing BatchNorm with LayerNorm would introduce another ingredient requiring its own baseline.

For dense projection, the sample grouping of projector BatchNorm necessarily differs if patches are projected individually. Keep this choice fixed in D0 and every D arm, record it, and avoid interpreting a dense-versus-pooled difference as exclusively a token-count effect. Dense prediction also needs explicit limits on batch size and candidate-rollout memory established by profiling, not guessed from compact LeWM.

### Required checks before any training matrix

1. All-kept C/P outputs, scalar loss, parameter gradients, and BatchNorm running statistics match the corresponding unmasked implementation within dtype-appropriate tolerances. A mask change does not alter the clean target or SIGReg on a fixed batch.
2. Mask counts match the requested budget; different clips receive independent masks; tube, independent, and segment modes exhibit their specified temporal correlations. Check marginal coverage near the beginning and end of a clip.
3. With action and position tensors held fixed, only the intended context values or rows change. Deliberately changing a future observation must not alter a past prediction. For numerical causality checks, freeze normalization statistics so batch-wide training normalization does not obscure attention-mask correctness; separately document normalization's existing cross-sample coupling.
4. Gradients reach the image encoder through both allowed learning paths. Neither PreJEPA's detach nor an accidental teacher encoder is inherited.
5. An entirely removed spatial set fails fast; exact fractions outside `[0,1)` are rejected except for a separately implemented action-only endpoint. Recency saturation and rounding are logged.
6. Dense outputs always restore all P positions in the same order. A two-step rollout accepts its own first prediction, produces finite action gradients where needed, and agrees with manually composed one-step prediction in evaluation mode.
7. Run one tiny allocated-GPU overfit case and a short profile before scaling. Record examples/second, peak memory, encoder/predictor time, effective batch, and actual mask counts. Follow the cluster rules and common guarded launcher; do not train on the login node.

## 8. Evaluation that tests the intended information loss

Follow the common planning protocol with full clean histories as the primary deployment condition. Use fixed masked histories only as an additional robustness diagnostic.

Report the common planning, probe, forecasting, collapse, and resource diagnostics. For dense models, include pooled and fixed-capacity spatial readouts so a stronger probe is not confused with a better representation.

Use dynamics-relevant and appearance-only interventions separately where the environment supports them. A changed texture or background should leave the physical state and action transition unchanged. An obstacle, object orientation, contact geometry, or goal marker may be static yet necessary for prediction or planning and must not be treated as disposable noise. If controlled rendering or ground-truth states are unavailable, mark the corresponding mechanism claim untested rather than substituting ordinary random augmentation as causal evidence.

Store configurations, revisions, data hashes, seeds, mask images, diagnostics, selection records, and evaluations in the common `experimental_results/` layout. Record what resource budget each comparison matches. Measure pooled-mask cost rather than promising savings.

## 9. Stop conditions and handoff

The branch owner implements and checks; the evaluation owner applies the common protocol; the research owner reviews expansion decisions. No person or compute allocation is assumed assigned.

Stop for invalid targets, broken causal access, collapse, normalization confounds, or unusable dense rollout. Do not expand a sweep on latent-loss improvement alone. Estimate replication needs before expanding inconclusive comparisons.

Deliver separate C and P result tables, mask visualizations, and cost profiles first. Resolve geometry, temporal persistence, and recency next; assess dense prediction separately. Propose integration with other branches only after understanding these comparisons.
