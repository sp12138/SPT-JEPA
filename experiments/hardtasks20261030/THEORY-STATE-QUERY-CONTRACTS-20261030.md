# State sufficiency is relative to the questions asked of it

Research synthesis, 2026-10-30. This note incorporates the completed information, rate, channel, spatial-readout and Cube metric audits. The pooling intervention at frozen source `b3f27e6` is pending; no pooling outcome is assumed. This is a mechanism interpretation and research specification, not a new algorithm, theorem or training authorization.

## What the interventions now support

**TwoRoom has a bottleneck in the learned state path, not a demonstrated lack of spatial information in its image encoder.** At update 2000, the learned 64-dimensional QueryPool gives agent x/y probe R² of .141/.255 and .113/.291; spatial pooling followed by a fixed 64-dimensional random projection gives .958/.938 and .951/.893. Most of that advantage already exists at initialization, including substantial displacement information. This rules out output dimension alone as the explanation. It does not prove that QueryPool contains no nonlinear position information, nor that random spatial features already constitute a learned world model. These readouts differ in conditioning, scale and inductive bias. [Spatial audit](../mechanism20261030/results/spatial/RESULTS.md).

The current residual is not background-dominated: the moving region occupies about 9.4% of pixels but accounts for roughly 84–91% of high-corruption error. Removing rate does not restore the state channel in either fresh or resumed 600-update controls. In resumed runs it strengthens dependence on actions and anchor while initial-state dependence remains negligible. Removing posterior sampling noise, forcing pure target noise, or combining both also fails to recover position readout at the fresh 600 gate. Thus neither rate removal nor these channel changes is an established rescue. All fresh gates remain inside learning-rate warmup, so these are bounded negative results, not impossibility claims. [Information](../mechanism20261030/results/information/RESULTS.md), [rate](../mechanism20261030/results/rate-controls/RESULTS.md), [channel controls](../mechanism20261030/results/channels/RESULTS.md).

**Cube exposes a different failure.** Temporal-forecast predicts its own endpoint cost accurately, while that cost is strongly associated with effector distance and weakly associated with block distance after the reported partial-rank controls. Deleting the fitted three-dimensional effector-readout span leaves refitted effector R² above .994 and gives no substantial planning rescue. This intervention destroys a particular fitted map, not effector information. Privileged block-position scoring also fails to rescue temporal-forecast consistently. Therefore neither “only rollout error” nor “subtract three nuisance directions” explains the result. Block-position readout remains less accurate than LeWM under the fixed probes, although capacity and training exposure differ. [Candidate audit](../mechanism20261030/results/cube-audit/RESULTS.md), [metric intervention](../mechanism20261030/results/metric/RESULTS.md).

## The distinction: conditional usefulness versus reusable state

The renderer receives side information U: noisy target, corruption time and anchor. For a fixed state representation Z and velocity target V, the standard squared-error projection identity gives

\[
\inf_f\mathbb E\|V-f(U)\|^2-\inf_g\mathbb E\|V-g(U,Z)\|^2
=\mathbb E\|\mathbb E[V\mid U,Z]-\mathbb E[V\mid U]\|^2.
\]

State is valuable to this loss only insofar as it improves prediction beyond those other inputs. This is not generally mutual information or an exact likelihood claim. An anchor can supply the scene's spatial reference; a forecast state can supply an action-dependent change. Their combination can answer the rendering question without placing a portable initial configuration in the encoder's state. The rate intervention's stronger action/anchor dependence without stronger initial-state dependence supports this particular division of labor.

The useful abstraction is **query-relative state sufficiency**. Let h be the available observation/action history, rather than assuming one image is Markov. A future query comprises an action sequence a, horizon k and observable test function ψ:

\[
Q_{a,k,\psi}(h)=\mathbb E[\psi(O_{t+k})\mid H_t=h,\operatorname{do}(A_{t:t+k-1}=a)].
\]

Two histories may share a state only if the required query family cannot distinguish them. Unsupported actions remain unidentified from offline data; this notation does not create counterfactual evidence. The three practical contracts are:

1. **Observation:** the state preserves distinctions needed to answer the queries. Frozen probes test particular readout families, not complete information sufficiency.
2. **Transition:** action-conditioned rollout transports those distinctions correctly over the relevant horizons. A good current-state probe, or a good forecast of a dominant factor, does not certify this for other factors.
3. **Goal readout:** the deployed cost ranks predicted answers according to the specified goal semantics. Exact dynamics and an injective encoder do not make arbitrary Euclidean latent distance task-correct.

These connect directly to predictive-state representations and value-equivalence, rather than constituting a new theory. The latter makes equivalence depend on a specified family of decision queries. Our contribution at this stage is the experimental localization of different broken contracts in the two tasks. [Predictive-state representations](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html), [Value equivalence](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html).

## Why the pooling test is informative, but not a proposed theory of control

A global learned query initially averages many token contributions. Similar local content at different addresses can then produce similar aggregate responses. An invariance useful for classifying an object can conceal where it is and how an action moves it relative to a wall or contact surface. In contrast, conditional reconstruction can retain the address in its anchor or noisy image and need not recover it from the pooled state.

This is an approximate aggregation mechanism, **not exact translation invariance of the ViT**. Positional embeddings, global attention and the CLS token can all carry location. The probe evidence localizes weak accessible spatial coding after pooling; it does not establish a mathematical symmetry of the complete network.

The pending experiment preserves parameters, initial tensors, four 16-dimensional slots and the training objective. Quadrant masks give each query one contiguous quarter of patches; shuffled masks give the same number of fixed, dispersed patches. Both retain CLS, and the encoder itself still mixes globally. Consequently:

- Quadrants outperforming shuffled partitions would support a benefit from spatial organization under these budgets.
- Both outperforming full pooling similarly would support partitioning, reduced averaging or signal strength, without isolating locality.
- Better probes without better action-conditioned answers or planning would improve only the observation contract.

The masks also change attention competition and gradient routing; they do not isolate “pure locality” from every other mechanism. A successful result would establish a useful fixed imaging prior, not algorithmic novelty. Cube's mature pooling probes nearly tie, so a universal pooling explanation is already unsupported.

## Prior art and the limit of an image goal

Decision-Metric Alignment already distinguishes decodable information from a planning-compatible metric and studies candidate-ranking diagnostics. Its DA-LeWM method adds inverse dynamics and demonstration-conditioned goal-action heads. Its diagnostic distinction applies here; those policy targets are outside the primary learning comparison we have chosen. We must not relabel the distinction itself as our innovation. [Wang et al., 2026](https://arxiv.org/html/2608.18746v1).

LeJEPA's identifiability analysis gives orthogonal recovery under specified stationary Gaussian-world assumptions. Its planning result additionally requires correctly transformed action-conditioned dynamics and compatible costs; Appendix D.2 explicitly separates encoder identification from learning those dynamics. Gaussian aggregate regularization alone does not establish our three contracts in arbitrary contact environments. LeWM's spread regularization supplies pressure unavailable to a renderer that does not benefit from its state, but it does not automatically identify object-only goal semantics. [Klindt, LeCun and Balestriero, 2026](https://arxiv.org/html/2605.26379v1).

The semantic limit is unavoidable: the same interaction data and full goal image can describe either “put the block here, with any arm pose” or “reproduce the block and arm configuration.” These tasks are observationally indistinguishable without an instruction, goal-equivalence examples, reward/success information or another stated convention. Even perfect dynamics cannot identify which predicate the user intends. Knowing benchmark semantics permits transparent evaluation and oracle diagnosis; it does not turn simulator coordinates or benchmark success labels into unprivileged world-model training data. A readout improvement must be reported separately from improved world learning.

## One research direction, with a rejection criterion

A principled next specification is to require the learned state to answer **addressed future observation queries through the action-conditioned predictor**, with no alternative scene-information route. For a query q consisting of a horizon and image address, a sample-level training term could have the form

\[
\mathbb E_{(o_{0:k},a_{0:k-1}),q}
\|D(F^{(k)}(E(o_0),a_{0:k-1}),q)-\psi_q(o_k)\|^2,
\]

where ψ is a fixed observation-derived target, such as the pixels at a sampled patch, and q is sampled independently of privileged object labels. The state must retain both content and the address needed for subsequent queries; the decoder receives neither anchor pixels nor a noisy target. Recorded actions specify what happened, not what was optimal. This is a predictive-reconstruction research family with substantial precedent, not a new method merely because it is written as queries. It can still allocate capacity poorly, and earlier anchor-cut failures show that removing a bypass alone is insufficient.

The falsifiable criterion is stronger than lower reconstruction error or higher state variance: under matched budget, a candidate must improve held-out spatial/displacement readout **and** action-conditioned future-query accuracy beyond constant-state/action-ignorant controls, then reduce physical selection regret on the same nonoracle candidate bank. Simulator labels remain evaluation-only. If only the first improves, reject the claim of stronger dynamics; if only predicted-versus-encoded cost agreement improves, reject the claim of better task alignment. No extension to long training is justified by this specification alone.
