# S3 revision: supervise what a trajectory does to a query

Status: a concrete proposed objective with CPU-verified algebra and an existing-data support audit, not an established novel algorithm or planning result. It does not modify the frozen S1 source. Generic event models, survival heads, reachability learning and trajectory preferences remain excluded as sufficient novelty claims.

## The change in prediction contract

A path should be represented by its **effect on every possible progress state of a query**, not just whether it succeeds when started from zero progress. This is the same distinction as representing a function rather than one evaluation of that function. It is useful when a later piece of behavior must compose with a preceding piece.

For the qualitative query “visit A, then later visit B,” define progress states 0 (no A), 1 (A occurred), and 2 (complete). An observation belongs to exactly one of A, B or O. Its known deterministic transition matrix is

\[
D_A=\begin{bmatrix}0&1&0\\0&1&0\\0&0&1\end{bmatrix},\quad
D_B=\begin{bmatrix}1&0&0\\0&0&1\\0&0&1\end{bmatrix},\quad D_O=I.
\]

For an observed segment w, its target is the complete matrix

\[
M(w)=D_{w_1}\cdots D_{w_H}.
\]

Only five different matrices are possible: I, A, B, AB and BA. Thus the full target is a five-class label, not a large image sequence. Its semantics is the transformation of query progress under this particular factual action sequence. No action is labeled optimal and no trajectory is declared better than another.

These five objects form a noncommutative finite monoid: composition is associative and has an identity, but inverses generally do not exist. This is classical automata theory; the research hypothesis concerns using its full transformation label to supervise an action-conditioned visual world model. Neither “discovering a monoid” nor predictive states in general is our novelty claim.

## A sample-level rule on the current data and model

1. Compute a fixed 8×8 area-pooled RGB feature ψ from each of the existing eight images. Randomly choose three distinct image indices in one clip, then randomly assign their feature vectors to prototype labels A, B and O. Do not sort the prototypes by time or pass their time indices to the model.
2. The query labels each observed image by its nearest prototype in ψ space. This is a deterministic observation-only partition with a fixed tie rule. No simulator state, contact annotation, environment success, reset, human preference or extra image is used. Duplicate prototypes are retained and reported as degenerate; they must not be quietly filtered to improve results.
3. Roll the native clean predictor forward from its permitted real prefix and the recorded actions. No future observed state is substituted into this rollout. Queries are supplied only to an auxiliary readout of that rollout, not to the dynamics predictor.
4. A small shared readout takes the ordered predicted states of a segment and its three query prototypes, and predicts the five-way class distribution \(\pi_\theta(m\mid\hat z_{i:j},q)\). Add

\[
L_{\rm query}=-\frac1{|\mathcal S|}\sum_{(i,j)\in\mathcal S}
\log \pi_\theta(M(w_{i:j})\mid\hat z_{i:j},q),
\]

where a predeclared collection of segments is used identically in both tasks: one full window and its two equal-length halves. Each segment target is computed from observed images, while its predictor input comes exclusively from the original causal rollout. The base TF/LeWM objective remains. The auxiliary readout is discarded for native planning, so observation and planner budgets stay unchanged. Extra head parameters and training computation must be counted.

**The halves are essential with this query sampler.** All three prototype frames occur somewhere in the complete window, so the whole-window class can only be AB or BA. Its full matrix then carries no extra information beyond the scalar completion label. The same is not true in the two halves, which can contain I, A or B without both events. A whole-window-only implementation would fail to isolate the claimed compositional distinction. `ORDERED_QUERY_RESULTS.md` reports the actual class support rather than hiding this degeneracy.

A direct equal-capacity control predicts the per-frame A/B/O symbols from the same predicted states and queries. A second control predicts only the scalar completion label from zero progress. These controls distinguish additional visual grounding, ordinary event classification, and the proposed compositional target. Merely winning against the no-head baseline would not isolate the mechanism.

The readout can be an explicitly ordered small MLP over a fixed-length padded segment, its binary segment mask and prototypes; that is an implementation choice, not the proposed innovation. A general sequence model is not needed. All segments use the same eight predicted states; no extra encoder observations or long online memory are introduced. Sample-level means all of these operations remain inside one logged clip, not across examples in the batch.

## What the target preserves, with exact separators

**A scalar success value is not closed under composition.** B alone and O alone both have completion 0 from progress state 0. Appending them after A yields AB (complete) and AO (incomplete). A scalar readout cannot supply the missing distinction. Their full matrices differ in row 1 and compose correctly. This is why the two rows not already absorbing must be supervised; the permanently absorbing row adds no learning information.

**Endpoint and occupation statistics are insufficient.** OABO and OBAO have the same first frame, final frame and symbol histogram, but ordered completion is 1 versus 0. The auxiliary target explicitly constrains information that an endpoint or bag-of-observations head can ignore. A sufficiently accurate full frame predictor already contains this information; this is a supervision difference, not an expressivity impossibility theorem against all existing world models.

**Composition is exact at the target level.** M(uv)=M(u)M(v) for every split. For deterministic matrix approximations with row-stochastic factors, the induced row-sum norm gives a telescoping error bound

\[
\|\hat M_1\cdots\hat M_K-M_1\cdots M_K\|_\infty
\le\sum_k\|\hat M_k-M_k\|_\infty.
\]

This controls monitor composition error for specified segment approximations. It does not bound latent rollout error or ensure a learned classifier will obey the identity. We are supervising an exact compositional label; we are not claiming that independently estimated uncertain segment distributions can always be multiplied.

**Qualitative timing invariance.** D_A²=D_A, D_B²=D_B and D_O=I. Repeating an identical hard symbol any number of times leaves the target unchanged. The target is sensitive to order and insensitive to dwell time inside a region. A fixed temporal-distance kernel changes when dwell times change; its supervised quantity is therefore different. The native dynamics objective must retain timing for control: invariance of this auxiliary target does not imply throwing time out of the state.

## An important rejected implementation: multiply marginal hazards

With probability 1/2, a future is AA; otherwise it is BB. Neither future completes A then B. Yet multiplying the per-time marginal matrices, each (D_A+D_B)/2, predicts completion 1/4. This is a false event introduced by breaking temporal dependence.

Therefore the proposed head predicts a distribution over the **five complete path transformations**, with a proper categorical loss. It does not multiply per-frame marginal event probabilities and call the result calibrated first-passage probability. Given sufficient model capacity, cross-entropy recovers P(M(w)=m | allowed prefix, action word, query) on the observed distribution. This is a statistical property of the objective, not proof that a particular small head or deterministic latent encoder can recover it.

One can still backpropagate through a soft finite-state monitor as a deterministic surrogate. The CPU file checks finite gradients for that construction, but that is a diagnostic alternative, not the proposed stochastic interpretation. The hard-symbol stuttering property also does not automatically hold for repeatedly multiplying a soft mixture.

## Both-task rationale and its limits

TwoRoom has order requirements such as moving through a passage before reaching another region. Cube has manipulation sequences in which approach, interaction, transport and release cannot be freely reordered. An observation-defined partition can expose some of these ordered transitions without naming them. Predicting the full transformation puts training pressure on the clean causal rollout to retain intermediate states needed for arbitrary progress contexts.

This is not a promise that three RGB prototypes identify doors or contacts. Arm appearance, lighting and camera changes may dominate the partition; short clips may not contain useful transitions; a monotone expert data distribution may let query-only or action-only shortcuts predict class. The small CPU real-data audit only asks whether the target is nondegenerate and distinguishes endpoint-preserving changes. It does not certify physical semantics, actual action interventions or control benefit. Shuffled words in that audit are mathematical sensitivity probes, never invented factual outcomes used for training.

As with TDWM, more retained dynamics information can make off-distribution action prediction harder. A legitimate gate must separate representation, recorded-action prediction and native planning. There is no claim that a new loss can infer outcomes of unsupported action sequences from missing evidence.

The completed episode-disjoint CPU shortcut audit (`ORDERED_QUERY_SHORTCUTS.md`, 128 train and 128 validation episodes per task) finds limited initial-state value. A fixed Gaussian-kernel readout using queries and actions achieves combined class accuracy 45.18% on TwoRoom and 48.27% on Cube; adding the initial image gives 47.49% and 49.71%. Paired episode bootstrap intervals for the increments are [+0.98,+3.68] and [+0.33,+2.57] percentage points. Linear readouts do not show a consistent increment. Training-majority baselines are 39.88% and 43.88%. This supports a small readout-dependent state contribution; it does **not** show that the proposed objective necessarily forces the world model to use its initial state. Kernel geometry and feature dimension also change when that image is added. The stronger state-forcing claim remains conditional and should not be used as the reason to launch a large experiment.

## Closest primary work and the honest novelty boundary

| Primary source | Existing result | Material distinction of the proposed rule |
|---|---|---|
| [Neural Event ODEs](https://arxiv.org/abs/2011.03902) | Learns event functions and chains hybrid dynamics. | Our model has no learned clock, mode boundary or ODE event solver. A fixed observation-query compiler supplies a complete path-transformation label to the existing predictor. Changing the query changes this label while physical event dynamics stay unchanged. |
| [C-Learning: Horizon-Aware Cumulative Accessibility Estimation, Naderian et al. 2021](https://arxiv.org/abs/2011.12363) | Learns cumulative goal reachability with an offline recurrence. | We predict the consequence of the supplied action word for all progress states, including the B-versus-O distinction. No optimal action or policy reachability target appears. Ordinary hit probability alone was explicitly rejected. |
| [PSRs and multiplicity automata, Thon & Jaeger](https://jmlr2020.csail.mit.edu/papers/v16/thon15a.html) | Observable operators and action-query predictive models have a long, unified history. | There is no general model-class novelty claim. The narrower proposal is a known finite query-transformation quotient as a factual-clip auxiliary target for an otherwise unchanged visual JEPA predictor. Classical PSRs can represent it; the exact training rule must still be audited for priority. |
| [Traj-LeWM](https://arxiv.org/html/2608.14125v1) | Learns a scalar trajectory cost using preferences and changes planning scores. | S3 predicts a factual transformation class, does not rank expert trajectories above perturbations, and keeps the native planner. Extra reset-based failure mining and success labels are absent. |
| [STL Neural Predictive Control](https://arxiv.org/abs/2309.05131) | Trains controllers to satisfy temporal specifications. | Here no action-generating controller is trained and neither satisfaction nor optimality is imposed on behavior: unsuccessful factual paths receive their correct class. |
| [TDWM](https://td-worldmodel.github.io/TD-WorldModel/) | Regresses within-clip similarities onto a temporal kernel; reports Scene experiments. | S3 targets noncommuting qualitative order and is hard-symbol stutter invariant, rather than prescribing similarity from time gaps. Both are sample-level temporal objectives, so that generic property is not a novelty claim. |

Citation clarification, 2026-10-31: Naderian et al.'s cumulative-accessibility paper above is distinct from Eysenbach, Salakhutdinov and Levine's [C-Learning: Learning to Achieve Goals via Recursive Classification](https://arxiv.org/abs/2011.08909), which estimates policy-conditioned future-state densities. These results must not be interchanged.

[WorldTest](https://proceedings.mlr.press/v306/warrier26a.html) further stresses that factual rollout queries do not identify every property of an environment. S3 is deliberately a factual path target, not an environment-level counterfactual oracle. [VQ-WM](https://arxiv.org/abs/2610.25600) already uses first-passage estimates for planning abstractions, reinforcing the rejection of “use first passage” as a sufficient new idea.

The proposal is materially different from an event-time head, one scalar success value, or expert trajectory preferences. Its foundational algebra is established, and a full prior-art novelty guarantee is impossible. It can be listed as a theory-backed proposed objective with a sharp tested property; it must not be listed as an empirically superior or conclusively original algorithm.
