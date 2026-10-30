# Four representation/optimization proposals and their admission audit

2026-10-30. Independently derived in response to the request for substantially different ideas. The preceding ten ideas in `experiments/ideas20261028/IDEAS_DRAFT.md` are excluded. Every learning target below comes from observations and recorded actions; simulator coordinates occur only in the explicitly separated frozen-data audit. No GPU, model training, model-code edit or long-run recommendation is part of this note.

**This is not a list of four accepted novel algorithms.** Deriving a plausible mechanism and then finding that it is known or violates the data assumptions is a result worth retaining. None has demonstrated a benchmark improvement. Novelty assessments are hypotheses from a targeted primary-source search, not exhaustive priority claims.

| Draft | Essential mechanism | Current admission decision | Evidence |
|---|---|---|---|
| A | Encode action-response coefficients after removing command-only response | Original form rejected by exact reparameterization; revised form held because action exogeneity fails and PSR prior art is close | Algebra + executable counterexample + real 256/256-episode audit |
| B | Grow the state dictionary using unrepresented action-pullback observables | Rejected as an original algorithm: direct Krylov/model-error-basis antecedents | Algebra + CPU check + primary-source collision |
| C | Open a dormant state/readout channel through a paired negative-curvature step that lowers the original objective | Retain as a focused optimization experiment, not a new anti-collapse theory | Exact local calculation + CPU check; real mixed Hessian unmeasured |
| D | Represent spatial addresses by amplitude-free annihilating-polynomial coefficients rather than amplitude-weighted moments | Retain as a higher-risk representation hypothesis; image-measure assumptions unresolved | Exact sparse-measure calculation + CPU conditioning/ambiguity failures |

The surviving questions concern three different properties: which response functions a state should represent (A's revised specification), whether the existing objective has an accessible information-bearing descent direction (C), and whether visual amplitude should determine spatial-coordinate scale (D). Only C and D currently justify a new bounded diagnostic. These are not three validated learning methods.

## A. Command-residual response coordinates

### Initial design and an exact reason to reject it

Let X be an initial image/history, A a recorded action sequence, Y a fixed observation-derived future query, and B(a)=E[Y|A=a]. A natural proposal is to predict R=Y−B(A), hoping to reserve state capacity for effects that need the initial scene. For a free action-conditioned decoder, however,

\[
\inf_D E\|Y-D(E(X),A)\|^2
=\inf_{D'} E\|Y-B(A)-D'(E(X),A)\|^2,
\quad D'=D-B.
\]

This equality holds for every fixed encoder if the decoder class is closed under that subtraction. The set of optimal encoders is unchanged. It can alter finite-network optimization, but it is not a new information-selection principle. Removing A from the decoder is not an automatic repair: the realized residual Y−B(A) can depend on the path used to reach an identical endpoint, so it need not be a function of a canonical endpoint state.

### A more concrete, distinct specification

Instead encode a current state's response function. Under an explicitly exogenous reference action distribution q, choose fixed orthonormal action functions φ_j, including φ_0=1, and fixed image queries ψ_l. Define

\[
e_{jl}(x)=E_{A\sim q}[\phi_j(A)\{E[\psi_l(Y)\mid X=x,A]-B_l(A)\}].
\]

One observed transition gives the target

\[
t_{jl}=\phi_j(A)\{\psi_l(Y)-\widehat B_l(A)\},\qquad
L_{\rm response}=\sum_{jl}(E_\theta(X)_{jl}-\operatorname{sg}(t_{jl}))^2.
\]

With independent exogenous actions, the correct fixed B, and squared regression, the population optimum is e(x); the constant encoder pays the variance of e(X) above the Bayes regression risk. No learned target encoder can collapse together with the student because the targets are fixed functions of recorded observations/actions and a separately fitted nuisance baseline. A transition model would then predict e(X') from e(X),A. This last Markov-closure property is an additional requirement, not implied by fitting finitely many coefficients.

For Y=(2A, X+AX), X and A independent equiprobable ±1, φ=(1,A), the command-only factor is removed and both retained response coefficients equal X. The executable check recovers these values exactly. In additive navigation, the zeroth coefficient can retain initial position; in a contact system, higher coefficients can retain configuration-dependent response. This gives a reason to investigate both tasks, not a guarantee of either. The query family and horizon can miss important variables; an interior position in a translation-invariant region may be invisible to derivative-only responses.

For comparison with the independently proposed within-clip action annihilator: global centering retains `s0−E[s0]+(b(s0)−E[b])U` in a trajectory `s0+b(s0)U`. Projection orthogonal to `[1,U]` removes that entire trajectory. The nullspaces and intended information differ, although both belong to residualization and should not be counted twice merely for implementation differences.

### The real data reject the convenient assumption

If A=X and Y=X, then B(A)=Y and all residual targets vanish despite X being indispensable under another action policy. This failure is not repaired by more accurate baseline fitting. Importance correction would require a reference policy with adequate support and a justified behavior-density estimator; it is not supplied by the original proposal.

`audit_action_confounding.py` uses the exact `BenchmarkWindows` episode/window indexing, 256 training and 256 disjoint validation episodes per task, and one sampled window per episode. It loads actions and physical labels only, never images or model weights. Standardization, ridge penalty and RBF bandwidth use training inputs only. Labels are diagnostic targets, not training inputs. The fixed RBF action-sequence probe gives:

| Task / diagnostic quantity | x R² | y R² | z R² |
|---|---:|---:|---:|
| TwoRoom initial agent position | .482 | .491 | — |
| Cube initial block position | .045 | .164 | .950 |
| Cube initial effector position | .062 | .119 | .935 |
| Cube block displacement | .644 | .835 | .964 |
| Cube effector displacement | .837 | .931 | .920 |

Thus future action sequences carry substantial initial-scene information. This is enough to falsify the exogenous-action premise on these windows. It does not identify causal effects or prove that every residualization is harmful. A linear full-sequence fit overfits some Cube coordinates at this sample size; the RBF result prevents treating that linear failure as independence. Constant-coordinate R² is recorded as null. Full results and episode IDs are in [the audit JSON](ideas10-action-confounding-20261030.json).

### Closest primary work and status

1. [Predictive State Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2017/hash/2bb0502c80b7432eee4c5847a5fd077b-Abstract.html) already learns states as predictions of future features and uses two-stage regression. Response coefficients are a finite representation of a conditional predictive function, so this is a serious conceptual overlap, not an unrelated citation.
2. [Double/debiased machine learning](https://academic.oup.com/ectj/article/21/1/C1/5056401) supplies residualization and cross-fitting ideas. Our simple residual target is not automatically a Neyman-orthogonal score and has no DML robustness guarantee.
3. [ResWM](https://arxiv.org/abs/2603.11110) changes actions to increments and encodes observation differences. That is functionally different from removing E[Y|A] or identifying response coefficients, but the distinction alone does not establish novelty.

**Decision:** reject the original objective as an information-theoretic innovation; hold the revised response-state proposal. The action-conditional score/Stein alternative being examined separately by root is a further derivation, not validated by A's failed assumptions.

## B. Residual observability closure

Start with an immutable observation query bank Ψ_0(x). For action a, define the conditional transition operator T_aψ(x)=E[ψ(X')|x,do(a)]. A representation that contains Ψ and closes under every relevant T_a can answer the associated multi-step queries. A direct constructive proposal is

\[
r_{a,j}=T_a\psi_j-\Pi_{\mathcal V}\,T_a\psi_j,
\qquad \mathcal V_{new}=\mathcal V+\operatorname{span}\{r_{a,j}\}.
\]

In a linear system x'=A_a x and initial observed query Cx, the minimal linear closure is the span of rows of `C A_{a_k}...A_{a_1}`. Induction proves sufficiency for every covered action word. A deterministic three-coordinate cyclic system grows from rank 1 to rank 2 to rank 3 in the CPU check. A constant dictionary has zero closure residual but fails the fixed observation query, proving why closure alone does not prevent collapse.

A practical sample-level version would fit residual query teachers using held-out logged transitions, append or replace dictionary coordinates, and train the predictor to transport them. This aims to preserve wall/contact distinctions that become visible only after actions, rather than every appearance detail. Cube and TwoRoom both motivate such delayed observability. But finite action coverage, history sufficiency, conditional-expectation estimation and a finite state budget remain essential. Small one-step closure residual does not by itself certify arbitrary long-horizon queries.

The close work is decisive:

1. [Parr et al., 2008](https://ics.uci.edu/~dechter/courses/ics-295/fall-2019/papers/2007-parr-icml08.pd.pdf) explicitly studies model-error basis functions and relates them to Krylov and Bellman-error constructions under the same initialization. This directly covers the supposed novelty of using residuals to grow a predictive feature basis.
2. [ResKoopNet](https://arxiv.org/abs/2501.00701) learns dictionaries by spectral residual minimization; neural dictionaries and residual-driven operator learning are already developed.
3. [Residual-Guided Dictionary Learning](https://arxiv.org/abs/2606.29083) combines operator residuals with conditioning penalties. Adding a condition-number audit is therefore not a new distinguishing mechanism either.

**Decision:** reject B as a new algorithm. Retain observability closure as a useful specification and existing-method reference. A deployment-specific fixed-budget dictionary replacement would need a genuinely different mechanism and comparison, not just a renamed residual loop.

## C. Paired opening of a dormant encoder/readout channel

The observed TwoRoom failure survives removal of rate and posterior noise at the measured gate. That does not prove that no information-bearing descent direction exists: encoder and readout can each need the other to move first. Restrict the *original priced loss* to an encoder direction e and a conditioning-readout direction w. A local model is

\[
J(e,w)=C-2\eta ew+\lambda e^2w^2+\sigma^2w^2+\tfrac12\beta\lambda e^2,
\qquad
H_0=\begin{pmatrix}\beta\lambda&-2\eta\\-2\eta&2\sigma^2\end{pmatrix}.
\]

At (0,0), each first derivative is zero; each isolated coordinate move increases this objective. Nevertheless, a joint direction has negative curvature iff `2η² > β λ σ²`. A sufficiently small step along that direction decreases the same objective. The strict inequality follows from the determinant with positive diagonal; the descent claim follows from the second-order expansion with bounded higher-order remainder. At a nonstationary checkpoint the linear term must also be evaluated, so a negative eigenvalue alone is not a step-acceptance rule.

**Proposed bounded mechanism:** restrict Hessian-vector products to the existing encoder output projection and renderer conditioning bridge; find a joint negative-curvature mode; test both signs and a small step on the unchanged sampled loss using identical random draws; accept only an actual decrease. Require the direction to retain its sign of curvature on a disjoint diagnostic set before claiming it reflects a persistent missing channel. No state-variance reward, gradient deletion, parameter expansion or changed rate is introduced. This differs from old I06, which modified encoder gradients to protect an existing first-order reconstruction direction. It is an optimizer experiment with additional compute, which must be matched.

The check with λ=1, σ=.1, β=.25, η=.4 has eigenvalues −.6732 and .9432. A paired step changes loss 1→.996658; its encoder-only/readout-only components give 1.000536/1.000057. With η=.001 both eigenvalues are positive, and this mechanism offers no escape. A partially learned Cube model could have analogous dormant object-response directions, but the real mixed Hessian has not been measured. If Cube has no such mode, this proposal has no demonstrated cross-task rationale.

Closest primary work:

1. [Saddle-free Newton](https://papers.neurips.cc/paper_files/paper/2014/hash/04192426585542c54b96ba14445be996-Abstract.html) establishes the general second-order saddle-escape motivation. Negative curvature is not our invention.
2. [Residual spectral instabilities in representation learning](https://arxiv.org/html/2610.11257v1) derives coupled encoder/decoder reactivation in collapsed VAE sectors through a conditional residual operator. This is extremely close to the mechanism, even though it does not implement the specific AS-JEPA bridge diagnostic or prove properties of its flow/rate objective.
3. [Lagging inference networks](https://arxiv.org/abs/1901.05534) addresses optimization-induced collapse through inference-network optimization. A paired Hessian step is functionally different from extra inference updates, but both address a known optimization problem.

**Decision:** retain only as a mechanism-discriminating optimizer control. A successful original-loss decrease is not sufficient for a novel algorithm claim, semantic recovery or planning superiority. Reject it as a cure if improvement is merely noise-fit, if real channels are locally stable, or if positive-control probes/selection regret do not improve.

## D. Spatial addresses from annihilating polynomials

### Why this differs from merely Fourier-pooling the tokens

Ordinary spatial Fourier moments retain address but weight it by appearance amplitude. To separate these properties, suppose a one-dimensional feature measure has K distinct atoms,

\[
\mu=\sum_{j=1}^K w_j\delta_{p_j},\quad
m_k=\sum_j w_j q_j^k,\quad q_j=e^{ip_j},\quad w_j\ne0.
\]

Its monic annihilating polynomial

\[
P(z)=\prod_j(z-q_j)=z^K+c_1z^{K-1}+\cdots+c_K
\]

satisfies a linear recurrence in the moments. With 2K noiseless moments and nonsingular Vandermonde/Hankel factors, the nodes and coefficients are identifiable. Let `H0[i,j]=m[i+j]`, `H1[i,j]=m[i+j+1]`. Writing `H0=V diag(w) Vᵀ` and `H1=V diag(w) diag(q) Vᵀ` shows that the generalized eigenvalues are q_j. This is classical Prony/finite-rate-of-innovation algebra, not a new theorem.

The proposed state coordinates are **c rather than m or the mass-weighted centroid**. They are unchanged by any nonzero reweighting of the same atoms. For K=2, `c1=−q1−q2`, `c2=q1q2`; the norm of the derivative with respect to either position is √2, independent of w. Under a common translation δ, `c_l' = exp(i l δ)c_l`, so address transport has an exact low-dimensional action. Coefficients avoid sorting roots when objects exchange order, although root collisions remain ill-conditioned. With two atoms weighted 100:1, the CPU check finds ordinary moment sensitivities 3.7417 and .03742, versus coefficient sensitivities 1.4142 and 1.4142.

This suggests a different spatial representation hypothesis: a small object should not receive a smaller coordinate merely because it contributes less visual mass. TwoRoom could retain the agent's position despite nearly unchanged global statistics; Cube could retain the block alongside the visually dominant arm. This does **not** determine whether the arm should count in an object-only goal, and 2D projected addresses do not automatically identify 3D pose or contact state.

### Concrete learning specification and its unresolved prerequisite

A possible state extractor would map an image to a small number of nonnegative feature measures, compute spatial moments in fixed directions, and obtain regularized polynomial coefficients. An action-conditioned predictor transports those coefficients. The sample-level supervision would combine an observation-anchored moment consistency term and multi-step coefficient prediction,

\[
L_D=\sum_k |\widehat m_k(c_t,w_t)-m_k^{obs}(O_t)|^2
    +\sum_h\|F^{(h)}(c_t,A_{t:t+h-1})-\operatorname{sg}(c_{t+h})\|^2.
\]

The first targets must be immutable observation-derived measurements, or otherwise independently anchored: allowing every feature measure and target to collapse together destroys the argument. Pixel-derived measures are legal and sample-level, but natural images are generally not sparse atomic measures. A learned feature front end might make them approximately sparse, yet introducing conventional landmark/slot detection would reintroduce its own collapse/assignment problem. This prerequisite is unresolved; the equation is a research specification, not implementation-ready training code. A planner using coefficient distance would also be a changed representation/readout and requires the same controlled evaluator.

The distinction from one soft-argmax channel is functional: one centroid cannot recover multiple differently weighted points, whereas sufficiently many moments can recover an unordered set before discarding amplitude. The distinction from direct Fourier features is amplitude invariance. Neither distinction implies robustness.

### Failure cases that the same checks deliberately expose

- **Small mass:** with the weaker atom's mass 1e−8, the Hankel condition number is 3.66e8. A fixed 1e−6 complex perturbation produces node error about 129. There is no uniformly stable recovery as mass vanishes. An algorithm cannot recover a visually unobservable object for free.
- **Colliding nodes:** the Vandermonde becomes singular; different factorizations merge. A nonzero separation/noise assumption is required.
- **Wrong measure order:** a K-atom model cannot generally represent a dense textured image or an articulated arm. Training residual can be small at fitted frequencies while unseen moments fail.
- **Two-dimensional pairing:** the point sets {(0,0),(1,1)} and {(0,1),(1,0)} have identical x/y marginal moments. Mixed-direction moments are necessary, and still require an identification argument for the chosen finite dictionary.
- **Learned-measure collapse:** all channels can select the same static wall or vanish. The sparse-measure theorem does not prevent this; it only applies after a faithful measure is supplied.

Closest primary work:

1. [Vetterli, Marziliano and Blu, 2002](https://bigwww.epfl.ch/publications/vetterli0201.html) supplies the actual finite-rate-of-innovation/annihilating-filter foundation. The recovery result must be credited to it.
2. [Deep Spatial Autoencoders](https://arxiv.org/abs/1509.06113) already learns spatial feature points for visuomotor control. The proposed multi-atom amplitude-free coefficient map differs from a single expected coordinate per feature channel, but ordinary learned landmark extraction is not a new contribution.
3. [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) learns operators in Fourier space. It does not make these polynomial coefficients the world state or remove mixture amplitudes, so using a Fourier transform alone is not the distinguishing mechanism.

[DeepPhase](https://i.cs.hku.hk/~taku/deepphase.pdf) is an additional close phase-representation reference; its temporal periodic motion embedding differs from spatial spectral-node identification. Targeted searches did not establish that the complete proposal is already published, but that is not proof of priority.

**Decision and one bounded falsification:** retain D as a hypothesis. Before GPU training, on a small fixed episode-disjoint image set compare low-order Hankel conditioning, held-out moment reconstruction, and address stability under harmless image-amplitude changes. Use physical labels only for an independent readout audit. If faithful, stable low-order measures are absent, reject this parameterization rather than hiding the failure with a large stack of semantic masks, per-task ranks or tuning constants. Only after that would a same-capacity short world-model gate be justified.

## Executable evidence and limits

- [check_representation_candidates.py](check_representation_candidates.py): exact residual reparameterization, confounded-action counterexample, observable closure, paired negative-curvature descent and positive-curvature failure, amplitude invariance, small-mass instability, and 2D marginal ambiguity. All assertions pass on CPU.
- [Numeric check output](ideas10-representation-theory-checks-20261030.json).
- [audit_action_confounding.py](audit_action_confounding.py) and [real-data output](ideas10-action-confounding-20261030.json): no GPU or learned world-model evaluation; physical labels never enter candidate training.

All positive mathematical statements have explicit idealized assumptions. None supplies a new theorem, a proof that current neural training will converge, a guarantee of goal semantics, or evidence of beating temporal-forecast/LeWM. The most useful outcome of this round is the admission filter itself: A's convenient data assumption is false, B's mechanism is already established, C targets a measurable optimization question, and D exposes a potentially useful separation between visual mass and spatial address together with a severe conditioning constraint.
