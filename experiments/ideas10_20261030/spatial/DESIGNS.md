# Spatial/contact/composition ideas: design, counterexample, rejection

Research date: 2026-10-30. This is a design and CPU-check artifact, not a claim of training gains or absolute novelty. The initial design/audit used CPU only. S1 was subsequently selected by root for an optional implementation and bounded GPU preflight; its exact implemented form is in `IMPLEMENTATION.md`. This agent did not launch GPU work. The later S3 refinement is in `S3_REFINED.md` and changes no frozen model code.

The starting evidence is specific: spatial pooling improves TwoRoom clean position readout primarily at initialization; its 600-step P0 rises from 6/8% to 12/16%, while Cube falls from 14/10% to 12/8%. DiT condition-use audits remain weak at this early stage, but they do not audit the clean latent planner, which bypasses DiT. Therefore neither another pooling arrangement nor blanket “state unused” reasoning is an adequate new mechanism.

The first independent designs were balanced action-loop response, contact resolvents, first-exit dynamics, and conservative trajectory gluing. The audit below deliberately rejects their generic forms. Three narrower research hypotheses are specified, but only S1 currently has a sufficiently distinct, existing-data operator and a direct real-data mechanism check. S2/S3 are provisional; their useful mathematical identities do not certify novel algorithms. S4 is rejected as a primary innovation. Do not silently count four established new methods.

## S1 — Action-annihilated outcome prediction

### The operation

For one ordinary logged window, let U_t be cumulative raw actions up to observed frame t, and let X=[1,U] be its temporal design matrix. Use the Moore–Penrose inverse to form

\[
P_A=I-XX^\dagger,\qquad Y_t=\psi(x_t),\qquad R_A=P_A Y.
\]

Here ψ is a fixed smooth image measurement, initially 4×4 cosine coefficients per RGB channel. It uses pixels only, no learned physical labels or pretrained teacher. Raw controls retain their physical zero. U must use the original action coordinates summed within the existing five-step blocks, not a flattened 25-dimensional block in an eight-frame window, which could exhaust the temporal nullspace.

A small head reads the model's causally rolled-out states, producing \(\hat Y_t=d_\theta(\hat z_t)\). Add the within-sample objective

\[
L_{\rm ann}=\frac{\|P_A(\hat Y-Y)\|_F^2}{\max(1,\operatorname{rank}P_A)\dim\psi}.
\]

The original TF or LeWM objective remains, including LeWM's native regularizer. The proposed term is sample-level: P_A is computed entirely inside one trajectory window. It has no batch negatives, covariance across samples, optimal-action labels, resets, or counterfactual outcomes. No new model input or planner cost is needed. The auxiliary head can be discarded at inference. Compare an equal-size head predicting unprojected ψ to isolate the operation from extra grounding/capacity.

### What is different from “predict the future”

The target is the part of visual evolution that a state-dependent *affine free response to the realized controls cannot explain*. Background constants and leading free effector motion cancel algebraically. Wall blocking, onset of contact, saturation and other nonlinear interactions can survive. This is not a claim that every surviving component is contact: perspective curvature, inertia, illumination changes, quantization and observation noise can survive too.

Do not replace the base representation loss by this target. A useful absolute position may be entirely annihilated; the proposed term teaches interaction-dependent outcome information, not the whole state.

### Property and constructive separation

1. **Exact nullspace.** For arbitrary state-dependent coefficients W(s), \(P_A XW(s)=0\). Cumulative-action linear coordinate changes U→UM+b, with invertible M, leave the projector unchanged. This does not include arbitrary recentering of *per-step* actions; that adds a time drift unless a time column is also included.

2. **Smooth/contact scaling.** Suppose q_t=q_0+εBU_t and ψ is twice continuously differentiable near q_0 with bounded Hessian. Taylor expansion gives \(P_A\psi(q_t)=O(\epsilon^2)\), because the constant and linear terms lie in col(X). For a contact kink \(q_t=\max(0,\epsilon U_t)\) crossing zero, \(P_Aq=\epsilon P_A\max(0,U)\), generically order ε. The ratio of contact to smooth residual therefore grows as 1/ε in this constructed local regime. If no kink is crossed, the contact example can also vanish. This is a conditional theorem, not a universal image-rendering law.

3. **Not global action-only residualization.** For q_t=s_0+b(s_0)U_t with s_0 independent of U, subtracting E[q|U] leaves s_0−E[s_0]+(b(s_0)−E[b])U. Our projector removes it identically. The other agent's response-coefficient state preserves this information; S1 deliberately does not. Their nullspaces and intended knowledge differ.

4. **Not old per-pixel predictability weighting.** A nonnegative diagonal temporal weight D cannot equal a nonzero P_A. Since P_A1=0, equality would imply 1ᵀD1=0, forcing D=0. Off-diagonal cancellation across a *single* trajectory is essential. The operation can of course be expressed as a known quadratic form; no new linear-algebra theorem is claimed.

5. **A real failure mode.** If the window's action design has full row rank, P_A=0 and the objective gives no signal. Even with nonzero rank, a constantly attached Cube moving affinely with the effector can be canceled. Contact onset and release, not every contact state, are the expected targets. Large camera movements or actuator nonlinearities can dominate the residual.

### What was numerically verified

`checks.py` checks affine annihilation, cumulative-coordinate invariance, distinctness from global action-mean removal, and scaling under five ε values. It also audits fixed smooth pixel features on 128 existing training episodes per task. Physical state/contact fields are read only to diagnose what the residual means; they are not candidate training targets. See `checks.json` and `RESULTS.md` for exact values and weak/negative observations.

### Both tasks and the first matched gate

TwoRoom: free translation is close to affine in cumulative controls, while blocked/corner/door transitions violate that relationship. Cube: effector free motion can be locally smooth while contact-induced block motion has a kink; this is a hypothesis needing real-data confirmation, not a guarantee for the five-dimensional control interface.

Four conditions would separate the mechanism at 600 updates: original model; equal-capacity full-ψ auxiliary prediction; P_A-projected prediction; a fixed random temporal projector of the same rank. The random control must use one predeclared seed and be constructed without altering outer RNG. Both tasks, common initial tensors/data order, one fixed normalization rule, unchanged native planner, all seeds retained. A gain only in auxiliary loss or clean probes is insufficient: require state/action-conditioned held-out outcome gains and then native P0. Any extra training compute must be counted.

### Three closest primary sources, and the boundary

| Source | Existing contribution relevant here | Why S1 is not the same operation |
|---|---|---|
| [MC-JEPA, Bardes et al. 2023](https://arxiv.org/abs/2307.12698) | Joint content and optical-flow representation learning. | Flow targets retain ordinary free translation; P_A cancels an affine control response. The free affine trajectory is a direct functional separator. Merely adding a motion head would not be novel. |
| [Homomorphism Autoencoder, Keurti et al. 2023](https://proceedings.mlr.press/v202/keurti23a.html) | Learns group-structured state/action transformations from observed transitions. | S1 assumes neither invertible physical actions nor group cancellation. A wall-projected forward/backward command has a nonzero defect. This rejects the initial idea of enforcing inverse consistency as though contact were a group. |
| [Causal Curiosity, Sontakke et al. 2021](https://proceedings.mlr.press/v139/sontakke21a.html) | Learns experiments that distinguish environmental causal factors. | S1 neither learns an exploration policy nor obtains new experiment outcomes: the temporal annihilator operates on the existing factual sequence. The original balanced-loop exploration sketch alone would not meet our matched-data contract. |

These are mechanism neighbors, not an exhaustive priority search. The defensible novelty hypothesis is the *per-window control-affine annihilator as a grounded JEPA world-model target*, with its nullspace and singularity-order prediction. It is not “contact discovery,” “residual prediction,” or “motion learning” in general.

## S2 — Rejection-informed conormal state (provisional)

### The basic idea that was rejected

Predicting a free step r and projecting it onto a learned feasible set, \(q'=\Pi_C(r)\), is a valuable physics prior but is not new. ContactNets already learns contact geometry without contact or force sensing. Simply placing such a layer inside LeWM is not an essentially different algorithm.

### The narrower candidate

Use only the same three-frame/action history already supplied to the planner. Encode *failed motion as information about a local feasible-action set*, even if the current image does not change. A proposed state contains a primal chart q and a compact dual constraint state N, rather than an image code alone. No additional online history is granted.

For a known local convex set C, one observed projection y=Π_C(r) implies

\[
n=r-y\in N_C(y),\qquad n^\top(v-y)\le0\ \forall v\in C.
\]

Thus an attempted action whose observed endpoint barely moves can still reveal a constraint normal. A candidate encoder infers a bounded set of such local normals from (x_{t-2:t},a_{t-2:t-1}); a shared transition operator uses them to constrain free predictions. The learning objectives are next-outcome prediction plus the samplewise projection/normal-cone consistency on observed transitions. There is no expert next-action target. The normal variables must be inferred, not taken from simulator contact labels.

**Functional counterexample.** Two hidden environments allow q≤0 or q≥0. Both show the same current q=0 image. A preceding positive attempted motion that returned q=0 identifies a right-side constraint; a preceding negative failed motion identifies a left-side constraint. A future positive command should remain at zero in one environment and move in the other. A frame-only representation aliases them; a history-conditioned conormal state need not. A sufficiently expressive ordinary history model can also solve this, so the counterexample does not prove architectural superiority.

**Verified conditional properties.** For one *fixed closed convex set*, the projection is idempotent and firmly nonexpansive:

\[
\|\Pi_Cx-\Pi_Cy\|^2\le\langle\Pi_Cx-\Pi_Cy,x-y\rangle.
\]

The CPU check tests 10,000 random pairs and a two-body non-penetration example. A wall command +.5 followed by −.5 from q=−.2 returns −.5, not −.2: invertible group action assumptions are inappropriate for this projected dynamics.

**Why both tasks.** Blocked TwoRoom motion reveals obstacles; Cube failed effector motion/contact can reveal interaction constraints. The important transfer is “learn from rejected action,” not “hard-code walls and cubes.”

**Unresolved issues that prevent promotion.** Latent charts inferred from pixels are not guaranteed to be physical coordinates. Global TwoRoom free space is nonconvex; local contact sets change with state, so the fixed-set nonexpansion theorem gives no global rollout guarantee. Existing three-frame history may not carry enough identification information. Zero free proposal or a degenerate chart can satisfy naive dual penalties. Grounded outcome learning and a free-proposal ablation are necessary. This remains a model hypothesis, not an accepted new contact method.

### Closest primary papers

| Source | Overlap and verdict |
|---|---|
| [ContactNets, Pfrommer et al. 2021](https://proceedings.mlr.press/v155/pfrommer21a.html) | Learns signed-distance/contact-Jacobian structure and uses complementarity-inspired losses; **already avoids force/contact sensing**. Reject any novelty claim based merely on learning contact implicitly or having no contact labels. |
| [Learning Contact Dynamics using Physically Structured Neural Networks, Hochlehnert et al. 2021](https://proceedings.mlr.press/v130/hochlehnert21a.html) | Structured contact dynamics and a study of touch feedback. Our no-touch visual history is a different information contract, not by itself a new mathematical mechanism. |
| [Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models, Zhong et al. 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/b7a8486459730bea9569414ef76cf03f-Abstract.html) | Already adds contact/friction/inequality mechanics to learned dynamics. A projection solver is not our new contribution. Only the observation-conditioned, finite-history *dual state update from rejected motion* remains a narrower research question. |

## S3 — Action-word first-passage prediction (provisional)

The generic survival-head version below remains insufficient as a novelty claim. `S3_REFINED.md` specifies the subsequent full query-transformation target, its five-class implementation contract, and a counterexample rejecting multiplication of marginal hazards. That revision is a proposed objective with CPU checks; it is not a trained result.

### The event-clock sketch that was rejected

Generic learned event times, variable-rate latent hierarchies and hybrid mode transitions already exist. Replacing fixed frame prediction with “predict the next event” is insufficient novelty.

### A different prediction contract

Predict the probability that a supplied *observation-defined query* is first satisfied during a specified action word, with proper treatment of trajectories that end before satisfaction. Training queries must be computed from fixed pixel features or ordinary observed frames. Do not use “touching,” “crossed the door,” block coordinates, or evaluator success labels unless reporting a separately privileged diagnostic.

For a Markov transition kernel P_a and query indicator g, define the killed kernel

\[
K_a^g=P_a\operatorname{diag}(1-g),\qquad
\Pr(T_g>H\mid\mu,a_{0:H-1})=\mu K_{a_0}^g\cdots K_{a_{H-1}}^g\mathbf1.
\]

The operator retains probability mass only along paths that have not yet satisfied the query. Composition acts on surviving state distributions; multiplying unconditional scalar success probabilities is generally wrong.

An auxiliary head can parameterize action-conditioned hazards on causally predicted states. For an observed first-hit index τ, or a right-censored window ending at H, the sample-level loss is

\[
L=-\sum_{t<\min(\tau,H+1)}\log(1-h_t)
 -\mathbf1\{\tau\le H\}\log h_\tau.
\]

Observation queries are training targets, not a policy's optimal-action labels. Use the same images/actions/window, and retain the native latent GoalMSE planner for the primary gate. A first-passage planning cost would be a separately reported controller extension. Queries can be sampled inside one sequence; no cross-batch negatives are necessary.

**Constructive separator from endpoint prediction.** On states S,G,D, action A toggles S↔G while B stays. Starting from S, AA and BB have the same endpoint distribution, but their probability of visiting G is one versus zero. Endpoint-only agreement cannot distinguish these path outcomes. A full frame predictor can recover them if accurate; we claim a different supervision target, not an impossibility theorem for all world models.

**Censoring check.** With geometric hazard .2 and maximum observation length 3, only .488 of trajectories contain an event. Correctly counting survival exposure recovers .2. Fitting only event-containing sequences yields .539823. This classical survival calculation explains why discarding uncompleted passages would be wrong.

**Why both tasks.** TwoRoom requires a valid passage before reaching the destination; Cube requires contact/motion and sustained completion rather than merely a plausible final frame. However, observation-defined query sets are not automatically those physical events. Goal images also do not identify whether arm pose or object pose is the intended task variable. These semantic identification problems remain and cannot be hidden in a privileged query detector.

### Closest primary papers and novelty boundary

| Source | Overlap and surviving distinction |
|---|---|
| [C-Learning: Horizon-Aware Cumulative Accessibility, Naderian et al. 2021](https://arxiv.org/abs/2011.12363) | Already learns horizon-dependent goal reachability and a recurrence from offline interaction. “Reach within H” is not novel. Our tentative distinction is predicting the outcome of the *specified action word*, rather than optimized policy reachability, and using it as reward-free world-model supervision. |
| [C-Learning via Recursive Classification, Eysenbach et al. 2021](https://arxiv.org/abs/2011.08909) | Models policy-conditioned future state distributions. Query prediction is established; a specific sequence-conditioned killed-path objective is the narrower candidate. If its implementation reduces to the existing objective plus an absorbing goal, reject the novelty claim. |
| [Neural Hybrid Automata, Poli et al. 2021](https://arxiv.org/abs/2106.04165) | Learns modes and stochastic event transitions. An event head alone is a rebrand; the candidate must enforce query-specific first-passage/survival composition. |

Additional hard exclusions: [Neural Event ODEs](https://arxiv.org/abs/2011.03902), [VTA](https://arxiv.org/abs/1910.00775), [THICK](https://openreview.net/pdf/3e5df2ed6659f21032c8784d5836ef8147d1413a.pdf), and [H-JEPA](https://arxiv.org/abs/2610.06805) already occupy substantial event/hierarchy territory. Killed kernels and survival likelihood are classical. This is a provisional training/API hypothesis, not a new probability theorem.

## S4 — Conservative trajectory gluing: rejected as a primary idea

The original sketch builds a directed graph of observed trajectory segments and composes them only when endpoint states are compatible, using a model to estimate join feasibility. This addresses topology and suboptimal-data recombination without optimal-action labels. But that broad mechanism is already well represented:

- [Search on the Replay Buffer, Eysenbach et al. 2019](https://arxiv.org/abs/1906.05253): learned goal-conditioned distances plus graph search over observations.
- [PALMER, 2022](https://papers.neurips.cc/paper_files/paper/2022/file/dd7a48c862b800f0537fe1d506e641b5-Paper-Conference.pdf): retrieves and recomposes observed trajectories using a learned perceptual reachability representation.
- [SAGAS, Liu et al. 2025](https://arxiv.org/abs/2512.00775): graph-assisted trajectory stitching with semantic/automaton planning.

Adding “conservative,” “topological,” or a confidence threshold does not establish a new operator. The CPU counterexample—equal position but opposite velocity gives endpoint distance zero and continuation distance ten after five steps—only shows why a sufficient history state is necessary. Existing history-aware graph models can address it. It does not rescue novelty.

## Status and useful next work

S1 is the priority distinct mechanism: it has an explicit sample-level operator, exact nullspace predictions, a constructed contact scaling test, and a direct existing-data audit. Its Cube support may be weak; report that rather than letting the TwoRoom mechanism dominate the narrative. The first useful GPU work, if root selects it, is a matched frozen-data predictive-head comparison, not a long training grid.

S2 and S3 have precise properties and falsifiable hypotheses but substantial prior-art and identifiability gaps. They can appear as provisional ideas with these gaps disclosed; they should not be counted as demonstrated novel methods. S4 and the generic versions of S2/S3 are rejected. None of the CPU checks establishes improved TwoRoom/Cube world modeling or planning.
