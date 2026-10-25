# Residual JEPA: focused prior-work and novelty assessment

Checked against public primary sources on 2026-10-08. This report supports the [author-facing proposal](residual-jepa-spt-proposal.md); it does not report experiments.

The broad architecture has substantial prior art. Sparse prediction followed by full-state integration, residual latent transitions, and separating a predictive state from reconstruction context have each appeared already. The potentially distinguishing hypothesis is narrower: **one globally shared subset of learned coordinates supplies all information used to compute every coordinate's increment**, while the complete previous state enters only through addition. The inspected work does not establish that this exact combination has already been evaluated, but the search cannot establish its absence from the literature.

## The claim being compared

Let the encoder produce \(z_t\in\mathbb R^d\). Select one index set \(J\), shared across frames, examples, and environments, and define

\[
s_t=z_t[J],\qquad
\Delta z_{t+1}=P(s_{\le t},a_{\le t}),\qquad
\hat z_{t+1}=z_t+\Delta z_{t+1}.
\]

The output of \(P\) has all \(d\) coordinates. The complement does not enter \(P\), any history-fusion module before \(P\), or a separate background-update network. A decoder can render the full predicted state. Actions are optional when unavailable. This report compares that strict version, including the revised **full-state identity skip**. It does not compare only the earlier complement-carry formulation.

This is a sparse set of parents of the **increment**, not a claim that unselected variables never influence the next state. Their identity paths explicitly preserve their values. It is also distinct from assuming that only a few output coordinates change.

| Kind of sparsity | What is restricted | Relationship to the proposed method |
| --- | --- | --- |
| Global parent subset | One set of input coordinates drives all updates in every example | The central hypothesis here |
| Sparse transition edges | Each output has a small parent set, potentially different for each output | Their union can contain every input coordinate |
| Local or dynamic selection | Important objects, tokens, or edges change with state and action | More flexible than a single global set |
| Sparse temporal changes | Few coordinates have nonzero increments at a step | Does not restrict which coordinates the predictor reads |
| Sparse mechanism changes | Few transition mechanisms change between environments | Concerns adaptation, not necessarily sparse within-environment dynamics |
| Sparse activations | Few feature values are nonzero for an example | Active coordinates can vary across examples |
| Spatial masking | Only selected image patches are processed | A fixed spatial location is not automatically a stable learned variable |

## Closest architectural comparisons

### 1. DDP-WM: Disentangled Dynamics Prediction for Efficient World Models

**Date and evidence:** first posted 2026-02-02; method inspected in full text v2, 2026-02-03, Sections 3.1 to 3.4, particularly Equations 1 to 3. Latest arXiv version is v4, 2026-03-05. ICML 2026 publication is separately verified in [official proceedings](https://proceedings.mlr.press/v306/yin26l.html).

DDP-WM uses frozen DINOv2 features, dynamically selects foreground patches, predicts their future features, and completes the background with an explicit residual cross-attention update. This is a close precedent for sparse expensive prediction plus full-state completion. However, all-token historical fusion precedes selection, and background features query the predicted foreground. Its increment therefore depends on unselected state. Its mask is state/action-dependent and spatial. The proposed strict, globally shared learned-coordinate bottleneck is a materially stronger restriction. Evaluations concern visual MPC in navigation and manipulation. [Inspected full text](https://arxiv.org/html/2602.01780v2)

### 2. Scope-WM: Scoped Computation for Efficient Visual World Models

**Date and evidence:** arXiv v1, 2026-09-27; full Method section inspected, especially Equations 10 to 19.

Scope-WM predicts selected foreground tokens and uses pooled foreground states and deltas to update the remaining tokens through gated residuals. It is another direct architectural comparison. Crucially, Equation 16 gives each update network its own current token as well as the foreground summary. The complement therefore participates in computing its increment. Although selector parameters are frozen after distillation, its selected set remains state/action-dependent. Features come from frozen DINOv2. The proposed method instead fixes coordinate identity globally and excludes complement information from every increment computation. Scope-WM evaluates efficient visual MPC; its results do not establish sufficiency of a global feature subset. [Full text](https://arxiv.org/html/2609.33218v1)

### 3. A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning

**Date and evidence:** first posted 2021-06-03; NeurIPS 2021. Full text v2, 2021-09-28, Section 4 and Appendix A.4 inspected; camera-ready v3 is dated 2021-11-04.

Conscious Planning already selects a small subset of an object-set state, applies dynamics to this bottleneck, then integrates predicted changes into the full state. Selection is action-conditioned and varies between steps. The integration module also queries the predicted bottleneck using the full current state, so it does not satisfy the proposed exclusion of unselected inputs. Training includes latent dynamics consistency and control objectives rather than future-frame reconstruction. Its gridworld generalization results support the broad rationale for limited predictive computation, but not fixed coordinate selection. It is a strong conceptual predecessor to acknowledge. [Inspected full text](https://arxiv.org/html/2106.02097v2), [official publication](https://proceedings.neurips.cc/paper/2021/hash/0c215f194276000be6a6df6528067151-Abstract.html)

### 4. LRC-JEPA: Disentangling Dynamics and Residual Context for Efficient World Models

**Date and evidence:** arXiv v1, 2026-09-28; full method and decoder description inspected, including Equations 11 and 12.

LRC-JEPA separates a compact predictive latent from learned residual-context embeddings. Only the former is propagated and used for planning; both support reconstruction. This is the closest JEPA precedent for the proposed information-separation narrative. However, Equation 12 reconstructs the same frame from its two branches through a cross-attention decoder. It does not add a subset-conditioned full increment to the previous complete state. The branches also are not a selected subset of one latent vector. Explicit regularizers address collapse. Its control evaluations make it a direct representational baseline, without establishing the strict global-parent hypothesis. [Full text](https://arxiv.org/html/2609.34375v1)

## Related JEPA and residual-transition designs

### 5. AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

**Date and evidence:** arXiv v1, 2026-09-24; full method inspected, especially Section III-B, Equations 1 and 2.

AD-WM directly uses a JEPA-style residual transition: predict a latent increment from current latent and action, then add it to the current latent. Thus neither the full-state skip nor residual JEPA dynamics alone is a defensible novelty claim. Its predictor reads the full latent, unlike the global-subset constraint here. It adds action-discriminative objectives and studies counterfactual MPC. The paper also distinguishes the residual parameterization's inductive bias from guarantees about action dependence. This makes it a relevant dense-residual comparison, separate from papers about selecting a sparse state. [Full text](https://arxiv.org/html/2609.30264v1)

### 6. Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models

**Date and evidence:** first posted 2026-05-29; full text v1 inspected. The [version record](https://arxiv.org/abs/2605.31111) lists v2 dated 2026-09-15; its revised results were not independently audited here.

SD-JEPA assigns fixed coordinate groups to progression and content, using a temporal triplet objective for the former and SIGReg for the latter. Fixed feature roles within a jointly learned JEPA latent therefore have precedent. Its predictor operates on the full latent rather than using only the progression coordinates to generate a full-state increment. It has no equivalent carry-and-render construction. The distinction is the proposed restriction on predictor inputs, not merely declaring a permanent subspace. Its planning experiments address usefulness of the progression/content split. [Inspected full text](https://arxiv.org/html/2605.31111v1)

## Sparse causal dynamics and stable learned variables

### 7. Variational Causal Dynamics: Discovering Modular World Models from Interventions

**Date and evidence:** arXiv v1, 2022-06-22; full formulation inspected, especially Equation 7 and the intervention-mask construction.

VCD jointly learns representations and a sparse transition graph across intervened environments. Each output factor has its own masked parents, and sparse mechanism changes support adaptation. This is relevant precedent for stable latent factors and sparse learned causal dynamics. It does not posit one small common parent set for every increment: the union of output-specific parent sets can be large. Its multi-environment intervention setting and modular transition assumptions provide evidence unavailable from reconstruction alone. The proposed residual bottleneck would need independent justification before borrowing VCD's causal interpretation. [Full text](https://arxiv.org/html/2206.11131v1)

### 8. Disentanglement via Mechanism Sparsity Regularization, and its nonparametric extension

**Date and evidence:** original first posted 2021-07-21, published at CLeaR 2022; full text v2 inspected. The extension first appeared 2024-01-10 and is published in JMLR 27(71), 2026; its v2 is dated 2026-07-31.

These works connect sparse temporal/action dependencies to identifiable or partially identifiable learned factors under explicit assumptions, including sufficient variation and appropriate graph structure. This is the relevant theoretical lineage for claiming that sparsity discovers variables, rather than merely compressing them. Their sparsity concerns dependency mechanisms, not a universal small input subset plus an identity skip. The assumptions and graphical criteria cannot be replaced by a generic reconstruction loss or a fixed mask. Formal proofs were not independently checked in this review. [Original full text](https://arxiv.org/html/2107.10098v2), [extension full text](https://arxiv.org/html/2401.04890v2), [journal record](https://www.jmlr.org/beta/papers/v27/24-0771.html)

### 9. SPARTAN: A Sparse Transformer World Model Attending to What Matters

**Date and evidence:** first posted 2024-11-11; latest v3, 2025-12-05. Full model, related-work discussion, and selected appendix material inspected.

SPARTAN learns sparse, context-dependent attention edges between object-factored tokens and supports adaptation to sparse interventions. It retains all objects and their residual self-connections. A sparse graph can therefore distribute computation across many different active objects, unlike one global parent subset. The paper explicitly distinguishes its sparse edges from Conscious Planning's small active subset. Its visual experiments use object factorization, including object-mask information, so they do not establish discovery of stable semantic coordinates from unrestricted video encoding. SPARTAN is a useful graph-sparsity comparator, rather than an implementation of the proposed node-subset restriction. [Full text v3](https://arxiv.org/html/2411.06890v3)

### 10. Causal Dynamics Learning for Task-Independent State Abstraction

**Date and evidence:** ICML 2022; first arXiv posting 2022-06-27. Official paper's formulation and abstraction discussion inspected.

CDL learns causal parents in state transitions and derives abstractions retaining controllable variables and relevant uncontrollable ancestors. Thus extracting a small state for planning from transition structure has direct precedent. Its guarantees concern an explicitly restricted class of actively accomplishable tasks; arbitrary reward functions can make otherwise discarded variables necessary. Its factored observable-state setting differs from learning a visual encoding and generating a full residual update from one global subset. For this proposal, it emphasizes that predictive compactness and planning sufficiency require separate statements about tasks and costs. [Official paper and PDF](https://proceedings.mlr.press/v162/wang22ae.html)

### 11. Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks

**Date and evidence:** arXiv v1, 2025-11-11; AAAI 2026. Full methodology and Jacobian results inspected.

The authors examine differentiable simulator Jacobians for robotic tasks and find global transition sparsity much less common than local, state-dependent sparsity. This is useful counterevidence to treating a globally sparse structure as automatic. Its scope matters: the analysis uses the benchmarks' physical coordinates, not all possible learned representations, and concerns transition dependencies rather than the proposed identity-subtracted increment. It therefore does not disprove the existence of a suitable learned global subset. It does motivate testing contact changes and multiple operating regimes rather than assuming a fixed subset transfers. [Full text](https://arxiv.org/html/2511.08086v1), [official publication](https://ojs.aaai.org/index.php/AAAI/article/view/39658)

## Adjacent evidence that should not be conflated with the claim

### 12. Sparse Imagination for Efficient Visual World Model Planning

**Date and evidence:** first posted 2025-06-02; ICLR 2026; full text v2, 2026-02-26, including Section 4.4 and Table 4 inspected.

This method uses randomized grouped attention and patch dropping to reduce visual planning computation. Its ablation compares masks held fixed within an episode with masks resampled during replanning, with mixed outcomes across settings. That is more relevant to the original spatial-mask question than a generic assertion that fixed masks must work. However, it tests retained spatial patches in pretrained features, not a globally shared set of learned semantic coordinates. It neither verifies the proposed bottleneck's performance nor establishes equivalence between temporal mask consistency and feature identity across environments. [Full text](https://arxiv.org/html/2506.01392v2)

### 13. MiMo: Improving Autoregressive Video Modeling with History Understanding

**Date and evidence:** ICLR 2026; official final paper, Figure 3 and Section 4, inspected through indexed PDF text.

MiMo uses random tube masking over clean history, alongside auxiliary history-understanding objectives for autoregressive video modeling. It is a direct precedent for retaining the same spatial support across multiple history frames. This is weaker than fixing a learned-variable subset globally across examples. No isolated fixed-versus-independently-resampled mask result was verified in the inspected material. It supports the feasibility of temporally aligned spatial masking, not a prediction that this SPT modification will preserve performance. [Official paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d30c2def27b5c6a5fb21a9aa5c16f8f-Abstract-Conference.html)

### 14. Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations

**Date and evidence:** first posted 2026-02-01; ICML 2026; v2 dated 2026-05-28. Abstract and full-text method overview inspected.

Rectified LpJEPA, coauthored by Yann LeCun, explicitly combines sparse representations with distribution-matching regularization against collapse. Its sparsity is sample-wise nonzero activation structure, with controllable expected support size. This differs from excluding the same coordinates from the predictor for every example. The paper is therefore relevant to the sparse-JEPA motivation and possible anti-collapse design, not evidence that a permanent predictive subset is sufficient. Evaluations address image representation learning, rather than the proposed residual video transition. [Full text](https://arxiv.org/html/2602.01456v2)

### 15. Unsupervised Learning of Disentangled Representations from Video (DrNet)

**Date and evidence:** arXiv v1, 2017-05-31; full architecture and temporal-prediction description inspected.

DrNet separates stationary content from changing pose, predicts pose sequences, and combines predicted pose with retained content for video generation. This is an older precedent for reusing appearance while predicting a smaller dynamic representation. Its temporal predictor is conditioned on content, so retained information is not excluded from dynamics. It uses separate encoders and adversarial disentanglement rather than a globally selected coordinate subset and full-state additive update. [Full text](https://arxiv.org/html/1705.10915v1)

### 16. RLA-WM: Learning Visual Feature-Based World Models via Residual Latent Action

**Date and evidence:** arXiv v1, 2026-05-08; full method inspected.

RLA-WM compresses changes between DINO feature states into a compact residual latent action and reconstructs future features using the current features and this transition code. Prediction of the code is conditioned on the current full features and actions. Its bottleneck compresses the transition output, whereas the proposed bottleneck restricts which state inputs can determine the transition. It supports compact change modeling, but not the global-parent exclusion claim. [Full text](https://arxiv.org/html/2605.07079v1)

## What follows for the research claim

The literature narrows the interesting question to whether a **single learned predictive subspace** can be sufficient across the intended data distribution, with no complement access in the update. The distinction is mathematically substantive, although greater restriction alone does not establish practical value or novelty.

For the first-order deterministic case, write \(R_Jz=z[J]\) and \(T(z,a)=z+P(R_Jz,a)\). Where derivatives exist,

\[
D_zT=I+(D_sF)R_J,
\qquad
\operatorname{rank}(D_zT-I)\leq |J|.
\]

Every column of the update Jacobian outside \(J\) is zero. This is an algebraic consequence of the proposed architecture, not a theorem supplied by the reviewed papers. For history-conditioned models, the analogous exclusion must hold for unselected coordinates at each input time. A graph with few parents per output does not generally satisfy this condition.

Three qualifications change how the result should be communicated:

1. **A fixed index set does not by itself identify physical variables.** With a freely learned encoder, choosing the first \(k\) coordinates can be equivalent to learning which \(k\) coordinates to retain, up to a permutation. The substantive constraint is the information capacity and exclusion of the predictive subspace. Semantics, minimality, and cross-environment stability require evidence beyond selecting indices.
2. **Autonomous selected-state rollout does not imply planning sufficiency.** Algebraically, \(\hat s_{t+1}=s_t+P(s_{\le t},a_{\le t})[J]\) is closed. But a planning cost may require carried variables. For example, velocity can determine position increments while a position goal still requires the current position. State sufficient for increments is not automatically state sufficient for every task.
3. **Reconstruction does not force the selected branch to be informative.** A sufficiently capable decoder can learn future prediction from the carried full state while ignoring a weak or collapsed update branch. Explicit latent alignment, appropriate anti-collapse treatment, and branch-use diagnostics are needed before describing the selected coordinates as a learned JEPA state.

The closest comparisons suggest four distinct empirical questions: whether the global bottleneck improves on a matched dense residual model; whether it remains competitive with dynamic selection; whether preventing complement access helps or harms; and whether the selected state supports the intended planning costs. The proposal's technical notes specify the experiment more concretely.

Suggested contribution wording:

> We investigate whether a globally shared subset of learned latent coordinates can carry the information needed to predict full-state increments, while a full-state residual path preserves the previous representation. In SPT, we test this restriction as a way to concentrate predictive information and reduce temporal-model computation.

Avoid claims of the first sparse world model, the first residual JEPA, the first dynamics/context decomposition, identified causal variables, or guaranteed anti-collapse. If experiments succeed, the strongest contribution would be evidence that the strict global bottleneck is useful under a clearly stated distribution and compute budget.

## Search coverage and limits

The search covered sparse causal world models, object/subset selection, causal state abstraction, sparse-mechanism identifiability, content/dynamics video decomposition, residual latent world models, sparse JEPA objectives, and persistent video masks. Related-work chains were followed from SPT/MiMo, LRC-JEPA, SPARTAN, Sparse Imagination, and Scope-WM to the closest architectural predecessors. Public arXiv full texts and official proceedings were used for substantive comparisons; secondary summaries were discovery aids only.

Additional screened neighbors included [Action-Sufficient State Representation Learning for Control with Structural Constraints](https://proceedings.mlr.press/v162/huang22f.html), [SlowVAE](https://arxiv.org/abs/2007.10930), and [data-driven discovery of coordinates and governing equations](https://arxiv.org/abs/1904.02107). These were screened at abstract or limited-method level and are not used to establish detailed distinctions here. They concern task-sufficient states, sparse temporal changes, and sparse equation terms, respectively.

The review checked relevant methods and selected evaluations, not every proof, appendix, implementation, or experimental claim. Version-specific evidence is identified above where the inspected version differs from the latest record. No numerical success probability for the proposed SPT modification is justified by this literature. Recent September 2026 preprints are particularly close, so any submission should refresh this search and compare their released implementations when available.

SPT itself is a project-local anonymous ICLR 2027 manuscript, *Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal Histories*. No public identifier was verified. The project handoff distinguishes SPT's existing sparse history conditioning from an explicit learned-latent residual transition; this report does not attribute that new transition to the baseline.
