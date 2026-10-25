# Codex session discussion log: sparse JEPA / SPT research line (Oct 8 to Oct 24, 2026)

Source: Codex CLI rollouts under `~/.codex/sessions/2026/10/` that mention the sparse_jepa repo. Only user messages and assistant final answers were read (tool output skipped). Forked subagent rollouts were read as supporting evidence for their parent session rather than as separate discussions. This log records how the thinking moved; it does not restate the finished documents, which are:

- `../archive/lewm-plans/dense-sparse-implementation-plan.md` (Oct 6 campaign spec) and `../archive/lewm-plans/{common-protocol,sparse-past,anti-collapse,multiple-future}.md` (Oct 4 LeWM exploration plans)
- `../superseded/residual-jepa-spt-proposal.md`, `residual-jepa-technical-notes.md`, `residual-jepa-prior-work.md` (Oct 8; the sessions that produced these are not in this set)
- `../spt-collaborator-handoff.md` (Oct 19 to 20, produced by session 2 below)
- `../anchored-state-jepa-design.md` (Oct 25, produced outside these Codex sessions; mentioned only for the trajectory)

Sessions in the adjacent `~/code/sparse-update` repo (the collaborator's sparse-update world model) are included because they are the same anti-collapse research line and reuse this repo's LeWM data protocol.

---

## 1. Oct 8 to Oct 15: dense-sparse LeWM campaign monitoring and results

Session `01a111de-8a39-7fc3-adfd-c6f2c66e4548` (cwd `sparse_jepa`, started 2026-10-08 09:10, last turn 2026-10-15).

**What the user asked.** Repeated "inspect the progress and restart runs if possible", one request to auto-launch the next batch, then "report the results" (Oct 12) and "summarize the results" (Oct 15). This is operational monitoring, but it is where the Oct 4 to Oct 6 plans turned into evidence, so the results matter for everything after.

**How it evolved.** The supervisor already resumed jobs from checkpoints and launched stages behind learning gates, so no manual restarts were needed. On Oct 11 the assistant found a planning shard stuck for two days on a draining node, cancelled that one unstarted job and resubmitted it elsewhere. Stage 1 (dense), Stage 2 (input masking 50/75/90% x two SIGReg weights x three seeds) and Stage 3 (stabilization variants) ran to completion: 51 core models with validation and diagnostics.

**Findings that shaped later thinking** (validation split, 200 tasks per seed, three seeds, ten-epoch budget; test split still sealed):

- Dense patch prediction tolerates heavy input masking but does not beat the CLS baseline. Ordinary SIGReg 0.09: 81.2% (0% mask), 79.7% (50%), 79.7% (75%), 82.7% (90%). The 90% gain of 1.5 points has a bootstrap 95% interval of about -3.5 to +6.5 points, so no improvement can be claimed. CLS controls: 84.7% at 0.09 and 93.5% at 0.0225.
- SIGReg weight matters more than masking. Weight 0.0225 was unreliable (one 50%-mask setting had seeds at 73.5%, 10.5%, 5.0%).
- Pooled-over-time SIGReg cost about 16 to 17 points versus ordinary SIGReg at 0.09.
- Removing SIGReg (prediction loss only) collapsed every run: 1.5 to 7% success, near-constant features, forecasts worse than persistence, no response to action shuffling, despite tiny training loss.

**Open at the end.** Conditional follow-ups from the implementation plan (target asymmetry, geometry/persistence, multi-future), final model selection and test evaluation were left pending. The practical takeaway the user carried forward is that the LeWM route needs an explicit anti-collapse regularizer and that masking alone neither helps nor hurts much, which is the backdrop for the user's later wish to waive SIGReg/EMA/stop-gradient through reconstruction instead.

---

## 2. Oct 19 to Oct 20: collaborator handoff for the SPT extension

Main session `01a1559e-f466-7293-9ebe-0d2514bc63eb` (cwd `sparse_jepa`, 2026-10-19 19:24, follow-up 2026-10-20 02:43). Forked reviewers: `01a1559f-c6cc-7303-b630-43857fa5cd4e` (code/manuscript audit), `01a1559f-e6ab-7ed0-b06b-200ad83f7610` (literature check), `01a155a1-484b-71c2-98f7-58e1a41befe4` (design review and later doc reviews).

**What the user asked.** Write a concise, intuitive handoff for a collaborator who is not theory-driven and had only heard the design over the phone. The user restated the three components added on top of SPT: (1) a partition of the latent space into a high-level dynamics part (H) and a low-level visual-detail part (L); (2) a residual connection in the latent next-state prediction; (3) a JEPA-like latent prediction loss. The story: H learns true dynamics without spending capacity on detail, L stores detail, and a reconstruction loss over both stabilizes the latent distribution canonically, so SIGReg, EMA and stop-gradient can all be dropped. The user wanted two algorithm boxes (residual first, then a general version with dimensionality control), a tuning space matched to the three components, and an evaluation that looks first at generation and action following on PushT (compared with JEPA models plus separate decoders) before planning accuracy. Internal derivations were explicitly not to be shared.

**How the user's thinking evolved inside the prompt.** The user called the residual design a crude, hard constraint: past L enters its future only linearly through the identity carry. The more general version would use two causal transformers, `H_next = F_H(H_history)` and `L_next = F_L(H_history, L_history)`, which is asymmetric but invites collapse (everything moves into L and H dies). The user reasoned that reconstruction should prevent this if the total dimensionality is right, because redundancy only appears when capacity is too large; hence the question of how to enforce a dimensionality match flexibly (a learned mask, echoing the earlier image-translation work at openreview U2g8OGONA_V). The user also proposed using actions to delineate the partitions: route actions only into H, note that changes come from both time and actions, and hope that same-start with/without-action contrasts can be obtained without paired training data, with theory tied to the mechanism-sparsity paper (arXiv 2401.04890). The central hypothesis: if actions connect only to H and total dimensionality is actively controlled, H will not collapse.

**What the reviewers contributed (and what was adopted).**

- Code audit: there is no SPT model source in this repo (the `spt` imports are `stable_pretraining`), so the handoff was framed as changes to the collaborator's SPT, not a diff of local code. SPT samples tokens after a full-chunk ViT, its loss is flow matching only, and its pretraining uses no actions. The sparse causal transformer does not output a full `[S, d]` grid, so the residual predictor needs a full-grid output head whose queries do not read L.
- Wording fix adopted: in the residual model both partitions change; H supplies the information that computes the update and L is carried by identity. Saying "only H changes" would contradict the Oct 8 notes.
- Supervise the full predicted state if it is rolled out; H-only latent MSE is a separate ablation. Rollouts must carry predicted, not observed, states.
- Literature check: the image-translation mask (`z_s = eps + m * f_u(eps)`, L1 with threshold) selects which coordinates may change; it does not limit total information, so only the "learn which coordinates change" idea transfers. The mechanism-sparsity theorems require assumptions (sufficient action variation, sparse learned edges) a deterministic transformer with reconstruction does not automatically meet, and dimensionality control is distinct from mechanism sparsity. The OpenReview PDF could not be fetched, so its theorem should not be cited as support.
- Collapse avoidance was reframed as a hypothesis the experiment tests, not a property of the design. Reasons: a flow decoder sees a noisy future target and can denoise well while ignoring the latent condition; reconstruction can be satisfied with H empty and L holding everything; nonzero H variance is not evidence that H is used.
- Same-state alternate-action tests can be simulator resets at evaluation time, so no paired training data is needed. Use the real "hold" action, not an all-zero vector (which may command motion to the origin).
- Two diagnostics were added: replace H history with another example's while holding actions fixed, and zero the residual update while keeping the carry.

**Oct 20 correction from the user.** The user clarified that the gating idea is global capacity pressure: if total representation dimension is constrained, every token must be used for reconstruction; implement a learned soft mask with an L1 penalty to zero out unnecessary tokens. The assistant revised the handoff accordingly.

Decisions after the correction:

- One global sigmoid gate per token, shared across examples and time, with `L_dim = mean |g|`; separate low-level budgets became optional rather than mandatory. Reason: matches the user's intent and is the simplest faithful test (precedent: Network Slimming).
- Latent MSE uses fixed denominators and attached targets, because `MSE(g*p, g*q) = g^2 * MSE(p, q)` lets shrinking gates fake lower loss.
- Validate by physically pruning low-gate tokens and re-measuring reconstruction and action following.

Rejected or demoted ideas:

- Hard-concrete gates and separate L budgets as the primary proposal (kept as alternatives): more machinery than the user's idea requires for a first test.
- Freezing L entirely: contradicts the settled full-state residual.
- Citing either paper as proof that collapse is prevented.

Open questions left:

- Small gates need not remove information: downstream weights and normalization (a following LayerNorm can undo a scalar gate) can compensate. Gate placement in the collaborator's SPT code was left to him.
- Efficient capacity use does not imply the intended partition: the minimum-capacity solution could still put scene information in L and use H only as an action conduit.
- Whether action routing plus capacity control actually makes H necessary for action-dependent dynamics is the central empirical uncertainty.
- No experiments were run in this session.

---

## 3. Oct 20 to Oct 21: signed sparse updates and latent actions (sparse-update repo)

Session `01a15a52-d962-79d2-9947-814918d3522b` (cwd `~/code/sparse-update`, a forked thread of `01a15a52-4466-76c0-a825-ee92bd4b457e`; first turn 2026-10-20 17:19, last answer 2026-10-21 11:42). Memos produced there: `sparse-update/docs/research/sparse_update_masking.md`, `latent_action_literature.md`, `october_setup_audit.md`.

**What the user asked.** The collaborator's line regularizes predicted updates (next state = current state + predicted update) toward a sparse distribution, following his rectified-sparsity work (arXiv 2608.22764). The user's critique: rectification removes negative values, and regularizing updates this way may let magnitudes grow and diverge. Proposal: replace that with a learned soft mask so updates stay signed while support is learned, and possibly make the mask input-dependent (citing SmartCLIP, arXiv 2507.22264). The user then questioned the principle: predicting updates is an equivalent parameterization of predicting the next state, so what is the principled anti-collapse regularizer on updates, does SIGReg still make sense there, and is regularizing updates better than regularizing states? Natural priors on updates: non-collapse, sparsity across adjacent steps, and predictability of actions, which resembles inverse-dynamics JEPA work. This led to a second direction: latent actions that unify temporal and action-driven change, with a discrete dictionary (VQ-like) giving entropy-style anti-collapse properties, either as a soft constraint or routed through the forward pass. The user asked for (1) first masking experiments and (2) a literature collection and positioning for the latent-action idea.

**How the thinking evolved and what was decided.**

- Masking design: a separate signed amplitude head plus a gate head over shared causal context, compared against soft gate, direct L1 and the existing fixed soft threshold. The fixed threshold already gives input-dependent support, so a learned static gate is a control, not a description of the baseline. Hard-concrete exact-zero gates became the main gated variant.
- Answer to "why not just L1 on the output": soft gates can trade scale with amplitude (`eps*m`, `v/eps`) and evade the penalty, which is the user's suspected advantage turned into a failure mode to measure.
- Sparsity alone does not prevent collapse: a constant encoder with zero updates satisfies prediction, L1 and gate penalties. Keep anti-collapse reference distributions independent of learned gates, and measure actual encoder differences, not only predicted updates.
- Latent-action positioning: the idea is close to existing work (Delta-JEPA decodes actions from latent differences; MotionJEPA stabilizes via visual-change targets; LAPO and the in-the-wild latent-action paper use frozen or separately grounded encoders). A discrete transition dictionary encourages reuse but gives the encoder no reason to preserve information on its own, so the first experiment should ground codes in observed actions or a stabilized visual-change branch, with an ungrounded VQ-only model as a failure control. Novelty was framed as a hypothesis (adaptive sparse support plus grounded transition codes), not an established gap.
- Experimental protocol: the exact DINO-WM PushT data was re-downloaded and verified because the original cluster paths were unavailable; a screen of nine dense cells (LR x SIGReg weight) on a 1,024-trajectory prefix, promotion of two dense candidates to full data, then nine cells per sparse family, then three paired seeds. Selection by planning success, then physical distance; never by total loss across regularization weights.

**Early evidence (Oct 21).** The first four dense cells all had 0/20 planning successes, with prediction-to-persistence ratios from 1.7 to over 50,000, so no dense anchor existed yet. The assistant warned that SIGReg on raw pre-mask updates can stay healthy while applied updates vanish.

**Open at the end.** No working dense anchor; whether update-space regularization can replace state-space regularization; whether latent actions can be the anti-collapse mechanism without a state regularizer.

---

## 4. Oct 24: what to learn, and how masking relates to disentangling dynamics from appearance

Main session `01a16d6c-0f5a-7b62-915e-ec84021ca5e3` (cwd `sparse_jepa`, 2026-10-24 10:19). Forked readers: `01a16d6c-6d45-7af0-8a8b-166f3db1d495` (SPT manuscript check), `01a16d6c-8491-71e0-a210-5505190866ff` (masking literature), `01a16d6c-9a54-7102-bb66-8e37ad93472e` (information-theoretic framing). No files changed.

**What the user asked.** A long reflective note. The user had realized the day before that autoregressive prediction differs from matching the whole series: it forces inference of the future from the past, and the unpredictable part cannot be pinned down. But persistent detail (textures, background) also transfers from past to future, even more reliably than dynamics, and same-step reconstruction does not separate it from dynamics. Using future information for reconstruction also breaks generation, where the future is unavailable. Ideas floated:

- Learn whatever the past can predict; take visual detail from the current state.
- A CLIP-like loss maximizing information between past and current/future, plus possibly an information-minimization term between current and future (acknowledged as conflicting).
- The persistent part as information shared between the immediate past and the entire history, with dynamics emerging only from long history; use the current frame as a contrastive negative and the full history as the positive.
- On top of SPT, add spatial tube masking and long temporal block masking with large ratios and large block sizes (lesson from MAE patch size), so low-level detail cannot be relied on.
- Questions about SPT's flow noise: is it part of the representation? Should the latent prediction loss be kept, given the causal transformer output summarizes the past and should not equal the next frame's encoding without a module in between? Does a history summary work for planning? Mask in latent or observation space? Has temporal MAE been done already?

**How the answer reframed it (main conclusions).**

- Three distinctions are being conflated: predictable vs unpredictable, persistent vs changing, relevant vs irrelevant to dynamics. Maximizing shared history-future information keeps persistent background (it is an excellent matching signal), so it does not select dynamics. "Dimensionality of mutual information" is not a meaningful quantity; the useful aim is predictive sufficiency under limited capacity, and the prediction target decides what counts as relevant.
- The user's "what long history adds" idea has a precise form, `I(older history; future | current observation, actions)`, but it measures the need for memory, not dynamics; it is zero for a fully observed Markov state with rich dynamics.
- Rejected: using the current frame as a negative for its own future, because it would repel useful shared position, geometry and contact state. A safer comparison is history-based versus current-frame-only prediction, or conditionally matched negatives.
- Masking is an inductive bias, not a definition of high-level information. Persistent appearance survives heavy temporal dropout better than velocity, so agreement across heavily masked histories may favor appearance. Prior work agrees: VideoMAE (90 to 95% tube masking, protects only low-motion patches), V-JEPA (causal block masking underperforms bidirectional; random 90% tube is much worse), MiMo (ICLR 2026, the closest precedent: tube-masked history plus auxiliary current/next codec-latent reconstruction with an AR diffusion generator; 25 to 50% masking works, higher needs unmasked fine-tuning). So temporal MAE on history has been done in close form.
- Where masking happens matters: SPT drops tokens after a full-chunk ViT, so retained tokens already contain masked-region information. Real information removal requires masking before the encoder mixes regions, which would be a substantive change to SPT. "4D masking" should be T x H x W; channels are a feature axis.
- "Latent reconstruction" has two meanings: SPT predicts fixed codec latents (still appearance), whereas JEPA predicts learned features (can omit detail but needs a collapse-prevention recipe).
- Flow noise is stochastic generation, not a low-level-detail partition: it samples plausible details but cannot recover observed appearance the conditioning discarded. A future-derived residual partition is allowed in training only with a history-conditioned prior at inference (PlaNet-style); otherwise the residual can carry everything.
- Keep the history representation distinct from the predicted future representation, with a predictor in between (CPC precedent). A history state can serve planning, but planning also needs action-conditioned transitions and a goal score; SPT re-encodes generated chunks rather than rolling a compact state.

**Recommended smallest design.** Encode causally masked history into a compact state; a separate predictor predicts future JEPA features; a flow decoder is conditioned on those predictions plus appearance carried from observed history (DrNet-like); keep a tested collapse-prevention mechanism; initially omit current-frame negatives and any explicitly learned future-residual partition. Diagnostics: texture changes should not change predicted motion, motion-history changes should, generation should work without future-derived inputs, and removing the history condition should hurt (otherwise the decoder is doing the predictive work).

**Open questions left.** Whether to keep a JEPA latent loss inside SPT and which anti-collapse mechanism to pair with it (this partially conflicts with the user's Oct 19 goal of waiving all such tricks); whether to move masking before the encoder; how to make a compact state that supports planning; how to formalize the persistent-vs-dynamic split without hand-picking what counts as nuisance.

---

## 5. Oct 24: why the upstream sparse-update model works and ours does not (sparse-update repo)

Session `01a16dc6-1c40-7231-bc3d-7794f56abcfa` (cwd `~/code/sparse-update`, 2026-10-24 11:57). Forked readers: `01a16dc7-601f-7fe0-ad29-d522b724b599` (local evidence), `01a16dc7-7ced-7410-aec0-3fbd79df6e06` (upstream comparison).

**What the user asked.** Pull the latest upstream commit (one version reportedly reaches about 80% PushT planning success) and explain why it works and ours does not.

**Findings.**

- Upstream works with a specific balance: Gaussian distribution matching on the encoded state (weight 0.5) plus a weak prior on the applied update (0.01). Removing the state regularizer drops success from 80% to 6%; raising the update weight to 0.1 drops it to 40%. The 80% models keep about 74 to 76% of update coordinates nonzero, and actual encoder differences stay dense, so this is mild sparsity, not a sparse transition.
- Our first masked campaign failed for a direct reason: SIGReg was applied to the raw update before the gate, so gates closed every applied update while the raw branch looked healthy (all 24 masked runs had zero applied updates, 2% success). Dense residual with update-SIGReg averaged 20%.
- Our newer state-SIGReg runs (direct predictor mean 20%, dense residual mean 8%) show that state regularization alone with SIGReg did not recover upstream. The regularizers (SIGReg vs RDMReg sliced-Wasserstein), their scales, the LR grid (upstream's 5e-5 was missing locally) and evaluation seeds/panels all differ.

**Decision.** Reproduce the upstream dense anchor and IS-Delta recipe exactly (state reg 0.5, update reg 0.01, theta 0.1, LR 1e-4, two epochs) before extending the masking comparison; evaluate on upstream seed 0 and then on the shared held-out panel. Pulling alone does not select the winning recipe because the config defaults still disable state regularization.

**Implication for the research line.** This is direct evidence against the Oct 20 hope that update-space regularization alone can prevent collapse: in the working recipe, state-space regularization carries most of the load.

**Open.** The 80% has not been reproduced locally; whether SIGReg itself contributes to the gap is unknown.

---


## Cross-session trajectory, Oct 4 to Oct 24

**Oct 4 to 6 (plans).** The line began as improvements to LeWM: three independent explorations (sparse past inputs, anti-collapse alternatives to SIGReg including pooled and flow-based variants, multiple-future targets) under a shared protocol, then a staged dense-sparse campaign spec.

**Oct 8 to 15 (evidence).** The 51-model campaign showed that input masking is roughly neutral for dense prediction, that the CLS baseline remains stronger, that SIGReg strength dominates, that pooled SIGReg hurts, and that removing SIGReg collapses every run. Masking did not deliver a benefit, and anti-collapse remained the crux.

**Oct 8 to 19 (pivot to SPT).** In parallel the user moved toward building on the collaborator's SPT (streaming video model with sparse causal histories and a flow decoder). The residual SPT proposal (a fixed predictive subset with a full-state identity carry) became, by Oct 19, a three-component design: H/L partition, residual transition, JEPA latent loss, with reconstruction meant to replace SIGReg/EMA/stop-gradient. The handoff fixed the residual model as the common reference and a global soft-mask L1 as the first capacity control, with collapse avoidance stated as a hypothesis.

**Oct 20 to 24 (sparse-update branch).** A second, collaborator-owned branch asked whether constraining updates instead of states could prevent collapse, and whether latent-action codes could unify temporal and action-driven change. The gated runs collapsed for a regularizer-placement reason, and the upstream 80% recipe turned out to depend on state regularization, so update-only regularization lost support. Latent-action ideas were found to overlap with Delta-JEPA, MotionJEPA and latent-action-model work.

**Oct 24 (reframing).** The user's reflection shifted the question from "which partition stores what" to "what should the representation be sufficient for". The answer separated predictability, persistence and dynamics-relevance; rejected current-frame negatives; showed masking is an inductive bias already explored by VideoMAE, V-JEPA and especially MiMo; flagged that SPT's post-encoder masking does not remove information; and proposed a smallest design (masked history, separate JEPA predictor, appearance carried from observed history into the flow decoder, an explicit collapse guard). The anchored-state design doc dated Oct 25 appears to continue from here (anchor route for appearance, transition prior, renderer, with masking explicitly not used as the information restriction), but it was not produced in these sessions.

**Still unresolved as of Oct 24.**

- Anti-collapse without SIGReg/EMA/stop-gradient: reconstruction alone permits an empty H, and every concrete result so far (LeWM no-SIGReg, upstream sparse-update ablation) says an explicit state-level guard matters. Whether a principled alternative (capacity control, action routing, grounded latent actions, an anchored appearance route) suffices is untested.
- Whether H actually carries action-dependent dynamics: no SPT experiment with the residual or soft-mask design had been run or reported back by the collaborator.
- Identifiability conditions: the mechanism-sparsity and latent-action identifiability results need assumptions not obviously met by the architecture; no theorem yet covers the proposed model.
- Masking's role: post-encoder SPT masking does not remove information; moving it earlier is a substantive change and its benefit is unproven.
- Planning interface: how a compact history state plus an action-conditioned transition would replace SPT's generate-and-re-encode loop for PushT planning.
- Sparse-update branch: the upstream 80% result is not yet reproduced locally, so the masking and latent-action comparisons have no trustworthy dense anchor.
- Dense-sparse LeWM campaign: conditional follow-ups and sealed test evaluation never ran.
