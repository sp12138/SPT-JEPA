# Discussion log: Claude session 6e5adacb (Anchored State JEPA design)

Source transcript: `~/.claude/projects/-mnt-weka-home-lingjing-kong-code-sparse-jepa/6e5adacb-10f8-4bd9-9e4d-d5e4f63b441d.jsonl`, 2026-10-24 17:19 UTC to 2026-10-25 14:09 UTC. Five user turns, five substantive assistant answers. Nothing was implemented or trained; the only artifact is the design doc `docs/lingjing/anchored-state-jepa-design.md` (originally `docs/experiments/` in sparse_jepa) (working name AS-JEPA, "the doc" below). A memory file `jepa-design-preferences.md` was also written during the session.

This log records how the design got to the doc, what was dropped on the way and why, and how the user likes to work. For the final architecture, objective, theory and experiment, read the doc; section pointers are given instead of repeating it.

## 1. Starting point (turn 1, 2026-10-24 17:19)

The user pasted a long voice-style brainstorm plus the answer from an earlier session, and asked to make the architecture concrete, scrutinize it and improve it. The stated goal, which stayed fixed all session: a JEPA world model that separates high-level from low-level representation, "ideally should be trained end-to-end", built on top of SPT (the repo's stochastic autoregressive video model: ViT encoder, causal transformer over 90%-masked history tokens, DiT flow decoder predicting a future chunk at random offset k).

The user's own ideas in that brainstorm, roughly in their words:

- What should be learned is "the prediction rule from the past to the future". Prediction from the past excludes the unpredictable part, but persistent appearance is also predictable ("even more persistent"), so prediction alone keeps it.
- Maximize information between past and current/future with a CLIP-like loss; "whatever cannot be predicted from the past will be resorted to from the current state".
- Time asymmetry as the key: "the persistent part could be formulated as the shared information between the immediate past and the entire history"; "dynamics would only emerge from the long history". Possible losses: immediate current state as a contrastive negative, or maximize past-future information while minimizing current-future information (user already suspected these conflict).
- Masking in SPT: spatial tube masking plus dropping long temporal stretches, "large masking ratio and, more importantly, large masking patch sizes" (the MAE lesson), to force high-level content.
- Decoder doubts: is the SPT flow noise "the low-level information or exogenous information" so that the causal-transformer context is the high-level part? Latent versus observation-space targets; is perfect reconstruction needed; should the causal transformer output match the next latent (user thought not, it is a summary of the past); planning needs per-step states.
- Suspected temporal MAE had been tried; asked for literature.

The earlier session's answer (embedded in turn 1) had already: split future information into static, dynamic, and unpredictable; mapped the idea to the conditional quantity I(F; H | X_t) (Crutchfield's elusivity); argued masking cannot remove persistent content because it is in every visible frame; argued flow noise is a sampler, not a representation; proposed a "content stream" (one frame fed only to the decoder) plus a capacity-limited history; pointed to LRC-JEPA (Sep 2026) as closest precedent.

## 2. Round 1: concrete "SPT-JEPA" (research-refine, turn 2 to 2026-10-24 17:41)

The user invoked the `research-refine` skill on: content stream + capacity-limited history stream + flow decoder. The assistant proposed:

- Per-chunk shared ViT with K_h = 4 learned state queries; state head outputs mu, sigma, noisy state h; a per-chunk rate (KL to N(0, I)).
- Content head on an anchor chunk (first chunk of the window), routed only to the decoder.
- Causal predictor over noisy states and actions gives belief c_t; a readout gives h_pred = h_t + delta at offset k.
- DiT decodes x_{t+k} from [h_pred; content]; plus L_JEPA = MSE(h_pred, stopgrad(mu(x_{t+k}))); loss = L_flow + lam L_JEPA + beta rate.
- Two independent photometric augmentation views (history under A1, anchor and target under A2) so appearance in h is "wrong, not merely redundant".
- Routing table (changing state and dynamics-relevant static factors to h; render-only static factors to content; unpredictable to flow noise; predictable but control-irrelevant stays in h by default).

Corrections made to the earlier sketch in this round, all of which survived into the doc:

- **Elusivity I(F; H | X_t) was the wrong target.** Conditioning on the current frame strips current positions too and leaves only velocity-like memory. The user's "immediate past versus whole history" idea works when conditioning on an *earlier anchor*: I(F; H | x_anchor) is the dynamic state.
- **The anchor must not be the current or recent frame,** or the decoder extrapolates motion itself and bypasses the state. DBSE evidence: fixed first-frame anchor works, random anchors leak dynamics.
- **c_t alone cannot plan;** per-chunk states are needed so goal images can be encoded without history.
- **Masking demoted to an optional knob;** random dropping does not make a network discard detail it can use when present.
- **Immediate current state must never be a contrastive negative,** since it shares exactly the state to keep. Same-clip contrastive became an optional alternative, then disappeared.
- **Rate penalty is required,** because no predictive loss penalizes predictable redundant information; token count alone is not a capacity limit.

Proposed PushT falsification experiment (render-only vs dynamics-relevant factors from PushT's variation space, WeakPolicy data, probes, swap test, colour-shifted goals). This experiment design carried through to doc Section 9 with modifications.

## 3. Round 2: user strips the heuristics (turn 3, 2026-10-24 22:26)

User: "I'd like to remove visual augmentation, EMA from the architecture." Asked that the design be inspired by information theory or causal representation learning (identifiability), while "Empirical success and elegance are more important though. Prefer simplicity as well."

Revision (the core of the final design): a variational state-space model reusing SPT's three modules.

- ViT becomes a per-chunk encoder q(z_t | x_t); causal transformer becomes a *prior* p(z_t | z_<t, a_<t); DiT becomes a renderer of x_t from z_t and anchor x_0.
- Single objective: flow loss + beta times a KL chain (KL to N(0, I) at t = 0, then KL to the learned prior). See doc Section 2.
- Removed in one stroke: augmentation, stop-gradient (assistant clarified the previous design used stop-gradient, not EMA), separate JEPA loss, separate rate term, readout, high-noise-bias guard. Reasons: entry cost at t = 0 removes appearance so augmentation is unnecessary; the KL is scale-invariant (pinned by N(0, I)) so it cannot be gamed by shrinking like an attached MSE; the renderer sees no history, so it cannot do the dynamics.
- Decoder now reads the *encoded current* state, not the predicted future state. The previous predicted-state design was what had needed the stop-gradient; the KL accounting now keeps transient detail out (bits-back break-even argument).
- Principles table introduced: Wyner-Ziv side information (anchor), description length (KL chain), bits-back (stochastic renderer), conditionally independent innovations (iVAE, LEAP, TDRL), actions as interventions (CITRIS, BISCUIT), residual prior mean (SlowVAE).
- Identifiability at this point was only component-wise (TDRL-style), framed as a "measurable bonus" via MCC. The user later found this insufficient (round 3).
- Risks listed: early posterior collapse (beta warm-up; KL per dim gives effective state size, replacing L1 gates from the residual-SPT handoff), unimodal prior, beta scale, contact factors dropped at large beta, latent teacher forcing (latent overshooting suggested; later removed). First experiment gained a transient random dot to test the noise route, and ablations "no anchor" and "deterministic MSE renderer".

## 4. Round 3: write it down, justify, sell, prove (turn 4, 2026-10-25 13:00)

User accepted the direction ("this proposal makes sense in general. Could you please note it down?") and asked for:

1. Component-by-component description with justification tied to principles. Now doc Section 3.
2. "an appealing story to convince people in the jepa community", who "are not a big fan of reconstructing pixels, so we really need compelling arguments". Now doc Section 4 (pitch: "we do not reconstruct pixels, we price them"; two alibis; SIGReg as the entry term without the price via the ELBO decomposition; anticipated objections).
3. Identifiability that matches the architecture: the component-wise conditions "might not directly support our architecture"; the user wants to show the *partitions* are disentangled, "especially the high level one", and "The theory formulation and our architecture design should align". Answer: three latent blocks (z, c, eps) with one cheap channel each, and a block-identifiability claim. Now doc Section 5; component-wise kept as Section 5.4 bonus.
4. Practicality: KL "might not be the best for the powerful models nowadays. They might be too restrictive"; wanted relaxations or approximations of the theory, "clean, simple and modern". Answer: same objective with different codes (Gaussian KL first, FSQ plus categorical prior as discrete exact-rate option, fixed-noise encoder plus latent flow prior for multimodal futures). Now doc Section 6 table.
5. Permission to modify SPT if needed. Answer: doc Section 7 (kept vs changed).

Assistant created the doc in this turn.

## 5. Round 4: anchor conditioning, prior, noise, multi-step, masking, theory detail (turn 5, 2026-10-25 14:01)

User questions and the resolutions (all reflected in the doc):

- **How does the anchor condition the renderer; "similar to image editing", would that literature help?** Yes: on static-camera PushT, channel-concatenate the anchor with the noisy target (InstructPix2Pix style); state tokens via cross-attention or adaLN; for camera motion, in-context anchor tokens (FLUX Kontext, OmniGen) or ReferenceNet cross-attention. Analogy adopted as part of the story: `x_t = edit(x_0, instruction = z_t)`; precedent is reference-plus-pose video generation (AnimateAnyone, MagicAnimate). Carry-over tools: state dropout for classifier-free guidance on the state (guidance scale as reliance diagnostic). Doc 3.2.
- **Why a KL in the prior, versus SPT's direct conditioning vector? User's reading: "simply the equivalence of simplicity bias ... can also be formalized into a sparsity constraint."** Assistant agreed in spirit but sharpened it: the KL is the expected description length under the predictor, so it is both the JEPA prediction loss (Gaussian NLL with learned variance) and a price on *surprise*, not on size; "sparse in surprise rather than in activation". An L1/active-dimension penalty is a cruder code. Recommended simplest implementation: fixed variances, giving three lines (fixed state noise, squared prediction error, squared norm on the first state). Doc 2 and 3.3.
- **Flow noise versus "we can only observe a realization of one specific future".** Resolution: in the new design the renderer's noise is the within-frame residual given current state and anchor, not alternative futures (those live in the prior). One sample per context suffices for conditional generative models, like one label per input for a classifier. Doc 3.4.
- **User: "maybe we can stick to the exact next frame and ... remove the multi-step future prediction ... If so, please remove it."** Accepted and removed (SPT random offset k and PlaNet latent overshooting both gone). Prior is one-step; renderer renders the current chunk. Justification: does not serve the separation argument; Dreamer v2/v3 do without overshooting. Drift is now a monitored risk rather than designed in.
- **Masking: obsolete? Could it implement the KL restriction? User leaned toward it: "the masking scheme has been specified in the SPT architecture. It works there, so perhaps that is more empirically appealing. I'm not too sure about the KL divergence."** Assistant held its position: SPT's post-ViT token masking hides nothing (a retained token can encode the whole chunk), has no entry cost and trains no predictor; at best it is an erasure channel (Information Dropout view), an alternative noise rather than a simpler one. The small state already gives SPT's compute saving (4 tokens vs 26 of 256). Masking kept only optionally for ViT compute. Doc Section 6 last paragraph. The user's empirical worry about the KL was answered by the fixed-variance and FSQ forms, not by masking. This is a point the user may revisit.
- **Causal conditions for block identifiability, relevant literature, proof idea.** Assumptions A1 to A4, three-step proof (matching, coupling through the state, rate minimality), literature map. Doc 5.2 to 5.5. The assistant framed step 3 as "the minimal-change argument of your partial-disentanglement paper" (Kong et al. 2022), i.e. the user is an author in this lineage; Kong et al. 2023 (MAE hierarchical latents) also appears in the map.

The session ended with the assistant offering: formal proposition plus proof for block identifiability, a PushT experiment plan with configs and beta sweep, or detailed probe and MCC evaluation design. The user had not chosen.

## 6. Settled decisions (see doc Section 11 for the canonical list)

- No visual augmentation, no EMA, no stop-gradient, no target network. Reason: user preference for principle over heuristics; the KL chain makes them unnecessary (scale-invariant, entry cost removes appearance).
- One evidence-bound objective: flow loss + beta times code-length chain. Reason: single principled loss that is simultaneously prediction loss and rate.
- Anchor is the first chunk of the window, is never rendered, enters only the renderer (channel concat on PushT). Reason: Wyner-Ziv side information; a recent anchor would let the renderer bypass the state; a prior that saw the anchor would let dynamics-relevant static factors (block shape) skip the state.
- Per-chunk encoder that sees only its own chunk. Reason: plannable from a goal image, Markov state, prevents "pose relative to anchor" leakage.
- Prior is anchor-blind, one-step, receives actions; renderer is history-blind and action-blind.
- Renderer is stochastic (flow matching) on a fixed target (pixels or frozen codec). Reasons: bits-back; REPA-E shows a learned target under a diffusion loss collapses.
- Multi-step prediction removed (user request, accepted).
- Masking is not the information restriction.
- First implementation: fixed-variance Gaussian code on PushT; FSQ second if beta is painful; latent flow prior for action-free multimodal video.
- Identifiability claim targets the *block* (high-level partition); component-wise is a bonus.

## 7. Rejected or dropped alternatives (with reason)

- Elusivity I(F; H | X_t) as the objective: conditions away current positions. Replaced by conditioning on an early anchor.
- Simultaneous maximize past-future / minimize current-future information: conflicting objectives; replaced by a single rate-distortion bound.
- Immediate current state as contrastive negative: false negatives, breaks temporal smoothness needed for planning, and it shares exactly the state to keep.
- Same-clip contrastive (conditional InfoNCE) loss: offered as optional in round 1, dropped in the simplification.
- Masking (spatial tubes, long temporal drops, SPT token masking) as the means to discard low-level info: masking removes local, short-lived info and keeps what is in every frame; post-ViT masking hides nothing.
- Treating flow noise as the low-level representation: noise is randomly paired with data during training, so it is a sampler; inversion codes (DiffAE) are tangled and unusable for planning.
- Explicit inferred "unpredictable part" code (latent action, EEN style): only if control of it is needed; absorbs distractor motion.
- Decoding from the causal transformer's history summary (plain SPT): renderer can bypass the state and the summary carries appearance.
- Decoding from the encoded future vs predicted future: round 1 used the predicted state (needed stop-gradient); round 2 switched to the encoded current state with KL accounting.
- Readout module, separate JEPA MSE, per-frame rate: subsumed by the prior and KL chain (per-frame rate charged static factors every frame; the chain charges once).
- Photometric augmentation views: unnecessary once entry cost removes appearance; user disliked it.
- Learned latent target through a diffusion loss: REPA-E collapse.
- Random or per-batch anchors: DBSE shows dynamics leak into the static code.
- Multi-step offsets and latent overshooting: no role in the separation story; user asked to remove.

## 8. Open questions and next steps at session end

From doc Section 11 plus loose ends in the conversation:

- Anchor refresh for video with camera motion (content revealed after the anchor enters the state).
- An action-relevance term to drop predictable but control-irrelevant dynamics (multi-step inverse dynamics as in AC-State, or Delta-JEPA action decoding).
- One versus several state tokens; chunk length as the slow/fast hierarchy knob (Kong et al. 2023 masking-level analogy); separate weights for entry versus surprise terms.
- Whether the user is satisfied that the fixed-variance KL or FSQ answers the "KL too restrictive for modern models" worry; the user's inclination toward SPT-style masking for empirical reasons was argued against but not explicitly conceded.
- Beta selection: by contact action-following, not flow loss; ramp from 0 to avoid posterior collapse. The theory needs beta = 1 with a minimum-rate tie-break or slightly above 1.
- Unconfirmed calibration: PushT state about 40 nats (assistant's rough estimate), used as a leak detector for the entry KL.
- Novelty: several close papers appeared Sep to Oct 2026 (LRC-JEPA, Flow-JEPA, VANE, FlowWM, Delta-JEPA, SD-JEPA); re-search before claiming novelty. LRC-JEPA is the baseline to beat (reported about 9 points higher planning success than a parameter-matched JEPA on PushT, Reacher, TwoRoom, Cube).
- Offered but not started: formal block-identifiability proposition with proof (use `proof-writer`); PushT experiment plan with configs and beta sweep (`experiment-plan`); probe and MCC evaluation design.
- Implementation notes: keep LeWM's `encode` and `rollout` signatures (`stable_worldmodel/wm/lewm/lewm.py`) so existing planners and `GoalMSE` work; the model would sit beside `lewm/` and `genie/` in `stable_worldmodel/wm/`; PushT variation space documented in `docs/envs/pusht.md`.
- Relation to prior docs: AS-JEPA supersedes `docs/experiments/residual-jepa-spt-proposal.md` for the dynamics-versus-appearance question; the fixed-slot ("fixed J") idea survives as learned state queries; KL per dimension replaces the L1 gates in the residual-SPT handoff.

## 9. How the user likes to work

- Brainstorming register: long, dictated stream-of-thought prompts (speech-to-text artifacts such as "young ETC" for LeCun et al., "color transformer" for causal transformer, "called representation learning" for causal representation learning). Read for intent.
- Wants replies "brief and concise", "intuitive and easy to understand", organized per point raised, with the option to "dig deeper into each point ... later". Round 1 explicitly: first organize and summarize the user's thoughts, then address every uncertainty.
- Principles over heuristics: removed augmentation and EMA; wants each component justified by information theory (rate-distortion, Wyner-Ziv, bits-back) or causal representation learning (block identifiability). But "Empirical success and elegance are more important"; "Prefer simplicity"; "clean, simple and modern".
- Theory must align with the architecture, not be bolted on; the target result is disentanglement of the partitions, especially the high-level one, rather than component-wise identifiability.
- Audience to convince: the JEPA community (V-JEPA, LeJEPA, LeCun line), which dislikes pixel reconstruction; reconstruction must be argued as priced and excused.
- Is open to modifying SPT but likes its intuition ("The auto-regressive model summarizes all the past, and we introduce additional exogenous information to produce the future") and values empirically proven components (hence the masking pushback).
- Values literature grounding and asked for broad searches; is comfortable with the identifiability literature (iVAE, TDRL, LEAP, partial disentanglement).
- Accepts being corrected when the reason is explained (elusivity, current-frame negative), and makes explicit removal requests ("If so, please remove it") that should be applied to the doc directly.
