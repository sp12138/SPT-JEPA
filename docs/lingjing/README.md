# Lingjing's research line: separating dynamics from appearance on SPT

Branch: `lingjing/anchored-state-jepa`. Moved here on 2026-10-25 from a private
LeWM / stable-worldmodel fork (`sparse_jepa`), where the ideas were developed.
Only plans, design notes and discussion logs were carried over; datasets,
checkpoints, results, PDFs and literature dumps stay outside this repository.
The collaborator implemented and ran the first round on branch
`research/ideas-20261028`; this folder holds our plans, reviews and the round-2
reference implementation.

The goal is a video world model, trained end to end on top of SPT's ViT, causal
transformer and DiT flow renderer, whose learned state holds what the dynamics
need and leaves persistent appearance and unpredictable detail to other
channels. The state must support latent planning, generation and probing.

## Where to start

- [../results-index.md](../results-index.md) (2026-11-21): one index of every
  result in this repository, by line of work (the original framework with the pixel
  renderer, the pixel-free variants, the evaluation assets), with protocols, sources
  and a map of all experiment folders.
- [decoder-free/handoff-two-tracks.md](decoder-free/handoff-two-tracks.md) (2026-11-18):
  the current pixel-free proposal. The anchor reader, a learned denoiser that
  estimates what the state carries beyond the anchor, with time-correlated posterior
  noise (`rho = 0.8`), anchor-relative coordinates and no initial-scale calibration.
  Section 4 is the PushT test that decides whether it works. It replaces Parts P and R
  of [decoder-free/handoff-anchor-reader.md](decoder-free/handoff-anchor-reader.md)
  (2026-11-16); Part G of that document, the anchor-gap diagnostic on existing
  checkpoints, is still requested. The Reach3D arms of Section 5 were closed by a
  measurement on frozen models (see the results index). Origin:
  [discussions/2026-11-15-latent-action-and-decoder-free-directions.md](discussions/2026-11-15-latent-action-and-decoder-free-directions.md).
- [as-jepa-evaluation-handoff.md](as-jepa-evaluation-handoff.md) (2026-11-16,
  revision 2): the evaluation design and story. A two-stage protocol for measuring
  how latent world models scale (own objective without action prediction, then frozen
  or fine-tuned probes), three measurements (policy alone, representation probes in
  physical units, policy plus latent planning), CEM as a validation subset, the
  contribution assessment, benchmarks, model series and a step-by-step agent plan on
  the RoboJEPA code base.
- [decoder-free/](decoder-free/) (2026-10-31): the theory of the pixel-free line
  ([decoder-free/story.md](decoder-free/story.md)) and its reference code and tests.
  The first launch spec ([decoder-free/spec.md](decoder-free/spec.md)), which used a
  closed-form Gaussian estimate of the same quantity, is superseded: all 62 PushT
  recipes built on it failed. Origin:
  [discussions/2026-10-31-decoder-free-as-jepa.md](discussions/2026-10-31-decoder-free-as-jepa.md).
- [as-jepa-round2-spec.md](as-jepa-round2-spec.md): the current handoff spec
  (2026-10-31, revision 3). Round 2 trains the renderer, with probability `lam` per
  frame, on the prior's forecast instead of the encoded state, which makes the
  objective reward information that predicts the future (sufficiency). The design
  exploration covers `lam`, priced or noiseless forecasts, one-step or open-loop
  horizons, correlated posterior noise (`rho`) and the code resolution (`sigma0`):
  - PushT: a free diagnostic of rollout contraction on existing checkpoints, matched
    LeWM, a forecast factorial, and the initial scale (with a sham control) and
    `sigma0` on the seeds where the original arm failed.
  - TwoRoom: why the state never left its initial scale, and whether starting it at
    `sigma0` fixes that; priced against noiseless forecasts; `sigma0`.
  - Cube: whether a past-only or constant anchor, or a smaller `sigma0`, moves the
    cube into the state and into the planning cost.

  The dot and colour purity tests wait for a later round. The spec also divides the
  work with the collaborator's harder-task campaign. Reference implementation and 24
  CPU contract tests in [round2/](round2/).
- [as-jepa-results-review-20261028.md](as-jepa-results-review-20261028.md) and
  [as-jepa-status-and-next-steps-20261029.md](as-jepa-status-and-next-steps-20261029.md):
  the review of the collaborator's first-round results and the plan that
  followed it (its Phase 1 is superseded by the round-2 spec).

0. [as-jepa-story.md](as-jepa-story.md): the canonical statement (2026-10-26):
   formulation, the objective and how the JEPA loss emerges from it, the
   architecture, the block-identifiability theory with proofs, and the
   narrative for the JEPA community.
1. [as-jepa-experiment-spec.md](as-jepa-experiment-spec.md): the handoff
   spec for the first round of experiments (2026-10-26, revised the same day):
   standard PushT only, AS-JEPA first in two code bases (L2 in LeWM's, S2 in
   SPT's) with a staged search over `beta`, `sigma0`, anchor gap and window,
   LeWM and SPT baselines as comparisons, named constants, shared data and
   evaluation, component-level implementation, gates, failure responses.
1. [anchored-state-jepa-design.md](anchored-state-jepa-design.md): the
   design notes and decision history (AS-JEPA, 2026-10-25). One objective, a flow loss plus a KL chain;
   no augmentation, EMA or stop-gradient. Section 11 lists settled decisions
   and open questions.
2. [as-jepa-spt-implementation-plan.md](as-jepa-spt-implementation-plan.md):
   how AS-JEPA is built (2026-10-25): LeWM's harness, encoder and predictor
   with SPT's `WorldDiT` renderer for the PushT suite, SPT's own stack at
   scale; the literal diff from LeWM; data, diagnostics, order of work.
3. [discussions/2026-10-24_to_10-25-as-jepa-design-session.md](discussions/2026-10-24_to_10-25-as-jepa-design-session.md):
   how that design was reached, turn by turn, including rejected alternatives,
   points the user did not explicitly settle (SPT-style masking), and offers
   not yet taken up (identifiability proof, PushT experiment plan, probe
   evaluation design).
4. [discussions/2026-10-08_to_10-24-codex-sessions.md](discussions/2026-10-08_to_10-24-codex-sessions.md):
   the earlier trajectory: LeWM dense-sparse campaign evidence (SIGReg carries
   most of the anti-collapse load), the residual-SPT collaborator handoff,
   sparse-update findings, and the Oct 24 masking/literature discussion.

## Other files

- [spt-collaborator-handoff.md](spt-collaborator-handoff.md) (2026-10-20): the
  residual-transition SPT extension handed to SPT collaborators. AS-JEPA
  supersedes it as the design direction, but it is the most concrete spec of
  edits to SPT modules.
- [superseded/](superseded/) (2026-10-08): the residual-SPT proposal, technical
  notes and prior-work assessment. The fixed-slot idea survives in AS-JEPA as
  learned state queries.
- [archive/lewm-plans/](archive/lewm-plans/) (2026-10-04 to 10-06): the LeWM
  exploration plans (sparse past, anti-collapse, multiple future, shared
  protocol) and the dense-sparse campaign spec. They target the LeWM codebase,
  not SPT, and are kept for context.

## Working conventions for this branch

- Keep plans, design changes and discussion logs in this folder and commit them
  to this branch; do not push to `main`.
- After a substantive discussion, add or extend a dated log in `discussions/`
  recording what was asked, what was decided and why, and what stays open.
- The repository's top-level `AGENTS.md` and `HANDOFF.md` describe the SPT
  maintainers' environment (EFS paths, specific dev machines, their running
  jobs). Do not touch their jobs or outputs; agree compute placement with the
  user before launching anything.
