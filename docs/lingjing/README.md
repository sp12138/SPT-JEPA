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

- [as-jepa-round2-spec.md](as-jepa-round2-spec.md): the current handoff spec
  (2026-10-31, revision 2). Round 2 trains the renderer, with probability `lam` per
  frame, on a `sigma0`-noisy sample of the prior's forecast instead of the encoded
  state, which makes the objective reward information that predicts the future
  (sufficiency). Three parts run in parallel:
  - PushT closure, about 100 GPU-hours: matched LeWM, `lam = 1` over four seeds,
    the initial state scale, the dot and colour purity tests, and a
    reconstruction-free control.
  - TwoRoom: why the state never left its initial scale, and whether starting it at
    `sigma0` fixes that.
  - Cube: whether a past-only or constant anchor moves the cube into the state and
    into the planning cost.

  It also divides the work with the collaborator's harder-task campaign. Reference
  implementation and 16 CPU contract tests in [round2/](round2/).
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
