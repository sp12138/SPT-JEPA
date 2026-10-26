# Lingjing's research line: separating dynamics from appearance on SPT

Branch: `lingjing/anchored-state-jepa`. Moved here on 2026-10-25 from a private
LeWM / stable-worldmodel fork (`sparse_jepa`), where the ideas were developed.
Only plans, design notes and discussion logs were carried over; datasets,
checkpoints, results, PDFs and literature dumps stay outside this repository.
Nothing in this folder has been implemented in SPT-JEPA yet.

The goal is a video world model, trained end to end on top of SPT's ViT, causal
transformer and DiT flow renderer, whose learned state holds what the dynamics
need and leaves persistent appearance and unpredictable detail to other
channels. The state must support latent planning, generation and probing.

## Where to start

0. [as-jepa-story.md](as-jepa-story.md): the canonical statement (2026-10-26):
   formulation, the objective and how the JEPA loss emerges from it, the
   architecture, the block-identifiability theory with proofs, and the
   narrative for the JEPA community.
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
