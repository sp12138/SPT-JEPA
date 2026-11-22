# Goal C: the staged evaluation protocol, run systematically

Confirmed by Lingjing on 2026-11-22. Session: the third shared session. Home
machines: two of the five, as listed in the shared `MACHINES.md`. Operating rules:
[README.md](README.md). Design:
[../as-jepa-evaluation-handoff.md](../as-jepa-evaluation-handoff.md).

## 1. What this direction pursues

A protocol for measuring how world models scale, and a study that uses it. Stage 1
is each model's own pretraining, with no action prediction. Stage 2 adapts the
model on fixed labelled data under a fixed contract, frozen, partially fine-tuned
or fully fine-tuned, and measures: a policy alone (E1), physical-state probes on
encoded and on predicted states (E2a, E2b), and policy plus latent planning (E3).
A small CEM row ties the results to the published planning numbers.

The design is settled. This direction is mostly systematic work: build the kit,
launch, debug, and fill the tables with care about what is matched. The design
choice that remains is the scope of Section 3.

## 2. Where it stands (2026-11-22)

- Reach3D reference sweep of public RoboJEPA predictors: 92 of 384 cases done.
  Strict 2 mm successes so far are near zero (22M 1 of 17, 50M 0 of 17, 100M 0 of
  11), so strict Reach3D success alone will not separate the models; the normalised
  return and the final distance have to be reported with it.
- The state probe and the policy head each passed its identity check; 13 of 40
  episodes cached for them.
- dev3 became unreachable at 02:50 UTC and the session paused on that. The machine
  manual now applies (README, Section 4).
- An existing finding of the kind the protocol is for: on the same checkpoints the
  planner's return fell (78.85 to 73.37) while the linear readout rose (81.37 to
  85.70).

## 3. Scope

Confirmed core, chosen so that the first multi-family result needs no new
pretraining and the hard benchmark is the one direction A targets:

| Dataset | Why | Measurements |
| --- | --- | --- |
| Meta-World MT50 | hard manipulation where planners give about zero for every latent world model; simulator state and goal images exist; six retrained public models already on the EFS | E1 in three regimes, E2, E3, CEM numbers already recorded |
| RoboCasa goal-image suite (Reach, ObjectReach, ObstacleReach, Push) | continuity with RoboJEPA, the released predictor series | E1, E2, E3, CEM on Reach and Push |
| PushT and Cube | pilot; shared with directions A and B | E1, E2, E3 |

| Model family | Source | Role |
| --- | --- | --- |
| RoboJEPA predictors (22M upward, as many sizes as the machines allow) | public weights | the predictor scales, the encoder is fixed |
| V-JEPA 2 / 2.1 encoders | public weights | the encoder scales |
| Six public latent world models retrained on Meta-World (LeWM, LeWM+IDM, SMWM, AC-MTM, LpWM, H-JEPA; 100 and 300 epochs, three seeds) | existing checkpoints, if still on the EFS | the first cross-family table: the planner puts all of them near 10 to 14%; does the staged protocol separate them? |
| AS-JEPA with the renderer | direction A | controlled arm |
| Decoder-free arm | direction B, when it has a recipe | controlled arm |
| Native LeWM (SIGReg) at several sizes | trained here | the nearest end-to-end JEPA world model |
| Scratch (no stage 1) | trained here | does pretraining help at all? |

Deferred until the core figures exist: RoboCasa365 and LIBERO-Plus (they need a
model pretrained on a large corpus), DROID and RoboArena, and a generative video
world model arm trained from scratch (the most expensive arm). When C2 is written
up, include a cost estimate for the video arm on Meta-World so that Lingjing's side
can decide whether to add it to the grid.

## 4. Milestones

| Id | Work | Done when |
| --- | --- | --- |
| C0 | Finish the Reach3D reference sweep and the table of existing curves (mandate steps 2 and 3), reporting return and final distance next to strict success. | milestone report |
| C1 | **The stage-2 kit**, usable by A and B: an adapter interface (`encode(frames) -> tokens`, `predict(tokens, actions) -> tokens`); the head trainer for the three regimes; the state-probe trainer; E1 and E3 loops with shared seeds; the E2b rollout probe; environment loops for RoboCasa goal-image, PushT, Cube and Meta-World; one config file holding every constant of the handoff's Section 3.4. | the handoff's checks pass (E3 with one proposal equals E1; E2b at horizon 0 equals E2a; the CEM row reproduces 0.887 on Reach); a README that lets A and B plug a model in; constants sent to Lingjing's side for one confirmation |
| C2 | **First results without new pretraining:** the six Meta-World models and the old AS-JEPA checkpoints under E1 (frozen), E2 and E3, next to their recorded CEM numbers; the RoboJEPA series under E1, E2, E3 and CEM; V-JEPA encoders under E1 and E2a. | tables, and a written answer to: do the staged measurements rank models differently from the planner, and does the predictor's size show in E2b and E3 but not in E1 and E2a? |
| C3 | **Controlled grid** on Meta-World: arms (AS-JEPA from A, decoder-free from B when ready, native LeWM, scratch) by three sizes by three data fractions, checkpoints at common compute points; frozen E1 with three head seeds, E2 and E3 everywhere; partial and full fine-tuning at the endpoints. | the handoff's figures 1 to 5 for Meta-World |
| C4 | **Validation of the protocol:** rank agreement of E3 gain and E2b with CEM across checkpoints; probe variants (lighter head, linear probe); the cost table in ms per decision. | figures 6 and 7 |
| C5 | The deferred scope, in the order Lingjing's side decides. | per decision |

C1 is the critical path for A and B. C0 runs on the GPUs while C1 is being written.

## 5. Rules specific to this direction

- A probe's or head's definition is not changed after results from it have been
  seen. If a constant must change, every affected number is rerun.
- No per-model tuning of the head: one learning rate chosen once on the reference
  model, checked at 0.3 and 3 times on two arms.
- The final checkpoint is reported, never the best on test episodes.
- Every table states what is and is not matched (data, compute, parameters,
  frames per window).
- A public model that cannot be reproduced within 5 points under its own settings
  is reported with the gap and not tuned.
- **Needs agreement first:** the constants of the contract (once), any change to
  the Reach3D development cases or the 2 mm rule, opening the 32 sealed cases,
  adding or dropping a dataset or model family.

## 6. Backlog (always available)

More head and probe seeds; the staged measurements on every checkpoint A and B
save; larger RoboJEPA sizes as cost allows; label-fraction curves; probe variants
on more checkpoints.

## 7. Goal text for the session to set

> Goal C (evaluation). Build the staged evaluation protocol for world models and
> run it systematically: stage 1 is each model's own pretraining without action
> prediction; stage 2 adapts it under a fixed contract (frozen, partial, full) and
> measures a policy (E1), physical-state probes on encoded and predicted states
> (E2), and policy plus latent planning (E3), with a small CEM row. Follow
> docs/lingjing/goals/README.md and C-evaluation.md on branch consolidate/20261121:
> milestones C0 to C5. C1, the stage-2 kit that directions A and B plug their
> models into, is the critical path; run C0 on the GPUs while you write it. You
> plan, launch, monitor, debug and write up everything yourself on your home
> machines (see MACHINES.md), and restore a machine that goes down by the machine
> manual. Keep their GPUs busy from your QUEUE.md in the priority order of the
> README; never launch a run only to occupy a GPU; lend GPUs you cannot use. Before each
> batch write the question, arms, decision rule and GPU-hours. Do not change a
> probe or head after seeing its results, do not tune per model, report the final
> checkpoint. When you need an answer from Lingjing's side, write the question and
> continue with another milestone or the backlog. The goal is not complete before
> C4; a milestone that cannot be met ends with a written account and you move on.
> Update STATUS.md hourly while anything runs, write a milestone report at each
> milestone, and post a five-line status in the chat.
