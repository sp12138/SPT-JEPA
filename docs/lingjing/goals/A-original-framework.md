# Goal A: the original framework, generation and a control representation in one model

Confirmed by Lingjing on 2026-11-22. Session: the first shared session. Home
machines: two of the five, as listed in the shared `MACHINES.md`. Operating rules:
[README.md](README.md).

## 1. What this direction pursues

In Lingjing's words (2026-11-21): "the original framework method that can handle
both generation and representation learning for planning". The canonical statement
([../as-jepa-story.md](../as-jepa-story.md), Summary): "a world model whose learned
state holds what the dynamics need and leaves out persistent appearance and
unpredictable detail, so that it supports latent planning, generation and probing
from single frames."

The target result: one recipe of this framework, trained without predicting
actions, whose state supports decent success on hard manipulation tasks and whose
renderer generates. The hard tasks are the six Meta-World manipulation tasks (push,
pick-place, drawer-open, door-open, peg-insert-side, bin-picking). Every world
model tried on them so far that does not learn a policy during its main training
scores about zero with a sampling planner: this framework 0%, and six retrained
public latent world models 5 successes in 2,160 cases.

## 2. Why the evaluation changes

A sampling planner (CEM) on a latent distance mixes the encoder, the predictor, the
hand-chosen distance and the sampler. A zero from it does not say that the state
lacks what control needs. Direction C's staged protocol
([../as-jepa-evaluation-handoff.md](../as-jepa-evaluation-handoff.md), Sections 0.2
and 3) separates these:

- **Stage 1:** the model trains with its own objective. Actions are inputs only.
- **Stage 2:** on fixed labelled task data, a fixed policy head is trained with the
  model frozen, with its last blocks trainable, or fully trainable.
- **E1:** the head alone, closed loop. **E2:** probes of physical state on encoded
  states (E2a) and on states the predictor rolls forward (E2b). **E3:** the head
  proposes 16 action chunks and the frozen predictor picks one; E3 minus E1 is what
  the world model itself adds.

This direction uses that protocol as its main evaluation, with CEM kept as a
reference row. The evaluation-stage rule set of 2026-11-22 (commit `77ee0551`)
applies to stage 2; `AGENTS.md` keeps governing stage 1.

## 3. Datasets

In this order, each stage deciding the next:

1. **PushT, Cube, TwoRoom with the checkpoints that already exist.** No new
   pretraining. The question is whether the staged measurements show something the
   planner hides. Example on Cube: the planner's cost follows the effector (partial
   rank 0.91) and not the cube (0.03); E2a says whether the cube's pose is in the
   state at all, and E1 says whether a head can use it. A benchmark stays in the
   suite only if the arms separate on it and it is not at its floor or ceiling.
2. **Meta-World MT50** (`lerobot/metaworld_mt50`, 2,500 scripted episodes, already
   on the EFS) as the main benchmark: the six hard tasks as the headline, all 50 as
   the broad table.
3. Larger suites (RoboCasa goal-image tasks) only through direction C's grid.

## 4. Where it stands (2026-11-22)

- **Recipe A-ref (F8):** renderer trained on one-step forecasts (`lam = 1`), noise
  correlated in time (`rho = 0.8`), `sigma0 = 0.1`, `beta = 0.25`.
- **PushT:** 90.44% / 90.89% over three seeds against native LeWM 84.89% / 78.67%
  at three epochs (unmatched: 8 frames per window against 4, parameter count).
- **Cube:** 18% / 20% against 26% / 27%; cost follows the effector.
- **TwoRoom:** the state is never used; a noise-scale batch was due to finish at
  13:00 UTC on 2026-11-22 and its reading belongs to A0.
- **Meta-World (older recipe, CEM):** Reach 87.5%, ReachWall 90%, six hard tasks 0%.
  A goal-conditioned action model trained end to end on the same data reaches 68%
  on the six tasks, so the demonstrations do contain the needed signal.

## 5. Milestones

| Id | Work | Done when |
| --- | --- | --- |
| A0 | Close the running TwoRoom batch and the Cube audit; write the milestone report for the mandate's four requirements; name A-ref as the reference recipe. TwoRoom gets at most one more written batch, then is parked with an account of what was ruled out. | report written |
| A1 | Staged measurements on existing PushT, Cube and TwoRoom checkpoints of A-ref and native LeWM: E2a and E2b first (state probes need no kit), then E1 and E3 when C's kit is released. | a table per benchmark and a written answer: which benchmarks separate the arms, and what the staged view shows that the planner did not |
| A2 | Meta-World stage 1: A-ref on MT50, two seeds, with native LeWM at the same budget (or the existing 100-epoch baseline checkpoints if they are still on the EFS). Generation numbers and E2a / E2b for effector and object poses. | checkpoints, probes and generation table |
| A3 | Meta-World stage 2 with C's kit: E1 frozen, partial and full; E3; a scratch arm (same head and encoder, no stage 1); label fractions 10%, 30%, 100%. | the table of Section 6, three head seeds |
| A4 | Design changes that the evidence calls for (Section 7). This rung may start as soon as the evidence exists; it does not wait for A3. | each change has a proposal entry, a screen and a written conclusion |
| A5 | Scale the chosen recipe: three model sizes by three data fractions on MT50, checkpoints at common compute points, handed to direction C. | curves delivered to C |

A1 and A2 start together: A2 trains on one machine while A1 measures on the other.

## 6. What success looks like on Meta-World

Working thresholds. Lingjing's side confirms or revises them when the first A3
numbers exist; until then they guide the work and do not end it:

1. Frozen E1 on the six hard tasks: at least 50% mean success, where the planner
   gives 0%. This shows the state holds what control needs.
2. Frozen E1 at least as high as native LeWM's under the same head and budget.
3. Pretraining helps: A-ref beats the scratch arm by at least 10 points at the 10%
   or 30% label fraction.
4. E3 minus E1 is not negative on the tasks with goal images, and positive on at
   least two of the six.
5. Generation numbers reported for the same checkpoints (256-clip FVD, LPIPS, SSIM).
6. The recipe is the same one that holds the PushT result.

A documented account of which of these cannot be met, and why, completes a
milestone but does not end the goal: it becomes the input to A4.

## 7. Design changes: what is open and how to propose one

The architecture stays close to the original design: a noisy state per frame, a
predictor that prices each state, a renderer that redraws the frame from the state
and an anchor. Within that, changes are welcome when they are principled, meaning
derived from the objective (describe the video in as few nats as possible given the
anchor) and not added as a patch.

Evidence that would call for a change, and the kind of change it points to:

| Evidence | Points to |
| --- | --- |
| E1 or E2a from the encoder's patch tokens is far above the 4 by 16 state tokens | the state's capacity or layout (more tokens, a spatial grid of state tokens) |
| Object pose is unreadable from both tokens and patches | what the renderer is asked to redraw: resolution (64 px today), the anchor rule |
| E2a is good and E2b degrades quickly with horizon | the predictor: horizon of the forecast the renderer sees, predictor capacity |
| The state is unused from the start (TwoRoom) | initial scale and `sigma0` relative to the measured signal size |

**You may explore a change without asking,** including a departure from the main
recipe's architecture, when two things are written in `PROPOSALS.md` before the
first run: the evidence that calls for it (measurements, on two seeds, not one
suggestive run), and the derivation from the objective (which term or component
changes, what the optimum becomes, what the state should now keep and drop). Start
with a two-seed screen at a short budget against the unchanged recipe; if the screen
supports the change, continue it at full length and through the staged
measurements. Keep the unchanged recipe running beside it as the comparison.

**Stays fixed** (these define the direction; write a question to Lingjing's side if
the evidence seems to point here): no action-prediction loss in stage 1; no EMA
target; no augmentation; no stop-gradient or gradient reweighting between renderer
and encoder used as a fix; no pretrained component; no auxiliary loss on the state
that is not derived from the objective.

**Reviewed by Lingjing's side:** which recipe is named the canonical one in a
milestone report. The review does not block further exploration.

## 8. Backlog (always available)

More training seeds behind the PushT and Meta-World claims; generation numbers for
every reported checkpoint; the staged measurements on every checkpoint this
direction has saved; the anchor-gap and rollout-gain diagnostics on Meta-World.

## 9. Goal text for the session to set

> Goal A (original framework). Pursue one recipe of the state, predictor and pixel
> renderer framework, trained without predicting actions, that under the staged
> protocol reaches decent success on Meta-World's six hard manipulation tasks and
> also generates. Follow docs/lingjing/goals/README.md and A-original-framework.md
> on branch consolidate/20261121: milestones A0 to A5 in order, A1 and A2 in
> parallel. You plan, launch, monitor, debug and write up everything yourself on
> your home machines (see MACHINES.md), and restore a machine that goes down by the
> machine manual. Keep their GPUs busy from your QUEUE.md in the priority order of
> the README; never launch a run only to occupy a GPU; lend GPUs you cannot use.
> Before each batch write the question, arms, decision rule and GPU-hours. Two
> seeds behind every statement. You may explore a principled change, including to
> the architecture, when strong evidence supports it: write the evidence and the
> derivation in PROPOSALS.md first, keep the fixed constraints of Section 7.
> When you need an answer from Lingjing's side, write the question and continue
> with another milestone or the backlog. The goal is not complete before A5 is
> delivered; a milestone that cannot be met ends with a written account and you
> move on. Update STATUS.md hourly while anything runs, write a milestone report at
> each milestone, and post a five-line status in the chat.
