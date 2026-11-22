# Goal B: a decoder-free method for the JEPA family, with a theory of what its state keeps

Confirmed by Lingjing on 2026-11-22. Session: the second shared session. Home
machine: one of the five, as listed in the shared `MACHINES.md`. Operating rules:
[README.md](README.md).

## 1. What this direction pursues

A world model trained from random weights with no pixel loss anywhere, no EMA
target, no augmentation and no stop-gradient used against collapse, whose objective
is derived from a principle and comes with statements about what the state keeps
and drops. It is the contribution addressed to the JEPA community, so it stays
decoder-free without exception.

The principle is the one behind the original framework: the state should hold what
the past and the actions predict and the anchor frame does not already give. With
an optimal renderer the original objective depends on the encoder only through
entropies of the state, so the renderer can in principle be replaced by an estimate
of the state's entropy given the anchor
([../decoder-free/story.md](../decoder-free/story.md)).

## 2. Where it stands (2026-11-22)

All on PushT at one epoch, two seeds. "Agent" and "block" are R² of position read
from the noisy state; the renderer model A-ref at the same budget has agent 0.44 /
0.45 and block 0.80 / 0.86.

| Objective | Agent | Block | Reading |
| --- | --- | --- | --- |
| Anchor reader (learned denoiser), 22 runs | under 0.06 | under 0.08 | one direction, the share of the target covered by the block, takes 96 to 99.6% of the state; it was already the largest direction of the random encoder |
| Per-window log-det reward, `rho = 0.8`, price 1 (L1) | 0.40 / 0.40 | 0.16 / 0.16 | agent appears, three informative directions |
| Same, price 0 (L2) | 0.62 / 0.65 | 0.19 / 0.28 | agent above the renderer model, block not captured |
| Peng's Reach3D rule (next code = current code + action embedding) | 0.99 | 0.39 / 0.41 | the simplest decoder-free rule; block partly |

Two things are established. First, the closed-form reward removes the coverage
direction and finds the agent, so the learned reader was a blocker. Second, no
decoder-free arm holds the block, the object the task is about. Open: whether the
objective's optimum keeps the block when the state starts with it (the test from
the renderer model's weights stopped at its reader-calibration check and never ran
its main arms).

Working hypotheses, not findings: (a) the entropy reward's gradient on a direction
grows with what that direction already carries, so content the random encoder
expresses weakly is never discovered, where a pixel error names the missing content
directly; (b) the per-window estimate credits only what varies inside one window,
and the block is still in most windows.

## 3. Milestones

| Id | Work | Done when |
| --- | --- | --- |
| B1 | **Is the optimum right?** Start the closed-form objective (L1 and L2 settings) from the renderer model's one-epoch weights, with a reward-off control. No reader, so no calibration check is needed. Evaluate block and agent at 0, 1,000, 3,000, 6,000 and 12,491 updates. | block stays near 0.8 with the reward on and falls with it off (the optimum keeps it; the problem is discovery), or it is lost with the reward on (the objective drops it) |
| B2 | **Why is the block not found from scratch?** Without training where possible: the share of windows in which the block moves; the reward's gradient along the block's direction at initialisation and at 1,000 updates, next to the agent's; what each informative direction of L1 and L2 correlates with; forecast error over change from the anchor for agent, block and coverage under each arm. | a written account that names the cause with a measurement behind it |
| B3 | **Principled candidates** that follow from B1 and B2 (Section 4), each screened at one epoch on two seeds against L2. | an arm with agent and block within 0.1 of the renderer model's values on both seeds, or a written account of why this family cannot reach it |
| B4 | The arm that passes B3 goes to three epochs with the conventional JEPA evaluation: planning on seeds 42 to 44 and 1042 to 1044, probes, latent rollouts against copying. Native LeWM and the Reach3D rule are the decoder-free comparison rows. | planning within 5 points of native LeWM (84.89% / 78.67%) on both seed sets |
| B5 | Second and third benchmark: Cube, then Meta-World with direction C's stage-2 kit (E1, E2). | the same recipe, tables on both |
| B6 | The theory document for the recipe that works: which propositions hold for the estimator actually used, what is proven, what is only measured. | document reviewed by Lingjing's side |

B1 and B2 run together. B5 on Meta-World waits for C's kit; Cube does not.

## 4. Design freedom and its limits

**Fixed:** no pixel term of any kind, in the loss or in a target; no EMA; no
augmentation; no stop-gradient as a collapse fix; no pretrained component; actions
are inputs and are never predicted; noise correlated in time (`rho = 0.8`) unless a
proposal argues otherwise.

**Open, with the evidence and a derivation written first:** the estimator of the
state's entropy given the anchor and what it conditions on, and, when strong
evidence supports it, the architecture around it (how the state is read from the
encoder, the predictor, the anchor pathway). You may explore these without asking.
Each candidate's entry in `PROPOSALS.md` states the evidence that calls for it, the
quantity it estimates, its bias, what the optimum keeps and drops, which
proposition of the story it changes, and whether it satisfies Peng's per-sample
rule. Start with a two-seed screen at one epoch against the best current arm; if it
supports the candidate, continue. Directions worth considering, none of them
decided:

- estimators between the per-window and the pooled one, which credit slowly
  changing content without rewarding a code that only identifies the episode (the
  known failure of the pooled estimate);
- several anchor gaps in one objective, so that content which moves rarely is
  still credited;
- changes that give the reward a gradient toward content the state does not yet
  express, as long as they are derived from the objective and are not a
  reconstruction in disguise.

**Write a question to Lingjing's side first, and continue with other work
meanwhile:** any statistic computed across different episodes in a batch; giving an
estimator the actions between anchor and frame; anything in the fixed list.

**Reviewed by Lingjing's side:** which recipe is named the canonical one, and the
theory statements made about it. The review does not block further exploration.

## 5. Evaluation

The conventional JEPA protocol (planning success as in LeWM, plus state probes) is
the main evaluation on PushT and Cube, because it needs nothing new and is what the
JEPA community reads. The staged protocol is added on Meta-World once C's kit is
out. Whichever protocol is used, native LeWM runs under the same one.

## 6. Backlog (always available)

More seeds behind any arm that is reported; the direction-content analysis on every
saved checkpoint; the Reach3D rule and native LeWM on each new benchmark as
comparison rows; small synthetic checks of each proposition before a training run
tests it.

## 7. Goal text for the session to set

> Goal B (decoder-free). Pursue a world model trained from random weights with no
> pixel loss, no EMA, no augmentation and no stop-gradient against collapse, whose
> objective is derived from a principle, whose state carries the task (agent and
> block on PushT) and plans as well as native LeWM, and which then holds on Cube and
> Meta-World. Follow docs/lingjing/goals/README.md and B-decoder-free.md on branch
> consolidate/20261121: milestones B1 to B6, B1 and B2 in parallel. You plan,
> launch, monitor, debug and write up everything yourself on your home machine (see
> MACHINES.md), borrow GPUs others lend, and restore a machine that goes down by
> the machine manual. Keep your GPUs busy from your QUEUE.md in the priority order
> of the README; never launch a run only to occupy a GPU. Before each batch write
> the question, arms, decision rule and GPU-hours. Two seeds behind every
> statement. You may explore a principled change to the estimator or, with strong
> evidence, the architecture: write the evidence and the derivation in PROPOSALS.md
> first, and keep the fixed constraints of Section 4. When you need an
> answer from Lingjing's side, write the question and continue with another
> milestone or the backlog. The goal is not complete before B6; a milestone that
> cannot be met ends with a written account and you move on. Update STATUS.md
> hourly while anything runs, write a milestone report at each milestone, and post
> a five-line status in the chat.
