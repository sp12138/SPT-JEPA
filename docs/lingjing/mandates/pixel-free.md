# Mandate B: find out whether a pixel-free objective can carry task content

Session: the second shared session. Machine: dev14. Campaign root:
`/efs/prescottsun/SPT-JEPA/runs/lingjing_pixel_free_20261122/` (new). Source clone to
start from: `/efs/prescottsun/SPT-JEPA/sources/lingjing-pixel-free-20261121` (branch
`runs/pixel-free-pusht-20261121`, commit `493e1428`). Common rules:
[README.md](README.md).

## The question

Can a model trained from scratch with no pixel loss anywhere, no EMA target and no
augmentation learn a state that carries the task (agent and block on PushT) and
plans as well as native LeWM? Lingjing's candidate is the anchor reader: the
encoder pays for what the predictor cannot predict from the past states and
actions, and is rewarded for what a small denoiser (the reader) cannot guess about
the state from an anchor frame. Specification:
`docs/lingjing/decoder-free/handoff-two-tracks.md`, Sections 2 to 4.

## Where it stands (2026-11-21)

The first PushT test ran on dev13 (`runs/lingjing_pixel_free_gate_20261121/`, eight
runs to 12,491 updates; evaluation under `gate-evaluation/`). No arm passed.

- With time-correlated noise (`rho = 0.8`) the state is alive: spread 0.56 to 2.18
  of `sigma0`, reader error 0.5 to 0.67 on training data and 0.2 to 0.5 held out.
  With independent noise it is near collapse (spread 0.30, reader error 0.02).
- The content is not the task. Noisy agent R² is 0.008 to 0.050 and noisy block R²
  0.018 to 0.059. The renderer model F8 at the same budget has 0.44 / 0.45 and
  0.80 / 0.86.
- In run Q1 seed 1 the state stayed near 0.14 of `sigma0` for 6,000 updates and
  then expanded to 2.9 by the end, with the prediction cost tripling and the
  entropy term growing without levelling off.
- The frame's index in its episode is partly readable from the state (R² about
  0.32, with an ill-conditioning warning). One reading: a counter is easy to
  predict from the past and hard to guess from a distant anchor, which is what
  this objective rewards. This is a hypothesis, not a finding.
- Peng's current Reach3D model is also pixel-free, with a different objective
  (next code = current code + a fixed-length embedding of the executed action).
  It has not been run on PushT.

## What success looks like

On PushT, two seeds: noisy agent and block probe R² at one epoch within 0.1 of
F8's values above; then planning at three epochs within 5 points of native LeWM
(84.89 / 78.67) on both sets of planning seeds. A clear account of why this family
of objectives cannot reach that is also a complete answer.

## Work, in order

1. **What does the state carry, and why does it run away?** No training. On the
   eight gate checkpoints: probes for episode frame index, time since the anchor,
   cumulative action, agent and block position and velocity, each with held-out
   episodes and a regularised fit; what the two largest information directions
   correlate with; and the same curve as Q1 seed 1 (spread, reader error,
   prediction cost, entropy term) for all eight runs. State which content explains
   the information beyond the anchor.
2. **A reference arm: Peng's Reach3D objective on PushT**, two seeds, one epoch,
   same probes. It tells us what the simplest pixel-free rule gives on a task where
   an object moves.
3. **Fixes that follow from step 1.** One is authorised in advance: if the state carries
   time, give the reader what is known for free about time (the signed number of
   frames between anchor and frame, and the frame's index when it is available at
   test time), so that it stops being rewarded. Others you propose from the
   measurements. Each fix is one arm against the unchanged Q1 at one epoch, two
   seeds.
4. **Any arm that meets the one-epoch criterion** continues to three epochs with
   the full PushT package (planning on seeds 42, 43, 44 and 1042, 1043, 1044,
   probes, latent rollouts against copying).

## Yours to decide

The reader's size, learning rate and loss weight; the weight on the prediction
cost among 0.3, 1 and 1.3; training length up to three epochs; the anchor gap
among values at least 8; which side information the reader receives, as long as it
is available without looking at the frame itself; all diagnostics. `rho = 0.8` and
the anchor-relative reader stay.

## Needs agreement first

Everything in rule 8 of the common rules, and: giving the reader the actions
between anchor and frame (it changes what the reward means), any term that looks at
pixels, any batch statistic across different episodes.

## Stop conditions

- An arm meets both criteria: write the recipe and the table, stop.
- Three written batches of fixes with no arm meeting the one-epoch criterion:
  write the closing report (what the state carried in each, what was ruled out),
  stop.
