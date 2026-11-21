# Mandate A: converge the original framework to one clean recipe

Session: the first shared session. Machine: dev13. Campaign root:
`/efs/prescottsun/SPT-JEPA/runs/lingjing_framework_pusht_20261121/` (continue there).
Common rules: [README.md](README.md).

## The question

The original framework trains an image encoder that gives a noisy state per frame,
a predictor that forecasts the next state from past states and actions, and a pixel
renderer that redraws each frame from its state and an anchor frame. Does one fixed
recipe of it give a state that plans well on PushT, TwoRoom and Cube, and also
generates? Today it is established only on PushT.

## Where it stands (2026-11-21)

- **Recipe under test (F8):** renderer trained on one-step forecasts (`lam = 1`),
  time-correlated posterior noise (`rho = 0.8`), `sigma0 = 0.1`, `beta = 0.25`,
  no initial-scale calibration.
- **PushT:** F8 at two epochs plans at 87.33 / 83.00 (two seeds). Seeds 1 and 2 are
  continuing to 37,473 updates and seed 3 is training from scratch, with chained
  evaluations. Native LeWM at three epochs: 84.89 / 78.67 (three seeds).
- **TwoRoom:** with F8's settings the state is unused at 2,000 updates (spread
  0.0015 of `sigma0`, no change in render error when the state is swapped), from
  the same initial weights as the earlier runs. Native LeWM: 80 / 72.
- **Cube:** two seeds running to 12,000 updates. Earlier temporal-forecast: 19 / 15
  against native 26 / 27.
- **Anchor gap:** measured, no dependence on PushT; no anchor change follows.

## What success looks like

One recipe, fixed across the three tasks except for each task's data settings, that
1. is within 5 points of native LeWM or better on PushT at three epochs, on both
   sets of planning seeds, over three training seeds;
2. on TwoRoom has a state the renderer uses at 2,000 updates and plans within 10
   points of native LeWM at 6,000;
3. on Cube plans within 5 points of native LeWM under the standard planner at
   12,000 updates;
4. has generation numbers (256-clip FVD, LPIPS, SSIM, pose rollout errors against
   copying) for its PushT checkpoints.
A documented account of which of these cannot be met, and why, is also a complete
answer.

## Work, in order

1. **Close the running batch.** When the chained evaluations finish, write the
   milestone report: the PushT comparison with a paired interval and the things
   that cannot be matched (frames per window 8 against 4, trained parameters,
   GPU-hours), the TwoRoom and Cube numbers next to the earlier ones, and the
   generation numbers.
2. **TwoRoom: why is the state never used?** This is the main open problem. Known:
   the state starts with a spread far below `sigma0` and stays there in every arm
   tried, including `beta = 0`; setting the initial spread to `sigma0` did not
   rescue it; a frozen random encoder gives a usable state. Establish the cause
   with measurements before trying fixes: the gradient that reaches the state head
   from the renderer and from the rate at initialisation and at 500 and 2,000
   updates; how much the renderer's output changes with the state against with the
   anchor; and the same quantities on PushT, where training works. Then test fixes
   that follow from what you measured.
3. **Cube.** If F8 trails native LeWM by more than 5 points, measure what the state
   tracks (cube against effector probes, clean and noisy) and what the planning
   cost follows, before changing anything.
4. **One recipe.** Carry whatever fixes TwoRoom or Cube back to PushT and confirm
   that requirement 1 still holds.

## Yours to decide

Initialisation scale and learning-rate warm-up of the state head; `sigma0` (it has
never been varied without the failed calibration); `beta` among the values already
used (1/16, 1/4); the anchor rule among those already implemented; window and batch
settings; training length up to the budgets above; all diagnostics.

## Needs agreement first

Everything in rule 8 of the common rules. In particular, do not make a state
"used" by cutting or reweighting gradients between the renderer and the encoder,
and do not add an auxiliary loss on the state.

## Stop conditions

- All four requirements met: write the recipe and the final table, stop.
- Three written batches on TwoRoom without a used state: write what was measured
  and what was ruled out, stop work on TwoRoom, and finish Cube and PushT.
