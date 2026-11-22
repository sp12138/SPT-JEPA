# Mandate C: build the evaluation framework and produce its first scaling result

Session: the third shared session. Machine: dev3. Campaign root:
`/efs/prescottsun/SPT-JEPA/runs/lingjing_evaluation_20261122/` (new). Common rules:
[README.md](README.md). Design: `docs/lingjing/as-jepa-evaluation-handoff.md`
(Sections 3 to 6); existing code: `docs/results-index.md`, Section 4. Both are on
branch `consolidate/20261121` of `github.com/sp12138/SPT-JEPA`, which dev machines
can fetch.

## The question

How do latent world models scale, and does the answer depend on how they are
evaluated? The protocol has two stages. Stage 1 is each model's own pretraining,
without predicting actions. Stage 2 freezes the model and adds fixed probes: a
readout or policy, probes for physical state on encoded and on predicted states,
and planning. The same probes are applied to models of different sizes.

One finding of this kind is already in the repository: on the same checkpoints,
between 1 and 8 hours of training, the planner's return fell (78.85 to 73.37) while
the linear readout's rose (81.37 to 85.70).

## Rules for the evaluation stage

`AGENTS.md` governs how a model is pretrained and is not changed by this mandate.
Decided by Lingjing on 2026-11-22, under the authority Peng delegated for this
project: the evaluation stage has its own rule set.

- Allowed, for evaluation only: a policy head trained on action sequences toward a
  later frame of the same episode; fine-tuning a copy of the encoder with action
  labels; probes trained on simulator state.
- Nothing trained in the evaluation stage feeds back into any model's pretraining.
- These results are reported in their own tables, never mixed with the
  from-scratch training results.
- The robot's joint readings (proprioception) may be used only for the labelled
  reproduction row of the public RoboJEPA reference under its own settings. Every
  comparison row is RGB only, with the commands actually issued as the action
  history.

## Work, in order

1. **Confirm the inventory (allowed now, read-only).** Check Section 4 of the
   results index against the code: the Reach simulator loop, per-case replay,
   frozen-feature caching, the RoboJEPA and V-JEPA 2.1 adapters, the CEM planner,
   the linear readout. Report what the index has wrong and what is on the EFS
   (public weights, case sets, logs).
2. **Reference rows from public models (allowed now; AGENTS.md permits RoboJEPA
   reproduction with its public weights, reported separately).** On the 64 Reach3D
   development cases, RGB only, with the 2 mm rule: each public RoboJEPA predictor
   size, evaluated twice, with RoboJEPA's own CEM planner and with the existing
   linear readout of the current executed action on frozen features. Start with
   the three smallest sizes; measure seconds per decision and GPU-hours and write
   the cost of the larger sizes before running them. Add the existing from-scratch
   mainline row (39 of 64) for context, labelled as a different training rule.
   This is the first scaling figure: success against predictor size, under two
   readouts.
3. **The same two readouts across training time and data (allowed now).** Collect
   from existing results, without new training, the curves the repository already
   has (training hours; 116 / 464 / 1,856 added trajectories; two seeds) into one
   table in the protocol's format, with what is and is not matched.
4. **State probes.** Log simulator state for the probe
   training logs, fit the probes on encoded states and on states the predictor
   rolls forward, and report error in physical units against horizon.
5. **Policy head and policy plus planning.** As in
   Section 6, Step 1 of the design, with its success checks.

## Yours to decide

All engineering, the order within steps 1 to 3, case batching, caching, which
public sizes fit the machine.

## Needs agreement first

Everything in rule 8 of the common rules; any change to the 64 development cases
or the 2 mm rule; opening the 32 sealed confirmation cases; changing a probe's
definition after results from it have been seen.

## Stop conditions

- A public model cannot be reproduced within 5 points of its recorded number under
  its own settings: report the gap and stop that row; do not tune it.
