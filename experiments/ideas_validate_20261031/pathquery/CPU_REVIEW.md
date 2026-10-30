# CPU implementation review

Eight contracts passed on dev13 with `CUDA_VISIBLE_DEVICES=''`: 5.56 seconds
inside unittest, 35.29 seconds including environment import. No GPU experiment
was launched by this implementation task. Native two-step GPU preflight and
600-update functional evaluation remain the next checks, owned by the parent.

Verified:

- Default `none` matches current Robust TF initialization, state dict, train-mode
  loss, every metric, every gradient and outer RNG exactly; no head is created.
- Three active modes preserve all baseline tensors and initialization RNG,
  share identical 70,533-parameter head initialization, and reload strictly.
- All 3,280 words of length zero through seven match an independent monitor
  oracle and the segment-composition law. Target pooling, nearest-prototype
  labels, full/half maps, exact ties and sample-locality match independent checks.
- Causal rollout makes exactly seven calls for eight frames. Changing observed
  future codes leaves logits unchanged; changing future query prototypes leaves
  the forecast unchanged. No query/target gradient enters future observations.
- All three modes have bit-identical forward logits. Their selected objectives
  equal independent cross entropy divided by log(K); inactive output rows have
  zero selected-loss gradient. Evaluation split-batch losses agree.
- Active loss encodes once, preserves native metrics and RNG, keeps finite full
  gradients, and clears its transient capture even on exceptions. Two ordinary
  SGD updates open native action-encoder gradients. Public pixel-only encode is
  unchanged; unsupported configurations fail explicitly.

The first run had seven passing contracts and one incorrect test expectation:
it manually opened only the native predictor output projection and expected
action-encoder gradients immediately. Inspecting native source showed a second
zero-initialized AdaLN modulation barrier. The final test uses two ordinary SGD
updates and passes without changing any model initialization. The first report
is preserved as `cpu-tests-first-incorrect-one-gate-assumption.{json,log}`.
Post-two-step real-preflight `action_encoder` expectations remain **active**.

Source SHA256:

- `as_jepa_lewm/pathquery.py`: `f218427d4c7f58d473580bb12206f945240350f7ed0803e104956b5b43b58638`
- `tests/as_jepa/test_pathquery.py`: `b9b4fc85d24f845f5545c1bb50afa9689a3d05db64875398bea4d7f422931379`

This verifies the implementation contracts, not task performance, learned
composition, or novelty. Shared parameter counts do not erase the disclosed
active-output-row/context differences among classification targets.
