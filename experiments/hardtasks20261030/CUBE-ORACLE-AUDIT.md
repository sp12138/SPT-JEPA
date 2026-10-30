# Cube: locating the loss of useful action ordering

This is a frozen-model diagnostic, not a new training method or benchmark claim.
It follows the completed first-epoch result: temporal-forecast 24/26%, lower-beta
22/26%, original 20/24%, and released LeWM 30% on the same 50 strict cases.
Their training budgets and data exposure are not matched to the released model.

The experiment fixes a library of raw action sequences and computes three costs:

1. **Physical outcome:** execute each sequence from the identical simulator
   state. Record cube-position error, 24-symmetry orientation error, and the
   strict three-step sustained-success criterion at every raw step.
2. **Encoded actual outcome:** encode the observed endpoint and measure the
   exact latent squared distance to the encoded goal.
3. **Predicted outcome:** roll the frozen world model over the same actions and
   measure the same goal distance on its predicted endpoint.

All three selectors see the same candidates. The gap between the best physical
candidate and the best actual-embedding candidate diagnoses the latent goal
metric. The further change caused by choosing the predicted-embedding candidate
diagnoses rollout ranking. The latter contribution is signed: model errors can
occasionally cancel a metric error, so a negative value is not an audit bug.
If the library contains no successful sequence, candidate coverage remains an
unresolved limitation, and neither selector can establish planning sufficiency.

The physical scalar is max(position_error/3cm, symmetry_angle/15deg), whose
threshold is the strict instantaneous pose criterion. Position, angle and
sustained success are always reported separately. An endpoint cost cannot stand
in for the temporal hold criterion. This is a 25-step first-plan experiment, not
the original 50-step closed-loop success rate.

The default 32 candidates contain zero actions, independent Gaussians in the
training action normalization, recorded held-out actions, two recorded-action
perturbation scales, and a duplicate recorded candidate. The duplicate must
reproduce all qpos values within 1e-7 after restoring the full MuJoCo integration
state. The recorded candidate's replay discrepancy and initial rendered-pixel
discrepancy are saved; interpretation must stop if these expose a reset or action
alignment defect. Recorded-action families are privileged diagnostic controls,
not proposed deployable behavior. Results are stratified into nonoracle and
recorded-action families. A small random library does not measure CEM coverage.

The optional `propose` stage runs the existing native CEM300x30 solver once per
case and model. Its 25 raw actions can be added to the common library through
`collect --proposals DIR` (repeatable). Collect all models' proposals before
using this option so every scored model sees the identical candidate set.
These are first-plan proposals; full MPC may recover or fail differently.

## Execution

Use the campaign environment, frozen source, pinned SWM/CLEAR imports and EGL
settings from the existing first-epoch plan. All commands below are arguments
to `python -B -m as_jepa_tools.cube_oracle_audit`; outputs must be on EFS.

Shared arguments:

```
--manifest /efs/prescottsun/SPT-JEPA/data/hardtasks-v1/cube-metadata/manifests/validation-strict.json
--data /efs/prescottsun/SPT-JEPA/data/hardtasks-v1/cube-metadata
```

First smoke:

```
collect --limit 2 --candidates 8 --output /efs/prescottsun/SPT-JEPA/reports/NEW-AUDIT/smoke-pool
score --limit 2 --pool /efs/prescottsun/SPT-JEPA/reports/NEW-AUDIT/smoke-pool --checkpoint CHECKPOINT --output /efs/prescottsun/SPT-JEPA/reports/NEW-AUDIT/smoke-model
```

After checking reset/replay identity and finite outputs, split collection across
eight slots with `--shards 8 --shard 0..7 --candidates 32`, sharing a fresh pool
directory. Score each checkpoint in its own output directory; scoring can also
be sharded. No existing output case is overwritten. Checkpoints:

```
/efs/prescottsun/SPT-JEPA/runs/hardtasks_cube_epoch1_20261030/cube-{original,lower-beta,temporal-forecast}-seed{1,2}-epoch1/model-step0011671
/efs/prescottsun/SPT-JEPA/weights/hardtasks/cube
```

Released LeWM uses its existing full-dataset action normalization. Our models use
the train-only card statistics, matching previous evaluations. The physical
library is the same raw actions despite different model normalizations.

## Interpretation

The mechanistic hypothesis is that **predictive information and decision
ordering are different requirements**. A representation can preserve robot pose,
contacts and appearance well enough for image prediction yet assign Euclidean
goal distance mostly to changes irrelevant to the cube success condition.
Likewise a predictor can have small average latent error but distort action
differences enough to choose the wrong candidate. These are hypotheses, not
findings established by the current success rates.

For true encoded cost C(a) and predicted cost C_hat(a), a uniform cost error
bound epsilon implies C(argmin C_hat) - min C <= 2 epsilon. The relevant quantity
is therefore prediction error relative to the gaps between candidate costs,
not absolute reconstruction loss. The audit reports rank correlations, selected
physical outcomes, centered action-effect error and horizon-wise latent error.
This decomposition is a diagnostic framework; it does not establish novelty.
