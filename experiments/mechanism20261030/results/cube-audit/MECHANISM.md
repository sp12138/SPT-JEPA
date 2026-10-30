# Cube mechanism review: accurate prediction of a misaligned goal geometry

This report interprets the frozen 50-case, 39-candidate audit. It does not replace the existing 50-step closed-loop benchmark (temporal-forecast 24/26%, released LeWM 30%). These are 25-step first-plan diagnostics. The released model differs in training budget, size and data exposure.

## Integrity

All 50 cases contain the same 39 raw-action candidates for all seven models: 32 base candidates plus one native CEM300x30 proposal from each model. All seven score outputs and eight collection shards use the same frozen audit source SHA, a34078ae8783998e619afbd4660c13e9be05eeda8f9adb7a6e21c087ec2e570f. Duplicate recorded-action rollouts reproduce every qpos value exactly. Recorded-action replay reaches the recorded goal within at most 3.07 mm. Initial rendered-pixel MSE is at most 16.58 on the uint8 scale; the image is therefore not bit-identical to its dataset rendering, although physical replay is close.

Every recorded sequence reaches the endpoint position/orientation tolerance, but only 56% already satisfy hold3 within the 25-step horizon. Across the entire library, 82% of cases have a sustained-success candidate. Therefore the 56% endpoint-physical-selector result is not a success upper bound: endpoint minimization and sustained success are different objectives. The nonoracle library has 22 candidates and 40% sustained-success coverage.

## The clearest signal is goal-metric factor weighting

Nonoracle cases, mean per-case rank statistics:

| Quantity | Temporal seed1 | Temporal seed2 | LeWM |
|---|---:|---:|---:|
| Actual latent goal cost vs effector distance | .952 | .957 | .452 |
| Effector partial rank, controlling cube position and angle | .928 | .935 | .408 |
| Cube-position partial rank, controlling effector and angle | .034 | .022 | .241 |
| Predicted cost vs actual latent cost | .948 | .951 | .556 |
| Centered action-effect MSE / actual candidate latent variance | .079 | .079 | .863 |
| Actual-embedding selector sustained success | 10% | 20% | 24% |
| Predicted-embedding selector sustained success | 20% | 12% | 22% |

The original AS-JEPA effector partial correlations are .797/.792; temporal forecasting increases them to .928/.935. Against LeWM, the temporal-minus-reference effector partial-rank differences are .520 [.440,.606] and .527 [.446,.613]. The corresponding cube-position differences are -.208 [-.341,-.077] and -.219 [-.350,-.097]. These intervals resample cases and are conditional on the frozen weights; they are not training-seed confidence intervals.

This is evidence that temporal forecasting predicts its own chosen geometry very consistently, while that geometry tracks the robot effector much more strongly than the object goal. Low normalized latent error is not evidence of better physical dynamics when different encoders represent different factors. Partial correlations alone do not establish causal irrelevance: robot pose is legitimately needed to predict contacts and motion. A separate frozen-metric projection intervention is being prepared to test whether these directions causally affect candidate selection, with a same-rank random deletion and post-projection block-information probes.

## Prediction consistency is still insufficient at the selected optimum

On all 39 candidates, the true-encoded selector gives 56% sustained success for every model and almost always chooses the recorded endpoint. The predicted selector gives temporal 20/22%, original 14/18%, lower-beta 14/24%, and LeWM 40%. Temporal predicted/actual cost rank remains .940/.946 versus LeWM .786. Thus an excellent global rank does not guarantee correct ordering near the minimum.

Temporal's own CEM proposal has lower predicted goal cost than the recorded action sequence in 44/50 and 37/50 cases; the model selects its own CEM proposal from the full common library in 44/50 and 29/50 cases. This is consistent with optimization selecting favorable prediction errors, but does not by itself prove an uncertainty mechanism. There is no justification for claiming that a scalar average rollout error explains success.

## Do not turn the diagnosis into an exaggerated failure claim

Actual execution of each model's own first CEM plan:

| Model | Mean endpoint cube distance | Endpoint position pass | Endpoint pose pass | Any-step pose pass | Hold3 success |
|---|---:|---:|---:|---:|---:|
| Original seed1 | 14.50 cm | 12% | 2% | 10% | 8% |
| Original seed2 | 15.02 cm | 16% | 12% | 14% | 10% |
| Temporal seed1 | 9.23 cm | 36% | 24% | 24% | 20% |
| Temporal seed2 | 9.40 cm | 28% | 18% | 22% | 10% |
| LeWM | 11.39 cm | 34% | 20% | 24% | 22% |

Temporal makes real progress over the internal original and achieves smaller mean endpoint position error than LeWM here. Its second seed loses more cases at the hold requirement. The metric diagnosis explains a limitation and possible mechanism; it does not establish that temporal forecasting is uniformly worse or that longer training cannot help.

## Abstraction and next decisive test

A world representation must preserve information needed for future dynamics. A goal metric must decide which differences count as failure. Using the same latent with unweighted Euclidean distance identifies these two requirements without proving they agree. A rate-distortion model can favor predictable, visually prominent motor state; making forecast reconstruction better may strengthen that preference. This mechanism is a hypothesis linking the observed effector dominance to the training objective, not yet a demonstrated causal account of the objective.

For a fixed encoded cost C and predicted cost C_hat, a uniform error bound epsilon implies encoded selection regret at most 2 epsilon. The relevant errors are therefore those compared with the cost gaps between competitive candidates, not only average reconstruction error. Separately, even exact prediction cannot repair an encoded cost that values the wrong goal factors.

The next diagnostic fits fixed train-only linear xyz readouts, removes the raw-latent effector readout subspace, compares a same-rank random removal, and reports retained block information. It uses privileged coordinates only as diagnostic instruments and does not train the encoder or provide expert optimal-action labels. Improvement after effector deletion would establish a contribution of that metric subspace in this candidate set; it would not by itself deliver a self-contained deployable algorithm.

## Artifacts

- results.json / RESULTS.md: all groups, paired case intervals and rank/selection decomposition.
- integrity-and-selection.json / selection-cases.json: source/candidate audit and individual selections.
- pose-hold-decomposition.json: position, orientation and temporal-hold outcomes.
- goal-metric-factors.png/svg: physical factors associated with the goal metric.
- first-plan-pose-hold.png/svg: execution outcomes of each native CEM first plan.
- oracle-selection.png/svg and oracle-ranks.png/svg: common-pool decomposition.
