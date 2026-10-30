# Independent Carrier review before the600 gate

Reviewed frozen source `dd2aeaa40e8a8050439ef036b402f289d006a257`, four completed calibrations and all12 prepared Carrier configurations. No model, optimizer, configuration or frozen snapshot was changed. The earlier eight CPU contracts were not rerun. One specifically authorized read-only precision measurement used dev2 GPU4, PID5705, for48.4 seconds, then exited successfully; GPU4 was confirmed at0MiB/0% afterward.

## Provenance and integration

All four calibrations contain128 distinct training episodes and zero validation overlap. Their source/helper/config/dataset-card/training-ID hashes match the authoritative frozen files. All three modes within each task/seed reuse exactly the same mean/scale arrays and calibration hash. No encoder warm start or legacy state calibration is enabled. All four standard-deviation floors have zero active coordinates.

| Task/seed | Minimum raw coordinate std | Mean raw coordinate std | Maximum |
|---|---:|---:|---:|
| TwoRoom1 | .114454 | .194195 | .329676 |
| TwoRoom2 | .094680 | .179205 | .274230 |
| Cube1 | .449731 | .791104 | 1.637074 |
| Cube2 | .473149 | .784068 | 1.530339 |

Same-seed encoder hashes match across tasks, as expected for the same fresh ViT initialization. The projection hash is identical everywhere. Actual exported seed1 preflight initial tensors match the calibration encoder hash, projection hash and normalization arrays for both tasks. Seed2 calibration/config provenance was checked; its later initial-model export still needs the campaign's normal comparison.

The factory dispatches Carrier in training, full-state health and information audits. Trainable parameters alone enter the existing optimizer. The health expectations correctly allow frozen encoder/pool and identity chart, while requiring active predictor/renderer/learned chart. The information assay explicitly differentiates frozen state **values**, without pretending those are encoder-parameter gradients. The benchmark planner loads the serialized Carrier target dynamically and keeps its original cost/controller. Legacy PushT-only planner/evaluation entry points do not support this target, but they are outside the current TwoRoom/Cube plan.

One minor reporting issue: the inherited `encoder_frozen` scalar measures the old scheduled-freeze window, not permanent `requires_grad=False`. It can be0 for Carrier. Use declared gradient expectations and unchanged frozen tensors as the authoritative checks.

Calibration is based on one frame per selected training episode/window, not a guarantee of exact unit variance on every training or validation distribution. This sampling choice is shared across all three arms and recorded; it neither supplies a semantic label nor guarantees representative coverage of rare contacts.

## Measured numerical train/evaluation distinction

Training enables TF32 (`float32_matmul_precision="high"`); calibration defaults to highest precision, and strict benchmark evaluation explicitly selects highest. Disabling autocast inside the frozen carrier does not disable TF32. The authorized assay compared these settings on the same first16 existing validation-manifest windows per task, using both initial and step2 bounded checkpoints from the frozen preflight. It also separated chart arithmetic from the encoder/projection by passing an identical normalized carrier into the chart.

| Task | Encoded-state RMS difference | RMS / observed coordinate std | RMS / sigma=.1 | Maximum absolute difference / sigma |
|---|---:|---:|---:|---:|
| TwoRoom | .0107501 | 1.1760% | .10750 | .59177 |
| Cube | .00312834 | .29467% | .03128 | .14503 |

Initial and step2 results agree to the reported precision. Raw-carrier RMS differences are .00183960 and .00224628, respectively. With identical chart inputs, step2 RMS differences are only8.5e-10 and2.0e-9; the measured discrepancy is almost entirely frozen encoder/projection arithmetic at this stage. All model tensors were verified unchanged.

This shared numerical difference does not show one arm has an advantage, nor establish a planning failure. It does mean that the carrier is not bitwise the same function across training and evaluation modes. The TwoRoom worst-coordinate shift is appreciable relative to posterior sigma and must not be described as exact alignment. After reviewing the measurement, the campaign owner retained the original high-training/highest-evaluation convention unchanged across all six arms. The three Carrier arms share this convention; no post-hoc precision variant is added to this gate. [Full measurement and provenance](precision-review.json); the EFS original is `/efs/prescottsun/SPT-JEPA/reports/carrier-precision-review-20261031/precision-review.json`.

## Endpoint interpretation criteria

1. **Verify the intervention actually occurred.** Frozen encoder, projection and normalization must remain unchanged. Identity chart parameters must remain unchanged. Learned charts must receive finite updates. Report raw/effective matrix norms, normalization activation, residual Lipschitz upper bound, and actual chart displacement/pair-distance change relative to calibrated u. Total parameter equality does not erase the identity arm's8,320 fewer trainable parameters.

2. **Separate factor normalization from preservation of information.** Raw W1 norm1.119 makes its normalization active from initialization. However the relevant global residual bound is `.5*||W1||2*||W2||2`. If W2 stays small, even the unconstrained chart can remain safely co-Lipschitz. Active factor normalization alone does not demonstrate an avoided destructive map. An upper bound exceeding.5 is also not proof that the true residual Lipschitz constant exceeds.5; the product bound can be loose. Actual loss of observed distinctions would be stronger evidence.

3. **Do not count standardized variance as learning.** Near-unit initial carrier variance is largely built into calibration. Clean physical probes, the same probes with the actual sigma noise, and generator/forecast state interventions answer different questions. The chart guarantees preservation relative to u in exact arithmetic, not preservation of every original-image factor, invariant noisy mutual information, or invariant linear-probe accuracy. A nonlinear invertible chart may change a linear probe without creating or deleting clean information.

4. **Require a learned transition to beat the appropriate copy baseline.** Compare teacher-forced prefix prediction against shifted observed carry, since both see the latest real state. Compare causal rollout against initial carry, since both begin from only frame0. Use per-horizon values and raw denominator energies alongside each arm's normalized errors. A smaller raw latent MSE across different charts may only reflect geometry. State/action intervention sensitivity is useful only together with better factual prediction than carry; a model that merely copies the initial state is state-sensitive too. Wrong-action comparisons against the original outcome are sensitivity checks, not ground-truth counterfactual dynamics scores.

5. **Keep planning separate.** Native P0 uses the same50 fixed cases and unchanged CEM/goal cost. Compare both seeds, report paired cases and uncertainty, and do not drop a losing task/seed. A P0 change without stronger outcome prediction may be a changed goal geometry. Conversely, a better learned transition without P0 gain leaves the Cube readout/goal-semantics issue unresolved.600 steps remain an early gate.

| Observed pattern | Supported interpretation |
|---|---|
| Identity improves over old TF; learned charts match it | Fixed measurement/calibration/frozen training is the useful change; no learned-chart gain established. |
| Both learned charts beat identity on dynamics and planning | Coordinate adaptation is useful under this budget; the bound's necessity remains unproven. |
| Bounded beats free, with free losing useful distinctions and worse dynamics/planning | Evidence supports the proposed preservation mechanism, subject to seeds/task support and actual chart activity. |
| Bounded beats free while both remain near identity and safely co-Lipschitz | An optimization/parameterization difference is plausible; do not attribute it to prevention of information deletion. |
| Probes improve, forecasts remain at carry | Observation information was retained/exposed but not transferred into causal dynamics. |
| Dynamics improve, P0 does not | World-model/readout separation remains; no successful end-to-end claim. |
| Only one seed/task wins or all arms remain near copy baselines | No robust promotion or automatic longer continuation is justified. |

The review found no source/config/statistical-leakage blocker. Numerical precision remains an explicitly measured limitation under the retained protocol. No mature superiority, academic novelty, or completed ten-idea result follows from this preflight review.

## Independent600-step endpoint review

Read-only snapshot at `2026-10-30T18:24:40Z` (October31 locally). All12 Carrier health, information, transition and planning reports were available. Every seed is retained. Sources and file hashes are in [the endpoint snapshot](endpoint-review-snapshot.json). No training, simulator evaluation or new GPU work was performed for this review.

**The strongest supported effect is the fixed carrier making useful TwoRoom state accessible. The distinctive information-preservation benefit of the bounded chart is not established.** Cube provides no cross-task promotion signal.

| Task / arm | Strict50-case success, seeds1/2 |
|---|---:|
| TwoRoom historical TF600 reference | 6 / 8% |
| TwoRoom identity | 44 / 30% |
| TwoRoom unconstrained | 34 / 30% |
| TwoRoom bounded | 46 / 42% |
| Cube historical TF600 reference | 14 / 10% |
| Cube identity | 4 / 6% |
| Cube unconstrained | 4 / 8% |
| Cube bounded | 10 / 6% |

The historical TF reference does not include the frozen carrier/calibration/chart and is not an equal trainable-capacity comparison. The three Carrier arms are the direct mechanism controls. Bounded versus identity wins/loses4/3 and8/2 matched TwoRoom cases; against unconstrained it wins/loses7/1 and8/2. Cube bounded versus identity is3/0 and1/1; versus unconstrained it is3/0 and1/2. These small case counts support a bounded-chart *performance hypothesis* in TwoRoom, not a broad stable superiority claim. Pooling both training seeds into100 independent examples would overstate replication: the same50 evaluation cases are reused.

### The free maps are already safely invertible

At600 the unconstrained residual Lipschitz upper bounds are .11558/.11209 on TwoRoom and .15022/.12933 on Cube. Thus every free endpoint has a global lower distance factor at least.8498. The bounded counterparts have residual bounds .09717/.09856 and .11441/.09942. W1 normalization is active, but the unconstrained function is far from the theoretical contraction limit and already satisfies a stronger preservation bound than the configured worst-case lower factor.5.

CPU analysis of existing observed states finds chart displacement RMS only.031–.041 in normalized carrier units. Median pair-distance ratios relative to identity are .977–.979 across distinct initial frames; the smallest observed free ratio is.9259. Adjacent-frame distance ratios also remain close to one. Therefore there is no endpoint evidence that unconstrained adaptation erased distinctions and the bound saved them. This statement concerns the audited endpoints, not an unmeasured claim about every intervening optimizer step.

### Put prediction errors into the same coordinates

Because all free and bounded endpoints certify a contractive residual, their saved predictions can be inverted into the identity carrier u without additional model rollouts. The CPU-only [analysis script](endpoint_alignment_review.py) loads only saved chart weights and `state-audit.npz`, performs32 fixed-point inverse iterations, and compares against the same stored identity targets. Recovered observed encodings agree within4.8e-7. [Common-coordinate results and provenance](endpoint-common-coordinates.json).

| Task / arm | Common-u rollout MSE / initial carry, seeds1/2 | Common-u teacher-forced MSE / shifted observed carry, seeds1/2 |
|---|---:|---:|
| TwoRoom identity | .58106 / .58539 | .81689 / .84411 |
| TwoRoom unconstrained | .58562 / .58247 | .82382 / .84314 |
| TwoRoom bounded | .58023 / .58213 | .81830 / .84013 |
| Cube identity | .52781 / .51760 | .73463 / .73544 |
| Cube unconstrained | .52520 / .51552 | .73118 / .73326 |
| Cube bounded | .52559 / .51524 | .73197 / .73304 |

All three arms genuinely predict better than their corresponding copy baselines. Positive learned-chart improvements **over identity** are below.7% relative, however, and bounded versus free has no consistent prediction advantage across tasks/seeds. Raw normalized errors in each learned chart initially made Cube adaptation look slightly worse; mapping into common u reverses that interpretation to tiny improvements. This is why a changed latent metric cannot be silently treated as a common outcome-quality measure. Even common-u error is a visual-carrier proxy, not privileged physical transition error.

In native-coordinate factual prediction, replacing the initial state raises TwoRoom rollout error by72–94%, and Cube by96–127%. Wrong action order raises error by.4–2.2% on TwoRoom and14.8–20.5% on Cube; other-episode actions give larger effects. The combination of beating carry and using initial state/actions is materially stronger than a nonzero gradient. It still does not establish correct counterfactual predictions for unobserved alternative action sequences.

### Preserved observation information is largely the original measurement

TwoRoom identity clean agent-x/y probe R² is .944/.943 and .928/.892; adding sigma=.1 noise retains .911/.901 and .899/.897. Learned-chart probe changes are around1e-4–1e-3. These results confirm a useful noisy observation channel; near-unit feature variance by itself would only confirm the fixed calibration.

Cube identity clean block-x/y/z R² is .584/.878/.345 and .588/.875/.385; the noisy probes remain similar. The block-height and x distinctions are substantially weaker than the mature learned Cube representations audited earlier. Block-x **displacement** linear R² is negative (−.174/−.244), unchanged by either chart. This does not prove nonlinear information is absent, but it limits the claim that the preserved frozen carrier already exposes all contact-relevant factors. A small invertible chart cannot manufacture missing information.

The renderer uses this channel much less than the latent predictor: at t=.95, wrong-initial-state changes global forecast denoising MSE by at most.63% in TwoRoom and.81% in Cube, with similar behavior at t=1. Wrong actions alter renderer error by roughly0% and.4–.6%, respectively. The clear latent transition/state benefit therefore does not imply comparably strong generative use or improved image outcomes. The fixed-state/predictive-prior path can be useful while this renderer remains mostly insensitive; causal attribution of that learning to individual loss terms has not been tested here.

The measured high-training/highest-evaluation carrier shift remains a shared limitation: RMS is1.176%/.295% of observed state standard deviation for TwoRoom/Cube in the initial/step2 precision audit. There is no new endpoint precision measurement, and no reason to call cross-mode carrier evaluation bitwise identical.

### Decision implication

The TwoRoom carrier family warrants a **targeted followup question**, because even identity improves substantially over historical TF600 and has meaningful factual dynamics. Any later controlled continuation must retain identity; testing the bound specifically also requires the unconstrained arm. This screen does not justify a long cross-task promotion of bounded F, removal of the identity control, or blanket continuation of the12 Carrier runs. Cube is below historical TF600 in both bounded seeds, with no substantial common-coordinate transition gain or newly exposed block information. The parent campaign's decision to hold all24 runs is consistent with this evidence.

The current conclusion is useful but narrower than the proposed mechanism: preserving an already informative spatial measurement removes a major TwoRoom obstacle; the particular invertibility bound has not demonstrated that it prevents a failure actually encountered by the matched free map. Small TwoRoom planning gains may instead reflect coordinate/readout or optimization differences. No new algorithm count, mature LeWM superiority, or validated general bound benefit is claimed.
