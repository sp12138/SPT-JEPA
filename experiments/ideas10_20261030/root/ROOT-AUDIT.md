# Independently designed candidates: results and admission limits

## R1 common-context decoder secants — reject as a new mechanism

For errors e_i, the proposed pairwise difference objective satisfies exactly

`sum_{i<j} ||e_i-e_j||² = n sum_i ||e_i-mean(e)||²`.

The executable check gives error0. This changes centering/weighting, not the fundamental reconstruction target. A state-independent decoder cannot fit nonconstant differences, but neither that observation nor a new name establishes a new information principle. Keep as a possible diagnostic control, not one of ten original algorithms.

## R2 conditional action-response coordinates — theory holds, implementation unresolved

The recorded design uses a conditional action density and integration by parts to make a state-only readout predict a weighted action derivative of the future observation. `v(a)=(1−a²)²` removes boundary terms on[-1,1]. The quadrature check has constant affine response1.7 for all11 tested initial states despite state-dependent behavior-policy means; clipping at a boundary produces a nonconstant response with variance0.150017. A derivative-only response still cannot identify free-space location: the affine coordinate variance is essentially zero. Observation grounding is necessary.

The real action audit samples256 training episodes/task,25 raw actions each. TwoRoom endpoint mass is62.31%/64.48% per coordinate; Cube's five coordinates have1.16%,6.31%,19.59%,9.52%,0%. Thus a naive continuous-density score on all observations is misspecified. A mixed distribution can still support an *interior-weighted* identity if its continuous component is modeled correctly and endpoint atoms are assigned zero operator value: v and v' vanish there. This does not recover derivatives or discontinuities at those atoms. The measured mean v is only.202/.195 in TwoRoom, so useful-signal/sample-efficiency claims need particular care. Conditional joint atoms, support gaps and state-dependent policy density cannot be ignored.

This is not an established causal intervention estimate from arbitrary logged multi-step actions. Later actions can respond to outcomes; a single-step sufficient conditioning state or justified sequential model is required. Estimating behavior scores adds another statistical problem and computational component. The known score-moment derivative identity is credited to [Score Function Features, Janzamin et al.](https://arxiv.org/abs/1412.6514); predictive-state ancestry is also close. The candidate's particular observation-grounded conditional response state has not passed a short learning gate or a decisive novelty comparison. Do not count it as a demonstrated two-task improvement.

## R3 dynamics-transported posterior — real mismatch found, novelty is limited

The design and exact conditional-Gaussian derivation are in `TRANSPORT-DESIGN-FIRST.md`. In a linear model with signal covariance Σ, fitting scalar-correlated posterior samples yields the population least-squares optimum

`M* = (AΣ + rho*sigma² I)(Σ + sigma² I)^−1`.

This is the correct conditional regressor for that artificial noisy joint distribution. It generally differs from A, which is the transition of the clean state used by our planner. Calling it simply an incorrect Bayes predictor would be misleading; the issue is the training-channel/deployment-map relationship.

With a90° rotation, Σ=.04I, sigma=.1 and rho=.8, the exact optimum is `[[.16,−.8],[.8,.16]]`, with squared Frobenius error.1312 relative to A. The extra squared transition error at the true A is.0328; Monte Carlo gives.0327679. The proposed observation-anchored transported posterior cancels this particular conditional-mean discrepancy exactly, including for nonlinear F. A separate unstable system expands a.1 perturbation to2.48 in four transitions, demonstrating why cancellation alone is not a stability result.

Eight frozen real-model audits, each32 fixed validation windows ×16 noise draws, all exit0. They decompose the squared mean error of the native rate into clean error, noise mismatch, and a cross term:

| Task/checkpoint | Seed | Clean MSE | Noise-mismatch energy | Total noisy mean MSE | Transport final/initial noise energy |
|---|---:|---:|---:|---:|---:|
| TwoRoom600 | 1 | .00001985 | .00038081 | .00040045 | 1.94 |
| TwoRoom600 | 2 | .00002505 | .00038257 | .00040756 | 1.95 |
| TwoRoom2000 | 1 | .00002209 | .00030513 | .00032714 | 1.61 |
| TwoRoom2000 | 2 | .00000270 | .00031819 | .00032108 | 1.65 |
| Cube600 | 1 | .00037386 | .00036953 | .00073775 | 1.88 |
| Cube600 | 2 | .00027395 | .00036912 | .00063719 | 1.88 |
| Cube11671 | 1 | .00152642 | .00032862 | .00186138 | 1.58 |
| Cube11671 | 2 | .00169963 | .00032040 | .00201642 | 1.55 |

In TwoRoom this noise term is much larger than the clean error, but that is also compatible with an almost constant encoder whose clean prediction is trivially easy. Earlier rate-off gates did not consistently rescue planning. The table therefore establishes an objective component, not that it causes the task failure. Cube's mature clean error dominates this term. Transported samples with a fixed total innovation budget did not explode on these frozen windows; they still have different marginal noise from the baseline.

The closest predecessors are material: [SRNN](https://arxiv.org/html/1605.07571v2), §3.3, already uses a residual correction to the prior mean and explicitly observes KL cancellation; [VCDT](https://proceedings.mlr.press/v97/ialongo19a.html) uses structured transition-coupled posterior families. Our specified correction `mu_next−F(mu_history,a)` ties that correction to a canonical observation encoder, unlike a free residual inference network. This is a concrete constrained implementation, but **the broad posterior family and cancellation are not new**. Hold as a useful method/control hypothesis; do not sell the algebra or the newly measured AS-JEPA mismatch as a fundamental new variational invention.

## Related sample-level world-model result

[TDWM's preliminary project report](https://td-worldmodel.github.io/TD-WorldModel/) uses within-clip temporal cosine targets and studies Scene. It reports that expert-only representation gains can hurt planning, with noisy re-roll data changing the outcome. Its reported scores are not our strict Cube benchmark. This directly reinforces the need to distinguish information retention, off-policy transition fidelity, and task readout; it also rules out presenting ordinary temporal discrimination as our new idea. The per-clip temporal target has revisit and nuisance-clock counterexamples, audited separately by the theory branch.
