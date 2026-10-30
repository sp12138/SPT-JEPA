# Final focused design pass: no additional mechanism admitted

This pass adds **zero** defensible new mechanisms to the idea count. It develops and rejects one correction-transfer family, with an exact check and a targeted primary-source audit. It does not reclassify older proposals, launch training, or change shared code.

## Updated empirical requirement

The [24-run gate](../results/relative600/RESULTS.md) makes the distinction concrete. Posterior grounding opens the query-specific encoder gradient and improves Cube's observed-code action-null error/reference ratio from 1.043/1.021 to .940/.892. However, the causal forecast ratio remains .9970/.9969 and its initial-state intervention effect is about 1e-5. The observed-code improvement did not transfer into this forecast query. This does not imply every forecast output is state-independent; it is a statement about the measured channel and budget.

Therefore another target-retention argument is insufficient. A useful next mechanism must explain how an observed distinction becomes a usable causal transition, or provide new evidence that identifies the missing transition. This motivated the following attempted design, formulated before searching its nearest literature.

## Attempted operator: learn the propagation of a state correction

For one factual action a, let z be the observed encoding, h the current rollout encoding, y the next observed encoding, and F the predictor. Define

    e = z-h,
    predicted correction = F(h+e,a)-F(h,a),
    observed correction = y-F(h,a),
    L_correct = ||predicted correction-observed correction||².

The intended interpretation was to teach the model how the state information missing from a rollout changes its next prediction, using recorded observations/actions only. It appears to target precisely the observed-to-forecast transfer gap.

But substitution gives the exact identity

    L_correct = ||F(z,a)-y||².

With consistent differentiation, it is simply teacher-forced one-step prediction, including the same gradients. The baseline rollout term cancels. A new interpretation of that loss is not a new learning rule. Selectively stopping gradients would break derivative equivalence, but would need a separately justified update rule; it cannot rescue a novelty claim merely by being forward-identical and backward-different.

The pure-Python test checks this equality on nonlinear predictors, differing true transitions, and 18 factual triples. Maximum numerical difference is 2.22e-16.

## Partial correction: a real change, but a harmful unsupported target

To avoid cancellation, consider partial correction at lambda in (0,1):

    L_lambda = ||F(h+lambda(z-h),a)
                - [(1-lambda)F(h,a)+lambda y]||².

When F(z,a)=y, this enforces linear interpolation of the transition between h and z. It is a genuine constraint beyond endpoint accuracy. If a scalar true transition f has bounded second derivative |f''|<=M on the segment, its interpolation residual obeys

    |f((1-lambda)h+lambda z)
      -(1-lambda)f(h)-lambda f(z)|
      <= (M/2) lambda(1-lambda) |z-h|².

For vector functions the corresponding bound follows from a bound on the bilinear Hessian operator along the segment. This is a smoothness assumption, not evidence that the interpolated latent is a realizable physical state.

A contact counterexample directly opposes the intended hard-task advantage. Let the true transition be f(s)=max(0,s), take h=-epsilon, z=epsilon and lambda=1/2. The perfect transition at the interpolation is f(0)=0, but the proposed target is epsilon/2. The perfect model receives loss epsilon²/4. The incorrect chord g(s)=(s+epsilon)/2 on this interval receives zero auxiliary loss at the midpoint while making physical squared error epsilon²/4 there. Its residual is first order in epsilon, whereas a smooth quadratic transition has second-order interpolation residual.

The check confirms these results for epsilon=1,.1,.01,.001. Thus the regularizer can penalize the correct contact switch precisely where a mechanism for Cube/TwoRoom should retain it. This does not prove interpolation can never help regularization; it defeats an unqualified correction-transfer rationale based only on endpoint pairs. Avoiding this problem would require actual outcomes at interpolated states or independently justified contact-aware geometry. Neither is supplied by the current logged sample.

## Primary-source audit after design

* [Self-Correcting Models for Model-Based Reinforcement Learning, Talvitie](https://arxiv.org/abs/1612.06018) already studies training from model-generated states to observed successor states, with positive and negative theoretical cases. A general claim that correcting imperfect rollout states is a new principle would conflict with this work. Our full-correction construction is even less distinct because it reduces exactly to teacher forcing.
* [Interpolation Consistency Training, Verma et al.](https://www.ijcai.org/proceedings/2019/504) explicitly matches predictions at interpolated inputs to interpolated endpoint predictions. Restricting an interpolation pair to the teacher/rollout states of one factual trajectory is a particular sampling rule, not yet a substantive new principle. Its use also inherits the need to justify interpolation in that state geometry.
* [Scheduled Sampling, Bengio et al.](https://research.google/pubs/scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks/) already changes sequence training between observed and model-generated histories. Endpoint selection alone is therefore a known control. The continuous partial-correction operator differs from discrete scheduled selection, but that difference is exactly the interpolation constraint tested above, not a demonstrated control advantage.

These are primary sources, checked after writing the operator. This audit does not prove that every possible correction update has been published. It shows that this specific proposal lacks the required substantive separation, while its most apparent extension has a counterexample aligned with the present tasks. Neither variant is admitted.

## The useful next discriminator is narrower than another algorithm label

Before proposing a forecast-repair mechanism, distinguish one-step transfer from multi-step accumulation on the frozen checkpoints. Encode the factual frames z_t, compute teacher-forced F(z_t,a_t), and apply the same already-trained readout and fixed query operator. Compare this with the existing observed-code and fully rolled-out curves. No optimal-action targets or added trajectory exposure are needed.

If even teacher-forced predictions lose the newly encoded distinction, ordinary one-step transfer/conditioning is the immediate unsolved problem; long-horizon error-correction theory does not explain it. If teacher-forced predictions preserve it and rollouts lose it, compounding/exposure mechanisms become better grounded. This is a suggested read-only diagnostic, not a new candidate or a launched experiment. It does not solve the independent Cube goal-readout problem.

Reproduction: `python3 experiments/ideas10_20261030/theory/check_correction_transfer.py`; output [exact checks](ideas10-correction-transfer-checks-20261031.json). All assertions pass. No task improvement, new algorithm, or ten-idea completion is claimed.
