# Revised interpretation after the frozen information audit

2026-10-30. This note supersedes the untested background-dominance explanation, not the immutable source of the running experiments. Evidence comes from `runs/mechanism_information_20261030`; the 16 fresh/resumed rate-on/off gates were still running when this interpretation was written. No new GPU task is launched here.

## What changed

The TwoRoom moving region occupies about 9.37% of pixels but contributes approximately 84–91% of the remaining high-corruption flow error. The current residual is therefore not background dominated. A small moving region can still have small absolute loss value, but that weaker observation does not establish why the model fails to learn it.

Temporal-forecast's clean-state position probes explain only about 0.11–0.29 of agent-coordinate variance. With sigma=.1 noise, their R2 is around -0.05. Removing noise and magnifying existing state differences 100 times barely changes the frozen renderer's error. This distinguishes three problems: the clean code contains weak spatial information, that signal does not survive the training channel, and the frozen renderer has not learned to use the amplified existing directions.

Priced-rate/flow state-gradient norm ratios are about .20/.31. Their cosine is close to zero. Moreover, flow's centered-state expansion derivative is positive for seed1 and close to zero for seed2; combining rate does not show a clear force expanding state differences. Thus the observed failure cannot currently be described as rate gradients overwhelming helpful flow gradients. A nonzero gradient norm is not necessarily a gradient that separates physical states.

Cube has a different failure. Its clean code exposes block xyz with R2 approximately .909/.951/.546 for seed1 and .909/.950/.720 for seed2, and end-effector coordinates above .979. Block displacement x/y probes are weaker than end-effector displacement probes. This motivates evaluating goal-cost and action-ranking alignment, rather than applying a global noncollapse cure to both environments.

## A narrower hypothesis: an encoder/readout channel that does not start learning

The information path is a composition: image -> encoder -> noisy code -> learned renderer. Learning a position code helps only if the renderer uses its direction; learning to use that direction helps only if the encoder supplies a reliable signal. Noise and alternative renderer inputs can weaken both incentives simultaneously. This is a training-dynamics hypothesis, distinct from claiming the final rate-distortion optimum excludes agent position.

A local scalar surrogate makes the distinction precise. Let psi(X) be a centered potential encoder feature with variance lambda. Let R be the renderer's residual after its existing side-information path, eta=E[R*psi(X)], let e control encoder feature amplitude, and let w be the renderer's readout. The noisy channel is `Z=e*psi(X)+sigma*n`. A simplified objective is

\[
 J(e,w)=C-2\eta ew+\lambda e^2w^2+\sigma^2w^2
         +\tfrac12\beta\lambda e^2.
\]

At e=w=0, both first derivatives vanish. The Hessian is

\[
 H=\begin{pmatrix}\beta\lambda&-2\eta\\-2\eta&2\sigma^2\end{pmatrix}.
\]

The collapsed solution is locally stable in this toy model if `beta*lambda*sigma^2 >= 2*eta^2`. More significantly, with beta=0 it is a saddle, but its negative eigenvalue is

\[
 \sigma^2-\sqrt{\sigma^4+4\eta^2}
 \approx -2\eta^2/\sigma^2
\]

when eta is small. Rate removal can therefore make the desired solution possible without making it quickly learnable. This is not a theorem about the full transformer model. It predicts which interventions discriminate weak residual-feature coupling from compression: target-side information influences eta; code noise influences sigma; spatial aggregation influences the available psi(X).

A frozen 100x amplification test cannot settle this training hypothesis because it changes e while the learned readout w remains fixed. The existing rate-off gates are also essential but do not alone identify the two information paths.

For a Gaussian noisy state, a useful measurement is

\[
 I(X;\mu(X)+\sigma\epsilon)
 \le \tfrac12\log\det(I+\operatorname{Cov}(\mu)/\sigma^2)
 \le \operatorname{tr}\operatorname{Cov}(\mu)/(2\sigma^2).
\]

Record the variance sum explicitly. Squaring the mean per-dimension standard deviation is not the covariance trace and cannot be substituted into this bound. The bound concerns one noisy observation; correlated temporal noise and action-conditioned prediction require a corresponding joint-channel analysis.

## First discriminator: where spatial information disappears

The AS encoder uses learned-query attention pooling. Queries initialize at standard deviation .02, giving nearly uniform pooling when initial attention logits are small. TwoRoom's changing factor is largely spatial translation; that can preserve global appearance statistics while changing position. Cube articulation can change more global statistics. This architecture hypothesis has not yet been tested.

At identical episode-disjoint image cases, fit diagnostic position probes on (a) spatially retained ViT patch features, (b) mean/CLS features, and (c) QueryPool output, at initialization and a 600-update checkpoint. Use fixed feature reductions and train-only selection of ridge strength. If position is readable before pooling but disappears after pooling, the next intervention should target spatial aggregation rather than add semantic losses. If all stages are weak, the problem arises earlier. Probes are diagnostics and must not become physical-state supervision for the proposed method.

## Second discriminator: one small factorial, conditional on gate results

If the 16 existing gates confirm that TwoRoom still fails with rate off, the next useful experiment is a 2x2 rather than another beta grid:

| Factor | Reference | Intervention |
|---|---|---|
| State-channel corruption | current posterior noise | no sampling noise, with the quadratic rate scale held fixed |
| Renderer target-side information | current corruption-time distribution | t=1, so renderer input is pure noise |

The reference fresh beta=.25 arm already exists. Three new conditions times two seeds give six 600-update TwoRoom jobs. Keep data, all other settings, initial tensors and draws matched. This is a causal diagnostic, not a proposed final algorithm.

- Recovery after removing state noise alone supports a channel-startup bottleneck.
- Recovery at t=1 alone supports competition with noisy-target information.
- Recovery only when both change supports a joint startup obstacle.
- Failure under both shifts attention toward spatial aggregation, available features or conditioning implementation.

At t=1, squared flow regression is effectively conditional mean image prediction after accounting for the known noise. It changes the learning task and may sacrifice distributional generation quality. Separating sampling sigma from rate sigma similarly breaks the original probabilistic correspondence. Neither diagnostic should be advertised as an elegant new objective merely because it improves a short gate.

The abstract lesson is a distinction between *having an incentive to encode information at a good solution* and *providing a usable gradient path that can create the information channel from initialization*. LeWM's explicit spread regularizer supplies encoder pressure independently of a renderer learning to read the state; whether this explains the matched performance gap still needs the causal tests above.
