# Why temporally correlated posterior noise may help an identity-carry prior

This is an exact calculation for a one-step linear Gaussian example, followed by a hypothesis about our model. It is not a new theorem or evidence that a candidate will improve PushT.

Let posterior means be mu_prev and mu_next, with fixed marginal noise sigma. The prior mean is A s_prev, with covariance sigma^2 I. Write delta = mu_next - A mu_prev.

For independent posterior noise, the expected transition KL is

    ||delta||^2 / (2 sigma^2) + 0.5 ||A||_F^2.

For a stationary Markov posterior with correlation rho, the next conditional posterior is

    q(s_next | s_prev, x_prev, x_next)
      = N(mu_next + rho (s_prev - mu_prev), sigma^2 (1-rho^2) I).

The expected conditional transition KL against the same prior is exactly

    ||delta||^2 / (2 sigma^2)
      + 0.5 ||A-rho I||_F^2
      + (d/2) [-rho^2 - log(1-rho^2)].

The last term is constant with respect to learned parameters. Thus independent posterior noise adds a Jacobian penalty toward zero, whereas the Markov posterior shifts the preferred linear dynamics toward rho I. An identity-carry initialization A=I pays mean-mismatch noise cost d/2 for independent noise, versus (d/2)(1-rho)^2 for the Markov posterior. At rho=.8 the mean-mismatch part is25times smaller; this excludes the covariance KL constant and does not mean the complete KL is25times smaller.

Both posteriors have the same per-frame marginal variance sigma^2 I. This intervention is not simply reducing each frame's marginal noise. It does change the joint posterior and therefore the rate objective. The code must include the conditional covariance KL; omitting it would misreport the rate. Constants must not count toward active dimensions.

For our nonlinear, multi-history predictor the above one-step expression is only a local guide. Markov-correlated history noise changes covariance terms across past states as well, so the one-step formula cannot be asserted as an exact objective for the whole transformer. Preserving slowly varying dynamics and reducing unnecessary shrinkage is a testable hypothesis, not an established explanation of an observed gain.

The forecast-rendering factor addresses a different issue: the original renderer is trained on posterior codes, but autonomous predictions use prior codes. Replacing a fixed fraction of rendering conditions by causal multi-step prior forecasts gives the generation loss a direct path through the rollout. No future RGB is supplied to that forecast. It can still harm goal embeddings or make the renderer ignore an inaccurate state; generation, counterfactual block motion and closed-loop planning must all be checked.

The factorial comparison separates posterior structure, forecast conditioning, their interaction and a simpler lower-beta retuning control. All have identical model size and sample/window-level objectives. Actual compute differs when extra prior unrolling is used.
