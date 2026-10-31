# Consistent-coordinate chart audit

Theory/CPU only; no model changes or GPU work. This is a correction of coordinate semantics, not a novelty claim or demonstrated performance improvement.

Let the fixed carrier be (u_t), and let (T_\theta) be a shared invertible map. Sample

\[
s_t=u_t+\sigma e_t,\quad e_t=\rho e_{t-1}+\sqrt{1-\rho^2}\eta_t,\quad z_t=T_\theta(s_t).
\]

For the native defaults \(\sigma=.1,\rho=.8\), the raw posterior conditional mean is \(b_t=u_t+\rho(s_{t-1}-u_{t-1})\) and its conditional variance is .0036. Define \(m_t=T_\theta^{-1}(F_\phi(z_{<t},a_{<t}))\), raw prior \(p_0=N(0,I)\), and \(p_t=N(m_t,\sigma^2I)\). Push **both** posterior and prior through the same T, including the entry prior. The native rate terms are then

\[
K_0=\tfrac12\|u_0\|^2+\tfrac d2(\sigma^2-1-\log\sigma^2),\qquad
K_t=\frac{\|b_t-m_t\|^2}{2\sigma^2}+\tfrac d2[-\rho^2-\log(1-\rho^2)].
\]

The last constant is .190825624 per dimension. Shared change-of-variables Jacobians cancel exactly in each conditional KL and the trajectory KL. No log determinant is needed when evaluating these raw-coordinate expressions. For nonlinear T, F's output is a transformed **location**, not generally the chart-distribution mean; applying the AR formula directly to chart coordinates is incorrect.

Under \(F_z=T\circ F_u\circ T^{-1}\), posterior-start and clean-start deterministic rollouts are conjugate. Retain the native forecast stream, deterministic rollout, posterior/forecast selection mask, corruption and anchor. With \(D_z(c)=D_u(T^{-1}c)\), the flow loss is pointwise identical too. The optional forecast-rate prior remains valid when its means use only initial sampled state and past actions; its comparison still uses raw \(b_t\). A stochastic prior draw must be \(T(m_t+\sigma\eta)\), not \(F_z+\sigma\eta\). Use raw goal distance after inverse mapping. This does **not** establish that the existing flow-plus-mixture total objective is an ELBO.

A scalar counterexample isolates the old incentive. Set \(u_0=1,u_1=2,F_u(s)=s,T_a(u)=au\). Keeping fixed additive chart noise, expected native entry-plus-transition rate falls from 52.51841 at a=1 to 14.64341 at a=.5, although inverse-mapped clean prediction error remains 1. Transporting noise and computing raw rate gives 52.51841 at both scales. Thus a pure coordinate contraction cannot lower the corrected rate while preserving raw predictions.

The current chart inverse is decorated `no_grad`; it cannot be reused unchanged for training. Correct differentiation requires
\[
dv/d\theta=J_T(v)^{-1}(dz/d\theta-\partial_\theta T_\theta(v)).
\]
Use differentiable fixed-point iterations or an implicit backward including both terms. For residual Lipschitz bound .5, inverse residual r certifies solution error at most 2||r|| in exact arithmetic. Check numerical residual and gradient accuracy relative to sigma; a fixed iteration count alone is insufficient. Inverse computation adds cost.

CPU scalar tests cover identity, contraction and nonlinear \(T(x)=x-.3\tanh x\): maximum rate/rollout/goal error 7.11e-15, density-ratio error 3.55e-15, mixture-decoder error 2.22e-16. Omitting the inverse's explicit parameter derivative produces .91762 instead of zero in a cancellation example. These are not native loss/gradient/RNG contracts.

Minimal fair test: frozen identity versus learned bounded T, both using this raw-coordinate formulation, identical carrier/calibration, original initial tensors, sample-level loss and planner. Instantiate the same chart in both; disclose 2,128 extra trainable parameters and inverse compute. Reuse old identity results only after exact native identity loss/gradient/RNG verification. Judge common-raw-coordinate forecasts and paired planning, not rate alone. No new GPU experiment is authorized here.

Normalizing-flow change of variables and contractive residual inverses are established ([Rezende & Mohamed](https://proceedings.mlr.press/v37/rezende15.html), [Behrmann et al.](https://proceedings.mlr.press/v97/behrmann19a.html)). With unrestricted conjugated predictor/decoder, T can be absorbed and adds no information. Any gain would concern finite-architecture parameterization/optimization; if conjugacy is enforced explicitly using unchanged raw networks, T cancels and is unidentifiable.
