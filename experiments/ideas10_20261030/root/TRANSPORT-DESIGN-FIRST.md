# R3: transport posterior noise through the action dynamics

Independent specification before a targeted nearest-prior search, 2026-10-31 Beijing. This is a candidate, not a novelty or performance claim. It must be distinguished from our old I03's fixed AR(1) noise and from published residual variational posteriors.

Current temporal-forecast ties noisy state samples with a scalar rho. The clean planner uses a nonlinear action transition F. Even when clean means obey mu_next=F(mu,a) exactly, a fixed Euclidean noise coupling generally does not obey that transition. For affine F(z)=Az+b, the expected squared transition discrepancy includes sigma²||A−rho I||_F². Thus the rate term can penalize the *correct* action-dependent Jacobian because posterior noise has the wrong transport law. This is a geometric mismatch, not insufficient state variance.

Propose a Markov posterior on the observed clip:

q(z0|o0)=N(mu0,sigma0²I),

q(zt|zprev,ot,oprev,a)=N(mut+F(zprev,a)−F(muprev,a),s²I).

The world-model prior remains p(zt|zprev,a)=N(F(zprev,a),s²I). For each conditional transition their means differ exactly by mut−F(muprev,a). Hence conditional KL is ||mut−F(muprev,a)||²/(2s²), independent of the sampled prior noise, for *any* F. It is a proper sequential variational posterior and prior, using only recorded observations/actions; no target action oracle, batch statistic or physical label is added. The observed correction becomes zero whenever the clean transition is correct, so q and p agree as entire trajectory laws in that case. The encoder and transition receive ordinary gradients of this one objective.

Rendering conditions use the corresponding posterior sample; forecast-render continues to use prior rollout. A clean-state predictor supplies the observation correction, a noisy-state predictor transports uncertainty. Their difference is a control variate dictated by the posterior definition, not a stop-gradient trick. Initial rate and image reconstruction remain. No module needs to be added. Predictor calls and posterior noise energy must be matched in direct controls.

Nontrivial separation: a rotation with A orthogonal and A != rho I is exact dynamics, yet scalar-AR noise incurs a positive extra rate. Transported posterior has zero conditional KL at exact clean prediction. This does not assert that eliminating noise-contraction pressure is always good: that pressure also regularizes extrapolation. A badly learned F can propagate noise explosively, and exact KL cancellation does not guarantee useful image information or semantically correct goals. Cubic F with broad initial noise is a deliberate negative control.

The transition noise s cannot silently be copied from fixed stationary marginals. For identity dynamics, variance accumulates sigma0²+t*s². One comparison should match the total clip innovation budget, e.g. s²=sigma0²/(window−1), and directly report realized variances. This is a changed posterior/prior family and needs its own exact rate accounting, not a claim that it is the original fixed-marginal KL. A deterministic s=0 choice is singular and is excluded.

TwoRoom and Cube relevance is a testable bridge: both have state/action-dependent blocked/contact dynamics. A fixed isotropic AR relation can misprice their derivatives; the current rate/noise ablations do not isolate this mismatch. A short test should measure the Jacobian-mismatch contribution on frozen checkpoints, then compare equal-size scalar coupling and transported coupling with matched noise budget before task evaluation. Passing a rotation example alone is insufficient real-task evidence.
