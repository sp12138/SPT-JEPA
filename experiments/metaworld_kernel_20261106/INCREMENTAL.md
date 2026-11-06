# From observation prediction to maintaining control

The first phase isolates a failure of the Gaussian target: its predicted motion is much smaller than the observed motion, and factual current-control inputs often predict worse than zero inputs. At1000 updates the point predictor beats copying on395–410/512 heldout windows across four seeds, whereas diagonal Gaussian goal models beat copying on33–52/512. All six manipulation tasks remain0 through the2500-update screen. At10000,point prediction reaches17/480=3.54% six-manip success,with Reach41.25/Wall21.25%;the two affine Gaussian arms remain0 on manipulation. Four nonlinear final evaluations are pending at08:51UTC. This is far below the78.75% reference,not a solution or a proof of impossibility.

An unrestricted drift can memorize the behavior policy in F(x,u)=b(x)+G(x)u. If the goal-conditioned observation mean is close to b(x), inversion produces a small control even when the demonstrated physical behavior requires a substantial control. Separately, a stationary image does not imply zero required force. A held gripper command can maintain contact without producing visible motion. Anchoring all zero-control motion to zero would be an unjustified solution for manipulation.

The next experiment instead models *changes from the agent's own previous command*. Let v_t=z_t-z_(t-1). A first-order expansion gives

    v_(t+1) approximately T(x_t) v_t + G(x_t) (u_t-u_(t-1)).

The identity transport is a constant-velocity approximation. A learned local linear operator can transport the observed velocity to account for changing image coordinates. Its output is linear in the observed velocity, so it cannot invent an arbitrary state-dependent drift when no motion was observed. The controller solves for the next bounded command while retaining the previous command in unobservable/null directions. This does not prove correct counterfactual dynamics: action coverage and the local approximation remain limitations.

Every target remains an actually observed NEXT random feature. Known current and past executed commands are inputs only. The previous command online comes solely from the agent's own actions. No action labels, demonstration action references, reconstruction, old learned weights, or pretrained teachers enter this phase.

The four independent-seed arms isolate: free drift without past actuation input; free drift with past actuation input; incremental dynamics with identity velocity transport; incremental dynamics with learned local velocity transport. The target-coordinate cache is the same frozen RANDOM map, not a learned first-phase teacher. All new trainable weights start randomly. The earlier RMS goal gate is removed for all four arms, and all use point future prediction; neither modification alone is claimed as a causal finding.

Each sample predicts the same next observation under known-current-control, unknown-control marginal, and future-goal information. The per-sample average over available queries is then averaged across samples, keeping the objective sample-separable. Goal-conditioned versus marginal control tests whether a goal adds information beyond the behavior's typical next observation. The marginal is not the zero-control distribution.

An initially considered goal/marginal density ratio cancels shared nuisance information, but also discards the behavior prior. If a scene already identifies its typical goal, the ratio can approach one while the demonstrated behavior still requires movement. Thus that ratio alone is not the primary controller. No claim of exact Bayes control or a novel Doob transform is made.

The primary controller matches the predicted goal-conditioned next observation using the learned forward response. The marginal-query controller is a paired ablation. Both use the same physical case bank,500native any-success,three imagined/executed five-native-step commands,RGB history and own past controls. Technical gates precede performance. Large improvement must be demonstrated against the old78.75% action-supervised six-task reference on complete common-case cohorts,followed by new-case validation.

Incremental nonlinear dynamic inversion and goal-conditioned observation planning have prior art. The present mechanism is motivated by the observed failures,not a claim that those broad concepts were invented here. Its effectiveness and publication-level novelty remain unestablished.

Primary prior-art references: [Incremental control with actuator dynamics](https://arxiv.org/abs/2201.09805),[robust flight control with INDI](https://repository.tudelft.nl/record/uuid%3Acf9119d3-8bbf-489b-b734-f25c4cd1402a),[causal confusion and targeted interventions](https://proceedings.neurips.cc/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html). These support the broad control/identifiability considerations;they do not validate the new manipulation experiment.
