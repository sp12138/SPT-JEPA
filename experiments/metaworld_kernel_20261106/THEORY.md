# What the successful action target may actually supply

## Current evidence, November 7, 09:39 Beijing time

The user now permits inverse learning when necessary. A useful matched gain
exists, although a broadly successful purely observation-supervised controller
has not been found. All numbers below use our independent LeRobot expert corpus,
RGB goals, and 500-native-step any-success evaluation. They are not public
JEPA-WMs scores or a canonical MetaWorld MT50 comparison.

The strict single-native-action candidate reaches 89.17% on six manipulation
tasks, 89.25% in the small 50-task screen, and 86.80% on the expanded 50-task
1,000-physical-case cohort, using model feedback between five-step real observations. It supervises only the known current a_t. The completed
controlled comparisons and the distinction from the older 20D models follow.

At the original 40k budget, pairing current/goal local features before aggregation
improves six manipulation tasks from 187/240 (77.92%) to 217/240 (90.42%). On the
stronger 50-task confirmation, there are 1,000 new physical cases, each evaluated
with two trained learners: 1,738/2,000 (86.90%) versus 1,623/2,000 (81.15%). The
paired physical-case bootstrap interval for the +5.75 percentage-point difference
is [4.10, 7.35], conditional on these two trained learners. An earlier four-case
per task screen gave 92.0% versus 83.75%; use the larger confirmation as the main
broad result. Four new scratch seed pairs also improve manipulation at 40k:
88.75% versus 76.04%. Their 120k schedule differs from the original 40k schedule;
only the within-pair comparison is matched.

These actors supervise a CURRENT f5 transition word: five known executed native
controls, 20 dimensions. They do not predict later macro-action labels. This is
not the same as supervising only the single native action a_t. A separate strict
4D actor supervises ONLY a_t, including no a_(t+1) label, and currently reaches
73.33% manipulation and 72.25% on the 50-task screen when its output is held for
five native steps. The distinction is material and must remain visible.

The inverse loss anchors learning in actuator coordinates. A small image change
from maintaining a grip may be cheap in observation space while a wrong grip
command remains costly in action space. It also learns a goal-conditioned
behavioral prior from an expert corpus. Thus the inverse gain does not establish
that a better world representation alone was learned, nor that the same method
will work with arbitrary failed exploration. Logged current controls are known
observations, not labels asserting the next optimal action.

The architecture ablations isolate a narrower, useful mechanism. For paired
patches c and g, the successful module applies a linear map to [c, g-c] before
global aggregation. [c, g] is an EXACT alternative parameterization:

    W_c c + W_d (g-c) = (W_c-W_d)c + W_d g.

Converting the initial linear weights makes the complete initial functions
equal. This concatenation control scores 89.58% versus 90.42%; the difference
interval includes zero. Merely giving the original separate token sets distinct
role initializations scores 77.92%. Evidence supports early ordered local
correspondence, not subtraction as a new function class. The paired module adds
parameters and halves decoder memory tokens, so the comparison is not equal
FLOPs. High-noise training, fixed sampling noise, extra own-intervention data,
and unstructured history inputs did not improve the completed full comparisons.
Adding the last four available a_t training states also produced no clear gain.

There is now a separate control-clock experiment. The strict 4D actor was trained
at native action time but originally deployed by holding one sample for five
steps. Instead, within each five-step real-camera interval, generate controls
sequentially with its OWN scratch observation-predictive world model:

    a_j ~ q(a | predicted_image_j, goal),
    predicted_image_(j+1) = F(predicted_image_j, a_j, own_past_control).

Only the first image is actually observed. Execute the five self-generated
controls, refresh the real camera, and repeat. No demonstrated future control,
actual intermediate future image, privileged state, or external teacher enters
this procedure. The actor's only action target remains the current native a_t.
The 20k world model uses future RGB targets and known current controls as inputs;
its own-data pretraining cost must be counted along with the 40k actor.

The conceptual separation is useful: the inverse actor supplies goal-conditioned
actuation on the behavioral support; the forward model supplies short-time
feedback between real observations. This avoids asking a terminal latent metric
to encode preparatory contacts, while also avoiding a multi-action label head.
It does not by itself establish novelty or equal-compute superiority: there are
five policy calls per camera interval plus four world predictions.

The initial small pilot was encouraging but insufficient: world feedback succeeded on
24/24 manipulation episodes versus 20/24 with copied images and 20/24 with the
original 224px image held constant. Both controls take the same five own action
draws. BinPicking is 4/4 versus 0/4. Reach is still 0/4, and all numbers reuse
only two physical cases per task crossed with two learners. Full 20-case eight
task and four-case 50-task evaluations, including zero and reversed imagined
controls, have now finished; see the controlled evidence below. The expansion
to the existing 1,000-case cohort has completed (details below).

Machine-readable provenance and active ownership are in RECOVERY.json. The
remote read-only comparison report is current-comparison-confirm.json under
reports/metaworld_kernel_20261106. All earlier experiments and failures remain
preserved; the following sections describe the earlier strict no-inverse phase.

### Completed native-time feedback evidence

The full current-a_t-only feedback comparison has now finished. All six
manipulation scores below contain 240 episodes: 20 common physical cases/task
crossed with two trained actors. Every method observes the real camera only
every five native steps and uses the same 500-step any-success budget.

| Deployment of the SAME frozen 4D flow actor | Six manipulations | All eight tasks |
|---|---:|---:|
| Hold one sampled action for five steps | 176/240 = 73.33% | 245/320 = 76.56% |
| Five action samples, unchanged actual 224px image | 179/240 = 74.58% | 246/320 = 76.88% |
| Five samples, copied intermediate 128→224px images | 176/240 = 73.33% | 245/320 = 76.56% |
| Predict images using zero imagined controls | 196/240 = 81.67% | 259/320 = 80.94% |
| Predict images using negated imagined controls | 199/240 = 82.92% | 250/320 = 78.13% |
| Predict images using the actor's OWN controls | 214/240 = 89.17% | 277/320 = 86.56% |

Versus five samples on the unchanged image, the manipulation gain is +14.58pp,
physical-case cluster CI [10.00, 19.17], conditional on two trained actors. The
50-task screen also finishes at 357/400 = 89.25% versus 301/400 = 75.25%, with
individual trained-seed gains +15pp and +13pp. Four cases/task is a screen;
evaluation on the existing 1,000-case comparison cohort completes at 1,736/2,000
=86.80% versus 1,421/2,000=71.05%. The +15.75pp interval is [14.20,17.30],
conditional on two trained actors, with per-actor gains +15.4pp and +16.1pp.
This is an expanded existing cohort, not untouched confirmation for this method.

The zero-control diagnostic does NOT imply a frozen or zero-flow image: the
stencil world model still reads known past controls and can predict their
transient/release effect. Negation also preserves the absolute-value action
features. Thus these negatives are narrower than disabling all dynamics.
They establish that correct current-action conditioning adds value, while the
remaining gains require more care than a claim of perfect physical fidelity.
The copied-image control separately excludes simple image resizing as the
complete explanation.

A deterministic one-current-action MSE actor with the same world model reaches
208/240 = 86.67%, versus 161/240 = 67.08% when held for five steps. It uses one
action-field pass per policy call, compared with twelve flow integration passes.
Both execute five policy calls between real observations. No Gaussian action
draws occur in the MSE actor. Copied images give only 162/240 = 67.50%, confirming
that deterministic control also benefits from advancing the predicted state.
However, its 50-task screen is only 335/400 = 83.75%, below flow's 89.25%.
Two queued MSE continuations were therefore declined before SGD (step0,
trained=False), not counted as trained models. The broad advantage of flow
does not uniquely prove multimodality as its cause.

The best existing 20D five-control model still has 217/240 = 90.42% manipulation
success. The stricter 4D feedback method is comparable, not a demonstrated
improvement over that score: the difference is -1.25pp, CI [-4.58, 2.08]. Its
advantage is recovering high performance without supervising subsequent native
controls. The 20k forward-model pretraining and additional inference cost remain
explicit. Four fresh paired native-action learners are now running to 120k,
with actual native2/600 full-state restores and physical entry checks passed.

### What the inverse target supplies, beyond a terminal image metric

An elementary conditioning argument helps separate desirability from local
physics. In an ideal fully observed Markov system with behavior policy pi,
define h_T(s) as the probability of reaching a specified goal neighborhood G
after T steps under that behavior. Where h_T(s)>0, Bayes' rule gives

    q_T(a | s, S_T in G)
      = pi(a | s) * E_{s' ~ P(.|s,a)}[h_(T-1)(s')] / h_T(s).

The first recorded action from a hindsight trajectory is a sample from this
goal-conditioned behavior distribution. It is not an assertion that the action
is optimal. A preparatory grip can be valuable because it raises future goal
probability, even if its immediate pixel displacement or terminal-image-distance
gradient is tiny. Inverse learning directly fits that control-coordinate
marginal; a short-range forward model instead fits local state propagation.
Replacing h with an arbitrary Euclidean latent distance need not preserve this
ordering. The formula is explanatory, not a new theorem or an exact derivation
of our horizon-mixture, partially observed pixel actor. In stochastic dynamics,
conditioning on a later goal also changes transition-noise probabilities.

This suggests the useful abstraction supported by the experiments: a learned
goal-conditioned actuation distribution carries long-range desirability, while
a short observation-predictive operator maintains the local feedback state.
The same 20k world model scored only 6.67% when paired with its old global
projection controller, yet is useful in this different role. That cross-controller
comparison is not matched inference compute and does not isolate a single cause.
The stronger causal evidence is the SAME-actor/SAME-camera frozen comparison
above, including unchanged, copied, zero-control, and reversed-control images.

The camera interval and native actuator interval were different clocks. Holding
q(o,g) for five steps substitutes constant actuation for q(F(o,a),g) at the
intermediate states. Repeated sampling on stale observations barely repairs
this mismatch. Four locally predicted states permit an approximate internal
feedback loop without using an actual intermediate observation or a future
demonstrated control as a reference. This is a practical mechanism with evidence;
novelty and superiority to public methods under matching data remain unproved.

### Mechanism tests completed after the first feedback result

Moving only the hindsight goal-image grid from five native steps to one did
not improve the 50-task screen: both matched seed5/6 groups score84.5%, delta0pp,
CI[-3.25,3.00]. Across four paired actor seeds, the six-manipulation delta is
-1.46pp, CI[-4.58,1.67]. Retain the simpler existing goal grid. Action labels
remain ONLYa_t throughout; all 256 default-data parity windows and 192 current
control access guards passed.

Training maturity is a distinct factor. For seeds3/4 with exactly the same
random initial tensors, sampler, current labels and forward models, fully
annealed40k training gives206/240=85.83%, versus189/240=78.75% at step40k of a
120k schedule. The +7.08pp interval is[2.50,11.67]. The change includes the
schedule horizon and its derived1%warmup; it does not establish a seed effect.
At120k the longer schedule gives205/240=85.42%, so more steps have not produced
a large gain over the40k fully annealed controls. All data/budgets remain visible.

A stronger goal-information control duplicates CURRENT before the CNN in both
image slots, retaining every parameter, initial tensor, label, sampler and
compute path. With the same40k schedule, ownF and physical cases, proper-goal
training reaches206/240=85.83% versus172/240=71.67% without a future goal image.
The +14.17pp interval is[9.17,19.58], conditional on two actor pairs. This supports
useful goal information beyond the current image; it does not prove arbitrary
new-goal generalization. An inconsistent goal-image substitution separately
changes every first command in32cases/arm; those are dependency diagnostics,
not valid donor-goal performance episodes. The broader information comparison
is still running and must retain the same distinction.

Within-real-observation latent coupling gives216/240 versus214/240 for old
actors, and188/240 versus189/240 for new actors. Swapping the two groups' own
forward models does not recover the newer actors' performance. A separate ODE
audit improves numerical agreement with Euler768 substantially, but Euler24
and midpoint12 give only small physical gains over Euler12. These are negative
results for strong claims about noise consistency or solver accuracy. They do
not justify additional controller parameters.

The original20D models'120k four-seed expansion finishes at3510/4000=87.75%
versus3241/4000=81.025% for absolute tokens. These models supervise FIVE native
controls, not just a_t, and are labelled comparators. The strict4D route86.80%
has recovered similar broad performance but has not substantially exceeded
this strongest measured result. Old chunk6 with the same real-camera5 feedback
scores186/240=77.5% on six manipulations; strict4D+F scores214/240=89.17%.
That end-to-end gain is real, but data anchors, model, label scope and forward
pretraining/inference cost differ; it is not a single-component causal ablation.

### Bounded next tests, with no advertised performance gain

The original TRAIN-only181608 current controls have exact endpoint masses
19.23%,26.21%,39.56%,49.76% across four actuator axes. Clipping a continuous latent
command induces boundary atoms: its saturated labels represent tail events,
not interior density observations. This motivates a one-pass conditional
logistic mixture likelihood with analytic boundary masses. It uses exactly the
same currenta_t label and visual architecture; no gripper-specific threshold,
future command or new data is added. Ordinary point likelihood is the matched
negative control, and one versus four components separates censoring from
multimodality. The mixture-density idea itself is established prior art
([Bishop1994](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf));
we claim a testable actuator-measure explanation, not invention of mixtures.

A separate scratch matcher learns current-to-goal patch correspondence and an
image-coordinate displacement before the existing flow decoder. Image-grid
coordinates are not object state. It adds query/key/fusion parameters, which
must be disclosed; this is not an equal-capacity or novelty claim. Two additional
fully annealed baseline seeds test robustness. Twelve actors total have a fixed
40k budget/LR40k, paired initial CNN/decoder tensors, original data and ownF3/4.
All12 native2/600 full restores, held-out gradients, sample-independence checks
and20-native-step physics entries pass. Their physical performance is pending;
there is no automatic120k continuation from these short tests.

Eight frozen follow-ups also compare the same mixture actors' stochastic draws
with the median of their highest-weight latent component, followed by the same
actuator clipping. This avoids averaging distinct components, while removing
sampling variation. It is not claimed to be the full mixture MAP, or an optimal
physical action. Both rules use the exact same trainedNNs/world/camera5/five
policy calls; no new label, threshold, teacher or task rule is added. Default
proposal/RNG parity and deterministic bounded median behavior pass on actual
saved weights and held-out image pairs. These are queued zero-SGD50-task screens;
no performance benefit has yet been measured.

### The remaining failure is not uniform across tasks

The matched current-only information control has now completed50tasks:
293/400=73.25% versus346/400=86.50% with a proper future goal image.
The +13.25pp case-cluster interval is[10.00,16.50], conditional on these two
trained actor pairs; both per-actor gains are positive(+20.5pp,+6.0pp). Both actors
use the same40k schedule, data, initial architecture and ownF. The aggregate
supports useful goal conditioning; it does not establish arbitrary new-goal
transfer or purely observation-supervised control.

Inside the ongoing1,000-case comparison, three tasks have fully completed40
matched episodes/arm with trace/caseSHA verification. World versus EXTRAactual
RGB every native step scores24/31 Basketball,31/30 BinPicking,1/15 SideHandlePull.
These are task-local diagnostics, not the final50-task score; extra observations
are a different information track. The world-feedback gap is task-dependent,
and the good six-task mean cannot justify assuming F is adequate everywhere.
Neither the gap nor a global pixelMSE uniquely identifies image blur as its cause.

A zero-SGD experiment therefore tests a general state-update operator. If
image_j(x)=image_0(C_j(x)) and a predicted backward map is w_j(x), then
C_(j+1)=C_j composed with w_j. Store the composed coordinate map and render the
source image once, rather than resampling its RGB repeatedly. Both the policy
and next forward-field query consume this once-rendered image. A128-source
control isolates repeated resampling, and a224-source arm also preserves original
image detail. The weights, goals, own noise, real camera5, five policy calls and
four F queries remain fixed. Additional coordinate-map sampling/render compute
is disclosed. Zero-motion224 identity is byte-exact; four half-pixel checkerboard
warps lose all contrast sequentially but preserve variance.25 when composed.
This synthetic fact is not evidence of a manipulation gain. Eight frozen50-task
screens on actors1..4 are queued with physical entry checks before formal scoring.

All twelve new actor variants have also completed40k and passed actual native
checkpoint/optimizer restores, held-out gradients and sample-independence checks.
Early completed BinPicking/PickPlace cases do not show a consistent censoring
benefit. Their complete physical comparisons and component-median controls are
still pending; none is promoted on training likelihood or selected tasks.

The next completed full-eight-task endpoints give six-manipulation success
85.42% for one censored logistic versus80.00% for its exact-initialized ordinary
point-likelihood control. Four-component censoring scores82.08% versus82.92%
for ordinary point likelihood. The same-seed mature Flow control is85.83%,
and additional Flow seeds5/6 score86.67%. Each listed arm has240 manipulation
episodes/320 total; the new50-task screens and deterministic-decision controls
remain pending. This does not support a universal censoring advantage or a
large improvement over Flow. The single-component model uses one action-field
pass instead of twelve; comparable performance could be an efficiency result,
but requires the full broad comparison. Do not select architectures per task.

### Completed broad negative tests and a targeted information repair

The50-task screens now reject a universal actuator-likelihood improvement:
censored4/point4/censored1/point1 score82.25/83.50/81.25/79.50%, respectively,
versus the matched Flow86.50%. Highest-component-median decisions give
82.00/81.75/82.75/79.00%; none fixes the gap. Each arm has400episodes/two
trained actors. One-pass computation is cheaper, but the physical performance
requirement is not met. All models and negatives are retained; none is promoted.

Replacing the fixed-chart goal features by soft matched features also scores
only71.25% over six manipulations, versus85.83% for the matched control. This
suggests a different failure to investigate: motion relative to a fixed camera
is useful control information. Making appearances align can discard that motion
before actuation is decided. The existing model also computes matches before
including image positions. Optimization and extra capacity remain alternative
explanations; the result does not uniquely establish information loss.

The targeted repair retains[c,g-c]AND[aligned_goal-c,coordinate_displacement]
through one fusion layer, and supplies the existing learned image-position
embeddings to the matcher. Its function class includes the original fixed-chart
model by setting additional feature weights to zero, although initialization
is not that exact function. There is one current-action Flow loss, no new data,
external teacher, state, future command or per-task rule. Two new seed3/4actors
have a fixed40k budget and the same oldF3/4. Actual old/default functions are
byte-identical on saved weights and real image pairs; both new CPU gradient and
current-action guards pass. Native2/physics gates await their owned GPU leases.
No repair performance has yet been measured. The same-frame and composed-state
operators remain separate tests, preserving their distinct mechanisms.

## Historical mechanism test before inverse learning was permitted

The objective is still unmet. The old action-supervised score is only a scalar
comparison; no inverse learner,its weights,outputs,or traces enter this study.
The following separates mathematical statements from empirical hypotheses.

Let a known executed control word `u` produce an observed image path
`Y_H = F_H(h,u)`,where `h` includes information inferred from observation and
own-control history. Near a reference word,an image prediction error induces
the control-space quadratic form

`delta_u^T G_H delta_u`,with
`G_H = sum_k J_k^T Sigma_k^{-1} J_k`,

where `J_k` is the derivative of the k-th future observation with respect to
the control word. This is a local statement,not a claim that image MSE is an
action loss. If `G_H` has a small eigenvalue,excellent image prediction gives
a weak bound on a control error in that direction. An action target instead
measures errors directly in the logged actuator coordinates. It also trains
a goal-conditioned controller,so its gain cannot be attributed solely to
representation quality. Logged demonstrations need not be optimal actions.

A preparation action can principally change *which later transitions are
possible*. It need not change the immediate image substantially. Increasing
the horizon can reveal that effect,but is not sufficient in general: for a
bilinear delayed effect `y_next = y + hidden_force * later_motion`,the local
derivative at the all-zero word can remain zero. Ordered,finite interventions
can distinguish these systems even when a zero-control linearization cannot.
This motivates latent memory,compositional observation likelihood,and symmetric
experiments over all control axes. It does not motivate a hand-coded gripper
rule,an action decoder,or pseudo-action labels.

There is a separate identification problem. If the demonstrations satisfy
`u = pi(h)`,then every model

`F_tilde(h,u) = F(h,u) + A(h,u) * (u - pi(h))`

has the same factual predictions on that support. A state-only predictor may
therefore fit well while exposing no useful control dependence. This does not
prove imitation through a forward model impossible: the demonstrated control
can remain a valid solution. It shows why fitting the observations alone need
not make the planning solution unique or make its derivatives physically right.
Same-context interventions constrain this otherwise unconstrained family.

The present evidence is narrower than this interpretation:

- Native image forecasting improved substantially over copying,yet the best
  native20k arm achieved only32/480=6.67% manipulation success.
- A CPU physical diagnostic of eligible own-forward trajectories found42/576
  signed control-order pairs differing by more than1cm at nearest-TCP contexts,
  versus0/576 at initial contexts. These are repeated pairs within shared
  physical contexts,not576 independent environments. Some failures occur before
  contact;some include contact/lift without completion. No unique failure cause
  follows from these observations.
- A matched original-data recurrent experiment,with the same five future RGB
  targets and paired initial weights,scored9/48=18.75% on six manipulation tasks
  at2500 updates when belief was composed. Resetting it on each observed
  transition scored0/48. The gain was door4/8 and drawer5/8;the other four tasks
  remained0. This is a screening result on four shared physical cases/task and
  two trained parents,not confirmation or an all-task solution.
- Frozen original-data models are now tested on exactly50 common,previously
  unseen own-intervention contexts. This audit separately measures factual
  forward error,odd control-response agreement,and whether a correct achieved
  RGB goal predicts better than a different branch goal from the same context.
  No audit optimizes a learner or predicts an action label.

The new corpus contains11600 own-command branches from400 contexts across50
tasks. All commands were self-generated experimental inputs;no expert command
was read/replayed,and no reward/state/success/task-ID tensor is a learner input.
Simulator state is used only to restore a branch and for separate diagnostics.
All branches of each exploration parent share one train/validation split and
benchmark physical cases are excluded. It is a separate extra-data track.
Original-data experiments never silently consume it. A comparison using it must
give the same corpus to its baselines and count collection/full pretraining
cost. Neither the data nor these results are equivalent to the gated official
JEPA-WMs public setting.

The previous 78.75% result used an action prediction target. It did not establish that image reconstruction is necessary, or that action labels are the only route to its useful information. The current strict experiment tests a different explanation: a distant goal needs to specify a *local transition*, while a forward model needs to specify how controls can realize that transition.

## A population equivalence, with explicit assumptions

Let x denote a sufficient observation history, g a future observation, z' the next observed feature, and u the already-executed current-transition control. Assume

1. The conditional mean of the next feature is control-affine: E[z' | x,u] = z(x) + b(x) + G(x)u.
2. After conditioning on x,u, the residual has zero conditional mean even when g is specified. This is stronger than an ordinary Markov assumption: conditioning on a future endpoint can select transition noise.
3. G has full column rank, and the relevant behavior controls lie in the allowed box.

Then the law of total expectation gives

    E[z' | x,g] - z(x) - b(x) = G(x) E[u | x,g].

Both the left-hand expectation and the forward dynamics can be learned using **only next-observation prediction targets**. Solving the four-dimensional forward inverse gives

    argmin_u ||b + Gu - (E[z' | x,g] - z)||_W^2 = E[u | x,g]

for every positive definite W. Thus the population conditional-mean controller learned by action MSE has an observation-target counterpart. This is a statement about sufficient statistics and identifiable dynamics, not a claim that the finite networks are equivalent or that the action is optimal. The controls in this equation are factual executed controls, never future action targets.

This suggests a concrete abstraction: **query one next-observation variable with two information sets, and reconcile the resulting predictions**. The goal query supplies local progress; the executed-control query supplies physical realizability. A terminal appearance distance conflates these roles and need not represent either one correctly.

## Why this can fail

Partial observation, contact discontinuities, stochastic endpoint selection, or behavior-dependent hidden state break the equivalence. An expert corpus may cover only one control near each state. For any on-support model F, F(x,u) + H(x,u)(u - pi(x)) agrees on expert data u=pi(x) while implying arbitrary off-support control effects. Low factual prediction error cannot distinguish them. Even deterministic dynamics do not fix inadequate action coverage.

Gaussian prediction can describe ambiguity but cannot manufacture information. A Gaussian boundary mean can average incompatible routes. Diagonal uncertainty also depends on the feature coordinates; it is not a coordinate-invariant geometry. Fixed random features avoid target collapse and pretrained priors, but may still represent the relevant object poorly. The experiment must separate these failure modes rather than equating small loss with understanding.

## The historical observation-only projection controller

The shared network predicts next fixed-random observation features in two modes. The control mode reads only the current transition's known executed command. The boundary mode reads the current history and a later goal image, with all controls zeroed. Its target is strictly between the two observed endpoints, so it cannot copy a provided target frame.

For affine Gaussian predictions, their overlap induces the projection weight (Sigma_forward + Sigma_boundary)^-1. The controller enumerates all 81 active faces of a four-command box and solves the resulting small quadratic programs. There is no learned action head, action reconstruction, action retrieval, old controller distillation, or pixel decoder. The nonlinear control arm tests whether the affine restriction, rather than the information factorization, is responsible for an outcome.

## Evidence needed

- Full-state restore, positive held-out gradients, frozen-coordinate equality, sample independence, and actual physics are technical gates.
- Boundary versus forward-only control tests the value of learned local progress under the same observation coordinates.
- Point versus Gaussian targets tests whether predictive ambiguity improves actual control.
- Affine versus nonlinear dynamics tests the analytic inverse restriction, with different inference compute disclosed.
- Factual prediction versus copy, sensitivity to current executed controls, the projection residual, and physical response to self-chosen commands diagnose failures. Model-scored alternatives alone are not physical counterfactual evidence.
- A performance claim requires common physical cases, independent training seeds, and complete success traces. The primary reference is the old 78.75% six-manipulation result, with its different model, target and budget disclosed. A few screening cases cannot establish a substantial improvement.

## Prior art boundary

Goal-conditioned observation prediction and learned visual servoing are established. GCP generated visual plans and executed them using action-supervised inverse models; this experiment instead inverts an observation-supervised forward model. This distinction meets the current supervision constraint, but alone does not establish a publishable novelty claim.

Primary sources: [GCP project and paper](https://orybkin.github.io/video-gcp/), [Learning Visual Servoing with Deep Features and Fitted Q-Iteration](https://arxiv.org/abs/1703.11000), [Analyzing Neural Jacobian Methods](https://arxiv.org/abs/2106.06083).
