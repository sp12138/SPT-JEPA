The previous frozen endpoint heads improved RGB forecasts and matched the actually executed input better than other realized controls in 95.7–97.3% of tested comparisons. Their expanded control results still achieved only 2/48 hard-task successes per objective, all on Drawer. Composing generated observations to imagine 120 steps then failed even the reaching tasks. These observations motivate testing whether the recurrent state behaves as an inferable physical state. They do not identify a unique failure cause.

This finite experiment compares three conditional latent world models, using the same two admissible factual parents and the same high-noise endpoint heads:

| Arm | Future latent during training | Decoder also reads current RGB |
|---|---|---|
| joint_anchor | Forward prior | Yes |
| belief_anchor | Posterior updated with actually observed future RGB | Yes |
| belief_state | Posterior updated with actually observed future RGB | No |

The objective is the conditional world negative ELBO:

`E_q[image reverse-chain NELBO] + KL(q(z_T | observed history, executed controls, observed x_T) || p(z_T | observed history, executed controls))`.

Both terms use nats divided by the same RGB dimension, 49,152. The KL has coefficient one, with no annealing, free bits, or batch regularization. The prior-only arm uses q=p. All distributions and normalization operate per sample. Gaussian scales have global numerical bounds, not task-specific settings. Image likelihood is a 64-step reverse-Gaussian-chain bound with its action-independent terminal prior omitted; it is not exact likelihood. Training uses one unbiased transition-KL estimate plus the endpoint term. Planning uses four stratified transition estimates plus the endpoint term, averaged over two latent samples.

Training inputs are recorded RGB histories and already executed forward controls. A uniformly selected valid recorded future image supplies the endpoint. A desired goal never enters training. At planning, a desired final image is evaluated as a hypothetical observation through the posterior and image bound. No action is predicted as a target or proposed by the posterior. The original parent's inverse pretraining reconstructs only the same first forward-input action; that inverse module is frozen in this phase. No expert paths/policies, protected evaluation trajectories, oracle data, state, reward, task labels, or external teachers enter training.

Each participant has 12,000 own factual-parent updates and 2,000 own head updates before this new phase. Six participants reuse two original trained parents; they are not six independent pretrainings. The phase intentionally starts a fresh AdamW optimizer at 1e-4, warmup100/cosine6000, global batch64, gradient clip1. The three arms share initial parameter values, data order, duration samples and all sampled noises within each seed. Four GPUs per learner split each global batch into four groups of16. Single-versus-four-GPU FP32 loss and gradient comparison passed on eight actual TRAIN windows; gradient relative L2 error was 5.11e-7. Complete model/Adam/scheduler/RNG/sample-cursor states are saved and restored between phases.

The six learners occupy four GPUs each on dev13/dev15/dev6 during training. Frozen evaluation follows in exclusive phases with one worker per physical GPU. Checkpoints2/500/2000 each receive 48 episodes: eight shared developmental physical cases repeated across six models. CEM64×4, horizon60, held controls5, execute15, 500-native-step any-success and ten image-renderer queries per candidate are common to all three arms. Extra posterior computation differs, so this is not an equal-FLOPs claim. It is not an official public benchmark, an untouched confirmation, or evidence of SOTA.

The finite maximum is 2,000 new updates per learner. A performance claim needs actual manipulation success across parents and additional cases, not a lower ELBO. There is no automatic long promotion. Existing negative attempts and all technical failures are retained. Hourly review remains a direct, interruptible wait in the current session; no external monitor or goal is created.

The 500-update pilot gave joint_anchor two hard successes (Drawer and Push), both from parent seed2. This triggered a frozen expansion of joint_anchor at2/500/2000 on32 shared physical cases:48 inherited and144 new model-case episodes,192 combined. It is developmental selection, not confirmation on untouched cases. A separate frozen solver probe fixes the500-update world and compares64×4 (inherited),300×15 and1024×8 CEM; these require17.578× and32× more trajectory forecasts. Its default64×4 implementation must reproduce20 actual actions, success flags and the first control bank exactly on both parents before formal evaluation. The new frozen workers may share GPUs with training: there remain24 physical cards.
