## Planner context versus the existing seven-step cold audit

Read-only audit of completed records and pinned code; zero model calls, simulations, probe fits or setting changes. The dynamic600 report contributes all12 task×seed×identity/static/state records. Both seeds of existing native, TF and F64 references were also checked.

| Actual quantity | TwoRoom | Cube |
|---|---:|---:|
| CEM future horizon, in model steps | 5 | 5 |
| Receding horizon, in model steps | 5 | 5 |
| Raw actions per model-step block | 5 | 5 |
| Maximum real context frames, including current | 3 | 3 |
| Raw action dimension | 2 | 5 |
| Blocked model action dimension | 10 | 25 |
| Raw environment actions searched / executed per plan | 25 / 25 | 25 / 25 |
| CEM candidates / refinement iterations / elites | 300 / 30 / 30 | 300 / 30 / 30 |
| Goal offset / total evaluation budget, raw steps | 25 / 50 | 25 / 50 |
| Training window / dataset frameskip | 8 / 5 | 8 / 5 |
| Predictor maximum input context | 7 | 7 |
| Objective time selection | Last predicted state | Last predicted state |

These plan values are read from actual completed `metrics.json`, not inferred from documentation. All12 dynamic600/control entries agree; the native/TF/F64 references agree too. The evaluator explicitly sets `history_len=3`. CEM's action dimension multiplies raw action dimension by5, and the chosen sequence is reshaped into25 separate raw actions. Thus this is a block of five potentially different actions per predicted transition, not necessarily one constant action repeated five times. The policy buffers observations every raw step, but normally replans only after its25-action queue empties; successful/terminated episodes can stop earlier.

At the first plan, the online history buffer contains one real starting image and zero past action blocks. No prior dataset frames or expert future action labels are inserted into that history. The first starting RGB is the manifest's dataset observation injected by the evaluation reset path. The model predicts five future states, corresponding to raw times5,10,15,20,25; predictor calls receive prefixes of lengths1,2,3,4,5. A separate goal RGB is the fixed dataset state25 raw steps after the selected start.

If an episode survives to the second plan at raw time25, the buffer supplies three real images from raw times15,20,25 and the two five-action blocks between them. Subsequent simulated states remain model predictions, with no additional real observations inside the search. This plan predicts times30,35,40,45,50. Predictor input lengths are3,4,5,6,7: by the final call there are three real context states and four previously predicted states, producing the fifth future state. The returned trajectory contains8 states, but only5 are new predictions. Input context length7 is not prediction depth7.

For synchronized benchmark starts and `mode=wait`, the50-step budget permits at most these two plans per surviving episode. The generic buffer can pad mixed-fill asynchronous histories, but the policy first requests only the maximum available context up to3. Consequently the synchronous first plan uses a genuinely short context, not three copies of its initial frame. No fresh episode is auto-reset into the wait-mode evaluation batch. The warm-start option supplies no unused tail here because receding horizon equals search horizon.

The shared `LeWM.rollout` uses `history_size=predictor.num_frames`, which the actual eight-frame model configurations set to7. It appends the known historical action blocks before candidate future blocks and slices observation/action prefixes together. All audited model families inherit this rollout; neither native's projector, TF's residual prediction, F64's carrier nor static/state dynamic readout supplies additional real history. Their predicted states and representations differ, not these information conditions.

The objective configuration selects `stable_worldmodel.planning.GoalMSE` with no additional arguments. Its pinned implementation scores only `predicted_emb[:, :, -1:]` against the last goal embedding and sums squared differences over latent coordinates. It does not average the five forecast states, sum intermediate state costs, or score the seventh cold-audit prediction. Actual metrics store the plan/solver but omit objective kwargs; the terminal/sum statement is established by the exact imported objective configuration plus its constructor/forward code, not by assuming a customary MPC objective.

Training and audit alignment:

- Eight training RGB frames are spaced5 raw steps apart, spanning35 raw transitions; the first seven action blocks leave the corresponding frames. The final placeholder action block is not a transition target. `num_frames=window−1=7` is explicit in the model constructor.
- Dynamic/static/identity600 configurations use `forecast_fraction=.5`. The frozen training loss has a seven-input noisy observed-prefix prediction and a causal forecast branch that starts from one noisy initial latent and rolls forward seven times. The clean audit starts from one noise-free actual encoding and rolls forward seven times, with recorded actions and prefixes1…7. It receives no new real observation after initialization. Its separate teacher-forced prefix control does receive new actual past encodings.
- The cold audit's first five horizons share the first planner call's one-initial-observation information pattern. They do not use the same cases/actions/goal objective: audit64 windows score predictions under recorded actions, whereas planning50 cases rank CEM candidates against a goal.
- The later planner call's three-real-frame/five-future prediction is not directly measured by that one-frame cold audit. Conversely, cold audit h7 is35 raw steps ahead and exceeds the planner's per-call25-step lookahead. The observed h7-versus-training-center comparison therefore cannot by itself establish which cost values the deployed planner ranks well or poorly.

There is **no search beyond the seven-step training future span** in this protocol: each plan predicts5 future steps, and even the maximum warm input context reaches only7. The full50-step environment budget consists of two separately replanned25-step segments, not one ten-step latent rollout. These distinctions are neither automatically a bug nor an established cause of current performance. No setting or evaluation change is proposed by this audit.

Exact source locations (`B=/efs/prescottsun/SPT-JEPA`):

| Responsibility | Source and lines |
|---|---|
| Actual evaluator override, model/cost/policy construction | `B/runs/dynamic_readout_source_20261031/source-v1/as_jepa_tools/benchmark_evaluate.py:31–54` |
| Training row stride and outgoing five-action blocks | Same frozen root, `as_jepa_tools/benchmark_data.py:20–54` |
| Predictor max context, residual prediction | Same frozen root, `as_jepa_lewm/model.py:83,122–124` |
| Dynamic readout changes predict only | Same frozen root, `as_jepa_lewm/spatial_carrier.py:165–171` |
| Training observed-prefix and seven-step forecast branches | Same frozen root, `as_jepa_lewm/robust.py:219–242` |
| Clean/teacher audit context boundaries | Same frozen root, `experiments/ideas_validate_20261031/audit.py:114–137` |
| Policy history growth and25-action queue | `B/sources/stable-worldmodel/stable_worldmodel/policy.py:374–403,425–442,470–518` |
| Strided real frames and intervening action blocks | Same pinned package, `buffer.py:134–185` |
| History-size choice and jointly truncated autoregression | Same pinned package, `wm/lewm/lewm.py:57–120` |
| Terminal latent squared-error cost | Same pinned package, `planning/objective.py:53–80`; `scripts/plan/config/objective/goal_mse.yaml` |
| Environment-step budget and dataset init/goal injection | Same pinned package, `world/world.py:434–440,581–626,643–658` |

The evaluator is byte-identical in frozen dynamic `d6d0a9f`, local `32fad0c`, native `7a1ab4f`, and TF/F64 `8e0f821`: SHA-256 `a205a1f1cd62cf22e3a4b9625a63842a9632c1d2e8a91bb294c963c6dbea8236`. Frozen snapshots remain distinct from the shared pinned SWM dependency. Exact dependency hashes and all12 dynamic planning-artifact hashes/actual fields are retained in `B/reports/planner-context-audit-20261031/evidence.json`.

Verified reference metrics: native TwoRoom6k at `runs/native_lewm6000_20261031/planning-tworoom-native-lewm-seed{1,2}` and Cube12k at `runs/native_cube12000_20261031/planning-cube-native-lewm-seed{1,2}`; TF TwoRoom6k at `runs/carrier_learn6000_20261031/planning-tworoom-temporal-forecast-seed{1,2}` and Cube12k at `runs/cube_tf12000_20261031/planning-cube-temporal-forecast-seed{1,2}`; F64 both tasks at `runs/carrier_learn6000_20261031/planning-{task}-carrier-identity-seed{1,2}`. This audit covers their original-command benchmark protocol; it does not substitute distinct controller-treatment results or claim anything about uninspected released-checkpoint histories.
